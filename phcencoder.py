#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
# Name        : phcencoder
# Description : Progressive Hierarchical Compression Encoder.
#               Encodes an image (Numpy array) into a PHCImage and decodes
#               a PHCImage into an image (Numpy array). Only RGB images are
#               allowed.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 25-April-2022 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORT
###############################################################################
import numpy as np
import gzip,snappy,lzma
from skimage import img_as_ubyte
from sklearn.cluster import KMeans
from anytree import LevelOrderGroupIter
from math import ceil
from scipy.spatial.distance import cdist
from bitarray import bitarray
from bitarray.util import int2ba,ba2int
from phcimage import PHCImage,PHCNode

###############################################################################
# PHCENCODER CLASS
###############################################################################
class PHCEncoder():

    ###########################################################################
    # CONSTRUCTOR
    # Create a PHCEncoder object.
    # Input  : bitsPerPaletteIndex - Number of bits to encode a palette index.
    #            That is, 2**bitsPerPaletteIndex is the maximum number of
    #            colors per node. From 1 to 8. Low values lead to smaller
    #            nodes but a larger number of them.
    #          bitsPerPaletteChannel - Number of bits to encode a color chan-
    #            nel. From 1 to 8. Values below 8 mean lossy compression.
    #          useKMeans - If True, the color clustering will be performed
    #            using KMeans. If False, a linear palette (from the darkest
    #            to the lightest color) will be created and each pixel assigned
    #            to the closest color. KMeans is a best choice (theoretically)
    #            in terms of quality but it is slower.
    #          theCompressor - Algorithm used to compress indexed pixels and
    #            palettes. Values are: 0 no compression, 1 snappy, 2 gzip,
    #            3 lzma - xz. The encoder checks if the compressed data is
    #            smaller than the uncompressed one. If not, then the data is
    #            not compressed. This rarely happens with indexed pixels but
    #            quite often with the palette.
    #          pctFit - Percentage of input data to fit if KMeans is selected.
    #          minFit - If KMeans is to be used and the percentage of input
    #            data leads to a number of samples below minFit, then minFit
    #            samples are used.
    ###########################################################################
    def __init__(self,bitsPerPaletteIndex=4,bitsPerPaletteChannel=8,useKMeans=False,theCompressor=1,pctFit=0.05,minFit=1000):
        self._bitsPerPaletteIndex=bitsPerPaletteIndex
        self._bitsPerPaletteChannel=bitsPerPaletteChannel
        self._useKMeans=useKMeans
        self._pctFit=pctFit
        self._minFit=minFit
        self._maxOrder=None
        self._theCompressor=theCompressor
        self._paletteChannelMask=(((1<<self._bitsPerPaletteChannel)-1)<<(8-self._bitsPerPaletteChannel))
        self._numPaletteEntries=(1<<self._bitsPerPaletteIndex)
        self._reduce_colors=[self._reduce_colors_linspace,self._reduce_colors_kmeans][int(useKMeans)]

    ###########################################################################
    # _RGB_TO_GRAY
    # Converts a list of RGB colors to a list of gray intensities.
    # Input  : rgbData - Nx3 matrix. N colors, 3 channels (RGB)
    # Output : grayData - N grayscale values.
    # Note   : The output range is the same as the input one, but the output
    #          is always float (even if values range from 0 to 255)
    ###########################################################################
    def _rgb_to_gray(self,rgbData):
        return np.sum(rgbData.astype('float')*[0.2125,0.7154,0.0721],axis=1)

    ###########################################################################
    # _SORT_PALETTE
    # Sorts a palette according to the corresponding colors grayscale.
    # Input  : thePalette - N x 3 matrix. The palette to sort.
    #          indexedPixels - The indexed image that uses thePalette
    # Output : thePalette - Sorted palette.
    #          indexedPixels - The indexed image adapted to the sorted palette.
    # Note   : The output palette is in the same scale as the input palette
    #          but with type uint8.
    ###########################################################################
    def _sort_palette(self,thePalette,indexedPixels):
        # Get the sort indices according to the corresponding grayscale palette
        idxSort=np.argsort(self._rgb_to_gray(thePalette))
        # Sort the palette according to these indices
        thePalette=thePalette[idxSort,:]
        # Change the indexed pixels values according to the sorted palette
        outPixels=np.zeros(indexedPixels.shape,dtype='uint8')
        for newIndex,oldIndex in enumerate(idxSort):
            outPixels[indexedPixels==oldIndex]=newIndex
        return thePalette,outPixels

    ###########################################################################
    # _PROCESS_ALL_COLORS
    # Given a set of RGB pixels, outputs the corresponding palette (including
    # all the present colors) and the indexed image.
    # Input  : rgbPixels - The RGB pixels
    # Output : thePalette - Sorted palette with all colors in rgbPixels
    #          indexedPixels - The indexed image.
    ###########################################################################
    def _process_all_colors(self,rgbPixels):
        # Mask the rgb pixels according to the desired number of bits
        maskedRGBPixels=rgbPixels & self._paletteChannelMask
        # Get the unique colors and the required indices
        grayPalette,palInd,indexedPixels=np.unique(self._rgb_to_gray(maskedRGBPixels),axis=0,return_index=True,return_inverse=True)
        # Get the palette with the unique colors
        thePalette=maskedRGBPixels[palInd]
        # Now sort the palette and renumber the indexed pixels accordingly
        thePalette,indexedPixels=self._sort_palette(thePalette,indexedPixels.astype('uint8'))
        return thePalette,indexedPixels

    ###########################################################################
    # _REDUCE_COLORS_LINSPACE
    # Given a set of rgbPixels outputs a palette involving only 2**bitsPerPa-
    # letteEntry and the corresponding indexed image. The palette is generated
    # linearly from the darkest color to the lightest one.
    # Input  : rgbPixels - The RGB pixels.
    # Output : thePalette - Sorted palette.
    #          indexedPixels - The indexed image.
    ###########################################################################
    def _reduce_colors_linspace(self,rgbPixels):
        # Get the grayscale pixels
        grayPixels=self._rgb_to_gray(rgbPixels)
        # Build the palete as a linear gradation from the darkest color to the
        # lightest one. Darkest and lightest are computes from the grayscale
        # palette.
        thePalette=np.linspace(rgbPixels[np.argmin(grayPixels),:],rgbPixels[np.argmax(grayPixels),:],self._numPaletteEntries,dtype='uint8')
        # Assign each pixel the the closet color
        indexedPixels=np.argmin(cdist(rgbPixels,thePalette),axis=1).astype('uint8')
        # Mask the palette
        thePalette=thePalette & self._paletteChannelMask
        return thePalette,indexedPixels

    ###########################################################################
    # _REDUCE_COLORS_KMEANS
    # Same as _REDUCE_COLORS_LINSPACE but using KMeans to obtain the palette.
    # Input  : rgbPixels - The RGB pixels.
    # Output : thePalette - Sorted palette.
    #          indexedPixels - The indexed image.
    ###########################################################################
    def _reduce_colors_kmeans(self,rgbPixels):
        # The number of samples used to perform K-Means is the maximum
        # between the percentage (_pctFit) and the minimum allowable numner
        # of pixels (minFit).
        numRandomSamples=min(round(max(self._minFit,rgbPixels.shape[0]*self._pctFit)),rgbPixels.shape[0])
        # Select such a number of random pixels from the provided image data.
        randomSamples=rgbPixels[np.random.choice(rgbPixels.shape[0],numRandomSamples,replace=False),:]
        # Perform K-Means
        kMeans=KMeans(n_clusters=self._numPaletteEntries,init='k-means++',random_state=0).fit(randomSamples)
        # Now assign each pixel to the corresponding centroid
        indexedPixels=kMeans.predict(rgbPixels)
        # The palette is just the K-Means centroids
        thePalette=kMeans.cluster_centers_.astype('uint8') & self._paletteChannelMask
        # Sort the palette
        thePalette,indexedPixels=self._sort_palette(thePalette,indexedPixels)
        return thePalette,indexedPixels

    ###########################################################################
    # _COMPUTE_INDEXATION_ERROR
    # Given an indexed image, it computes the error introduced by each of its
    # palette entries when compared to the actual (source) image. The error is
    # the sum of squared differences.
    # Input  : thePalette - Palette of the indexed image.
    #          indexedPixels - Indexed image.
    #          rgbPixels - Original image data.
    # Output : indexErrors - Error for each palette entry.
    ###########################################################################
    def _compute_indexation_error(self,thePalette,indexedPixels,rgbPixels):
        indexErrors=np.zeros(thePalette.shape[0],dtype='int64')
        for idxPalette,valPalette in enumerate(thePalette):
            indexErrors[idxPalette]=np.sum((rgbPixels[indexedPixels==idxPalette]-valPalette)**2)
        return indexErrors

    ###########################################################################
    # _COMPRESS
    # Compresses the provided data.
    # Input  : theData - Data to compress
    # Output : theData - Compressed data
    # Note   : Compression is performed depending on self._theCompressor, which
    #          can be:
    #            + 0: No compression
    #            + 1: Snappy
    #            + 2: Gzip
    #            + 3: LZMA-XZ
    ###########################################################################
    def _compress(self,theData):
        if self._theCompressor==1:
            return snappy.compress(theData)
        if self._theCompressor==2:
            return gzip.compress(theData,compresslevel=9)
        if self._theCompressor==3:
            return lzma.compress(theData,preset=9)
        return theData

    ###########################################################################
    # _DECOMPRESS
    # Decompresses the provided data.
    # Input  : theData - Data to depack
    # Output : theData - Decompressed data
    # Note   : Decompression is performed depending on self._theCompressor.
    #          See _compress to learn about the possible compression algorithms
    ###########################################################################
    def _decompress(self,theData):
        if self._theCompressor==1:
            return snappy.decompress(theData)
        if self._theCompressor==2:
            return gzip.decompress(theData)
        if self._theCompressor==3:
            return lzma.decompress(theData)
        return theData

    ###########################################################################
    # _UNPACK_BYTES
    # Unpacks data bytes. The first byte states if the data is compressed or
    # not. If it is, it is depacked. Note that this byte being 0 (data not
    # compressed) and the compression method being 0 are different things.
    # A compression method of 0 is meant for testing purposes. This byte being
    # 0 is decided automatically if the compressed data is larger than the
    # uncompressed one.
    # Input  : pckData - Packed data with the first byte as described before.
    # Output : rawBits - Depacked data as a bitarray.
    ###########################################################################
    def _unpack_bytes(self,pckData):
        # If data was not compressed, return it.
        if pckData[0]==0:
            outData=pckData[1:]
        # Otherwise, depack it
        else:
            outData=self._decompress(pckData[1:])
        # Convert it to a bitarray
        rawBits=bitarray()
        rawBits.frombytes(outData)
        return rawBits

    ###########################################################################
    # _UNPACK_DATA
    # Unpacks palette and indexed pixels.
    # Input  : pckPalette - Packed palette
    #          pckIndexedPixels - Packed indexed image
    # Output : thePalette - Unpacked palette
    #          indexedPixels - Unpacked indexed image
    ###########################################################################
    def _unpack_data(self,pckPalette,pckIndexedPixels):
        # Unpack the palette bytes and prepare space for the unpacked version
        rawBits=self._unpack_bytes(pckPalette)
        thePalette=np.zeros(ceil(len(rawBits)/(self._bitsPerPaletteChannel*3))*3,dtype='uint8')
        # Build the unpacked palette
        for i,j in enumerate(range(0,len(rawBits),self._bitsPerPaletteChannel)):
            theValue=rawBits[j:j+self._bitsPerPaletteChannel]+bitarray('0')*(8-self._bitsPerPaletteChannel)
            thePalette[i]=(ba2int(theValue))
        # Reshape it to be Nx3
        thePalette=thePalette.reshape((-1,3))
        # Get the indexed pixels bytes and prepare space for the unpacked vers.
        rawBits=self._unpack_bytes(pckIndexedPixels)
        numIndices=ba2int(rawBits[:32])
        indexedPixels=np.zeros(numIndices,dtype='uint8')
        # Build the unpacked indexed pixels
        for i in range(numIndices):
            theValue=rawBits[32+(i*self._bitsPerPaletteIndex):32+((i+1)*self._bitsPerPaletteIndex)]
            indexedPixels[i]=(ba2int(theValue))
        return thePalette,indexedPixels

    ###########################################################################
    # _PACK_DATA
    # Packs palette and indexed pixels
    # Input  : thePalette - Palette to pack
    #          indexedPixels - Indexed pixels to pack
    # Output : outPalette - Packed palette
    #          outIndices - Packed indexed pixels
    #          lenPackedPalette - Size (bytes) of the packed palette
    #          lenPackedIndices - Size (bytes) of the packed indices
    #          lenUnpackedPalette - Size (bytes) of the unpacked palette
    #          lenUnpackedIndices - Size (bytes) of the unpacked indices
    # Note   : The palette and the indices are either compressed or not. Both
    #          sizes are provided for comparison purposes.
    ###########################################################################
    def _pack_data(self,thePalette,indexedPixels):
        # Prepare the palette and indexedpixels bit arrays.
        palData=bitarray()
        indData=bitarray()
        # Prepare the codes used for compressed and uncompressed data.
        pckCode=bitarray('00000001').tobytes()
        unpckCode=bitarray('00000000').tobytes()
        # For each palette item
        for curVal in thePalette:
            # Mask the components to have the desired bits.
            rVal=int(curVal[0] >> (8-self._bitsPerPaletteChannel))
            gVal=int(curVal[1] >> (8-self._bitsPerPaletteChannel))
            bVal=int(curVal[2] >> (8-self._bitsPerPaletteChannel))
            # Store the masked RGB as a bitarray
            palData+=int2ba(rVal,self._bitsPerPaletteChannel)+int2ba(gVal,self._bitsPerPaletteChannel)+int2ba(bVal,self._bitsPerPaletteChannel)
        # Convert it to bytes
        bytesPalette=palData.tobytes()
        # Compress
        packedPalette=pckCode+self._compress(bytesPalette)
        # Also build the uncompressed version
        unpackedPalette=unpckCode+bytesPalette
        paletteIndexMask=int(self._numPaletteEntries-1)
        # Now convert the indexed pixels to bitarray
        numIndices=int2ba(indexedPixels.shape[0],32).tobytes()
        for curVal in indexedPixels:
            indData+=int2ba(int(curVal & paletteIndexMask),self._bitsPerPaletteIndex)
        # Convert it to bytes
        bytesIndices=indData.tobytes()
        # Compress it
        packedIndices=pckCode+self._compress(numIndices+bytesIndices)
        # Also build the uncompressed version
        unpackedIndices=unpckCode+numIndices+bytesIndices
        # Get the lengths
        lenPackedPalette=len(packedPalette)
        lenUnpackedPalette=len(unpackedPalette)
        lenPackedIndices=len(packedIndices)
        lenUnpackedIndices=len(unpackedIndices)
        # Decide which ones (compressed or uncompressed) to return depending
        # on their size.
        if lenPackedPalette<lenUnpackedPalette:
            outPalette=packedPalette
        else:
            outPalette=unpackedPalette
        if lenPackedIndices<lenUnpackedIndices:
            outIndices=packedIndices
        else:
            outIndices=unpackedIndices
        return outPalette,outIndices,lenPackedPalette,lenPackedIndices,lenUnpackedPalette,lenUnpackedIndices

    ###########################################################################
    # _ENCODE_NODE
    # Given a set of rgbPixels, builds the corresponding sub-tree recursively.
    # Input  : rgbPixels - RGB pixels to encode.
    #          parentPaletteIndex - Palette index of the parent node whose
    #                      pixels are refined by this node or -1 if this is
    #                      the first (root) node.
    #          baseError - The whole image error if the rgbPixels here are not
    #                      considered.
    #          theLevel - Recursion level. Used internally to debug large
    #                     recursion depths.
    # Output : theNode - The encoded sub-tree
    ###########################################################################
    def _encode_node(self,rgbPixels,parentPaletteIndex,baseError,theLevel=0):
        # Start by processing all the existing colors
        thePalette,indexedPixels=self._process_all_colors(rgbPixels)
        # If all the existing colors can be represented using the desired
        # number of bits per palette entry, that's it: recursion ends here.
        if thePalette.shape[0]<=self._numPaletteEntries:
            # Compute he node error as the base error plus this palette error.
            nodeError=baseError+np.sum(self._compute_indexation_error(thePalette,indexedPixels,rgbPixels))
            # Pack the data
            thePalette,indexedPixels,lenPackedPalette,lenPackedIndices,lenUnpackedPalette,lenUnpackedIndices=self._pack_data(thePalette,indexedPixels)
            # Return the node with this data
            return PHCNode(parentPaletteIndex,thePalette,indexedPixels,lenPackedPalette,lenPackedIndices,lenUnpackedPalette,lenUnpackedIndices,nodeError)
        # If too many colors, reduce them (KMeans or linear interpolation)
        thePalette,indexedPixels=self._reduce_colors(rgbPixels)
        # Get decoupled palette error
        indexErrors=self._compute_indexation_error(thePalette,indexedPixels,rgbPixels)
        # The node error is the base error plus the (coupled) palette error
        nodeError=baseError+np.sum(indexErrors)
        # Pack the palette and the indexed pixels
        pckPalette,pckIndices,lenPackedPalette,lenPackedIndices,lenUnpackedPalette,lenUnpackedIndices=self._pack_data(thePalette,indexedPixels)
        # Build the node
        parentNode=PHCNode(parentPaletteIndex,pckPalette,pckIndices,lenPackedPalette,lenPackedIndices,lenUnpackedPalette,lenUnpackedIndices,nodeError)
        # Process the (possible) node children.
        for idxPalette in range(thePalette.shape[0]):
            # Get the rgbPixels of thie candidate child
            rgbChildPixels=rgbPixels[indexedPixels==idxPalette]
            # If there are pixels...
            if rgbChildPixels.shape[0]>0:
                # The child base error is the parent error minus the part of
                # the error that the child is going to change.
                childBaseError=nodeError-indexErrors[idxPalette]
                # Recursively encode this child
                childNode=self._encode_node(rgbChildPixels,idxPalette,childBaseError,theLevel+1)
                # Set its parent to this node
                childNode.parent=parentNode
        # Return the sub-tree
        return parentNode

    ###########################################################################
    # _DECODE_NODE
    # Recursively decodes the sub-tree whose root is the provided node.
    # Input  : theNode - A PHCNode which is the root of the sub-tree to decode.
    # Output : rgbData - rgbData corresponding to the subtree.
    # Note   : If self._maxOrder is None, the whole subtree is decoded. If it
    #          is an integer, only nodes whose "_theOrder" field is lower or
    #          equal to _maxOrder are decoded.
    ###########################################################################
    def _decode_node(self,theNode):
        # Compute the packed size
        self._packedSize+=theNode.get_size()
        # Unpack the node data
        thePalette,indexedPixels=self._unpack_data(theNode._thePalette,theNode._indexedPixels)
        # Prepare space for the decoded rgb pixels
        outData=np.zeros((indexedPixels.shape[0],thePalette.shape[1]),dtype='uint8')
        # If the node is a leaf or the maxOrder is reached
        if theNode.is_leaf or ((not (self._maxOrder is None)) and theNode._theOrder>=self._maxOrder):
            # Prepare the output data and exit.
            for idxPalette,valPalette in enumerate(thePalette):
                outData[indexedPixels==idxPalette,:]=valPalette
            return outData
        # Create a list with all possible palette indices
        idxPalette=list(range(thePalette.shape[0]))
        # For each child node
        for curChild in theNode.children:
            # Decode it
            childData=self._decode_node(curChild)
            # It there is decoded data
            if not (childData is None):
                # Put this data into the decoded output
                outData[indexedPixels==curChild._parentPaletteIndex,:]=childData
                # Remove this palette index from the list
                idxPalette.remove(curChild._parentPaletteIndex)
        # For each children that provided no data
        for curIndex in idxPalette:
            # Put this node data into the decoded output
            outData[indexedPixels==curIndex,:]=thePalette[curIndex,:]
        return outData

    ###########################################################################
    # ENCODE
    # Encodes the provided image.
    # Input  : theImage - Image to encode (HxWxC Numpy array)
    # Output : outImage - Encoded PHCImage
    ###########################################################################
    def encode(self,theImage):
        # Convert the image to ubyte and reshape it
        rgbPixels=img_as_ubyte(theImage).reshape((-1,theImage.shape[2]))
        # Recursively encode the image
        rootNode=self._encode_node(rgbPixels,-1,0)
        # Assign the _theOrder field level by level. Within each level, the
        # node error (in descending order) determines the order.
        theOrder=0
        for theNodes in LevelOrderGroupIter(rootNode):
            # At each tree level, sort the nodes according to their error
            sortedNodes=sorted(theNodes, key=lambda x: x._theError, reverse=True)
            # Assign the order
            for theNode in sortedNodes:
                theNode._theOrder=theOrder
                theOrder+=1
        # Return the encoded image
        return PHCImage(theImage.shape,rootNode,self._bitsPerPaletteIndex,self._bitsPerPaletteChannel,self._theCompressor)

    ###########################################################################
    # DECODE
    # Decodes a maximum of maxOrder nodes of the provided encoded image.
    # Input  : phcImage - Encoded image (PHCImage object)
    #          maxOrder - Maximum number of nodes to decode.
    # Output : theImage - Decoded image (HxWxC Numpy array)
    ###########################################################################
    def decode(self,phcImage,maxOrder=None):
        # Get the encoding parameters from the image
        self._bitsPerPaletteIndex=phcImage._bitsPerPaletteIndex
        self._bitsPerPaletteChannel=phcImage._bitsPerPaletteChannel
        self._theCompressor=phcImage._theCompressor
        self._paletteChannelMask=(((1<<self._bitsPerPaletteChannel)-1)<<(8-self._bitsPerPaletteChannel))
        self._numPaletteEntries=(1<<self._bitsPerPaletteIndex)
        self._packedSize=0
        self._maxOrder=maxOrder
        # Decode the image recursively
        rgbPixels=self._decode_node(phcImage._rootNode)
        # Reshape the decoded pixels and return them
        return rgbPixels.reshape(phcImage._imgShape)