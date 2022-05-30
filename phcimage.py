#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
# Name        : phcimage and phcnode
# Description : Progressive Hierarchically Compressed Image and
#               Progressice Hierarchically Compressed Node.
#               PHCImage encapsulates methods to load and save a PHC image
#               as well as to print and plot the tree structure. It does NOT
#               include methods to display or to convert from/to standard
#               image formats. The from/to conversion is performed by the
#               PHCEncoder.
#               PHCNode defines the nodes used to build the tree in a
#               PHC Image. It inherits from NodeMixin so it can take advantage
#               of the anytree library. Some information within the node might
#               be compressed. This class does not provide methods to extract
#               that information. This is achieved by the PHCEncoder class.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 25-April-2022 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import sys,os,json,itertools
from anytree import RenderTree,LevelOrderGroupIter,NodeMixin,LevelOrderIter
from anytree.exporter import DotExporter

###############################################################################
# PHCNODE CLASS
###############################################################################
class PHCNode(NodeMixin):

    ###########################################################################
    # CLASS MEMBERS
    ###########################################################################
    # Used to create a unique (sequential) identifier for each object of this
    # class. It must start with 1 (since 0 is identifies the tree root parent
    # which, by definition, does not exist).
    idIterator=itertools.count(1)

    ###########################################################################
    # CONSTRUCTOR
    # Creates a new node object.
    # Input  : parentPaletteIndex - Index of the palette of the parent node
    #            that is refined by this node.
    #          thePalette - This node palette. May be compressed.
    #          indexedPixels - This node indexed pixels. Each indexed pixel is
    #            an index to the palette. May be compressed.
    #          lenPackedPalette - Size (bytes) of the compressed palette.
    #          lenPackedIndices - Size (bytes) of the compressed indexed pixels
    #          lenUnpackedPalette - Size (bytes) of the uncompressed palette.
    #          lenUnpackedIndices - Size (bytes) of the uncompressed indexed
    #            pixels. Note that the packed/unpacked sizes must be provided
    #            independently of the actual data being compressed or not. This
    #            is to help in obtaining image stats.
    #          theError - Error of the whole image if this node and all of
    #            its ancestors are processed. Used to decide the order in which
    #            the nodes should be transmitted. The proposed error measure
    #            is the sum of squared differences between the reconstructed
    #            image and the original one.
    ###########################################################################
    def __init__(self,parentPaletteIndex,thePalette,indexedPixels,lenPackedPalette,lenPackedIndices,lenUnpackedPalette,lenUnpackedIndices,theError):
        super(PHCNode,self).__init__()
        self._parentPaletteIndex=parentPaletteIndex
        self._thePalette=thePalette
        self._indexedPixels=indexedPixels
        self._theIdentifier=next(PHCNode.idIterator)
        self._lenPackedPalette=lenPackedPalette
        self._lenPackedIndices=lenPackedIndices
        self._lenUnpackedPalette=lenUnpackedPalette
        self._lenUnpackedIndices=lenUnpackedIndices
        self._theError=theError
        self._theOrder=None
        self.name='I%d,S%d'%(self._parentPaletteIndex,self._lenPackedIndices+self._lenPackedPalette)

    ###########################################################################
    # GET_SIZE
    # Returns the size (bytes) of this node (not the subtree, just this node)
    # Output : theSize - Node size in bytes
    ###########################################################################
    def get_size(self):
        return min(self._lenPackedIndices,self._lenUnpackedIndices)+min(self._lenPackedPalette,self._lenUnpackedPalette)

###############################################################################
# PHCIMAGE CLASS
###############################################################################
class PHCImage():

    ###########################################################################
    # CLASS MEMBERS
    ###########################################################################
    # Byte encoding lengths
    LEN_IDENTIFIER_BYTES=4      # Node identifier length (bytes)
    LEN_PALINDEX_BYTES=1        # Palette index length (bytes)

    # Save/Load file extensions
    PIX_EXTENSION='.PIX'        # Indexed pixels file extension
    PAL_EXTENSION='.PAL'        # Palettes file extension
    JSON_EXTENSION='.JSON'      # Meta-data (JSON format) extension

    ###########################################################################
    # CONSTRUCTOR
    # Stores the input parameters.
    # Input  : imgShape - The image shape (height, width, color channels).
    #          rootNode - The root of the encoded image tree. Must be PHCNode.
    #          bitsPerPaletteIndex - Bits to encode an index to the palette.
    #                                This means that at each node,
    #                                2**bitsPerPaletteIndex different colors
    #                                are possible. From 1 to 8.
    #          bitsPerPaletteChannel - Bits to encode each color channel.
    #                                From 1 to 8.
    #          theCompressor - Identifies the algorithm used to compress inde-
    #                          xed pixels and (maybe) the palette data as fo-
    #                          llows: 0- no compression, 1- snappy, 2- gzip,
    #                          3- lzma - xz
    ###########################################################################
    def __init__(self,imgShape=[480,640,3],rootNode=None,bitsPerPaletteIndex=4,bitsPerPaletteChannel=8,theCompressor=0):
        self._imgShape=imgShape
        self._bitsPerPaletteChannel=bitsPerPaletteChannel
        self._bitsPerPaletteIndex=bitsPerPaletteIndex
        self._theCompressor=theCompressor
        self._rootNode=rootNode

    ###########################################################################
    # SAVE
    # Saves the image data to disk. The image is saved as a series of files
    # within a folder whose name is specified (baseName). The saving format is:
    # * SPECS.json: Image meta-data.
    # * For each tree node, the following files:
    #   + 'LxxxxxNyyyyy.pix', where xxxxx is the tree level and yyyyy is the
    #     node order according to descending error. Contains the compressed
    #     indexed pixels plus a header (see code).
    #   + 'LxxxxxNyyyyy.pal', where xxxxx is the tree level and yyyyy is the
    #     node order according to descending error. Contains the (possibly)
    #     compressed palette.
    #   + 'LxxxxxNyyyyy.json', where xxxxx is the tree level and yyyyy is the
    #     node order according to descending error. Contains the node meta-data
    #     (see code).
    # Input  : baseName - Folder where the image data is to be saved.
    ###########################################################################
    def save(self,baseName):
        # The image data is composed of several files. They will be saved
        # in a folder named baseName. To prevent overwriting, if such folder
        # or a file with the same name exist, saving is aborted.
        if os.path.exists(baseName):
            sys.exit('[ERROR] FILE OR FOLDER %s ALREADY EXISTS. ABORTING.'%baseName)
        # Create the folder
        os.mkdir(baseName)
        # Prepare a meta-data dictionary
        headerDict={'imgShape':self._imgShape,
                    'bitsPerPaletteChannel':self._bitsPerPaletteChannel,
                    'bitsPerPaletteIndex':self._bitsPerPaletteIndex,
                    'theCompressor':self._theCompressor
                    }
        # Save the meta-data as JSON
        jsonFileName=os.path.join(baseName,'SPECS'+PHCImage.JSON_EXTENSION)
        with open(jsonFileName,'wt') as outFile:
            json.dump(headerDict,outFile)
        # Traverse the tree in level order
        for theLevel,theNodes in enumerate(LevelOrderGroupIter(self._rootNode)):
            # Sort the nodes in this level according to their goodness
            sortedNodes=sorted(theNodes, key=lambda x: x._theError, reverse=True)
            # For each node in this tree level
            for iNode,curNode in enumerate(sortedNodes):
                # Encode the node identifier
                bytesID=curNode._theIdentifier.to_bytes(PHCImage.LEN_IDENTIFIER_BYTES,byteorder='big')
                # Encode the parent palette index
                bytesParentPaletteIndex=curNode._parentPaletteIndex.to_bytes(PHCImage.LEN_PALINDEX_BYTES,byteorder='big',signed=True)
                # Encode the parent ID (or zero if the node is root)
                if curNode.is_root:
                    parentID=0
                else:
                    parentID=curNode.parent._theIdentifier
                bytesParentID=parentID.to_bytes(PHCImage.LEN_IDENTIFIER_BYTES,byteorder='big')
                # Create the header by concatenating the node identifier, the
                # parent palette index and the parent ID.
                curHeader=bytesID+bytesParentPaletteIndex+bytesParentID
                # Build the file names
                fileName=os.path.join(baseName,'L%05dN%05d'%(theLevel,iNode))
                pixFileName=fileName+PHCImage.PIX_EXTENSION
                palFileName=fileName+PHCImage.PAL_EXTENSION
                jsonFileName=fileName+PHCImage.JSON_EXTENSION
                # Save the header and the indexed pixels
                with open(pixFileName,'wb') as outFile:
                    outFile.write(curHeader+curNode._indexedPixels)
                # Save the palette
                with open(palFileName,'wb') as outFile:
                    outFile.write(curNode._thePalette)
                # Prepare the node meta-data dictionary
                xtraDict={'lenPackedPalette':int(curNode._lenPackedPalette),
                          'lenPackedIndices':int(curNode._lenPackedIndices),
                          'lenUnpackedPalette':int(curNode._lenUnpackedPalette),
                          'lenUnpackedIndices':int(curNode._lenUnpackedIndices),
                          'theOrder':int(curNode._theOrder),
                          'theError':int(curNode._theError)}
                # Save it as a JSON file
                with open(jsonFileName,'wt') as outFile:
                    json.dump(xtraDict,outFile)

    ###########################################################################
    # LOAD
    # Loads "maxOrder" nodes from the image data from the specified folder.
    # Nodes are loaded first level-wise and then counter-error-wise (see SAVE
    # andthe CODE).
    # Input  : baseName - Folder with the image data to load.
    #          maxOrder - Number of nodes to load or None to load them all.
    ###########################################################################
    def load(self,baseName,maxOrder=None):
        # Check if the image folder exists.
        if not os.path.exists(baseName):
            sys.exit('[ERROR] FOLDER %s DOES NOT EXIST. ABORTING.'%baseName)
        # Load the meta-data
        jsonFileName=os.path.join(baseName,'SPECS'+PHCImage.JSON_EXTENSION)
        with open(jsonFileName,'rt') as inFile:
            headerDict=json.load(inFile)
        self._imgShape=headerDict['imgShape']
        self._imgShape=headerDict['imgShape']
        self._bitsPerPaletteChannel=headerDict['bitsPerPaletteChannel']
        self._bitsPerPaletteIndex=headerDict['bitsPerPaletteIndex']
        self._theCompressor=headerDict['theCompressor']
        # Get all the files in the image directory and initialize some loop
        # control and storage variables.
        allFiles=os.listdir(baseName)
        allLevelsExplored=False
        curLevel=0
        theNodes=[]
        # Loop for all image levels.
        while not allLevelsExplored:
            # Get this level files (level number is encoded in the filename).
            # Note that the -len(PHCImage.PIX_EXTENSION) is to remove the file extension (PHCImage.PIX_EXTENSION)
            thisLevelFiles=sorted([x[:-len(PHCImage.PIX_EXTENSION)] for x in allFiles if x.startswith('L%05d'%curLevel) and x.endswith(PHCImage.PIX_EXTENSION)])
            # If there are no files in this level, that's all.
            if len(thisLevelFiles)==0:
                allLevelsExplored=True
                break
            # For each file in this level
            for fileName in thisLevelFiles:
                # Build the file names
                pixFileName=fileName+PHCImage.PIX_EXTENSION
                palFileName=fileName+PHCImage.PAL_EXTENSION
                jsonFileName=fileName+PHCImage.JSON_EXTENSION
                # Read and store the node meta-data
                with open(os.path.join(baseName,jsonFileName),'rt') as inFile:
                    xtraDict=json.load(inFile)
                    lenPackedPalette=xtraDict['lenPackedPalette']
                    lenPackedIndices=xtraDict['lenPackedIndices']
                    lenUnpackedPalette=xtraDict['lenUnpackedPalette']
                    lenUnpackedIndices=xtraDict['lenUnpackedIndices']
                    theOrder=xtraDict['theOrder']
                    theError=xtraDict['theError']
                # Read this node header and the indexed pixels
                with open(os.path.join(baseName,pixFileName),'rb') as inFile:
                    theData=inFile.read()
                    # Compute the indices to separate header from data and
                    # to separate each header component
                    headerLen=PHCImage.LEN_IDENTIFIER_BYTES*2+PHCImage.LEN_PALINDEX_BYTES
                    endIdentifier=PHCImage.LEN_IDENTIFIER_BYTES
                    endPalIndex=PHCImage.LEN_IDENTIFIER_BYTES+PHCImage.LEN_PALINDEX_BYTES
                    theHeader=theData[:headerLen]
                    theIdentifier=int.from_bytes(theHeader[:endIdentifier],byteorder='big')
                    parentPaletteIndex=int.from_bytes(theHeader[endIdentifier:endPalIndex],byteorder='big')
                    parentID=int.from_bytes(theHeader[endPalIndex:],byteorder='big')
                    indexedPixels=theData[headerLen:]
                # Read this node palette
                with open(os.path.join(baseName,palFileName),'rb') as inFile:
                    thePalette=inFile.read()
                # Create the node
                curNode=PHCNode(parentPaletteIndex,thePalette,indexedPixels,lenPackedPalette,lenPackedIndices,lenUnpackedPalette,lenUnpackedIndices,theError)
                # Overwrite the identifier
                curNode._theIdentifier=theIdentifier
                # Store the loaded order
                curNode._theOrder=theOrder
                # Assign the node to its parent if possible
                parentFound=False
                for otherNode in theNodes:
                    if otherNode._theIdentifier==parentID:
                        curNode.parent=otherNode
                        parentFound=True
                        break
                if not parentFound and len(theNodes)!=0:
                    print('[NODE WITHOUT PARENT]')
                theNodes.append(curNode)
                if (not maxOrder is None) and curNode._theOrder>=maxOrder:
                    allLevelsExplored=True
                    break
            # Advance tree level
            curLevel+=1
        # Search all possible root nodes and assign the first one as the
        # image root node (this could be improved to check if there is more
        # than one root node).
        rootNode=[n for n in theNodes if n.is_root]
        self._rootNode=rootNode[0]

    ###########################################################################
    # GET_NUM_NODES
    # Counts the number of nodes in the image tree
    ###########################################################################
    def get_num_nodes(self):
        return sum(1 for _ in LevelOrderIter(self._rootNode))

    ###########################################################################
    # PRINT
    # Prints the tree structure.
    ###########################################################################
    def print(self):
        for pre, _, node in RenderTree(self._rootNode):
            print("%s%s" % (pre, node.name))

    ###########################################################################
    # PLOT
    # Saves the tree structure as an image. Requires GraphViz.
    ###########################################################################
    def plot(self,fileName):
        if os.path.exists(fileName):
            sys.exit('[ERROR] FILE %s ALREADY EXISTS. ABORTING.'%fileName)
        DotExporter(self._rootNode).to_picture(fileName)