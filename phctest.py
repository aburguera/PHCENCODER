#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
# Name        : phctest
# Description : Progressive Hierarchical Compression Test.
#               Usage example of PHCEncoder and PHCImage. The most part of the
#               code is to obtain stats. To encode an image, simply do:
#               > theEncoder=PHCEncoder() # Specify parameters or leave default
#               > encodedImage=theEncoder.encode()
#               To decode an image simply do:
#               > theEncoder=PHCEncoder()
#               > theImage=theEncoder.decode()
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 28-April-2022 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

from phcimage import PHCImage
from phcencoder import PHCEncoder
from skimage.io import imread
from os.path import getsize,exists
import numpy as np
import matplotlib.pyplot as plt
from anytree import LevelOrderGroupIter

###############################################################################
# GLOBAL PARAMETERS
###############################################################################

# =============================================================================
# PHC ENCODER PARAMETERS (SEE PHCENCODER)
# =============================================================================

BITS_PER_PALETTE_INDEX=2
BITS_PER_PALETTE_CHANNEL=8
COMPRESSION_ALGORITHM=1
USE_KMEANS=False

# =============================================================================
# BASIC EVALUATION PARAMETERS
# =============================================================================

NUM_TESTS=100

# =============================================================================
# PATHS
# =============================================================================

# File name of the image to test
testImageFileName="IMG/IMG.png"
# Folder name to save the PHC image
testImageFolderName="IMG/IMG"

###############################################################################
# AUXILIARY FUNCTIONS
###############################################################################

# =============================================================================
# Compute the mean squared error of the provided image with respect to the
# provided ground truth.
# Input  : theImage - Image to evaluate.
#          groundTruth - Correct image.
# Output : theError - Mean squared error.
# =============================================================================

def compute_error(theImage,groundTruth):
    return np.mean((theImage.astype('float')-groundTruth.astype('float'))**2)

###############################################################################
# MAIN CODE
###############################################################################

# =============================================================================
# LOAD IMAGE
# =============================================================================

# Get the test image
print('* LOADING BASE IMAGE')
groundTruth=imread(testImageFileName)
gtSize=getsize(testImageFileName)
rawSize=np.prod(groundTruth.shape)

# =============================================================================
# ENCODE AND SAVE IMAGE
# =============================================================================

# Create the PHC encoder
theEncoder=PHCEncoder(bitsPerPaletteIndex=BITS_PER_PALETTE_INDEX,theCompressor=COMPRESSION_ALGORITHM,bitsPerPaletteChannel=BITS_PER_PALETTE_CHANNEL,useKMeans=USE_KMEANS)
# Encode and save
if not exists(testImageFolderName):
    print('* ENCODING THE IMAGE')
    encodedImage=theEncoder.encode(groundTruth)
    print('* SAVING THE ENCODED IMAGE')
    encodedImage.save(testImageFolderName)
# Load the image
else:
    print('* LOADING THE ENCODED IMAGE')
    encodedImage=PHCImage()
    encodedImage.load(testImageFolderName)

# =============================================================================
# SEARCH NUMBER OF NODES WITH SIZE SIMILAR TO SOURCE IMAGE
# =============================================================================

# Search the point at which the PHC image size is similar to the rawSize
print('* SEARCHING THE NUMBER OF NODES WITH SIMILAR SIZE TO RAW DATA')
bUpper=encodedImage.get_num_nodes()
bLower=0
while True:
    bMid=int((bUpper+bLower)/2)
    outImage=theEncoder.decode(encodedImage,bMid)
    if theEncoder._packedSize>=rawSize and theEncoder._packedSize<=rawSize*1.05:
        bUpper=bMid
        break
    if theEncoder._packedSize>rawSize:
        bUpper=bMid
    else:
        bLower=bMid
theError=compute_error(outImage, groundTruth)
print('  + WITH %d NODES THE SIZE IS %d BYTES WITH AN ERROR OF %f'%(bUpper,theEncoder._packedSize,theError))

# =============================================================================
# COMPUTE SIZES AND ERRORS
# =============================================================================

# Compute the error decoding the image using a number of nodes ranging from 0
# to the point found before (bUpper).
print('* COMPUTING ENCODED IMAGE ERRORS AND SIZES')
allErrors=[]
allSizes=[]
for maxOrder in np.linspace(0,bUpper,NUM_TESTS):
    # Decode the image using maxOrder nodes
    outImage=theEncoder.decode(encodedImage,maxOrder)
    # Compute the error and get the size
    curError=compute_error(outImage,groundTruth)
    curSize=theEncoder._packedSize
    # Store the data
    allErrors.append(curError)
    allSizes.append(curSize)

# =============================================================================
# COMPUTE INDIVIDUAL MESSAGE (NODE) SIZES
# =============================================================================

print('* COMPUTING MESSAGE SIZES')
messageSizes=[]
isDone=False
for curLevel in LevelOrderGroupIter(encodedImage._rootNode):
    for curNode in curLevel:
        messageSizes.append(curNode.get_size())
        if curNode._theOrder>=bUpper:
            isDone=True
            break
    if isDone:
        break

# =============================================================================
# PLOT SOME RESULTS
# =============================================================================

# Plot encoded messages errors and sizes
plt.figure()
plt.plot(allSizes,allErrors,'k')
plt.plot([rawSize,rawSize],[0,max(allErrors)],'r')
plt.plot([gtSize,gtSize],[0,max(allErrors)],'b')
plt.grid('on')
plt.axis([0,max(allSizes),0,max(allErrors)])
plt.legend(['PHC','RAW','PNG'])
plt.xlabel('TOTAL SIZE (BYTES)')
plt.ylabel('MEAN SQUARED ERROR')
plt.show()

# Plot message sizes
plt.figure()
plt.plot(messageSizes,'r')
plt.grid('on')
plt.axis([0,len(messageSizes)-1,0,max(messageSizes)])
plt.xlabel('MESSAGE NUMBER')
plt.ylabel('MESSAGE SIZE (BYTES)')
plt.show()