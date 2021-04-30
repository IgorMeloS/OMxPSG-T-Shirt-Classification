# =============================================================================
# Script to automate the training process of a Haar Cascade.
# =============================================================================

###### Importing Libraries ######
from getopt import getopt
from sys import argv
from os import listdir, path, makedirs, remove
from re import sub
from subprocess import check_call
from time import sleep
from shutil import rmtree, copytree, copy
import cv2 as cv


# =============================================================================
# Defining the main directory and the directories for the original images.
# This is a safe way to conserve your original data.
# Positive images contain the object to be recognized.
# Negative images contain the images background.
# =============================================================================

MAIN_DATA_DIR = path.abspath("dataset")
ORG_POS = "dataset/orig-pos"
ORG_NEG = "dataset/orig-neg"

"""
    ### Import to know ###
    
    To avoid any kind of problem with the training process, it's recommended
    to use the absolute path to all directories and files.
    
"""
# =============================================================================
# Defining the directories for the training sets.
# =============================================================================

TR_POS_IMAGES = path.abspath("dataset/pos") # Training positive images
TR_NEG_IMAGES = path.abspath("dataset/neg") # Training negative images
SAMPLE_IMAGES = MAIN_DATA_DIR +"/positive-samples" # Folder to the samples images cropped and annotated
DIR_HAAR_DATA = MAIN_DATA_DIR + "/haar-cascade-data" # Folder to store the Haar cascade and its training stages and parameters

# =============================================================================
# Defining some important files as background, sample and vector.
# =============================================================================

NEG_COLLECTION = TR_NEG_IMAGES + "/" + TR_NEG_IMAGES.split("/")[-1]+".txt" # Negative images list
SAMPLE_COLLECTION = SAMPLE_IMAGES+ "/" + SAMPLE_IMAGES.split("/")[-1]+".txt" # Annotation file for the positive images with their background.
SAMPLE_VEC = SAMPLE_IMAGES + ".vec" # File for vectorized images


# =============================================================================
# Starting the script for the full training process 
# =============================================================================

if __name__ == "__main__":
    (createSamples, createVec, trainCascade) = (False, False, False)
    opts, args = getopt(argv[1:],"svt",["createSamples","createVec","trainCascade"])
    for opt, arg in opts:
        if(opt in ("--createSamples","-s")):
            createSamples = True
        elif(opt in ("--createVec","-v")):
            createVec = True
        elif(opt in ("--trainCascade","-t")):
            trainCascade = True
    
    # =============================================================================
    # Images preprocessing
    #  - Copy to the train directory
    #  - Changing the color scale
    #  - Resizing 
    # =============================================================================
    
    ##### Defining function to process all negative images ######
    
    def resizeNegImages():
        
        imagePath = listdir(ORG_NEG) # list of images from the directory orig-neg
        identify = 1 # counter for each image
        
        if not path.isdir(TR_NEG_IMAGES):
            makedirs(TR_NEG_IMAGES) # To create the directory with the negative images for the training process
        
        for i in imagePath:
            i.replace(i, TR_NEG_IMAGES + "/" + str(identify) + ".JPEG") # To replace the image from the directory orig-neg into the repository neg
            copy(ORG_NEG + "/" + i, i.replace(i, TR_NEG_IMAGES + "/" + str(identify) + ".JPEG")) # To copy the image in the repository neg
            img = cv.imread(TR_NEG_IMAGES + "/" + str(identify) + ".JPEG", cv.IMREAD_GRAYSCALE) # To change the color scale into gray scale
            resize = cv.resize(img, (100, 100)) # To resize the image
            cv.imwrite(TR_NEG_IMAGES + "/" + str(identify) + ".JPEG", resize) # To write the processed image
            identify += 1 # to increase the counter that labeling each image
            
    resizeNegImages() # applying the function
    
   ##### Defining function to process all positive images ######
    
    def resizePosImages():
        imagePath = listdir(ORG_POS)
        identify = 1
        if not path.isdir(TR_POS_IMAGES):
            makedirs(TR_POS_IMAGES)
            
        for i in imagePath:
            i.replace(i, TR_POS_IMAGES + "/" + str(identify) + ".JPEG")
            copy(ORG_POS + "/" + i, i.replace(i, TR_POS_IMAGES + "/" + str(identify) + ".JPEG"))
            img = cv.imread(TR_POS_IMAGES + "/" + str(identify) + ".JPEG", cv.IMREAD_GRAYSCALE)
            resize = cv.resize(img, (100, 100))
            cv.imwrite(TR_POS_IMAGES + "/" + str(identify) + ".JPEG", resize)
            identify += 1
            
    resizePosImages()
   
    ##### To be sure that we have the output directory #####
    if(not path.isdir(SAMPLE_IMAGES)):
        makedirs(DIR_SAMPLE_IMAGES)
    
    ##### Creating the background file ######
    negImageCollectionFile = open(NEG_COLLECTION , "w")
    negImageFilenames = [f for f in listdir(TR_NEG_IMAGES) if path.isfile(path.join(TR_NEG_IMAGES,f)) and f.endswith("JPEG")]
    for f in negImageFilenames:
        negImageCollectionFile.write(TR_NEG_IMAGES + "/" + f)
        negImageCollectionFile.write("\n")
    negImageCollectionFile.close()
    
    ##### Making the list with the positve images #####
    posImageFilenames = [f for f in listdir(DIR_POS_IMAGES) if path.isfile(path.join(DIR_POS_IMAGES,f))]
    
    ##### Creating the samples ######
    if(createSamples):
        SampleCollectionFile = open(DIR_SAMPLE_IMAGES + "/"+ "sample.lst", "w")
        count = 1
        for f in posImageFilenames:
            file = "sample-" + str(count) + ".lst"
            check_call(["opencv_createsamples",
                "-img", path.join(DIR_POS_IMAGES,f),
                "-bg", 'neg.txt',
                "-info", path.join(DIR_SAMPLE_IMAGES, file),
                "-num", str(round(250*0.8)),
                "-maxxangle", "0.0",
                "-maxyangle", "0.0",
                "-maxzangle", "0.3",
                "-bgcolor", "255",
                "-bgthresh", "8",
                "-w", "40",
                "-h", "40"])
            with open(DIR_SAMPLE_IMAGES + "/" + file, "r") as ss:
                for line in ss:
                    SampleCollectionFile.write(line)
                    #SampleCollectionFile.write("\n")
            count += 1
            remove(DIR_SAMPLE_IMAGES + "/" + file)
        SampleCollectionFile.close()
    
    ##### Vectorizing all sample images #####
    if(createVec):
         check_call(["opencv_createsamples",
            "-info", DIR_SAMPLE_IMAGES + "/sample.lst",
            "-vec", FILE_SAMPLE_VEC,
            "-bg", "neg.txt",
            "-num", str(sum(1 for line in open(DIR_SAMPLE_IMAGES + "/sample.lst", "r"))),
            "-w", "40",
            "-h", "40"])
         sleep(1)
    
    ##### Training the Haar Cascades #####
    if(trainCascade):
        if(not path.isdir(DIR_HAAR_DATA)):
            makedirs(DIR_HAAR_DATA)

        check_call(["opencv_traincascade",
            "-data", DIR_HAAR_DATA,
            "-vec", FILE_SAMPLE_VEC,
            "-bg", "neg.txt",
            "-numPos", str(max(50, sum(1 for line in open(DIR_SAMPLE_IMAGES + "/sample.lst", "r")))/2),
            "-numNeg", str(max(50, sum(1 for line in open(DIR_SAMPLE_IMAGES + "/sample.lst", "r")))/4),
            "-numStages", "10",
            "-precalcValBufSize", "1024",
            "-precalcIdxBufSize", "1024",
            "-featureType", "HAAR",
            "-w", "40",
            "-h", "40",
            "-minHitRate", "0.995",
            "-maxFalseAlarmRate", "0.5"])
        sleep(1)