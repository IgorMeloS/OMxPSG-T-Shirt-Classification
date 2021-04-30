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
HAAR_DATA = MAIN_DATA_DIR + "/haar-cascade-data" # Folder to store the Haar cascade and its training stages and parameters

# =============================================================================
# Defining some important files as background, sample and vector.
# =============================================================================

NEG_COLLECTION = TR_NEG_IMAGES + "/" + TR_NEG_IMAGES.split("/")[-1]+".txt" # Negative images list
ANOT = SAMPLE_IMAGES + "/"+ "sample.lst"# Annotation file for the positive images with their background.
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
    
    ##### Defining the function to process all negative images ######
    
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
        return identify # this returns the number of negative images
            
    neg_num = resizeNegImages() # applying the function 
    
   ##### Defining the function to process all positive images ######
    
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
   
    # =============================================================================
    # Creating the background file.
    # - The list with all absolute path for each negative image.
    # =============================================================================
    
    bgFile = open(NEG_COLLECTION , "w") # Creating the file
    listNegimg = [f for f in listdir(TR_NEG_IMAGES) if path.isfile(path.join(TR_NEG_IMAGES,f)) and f.endswith("JPEG")] # Creating the list with negative images
    
    for f in listNegimg:
        bgFile.write(f)
        bgFile.write("\n")
    bgFile.close()
   
    # =============================================================================
    # STEP 1
    # Creating the sample with the annotated images
    # - Sample list for the positive images annotated in their backgrounds.
    # - Each positive image is annotated in a certain number of negative images.
    # - For each positive image there is an associated file with the annotations.
    # - We merge all files into one single file, a big list with the annotated 
    #   images.
    # =============================================================================
    
    if(createSamples):
        
        if(not path.isdir(SAMPLE_IMAGES)):
            makedirs(SAMPLE_IMAGES) # To create the folder to store the sample of cropped images, in the case that there is not the specific folder
            
        listPosimg = [f for f in listdir(TR_POS_IMAGES) if path.isfile(path.join(TR_POS_IMAGES,f))] # To create a list with the name of all positive images   
        AnotFile = open(ANOT, "w") # Creating the the file to write all annotations
        count = 1 # counter for each positive image
        
        for f in listPosimg:
            file = "sample-" + str(count) + ".lst" # To create the file to write the annotation for each positive image
            check_call(["opencv_createsamples", # OpenCV command to create the samples
                "-img", path.join(TR_POS_IMAGES,f), # Directory for each positive image
                "-bg", NEG_COLLECTION, # The background file
                "-info", path.join(SAMPLE_IMAGES, file), # File for each image with their annotations
                "-num", str(round(neg_num*0.8)), # Number of negative image considered as background for the current positve image
                "-maxxangle", "0.0", # rotation in x axis
                "-maxyangle", "0.0", # rotation in y axis
                "-maxzangle", "0.3", # rotation in z axis
                "-bgcolor", "255", # limit of BGC color
                "-bgthresh", "8", # Limit for the background
                "-w", "40", # width for the cascade
                "-h", "40"]) # height for the cascade
            with open(SAMPLE_IMAGES + "/" + file, "r") as ss: # Opening the annotation file for the current image
                for line in ss: # loop for to read all lines inside the annotation file for the current image
                    AnotFile.write(line) # writting the annotations inside the big list of annotation
            count += 1 # increasing the counte
            remove(SAMPLE_IMAGES + "/" + file) # removing the annotation file for the current image
        AnotFile.close() # closing the big list of annotation.
    
    # =============================================================================
    # STEP 2
    # Vectorizing all the positive images
    # =============================================================================
    
    if(createVec):
        
         check_call(["opencv_createsamples",
            "-info", ANOT, # Calling the big list with the annotations
            "-vec", SAMPLE_VEC, # File to write the vectorized images
            "-bg", NEG_COLLECTION, # Background file
            "-num", str(sum(1 for line in open(ANOT, "r"))), # The number of annotated images
            "-w", "40",
            "-h", "40"])
         sleep(1)        
        
    # =============================================================================
    # STEP 3
    # Training the Haar Cascade
    # =============================================================================
    
    if(trainCascade):
        
        if(not path.isdir(HAAR_DATA)):
            makedirs(HAAR_DATA) # To create the folder to store the Haar parameters in xml format, in the case that there is not the specific folder

        check_call(["opencv_traincascade", # OpenCV command
            "-data", HAAR_DATA, # directory to save the haar cascade
            "-vec", SAMPLE_VEC, # The vectorized file
            "-bg", NEG_COLLECTION, # The background file
            "-numPos", str(max(50, sum(1 for line in open(DIR_SAMPLE_IMAGES + "/sample.lst", "r")))/2), # Number of positive samples used in training for every classifier stage.
            "-numNeg", str(max(50, sum(1 for line in open(DIR_SAMPLE_IMAGES + "/sample.lst", "r")))/4), # Number of negative samples used in training for every classifier stage.
            "-numStages", "20", # Number of cascade stages to be trained.
            "-precalcValBufSize", "1024", # Size of buffer for precalculated feature values (in Mb)
            "-precalcIdxBufSize", "1024", # Size of buffer for precalculated feature indices (in Mb). 
            "-featureType", "HAAR", # Type of features: HAAR - Haar-like features, LBP - local binary patterns.
            "-w", "40",
            "-h", "40",
            "-minHitRate", "0.995", # Minimal desired hit rate for each stage of the classifier
            "-maxFalseAlarmRate", "0.5"]) # Maximal desired false alarm rate for each stage of the classifier.
        sleep(1)