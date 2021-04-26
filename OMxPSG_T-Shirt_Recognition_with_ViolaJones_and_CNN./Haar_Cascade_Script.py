#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:38:52 2021

@author: igor
"""


####### Importing Libraries ######
from getopt import getopt
from sys import argv
from os import listdir, path, makedirs, remove
from re import sub
from subprocess import check_call
from time import sleep
from shutil import rmtree, copytree, copy
import cv2 as cv

######## Defining the directories for the original images ########

POS_IMG = "dataset/orig-pos"
NEG_IMG = "dataset/orig-neg"

######## Defining the directories for the training sets ########

DIR_PROCESSING_DATA = "dataset"
DIR_POS_IMAGES = "dataset/pos"
DIR_POS_IMAGES_CROPPED = DIR_PROCESSING_DATA+"/positive-clean-cropped"
DIR_DATA = "dataset"
DIR_NEG_IMAGES = "dataset/neg"
DIR_SAMPLE_IMAGES = DIR_DATA+"/positive-clean-cropped-samples"
DIR_HAAR_DATA = DIR_DATA+"/clean-cropped-haardata"

FILE_NEG_COLLECTION = DIR_NEG_IMAGES+"/"+DIR_NEG_IMAGES.split("/")[-1]+".txt"
FILE_SAMPLE_COLLECTION = DIR_SAMPLE_IMAGES+"/"+DIR_SAMPLE_IMAGES.split("/")[-1]+".txt"
FILE_SAMPLE_VEC = DIR_SAMPLE_IMAGES+".vec"

####### Starting the script for the full training process ########

if __name__=="__main__":
    (createSamples, createVec, trainCascade) = (False, False, False)
    opts, args = getopt(argv[1:],"svt",["createSamples","createVec","trainCascade"])
    for opt, arg in opts:
        if(opt in ("--createSamples","-s")):
            createSamples = True
        elif(opt in ("--createVec","-v")):
            createVec = True
        elif(opt in ("--trainCascade","-t")):
            trainCascade = True
    
    ######### Resizing images and changing image colors #########
    
    ##### Negatives Images ######
    def resizeNegImages():
        
        imagePath = listdir(NEG_IMG)
        identify = 1
        if not path.isdir(DIR_NEG_IMAGES):
            makedirs(DIR_NEG_IMAGES)
        for i in imagePath:
            
            i.replace(i, DIR_NEG_IMAGES + "/" + str(identify) + ".JPEG")
            copy(NEG_IMG + "/" + i, i.replace(i, DIR_NEG_IMAGES + "/" + str(identify) + ".JPEG"))
            img = cv.imread(DIR_NEG_IMAGES + "/" + str(identify) + ".JPEG", cv.IMREAD_GRAYSCALE)
            resize = cv.resize(img, (100, 100))
            cv.imwrite(DIR_NEG_IMAGES + "/" + str(identify) + ".JPEG", resize)
            
            identify += 1
        return identify
    nun_neg = resizeNegImages()
    
    ##### Positive Images #####
    def resizePosImages():
        
        imagePath = listdir(POS_IMG)
        identify = 1
        
        if not path.isdir(DIR_POS_IMAGES):
            makedirs(DIR_POS_IMAGES)
        for i in imagePath:
            
            i.replace(i, DIR_POS_IMAGES + "/" + str(identify) + ".JPEG")
            copy(POS_IMG + "/" + i, i.replace(i, DIR_POS_IMAGES + "/" + str(identify) + ".JPEG"))
            img = cv.imread(DIR_POS_IMAGES + "/" + str(identify) + ".JPEG", cv.IMREAD_GRAYSCALE)
            resize = cv.resize(img, (100, 100))
            cv.imwrite(DIR_POS_IMAGES + "/" + str(identify) + ".JPEG", resize)
            
            identify += 1
        return identify
    nun_pos = resizePosImages()
   
    ##### To be sure that we have the output directory #####
    if(not path.isdir(DIR_SAMPLE_IMAGES)):
        makedirs(DIR_SAMPLE_IMAGES)
    
    ##### Creating the background file ######
    negImageCollectionFile = open("neg.txt", "w")
    negImageFilenames = [f for f in listdir(DIR_NEG_IMAGES) if path.isfile(path.join(DIR_NEG_IMAGES,f)) and f.endswith("JPEG")]
    for f in negImageFilenames:
        negImageCollectionFile.write("/home/igor/Documents/Artificial Inteligence/Deep-Learning/Artificial Intelligence Formation/4 - Computer Vision/Haar Cascade/dataset/neg/" + f)
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
                "-num", str(round(1600*0.8)),
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
            #"-numPos", str(50),
            "-numNeg", str(max(50, sum(1 for line in open(DIR_SAMPLE_IMAGES + "/sample.lst", "r")))/4),
            "-numStages", "20",
            "-precalcValBufSize", "1024",
            "-precalcIdxBufSize", "1024",
            "-featureType", "HAAR",
            "-w", "40",
            "-h", "40",
            "-minHitRate", "0.995",
            "-maxFalseAlarmRate", "0.5"])
        sleep(1)