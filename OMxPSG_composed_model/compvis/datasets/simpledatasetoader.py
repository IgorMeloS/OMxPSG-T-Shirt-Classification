# =============================================================================
# Image Loader class
# Load images and process them, if required
# =============================================================================

import numpy as np
import cv2 as cv
import os

class SimpleDatasetLoader:
    """Simple dataset load. This class load your data from disk.
    Argument:
            preprocessor: list of callable image preprocessor (resize, image_to_array etc...).
            If any preprocessor function is called, it will be applyed to the image to be returned.
    """
    def __init__(self, preprocessors=None):
        # defining local variables
        # calling an image prepocessor
        self.preprocessors = preprocessors
        
        if self.preprocessors is None:
            #If there is not image preprocessor, make an empty list
            self.preprocessors = []
        # defining the function to load your set of images
       
    def load(self, imagePaths, verbose = -1):
        """Function to load the images from disk.
        Args:
            imagePaths: path to load the image from disk
            verbose: interger number, by default -1.
        return: this function returns two arrays that contain the list of image features and labels.
        """
        # the variable imagePaths tell to us where the dataset is located
        data = [] # list of data points (images)
        labels = [] # list of labels
            
        for (i, imagePath) in enumerate(imagePaths):
            # Function to read image by image
                
            image = cv.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # The variable label is set for the case that you've organized
            # your dataset in the follow way
            # Dataset/class/img.jpg
                
            if self.preprocessors is not None:
                # processing the image with Preprocessor class
                for p in self.preprocessors:
                    image = p.preprocess(image)
                  
            data.append(image) # Appending the image into vector of features
            labels.append(label) # list of labels
                
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i +1, len(imagePaths)))
        # Returning a tuple with the images and labels
        return (np.array(data), np.array(labels))