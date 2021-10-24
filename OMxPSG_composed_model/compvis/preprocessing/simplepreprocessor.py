# =============================================================================
# Simple image preprocess
# Class SimplePreprocessor
# Atribute preprocess - resizes image ignoring the aspect ratio
# =============================================================================

import cv2 as cv

class SimplePreprocessor:
    """Simple image preprocessor.
    Args:
        width: width to resize the image.
        height: height to resize the image.
        inter: interpolation method. By default cv2.INTER_AREA
    """
    def __init__(self, width, height, inter=cv.INTER_AREA):
        
        # Local variables for the class and its functions
        self.width = width
        self.height = height
        self.inter = inter # interpolation method
        
    def preprocess(self, image):
        """Preprocess function. This function resizes the image without considering the aspect ratio.
        Args:
            image: the path of the image to be resized.
        return: this function returns the resized image.
        """
        img = cv.resize(image, (self.width, self.height), interpolation=self.inter) # object to resize
        return img