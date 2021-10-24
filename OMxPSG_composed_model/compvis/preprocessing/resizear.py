# =============================================================================
# Aspect-Aware preprocessing
# =============================================================================

# Importing the libraries
import imutils
import cv2

# Creating the class
class ResizeAR:
    """Image resizing conserving the aspect ratio proportional to the height or width.
        Args:
            width: new image width.
            height: new image height.
            inter: Any interpolation method from OpenCV.
    """
    def __init__(self, width, height, inter = cv2.INTER_AREA):
        # storing the variables for the resizing proprocess
        self.width = width
        self.height = height
        self.inter = inter
    # Building the process functions
    def preprocess(self, image):
        """Preprocess fucntion to extratct the dimensions
        of each image, crop and resize them conserving the aspect ratio.
        Args:
            image: image path.
        >>>
            NOTE: To resize the images, this atribute considers the lower dimension to be zoomed and the
            highest to be cropped.
                If width > height the ratio considered is dw = (image width(input) - width (desired width))/2.
                If height > width the ratio considered is dh = (image height(input) - height(desired height))/2.
        """
        # Extrating the the image dimensions and the deltas to cropp
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0
        # =============================================================================
        # To resize the images, we consider the lower dimension to be zoomed and the
        # highest to be cropped. We start with the if condition to verify the dimensions        
        # =============================================================================
        if w < h:
            image = imutils.resize(image, width = self.width, inter = self.inter)
            dH = int((image.shape[0] - self.height)/2.0)
        else:
            image = imutils.resize(image, height = self.height, inter = self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)
        # Executing the cropping process with the deltas
        # re-grabbing the coordinates after the first transfomartion
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]
        
        # finilly we resize to ensure the spatial dimensions
        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)
