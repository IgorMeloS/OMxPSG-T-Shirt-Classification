# OM x PSG T-shirt recognition using Viola Jones algorithm and CNN classifier.

This method is composed by two models, one model detects t-shirt and extracts the ROI (region of interest). The second model is classification model trained with transfer learning.
The classification is made inside the ROI, the classes to be predict are OM and PSG. We build this approach with the follow steps

## Step1
Training the Haar cascade classifier
## `Haar_Cascade_Script.py`
To run this code `python3 Haar_Cascade_Script.py -svt`

The training of the Haar requires two set of images, positive (contains objects to be detected) and negative (images without the object). The training is composed by three stages

- create a sample of images (create images with positive images over a negative (background))
- create a vector of features
- training the model (to this project the model was trained over 20 step train)

## Step 2
Testing the Haar cascade with Viola-Jones algorithm
## `Testing_cascade.ipynb`

## Step 3
Training the classification model with fine tuning. The backbone was VGG16 with the weights from ImageNet
##`OMxPSG_CNN_class.ipynb`

## Step 4
Build a the t-shirt detector merging Viola-Jones algorithm and the classification
##`OMxPSG_tshirt_detect.ipynb`
