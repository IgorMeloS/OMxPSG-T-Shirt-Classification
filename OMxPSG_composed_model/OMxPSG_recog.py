#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:30:34 2021

@author: igor
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:07:41 2021

@author: igor
"""

# Importing the library

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

#Loading the pre-trained cascade method for faces and eyes
tshirt = cv2.CascadeClassifier('cascade.xml') #Object to read the pre-trained frontal face detectp
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #Object to read the pre-trained eyes detector
model = load_model("tune.h5")



def detect(gray, frame):
   tshirts =  tshirt.detectMultiScale(gray, 5.0, 120, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

   for (x, y, w, h) in tshirts:
       cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = frame[y:y+h, x:x+w]
       roi_gray = cv2.resize(roi_gray, (160, 160))
       roi_gray = roi_gray.astype("float") / 255.0
       roi_gray = img_to_array(roi_gray)
       #roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
       roi_gray = np.expand_dims(roi_gray, axis=0)  
       roi_color = cv2.resize(roi_color, (160, 160))
       roi_color = roi_color.astype("float") / 255.0
       roi_gray = img_to_array(roi_color)
       roi_color = np.expand_dims(roi_color, axis=0)
       pred = model.predict(roi_color)
       label = "PSG" if pred > 0.5 else "OM"
       
       cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
   return frame
    
    

# Detecting faces with webcam

video_capture = cv2.VideoCapture(0)

while True:
    
    _, frame = video_capture.read()
    
    #frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameClone = frame.copy()
    canvas = detect(gray, frameClone)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()