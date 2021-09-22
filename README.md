# OM x PSG T-Shirt Detection

## Overview

This project intends to build three different approaches for object detection. The objects to be detected are football jerseys (t-shirts) of the two most popular French teams, the Olympique de Marseille and the Paris Saint-Germain.


**About the project**: The idea behind this project started with my training on Computer Vision (CV). In the beginning of my Artificial Intelligence journey, I wanted deploy a model of Convolutional Neural Network (CNN) to predict images from the both French teams. But, in this moment, I didn't have enough knowledge about CV. I’ve postulated for a job, to work with CV and, I presented this idea as personal project. The experience was a little bite different I had imagined. When I was asked about image pre-processing, about labelization and especially if, my model was capable to detect object with video stream, I didn’t have the good answers.

After this experience, I was demotivated (I imagined that the classification model was enough), but I decided to go into a deep formation on CV and fill the empty blanks. During this process, I've studied many aspects about image preprocessing, CNN architectures (for classification and detection tasks) and OpenCV skills. The results presented here is a part of my studies and a nice little reward.

As said before, this project contains three different approaches, each of them corresponds to a stage of my training on CV. The folders to each approach are organized as:

- [OM x PSG T-Shirt Recognition with Viola-Jones algorithm and CNN.](https://github.com/IgorMeloS/OMxPSG-T-Shirt-Recognition/tree/main/OMxPSG_T-Shirt_Recognition_with_ViolaJones_and_CNN.)
  - The first attempt to detect the t-shirt was made with two methods. We consider the Viola-Jones algorithm to extract the Region of Interest (ROI), in this case any t-shirt. For the classification of OM and PSG t-shirts, we consider the principle of transfer learning to train the CNN. Once we have the two models, we merge them. Inside the ROI obtained with the Viola-Jones algorithm, we predict the CNN model for the two classes. The result was HORRIBLE !!!!
- [OM x PSG T-Shirt Recognition with SSD.](https://github.com/IgorMeloS/OMxPSG-T-Shirt-Recognition/tree/main/OMxPSG_T-Shirt_Recognition_with_ViolaJones_and_CNN.)
  - The second approach was successful. Using the TensorFlow Object Detection API, we deploy the Single Shot Detection (SSD), using the pre-trained model.
- [OM x PSG T-Shirt Recognition with YOLO.](https://github.com/IgorMeloS/OMxPSG-T-Shirt-Recognition/tree/main/OMxPSG_T-Shirt_Recognition_with_ViolaJones_and_CNN.)
  - The last approach was so successful as the second. Considering the YOLO version 5 from the Ultralytics, here we use PyTorch. I've trained the model with the weight from the COCO dataset.

**A more detailed explanation about each method and the metrics results can be found inside each above folder.**

![OM x PSG T-Shirt Recognition!](Image/om-psg-classico.jpg "OM x PSG")

## Motivation

Football (soccer) the most popular sport around the world, capable to induce several sentiments as happiness or angry. Everyone, once time in your life have watched a football match. Some people might have a favorite team or not, but the fact is, we play football on the five continents, and your world cup is a big party of the sport. This sport played by 22 soccer players inside four lines, became a worldwide passion. On the other hand, we cannot to forget, the modern football is also a market environment with financial transactions and strategies. For many reasons, football is always a present subject in our life, we found it by the TV, radio, internet, friends, neighbors, and others sources. Soccer is everywhere.

The rivalry is another interesting feature of the soccer. In the football world, we can see many rivalries developed during your long history. A derby, for example, is a match between two clubs knew by your rivalry. For many football lovers, the derby is the most important match in the season, because there are others things beyond the sport, the pride of the club and fans. Around the world we can to found rivalries between countries as Brazil x Argentine. Between football clubs from different cities as Barcelona x Real Madrid, clubs from the same city as Manchester United x Manchester City. But also, the rivalries are present for the college, neighborhoods, and friends teams.

In France, it couldn’t be different, the rivalry is present in the hexagon. The well-known “Le classique” has place every year, figured by the two most popular French teams, the Olympique de Marseille (OM) and the Paris Saint-Germain (PSG). The Olympique de Marseille is the most popular team in France, in your list of titles contains a Champions League, the most important competition for the football clubs. In your long existence, the club of the south spread across France due to your glorious history, the gold star above your logo has an important weight for the club.

The PSG, on the other hand, is the richest French team. The club wants to enter the hall of great teams of Europe. Always presenting good performances at international level, the PSG did never reach your main goal, become a European champion. Every year, the PSG is faced to the international failure. Regardless of your constant European failure, the club of the capital dominates the football at national level. The PSG became the second most popular French club, due to a considerable number of star players, a strong advertising campaign, and evidently, a good football.

Motivated by this French rivalry, OM x PSG, I want to bring it to the world of computer vision. Several applications can be deployed with computer vision, as object recognition, image segmentation, for example. We can employ it to many fields, industry, medicine, traffic control, agriculture. (To fill).

## Dataset
