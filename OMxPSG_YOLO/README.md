# OM x PSG T-Shirt Detection with YOLOv5

## Installing yolov5

```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```

## Dataset

The dataset for the object detection come from the original [dataset](https://github.com/IgorMeloS/OMxPSG-T-Shirt-Detection/tree/main/dataset), if you want to train the model by yourself, you must select the images for your training.

This project considers 320 of each team. Example of images for detection task.
![Image for object detection!](images/OM-484 "OMxPSG")
![Image for object detection!](images/PSG-1152 "OMxPSG")

## Annotation

The annotations was made with [LabelImg](https://github.com/tzutalin/labelImg). The output is a txt file, that's the appropriate format to train YOLO models. The information about annotation look sound this

```
0 0.5078125 0.5765625 0.57421875 0.640625
```

Here, we can find two example of annotation files ([1](https://github.com/IgorMeloS/OMxPSG-T-Shirt-Detection/blob/main/OMxPSG_YOLO/images/OM-484.txt) and [2](https://github.com/IgorMeloS/OMxPSG-T-Shirt-Detection/blob/main/OMxPSG_YOLO/images/PSG-1152.txt)).
