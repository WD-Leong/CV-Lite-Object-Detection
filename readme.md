# Custom Object Detection

This repository contains my experiments with custom object detection models whose weights are trained from scratch, that is to say, no pre-trained backbone model is used. The model is relatively small and trained on the [VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset. To run the object detection, first run
```
python process_voc_object_detection.py
```
followed by
```
python voc_object_detection.py
```
to train the object detector.

Some detection results are shown below.

Sample Result 1:
![Result 12](obj_detection_result_12.jpg)

Sample Result 2:
![Result 24](obj_detection_result_24.jpg)

Detection Result with Heatmap (black boxes indicate ground truth while red boxes indicate the predicted bounding box):
![Result with Heatmap](obj_detection_result.jpg)


