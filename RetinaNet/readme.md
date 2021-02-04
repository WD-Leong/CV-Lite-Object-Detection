# RetinaNet
This repository contains the implementation of [RetinaNet](https://arxiv.org/abs/1708.02002) as outlined [here](https://keras.io/examples/vision/retinanet/). Currently, two backbones are supported - (i) MobileNet v2 and (ii) ResNet-50.

## Formatting the Data
To format the data, run
```
python format_VOC_annotations_retinanet.py
```

## Training the Model
To train the model, run
```
python train_retinanet.py
```

## Inference
To infer using the trained model, run
```
python infer_retinanet.py
```

Please note that this is still Work-in-Progress.
