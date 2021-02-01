
import numpy as np
import pickle as pkl
import tensorflow as tf
from utils import visualize_detections
from data_preprocess import resize_and_pad_image
from retinanet import get_backbone, RetinaNet, DecodePredictions

# Custom function to load image. #
# Custom function to parse the data. #
def _parse_image(filename):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    return tf.cast(image_decoded, tf.float32)

def prepare_image(image, preprocess_input=None):
    image, _, ratio = resize_and_pad_image(
        image, min_side=384.0, max_side=384.0, jitter=None)
    
    if preprocess_input is None:
        image = tf.keras.applications.resnet.preprocess_input(image)
    else:
        image = preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

# Load the data. #
tmp_path = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/"
load_pkl_file = tmp_path + "voc_data.pkl"
with open(load_pkl_file, "rb") as tmp_load:
    id_2_label  = pkl.load(tmp_load)
    voc_dataset = pkl.load(tmp_load)

# Build model. #
backbone_name = "mobilenetv2"
if backbone_name == "mobilenetv2":
    preprocess_input = \
        tf.keras.applications.mobilenet_v2.preprocess_input
else:
    preprocess_input = None

box_dims = [float(300 / (2**x)) for x in range(5)]
box_dims = box_dims[::-1]

num_classes = len(id_2_label)
mobilenet_backbone = get_backbone(model_name=backbone_name)

retinanet_model = RetinaNet(
    num_classes, mobilenet_backbone)
model_optimizer = tf.optimizers.SGD(
    learning_rate=1.0e-3, momentum=0.9)
model_decoder = DecodePredictions(
    confidence_threshold=0.10, box_dims=box_dims)

# Loading weights. #
voc_path = "C:/Users/admin/Desktop/TF_Models/voc_2012_model/"
ckpt_model = voc_path + "voc_retinanet_mobilenetv2"

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    retinanet_model=retinanet_model, 
    model_optimizer=model_optimizer)
ck_manager = tf.train.CheckpointManager(
    checkpoint, directory=ckpt_model, max_to_keep=1)

checkpoint.restore(ck_manager.latest_checkpoint)
if ck_manager.latest_checkpoint:
    print("Model restored from {}".format(
        ck_manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")
st_step = checkpoint.step.numpy().astype(np.int32)

# Building inference model. #
image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = retinanet_model(image, training=False)
detections  = model_decoder(image, predictions)

inference_model = tf.keras.Model(
    inputs=image, outputs=detections)

# Generating detections. #
print("Testing Model", "(" + str(st_step), "iterations).")

image_file = "kite.jpg"
raw_image  = _parse_image(image_file)
input_image, ratio = prepare_image(
    raw_image, preprocess_input=preprocess_input)

tmp_detect = inference_model.predict(input_image)
n_detected = tmp_detect[3][0]
class_names = [id_2_label[
    int(x)] for x in tmp_detect[2][0][:n_detected]]

visualize_detections(
    raw_image, 
    tmp_detect[0][0][:n_detected] / ratio,
    class_names, tmp_detect[1][0][:n_detected])
print(str(n_detected), "objects detected.")