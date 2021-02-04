
import numpy as np
import pickle as pkl
import tensorflow as tf
from matplotlib import pyplot as plt
from utils import swap_xy, visualize_detections
from fcos import build_model, prediction_to_corners

# Custom function to load image. #
# Custom function to parse the data. #
def _parse_image(filename):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    return tf.cast(image_decoded, tf.float32)

def prepare_image(image, img_w=384, img_h=384):
    img_dims = [int(image.shape[0]), 
                int(image.shape[1])]
    w_ratio  = img_dims[0] / img_w
    h_ratio  = img_dims[1] / img_h
    
    img_resized = tf.image.resize(image, [img_w, img_h])
    img_resized = img_resized / 127.5 - 1.0
    return tf.expand_dims(img_resized, axis=0), w_ratio, h_ratio

def image_detections(
    image, model, num_classes, center=False, 
    iou_thresh=0.5, cls_thresh=0.05, 
    max_detections=100, max_total_size=100):
    strides = [8, 16, 32, 64, 128]
    
    tmp_predict = model(image, training=False)
    tmp_outputs = []
    for n_layer in range(len(tmp_predict)):
        tmp_output = tmp_predict[n_layer]
        
        processed_output = tmp_output.numpy()[0]
        processed_bboxes = prediction_to_corners(
            processed_output[..., :4], strides[n_layer])
        
        processed_output[..., :4] = processed_bboxes
        tmp_outputs.append(
            processed_output.reshape(-1, num_classes+5))
    
    tmp_outputs = np.concatenate(tmp_outputs, axis=0)
    tmp_boxes   = np.expand_dims(
        np.expand_dims(tmp_outputs[:, :4], axis=1), axis=0)
    
    if center:
        cen_scores  = tf.nn.sigmoid(tmp_outputs[:, 4])
        tmp_scores  = np.expand_dims(np.multiply(
            np.expand_dims(cen_scores, axis=1), 
            tf.nn.sigmoid(tmp_outputs[:, 5:])), axis=0)
    else:
        tmp_scores  = np.expand_dims(
            tf.nn.sigmoid(tmp_outputs[:, 5:]), axis=0)
    tmp_detect  = tf.image.combined_non_max_suppression(
        tmp_boxes, tmp_scores, max_detections, 
        max_total_size=max_total_size, clip_boxes=False, 
        iou_threshold=iou_thresh, score_threshold=cls_thresh)
    return tmp_detect

def detect_heatmap(
    image, model, center=True, img_rows=384, img_cols=384):
    img_w = int(image.shape[0])
    img_h = int(image.shape[1])
    strides = [8, 16, 32, 64, 128]
    
    img_resized = tf.image.resize(
        image, [img_rows, img_cols])
    img_resized = tf.expand_dims(img_resized, axis=0)
    img_resized = img_resized / 127.5 - 1.0
    tmp_predict = model(img_resized, training=False)
    
    tmp_heatmap = []
    for n_layer in range(len(tmp_predict)):
        stride = strides[n_layer]
        tmp_array = np.zeros(
            [int(img_rows/8), int(img_cols/8)])
        
        tmp_output = tmp_predict[n_layer]
        tmp_output = tmp_output.numpy()[0]
        cls_output = tf.nn.sigmoid(tmp_output[..., 4:])
        
        max_probs  = tf.reduce_max(cls_output, axis=2)
        down_scale = int(stride / 8)
        
        if center:
            cen_output = tf.nn.sigmoid(tmp_output[..., 4])
            tmp_array[0::down_scale, 0::down_scale] = \
                np.multiply(cen_output, max_probs)
        else:
            tmp_array[0::down_scale, 
                      0::down_scale] = max_probs
        
        tmp_array = tf.expand_dims(tmp_array, axis=2)
        obj_probs = tf.image.resize(
            tf.expand_dims(tmp_array, axis=0), [img_w, img_h])
        obj_probs = tf.squeeze(obj_probs, axis=3)
        tmp_heatmap.append(obj_probs)
    
    tmp_heatmap = tf.concat(tmp_heatmap, axis=0)
    tmp_heatmap = tf.reduce_max(tmp_heatmap, axis=0)
    
    fig, ax = plt.subplots(1)
    tmp_img = np.array(image, dtype=np.uint8)
    ax.imshow(tmp_img)
    tmp = ax.imshow(tmp_heatmap, "jet", alpha=0.50)
    fig.colorbar(tmp, ax=ax)
    
    fig.suptitle("Detection Heatmap")
    fig.savefig("heatmap.jpg", dpi=199)
    plt.close()
    del fig, ax
    return None

# Load the data. #
tmp_path = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/"
load_pkl_file = tmp_path + "voc_data.pkl"
with open(load_pkl_file, "rb") as tmp_load:
    id_2_label  = pkl.load(tmp_load)
    voc_dataset = pkl.load(tmp_load)

# Build model. #
backbone_name = "mobilenetv2"
num_classes = len(id_2_label)

fcos_model = build_model(
    num_classes, backbone_model="mobilenetv2")
model_optimizer = tf.optimizers.Adam()

# Loading weights. #
voc_path = "C:/Users/admin/Desktop/TF_Models/voc_2012_model/"
ckpt_model = voc_path + "voc_fcos_mobilenetv2"

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    fcos_model=fcos_model, 
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

# Generating detections. #
print("Testing Model", "(" + str(st_step), "iterations).")
cls_thresh = 0.25
iou_thresh = 0.50

#image_file = "test_image.jpg"
image_file = voc_dataset[10]["image"]
raw_image  = _parse_image(image_file)
input_image, w_ratio, h_ratio = \
    prepare_image(raw_image, img_w=384, img_h=384)

tmp_detect = image_detections(
    input_image, fcos_model, num_classes, 
    cls_thresh=cls_thresh, iou_thresh=iou_thresh)

n_detected  = tmp_detect[3][0]
bbox_ratio  = np.array(
    [w_ratio, h_ratio, w_ratio, h_ratio])
bbox_detect = tmp_detect[0][0][:n_detected] * bbox_ratio
class_names = [id_2_label[
    int(x)] for x in tmp_detect[2][0][:n_detected]]
print(class_names, bbox_detect.numpy())

detect_heatmap(raw_image, fcos_model)
visualize_detections(
    raw_image, swap_xy(bbox_detect), 
    class_names, tmp_detect[1][0][:n_detected])
print("Total of", str(n_detected.numpy()), "objects detected.")

