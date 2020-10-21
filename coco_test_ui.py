import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st
import tensorflow as tf
import tf_object_detection as tf_obj_detector

from PIL import Image
from matplotlib import pyplot as plt

# File Selector. #
def image_selector(folder_path="."):
    filenames = [x for x in os.listdir(folder_path) \
                 if str(x).endswith("jpg") or str(x).endswith("jpeg")]
    sel_image = st.sidebar.selectbox("Select an image", filenames)
    print(os.path.join(folder_path, sel_image))
    return os.path.join(folder_path, sel_image)

# Load the data. #
def load_data():
    load_pkl_file = "C:/Users/admin/Desktop/Data/COCO/"
    load_pkl_file += "coco_annotations_320.pkl"
    with open(load_pkl_file, "rb") as tmp_load:
        img_scale = pkl.load(tmp_load)
        coco_obj_list = pkl.load(tmp_load)
    
    # Remember to add 1 more class for background. #
    coco_label = pd.read_csv(
        "C:/Users/admin/Desktop/Data/COCO/labels.csv")
    list_label = sorted([
        coco_label.iloc[x]["name"] \
        for x in range(len(coco_label))])
    list_label = ["background"] + list_label
    label_dict = dict([(
        x, list_label[x]) for x in range(len(list_label))])
    index_dict = dict([(
        list_label[x], x) for x in range(len(list_label))])
    return img_scale, coco_obj_list, label_dict, index_dict

# Load the COCO model. #
def load_coco_model():
    # Load the weights if continuing from a previous checkpoint. #
    coco_model = tf_obj_detector.build_model(
        n_filters, n_classes, tmp_pi=0.95, 
        img_rows=img_width, img_cols=img_height, seperable=False)
    optimizer = tf.keras.optimizers.Adam()
    
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0), 
        coco_model=coco_model, 
        optimizer=optimizer)
    ck_manager = tf.train.CheckpointManager(
        checkpoint, directory=ckpt_model, max_to_keep=1)
    
    checkpoint.restore(ck_manager.latest_checkpoint)
    if ck_manager.latest_checkpoint:
        print("Model restored from {}".format(
            ck_manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    # Print out the model summary. #
    print(coco_model.summary())
    print("-" * 50)
    return coco_model

# Custom function to parse the data. #
def _parse_image(
    filename, img_rows=320, img_cols=320):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = \
        tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize(
        image_decoded, [img_width, img_height])
    image_resized = tf.ensure_shape(
        image_resized, shape=(img_width, img_height, 3))
    return image_resized

def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    
    boxes1_area = np.multiply(
        boxes1[..., 2] - boxes1[..., 0], 
        boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = np.multiply(
        boxes2[..., 2] - boxes2[..., 0], 
        boxes2[..., 3] - boxes2[..., 1])
    
    left_up   = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_dwn = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_dwn - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    
    ious = np.maximum(
        1.0 * inter_area / union_area, np.finfo(np.float32).eps)
    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, width, height, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    
    tmp_bboxes = bboxes
    tmp_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    tmp_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    
    best_bboxes = []
    for tmp_cls in classes_in_img:
        cls_mask = (tmp_bboxes[:, 5] == tmp_cls)
        cls_bboxes = tmp_bboxes[cls_mask]
        
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou_bboxes = bboxes_iou(
                best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            box_weight = np.ones((len(iou_bboxes),), dtype=np.float32)
            
            assert method in ['nms', 'soft-nms']
            
            if method == 'nms':
                iou_mask = iou_bboxes > iou_threshold
                box_weight[iou_mask] = 0.0
            
            if method == 'soft-nms':
                box_weight = np.exp(-(1.0 * iou_bboxes ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * box_weight
            
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]
    return best_bboxes

def obj_detect_results(
    img_in_file, voc_model, labels, img_box=None, 
    heatmap=True, thresh=0.50, img_scale=None, 
    img_rows=448, img_cols=448, overlap_thresh=0.60, 
    save_img_file="object_detection_result.jpg"):
    if img_scale is None:
        if max(img_rows, img_cols) >= 512:
            max_scale = max(img_rows, img_cols)
        else:
            max_scale = 512
        img_scale = [64, 128, 256, max_scale]
    else:
        if len(img_scale) != 4:
            raise ValueError("img_scale must be size 4.")
    
    # Read the image. #
    image_resized = tf.expand_dims(
        _parse_image(img_in_file, 
                     img_rows=img_rows, 
                     img_cols=img_cols), axis=0)
    
    tmp_output = \
        voc_model.predict(image_resized)
    reg_output = tmp_output[0, :, :, :, :4]
    cls_output = tmp_output[0, :, :, :, 4:]
    cls_probs  = tf.nn.sigmoid(cls_output)
    n_classes  = cls_output.shape[3]
    
    # Plot the bounding boxes on the image. #
    fig, ax = plt.subplots(1)
    tmp_img = np.array(
        Image.open(img_in_file), dtype=np.uint8)
    ax.imshow(tmp_img)
    
    img_width   = tmp_img.shape[0]
    img_height  = tmp_img.shape[1]
    tmp_w_ratio = img_width / img_rows
    tmp_h_ratio = img_height / img_cols
    
    if heatmap:
        if n_classes > 1:
            obj_probs = tf.reduce_max(
                cls_probs[:, :, :, 1:], axis=[2, 3])
        else:
            obj_probs = tf.reduce_max(
                cls_probs[:, :, :, 0], axis=2)
        
        obj_probs = tf.image.resize(tf.expand_dims(
            obj_probs, axis=2), [img_width, img_height])
        tmp = ax.imshow(tf.squeeze(
            obj_probs, axis=2), "jet", alpha=0.50)
        fig.colorbar(tmp, ax=ax)
    
    tmp_obj_detect = []
    for n_sc in range(4):
        if n_sc == 3:
            if max(img_rows, img_cols) <= img_scale[n_sc]:
                box_scale = max(img_rows, img_cols)
            else:
                box_scale = img_scale[n_sc]
        else:
            box_scale = img_scale[n_sc]
        
        if n_classes > 1:
            prob_max = tf.reduce_max(
                cls_probs[:, :, n_sc, 1:], axis=2)
            pred_label = 1 + tf.math.argmax(
                cls_probs[:, :, n_sc, 1:], axis=2)
        else:
            prob_max = cls_probs[:, :, n_sc, 0]
        tmp_thresh = \
            np.where(prob_max >= thresh, 1, 0)
        tmp_coords = np.nonzero(tmp_thresh)
        
        for n_box in range(len(tmp_coords[0])):
            x_coord = tmp_coords[0][n_box]
            y_coord = tmp_coords[1][n_box]
            
            tmp_boxes = reg_output[x_coord, y_coord, n_sc, :]
            tmp_probs = int(
                prob_max[x_coord, y_coord].numpy()*100)
            if n_classes > 1:
                tmp_label = pred_label[x_coord, y_coord].numpy()
            else:
                tmp_label = 0
            
            x_centroid = tmp_w_ratio * (x_coord + tmp_boxes[0])*8
            y_centroid = tmp_h_ratio * (y_coord + tmp_boxes[1])*8
            box_width  = tmp_w_ratio * box_scale * tmp_boxes[2]
            box_height = tmp_h_ratio * box_scale * tmp_boxes[3]
            
            if box_width > img_width:
                box_width = img_width
            if box_height > img_height:
                box_height = img_height
            
            # Output prediction is transposed. #
            x_lower = x_centroid - box_width/2
            y_lower = y_centroid - box_height/2
            if x_lower < 0:
                x_lower = 0
            if y_lower < 0:
                y_lower = 0
            
            tmp_bbox = np.array([
                y_lower, x_lower, 
                box_height, box_width, tmp_probs, tmp_label])
            tmp_obj_detect.append(np.expand_dims(tmp_bbox, axis=0))
        
        bboxes_raw = np.concatenate(
            tuple(tmp_obj_detect), axis=0)
        bboxes_nms = nms(bboxes_raw, 0.213, method='nms')
        for tmp_obj in bboxes_nms:
            box_width  = tmp_obj[2] - tmp_obj[0]
            box_height = tmp_obj[3] - tmp_obj[1]
            box_patch  = plt.Rectangle(
                (tmp_obj[0], tmp_obj[1]), box_width, box_height, 
                linewidth=1, edgecolor="red", fill=None)
            
            tmp_label = str(labels[int(tmp_obj[5])])
            tmp_text  = \
                tmp_label + ": " + str(tmp_obj[4]) + "%"
            ax.add_patch(box_patch)
            ax.text(tmp_obj[0], tmp_obj[1], 
                    tmp_text, fontsize=5, color="red")
    print(str(len(bboxes_nms)), "objects detected.")
    
    # True image is not transposed. #
    if img_box is not None:
        for n_sc in range(4):
            if n_sc == 3:
                if max(img_rows, img_cols) <= img_scale[n_sc]:
                    box_scale = max(img_rows, img_cols)
                else:
                    box_scale = img_scale[n_sc]
            else:
                box_scale = img_scale[n_sc]
            
            tmp_true_box = np.nonzero(img_box[:, :, n_sc, 4])
            for n_box in range(len(tmp_true_box[0])):
                x_coord = tmp_true_box[0][n_box]
                y_coord = tmp_true_box[1][n_box]
                tmp_boxes = img_box[x_coord, y_coord, n_sc, :4]
                
                x_centroid = tmp_w_ratio * (x_coord + tmp_boxes[0])*8
                y_centroid = tmp_h_ratio * (y_coord + tmp_boxes[1])*8
                box_width  = tmp_w_ratio * box_scale * tmp_boxes[2]
                box_height = tmp_h_ratio * box_scale * tmp_boxes[3]
                
                x_lower = x_centroid - box_width/2
                y_lower = y_centroid - box_height/2
                box_patch = plt.Rectangle(
                    (y_lower.numpy(), x_lower.numpy()), 
                    box_height.numpy(), box_width.numpy(), 
                    linewidth=1, edgecolor="black", fill=None)
                ax.add_patch(box_patch)
    return fig

# Load the COCO data. #
print("Loading the file.")
start_time = time.time()
img_scale, coco_obj_list, label_dict, index_dict = load_data()

# Define the Neural Network. #
#img_width  = 448
#img_height = 448
img_width  = 320
img_height = 320
n_filters  = 32

init_lr    = 1.0e-3
cls_loss   = "focal"
decay_rate = 0.95
max_epochs = 100
batch_size = 96
sub_batch  = 12
max_steps  = 1 +\
    int(max_epochs * len(coco_obj_list) / batch_size)
n_classes  = len(label_dict)

display_step = 25
down_width   = int(img_width/8)
down_height  = int(img_height/8)
dense_shape  = [down_height, down_width, 4, n_classes+4]

# Define the checkpoint callback function. #
coco_path   = "C:/Users/admin/Desktop/TF_Models/coco_model/"
if cls_loss == "focal":
    train_loss  = coco_path + "coco_losses_v0.csv"
    ckpt_path   = coco_path + "coco_v0.ckpt"
    ckpt_dir    = os.path.dirname(ckpt_path)
    ckpt_model  = coco_path + "coco_keras_model_v0"
else:
    train_loss  = coco_path + "coco_losses_sigmoid_v1.csv"
    ckpt_path   = coco_path + "coco_sigmoid_v1.ckpt"
    ckpt_dir    = os.path.dirname(ckpt_path)
    ckpt_model  = coco_path + "coco_keras_model_sigmoid_v1"

coco_model = load_coco_model()
elapsed_tm = (time.time() - start_time) / 60
print("Elapsed Time:", str(round(elapsed_tm, 3)), "mins.")

# App Title. #
st.title("Coco Wide and Shallow Object Detection Model Test")

# Load the file. #
img_input_file = image_selector(
    folder_path="C:/Users/admin/Desktop/Codes/")
save_img_file  = \
    "C:/Users/admin/Desktop/Data/Results/coco_test_result.jpg"

init_flag  = True
run_button = st.sidebar.button("Run")
if run_button:
    if not init_flag:
        del fig
    start_time = time.time()
    
    fig = obj_detect_results(
        img_input_file, coco_model, label_dict, 
        heatmap=False, img_rows=img_width, img_cols=img_height, 
        img_scale=img_scale, img_box=None, thresh=0.30, 
        overlap_thresh=0.75, save_img_file=save_img_file)
    st.pyplot(fig)
    
    init_flag  = False
    elapsed_tm = (time.time() - start_time) / 60
    st.write("Elapsed Time:", str(round(elapsed_tm, 3)), "mins.")

save_button = st.sidebar.button("Save")
if save_button:
    if init_flag:
        st.write("No output image available.")
    else:
        fig.savefig(save_img_file, dpi=199)
