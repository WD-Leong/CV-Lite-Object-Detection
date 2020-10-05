# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:30:14 2020

@author: admin
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from PIL import Image
import matplotlib.pyplot as plt

def _parse_image(
    filename, img_rows=448, img_cols=448):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = \
        tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize(
        image_decoded, [img_rows, img_cols])
    image_resized = tf.ensure_shape(
        image_resized ,shape=(img_rows, img_cols, 3))
    return image_resized

def bbox_ciou(boxes1, boxes2, eps=1e-7):
    boxes1_coor = tf.concat([
        boxes1[..., :2] - boxes1[..., 2:] * 0.5,
        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([
        boxes2[..., :2] - boxes2[..., 2:] * 0.5,
        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    enclose_left_up = tf.maximum(
        boxes1_coor[..., :2], boxes2_coor[..., :2])
    enclose_right_down = tf.minimum(
        boxes1_coor[..., 2:], boxes2_coor[..., 2:])

    inter_section = tf.maximum(
        enclose_right_down - enclose_left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = tf.math.divide_no_nan(inter_area, union_area)

    left_up = tf.minimum(
        boxes1_coor[..., :2], boxes2_coor[..., :2])
    right_down = tf.maximum(
        boxes1_coor[..., 2:], boxes2_coor[..., 2:])
    c = tf.maximum(right_down - left_up, 0.0)
    c = tf.pow(c[..., 0], 2) + tf.pow(c[..., 1], 2)
    
    u = tf.add(
        (boxes1[..., 0]-boxes2[..., 0]) * (boxes1[..., 0]-boxes2[..., 0]), 
        (boxes1[..., 1]-boxes2[..., 1]) * (boxes1[..., 1]-boxes2[..., 1]))
    d = tf.math.divide_no_nan(u, c)
    
    ar_gt = tf.math.divide_no_nan(boxes2[..., 2] , boxes2[..., 3])
    ar_pred = tf.math.divide_no_nan(boxes1[..., 2], boxes1[..., 3])
    
    pi = tf.convert_to_tensor(np.pi)
    ar_loss = tf.multiply(
        tf.math.divide_no_nan(4.0, pi * pi), 
        tf.pow((tf.atan(ar_gt) - tf.atan(ar_pred)), 2))
    alpha = tf.math.divide_no_nan(ar_loss ,(1 - iou + ar_loss))
    ciou_term = d + alpha * ar_loss
    ciou = iou - ciou_term   #   1
    ciou = tf.clip_by_value(
        ciou, clip_value_min=-1.0, clip_value_max=0.99)
    return ciou

def cnn_block(
    x_cnn_input, n_filters, 
    ker_sz1, ker_sz2, stride_1, stride_2, 
    blk_name, seperable=True, batch_norm=False):
    cnn_stride_1 = (stride_1, stride_1)
    cnn_stride_2 = (stride_2, stride_2)
    
    if seperable:
        x_cnn_1 = layers.SeparableConv2D(
            n_filters, (ker_sz1, ker_sz1), 
            strides=cnn_stride_1, padding="same", 
            activation="relu", name=blk_name+"_cnn1")(x_cnn_input)
        x_cnn_2 = layers.SeparableConv2D(
            n_filters, (ker_sz2, ker_sz2), 
            strides=cnn_stride_2, padding='same', 
            activation="linear", name=blk_name+"_cnn2")(x_cnn_1)
    else:
        x_cnn_1 = layers.Conv2D(
            n_filters, (ker_sz1, ker_sz1), 
            strides=cnn_stride_1, padding="same", 
            activation="relu", name=blk_name+"_cnn1")(x_cnn_input)
        x_cnn_2 = layers.Conv2D(
            n_filters, (ker_sz2, ker_sz2), 
            strides=cnn_stride_2, padding='same', 
            activation="linear", name=blk_name+"_cnn2")(x_cnn_1)
    
    if batch_norm:
        x_bnorm = layers.BatchNormalization(
            name=blk_name+"_batch_norm")(x_cnn_2)
        x_cnn_out = layers.ReLU(name=blk_name+"_ReLU")(x_bnorm)
    else:
        x_cnn_out = layers.ReLU(name=blk_name+"_ReLU")(x_cnn_2)
    return x_cnn_out

def build_model(
    n_filters, n_classes, 
    tmp_pi=0.99, img_rows=448, img_cols=448, 
    seperable=True, batch_norm=True):
    tmp_b  = tf.math.log((1.0-tmp_pi)/tmp_pi)
    
    b_focal = tf.Variable(
        tmp_b, trainable=True, name="b_focal")
    x_input = tf.keras.Input(
        shape=(img_rows, img_cols, 3), name="x_input")
    
    # Block 1. #
    x_cnn_out_1 = cnn_block(
        x_input, n_filters, 5, 3, 1, 2, 
        "block_1", batch_norm=batch_norm)
    
    # Block 2. #
    x_cnn_out_2 = cnn_block(
        x_cnn_out_1, 2*n_filters, 3, 3, 1, 2, "block_2")
    
    # Block 3. #
    x_blk3_out  = cnn_block(
        x_cnn_out_2, 4*n_filters, 3, 3, 1, 2, 
        "block_3", seperable=seperable, batch_norm=batch_norm)
    x_cnn_out_3 = layers.Conv2D(
        2*n_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="cnn_out_3")(x_blk3_out)
    
    if batch_norm:
        x_bn_small  = layers.BatchNormalization(
            name="cnn_small_batch_norm")(x_cnn_out_3)
        x_cnn_small = layers.ReLU(name="cnn_small_relu")(x_bn_small)
    else:
        x_cnn_small = layers.ReLU(name="cnn_small_relu")(x_cnn_out_3)
    
    # Block 4. #
    x_blk4_out  = cnn_block(
        x_cnn_out_3, 8*n_filters, 3, 3, 1, 2, 
        "block_4", seperable=seperable, batch_norm=batch_norm)
    x_cnn_out_4 = layers.Conv2D(
        4*n_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="cnn_out_4")(x_blk4_out)
    
    if batch_norm:
        x_bn_medium  = layers.BatchNormalization(
            name="cnn_medium_batch_norm")(x_cnn_out_4)
        x_cnn_medium = layers.ReLU(
            name="cnn_medium_relu")(x_bn_medium)
    else:
        x_cnn_medium = layers.ReLU(
            name="cnn_medium_relu")(x_cnn_out_4)
    
    # Block 5. #
    x_cnn_out_5  = cnn_block(
        x_cnn_out_4, 8*n_filters, 3, 3, 1, 2, 
        "block_5", seperable=seperable, batch_norm=batch_norm)
    
    if batch_norm:
        x_bn_large  = layers.BatchNormalization(
            name="cnn_large_batch_norm")(x_cnn_out_5)
        x_cnn_large = layers.ReLU(
            name="cnn_large_relu")(x_bn_large)
    else:
        x_cnn_large = layers.ReLU(
            name="cnn_large_relu")(x_cnn_out_5)
    
    # Block 5. #
    x_cnn_out_6  = cnn_block(
        x_cnn_out_5, 16*n_filters, 3, 3, 1, 2, 
        "block_6", seperable=seperable, batch_norm=batch_norm)
    x_upsample_3 = layers.Conv2DTranspose(
        8*n_filters, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="upsample_3")(x_cnn_out_6)
    x_ups_cnn_3  = layers.Conv2D(
        8*n_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="ups_cnn_3")(x_upsample_3)
    x_ups_relu_3 = layers.ReLU(name="ups_relu_3")(x_ups_cnn_3)
    x_upsample_4 = layers.Conv2DTranspose(
        4*n_filters, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="upsample_4")(x_ups_relu_3)
    x_ups_cnn_4  = layers.Conv2D(
        4*n_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="ups_cnn_4")(x_upsample_4)
    x_ups_relu_4 = layers.ReLU(name="ups_relu_4")(x_ups_cnn_4)
    x_upsample_5 = layers.Conv2DTranspose(
        2*n_filters, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="upsample_5")(x_ups_relu_4)
    x_ups_cnn_5  = layers.Conv2D(
        2*n_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="ups_cnn_5")(x_upsample_5)
    x_cnn_vlarge = layers.ReLU(
        name="cnn_vlarge_relu")(x_ups_cnn_5)
    
    # Feature Pyramid Network. #
    x_fpn_large  = x_ups_relu_3 + x_cnn_large
    x_upsample_6 = layers.Conv2DTranspose(
        4*n_filters, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="upsample_6")(x_fpn_large)
    x_ups_cnn_6  = layers.Conv2D(
        4*n_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="ups_cnn_6")(x_upsample_6)
    x_ups_relu_6 = layers.ReLU(name="ups_relu_6")(x_ups_cnn_6)
    x_upsample_7 = layers.Conv2DTranspose(
        2*n_filters, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="upsample_7")(x_ups_relu_6)
    x_ups_cnn_7  = layers.Conv2D(
        2*n_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="ups_cnn_7")(x_upsample_7)
    x_ups_relu_7 = layers.ReLU(name="ups_relu_7")(x_ups_cnn_7)
    
    x_fpn_medium = x_ups_relu_6 + x_cnn_medium
    x_upsample_8 = layers.Conv2DTranspose(
        2*n_filters, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="upsample_8")(x_fpn_medium)
    x_ups_cnn_8  = layers.Conv2D(
        2*n_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="ups_cnn_8")(x_upsample_8)
    x_ups_relu_8 = layers.ReLU(name="ups_relu_8")(x_ups_cnn_8)
    
    x_fpn_small = x_ups_relu_8 + x_cnn_small
    x_reg_small = layers.Conv2D(
        4, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="reg_small")(x_fpn_small)
    x_cls_small = layers.Conv2D(
        n_classes, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="cls_small")(x_fpn_small)
    
    x_reg_medium = layers.Conv2D(
        4, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="reg_medium")(x_ups_relu_8)
    x_cls_medium = layers.Conv2D(
        n_classes, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="cls_medium")(x_ups_relu_8)
    
    x_reg_large = layers.Conv2D(
        4, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="reg_large")(x_ups_relu_7)
    x_cls_large = layers.Conv2D(
        n_classes, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="cls_large")(x_ups_relu_7)
    
    x_reg_vlarge = layers.Conv2D(
        4, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="reg_vlarge")(x_cnn_vlarge)
    x_cls_vlarge = layers.Conv2D(
        n_classes, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="cls_vlarge")(x_cnn_vlarge)
    
    # Concatenate the outputs. #
    reg_heads = tf.concat(
        [tf.expand_dims(x_reg_small, axis=3), 
         tf.expand_dims(x_reg_medium, axis=3), 
         tf.expand_dims(x_reg_large, axis=3), 
         tf.expand_dims(x_reg_vlarge, axis=3)], axis=3)
    cls_heads = tf.concat(
        [tf.expand_dims(x_cls_small, axis=3), 
         tf.expand_dims(x_cls_medium, axis=3), 
         tf.expand_dims(x_cls_large, axis=3), 
         tf.expand_dims(x_cls_vlarge, axis=3)], axis=3)
    
    x_outputs = tf.concat(
        [tf.nn.sigmoid(reg_heads), cls_heads+b_focal], axis=4)
    obj_model = tf.keras.Model(
        inputs=x_input, outputs=x_outputs)
    return obj_model

def sigmoid_loss(labels, logits):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(labels, tf.float32), logits=logits)

def focal_loss(
    labels, logits, alpha=0.25, gamma=2.0):
    labels = tf.cast(labels, tf.float32)
    tmp_log_logits  = tf.math.log(1.0 + tf.exp(-1.0 * tf.abs(logits)))
    
    tmp_abs_term = tf.math.add(
        tf.multiply(labels * alpha * tmp_log_logits, 
                    tf.pow(1.0 - tf.nn.sigmoid(logits), gamma)), 
        tf.multiply(tf.pow(tf.nn.sigmoid(logits), gamma), 
                    (1.0 - labels) * (1.0 - alpha) * tmp_log_logits))
    
    tmp_x_neg = tf.multiply(
        labels * alpha * tf.minimum(logits, 0), 
        tf.pow(1.0 - tf.nn.sigmoid(logits), gamma))
    tmp_x_pos = tf.multiply(
        (1.0 - labels) * (1.0 - alpha), 
        tf.maximum(logits, 0) * tf.pow(tf.nn.sigmoid(logits), gamma))
    
    foc_loss_stable = tmp_abs_term + tmp_x_pos - tmp_x_neg
    return tf.reduce_sum(foc_loss_stable, axis=[1, 2, 3, 4])

def one_class_focal_loss(
    labels, logits, alpha=0.25, gamma=2.0, eps=1.0e-9):
    labels = tf.cast(labels, tf.float32)
    pred_prob = tf.nn.sigmoid(logits)
    focal_pos = labels*alpha*tf.multiply(
        tf.pow(1.0-pred_prob, gamma), tf.math.log(pred_prob + eps))
    focal_neg = (1.0-labels)*(1.0-alpha)*tf.multiply(
        tf.pow(pred_prob, gamma), tf.math.log(1.0 - pred_prob + eps))
    
    focal_loss = focal_pos + focal_neg
    return -1.0 * tf.reduce_sum(focal_loss, axis=[1, 2, 3, 4])

def model_loss(
    bboxes, masks, outputs, img_size=448, 
    reg_lambda=0.10, loss_type="sigmoid", eps=1.0e-6):
    input_size = tf.cast(img_size, tf.float32)
    reg_weight = tf.expand_dims(masks, axis=4)
    reg_output = outputs[:, :, :, :, :4]
    cls_output = outputs[:, :, :, :, 4:]
    cls_labels = tf.cast(bboxes[:, :, :, :, 4:], tf.int32)
    
#    ciou = bbox_ciou(reg_output, bboxes[:, :, :, :4])
#    bbox_loss_scale = \
#        2.0 - 1.0 * tf.multiply(
#            bboxes[:, :, :, 2], 
#            bboxes[:, :, :, 3]) / (input_size ** 2)
#    total_reg_loss  = tf.reduce_sum(
#        masks * bbox_loss_scale * (1.0 - ciou))
    
    if loss_type == "sigmoid":
        total_cls_loss  = tf.reduce_sum(
            sigmoid_loss(cls_labels, cls_output))
    else:
        total_cls_loss  = tf.reduce_sum(
            focal_loss(cls_labels, cls_output))
    total_reg_loss  = tf.reduce_sum(tf.multiply(
        tf.abs(bboxes[:, :, :, :, :4] - reg_output), reg_weight))
    return total_cls_loss, total_reg_loss

def train_step(
    voc_model, sub_batch_sz, 
    images, bboxes, masks, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0, 
    cls_lambda=5.0, loss_type="focal"):
    optimizer.lr.assign(learning_rate)
    
    batch_size = images.shape[0]
    if batch_size <= sub_batch_sz:
        n_sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        n_sub_batch = int(batch_size / sub_batch_sz)
    else:
        n_sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = voc_model.trainable_variables
    acc_gradients = [tf.zeros_like(var) for var in model_params]
    
    tmp_reg_loss = 0.0
    tmp_cls_loss = 0.0
    for n_sub in range(n_sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (n_sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_images = images[id_st:id_en, :, :, :]
        tmp_bboxes = bboxes[id_st:id_en, :, :, :, :]
        tmp_masks  = masks[id_st:id_en, :, :, :]
        
        with tf.GradientTape() as voc_tape:
            tmp_output = voc_model(tmp_images, training=True)
            tmp_losses = model_loss(
                tmp_bboxes, tmp_masks, tmp_output, loss_type=loss_type)
            
            tmp_cls_loss += tmp_losses[0]
            tmp_reg_loss += tmp_losses[1]
            total_losses = \
                cls_lambda*tmp_losses[0] + tmp_losses[1]
        
        # Accumulate the gradients. #
        tmp_gradients = \
            voc_tape.gradient(total_losses, model_params)
        acc_gradients = [
            (acc_grad+grad) for \
            acc_grad, grad in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_reg_loss  = tmp_reg_loss / batch_size
    avg_cls_loss  = tmp_cls_loss / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clipped_gradients, _ = \
        tf.clip_by_global_norm(acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return avg_cls_loss, avg_reg_loss

def obj_detect_results(
    img_in_file, voc_model, labels, img_box=None, 
    heatmap=True, thresh=0.50, img_rows=448, img_cols=448, 
    img_scale=None, save_img_file="object_detection_result.jpg"):
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
    
    n_obj_detected = 0
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
                tmp_label = str(labels[
                    pred_label[x_coord, y_coord].numpy()])
            else:
                tmp_label = str(labels[0])
            
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
            
            box_patch = plt.Rectangle(
                (y_lower, x_lower), box_height, box_width, 
                linewidth=1, edgecolor="red", fill=None)
            
            n_obj_detected += 1
            tmp_text = \
                tmp_label + ": " + str(tmp_probs) + "%"
            ax.add_patch(box_patch)
            ax.text(y_lower, x_lower, tmp_text, 
                    fontsize=10, color="red")
    print(str(n_obj_detected), "objects detected.")
    
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
    
    fig.savefig(save_img_file, dpi=199)
    plt.close()
    del fig, ax
    return None

