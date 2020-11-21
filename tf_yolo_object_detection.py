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

def cnn_block(
    x_cnn_input, n1_filter, n2_filter, 
    ker_sz1, ker_sz2, stride_1, stride_2, 
    blk_name, seperable=True, batch_norm=False):
    in_filter = \
        x_cnn_input.get_shape().as_list()[-1]
    
    cnn_stride_1 = (stride_1, stride_1)
    cnn_stride_2 = (stride_2, stride_2)
    
    if seperable:
        x_cnn_1 = layers.SeparableConv2D(
            n1_filter, (ker_sz1, ker_sz1), 
            strides=cnn_stride_1, padding="same", 
            activation="relu", name=blk_name+"_cnn1")(x_cnn_input)
        x_cnn_2 = layers.SeparableConv2D(
            n2_filter, (ker_sz2, ker_sz2), 
            strides=cnn_stride_2, padding='same', 
            activation="linear", name=blk_name+"_cnn2")(x_cnn_1)
    else:
        x_cnn_1 = layers.Conv2D(
            n1_filter, (ker_sz1, ker_sz1), 
            strides=cnn_stride_1, padding="same", 
            activation="relu", name=blk_name+"_cnn1")(x_cnn_input)
        x_cnn_2 = layers.Conv2D(
            n2_filter, (ker_sz2, ker_sz2), 
            strides=cnn_stride_2, padding='same', 
            activation="linear", name=blk_name+"_cnn2")(x_cnn_1)
    
    if batch_norm:
        x_bnorm = layers.BatchNormalization(
            name=blk_name+"_batch_norm")(x_cnn_2)
        
        if in_filter == n2_filter:
            x_cnn_out = layers.ReLU(
                name=blk_name+"_ReLU")(x_cnn_input+x_bnorm)
        else:
            x_cnn_out = layers.ReLU(
                name=blk_name+"_ReLU")(x_bnorm)
    else:
        if in_filter == n2_filter:
            x_cnn_out = layers.ReLU(
                name=blk_name+"_ReLU")(x_cnn_input+x_cnn_2)
        else:
            x_cnn_out = layers.ReLU(
                name=blk_name+"_ReLU")(x_cnn_2)
    return x_cnn_out

def build_model(
    n_filters, n_classes, tmp_pi=0.99, 
    n0_filter=32, img_rows=448, img_cols=448, 
    n_repeats=None, seperable=True, batch_norm=True):
    b_focal = tf.Variable(tf.math.log(
        (1.0-tmp_pi)/tmp_pi), trainable=True, name="b_focal")
    x_input = tf.keras.Input(
        shape=(img_rows, img_cols, 3), name="x_input")
    batch_sz = tf.shape(x_input)[0]
    
    if n_repeats is None:
        n_repeats = [1 for z in range(6)]
    
    # Block 0. #
    x_blk0_out = layers.Conv2D(
        n0_filter, (3,3), 
        strides=(1,1), padding='same', 
        activation=None, name="block0_cnn1")(x_input)
    
    # Block 1. #
    x_cnn1_in = x_blk0_out
    for tmp_rep in range(n_repeats[0]):
        blk_name = "blk1_" + str(tmp_rep+1)
        tmp_output = cnn_block(
            x_cnn1_in, int(n0_filter/2), 
            n0_filter, 3, 3, 1, 1, blk_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_cnn1_in = tmp_output
    x_cnn1_out = tmp_output
    
    # Downsampling. #
    x_feature_1 = layers.Conv2D(
        n_filters, (3,3), 
        strides=(2,2), padding='same', 
        activation=None, name="block1_out")(x_cnn1_out)
    
    # Block 2. #
    x_cnn2_in = x_feature_1
    for tmp_rep in range(n_repeats[1]):
        blk_name = "blk2_" + str(tmp_rep+1)
        tmp_output = cnn_block(
            x_cnn2_in, n0_filter, 
            n_filters, 3, 3, 1, 1, blk_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_cnn2_in = tmp_output
    x_cnn2_out = tmp_output
    
    # Downsampling. #
    x_blk2_out = layers.Conv2D(
        2*n_filters, (3,3), strides=(2,2), 
        padding='same', use_bias=False, 
        activation=None, name="block2_out")(x_cnn2_out)
    
    feature_shp = \
        [batch_sz, int(img_rows/4), int(img_cols/4), 4*n_filters]
    x_feature_2 = tf.concat([
        x_blk2_out, tf.reshape(x_feature_1, feature_shp)], axis=3)
    
    # Block 3. #
    x_cnn3_in = x_feature_2
    for tmp_rep in range(n_repeats[2]):
        blk_name = "blk3_" + str(tmp_rep+1)
        tmp_output = cnn_block(
            x_cnn3_in, n_filters, 
            2*n_filters, 3, 3, 1, 1, blk_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_cnn3_in = tmp_output
    x_cnn3_out = tmp_output
    
    # Downsampling. #
    x_blk3_out = layers.Conv2D(
        4*n_filters, (3,3),  strides=(2,2), 
        padding='same', use_bias=False, 
        activation=None, name="block3_out")(x_cnn3_out)
    
    feature_shp = \
        [batch_sz, int(img_rows/8), int(img_cols/8), 8*n_filters]
    x_feature_3 = tf.concat([
        x_blk3_out, tf.reshape(x_blk2_out, feature_shp)], axis=3)
    
    # Block 4. #
    x_cnn4_in = x_feature_3
    for tmp_rep in range(n_repeats[3]):
        blk_name = "blk4_" + str(tmp_rep+1)
        tmp_output = cnn_block(
            x_cnn4_in, 2*n_filters, 
            4*n_filters, 3, 3, 1, 1, blk_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_cnn4_in = tmp_output
    x_cnn4_out = tmp_output
    
    # Downsampling. #
    x_blk4_out = layers.Conv2D(
        8*n_filters, (3,3),  strides=(2,2), 
        padding='same', use_bias=False, 
        activation=None, name="block4_out")(x_cnn4_out)
    
    feature_shp = \
        [batch_sz, int(img_rows/16), int(img_cols/16), 16*n_filters]
    x_feature_4 = tf.concat([
        x_blk4_out, tf.reshape(x_blk3_out, feature_shp)], axis=3)
    
    # Block 5. #
    x_cnn5_in = x_feature_4
    for tmp_rep in range(n_repeats[4]):
        blk_name = "blk5_" + str(tmp_rep+1)
        tmp_output = cnn_block(
            x_cnn5_in, 4*n_filters, 
            8*n_filters, 3, 3, 1, 1, blk_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_cnn5_in = tmp_output
    x_cnn5_out = tmp_output
    
    # Downsampling. #
    x_blk5_out = layers.Conv2D(
        16*n_filters, (3,3),  strides=(2,2), 
        padding='same', use_bias=False, 
        activation=None, name="block5_out")(x_cnn5_out)
    
    feature_shp = \
        [batch_sz, int(img_rows/32), int(img_cols/32), 32*n_filters]
    x_feature_5 = tf.concat([
        x_blk5_out, tf.reshape(x_blk4_out, feature_shp)], axis=3)
    
    # Block 6. #
    x_cnn6_in = x_feature_5
    for tmp_rep in range(n_repeats[5]):
        blk_name = "blk6_" + str(tmp_rep+1)
        tmp_output = cnn_block(
            x_cnn6_in, 8*n_filters, 
            16*n_filters, 3, 3, 1, 1, blk_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_cnn6_in = tmp_output
    x_cnn6_out = tmp_output
    
    # Downsampling. #
    x_blk6_out = layers.Conv2D(
        32*n_filters, (3,3),  strides=(2,2), 
        padding='same', use_bias=False, 
        activation=None, name="block6_out")(x_cnn6_out)
    
    # Regression and Classification heads. #
    x_reg_small = tf.nn.sigmoid(layers.Conv2D(
        4, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="reg_small")(x_blk3_out))
    x_cls_small = layers.Conv2D(
        n_classes, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="cls_small")(x_blk3_out)
    x_out_small = tf.concat(
        [x_reg_small, x_cls_small+b_focal], axis=3)
    
    x_reg_medium = tf.nn.sigmoid(layers.Conv2D(
        4, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="reg_medium")(x_blk4_out))
    x_cls_medium = layers.Conv2D(
        n_classes, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="cls_medium")(x_blk4_out)
    x_out_medium = tf.concat(
        [x_reg_medium, x_cls_medium+b_focal], axis=3)
    
    x_reg_large = tf.nn.sigmoid(layers.Conv2D(
        4, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="reg_large")(x_blk5_out))
    x_cls_large = layers.Conv2D(
        n_classes, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="cls_large")(x_blk5_out)
    x_out_large = tf.concat(
        [x_reg_large, x_cls_large+b_focal], axis=3)
    
    x_reg_vlarge = tf.nn.sigmoid(layers.Conv2D(
        4, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="reg_vlarge")(x_blk6_out))
    x_cls_vlarge = layers.Conv2D(
        n_classes, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="cls_vlarge")(x_blk6_out)
    x_out_vlarge = tf.concat(
        [x_reg_vlarge, x_cls_vlarge+b_focal], axis=3)
    
    # Concatenate the outputs. #
    x_output = [
        x_out_small, x_out_medium, 
        x_out_large, x_out_vlarge]
    obj_model = tf.keras.Model(
        inputs=x_input, outputs=x_output)
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
    return tf.reduce_sum(foc_loss_stable, axis=[1, 2, 3])

def model_loss(
    bboxes, masks, outputs, loss_type="sigmoid"):
    total_reg_loss = 0.0
    total_cls_loss = 0.0
    for id_sc in range(len(outputs)):
        reg_weight = tf.expand_dims(masks[id_sc], axis=3)
        reg_output = outputs[id_sc][:, :, :, :4]
        cls_output = outputs[id_sc][:, :, :, 4:]
        cls_labels = tf.cast(
            bboxes[id_sc][:, :, :, 4:], tf.int32)
        
        if loss_type == "sigmoid":
            total_cls_loss += tf.reduce_sum(
                sigmoid_loss(cls_labels, cls_output))
        else:
            total_cls_loss += tf.reduce_sum(
                focal_loss(cls_labels, cls_output))
        total_reg_loss += tf.reduce_sum(tf.multiply(tf.abs(
            bboxes[id_sc][:, :, :, :4] - reg_output), reg_weight))
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
        
        tmp_bboxes = []
        tmp_masks  = []
        for id_sc in range(len(bboxes)):
            tmp_masks.append(masks[id_sc][id_st:id_en, :, :])
            tmp_bboxes.append(bboxes[id_sc][id_st:id_en, :, :, :])
        
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
    img_in_file, voc_model, labels, 
    heatmap=True, img_box=None, thresh=0.50, 
    img_rows=448, img_cols=448, img_scale=None, 
    img_title=None, save_img_file="object_detection_result.jpg"):
    if img_scale is None:
        if max(img_rows, img_cols) >= 512:
            max_scale = max(img_rows, img_cols)
        else:
            max_scale = 512
        img_scale = [64, 128, 256, max_scale]
    else:
        if len(img_scale) != 4:
            raise ValueError("img_scale must be size 4.")
    dwn_scale = [8, 16, 32, 64]
    
    # Read the image. #
    image_resized = tf.expand_dims(_parse_image(
        img_in_file, img_rows=img_rows, img_cols=img_cols), axis=0)
    
    tmp_output = \
        voc_model.predict(image_resized)
    n_classes  = tmp_output[0][0, :, :, 4:].shape[2]
    
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
        tmp_probs = []
        for n_sc in range(len(tmp_output)):
            tmp_array = np.zeros(
                [int(img_rows/8), int(img_cols/8)])
            down_scale = int(dwn_scale[n_sc] / 8)
            cls_output = tmp_output[n_sc][0, :, :, 4:]
            cls_probs  = tf.nn.sigmoid(cls_output)
            
            if n_classes > 1:
                obj_probs = tf.reduce_max(
                    cls_probs[:, :, 1:], axis=2)
            else:
                obj_probs = cls_probs[:, :, 0]
            tmp_array[int(down_scale/2)::down_scale, 
                      int(down_scale/2)::down_scale] = obj_probs
            
            tmp_array = tf.expand_dims(tmp_array, axis=2)
            obj_probs = tf.squeeze(tf.image.resize(tf.expand_dims(
                tmp_array, axis=0), [img_width, img_height]), axis=3)
            tmp_probs.append(obj_probs)
        
        tmp_probs = tf.concat(tmp_probs, axis=0)
        tmp_probs = tf.reduce_max(tmp_probs, axis=0)
        
        tmp = ax.imshow(tmp_probs, "jet", alpha=0.50)
        fig.colorbar(tmp, ax=ax)
    
    n_obj_detected = 0
    for n_sc in range(4):
        down_scale = dwn_scale[n_sc]
        
        reg_output = tmp_output[n_sc][0, :, :, :4]
        cls_output = tmp_output[n_sc][0, :, :, 4:]
        cls_probs  = tf.nn.sigmoid(cls_output)
        if n_sc == 3:
            if max(img_rows, img_cols) <= img_scale[n_sc]:
                box_scale = max(img_rows, img_cols)
            else:
                box_scale = img_scale[n_sc]
        else:
            box_scale = img_scale[n_sc]
        
        if n_classes > 1:
            prob_max = tf.reduce_max(
                cls_probs[:, :, 1:], axis=2)
            pred_label = 1 + tf.math.argmax(
                cls_probs[:, :, 1:], axis=2)
        else:
            prob_max = cls_probs[:, :, 0]
        tmp_thresh = \
            np.where(prob_max >= thresh, 1, 0)
        tmp_coords = np.nonzero(tmp_thresh)
        
        for n_box in range(len(tmp_coords[0])):
            x_coord = tmp_coords[0][n_box]
            y_coord = tmp_coords[1][n_box]
            
            tmp_boxes = reg_output[x_coord, y_coord, :]
            tmp_probs = int(
                prob_max[x_coord, y_coord].numpy()*100)
            if n_classes > 1:
                tmp_label = str(labels[
                    pred_label[x_coord, y_coord].numpy()])
            else:
                tmp_label = str(labels[0])
            
            x_centroid = \
                tmp_w_ratio * (x_coord + tmp_boxes[0])*down_scale
            y_centroid = \
                tmp_h_ratio * (y_coord + tmp_boxes[1])*down_scale
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
            down_scale = dwn_scale[n_sc]
            
            if n_sc == 3:
                if max(img_rows, img_cols) <= img_scale[n_sc]:
                    box_scale = max(img_rows, img_cols)
                else:
                    box_scale = img_scale[n_sc]
            else:
                box_scale = img_scale[n_sc]
            
            tmp_true_box = np.nonzero(img_box[n_sc][:, :, 4])
            for n_box in range(len(tmp_true_box[0])):
                x_coord = tmp_true_box[0][n_box]
                y_coord = tmp_true_box[1][n_box]
                tmp_boxes = img_box[n_sc][x_coord, y_coord, :4]
                
                x_centroid = \
                    tmp_w_ratio * (x_coord + tmp_boxes[0])*down_scale
                y_centroid = \
                    tmp_h_ratio * (y_coord + tmp_boxes[1])*down_scale
                box_width  = tmp_w_ratio * box_scale * tmp_boxes[2]
                box_height = tmp_h_ratio * box_scale * tmp_boxes[3]
                
                x_lower = x_centroid - box_width/2
                y_lower = y_centroid - box_height/2
                box_patch = plt.Rectangle(
                    (y_lower.numpy(), x_lower.numpy()), 
                    box_height.numpy(), box_width.numpy(), 
                    linewidth=1, edgecolor="black", fill=None)
                ax.add_patch(box_patch)
    
    if img_title is not None:
        fig.suptitle(img_title)
    fig.savefig(save_img_file, dpi=199)
    plt.close()
    del fig, ax
    return None

