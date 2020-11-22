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
    ker_sz1, ker_sz2, stride_1, stride_2, blk_name, 
    seperable=True, batch_norm=True, norm_order="first"):
    cnn_stride_1 = (stride_1, stride_1)
    cnn_stride_2 = (stride_2, stride_2)
    
    if norm_order == "first":
        tmp_input = layers.BatchNormalization(
            name=blk_name+"_batch_norm")(x_cnn_input)
    else:
        tmp_input = x_cnn_input
    
    if seperable:
        x_cnn_1 = layers.SeparableConv2D(
            n1_filter, (ker_sz1, ker_sz1), 
            strides=cnn_stride_1, padding="same", 
            activation="relu", name=blk_name+"_cnn1")(tmp_input)
        x_cnn_2 = layers.SeparableConv2D(
            n2_filter, (ker_sz2, ker_sz2), 
            strides=cnn_stride_2, padding='same', 
            activation=None, name=blk_name+"_cnn2")(x_cnn_1)
    else:
        x_cnn_1 = layers.Conv2D(
            n1_filter, (ker_sz1, ker_sz1), 
            strides=cnn_stride_1, padding="same", 
            activation="relu", name=blk_name+"_cnn1")(tmp_input)
        x_cnn_2 = layers.Conv2D(
            n2_filter, (ker_sz2, ker_sz2), 
            strides=cnn_stride_2, padding='same', 
            activation=None, name=blk_name+"_cnn2")(x_cnn_1)
    
    if norm_order == "last":
        x_bnorm = layers.BatchNormalization(
            name=blk_name+"_batch_norm")(x_cnn_2)
        x_cnn_out = layers.ReLU(
            name=blk_name+"_ReLU")(x_bnorm)
    else:
        x_cnn_out = layers.ReLU(
            name=blk_name+"_ReLU")(x_cnn_2)
    return x_cnn_out

def build_model(
    n_filters, n_classes, tmp_pi=0.99, 
    n_rounds=1, seperable=True, batch_norm=True):
    b_focal = tf.Variable(tf.math.log(
        (1.0-tmp_pi)/tmp_pi), trainable=True, name="b_focal")
    x_input = tf.keras.Input(
        shape=(None, None, 3), name="x_input")
    x_shape = tf.shape(x_input)
    
    batch_sz = tf.shape(x_input)[0]
    img_rows = tf.cast(x_shape[1]/8, tf.int32)
    img_cols = tf.cast(x_shape[2]/8, tf.int32)
    
    curr_input = x_input
    for n_round in range(n_rounds):
        # Encoder network. #
        # Block 0. #
        blk0_name = "block0_cnn1_round_" + str(n_round+1)
        x_blk0_out = layers.Conv2D(
            n_filters, (3,3), strides=(1,1), 
            padding="same", use_bias=True, 
            activation="relu", name=blk0_name)(curr_input)
        
        # Block 1. #
        n0_filters = int(n_filters/2)
        blk1_name  = "block_1_round_" + str(n_round+1)
        x_cnn1_out = cnn_block(
            x_input, n0_filters, 
            n_filters, 3, 3, 1, 1, blk1_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_res1_out = x_blk0_out + x_cnn1_out
        
        # Downsampling. #
        blk1_name  += "_out"
        x_blk1_out = layers.Conv2D(
            2*n_filters, (3,3), strides=(2,2), 
            padding="same", use_bias=True, 
            activation="relu", name=blk1_name)(x_res1_out)
        
        # Block 2. #
        blk2_name  = "block_2_round_" + str(n_round+1)
        x_cnn2_out = cnn_block(
            x_blk1_out, n_filters, 
            2*n_filters, 3, 3, 1, 1, blk2_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_res2_out = x_blk1_out + x_cnn2_out
        
        # Downsampling. #
        blk2_name  += "_out"
        x_blk2_out = layers.Conv2D(
            4*n_filters, (3,3), strides=(2,2), 
            padding="same", use_bias=True, 
            activation="relu", name=blk2_name)(x_res2_out)
        
        # Block 3. #
        blk3_name  = "block_3_round_" + str(n_round+1)
        x_cnn3_out = cnn_block(
            x_blk2_out, 2*n_filters, 
            4*n_filters, 3, 3, 1, 1, blk3_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_res3_out = x_blk2_out + x_cnn3_out
        
        # Downsampling. #
        blk3_name  += "_out"
        x_blk3_out = layers.Conv2D(
            8*n_filters, (3,3),  strides=(2,2), 
            padding="same", use_bias=True, 
            activation="relu", name=blk3_name)(x_res3_out)
        
        # Block 4. #
        blk4_name  = "block_4_round_" + str(n_round+1)
        x_cnn4_out = cnn_block(
            x_blk3_out, 4*n_filters, 
            8*n_filters, 3, 3, 1, 1, blk4_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_res4_out = x_blk3_out + x_cnn4_out
        
        # Downsampling. #
        blk4_name  += "_out"
        x_blk4_out = layers.Conv2D(
            16*n_filters, (3,3),  strides=(2,2), 
            padding="same", use_bias=True, 
            activation="relu", name=blk4_name)(x_res4_out)
        
        # Block 5. #
        blk5_name  = "block_5_round_" + str(n_round+1)
        x_cnn5_out  = cnn_block(
            x_blk4_out, 8*n_filters, 
            16*n_filters, 3, 3, 1, 1, blk5_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_res5_out = x_blk4_out + x_cnn5_out
        
        # Downsampling. #
        blk5_name  += "_out"
        x_blk5_out = layers.Conv2D(
            32*n_filters, (3,3),  strides=(2,2), 
            padding="same", use_bias=True, 
            activation="relu", name=blk5_name)(x_res5_out)
        
        # Block 6. #
        blk6_name  = "block_6_round_" + str(n_round+1)
        x_cnn6_out = cnn_block(
            x_blk5_out, 16*n_filters, 
            32*n_filters, 3, 3, 1, 1, blk6_name, 
            seperable=seperable, batch_norm=batch_norm)
        x_res6_out = x_blk5_out + x_cnn6_out
        
        # Downsampling. #
        blk6_name  += "_out"
        x_blk6_out = layers.Conv2D(
            64*n_filters, (3,3),  strides=(2,2), 
            padding="same", use_bias=True, 
            activation="relu", name=blk6_name)(x_res6_out)
        
        # Decoder network. #
        # Upsample network for 6th layer. #
        x_ups6_out = layers.UpSampling2D(
            interpolation="bilinear")(x_blk6_out)
        blk6_name  = "dec_6_in_round_" + str(n_round+1)
        x_dec6_in  = layers.Conv2D(
            32*n_filters, (3,3),  strides=(1,1), 
            padding="same", use_bias=True, 
            activation="relu", name=blk6_name)(x_ups6_out)
        
        blk6_name  = "dec_6_out_round_" + str(n_round+1)
        x_res6_in  = x_res6_out + x_dec6_in
        x_dec6_out = cnn_block(
            x_res6_in, 16*n_filters, 
            32*n_filters, 3, 3, 1, 1, blk6_name, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Upsample network for 5th layer. #
        x_ups5_out = layers.UpSampling2D(
            interpolation="bilinear")(x_dec6_out)
        blk5_name  = "dec_5_in_round_" + str(n_round+1)
        x_dec5_in  = layers.Conv2D(
            16*n_filters, (3,3),  strides=(1,1), 
            padding="same", use_bias=True, 
            activation="relu", name=blk5_name)(x_ups5_out)
        
        blk5_name  = "dec_5_out_round_" + str(n_round+1)
        x_res5_in  = x_res5_out + x_dec5_in
        x_dec5_out = cnn_block(
            x_res5_in, 8*n_filters, 
            16*n_filters, 3, 3, 1, 1, blk5_name, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Upsample network for 4th layer. #
        x_ups4_out = layers.UpSampling2D(
            interpolation="bilinear")(x_dec5_out)
        blk4_name  = "dec_4_in_round_" + str(n_round+1)
        x_dec4_in  = layers.Conv2D(
            8*n_filters, (3,3),  strides=(1,1), 
            padding="same", use_bias=True, 
            activation="relu", name=blk4_name)(x_ups4_out)
        
        blk4_name  = "dec_4_out_round_" + str(n_round+1)
        x_res4_in  = x_res4_out + x_dec4_in
        x_dec4_out = cnn_block(
            x_res4_in, 4*n_filters, 
            8*n_filters, 3, 3, 1, 1, blk4_name, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Upsample network for 3rd layer. #
        x_ups3_out = layers.UpSampling2D(
            interpolation="bilinear")(x_dec4_out)
        blk3_name  = "dec_3_in_round_" + str(n_round+1)
        x_dec3_in  = layers.Conv2D(
            4*n_filters, (3,3),  strides=(1,1), 
            padding="same", use_bias=True, 
            activation="relu", name=blk3_name)(x_ups3_out)
        
        blk3_name  = "dec_3_out_round_" + str(n_round+1)
        x_res3_in  = x_res3_out + x_dec3_in
        x_dec3_out = cnn_block(
            x_res3_in, 2*n_filters, 
            4*n_filters, 3, 3, 1, 1, blk3_name, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Upsample network for 2nd layer. #
        x_ups2_out = layers.UpSampling2D(
            interpolation="bilinear")(x_dec3_out)
        blk2_name  = "dec_2_in_round_" + str(n_round+1)
        x_dec2_in  = layers.Conv2D(
            2*n_filters, (3,3),  strides=(1,1), 
            padding="same", use_bias=True, 
            activation="relu", name=blk2_name)(x_ups2_out)
        
        blk2_name  = "dec_2_out_round_" + str(n_round+1)
        x_res2_in  = x_res2_out + x_dec2_in
        x_dec2_out = cnn_block(
            x_res2_in, n_filters, 
            2*n_filters, 3, 3, 1, 1, blk2_name, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Upsample network for 1st layer. #
        x_ups1_out = layers.UpSampling2D(
            interpolation="bilinear")(x_dec2_out)
        blk1_name  = "dec_1_in_round_" + str(n_round+1)
        x_dec1_in  = layers.Conv2D(
            n_filters, (3,3),  strides=(1,1), 
            padding="same", use_bias=True, 
            activation="relu", name=blk1_name)(x_ups1_out)
        
        blk1_name  = "dec_1_out_round_" + str(n_round+1)
        x_res1_in  = x_res1_out + x_dec1_in
        x_dec1_out = cnn_block(
            x_res1_in, n0_filters, 
            n_filters, 3, 3, 1, 1, blk1_name, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Set the current input to the decoder output. #
        curr_input = x_dec1_out
    
    # Concatenate the different scales output. #
    dec1_reshape = [batch_sz, img_rows, img_cols, 64*n_filters]
    dec2_reshape = [batch_sz, img_rows, img_cols, 32*n_filters]
    dec3_reshape = [batch_sz, img_rows, img_cols, 16*n_filters]
    dec5_reshape = [batch_sz, img_rows, img_cols, 4*n_filters]
    dec6_reshape = [batch_sz, img_rows, img_cols, 2*n_filters]
    
    x_dec1_features = tf.reshape(
        x_dec1_out, dec1_reshape, name="dec1_reshape")
    x_dec2_features = tf.reshape(
        x_dec2_out, dec2_reshape, name="dec2_reshape")
    x_dec3_features = tf.reshape(
        x_dec3_out, dec3_reshape, name="dec3_reshape")
    x_dec5_features = tf.reshape(
        x_dec5_out, dec5_reshape, name="dec5_reshape")
    x_dec6_features = tf.reshape(
        x_dec6_out, dec6_reshape, name="dec6_reshape")
    
    x_features = tf.concat(
        [x_dec1_features, x_dec2_features, 
         x_dec3_features, x_dec4_out, 
         x_dec5_features, x_dec6_features], axis=3)
    
    # 4 scales with 4 regression coordinates with  #
    # n_classes classification probabilities gives #
    # = 4*(4 + n_classes) output filters.          #
    output_reshape = [
        batch_sz, img_rows, img_cols, 4, 4 + n_classes]
    
    x_head_out = layers.Conv2D(
        4*(4 + n_classes), (3, 3), 
        strides=(1, 1), padding="same", 
        activation=None, name="head_out")(x_features)
    x_head_out = tf.reshape(
        x_head_out, output_reshape, name="head_out_reshape")
    
    # Get the regression and classification outputs. #
    reg_heads = \
        tf.nn.sigmoid(x_head_out[:, :, :, :, :4])
    cls_heads = x_head_out[:, :, :, :, 4:] + b_focal
    
    x_outputs = tf.concat([reg_heads, cls_heads], axis=4)
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
    return tf.reduce_sum(foc_loss_stable, axis=[1, 2, 3])

def model_loss(
    bboxes, masks, outputs, loss_type="sigmoid"):
    reg_weight = tf.expand_dims(masks, axis=4)
    reg_output = outputs[:, :, :, :, :4]
    cls_output = outputs[:, :, :, :, 4:]
    cls_labels = tf.cast(bboxes[:, :, :, :, 4:], tf.int32)
    
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
    cls_lambda=2.5, loss_type="focal"):
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
    img_in_file, voc_model, labels, 
    heatmap=True, thresh=0.50, 
    transpose=False, img_title=None, img_box=None, 
    img_rows=448, img_cols=448, img_scale=None, 
    save_img_file="object_detection_result.jpg"):
    if img_scale is None:
        if max(img_rows, img_cols) < 512:
            max_scale = max(img_rows, img_cols)
        else:
            max_scale = 512
        img_scale = [64, 128, 256, max_scale]
    else:
        if len(img_scale) != 4:
            raise ValueError("img_scale must be size 4.")
    
    # Read the image. #
    image_resized = tf.expand_dims(_parse_image(
        img_in_file, img_rows=img_rows, img_cols=img_cols), axis=0)
    
    tmp_output = \
        voc_model.predict(image_resized)
    if transpose:
        tmp_output = tf.transpose(tmp_output, [0, 2, 1, 3, 4])
        if img_box is not None:
            tmp_box = tf.transpose(img_box, [1, 0, 2, 3])
            img_box = tmp_box.numpy()
            
            img_box[:, :, :, 0] = tmp_box[:, :, :, 1]
            img_box[:, :, :, 1] = tmp_box[:, :, :, 0]
            img_box[:, :, :, 2] = tmp_box[:, :, :, 3]
            img_box[:, :, :, 3] = tmp_box[:, :, :, 2]
            img_box = tf.constant(img_box)
            del tmp_box
    
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
    
    if img_title is not None:
        fig.suptitle(img_title)
    fig.savefig(save_img_file, dpi=199)
    plt.close()
    del fig, ax
    return None

