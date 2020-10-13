
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

def mb_conv_block(
    x_cnn_input, kernel_size, strides, 
    in_filters, out_filters, expand_ratio, blk_name):
    mb_ker_size = (kernel_size, kernel_size)
    mb_strides  = (strides, strides)
    exp_filters = int(expand_ratio * in_filters)
    
    x_mb_input = layers.Conv2D(
        exp_filters, (1,1), strides=(1,1), 
        padding="same", use_bias=False, 
        activation=None, name=blk_name+"_in_conv")(x_cnn_input)
    
    x_mb_input = layers.BatchNormalization(
        name=blk_name+"_bn_input")(x_mb_input)
    x_mb_input = layers.ReLU(
        name=blk_name+"_relu_input")(x_mb_input)
    
    # Depthwise Convolution. #
    x_dw_out = layers.SeparableConv2D(
        exp_filters, mb_ker_size, 
        strides=mb_strides, padding="same", 
        use_bias=False, activation=None, 
        name=blk_name+"seperable_conv")(x_mb_input)
    x_dw_out = layers.BatchNormalization(
        name=blk_name+"_bn_depth")(x_dw_out)
    x_dw_out = layers.ReLU(
        name=blk_name+"_relu_depth")(x_dw_out)
    
    # Output. #
    x_mb_output = layers.Conv2D(
        out_filters, (1,1), strides=(1,1), 
        padding="same", use_bias=False, 
        activation=None, name=blk_name+"_out_conv")(x_dw_out)
    x_mb_output = layers.BatchNormalization(
        name=blk_name+"_bn_output")(x_mb_output)
    
    # Residual Connection. #
    if in_filters == out_filters:
        x_mb_output = x_cnn_input + x_mb_output
    return x_mb_output

def build_model(
    n_repeats, w_ratio, n_classes, tmp_pi=0.99, 
    img_rows=448, img_cols=448, batch_norm=True):
    tmp_b = tf.math.log((1.0-tmp_pi)/tmp_pi)
    img_w = int(img_rows/8)
    img_h = int(img_cols/8)
    
    b_focal = tf.Variable(
        tmp_b, trainable=True, name="b_focal")
    x_input = tf.keras.Input(
        shape=(img_rows, img_cols, 3), name="x_input")
    
    # Block 0. #
    n0_filters = int(w_ratio * 32)
    x_blk0_out = layers.Conv2D(
        n0_filters, (3,3), strides=(2,2), 
        padding="same", use_bias=False, 
        activation=None, name="blk_0")(x_input)
    
    # EfficientNet Backbone. #
    # Block 1. #
    n1_filters = int(w_ratio * 16)
    x_blk1_in  = x_blk0_out
    for n_round in range(n_repeats):
        blk_name = "blk1_depth_" + str(n_round+1)
        if n_round == 0:
            in_filters = n0_filters
        else:
            in_filters = n1_filters
        
        # MB Block. #
        tmp_out = mb_conv_block(
            x_blk1_in, 3, 1, 
            in_filters, n1_filters, 1, blk_name)
        
        # Residual Connection. #
        if n_round == 0:
            x_blk1_out = tmp_out
        else:
            x_blk1_out = x_blk1_in + tmp_out
        x_blk1_in = x_blk1_out
    
    # Squeeze and Excite layer. #
    x_blk1_squeeze = \
        layers.GlobalAveragePooling2D()(x_blk1_out)
    x_blk1_sigmoid = tf.nn.sigmoid(tf.expand_dims(
        tf.expand_dims(x_blk1_squeeze, axis=1), axis=1))
    x_blk1_out = x_blk1_sigmoid * x_blk1_out
    
    # Block 2. #
    n2_filters = int(w_ratio * 24)
    x_blk2_in  = x_blk1_out
    for n_round in range(n_repeats):
        blk_name = "blk2_depth_" + str(n_round+1)
        if n_round == 0:
            in_filters = n1_filters
        else:
            in_filters = n2_filters
        
        # MB Block. #
        tmp_out = mb_conv_block(
            x_blk2_in, 3, 1, 
            in_filters, n2_filters, 6, blk_name)
        
        # Residual Connection. #
        if n_round == 0:
            x_blk2_out = tmp_out
        else:
            x_blk2_out = x_blk2_in + tmp_out
        x_blk2_in = x_blk2_out
    
    # Squeeze and Excite layer. #
    x_blk2_squeeze = \
        layers.GlobalAveragePooling2D()(x_blk2_out)
    x_blk2_sigmoid = tf.nn.sigmoid(tf.expand_dims(
        tf.expand_dims(x_blk2_squeeze, axis=1), axis=1))
    x_blk2_out = x_blk2_sigmoid * x_blk2_out
    
    # Downsample. #
    x_blk2_out = layers.Conv2D(
        n2_filters, (3,3), strides=(2,2), 
        padding="same", use_bias=False, 
        activation=None, name="blk2_down")(x_blk2_out)
    
    # Block 3. #
    n3_filters = int(w_ratio * 40)
    x_blk3_in  = x_blk2_out
    for n_round in range(n_repeats):
        blk_name = "blk3_depth_" + str(n_round+1)
        if n_round == 0:
            in_filters = n2_filters
        else:
            in_filters = n3_filters
        
        # MB Block. #
        tmp_out = mb_conv_block(
            x_blk3_in, 3, 1, 
            in_filters, n3_filters, 6, blk_name)
        
        # Residual Connection. #
        if n_round == 0:
            x_blk3_out = tmp_out
        else:
            x_blk3_out = x_blk3_in + tmp_out
        x_blk3_in = x_blk3_out
    
    # Squeeze and Excite layer. #
    x_blk3_squeeze = \
        layers.GlobalAveragePooling2D()(x_blk3_out)
    x_blk3_sigmoid = tf.nn.sigmoid(tf.expand_dims(
        tf.expand_dims(x_blk3_squeeze, axis=1), axis=1))
    x_blk3_out = x_blk3_sigmoid * x_blk3_out
    
    # Downsample. #
    x_blk3_out = layers.Conv2D(
        n3_filters, (3,3), strides=(2,2), 
        padding="same", use_bias=False, 
        activation=None, name="blk3_down")(x_blk3_out)
    
    # Block 4. #
    n4_filters = int(w_ratio * 80)
    x_blk4_in  = x_blk3_out
    for n_round in range(n_repeats):
        blk_name = "blk4_depth_" + str(n_round+1)
        if n_round == 0:
            in_filters = n3_filters
        else:
            in_filters = n4_filters
        
        # MB Block. #
        tmp_out = mb_conv_block(
            x_blk4_in, 3, 1, 
            in_filters, n4_filters, 6, blk_name)
        
        # Residual Connection. #
        if n_round == 0:
            x_blk4_out = tmp_out
        else:
            x_blk4_out = x_blk4_in + tmp_out
        x_blk4_in = x_blk4_out
    
    # Squeeze and Excite layer. #
    x_blk4_squeeze = \
        layers.GlobalAveragePooling2D()(x_blk4_out)
    x_blk4_sigmoid = tf.nn.sigmoid(tf.expand_dims(
        tf.expand_dims(x_blk4_squeeze, axis=1), axis=1))
    x_blk4_out = x_blk4_sigmoid * x_blk4_out
    
    # Downsample. #
    x_blk4_out = layers.Conv2D(
        n4_filters, (3,3), strides=(2,2), 
        padding="same", use_bias=False, 
        activation=None, name="blk4_down")(x_blk4_out)
    
    # Block 5. #
    n5_filters = int(w_ratio * 112)
    x_blk5_in  = x_blk4_out
    for n_round in range(n_repeats):
        blk_name = "blk5_depth_" + str(n_round+1)
        if n_round == 0:
            in_filters = n4_filters
        else:
            in_filters = n5_filters
        
        # MB Block. #
        tmp_out = mb_conv_block(
            x_blk5_in, 3, 1, 
            in_filters, n5_filters, 6, blk_name)
        
        # Residual Connection. #
        if n_round == 0:
            x_blk5_out = tmp_out
        else:
            x_blk5_out = x_blk5_in + tmp_out
        x_blk5_in = x_blk5_out
    
    # Squeeze and Excite layer. #
    x_blk5_squeeze = \
        layers.GlobalAveragePooling2D()(x_blk5_out)
    x_blk5_sigmoid = tf.nn.sigmoid(tf.expand_dims(
        tf.expand_dims(x_blk5_squeeze, axis=1), axis=1))
    x_blk5_out = x_blk5_sigmoid * x_blk5_out
    
    # Downsample. #
    x_blk5_out = layers.Conv2D(
        n5_filters, (3,3), strides=(2,2), 
        padding="same", use_bias=False, 
        activation=None, name="blk5_down")(x_blk5_out)
    
    # Block 6. #
    n6_filters = int(w_ratio * 192)
    x_blk6_in  = x_blk5_out
    for n_round in range(n_repeats):
        blk_name = "blk6_depth_" + str(n_round+1)
        if n_round == 0:
            in_filters = n5_filters
        else:
            in_filters = n6_filters
        
        # MB Block. #
        tmp_out = mb_conv_block(
            x_blk6_in, 3, 1, 
            in_filters, n6_filters, 6, blk_name)
        
        # Residual Connection. #
        if n_round == 0:
            x_blk6_out = tmp_out
        else:
            x_blk6_out = x_blk6_in + tmp_out
        x_blk6_in = x_blk6_out
    
    # Squeeze and Excite layer. #
    x_blk6_squeeze = \
        layers.GlobalAveragePooling2D()(x_blk6_out)
    x_blk6_sigmoid = tf.nn.sigmoid(tf.expand_dims(
        tf.expand_dims(x_blk6_squeeze, axis=1), axis=1))
    x_blk6_out = x_blk6_sigmoid * x_blk6_out
    
    # Downsample. #
    x_blk6_out = layers.Conv2D(
        n6_filters, (3,3), strides=(2,2), 
        padding="same", use_bias=False, 
        activation=None, name="blk6_down")(x_blk6_out)
    
    # Upsample network for last layer. #
    x_upsample_1 = layers.UpSampling2D(
        interpolation="bilinear")(x_blk6_out)
    x_ups_cnn_1  = layers.Conv2D(
        n5_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="ups_cnn_1")(x_upsample_1)
    x_ups_norm_1 = layers.BatchNormalization(
        name="ups_cnn_bnorm_1")(x_ups_cnn_1)
    x_ups_relu_1 = layers.ReLU(name="ups_relu_1")(x_ups_norm_1)
    
    x_upsample_2 = layers.UpSampling2D(
        interpolation="bilinear")(x_ups_relu_1)
    x_ups_cnn_2  = layers.Conv2D(
        n4_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="ups_cnn_2")(x_upsample_2)
    x_ups_norm_2 = layers.BatchNormalization(
        name="ups_cnn_bnorm_2")(x_ups_cnn_2)
    x_ups_relu_2 = layers.ReLU(name="ups_relu_2")(x_ups_norm_2)
    
    x_upsample_3 =layers.UpSampling2D(
        interpolation="bilinear")(x_ups_relu_2)
    x_ups_cnn_3  = layers.Conv2D(
        n3_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="ups_cnn_3")(x_upsample_3)
    x_ups_norm_3 = layers.BatchNormalization(
        name="ups_cnn_bnorm_3")(x_ups_cnn_3)
    x_cnn_vlarge = layers.ReLU(
        name="cnn_vlarge_relu")(x_ups_norm_3)
    
    # Feature Pyramid Network. #
    x_fpn_large  = x_ups_relu_1 + x_blk5_out
    x_fpn_ups_1 = layers.UpSampling2D(
        interpolation="bilinear")(x_fpn_large)
    x_fpn_cnn_1  = layers.Conv2D(
        n4_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="fpn_cnn_1")(x_fpn_ups_1)
    x_fpn_norm_1 = layers.BatchNormalization(
        name="fpn_bnorm_1")(x_fpn_cnn_1)
    x_fpn_relu_1 = layers.ReLU(name="fpn_relu_1")(x_fpn_norm_1)
    
    x_fpn_ups_2  = layers.UpSampling2D(
        interpolation="bilinear")(x_fpn_relu_1)
    x_fpn_cnn_2  = layers.Conv2D(
        n3_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="fpn_cnn_2")(x_fpn_ups_2)
    x_fpn_norm_2 = layers.BatchNormalization(
        name="fpn_bnorm_2")(x_fpn_cnn_2)
    x_fpn_relu_2 = layers.ReLU(name="fpn_relu_2")(x_fpn_norm_2)
    
    x_fpn_medium = x_fpn_relu_1 + x_blk4_out
    x_fpn_ups_3  = layers.UpSampling2D(
        interpolation="bilinear")(x_fpn_medium)
    x_fpn_cnn_3  = layers.Conv2D(
        n3_filters, (3, 3), strides=(1, 1), padding="same", 
        activation="linear", name="fpn_cnn_3")(x_fpn_ups_3)
    x_fpn_norm_3 = layers.BatchNormalization(
        name="fpn_bnorm_3")(x_fpn_cnn_3)
    x_fpn_relu_3 = layers.ReLU(name="fpn_relu_3")(x_fpn_norm_3)
    
    x_features = tf.concat(
        [x_blk3_out, x_fpn_relu_3, 
         x_fpn_relu_2, x_cnn_vlarge], axis=3)
    
    # 4 scales with 4 coordinates = 16 filters for regression head. #
    x_cnn_output = layers.Conv2D(
        4*(4 + n_classes), (3, 3), 
        strides=(1, 1), padding="same", 
        activation="linear", name="cnn_output")(x_features)
    
    # Reshape the outputs. #
    batch_size = tf.shape(x_input)[0]
    out_shape  = [batch_size, img_w, img_h, 4, 4 + n_classes]
    
    out_heads = tf.reshape(x_cnn_output, out_shape)
    reg_heads = tf.nn.sigmoid(out_heads[:, :, :, :, :4])
    cls_heads = out_heads[:, :, :, :, 4:] + b_focal
    
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
    return tf.reduce_sum(foc_loss_stable, axis=[1, 2, 3, 4])

def model_loss(
    bboxes, masks, outputs, img_size=448, 
    reg_lambda=0.10, loss_type="sigmoid", eps=1.0e-6):
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
    heatmap=True, thresh=0.50, img_title=None, 
    img_rows=448, img_cols=448, img_scale=None, 
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
    image_resized = tf.expand_dims(_parse_image(
        img_in_file, img_rows=img_rows, img_cols=img_cols), axis=0)
    
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
    
    if img_title is not None:
        fig.suptitle(img_title)
    fig.savefig(save_img_file, dpi=199)
    plt.close()
    del fig, ax
    return None

