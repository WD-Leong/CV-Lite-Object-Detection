import numpy as np
import tensorflow as tf
from tf_bias_layer import BiasLayer
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

def cnn_block(
    x_cnn_input, n_filters, ker_sz, stride, 
    blk_name, n_repeats=1, seperable=True, 
    batch_norm=True, norm_order="norm_first"):
    n_channels = 2*n_filters
    kernel_sz  = (ker_sz, ker_sz)
    cnn_stride = (stride, stride)
    
    tmp_input = x_cnn_input
    for n_repeat in range(n_repeats):
        bnorm_name = "_bn_" + str(n_repeat)
        cnn_name = blk_name + "_cnn_" + str(n_repeat)
        bot_name = blk_name + "_bot_" + str(n_repeat)
        out_name = blk_name + "_out_" + str(n_repeat)
        act_name = blk_name + "_relu_" + str(n_repeat)
        
        if norm_order == "norm_first":
            if batch_norm:
                tmp_input = layers.BatchNormalization(
                    name=blk_name+bnorm_name)(tmp_input)
        
        if seperable:
            # Bottleneck layer. #
            tmp_output = layers.SeparableConv2D(
                n_filters, (1, 1), 
                strides=(1, 1), padding="same", 
                activation=None, name=bot_name)(tmp_input)
            
            tmp_output = layers.SeparableConv2D(
                n_filters, kernel_sz, 
                strides=cnn_stride, padding="same", 
                activation=None, name=cnn_name)(tmp_output)
            
            tmp_output = layers.SeparableConv2D(
                n_channels, (1, 1), 
                strides=(1, 1), padding="same", 
                activation=None, name=out_name)(tmp_output)
        else:
            tmp_output = layers.Conv2D(
                n_filters, (1, 1), 
                strides=(1, 1), padding="same", 
                activation=None, name=bot_name)(tmp_input)
            
            tmp_output = layers.Conv2D(
                n_filters, kernel_sz, 
                strides=cnn_stride, padding="same", 
                activation=None, name=cnn_name)(tmp_output)
            
            tmp_output = layers.Conv2D(
                n_channels, (1, 1), 
                strides=(1, 1), padding="same", 
                activation=None, name=out_name)(tmp_output)
        
        if norm_order == "norm_last":
            if batch_norm:
                tmp_bnorm = layers.BatchNormalization(
                    name=blk_name+bnorm_name)(tmp_output)
                tmp_relu  = layers.ReLU(name=act_name)(tmp_bnorm)
            else:
                tmp_relu = layers.ReLU(name=act_name)(tmp_output)
        else:
            tmp_relu = layers.ReLU(name=act_name)(tmp_output)
        
        # Residual Layer. #
        if n_repeat == 0:
            res_output = tmp_relu
        else:
            res_output = tmp_relu + tmp_input
        tmp_input = res_output
    return res_output

def downsample_block(x_cnn_input, name="max_pool"):
    return layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), 
        padding="same", name=name)(x_cnn_input)

def build_model(
    n_classes, tmp_pi=0.99, n_filters=128, 
    n_stacks=1, n_repeats=2, seperable=True, 
    batch_norm=True, norm_order="norm_first"):
    tmp_b = tf.math.log((1.0-tmp_pi)/tmp_pi)
    
    b_focal = BiasLayer(
        bias_init=tmp_b, 
        trainable=True, name="b_focal")
    x_input = tf.keras.Input(
        shape=(None, None, 3), name="x_input")
    
    # Block 0. #
    kernel_sz_0 = (7,7)
    if seperable:
        x_blk0_out = layers.SeparableConv2D(
            n_filters, kernel_sz_0, 
            strides=(2,2), padding="same", 
            activation=None, name="cnn_block_0")(x_input)
    else:
        x_blk0_out = layers.Conv2D(
            n_filters, kernel_sz_0, 
            strides=(2,2), padding="same", 
            activation=None, name="cnn_block_0")(x_input)
    
    # Process one more time. #
    x_cnn1_out = cnn_block(
        x_blk0_out, n_filters, 3, 1, "cnn_block_1", 
        n_repeats=n_repeats, norm_order=norm_order, 
        seperable=seperable, batch_norm=batch_norm)
    
    x_blk1_out = downsample_block(
        x_cnn1_out, name="max_pool_1")
    
    # Hourglass network. #
    x_stack_input = x_blk1_out
    for n_stack in range(n_stacks):
        stack_name = "stack_" + str(n_stack+1) + "_"
        
        enc_1_name = stack_name + "enc_block_1"
        enc_2_name = stack_name + "enc_block_2"
        enc_3_name = stack_name + "enc_block_3"
        enc_4_name = stack_name + "enc_block_4"
        
        dec_1_name = stack_name + "dec_block_1"
        dec_2_name = stack_name + "dec_block_2"
        dec_3_name = stack_name + "dec_block_3"
        dec_4_name = stack_name + "dec_block_4"
        
        out_1_name = stack_name + "dec_out_1"
        out_2_name = stack_name + "dec_out_2"
        out_3_name = stack_name + "dec_out_3"
        out_4_name = stack_name + "dec_out_4"
        
        # Encoder Network. #
        # Block 1. #
        x_enc1_cnn = cnn_block(
            x_stack_input, n_filters, 3, 1, enc_1_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Residual layer. #
        x_enc1_res = x_stack_input + x_enc1_cnn
        x_enc1_out = downsample_block(
            x_enc1_res, name=stack_name+"enc_out_1")
        
        # Block 2. #
        x_enc2_cnn= cnn_block(
            x_enc1_out, n_filters, 3, 1, enc_2_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Residual layer. #
        x_enc2_res = x_enc1_out + x_enc2_cnn
        x_enc2_out = downsample_block(
            x_enc2_res, name=stack_name+"enc_out_2")
        
        # Block 3. #
        x_enc3_cnn = cnn_block(
            x_enc2_out, n_filters, 3, 1, enc_3_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Residual layer. #
        x_enc3_res = x_enc2_out + x_enc3_cnn
        x_enc3_out = downsample_block(
            x_enc3_res, name=stack_name+"enc_out_3")
        
        # Block 4. #
        enc_4a_name = enc_4_name + "a"
        x_enc4a_cnn = cnn_block(
            x_enc3_out, n_filters, 3, 1, enc_4a_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        enc_4b_name = enc_4_name + "b"
        x_enc4b_cnn = cnn_block(
            x_enc4a_cnn, n_filters, 3, 1, enc_4b_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        x_enc4_cnn = cnn_block(
            x_enc4b_cnn, n_filters, 3, 1, enc_4_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Residual layer. #
        x_enc4_res = x_enc3_out + x_enc4_cnn
        x_enc4_out = downsample_block(
            x_enc4_res, name=stack_name+"enc_out_4")
        
        # Decoder Network. #
        # Block 1. #
        x_ups1_out = layers.UpSampling2D(
            interpolation="bilinear")(x_enc4_out)
        
        x_enc_dec1 = cnn_block(
            x_enc3_out, n_filters, 3, 1, dec_1_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        x_dec1_res = x_enc_dec1 + x_ups1_out
        x_dec1_out = cnn_block(
            x_dec1_res, n_filters, 3, 1, out_1_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Block 2. #
        x_ups2_out = layers.UpSampling2D(
            interpolation="bilinear")(x_dec1_out)
        
        x_enc_dec2 = cnn_block(
            x_enc2_out, n_filters, 3, 1, dec_2_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        x_dec2_res = x_enc_dec2 + x_ups2_out
        x_dec2_out = cnn_block(
            x_dec2_res, n_filters, 3, 1, out_2_name,  
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Block 3. #
        x_ups3_out = layers.UpSampling2D(
            interpolation="bilinear")(x_dec2_out)
        
        x_enc_dec3 = cnn_block(
            x_enc1_out, n_filters, 3, 1, dec_3_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        x_dec3_res = x_enc_dec3 + x_ups3_out
        x_dec3_out = cnn_block(
            x_dec3_res, n_filters, 3, 1, out_3_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Block 4. #
        x_ups4_out = layers.UpSampling2D(
            interpolation="bilinear")(x_dec3_out)
        
        x_enc_dec4 = cnn_block(
            x_stack_input, n_filters, 3, 1, dec_4_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        x_dec4_res = x_enc_dec4 + x_ups4_out
        x_dec4_out = cnn_block(
            x_dec4_res, n_filters, 3, 1, out_4_name, 
            n_repeats=n_repeats, norm_order=norm_order, 
            seperable=seperable, batch_norm=batch_norm)
        
        # Output of this stack is passed as input #
        # of the next stack in a stacked network. #
        x_stack_input = x_dec4_out
    
    # Output CNN layer. #
    x_cnn_out = layers.Conv2D(
        4 + n_classes, (3, 3), 
        strides=(1, 1), padding="same", 
        activation=None, name="cnn_out")(x_dec4_out)
    
    # Get the regression and classification outputs. #
    reg_heads = x_cnn_out[:, :, :, :4]
    cls_heads = b_focal(x_cnn_out[:, :, :, 4:])
    
    x_outputs = tf.concat(
        [reg_heads, cls_heads], axis=3)
    obj_model = tf.keras.Model(
        inputs=x_input, outputs=x_outputs)
    return obj_model

def prediction_to_corners(xy_pred, stride):
    feat_dims  = [tf.shape(xy_pred)[0], 
                  tf.shape(xy_pred)[1]]
    bbox_shape = [int(xy_pred.shape[0]), 
                  int(xy_pred.shape[1]), 4]
    bbox_coord = np.zeros(bbox_shape)
    
    ch = tf.range(0., tf.cast(
        feat_dims[0], tf.float32), dtype=tf.float32) + 0.5
    cw = tf.range(0., tf.cast(
        feat_dims[1], tf.float32), dtype=tf.float32) + 0.5
    [grid_x, grid_y] = tf.meshgrid(cw, ch)
    
    pred_x_low = grid_x - xy_pred[..., 2]
    pred_x_upp = grid_x + xy_pred[..., 3]
    pred_y_low = grid_y - xy_pred[..., 0]
    pred_y_upp = grid_y + xy_pred[..., 1]
    
    bbox_coord[:, :, 0] = pred_y_low
    bbox_coord[:, :, 2] = pred_y_upp
    bbox_coord[:, :, 1] = pred_x_low
    bbox_coord[:, :, 3] = pred_x_upp
    return stride*bbox_coord

def format_data(
    gt_labels, img_dim, 
    num_classes, img_pad=None, stride=8):
    """
    gt_labels: Normalised Gound Truth Bounding Boxes (y, x, h, w).
    num_targets is for debugging purposes.
    """
    if img_pad is None:
        img_pad = img_dim
    
    gt_height = gt_labels[:, 2]*img_dim[0]
    gt_width  = gt_labels[:, 3]*img_dim[1]
    gt_height = gt_height.numpy()
    gt_width  = gt_width.numpy()
    
    # Format the ground truth map. #
    h_max = int(img_pad[1] / stride)
    w_max = int(img_pad[0] / stride)
    pad_y = int((img_pad[1] - img_dim[1]) / 2.0)
    pad_x = int((img_pad[0] - img_dim[0]) / 2.0)
    
    tmp_output = np.zeros(
        [h_max, w_max, num_classes+4])
    if len(gt_labels) == 0:
        # This should not be occurring. #
        num_targets = 0
    else:
        # Sort the labels by area from largest to smallest.     #
        # Then the smallest area will automatically overwrite   #
        # any overlapping grid positions since it is the last   #
        # to be filled up.                                      #
        
        # For FCOS, fill up all grid positions which the bounding #
        # box occupies in the feature map. Note that we also clip #
        # the (l, r, t, b) values as it may have negative values  #
        # due to the integer floor operation. This could cause    #
        # the (l, r, t, b) computation to return negative values  #
        # and usually occurs when the object is near or at the    #
        # edge of the image.                                      #
        
        tmp_labels = gt_labels.numpy()
        if len(tmp_labels) == 1:
            tmp_sorted = tmp_labels
        else:
            tmp_box_areas = np.multiply(
                tmp_labels[:, 2]*img_dim[0], 
                tmp_labels[:, 3]*img_dim[1])
            
            tmp_sorted = \
                tmp_labels[np.argsort(tmp_box_areas)]
        
        for n_label in range(len(tmp_sorted)):
            tmp_label = tmp_sorted[n_label]
            tmp_coord = [
                (tmp_label[0] - 0.5*tmp_label[2])*img_dim[0], 
                (tmp_label[1] - 0.5*tmp_label[3])*img_dim[1], 
                (tmp_label[0] + 0.5*tmp_label[2])*img_dim[0], 
                (tmp_label[1] + 0.5*tmp_label[3])*img_dim[1]]
            
            tmp_y_cen = (tmp_coord[0] + tmp_coord[2]) / 2.0
            tmp_x_cen = (tmp_coord[1] + tmp_coord[3]) / 2.0
            tmp_y_cen = int((pad_y + tmp_y_cen) / stride)
            tmp_x_cen = int((pad_x + tmp_x_cen) / stride)
            idx_class = 4 + int(tmp_label[4])
            
            # Set bounding box coordinates. #
            box_offsets = [
                tmp_y_cen + 0.5 - (pad_y + tmp_coord[0])/stride, 
                (pad_y + tmp_coord[2])/stride - tmp_y_cen - 0.5, 
                tmp_x_cen + 0.5 - (pad_x + tmp_coord[1])/stride, 
                (pad_x + tmp_coord[3])/stride - tmp_x_cen - 0.5]
            
            tmp_output[tmp_y_cen, tmp_x_cen, :4] = box_offsets
            tmp_output[
                tmp_y_cen, tmp_x_cen, idx_class] = 1.0
        
        num_targets = len(tmp_labels)
    return tmp_output, num_targets

def smooth_l1_loss(xy_true, xy_pred, mask=1.0, delta=1.0):
    mask = tf.expand_dims(mask, axis=-1)
    raw_diff = xy_true - xy_pred
    sq_diff  = tf.square(raw_diff)
    abs_diff = tf.abs(raw_diff)
    
    smooth_l1_loss = tf.where(
        tf.less(abs_diff, delta), 
        0.5 * sq_diff, abs_diff)
    smooth_l1_loss = tf.reduce_sum(tf.reduce_sum(
        tf.multiply(smooth_l1_loss, mask), axis=-1))
    return smooth_l1_loss

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
    return tf.reduce_sum(foc_loss_stable)

def model_loss(y_true, y_pred):
    """
    y_true: Normalised Gound Truth Bounding Boxes (x, y, w, h).
    """
    tmp_obj  = tf.reduce_max(
        y_true[..., 4:], axis=-1)
    tmp_mask = tf.cast(tmp_obj > 0, tf.float32)
    
    cls_loss = focal_loss(
        y_true[..., 4:], y_pred[..., 4:])
    
    reg_loss = smooth_l1_loss(
        y_true[..., :4], y_pred[..., :4], mask=tmp_mask)
    return cls_loss, reg_loss

def train_step(
    voc_model, sub_batch_sz, 
    images, bboxes, optimizer, 
    cls_lambda=2.5, reg_lambda=1.0, 
    learning_rate=1.0e-3, grad_clip=1.0):
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
        tmp_bboxes = bboxes[id_st:id_en, :, :, :]
        
        with tf.GradientTape() as voc_tape:
            tmp_output = voc_model(tmp_images, training=True)
            tmp_losses = model_loss(tmp_bboxes, tmp_output)
            
            tmp_cls_loss += tmp_losses[0]
            tmp_reg_loss += tmp_losses[1]
            total_losses = tf.add(
                cls_lambda*tmp_losses[0], 
                reg_lambda*tmp_losses[1])
        
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
    downsample=8, iou_thresh=0.213, img_box=None, 
    img_title=None, img_rows=448, img_cols=448, 
    save_img_file="object_detection_result.jpg"):
    # Read the image. #
    image_resized = _parse_image(
        img_in_file, img_rows=img_rows, img_cols=img_cols)
    
    tmp_output = voc_model.predict(
        tf.expand_dims(image_resized, axis=0))
    reg_output = prediction_to_corners(
        tmp_output[0, :, :, :4], downsample)
    cls_output = tmp_output[0, :, :, 4:]
    cls_probs  = tf.nn.sigmoid(cls_output)
    n_classes  = cls_output.shape[2]
    
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
            obj_probs = tf.reduce_max(cls_probs, axis=2)
        else:
            obj_probs = cls_probs[:, :, 0]
        
        obj_probs = tf.image.resize(tf.expand_dims(
            obj_probs, axis=2), [img_width, img_height])
        tmp = ax.imshow(tf.squeeze(
            obj_probs, axis=2), "jet", alpha=0.50)
        fig.colorbar(tmp, ax=ax)
    
    if n_classes > 1:
        prob_max = tf.reduce_max(cls_probs, axis=2)
        pred_label = tf.math.argmax(cls_probs, axis=2)
    else:
        prob_max = cls_probs[:, :, 0]
    tmp_thresh = \
        np.where(prob_max >= thresh, 1, 0)
    tmp_coords = np.nonzero(tmp_thresh)
    
    tmp_obj_detect = []
    for n_box in range(len(tmp_coords[0])):
        x_coord = tmp_coords[0][n_box]
        y_coord = tmp_coords[1][n_box]
        
        tmp_boxes = reg_output[x_coord, y_coord, :]
        tmp_probs = int(
            prob_max[x_coord, y_coord].numpy()*100)
        if n_classes > 1:
            tmp_label = \
                pred_label[x_coord, y_coord].numpy()
        else:
            tmp_label = 0
        
        x_low = tmp_h_ratio * tmp_boxes[1]
        y_low = tmp_w_ratio * tmp_boxes[0]
        x_upp = tmp_h_ratio * tmp_boxes[3]
        y_upp = tmp_w_ratio * tmp_boxes[2]
        
        box_w = x_upp - x_low
        box_h = y_upp - y_low
        if box_w > img_width:
            box_w = img_width
        if box_h > img_height:
            box_h = img_height
        
        # Output prediction is transposed. #
        if x_low < 0:
            x_low = 0
        if y_low < 0:
            y_low = 0
        
        tmp_bbox = np.array([
            x_low, y_low, box_w, box_h, tmp_probs, tmp_label])
        tmp_obj_detect.append(np.expand_dims(tmp_bbox, axis=0))
    
    if len(tmp_obj_detect) > 0:
        bboxes_raw = np.concatenate(
            tuple(tmp_obj_detect), axis=0)
        bboxes_nms = nms(bboxes_raw, iou_thresh, method='nms')
        for tmp_obj in bboxes_nms:
            box_w = tmp_obj[2] - tmp_obj[0]
            box_h = tmp_obj[3] - tmp_obj[1]
            box_patch  = plt.Rectangle(
                (tmp_obj[0], tmp_obj[1]), box_w, box_h, 
                linewidth=1, edgecolor="red", fill=None)
            
            tmp_label = str(labels[int(tmp_obj[5])])
            tmp_text  = \
                tmp_label + ": " + str(tmp_obj[4]) + "%"
            
            ax.add_patch(box_patch)
            ax.text(tmp_obj[0], tmp_obj[1], 
                    tmp_text, fontsize=5, color="red")
        print(str(len(bboxes_nms)), "objects detected.")
    else:
        print("0 objects detected.")
    
    # True image is not transposed. #
    if img_box is not None:
        tmp_true_box = np.nonzero(
            tf.reduce_max(img_box[:, :, 4:], axis=2))
        tmp_bbox_reg = prediction_to_corners(
            img_box[:, :, :4], downsample)
        for n_box in range(len(tmp_true_box[0])):
            x_coord = tmp_true_box[0][n_box]
            y_coord = tmp_true_box[1][n_box]
            tmp_boxes = tmp_bbox_reg[x_coord, y_coord, :4]
            
            x_low = tmp_w_ratio * tmp_boxes[1]
            y_low = tmp_h_ratio * tmp_boxes[0]
            x_upp = tmp_w_ratio * tmp_boxes[3]
            y_upp = tmp_h_ratio * tmp_boxes[2]
            
            box_w = x_upp - x_low
            box_h = y_upp - y_low
            box_patch = plt.Rectangle(
                (x_low, y_low), 
                box_w, box_h, linewidth=1, 
                edgecolor="white", fill=None)
            ax.add_patch(box_patch)
    
    if img_title is not None:
        fig.suptitle(img_title)
    fig.savefig(save_img_file, dpi=199)
    plt.close()
    del fig, ax
    return None

def show_object_boxes(
    img_array, img_box, img_dims, 
    downsample=8, save_img_file="ground_truth.jpg"):
    # Plot the bounding boxes on the image. #
    fig, ax = plt.subplots(1)
    ax.imshow(img_array)
    
    tmp_w_ratio = 1.0
    tmp_h_ratio = 1.0
    
    obj_probs = tf.reduce_max(img_box[:, :, 4:], axis=2)
    obj_probs = tf.image.resize(tf.expand_dims(
        obj_probs, axis=2), [img_dims, img_dims])
    tmp = ax.imshow(tf.squeeze(
        obj_probs, axis=2), "jet", alpha=0.50)
    fig.colorbar(tmp, ax=ax)
    
    # True image is not transposed. #
    tmp_true_box = np.nonzero(
        tf.reduce_max(img_box[:, :, 4:], axis=2))
    tmp_bbox_reg = prediction_to_corners(img_box, downsample)
    for n_box in range(len(tmp_true_box[0])):
        x_coord = tmp_true_box[0][n_box]
        y_coord = tmp_true_box[1][n_box]
        tmp_boxes = tmp_bbox_reg[x_coord, y_coord, :4]
        
        x_low = tmp_w_ratio * tmp_boxes[1]
        y_low = tmp_h_ratio * tmp_boxes[0]
        x_upp = tmp_w_ratio * tmp_boxes[3]
        y_upp = tmp_h_ratio * tmp_boxes[2]
        box_w = x_upp - x_low
        box_h = y_upp - y_low
        box_patch = plt.Rectangle(
            (x_low, y_low), 
            box_w, box_h, linewidth=1, 
            edgecolor="white", fill=None)
        ax.add_patch(box_patch)
    
    fig.suptitle("Ground Truth")
    fig.savefig(save_img_file, dpi=199)
    plt.close()
    del fig, ax
    return None
