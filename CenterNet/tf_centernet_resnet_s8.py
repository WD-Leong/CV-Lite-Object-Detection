
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
    image_decoded = tf.cast(image_decoded, tf.float32)
    image_decoded = image_decoded / 255.0
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

def build_model(
    num_classes, n_scales=5, backbone_model="resnet50"):
    """
    Builds Backbone Model with pre-trained imagenet weights.
    """
    # Define the focal loss bias. #
    b_focal = tf.constant_initializer(
        np.log(0.01 / 0.99))
    
    # Classification and Regression Feature Layers. #
    cls_cnn = []
    reg_cnn = []
    for n_layer in range(4):
        cls_cnn.append(layers.Conv2D(
            256, 3, padding="same", 
            activation=None, use_bias=False, 
            name="cls_layer_" + str(n_layer+1)))
        
        reg_cnn.append(layers.Conv2D(
            256, 3, padding="same", 
            activation=None, use_bias=False, 
            name="reg_layer_" + str(n_layer+1)))
    
    # Backbone Network. #
    if backbone_model.lower() == "resnet50":
        backbone = tf.keras.applications.ResNet50(
            include_top=False, input_shape=[None, None, 3])
        
        c3_c5_layer_names = [
            "conv3_block4_out", 
            "conv4_block6_out", "conv5_block3_out"]
    if backbone_model.lower() == "resnet101":
        backbone = tf.keras.applications.ResNet101(
            include_top=False, input_shape=[None, None, 3])
        
        c3_c5_layer_names = [
            "conv3_block4_out", 
            "conv4_block23_out", "conv5_block3_out"]
    else:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, input_shape=[None, None, 3])
        
        c3_c5_layer_names = [
            "block_6_expand", "block_13_expand", "Conv_1"]
    
    # Extract the feature maps. #
    feature_maps = [
        backbone.get_layer(layer_name).output
            for layer_name in c3_c5_layer_names]
    
    c3_output = feature_maps[0]
    c4_output = feature_maps[1]
    c5_output = feature_maps[2]
    
    # Feature Pyramid Network Feature Maps. #
    p3_1x1 = layers.Conv2D(
        256, 1, 1, "same", name="c3_1x1")(c3_output)
    p4_1x1 = layers.Conv2D(
        256, 1, 1, "same", name="c4_1x1")(c4_output)
    p5_1x1 = layers.Conv2D(
        256, 1, 1, "same", name="c5_1x1")(c5_output)
    
    # P6 to P7. #
    p6_output = layers.Conv2D(
        256, 3, 2, "same", name="c6_3x3")(p5_1x1)
    p6_relu   = tf.nn.relu(p6_output)
    p7_output = layers.Conv2D(
        256, 3, 2, "same", name="c7_3x3")(p6_relu)
    
    # Upsampling and Residual Connections. #
    p6_residual = p6_relu + layers.UpSampling2D(
        size=(2, 2), name="ups_P7")(p7_output)
    p5_residual = p5_1x1 + layers.UpSampling2D(
        size=(2, 2), name="ups_P6")(p6_residual)
    p4_residual = p4_1x1 + layers.UpSampling2D(
        size=(2, 2), name="ups_P5")(p5_residual)
    p3_residual = p3_1x1 + layers.UpSampling2D(
        size=(2, 2), name="ups_P4")(p4_residual)
    
    # CNN Feature Map layer. #
    x_cnn_features = layers.Conv2D(
        256, 3, 1, "same", 
        name="cnn_feature_map")(p3_residual)
    
    # Output Layers. #
    cls_outputs = []
    for n_scale in range(n_scales):
        layer_cls_output = x_cnn_features
        for n_layer in range(4):
            layer_cls_output = \
                cls_cnn[n_layer](layer_cls_output)
        
        cnn_cls_name = "cnn_cls_output_" + str(n_scale+1)
        tmp_output = tf.nn.relu(layer_cls_output)
        cls_output = layers.Conv2D(
            num_classes, 3, 1, 
            bias_initializer=b_focal, 
            padding="same", name=cnn_cls_name)(tmp_output)
        cls_outputs.append(tf.expand_dims(cls_output, axis=3))
    
    reg_outputs = []
    for n_scale in range(n_scales):
        layer_reg_output = x_cnn_features
        for n_layer in range(4):
            layer_reg_output = \
                reg_cnn[n_layer](layer_reg_output)
        
        cnn_reg_name = "cnn_reg_output_" + str(n_scale+1)
        tmp_output = tf.nn.relu(layer_reg_output)
        reg_output = layers.Conv2D(
            4, 3, 1, use_bias=True, 
            padding="same", name=cnn_reg_name)(tmp_output)
        reg_outputs.append(
            tf.expand_dims(tf.nn.sigmoid(reg_output), axis=3))
    
    cls_outputs = tf.concat(cls_outputs, axis=3)
    reg_outputs = tf.concat(reg_outputs, axis=3)
    
    x_output = tf.concat([
        reg_outputs, cls_outputs], axis=4)
    return tf.keras.Model(
        inputs=backbone.input, outputs=x_output)

def prediction_to_corners(xy_pred, box_scales, stride=8):
    feat_dims  = [tf.shape(xy_pred)[0], 
                  tf.shape(xy_pred)[1]]
    bbox_shape = [int(xy_pred.shape[0]), 
                  int(xy_pred.shape[1]), 
                  int(len(box_scales)), 4]
    bbox_coord = np.zeros(bbox_shape)
    
    ch = tf.range(0., tf.cast(
        feat_dims[0], tf.float32), dtype=tf.float32)
    cw = tf.range(0., tf.cast(
        feat_dims[1], tf.float32), dtype=tf.float32)
    [grid_x, grid_y] = tf.meshgrid(cw, ch)
    
    for n_scale in range(len(box_scales)):
        pred_x_cen = (grid_x + xy_pred[:, :, n_scale, 1])*stride
        pred_y_cen = (grid_y + xy_pred[:, :, n_scale, 0])*stride
        pred_box_w = \
            xy_pred[:, :, n_scale, 3] * box_scales[n_scale]
        pred_box_h = \
            xy_pred[:, :, n_scale, 2] * box_scales[n_scale]
        
        pred_x_low = pred_x_cen - pred_box_w / 2.0
        pred_x_upp = pred_x_cen + pred_box_w / 2.0
        pred_y_low = pred_y_cen - pred_box_h / 2.0
        pred_y_upp = pred_y_cen + pred_box_h / 2.0
    
        bbox_coord[:, :, n_scale, 0] = pred_y_low
        bbox_coord[:, :, n_scale, 2] = pred_y_upp
        bbox_coord[:, :, n_scale, 1] = pred_x_low
        bbox_coord[:, :, n_scale, 3] = pred_x_upp
    return bbox_coord

def format_data(
    gt_labels, box_scales, img_dim, 
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
    
    num_scales = len(box_scales)
    tmp_output = np.zeros(
        [h_max, w_max, num_scales, num_classes+4])
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
            
            box_h = tmp_coord[2] - tmp_coord[0]
            box_w = tmp_coord[3] - tmp_coord[1]
            box_d = max(box_h, box_w)
            id_sc = min([n_sc for n_sc in range(
                num_scales) if box_d < box_scales[n_sc]])
            
            tmp_scale = box_scales[id_sc]
            raw_y_cen = (tmp_coord[0] + tmp_coord[2]) / 2.0
            raw_x_cen = (tmp_coord[1] + tmp_coord[3]) / 2.0
            tmp_y_cen = int((pad_y + raw_y_cen) / stride)
            tmp_x_cen = int((pad_x + raw_x_cen) / stride)
            idx_class = 4 + int(tmp_label[4])
            
            # Compute the bounding box offsets. #
            tmp_y_off = (pad_y + raw_y_cen - tmp_y_cen*stride)
            tmp_x_off = (pad_x + raw_x_cen - tmp_x_cen*stride)
            
            box_offsets = [
                tmp_y_off / stride, tmp_x_off / stride, 
                box_h / tmp_scale, box_w / tmp_scale]
            
            tmp_output[
                tmp_y_cen, tmp_x_cen, id_sc, :4] = box_offsets
            tmp_output[
                tmp_y_cen, tmp_x_cen, id_sc, idx_class] = 1.0
        
        num_targets = len(tmp_labels)
    return tmp_output, num_targets

def smooth_l1_loss(
    xy_true, xy_pred, mask=1.0, delta=1.0):
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
    tmp_log_logits = tf.math.log(
        1.0 + tf.exp(-1.0 * tf.abs(logits)))
    
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
    n_scales = int(tf.shape(y_pred)[3])
    
    cls_loss = 0.0
    reg_loss = 0.0
    for n_scale in range(n_scales):
        tmp_obj  = tf.reduce_max(
            y_true[:, :, :, n_scale, 4:], axis=-1)
        tmp_mask = tf.cast(tmp_obj > 0, tf.float32)
        
        cls_loss += focal_loss(
            y_true[:, :, :, n_scale, 4:], 
            y_pred[:, :, :, n_scale, 4:])
        
        reg_loss += smooth_l1_loss(
            y_true[:, :, :, n_scale, :4], 
            y_pred[:, :, :, n_scale, :4], mask=tmp_mask)
    return cls_loss, reg_loss

def train_step(
    model, sub_batch_sz, 
    images, bboxes, optimizer, 
    cls_lambda=1.0, reg_lambda=1.0, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = images.shape[0]
    if batch_size <= sub_batch_sz:
        n_sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        n_sub_batch = int(batch_size / sub_batch_sz)
    else:
        n_sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
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
        
        with tf.GradientTape() as grad_tape:
            tmp_output = model(tmp_images, training=True)
            tmp_losses = model_loss(tmp_bboxes, tmp_output)
            
            tmp_cls_loss += tmp_losses[0]
            tmp_reg_loss += tmp_losses[1]
            total_losses = tf.add(
                cls_lambda*tmp_losses[0], 
                reg_lambda*tmp_losses[1])
        
        # Accumulate the gradients. #
        tmp_gradients = \
            grad_tape.gradient(total_losses, model_params)
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
    img_in_file, model, box_scales, 
    labels, heatmap=True, thresh=0.50, 
    downsample=8, iou_thresh=0.213, img_box=None, 
    img_title=None, img_rows=448, img_cols=448, 
    save_img_file="object_detection_result.jpg"):
    # Read the image. #
    image_resized = _parse_image(
        img_in_file, img_rows=img_rows, img_cols=img_cols)
    
    tmp_output = model.predict(
        tf.expand_dims(image_resized, axis=0))
    reg_output = prediction_to_corners(
        tmp_output[0, :, :, :, :4], 
        box_scales, stride=downsample)
    cls_output = tmp_output[0, :, :, :, 4:]
    
    cls_probs = tf.nn.sigmoid(cls_output)
    n_scales  = cls_output.shape[2]
    n_classes = cls_output.shape[3]
    
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
                cls_probs, axis=[2, 3])
        else:
            obj_probs = tf.reduce_max(
                cls_probs[:, :, :, 0], axis=2)
        
        obj_probs = tf.image.resize(tf.expand_dims(
            obj_probs, axis=2), [img_width, img_height])
        tmp = ax.imshow(tf.squeeze(
            obj_probs, axis=2), "jet", alpha=0.50)
        fig.colorbar(tmp, ax=ax)
    
    tmp_obj_detect = []
    for n_scale in range(n_scales):
        if n_classes > 1:
            prob_max = tf.reduce_max(
                cls_probs[:, :, n_scale, :], axis=2)
            pred_label = tf.math.argmax(
                cls_probs[:, :, n_scale, :], axis=2)
        else:
            prob_max = cls_probs[:, :, n_scale, 0]
        
        tmp_thresh = \
            np.where(prob_max >= thresh, 1, 0)
        tmp_coords = np.nonzero(tmp_thresh)
        
        for n_box in range(len(tmp_coords[0])):
            x_coord = tmp_coords[0][n_box]
            y_coord = tmp_coords[1][n_box]
            
            tmp_boxes = \
                reg_output[x_coord, y_coord, n_scale, :]
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
        tmp_bbox_reg = prediction_to_corners(
            img_box[:, :, :, :4], 
            box_scales, stride=downsample)
        
        for n_scale in range(n_scales):
            tmp_true_box = np.nonzero(tf.reduce_max(
                img_box[:, :, n_scale, 4:], axis=2))
            for n_box in range(len(tmp_true_box[0])):
                x_coord = tmp_true_box[0][n_box]
                y_coord = tmp_true_box[1][n_box]
                
                tmp_boxes = \
                    tmp_bbox_reg[x_coord, y_coord, n_scale, :4]
                
                x_low = tmp_h_ratio * tmp_boxes[1]
                y_low = tmp_w_ratio * tmp_boxes[0]
                x_upp = tmp_h_ratio * tmp_boxes[3]
                y_upp = tmp_w_ratio * tmp_boxes[2]
                
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
    img_array, img_box, img_dims, box_scales, 
    downsample=8, save_img_file="ground_truth.jpg"):
    # Plot the bounding boxes on the image. #
    fig, ax = plt.subplots(1)
    ax.imshow(img_array)
    
    n_scales = int(img_box.shape[2])
    tmp_w_ratio = 1.0
    tmp_h_ratio = 1.0
    
    obj_probs = tf.reduce_max(
        img_box[:, :, :, 4:], axis=[2, 3])
    obj_probs = tf.image.resize(tf.expand_dims(
        obj_probs, axis=2), [img_dims, img_dims])
    tmp = ax.imshow(tf.squeeze(
        obj_probs, axis=2), "jet", alpha=0.50)
    fig.colorbar(tmp, ax=ax)
    
    # True image is not transposed. #
    tmp_bbox_reg = prediction_to_corners(
        img_box[:, :, :, :4], box_scales, stride=downsample)
    for n_scale in range(n_scales):
        tmp_true_box = np.nonzero(tf.reduce_max(
            img_box[:, :, n_scale, 4:], axis=2))
        
        for n_box in range(len(tmp_true_box[0])):
            x_coord = tmp_true_box[0][n_box]
            y_coord = tmp_true_box[1][n_box]
            
            tmp_boxes = \
                tmp_bbox_reg[x_coord, y_coord, n_scale, :4]
            
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
