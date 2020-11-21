# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:11:14 2020

@author: admin
"""

import numpy as np
import tensorflow as tf

def _parse_image(
    filename, img_rows=320, img_cols=320):
    # Image is transposed when loaded. #
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = \
        tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_decoded = tf.transpose(image_decoded, [1, 0, 2])
    image_resized = tf.image.resize(
        image_decoded, [img_rows, img_cols])
    image_resized = tf.ensure_shape(
        image_resized, shape=(img_rows, img_cols, 3))
    return image_resized

def image_augment(img_in, img_bbox, p=0.5):
    if np.random.uniform() >= p:
        p_tmp = np.random.uniform()
        if p_tmp <= 0.333:
            # Distort colour. #
            tmp_in = img_in.numpy()
            
            tmp_in[:, :, 0] = 1.0 - tmp_in[:, :, 0]
            tmp_in[:, :, 1] = 1.0 - tmp_in[:, :, 1]
            tmp_in[:, :, 2] = 1.0 - tmp_in[:, :, 2]
            return (tf.constant(tmp_in), img_bbox)
        elif p_tmp <= 0.667:
            # Flip left-right. #
            img_in = img_in[:, ::-1, :]
            
            tmp_bbox = img_bbox.numpy()
            img_bbox = tmp_bbox[:, ::-1, :, :]
            
            img_bbox[:, :, :, 1] = 1.0 - img_bbox[:, :, :, 1]
            return (img_in, tf.constant(img_bbox))
        else:
            # Rotate by either 90 degrees of 270 degrees. #
            img_in = tf.transpose(img_in, [1, 0, 2])
            
            img_bbox = tf.transpose(img_bbox, [1, 0, 2, 3])
            tmp_bbox = img_bbox.numpy()
            img_bbox = tmp_bbox
            
            img_bbox[:, :, :, 0] = tmp_bbox[:, :, :, 1]
            img_bbox[:, :, :, 1] = tmp_bbox[:, :, :, 0]
            img_bbox[:, :, :, 2] = tmp_bbox[:, :, :, 3]
            img_bbox[:, :, :, 3] = tmp_bbox[:, :, :, 2]
            
            p_rot = np.random.uniform()
            if p_rot >= 0.50:
                # Rotate by 270 degrees by flipping up-down. #
                img_in = img_in[::-1, :, :]
                
                img_bbox = img_bbox[::-1, :, :, :]
                img_bbox[:, :, :, 0] = 1.0 - img_bbox[:, :, :, 0]
            
            img_bbox = tf.constant(img_bbox)
            return (img_in, img_bbox)
    else:
        return (img_in, img_bbox)

def img_mosaic(
    img_files, img_dims, bboxes, img_scale, 
    n_classes, input_rows=320, input_cols=320):
    n_imgs = len(img_files)
    
    if n_imgs == 1:
        img_w = img_dims[0][0]
        img_h = img_dims[0][1]
        img_coord  = [(0, 0)]
        img_config = [(img_w, img_h)]
    elif n_imgs == 2:
        if np.random.uniform() < 0.5:
            img_w = img_dims[0][0]
            img_h = img_dims[0][1] + img_dims[1][1]
            img_coord  = [
                (0, 0), (0, img_dims[0][1])]
            img_config = [
                (img_w, img_dims[0][1]), (img_w, img_dims[1][1])]
        else:
            img_w = img_dims[0][0] + img_dims[1][0]
            img_h = img_dims[0][1]
            img_coord  = [
                (0, 0), (img_dims[0][0], 0)]
            img_config = [
                (img_dims[0][0], img_h), (img_dims[1][0], img_h)]
    elif n_imgs == 3:
        if np.random.uniform() < 0.5:
            img_w = img_dims[0][0] + img_dims[2][0]
            img_h = img_dims[0][1] + img_dims[1][1]
            img_coord  = [
                (0, 0), 
                (0, img_dims[0][1]), 
                (img_dims[0][0], 0)]
            img_config = [
                (img_dims[0][0], img_dims[0][1]), 
                (img_w, img_dims[1][1]), 
                (img_dims[2][0], img_dims[0][1])]
        else:
            img_w = img_dims[0][0] + img_dims[1][0]
            img_h = img_dims[0][1] + img_dims[2][1]
            img_coord  = [
                (0, 0), 
                (img_dims[0][0], 0), 
                (0, img_dims[0][1])]
            img_config = [
                (img_dims[0][0], img_dims[0][1]), 
                (img_dims[1][0], img_dims[0][1]), 
                (img_w, img_dims[2][1])]
    elif n_imgs == 4:
        if np.random.uniform() < 1.0:
            img_w = img_dims[0][0] + img_dims[2][0]
            img_h = img_dims[0][1] + img_dims[1][1]
            img_coord  = [
                (0, 0), 
                (0, img_dims[0][1]), 
                (img_dims[0][0], 0), 
                (img_dims[0][0], img_dims[0][1])]
            img_config = [
                (img_dims[0][0], img_dims[0][1]), 
                (img_dims[0][0], img_dims[1][1]), 
                (img_dims[2][0], img_dims[0][1]), 
                (img_dims[2][0], img_dims[1][1])]
        else:
            img_w = img_dims[0][0] + img_dims[1][0]
            img_h = img_dims[0][1] + img_dims[2][1]
            img_coord  = [
                (0, 0), 
                (img_dims[0][0], 0), 
                (0, img_dims[0][1]), 
                (img_dims[0][0], img_dims[0][1])]
            img_config = [
                (img_dims[0][0], img_dims[0][1]), 
                (img_dims[1][0], img_dims[0][1]), 
                (img_dims[0][0], img_dims[2][1]), 
                (img_dims[1][0], img_dims[2][1])]
    
    img_array = np.zeros(
        [int(img_w), int(img_h), 3], dtype=np.float32)
    dense_shape = [
        int(input_rows/8), int(input_cols/8), 4, n_classes+4]
    
    tmp_values  = []
    tmp_indices = []
    tmp_w_ratio = img_w / input_rows
    tmp_h_ratio = img_h / input_cols
    for n_img in range(n_imgs):
        xmin = int(img_coord[n_img][0])
        ymin = int(img_coord[n_img][1])
        
        img_file = img_files[n_img]
        tmp_bbox = bboxes[n_img]
        tmp_dims = img_dims[n_img]
        tmp_config = img_config[n_img]
        
        tmp_rows = int(tmp_config[0])
        tmp_cols = int(tmp_config[1])
        tmp_img  = _parse_image(
            img_file, img_rows=tmp_rows, img_cols=tmp_cols)
        
        xmax = xmin + tmp_rows
        ymax = ymin + tmp_cols
        img_array[xmin:xmax, ymin:ymax, :] = tmp_img.numpy()
        
        scale_w = tmp_rows / tmp_dims[0]
        scale_h = tmp_cols / tmp_dims[1]
        if len(tmp_bbox) > 0:
            for tmp_box in tmp_bbox:
                tmp_x_cen  = tmp_box[0] * scale_w
                tmp_y_cen  = tmp_box[1] * scale_h
                tmp_x_cen  = (xmin + tmp_x_cen) / tmp_w_ratio
                tmp_y_cen  = (ymin + tmp_y_cen) / tmp_h_ratio
                tmp_label  = tmp_box[4]
                tmp_width  = tmp_box[2] * scale_w / tmp_w_ratio
                tmp_height = tmp_box[3] * scale_h / tmp_h_ratio
                
                if tmp_width < 0 or tmp_height < 0:
                    continue
                elif tmp_width < img_scale[0] \
                    and tmp_height < img_scale[0]:
                    id_sc = 0
                    box_scale = img_scale[0]
                elif tmp_width < img_scale[1] \
                    and tmp_height < img_scale[1]:
                    id_sc = 1
                    box_scale = img_scale[1]
                elif tmp_width < img_scale[2] \
                    and tmp_height < img_scale[2]:
                    id_sc = 2
                    box_scale = img_scale[2]
                else:
                    id_sc = 3
                    box_scale = img_scale[3]
                
                tmp_w_reg = tmp_width / box_scale
                tmp_h_reg = tmp_height / box_scale
                
                tmp_w_cen = int(tmp_x_cen/8)
                tmp_h_cen = int(tmp_y_cen/8)
                tmp_w_off = (tmp_x_cen - tmp_w_cen*8) / 8
                tmp_h_off = (tmp_y_cen - tmp_h_cen*8) / 8
                
                val_array = [tmp_w_off, tmp_h_off, tmp_w_reg, tmp_h_reg, 1]
                for dim_arr in range(5):
                    tmp_index = [tmp_w_cen, tmp_h_cen, id_sc, dim_arr]
                    if tmp_index not in tmp_indices:
                        tmp_indices.append(tmp_index)
                        tmp_values.append(val_array[dim_arr])
                
                id_obj = tmp_label + 4
                tmp_index = [tmp_w_cen, tmp_h_cen, id_sc, id_obj]
                if tmp_index not in tmp_indices:
                    tmp_values.append(1)
                    tmp_indices.append(tmp_index)
    
    bbox_sparse = tf.sparse.SparseTensor(
        tmp_indices, tmp_values, dense_shape)
    img_bboxes  = tf.sparse.to_dense(
        tf.sparse.reorder(bbox_sparse))
    
    image_resized = tf.image.resize(
        img_array, [input_rows, input_cols])
    image_resized = tf.ensure_shape(
        image_resized, shape=(input_rows, input_cols, 3))
    return img_array, image_resized, img_bboxes
