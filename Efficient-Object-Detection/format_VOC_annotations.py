# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:13:59 2020

@author: admin
"""

import time
import pandas as pd
import pickle as pkl
import tensorflow as tf

# Load the VOC 2012 dataset. #
tmp_path = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/"
tmp_pd_file = tmp_path + "voc_2012_objects.csv"
raw_voc_df  = pd.read_csv(tmp_pd_file)

tmp_df_cols = ["filename", "width", "height", 
               "xmin", "xmax", "ymin", "ymax", "label"]
raw_voc_df  = pd.DataFrame(raw_voc_df, columns=tmp_df_cols)
image_files = sorted(list(pd.unique(raw_voc_df["filename"])))
image_files = pd.DataFrame(image_files, columns=["filename"])
print("Total of", str(len(image_files)), "images in VOC dataset.")

# The VOC 2012 class names. #
class_names = list(
    sorted(list(pd.unique(raw_voc_df["label"]))))
class_names = ["object"] + [str(x) for x in class_names]
class_dict  = dict(
    [(class_names[x], x) for x in range(len(class_names))])
label_dict  = dict(
    [(x, class_names[x]) for x in range(len(class_names))])

raw_voc_df["img_label"] = [
    class_dict.get(str(
        raw_voc_df.iloc[x]["label"])) \
            for x in range(len(raw_voc_df))]
raw_voc_df["x_centroid"] = \
    (raw_voc_df["xmax"] + raw_voc_df["xmin"]) / 2
raw_voc_df["y_centroid"] = \
    (raw_voc_df["ymax"] + raw_voc_df["ymin"]) / 2

start_time = time.time()
img_dims   = [(256, 256), (320, 320), (384, 384)]
n_classes  = len(class_names)

img_scale = []
for n_scale in range(len(img_dims)):
    min_scale = min(img_dims[n_scale])
    tmp_scale = [int(
        min_scale/(2**x)) for x in range(4)]
    img_scale.append(tmp_scale[::-1])
del tmp_scale

voc_objects = []
for n_scale in range(len(img_dims)):
    tmp_objects = []
    for n_img in range(len(image_files)):
        tot_obj  = 0
        img_file = image_files.iloc[n_img]["filename"]
        
        img_width   = img_dims[n_scale][0]
        img_height  = img_dims[n_scale][1]
        max_scale   = max(img_width, img_height)
        tmp_scale   = img_scale[n_scale]
        down_width  = int(img_width/8)
        down_height = int(img_height/8)
        dense_shape = [down_height, down_width, 4, n_classes+4]
        
        tmp_filter = raw_voc_df[
            raw_voc_df["filename"] == img_file]
        tmp_filter = tmp_filter[[
            "width", "height", "img_label", "xmin", "xmax", 
            "ymin", "ymax", "x_centroid", "y_centroid"]]
        
        tmp_w_ratio = tmp_filter.iloc[0]["width"] / img_width
        tmp_h_ratio = tmp_filter.iloc[0]["height"] / img_height
        
        sparse_tensor_list = []
        if len(tmp_filter) > 0:
            tmp_indices = []
            tmp_values  = []
            for n_obj in range(len(tmp_filter)):
                tmp_object = tmp_filter.iloc[n_obj]
                tmp_label  = int(tmp_object["img_label"])
                
                tmp_w_min = tmp_object["xmin"] / tmp_w_ratio
                tmp_w_max = tmp_object["xmax"] / tmp_w_ratio
                tmp_h_min = tmp_object["ymin"] / tmp_h_ratio
                tmp_h_max = tmp_object["ymax"] / tmp_h_ratio
                
                tmp_x_cen = tmp_object["x_centroid"] / tmp_w_ratio
                tmp_y_cen = tmp_object["y_centroid"] / tmp_h_ratio
                
                tmp_width  = tmp_w_max - tmp_w_min
                tmp_height = tmp_h_max - tmp_h_min
                
                if tmp_width < 0 or tmp_height < 0:
                    continue
                elif tmp_width < tmp_scale[0] \
                    and tmp_height < tmp_scale[0]:
                    id_sc = 0
                    box_scale = tmp_scale[0]
                elif tmp_width < tmp_scale[1] \
                    and tmp_height < tmp_scale[1]:
                    id_sc = 1
                    box_scale = tmp_scale[1]
                elif tmp_width < tmp_scale[2] \
                    and tmp_height < tmp_scale[2]:
                    id_sc = 2
                    box_scale = tmp_scale[2]
                else:
                    id_sc = 3
                    box_scale = tmp_scale[3]
                
                # The regression and classification outputs #
                # are 40 x 40 pixels, so divide by 8.       #
                tmp_w_reg = (tmp_w_max - tmp_w_min) / box_scale
                tmp_h_reg = (tmp_h_max - tmp_h_min) / box_scale
                
                tmp_w_cen = int(tmp_x_cen/8)
                tmp_h_cen = int(tmp_y_cen/8)
                tmp_w_off = (tmp_x_cen - tmp_w_cen*8) / 8
                tmp_h_off = (tmp_y_cen - tmp_h_cen*8) / 8
                
                val_array = [tmp_h_off, tmp_w_off, tmp_h_reg, tmp_w_reg, 1]
                for dim_arr in range(5):
                    tmp_index = [tmp_h_cen, tmp_w_cen, id_sc, dim_arr]
                    if tmp_index not in tmp_indices:
                        tmp_indices.append(tmp_index)
                        tmp_values.append(val_array[dim_arr])
                
                id_obj = tmp_label + 4
                tmp_index = [tmp_h_cen, tmp_w_cen, id_sc, id_obj]
                if tmp_index not in tmp_indices:
                    tmp_values.append(1)
                    tmp_indices.append(tmp_index)
            
            if len(tmp_indices) > 0:
                tmp_dims = (img_width, img_height)
                tmp_sparse_tensor = tf.sparse.SparseTensor(
                    tmp_indices, tmp_values, dense_shape)
                tmp_objects.append((
                    img_file, tmp_dims, tmp_sparse_tensor))
            
        if (n_img+1) % 2500 == 0:
            print(str(n_img+1), "annotations processed", 
                  "at scale", str(n_scale+1) + ".")
    voc_objects.append(tmp_objects)

elapsed_tm = (time.time() - start_time) / 60
print("Total of", str(len(voc_objects[0])), "images.")
print("Elapsed Time:", str(round(elapsed_tm, 3)), "mins.")

print("Saving the file.")
save_pkl_file = tmp_path + "voc_annotations_multiscale.pkl"
with open(save_pkl_file, "wb") as tmp_save:
    pkl.dump(img_scale, tmp_save)
    pkl.dump(label_dict, tmp_save)
    pkl.dump(voc_objects, tmp_save)
