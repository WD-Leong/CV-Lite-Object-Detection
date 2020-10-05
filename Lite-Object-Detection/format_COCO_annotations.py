import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf

# Load the COCO dataset. #
tmp_pd_file = \
    "C:/Users/admin/Desktop/Data/COCO/object_boxes.csv"
raw_coco_df = pd.read_csv(tmp_pd_file)

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

tmp_col_df  = ["filename", "img_width", "img_height", "id", 
              "x_lower", "y_lower", "box_width", "box_height"]
image_files = sorted(list(pd.unique(raw_coco_df["filename"])))
image_files = pd.DataFrame(image_files, columns=["filename"])
print("Total of", str(len(image_files)), "images in COCO dataset.")

raw_coco_df["x_centroid"] = \
    raw_coco_df["x_lower"] + raw_coco_df["box_width"]/2
raw_coco_df["y_centroid"] = \
    raw_coco_df["y_lower"] + raw_coco_df["box_height"]/2

# Define the Neural Network. #
#img_width  = 448
#img_height = 448
img_width  = 320
img_height = 320
max_scale  = max(img_width, img_height)
img_scale  = [40, 80, 160, 320]
n_classes  = len(label_dict)

down_width   = int(img_width/8)
down_height  = int(img_height/8)
dense_shape  = [down_height, down_width, 4, n_classes+4]

# Find a way to remove duplicate indices from the data. #
# Total output classes is n_classes + centerness (1) + regression (4). #
print("Formatting the object detection bounding boxes.")
start_time = time.time()

train_list = []
for n_img in range(len(image_files)):
    tot_obj  = 0
    img_file = image_files.iloc[n_img]["filename"]
    
    tmp_filter = raw_coco_df[
        raw_coco_df["filename"] == img_file]
    tmp_filter = tmp_filter[[
        "img_width", "img_height", "id", "x_lower", "y_lower", 
        "box_width", "box_height", "x_centroid", "y_centroid"]]
    
    tmp_w_ratio = tmp_filter.iloc[0]["img_width"] / img_width
    tmp_h_ratio = tmp_filter.iloc[0]["img_height"] / img_height
    if len(tmp_filter) > 0:
        tmp_indices = []
        tmp_values  = []
        for n_obj in range(len(tmp_filter)):
            tmp_object = tmp_filter.iloc[n_obj]
            tmp_label  = int(index_dict[coco_label[
                coco_label["id"] == tmp_object["id"]].iloc[0]["name"]])
            
            tmp_width  = tmp_object["box_width"] / tmp_w_ratio
            tmp_height = tmp_object["box_height"] / tmp_h_ratio
            
            if tmp_width < 0 or tmp_height < 0:
                continue
            elif tmp_width < img_scale[0] and tmp_height < img_scale[0]:
                id_sc = 0
                box_scale = img_scale[0]
            elif tmp_width < img_scale[1] and tmp_height < img_scale[1]:
                id_sc = 1
                box_scale = img_scale[1]
            elif tmp_width < img_scale[2] and tmp_height < img_scale[2]:
                id_sc = 2
                box_scale = img_scale[2]
            else:
                id_sc = 3
                if max_scale < img_scale[3]:
                    box_scale = max_scale
                else:
                    box_scale = img_scale[3]
            
            # The regression and classification outputs #
            # are 104 x 104 pixels, so divide by 8.     #
            tmp_x_cen = tmp_object["x_centroid"] / tmp_w_ratio
            tmp_y_cen = tmp_object["y_centroid"] / tmp_h_ratio
            tmp_w_cen = int(tmp_x_cen/8)
            tmp_h_cen = int(tmp_y_cen/8)
            
            tmp_w_off = (tmp_x_cen - tmp_w_cen*8) / 8
            tmp_h_off = (tmp_y_cen - tmp_h_cen*8) / 8
            tmp_w_reg = tmp_width / box_scale
            tmp_h_reg = tmp_height / box_scale
            
            val_array = [tmp_h_off, tmp_w_off, tmp_h_reg, tmp_w_reg, 1]
            for dim_arr in range(5):
                tmp_index_list = [tmp_h_cen, tmp_w_cen, id_sc, dim_arr]
                if tmp_index_list not in tmp_indices:
                    tmp_values.append(val_array[dim_arr])
                    tmp_indices.append(tmp_index_list)
            
            id_obj = tmp_label + 4
            tmp_index_list = [tmp_h_cen, tmp_w_cen, id_sc, id_obj]
            if tmp_index_list not in tmp_indices:
                tmp_values.append(1)
                tmp_indices.append(tmp_index_list)
        
        if len(tmp_indices) > 0:
            tmp_sparse_tensor = tf.sparse.SparseTensor(
                tmp_indices, tmp_values, dense_shape)
            train_list.append((img_file, tmp_sparse_tensor))
        
        if (n_img+1) % 2500 == 0:
            print(str(n_img+1), "annotations processed.")

elapsed_tm = (time.time() - start_time) / 60
print("Elapsed Time:", str(round(elapsed_tm, 3)), "mins.")

print("Saving the file.")
save_pkl_file = "C:/Users/admin/Desktop/Data/COCO/"
save_pkl_file += "coco_annotations_320.pkl"
with open(save_pkl_file, "wb") as tmp_save:
    pkl.dump(img_scale, tmp_save)
    pkl.dump(train_list, tmp_save)
