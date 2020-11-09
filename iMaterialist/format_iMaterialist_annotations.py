
import time
import json
import pandas as pd
import pickle as pkl
import tensorflow as tf

# Load the VOC 2012 dataset. #
tmp_path = "C:/Users/admin/Desktop/Data/iMaterialist/"
tmp_pd_file = tmp_path + "iMaterialist.csv"
raw_data_df = pd.read_csv(tmp_pd_file)

image_files = sorted(list(pd.unique(raw_data_df["filename"])))
image_files = pd.DataFrame(image_files, columns=["filename"])
print("Total of", str(len(image_files)), 
      "images in iMaterialist dataset.")

# The iMaterialist class names. #
tmp_json_file = tmp_path 
tmp_json_file += "instances_attributes_train2020.json"
with open(tmp_json_file) as tmp_file_open:
    tmp_json = json.load(tmp_file_open)

# Add in the objectness category to number of classes. #
tmp_categories = tmp_json["categories"]
label_dict = dict([
    (x["id"], x["name"]) for x in tmp_categories])

start_time = time.time()
img_dims   = [(256, 256), (320, 320), (384, 384)]
n_classes  = len(label_dict) + 1

img_scale = []
for n_scale in range(len(img_dims)):
    min_scale = min(img_dims[n_scale])
    tmp_scale = [int(
        min_scale/(2**x)) for x in range(4)]
    img_scale.append(tmp_scale[::-1])
del tmp_scale

iMat_objects = []
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
        
        tmp_filter = raw_data_df[
            raw_data_df["filename"] == img_file]
        tmp_filter = tmp_filter[[
            "img_width", "img_height", "category_id", 
            "x_cen", "y_cen", "box_width", "box_height"]]
        
        tmp_w_ratio = tmp_filter.iloc[0]["img_width"] / img_width
        tmp_h_ratio = tmp_filter.iloc[0]["img_height"] / img_height
        
        sparse_tensor_list = []
        if len(tmp_filter) > 0:
            tmp_indices = []
            tmp_values  = []
            for n_obj in range(len(tmp_filter)):
                tmp_object = tmp_filter.iloc[n_obj]
                tmp_label  = int(tmp_object["category_id"])
                tmp_x_cen  = tmp_object["x_cen"] / tmp_w_ratio
                tmp_y_cen  = tmp_object["y_cen"] / tmp_h_ratio
                
                tmp_width  = tmp_object["box_width"] / tmp_w_ratio
                tmp_height = tmp_object["box_height"] / tmp_h_ratio
                
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
                tmp_w_reg = tmp_width / box_scale
                tmp_h_reg = tmp_height / box_scale
                
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
                
                # The labels are zero-indexed so add 5. #
                id_obj = tmp_label + 5
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
    iMat_objects.append(tmp_objects)

elapsed_tm = (time.time() - start_time) / 60
print("Total of", str(len(iMat_objects[0])), "images.")
print("Elapsed Time:", str(round(elapsed_tm, 3)), "mins.")

print("Saving the file.")
save_pkl_file = tmp_path + "iMat_annotations_multiscale.pkl"
with open(save_pkl_file, "wb") as tmp_save:
    pkl.dump(img_scale, tmp_save)
    pkl.dump(label_dict, tmp_save)
    pkl.dump(iMat_objects, tmp_save)
