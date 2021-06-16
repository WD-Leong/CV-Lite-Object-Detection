import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf

# Custom function. #
def compute_centerness(l, r, b, t):
    return np.multiply(
        np.sqrt(min(l, r) / max(l, r)), 
        np.sqrt(min(b, t) / max(b, t)))

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
list_label = ["objectness"] + list_label
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
#img_dims  = [(128, 128), (192, 192), (256, 256), 
#             (320, 320), (384, 384), (448, 448), 
#             (512, 512), (576, 576), (640, 640)]

img_dims  = [(448, 448)]
n_classes = len(label_dict)

num_scale = 5
img_scale = []
for n_scale in range(len(img_dims)):
    min_scale = min(img_dims[n_scale])
    tmp_scale = [int(
        min_scale/(2**x)) for x in range(num_scale)]
    img_scale.append(tmp_scale[::-1])
del tmp_scale

# Find a way to remove duplicate indices from the data. #
# Total output classes is n_classes + centerness (1) + regression (4). #
print("Formatting the object detection bounding boxes.")
start_time = time.time()

train_objects = []
for n_scale in range(len(img_dims)):
    tmp_objects = []
    for n_img in range(len(image_files)):
        print(n_img)
        tot_obj  = 0
        img_file = image_files.iloc[n_img]["filename"]
        
        img_width   = img_dims[n_scale][0]
        img_height  = img_dims[n_scale][1]
        max_scale   = max(img_width, img_height)
        tmp_scale   = img_scale[n_scale]
        down_width  = int(img_width/8)
        down_height = int(img_height/8)
        dense_shape = [
            down_height, down_width, num_scale, n_classes+4]
        
        tmp_filter = raw_coco_df[
            raw_coco_df["filename"] == img_file]
        tmp_filter = tmp_filter[[
            "img_width", "img_height", "id", "x_lower", "y_lower", 
            "box_width", "box_height", "x_centroid", "y_centroid"]]
        
        tmp_w_ratio = tmp_filter.iloc[0]["img_width"] / img_width
        tmp_h_ratio = tmp_filter.iloc[0]["img_height"] / img_height
        if len(tmp_filter) > 0:
            tmp_mask = np.zeros(
                [img_width, img_height, num_scale], dtype=np.int32)
            
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
                elif tmp_width < tmp_scale[3] \
                    and tmp_height < tmp_scale[3]:
                    id_sc = 3
                    box_scale = tmp_scale[3]
                else:
                    id_sc = 4
                    box_scale = tmp_scale[4]
                
                # Feature map is at stride 8. #
                tmp_x_low = int(
                    tmp_object["x_lower"] / tmp_w_ratio)
                tmp_x_upp = int(tmp_x_low + tmp_width)
                tmp_y_low = int(
                    tmp_object["y_lower"] / tmp_h_ratio)
                tmp_y_upp = int(tmp_y_low + tmp_height)
                
                tmp_x_idx = [z for z in range(tmp_x_low, tmp_x_upp)]
                tmp_y_idx = [z for z in range(tmp_y_low, tmp_y_upp)]
                tmp_index = [(
                    z0, z1) for z0 in tmp_x_idx for z1 in tmp_y_idx]
                
                tmp_l = [(x-tmp_x_low) for x, y in tmp_index]
                tmp_r = [(tmp_x_upp-x) for x, y in tmp_index]
                tmp_b = [(y-tmp_y_low) for x, y in tmp_index]
                tmp_t = [(tmp_y_upp-y) for x, y in tmp_index]
                tmp_c = [compute_centerness(
                    tmp_l[z], tmp_r[z], 
                    tmp_b[z], tmp_t[z]) for z in range(len(tmp_l))]
                
                # Get the non-overlapping indices. #
                tmp_arr = np.zeros(
                    [img_width, img_height, num_scale], dtype=np.int32)
                tmp_arr[tmp_x_low:tmp_x_upp, 
                        tmp_y_low:tmp_y_upp, id_sc] = 1
                
                tmp_nnz = np.nonzero(tmp_arr - tmp_arr*tmp_mask)
                tmp_x_nnz  = tmp_nnz[0]
                tmp_y_nnz  = tmp_nnz[1]
                tmp_sc_nnz = tmp_nnz[2]
                for n_idx in range(len(tmp_x_nnz)):
                    tmp_x  = tmp_x_nnz[n_idx]
                    tmp_y  = tmp_y_nnz[n_idx]
                    tmp_sc = tmp_sc_nnz[n_idx]
                    
                    tmp_l = tmp_x - tmp_x_low
                    tmp_r = tmp_x_upp - tmp_x
                    tmp_b = tmp_y - tmp_y_low
                    tmp_t = tmp_y_upp - tmp_y
                    tmp_c = compute_centerness(tmp_l, tmp_r, tmp_b, tmp_t)
                    
                    tmp_reg = [tmp_b, tmp_t, tmp_l, tmp_r, tmp_c, 1]
                    for dim_arr in range(6):
                        tmp_index_list = [tmp_y, tmp_x, id_sc, dim_arr]
                        tmp_values.append(tmp_reg[dim_arr])
                        tmp_indices.append(tmp_index_list)
                    
                    id_obj = tmp_label + 4
                    tmp_index_list = [tmp_y, tmp_x, id_sc, id_sc, id_obj]
                    tmp_values.append(1)
                    tmp_indices.append(tmp_index_list)
                del tmp_index, tmp_arr
            
            if len(tmp_indices) > 0:
                tmp_dims = (img_width, img_height)
                tmp_sparse_tensor = tf.sparse.SparseTensor(
                    tmp_indices, tmp_values, dense_shape)
                tmp_objects.append((
                    img_file, tmp_dims, tmp_sparse_tensor))
        
        if (n_img+1) % 2500 == 0:
            print(str(n_img+1), "annotations processed", 
                  "at the", str(n_scale+1), "scale.")
    
    # Append to the annotated files. #
    train_objects.append(tmp_objects)

elapsed_tm = (time.time() - start_time) / 60
print("Elapsed Time:", str(round(elapsed_tm, 3)), "mins.")
print("Total of", str(train_objects[1]), "images.")

print("Saving the file.")
save_pkl_file = "C:/Users/admin/Desktop/Data/COCO/"
save_pkl_file += "coco_annotations_fcos.pkl"
with open(save_pkl_file, "wb") as tmp_save:
    pkl.dump(img_scale, tmp_save)
    pkl.dump(train_objects, tmp_save)
