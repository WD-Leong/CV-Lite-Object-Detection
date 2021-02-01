
import time
import numpy as np
import pandas as pd
import pickle as pkl

# Parameters. #
min_side = 384
max_side = 384
l_jitter = 240
u_jitter = 384

# Load the VOC 2012 dataset. #
print("Loading the data.")
start_tm = time.time()

tmp_path = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/"
tmp_pd_file = tmp_path + "voc_2012_objects.csv"
raw_voc_df  = pd.read_csv(tmp_pd_file)

tmp_df_cols = ["filename", "width", "height", 
               "xmin", "xmax", "ymin", "ymax", "label"]
raw_voc_df  = pd.DataFrame(raw_voc_df, columns=tmp_df_cols)
image_files = sorted(list(pd.unique(raw_voc_df["filename"])))
image_files = pd.DataFrame(image_files, columns=["filename"])

# The VOC data class names. #
class_names = list(
    sorted(list(pd.unique(raw_voc_df["label"]))))
class_names = ["object"] + [str(x) for x in class_names]
label_2_id  = dict(
    [(class_names[x], x) for x in range(len(class_names))])
id_2_label  = dict(
    [(x, class_names[x]) for x in range(len(class_names))])

elapsed_tm = (time.time() - start_tm) / 60.0
print("Total of", str(len(image_files)), "images in VOC dataset.")
print("Elapsed time:", str(elapsed_tm), "mins.")

# Format the data. #
print("Formatting VOC data.")
start_tm = time.time()

voc_objects = []
for n_img in range(len(image_files)):
    img_file = image_files.iloc[n_img]["filename"]
        
    tmp_filter = raw_voc_df[
        raw_voc_df["filename"] == img_file]
    tmp_filter = tmp_filter[[
        "width", "height", "label", 
        "xmin", "xmax", "ymin", "ymax"]]
    n_objects  = len(tmp_filter)
    
    tmp_bboxes = []
    tmp_labels = []
    for n_obj in range(n_objects):
        tmp_object = tmp_filter.iloc[n_obj]
        img_width  = tmp_object["width"]
        img_height = tmp_object["height"]
        box_x_min  = tmp_object["xmin"] / img_width
        box_x_max  = tmp_object["xmax"] / img_width
        box_y_min  = tmp_object["ymin"] / img_height
        box_y_max  = tmp_object["ymax"] / img_height
        
        tmp_bbox = np.array([
            box_x_min, box_y_min, 
            box_x_max, box_y_max])
        tmp_label = np.array([
            label_2_id[tmp_object["label"]]])
        
        tmp_labels.append(np.expand_dims(tmp_label, axis=0))
        tmp_bboxes.append(np.expand_dims(tmp_bbox, axis=0))
    
    tmp_labels = np.concatenate(tmp_labels, axis=0)
    tmp_labels = tmp_labels.reshape((n_objects,))
    tmp_bboxes = np.concatenate(tmp_bboxes, axis=0)
    tmp_objects = {"bbox": tmp_bboxes, 
                   "label": tmp_labels}
    
    voc_objects.append({
        "image": img_file, 
        "min_side": min_side, 
        "max_side": max_side, 
        "l_jitter": l_jitter, 
        "u_jitter": u_jitter, 
        "objects": tmp_objects})
    
    if (n_img+1) % 1000 == 0:
        elapsed_tm = (time.time() - start_tm) / 60.0
        print(str(n_img+1), "images processed", 
              "(" + str(elapsed_tm), "mins).")

elapsed_tm = (time.time() - start_tm) / 60.0
print("Total of", str(len(voc_objects)), "images.")
print("Elapsed Time:", str(round(elapsed_tm, 3)), "mins.")

print("Saving the file.")
save_pkl_file = tmp_path + "voc_data.pkl"
with open(save_pkl_file, "wb") as tmp_save:
    pkl.dump(id_2_label, tmp_save)
    pkl.dump(voc_objects, tmp_save)
print("VOC data processed.")
