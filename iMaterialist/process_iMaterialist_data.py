
import json
import pandas as pd

tmp_path = "C:/Users/admin/Desktop/Data/iMaterialist/"
img_path = tmp_path + "train2020/train/"

tmp_json_file = tmp_path + "instances_attributes_train2020.json"
with open(tmp_json_file) as tmp_file_open:
    tmp_json = json.load(tmp_file_open)

tmp_categories  = tmp_json["categories"]
tmp_image_files = tmp_json["images"]
tmp_annotations = tmp_json["annotations"]

label_dict = dict([
    (x["id"], x["name"]) for x in tmp_categories])
name_to_id = dict([
    (x["name"], x["id"]) for x in tmp_categories])
image_dict = dict([
    (x["id"], x["file_name"]) for x in tmp_image_files])

tmp_data = []
image_df = pd.DataFrame(tmp_image_files)
for tmp_annotation in tmp_annotations:
    tmp_img_file = \
        img_path + image_dict[tmp_annotation["image_id"]]
    tmp_category = tmp_annotation["category_id"]
    
    tmp_img_meta = image_df[
        image_df["id"] == tmp_annotation["image_id"]].iloc[0]
    tmp_img_width  = tmp_img_meta["width"]
    tmp_img_height = tmp_img_meta["height"]
    
    tmp_bbox = tmp_annotation["bbox"]
    tmp_xcen = tmp_bbox[0] + tmp_bbox[2] / 2.0
    tmp_ycen = tmp_bbox[1] + tmp_bbox[3] / 2.0
    tmp_data.append((
        tmp_img_file, tmp_img_width, tmp_img_height, 
        tmp_xcen, tmp_ycen, tmp_bbox[2], tmp_bbox[3], tmp_category))

tmp_cols_df = [
    "filename", "img_width", "img_height", 
    "x_cen", "y_cen", "box_width", "box_height", "category_id"]
tmp_data_df = pd.DataFrame(tmp_data, columns=tmp_cols_df)
tmp_data_df.to_csv(tmp_path + "iMaterialist.csv", index=False)
