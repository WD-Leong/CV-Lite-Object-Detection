# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:47:37 2020

@author: admin
"""

import os
import pandas as pd
from bs4 import BeautifulSoup

tmp_path  = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/Annotations"
tmp_files = [(tmp_path + "/" + x) for x in os.listdir(tmp_path)]

tmp_list = []
for tmp_file in tmp_files:
    tmp_bs = BeautifulSoup(open(tmp_file).read(), features="lxml")
    
    tmp_img = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/JPEGImages/"
    tmp_img += tmp_bs.find("filename").text
    
    tmp_dim = tmp_bs.find("size")
    tmp_w = int(tmp_dim.find("width").text)
    tmp_h = int(tmp_dim.find("height").text)
    
    tmp_objects = tmp_bs.find_all("object")
    if len(tmp_objects) > 0:
        for tmp_object in tmp_objects:
            tmp_name = tmp_object.find("name").text
            tmp_xmin = int(float(tmp_object.find("xmin").text))
            tmp_xmax = int(float(tmp_object.find("xmax").text))
            tmp_ymin = int(float(tmp_object.find("ymin").text))
            tmp_ymax = int(float(tmp_object.find("ymax").text))
            
            tmp_list.append((tmp_img, tmp_w, tmp_h, tmp_name, 
                             tmp_xmin, tmp_xmax, tmp_ymin, tmp_ymax))

# Convert into a DataFrame. #
tmp_cols = ["filename", "width", "height", 
            "label", "xmin", "xmax", "ymin", "ymax"]
tmp_df = pd.DataFrame(tmp_list, columns=tmp_cols)

# Save the file. #
tmp_pd_file = \
    "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/voc_2012_objects.csv"
tmp_df.to_csv(tmp_pd_file, index=False)
