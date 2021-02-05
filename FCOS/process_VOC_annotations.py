
import os
import pandas as pd
from bs4 import BeautifulSoup

tmp_path = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/Annotations/"
img_path = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/JPEGImages/"
tmp_xmls = os.listdir(tmp_path)

tmp_count = 0
tmp_objects = []
for tmp_xml in tmp_xmls:
    with open(tmp_path+tmp_xml) as xml_file:
        tmp_soup = BeautifulSoup(xml_file.read())
        
        tmp_image  = img_path+tmp_soup.find("filename").text
        tmp_object = tmp_soup.find("object").find("name").text
        tmp_objects.append((tmp_image, tmp_object))
    
    tmp_count += 1
    if tmp_count % 1000 == 0:
        print(str(tmp_count/len(tmp_xmls)*100) +\
              "% of annotations processed.")

tmp_pd_file = \
    "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/voc_2012_annotations.csv"
tmp_objects_df = pd.DataFrame(
    tmp_objects, columns=["image_file", "object_class"])
tmp_objects_df.to_csv(tmp_pd_file, index=False)


