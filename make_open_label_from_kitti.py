import enum
import glob
import os
import random
from logging import root

import numpy as np
import torch
from nuimages import NuImages
from nuimages.utils.utils import (annotation_name, get_font, mask_decode,
                                  name_to_index_mapping)
from PIL import Image
from skimage.transform import resize

from utils.utils import m_resize
import json

data_dir = "./data/kitti_tiny/training/"
label_list = sorted(glob.glob(data_dir + 'label_2/*.txt')) 

def category_to_class(category, filter_labels, labels_to_class):
    label = -1
    for i, l in enumerate(filter_labels):
        if l == category:
            label = i
    if label != -1 and labels_to_class[label] != -1:
        return label
    return -1

def get_bbox_class(ann, data_format):
    class_str = ann[0]
    if data_format == 'kitti':
        x1, y1, x2, y2 = float(
            ann[4]), float(
            ann[5]), float(
            ann[6]), float(
                ann[7])
    elif data_format == 'coco' or data_format == 'nuimage':
        x1, y1, x2, y2 = float(
            ann[1]), float(
            ann[2]), float(
            ann[3]), float(
                ann[4])
    return class_str, [x1, y1, x2, y2]

frames = []
for ind, label_path in enumerate(label_list):
    f = open(label_path, 'r')
    image_file_name = label_path.split(os.path.sep)[-1].replace('txt', 'jpeg')
    attr_dict = {"text": [{"view": "Front"}], "boolean": [],"others": [{"score": 0},{"bbox_id": 0},{"person_id": 0},{"lane_id": 0}]}
    objects = [] 
    for i, line in enumerate(f.readlines()):
        ann = line.rstrip().split(' ')
        class_str, bbox = get_bbox_class(ann, 'kitti')
        if class_str != 'DontCare':
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0    
        bbox = [{"name": class_str, "val": [x1, y1, x2-x1, y2-y1], "attributes":attr_dict}]
        objects.append({str(i):{"object_data":{"bbox":bbox}}})
    frames.append({str(ind):{"file":image_file_name, "objects":objects}})
annotation_dict = {"openlabel":{"frames": frames}}
with open(os.path.join(data_dir, 'OL_annotation.json'), 'w') as f:
    json.dump(annotation_dict, f)
