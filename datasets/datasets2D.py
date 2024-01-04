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
from torch.utils.data import Dataset

from utils.utils import m_resize, unpack_pre_annotation
import json
from collections import defaultdict


def write_annotations(ind, data_dict, pre_annotation, pre_label_dir, labels, opt):
    img_path = data_dict['im_path']
    img = np.array(Image.open(img_path))
    h, w, _ = img.shape
    pad_x = max(h - w, 0) * (opt.img_size / max(img.shape))
    pad_y = max(w - h, 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x
    sep = '\\' if os.name =='nt' else '/'
    img_filename = img_path.split(sep)[-1]
    if opt.data_format == 'kitti':
        label_filename = img_filename.replace('jpeg', 'txt')
        with open(os.path.join(pre_label_dir, label_filename), 'w') as f:
            if pre_annotation is not None:
                pre_annotation = pre_annotation.cpu().numpy()
                for pre_ann in pre_annotation:
                    line = ['-1.0'] * 15; line[1], line[2] = '0.0', '3'
                    c, x1, y1, x2, y2 = unpack_pre_annotation(pre_ann, pad_x, pad_y, unpad_h, unpad_w, h, w, labels)
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
                    line[0], line[4], line[5], line[6], line[7] = c, str(x1), str(y1), str(x2), str(y2)
                    f.write(' '.join(line) + '\n')
                
            # else:
            line = ['-1.0'] * 15; line[0], line[1], line[2] = 'DontCare', '0.0', '3' 
            f.write(' '.join(line) + '\n')
    elif opt.data_format == 'openlabel':
        objects = [] 
        attr_dict = {"text": [{"view": "Front"}], "boolean": [],"others": [{"score": 0},{"person_id": 0},{"bbox_id": 0},{"lane_id": 0}]}
        if pre_annotation is not None:
                pre_annotation = pre_annotation.cpu().numpy()
                for i , pre_ann in enumerate(pre_annotation):
                    c, x1, y1, x2, y2 = unpack_pre_annotation(pre_ann, pad_x, pad_y, unpad_h, unpad_w, h, w, labels)
                    width, height = x2 - x1, y2 - y1
                    bbox = [{"name": c, "val": [x1, y1, width, height], "attributes":attr_dict}]
                    objects.append({str(i):{"object_data":{"bbox":bbox}}})
        else:
            bbox = [{"name": 'DontCare', "val": [0.0, 0.0, 0.0, 0.0], "attributes":attr_dict}]
            objects.append({"0":{"object_data":{"bbox":bbox}}})
        return {str(ind):{"file":img_filename, "objects":objects}}

def collate_fn(list_data):
    rt_data_dict = defaultdict(list)
    for data_dict in list_data:
        for k , v in data_dict.items():
            rt_data_dict[k].append(v)
            
    rt_data_dict = {k:torch.stack(v, dim=0) if  torch.is_tensor(v[0]) else v for k, v in rt_data_dict.items() }
    return rt_data_dict

def category_to_class(category, filter_labels, labels_to_class):
    label = -1
    for i, l in enumerate(filter_labels):
        if l == category:
            label = i
    if label != -1 and labels_to_class[label] != -1:
        return label
    return -1


class NuImageDataset(Dataset):
    def __init__(self, root_dir, filter_lables, labels_to_classes,
                 img_size=416, resize_tuple=None):
        self.root_dir = root_dir
        self.filter_labels = filter_lables
        self.labels_to_classes = labels_to_classes
        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        nuim = NuImages(
            dataroot=self.root_dir,
            version='v1.0-mini',
            verbose=False,
            lazy=False)
        self.sample = nuim.sample
        self.sample_data = nuim.sample_data
        self.object_ann = nuim.object_ann
        self.category = nuim.category
        self.resize_tuple = resize_tuple

    def __getitem__(self, index):
        # reading image
        sample = self.sample[index]
        key_camera_token = sample['key_camera_token']
        sample_data = [
            s for s in self.sample_data if s['token'] == key_camera_token][0]
        im_path = os.path.join(self.root_dir, sample_data['filename'])
        im_resized, ratio = m_resize(Image.open(im_path), self.resize_tuple)
        img = np.array(im_resized)
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (
            (pad1, pad2), (0, 0), (0, 0)) if h <= w else (
            (0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = torch.from_numpy(input_img).float()
        # reading label

        object_anns = [
            o for o in self.object_ann if o['sample_data_token'] == key_camera_token]
        labels = np.zeros((self.max_objects, 5))
        idx = 0
        for ann in object_anns:
            if idx < self.max_objects:
                category_token = ann['category_token']
                category_name = [
                    c for c in self.category if c['token'] == category_token][0]
                c = category_to_class(
                    category_name['name'], self.filter_labels)
                if c != -1 and self.labels_to_classes[c] != -1:
                    labels[idx, 0] = c
                    bbox = [a * (ratio if ratio is not None else 1)
                            for a in ann['bbox']]
                    labels[idx, 1] = (
                        (bbox[0] + bbox[2] + 2 * pad[1][0]) / 2) / padded_w
                    labels[idx, 2] = (
                        (bbox[1] + bbox[3] + 2 * pad[0][0]) / 2) / padded_h
                    labels[idx, 3] = (bbox[2] - bbox[0]) / padded_w
                    labels[idx, 4] = (bbox[3] - bbox[1]) / padded_h
                    idx = idx + 1
        filled_labels = torch.from_numpy(labels)
        return im_path, input_img, filled_labels

    def __len__(self):
        return len(self.sample)


def coco_label_to_label(self, coco_label):
    return self.coco_labels_inverse[coco_label]


def get_image_labels_path(root_dir, data_format):
    if data_format == 'kitti':
        root_dir = os.path.join(root_dir, 'image_2')
        images_path = sorted(glob.glob(f'{root_dir}/*.*'))
        labels_path = sorted(glob.glob(f'{root_dir}/*.*' .replace('image','label')))
    elif data_format == 'nuimage':
        images_path = []
        labels_path = []
        if not os.path.exists(os.path.join(root_dir, 'images_path.txt')):
            nuim = NuImages(
                dataroot=root_dir,
                version='v1.0-mini',
                verbose=False,
                lazy=False)
            all_sample = nuim.sample
            sample_data = nuim.sample_data
            object_ann = nuim.object_ann
            category = nuim.category
            label_dir = os.path.join(root_dir, 'labels')
            os.makedirs(label_dir, exist_ok=True)
            for idx in range(len(all_sample)):
                sample = all_sample[idx]
                key_camera_token = sample['key_camera_token']
                data = [s for s in sample_data if s['token']
                        == key_camera_token][0]
                im_path = os.path.join(root_dir, data['filename'])
                images_path.append(im_path)
                l_path = os.path.join(label_dir, '{0:0>5}.txt'.format(idx))
                labels_path.append(l_path)
                object_anns = [
                    o for o in object_ann if o['sample_data_token'] == key_camera_token]
                f = open(l_path, 'w')
                for ann in object_anns:
                    bbox = ann['bbox']
                    category_token = ann['category_token']
                    category_name = [
                        c for c in category if c['token'] == category_token][0]
                    line_to_write = category_name['name'] + ' ' + str(bbox[0]) + ' ' + str(
                        bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + '\n'
                    f.write(line_to_write)
                f.close()
            with open(os.path.join(root_dir, 'labels_path.txt'), 'w') as f:
                f.writelines('\n'.join(labels_path) + '\n')
            with open(os.path.join(root_dir, 'images_path.txt'), 'w') as f:
                f.writelines('\n'.join(images_path) + '\n')
        else:
            with open(os.path.join(root_dir, 'labels_path.txt'), 'r') as f:
                labels_path = f.readlines()
            with open(os.path.join(root_dir, 'images_path.txt'), 'r') as f:
                images_path = f.readlines()
    elif data_format == 'openlabel':
        img_root_dir = os.path.join(root_dir, 'image_2')
        images_path = sorted(glob.glob(f'{img_root_dir}/*.*'))
        labels_path = []
        with open(os.path.join(root_dir, 'OL_annotation.json'), 'r') as f:
            ann_dict = json.load(f)
            labels_path = ann_dict['openlabel']['frames']

    elif data_format == 'coco':
        coco = COCO(
            os.path.join(
                root_dir,
                'annotations',
                'instances_' +
                'minitrain2017' +
                '.json'))
        categories = coco.loadCats(coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        id_category = {}
        for c in categories:
            id_category[c['id']] = c['name'].replace(' ', '_')
        image_ids = coco.getImgIds()
        images_path = []
        labels_path = []
        if not os.path.exists(os.path.join(root_dir, 'images_path.txt')):
            label_dir = os.path.join(root_dir, 'labels')
            os.makedirs(label_dir, exist_ok=True)
            for idx, id in enumerate(image_ids):
                image_info = coco.loadImgs(id)[0]
                path = os.path.join(
                    root_dir,
                    'images',
                    'minitrain2017',
                    image_info['file_name'])
                images_path.append(path)
                l_path = os.path.join(label_dir, '{0:0>5}.txt'.format(idx))
                labels_path.append(l_path)
                annotations_ids = coco.getAnnIds(imgIds=id, iscrowd=False)
                f = open(l_path, 'w')
                coco_annotations = coco.loadAnns(annotations_ids)
                for a in coco_annotations:
                    if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                        continue

                    x1, y1, x2, y2 = a['bbox'][0], a['bbox'][1], a['bbox'][2] + \
                        a['bbox'][0], a['bbox'][3] + a['bbox'][1]
                    c = id_category[a['category_id']]
                    line_to_write = c + ' ' + \
                        str(x1) + ' ' + str(y1) + ' ' + \
                        str(x2) + ' ' + str(y2) + '\n'
                    f.write(line_to_write)
                f.close()

            with open(os.path.join(root_dir, 'labels_path.txt'), 'w') as f:
                f.writelines('\n'.join(labels_path) + '\n')
            with open(os.path.join(root_dir, 'images_path.txt'), 'w') as f:
                f.writelines('\n'.join(images_path) + '\n')
        else:
            with open(os.path.join(root_dir, 'labels_path.txt'), 'r') as f:
                labels_path = f.readlines()
            with open(os.path.join(root_dir, 'images_path.txt'), 'r') as f:
                images_path = f.readlines()
    else:
        raise ('data format is not found!')
    return images_path, labels_path


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


class Image2DAnnotationDataset(Dataset):
    img_extension = ''
    def __init__(self, root_dir, filter_lables, labels_to_classes, data_format, img_size=416, resize_tuple=None, img_root_dir=None, labelled_filenames=None, parent=None):
        self.root_dir = root_dir
        self.images_path, self.labels_path = get_image_labels_path(root_dir, data_format)
        if Image2DAnnotationDataset.img_extension == '':
            Image2DAnnotationDataset.img_extension = self.images_path[0].split('.')[-1]
        if len(self.images_path) == 0 and parent is not None:
            parent.exit_(1, f'No data found in {root_dir}')
        self.sep = os.path.sep
        if labelled_filenames is not None:
            self.images_path = [img for img in self.images_path if img.split(self.sep)[-1].split('.')[0] not in  labelled_filenames]
            if data_format == 'kitti':
                self.labels_path = [l for l in self.labels_path if l.split(self.sep)[-1].split('.')[0] not in labelled_filenames]
            elif data_format == 'openlabel':
                self.labels_path = [l for l in self.labels_path if l[list(l.keys())[0]]['file'].split('.')[0] not in labelled_filenames]
                
        self.filter_labels = filter_lables
        self.labels_to_classes = labels_to_classes
        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        self.resize_tuple = resize_tuple
        self.data_format = data_format
        self.img_root_dir = img_root_dir

    def __getitem__(self, index):
        # reading image
        if len(self.labels_path) == 0:
            label_path = None
        else:
            if self.data_format == 'openlabel':
                label_path = self.labels_path[index]
            else:
                label_path = self.labels_path[index].rstrip()
        if self.img_root_dir is not None:
            if self.data_format == 'kitti':
                img_name = label_path.split(self.sep)[-1].replace('txt', Image2DAnnotationDataset.img_extension)
                
                im_path = os.path.join(self.img_root_dir, 'image_2', img_name)
            elif self.data_format == 'openlabel':
                label_file_name = label_path[list(label_path.keys())[0]]['file'].replace('txt', Image2DAnnotationDataset.img_extension)
                im_path = os.path.join(self.img_root_dir, 'image_2', label_file_name)
        else:
            im_path = self.images_path[index].rstrip()
        im_resized, ratio = m_resize(Image.open(im_path), self.resize_tuple)
        img = np.array(im_resized)
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1).repeat(3, -1)
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = torch.from_numpy(input_img).float()
        # reading label
        labels = np.zeros((self.max_objects, 5))
        if label_path is not None:
            if self.data_format == 'openlabel':
                idx = 0
                object_list = label_path[list(label_path.keys())[0]]['objects']
                for ind, obj in enumerate(object_list):
                    bbox = obj[str(ind)]['object_data']['bbox'][0]
                    class_str = bbox['name']
                    if idx < self.max_objects:
                        c = category_to_class(class_str, self.filter_labels, self.labels_to_classes)
                        if c!= -1:
                            x1, y1, width, height = bbox['val']
                            labels[idx, 0] = c
                            labels[idx, 1] = (x1 + width/2 + pad[1][0]) / padded_w
                            labels[idx, 2] = (y1 + height/2 + pad[0][0]) / padded_h
                            labels[idx, 3] = width / padded_w
                            labels[idx, 4] = height / padded_h
                            idx = idx + 1
            else:
                idx = 0
                f = open(label_path, 'r')
                for line in f.readlines():
                    ann = line.rstrip().split(' ')
                    class_str, bbox = get_bbox_class(ann, self.data_format)
                    if idx < self.max_objects:
                        c = category_to_class(class_str, self.filter_labels, self.labels_to_classes)
                        if c != -1:
                            labels[idx, 0] = c
                            bbox = [b * (ratio if ratio is not None else 1) for b in bbox]
                            labels[idx, 1] = ((bbox[0] + bbox[2] + 2 * pad[1][0]) / 2) / padded_w
                            labels[idx, 2] = ((bbox[1] + bbox[3] + 2 * pad[0][0]) / 2) / padded_h
                            labels[idx, 3] = (bbox[2] - bbox[0]) / padded_w
                            labels[idx, 4] = (bbox[3] - bbox[1]) / padded_h
                            idx = idx + 1
        filled_labels = torch.from_numpy(labels)
        rt_data_dict= dict(im_path = im_path,
         input_img = input_img, filled_labels = filled_labels)
        return rt_data_dict

    def __len__(self):
        dataset_len = max(len(self.images_path), len(self.labels_path))
        return dataset_len
