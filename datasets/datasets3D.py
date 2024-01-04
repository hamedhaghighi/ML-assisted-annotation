import numpy as np
import os
import glob
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from datasets.utils import  read_points, bbox_camera2lidar
from datasets.data_aug import point_range_filter
from tqdm import tqdm, trange
import cv2
from datasets.utils import read_points, write_points, read_calib, read_label, \
    write_pickle, remove_outside_points, get_points_num_in_bbox, \
    points_in_bboxes_v2, keep_bbox_from_image_range, keep_bbox_from_lidar_range, write_label

CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR)
pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

def write_annotations(j, data_dict, result, pre_label_dir, label_names, opt=None):
    format_result = {
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': []
    }
    calib_info = data_dict['batched_calib_info'][0]
    tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
    r0_rect = calib_info['R0_rect'].astype(np.float32)
    P2 = calib_info['P2'].astype(np.float32)
    image_shape = data_dict['batched_img_info'][0]['image_shape']
    idx = data_dict['batched_img_info'][0]['image_idx']
    result_filter = keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape)
    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)

    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']
    bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
    for lidar_bbox, label, score, bbox2d, camera_bbox in \
        zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
        format_result['name'].append(label_names[label])
        format_result['truncated'].append(0.0)
        format_result['occluded'].append(0)
        alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
        format_result['alpha'].append(alpha)
        format_result['bbox'].append(bbox2d)
        format_result['dimensions'].append(camera_bbox[3:6])
        format_result['location'].append(camera_bbox[:3])
        format_result['rotation_y'].append(camera_bbox[6])
        format_result['score'].append(score)
    
    write_label(format_result, os.path.join(pre_label_dir, f'{idx:06d}.txt'))
    return format_result
    
def collate_fn(list_data):
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list, batched_names_list = [], []
    batched_difficulty_list = []
    batched_img_list, batched_calib_list = [], []
    for data_dict in list_data:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']
        difficulty = data_dict['difficulty']
        image_info, calbi_info = data_dict['image_info'], data_dict['calib_info']

        batched_pts_list.append(torch.from_numpy(pts))
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))
        batched_names_list.append(gt_names) # List(str)
        batched_difficulty_list.append(torch.from_numpy(difficulty))
        batched_img_list.append(image_info)
        batched_calib_list.append(calbi_info)
    
    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_difficulty=batched_difficulty_list,
        batched_img_info=batched_img_list,
        batched_calib_info=batched_calib_list
    )

    return rt_data_dict


def judge_difficulty(annotation_dict):
    truncated = annotation_dict['truncated']
    occluded = annotation_dict['occluded']
    bbox = annotation_dict['bbox']
    height = bbox[:, 3] - bbox[:, 1]

    MIN_HEIGHTS = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.30, 0.50]
    difficultys = []
    for h, o, t in zip(height, occluded, truncated):
        difficulty = -1
        for i in range(2, -1, -1):
            if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                difficulty = i
        difficultys.append(difficulty)
    return np.array(difficultys, dtype='int')

class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


class Image3DAnnotationDataset(Dataset):
    def __init__(self, root_dir, labels, labels_to_classes, lidars_root_dir=None, labelled_filenames=None, parent=None):
        self.sep = os.path.sep
        self.lidars_path = sorted(glob.glob(f'{root_dir}/velodyne/*.*'))
        self.labels_path = sorted(glob.glob(f'{root_dir}/label_2/*.*'))
        self.images_path = sorted(glob.glob(f'{root_dir}/image_2/*.*' ))
        self.calibs_path = sorted(glob.glob(f'{root_dir}/calib/*.*' ))
        if len(self.lidars_path) == 0  and len(self.labels_path) == 0 and parent is not None:
            parent.exit_(1, f'No data found in {root_dir}')
        if labelled_filenames is not None:
            self.lidars_path = [l for l in self.lidars_path if l.split(self.sep)[-1].split('.')[0] not in labelled_filenames]
            self.labels_path = [l for l in self.labels_path if l.split(self.sep)[-1].split('.')[0] not in labelled_filenames]
            self.images_path = [l for l in self.images_path if l.split(self.sep)[-1].split('.')[0] not in labelled_filenames]
            self.calibs_path = [l for l in self.calibs_path if l.split(self.sep)[-1].split('.')[0] not in labelled_filenames]
        self.root_dir = root_dir
        self.lidars_root_dir = lidars_root_dir
        self.labels = labels
        self.labels_to_classes = labels_to_classes
        self.label_to_i = {c:i for i,c in enumerate(labels)}
        kitti_infos_dict = {}
        db = True
        for i in range(self.__len__()):
            cur_info_dict={}
            if self.lidars_root_dir is None:
                img_path = self.images_path[i]
                lidar_path = self.lidars_path[i]
                calib_path = self.calibs_path[i]
            else:
                fname = self.labels_path[i].split(self.sep)[-1]                
                lidar_path = os.path.join(self.lidars_root_dir, 'velodyne', fname.replace('txt', 'bin'))
                img_path = os.path.join(self.lidars_root_dir, 'image_2', fname.replace('txt', 'png'))
                calib_path = os.path.join(self.lidars_root_dir, 'calib', fname)
            id = lidar_path.split(self.sep)[-1].split('.')[0]
            cur_info_dict['velodyne_path'] = self.sep.join(lidar_path.split(self.sep)[-2:])
            img = cv2.imread(img_path)
            image_shape = img.shape[:2]
            cur_info_dict['image'] = {
                'image_shape': image_shape,
                'image_path': self.sep.join(img_path.split(self.sep)[-2:]), 
                'image_idx': int(id),
            }

            calib_dict = read_calib(calib_path)
            cur_info_dict['calib'] = calib_dict

            lidar_points = read_points(lidar_path)
            reduced_lidar_points = remove_outside_points(
                points=lidar_points, 
                r0_rect=calib_dict['R0_rect'], 
                tr_velo_to_cam=calib_dict['Tr_velo_to_cam'], 
                P2=calib_dict['P2'], 
                image_shape=image_shape)

            if self.lidars_root_dir is None:
                saved_reduced_path = os.path.join(root_dir, 'velodyne_reduced')
                os.makedirs(saved_reduced_path, exist_ok=True)
                saved_reduced_points_name = os.path.join(saved_reduced_path, f'{id}.bin')
                write_points(reduced_lidar_points, saved_reduced_points_name)

            if len(self.labels_path) > 0:
                label_path = self.labels_path[i]
                annotation_dict = read_label(label_path)
                annotation_dict['difficulty'] = judge_difficulty(annotation_dict)
                annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(
                    points=reduced_lidar_points,
                    r0_rect=calib_dict['R0_rect'], 
                    tr_velo_to_cam=calib_dict['Tr_velo_to_cam'],
                    dimensions=annotation_dict['dimensions'],
                    location=annotation_dict['location'],
                    rotation_y=annotation_dict['rotation_y'],
                    name=annotation_dict['name'])
                cur_info_dict['annos'] = annotation_dict

                # if db:
                #     indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
                #         points_in_bboxes_v2data_infosnnotation_dict['dimensions'].astype(np.float32),
                #             location=annotation_dict['location'].astype(np.float32),
                #             rotation_y=annotation_dict['rotation_y'].astype(np.float32),
                #             name=annotation_dict['name']    
                #         )
                #     for j in range(n_valid_bbox):
                #         db_points = lidar_points[indices[:, j]]
                #         db_points[:, :3] -= bboxes_lidar[j, :3]
                #         db_points_saved_name = os.path.join(db_points_saved_path, f'{int(id)}_{name[j]}_{j}.bin')
                #         write_points(db_points, db_points_saved_name)

                #         db_info={
                #             'name': name[j],
                #             'path': os.path.join(os.path.basename(db_points_saved_path), f'{int(id)}_{name[j]}_{j}.bin'),
                #             'box3d_lidar': bboxes_lidar[j],
                #             'difficulty': annotation_dict['difficulty'][j], 
                #             'num_points_in_gt': len(db_points), 
                #         }
                #         if name[j] not in kitti_dbinfos_train:
                #             kitti_dbinfos_train[name[j]] = [db_info]
                #         else:
                #             kitti_dbinfos_train[name[j]].append(db_info)
            
            kitti_infos_dict[int(id)] = cur_info_dict
        # writing the dict
        # saved_path = os.path.join(root_dir, f'{prefix}_infos_{data_type}.pkl')
        # write_pickle(kitti_infos_dict, saved_path)
        # if db:
        #     saved_db_path = os.path.join(root_dir, f'{prefix}_dbinfos_train.pkl')
        #     write_pickle(kitti_dbinfos_train, saved_db_path)


        self.data_infos = kitti_infos_dict
        self.sorted_ids = list(self.data_infos.keys())
        # db_infos = read_pickle(os.path.join(root_dir, 'kitti_dbinfos_train.pkl'))
        # db_infos = self.filter_db(db_infos)
        # db_sampler = {}
        # for cat_name in self.CLASSES:
        #     db_sampler[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
        # self.data_aug_config=dict(
        #     db_sampler=dict(
        #         db_sampler=db_sampler,
        #         sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10)
        #         ),
        #     object_noise=dict(
        #         num_try=100,
        #         translation_std=[0.25, 0.25, 0.25],
        #         rot_range=[-0.15707963267, 0.15707963267]
        #         ),
        #     random_flip_ratio=0.5,
        #     global_rot_scale_trans=dict(
        #         rot_range=[-0.78539816, 0.78539816],
        #         scale_ratio_range=[0.95, 1.05],
        #         translation_std=[0, 0, 0]
        #         ), 
        #     point_range_filter=[0, -39.68, -3, 69.12, 39.68, 1],
        #     object_range_filter=[0, -39.68, -3, 69.12, 39.68, 1]             
        # )

    def filter_labels(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info['name']) if name in self.labels and self.labels_to_classes[self.label_to_i[name]] != -1]
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        # 1. filter_by_difficulty
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item['difficulty'] != -1]

        # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10)
        filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10)
        for cat in self.label_to_i:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
        
        return db_infos

    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_ids[index]]
        image_info, calib_info, annos_info = \
            data_info['image'], data_info['calib'], data_info['annos']
    
        # point cloud input
        # TODO you may need to change the line below
        velodyne_path = data_info['velodyne_path'].replace('velodyne', 'velodyne_reduced')
        # velodyne_path = data_info['velodyne_path']
        pts_path = os.path.join(self.root_dir if self.lidars_root_dir is None else self.lidars_root_dir, velodyne_path)
        pts = read_points(pts_path)
        
        # calib input: for bbox coordinates transformation between Camera and Lidar.
        # because
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)

        # annotations input
        annos_info = self.filter_labels(annos_info)
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
        gt_labels = [self.label_to_i[name] for name in annos_name]
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': np.array(gt_labels), 
            'gt_names': annos_name,
            'difficulty': annos_info['difficulty'],
            'image_info': image_info,
            'calib_info': calib_info,
            'lidar_path': velodyne_path
        }
        # if self.split in ['train', 'trainval']:
        #     data_dict = data_augment(self.CLASSES, self.root_dir, data_dict, self.data_aug_config)
        # else:
        data_dict = point_range_filter(data_dict, point_range=[0, -39.68, -3, 69.12, 39.68, 1])

        return data_dict

    def __len__(self):
        dataset_len = max(len(self.lidars_path), len(self.labels_path))
        return dataset_len
 

if __name__ == '__main__':
    
    kitti_data = Image3DAnnotationDataset(root_dir='/mnt/ssd1/lifa_rdata/det/kitti')
    kitti_data.__getitem__(9)
