import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

import cv2

from datasets.data_aug import point_range_filter
from datasets.utils import (
    bbox3d2corners_camera,
    bbox_camera2lidar,
    bbox_lidar2camera,
    get_points_num_in_bbox,
    keep_bbox_from_image_range,
    keep_bbox_from_lidar_range,
    points_camera2image,
    read_calib,
    read_label,
    read_points,
    remove_outside_points,
    write_label,
    write_points,
)

CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR)
pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)


def bbox3d2camera2d(lidar_bboxes, image_shape, tr_velo_to_cam, r0_rect, P2):
    """
    lidar_bboxes: (N, 7)
    return bboxes_2d: (N, 4)
    """
    h, w = image_shape
    camera_bboxes = bbox_lidar2camera(lidar_bboxes, tr_velo_to_cam, r0_rect)  # (n, 7)
    alpha = camera_bboxes[:, 6] - np.arctan2(camera_bboxes[:, 0], camera_bboxes[:, 2])
    bboxes_points = bbox3d2corners_camera(camera_bboxes)  # (n, 8, 3)
    image_points = points_camera2image(bboxes_points, P2)  # (n, 8, 2)
    image_x1y1 = np.min(image_points, axis=1)  # (n, 2)
    image_x1y1 = np.maximum(image_x1y1, 0)
    image_x2y2 = np.max(image_points, axis=1)  # (n, 2)
    image_x2y2 = np.minimum(image_x2y2, [w, h])
    bboxes2d = np.concatenate([image_x1y1, image_x2y2], axis=-1)  # (n, 4)

    # keep_flag = (image_x1y1[:, 0] < w) & (image_x1y1[:, 1] < h) & (image_x2y2[:, 0] > 0) & (image_x2y2[:, 1] > 0)
    return bboxes2d, alpha


def create_kitti_xml(xml_path, annotations, calib_dict):

    root = ET.Element(
        "boost_serialization", version="9", signature="serialization::archive"
    )
    tracklets = ET.SubElement(
        root, "tracklets", version="0", tracking_level="0", class_id="0"
    )

    count_elem = ET.SubElement(tracklets, "count")
    # count_elem.text = str(len(annotations))

    item_version_elem = ET.SubElement(tracklets, "item_version")
    item_version_elem.text = "1"
    total_counts = 0
    for id, annotation in annotations.items():
        calib_info = calib_dict[id]
        tr_velo_to_cam = calib_info["Tr_velo_to_cam"].astype(np.float32)
        r0_rect = calib_info["R0_rect"].astype(np.float32)
        gt_bboxes_3d = np.concatenate(
            [
                annotation["location"],
                annotation["dimensions"],
                annotation["rotation_y"][:, None],
            ],
            axis=1,
        ).astype(np.float32)
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes_3d, tr_velo_to_cam, r0_rect)
        for j in range(len(annotation["name"])):
            total_counts += 1
            item_elem = ET.SubElement(
                tracklets, "item", version="1", tracking_level="0", class_id="1"
            )

            ET.SubElement(item_elem, "objectType").text = annotation["name"][j]
            ET.SubElement(item_elem, "h").text = str(gt_bboxes_3d[j][3])
            ET.SubElement(item_elem, "w").text = str(gt_bboxes_3d[j][4])
            ET.SubElement(item_elem, "l").text = str(gt_bboxes_3d[j][5])
            ET.SubElement(item_elem, "first_frame").text = str(int(id))
            poses_elem = ET.SubElement(item_elem, "poses")
            poses_count_elem = ET.SubElement(poses_elem, "count")
            poses_count_elem.text = "1"
            poses_item_version_elem = ET.SubElement(poses_elem, "item_version")
            poses_item_version_elem.text = "0"
            pose_elem = ET.SubElement(
                poses_elem, "item", version="1", tracking_level="0", class_id="3"
            )
            ET.SubElement(pose_elem, "tx").text = str(gt_bboxes_3d[j][0])
            ET.SubElement(pose_elem, "ty").text = str(gt_bboxes_3d[j][1])
            ET.SubElement(pose_elem, "tz").text = str(
                gt_bboxes_3d[j][2] + gt_bboxes_3d[j][5] / 2
            )
            ET.SubElement(pose_elem, "rx").text = "0.0"
            ET.SubElement(pose_elem, "ry").text = (
                "0.0"  # Assuming rotation_y is not available in annotations
            )
            ET.SubElement(pose_elem, "rz").text = str(
                np.pi * 0.5 - gt_bboxes_3d[j][6]
            )  # Assuming rotation_z is not available in annotations
            ET.SubElement(pose_elem, "state").text = "2"
            ET.SubElement(pose_elem, "occlusion").text = str(annotation["occluded"][j])
            # ET.SubElement(pose_elem, "occlusion").text = "0"
            ET.SubElement(pose_elem, "occlusion_kf").text = "0"
            ET.SubElement(pose_elem, "truncation").text = str(
                int(annotation["truncated"][j])
            )
            ET.SubElement(pose_elem, "amt_occlusion").text = "-1"
            ET.SubElement(pose_elem, "amt_border_l").text = "-1"
            ET.SubElement(pose_elem, "amt_border_r").text = "-1"
            ET.SubElement(pose_elem, "amt_occlusion_kf").text = "-1"
            ET.SubElement(pose_elem, "amt_border_kf").text = "-1"
            ET.SubElement(item_elem, "finished").text = "1"

    count_elem.text = str(total_counts)
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)


def read_kitti_xml(xml_path, data_root_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = {}
    for item in root.findall(".//item"):
        if item.find("first_frame") is not None:
            id = int(item.find("first_frame").text)
            name = item.find("objectType").text
            dimensions = [
                float(item.find("h").text),
                float(item.find("w").text),
                float(item.find("l").text),
            ]
            location_elem = item.find(".//poses/item")
            location = [
                float(location_elem.find("tx").text),
                float(location_elem.find("ty").text),
                float(location_elem.find("tz").text) - dimensions[2] / 2,
            ]
            rotation_y = np.pi * 0.5 - float(location_elem.find("rz").text)
            occluded = int(location_elem.find("occlusion").text)
            truncated = int(location_elem.find("truncation").text)

            if id not in annotations:
                annotations[id] = {
                    "name": [],
                    "dimensions": [],
                    "location": [],
                    "rotation_y": [],
                    "occluded": [],
                    "truncated": [],
                    "alpha": [],
                    "score": [],
                    "bbox": [],
                }

            annotations[id]["name"].append(name)
            annotations[id]["dimensions"].append(dimensions)
            annotations[id]["location"].append(location)
            annotations[id]["rotation_y"].append(rotation_y)
            annotations[id]["occluded"].append(occluded)
            annotations[id]["truncated"].append(truncated)
            annotations[id]["alpha"].append(0.0)
            annotations[id]["score"].append(0.0)
            annotations[id]["bbox"].append([0.0, 0.0, 0.0, 0.0])

    for k, v in annotations.items():
        annotations[k] = {k1: np.array(v1) for k1, v1 in v.items()}
    for k, v in annotations.items():
        gt_bboxes = np.concatenate(
            [v["location"], v["dimensions"], v["rotation_y"][:, None]], axis=1
        ).astype(np.float32)
        calib_dict = read_calib(os.path.join(data_root_dir, "calib", f"{k:06d}.txt"))
        img = cv2.imread(os.path.join(data_root_dir, "image_2", f"{k:06d}.png"))
        bboxes_2d, alpha = bbox3d2camera2d(
            gt_bboxes,
            img.shape[:2],
            calib_dict["Tr_velo_to_cam"],
            calib_dict["R0_rect"],
            calib_dict["P2"],
        )
        annotations[k]["bbox"] = bboxes_2d
        annotations[k]["alpha"] = alpha
        gt_bboxes = bbox_lidar2camera(
            gt_bboxes, calib_dict["Tr_velo_to_cam"], calib_dict["R0_rect"]
        )
        annotations[k]["location"] = gt_bboxes[:, :3]
        annotations[k]["dimensions"] = gt_bboxes[:, 3:6]
        annotations[k]["rotation_y"] = gt_bboxes[:, -1]
    return annotations


def convert_kitti_xml_to_txt(xml_path, txt_folder, data_root_dir):

    annotations_dict = read_kitti_xml(xml_path, data_root_dir)
    for idx, annotations in annotations_dict.items():
        write_label(annotations, os.path.join(txt_folder, f"{idx:06d}.txt"))


def format_annotations(j, data_dict, result, label_names):
    format_result = {
        "name": [],
        "truncated": [],
        "occluded": [],
        "alpha": [],
        "bbox": [],
        "dimensions": [],
        "location": [],
        "rotation_y": [],
        "score": [],
    }
    calib_info = data_dict["batched_calib_info"][0]
    tr_velo_to_cam = calib_info["Tr_velo_to_cam"].astype(np.float32)
    r0_rect = calib_info["R0_rect"].astype(np.float32)
    P2 = calib_info["P2"].astype(np.float32)
    image_shape = data_dict["batched_img_info"][0]["image_shape"]
    idx = data_dict["batched_img_info"][0]["image_idx"]
    result_filter = keep_bbox_from_image_range(
        result, tr_velo_to_cam, r0_rect, P2, image_shape
    )
    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    if len(result_filter["labels"]) == 0:
        result_filter = {
            "lidar_bboxes": np.zeros((1, 7)),
            "labels": np.zeros(1, dtype=np.int32) + -1,
            "scores": np.zeros(1),
            "bboxes2d": np.zeros((1, 4)),
            "camera_bboxes": np.zeros((1, 7)),
        }

    lidar_bboxes = result_filter["lidar_bboxes"]
    labels, scores = result_filter["labels"], result_filter["scores"]
    bboxes2d, camera_bboxes = result_filter["bboxes2d"], result_filter["camera_bboxes"]
    for _, label, score, bbox2d, camera_bbox in zip(
        lidar_bboxes, labels, scores, bboxes2d, camera_bboxes
    ):
        format_result["name"].append(label_names[label] if label != -1 else "DontCare")
        format_result["truncated"].append(0.0)
        format_result["occluded"].append(0)
        alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
        format_result["alpha"].append(alpha)
        format_result["bbox"].append(bbox2d)
        format_result["dimensions"].append(camera_bbox[3:6])
        format_result["location"].append(camera_bbox[:3])
        format_result["rotation_y"].append(camera_bbox[6])
        format_result["score"].append(score)
    return format_result


def collate_fn(list_data):
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list, batched_names_list = [], []
    batched_difficulty_list = []
    batched_img_list, batched_calib_list = [], []
    for data_dict in list_data:
        pts, gt_bboxes_3d = data_dict["pts"], data_dict["gt_bboxes_3d"]
        gt_labels, gt_names = data_dict["gt_labels"], data_dict["gt_names"]
        difficulty = data_dict["difficulty"]
        image_info, calbi_info = data_dict["image_info"], data_dict["calib_info"]

        batched_pts_list.append(torch.from_numpy(pts))
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))
        batched_names_list.append(gt_names)  # List(str)
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
        batched_calib_info=batched_calib_list,
    )

    return rt_data_dict


def judge_difficulty(annotation_dict):
    truncated = annotation_dict["truncated"]
    occluded = annotation_dict["occluded"]
    bbox = annotation_dict["bbox"]
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
    return np.array(difficultys, dtype="int")


class BaseSampler:
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
            ret = self.sampled_list[self.indices[self.idx : self.idx + num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx :]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


class Image3DAnnotationDataset(Dataset):
    def __init__(
        self,
        root_dir,
        labels,
        labels_to_classes,
        lidars_root_dir=None,
        labelled_filenames=None,
        parent=None,
    ):
        self.sep = os.path.sep
        self.lidars_path = sorted(
            glob.glob(os.path.join(root_dir, "velodyne", "*.bin*"))
        )
        self.labels_path = sorted(
            glob.glob(os.path.join(root_dir, "label_2", "*.txt*"))
        )
        self.images_path = sorted(
            glob.glob(os.path.join(root_dir, "image_2", "*.png*"))
        )
        self.calibs_path = sorted(glob.glob(os.path.join(root_dir, "calib", "*.txt*")))
        if (
            len(self.lidars_path) == 0
            and len(self.labels_path) == 0
            and parent is not None
        ):
            parent.exit_(1, f"No data found in {root_dir}")
        if labelled_filenames is not None:
            self.lidars_path = [
                l
                for l in self.lidars_path
                if l.split(self.sep)[-1].split(".")[0] not in labelled_filenames
            ]
            self.labels_path = [
                l
                for l in self.labels_path
                if l.split(self.sep)[-1].split(".")[0] not in labelled_filenames
            ]
            self.images_path = [
                l
                for l in self.images_path
                if l.split(self.sep)[-1].split(".")[0] not in labelled_filenames
            ]
            self.calibs_path = [
                l
                for l in self.calibs_path
                if l.split(self.sep)[-1].split(".")[0] not in labelled_filenames
            ]
        self.root_dir = root_dir
        self.lidars_root_dir = lidars_root_dir
        self.labels = labels
        self.labels_to_classes = labels_to_classes
        self.label_to_i = {c: i for i, c in enumerate(labels)}
        kitti_infos_dict = {}
        db = True
        for i in range(self.__len__()):
            cur_info_dict = {}
            if self.lidars_root_dir is None:
                img_path = self.images_path[i]
                lidar_path = self.lidars_path[i]
                calib_path = self.calibs_path[i]
            else:
                fname = self.labels_path[i].split(self.sep)[-1]
                lidar_path = os.path.join(
                    self.lidars_root_dir, "velodyne", fname.replace("txt", "bin")
                )
                img_path = os.path.join(
                    self.lidars_root_dir, "image_2", fname.replace("txt", "png")
                )
                calib_path = os.path.join(self.lidars_root_dir, "calib", fname)
            id = lidar_path.split(self.sep)[-1].split(".")[0]
            cur_info_dict["velodyne_path"] = self.sep.join(
                lidar_path.split(self.sep)[-2:]
            )
            img = cv2.imread(img_path)
            image_shape = img.shape[:2]
            cur_info_dict["image"] = {
                "image_shape": image_shape,
                "image_path": self.sep.join(img_path.split(self.sep)[-2:]),
                "image_idx": int(id),
            }

            calib_dict = read_calib(calib_path)
            cur_info_dict["calib"] = calib_dict

            lidar_points = read_points(lidar_path)
            reduced_lidar_points = remove_outside_points(
                points=lidar_points,
                r0_rect=calib_dict["R0_rect"],
                tr_velo_to_cam=calib_dict["Tr_velo_to_cam"],
                P2=calib_dict["P2"],
                image_shape=image_shape,
            )

            if self.lidars_root_dir is None:
                saved_reduced_path = os.path.join(root_dir, "velodyne_reduced")
                os.makedirs(saved_reduced_path, exist_ok=True)
                saved_reduced_points_name = os.path.join(
                    saved_reduced_path, f"{id}.bin"
                )
                write_points(reduced_lidar_points, saved_reduced_points_name)

            if len(self.labels_path) > 0:
                label_path = self.labels_path[i]
                annotation_dict = read_label(label_path)
                annotation_dict["difficulty"] = judge_difficulty(annotation_dict)
                annotation_dict["num_points_in_gt"] = get_points_num_in_bbox(
                    points=reduced_lidar_points,
                    r0_rect=calib_dict["R0_rect"],
                    tr_velo_to_cam=calib_dict["Tr_velo_to_cam"],
                    dimensions=annotation_dict["dimensions"],
                    location=annotation_dict["location"],
                    rotation_y=annotation_dict["rotation_y"],
                    name=annotation_dict["name"],
                )
                cur_info_dict["annos"] = annotation_dict

            kitti_infos_dict[int(id)] = cur_info_dict

        self.data_infos = kitti_infos_dict
        self.sorted_ids = list(self.data_infos.keys())

    def filter_labels(self, annos_info):
        keep_ids = [
            i
            for i, name in enumerate(annos_info["name"])
            if name in self.labels
            and self.labels_to_classes[self.label_to_i[name]] != -1
        ]
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        # 1. filter_by_difficulty
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item["difficulty"] != -1]

        # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10)
        filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10)
        for cat in self.label_to_i:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [
                item for item in db_infos[cat] if item["num_points_in_gt"] >= filter_thr
            ]

        return db_infos

    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_ids[index]]
        image_info, calib_info, annos_info = (
            data_info["image"],
            data_info["calib"],
            data_info["annos"],
        )

        # point cloud input
        velodyne_path = data_info["velodyne_path"].replace(
            "velodyne", "velodyne_reduced"
        )
        # velodyne_path = data_info['velodyne_path']
        pts_path = os.path.join(
            self.root_dir if self.lidars_root_dir is None else self.lidars_root_dir,
            velodyne_path,
        )
        pts = read_points(pts_path)

        # calib input: for bbox coordinates transformation between Camera and Lidar.
        # because
        tr_velo_to_cam = calib_info["Tr_velo_to_cam"].astype(np.float32)
        r0_rect = calib_info["R0_rect"].astype(np.float32)

        # annotations input
        annos_info = self.filter_labels(annos_info)
        annos_name = annos_info["name"]
        annos_location = annos_info["location"]
        annos_dimension = annos_info["dimensions"]
        rotation_y = annos_info["rotation_y"]
        gt_bboxes = np.concatenate(
            [annos_location, annos_dimension, rotation_y[:, None]], axis=1
        ).astype(np.float32)
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
        gt_labels = [self.label_to_i[name] for name in annos_name]
        data_dict = {
            "pts": pts,
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels": np.array(gt_labels),
            "gt_names": annos_name,
            "difficulty": annos_info["difficulty"],
            "image_info": image_info,
            "calib_info": calib_info,
            "lidar_path": velodyne_path,
        }
        # if self.split in ['train', 'trainval']:
        #     data_dict = data_augment(self.CLASSES, self.root_dir, data_dict, self.data_aug_config)
        # else:
        data_dict = point_range_filter(
            data_dict, point_range=[0, -39.68, -3, 69.12, 39.68, 1]
        )

        return data_dict

    def __len__(self):
        dataset_len = max(len(self.lidars_path), len(self.labels_path))
        return dataset_len


if __name__ == "__main__":

    kitti_data = Image3DAnnotationDataset(root_dir="/mnt/ssd1/lifa_rdata/det/kitti")
    kitti_data.__getitem__(9)
