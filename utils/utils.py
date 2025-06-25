from __future__ import division

import datetime
import math
import os
import pickle
import random
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullLocator
from PIL import Image
from PyQt5.QtWidgets import QMessageBox
from torch.utils.tensorboard import SummaryWriter


class Log:
    def __init__(self, opt, resume):
        self.labelled_filenames = []
        self.selected_filenames = []
        self.opt = opt
        self.log_dir = os.path.join(self.opt.checkpoint_dir, "log.pkl")
        if os.path.exists(self.log_dir):
            self.load()
        else:
            self.save()

    def load(self):
        with open(self.log_dir, "rb") as f:
            dict = pickle.load(f)
        self.labelled_filenames = dict["labelled_filenames"]
        self.selected_filenames = dict["selected_filenames"]

    def save(self):
        with open(self.log_dir, "wb") as f:
            dict = {
                "labelled_filenames": self.labelled_filenames,
                "selected_filenames": self.selected_filenames,
            }
            pickle.dump(dict, f)

    def update_selected(self, filenames):
        self.selected_filenames = filenames
        self.save()

    def update_labelled(self):
        self.labelled_filenames.extend(self.selected_filenames)
        self.selected_filenames = []
        self.save()


class Visualizer:
    """ """

    def __init__(self, checkpoint_dir):
        self.tb_dir = os.path.join(
            checkpoint_dir, "TB", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.makedirs(self.tb_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter(self.tb_dir)

        self.index = {}
        self.log_text = ""

    def plot(self, tag, loss, step):
        """
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        self.writer.add_scalar(tag, loss, step)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def img(self, name, img_):
        """
        self.img('input_img',t.Tensor(64,64))
        """

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(), win=name, opts=dict(title=name))

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        self.img(
            name, tv.utils.make_grid(input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0))
        )

    def log(self, info, win="log_text"):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += "[{time}] {info} <br>".format(
            time=time.strftime("%m%d_%H%M%S"), info=info
        )
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


def check_opt(opt, print_fn):
    resp = True
    conf_var_names = set(
        [
            "exp_name",
            "model_name",
            "data_format",
            "labels",
            "labels_to_classes",
            "use_cuda",
            "data_dir",
            "annotation_tool",
        ]
    )
    missed_keys = list(set(conf_var_names).difference(opt.keys()))
    if len(missed_keys) > 0:
        resp = False
        print_fn(
            f"The selected config file does not contain the following keys: {', '.join(missed_keys)}",
            is_error=True,
        )
    is_two_d = "yolo" in opt["model_name"]
    data_dir = opt["data_dir"]
    if (
        is_two_d
        and opt["data_format"] == "kitti"
        and not os.path.exists(os.path.join(opt["data_dir"], "image_2"))
    ):
        resp = False
        print_fn(
            f"The path {data_dir} does not contain image_2. Please check the structure of kitti 2D object detection dataset.",
            is_error=True,
        )
    if not is_two_d and (
        not os.path.exists(os.path.join(opt["data_dir"], "velodyne"))
        or not os.path.exists(os.path.join(data_dir, "calib"))
    ):
        resp = False
        print_fn(
            f"The path {data_dir} does not contain velodyne or calib. Please check the structure of kitti 3D object detection dataset.",
            is_error=True,
        )
    if not is_two_d and opt["data_format"] != "kitti":
        resp = False
        print_fn(
            f"Only kitti data format is supported for 3D object detection.",
            is_error=True,
        )
    if not is_two_d and not opt["use_cuda"]:
        resp = False
        print_fn(f"The cuda should be enabled for 3D object detection.", is_error=True)
    return resp


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = (
        np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1)
        + area
        - iw * ih
    )

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def non_max_suppression(
    prediction, num_classes, classes_to_labels=None, conf_thres=0.5, nms_thres=0.4
):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf,
        # class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1
        )
        # map and filter based on classes to labels
        if classes_to_labels is not None:
            detections[:, -1] = classes_to_labels[detections[:, -1].long()].float()
            detections = detections[detections[:, -1] >= 0.0]

        if not detections.size(0):
            continue
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max
                # detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections
                if output[image_i] is None
                else torch.cat((output[image_i], max_detections))
            )

    return output


def build_targets(
    pred_boxes,
    pred_conf,
    pred_cls,
    target,
    anchors,
    num_anchors,
    num_classes,
    grid_size,
    ignore_thres,
    img_dim,
):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)
    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(
                np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1)
            )
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero
            # (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])


def draw_bbox(img_path, detections, classes, kitti_img_size=416, resize_tuple=None):
    resized_pil, _ = m_resize(Image.open(img_path), resize_tuple)
    img = np.array(resized_pil)
    cmap = plt.get_cmap("PuBuGn_r")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    # plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # kitti_img_size = 11*32
    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (kitti_img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (kitti_img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = kitti_img_size - pad_y
    unpad_w = kitti_img_size - pad_x
    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if (x1 + y1 + x2 + y2) != 0:
                # print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                # Rescale coordinates to original dimensions
                box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
                box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]))
                y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
                x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle(
                    (x1, y1),
                    box_w,
                    box_h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1 - 30,
                    s=classes[int(cls_pred)] + " " + str("%.4f" % cls_conf.item()),
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()
    plt.close()
    # plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    # plt.close()
    return


def m_resize(PIL_img, resize_tuple):
    if resize_tuple is not None:
        w, h = PIL_img.size
        ratio = (resize_tuple[0] / w, resize_tuple[1] / h)
        ind = np.argmin(
            np.array([abs(1 - resize_tuple[0] / w), abs(1 - resize_tuple[1] / h)])
        )
        return PIL_img.resize((int(ratio[ind] * w), int(ratio[ind] * h))), ratio[ind]
    return PIL_img, None


def convert_target_to_detection(target, img_size):
    if target.sum() != 0:
        output = torch.zeros(target.shape[0], 7)
        target[:, 1:] = target[:, 1:] * img_size
        output[:, 0] = target[:, 1] - target[:, 3] / 2
        output[:, 1] = target[:, 2] - target[:, 4] / 2
        output[:, 2] = target[:, 1] + target[:, 3] / 2
        output[:, 3] = target[:, 2] + target[:, 4] / 2
        output[:, -1] = target[:, 0]
        return output
    return None


def make_ordinal(n):
    """
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    """
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix


def set_device(pyt_element, is_cuda):
    if is_cuda:
        pyt_element = pyt_element.cuda()
    return pyt_element


def show_msgbox(parent, msg, button=None, type="info", is_gui=True):
    if is_gui:
        msgBox = QMessageBox()
        msgBox.setStyleSheet(
            "QMessageBox { background-color: #353535;}"
            "QMessageBox QLabel { color: white; }"
            "QMessageBox QPushButton { background-color: #505050; color: white; }"
        )
        icon = QMessageBox.Information if type == "info" else QMessageBox.Critical
        msgBox.setIcon(icon)
        msgBox.setText(msg)
        # msgBox.setInformativeText( "Do you really want to disable safety enforcement?" )
        if button == "OK":
            msgBox.addButton(QMessageBox.Ok)
        elif button == "yes/no":
            msgBox.addButton(QMessageBox.No)
            msgBox.addButton(QMessageBox.Yes)
        ret = msgBox.exec_()
        return ret
    else:
        print(msg)


def unpack_pre_annotation(pre_annotation, pad_x, pad_y, unpad_h, unpad_w, h, w, labels):
    x1, y1, x2, y2, _, _, c = pre_annotation
    c = labels[int(c)]
    y1 = ((y1 - pad_y // 2) / unpad_h) * h
    x1 = ((x1 - pad_x // 2) / unpad_w) * w
    y2 = ((y2 - pad_y // 2) / unpad_h) * h
    x2 = ((x2 - pad_x // 2) / unpad_w) * w
    return c, x1, y1, x2, y2
