from __future__ import division

from ast import Num
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from PIL import Image
from torch.autograd import Variable

from utils.parse_config import *
from utils.utils import (
    bbox_iou_numpy,
    build_targets,
    calc_bbox_correction,
    compute_ap,
    non_max_suppression,
    set_device,
)


def create_modules(module_defs, img_size):
    """
    Constructs module list of layer blocks from module configuration in module_defs.
    This function is used to dynamically build the model architecture from a config file.
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        # Add convolutional, batch norm, activation, pooling, upsample, etc. layers as specified in config
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(
                scale_factor=int(module_def["stride"]), mode="nearest"
            )
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            # Route layers concatenate outputs from previous layers
            filters = 0
            for layer_i in layers:
                if layer_i > 0:
                    filters += output_filters[layer_i + 1]
                else:
                    filters += output_filters[layer_i]
            modules.add_module("route_%d" % i, EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            # YOLO detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module("yolo_%d" % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(reduction="mean")  # Coordinate loss
        self.bce_loss = nn.BCELoss(reduction="mean")  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None, classes_to_labels=None):
        # --- Unpack YOLO predictions and compute losses or output boxes ---
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        prediction = (
            x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()
        )  # nB, nA, nG, nG, n_attrs
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = (
            torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        )
        scaled_anchors = FloatTensor(
            [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        )
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))
        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:
            # --- Compute YOLO losses for training ---
            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )
            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals) if nProposals else 0

            # Handle masks
            mask = mask.type(torch.bool)
            conf_mask = conf_mask.type(torch.bool)
            # mask = Variable(mask.type(ByteTensor))
            # conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = (conf_mask.int() - mask.int()).bool()
            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(
                pred_conf[conf_mask_false], tconf[conf_mask_false]
            ) + self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])

            loss_cls = (1 / nB) * self.ce_loss(
                pred_cls[mask], torch.argmax(tcls[mask], 1)
            )
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, opt, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs, img_size)
        self.img_size = img_size
        self.num_classes = len(opt.labels)
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]
        self.opt = opt
        self.is_cuda = torch.cuda.is_available() and opt.use_cuda

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(
            zip(self.module_defs, self.module_list)
        ):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3

        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path, cutoff=-1, match_saved_weights=True):
        """Parses and loads the weights stored in 'weights_path'"""
        # Parses and loads the weights stored in 'weights_path'
        # @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        if weights_path.endswith("darknet53.conv.74"):
            cutoff = 75
        # Open the weights file
        fp = open(weights_path, "rb")
        # First five are header values
        header = np.fromfile(fp, dtype=np.int32, count=5)

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()
        ptr = 0
        for i, (module_def, module) in enumerate(
            zip(self.module_defs[:cutoff], self.module_list[:cutoff])
        ):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.bias
                    )
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.weight
                    )
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_mean
                    )
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_var
                    )
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    if (
                        self.module_defs[i + 1]["type"] == "yolo"
                        and not match_saved_weights
                    ):
                        ptr += 39
                    else:
                        num_b = conv_layer.bias.numel()
                        conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                            conv_layer.bias
                        )
                        conv_layer.bias.data.copy_(conv_b)
                        ptr += num_b
                # Load conv. weights
                if (
                    self.module_defs[i + 1]["type"] == "yolo"
                    and not match_saved_weights
                ):
                    ptr += (
                        39
                        * conv_layer.weight.shape[1]
                        * conv_layer.weight.shape[2]
                        * conv_layer.weight.shape[3]
                    )
                else:
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(
                        conv_layer.weight
                    )
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w

    """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """

    def save_weights(self, path, cutoff=-1):

        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(
            zip(self.module_defs[:cutoff], self.module_list[:cutoff])
        ):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

    def evaluate_MAF(
        self, data_loader, classes_to_labels, pre_annotation_list=None, progressBar=None
    ):
        self.eval()
        all_detections = []
        all_annotations = []
        class_correction = 0
        bbox_correction = 0.0
        bbox_creation = 0
        bbox_creation_h = 0.0
        bbox_creation_w = 0.0
        bbox_deletion = 0
        bbox_n_correction = 0
        bbox_correct = 0
        image_shape_list = []
        if progressBar is not None:
            progressBar.show()
        for batch_i, (img_paths, imgs, targets) in enumerate(data_loader):
            # progressBar.setValue(((batch_i + 1)/len(data_loader))*100)
            if pre_annotation_list is None:
                imgs = set_device(imgs, self.is_cuda)
                with torch.no_grad():
                    outputs = self.__call__(imgs)
                    outputs = non_max_suppression(
                        outputs,
                        80,
                        classes_to_labels,
                        conf_thres=self.opt.conf_thres,
                        nms_thres=self.opt.nms_thres,
                    )
            else:
                outputs = pre_annotation_list[
                    batch_i * self.opt.batch_size : (batch_i + 1) * self.opt.batch_size
                ]

            for img_path, output, annotations in zip(img_paths, outputs, targets):
                # all_detections.append([np.array([])
                #                       for _ in range(self.num_classes)])
                img = np.array(Image.open(img_path))
                h, w, _ = img.shape
                image_shape_list.append((h, w))
                if output is not None:
                    # Get predicted boxes, confidence scores and label
                    output = output.cpu().numpy()
                    pred_box = output[:, :4]
                    pred_label = output[:, [-1]]
                    scores = output[:, 4]
                    # Order by confidence
                    sort_i = np.argsort(scores)
                    output = output[sort_i]
                    all_detections.append(
                        np.concatenate([pred_box, pred_label], axis=-1)
                    )
                else:
                    all_detections.append(None)

                # all_annotations.append([np.array([])
                #                        for _ in range(self.num_classes)])
                if any(annotations[:, -1] > 0):

                    annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
                    _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                    # Reformat to x1, y1, x2, y2 and rescale to image
                    # dimensions
                    annotation_boxes = np.empty_like(_annotation_boxes)
                    annotation_boxes[:, 0] = (
                        _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                    )
                    annotation_boxes[:, 1] = (
                        _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                    )
                    annotation_boxes[:, 2] = (
                        _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                    )
                    annotation_boxes[:, 3] = (
                        _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                    )
                    annotation_boxes *= self.opt.img_size
                    all_annotations.append(
                        np.concatenate(
                            [
                                annotation_boxes,
                                np.expand_dims(annotation_labels, axis=1),
                            ],
                            axis=-1,
                        )
                    )
                else:
                    all_annotations.append(None)
        for im_shape, annotation, detection in zip(
            image_shape_list, all_annotations, all_detections
        ):
            h, w = im_shape
            pad_x = max(h - w, 0) * (self.opt.img_size / max(im_shape))
            pad_y = max(w - h, 0) * (self.opt.img_size / max(im_shape))
            unpad_h = self.opt.img_size - pad_y
            unpad_w = self.opt.img_size - pad_x
            if annotation is not None:
                for *bbox, label in annotation:
                    if detection is not None and len(detection) > 0:
                        overlaps = bbox_iou_numpy(
                            np.expand_dims(bbox, axis=0), detection[:, :-1]
                        )
                        assigned_detection = np.argmax(overlaps, axis=1)[0]
                        max_overlap = overlaps[0, assigned_detection]
                        if max_overlap >= self.opt.iou_thres:
                            if label != detection[assigned_detection, -1]:
                                class_correction += 1

                            # Image height and width after padding is removed
                            bbox_correction += calc_bbox_correction(
                                detection[assigned_detection, :-1],
                                bbox,
                                w / unpad_w,
                                h / unpad_h,
                            )
                            detection = np.delete(detection, assigned_detection, axis=0)
                            bbox_n_correction += 1
                            if max_overlap >= 0.9:
                                bbox_correct += 1
                        else:
                            bbox_creation += 1
                            bbox_creation_w += (np.abs(bbox[2] - bbox[0])) * w / unpad_w
                            bbox_creation_h += (np.abs(bbox[3] - bbox[1])) * h / unpad_h
                    else:
                        bbox_creation += 1
                        bbox_creation_w += (np.abs(bbox[2] - bbox[0])) * w / unpad_w
                        bbox_creation_h += (np.abs(bbox[3] - bbox[1])) * h / unpad_h
                bbox_deletion += len(detection) if detection is not None else 0
        result_dict = {
            "bbox_creation": bbox_creation,
            "bbox_correct": bbox_correct,
            "bbox_creation_w": bbox_creation_w,
            "bbox_creation_h": bbox_creation_h,
            "bbox_correction": bbox_correction,
            "bbox_n_correction": bbox_n_correction,
            "class_correction": class_correction,
            "bbox_deletion": bbox_deletion,
            "n_samples": len(all_annotations),
        }
        return result_dict

    def evaluate(
        self, data_loader, classes_to_labels, pre_annotation_list=None, progressBar=None
    ):
        self.eval()
        all_detections = []
        all_annotations = []
        if progressBar is not None:
            progressBar.show()
        for batch_i, (_, imgs, targets) in enumerate(data_loader):
            # progressBar.setValue(((batch_i + 1)/len(data_loader))*100)
            if pre_annotation_list is None:
                imgs = set_device(imgs, self.is_cuda)
                with torch.no_grad():
                    outputs = self.__call__(imgs)
                    outputs = non_max_suppression(
                        outputs,
                        80,
                        classes_to_labels,
                        conf_thres=self.opt.conf_thres,
                        nms_thres=self.opt.nms_thres,
                    )
            else:
                outputs = pre_annotation_list[
                    batch_i * self.opt.batch_size : (batch_i + 1) * self.opt.batch_size
                ]

            for output, annotations in zip(outputs, targets):
                all_detections.append([np.array([]) for _ in range(self.num_classes)])
                if output is not None:
                    # Get predicted boxes, confidence scores and labels
                    pred_boxes = output[:, :5].cpu().numpy()
                    scores = output[:, 4].cpu().numpy()
                    pred_labels = output[:, -1].cpu().numpy()

                    # Order by confidence
                    sort_i = np.argsort(scores)
                    pred_labels = pred_labels[sort_i]
                    pred_boxes = pred_boxes[sort_i]

                    for label in range(self.num_classes):
                        all_detections[-1][label] = pred_boxes[pred_labels == label]

                all_annotations.append([np.array([]) for _ in range(self.num_classes)])
                if any(annotations[:, -1] > 0):

                    annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
                    _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                    # Reformat to x1, y1, x2, y2 and rescale to image
                    # dimensions
                    annotation_boxes = np.empty_like(_annotation_boxes)
                    annotation_boxes[:, 0] = (
                        _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                    )
                    annotation_boxes[:, 1] = (
                        _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                    )
                    annotation_boxes[:, 2] = (
                        _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                    )
                    annotation_boxes[:, 3] = (
                        _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                    )
                    annotation_boxes *= self.opt.img_size

                    for label in range(self.num_classes):
                        all_annotations[-1][label] = annotation_boxes[
                            annotation_labels == label, :
                        ]

        average_precisions = {}
        for label in range(self.num_classes):
            if self.opt.labels_to_classes[label] != -1:
                true_positives = []
                scores = []
                num_annotations = 0

                for i in range(len(all_annotations)):
                    detections = all_detections[i][label]
                    annotations = all_annotations[i][label]

                    num_annotations += annotations.shape[0]
                    detected_annotations = []

                    for *bbox, score in detections:
                        scores.append(score)
                        if annotations.shape[0] == 0:
                            true_positives.append(0)
                            continue

                        overlaps = bbox_iou_numpy(
                            np.expand_dims(bbox, axis=0), annotations
                        )
                        assigned_annotation = np.argmax(overlaps, axis=1)
                        max_overlap = overlaps[0, assigned_annotation]

                        if (
                            max_overlap >= self.opt.iou_thres
                            and assigned_annotation not in detected_annotations
                        ):
                            true_positives.append(1)
                            detected_annotations.append(assigned_annotation)
                        else:
                            true_positives.append(0)

                # no annotations -> AP for this class is 0
                if num_annotations == 0:
                    average_precisions[label] = 0
                    continue

                true_positives = np.array(true_positives)
                false_positives = np.ones_like(true_positives) - true_positives
                # sort by score
                indices = np.argsort(-np.array(scores))
                false_positives = false_positives[indices]
                true_positives = true_positives[indices]
                # compute false positives and true positives
                false_positives = np.cumsum(false_positives)
                true_positives = np.cumsum(true_positives)

                # compute recall and precision
                recall = true_positives / num_annotations
                precision = true_positives / np.maximum(
                    true_positives + false_positives, np.finfo(np.float64).eps
                )

                # compute average precision
                average_precision = compute_ap(recall, precision)
                average_precisions[label] = average_precision

        mAP = np.mean(list(average_precisions.values()))

        return mAP, average_precisions

    def freeze_parameters(self, completed_percentage, freeze_backbone, is_training):
        if freeze_backbone and is_training:
            if completed_percentage < 0.5:
                for i, (name, p) in enumerate(self.named_parameters()):
                    if int(name.split(".")[1]) < 75:  # if layer < 75
                        p.requires_grad = False
            elif completed_percentage >= 0.5:
                for i, (name, p) in enumerate(self.named_parameters()):
                    if int(name.split(".")[1]) < 75:  # if layer < 75
                        p.requires_grad = True
