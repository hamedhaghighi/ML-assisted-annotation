import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.anchors import Anchors, anchor_target, anchors2bboxes
from models.utils import (
    get_score_thresholds,
    iou2d,
    iou3d_camera,
    iou_bev,
    limit_period,
)
from ops import Voxelization, nms_cuda


class Loss(nn.Module):
    def __init__(
        self, alpha=0.25, gamma=2.0, beta=1 / 9, cls_w=1.0, reg_w=2.0, dir_w=0.2
    ):
        super().__init__()
        self.alpha = 0.25
        self.gamma = 2.0
        self.cls_w = cls_w
        self.reg_w = reg_w
        self.dir_w = dir_w
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none", beta=beta)
        self.dir_cls = nn.CrossEntropyLoss()

    def forward(
        self,
        bbox_cls_pred,
        bbox_pred,
        bbox_dir_cls_pred,
        batched_labels,
        num_cls_pos,
        batched_bbox_reg,
        batched_dir_labels,
    ):
        """
        bbox_cls_pred: (n, 3)
        bbox_pred: (n, 7)
        bbox_dir_cls_pred: (n, 2)
        batched_labels: (n, )
        num_cls_pos: int
        batched_bbox_reg: (n, 7)
        batched_dir_labels: (n, )
        return: loss, float.
        """
        # 1. bbox cls loss
        # focal loss: FL = - \alpha_t (1 - p_t)^\gamma * log(p_t)
        #             y == 1 -> p_t = p
        #             y == 0 -> p_t = 1 - p
        nclasses = bbox_cls_pred.size(1)
        batched_labels = F.one_hot(batched_labels, nclasses + 1)[
            :, :nclasses
        ].float()  # (n, 3)

        bbox_cls_pred_sigmoid = torch.sigmoid(bbox_cls_pred)
        weights = self.alpha * (1 - bbox_cls_pred_sigmoid).pow(
            self.gamma
        ) * batched_labels + (1 - self.alpha) * bbox_cls_pred_sigmoid.pow(
            self.gamma
        ) * (
            1 - batched_labels
        )  # (n, 3)
        cls_loss = F.binary_cross_entropy(
            bbox_cls_pred_sigmoid, batched_labels, reduction="none"
        )
        cls_loss = cls_loss * weights
        cls_loss = cls_loss.sum() / num_cls_pos

        # 2. regression loss
        reg_loss = self.smooth_l1_loss(bbox_pred, batched_bbox_reg)
        reg_loss = reg_loss.sum() / reg_loss.size(0)

        # 3. direction cls loss
        dir_cls_loss = self.dir_cls(bbox_dir_cls_pred, batched_dir_labels)

        # 4. total loss
        total_loss = (
            self.cls_w * cls_loss + self.reg_w * reg_loss + self.dir_w * dir_cls_loss
        )

        loss_dict = {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "dir_cls_loss": dir_cls_loss,
            "total_loss": total_loss,
        }
        return loss_dict


def pre_process_tensors(data_dict, is_cuda=True):
    if is_cuda:
        # move the tensors to the cuda
        for key in data_dict:
            for j, item in enumerate(data_dict[key]):
                if torch.is_tensor(item):
                    data_dict[key][j] = data_dict[key][j].cuda()
    batched_pts = data_dict["batched_pts"]
    batched_gt_bboxes = data_dict["batched_gt_bboxes"]
    batched_labels = data_dict["batched_labels"]
    batched_difficulty = data_dict["batched_difficulty"]
    return data_dict


class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )

    @torch.no_grad()
    def forward(self, batched_pts):
        """
        batched_pts: list[tensor], len(batched_pts) = bs
        return:
               pillars: (p1 + p2 + ... + pb, num_points, c),
               coors_batch: (p1 + p2 + ... + pb, 1 + 3),
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        """
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        pillars = torch.cat(pillars, dim=0)  # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(
            npoints_per_pillar, dim=0
        )  # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)  # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        """
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        """
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = (
            pillars[:, :, :3]
            - torch.sum(pillars[:, :, :3], dim=1, keepdim=True)
            / npoints_per_pillar[:, None, None]
        )  # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (
            coors_batch[:, None, 1:2] * self.vx + self.x_offset
        )  # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (
            coors_batch[:, None, 2:3] * self.vy + self.y_offset
        )  # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat(
            [pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1
        )  # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center  # tmp
        features[:, :, 1:2] = y_offset_pi_center  # tmp
        # In consitent with mmdet3d.
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)  # (num_points, )
        mask = (
            voxel_ids[:, None] < npoints_per_pillar[None, :]
        )  # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(
            0, 2, 1
        ).contiguous()  # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(
            self.bn(self.conv(features))
        )  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[
            0
        ]  # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros(
                (self.x_l, self.y_l, self.out_channel),
                dtype=torch.float32,
                device=device,
            )
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(
            batched_canvas, dim=0
        )  # (bs, in_channel, self.y_l, self.x_l)
        return batched_canvas


class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(
                nn.Conv2d(
                    in_channel,
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    bias=False,
                    padding=1,
                )
            )
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(
                    nn.Conv2d(
                        out_channels[i], out_channels[i], 3, bias=False, padding=1
                    )
                )
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        """
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(
                nn.ConvTranspose2d(
                    in_channels[i],
                    out_channels[i],
                    upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False,
                )
            )
            decoder_block.append(
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01)
            )
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        """
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()

        self.conv_cls = nn.Conv2d(in_channel, n_anchors * n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors * 7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors * 2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        """
        x: (bs, 384, 248, 216)
        return:
              bbox_cls_pred: (bs, n_anchors*3, 248, 216)
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        """
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class PointPillars(nn.Module):
    def __init__(
        self,
        opt,
        nclasses=3,
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        max_num_points=32,
        max_voxels=(16000, 40000),
    ):
        super().__init__()
        self.nclasses = nclasses
        self.pillar_layer = PillarLayer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )
        self.pillar_encoder = PillarEncoder(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            in_channel=9,
            out_channel=64,
        )
        self.backbone = Backbone(
            in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5]
        )
        self.neck = Neck(
            in_channels=[64, 128, 256],
            upsample_strides=[1, 2, 4],
            out_channels=[128, 128, 128],
        )
        self.head = Head(in_channel=384, n_anchors=2 * nclasses, n_classes=nclasses)

        # anchors
        ranges = [
            [0, -39.68, -0.6, 69.12, 39.68, -0.6],
            [0, -39.68, -0.6, 69.12, 39.68, -0.6],
            [0, -39.68, -1.78, 69.12, 39.68, -1.78],
        ]
        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
        rotations = [0, 1.57]
        self.anchors_generator = Anchors(
            ranges=ranges, sizes=sizes, rotations=rotations
        )

        # train
        self.assigners = [
            {"pos_iou_thr": 0.5, "neg_iou_thr": 0.35, "min_iou_thr": 0.35},
            {"pos_iou_thr": 0.5, "neg_iou_thr": 0.35, "min_iou_thr": 0.35},
            {"pos_iou_thr": 0.6, "neg_iou_thr": 0.45, "min_iou_thr": 0.45},
        ]
        self.opt = opt
        self.is_cuda = torch.cuda.is_available() and opt.use_cuda
        # val and test
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50
        self.Loss = Loss()

    def get_output(self, data_dict, classes_to_labels=None):
        if self.is_cuda:
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()
        batched_pts = data_dict["batched_pts"]
        batched_gt_bboxes = data_dict["batched_gt_bboxes"]
        batched_labels = data_dict["batched_labels"]
        batched_difficulty = data_dict["batched_difficulty"]
        ret_results = []
        batch_results = self.__call__(
            batched_pts=batched_pts,
            mode="val",
            batched_gt_bboxes=batched_gt_bboxes,
            batched_gt_labels=batched_labels,
        )
        for res in batch_results:
            keep_ind = []
            if isinstance(res, dict) and classes_to_labels is not None:
                keep_ind = [
                    i for i, l in enumerate(res["labels"]) if classes_to_labels[l] != -1
                ]
            if len(keep_ind) > 0:
                ret_results.append({k: v[keep_ind] for k, v in res.items()})
            else:
                bboxes = np.zeros((1, 7))
                labels = np.zeros(1) + -1
                scores = np.zeros(1)
                ret_results.append(
                    {"lidar_bboxes": bboxes, "labels": labels, "scores": scores}
                )
        return ret_results

    def get_bbox_scores(self, data_dict, classes_to_labels):
        results_dict = self.get_output(data_dict, classes_to_labels)
        return [res["scores"].mean() for res in results_dict]

    def evaluate(self, det_results, gt_results, CLASSES, print_fn):
        assert len(det_results) == len(gt_results)
        # 1. calculate iou
        ious = {"bbox_2d": [], "bbox_bev": [], "bbox_3d": []}
        ids = list(sorted(gt_results.keys()))
        for id in ids:
            gt_result = gt_results[id]["annos"]
            det_result = det_results[id]

            # 1.1, 2d bboxes iou
            gt_bboxes2d = gt_result["bbox"].astype(np.float32)
            det_bboxes2d = det_result["bbox"].astype(np.float32)
            iou2d_v = iou2d(
                torch.from_numpy(gt_bboxes2d).cuda(),
                torch.from_numpy(det_bboxes2d).cuda(),
            )
            ious["bbox_2d"].append(iou2d_v.cpu().numpy())

            # 1.2, bev iou
            gt_location = gt_result["location"].astype(np.float32)
            gt_dimensions = gt_result["dimensions"].astype(np.float32)
            gt_rotation_y = gt_result["rotation_y"].astype(np.float32)
            det_location = det_result["location"].astype(np.float32)
            det_dimensions = det_result["dimensions"].astype(np.float32)
            det_rotation_y = det_result["rotation_y"].astype(np.float32)

            gt_bev = np.concatenate(
                [
                    gt_location[:, [0, 2]],
                    gt_dimensions[:, [0, 2]],
                    gt_rotation_y[:, None],
                ],
                axis=-1,
            )
            det_bev = np.concatenate(
                [
                    det_location[:, [0, 2]],
                    det_dimensions[:, [0, 2]],
                    det_rotation_y[:, None],
                ],
                axis=-1,
            )
            iou_bev_v = iou_bev(
                torch.from_numpy(gt_bev).cuda(), torch.from_numpy(det_bev).cuda()
            )
            ious["bbox_bev"].append(iou_bev_v.cpu().numpy())

            # 1.3, 3dbboxes iou
            gt_bboxes3d = np.concatenate(
                [gt_location, gt_dimensions, gt_rotation_y[:, None]], axis=-1
            )
            det_bboxes3d = np.concatenate(
                [det_location, det_dimensions, det_rotation_y[:, None]], axis=-1
            )
            iou3d_v = iou3d_camera(
                torch.from_numpy(gt_bboxes3d).cuda(),
                torch.from_numpy(det_bboxes3d).cuda(),
            )
            ious["bbox_3d"].append(iou3d_v.cpu().numpy())

        MIN_IOUS = {
            "Pedestrian": [0.5, 0.5, 0.5],
            "Cyclist": [0.5, 0.5, 0.5],
            "Car": [0.7, 0.7, 0.7],
        }
        MIN_HEIGHT = [40, 25, 25]

        overall_results = {}
        for e_ind, eval_type in enumerate(["bbox_2d", "bbox_bev", "bbox_3d"]):
            eval_ious = ious[eval_type]
            eval_ap_results, eval_aos_results = {}, {}
            for cls in CLASSES:
                eval_ap_results[cls] = []
                eval_aos_results[cls] = []
                CLS_MIN_IOU = MIN_IOUS[cls][e_ind]
                # for difficulty in [0, 1, 2]:
                # 1. bbox property
                total_gt_ignores, total_det_ignores, total_dc_bboxes, total_scores = (
                    [],
                    [],
                    [],
                    [],
                )
                total_gt_alpha, total_det_alpha = [], []
                for id in ids:
                    gt_result = gt_results[id]["annos"]
                    det_result = det_results[id]

                    # 1.1 gt bbox property
                    cur_gt_names = gt_result["name"]
                    cur_difficulty = gt_result["difficulty"]
                    gt_ignores, dc_bboxes = [], []
                    for j, cur_gt_name in enumerate(cur_gt_names):
                        # ignore = cur_difficulty[j] < 0 or cur_difficulty[j] > difficulty
                        ignore = False
                        if cur_gt_name == cls:
                            valid_class = 1
                        elif cls == "Pedestrian" and cur_gt_name == "Person_sitting":
                            valid_class = 0
                        elif cls == "Car" and cur_gt_name == "Van":
                            valid_class = 0
                        else:
                            valid_class = -1

                        if valid_class == 1 and not ignore:
                            gt_ignores.append(0)
                        elif valid_class == 0 or (valid_class == 1 and ignore):
                            gt_ignores.append(1)
                        else:
                            gt_ignores.append(-1)

                        if cur_gt_name == "DontCare":
                            dc_bboxes.append(gt_result["bbox"][j])
                    total_gt_ignores.append(gt_ignores)
                    total_dc_bboxes.append(np.array(dc_bboxes))
                    total_gt_alpha.append(gt_result["alpha"])

                    # 1.2 det bbox property
                    cur_det_names = det_result["name"]
                    cur_det_heights = (
                        det_result["bbox"][:, 3] - det_result["bbox"][:, 1]
                    )
                    det_ignores = []
                    for j, cur_det_name in enumerate(cur_det_names):
                        # if cur_det_heights[j] < MIN_HEIGHT[difficulty]:
                        #     det_ignores.append(1)
                        if cur_det_name == cls:
                            det_ignores.append(0)
                        else:
                            det_ignores.append(-1)
                    total_det_ignores.append(det_ignores)
                    total_scores.append(det_result["score"])
                    total_det_alpha.append(det_result["alpha"])

                # 2. calculate scores thresholds for PR curve
                tp_scores = []
                for i, id in enumerate(ids):
                    cur_eval_ious = eval_ious[i]
                    gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                    scores = total_scores[i]
                    nn, mm = cur_eval_ious.shape
                    assigned = np.zeros((mm,), dtype=np.bool_)
                    for j in range(nn):
                        if gt_ignores[j] == -1:
                            continue
                        match_id, match_score = -1, -1
                        for k in range(mm):
                            if (
                                not assigned[k]
                                and det_ignores[k] >= 0
                                and cur_eval_ious[j, k] > CLS_MIN_IOU
                                and scores[k] > match_score
                            ):
                                match_id = k
                                match_score = scores[k]
                        if match_id != -1:
                            assigned[match_id] = True
                            if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                tp_scores.append(match_score)
                total_num_valid_gt = np.sum(
                    [
                        np.sum(np.array(gt_ignores) == 0)
                        for gt_ignores in total_gt_ignores
                    ]
                )
                score_thresholds = get_score_thresholds(tp_scores, total_num_valid_gt)

                # 3. draw PR curve and calculate mAP
                tps, fns, fps, total_aos = [], [], [], []

                for score_threshold in score_thresholds:
                    tp, fn, fp = 0, 0, 0
                    aos = 0
                    for i, id in enumerate(ids):
                        cur_eval_ious = eval_ious[i]
                        gt_ignores, det_ignores = (
                            total_gt_ignores[i],
                            total_det_ignores[i],
                        )
                        gt_alpha, det_alpha = total_gt_alpha[i], total_det_alpha[i]
                        scores = total_scores[i]

                        nn, mm = cur_eval_ious.shape
                        assigned = np.zeros((mm,), dtype=np.bool_)
                        for j in range(nn):
                            if gt_ignores[j] == -1:
                                continue
                            match_id, match_iou = -1, -1
                            for k in range(mm):
                                if (
                                    not assigned[k]
                                    and det_ignores[k] >= 0
                                    and scores[k] >= score_threshold
                                    and cur_eval_ious[j, k] > CLS_MIN_IOU
                                ):

                                    if (
                                        det_ignores[k] == 0
                                        and cur_eval_ious[j, k] > match_iou
                                    ):
                                        match_iou = cur_eval_ious[j, k]
                                        match_id = k
                                    elif det_ignores[k] == 1 and match_iou == -1:
                                        match_id = k

                            if match_id != -1:
                                assigned[match_id] = True
                                if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                    tp += 1
                                    if eval_type == "bbox_2d":
                                        aos += (
                                            1
                                            + np.cos(gt_alpha[j] - det_alpha[match_id])
                                        ) / 2
                            else:
                                if gt_ignores[j] == 0:
                                    fn += 1

                        for k in range(mm):
                            if (
                                det_ignores[k] == 0
                                and scores[k] >= score_threshold
                                and not assigned[k]
                            ):
                                fp += 1

                        # In case 2d bbox evaluation, we should consider dontcare bboxes
                        if eval_type == "bbox_2d":
                            dc_bboxes = total_dc_bboxes[i]
                            det_bboxes = det_results[id]["bbox"]
                            if len(dc_bboxes) > 0:
                                ious_dc_det = (
                                    iou2d(
                                        torch.from_numpy(det_bboxes),
                                        torch.from_numpy(dc_bboxes),
                                        metric=1,
                                    )
                                    .numpy()
                                    .T
                                )
                                for j in range(len(dc_bboxes)):
                                    for k in range(len(det_bboxes)):
                                        if (
                                            det_ignores[k] == 0
                                            and scores[k] >= score_threshold
                                            and not assigned[k]
                                        ):
                                            if ious_dc_det[j, k] > CLS_MIN_IOU:
                                                fp -= 1
                                                assigned[k] = True

                    tps.append(tp)
                    fns.append(fn)
                    fps.append(fp)
                    if eval_type == "bbox_2d":
                        total_aos.append(aos)
                tps, fns, fps = np.array(tps), np.array(fns), np.array(fps)

                recalls = tps / (tps + fns)
                precisions = tps / (tps + fps)
                for i in range(len(score_thresholds)):
                    precisions[i] = np.max(precisions[i:])

                sums_AP = 0
                for i in range(0, len(score_thresholds), 4):
                    sums_AP += precisions[i]
                mAP = sums_AP / np.ceil(len(score_thresholds) / 4) * 100
                eval_ap_results[cls].append(mAP)

                if eval_type == "bbox_2d":
                    total_aos = np.array(total_aos)
                    similarity = total_aos / (tps + fps)
                    for i in range(len(score_thresholds)):
                        similarity[i] = np.max(similarity[i:])
                    sums_similarity = 0
                    for i in range(0, len(score_thresholds), 4):
                        sums_similarity += similarity[i]
                    mSimilarity = (
                        sums_similarity / np.ceil(len(score_thresholds) / 4) * 100
                    )
                    eval_aos_results[cls].append(mSimilarity)

            print_fn(f"=========={eval_type.upper()}==========")
            print_fn(f"=========={eval_type.upper()}==========")
            for k, v in eval_ap_results.items():
                print_fn(f"{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f}")
                print_fn(f"{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f}")
            if eval_type == "bbox_2d":
                print_fn(f"==========AOS==========")
                print_fn(f"==========AOS==========")
                for k, v in eval_aos_results.items():
                    print_fn(f"{k} AOS@{MIN_IOUS[k][e_ind]}: {v[0]:.4f}")
                    print_fn(f"{k} AOS@{MIN_IOUS[k][e_ind]}: {v[0]:.4f}")

            overall_results[eval_type] = np.mean(list(eval_ap_results.values()), 0)
            if eval_type == "bbox_2d":
                overall_results["AOS"] = np.mean(list(eval_aos_results.values()), 0)

        print_fn(f"\n==========Overall==========")
        print_fn(f"\n==========Overall==========")
        for k, v in overall_results.items():
            print_fn(f"{k} AP: {v[0]:.4f}")
            print_fn(f"{k} AP: {v[0]:.4f}")
        return overall_results

    def m_train(
        self,
        data_dict,
        optimizer,
        scheduler,
        labels_to_classes,
        vis,
        len_dataloader,
        epoch,
        batch_i,
        nclasses,
        print_msg_fn,
        total_steps,
    ):
        if self.is_cuda:
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()
        batched_pts = data_dict["batched_pts"]
        batched_gt_bboxes = data_dict["batched_gt_bboxes"]
        batched_labels = data_dict["batched_labels"]
        batched_difficulty = data_dict["batched_difficulty"]
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = self.__call__(
            batched_pts=batched_pts,
            mode="train",
            batched_gt_bboxes=batched_gt_bboxes,
            batched_gt_labels=batched_labels,
        )
        bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, nclasses)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
        # changing target labels to clasess
        batched_bbox_labels = anchor_target_dict["batched_labels"].reshape(-1)
        if labels_to_classes is not None:
            # we dont want to change locations where there is no annotation
            has_annotation_ind = batched_bbox_labels != len(labels_to_classes)
            if self.is_cuda:
                tensor_labels_to_class = torch.tensor(labels_to_classes).cuda()
            # x, y = torch.nonzero(has_annotation_ind, as_tuple=True)
            batched_bbox_labels[has_annotation_ind] = tensor_labels_to_class[
                batched_bbox_labels[has_annotation_ind].long()
            ].to(batched_bbox_labels)
        batched_label_weights = anchor_target_dict["batched_label_weights"].reshape(-1)
        batched_bbox_reg = anchor_target_dict["batched_bbox_reg"].reshape(-1, 7)
        # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
        batched_dir_labels = anchor_target_dict["batched_dir_labels"].reshape(-1)
        # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
        pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < nclasses)
        bbox_pred = bbox_pred[pos_idx]
        batched_bbox_reg = batched_bbox_reg[pos_idx]
        # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
        bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(
            batched_bbox_reg[:, -1].clone()
        )
        batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(
            batched_bbox_reg[:, -1].clone()
        )
        bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
        batched_dir_labels = batched_dir_labels[pos_idx]

        num_cls_pos = (batched_bbox_labels < nclasses).sum()
        bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
        batched_bbox_labels[batched_bbox_labels < 0] = nclasses
        batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

        loss_dict = self.Loss(
            bbox_cls_pred=bbox_cls_pred,
            bbox_pred=bbox_pred,
            bbox_dir_cls_pred=bbox_dir_cls_pred,
            batched_labels=batched_bbox_labels,
            num_cls_pos=num_cls_pos,
            batched_bbox_reg=batched_bbox_reg,
            batched_dir_labels=batched_dir_labels,
        )

        loss = loss_dict["total_loss"]
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
        optimizer.step()
        scheduler.step()

        if batch_i % 1 == 0:
            for tag, value in loss_dict.items():
                vis.plot("losses_" + tag, value, total_steps)
            vis.plot("total_loss", loss.item(), total_steps)
            if not self.opt.background_training:
                msg = "[Epoch %d/%d, Batch %d/%d] [Losses: " % (
                    epoch,
                    self.opt.epochs,
                    batch_i,
                    len_dataloader,
                )
                for k, v in loss_dict.items():
                    msg += "%s: %f, " % (k, v)
                msg += "total: %f]" % (loss.item())
                print_msg_fn(msg)

    def get_predicted_bboxes_single(
        self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors
    ):
        """
        bbox_cls_pred: (n_anchors*3, 248, 216)
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return:
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, )
        """
        # 0. pre-process
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)

        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]
        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat(
            [
                bbox_pred2d_xy - bbox_pred2d_lw / 2,
                bbox_pred2d_xy + bbox_pred2d_lw / 2,
                bbox_pred[:, 6:],
            ],
            dim=-1,
        )  # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]

            # 3.2 nms core
            if self.is_cuda:
                keep_inds = nms_cuda(
                    boxes=cur_bbox_pred2d,
                    scores=cur_bbox_cls_pred,
                    thresh=self.nms_thr,
                    pre_maxsize=None,
                    post_max_size=None,
                )

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(
                cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi
            ).to(
                cur_bbox_pred
            )  # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(
                torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i
            )
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return [], [], []

        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            "lidar_bboxes": ret_bboxes.detach().cpu().numpy(),
            "labels": ret_labels.detach().cpu().numpy(),
            "scores": ret_scores.detach().cpu().numpy(),
        }
        return result

    def get_predicted_bboxes(
        self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors
    ):
        """
        bbox_cls_pred: (bs, n_anchors*3, 248, 216)
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return:
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ]
        """
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(
                bbox_cls_pred=bbox_cls_pred[i],
                bbox_pred=bbox_pred[i],
                bbox_dir_cls_pred=bbox_dir_cls_pred[i],
                anchors=batched_anchors[i],
            )
            results.append(result)
        return results

    def forward(
        self, batched_pts, mode="test", batched_gt_bboxes=None, batched_gt_labels=None
    ):  #
        batch_size = len(batched_pts)
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c),
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3),
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        # xs:  [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        xs = self.backbone(pillar_features)

        # x: (bs, 384, 248, 216)
        x = self.neck(xs)

        # bbox_cls_pred: (bs, n_anchors*3, 248, 216)
        # bbox_pred: (bs, n_anchors*7, 248, 216)
        # bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]
        if mode == "train":
            anchor_target_dict = anchor_target(
                batched_anchors=batched_anchors,
                batched_gt_bboxes=batched_gt_bboxes,
                batched_gt_labels=batched_gt_labels,
                assigners=self.assigners,
                nclasses=self.nclasses,
            )

            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode == "val":
            results = self.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred,
                bbox_pred=bbox_pred,
                bbox_dir_cls_pred=bbox_dir_cls_pred,
                batched_anchors=batched_anchors,
            )
            return results

        elif mode == "test":
            results = self.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred,
                bbox_pred=bbox_pred,
                bbox_dir_cls_pred=bbox_dir_cls_pred,
                batched_anchors=batched_anchors,
            )
            return results
        else:
            raise ValueError
