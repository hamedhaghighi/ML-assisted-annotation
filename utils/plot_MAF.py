import os
import pickle
from cProfile import label
from distutils.command.config import config
from re import M
from turtle import color

import matplotlib.pyplot as plt
import numpy as np

def title_change(s):
    classes = ['Car', 'Truck', 'Pedestrian']
    for c in classes:
        if c in s:
            return c
    return 'Average'

def average_mAP_improvement(mAP_list):
    max = 0.0
    # for i , m in enumerate(mAP_list):
    #     if m > max:
    #         max = m
    #     else:
    #         mAP_list[i] = max
    mAP_list = np.array(mAP_list).cumsum()
    mAP_list_t = mAP_list.copy()
    for i, m in enumerate(mAP_list):
        if i > 0:
            mAP_list[i] += (len(mAP_list) - i) * \
                (mAP_list_t[i] - mAP_list_t[i - 1])
        else:
            mAP_list[i] += len(mAP_list - i) * mAP_list_t[i]
    mAP_list = mAP_list / len(mAP_list)
    mAP_list = ((mAP_list - mAP_list[0]) / mAP_list[0]) * 100
    return mAP_list.tolist()


exp_name_1 = 'calc_MAF'


def convert_to_percentage(vector, fraction):
    return [str(int(v * fraction * 100)) + '%' for v in vector]


kitti_num_samples = 7481
fraction_1 = 100 / int(1.0 * kitti_num_samples)
fraction_2 = 500 / int(1.0 * kitti_num_samples)
fraction_3 = 1000 / int(1.0 * kitti_num_samples)

with open(os.path.join('checkpoints', exp_name_1, 'random_subset_MAF.pkl'), 'rb') as f:
    MAF_dict = pickle.load(f)
# with open(os.path.join('checkpoints', exp_name_1, 'random_subset_avg_p_dict.pkl'), 'rb') as f:
#     subset_avg_p_dict = pickle.load(f)

MAF_dict = {k: np.array(v) for k ,v in MAF_dict.items()}
n_gt_bbx= MAF_dict['bbox_n_correction'] + MAF_dict['bbox_creation']
n_pred_bbx= MAF_dict['bbox_n_correction'] + MAF_dict['bbox_deletion']
MAF_dict['bbox_correction'] = MAF_dict['bbox_correction'] / MAF_dict['bbox_n_correction']
MAF_dict['class_correction'] = MAF_dict['class_correction'] / MAF_dict['bbox_n_correction']
MAF_dict['bbox_creation'] = MAF_dict['bbox_creation'] / n_gt_bbx
MAF_dict['bbox_deletion'] = MAF_dict['bbox_deletion'] / n_pred_bbx
MAF_dict.pop('bbox_n_correction');MAF_dict.pop('bbox_creation_h');MAF_dict.pop('bbox_creation_w');MAF_dict.pop('n_samples')
for k , v in MAF_dict.items():
    # plt.subplot(3, 2, 1)
    plt.figure()
    l_v = len(v)
    # l_v = 31
    width = 0.5
    x = np.arange(l_v).tolist()
    plt.plot(x, v)
    # plt.bar(x, v, width=width)
    x_ticks = x[::5]
    # x_ticks.append(x[-1])
    x_ticks_labels = convert_to_percentage(x_ticks, fraction_1)
    # x_ticks_labels.append('100%')
    # y_ticks_array = np.arange(-50, 105, 10)
    plt.xticks(x_ticks, x_ticks_labels)
    # plt.yticks(y_ticks_array, [str(a) + '%' for a in y_ticks_array])
    # plt.title('Average mAP gain')
    plt.xlabel('Percentage of labelling data')
    plt.ylabel(k)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.8)
plt.show()

# plt.subplot(3, 2, 3)
# v1 = average_mAP_improvement(val_avg_p_dict_500['val_mAP'])
# l_v = len(v1)
# # l_v = 31
# width = 0.5
# x = np.arange(l_v).tolist()
# y = v1[:l_v]
# plt.bar(x, y, width=width)
# x_ticks = x
# # x_ticks.append(x[-1])
# x_ticks_labels = convert_to_percentage(x_ticks, fraction_2)
# # x_ticks_labels.append('100%')
# y_ticks_array = np.arange(0, 105, 10)
# plt.xticks(x_ticks, x_ticks_labels)
# plt.yticks(y_ticks_array, [str(a) + '%' for a in y_ticks_array])
# # plt.title('Average mAP gain')
# plt.xlabel('Percentage of labelling data')
# plt.ylabel('Percentage of mAP gain')
# # plt.legend(loc='lower right')
# plt.grid(alpha=0.8)


# plt.subplot(3, 2, 5)
# v1 = average_mAP_improvement(val_avg_p_dict_1000['val_mAP'])
# l_v = len(v1)
# # l_v = 31
# width = 0.5
# x = np.arange(l_v).tolist()
# y = v1[:l_v]
# plt.bar(x, y, width=width)
# x_ticks = x
# # x_ticks.append(x[-1])
# x_ticks_labels = convert_to_percentage(x_ticks, fraction_3)
# # x_ticks_labels.append('100%')
# y_ticks_array = np.arange(0, 105, 10)
# plt.xticks(x_ticks, x_ticks_labels)
# plt.yticks(y_ticks_array, [str(a) + '%' for a in y_ticks_array])
# # plt.title('Average mAP gain')
# plt.xlabel('Percentage of labelling data')
# plt.ylabel('Percentage of mAP gain')
# # plt.legend(loc='lower right')
# plt.grid(alpha=0.8)

# plt.show()
