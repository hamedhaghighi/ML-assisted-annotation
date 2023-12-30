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


exp_name_1 = 'calc_MAF_no_training'
exp_name_2 = 'calc_MAF_w_mlada'


def convert_to_percentage(vector, fraction):
    return [str(int(v * fraction * 100)) + '%' for v in vector]


kitti_num_samples = 7481
fraction_1 = 100 / int(1.0 * kitti_num_samples)
fraction_2 = 500 / int(1.0 * kitti_num_samples)
fraction_3 = 1000 / int(1.0 * kitti_num_samples)

dict_list = []
labels_list = ['MLADA_wo_training', 'MLADA_w_training', 'No_MLADA']
with open(os.path.join('checkpoints', exp_name_1, 'random_subset_MAF.pkl'), 'rb') as f:
    dict_list.append(pickle.load(f))
with open(os.path.join('checkpoints', exp_name_2, 'random_subset_MAF.pkl'), 'rb') as f:
    dict_list.append(pickle.load(f))
with open(os.path.join('checkpoints', exp_name_2, 'random_subset_MAF_no_MLADA.pkl'), 'rb') as f:
    dict_list.append(pickle.load(f))
# with open(os.path.join('checkpoints', exp_name_1, 'subset_avg_p_dict.pkl'), 'rb') as f:
#     a = pickle.load(f)
MAF_dict = {}
for dict_ in dict_list:
    for k , v in dict_.items():
        if k in MAF_dict:
            MAF_dict[k]= np.concatenate([MAF_dict[k], np.array(dict_[k])[None, :]], axis=0)
        else:
            MAF_dict[k] = np.array(dict_[k])[None, :]

# with open(os.path.join('checkpoints', exp_name_1, 'random_subset_avg_p_dict.pkl'), 'rb') as f:
#     subset_avg_p_dict = pickle.load(f)

# MAF_dict = {k: np.array(v) for k ,v in MAF_dict.items()}
n_gt_bbx= MAF_dict['bbox_n_correction'] + MAF_dict['bbox_creation']
n_pred_bbx= MAF_dict['bbox_n_correction'] + MAF_dict['bbox_deletion']
MAF_dict['bbox_displacement'] = MAF_dict['bbox_correction'] / MAF_dict['bbox_n_correction']
# MAF_dict['bbox_n_correction'] = MAF_dict['bbox_n_correction'] / n_pred_bbx
MAF_dict['bbox_detected'] = MAF_dict['bbox_correct'] / n_pred_bbx
MAF_dict['class_correction'] = MAF_dict['class_correction'] / MAF_dict['bbox_n_correction']
MAF_dict['bbox_creation'] = MAF_dict['bbox_creation'] / n_gt_bbx
MAF_dict['bbox_deletion'] = MAF_dict['bbox_deletion'] / n_pred_bbx
MAF_dict.pop('bbox_n_correction');MAF_dict.pop('bbox_correction')
MAF_dict.pop('bbox_creation_h');MAF_dict.pop('bbox_creation_w');MAF_dict.pop('n_samples');MAF_dict.pop('bbox_correct')
for k , v in MAF_dict.items():
    # plt.subplot(3, 2, 1)
    fig = plt.figure()
    width = 0.5
    x = np.arange(1, v.shape[1] + 1).tolist()
    for i, v_i in enumerate(v):
        if not any(np.isnan(v_i)):
            plt.plot(x, v_i, label=labels_list[i])
    # plt.bar(x, v, width=width)
    x_ticks = x[::5]
    x_ticks.append(x[-1])
    x_ticks_labels = convert_to_percentage(x_ticks, fraction_1)
    # x_ticks_labels.append('100%')
    # y_ticks_array = np.arange(-50, 105, 10)
    plt.xticks(x_ticks, x_ticks_labels)
    # plt.yticks(y_ticks_array, [str(a) + '%' for a in y_ticks_array])
    # plt.title(k + " vs Percentage of labelling data")
    plt.xlabel('Percentage of labelling data')
    plt.ylabel(k)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.8)
    fig.savefig(os.path.join('checkpoints/results_fig', k + '.png'))
    # plt.show()

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