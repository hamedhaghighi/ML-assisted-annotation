import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


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
            mAP_list[i] += (len(mAP_list) - i) * (mAP_list_t[i] - mAP_list_t[i - 1])
        else:
            mAP_list[i] += len(mAP_list - i) * mAP_list_t[i]
    mAP_list = mAP_list / len(mAP_list)
    mAP_list = ((mAP_list - mAP_list[0]) / mAP_list[0]) * 100
    return mAP_list.tolist()


# with open('checkpoints/random_mapdict.pkl', 'rb') as f:
#     random_dict = pickle.load(f)
# with open('checkpoints/conf_mapdict.pkl', 'rb') as f:
#     conf_dict = pickle.load(f)


# for k, v in conf_dict.items():
#     if k == 'mAP':
#         length = min(len(conf_dict[k]), len(random_dict[k]))
#         length=40
#         plt.figure()
#         plt.plot(np.arange(length), v[:length], label='Confidence-based Selection')
#         plt.plot(np.arange(length), random_dict[k][:length], label='Random Selection')
#         # plt.title(k)
#         plt.xlabel('Active Learning Iterations')
#         plt.ylabel('mAP')
#         plt.legend()
#         plt.grid(alpha=0.8)
#         plt.show()


exp_name_1 = "exp_sub_100_e_5_freeze_b_8"
exp_name_2 = "exp_sub_500_e_5_freeze"
exp_name_3 = "exp_sub_1000_e_20_freeze_b_32"


def convert_to_percentage(vector, fraction):
    return [str(int(v * fraction * 100)) + "%" for v in vector]


kitti_num_samples = 7481
fraction_1 = 100 / int(0.8 * kitti_num_samples)
fraction_2 = 500 / int(0.8 * kitti_num_samples)
fraction_3 = 1000 / int(0.8 * kitti_num_samples)

with open(
    os.path.join("checkpoints", exp_name_1, "random_val_avg_p_dict.pkl"), "rb"
) as f:
    val_avg_p_dict_100 = pickle.load(f)
with open(
    os.path.join("checkpoints", exp_name_2, "random_val_avg_p_dict.pkl"), "rb"
) as f:
    val_avg_p_dict_500 = pickle.load(f)
with open(
    os.path.join("checkpoints", exp_name_3, "random_val_avg_p_dict.pkl"), "rb"
) as f:
    val_avg_p_dict_1000 = pickle.load(f)


def title_change(s):
    classes = ["Car", "Truck", "Pedestrian"]
    for c in classes:
        if c in s:
            return c
    return "Average"


# n = len(val_avg_p_dict_100)
# i = 1
# for (k1 , v1) , (k2, v2), (k3, v3) in zip(val_avg_p_dict_100.items(),val_avg_p_dict_500.items(),val_avg_p_dict_1000.items()):
#     plt.subplot(2, n//2, i)
#     width = 0.5
#     x1, x2, x3 = (np.arange(len(v1)) * fraction_1 * 100).tolist(),(np.arange(len(v2)) * fraction_2 * 100).tolist(),(np.arange(len(v3)) * fraction_3 * 100).tolist()
#     plt.plot(x1, v1, label='Subset size of 100', color='blue')
#     plt.plot(x2, v2, label='Subset size of 500', color='red')
#     plt.plot(x3, v3, label='Subset size of 1000', color='green')
#     # plt.bar(np.arange(l_v) + width, v2[:l_v], label='10 epochs', width=width)
#     # plt.bar(np.arange(l_v) + 2 * width, v3[:l_v], label='20 epochs', width=width)
#     x_ticks = np.arange(0, 110, 10)
#     # x_ticks.append(x[-1])
#     # x_ticks_labels = convert_to_percentage(x_ticks, fraction)
#     # x_ticks_labels.append('100%')
#     plt.xticks(x_ticks, [str(a) + '%' for a in x_ticks])
#     plt.yticks(np.arange(0, max(v1), 0.05))
#     plt.title(title_change(k1))
#     plt.xlabel('Percentage of labelling dataset')
#     plt.ylabel('mAP')
#     plt.legend(loc='lower right')
#     plt.grid(alpha=0.8)
#     i = i + 1

# plt.show()


plt.subplot(3, 2, 1)
v1 = average_mAP_improvement(val_avg_p_dict_100["val_mAP"])
l_v = len(v1)
# l_v = 31
width = 0.5
x = np.arange(l_v).tolist()
y = v1[:l_v]
plt.bar(x, y, width=width)
x_ticks = x[::5]
# x_ticks.append(x[-1])
x_ticks_labels = convert_to_percentage(x_ticks, fraction_1)
# x_ticks_labels.append('100%')
y_ticks_array = np.arange(-50, 105, 10)
plt.xticks(x_ticks, x_ticks_labels)
plt.yticks(y_ticks_array, [str(a) + "%" for a in y_ticks_array])
# plt.title('Average mAP gain')
plt.xlabel("Percentage of labelling data")
plt.ylabel("Percentage of mAP gain")
# plt.legend(loc='lower right')
plt.grid(alpha=0.8)


plt.subplot(3, 2, 3)
v1 = average_mAP_improvement(val_avg_p_dict_500["val_mAP"])
l_v = len(v1)
# l_v = 31
width = 0.5
x = np.arange(l_v).tolist()
y = v1[:l_v]
plt.bar(x, y, width=width)
x_ticks = x
# x_ticks.append(x[-1])
x_ticks_labels = convert_to_percentage(x_ticks, fraction_2)
# x_ticks_labels.append('100%')
y_ticks_array = np.arange(0, 105, 10)
plt.xticks(x_ticks, x_ticks_labels)
plt.yticks(y_ticks_array, [str(a) + "%" for a in y_ticks_array])
# plt.title('Average mAP gain')
plt.xlabel("Percentage of labelling data")
plt.ylabel("Percentage of mAP gain")
# plt.legend(loc='lower right')
plt.grid(alpha=0.8)


plt.subplot(3, 2, 5)
v1 = average_mAP_improvement(val_avg_p_dict_1000["val_mAP"])
l_v = len(v1)
# l_v = 31
width = 0.5
x = np.arange(l_v).tolist()
y = v1[:l_v]
plt.bar(x, y, width=width)
x_ticks = x
# x_ticks.append(x[-1])
x_ticks_labels = convert_to_percentage(x_ticks, fraction_3)
# x_ticks_labels.append('100%')
y_ticks_array = np.arange(0, 105, 10)
plt.xticks(x_ticks, x_ticks_labels)
plt.yticks(y_ticks_array, [str(a) + "%" for a in y_ticks_array])
# plt.title('Average mAP gain')
plt.xlabel("Percentage of labelling data")
plt.ylabel("Percentage of mAP gain")
# plt.legend(loc='lower right')
plt.grid(alpha=0.8)

plt.show()
