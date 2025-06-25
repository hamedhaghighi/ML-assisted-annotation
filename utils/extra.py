####### dataset & evaluation settings ####################################
opt["model_name"] = "yolov3-coco"  # path to current model config file
opt["data_dir"] = "./data/kitti_tiny/training"  # path to data config file
# data format for loading data choose from ['kitti', 'coco', 'nuimage']
opt["data_format"] = "kitti"
opt["iou_thres"] = 0.5  # iou threshold required to qualify as detected
opt["conf_thres"] = 0.8  # object confidence threshold
opt["nms_thres"] = 0.4  # iou thresshold for non-maximum suppression
####### training settings ########################################
opt["epochs"] = 3  # Number of epochs
opt["batch_size"] = 8  # size of each image batch
opt["subset_size"] = 25  # size of the subset
# performance threshold for continuing the model refinement
opt["performance_thres"] = 0.5
opt["img_size"] = 416  # size of each image dimension
# directory where model checkpoints are saved
opt["checkpoint_dir"] = "checkpoints"
opt["query_mode"] = "random"  # mode of active learning subset selection
opt["exp_name"] = "random"  # name of this experiment
# general settings
opt["use_cuda"] = True  # use cuda for processings
opt["visualise_detection"] = False  # visualise detection for debugging
opt["n_cpu"] = 4  # number of cpu threads to use during batch generation


# _, losses_list = calc_loss(model, val_loader, opt.epochs, optimizer, vis, total_steps, opt.checkpoint_dir, i, labels_to_classes, is_training=False)
# for k , v in losses_list.items():
#     avg_losses_list[k].append(v)
# for tag, loss in avg_losses_list.items():
#     vis.plot('losses_' + tag, loss[-1], i)
