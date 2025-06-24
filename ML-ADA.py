import json
import os
import wget
import pickle
import random
import shutil
import sys
import argparse
import threading
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import Subset
from glob import glob
from models import Darknet
from datasets import Image2DAnnotationDataset
from utils.utils import (Visualizer, convert_target_to_detection, draw_bbox, make_ordinal,
                         non_max_suppression, weights_init_normal, set_device, unpack_pre_annotation, show_msgbox, Log)
from annotation_toolkit import AnnotationToolkit
from time import sleep
is_gui = None
if len(sys.argv) > 1 :
    if sys.argv[1] == '--gui':
        is_gui = True
        from annotation_toolkit import InstructionUI
        from PyQt5.QtWidgets import QMessageBox
        from PyQt5.QtWidgets import (QApplication, QMainWindow)
        from PyQt5.QtCore import QThread, pyqtSignal
        from ui_files.main_window_ui import Ui_MainWindow
        from config_setup import ConfigUI
        inhertance_classes = (QMainWindow, Ui_MainWindow)
    elif len(sys.argv) == 3 and sys.argv[1] == '--cfg':
        is_gui = False
        inhertance_classes = ()
if is_gui is None:
    print('Please use --gui or --cfg options.')
    exit(1)


    


class MargParse():
    def __init__(self, opt):
        self.img_size = 416
        self.background_training = False
        self.n_cpu = 4
        self.iou_thres = 0.5
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.checkpoint_dir = 'checkpoints'
        for k, v in opt.items():
            setattr(self, k, v)
        if self.annotation_tool == 'cvat_api':
            self.annotation_tool = 'cvat'
            self.use_api = True
        elif self.annotation_tool == 'cvat_manual':
            self.annotation_tool = 'cvat'
            self.use_api = False            
        elif self.annotation_tool == 'general':
            self.use_api = False   


class MLADA(*inhertance_classes):
    if is_gui:
        textBrowser_signal = pyqtSignal(str)
        progress_bar_signal = pyqtSignal(int)

    def __init__(self, opt, parent=None):
        if is_gui:
            super().__init__(parent)
            self.setupUi(self)
            self.show()
            self.subset_selection_thread = QThread()
            self.subset_selection_thread.started.connect(self.annotate_next_subset)
            self.pushButton.clicked.connect(lambda: self.start_thread(self.subset_selection_thread))
            self.textBrowser_signal.connect(self.receive_texbrowser_signal)
        else:
            print('\n\n***************************************'
                + ' Welcome to Machine Learning-Assisted Data Annotation (ML-ADA) *'
                + '*************************************************\n')
        print(opt)
        opt_dict = opt
        opt = MargParse(opt)
        # Setting random seeds to permit repeatability
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.training_thread = None
        self.training_epoch = 0
        self.training_batch = 0
        self.do_training = opt.do_training
        self.training_iteration = 0
        self.training_dataloader_len = 0
        self.loop_iteration = 0
        self.total_time = 0
        self.opt = opt
        img_size = opt.img_size
        self.is_gui = is_gui
        self.extension = 'jpeg'
        self.skip_rt = False
        cfg = f'config/{opt.model_name}.cfg'
        opt.checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.exp_name)
        resume = False
        if os.path.exists(opt.checkpoint_dir):
            while(True):
                ret = show_msgbox(self, f'The experiment name {opt.exp_name} already exists. Do you want to resume annotation?\nPress yes to resume it, no to overwrite the project:', 'yes/no', 'info', self.is_gui)
                if self.is_gui:
                    if ret == QMessageBox.Yes:
                        resume = True
                        break
                    elif ret == QMessageBox.No:
                        shutil.rmtree(opt.checkpoint_dir)
                        self.print_msg('The previous project deleted.')
                        break
                else:
                    resp = input()
                    if resp == 'yes':
                        resume = True
                        break
                    elif resp == 'no':
                        shutil.rmtree(opt.checkpoint_dir)
                        self.print_msg('The previous project deleted.')
                        break
                sleep(1)
        os.makedirs(opt.checkpoint_dir, exist_ok=True)
        with open(f"{os.path.join(opt.checkpoint_dir, 'config.json')}", 'w') as f:
            json.dump(opt_dict, f)
        self.print_msg(f"Config file saved in {os.path.join(opt.checkpoint_dir, 'config.json')}")
        weights_path = os.path.join(f'{opt.checkpoint_dir}','best.weights') if resume else os.path.join(f'checkpoints',f'{opt.model_name}.weights')
        if not os.path.exists(weights_path):
            if resume:
                self.exit_(1, f'Cannot find {weights_path}')
            self.print_msg(f'Downloading the {opt.model_name} weights ...')
            wget.download('https://pjreddie.com/media/files/yolov3.weights', weights_path)
        if not (resume and os.path.exists(opt.checkpoint_dir)):
            shutil.copy(weights_path, os.path.join(f'{opt.checkpoint_dir}','best.weights'))
        self.logger = Log(opt, resume)
        # creating class-to-label and lable-to-class maps
        model = Darknet(cfg, opt, img_size=img_size)
        classes = model.hyperparams['classes'].split(',')
        model.apply(weights_init_normal)
        model.load_weights(weights_path)
        self.model = model
        # resize_tuple = (1224, 370)
        self.resize_tuple = None
        # creating class-to-label and lable-to-class maps
        classes_dict = {c: i for i, c in enumerate(classes)}
        # The user should define its own labels and the labels-to-class map. If
        # doesn't match any class put -1 in the list.
        labels = opt.labels
        labels_to_classes = opt.labels_to_classes
        #######################################################################
        classes_to_labels = -1 * np.ones(len(classes_dict))

        for i, l in enumerate(labels_to_classes):
            if l != -1:
                labels_to_classes[i] = classes_dict[l]
                classes_to_labels[classes_dict[l]] = i

        self.classes_to_labels = classes_to_labels
        self.labels_to_classes = labels_to_classes
        self.labels = labels
        self.num_classes = len(labels)
        resize_tuple = None
        # Initiating Datasets
        unlabelled_set = Image2DAnnotationDataset(
            opt.data_dir, labels, labels_to_classes, opt.data_format, opt.img_size, resize_tuple, None, self.logger.labelled_filenames if resume else None, self, self.extension)
        filenames = list(map(lambda l: l.split(os.path.sep)[-1].split('.')[0], unlabelled_set.images_path))
        self.f2i = {f:i for i, f in enumerate(filenames)}
   

        random_ordered_indices = np.arange(len(unlabelled_set))
        random.shuffle(random_ordered_indices)
        self.unlabelled_set = unlabelled_set
        self.len_total_dataset = len(self.unlabelled_set) + len(self.logger.labelled_filenames)
        self.total_iterations = int(
            np.ceil(self.len_total_dataset / self.opt.subset_size))
    # Instantiating visualiser

        vis = Visualizer(opt.checkpoint_dir)
        self.vis = vis
        # Instantiating and loading the model
        self.is_cuda = torch.cuda.is_available() and opt.use_cuda
        model = set_device(model, self.is_cuda)
        self.classes_to_labels = set_device(
            torch.from_numpy(self.classes_to_labels), self.is_cuda)

        self.val_best_mAP = 0.0
        self.subset_best_mAP = 0.0

        self.val_avg_p_dict = defaultdict(list)
        self.subset_avg_p_dict = defaultdict(list)
        self.subset_MAF = defaultdict(list)
        self.subset_MAF_no_MLADA = defaultdict(list)

        self.subset_size = opt.subset_size
        self.total_steps = 0
        if is_gui:
            self.print_msg("Press 'Next Subset' to start the annotation process for the next subset.")




    def start_thread(self, thread):
        if thread.isFinished:
            thread.start()


    def subset_selection(self):
        self.model.eval()
        len_unlabelled_set = len(self.unlabelled_set)
        self.loop_iteration = int(
            np.ceil((self.len_total_dataset - len_unlabelled_set) / self.opt.subset_size))
        if self.loop_iteration == self.total_iterations:
            self.print_msg(f"All the dataset is annotated!! The annotation files can be found in {os.path.join(self.opt.checkpoint_dir, 'total', 'label_2')}")
            return None
        self.print_msg(
            f'Annotating the {make_ordinal(self.loop_iteration + 1)} subset of data from total {self.total_iterations} subsets ...')
        selection_size = min(len_unlabelled_set, self.opt.subset_size)
        if self.loop_iteration == 0 and len(self.logger.selected_filenames) > 0:
            subset_indices = [self.f2i[f] for f in self.logger.selected_filenames]
        elif self.opt.query_mode == 'random':
            subset_indices = np.random.choice(
                np.arange(len_unlabelled_set), selection_size, replace=False)

        elif self.opt.query_mode == 'conf':
            sample_scores = np.ones(len_unlabelled_set, dtype=np.float32)
            dataloader = torch.utils.data.DataLoader(
                self.unlabelled_set, batch_size=self.opt.batch_size, shuffle=False, num_workers=self.opt.n_cpu)
            output_list = []
            if self.is_gui:
                self.progressBar.show()
                self.progressBar.setValue(0.0)
            with torch.no_grad():
                for batch_i, (_, imgs, _) in enumerate(dataloader):
                    if self.is_gui:
                        self.progressBar.setValue(((batch_i + 1)/len(dataloader))*100)
                    imgs = set_device(imgs)
                    try:
                        outputs = self.model(imgs)
                    except BaseException:
                        return None
                    outputs = non_max_suppression(
                        outputs, 80, conf_thres=self.opt.conf_thres, nms_thres=self.opt.nms_thres)
                    output_list.extend(outputs)
            for i, output in enumerate(output_list):
                if output is not None:
                    box_scores = output[:, 4].cpu().numpy()
                    sample_scores[i] = box_scores.mean()
            subset_indices = np.argsort(sample_scores)[:selection_size]
        order_indx = np.argsort([self.unlabelled_set[i][0] for i in subset_indices])
        subset_indices = subset_indices[order_indx]
        unlabelled_sub_dataset = Subset(self.unlabelled_set, subset_indices)
        remaining_indices = set(
            np.arange(len_unlabelled_set)).difference(subset_indices)
        self.unlabelled_set = Subset(
            self.unlabelled_set, list(remaining_indices))
        return unlabelled_sub_dataset

    def visualise_annotation(self, show_target=False):
        img_path, img, target = self.unlabelled_set[0]
        self.model.eval()
        img = set_device(img, self.is_cuda).unsqueeze(axis=0)
        with torch.no_grad():
            output = self.model(img)
            output = non_max_suppression(
                output, 80, self.classes_to_labels, conf_thres=self.opt.conf_thres, nms_thres=self.opt.nms_thres)
        if show_target:
            output_ = convert_target_to_detection(
                target, self.opt.img_size)
            if output_ is not None:
                draw_bbox(img_path, output_, self.labels,
                            self.opt.img_size, self.resize_tuple)
        if output[0] is not None:
            draw_bbox(img_path, output[0], self.labels,
                        self.opt.img_size, self.resize_tuple)

    def pre_annotate(self, unlabelled_sub_dataset):
        data_loader = torch.utils.data.DataLoader(
            unlabelled_sub_dataset, batch_size=self.opt.batch_size, shuffle=False, num_workers=self.opt.n_cpu)
        self.model.eval()
        all_outputs = []
        if self.is_gui:
            self.progressBar.show()
        s = time.time()
        for batch_i, (imgs_path, imgs, _) in enumerate(data_loader):
            if self.is_gui:
                self.progressBar.setValue(((batch_i + 1)/len(data_loader))*100)
            imgs = set_device(imgs, self.is_cuda)

            with torch.no_grad():
                outputs = self.model(imgs)
                outputs = non_max_suppression(outputs, 80, self.classes_to_labels, conf_thres=self.opt.conf_thres, nms_thres=self.opt.nms_thres)
            all_outputs.extend(outputs)
        d = time.time() - s
        self.print_msg(f"Total time for pre-annotation is {d}")
        self.vis.plot('pre_annotate_time', d, self.loop_iteration)
        self.total_time_iter += d
        return all_outputs

    def evaluate(self, pre_annotation_list):
        self.print_msg("\nEvaluation .....\n")
        tag = 'subset'
        if tag != 'subset':
            pre_annotation_list = None
        labelled_sub_dataset = Image2DAnnotationDataset(os.path.join(self.opt.checkpoint_dir, 'temp' if tag == 'subset' else 'total'), self.labels,
                                                        self.labels_to_classes, self.opt.data_format, self.opt.img_size, self.resize_tuple, img_root_dir=self.opt.data_dir, extension=self.extension)
        data_loader = torch.utils.data.DataLoader(
            labelled_sub_dataset, batch_size=self.opt.batch_size, shuffle=False, num_workers=self.opt.n_cpu)
        s = time.time()
        mAP, average_precisions = self.model.evaluate(data_loader, self.classes_to_labels, pre_annotation_list, self.progressBar if self.is_gui else None)
        d = time.time() - s
        self.total_time_iter += d
        MAF = self.model.evaluate_MAF(data_loader, self.classes_to_labels,pre_annotation_list, self.progressBar if self.is_gui else None)
        MAF_no_MLADA = self.model.evaluate_MAF(data_loader, self.classes_to_labels,[None] * len(pre_annotation_list),  None)

        if ( mAP > self.val_best_mAP):
            self.val_best_mAP = mAP
            self.model.save_weights("%s/kitti_best.weights" %
                                    (self.opt.checkpoint_dir))
            # self.print_msg(f"New Best mAP appear !!! {round(self.val_best_mAP, 2)}")
        print_str = f"Iteration: {self.loop_iteration}"
        for k, v in MAF.items():
            print_str = f"{print_str}, {k}:{round(v, 2)}"
            self.vis.plot(k, v, self.loop_iteration)
            self.subset_MAF[k].append(v)
        for k, v in MAF_no_MLADA.items():
            self.vis.plot('No_MLADA_'+k, v, self.loop_iteration)
            self.subset_MAF_no_MLADA[k].append(v)
        self.print_msg(print_str)
        avg_precision = self.subset_avg_p_dict
        for k, v in average_precisions.items():
            avg_precision[tag + '_mAP_' + self.labels[k]].append(v)
        avg_precision[tag + '_mAP'].append(mAP)
        print_str = f"Iteration: {self.loop_iteration}"
        for k, v in avg_precision.items():
            print_str = f"{print_str}, {k}:{round(v[-1], 2)}"
            self.vis.plot(k, v[-1], self.loop_iteration)
        self.print_msg(print_str)
        self.print_msg(f"Total time for evaluation is {d}")
        self.vis.plot('evaluation_time', d, self.loop_iteration)

        with open(os.path.join(self.opt.checkpoint_dir, self.opt.query_mode + '_' + tag + '_avg_p_dict.pkl'), 'wb') as f:
            pickle.dump(avg_precision, f)
        with open(os.path.join(self.opt.checkpoint_dir, self.opt.query_mode + '_' + tag + '_MAF.pkl'), 'wb') as f:
            pickle.dump(self.subset_MAF, f)
        with open(os.path.join(self.opt.checkpoint_dir, self.opt.query_mode + '_' + tag + '_MAF_no_MLADA.pkl'), 'wb') as f:
            pickle.dump(self.subset_MAF_no_MLADA, f)
        self.subset_avg_p_dict = avg_precision
        ###### check performance ##############################################
        self.check_performance()
        return

    def fine_tune_thread(self, is_training=True):

        self.print_msg("\nTraining .....\n")
        self.training_epoch = 0
        self.training_batch = 0
        total_data_dir = os.path.join(self.opt.checkpoint_dir, 'total')
        train_set = Image2DAnnotationDataset(total_data_dir, self.labels, self.labels_to_classes,
                                             self.opt.data_format, self.opt.img_size, self.resize_tuple, img_root_dir=self.opt.data_dir, extension=self.extension)
        if len(train_set) == 0:
            self.exit_(1, f'No data found in {total_data_dir}')
        
        dataloader = torch.utils.data.DataLoader(
            train_set, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.n_cpu)
        self.training_dataloader_len = len(dataloader)
        self.model.train(True)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        freeze_backbone = True
        accumulated_batches = 4
        losses_list = defaultdict(list)
        if self.is_gui:
            self.progressBar.show()
        s = time.time()
        for epoch in range(self.opt.epochs if is_training else 1):
            if self.is_gui:
                self.progressBar.setValue(((epoch + 1)/self.opt.epochs)*100)    
            # train
            # Freeze darknet53.conv.74 layers for first some epochs
            self.model.freeze_parameters(self.training_iteration / self.total_iterations, freeze_backbone, is_training)
            if is_training:
                optimizer.zero_grad()
            for batch_i, (_, imgs, targets) in enumerate(dataloader):
                self.training_batch = batch_i
                imgs = set_device(imgs, self.is_cuda)
                targets = set_device(targets, self.is_cuda)
                if self.labels_to_classes is not None:
                    no_annotation_ind = targets.sum(-1) != 0.0
                    tensor_labels_to_class = set_device(torch.tensor(
                        self.labels_to_classes), self.is_cuda)
                    x, y = torch.nonzero(no_annotation_ind, as_tuple=True)
                    targets[x, y, 0] = tensor_labels_to_class[targets[x, y, 0].long()].double()

                if is_training:
                    loss = self.model(imgs, targets)
                    loss.backward()
                    if ((batch_i + 1) % accumulated_batches ==0) or (batch_i == len(dataloader) - 1):
                        optimizer.step()
                        optimizer.zero_grad()
                    self.total_steps += 1

                    for tag, value in self.model.losses.items():
                        self.vis.plot('losses_' + tag, value, self.total_steps)
                    self.vis.plot('total_loss', loss.item(),
                                    self.total_steps)
                    self.vis.plot('epoch', epoch, self.total_steps)
                    if not self.opt.background_training:
                        msg = "[Epoch %d/%d, Batch %d/%d] [Losses: " % (
                                epoch,
                                self.opt.epochs,
                                batch_i,
                                len(dataloader)
                            )
                        for k, v in self.model.losses.items():
                            msg += '%s: %f, ' % (k , v)
                        msg += 'total: %f]' % (loss.item())
                        self.print_msg(msg)
                else:
                    with torch.no_grad():
                        loss = self.model(imgs, targets)
                        for k, v in self.model.losses.items():
                            losses_list[k].append(v)
                        losses_list['total_loss'].append(loss.item())

            losses_list = None if is_training else {
                k: np.array(v).mean() for k, v in losses_list.items()}
        d = time.time() - s
        self.print_msg(f"Duration of training is {d}")
        self.vis.plot('training_time', d, self.loop_iteration)
        self.total_time_iter += d
        self.training_iteration += 1
        self.model.save_weights("%s/best.weights" % (self.opt.checkpoint_dir))

        return losses_list


    def check_performance(self):
        if self.val_best_mAP > self.opt.performance_thres:
            self.do_training = False

    def export_annotation(self, dataset, pre_annotation_list):
        pre_label_dir = os.path.join(self.opt.checkpoint_dir, 'temp', 'pre_label_2')
        total_pre_label_dir = os.path.join(self.opt.checkpoint_dir, 'total', 'pre_label_2')
        os.makedirs(pre_label_dir, exist_ok=True)
        os.makedirs(total_pre_label_dir, exist_ok=True)
        frames = []
        for (ind, ((img_path, _, _), pre_annotation)) in enumerate(zip(dataset, pre_annotation_list)):

            img = np.array(Image.open(img_path))
            h, w, _ = img.shape
            pad_x = max(h - w, 0) * (self.opt.img_size / max(img.shape))
            pad_y = max(w - h, 0) * (self.opt.img_size / max(img.shape))
            # Image height and width after padding is removed
            unpad_h = self.opt.img_size - pad_y
            unpad_w = self.opt.img_size - pad_x
            sep = '\\' if os.name =='nt' else '/'
            img_filename = img_path.split(sep)[-1]
            label_filename = img_filename.replace('jpeg', 'txt')
            with open(os.path.join(pre_label_dir, label_filename), 'w') as f:
                if pre_annotation is not None:
                    pre_annotation = pre_annotation.cpu().numpy()
                    for pre_ann in pre_annotation:
                        line = ['-1.0'] * 15; line[1], line[2] = '0.0', '3'
                        c, x1, y1, x2, y2 = unpack_pre_annotation(pre_ann, pad_x, pad_y, unpad_h, unpad_w, h, w, self.labels)
                        x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
                        line[0], line[4], line[5], line[6], line[7] = c, str(x1), str(y1), str(x2), str(y2)
                        f.write(' '.join(line) + '\n')
                    
                # else:
                line = ['-1.0'] * 15; line[0], line[1], line[2] = 'DontCare', '0.0', '3' 
                f.write(' '.join(line) + '\n')


        for filename in os.listdir(pre_label_dir):
            src_file = os.path.join(pre_label_dir, filename)
            dst_file = os.path.join(total_pre_label_dir, filename)
            shutil.copy(src_file, dst_file)



    def refine_pre_labels(self, dataset):
        tmp_image_dir = os.path.join(self.opt.checkpoint_dir, 'temp', 'image_2')
        gt_label_dir = os.path.join(self.opt.checkpoint_dir, 'temp', 'label_2')
        gt_label_dir_t = os.path.join(self.opt.checkpoint_dir, 'total', 'label_2')
        os.makedirs(gt_label_dir, exist_ok=True)
        os.makedirs(gt_label_dir_t, exist_ok=True)
        os.makedirs(tmp_image_dir, exist_ok=True)
        if self.opt.data_format == 'kitti':
            for img_path, _, _ in dataset:
                shutil.copy(img_path, tmp_image_dir)
                label_path = img_path.replace('image', 'label').replace(self.extension, 'txt')
                shutil.copy(label_path, gt_label_dir)


            ############### use tookit to manually refine the subset annotations #######################################################
            filenames = list(map(lambda l: l.split(os.path.sep)[-1].split('.')[0], [d[0] for d in dataset]))
            self.logger.update_selected(filenames)
            ############### TODO: remove this when on the last release ##################################################################
            for img_path, _, _ in dataset:
                label_path = img_path.replace('image', 'label').replace(self.extension, 'txt')
                shutil.copy(label_path, gt_label_dir_t)
            # os.makedirs(os.path.join(self.opt.checkpoint_dir,'Refined_Subset'), exist_ok=True)
            # shutil.copytree(os.path.join(self.opt.checkpoint_dir, 'total', 'label_2'), os.path.join(self.opt.checkpoint_dir,'Refined_Subset', 'label_2'))
            # shutil.make_archive(os.path.join(self.opt.checkpoint_dir, 'Refined_Subset'), 'zip', os.path.join(self.opt.checkpoint_dir,'Refined_Subset'))
            # shutil.rmtree(os.path.join(self.opt.checkpoint_dir, 'Refined_Subset'))

            ########################################################################################################################################

    def print_manual(self):
        print_str = '\nPlease type the following commands:\n\n'
        print_str += "'ns' for annotating Next Subset.\n"
        print_str += "'vs' for Visualise Annotation.\n"
        print_str += "'rt' flip skip re-training.\n"
        print_str += "'db' for Disabling the Background re-training.\n"
        print_str += "'h' for Help.\n"
        print_str += "'q' for Quitting the program.\n"
        print(print_str)
    

    def annotate_next_subset(self):
        self.total_time_iter = 0
        unlabelled_sub_dataset = self.subset_selection()
        # Pre-Annotate ###############################
        pre_annotation_list = self.pre_annotate(unlabelled_sub_dataset)
        self.export_annotation(unlabelled_sub_dataset,pre_annotation_list)
        self.refine_pre_labels(unlabelled_sub_dataset)
        self.evaluate(pre_annotation_list)
        if self.do_training:
            self.fine_tune_thread()
        shutil.rmtree(os.path.join(self.opt.checkpoint_dir, 'temp'))
        self.print_msg('\nTraining is finished.\n')
        ## log the filnames labelled
        self.logger.update_labelled()
        self.print_msg(f"total time for iteration {self.loop_iteration} is {self.total_time_iter}")
        self.vis.plot('total_time_iter', self.total_time_iter, self.loop_iteration)
        self.total_time += self.total_time_iter
        self.print_msg(f"total time for all iterations till now is {self.total_time}")
        self.vis.plot('total_time', self.total_time, self.loop_iteration)
        return
    
    def run_loop(self):
        for _ in range(self.total_iterations):
                self.annotate_next_subset()
                # self.visualise_annotation(show_target=True)
                

    def receive_texbrowser_signal(self, msg):
        self.textBrowser.append(msg)
        self.textBrowser.repaint()

    def print_msg(self, msg, is_error=False):
        if self.is_gui:
            if is_error:
                show_msgbox(self, msg, 'OK', 'error', True)
            else:
                self.textBrowser_signal.emit(msg)
        else:
            print(msg)

    def exit_(self, code, msg=None):
        if hasattr(self, 'annotation_toolkit'):
            if self.annotation_toolkit.driver is not None:
                self.annotation_toolkit.driver.close()
        if code == 1:
            if msg is not None:
                self.print_msg(msg, is_error=True)
            exit(1)
        else:
            exit(0)

    def closeEvent(self,event):
        self.exit_(0)

def start_MLADA(config_window):
    mainwindow = MLADA(opt=config_window.get_all_variables())

if __name__ == '__main__':
    if is_gui :
        app = QApplication(sys.argv)
        config_window = ConfigUI()
        config_window._closed.connect(lambda: start_MLADA(config_window))
        config_window.show()
        sys.exit(app.exec())
    else:
        cfg_file_path = sys.argv[2]
        if os.path.exists(cfg_file_path):
            with open(cfg_file_path) as outfile:
                opt = json.load(outfile)
            conf_var_names = set(['exp_name', 'model_name', 'data_format', 'labels', 'labels_to_classes', 'use_cuda', 'data_dir', 'annotation_tool'])
            missed_keys = list(set(conf_var_names).difference(opt.keys()))
            if len(missed_keys) > 0:
                print(f"The selected config file does not contain the following keys: {', '.join(missed_keys)}")
                exit(1)
            if opt['data_format'] == 'kitti' and not os.path.exists(os.path.join(opt['data_dir'],'image_2')) or not os.path.exists(os.path.join(opt['data_dir'],'label_2')):
                print(f'The path {opt.data_dir} does not contain image_2 or label_2 folders. Please check the structure of kitti dataset.')
                exit(1)
            mainwindow = MLADA(opt=opt)
            mainwindow.run_loop()
        else:
            print(f'Cannot find config file {cfg_file_path}!')
            exit(1)
