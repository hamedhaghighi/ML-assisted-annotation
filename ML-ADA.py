"""
ML-ADA (Machine Learning Assisted Data Annotation) - Main Application Module

This module provides the core functionality for ML-assisted annotation of 2D and 3D datasets.
It supports both GUI and command-line interfaces, with automatic model training and
annotation refinement capabilities.

Key Features:
- 2D object detection using YOLO models
- 3D object detection using PointPillars
- Integration with CVAT annotation tool
- Automatic model fine-tuning
- Batch annotation processing
- Progress tracking and logging
"""

import json
import os
import pickle
import random
import shutil
import sys
import threading
import time
from collections import defaultdict
from copy import deepcopy
from glob import glob
from time import sleep

import numpy as np
import torch
import wget
from PyQt5.QtWidgets import QMessageBox
from torch.utils.data import Subset

# Import custom modules for annotation, datasets, and models
from annotation_toolkit import AnnotationToolkit
from datasets import get_dataset
from datasets.datasets2D import collate_fn as collate_fn_2d
from datasets.datasets2D import write_annotations as write_annotations_2d
from datasets.datasets3D import collate_fn as collate_fn_3d
from datasets.datasets3D import (
    convert_kitti_xml_to_txt,
    create_kitti_xml,
    format_annotations,
)
from models.pointpillars import PointPillars
from models.yolo import Darknet
from utils.utils import (
    Log,
    Visualizer,
    check_opt,
    convert_target_to_detection,
    draw_bbox,
    make_ordinal,
    non_max_suppression,
    set_device,
    show_msgbox,
    weights_init_normal,
)

# Determine if running in GUI or CLI mode based on command line arguments
is_gui = None
if len(sys.argv) > 1:
    if sys.argv[1] == "--gui":
        is_gui = True
        # Import GUI-specific modules only when needed
        from PyQt5.QtCore import QThread, pyqtSignal
        from PyQt5.QtWidgets import QApplication, QMainWindow

        from config_setup import ConfigUI
        from ui_files.main_window_ui import Ui_MainWindow

        inhertance_classes = (QMainWindow, Ui_MainWindow)
    elif len(sys.argv) == 3 and sys.argv[1] == "--cfg":
        is_gui = False
        inhertance_classes = ()
if is_gui is None:
    print("Please use --gui or --cfg options.")
    exit(1)


class MargParse:
    """
    Configuration parser that sets default values and processes annotation tool options.

    This class handles the conversion of configuration dictionaries to object attributes
    with sensible defaults for ML-ADA parameters.
    """

    def __init__(self, opt):
        # Set default values for common parameters
        self.img_size = 416
        self.background_training = False
        self.n_cpu = 4
        self.iou_thres = 0.5
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.checkpoint_dir = "checkpoints"

        # Override defaults with provided options
        for k, v in opt.items():
            setattr(self, k, v)

        # Handle annotation tool configuration
        if self.annotation_tool == "cvat_api":
            self.annotation_tool = "cvat"
            self.use_api = True
        elif self.annotation_tool == "cvat_manual":
            self.annotation_tool = "cvat"
            self.use_api = False
        elif self.annotation_tool == "general":
            self.use_api = False


class MLADA(*inhertance_classes):
    """
    Main ML-ADA application class that orchestrates the annotation workflow.

    This class handles both GUI and CLI modes, manages model training, dataset processing,
    and coordinates the annotation pipeline from pre-annotation to fine-tuning.
    """

    if is_gui:
        # GUI-specific signals for progress updates and text display
        textBrowser_signal = pyqtSignal(str)
        progress_bar_signal = pyqtSignal(int)

    def init_func(self, opt):
        """
        Initialize the ML-ADA application with configuration options.

        Args:
            opt: Configuration dictionary containing experiment parameters
        """
        if is_gui:
            # Set up GUI threading for subset selection
            self.subset_selection_thread = QThread()
            self.subset_selection_thread.started.connect(self.annotate_next_subset)
            self.pushButton.clicked.connect(
                lambda: self.start_thread(self.subset_selection_thread)
            )
            self.textBrowser_signal.connect(self.receive_texbrowser_signal)

        # Validate configuration options
        if not check_opt(opt, self.print_msg):
            self.exit_(1)

        opt_dict = opt
        opt = MargParse(opt)

        if not isinstance(opt.exp_name, str):
            exit(1)

        # Set random seeds for reproducibility
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize training state variables
        self.training_thread = None
        self.training_model = None
        self.training_epoch = 0
        self.training_batch = 0
        self.do_training = True
        self.training_iteration = 0
        self.training_dataloader_len = 0
        self.loop_iteration = 0

        # Store configuration and determine model type
        self.opt = opt
        img_size = opt.img_size
        self.is_gui = is_gui
        self.is_two_d = (
            "yolo" in opt.model_name
        )  # True for 2D YOLO, False for 3D PointPillars
        self.velo_folder_name = "velodyne_reduced"
        self.image_shape = None
        self.skip_rt = False

        # Set up model configuration and checkpoint directory
        cfg = f"config/{opt.model_name}.cfg"
        opt.checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.exp_name)

        # Initialize annotation toolkit
        self.annotation_toolkit = AnnotationToolkit(
            self.opt, is_gui, self.is_two_d, self.textBrowser_signal if is_gui else None
        )
        self.calib_info = None

        # Handle experiment resumption
        resume = False
        if os.path.exists(opt.checkpoint_dir):
            while True:
                ret = show_msgbox(
                    self,
                    f"The experiment name '{opt.exp_name}' already exists. Do you want to resume annotation?\nPress yes to resume it, no to overwrite the project:",
                    "yes/no",
                    "info",
                    self.is_gui,
                )
                if self.is_gui:
                    if ret == QMessageBox.Yes:
                        resume = True
                        break
                    elif ret == QMessageBox.No:
                        shutil.rmtree(opt.checkpoint_dir)
                        self.annotation_toolkit.delete_task()
                        self.print_msg("The previous project deleted.")
                        break
                else:
                    resp = input()
                    if resp == "yes":
                        resume = True
                        break
                    elif resp == "no":
                        shutil.rmtree(opt.checkpoint_dir)
                        self.annotation_toolkit.delete_task()
                        self.print_msg("The previous project deleted.")
                        break
                sleep(1)

        # Create checkpoint directory and save configuration
        os.makedirs(opt.checkpoint_dir, exist_ok=True)
        with open(f"{os.path.join(opt.checkpoint_dir, 'config.json')}", "w") as f:
            json.dump(opt_dict, f)
        self.print_msg(
            f"Config file saved in {os.path.join(opt.checkpoint_dir, 'config.json')}"
        )

        # Handle model weights download and loading
        weights_path = (
            os.path.join(f"{opt.checkpoint_dir}", "best.weights")
            if resume
            else os.path.join(f"checkpoints", f"{opt.model_name}.weights")
        )
        if not os.path.exists(weights_path):
            if resume:
                self.exit_(1, f"Cannot find {weights_path}")
            self.print_msg(f"Downloading the {opt.model_name} weights ...")
            weights_url = (
                "https://pjreddie.com/media/files/yolov3.weights"
                if self.is_two_d
                else "https://livewarwickac-my.sharepoint.com/:u:/g/personal/u2039803_live_warwick_ac_uk/EeyHT4eBIQJAvsAwtxIrfN0BSak10rUUNgm49SzrhlKv0A?e=iTbyGM"
            )
            wget.download(weights_url, weights_path)
        if not (resume and os.path.exists(opt.checkpoint_dir)):
            shutil.copy(
                weights_path, os.path.join(f"{opt.checkpoint_dir}", "best.weights")
            )

        # Initialize logging system
        self.logger = Log(opt, resume)

        # Load and configure the appropriate model (2D YOLO or 3D PointPillars)
        if self.is_two_d:
            model = Darknet(cfg, opt, img_size)
            model.apply(weights_init_normal)
            model.load_weights(weights_path)
            classes = model.hyperparams["classes"].split(",")
        else:
            model = PointPillars(opt)
            model.load_state_dict(torch.load(weights_path))
            classes = ["Pedestrian", "Cyclist", "Car"]

        self.nclasses = len(classes)
        self.model = model
        self.resize_tuple = None

        # Create class-to-label and label-to-class mapping dictionaries
        classes_dict = {c: i for i, c in enumerate(classes)}
        # The user should define its own labels and the labels-to-class map. If
        # doesn't match any class put -1 in the list.
        labels = opt.labels
        labels_to_classes = opt.labels_to_classes

        # Initialize mapping arrays with -1 (no mapping)
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

        # Initialize dataset with unlabelled data
        unlabelled_set = get_dataset(
            opt.data_dir,
            labels,
            labels_to_classes,
            opt.data_format,
            opt.img_size,
            resize_tuple,
            None,
            self.logger.labelled_filenames if resume else None,
            self,
            is_two_d=self.is_two_d,
        )
        self.annotation_toolkit.create_task()

        # Create filename to index mapping for efficient lookup
        filenames = list(
            map(
                lambda l: l.split(os.path.sep)[-1].split(".")[0],
                unlabelled_set.images_path,
            )
        )
        self.f2i = {f: i for i, f in enumerate(filenames)}

        # Shuffle dataset indices for random sampling
        random_ordered_indices = np.arange(len(unlabelled_set))
        random.shuffle(random_ordered_indices)
        self.unlabelled_set = unlabelled_set
        self.len_total_dataset = len(self.unlabelled_set) + len(
            self.logger.labelled_filenames
        )
        self.total_iterations = int(
            np.ceil(self.len_total_dataset / self.opt.subset_size)
        )
        # Initialize visualizer for plotting and logging
        vis = Visualizer(opt.checkpoint_dir)
        self.vis = vis

        # Set up device (CPU/GPU) and move model to appropriate device
        self.is_cuda = torch.cuda.is_available() and opt.use_cuda
        model = set_device(model, self.is_cuda)
        self.classes_to_labels = set_device(
            torch.from_numpy(self.classes_to_labels), self.is_cuda
        )

        # Initialize performance tracking variables
        self.val_best_mAP = 0.0
        self.subset_best_mAP = 0.0

        self.val_avg_p_dict = defaultdict(list)
        self.subset_avg_p_dict = defaultdict(list)

        self.subset_size = opt.subset_size
        self.total_steps = 0
        if is_gui:
            self.print_msg(
                "Press 'Next Subset' to start the annotation process for the next subset."
            )

    def __init__(self, opt, parent=None):
        """
        Initialize ML-ADA application with GUI or CLI mode.

        Args:
            opt: Configuration dictionary
            parent: Parent widget for GUI mode
        """
        if is_gui:
            # Initialize GUI components and start initialization in separate thread
            super().__init__(parent)
            self.setupUi(self)
            self.show()
            self.init_thread = QThread()
            self.init_thread.started.connect(lambda: self.init_func(opt))
            self.start_thread(self.init_thread)
        else:
            # CLI mode - print welcome message and initialize directly
            print(
                "\n\n***************************************"
                + " Welcome to Machine Learning-Assisted Data Annotation (ML-ADA) *"
                + "*************************************************\n"
            )
            self.init_func(opt)

    def start_thread(self, thread):
        """
        Start a QThread if it's not already running.

        Args:
            thread: QThread instance to start
        """
        if thread.isFinished:
            thread.start()

    def subset_selection(self):
        """
        Select the next subset of unlabelled data for annotation.

        This method implements different selection strategies:
        - Random selection
        - Confidence-based selection (lowest confidence samples first)

        Returns:
            Subset: Selected subset of unlabelled data
        """
        self.model.eval()
        len_unlabelled_set = len(self.unlabelled_set)
        self.loop_iteration = int(
            np.ceil(
                (self.len_total_dataset - len_unlabelled_set) / self.opt.subset_size
            )
        )

        # Check if all data has been annotated
        if self.loop_iteration == self.total_iterations:
            self.print_msg(
                f"All the dataset is annotated!! The annotation files can be found in {os.path.join(self.opt.checkpoint_dir, 'total', 'label_2')}"
            )
            exit(0)
            return None

        self.print_msg(
            f"Annotating the {make_ordinal(self.loop_iteration + 1)} subset of data from total {self.total_iterations} subsets ..."
        )

        selection_size = min(len_unlabelled_set, self.opt.subset_size)

        # Handle resumption from previous session
        if self.loop_iteration == 0 and len(self.logger.selected_filenames) > 0:
            subset_indices = np.array(
                [self.f2i[f] for f in self.logger.selected_filenames]
            )
        elif self.opt.query_mode == "random":
            # Random selection strategy
            subset_indices = np.random.choice(
                np.arange(len_unlabelled_set), selection_size, replace=False
            )
        elif self.opt.query_mode == "conf":
            # Confidence-based selection - select samples with lowest confidence
            sample_scores = []
            dataloader = torch.utils.data.DataLoader(
                self.unlabelled_set,
                batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=self.opt.n_cpu,
                collate_fn=collate_fn_2d if self.is_two_d else collate_fn_3d,
            )
            if self.is_gui:
                self.progressBar.show()
                self.progressBar.setValue(0)

            # Compute confidence scores for all samples
            with torch.no_grad():
                for batch_i, data_dict in enumerate(dataloader):
                    if self.is_gui:
                        self.progressBar.setValue(
                            int(((batch_i + 1) / len(dataloader)) * 100)
                        )
                    sample_scores.extend(
                        self.model.get_bbox_scores(data_dict, self.classes_to_labels)
                    )
            # Select samples with lowest confidence scores
            subset_indices = np.argsort(sample_scores)[:selection_size]

        # Sort indices by file path for consistent ordering
        if self.is_two_d:
            order_indx = np.argsort(
                [self.unlabelled_set[i]["im_path"] for i in subset_indices]
            )
        else:
            order_indx = np.argsort(
                [self.unlabelled_set[i]["lidar_path"] for i in subset_indices]
            )
        subset_indices = subset_indices[order_indx]

        # Create subset and update unlabelled set
        unlabelled_sub_dataset = Subset(self.unlabelled_set, subset_indices)
        remaining_indices = set(np.arange(len_unlabelled_set)).difference(
            subset_indices
        )
        self.unlabelled_set = Subset(self.unlabelled_set, list(remaining_indices))
        return unlabelled_sub_dataset

    def visualise_annotation(self, show_target=False):
        """
        Visualize model predictions and ground truth annotations.

        Args:
            show_target: If True, also show ground truth annotations
        """
        data_dict = self.unlabelled_set[0]
        img_path = data_dict["im_path"]
        img = data_dict["input_img"]
        target = data_dict["filled_labels"]

        # Run model inference
        self.model.eval()
        img = set_device(img, self.is_cuda).unsqueeze(axis=0)
        with torch.no_grad():
            output = self.model(img)
            output = non_max_suppression(
                output,
                80,
                self.classes_to_labels,
                conf_thres=self.opt.conf_thres,
                nms_thres=self.opt.nms_thres,
            )

        # Draw ground truth annotations if requested
        if show_target:
            output_ = convert_target_to_detection(target, self.opt.img_size)
            if output_ is not None:
                draw_bbox(
                    img_path, output_, self.labels, self.opt.img_size, self.resize_tuple
                )

        # Draw model predictions
        if output[0] is not None:
            draw_bbox(
                img_path, output[0], self.labels, self.opt.img_size, self.resize_tuple
            )

    def pre_annotate(self, unlabelled_sub_dataset):
        """
        Generate pre-annotations for a subset of unlabelled data.

        Args:
            unlabelled_sub_dataset: Subset of unlabelled data to annotate

        Returns:
            list: List of model predictions for each sample
        """
        data_loader = torch.utils.data.DataLoader(
            unlabelled_sub_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.n_cpu,
            collate_fn=collate_fn_2d if self.is_two_d else collate_fn_3d,
        )
        self.model.eval()
        all_outputs = []

        if self.is_gui:
            self.progressBar.show()

        # Process each batch and collect predictions
        for batch_i, data_dict in enumerate(data_loader):
            if self.is_gui:
                self.progressBar.setValue(int(((batch_i + 1) / len(data_loader)) * 100))
            with torch.no_grad():
                outputs = self.model.get_output(data_dict, self.classes_to_labels)
            all_outputs.extend(outputs)

        return all_outputs

    def evaluate(self, pre_annotation_list=None, tag="subset"):
        """
        Evaluate model performance on labelled data.

        Args:
            pre_annotation_list: Optional list of pre-annotations to evaluate
            tag: Evaluation tag for logging ("subset" or "validation")
        """
        if tag != "subset":
            pre_annotation_list = None
        self.print_msg("\nEvaluation .....\n")

        # Load labelled dataset for evaluation
        labelled_sub_dataset = get_dataset(
            os.path.join(
                self.opt.checkpoint_dir, "temp" if tag == "subset" else "total"
            ),
            self.labels,
            self.labels_to_classes,
            self.opt.data_format,
            self.opt.img_size,
            self.resize_tuple,
            self.opt.data_dir,
            None,
            self,
            is_two_d=self.is_two_d,
        )
        data_loader = torch.utils.data.DataLoader(
            labelled_sub_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.n_cpu,
            collate_fn=collate_fn_2d if self.is_two_d else collate_fn_3d,
        )
        if self.is_two_d:
            mAP, average_precisions = self.model.evaluate(
                data_loader,
                self.classes_to_labels,
                pre_annotation_list,
                self.progressBar if self.is_gui else None,
            )
            overall_result = self.subset_avg_p_dict
            for k, v in average_precisions.items():
                overall_result[tag + "_mAP_" + self.labels[k]].append(v)
            overall_result[tag + "_mAP"].append(mAP)
            print_str = f"Iteration: {self.loop_iteration}"
            for k, v in overall_result.items():
                print_str = f"{print_str}\n{k}:{round(v[-1], 2)}"
                self.vis.plot(k, v[-1], self.loop_iteration)
            self.print_msg(print_str)
            with open(
                os.path.join(
                    self.opt.checkpoint_dir,
                    self.opt.query_mode + "_" + tag + "_avg_p_dict.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(overall_result, f)
            self.subset_avg_p_dict = overall_result
        else:
            overall_result = self.model.evaluate(
                pre_annotation_list,
                labelled_sub_dataset.data_infos,
                self.labels,
                self.print_msg,
            )
            mAP = overall_result["bbox_3d"].mean() / 100
        if mAP > self.val_best_mAP:
            self.val_best_mAP = mAP
            if self.is_two_d:
                self.model.save_weights(
                    os.path.join(self.opt.checkpoint_dir, "kitti_best.weights")
                )
            else:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.opt.checkpoint_dir, "best.weights"),
                )

            # self.print_msg(f"New Best mAP appear !!! {round(self.val_best_mAP, 2)}")

        ###### check performance ##############################################
        self.check_performance()
        return

    def fine_tune_thread(self):

        self.print_msg("\nTraining .....\n")
        self.training_epoch = 0
        self.training_batch = 0
        total_data_dir = os.path.join(self.opt.checkpoint_dir, "total")
        train_set = get_dataset(
            total_data_dir,
            self.labels,
            self.labels_to_classes,
            self.opt.data_format,
            self.opt.img_size,
            self.resize_tuple,
            self.opt.data_dir,
            None,
            self,
            is_two_d=self.is_two_d,
        )
        if len(train_set) == 0:
            self.exit_(1, f"No data found in {total_data_dir}")

        dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.n_cpu,
            collate_fn=collate_fn_2d if self.is_two_d else collate_fn_3d,
        )
        self.training_dataloader_len = len(dataloader)

        training_model = set_device(deepcopy(self.model), self.is_cuda)
        optimizer = (
            torch.optim.Adam(
                filter(lambda p: p.requires_grad, training_model.parameters())
            )
            if self.is_two_d
            else torch.optim.AdamW(
                training_model.parameters(),
                lr=0.00025,
                betas=(0.95, 0.99),
                weight_decay=0.01,
            )
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.00025 * 10,
            total_steps=self.training_dataloader_len * self.opt.epochs,
            pct_start=0.4,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.95 * 0.895,
            max_momentum=0.95,
            div_factor=10,
        )
        training_model.train(True)
        freeze_backbone = True
        if self.is_gui:
            self.progressBar.show()
        for epoch in range(self.opt.epochs):
            if self.is_gui:
                self.progressBar.setValue(int(((epoch + 1) / self.opt.epochs) * 100))
            if self.is_two_d:
                # Freeze darknet53.conv.74 layers for first some epochs
                training_model.freeze_parameters(
                    self.training_iteration / self.total_iterations, freeze_backbone
                )
            optimizer.zero_grad()
            for batch_i, data_dict in enumerate(dataloader):
                self.total_steps += 1
                if self.is_two_d:
                    training_model.m_train(
                        data_dict,
                        optimizer,
                        self.labels_to_classes,
                        self.vis,
                        len(dataloader),
                        epoch,
                        batch_i,
                        self.print_msg,
                        self.total_steps,
                    )
                else:
                    training_model.m_train(
                        data_dict,
                        optimizer,
                        scheduler,
                        self.labels_to_classes,
                        self.vis,
                        len(dataloader),
                        epoch,
                        batch_i,
                        self.nclasses,
                        self.print_msg,
                        self.total_steps,
                    )
        self.training_iteration += 1
        if self.is_two_d:
            training_model.save_weights(
                os.path.join(self.opt.checkpoint_dir, "best.weights")
            )
        else:
            torch.save(
                training_model.state_dict(),
                os.path.join(self.opt.checkpoint_dir, "best.weights"),
            )
        self.training_model = training_model
        return

    def fine_tune(self):
        """
        Orchestrate model fine-tuning process with optional background training.

        This method handles the training workflow, including background training
        management and model updates after training completion.
        """
        if self.do_training:
            if self.opt.background_training:
                # Handle background training thread management
                if self.training_thread is not None and self.training_thread.is_alive():
                    self.print_msg(
                        f"The training thread is still running (epoch {self.training_epoch} from {self.opt.epochs}), do you like to wait for training to be finished?"
                    )
                    while True:
                        cmd = input(
                            "\n Type 'yes' for waiting  and 'no' for continue annotating with current model:\n"
                        )
                        if cmd == "yes":
                            # Wait for training to complete
                            while self.training_thread.is_alive():
                                time.sleep(1)
                                self.print_msg(
                                    f"epoch {self.training_epoch} from {self.opt.epochs} total epochs and batch no {self.training_batch} from {self.training_dataloader_len} total batches"
                                )
                            self.training_thread.join()
                            self.model = set_device(self.training_model, self.is_cuda)
                            break
                        elif cmd == "no":
                            return
                elif (
                    self.training_thread is not None
                    and not self.training_thread.is_alive()
                ):
                    # Training completed, update model
                    self.training_thread.join()
                    self.model = set_device(self.training_model, self.is_cuda)

                # Start new training thread
                self.training_thread = threading.Thread(target=self.fine_tune_thread)
                self.training_thread.start()

            else:
                # Synchronous training
                if self.training_model is not None:
                    self.model = set_device(self.training_model, self.is_cuda)
                self.fine_tune_thread()
        else:
            self.print_msg(
                "Re-training is disabled as the desired performance has been achieved."
            )
        self.print_msg("\nTraining is finished.\n")
        return

    def check_performance(self):
        """
        Check if model performance meets the threshold to disable further training.
        """
        if self.val_best_mAP > self.opt.performance_thres:
            self.do_training = False

    def export_annotation(self, dataset, pre_annotation_list):
        """
        Export pre-annotations to the appropriate format for annotation tools.

        Args:
            dataset: Dataset containing the samples
            pre_annotation_list: List of model predictions

        Returns:
            dict: Formatted results for 3D annotations
        """
        pre_label_dir = os.path.join(self.opt.checkpoint_dir, "temp", "pre_label_2")
        os.makedirs(pre_label_dir, exist_ok=True)
        format_results = {}
        calib_dict = {}
        frames = []

        # Process each sample and its predictions
        for ind, (data_dict, pre_annotation) in enumerate(
            zip(dataset, pre_annotation_list)
        ):
            if self.is_two_d:
                # Handle 2D annotations
                ret = write_annotations_2d(
                    ind, data_dict, pre_annotation, pre_label_dir, self.labels, self.opt
                )
                if self.opt.data_format == "openlabel":
                    frames.append(ret)
            else:
                # Handle 3D annotations
                data_dict = collate_fn_3d([data_dict])
                format_result = format_annotations(
                    ind, data_dict, pre_annotation, self.labels
                )
                idx = data_dict["batched_img_info"][0]["image_idx"]
                format_results[idx] = {k: np.array(v) for k, v in format_result.items()}
                calib_dict[idx] = data_dict["batched_calib_info"][0]

        # Export in appropriate format
        if self.is_two_d and self.opt.data_format == "openlabel":
            annotation_dict = {"openlabel": {"frames": frames}}
            with open(
                os.path.join(self.opt.checkpoint_dir, "Subset_OL_annotations.json"), "w"
            ) as f:
                json.dump(annotation_dict, f)
        elif not self.is_two_d:
            create_kitti_xml(
                os.path.join(pre_label_dir, "tracklet_labels.xml"),
                format_results,
                calib_dict,
            )
            return format_results

    def refine_pre_labels(self, dataset):
        """
        Prepare pre-annotations for manual refinement in annotation tools.

        This method creates the necessary directory structure and files
        for uploading to annotation tools like CVAT.

        Args:
            dataset: Dataset containing the samples to be annotated
        """

        def make_data_dir(folder, subfolder):
            """Helper function to create directory structure."""
            dir_ = os.path.join(self.opt.checkpoint_dir, folder, subfolder)
            os.makedirs(dir_, exist_ok=True)
            return dir_

        # Set up directory structure
        data_folder_name = "image_2" if self.is_two_d else self.velo_folder_name
        tmp_data_dir = make_data_dir("temp", data_folder_name)
        make_data_dir("temp", "label_2")
        gt_label_dir_t = make_data_dir("total", "label_2")

        if self.opt.data_format == "kitti":
            # Copy data files to temporary directory
            for data_dict in dataset:
                data_path = (
                    data_dict["im_path"]
                    if self.is_two_d
                    else os.path.join(self.opt.data_dir, data_dict["lidar_path"])
                )
                shutil.copy(data_path, tmp_data_dir)

            # Create subset archive for annotation tool upload
            os.makedirs(os.path.join(self.opt.checkpoint_dir, "Subset"), exist_ok=True)
            if self.is_two_d:
                shutil.copytree(
                    os.path.join(self.opt.checkpoint_dir, "temp", "pre_label_2"),
                    os.path.join(self.opt.checkpoint_dir, "Subset", "label_2"),
                )
            else:
                shutil.copy(
                    os.path.join(
                        self.opt.checkpoint_dir,
                        "temp",
                        "pre_label_2",
                        "tracklet_labels.xml",
                    ),
                    os.path.join(self.opt.checkpoint_dir, "Subset"),
                )
            if not self.opt.use_api and self.is_two_d:
                shutil.copytree(
                    os.path.join(self.opt.checkpoint_dir, "temp", data_folder_name),
                    os.path.join(self.opt.checkpoint_dir, "Subset", data_folder_name),
                )

            # Create zip archive and clean up
            shutil.make_archive(
                os.path.join(self.opt.checkpoint_dir, "Subset"),
                "zip",
                os.path.join(self.opt.checkpoint_dir, "Subset"),
            )
            shutil.rmtree(os.path.join(self.opt.checkpoint_dir, "Subset"))

            # Upload to annotation tool and start correction process
            if self.loop_iteration > 0 or len(self.logger.selected_filenames) == 0:
                self.annotation_toolkit.upload_annotations()
            filenames = list(
                map(
                    lambda l: l.split(os.path.sep)[-1].split(".")[0],
                    [
                        d["im_path"] if self.is_two_d else d["lidar_path"]
                        for d in dataset
                    ],
                )
            )
            self.logger.update_selected(filenames)
            self.annotation_toolkit.correct_annotations()

        elif self.is_two_d and self.opt.data_format == "openlabel":
            # Handle OpenLabel format for 2D annotations
            os.makedirs(
                os.path.join(self.opt.checkpoint_dir, "Subset", "images"), exist_ok=True
            )
            for data_dict in dataset:
                img_path = data_dict["im_path"]
                shutil.copy(
                    img_path, os.path.join(self.opt.checkpoint_dir, "Subset", "images")
                )
            shutil.move(
                os.path.join(self.opt.checkpoint_dir, "Subset_OL_annotations.json"),
                os.path.join(self.opt.checkpoint_dir, "Subset"),
            )
            shutil.make_archive(
                os.path.join(self.opt.checkpoint_dir, "Subset"),
                "zip",
                os.path.join(self.opt.checkpoint_dir, "Subset"),
            )
            shutil.rmtree(os.path.join(self.opt.checkpoint_dir, "Subset"))
            filenames = list(
                map(
                    lambda l: l.split(os.path.sep)[-1].split(".")[0],
                    [d["im_path"] for d in dataset],
                )
            )
            print(filenames)
            self.logger.update_selected(filenames)
            self.annotation_toolkit.correct_annotations()

    def print_manual(self):
        """Display available commands for CLI mode."""
        print_str = "\nPlease type the following commands:\n\n"
        print_str += "'ns' for annotating Next Subset.\n"
        print_str += "'vs' for Visualise Annotation.\n"
        print_str += "'rt' skip re-training.\n"
        print_str += "'db' for Disabling the Background re-training.\n"
        print_str += "'h' for Help.\n"
        print_str += "'q' for Quitting the program.\n"
        print(print_str)

    def annotate_next_subset(self):
        """
        Execute the complete annotation workflow for the next subset.

        This is the main workflow method that:
        1. Selects the next subset of unlabelled data
        2. Generates pre-annotations
        3. Exports annotations for manual refinement
        4. Processes refined annotations
        5. Evaluates model performance
        6. Optionally fine-tunes the model
        """
        # Select next subset for annotation
        unlabelled_sub_dataset = self.subset_selection()
        if unlabelled_sub_dataset is None:
            if self.is_gui:
                self.subset_selection_thread.quit()
            return

        # Generate pre-annotations using the current model
        pre_annotation_list = self.pre_annotate(unlabelled_sub_dataset)
        format_results = self.export_annotation(
            unlabelled_sub_dataset, pre_annotation_list
        )
        self.refine_pre_labels(unlabelled_sub_dataset)

        # Process refined annotations from annotation tool
        if self.opt.data_format == "kitti":
            # Extract and process KITTI format annotations
            os.makedirs(os.path.join(self.opt.checkpoint_dir, "unzipped"))
            shutil.unpack_archive(
                os.path.join(self.opt.checkpoint_dir, "Refined_Subset.zip"),
                os.path.join(self.opt.checkpoint_dir, "unzipped"),
                "zip",
            )
            if not self.is_two_d:
                # Convert XML to TXT for 3D annotations
                os.makedirs(
                    os.path.join(self.opt.checkpoint_dir, "unzipped", "label_2"),
                    exist_ok=True,
                )
                convert_kitti_xml_to_txt(
                    glob(os.path.join(self.opt.checkpoint_dir, "unzipped", "*.xml"))[0],
                    os.path.join(self.opt.checkpoint_dir, "unzipped", "label_2"),
                    self.opt.data_dir,
                )

            # Find all label files
            if self.is_two_d:
                label_fname_list = glob(
                    os.path.join(
                        self.opt.checkpoint_dir, "unzipped", "**", "label_2", "*.txt"
                    )
                )
            else:
                label_fname_list = glob(
                    os.path.join(self.opt.checkpoint_dir, "unzipped", "**", "*.txt")
                )

            # Copy processed annotations to appropriate directories
            flag_dict = {k: False for k in self.logger.selected_filenames}
            for f in label_fname_list:
                k = f.split(os.path.sep)[-1].split(".")[0]
                if k in self.logger.selected_filenames:
                    shutil.copy(
                        f, os.path.join(self.opt.checkpoint_dir, "total", "label_2")
                    )
                    shutil.copy(
                        f, os.path.join(self.opt.checkpoint_dir, "temp", "label_2")
                    )
                    flag_dict[k] = True

            # Check for missing annotations
            missed_id_list = []
            for k, v in flag_dict.items():
                if not v:
                    missed_id_list.append('"' + k + '"')
            shutil.rmtree(os.path.join(self.opt.checkpoint_dir, "unzipped"))
            os.remove(os.path.join(self.opt.checkpoint_dir, "Refined_Subset.zip"))
            if len(missed_id_list) > 0:
                self.exit_(
                    1,
                    f"Please re-run ML-ADA with the same config and annotate the images with following ids: {', '.join(missed_id_list)}.\nClosing ...",
                )

        elif self.opt.data_format == "openlabel":
            # Process OpenLabel format annotations
            with open(
                os.path.join(
                    self.opt.checkpoint_dir, "Refined_Subset_OL_annotations.json"
                ),
                "r",
            ) as f:
                refined_ann_dict = json.load(f)
                new_frame_dict_list = refined_ann_dict["openlabel"]["frames"]

            # Merge with existing annotations if available
            if os.path.exists(
                os.path.join(self.opt.checkpoint_dir, "total", "OL_annotation.json")
            ):
                with open(
                    os.path.join(
                        self.opt.checkpoint_dir, "total", "OL_annotation.json"
                    ),
                    "r",
                ) as f:
                    total_ann_dict = json.load(f)
                total_ann_dict["openlabel"]["frames"].extend(new_frame_dict_list)
            else:
                total_ann_dict = refined_ann_dict

            # Save merged annotations
            with open(
                os.path.join(self.opt.checkpoint_dir, "total", "OL_annotation.json"),
                "w",
            ) as f:
                json.dump(total_ann_dict, f)

        # Evaluate model performance on the newly annotated subset
        if self.is_two_d:
            self.evaluate(pre_annotation_list=pre_annotation_list)
        else:
            self.evaluate(format_results)

        # Fine-tune model if enabled
        do_rt = not self.checkBox.isChecked() if self.is_gui else not self.skip_rt
        if do_rt:
            self.fine_tune()

        # Clean up and update logging
        shutil.rmtree(os.path.join(self.opt.checkpoint_dir, "temp"))
        self.logger.update_labelled()
        self.annotation_toolkit.destroy_annotation()

        if self.is_gui:
            self.subset_selection_thread.quit()
            self.print_msg(
                "Press 'Next Subset' to start the annotation process for the next subset."
            )
        return

    def run_loop(self):
        """
        Main command loop for CLI mode.

        Handles user input and executes corresponding commands.
        """
        self.print_manual()
        while True:
            cmd = input("ML-ADA>")
            if cmd == "ns":
                self.annotate_next_subset()
            elif cmd == "vs":
                # Visualize current model predictions
                self.visualise_annotation(show_target=True)
            elif cmd == "rt":
                # Toggle re-training skip flag
                self.skip_rt = not self.skip_rt
            elif cmd == "db":
                # Toggle background training
                self.opt.background_training = not self.opt.background_training
            elif cmd == "h":
                self.print_manual()
            elif cmd == "q":
                exit(0)
            else:
                print("Command not recognised!")

    def receive_texbrowser_signal(self, msg):
        """
        Handle text browser signal for GUI mode.

        Args:
            msg: Message to display in the text browser
        """
        self.textBrowser.append(msg)
        self.textBrowser.repaint()

    def print_msg(self, msg, is_error=False):
        """
        Print message to appropriate output (GUI or console).

        Args:
            msg: Message to print
            is_error: Whether this is an error message
        """
        if is_gui:
            if is_error:
                show_msgbox(self, msg, "OK", "error", True)
            else:
                self.textBrowser_signal.emit(msg + "\n")
        else:
            print(msg + "\n")

    def exit_(self, code, msg=None):
        """
        Clean exit with proper cleanup.

        Args:
            code: Exit code (0 for success, 1 for error)
            msg: Optional error message
        """
        if hasattr(self, "annotation_toolkit"):
            if self.annotation_toolkit.driver is not None:
                self.annotation_toolkit.driver.close()
        if code == 1:
            if msg is not None:
                self.print_msg(msg, is_error=True)
            exit(1)
        else:
            exit(0)

    def closeEvent(self, event):
        """Handle GUI window close event."""
        self.exit_(0)


def start_MLADA(config_window):
    """
    Start ML-ADA application with configuration from GUI.

    Args:
        config_window: Configuration window containing experiment parameters
    """
    mainwindow = MLADA(opt=config_window.get_all_variables())


if __name__ == "__main__":
    """
    Main entry point for ML-ADA application.

    Supports both GUI and CLI modes:
    - GUI mode: --gui flag launches configuration window
    - CLI mode: --cfg <config_file> loads configuration from JSON file
    """
    if is_gui:
        # GUI mode - launch configuration window
        app = QApplication(sys.argv)
        config_window = ConfigUI()
        config_window.accepted.connect(lambda: start_MLADA(config_window))
        config_window.exec()
        sys.exit(app.exec())
    else:
        # CLI mode - load configuration from file
        cfg_file_path = sys.argv[2]
        if os.path.exists(cfg_file_path):
            with open(cfg_file_path) as outfile:
                opt = json.load(outfile)
            mainwindow = MLADA(opt=opt)
            mainwindow.run_loop()
        else:
            print(f"Cannot find config file {cfg_file_path}!")
            exit(1)
