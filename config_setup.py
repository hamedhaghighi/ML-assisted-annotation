import os
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox, QFileDialog
)
from utils.parse_config import parse_model_config
from utils.utils import show_msgbox

from ui_files.confi_ui import Ui_Dialog
from ui_files.advanced_conf_ui import Ui_Dialog as Advanced_Ui_Dialog
import numpy as np
import shutil
import json

class Advanced_ConfigUI(QDialog, Advanced_Ui_Dialog):
    def __init__(self, query_mode_list, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        

        


    def set_var(self, name, val):
        self.var_dict[name] = val

        



class ConfigUI(QDialog, Ui_Dialog):
    _closed = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.model_name_list = ['', 'yolov3-coco']
        self.data_format_list = ['', 'kitti', 'openlabel']
        self.query_mode_list = ['random', 'confidence_based']
        self.advanced_conf_var_dict = {'epochs':5, 'batch_size':32, 'subset_size':25, 'performance_thres':0.5, 'query_mode':'random'}
        self.conf_var_names = ['exp_name', 'model_name', 'data_format', 'labels', 'labels_to_classes', 'use_cuda', 'data_dir', 'annotation_tool']
        self.var_dict = {name: {'val':0 , 'flag':False} for name in self.conf_var_names}
        self.var_dict['use_cuda']['flag'] = True
        for k , v in self.advanced_conf_var_dict.items():
            self.var_dict[k] = {'val':v, 'flag': True}
        self.advanced_conf_ui = Advanced_ConfigUI(self.query_mode_list)
        self.exp_name.editingFinished.connect(self.process_exp_name)
        for m in self.model_name_list:
            self.model_name.addItem(m)
        for df in self.data_format_list:
            self.data_format.addItem(df)
        for qm in self.query_mode_list:
            self.query_mode.addItem(qm)

        self.epochs.setValue(self.advanced_conf_var_dict['epochs'])
        self.batch_size.setValue(self.advanced_conf_var_dict['batch_size'])
        self.subset_size.setValue(self.advanced_conf_var_dict['subset_size'])
        self.performance_thresh.setValue(self.advanced_conf_var_dict['performance_thres'] * 100)
        

        self.model_name.activated.connect(self.process_model_name)
        self.browse.clicked.connect(self.open_file_dialog)
        self.add_labels.clicked.connect(self.process_labels)
        self.clear_labels.clicked.connect(self.clear_labels_list)
        self.use_cuda.stateChanged.connect(lambda: self.set_var('use_cuda', self.use_cuda.isChecked()))
        self.data_format.activated.connect(lambda: self.set_var('data_format', self.data_format.currentText()))

        self.epochs.valueChanged.connect(lambda: self.set_var('epochs', self.epochs.value()))
        self.batch_size.valueChanged.connect(lambda: self.set_var('batch_size', self.batch_size.value()))
        self.subset_size.valueChanged.connect(lambda: self.set_var('subset_size', self.subset_size.value()))
        self.performance_thresh.valueChanged.connect(lambda: self.set_var('performance_thres', self.performance_thresh.value()/ 100.0))
        self.query_mode.activated.connect(lambda: self.set_var('query_mode', self.query_mode.currentText()))
        self.import_config.clicked.connect(self.read_cfg)
        self.first_ok.clicked.connect(self.save_conf)
        self.cvat_api_button.clicked.connect(lambda: self.set_var('annotation_tool', 'cvat_api'))
        self.cvat_manual_button.clicked.connect(lambda: self.set_var('annotation_tool', 'cvat_manual'))
        self.general_button.clicked.connect(lambda: self.set_var('annotation_tool', 'general'))

        self.labels_list_var = []
        self.labels_to_classes_var = []

    def set_var(self, name, val):
        self.var_dict[name]['val'] = val
        self.var_dict[name]['flag'] = True
        if (all([f['flag'] for f in list(self.var_dict.values())])):
            self.first_ok.setEnabled(True)
        # print(val)

    def save_conf(self):
        self.close()

    def closeEvent(self, event):
        self._closed.emit()



    def process_model_name(self):
        model_name = self.model_name.currentText()
        if model_name != "":
            cfg = f'config/{model_name}.cfg'
            hyperparams = parse_model_config(cfg).pop(0)
            classes = hyperparams['classes'].split(',')
            self.labels_to_classes.clear()
            self.labels_to_classes.addItem('')
            for c in classes:
                self.labels_to_classes.addItem(c)
            self.set_var('model_name', model_name)

    def process_exp_name(self):
        exp_name = self.exp_name.text()
        if exp_name != '':
            # checkpoint_dir = os.path.join('checkpoints', exp_name)
            # if os.path.exists(checkpoint_dir):
            #     msg ="The experiment name already exists.\nDo you like to resume annotating? Press 'Yes' for resuming and 'No' for overwriting."
            #     ret = show_msgbox(self, msg, button='yes/no')
            #     if ret == QMessageBox.Yes:
            #         self.set_var('resume', True)
            #     else:
            #         self.set_var('resume', False)
            # else:
            #     self.set_var('resume', False)
            self.set_var('exp_name', self.exp_name.text())
        

    def open_file_dialog(self):
        if not self.var_dict['data_format']['flag']:
            show_msgbox(self, 'Please select data format before browsing the directory.')
        else:
            data_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            if self.var_dict['data_format']['val'] == 'kitti' and not os.path.exists(os.path.join(data_dir,'image_2')) or not os.path.exists(os.path.join(data_dir,'label_2')):
                show_msgbox(self, f'The path {data_dir} does not contain image_2 or label_2 folders. Please check the structure of kitti dataset.')
            else:
                self.data_dir.setText(data_dir)
                self.set_var('data_dir', data_dir)


    def process_labels(self):
        labels = self.labels.text()
        labels_to_classes = self.labels_to_classes.currentText()
        if labels != '' and labels_to_classes!='':
            self.labels_list_var.append(labels)
            self.labels_to_classes_var.append(labels_to_classes)
            self.set_var('labels', self.labels_list_var)
            self.set_var('labels_to_classes', self.labels_to_classes_var)
            txt  = []
            for l , l2c in zip(self.labels_list_var, self.labels_to_classes_var):
                txt.append(l + '=>' + l2c)
            self.labels_to_classes_list.setText('\n'.join(txt))

            self.labels.clear()
            self.labels_to_classes.setCurrentIndex(0)

    def clear_labels_list(self):
        self.labels_list_var = []
        self.labels_to_classes_var = []
        self.labels.clear()
        self.labels_to_classes.setCurrentIndex(0)
        self.labels_to_classes_list.clear()
    
    def get_all_variables(self):
        return {k:v['val'] for k ,v in self.var_dict.items()}

    def read_cfg(self):
        cfg_file_path,_ = QFileDialog.getOpenFileName(self, 'Select Config File', filter='*.json')
        ###### TODO: change this on the latest release
        # cfg_file_path = 'config/config.json'
        if os.path.exists(cfg_file_path):
            with open(cfg_file_path) as outfile:
                opt = json.load(outfile)
            for k , v in opt.items():
                self.var_dict[k]={'val':v, 'flag':True}
            missed_keys = []
            for k, v in self.var_dict.items():
                if not v['flag']:
                    missed_keys.append('"' + k + '"')
            if len(missed_keys) > 0:
                show_msgbox(self, f"The selected config file does not contain the following keys: {', '.join(missed_keys)}")
            else:
                self.close()
        