from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from ui_files.instruction_ui import Ui_Dialog
from login import Ui_Login as login_ui
from PyQt5.QtWidgets import QDialog
import os
import requests
import json
from pprint import pprint
from glob import glob
from cvat_sdk.api_client import Configuration, ApiClient, models, apis, exceptions
from cvat_sdk.api_client.models import *
from copy import copy
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from time import sleep
from utils.utils import show_msgbox
from PyQt5.QtWidgets import QMessageBox
from getpass import getpass
import traceback

class InstructionUI(QDialog, Ui_Dialog):
    def __init__(self, parent, message):
        super().__init__()
        
        self.setupUi(self)
        html_text = f'<html> \
<head> \
</head> \
<body> \
<p>{message}</p>\
</body> \
</html> '
        self.label.setText(html_text)
        self.pushButton.clicked.connect(self.close)
        self.label.linkActivated.connect(self.link)

    def link(self, linkStr):
        QDesktopServices.openUrl(QUrl(linkStr))


class LoginUI(QDialog, login_ui):
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)


class AnnotationToolkit:
    def __init__(self, opt, is_gui, text_browser_signal=None):
        self.tool_name = opt.annotation_tool
        self.checkpoint_dir = opt.checkpoint_dir
        self.is_gui = is_gui
        self.text_browser_signal = text_browser_signal
        self.use_api = opt.use_api
        self.opt = opt
        if self.use_api:
            if is_gui:
                self.print_msg('Please enter your CVAT account credential on the opened window.')
                self.login_ui_cl = LoginUI(self)
                self.login_ui_cl.pushButton.clicked.connect(self.get_user_pass)
                self.login_ui_cl.exec()
            else:
                self.print_msg('Please enter your CVAT account credential below:')
                self.user = input('username:')
                self.password = getpass()
            try:
                # TODO: remove this line
                self.configuration = Configuration(host="https://app.cvat.ai",username=self.user, password=self.password)
            except:
                raise Exception('Authentication Failed!')
            self.task_id = None
            self.driver = None

    def get_user_pass(self):
        self.user = self.login_ui_cl.lineEdit.text()
        self.password = self.login_ui_cl.lineEdit_2.text()
        self.login_ui_cl.close()

    def try_func(self, fn, is_upload_annotation=False, **kwargs):
        max_try = 3
        for i in range(max_try):
            try:
                if is_upload_annotation:
                    task_annotations_update_request = TaskAnnotationsUpdateRequest(annotation_file=open(os.path.join(self.checkpoint_dir, 'Subset.zip'), 'rb'))
                    kwargs['task_annotations_update_request'] = task_annotations_update_request
                data, response = fn(**kwargs)
                if response.status in [200, 201, 202] and response.data != b'':
                    return data, response
                else:
                    self.print_msg(f'Api call was not successful! {response.data}. Trying again ...')
            except Exception as e:
                self.print_msg(traceback.format_exc())
                self.print_msg(f'Exception when calling CVAT TasksApi {e}')
        self.print_msg("Couldn't make the api call after maximum tries. Please try again.")
        exit(1)

    def create_task(self):
        if self.use_api:
            if self.tool_name == 'cvat':
                with ApiClient(self.configuration) as api_client:
                    data, _ = self.try_func(api_client.tasks_api.list, name=self.opt.exp_name)
                    if len(data.results) > 0 :
                        self.task_id = data.results[0].id
                    else:
                        self.print_msg(f'Creating an annotation task on CVAT ...')
                        labels  = copy(self.opt.labels)
                        labels.append('DontCare')
                        task_spec = {"name": self.opt.exp_name, 'labels': []}
                        for label in labels:
                            task_spec['labels'].append({'name':label})

                        task, _ = self.try_func(api_client.tasks_api.create, task_write_request=task_spec)
                        self.task_id = task.id
                        root_files = self.opt.data_dir
                        if self.opt.data_format == 'kitti':
                            root_files = os.path.join(root_files, 'image_2')
                        task_data = models.DataRequest(image_quality=75, client_files=[open(img_path, 'rb') for img_path in list(glob(os.path.join(root_files, '*.jpeg')))],)
                        self.try_func(api_client.tasks_api.create_data, id=task.id, data_request=task_data, _content_type="multipart/form-data", _check_status=False, _parse_response=False)
                        self.print_msg(f'Annotation task with id:{self.task_id} is created on CVAT.')
            else:
                raise NotImplemented;
        else:
            if self.is_gui:
                instruction_ui = InstructionUI(self, self.get_messages('labeling_tool'))
                instruction_ui.exec()
            else:
                _str = self.get_messages('labeling_tool').split(' ')
                _str = filter(lambda s: s!='', _str)
                self.print_msg(' '.join(_str))
                
    def upload_annotations(self):
        # If an object is modified on the server, the local object is not updated automatically.
        # api_client.jobs_api.create_annotations(job_id, annotation_file_request, format='KITTI 1.0', _content_type='multipart/form-data')
        if self.use_api:
            if self.tool_name == 'cvat':
                with ApiClient(self.configuration) as api_client:
                    self.try_func(api_client.tasks_api.update_annotations, is_upload_annotation=True, id=self.task_id,\
                                format='KITTI 1.0', _content_type='multipart/form-data')
                    

    def correct_annotations(self):
        if self.use_api:
            if self.tool_name == 'cvat':
                with ApiClient(self.configuration) as api_client:
                    data, _ = self.try_func(api_client.jobs_api.list, task_id=self.task_id)
                    job_id = data.results[0].id       
                    url = f'https://app.cvat.ai/tasks/{self.task_id}/jobs/{job_id}'
                    if self.driver is None:
                        options = webdriver.ChromeOptions()
                        options.add_argument("--start-maximized")
                        self.driver = webdriver.Chrome(options=options)
                        self.driver.get(url)
                        WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="credential"]')))
                        username = self.driver.find_element(By.XPATH, '//*[@id="credential"]')
                        username.send_keys(self.user)
                        password = self.driver.find_element(By.XPATH, '//*[@id="password"]')
                        password.send_keys(self.password)
                        self.driver.find_element(By.XPATH, '//*[@id="root"]/section/section/main/div/div[2]/div/div/div/form/div[3]/div/div/div/button').click()
                    else:
                        self.driver.get(url)
                    # self.print_msg("Type \"yes\" when the annotation refinement is finished:")
                    msg = "Use the opened browser window to manually refine the annotations.\nSet the filter to 'Label is not null' to only refine the annotated images.\nSave the project and then press ok once finished."
                    while True:
                        ret = show_msgbox(self, msg, 'OK', 'info', self.is_gui)
                        if self.is_gui:
                            if ret == QMessageBox.Ok:
                                break
                        elif input() == 'ok':
                            break
                    data , response = self.try_func(api_client.tasks_api.retrieve_annotations, id=self.task_id, format='KITTI 1.0', filename='subset.zip', action='download', _parse_response=False)
                    with open(os.path.join(self.checkpoint_dir, 'Refined_Subset.zip'), 'wb') as output_file:
                        output_file.write(response.data)
                    
            else:
                raise NotImplemented

        else:
            if self.is_gui:
                instruction_ui = InstructionUI(self, self.get_messages('label_refinement'))
                instruction_ui.exec()
            else:
                _str = self.get_messages('label_refinement').split(' ')
                _str = filter(lambda s: s!='', _str)
                self.print_msg(' '.join(_str))
                msg = 'Press ok button when you are done'
                while True:
                    if input(msg) == 'ok':
                        break 
        self.check_annotation()

    def destroy_annotation(self):
        with ApiClient(self.configuration) as api_client:
            api_client.tasks_api.destroy_annotations(id=self.task_id)

    def delete_task(self):
        with ApiClient(self.configuration) as api_client:
            data, _ = self.try_func(api_client.tasks_api.list, name=self.opt.exp_name)
            if len(data.results) > 0 :
                task_id = data.results[0].id
                api_client.tasks_api.destroy(id=task_id)

        
    def get_messages(self, key):
        def get_a_ref(url, p):
            if self.is_gui:
                return f'<a href=\"{url}\" target=\"_blank\">{p}</a>'
            return url
        messages_dict = {}
        br = '<br>' if self.is_gui else '\n'
        if self.tool_name == 'cvat':
            label_refinement_msg = f"Manually refine labels, following the instructions below:{br}\
            1- Upload pre-annotations \"{os.path.join(self.checkpoint_dir,'Subset.zip')}\" into the manual annotation tool (see the guidelines {get_a_ref('https://opencv.github.io/cvat/docs/manual/advanced/export-import-datasets/#upload-annotations', 'here')}).{br}\
            2- Refine the annotations using the tool (see guidelines {get_a_ref('https://opencv.github.io/cvat/docs/manual/advanced/annotation-with-rectangles/', 'here')}).{br}\
            3- Export the dataset as \"Refined_Subset.zip\" to \"{self.checkpoint_dir}\" directory (see the guidelines {get_a_ref('https://opencv.github.io/cvat/docs/manual/advanced/export-import-datasets/#export-dataset', 'here')}). {br}\
            4- Delete the annotation in the task to avoid the confusion with the next annotation subset. {br} \
            5- Press OK button when you are done. "

            labeling_tool_msg = f"Set up the manual annotation tool (CVAT), following the instructions below: {br}\
            1- Open {get_a_ref('https://app.cvat.ai', 'CVAT tool')} in your browser. {br} \
            2- Create an annotation task (see the guidelines {get_a_ref('https://opencv.github.io/cvat/docs/manual/basics/create_an_annotation_task/', 'here')}).{br} \
            3- Import your dataset (see the guidelines {get_a_ref('https://opencv.github.io/cvat/docs/manual/advanced/export-import-datasets/#import-dataset', 'here')}). {br}"
        elif self.tool_name == 'ilabel':
            label_refinement_msg = f"Manually refine labels in ILabel, following the instructions below:{br}\
            1- Create \"Open-Label Pre-annotated Project\" by entering your image dataset directory and ML-ADA genererated annotation path as \"{os.path.join(checkpoint_dir,'subset_OL_annotation.json')}\". {br}\
            2- Refine the annotations using the tool.{br}\
            3- Export the the refined labels as \"refined_OL_annotation.json\" to \"{self.checkpoint_dir}\" directory.{br}\
            4- Press OK button when you are done. "

            labeling_tool_msg = None
        messages_dict['label_refinement'] = label_refinement_msg
        messages_dict['labeling_tool'] = labeling_tool_msg
        return messages_dict[key]

    def check_annotation(self):
        file_exist = lambda path: os.path.exists(path)
        subset_ann_path = os.path.join(self.checkpoint_dir, 'Refined_Subset.zip')
        if not file_exist(subset_ann_path):
            while(True):
                ret = show_msgbox(self, f'{subset_ann_path} not found. Please try again and Press ok while finished.', 'OK', 'error', self.is_gui)
                if self.is_gui:
                    if ret == QMessageBox.Ok and file_exist(subset_ann_path):
                        break
                elif input() == 'ok' and file_exist(subset_ann_path):
                    break
                sleep(1)

            
    def print_msg(self, msg):
        if self.is_gui:
            self.text_browser_signal.emit(msg)
        else:
            print(msg)