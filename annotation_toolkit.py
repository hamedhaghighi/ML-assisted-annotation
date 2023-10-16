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
        if self.use_api and self.tool_name == 'cvat':
            if is_gui:
                self.print_msg('Please enter your CVAT.ai account credential on the opened window.')
                self.login_ui_cl = LoginUI(self)
                self.login_ui_cl.pushButton.clicked.connect(self.get_user_pass)
                self.login_ui_cl.exec()
            else:
                self.print_msg('Please enter your CVAT.ai account credential below:')
                self.user = input('username:')
                self.password = getpass()
                # TODO: remove this line
            self.configuration = Configuration(host="https://app.cvat.ai",username=self.user, password=self.password)
            try:
                ـ = ApiClient(self.configuration)
            except:
                self.print_msg('Authentication Failed! Please re-run the app', True)
                exit(1)
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
                self.print_msg(f'Exception when calling CVAT.ai TasksApi {e}')
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
                        labels  = copy(self.opt.labels)
                        labels.append('DontCare')
                        task_spec = {"name": self.opt.exp_name, 'labels': []}
                        for label in labels:
                            task_spec['labels'].append({'name':label})
                        self.print_msg(f'Creating annotation task on CVAT.ai ...')
                        task, _ = self.try_func(api_client.tasks_api.create, task_write_request=task_spec)
                        self.task_id = task.id
                        self.print_msg(f'Annotation task with id:{self.task_id} is created on CVAT.ai.')
                        root_files = self.opt.data_dir
                        if self.opt.data_format == 'kitti':
                            root_files = os.path.join(root_files, 'image_2')
                        self.print_msg('Uploading image dataset to the task ...')
                        task_data = models.DataRequest(image_quality=75, client_files=[open(img_path, 'rb') for img_path in list(glob(os.path.join(root_files, '*')))],)
                        self.try_func(api_client.tasks_api.create_data, id=task.id, data_request=task_data, _content_type="multipart/form-data", _check_status=False, _parse_response=False)
                        self.print_msg('Image dataset is uploaded successfully.')

        else:
            if self.tool_name == 'cvat':
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
                self.print_msg('Uploading subset pre-annotation to the task.')
                with ApiClient(self.configuration) as api_client:
                    self.try_func(api_client.tasks_api.update_annotations, is_upload_annotation=True, id=self.task_id,\
                                format='KITTI 1.0', _content_type='multipart/form-data')
                self.print_msg('The subset pre-annotation is uploaded to the task.')
                    

    def correct_annotations(self):
        if self.use_api:
            if self.tool_name == 'cvat':
                if self.is_gui:
                    self.print_msg(self.get_messages('api_msg'))
                    instruction_ui = InstructionUI(self, self.get_messages('api_msg'))
                    instruction_ui.exec()
                else:
                    _str = self.get_messages('api_msg').split(' ')
                    _str = filter(lambda s: s!='', _str)
                    self.print_msg(' '.join(_str))
                    while True:
                        if input() == 'ok':
                            break 
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
                    msg = "Are you done with manual annotation?"
                    while True:
                        ret = show_msgbox(self, msg, 'yes/no', 'info', self.is_gui)
                        if self.is_gui:
                            if ret == QMessageBox.Yes:
                                while(True):
                                    ret = show_msgbox(self, 'Have you saved the project on CVAT.ai?', 'yes/no', 'info', self.is_gui)
                                    if ret == QMessageBox.Yes:
                                        break
                                break
                        else:
                            self.print_msg("Press 'yes' or 'no' to continue ...")
                            if input() == 'yes':
                                while(True):
                                    self.print_msg('Have you saved the project on CVAT.ai?')
                                    self.print_msg("Press 'yes' or 'no' to continue ...")
                                    if input() == 'yes':
                                        break
                                break
                    self.print_msg('Downloading the refined annotations ...')
                    data , response = self.try_func(api_client.tasks_api.retrieve_annotations, id=self.task_id, format='KITTI 1.0', filename='subset.zip', action='download', _parse_response=False)
                    with open(os.path.join(self.checkpoint_dir, 'Refined_Subset.zip'), 'wb') as output_file:
                        output_file.write(response.data)
                    self.print_msg('The refined annotations are downloaded.')
                    
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
                while True:
                    if input() == 'ok':
                        break 
        self.check_annotation()

    def destroy_annotation(self):
        if self.use_api and self.tool_name == 'cvat':
            with ApiClient(self.configuration) as api_client:
                api_client.tasks_api.destroy_annotations(id=self.task_id)

    def delete_task(self):
        if self.use_api and self.tool_name == 'cvat':
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
            2- Use the right arrow on the keyboard to view the next annotated image.{br}\
            3- Refine the annotations using the tool (see guidelines {get_a_ref('https://opencv.github.io/cvat/docs/manual/advanced/annotation-with-rectangles/', 'here')}).{br}\
            4- Save the project and export the dataset as \"Refined_Subset.zip\" to \"{self.checkpoint_dir}\" directory (see the guidelines {get_a_ref('https://opencv.github.io/cvat/docs/manual/advanced/export-import-datasets/#export-dataset', 'here')}). {br}\
            5- Delete the annotation in the task to avoid the confusion with the next annotation subset. {br} \
            6- Press OK button when you are done. "

            labeling_tool_msg = f"Set up the manual annotation tool (CVAT.ai), following the instructions below: {br}\
            1- Open {get_a_ref('https://app.cvat.ai', 'CVAT.ai tool')} in your browser. {br} \
            2- Create an annotation task (see the guidelines {get_a_ref('https://opencv.github.io/cvat/docs/manual/basics/create_an_annotation_task/', 'here')}). {br}\
            3- Include the predefined labels from the config file in addition to the 'DontCare' label into the task.{br} \
            4- Import your dataset (see the guidelines {get_a_ref('https://opencv.github.io/cvat/docs/manual/advanced/export-import-datasets/#import-dataset', 'here')})."
            api_msg = f"ML-ADA is going to pop up CVAT.ai app on the chrome. Please use the following guidelines to manually refine the annotations: {br} \
                1- Use the right arrow on the keyboard to view the next annotated image.{br}\
                2- Save the project once refining is done and continue on ML-ADA.{br}\
                Press ok to continue ...   "
        elif self.tool_name == 'general':
            subset_ann_filename = 'Subset.zip'
            refined_ann_filename = 'Refined_Subset.zip' if self.opt.data_format == 'kitti' else 'Refined_Subset_OL_annotations.json'
            label_refinement_msg = f"Use the following instructions to manually refine the pre-annotations:{br}\
            1- Create an annotation task by uploading the current subset images and pre-annotations provided in \"{os.path.join(self.checkpoint_dir, subset_ann_filename)}\". {br}\
            2- Refine the annotations using the tool.{br}\
            3- Export the the refined annotations as \"{refined_ann_filename}\" to \"{self.checkpoint_dir}\" directory.{br}\
            4- Press OK button once you are finished refining. "
            labeling_tool_msg = None
            api_msg = None
        messages_dict['label_refinement'] = label_refinement_msg
        messages_dict['labeling_tool'] = labeling_tool_msg
        messages_dict['api_msg'] = api_msg
        return messages_dict[key]

    def check_annotation(self):
        file_exist = lambda path: os.path.exists(path)
        subset_ann_path = os.path.join(self.checkpoint_dir, 'Refined_Subset.zip' if self.opt.data_format == 'kitti' else 'Refined_Subset_OL_annotations.json')
        if not file_exist(subset_ann_path):
            while(True):
                ret = show_msgbox(self, f'{subset_ann_path} has not been found. Please export the refined annotations as instructed and press ok once done.', 'OK', 'error', self.is_gui)
                if self.is_gui:
                    if ret == QMessageBox.Ok and file_exist(subset_ann_path):
                        break
                elif input() == 'ok' and file_exist(subset_ann_path):
                    break
                sleep(1)

            
    def print_msg(self, msg, is_error=False):
        if self.is_gui:
            if is_error:
                show_msgbox(self, msg, 'OK', 'error', True)
            else:
                self.text_browser_signal.emit(msg)
        else:
            print(msg)