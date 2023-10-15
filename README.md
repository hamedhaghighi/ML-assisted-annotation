# Machine Learning-Assisted Data Annotation (ML-ADA) for 2D Object Detection with Fine-tuning In the Loop
ML-ADA is a semi-automatic data annotation tool that uses machine learning (ML) models to pre-annotate data. Simultaneously, it actively fine-tunes the models with the refined annotations provided by the user. The primary goal is to reduce the cost of manual annotation efforts by improving the model's performance. Currently, ML-ADA is enabled for the 2D object detection task and leverages Yolov3 as an ML model. Also it is only compatible with the popular kitti data structure and has been fully integrated into [CVAT](https://app.cvat.ai) tool, allowing users to manually refine annotations.

## **Table of Contents**
- [Pipeline Overview](#Pipeline-Overview)
- [Installation](#Installation)
- [Usage](#Usage)
- [Configuration Guide](#configuration-guide)
- [Credit](#credit)


## **ML-ADA Pipeline Overview**
we have provided an overview of our generic annotation pipeline below. The input to the pipeline is an unlabelled dataset and a selected ML model with specified configurations. The pipeline follows the next 6 steps to output the labeled dataset:  

- **Select a subset of data:** A subset of dataset is selected for the current round of labelling. The order of subset selection can be defined either randomly, by user’s preference (e.g., to prioritise images with new labels) or by active-learning prioritisation.  

- **Run the ML model to label the data:** The ML model processes the subset of data and produces pre-annotation.  

- **Create/Refine/Review labels by user:** The user can manually create, refine, or review the produced labels resulting in the final labels for the current subset.  

- **Evaluate the model:** The ML model performance is evaluated on this subset by comparing the produced labels with the manually refined ones.  

- **Check the performance:** The performance score of the model is checked against a threshold. If the score surpasses the threshold, next round of labelling will be repeated from step 1, otherwise, the model is re-trained in the following step.  

- **Re-train the model:** The model is re-trained on this subset using the hyper-parameters defined by the user. The next round of labelling will be initiated when the training is finished. 

![pipeline](assets/pipeline.png)

*Figure1. An overview of the ML-ADA pipeline*

## **Installation**
### **Installation via Docker**

**Step 1.** Build the docker using the provided Dockerfile.
```shell
docker build . -f ./docker/Dockerfile-dev -t mlada
```
**Step 2.** Run the container, mount X11 folder, and attach the bash to the container.
```shell
docker run -v /tmp/.X11-unix/:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY -v /run/user/1000/gdm/Xauthority:/root/.Xauthority -h $HOSTNAME -i -t mlada bash
```



### **Manual Installation**
**Step 1.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 2.** Create a conda environment and activate it.

```shell
conda env create -f environment.yml
conda activate mlada
```

**Step 3.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

**Step 4.** Install pyqt.
```shell
conda install -c anaconda pyqt
```

**Step 5.** Clone the github repository:

```shell
git clone https://github.com/hamedhaghighi/ML-assisted-annotation.git
```

**Step 6.** Change the current dir to the project's

```shell
cd ML-assisted-annotation
```

## **Usage**

Use the following steps to run the tool either on Docker or your platform:

**Step 1.** Register an account on [CVAT](https://app.cvat.ai/auth/register) manual annotation tool.

**Step 2.** Create a json configuration file similar to the one in `config/config.json`. Check the [configuration guide](#configuration-guide) for more information.

**Step 3.** Open the Terminal and activate conda.
```shell
conda activate mlada
```

**Step 4.** Run ML-ADA using the terminal.

To enable GUI, use:

```shell
python ML-ADA.py --gui
```

To enable the command line interface, use:

```shell
python ML-ADA.py --cfg CONFIG_PATH
```
In the case of GUI, either manually set the paramters or browse the config files using "Import Config" button on the opened window.

## **Configuration Guide**

Following is the description of each parameter in the json configuration file:

| Parameter | Description |
| :---        |    :----  |
| data_dir | path to the dataset, e.g. ./data/kitti |
| data_format | format of the dataset for importing and exporting, e.g. kitti |
| exp_name | name of the current experiment, e.g. kitti_annotation |
| use_cuda | true to use cuda,  false to use cpu |
| labels |  your object labels list, e.g. ["Car","Truck", "Pedestrian", "static_object"] |
| labels_to_classes | The mapping between your labels and the labels of the dataset on which the model is pre-trained, e.g. ["car", "truck", "person", -1] |

Note that the folder structure of your dataset should follow the "data_format" parameter.
For instance, folder structure for "kitti" data format should look like the following:

```
[ROOTDIR]
├── image_2
│   ├── 000000.jpeg
│   ├── 000001.jpeg
│   ├── ...
├── label_2
│   ├── 000000.txt
│   ├── 000001.txt
│   ├── ...
```

Rest of the parameters in the config file are for advanced users and you may not need to change them. Following is the explanation of these parameters:

| Parameter | Description |
| :---        |    :----  |
| model_name | Name of the an object detector model including the dataset on which the model is trained on, e.g. yolov3-coco |
| iou_thres | iou threshold required to qualify as detected by evaluation metrics, e.g. 0.5 |
| conf_thres | object confidence threshold used during the detection, e.g. 0.8  |
| nms_thres | iou thresshold for non-maximum suppression used during the detection, e.g. 0.4,  |
| epochs | number of epochs for model re-training, e.g. 10 |
| batch_size | batch size that used for model re-training, e.g. e.g. 32 |
| subset_size | number of images that is annotated during each run of the program pipeline, e.g. 100 |
| performance_thres | Re-training is discontinued when the performance reaches the performance thereshold, e.g. 0.7 |
| img_size | the images are resized to img_size while feeding into the model, e.g. 416 |
| checkpoint_dir | directory for saving the current run logs, e.g. ./checkpoint |
| query_mode | method for selecting the next subset, e.g. random |
| background_training | true to enable the re-training to be run in background,  false to disable it |
| n_cpu | number of cpu threads to be used in reading the dataset from drive, e.g. 4 |

## **Credit**

Yolov3 implementation from [packyan](https://github.com/packyan/PyTorch-YOLOv3-kitti)
