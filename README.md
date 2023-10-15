# Machine Learning-Assisted Data Annotation (ML-ADA) for 2D Object Detection with Fine-tuning In the Loop
ML-ADA is a semi-automatic data annotation tool that uses machine learning (ML) models to pre-annotate data. Simultaneously, it actively fine-tunes the models with the refined annotations provided by the user. Currently, ML-ADA supports:

- 2D object detection
- confidence-based sampling strategy for active learning
- Kitti and OpenLabel data annotation formats
- Yolov3 for object detection
- API-based integration with [CVAT.ai](https://app.cvat.ai).


![pipeline](assets/pipeline-3.png)
\*Manual annotation tool is an external tool and it is wrapped by ML-ADA.
## **Table of Contents**
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Guide](#configuration-guide)
- [Credit](#credit)
 
<!-- 
*Figure1. An overview of the ML-ADA pipeline* -->
## **Installation**
**Step 1.** Clone the github repository:

```shell
git clone https://github.com/hamedhaghighi/ML-assisted-annotation.git
```

**Step 2.** Change the current dir to the project's

```shell
cd ML-assisted-annotation
```
**Step 3.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 4.** Create a conda environment and activate it.

```shell
conda env create -f environment.yml
conda activate mlada
```

**Step 5.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

**Step 6.** Install pyqt.
```shell
conda install -c anaconda pyqt
```
If you encounter an error related to 'libffi.so.6' library. You can resolve it by downloading the library from this [link](https://mirrors.kernel.org/ubuntu/pool/main/libf/libffi/libffi6_3.2.1-8_amd64.deb) and then proceed to install it.
```shell
sudo apt install ./libffi6_3.2.1-8_amd64.deb
```
## **Usage**

Use the following steps to run the tool either on Docker or your platform:

**Step 1.** Register an account on [CVAT.ai](https://app.cvat.ai/auth/register) manual annotation tool.

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
| annotation_tool | name of the manual annotation tool, could be either "cvat_api", "cvat_manual", or "general" |
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
