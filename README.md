# AirDet: Few-Shot Detection without Fine-tuning for Autonomous Exploration

### Anonymous ECCV submission

### Paper ID 4293

###  

## Abstract

Few-shot object detection has attracted increasing attention and rapidly progressed in recent years. However, the requirement of an exhaustive offline fine-tuning stage in existing methods is time-consuming and significantly hinders their usage in online applications such as autonomous exploration of low-power robots. We find that their major limitation is that the little but valuable information from a few support images is not fully exploited. To solve this problem, we propose a brand new architecture, AirDet, and surprisingly find that, by learning \textit{class-agnostic relation} with the support images in all modules, including cross-scale object proposal network, shots aggregation module, and localization network, AirDet without fine-tuning achieves comparable or even better results than the exhaustively fine-tuned methods, reaching up to \textbf{30-40\%} improvements. We also present solid results of onboard tests on real-world exploration data from the DARPA Subterranean Challenge, which strongly validate the feasibility of AirDet in robotics. To the best of our knowledge, AirDet is the first feasible few-shot detection method for autonomous exploration of low-power robots. The source code, pre-trained models, along with the real-world data for exploration, are released at {\tt this anonymous link}.



## Overview

We provide official implementation here to reproduce the results of ResNet101 backbone on COCO-2017 validation and VOC-2012 validation dataset **w/o** fine-tuning.



## Installation

Please create a python environment including:

Python                  3.6.9

numpy                   1.19.2

detectron2              0.2

CUDA compiler           CUDA 10.2

PyTorch                 1.5.1

Pillow                  8.3.1

torchvision             0.6.0

fvcore                  0.1.5

cv2                     4.5.4

We'll also provide the official docker image in the future for faster reproduction.



## Dataset Preparation

### 1. Download official datasets

[MS COCO 2017](https://cocodataset.org/#home)

[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

[COCO format VOC annotations](https://s3.amazonaws.com/images.cocodataset.org/external/external_PASCAL_VOC.zip)

Expected dataset Structure:

```shell
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
```

```shell
VOC20{12}/
  annotations/
  	json files
  JPEGImages/
```

### 2. Generate supports 

Download and unzip support [COCO json](https://mega.nz/file/QEETwCLJ#A8m0R7NhJ-MUNuT1fhzEgRIg6t5R69u5rAaBHTsqgUw) files in

```shell
datasets/
  coco/
    new_annotations/
```

Download and unzip support [VOC json](https://mega.nz/file/BBcjjYwY#1S3Utg99D_WyfzN5qq0UfeuFrlh7Eum2jZs9U7GHhJY) files in

```shell
datasets/
  voc/
    new_annotations/
```

Run the script

```shell
cd datasets
bash generate_support_data.sh
```

You may modify 4_gen_support_pool_10_shot.py line 190, 213, and 269 with different shots (default is 1 shot).



## Reproduce

### Base training

Download base [R-101 model](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl) in /output

start training

```shell
bash train.sh
```

We also provide official trained model [here](https://mega.nz/file/1YcBWQ4R#mCtaHS9RE2pzmPlAmOAAtk-IghBNiW95oSX4Lfktw4Y)

Put the model in /output/R101/

### Inference w/o fine-tuning

```shell
bash test.sh
```

You'll get the results in /log



## Acknowledgement

Our code is built on top of [FewX](https://github.com/fanq15/FewX), we express our sincere gratitude for the authors.
