# yolo_1_pytorch

###### simplest implementation of yolo v1 via pytorch √
##### Language: [中文](中文.md)
##### paper: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
##### CSDN blog: [博客解析](https://muzhan.blog.csdn.net/article/details/82588059)
This repo is a brief implementation of yolo v1. You can easily train the model and visualize the result.
![imgs](https://github.com/leviome/yolo_1_pytorch/blob/master/imgs/yolo.png)
```
git clone https://github.com/leviome/yolo_1_pytorch.git
cd yolo_1_pytorch
```

Environment:
---
- *Python3*
- *Pytorch>=1.3*
- *cv2*
- *matplotlib*

Dataset preparation
---
1. Download voc2007 dataset:
```
wget -c http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget -c http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
wget -c http://pjreddie.com/media/files/VOCdevkit_08-Jun-2007.tar
```
2. Extract all tars:
```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```
3. put the data into dataset/voc2007 and make the folder structure look like:
```
dataset
├── voc2007
│   ├── Annotations
│   ├── ImageSets
│   ├── JPEGImages
│   ├── Label
│   ├── SegmentationClass
│   └── SegmentationObject
└── voc2012
```
4. fit voc dataset to yolo model as pytorch dataset format:
```
python fit_voc_to_yolo.py
```
Train
---
```
python train.py
```
Detect single image
---
```
python detect.py
```
Demo
---
![imgs](https://github.com/leviome/yolo_1_pytorch/blob/master/imgs/demo1.png)
![imgs](https://github.com/leviome/yolo_1_pytorch/blob/master/imgs/demo2.png)
