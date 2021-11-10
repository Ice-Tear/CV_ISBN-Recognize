# CV_ISBN-Recognize

《计算机视觉》 实验+结课作业 isbn识别

## 简介

本项目为研一《计算机视觉》实验+结课作业

### 实验

图像的基本操作，包括：图像灰度化，灰度直方图，二值化，边缘检测，图像特征提取

### 结课作业

本项目的目的是将100张图中的isbn号识别出来（不通过条形码）并计算正确率

项目整个流程为：图像预处理，ROI提取（水平投影），字符分割（垂直投影），svm训练，字符识别

**关键词**：图像校正；直方图均衡化；拉普拉斯锐化；gamma锐化；SVM；数字字符识别

**注：本项目是针对老师的测试集进行预处理+识别的，并不具有普适性**

## 环境

* **Language**：python 3.8
* **Libraries**：opencv_python,numpy,scipy,matplotlib,pandas,scikit_learn,joblib,scikit_image,numba

## 使用

### 测试方法

1. 将测试用图放入resource/test_resource/test_Origin中
2. 运行main.py
3. 在resource/test_resource/test_Error中查看识别失败的图片及其信息

### 训练方法

1. 将训练用图放入resource/train_resource/train_Origin中
2. 运行svm_train.py

**注**：.gitkeep文件只为保留目录结构，若报错请自行删除。

## 正确率

图片识别准确率：99%

字符识别准确率：99%

## 写在最后

此项目为本人学习python第一个项目，在这里只提供一个识别isbn的思路，代码写的很烂，请见谅。

若有问题请联系邮箱*lzh961031@126.com*
