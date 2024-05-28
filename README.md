
# Object Detection Model Implementation 

## Introduction

Welcome to the GitHub repository of my internship project at the university of Sønderbog in Denmark for creating an artificial intelligence model for traffic sign detection. This project aims to develop a machine learning model capable of detecting and recognizing, initially, road traffic signs from images and, subsequently, vandalized signs. This GitHub contains the various methods that allowed me to develop the model and the different techniques used to improve the model's performance.

## Context

Traffic sign detection is a crucial task for autonomous driving systems and road safety applications. This project aims to create a detection model based on deep learning.

Traffic signs are crucial for maintaining road safety.

Non-compliant signs can cause:

- Confusion for drivers, cyclists, and pedestrians.
- An increase in the number of accidents and traffic violations.

## Objectives

1. Collect and preprocess a dataset of traffic sign images.
2. Develop a deep learning model for the detection and recognition of traffic signs.
3. Evaluate the model's performance.
4. Improve the model's performance.
5. Integrate the model into a drone.


## Table of Contents

- [Requirements](#Requirements)
- [Training](#Training)
- [Improvement](#Improvement)


## Requirements

Before starting, make sure you have the following installed:

- Python 3.10.3
- Nvidia CUDA 11.7, to use your machine's GPU, essential for training (link for download: https://developer.nvidia.com/cuda-11-7-0-download-archive)
- PyTorch to load and use the models (link for download: https://pytorch.org/get-started/locally/)
- Nvidia TensorRT, a technique to export the model in a different format to improve model performance (https://developer.nvidia.com/tensorrt)
- Nvidia CuDNN 8.4.1 (link for download: https://developer.nvidia.com/rdp/cudnn-archive)

## Training

You will find in the folder `scripts` the two "training" files that allow you to train the DETR and YOLO architecture models. These are the two models that I analyzed, used, and compared during this internship.

I turned to the YOLO model, where I obtained the best performance.

## Improvement

To improve my model's performance, I studied the knowledge distillation method, which involves transferring knowledge from a complex model to a lighter model to reduce the model's size without affecting accuracy.

Next, you will find in the file: /scripts/export_to_TensorRT_format.py
A script to export the model in TensorRT format to double the inference speed without changing the accuracy.

The "scripts" folder contains all methods that I used for improving my model performance during this internship.