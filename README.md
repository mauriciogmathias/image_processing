# image_processing

# Image Processing and Deep Learning with OpenCV, Tkinter, and TensorFlow

This repository contains three separate image processing and computer vision projects using different techniques and frameworks:
- **Averaging and Edge Detection with OpenCV**: A basic introduction to image filtering and edge detection using OpenCV.
- **Real-Time Face Detection with OpenCV and Tkinter**: A face detection application using OpenCV's DNN module with a graphical interface built with Tkinter.
- **DeepLab Semantic Segmentation with TensorFlow**: A deep learning model implementation for semantic segmentation using TensorFlow.
  
---

## Overview

This repository showcases different computer vision techniques using OpenCV, Tkinter, and TensorFlow:

1. **Averaging and Edge Detection with OpenCV**:
   - This script demonstrates image filtering and edge detection techniques like averaging, Laplacian filtering, and Sobel filtering.
   - It uses OpenCV to perform basic image processing and displays the results using Matplotlib.

2. **Real-Time Face Detection with Tkinter**:
   - This script captures video from the webcam and uses a deep learning-based face detection model (SSD with ResNet architecture) from OpenCV's DNN module.
   - A GUI is built using Tkinter, which allows users to start the camera feed, detect faces, and display the results in a new window.

3. **DeepLab Semantic Segmentation with TensorFlow**:
   - This script loads the pre-trained DeepLab model from TensorFlow, a state-of-the-art model for semantic segmentation.
   - The script downloads the model, runs inference on input images, and visualizes the segmentation results using Matplotlib.

---

## Installation

### Dependencies

To run the code in this repository, you need the following libraries:

- **Python 3.x**
- **OpenCV** (`cv2`)
- **NumPy**
- **Matplotlib**
- **Pillow (PIL)** (for handling images in Tkinter)
- **Tkinter** (for building the GUI in the face detection script)
- **Requests** (for downloading the model and files from the internet)
- **TensorFlow** (for the DeepLab model)
