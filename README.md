# Multi Person Detection and Tracking

This repository contains a Python script for person detection and tracking using the YOLOv3 object detection model and OpenCV. The script processes a video stream or video file and detects and tracks people in real-time. Note that the script currently runs on CPU, so the frame rate may be limited compared to GPU-accelerated implementations.

## Features

- Real-time person detection and tracking
- Trajectory visualization for tracked persons
- Toggle processing frames on and off using the 'a' key
- Display frames per second (FPS) on the video feed

## Prerequisites

Before running the script, ensure you have the following prerequisites installed:

- Python (3.x recommended)
- OpenCV (cv2)
- NumPy
- CUDA (optional for GPU acceleration)
- YOLOv3 weights and configuration files
- COCO names file

You can install the required Python libraries using pip:

```bash
pip install opencv-python numpy
```






# Setup
```bash
git clone https://github.com/pardhu-nadella/multi-person-detection-and-tracking.git
cd multi-person-detection-and-tracking
```



# Usage
To run the person detection and tracking script, use the following command:
```bash
python main.py
```

You can change the video source by modifying the cap variable. For example, to use a webcam, set cap to:
```bash
cap = cv2.VideoCapture(0)
```

To use a video file, provide the file path:
```bash
cap = cv2.VideoCapture("path_to_an_mp4_file.mp4")
```


# Controls
Press 'a' to toggle processing frames on and off.
Press 'q' to exit the application.



# Notes
The script is set to run on CPU by default. To enable GPU acceleration, uncomment the net.setPreferableBackend and net.setPreferableTarget lines accordingly.

```bash
# For GPU acceleration
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```
