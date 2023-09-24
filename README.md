# Person Detection and Tracking

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

