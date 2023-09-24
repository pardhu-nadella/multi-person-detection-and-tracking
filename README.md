# Multi Person Detection and Tracking

This repository contains a Python script for person detection and tracking using the YOLOv3 object detection model and OpenCV. The script processes a video stream or video file and detects and tracks people in real-time. Note that the script currently runs on CPU, so the frame rate may be limited compared to GPU-accelerated implementations.

## Versions

- personTrackingV1.py -> Contains bare minimum backbone of multi person detection and tracking by creating an object for every person and plotting their prajectory.
- personTrackingV2.py -> Updated version of personTrackingV1.py. Counting the number of people going in and coming out is done. Didn't give good enough results.
- main.py             -> Thresholding is done to eliminate the multiple people being detected.

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
- YOLOv3 weights and configuration files (Included in this repository)
- COCO names file (Included in this repository)

You can install the required Python libraries using pip:

```bash
pip install opencv-python numpy
```






# Setup
```bash
git clone https://github.com/pardhu-nadella/multi-person-detection-and-tracking.git
cd multi-person-detection-and-tracking
```
Download the dependencies for this project in the same working directory.
<a href="https://example.com" class="button">Download Dependencies</a>



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
(Lines 16 and 17 in main.py)


# Controls
- Press 'a' to toggle processing frames on and off.
- Press 'q' to exit the application.



# Notes
The script is set to run on CPU by default. To enable GPU acceleration, uncomment the net.setPreferableBackend and net.setPreferableTarget lines (lines 28 and 29 in main.py) accordingly.

```bash
# For GPU acceleration
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

# Example Output
<a href="https://youtu.be/8HyhrgQ-PSM">Follow this link for video output</a>





You're encouraged to actively contribute to this personal repository by enhancing its functionality, introducing innovative features, or optimizing it in any way you can. If you encounter any challenges or have valuable insights to share, please don't hesitate to open a GitHub issue. Your input is highly appreciated, and your collaboration in refining this project for even better results is welcome.

For engaging discussions and further communication, please feel free to connect with the repository owner on LinkedIn. Your feedback is valued. Here's to a successful journey in the realm of person detection and tracking!
