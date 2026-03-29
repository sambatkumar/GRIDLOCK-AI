# GRIDLOCK-AI

AI-powered helmet detection system using YOLOv8 integrated with a simulation environment to dynamically control vehicle speed based on safety compliance.

## Features

* Real-time helmet detection using webcam input
* YOLOv8-based safety compliance detection
* Simulation-based speed restriction logic
* Visual alerts and UI feedback
* Training pipeline for custom dataset

## Tech Stack

* Python
* YOLOv8
* OpenCV
* Pygame
* PyTorch

## Note

Vehicle speed in this project is simulated. The real-time component is helmet detection using computer vision.

## Files

* `bike.py` — simulation + detection integration
* `detect.py` — standalone helmet detection pipeline
* `train.py` — YOLOv8 training script
* `helmet_dataset.yaml` — dataset config
* `requirements.txt` — dependencies
