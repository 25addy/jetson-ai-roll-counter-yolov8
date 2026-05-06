# Jetson AI Roll Counter - YOLOv8 Industrial Vision System

AI camera-based roll counting system running on **NVIDIA Jetson Orin Nano**, using YOLOv8, OpenCV, PySide6 touchscreen UI, TensorRT model inference, and optional MySQL production logging.

## Features

- Real-time roll detection using YOLOv8
- NVIDIA Jetson Orin Nano deployment
- TensorRT `.engine` model support for faster inference
- Live camera preview with detection boxes
- 3-zone counting logic: Top Zone, Count Zone, Exit Zone
- Direction lock to reduce false counting
- Pallet number search using MySQL when enabled
- Target quantity tracking
- Start, stop, pause, reset, and add roll functions
- CSV backup logging
- Optional MySQL production log saving
- TensorRT / ONNX / PT model support

## Hardware

- NVIDIA Jetson Orin Nano
- USB camera / industrial USB camera
- Touchscreen monitor
- Conveyor system
- Local MySQL server when database logging is enabled

## Tech Stack

- Python
- PySide6
- OpenCV
- Ultralytics YOLOv8
- MySQL
- TensorRT / ONNX / PyTorch
- Ubuntu on NVIDIA Jetson Orin Nano

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
