# AI Roll Counter - YOLOv8 Industrial Vision System

AI camera-based roll counting system using YOLOv8, OpenCV, PySide6 touchscreen UI, and optional MySQL production logging.

## Features

- Real-time roll detection using YOLOv8
- Live camera preview with detection boxes
- 3-zone counting logic: Top Zone, Count Zone, Exit Zone
- Direction lock to reduce false counting
- Pallet number search using MySQL when enabled
- Target quantity tracking
- Start, stop, pause, reset, and add roll functions
- CSV backup logging
- Optional MySQL production log saving
- TensorRT / ONNX / PT model support

## Tech Stack

- Python
- PySide6
- OpenCV
- Ultralytics YOLOv8
- MySQL
- TensorRT / ONNX
- Ubuntu / Jetson

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Put your model file in:

```text
models/best.engine
```

Run:

```bash
python main.py
```

## Environment Variables

This GitHub version does not include private database credentials or company paths. Use `.env.example` as a guide and set your own environment variables before running with MySQL enabled.

## Note

This is a portfolio/demo version. Private database credentials, model files, and company production details are not included.
