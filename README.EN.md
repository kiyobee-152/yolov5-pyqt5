# YOLOv5-PyQt5: Deep Learning Belt Conveyor Anchor Detection System

[中文版](README.md) | **English**

## Project Overview

This is a deep learning-based belt conveyor anchor rod detection system built with YOLOv5 and PyQt5. It provides complete functionality including video stream processing, universal model interface, interactive GUI, and post-processing features.

## Main Features

### 1. Video Stream Processing Module
- **Video File Support**: Supports multiple video formats (MP4, AVI, FLV, etc.)
- **Real-time Camera Detection**: Supports USB camera real-time video stream detection
- **Image Preprocessing**: 
  - Resolution adjustment
  - Frame rate control
  - Image enhancement (brightness, contrast, saturation adjustment)

### 2. Universal Model Interface
- **Multiple Model Format Support**: 
  - ONNX models (currently implemented)
  - PyTorch models (interface reserved for future expansion)
- **Dynamic Model Loading**: Supports runtime model switching
- **Adjustable Parameters**: Supports dynamic adjustment of confidence and IOU thresholds

### 3. Interactive Software Interface
- **Data Input**:
  - Image file detection
  - Video file detection
  - Real-time camera detection
- **Detection Results Display**:
  - Real-time video display
  - Bounding boxes and label rendering
  - Detection statistics (current frame and cumulative statistics)
- **Post-processing Operations**:
  - Save current detection image
  - Export detection reports (TXT format)
  - Export CSV data
  - Export JSON data
  - Clear detection history

### 4. Post-processing Features
- **Result Saving**: Automatically records all detection results
- **Statistical Information**: Real-time statistics of detections per category
- **Report Export**: Supports multiple export formats (TXT, CSV, JSON)
- **History Records**: Saves detection timestamps, frame IDs, positions, and other information

## System Architecture

```
Belt Conveyor Anchor Detection System
├── main.py                 # Main program, GUI interface
├── model_interface.py      # Universal model interface module
├── video_processor.py      # Video preprocessing module
├── post_processor.py       # Post-processing module
├── Yolov5OnnxruntimeDet.py # YOLOv5 ONNX detector (legacy code compatibility)
├── yolov5_utils.py         # YOLOv5 utility functions
└── weights/                # Model weights directory
    ├── yolov5s.onnx       # ONNX model file
    └── class_names.txt     # Class names file
```

## Installation

### Dependencies

```bash
pip install PyQt5
pip install opencv-python
pip install numpy
pip install torch
pip install onnxruntime
pip install torchvision
```

Or install all at once:

```bash
pip install -r yolov5-pyqt5/requirements.txt
```

## Usage

### 1. Prepare Model Files
Place your trained ONNX model file in the `weights/` directory and ensure the `class_names.txt` file exists.

### 2. Run the Program
```bash
cd yolov5-pyqt5
python main.py
```

### 3. Usage Steps
1. **Select Model**: Choose the model to use from the dropdown on the left side of the interface
2. **Adjust Parameters**: 
   - Adjust confidence threshold (0.0-1.0)
   - Adjust IOU threshold (0.0-1.0)
3. **Start Detection**:
   - Click "Select Image" for image detection
   - Click "Select Video" for video file detection
   - Click "Open Camera" for real-time detection
4. **View Results**: Check detection results in the statistics area on the right side
5. **Export Results**: 
   - Click "Save Current Image" to save the current detection image
   - Click "Export Report" to export TXT format report
   - Click "Export CSV" to export CSV format data
   - Use menu bar "File" -> "Export JSON" to export JSON format data

## Technical Features

1. **Modular Design**: Adopts modular architecture for easy extension and maintenance
2. **Universal Interface**: Provides unified model interface supporting multiple model formats
3. **Real-time Processing**: Supports real-time video stream detection with controllable frame rate
4. **Complete Records**: Automatically records all detection results, supports multiple export formats
5. **User-friendly**: Intuitive GUI interface with simple operation

## Extension Guide

### Adding New Model Support
1. Inherit the `BaseDetector` class in `model_interface.py`
2. Implement `load_model()` and `inference_image()` methods
3. Add model type judgment in the `create_detector()` function

### Custom Preprocessing
Add custom preprocessing logic in the `process_frame()` method of `video_processor.py`.

### Custom Post-processing
Add new export formats or processing features in `post_processor.py`.

## Notes

1. Ensure the model file format is correct and the class names file matches the model
2. The first run will automatically load the model, which may take some time
3. Confidence and IOU thresholds can be dynamically adjusted during detection
4. Export functionality requires prior detection - there must be detection records to export

## System Requirements

- Python 3.7+
- Windows/Linux/MacOS
- CUDA-capable GPU (optional, for accelerated inference)

## License

This project is for learning and research purposes only.
