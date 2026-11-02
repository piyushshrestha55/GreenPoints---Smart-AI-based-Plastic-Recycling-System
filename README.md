
# GreenPoints - Smart AI-Based Plastic Recycling System

GreenPoints is a student project that explores how computer vision can support plastic bottle recycling through an AI-based recycling platform.

The current prototype uses a webcam to inspect objects placed on a scanning platform, detect bottles, and pause scanning when a hand is detected in the platform area.

## Features

- Real-time webcam feed using OpenCV.
- Bottle detection using the Ultralytics YOLO model.
- Minimum bottle confidence threshold of 80%.
- Rejection of multiple detected bottles.
- Rejection of other detected objects on the scanning platform.
- Hand detection using MediaPipe.
- Pauses scanning when a hand enters the defined scanning area.
- Point counting and session timer.
- Simulated trapdoor opening and closing.
- Waits for the platform to become empty before scanning the next item.

## Technologies Used

- Python
- Ultralytics YOLO
- OpenCV
- MediaPipe
- NumPy

## Requirements

- Python 3.12 recommended for the current development setup.
- A working webcam.
- Internet access for the initial installation and model download.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/piyushshrestha55/GreenPoints---Smart-AI-based-Plastic-Recycling-System.git
cd GreenPoints---Smart-AI-based-Plastic-Recycling-System
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Run the prototype

If the main Python file is named `greenpoints.py`:

```powershell
.\.venv\Scripts\python.exe greenpoints.py
```

If the file has a different name, replace `greenpoints.py` with the actual filename.

Press `Q` to close the webcam window.

## Model

The prototype uses the pretrained YOLO model `yolo26n.pt`.

Ultralytics can download the model automatically when it is not present locally, provided internet access is available.

## Current Limitations

- The standard YOLO bottle class does not reliably distinguish plastic bottles from glass bottles.
- Hand detection may miss hands because of camera angle, lighting, or occlusion.
- Bottle confidence alone does not establish that a bottle is empty.
- Water-level verification is not yet implemented.
- Trapdoor operation is simulated in software; physical motor control is not implemented.
- Successful physical deposit is not independently verified.

This project is a prototype and requires further testing before use with physical recycling hardware.

## Future Improvements

- Train a custom model for plastic bottle verification.
- Explore empty and partially filled bottle classification.
- Improve hand detection and scanning-area calibration.
- Integrate an ESP32 and a physical trapdoor.
- Add deposit verification and a persistent points database.
- Develop a user interface for recycling sessions and points.

## Project Status

Student prototype under development.
