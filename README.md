# PPE Detection System - Setup Instructions

## Requirements
- Python 3.9+
- Anaconda (recommended)
- Webcam

## Setup Steps

### 1. Copy your model
Put your `best.pt` file in this folder (PPE_Detection_App/).

### 2. Install dependencies
Open Anaconda Prompt and run:
```
cd path\to\PPE_Detection_App
pip install flask ultralytics opencv-python numpy
```

### 3. Run the app
```
python app.py
```

### 4. Open browser
Go to: http://localhost:5000

## Features
- Live webcam detection
- Upload and process video files
- SAFE / VIOLATION indicator
- Live class detection counters
- Violations over time chart
- Detections per class bar chart
- FPS and inference time display
- Auto screenshot on violation (saved to screenshots/ folder)
- Reset stats button

## Folder Structure
```
PPE_Detection_App/
  app.py              - Flask backend
  best.pt             - Your trained YOLOv8s model (put here!)
  requirements.txt    - Python dependencies
  templates/
    index.html        - Dashboard frontend
  screenshots/        - Auto-saved violation screenshots
  uploads/            - Uploaded videos
```
