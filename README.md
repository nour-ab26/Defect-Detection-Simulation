# 🛠️ Defect Detection Simulation using YOLOv8

This repository contains a simple simulation of an AI-based defect detection pipeline using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). 
The goal is to simulate how such a system might work on embedded or edge AI devices.

## 📁 What's in this folder

- `defect_simulation.py`: The main script simulating defect detection on a static image.
- `GUI_simulation.py`: A graphical user interface version of the simulation for interactive use.
- `yolov8s.pt`: The pre-trained YOLOv8 small model used for object/defect detection.
- `README.md`: This file, describing the project.
- `defect_detected_output.jpg`: The output image saved after running the detection script, showing detected objects.

## 🚀 What the scripts do

### defect_simulation.py
- Loads an input image (simulated camera feed).
- Runs object detection using YOLOv8.
- Simulates a defect detection decision based on detections.
- Saves the image with detection boxes drawn and prints status messages.

### GUI_simulation.py
- Provides an interactive GUI to load images, run detection, and view results in real-time.
- Useful for demo purposes or quick testing without running command-line scripts.

## ⚙️ How to run

1. Install dependencies:
   ```bash
   pip install ultralytics opencv-python
2. Update the image path in defect_simulation.py or use the GUI to select images.
3. Run the simulation script:
   ```bash
   python defect_simulation.py
4. Or run the GUI version:
   ```bash
   python GUI_simulation.py
5. Check the output image (defect_detected_output.jpg) or the GUI window to see detection results.

## 🔄 Model Info
- The detection model is yolov8s.pt (small YOLOv8).

- You can swap in other YOLOv8 models (like yolov8n.pt for faster, smaller inference).

- Custom trained models can be used by replacing the .pt file path in the scripts.

## 🧠 Why I made this
This project is a demo to better understand and simulate how AI-driven defect detection can be implemented, especially with embedded/edge AI in mind. It highlights core concepts such as inference time, decision logic, and output signaling, and includes notes about embedded deployment considerations.

## 📌 Use Cases
- Learning and experimentation with object detection and AI pipelines.

- Prototype defect detection workflows before real hardware deployment.

- Interactive demos using the GUI for quick testing.




