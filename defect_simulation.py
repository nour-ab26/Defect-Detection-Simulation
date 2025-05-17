import cv2
import time
from ultralytics import YOLO # Requires 'pip install ultralytics'

# --- Simulation Setup ---
# Slide 7: Image Acquisition (Simulated)
# Replace with path to your sample image
# Image should contain something a pre-trained YOLO might detect,
# or use an image from a small custom defect dataset.
image_path = 'path/to/your/defect_image.jpg'
output_path = 'defect_detected_output.jpg'

# Slide 8, 12: Load AI Model
# Using a small YOLOv8 model (yolov8s.pt) as it's closer to what
# might be optimized for edge. For real edge, you'd use yolov8n.pt or smaller.
# Download automatically on first run.
model = YOLO('yolov8s.pt')

# --- Embedded Angle Discussion ---
# print(f"Model loaded. Size implications for embedded device (Slide 13): {model.model.model_size / (1024*1024):.2f} MB")
# Note: .model_size is not a direct attribute. You'd typically check file size.
# Discuss that this model needs conversion/quantization for platforms like Coral/STM32 (Slide 11, 12).

try:
    # --- Pipeline Step 1: Image Acquisition (Simulated) ---
    print(f"Loading image from {image_path} (Simulating camera feed - Slide 7)...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        exit()

    # --- Pipeline Step 2: Preprocessing (Slide 7, 8) ---
    # YOLOv8 handles internal preprocessing like resizing and normalization.
    print("Preprocessing image (resizing, normalization handled by YOLO - Slide 7, 8)...")
    # Model expects RGB, OpenCV reads BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Pipeline Step 3: Inference (Slide 7, 8) ---
    print("Running AI Inference (Detecting objects/potential defects - Slide 7, 8)...")
    start_time = time.time()
    # Predict returns a list of Results objects, one for each image (here just one)
    results = model(img_rgb, verbose=False) # Set verbose=True to see more model output
    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000

    # --- Embedded Angle Discussion ---
    print(f"Inference Time: {inference_time_ms:.2f} ms")
    print(f"(Crucial for Real-Time Defect Detection on Embedded Systems - Slide 9, 15)")
    # Discuss how this time would be higher/lower depending on the specific embedded hardware (Slide 12)

    # --- Pipeline Step 4 & 5: Decision & Postprocessing (Slide 7, 8) ---
    print("Analyzing results and making decision (Slide 7, 8)...")
    defect_detected = False
    # You'd need logic here based on your *actual* defect classes if you trained a model.
    # For a generic model, we can just show general object detection.
    # If you had a custom model with class_names = ['crack', 'solder_issue', ...],
    # you would check if any detected object's class is in this list.

    # Let's just draw all detections for demonstration
    # Assuming 'defect' classes would be among the detected objects if model was trained accordingly
    # If using a generic COCO model, interpret detected objects as potential "features" or "regions of interest".
    # To truly simulate defect detection, you NEED a model trained on defects.
    # For this *simulation*, we'll just show the detections and indicate if *any* object was detected.
    if results and results[0].boxes: # Check if any results and bounding boxes exist
        detections = results[0].boxes.xyxy # Get box coordinates
        confidences = results[0].boxes.conf # Get confidence scores
        class_ids = results[0].boxes.cls # Get class IDs

        print(f"Detected {len(detections)} objects.")

        # --- Simulate Decision: Did we find something of interest (potentially a defect)? ---
        # In a real project, you'd check `class_ids` against your defect class list
        # and filter by `confidences` against a threshold.
        if len(detections) > 0:
             defect_detected = True
             print("Decision: Potential issue detected based on model output.") # Refine this message based on your model/data

        # Draw results on the image
        img_result = results[0].plot() # YOLOv8 .plot() draws detections on the image

    # --- Pipeline Step 6: Output (Simulated) (Slide 7, 15) ---
    print("\n--- Simulation Output ---")
    if defect_detected:
        print("Output Signal: DEFECT DETECTED!") # Simulating turning on a light/GPIO/flag
        # In real embedded, this would be GPIO control, message queue, etc. (Slide 15)
    else:
        print("Output Signal: No Defect Detected.")

    # Display or save the output image
    cv2.imwrite(output_path, img_result if 'img_result' in locals() else img) # Save processed image or original if no detections
    print(f"Output image saved to {output_path} (Visualizing detection - Slide 15).")

    # Optional: Display image (might not work in all environments, requires GUI)
    # cv2.imshow("Defect Detection Simulation", img_result if 'img_result' in locals() else img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


except Exception as e:
    print(f"An error occurred during the simulation: {e}")
    print("Please ensure you have installed ultralytics (`pip install ultralytics`) and have a sample image.")


# --- Embedded Angle Summary Discussion ---
print("\n--- Embedded Considerations ---")
print("This simulation ran on a PC. For embedded deployment (Slide 12), consider:")
print("- Model Optimization (Quantization, Pruning) for smaller size and faster inference on limited hardware (Slide 11, 13).")
print("- Using frameworks like TensorFlow Lite or optimized PyTorch Mobile (or platform-specific SDKs) (Slide 11).")
print("- The actual inference time on the target hardware (Jetson, Coral, STM32) (Slide 12, 15).")
print("- Interfacing with real camera hardware and output peripherals (GPIO for flags, communication for logging) (Slide 7, 15).")
print("- Power consumption (Slide 9).")
print("- Data handling and storage on the device (Slide 9).")
print("- Model updates and maintenance in deployed systems (Slide 13).")
print("This demo highlights the core AI inference step which is central to the pipeline running on the edge (Slide 9, 15).")