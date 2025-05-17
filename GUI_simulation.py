import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk # Need Pillow: pip install Pillow
import cv2
import time
from ultralytics import YOLO # Need ultralytics: pip install ultralytics
import numpy as np

# --- Global Variables ---
# Slide 8, 12: Load AI Model
# Using a small YOLOv8 model (yolov8s.pt)
# Load the model once when the app starts
try:
    print("Loading YOLOv8 model... This may take a moment the first time.")
    model = YOLO('yolov8s.pt')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None # Handle case where model loading fails

selected_image_path = None
# Max dimensions for displaying images in the GUI
MAX_DISPLAY_SIZE = (600, 600)

# --- Helper Function to Resize Image for Display ---
def resize_image_for_display(img_cv2, max_size):
    """Resizes an OpenCV image (NumPy array) to fit within max_size while maintaining aspect ratio."""
    h, w = img_cv2.shape[:2]
    max_w, max_h = max_size

    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_cv2

    return img_resized

# --- Function to Display Image in Tkinter Label ---
def display_image_on_label(img_cv2, label):
    """Converts OpenCV image to PhotoImage and displays it on a Tkinter Label."""
    if img_cv2 is None:
        label.config(image=None)
        label.image = None
        return

    # Convert BGR (OpenCV) to RGB (PIL)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Resize for display
    img_pil_resized = resize_image_for_display(np.array(img_pil), MAX_DISPLAY_SIZE)
    img_pil_resized = Image.fromarray(cv2.cvtColor(img_pil_resized, cv2.COLOR_BGR2RGB)) # Need RGB PIL again

    # Convert PIL image to Tkinter PhotoImage
    img_tk = ImageTk.PhotoImage(image=img_pil_resized)

    # Update the label
    label.config(image=img_tk)
    label.image = img_tk # Keep a reference to prevent garbage collection

# --- GUI Functions ---

# Slide 7: Simulate Image Acquisition (Loading File)
def select_image():
    global selected_image_path
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*"))
    )
    if file_path:
        selected_image_path = file_path
        lbl_file_path.config(text=f"Selected: {file_path}")
        lbl_status.config(text="Image selected. Click 'Run Detection'.")
        lbl_inference_time.config(text="")

        # Display the selected input image
        img_input = cv2.imread(selected_image_path)
        if img_input is not None:
            display_image_on_label(img_input, lbl_image_display)
        else:
            messagebox.showerror("Error", f"Could not load image from {file_path}")
            lbl_image_display.config(image=None) # Clear display
            lbl_image_display.image = None
            selected_image_path = None


# Slide 7, 8, 15: AI Pipeline (Preprocessing, Inference, Decision, Output)
def run_detection():
    if model is None:
        messagebox.showerror("Error", "AI model failed to load. Cannot run detection.")
        return

    if not selected_image_path:
        messagebox.showwarning("Warning", "Please select an image first.")
        return

    lbl_status.config(text="Processing...")
    lbl_inference_time.config(text="")
    root.update() # Update the GUI to show status change

    try:
        # --- Pipeline Step 1: Image Acquisition (Loaded previously) ---
        img_cv2 = cv2.imread(selected_image_path)
        if img_cv2 is None:
            messagebox.showerror("Error", f"Could not reload image from {selected_image_path}")
            lbl_status.config(text="Error loading image.")
            display_image_on_label(None, lbl_image_display) # Clear display
            return

        # --- Pipeline Step 2: Preprocessing (Handled internally by YOLO) ---
        # YOLO expects RGB
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        # Resizing/Normalization are handled when passing to model

        # --- Pipeline Step 3: Inference (Slide 7, 8) ---
        start_time = time.time()
        # Passing img_rgb (PIL Image or numpy array)
        results = model(img_rgb, verbose=False)
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000

        # --- Embedded Angle Discussion ---
        lbl_inference_time.config(text=f"Inference Time: {inference_time_ms:.2f} ms (Edge Speed - Slide 9)")
        # Discuss how this time relates to embedded performance (Slide 12, 13, 15)

        # --- Pipeline Step 4 & 5: Decision & Postprocessing (Slide 7, 8) ---
        defect_detected = False
        img_output_cv2 = img_cv2.copy() # Start with a copy of the original image

        if results and results[0].boxes: # Check if any results and bounding boxes exist
            detections = results[0].boxes.xyxy # Get box coordinates
            confidences = results[0].boxes.conf # Get confidence scores
            class_ids = results[0].boxes.cls # Get class IDs
            # class_names = model.names # Get class names from the model

            print(f"Detected {len(detections)} objects.")

            # --- Simulate Decision: Check if *any* object was detected ---
            # In a real project, you'd check `class_ids` against your specific defect class list
            # and filter by `confidences` against a higher threshold for defects.
            if len(detections) > 0:
                 defect_detected = True
                 # In a real system: Check if detected class_id is one of your defect classes
                 # Example: if any(model.names[int(cls_id)] in ['crack', 'solder_issue'] for cls_id in class_ids):
                 #    defect_detected = True
                 print("Decision: Potential issue detected based on model output (any object found).")


            # Draw results on the output image copy
            # YOLOv8 .plot() method is convenient for this
            img_output_cv2 = results[0].plot()


        # --- Pipeline Step 6: Output (Visualized) (Slide 7, 15) ---
        if defect_detected:
            status_text = "Result: DEFECT DETECTED!" # Simulating output signal
            lbl_status.config(text=status_text, fg="red") # Use red color for defect
            print(status_text)
        else:
            status_text = "Result: No Defect Detected."
            lbl_status.config(text=status_text, fg="green") # Use green color
            print(status_text)

        # Display the output image with detections
        display_image_on_label(img_output_cv2, lbl_image_display)

    except Exception as e:
        messagebox.showerror("Processing Error", f"An error occurred during detection: {e}")
        lbl_status.config(text="Error during detection.", fg="black")
        lbl_inference_time.config(text="")
        # Optionally clear the image display on error
        # lbl_image_display.config(image=None)
        # lbl_image_display.image = None


# --- Set up the main GUI window ---
root = tk.Tk()
root.title("Embedded AI Defect Detection Simulation")

# Use grid layout
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(5, weight=1) # Row where the image is displayed

# --- GUI Widgets ---

# Frame for controls on the left
frame_controls = tk.Frame(root)
frame_controls.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

lbl_title = tk.Label(frame_controls, text="AI Defect Detection Simulation", font=("Helvetica", 14, "bold"))
lbl_title.pack(pady=(0, 10))

btn_select_image = tk.Button(frame_controls, text="Select Image File (Simulate Acquisition)", command=select_image)
btn_select_image.pack(pady=5)

lbl_file_path = tk.Label(frame_controls, text="No image selected.", wraplength=300)
lbl_file_path.pack(pady=5)

btn_run_detection = tk.Button(frame_controls, text="Run Defect Detection", command=run_detection)
btn_run_detection.pack(pady=10)

lbl_status_label = tk.Label(frame_controls, text="Status:")
lbl_status_label.pack(pady=(10,0))
lbl_status = tk.Label(frame_controls, text="Waiting for image selection...", font=("Helvetica", 10, "bold"))
lbl_status.pack(pady=(0, 5))

lbl_time_label = tk.Label(frame_controls, text="Inference Time:")
lbl_time_label.pack(pady=(10,0))
lbl_inference_time = tk.Label(frame_controls, text="", font=("Helvetica", 10))
lbl_inference_time.pack(pady=(0, 5))

# Placeholder for Embedded Angle Discussion (Text labels)
lbl_embedded_angles = tk.Label(frame_controls, text="Embedded Considerations (Slide 9-15):\n- Real-Time (Check time above)\n- Local Processing (Secure)\n- Low Power (Target hardware dependent)\n- Optimization needed for Edge platforms (Jetson, Coral, STM32)",
                              font=("Helvetica", 9), justify="left", wraplength=300)
lbl_embedded_angles.pack(pady=(20, 0))


# Label to display the image on the right
lbl_image_display = tk.Label(root, text="Image will appear here")
lbl_image_display.grid(row=0, column=1, rowspan=6, sticky="nsew", padx=10, pady=10) # Span multiple rows

# --- Start the GUI Event Loop ---
print("Starting GUI...")
root.mainloop()
print("GUI closed.")