import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import time
ß
# Set page configuration
st.set_page_config(
    page_title="YOLOapp - Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("YOLOapp - Real-time Object Detection")
st.markdown("Use your iPhone camera to detect objects in real-time!")

# Initialize session state for storing the model
if 'model' not in st.session_state:
    with st.spinner("Loading YOLO model... This might take a moment."):
        st.session_state.model = YOLO("yolov8n.pt")  # Using the small version of YOLOv8

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Camera Detection", "About"])

with tab1:
    # Camera input
    st.subheader("Point your camera at objects")
    img_file_buffer = st.camera_input("Take a picture to detect objects")
    
    # Display confidence threshold slider
    confidence = st.slider("Detection Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    
    if img_file_buffer is not None:
        # Convert the image buffer to an OpenCV image
        bytes_data = img_file_buffer.getvalue()
        
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(bytes_data)
            tmp_file_path = tmp_file.name
        
        # Perform prediction with YOLO
        start_time = time.time()
        results = st.session_state.model.predict(tmp_file_path, conf=confidence)
        end_time = time.time()
        
        # Process and display the result
        result = results[0]
        image = Image.open(tmp_file_path)
        
        # Convert to numpy array for OpenCV processing
        image_np = np.array(image)
        
        # Get detection information
        boxes = result.boxes
        
        # Draw bounding boxes and labels on the image
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class name and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{result.names[cls]}: {conf:.2f}"
            
            # Draw rectangle and text
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the result image
        st.image(image_np, caption="Detection Result", use_column_width=True)
        
        # Display detection information
        st.subheader("Detection Results")
        inference_time = end_time - start_time
        st.write(f"Inference Time: {inference_time:.4f} seconds")
        
        # Show what was detected
        if len(boxes) > 0:
            detected_objects = {}
            
            for box in boxes:
                cls = int(box.cls[0])
                class_name = result.names[cls]
                conf = float(box.conf[0])
                
                if class_name in detected_objects:
                    if conf > detected_objects[class_name]:
                        detected_objects[class_name] = conf
                else:
                    detected_objects[class_name] = conf
            
            # Display detected objects with confidence
            st.write("Detected Objects:")
            for obj, conf in detected_objects.items():
                st.write(f"- {obj} (Confidence: {conf:.2f})")
        else:
            st.write("No objects detected.")

with tab2:
    st.subheader("About YOLOapp")
    st.write("""
    This application uses YOLOv8, a state-of-the-art object detection model, to identify objects in real-time using your iPhone's camera.
    
    ### How to use:
    1. Allow camera access when prompted
    2. Point your camera at objects you want to identify
    3. Take a picture using the camera button
    4. View the detected objects with bounding boxes
    5. Adjust the confidence threshold slider to filter detections
    
    ### Technical Details:
    - Model: YOLOv8 (nano version)
    - Capable of detecting 80 different object classes
    - Processing happens on the server, not on your device
    """)