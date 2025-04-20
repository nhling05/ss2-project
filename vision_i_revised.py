#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-I: A Vision Enhancement System for Visually Impaired Users (YOLOv8 Version)
"""

import numpy as np
import os
import cv2
import pyttsx3
import time
from PIL import Image
from ultralytics import YOLO

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Configure voice properties
try:
    voices = engine.getProperty('voices')
    # Set to English voice
    for voice in voices:
        if 'english' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    # Adjust speaking rate
    engine.setProperty('rate', 150)
except Exception as e:
    print(f"Voice configuration error: {e}")

# Load YOLOv8 model
def load_model():
    """Load the YOLOv8 model for object detection"""
    try:
        # Load YOLOv8n model (smallest and fastest version)
        print("Loading YOLO model...")
        model = YOLO('yolov8n.pt')
        print("YOLO model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Calculate distance based on bounding box size
def calculate_distance(box):
    """Calculate approximate distance based on bounding box size"""
    x1, y1, x2, y2 = box
    # Based on the width of the bounding box
    width = abs(x2 - x1)
    normalized_width = width / 640  # Assuming frame width is 640
    # Formula similar to original code
    return round(((1 - normalized_width) ** 4), 1)

# Provide voice feedback based on distance and object class
def speak_feedback(object_name, distance, is_close=False):
    """Generate voice feedback based on detected object and distance"""
    if is_close:
        engine.say(f"Warning! {object_name} is very close at approximately {distance} units")
    else:
        engine.say(f"Detected {object_name} at approximately {distance} units")
    engine.runAndWait()

# Function to detect dollar bills
def detect_dollar_bills(image):
    """Detect and identify dollar bills in the image"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_bills = []
        
        # Process each contour
        for contour in contours:
            # Calculate area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter small contours (noise)
            if area < 10000:  # Adjust based on expected bill size
                continue
                
            # Approximate contour to polygon
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Dollar bills are rectangles (4 points)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check aspect ratio (dollar bills have ~2.4:1 ratio)
                aspect_ratio = float(w) / h
                if 1.9 < aspect_ratio < 2.7:  # Allow some tolerance
                    potential_bills.append((x, y, w, h))
                    
        if potential_bills:
            # Process each potential bill
            for (x, y, w, h) in potential_bills:
                bill_img = image[y:y+h, x:x+w]
                
                # Calculate dominant colors (for denomination recognition)
                bill_img_rgb = cv2.cvtColor(bill_img, cv2.COLOR_BGR2RGB)
                bill_img_rgb = cv2.resize(bill_img_rgb, (100, 40))  # Resize for faster processing
                
                # Reshape the image to be a list of pixels
                pixels = bill_img_rgb.reshape((-1, 3))
                pixels = np.float32(pixels)
                
                # Define criteria and apply kmeans
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
                # Convert back to uint8
                centers = np.uint8(centers)
                
                # Count percentage of each color
                color_percentages = []
                for i in range(5):
                    color_count = np.count_nonzero(labels == i)
                    color_percentages.append(color_count / labels.size)
                
                # Draw rectangle around the bill on display image
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Simple color-based denomination recognition
                # This is a very basic approach - for production you'd need a more sophisticated model
                # Check green intensity (dominant in all bills)
                green_intensity = np.mean(bill_img[:, :, 1])
                
                # Check for color signatures
                # These thresholds are approximate and would need calibration
                # Different denominations have different color tints
                if np.mean(bill_img[:, :, 0]) > 150:  # High blue
                    denomination = "hundred dollar"
                elif np.mean(bill_img[:, :, 2]) > 140:  # High red
                    denomination = "fifty dollar"
                elif green_intensity > 110:
                    denomination = "twenty dollar"
                elif green_intensity > 100:
                    denomination = "ten dollar"
                elif green_intensity > 90:
                    denomination = "five dollar"
                else:
                    denomination = "one dollar"
                
                # Label the bill
                cv2.putText(
                    image, 
                    f'{denomination} bill', 
                    (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
                
                # Announce detection
                engine.say(f"Detected {denomination} bill")
                engine.runAndWait()
                
                return denomination
        
        else:
            engine.say("No dollar bills detected")
            engine.runAndWait()
            return None
            
    except Exception as e:
        print(f"Dollar bill detection error: {e}")
        engine.say("Error in dollar bill detection")
        engine.runAndWait()
        return None

# Main function to run the Vision-I system
def run_vision_system():
    """Main function that runs the Vision-I system"""
    # Setup
    model = load_model()
    if model is None:
        print("Could not load model. Exiting...")
        return
    
    # Start video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    
    # Check if camera is working
    if not cap.isOpened():
        print("Cannot open camera. Please check webcam connection.")
        return
    
    # Welcome message
    engine.say("Vision-I system is starting. Press 'r' for dollar bill detection, 'q' to quit, 'd' to detect and read objects.")
    engine.runAndWait()
    
    print("Vision-I system is running. Press 'q' to quit, 'r' for dollar bill detection, 'd' to detect and read objects.")
    
    # Time of last alert
    last_alert_time = 0
    alert_cooldown = 3  # Time between alerts (seconds)
    confidence_threshold = 0.45  # Confidence threshold
    
    # Tracking objects to avoid repeated alerts
    detected_objects = {}
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from camera. Exiting...")
            break
        
        # Clone frame for display
        display_frame = frame.copy()
        
        # Run detection with larger image size to improve accuracy
        results = model(frame, imgsz=640, conf=confidence_threshold)
        
        # Process results
        detected_count = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence
                confidence = float(box.conf[0])
                
                # Get object type
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Draw bounding box and object name
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display object name and confidence
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(
                    display_frame, 
                    label, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
                
                # Calculate center point and distance
                mid_x = (x1 + x2) / 2 / frame.shape[1]  # Normalize
                mid_y = (y1 + y2) / 2 / frame.shape[0]  # Normalize
                distance = calculate_distance([x1, y1, x2, y2])
                
                # Display distance on screen
                cv2.putText(
                    display_frame, 
                    f'Dist: {distance}', 
                    (int((x1 + x2)/2), int((y1 + y2)/2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    2
                )
                
                # Count detected objects
                detected_count += 1
                
                # Check if object is too close (in the center and near)
                is_close = distance <= 0.5 and (0.3 < mid_x < 0.7)
                current_time = time.time()
                
                # Create ID for object based on position and name
                object_id = f"{class_name}_{int(mid_x*100)}_{int(mid_y*100)}"
                
                # Check if this object has been alerted recently
                if is_close and (current_time - last_alert_time) > alert_cooldown:
                    if object_id not in detected_objects or (current_time - detected_objects.get(object_id, 0)) > alert_cooldown:
                        cv2.putText(
                            display_frame, 
                            'WARNING!!!', 
                            (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, 
                            (0, 0, 255), 
                            3
                        )
                        # Add debug print to check
                        print(f"Alert: {class_name} is close at distance {distance}")
                        
                        # Update last alert time
                        last_alert_time = current_time
                        detected_objects[object_id] = current_time
                        
                        # Play alert sound
                        speak_feedback(class_name, distance, is_close=True)
        
        # Display number of detected objects
        cv2.putText(
            display_frame, 
            f'Detected: {detected_count} objects', 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Display processed frame
        cv2.imshow('Vision-I: Visual Assistance System', cv2.resize(display_frame, (800, 600)))
        
        # Check for keyboard commands
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('r'):  # Press 'r' for dollar bill detection
            detect_dollar_bills(frame.copy())
        elif key == ord('d'):  # Press 'd' to detect objects and read
            # Read all detected objects
            if detected_count > 0:
                engine.say(f"Detected {detected_count} objects in the frame")
                
                # Create list of objects to read
                objects_to_read = []
                for result in results:
                    for box in result.boxes:
                        if float(box.conf[0]) >= confidence_threshold:
                            class_name = result.names[int(box.cls[0])]
                            objects_to_read.append(class_name)
                
                if objects_to_read:
                    # Create a unique set to avoid duplicates
                    unique_objects = list(set(objects_to_read))
                    objects_text = ", ".join(unique_objects)
                    engine.say(f"Objects are: {objects_text}")
                
                engine.runAndWait()
            else:
                engine.say("No objects detected")
                engine.runAndWait()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    engine.say("Vision-I system shutting down. Goodbye.")
    engine.runAndWait()

# Run the system if this script is executed directly
if __name__ == "__main__":
    run_vision_system()