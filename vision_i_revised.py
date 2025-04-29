#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-I: Cải tiến Hệ thống Hỗ trợ Thị lực cho Người Khiếm thị (YOLOv8 Version)
Phiên bản tự động không cần PyAudio với cải tiến hiệu suất
"""

import numpy as np
import os
import cv2
import pyttsx3
import time
from threading import Thread
import webbrowser
from PIL import Image
from ultralytics import YOLO
import sys
import socket
import re
import keyboard
from droidcam_connector import get_droidcam_capture

# Hằng số hệ thống
CONFIDENCE_THRESHOLD = 0.65
ALERT_COOLDOWN = 6
ANNOUNCEMENT_COOLDOWN = 12
MAX_OBJECTS_TO_READ = 2
CENTER_ZONE = 0.25
FRAME_SKIP = 2
MAX_WARNINGS_PER_OBJECT = 2  # New constant for max warnings per object

# Khởi tạo bộ đọc giọng nói
engine = pyttsx3.init()

# Cấu hình thuộc tính giọng nói
try:
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'english' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.setProperty('rate', 150)
except Exception as e:
    print(f"Lỗi cấu hình giọng nói: {e}")

# Hàm đọc thông báo
def speak(text, priority=False):
    global is_speaking
    if is_speaking and not priority:
        return
    try:
        is_speaking = True
        print(f"Đọc: {text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Lỗi đọc thông báo: {e}")
    finally:
        is_speaking = False

# Khởi tạo biến theo dõi trạng thái hệ thống
is_speaking = False
is_running = True
last_speech_time = 0
last_announcement_time = 0
last_dollar_detection_time = 0
last_space_press_time = 0
space_press_count = 0
system_mode = 0  # 0: Phát hiện vật thể, 1: Phát hiện tiền
already_announced_objects = set()
center_objects_history = {}
warning_counts = {}  # Track number of warnings per object
suppressed_objects = set()  # Objects with suppressed warnings
frame_count = 0

# Tải mô hình YOLOv8
def load_model():
    try:
        print("Đang tải mô hình YOLO...")
        model = YOLO('yolov8n.pt')
        print("Đã tải mô hình YOLO thành công")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        speak("Error loading model. Please check your internet connection or model installation.")
        return None

# Tính khoảng cách dựa trên kích thước hộp giới hạn
def calculate_distance(box):
    x1, y1, x2, y2 = box
    width = abs(x2 - x1)
    normalized_width = width / 640
    if normalized_width > 0.6:
        return "very close"
    elif normalized_width > 0.4:
        return "close"
    elif normalized_width > 0.25:
        return "moderate distance"
    elif normalized_width > 0.15:
        return "far"
    else:
        return "very far"

# Phát hiện tiền giấy
def detect_dollar_bills(image):
    global last_dollar_detection_time
    current_time = time.time()
    if current_time - last_dollar_detection_time < 3:
        return None
    last_dollar_detection_time = current_time
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        potential_bills = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < 10000:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 1.9 < aspect_ratio < 2.7:
                    potential_bills.append((x, y, w, h))
        if potential_bills:
            for (x, y, w, h) in potential_bills:
                bill_img = image[y:y+h, x:x+w]
                bill_img_rgb = cv2.cvtColor(bill_img, cv2.COLOR_BGR2RGB)
                bill_img_rgb = cv2.resize(bill_img_rgb, (100, 40))
                pixels = bill_img_rgb.reshape((-1, 3))
                pixels = np.float32(pixels)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                color_percentages = []
                for i in range(5):
                    color_count = np.count_nonzero(labels == i)
                    color_percentages.append(color_count / labels.size)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                green_intensity = np.mean(bill_img[:, :, 1])
                if np.mean(bill_img[:, :, 0]) > 150:
                    denomination = "hundred dollar"
                elif np.mean(bill_img[:, :, 2]) > 140:
                    denomination = "fifty dollar"
                elif green_intensity > 110:
                    denomination = "twenty dollar"
                elif green_intensity > 100:
                    denomination = "ten dollar"
                elif green_intensity > 90:
                    denomination = "five dollar"
                else:
                    denomination = "one dollar"
                cv2.putText(
                    image, 
                    f'{denomination} bill', 
                    (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
                speak(f"Detected {denomination} bill", priority=True)
                return denomination
        return None
    except Exception as e:
        print(f"Lỗi phát hiện tiền giấy: {e}")
        return None

def find_droidcam():
    return get_droidcam_capture()

# Thông báo các vật thể được phát hiện
def announce_objects(objects=None, force=False):
    global already_announced_objects, last_announcement_time
    current_time = time.time()
    if not force and (current_time - last_announcement_time) < ANNOUNCEMENT_COOLDOWN:
        return
    if objects and len(objects) > 0:
        if force:
            objects_to_announce = objects
        else:
            objects_to_announce = [obj for obj in objects if obj not in already_announced_objects]
        if len(objects_to_announce) > MAX_OBJECTS_TO_READ:
            objects_to_announce = objects_to_announce[:MAX_OBJECTS_TO_READ]
        if objects_to_announce:
            already_announced_objects.update(objects_to_announce)
            if len(objects_to_announce) == 1:
                speak(f"Detected {objects_to_announce[0]}")
            else:
                objects_text = ", ".join(objects_to_announce)
                speak(f"Detected {objects_text}")
            last_announcement_time = current_time
            if len(already_announced_objects) > 15:
                already_announced_objects.clear()

# Phát hiện và thông báo vật thể ở trung tâm
def detect_center_objects(objects_with_data):
    global center_objects_history, last_speech_time, warning_counts, suppressed_objects
    current_time = time.time()
    objects_in_center = []
    center_threshold = CENTER_ZONE
    for obj_name, distance, mid_x, mid_y, confidence in objects_with_data:
        if abs(mid_x - 0.5) < center_threshold and abs(mid_y - 0.5) < center_threshold:
            objects_in_center.append((obj_name, distance, confidence))
    if objects_in_center:
        objects_in_center.sort(key=lambda x: x[2], reverse=True)
        top_object = objects_in_center[0]
        obj_name, distance, _ = top_object
        if obj_name in suppressed_objects:
            return
        warning_count = warning_counts.get(obj_name, 0)
        if warning_count >= MAX_WARNINGS_PER_OBJECT:
            return
        last_time = center_objects_history.get(obj_name, 0)
        if (current_time - last_time) > ALERT_COOLDOWN:
            message = f"{obj_name} is {distance} in front of you"
            speak(message)
            center_objects_history[obj_name] = current_time
            last_speech_time = current_time
            warning_counts[obj_name] = warning_count + 1

# Tạo desktop shortcut
def create_desktop_shortcut():
    try:
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        shortcut_path = os.path.join(desktop, 'Vision-I.bat')
        script_path = os.path.abspath(__file__)
        with open(shortcut_path, 'w') as f:
            f.write('@echo off\n')
            f.write(f'python "{script_path}"\n')
            f.write('pause\n')
        print(f"Đã tạo lối tắt tại: {shortcut_path}")
        speak("Desktop shortcut created successfully", priority=True)
    except Exception as e:
        print(f"Lỗi khi tạo lối tắt: {e}")
        speak("Error creating desktop shortcut", priority=True)

# Theo dõi phím bấm từ bàn phím
def keyboard_listener():
    global is_running, system_mode, last_space_press_time, space_press_count, suppressed_objects
    speak("Press spacebar once to change modes, twice to suppress current object warnings. Press escape to exit. Press R to repeat detections.", priority=True)
    while is_running:
        try:
            if keyboard.is_pressed('esc'):
                print("Phím ESC được nhấn. Đang thoát...")
                is_running = False
                speak("Shutting down the system", priority=True)
                time.sleep(1)
                break
            if keyboard.is_pressed('space'):
                current_time = time.time()
                if (current_time - last_space_press_time) < 0.5:
                    space_press_count += 1
                else:
                    space_press_count = 1
                last_space_press_time = current_time
                if space_press_count == 2:
                    # Suppress warnings for the most recently warned object
                    if center_objects_history:
                        recent_object = max(center_objects_history, key=center_objects_history.get)
                        suppressed_objects.add(recent_object)
                        speak(f"Warnings for {recent_object} suppressed", priority=True)
                    space_press_count = 0  # Reset after double press
                elif space_press_count == 1:
                    # Switch modes
                    system_mode = (system_mode + 1) % 2
                    if system_mode == 0:
                        speak("Object detection mode activated", priority=True)
                    elif system_mode == 1:
                        speak("Dollar detection mode activated", priority=True)
                time.sleep(0.3)
            if keyboard.is_pressed('r'):
                speak("Repeating the last detections", priority=True)
                announce_objects(force=True)
                time.sleep(0.5)
            time.sleep(0.1)
        except Exception as e:
            print(f"Lỗi trong keyboard_listener: {e}")
            time.sleep(0.1)

# Hàm chính để chạy hệ thống Vision-I
def run_vision_system():
    global is_running, system_mode, last_speech_time, frame_count
    model = load_model()
    if model is None:
        print("Không thể tải mô hình. Đang thoát...")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--create-shortcut":
        create_desktop_shortcut()
        return
    cap = find_droidcam()
    if not cap.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra kết nối webcam hoặc DroidCam.")
        speak("Cannot open camera. Please check camera connection.", priority=True)
        return
    speak("Vision-I system is starting. Press spacebar once to change modes, twice to suppress object warnings. Press escape to exit.", priority=True)
    print("Hệ thống Vision-I đang chạy. Sử dụng phím Space để điều khiển, phím ESC để thoát.")
    keyboard_thread = Thread(target=keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    last_alert_time = 0
    detected_objects = {}
    recent_objects = []
    while is_running:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Không thể nhận khung hình từ camera. Đang thử lại...")
                time.sleep(1)
                continue
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                cv2.imshow('Vision-I: Hệ thống Hỗ trợ Thị lực', cv2.resize(frame, (800, 600)))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            display_frame = frame.copy()
            if system_mode == 1:
                detect_dollar_bills(display_frame)
                cv2.putText(
                    display_frame, 
                    'Dollar detection mode', 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 255), 
                    2
                )
                cv2.imshow('Vision-I: Hệ thống Hỗ trợ Thị lực', cv2.resize(display_frame, (800, 600)))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            process_frame = cv2.resize(frame, (640, 480))
            results = model(process_frame, imgsz=640, conf=CONFIDENCE_THRESHOLD)
            detected_count = 0
            detected_names = []
            objects_with_distances = []
            objects_with_data = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    scale_x = frame.shape[1] / 640
                    scale_y = frame.shape[0] / 480
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    if class_name in ["cell phone", "remote", "mouse", "keyboard", "book"]:
                        if (x2 - x1) * (y2 - y1) < (frame.shape[0] * frame.shape[1]) / 20:
                            continue
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
                    mid_x = (x1 + x2) / 2 / frame.shape[1]
                    mid_y = (y1 + y2) / 2 / frame.shape[0]
                    distance_desc = calculate_distance([x1, y1, x2, y2])
                    cv2.putText(
                        display_frame, 
                        f'Dist: {distance_desc}', 
                        (int((x1 + x2)/2), int((y1 + y2)/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        2
                    )
                    detected_count += 1
                    detected_names.append(class_name)
                    objects_with_distances.append((class_name, distance_desc))
                    objects_with_data.append((class_name, distance_desc, mid_x, mid_y, confidence))
                    is_close = ("close" in distance_desc) and (0.3 < mid_x < 0.7)
                    current_time = time.time()
                    object_id = f"{class_name}_{int(mid_x*100)}_{int(mid_y*100)}"
                    if is_close and system_mode == 0 and (current_time - last_alert_time) > ALERT_COOLDOWN:
                        if object_id not in detected_objects or (current_time - detected_objects.get(object_id, 0)) > ALERT_COOLDOWN:
                            if class_name not in suppressed_objects and warning_counts.get(class_name, 0) < MAX_WARNINGS_PER_OBJECT:
                                cv2.putText(
                                    display_frame, 
                                    'WARNING!!!', 
                                    (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1.0, 
                                    (0, 0, 255), 
                                    3
                                )
                                print(f"Cảnh báo: {class_name} đang rất gần ở khoảng cách {distance_desc}")
                                last_alert_time = current_time
                                detected_objects[object_id] = current_time
                                speak(f"Warning! {class_name} is {distance_desc} in front of you", priority=True)
            if detected_names:
                recent_objects = list(set(detected_names))
                announce_objects(recent_objects)
                detect_center_objects(objects_with_data)
            cv2.putText(
                display_frame, 
                f'Detected: {detected_count} objects', 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            mode_names = ["Object Detection", "Dollar Detection"]
            mode_text = mode_names[system_mode]
            cv2.putText(
                display_frame, 
                f'Mode: {mode_text}', 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            time_since_last = time.time() - last_announcement_time
            cv2.putText(
                display_frame, 
                f'Next announcement in: {max(0, ANNOUNCEMENT_COOLDOWN - int(time_since_last))}s', 
                (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            cv2.imshow('Vision-I: Hệ thống Hỗ trợ Thị lực', cv2.resize(display_frame, (800, 600)))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        except Exception as e:
            print(f"Lỗi trong vòng lặp chính: {e}")
            time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()
    speak("Vision-I system shutting down. Goodbye.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-shortcut":
        create_desktop_shortcut()
    else:
        run_vision_system()