#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-I: Cải tiến Hệ thống Hỗ trợ Thị lực cho Người Khiếm thị (YOLOv8 Version)
"""

import numpy as np
import os
import cv2
import pyttsx3
import time
from threading import Thread, Lock
import webbrowser
from PIL import Image
from ultralytics import YOLO
import sys
import socket
import re
import keyboard
from droidcam_connector import get_droidcam_capture 

# Hằng số hệ thống chung
CONFIDENCE_THRESHOLD = 0.65
ALERT_COOLDOWN = 6
ANNOUNCEMENT_COOLDOWN = 12
MAX_OBJECTS_TO_READ = 2
CENTER_ZONE = 0.25
FRAME_SKIP = 2
MAX_WARNINGS_PER_OBJECT = 2

# Hằng số cho tiền USD
MONEY_CONFIDENCE_THRESHOLD_USD = 0.45
MONEY_ANNOUNCEMENT_COOLDOWN_USD = 5
MONEY_ANNOUNCEMENT_REPEAT_LIMIT_USD = 3
MONEY_ANNOUNCEMENT_RESET_TIMEOUT_USD = 15

# Hằng số cho tiền VND
MONEY_CONFIDENCE_THRESHOLD_VND = 0.50
MONEY_ANNOUNCEMENT_COOLDOWN_VND = 5
MONEY_ANNOUNCEMENT_REPEAT_LIMIT_VND = 3
MONEY_ANNOUNCEMENT_RESET_TIMEOUT_VND = 15

# Ánh xạ tên lớp tiền giấy USD
CLASS_MAPPING_USD = {
    'one-front': 'one', 'one-back': 'one',
    'five-front': 'five', 'five-back': 'five',
    'ten-front': 'ten', 'ten-back': 'ten',
    'twenty-front': 'twenty', 'twenty-back': 'twenty',
    'fifty-front': 'fifty', 'fifty-back': 'fifty',
    'hundred-front': 'hundred', 'hundred-back': 'hundred'
}

engine = pyttsx3.init()
try:
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'english' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.setProperty('rate', 150)
except Exception as e:
    print(f"Lỗi cấu hình giọng nói: {e}")

is_speaking = False
speak_lock = Lock()

def _speak_task(text_to_speak):
    global is_speaking
    try:
        engine.say(text_to_speak)
        engine.runAndWait()
    except Exception as e: print(f"Lỗi trong _speak_task: {e}")
    finally:
        with speak_lock: is_speaking = False

def speak(text, priority=False):
    global is_speaking
    with speak_lock:
        if is_speaking and not priority: return
        is_speaking = True
    print(f"Đọc: {text}")
    tts_thread = Thread(target=_speak_task, args=(text,)); tts_thread.daemon = True; tts_thread.start()

is_running = True
last_announcement_time = 0
system_mode = 0  # 0: Phát hiện vật thể, 1: Phát hiện tiền (cả USD và VND)
already_announced_objects = set()
center_objects_history = {}
warning_counts = {}
suppressed_objects = set()
frame_count = 0

money_detection_counts_usd = {}
last_money_announced_time_usd = {}
money_detection_counts_vnd = {}
last_money_announced_time_vnd = {}

def load_models():
    try:
        print("Đang tải mô hình YOLO...")
        general_model = YOLO('yolov8n.pt')
        print("Đã tải mô hình YOLO chung thành công")

        dollar_model_path = os.path.join('models', 'best_money.pt')
        dollar_model = YOLO(dollar_model_path) if os.path.exists(dollar_model_path) else None
        if dollar_model: print("Đã tải mô hình nhận dạng tiền USD thành công")
        else: print("CẢNH BÁO: Không tìm thấy mô hình tiền USD ('best_money.pt').")

        vnd_model_path = os.path.join('models', 'best_vnd.pt')
        vnd_model = YOLO(vnd_model_path) if os.path.exists(vnd_model_path) else None
        if vnd_model: print("Đã tải mô hình nhận dạng tiền VND thành công")
        else: print("CẢNH BÁO: Không tìm thấy mô hình tiền VND ('best_vnd.pt').")

        return general_model, dollar_model, vnd_model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None, None, None

def calculate_distance(box): # Giữ nguyên
    x1, _, x2, _ = box; width = abs(x2 - x1); normalized_width = width / 640
    if normalized_width > 0.6: return "very close"
    elif normalized_width > 0.4: return "close"
    elif normalized_width > 0.25: return "moderate distance"
    elif normalized_width > 0.15: return "far"
    else: return "very far"

last_dollar_detection_time_heuristic = 0
def detect_dollar_bills_heuristic(image):
    global last_dollar_detection_time_heuristic
    current_time = time.time()
    if current_time - last_dollar_detection_time_heuristic < 3: return None
    last_dollar_detection_time_heuristic = current_time
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        potential_bills = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000: continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 1.9 < aspect_ratio < 2.7: potential_bills.append((x, y, w, h))
        if potential_bills:
            (x, y, w, h) = potential_bills[0]
            bill_img = image[y:y+h, x:x+w]
            green_intensity = np.mean(bill_img[:, :, 1])
            blue_intensity = np.mean(bill_img[:, :, 0])
            red_intensity = np.mean(bill_img[:, :, 2])
            denomination = "unknown dollar"
            if blue_intensity > 150: denomination = "hundred dollar"
            elif red_intensity > 140: denomination = "fifty dollar"
            elif green_intensity > 110: denomination = "twenty dollar"
            elif green_intensity > 100: denomination = "ten dollar"
            elif green_intensity > 90: denomination = "five dollar"
            else: denomination = "one dollar"
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f'{denomination} (heuristic)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return denomination.split(" ")[0]
        return None
    except Exception as e:
        print(f"Lỗi phát hiện USD (heuristic): {e}")
        return None

def find_droidcam(): return get_droidcam_capture() # Giữ nguyên
def announce_objects(objects_to_announce_names=None, force=False): # Giữ nguyên
    global already_announced_objects, last_announcement_time; current_time = time.time()
    if not force and (current_time - last_announcement_time) < ANNOUNCEMENT_COOLDOWN: return
    if objects_to_announce_names:
        actual_objects_to_read = [obj for obj in objects_to_announce_names if force or obj not in already_announced_objects]
        if len(actual_objects_to_read) > MAX_OBJECTS_TO_READ: actual_objects_to_read = actual_objects_to_read[:MAX_OBJECTS_TO_READ]
        if actual_objects_to_read:
            already_announced_objects.update(actual_objects_to_read)
            speak(f"Detected {', '.join(actual_objects_to_read)}"); last_announcement_time = current_time

def detect_center_objects(objects_with_data_list): # Giữ nguyên
    global center_objects_history, warning_counts, suppressed_objects; current_time = time.time(); objects_in_center = []
    for obj_name, distance, mid_x_norm, _, _ in objects_with_data_list: #confidence, mid_y_norm không dùng trực tiếp ở đây
        if abs(mid_x_norm - 0.5) < CENTER_ZONE : objects_in_center.append((obj_name, distance, _)) # _ là confidence
    if objects_in_center:
        objects_in_center.sort(key=lambda x: x[2], reverse=True)
        top_object_name, top_object_distance, _ = objects_in_center[0]
        if top_object_name in suppressed_objects or warning_counts.get(top_object_name, 0) >= MAX_WARNINGS_PER_OBJECT: return
        last_time_warned = center_objects_history.get(top_object_name, 0)
        if (current_time - last_time_warned) > ALERT_COOLDOWN:
            speak(f"{top_object_name} is {top_object_distance} in front of you", priority=True)
            center_objects_history[top_object_name] = current_time
            warning_counts[top_object_name] = warning_counts.get(top_object_name, 0) + 1

def create_desktop_shortcut(): # Giữ nguyên
    try:
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        shortcut_path = os.path.join(desktop, 'Vision-I.bat')
        script_path = os.path.abspath(__file__)
        with open(shortcut_path, 'w') as f:
            f.write(f'@echo off\npython "{script_path}"\npause\n')
        print(f"Đã tạo lối tắt tại: {shortcut_path}"); speak("Desktop shortcut created", priority=True)
    except Exception as e: print(f"Lỗi tạo lối tắt: {e}"); speak("Error creating shortcut", priority=True)


def keyboard_listener():
    global is_running, system_mode, suppressed_objects, already_announced_objects, center_objects_history, warning_counts
    global money_detection_counts_usd, last_money_announced_time_usd
    global money_detection_counts_vnd, last_money_announced_time_vnd

    last_actual_press_time = 0.0
    space_key_physically_down = False
    DOUBLE_PRESS_WINDOW = 0.4

    speak("Press spacebar to change modes, double press to suppress. R to repeat. ESC to exit.", priority=True)

    while is_running:
        try:
            if keyboard.is_pressed('esc'):
                is_running = False; speak("Shutting down", priority=True); time.sleep(1); break

            if keyboard.is_pressed('space'):
                if not space_key_physically_down:
                    space_key_physically_down = True; current_press_time = time.time()
                    if (current_press_time - last_actual_press_time) < DOUBLE_PRESS_WINDOW: # Double Press
                        if center_objects_history:
                            try:
                                recent_object = max(center_objects_history, key=center_objects_history.get)
                                if recent_object not in suppressed_objects:
                                    suppressed_objects.add(recent_object); speak(f"Warnings for {recent_object} suppressed", priority=True)
                                else:
                                    suppressed_objects.remove(recent_object); speak(f"Warnings for {recent_object} re-enabled", priority=True)
                            except ValueError: speak("No object history.", priority=True)
                        else: speak("No object in center.", priority=True)
                        last_actual_press_time = 0
                    else: # Single Press
                        system_mode = (system_mode + 1) % 2 # >>> THAY ĐỔI: Chỉ còn 2 chế độ (0 và 1)

                        already_announced_objects.clear(); center_objects_history.clear(); warning_counts.clear()
                        money_detection_counts_usd.clear(); last_money_announced_time_usd.clear()
                        money_detection_counts_vnd.clear(); last_money_announced_time_vnd.clear()

                        if system_mode == 0: speak("Object detection mode activated", priority=True)
                        elif system_mode == 1: speak("Currency detection mode activated", priority=True) # Thông báo chung
                        last_actual_press_time = current_press_time
            else:
                if space_key_physically_down: space_key_physically_down = False

            if keyboard.is_pressed('r'): # Giữ nguyên logic phím R
                speak("Repeating last detections", priority=True)
                if system_mode == 0 and already_announced_objects:
                     announce_objects(list(already_announced_objects), force=True)
                else: speak("No recent object detections to repeat.", priority=True)
                time.sleep(0.3)
            time.sleep(0.05)
        except: time.sleep(0.1) # Bắt lỗi chung cho keyboard listener


def run_vision_system():
    global is_running, system_mode, frame_count
    global money_detection_counts_usd, last_money_announced_time_usd
    global money_detection_counts_vnd, last_money_announced_time_vnd
    global already_announced_objects, last_announcement_time

    general_model, dollar_model, vnd_model = load_models()

    if general_model is None:
        speak("Failed to load general model. Exiting.", priority=True); time.sleep(2); return

    if len(sys.argv) > 1 and sys.argv[1] == "--create-shortcut":
        create_desktop_shortcut(); return

    cap = find_droidcam()
    if not cap or not cap.isOpened():
        speak("Cannot open camera.", priority=True); time.sleep(2); return

    speak("Vision-I system is starting.", priority=True)
    print("Hệ thống Vision-I đang chạy...")

    keyboard_thread = Thread(target=keyboard_listener); keyboard_thread.daemon = True; keyboard_thread.start()

    while is_running:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Không thể nhận khung hình..."); time.sleep(0.5); cap.release(); cap = find_droidcam()
                if not cap or not cap.isOpened(): speak("Camera lost.", priority=True); is_running = False; break
                else: speak("Re-established camera.", priority=True)
                continue

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                cv2.imshow('Vision-I', cv2.resize(frame, (800, 600)))
                if cv2.waitKey(1) & 0xFF == 27: is_running = False; break
                continue

            display_frame = frame.copy()
            current_time = time.time()

            # Chế độ 0: Phát hiện vật thể
            if system_mode == 0:
                process_frame_obj = cv2.resize(frame, (640, 480))
                results_obj = general_model(process_frame_obj, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False)
                detected_object_names_this_frame = []
                objects_data_for_center_warning = []

                for result_o in results_obj:
                    for box_o in result_o.boxes:
                        x1o, y1o, x2o, y2o = map(int, box_o.xyxy[0].cpu().numpy()) # Chuyển sang int sớm hơn
                        # Scale về frame gốc
                        scale_x_o = display_frame.shape[1] / 640
                        scale_y_o = display_frame.shape[0] / 480
                        x1d, y1d, x2d, y2d = int(x1o * scale_x_o), int(y1o * scale_y_o), int(x2o * scale_x_o), int(y2o * scale_y_o)

                        confidence_o = float(box_o.conf[0])
                        class_name_o = general_model.names[int(box_o.cls[0])] # Lấy tên từ model object

                        cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), (0, 255, 0), 2)
                        cv2.putText(display_frame, f'{class_name_o}: {confidence_o:.2f}', (x1d, y1d - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        mid_x_norm_o = (x1o + x2o) / 2 / 640 # Dùng tọa độ của process_frame_obj
                        mid_y_norm_o = (y1o + y2o) / 2 / 480
                        distance_desc_o = calculate_distance([x1o, y1o, x2o, y2o])

                        if class_name_o not in detected_object_names_this_frame:
                             detected_object_names_this_frame.append(class_name_o)
                        objects_data_for_center_warning.append((class_name_o, distance_desc_o, mid_x_norm_o, mid_y_norm_o, confidence_o))
                
                if detected_object_names_this_frame: announce_objects(detected_object_names_this_frame)
                if objects_data_for_center_warning: detect_center_objects(objects_data_for_center_warning)
                cv2.putText(display_frame, f'Vat the: {len(objects_data_for_center_warning)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # >>> THAY ĐỔI: Chế độ 1: Phát hiện tiền (cả USD và VND)
            elif system_mode == 1:
                # Xử lý USD
                denoms_to_clear_usd = [d for d, t in last_money_announced_time_usd.items() if current_time - t > MONEY_ANNOUNCEMENT_RESET_TIMEOUT_USD]
                for d in denoms_to_clear_usd:
                    if d in money_detection_counts_usd: del money_detection_counts_usd[d]
                    if d in last_money_announced_time_usd: del last_money_announced_time_usd[d]

                if dollar_model is not None:
                    process_frame_money = cv2.resize(frame, (640, 640)) # Chung kích thước cho các model tiền
                    results_money_usd = dollar_model(process_frame_money, conf=MONEY_CONFIDENCE_THRESHOLD_USD, verbose=False)
                    unique_denoms_usd_this_frame = set()

                    for res_usd in results_money_usd:
                        for box_usd in res_usd.boxes:
                            x1m, y1m, x2m, y2m = map(int, box_usd.xyxy[0].cpu().numpy())
                            # Scale về frame gốc
                            scale_x_disp = display_frame.shape[1] / 640
                            scale_y_disp = display_frame.shape[0] / 640
                            x1d, y1d, x2d, y2d = int(x1m*scale_x_disp), int(y1m*scale_y_disp), int(x2m*scale_x_disp), int(y2m*scale_y_disp)

                            conf_usd = float(box_usd.conf[0])
                            model_class_name_usd = dollar_model.names[int(box_usd.cls[0])]
                            denomination_usd = CLASS_MAPPING_USD.get(model_class_name_usd, model_class_name_usd.split('-')[0])
                            unique_denoms_usd_this_frame.add(denomination_usd)

                            cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), (0, 0, 255), 2) # Đỏ cho USD
                            cv2.putText(display_frame, f'{denomination_usd}$: {conf_usd:.2f}', (x1d, y1d - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            count_usd = money_detection_counts_usd.get(denomination_usd, 0)
                            last_announced_usd = last_money_announced_time_usd.get(denomination_usd, 0)
                            if count_usd < MONEY_ANNOUNCEMENT_REPEAT_LIMIT_USD and \
                               (current_time - last_announced_usd > MONEY_ANNOUNCEMENT_COOLDOWN_USD):
                                speak(f"Detected {denomination_usd} dollar", priority=True)
                                money_detection_counts_usd[denomination_usd] = count_usd + 1
                                last_money_announced_time_usd[denomination_usd] = current_time
                    for d in unique_denoms_usd_this_frame: last_money_announced_time_usd[d] = current_time
                elif not vnd_model : # Chỉ hiện heuristic nếu cả 2 model tiền đều không có
                    denom_heuristic = detect_dollar_bills_heuristic(display_frame) # Fallback heuristic cho USD
                    if denom_heuristic:
                        if denom_heuristic not in money_detection_counts_usd or \
                           current_time - last_money_announced_time_usd.get(denom_heuristic, 0) > MONEY_ANNOUNCEMENT_COOLDOWN_USD : # Giả sử dùng cooldown USD
                            speak(f"Detected {denom_heuristic} dollar (heuristic)", priority=True)
                            money_detection_counts_usd[denom_heuristic] = money_detection_counts_usd.get(denom_heuristic, 0) + 1
                            last_money_announced_time_usd[denom_heuristic] = current_time


                # Xử lý VND (chạy ngay sau USD trong cùng chế độ tiền)
                denoms_to_clear_vnd = [d for d, t in last_money_announced_time_vnd.items() if current_time - t > MONEY_ANNOUNCEMENT_RESET_TIMEOUT_VND]
                for d in denoms_to_clear_vnd:
                    if d in money_detection_counts_vnd: del money_detection_counts_vnd[d]
                    if d in last_money_announced_time_vnd: del last_money_announced_time_vnd[d]
                
                if vnd_model is not None:
                    process_frame_money_vnd = cv2.resize(frame, (640, 640)) # Có thể dùng lại process_frame_money nếu kích thước model giống nhau
                    results_money_vnd = vnd_model(process_frame_money_vnd, conf=MONEY_CONFIDENCE_THRESHOLD_VND, verbose=False)
                    unique_denoms_vnd_this_frame = set()

                    for res_vnd in results_money_vnd:
                        for box_vnd in res_vnd.boxes:
                            x1m, y1m, x2m, y2m = map(int, box_vnd.xyxy[0].cpu().numpy())
                            # Scale về frame gốc
                            scale_x_disp = display_frame.shape[1] / 640
                            scale_y_disp = display_frame.shape[0] / 640
                            x1d, y1d, x2d, y2d = int(x1m*scale_x_disp), int(y1m*scale_y_disp), int(x2m*scale_x_disp), int(y2m*scale_y_disp)

                            conf_vnd = float(box_vnd.conf[0])
                            denomination_vnd = vnd_model.names[int(box_vnd.cls[0])]
                            # Nếu có CLASS_MAPPING_VND:
                            # denomination_vnd = CLASS_MAPPING_VND.get(model_class_name_vnd, model_class_name_vnd.split('-')[0])
                            unique_denoms_vnd_this_frame.add(denomination_vnd)

                            cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), (255, 0, 0), 2) # Xanh dương cho VND
                            cv2.putText(display_frame, f'{denomination_vnd} VND: {conf_vnd:.2f}', (x1d, y1d - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
                            count_vnd = money_detection_counts_vnd.get(denomination_vnd, 0)
                            last_announced_vnd = last_money_announced_time_vnd.get(denomination_vnd, 0)
                            if count_vnd < MONEY_ANNOUNCEMENT_REPEAT_LIMIT_VND and \
                               (current_time - last_announced_vnd > MONEY_ANNOUNCEMENT_COOLDOWN_VND):
                                speak_text_vnd = f"Detected {denomination_vnd.replace('k', ' thousand')} Dong"
                                if '00' not in denomination_vnd and 'k' not in denomination_vnd: speak_text_vnd = f"Detected {denomination_vnd} Dong"
                                speak(speak_text_vnd, priority=True)
                                money_detection_counts_vnd[denomination_vnd] = count_vnd + 1
                                last_money_announced_time_vnd[denomination_vnd] = current_time
                    for d in unique_denoms_vnd_this_frame: last_money_announced_time_vnd[d] = current_time
                
                if dollar_model is None and vnd_model is None:
                     cv2.putText(display_frame, "Currency Models not loaded", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                cv2.putText(display_frame, 'Che do Phat hien Tien Te', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


            # Hiển thị thông tin chung
            mode_names = ["Phat hien Vat the", "Phat hien Tien te"] 
            mode_text = mode_names[system_mode]
            cv2.putText(display_frame, f'Che do: {mode_text}', (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if system_mode == 0 : # Cooldown chỉ cho object detection
                time_since_last_obj_ann = current_time - last_announcement_time
                next_ann_in = max(0, ANNOUNCEMENT_COOLDOWN - int(time_since_last_obj_ann))
                cv2.putText(display_frame, f'Thong bao sau: {next_ann_in}s', (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

            cv2.imshow('Vision-I: He thong Ho tro Thi luc', cv2.resize(display_frame, (800, 600)))
            key = cv2.waitKey(1) & 0xFF
            if key == 27: is_running = False; break
        except Exception as e:
            print(f"Lỗi trong vòng lặp chính: {e}"); import traceback; traceback.print_exc(); time.sleep(1)

    if cap and cap.isOpened(): cap.release()
    cv2.destroyAllWindows()
    speak("Vision-I system shutting down.", priority=True)
    print("Hệ thống đã tắt.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-shortcut": create_desktop_shortcut()
    else: run_vision_system()