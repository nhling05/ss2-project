#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-I: Cải tiến Hệ thống Hỗ trợ Thị lực cho Người Khiếm thị (YOLOv8 Version)
Phiên bản tự động không cần PyAudio với cải tiến hiệu suất và nhận dạng tiền giấy
"""

import numpy as np
import os
import cv2
import pyttsx3
import time
from threading import Thread, Lock # Thêm Lock
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
MAX_WARNINGS_PER_OBJECT = 2
MONEY_CONFIDENCE_THRESHOLD = 0.45  # Ngưỡng tin cậy thấp hơn cho phát hiện tiền
MONEY_ANNOUNCEMENT_COOLDOWN = 5    # Thời gian nghỉ ngắn hơn cho thông báo tiền
MONEY_ANNOUNCEMENT_REPEAT_LIMIT = 3 # Giới hạn số lần đọc cho mỗi mệnh giá tiền
MONEY_ANNOUNCEMENT_RESET_TIMEOUT = 15 # Sau 15s không thấy mệnh giá, reset bộ đếm

# Ánh xạ tên lớp tiền giấy
CLASS_MAPPING = {
    'one-front': 'one',
    'one-back': 'one',
    'five-front': 'five',
    'five-back': 'five',
    'ten-front': 'ten',
    'ten-back': 'ten',
    'twenty-front': 'twenty',
    'twenty-back': 'twenty',
    'fifty-front': 'fifty',
    'fifty-back': 'fifty'
}

# Khởi tạo bộ đọc giọng nói
engine = pyttsx3.init()

# Cấu hình thuộc tính giọng nói
try:
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'english' in voice.name.lower(): # Giữ nguyên logic chọn giọng
            engine.setProperty('voice', voice.id)
            break
    engine.setProperty('rate', 150)
except Exception as e:
    print(f"Lỗi cấu hình giọng nói: {e}")

# Biến theo dõi trạng thái hệ thống và TTS
is_speaking = False
speak_lock = Lock() # Khóa cho việc cập nhật is_speaking

# Hàm chạy tác vụ đọc TTS trong một luồng riêng
def _speak_task(text_to_speak):
    global is_speaking
    try:
        engine.say(text_to_speak)
        engine.runAndWait()
    except Exception as e:
        print(f"Lỗi trong _speak_task: {e}")
    finally:
        with speak_lock:
            is_speaking = False

# Hàm đọc thông báo (đã sửa để chạy TTS ở luồng riêng)
def speak(text, priority=False):
    global is_speaking
    with speak_lock:
        if is_speaking and not priority:
            return # Nếu đang nói và không phải ưu tiên, bỏ qua
        is_speaking = True # Đặt is_speaking sớm để các lệnh gọi sau biết

    print(f"Đọc: {text}")
    tts_thread = Thread(target=_speak_task, args=(text,))
    tts_thread.daemon = True # Để thread tự thoát khi chương trình chính thoát
    tts_thread.start()

# Khởi tạo biến theo dõi trạng thái hệ thống
is_running = True
last_speech_time = 0
last_announcement_time = 0
last_dollar_detection_time = 0 # Dùng cho detect_dollar_bills
last_space_press_time = 0
space_press_count = 0
system_mode = 0  # 0: Phát hiện vật thể, 1: Phát hiện tiền
already_announced_objects = set()
center_objects_history = {}
warning_counts = {}  # Track number of warnings per object
suppressed_objects = set()  # Objects with suppressed warnings
frame_count = 0

# Biến cho logic đếm tiền
money_detection_counts = {}  # Ví dụ: {"one": 1, "five": 3}
last_money_announced_time = {} # Ví dụ: {"one": timestamp, "five": timestamp} để cooldown và reset

# Tải mô hình YOLOv8
def load_models():
    try:
        print("Đang tải mô hình YOLO...")
        # Tải mô hình nhận dạng đối tượng chung
        general_model = YOLO('yolov8n.pt')
        print("Đã tải mô hình YOLO thành công")
        
        # Tải mô hình nhận dạng tiền giấy chuyên biệt
        dollar_model_path = os.path.join('models', 'best_money.pt')
        if os.path.exists(dollar_model_path):
            dollar_model = YOLO(dollar_model_path)
            print("Đã tải mô hình nhận dạng tiền thành công")
        else:
            print("Không tìm thấy mô hình nhận dạng tiền, tính năng nhận diện tiền có thể bị hạn chế hoặc sử dụng phương pháp xử lý ảnh thay thế.")
            dollar_model = None
            
        return general_model, dollar_model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        # speak("Error loading model. Please check your model installation.") # Cân nhắc nếu engine TTS có thể chưa sẵn sàng
        return None, None

# Tính khoảng cách dựa trên kích thước hộp giới hạn
def calculate_distance(box): # box là [x1, y1, x2, y2] của đối tượng trong frame đã resize cho model
    # Hàm này cần được điều chỉnh nếu kích thước frame đầu vào của model thay đổi
    # Hiện tại, giả sử input model là 640 (width) cho general model
    # và có thể là kích thước khác cho money model (cần nhất quán)
    # Tuy nhiên, logic hiện tại dựa trên normalized_width = width / 640
    # Nếu box truyền vào là tọa độ trên frame gốc, thì 640 nên là frame.shape[1]
    # Nếu box truyền vào là tọa độ trên frame đã resize cho model (ví dụ 640x480), thì width/640 là hợp lý.
    x1, y1, x2, y2 = box
    width = abs(x2 - x1)
    # Giả sử width này là từ frame đã resize (ví dụ 640x480 cho general model)
    # Nếu `calculate_distance` được gọi với tọa độ trên frame gốc,
    # cần chuẩn hóa dựa trên `frame.shape[1]`
    normalized_width = width / 640 # Cần xem xét lại hằng số này
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

# Phát hiện tiền giấy bằng phương pháp xử lý ảnh (giữ nguyên logic gốc)
def detect_dollar_bills(image):
    global last_dollar_detection_time
    current_time = time.time()
    if current_time - last_dollar_detection_time < 3: # Cooldown 3s riêng của hàm này
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
            if area < 10000: # Ngưỡng diện tích tối thiểu
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4: # Kiểm tra hình tứ giác
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 1.9 < aspect_ratio < 2.7: # Kiểm tra tỷ lệ khung hình của tờ tiền
                    potential_bills.append((x, y, w, h))
        
        if potential_bills:
            # Giả sử chỉ xử lý tờ tiền lớn nhất hoặc đầu tiên tìm thấy
            (x, y, w, h) = potential_bills[0] # Có thể cải tiến để xử lý nhiều tờ
            bill_img = image[y:y+h, x:x+w]
            
            # Phần heuristic dựa trên màu sắc (cần cải thiện độ chính xác)
            # bill_img_rgb = cv2.cvtColor(bill_img, cv2.COLOR_BGR2RGB)
            # bill_img_rgb = cv2.resize(bill_img_rgb, (100, 40)) # Resize để chuẩn hóa
            # pixels = bill_img_rgb.reshape((-1, 3))
            # pixels = np.float32(pixels)
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            # _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # centers = np.uint8(centers)
            # color_percentages = []
            # for i in range(5):
            #     color_count = np.count_nonzero(labels == i)
            #     color_percentages.append(color_count / labels.size)

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            green_intensity = np.mean(bill_img[:, :, 1]) # Kênh G trong BGR
            blue_intensity = np.mean(bill_img[:, :, 0])  # Kênh B
            red_intensity = np.mean(bill_img[:, :, 2])   # Kênh R

            denomination = "unknown dollar" # Mặc định
            if blue_intensity > 150: # Heuristic cho $100 (màu xanh dương đặc trưng)
                denomination = "hundred dollar"
            elif red_intensity > 140: # Heuristic cho $50 (màu đỏ/cam)
                denomination = "fifty dollar"
            elif green_intensity > 110: # Heuristic cho $20
                denomination = "twenty dollar"
            elif green_intensity > 100: # Heuristic cho $10
                denomination = "ten dollar"
            elif green_intensity > 90: # Heuristic cho $5
                denomination = "five dollar"
            else: # Heuristic cho $1
                denomination = "one dollar"
            
            cv2.putText(image, f'{denomination} bill (heuristic)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Hàm này vẫn tự speak, logic giới hạn 3 lần đọc không áp dụng ở đây trừ khi sửa đổi lớn.
            speak(f"Detected {denomination} bill by image processing", priority=True)
            return denomination.split(" ")[0] # Trả về mệnh giá "one", "five",...
        return None
    except Exception as e:
        print(f"Lỗi phát hiện tiền giấy (image processing): {e}")
        return None

# Phương pháp kết hợp sử dụng cả mô hình ML và xử lý ảnh (hàm này không được dùng trong vòng lặp chính hiện tại)
def detect_dollars_combined(frame, dollar_model_instance):
    detected_bills_info = [] # Sẽ chứa (bill_info_string, box_coords, confidence)
    
    if dollar_model_instance is not None:
        process_frame_ml = cv2.resize(frame, (640, 640)) # Kích thước cho model tiền
        results = dollar_model_instance(process_frame_ml, conf=MONEY_CONFIDENCE_THRESHOLD)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1_ml, y1_ml, x2_ml, y2_ml = box.xyxy[0].cpu().numpy()
                scale_x = frame.shape[1] / 640
                scale_y = frame.shape[0] / 640
                x1, x2 = int(x1_ml * scale_x), int(x2_ml * scale_x)
                y1, y2 = int(y1_ml * scale_y), int(y2_ml * scale_y)
                
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                denomination = class_name.split('-')[0]
                side = "front side" if "front" in class_name else "back side"
                
                bill_name_vi = class_name.replace('-front', ' mặt trước').replace('-back', ' mặt sau')
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{bill_name_vi}: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                bill_info_to_speak = f"{denomination} dollar bill" #, {side}" Bỏ side để đơn giản
                detected_bills_info.append((bill_info_to_speak, (x1,y1,x2,y2), confidence, denomination))
    
    if not detected_bills_info: # Nếu model ML không phát hiện gì
        traditional_result_denom = detect_dollar_bills(frame) # Hàm này tự vẽ và speak
        if traditional_result_denom:
            # Không có box, confidence từ hàm này, chỉ có denomination
            # Cần điều chỉnh nếu muốn tích hợp đầy đủ
            # detected_bills_info.append((f"{traditional_result_denom} dollar bill", None, 0.5, traditional_result_denom))
            pass # Vì detect_dollar_bills đã speak rồi

    return detected_bills_info # Trả về danh sách các bill đã phát hiện bằng ML


def find_droidcam(): # Giữ nguyên
    return get_droidcam_capture()

# Thông báo các vật thể được phát hiện
def announce_objects(objects_to_announce_names=None, force=False):
    global already_announced_objects, last_announcement_time
    current_time = time.time()

    if not force and (current_time - last_announcement_time) < ANNOUNCEMENT_COOLDOWN:
        return

    if objects_to_announce_names and len(objects_to_announce_names) > 0:
        actual_objects_to_read = []
        if force:
            actual_objects_to_read = objects_to_announce_names # Nếu force, đọc lại hết
            already_announced_objects.clear() # Xóa lịch sử để đảm bảo đọc lại
        else:
            actual_objects_to_read = [obj_name for obj_name in objects_to_announce_names if obj_name not in already_announced_objects]
        
        if len(actual_objects_to_read) > MAX_OBJECTS_TO_READ:
            actual_objects_to_read = actual_objects_to_read[:MAX_OBJECTS_TO_READ]

        if actual_objects_to_read:
            already_announced_objects.update(actual_objects_to_read)
            if len(actual_objects_to_read) == 1:
                speak(f"Detected {actual_objects_to_read[0]}")
            else:
                objects_text = ", ".join(actual_objects_to_read)
                speak(f"Detected {objects_text}")
            last_announcement_time = current_time
            
            # Xóa bớt lịch sử nếu quá nhiều để tránh đầy bộ nhớ
            if len(already_announced_objects) > 15: # Ngưỡng tùy chỉnh
                # Xóa các đối tượng cũ nhất, hoặc đơn giản là clear()
                # Để đơn giản, có thể clear sau một khoảng thời gian thay vì số lượng
                pass # Hiện tại chỉ update, chưa có logic xóa bớt phức tạp

# Phát hiện và thông báo vật thể ở trung tâm
def detect_center_objects(objects_with_data_list): # objects_with_data_list là list of tuples
    global center_objects_history, last_speech_time, warning_counts, suppressed_objects
    current_time = time.time()
    objects_in_center = []
    # center_threshold = CENTER_ZONE # Đã là global

    for obj_name, distance, mid_x_norm, mid_y_norm, confidence in objects_with_data_list:
        # mid_x_norm, mid_y_norm là tọa độ chuẩn hóa (0-1) của tâm đối tượng
        if abs(mid_x_norm - 0.5) < CENTER_ZONE and abs(mid_y_norm - 0.5) < CENTER_ZONE:
            objects_in_center.append((obj_name, distance, confidence))
    
    if objects_in_center:
        objects_in_center.sort(key=lambda x: x[2], reverse=True) # Sắp xếp theo confidence giảm dần
        top_object_name, top_object_distance, _ = objects_in_center[0]

        if top_object_name in suppressed_objects: # Nếu đối tượng đã bị tắt thông báo
            return

        warning_count_for_object = warning_counts.get(top_object_name, 0)
        if warning_count_for_object >= MAX_WARNINGS_PER_OBJECT: # Đã cảnh báo đủ số lần
            return

        last_time_warned = center_objects_history.get(top_object_name, 0)
        if (current_time - last_time_warned) > ALERT_COOLDOWN: # Đủ thời gian nghỉ giữa các cảnh báo
            message = f"{top_object_name} is {top_object_distance} in front of you"
            speak(message, priority=True) # Ưu tiên cho cảnh báo trung tâm
            center_objects_history[top_object_name] = current_time
            last_speech_time = current_time # Cập nhật thời gian nói chung
            warning_counts[top_object_name] = warning_count_for_object + 1

# Tạo desktop shortcut
def create_desktop_shortcut(): # Giữ nguyên
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

# ... (Các phần import và khai báo khác giữ nguyên) ...

# Theo dõi phím bấm từ bàn phím
def keyboard_listener():
    # Khai báo global cho các biến mà hàm này sẽ thay đổi giá trị của chúng ở phạm vi toàn cục
    global is_running, system_mode, suppressed_objects, already_announced_objects, center_objects_history, warning_counts, money_detection_counts, last_money_announced_time
    
    # Biến cục bộ cho logic xử lý phím space trong listener này
    last_actual_press_time = 0.0  # Thời điểm của lần nhấn phím space thực sự cuối cùng
    space_key_physically_down = False # Cờ theo dõi trạng thái vật lý của phím space (đang được nhấn hay không)

    # Thời gian chờ cho double press (ví dụ: 400ms)
    DOUBLE_PRESS_WINDOW = 0.4 

    speak("Press spacebar once to change modes, twice to suppress/unsuppress warnings for center object. Press R to repeat detections. Press escape to exit.", priority=True)
    
    while is_running:
        try:
            # Xử lý phím ESC để thoát
            if keyboard.is_pressed('esc'):
                print("Phím ESC được nhấn. Đang thoát...")
                is_running = False
                speak("Shutting down the system", priority=True)
                time.sleep(1) # Đợi chút để speak kịp chạy
                break
            
            # Xử lý phím Space
            if keyboard.is_pressed('space'):
                if not space_key_physically_down:  # Phát hiện khi phím VỪA ĐƯỢC NHẤN XUỐNG
                    space_key_physically_down = True # Đánh dấu phím đang được giữ
                    current_press_time = time.time()

                    # Kiểm tra double press
                    if (current_press_time - last_actual_press_time) < DOUBLE_PRESS_WINDOW:
                        # Đây là DOUBLE PRESS
                        if center_objects_history:
                            try:
                                recent_object_to_suppress = max(center_objects_history, key=center_objects_history.get)
                                if recent_object_to_suppress not in suppressed_objects:
                                    suppressed_objects.add(recent_object_to_suppress)
                                    speak(f"Warnings for {recent_object_to_suppress} suppressed (Double Press)", priority=True)
                                else:
                                    suppressed_objects.remove(recent_object_to_suppress)
                                    speak(f"Warnings for {recent_object_to_suppress} re-enabled (Double Press)", priority=True)
                            except ValueError: # Trường hợp center_objects_history rỗng
                                speak("No object history to suppress/unsuppress (Double Press).", priority=True)
                        else:
                            speak("No object in center to suppress/unsuppress (Double Press).", priority=True)
                        
                        # Reset last_actual_press_time để lần nhấn tiếp theo (nếu có) không bị coi là double press nữa
                        # và sẽ bắt đầu một chuỗi single press mới.
                        last_actual_press_time = 0 
                    else:
                        # Đây là SINGLE PRESS (hoặc là lần nhấn đầu tiên của một double press tiềm năng)
                        system_mode = (system_mode + 1) % 2
                        
                        # Reset các trạng thái liên quan khi chuyển chế độ
                        money_detection_counts.clear()
                        last_money_announced_time.clear()
                        already_announced_objects.clear()
                        center_objects_history.clear()
                        warning_counts.clear()
                        # suppressed_objects.clear() # Cân nhắc việc này, có thể người dùng muốn giữ lại
                        
                        if system_mode == 0:
                            speak("Object detection mode activated (Single Press)", priority=True)
                        elif system_mode == 1:
                            speak("Dollar detection mode activated (Single Press)", priority=True)
                        
                        # Ghi lại thời điểm của lần nhấn này, nó có thể là lần đầu của một double press
                        last_actual_press_time = current_press_time
            else: # Phím space không được nhấn
                if space_key_physically_down: # Nếu trước đó phím đang được giữ thì giờ là lúc nó được nhả ra
                    space_key_physically_down = False # Đánh dấu phím đã được nhả

            # Xử lý phím 'R' để lặp lại phát hiện
            if keyboard.is_pressed('r'):
                # Thêm một chút debounce hoặc cooldown cho phím R nếu cần
                # (Hiện tại chưa có, nếu nhấn giữ R sẽ gọi liên tục)
                speak("Repeating the last detections", priority=True)
                if already_announced_objects: # Đảm bảo có gì đó để lặp lại
                     announce_objects(list(already_announced_objects), force=True)
                else:
                     speak("No recent detections to repeat.", priority=True)
                time.sleep(0.3) # Cooldown ngắn sau khi nhấn R để tránh lặp quá nhanh

            time.sleep(0.05)  # Giảm tải CPU cho vòng lặp listener, không nên quá lớn để không miss event

        except Exception as e:
            # print(f"Lỗi trong keyboard_listener: {e}") # Bỏ comment nếu cần debug
            time.sleep(0.1)


# Hàm chính để chạy hệ thống Vision-I
def run_vision_system():
    global is_running, system_mode, last_speech_time, frame_count
    global money_detection_counts, last_money_announced_time, already_announced_objects, last_announcement_time
    global center_objects_history, warning_counts, suppressed_objects # Đảm bảo các biến này global nếu được thay đổi ở keyboard_listener

    general_model, dollar_model = load_models()
    
    if general_model is None:
        print("Không thể tải mô hình nhận dạng đối tượng chung. Đang thoát...")
        speak("Failed to load the general object detection model. Exiting.", priority=True)
        time.sleep(2) # Đợi speak chạy xong
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--create-shortcut":
        create_desktop_shortcut()
        return
        
    cap = find_droidcam()
    if not cap or not cap.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra kết nối webcam hoặc DroidCam.")
        speak("Cannot open camera. Please check camera connection.", priority=True)
        time.sleep(2)
        return
        
    speak("Vision-I system is starting.", priority=True) # Thông báo khởi động ngắn gọn hơn
    print("Hệ thống Vision-I đang chạy. Sử dụng phím Space để điều khiển, phím ESC để thoát.")
    
    keyboard_thread = Thread(target=keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    last_alert_time = 0 # Dùng cho cảnh báo vật thể gần trong chế độ 0
    detected_objects_warnings = {} # Dùng riêng cho logic cảnh báo vật thể gần (object_id_warning)

    while is_running:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Không thể nhận khung hình từ camera. Đang thử lại...")
                time.sleep(0.5) # Đợi chút rồi thử lại
                # Thử kết nối lại camera nếu lỗi liên tục
                cap.release()
                cap = find_droidcam()
                if not cap or not cap.isOpened():
                    speak("Camera connection lost and could not be re-established. Shutting down.", priority=True)
                    is_running = False
                    break
                else:
                    speak("Re-established camera connection.", priority=True)
                continue
            
            frame_count += 1
            if frame_count % FRAME_SKIP != 0: # Bỏ qua frame để giảm tải
                # Vẫn hiển thị frame gốc để mượt hơn
                cv2.imshow('Vision-I: Hệ thống Hỗ trợ Thị lực', cv2.resize(frame, (800, 600)))
                if cv2.waitKey(1) & 0xFF == 27: # Vẫn cho phép thoát bằng ESC ở đây
                    is_running = False
                    break
                continue
            
            display_frame = frame.copy()
            current_time = time.time()

            # Chế độ phát hiện tiền
            if system_mode == 1:
                # Dọn dẹp money_detection_counts
                denominations_to_clear_from_count = []
                for denom, last_seen_or_announced_time in last_money_announced_time.items():
                    if current_time - last_seen_or_announced_time > MONEY_ANNOUNCEMENT_RESET_TIMEOUT:
                        denominations_to_clear_from_count.append(denom)
                for denom in denominations_to_clear_from_count:
                    if denom in money_detection_counts:
                        del money_detection_counts[denom]
                        print(f"Resetting announcement count for {denom} dollar bill due to timeout.")
                    if denom in last_money_announced_time: # Cũng xóa khỏi đây để logic được sạch
                         del last_money_announced_time[denom]


                if dollar_model is not None:
                    process_frame_money = cv2.resize(frame, (640, 640)) # Kích thước cho model tiền
                    results_money = dollar_model(process_frame_money, conf=MONEY_CONFIDENCE_THRESHOLD, verbose=False)
                    
                    unique_denominations_detected_this_frame = set()

                    for result_m in results_money:
                        boxes_m = result_m.boxes
                        for box_m in boxes_m:
                            x1_orig_m, y1_orig_m, x2_orig_m, y2_orig_m = box_m.xyxy[0].cpu().numpy()
                            # Scale tọa độ về frame gốc
                            scale_x_m = display_frame.shape[1] / 640
                            scale_y_m = display_frame.shape[0] / 640
                            x1m, x2m = int(x1_orig_m * scale_x_m), int(x2_orig_m * scale_x_m)
                            y1m, y2m = int(y1_orig_m * scale_y_m), int(y2_orig_m * scale_y_m)
                            
                            confidence_m = float(box_m.conf[0])
                            class_id_m = int(box_m.cls[0])
                            class_name_model = result_m.names[class_id_m]
                            
                            denomination = class_name_model.split('-')[0]
                            unique_denominations_detected_this_frame.add(denomination)

                            bill_name_vi = class_name_model.replace('-front', ' mặt trước').replace('-back', ' mặt sau')
                            cv2.rectangle(display_frame, (x1m, y1m), (x2m, y2m), (0, 0, 255), 2) # Màu đỏ cho tiền
                            label_m = f'{bill_name_vi}: {confidence_m:.2f}'
                            cv2.putText(display_frame, label_m, (x1m, y1m - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            current_det_count = money_detection_counts.get(denomination, 0)
                            last_announced_denom_time = last_money_announced_time.get(denomination, 0)

                            if current_det_count < MONEY_ANNOUNCEMENT_REPEAT_LIMIT:
                                if current_time - last_announced_denom_time > MONEY_ANNOUNCEMENT_COOLDOWN:
                                    speak(f"Detected {denomination} dollar bill", priority=True)
                                    money_detection_counts[denomination] = current_det_count + 1
                                    last_money_announced_time[denomination] = current_time
                                    # print(f"Announced {denomination}, count: {money_detection_counts[denomination]}")
                            # else:
                                # print(f"Limit reached for {denomination}")
                    
                    for denom in unique_denominations_detected_this_frame: # Cập nhật thời gian thấy cuối cùng
                        last_money_announced_time[denom] = current_time
                
                else: # dollar_model is None, dùng image processing
                    denom_from_img_proc = detect_dollar_bills(display_frame) # Hàm này tự speak và vẽ
                    if denom_from_img_proc:
                        # Logic đếm 3 lần chưa áp dụng tốt cho nhánh này nếu detect_dollar_bills tự speak
                        # Để đơn giản, chỉ cập nhật thời gian thấy để có thể reset nếu cần
                        last_money_announced_time[denom_from_img_proc] = current_time


                cv2.putText(display_frame, 'Che do Phat hien Tien', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Chế độ phát hiện đối tượng thông thường
            else: # system_mode == 0
                process_frame_obj = cv2.resize(frame, (640, 480)) # Kích thước cho model chung
                results_obj = general_model(process_frame_obj, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False) # imgsz có thể là (480,640)
                
                detected_object_names_this_frame = [] # Tên các đối tượng trong frame này để thông báo chung
                objects_data_for_center_warning = [] # Dữ liệu cho cảnh báo trung tâm

                for result_o in results_obj:
                    boxes_o = result_o.boxes
                    for box_o in boxes_o:
                        x1_orig_o, y1_orig_o, x2_orig_o, y2_orig_o = box_o.xyxy[0].cpu().numpy()
                        # Scale về frame gốc
                        scale_x_o = display_frame.shape[1] / 640 # Kích thước width của process_frame_obj
                        scale_y_o = display_frame.shape[0] / 480 # Kích thước height của process_frame_obj
                        x1o, x2o = int(x1_orig_o * scale_x_o), int(x2_orig_o * scale_x_o)
                        y1o, y2o = int(y1_orig_o * scale_y_o), int(y2_orig_o * scale_y_o)

                        confidence_o = float(box_o.conf[0])
                        class_id_o = int(box_o.cls[0])
                        class_name_o = result_o.names[class_id_o]

                        # Lọc các vật thể nhỏ không quan trọng
                        # if class_name_o in ["cell phone", "remote", "mouse", "keyboard", "book", "cup", "bottle"]:
                        #     object_area_on_original = (x2o - x1o) * (y2o - y1o)
                        #     frame_area = display_frame.shape[0] * display_frame.shape[1]
                        #     if object_area_on_original < frame_area / 100: # Ví dụ: nhỏ hơn 1% diện tích frame
                        #         continue
                        
                        cv2.rectangle(display_frame, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
                        label_o = f'{class_name_o}: {confidence_o:.2f}'
                        cv2.putText(display_frame, label_o, (x1o, y1o - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Tọa độ tâm chuẩn hóa trên process_frame_obj (để đưa vào detect_center_objects)
                        mid_x_norm_o = (x1_orig_o + x2_orig_o) / 2 / 640
                        mid_y_norm_o = (y1_orig_o + y2_orig_o) / 2 / 480
                        
                        # Tính khoảng cách dựa trên box trên process_frame_obj
                        distance_desc_o = calculate_distance([x1_orig_o, y1_orig_o, x2_orig_o, y2_orig_o])
                        
                        # cv2.putText(display_frame, f'Dist: {distance_desc_o}', (x1o, y1o - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

                        if class_name_o not in detected_object_names_this_frame:
                             detected_object_names_this_frame.append(class_name_o)
                        
                        objects_data_for_center_warning.append((class_name_o, distance_desc_o, mid_x_norm_o, mid_y_norm_o, confidence_o))
                        
                        # Logic cảnh báo vật thể gần (giữ nguyên, nhưng dùng detected_objects_warnings)
                        is_close = ("close" in distance_desc_o or "very close" in distance_desc_o) and \
                                   (0.5 - CENTER_ZONE < mid_x_norm_o < 0.5 + CENTER_ZONE) # Kiểm tra vùng trung tâm
                        
                        if is_close and (current_time - last_alert_time) > ALERT_COOLDOWN: # Cooldown chung cho loại cảnh báo này
                            # Tạo ID cho đối tượng dựa trên tên và vị trí tương đối (để tránh cảnh báo lặp lại cho cùng 1 vật thể)
                            # object_instance_id = f"{class_name_o}_{int(mid_x_norm_o*10)}_{int(mid_y_norm_o*10)}"
                            
                            # Chỉ cảnh báo nếu đối tượng này (tên lớp) chưa bị tắt và chưa cảnh báo đủ số lần
                            if class_name_o not in suppressed_objects and warning_counts.get(class_name_o, 0) < MAX_WARNINGS_PER_OBJECT:
                                # if object_instance_id not in detected_objects_warnings or \
                                #    (current_time - detected_objects_warnings.get(object_instance_id, 0)) > ALERT_COOLDOWN * 2: # Cooldown dài hơn cho cùng 1 instance

                                    # Kiểm tra lại logic warning_counts và center_objects_history
                                    # Hàm detect_center_objects đã xử lý việc này, có thể không cần logic warning ở đây nữa
                                    # mà chỉ dựa vào detect_center_objects
                                    pass # Để detect_center_objects xử lý cảnh báo

                if detected_object_names_this_frame:
                    announce_objects(detected_object_names_this_frame) # Thông báo chung các loại đối tượng
                if objects_data_for_center_warning:
                    detect_center_objects(objects_data_for_center_warning) # Xử lý cảnh báo đối tượng ở trung tâm

                cv2.putText(display_frame, f'Detected: {len(objects_data_for_center_warning)} objects', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Hiển thị thông tin chung
            mode_names = ["Phat hien Vat the", "Phat hien Tien"]
            mode_text = mode_names[system_mode]
            cv2.putText(display_frame, f'Che do: {mode_text}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if system_mode == 0 : # Chỉ hiển thị cooldown cho object detection
                time_since_last_obj_ann = current_time - last_announcement_time
                next_ann_in = max(0, ANNOUNCEMENT_COOLDOWN - int(time_since_last_obj_ann))
                cv2.putText(display_frame, f'Thong bao sau: {next_ann_in}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Vision-I: He thong Ho tro Thi luc', cv2.resize(display_frame, (800, 600)))
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # Phím ESC
                is_running = False
                break
        
        except Exception as e:
            print(f"Lỗi trong vòng lặp chính: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    # Dọn dẹp
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    speak("Vision-I system shutting down. Goodbye.", priority=True)
    print("Hệ thống đã tắt.")
    # Chờ các luồng phụ (như keyboard_listener, tts_thread) kết thúc nếu cần
    # Tuy nhiên, với daemon=True, chúng sẽ tự thoát.
    # keyboard_thread.join() # Không cần thiết nếu daemon=True


if __name__ == "__main__":
    # Ghi đè sys.stdout và sys.stderr để tránh lỗi Unicode khi in ra console trên một số hệ thống
    # sys.stdout = open(os.devnull, 'w', encoding='utf-8') # Chuyển hướng stdout
    # sys.stderr = open(os.devnull, 'w', encoding='utf-8') # Chuyển hướng stderr
    # Hoặc cấu hình console của bạn để hỗ trợ UTF-8 (ví dụ: chcp 65001 trên Windows)

    if len(sys.argv) > 1 and sys.argv[1] == "--create-shortcut":
        create_desktop_shortcut()
    else:
        run_vision_system()