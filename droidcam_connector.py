import cv2
import time

def get_droidcam_capture():
    """Hàm nâng cao để kết nối DroidCam với khả năng thử lại và xử lý lỗi tốt hơn"""
    print("Đang tìm và kết nối với camera...")
    
    # Thử các URL DroidCam với thời gian chờ
    for attempt in range(3):
        print(f"\nNỗ lực kết nối DroidCam {attempt+1}/3:")
        
        # Danh sách các URL DroidCam thông dụng để thử
        droidcam_urls = [
            "http://192.168.0.113:4747/video",
            "http://192.168.0.113:4747/videofeed",
            "http://192.168.0.113:4747/mjpegfeed",
            "http://127.0.0.1:4747/video",
            "http://localhost:4747/video",
            "http://192.168.0.113:8080/video"  # Một số phiên bản DroidCam sử dụng cổng 8080
        ]
        
        for url in droidcam_urls:
            try:
                print(f"Đang thử kết nối với {url}")
                cap = cv2.VideoCapture(url)
                
                # Đợi kết nối thiết lập
                time.sleep(2)
                
                if cap.isOpened():
                    # Kiểm tra xem có thể đọc khung hình không
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"Kết nối thành công với {url}")
                        return cap
                    else:
                        print("Mở được nhưng không đọc được khung hình")
                        cap.release()
                else:
                    print(f"Không thể mở kết nối tới {url}")
            except Exception as e:
                print(f"Lỗi khi kết nối {url}: {e}")
        
        # Thử các camera cục bộ
        print("Thử tìm camera cục bộ...")
        for i in range(3):  # Thử tối đa 3 camera
            try:
                cap = cv2.VideoCapture(i)
                
                # Đợi kết nối thiết lập
                time.sleep(1)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"Đã tìm thấy camera cục bộ ở chỉ số {i}")
                        return cap
                    else:
                        print(f"Không đọc được khung hình từ camera {i}")
                        cap.release()
            except Exception as e:
                print(f"Lỗi khi kiểm tra camera {i}: {e}")
        
        # Đợi một chút trước khi thử lại
        print(f"Không tìm thấy camera trong lần thử {attempt+1}, đợi trước khi thử lại...")
        time.sleep(3)
    
    # Nếu sau tất cả các nỗ lực, vẫn không tìm thấy camera, thử camera mặc định
    print("\nSử dụng camera mặc định (chỉ số 0) sau nhiều lần thử không thành công")
    return cv2.VideoCapture(0)

# Để kiểm tra nếu file được chạy trực tiếp
if __name__ == "__main__":
    print("Kiểm tra kết nối DroidCam...")
    cap = get_droidcam_capture()
    
    if cap.isOpened():
        print("Kết nối camera thành công! Đọc 10 khung hình để kiểm tra...")
        
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                # Hiển thị khung hình
                cv2.imshow('DroidCam Test', frame)
                
                # Đợi 100ms, cho phép hiển thị và xử lý sự kiện
                key = cv2.waitKey(100)
                if key == 27:  # Phím ESC
                    break
                    
                print(f"Đã đọc khung hình {i+1}")
            else:
                print(f"Không đọc được khung hình {i+1}")
            
            # Đợi 0.5 giây giữa các lần đọc
            time.sleep(0.5)
        
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()
        print("Kiểm tra kết nối hoàn tất")
    else:
        print("Không thể kết nối với bất kỳ camera nào")