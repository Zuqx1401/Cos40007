# ============================================================================
# COS40007 - PHÁT HIỆN RÁC THẢI VEN ĐƯỜNG - NHÓM 10
# File chính (MAIN.PY)
# 
# THÀNH VIÊN NHÓM:
# - Võ Nam Thịnh (103806638)
# - Nguyễn Duy Quang
# - Trần Việt Dũng
# - Lê Nguyễn Thái Sơn
# ============================================================================

"""
HỆ THỐNG PHÁT HIỆN RÁC THẢI VEN ĐƯỜNG (MAIN)
==============================================

Đây là tệp chính (entry point) của hệ thống.
Chương trình sẽ chạy vòng lặp chính tại đây, quản lý:
1. Ghi hình từ camera
2. Suy luận (inference) từ mô hình YOLOv5
3. Vẽ kết quả lên khung hình
4. Hiển thị kết quả
5. Xử lý thoát an toàn

Để chạy chương trình:
    python main.py
    
Để thoát chương trình:
    Nhấn phím ESC
"""

# ============================================================================
# BƯỚC 1: NHẬP CÁC THƯ VIỆN CẦN THIẾT
# ============================================================================

import sys
import time
import cv2
import torch
import numpy as np
from pathlib import Path

# Nhập các module tự viết
from config import (
    WINDOW_TITLE, EXIT_TIMEOUT, CONFIDENCE_THRESHOLD, 
    IOU_THRESHOLD, DEBUG_MODE, UPLOAD_KEY_CODE, RESULTS_DIR, DEFAULT_INPUT_DIR,
    FRAME_WIDTH
)
from model import load_model, set_model_classes, warmup_model
from camera import initialize_camera, capture_frame, release_camera, get_camera_properties
from gui import (
    draw_bounding_boxes, display_frame, create_window, close_window, 
    close_all_windows, wait_key, is_exit_key_pressed, add_info_text, select_image_file
)

# ============================================================================
# BƯỚC 2: ĐỊNH NGHĨA CÁC BIẾN TOÀN CỤC
# ============================================================================

# Biến để lưu thời gian và tính FPS
start_time = None
frame_count = 0
fps = 0

# ============================================================================
# BƯỚC 3: ĐỊNH NGHĨA CÁC HÀM PHỤ TRỢ
# ============================================================================

def calculate_fps(frame_number, elapsed_time):
    """
    Tính toán FPS (Frames Per Second) - số khung hình xử lý mỗi giây.
    
    Công thức: FPS = số khung hình / thời gian trôi (giây)
    
    Tham số:
        frame_number (int): Số khung hình đã xử lý
        elapsed_time (float): Thời gian trôi (giây)
        
    Returns:
        float: FPS hiện tại
    """
    
    if elapsed_time > 0:
        return frame_number / elapsed_time
    return 0


def perform_inference(model, frame):
    """
    Thực hiện suy luận (inference) trên một khung hình.
    
    Bước:
    1. Đưa khung hình vào mô hình
    2. Mô hình sẽ chia khung hình thành lưới và phát hiện các đối tượng
    3. Trả về kết quả phát hiện
    
    Tham số:
        model: Mô hình YOLOv11
        frame (numpy array): Khung hình từ camera
        
    Returns:
        results: Kết quả phát hiện từ mô hình
    """
    
    try:
        # Thực hiện suy luận với YOLOv11 (Ultralytics)
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=getattr(model, 'filter_classes', None),
            verbose=False
        )
        
        return results
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi thực hiện inference: {str(e)}")
        return None


def handle_image_upload(model):
    """
    Xử lý việc tải ảnh lên và chạy nhận diện trên ảnh đó.
    Cho phép chọn nhiều ảnh liên tục cho đến khi người dùng hủy.
    """
    while True:
        print("\n[INFO] Đang mở hộp thoại chọn tệp... (Nhấn 'Cancel' để quay lại menu chính)")
        
        # Sử dụng thư mục mặc định từ tệp config nếu có, nếu không để OS tự quyết định
        initial_dir = None
        if DEFAULT_INPUT_DIR and Path(DEFAULT_INPUT_DIR).exists():
            initial_dir = str(Path(DEFAULT_INPUT_DIR).absolute())
        
        image_path = select_image_file(initial_dir=initial_dir)
        
        if not image_path:
            print("[INFO] Quay lại menu chính.")
            break
            
        # Đọc hình ảnh
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Không thể đọc tệp ảnh tại: {image_path}")
            continue

        # Resize ảnh nếu cần để xử lý đồng nhất
        h, w = frame.shape[:2]
        if w > FRAME_WIDTH:
            scale = FRAME_WIDTH / w
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)

        annotated_frame = frame.copy()
        results = perform_inference(model, annotated_frame)
        num_detections = count_detections(results)
        annotated_frame = draw_bounding_boxes(annotated_frame, results)
        
        status_folder = "has_rubbish" if num_detections > 0 else "no_rubbish"
        output_dir = Path(RESULTS_DIR) / status_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"detected_{Path(image_path).name}"
        cv2.imwrite(str(output_path), annotated_frame)
        
        print(f"[SUCCESS] Đã xử lý xong ảnh: {Path(image_path).name}")
        print(f"[INFO] Kết quả (Detections: {num_detections}) đã lưu tại: {output_path}")

        # Hiển thị cửa sổ kết quả (Bảng thứ 2) và đợi người dùng xem xong nhấn phím
        create_window(WINDOW_TITLE)
        display_frame(annotated_frame, WINDOW_TITLE)
        
        print("[HƯỚNG DẪN] Nhấn phím bất kỳ hoặc ESC: Chọn ảnh khác | GIỮ LÌ ESC 6 GIÂY: THOÁT HỆ THỐNG")
        
        should_quit_all = False
        start_hold = None
        
        while True:
            # Giảm thời gian chờ xuống 50ms để bắt sự kiện phím chính xác hơn
            key = cv2.waitKey(50) & 0xFF
            
            if key == 27:  # Phím ESC đang được nhấn giữ
                if start_hold is None:
                    start_hold = time.time()
                
                elapsed = time.time() - start_hold
                sys.stdout.write(f"\r[THOÁT] Đang giữ ESC để đóng hệ thống: {elapsed:.1f}/6.0s ")
                sys.stdout.flush()
                
                if elapsed >= 6.0:
                    should_quit_all = True
                    break
            elif key != 255 or (start_hold is not None):  # Thả ESC hoặc nhấn phím khác
                sys.stdout.write("\r" + " " * 60 + "\r")
                break
                
        close_window(WINDOW_TITLE)
        if should_quit_all:
            return True


def count_detections(results):
    """
    Đếm số lượng đối tượng được phát hiện trong một khung hình.
    
    Tham số:
        results: Kết quả từ mô hình
        
    Returns:
        int: Số lượng đối tượng phát hiện được
    """
    
    try:
        if results is None or len(results) == 0:
            return 0
        return len(results[0].boxes)
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi đếm phát hiện: {str(e)}")
        return 0


# ============================================================================
# BƯỚC 4: HÀM CHÍNH (MAIN FUNCTION)
# ============================================================================

def main():
    """
    Hàm chính - chứa vòng lặp chính của chương trình.
    
    Quy trình:
    1. Khởi tạo camera
    2. Tải mô hình
    3. Vòng lặp chính:
        a. Ghi khung hình từ camera
        b. Thực hiện suy luận
        c. Vẽ kết quả
        d. Hiển thị khung hình
        e. Kiểm tra phím thoát
    4. Thoát an toàn
    """
    
    global start_time, frame_count, fps
    
    # Khởi tạo các biến
    camera = None
    model = None
    start_time = time.time()
    frame_count = 0

    try:
        print("\n" + "="*70)
        print("HỆ THỐNG PHÁT HIỆN RÁC THẢI VEN ĐƯỜNG - NHÓM 10")
        print("="*70)
        
        # Tải mô hình một lần duy nhất khi khởi động
        print("\n[BƯỚC 1] TẢI MÔ HÌNH YOLOv5")
        model = load_model()
        model = set_model_classes(model)
        warmup_model(model)

        while True:
            print("\n" + "-"*70)
            print("MENU CHỌN CHẾ ĐỘ HOẠT ĐỘNG:")
            print("1. Chạy Camera thời gian thực")
            print("2. Tải ảnh lên từ máy tính (Image Upload)")
            print("3. Thoát hệ thống")
            
            mode = input("\nNhập lựa chọn của bạn: ").strip().lower()

            if mode == '3' or mode == 'q':
                print("[INFO] Đang thoát chương trình...")
                break

            if mode == '2':
                # Nếu hàm trả về True (do giữ ESC 6s), thoát hẳn main
                if handle_image_upload(model):
                    return
                continue

            if mode == '1':
                print("\n[BƯỚC 2] KHỞI TẠO CAMERA")
                try:
                    camera = initialize_camera()
                except Exception as e:
                    print(f"\n[ERROR] Không thể khởi tạo camera: {str(e)}")
                    continue
                
                create_window(WINDOW_TITLE)
                
                print("\n[BƯỚC 3] BẮT ĐẦU NHẬN DIỆN CAMERA")
                print("Nhấn 'q' hoặc ESC để QUAY LẠI MENU CHÍNH")
                
                start_time = time.time()
                frame_count = 0
                should_exit_camera = False
                
                while not should_exit_camera:
                    ret, frame = capture_frame(camera)
                    if not ret or frame is None:
                        continue
                    
                    results = perform_inference(model, frame)
                    
                    if results is not None:
                        frame = draw_bounding_boxes(frame, results)
                        num_detections = count_detections(results)
                    else:
                        num_detections = 0
                    
                    frame_count += 1
                    fps = calculate_fps(frame_count, time.time() - start_time)
                    
                    fps_text = f"FPS: {fps:.1f} | Detections: {num_detections}"
                    frame = add_info_text(frame, fps_text, position=(10, 30), font_scale=0.7, thickness=2)
                    frame = add_info_text(frame, "Press ESC to back to menu", position=(10, frame.shape[0] - 10), 
                                         font_scale=0.5, thickness=1, color=(0, 255, 255))
                    
                    display_frame(frame, WINDOW_TITLE)
                    
                    key = wait_key(1)
                    if is_exit_key_pressed(key):
                        print("[INFO] Quay lại menu chính.")
                        should_exit_camera = True
                
                # Giải phóng tài nguyên camera sau mỗi lần dùng xong để quay lại menu
                total_time = time.time() - start_time
                print(f"\n[THỐNG KÊ CAMERA] FPS trung bình: {frame_count/total_time:.2f}")
                release_camera(camera)
                close_all_windows()
            else:
                print("[WARNING] Lựa chọn không hợp lệ, vui lòng chọn lại.")

    # ========================================================================
    # KHỐI XỬ LÝ LỖI
    # ========================================================================
    
    except KeyboardInterrupt:
        # Xử lý khi người dùng nhấn Ctrl+C
        print("\n[INFO] Người dùng ngắt lại chương trình (Ctrl+C)")
    
    except Exception as e:
        # Xử lý các lỗi không mong muốn
        print(f"\n[FATAL ERROR] Lỗi không mong muốn: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # ====================================================================
        # GIẢI PHÓNG CÁC TÀI NGUYÊN
        # ====================================================================
        
        print("\n[BƯỚC 6] GIẢI PHÓNG TÀI NGUYÊN")
        print("-" * 70)
        
        # Giải phóng camera
        if camera is not None:
            release_camera(camera)
        
        # Đóng tất cả các cửa sổ OpenCV
        try:
            close_all_windows()
        except Exception as e:
            print(f"[WARNING] Lỗi khi đóng cửa sổ: {str(e)}")
        
        print("\n" + "="*70)
        print("HỆ THỐNG ĐÃ ĐÓNG AN TOÀN")
        print("="*70 + "\n")


# ============================================================================
# BƯỚC 5: CHẠY CHƯƠNG TRÌNH
# ============================================================================

if __name__ == "__main__":
    """
    Điểm bắt đầu của chương trình.
    
    __name__ == "__main__" đảm bảo rằng hàm main() chỉ được gọi khi
    tệp này là tệp chính được chạy, không phải khi được import bởi tệp khác.
    """
    
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Lỗi Fatal: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
