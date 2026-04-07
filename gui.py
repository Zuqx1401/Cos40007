# ============================================================================
# COS40007 - PHÁT HIỆN RÁC THẢI VEN ĐƯỜNG - NHÓM 10
# Module hiển thị và GUI (GUI.PY)
# ============================================================================

"""
MODULE HIỂN THỊ VÀ GIAO DIỆN (GUI & DISPLAY)
=============================================
Module này chịu trách nhiệm:
- Vẽ khung bao (bounding boxes) xung quanh các đối tượng phát hiện được
- Hiển thị nhãn (label) tên lớp và độ tin cậy (confidence score)
- Quản lý cửa sổ hiển thị
- Xử lý các sự kiện bàn phím
"""

import cv2
import numpy as np
from config import (
    BOUNDING_BOX_THICKNESS, FONT_SCALE, TEXT_THICKNESS, TEXT_PADDING,
    CLASS_COLORS, WINDOW_TITLE, EXIT_KEY_CODE
)

# ============================================================================
# Hàm vẽ khung bao và thông tin
# ============================================================================

def draw_bounding_boxes(frame, results):
    """
    Vẽ các khung bao (bounding boxes) và thông tin dự đoán lên khung hình.
    
    Tham số:
        frame (numpy array): Khung hình gốc từ camera
        results: Kết quả phát hiện từ mô hình YOLOv5
        
    Returns:
        frame (numpy array): Khung hình sau khi vẽ các khung bao
    """
    
    try:
        # Kiểm tra nếu không có kết quả (YOLOv11 trả về một danh sách các đối tượng Results)
        if results is None or len(results) == 0:
            return frame
        
        # Lấy kết quả đầu tiên từ danh sách
        result = results[0]
        boxes = result.boxes  # Đối tượng chứa thông tin khung bao của YOLOv11
        
        if len(boxes) == 0:
            return frame
        
        # Duyệt qua từng khung bao được phát hiện
        for box in boxes:
            # Lấy tọa độ x1, y1, x2, y2
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            
            # Lấy độ tin cậy và tên lớp
            confidence = float(box.conf[0])
            class_idx = int(box.cls[0])
            class_name = result.names[class_idx]
            
            color = CLASS_COLORS.get(class_name, (0, 255, 0))
            
            # Vẽ khung bao (rectangle)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOUNDING_BOX_THICKNESS)
            
            # Tạo văn bản hiển thị: "class_name confidence%"
            label_text = f"{class_name} {confidence:.2f}"
            
            # Lấy kích thước của văn bản để vẽ nền cho chữ
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label_text, font, FONT_SCALE, TEXT_THICKNESS)
            text_width = text_size[0][0]
            text_height = text_size[0][1]
            
            # Tính toán vị trí nên để văn bản (trên cạnh trên của khung bao)
            x_text = x1
            y_text = max(y1 - TEXT_PADDING, text_height + TEXT_PADDING)
            
            # Vẽ nền cho văn bản (để dễ đọc hơn)
            cv2.rectangle(
                frame,
                (x_text, y_text - text_height - TEXT_PADDING),
                (x_text + text_width + TEXT_PADDING, y_text + TEXT_PADDING),
                color,
                -1  # -1 có nghĩa là tô đầy (filled)
            )
            
            # Vẽ văn bản (label và confidence)
            cv2.putText(
                frame,
                label_text,
                (x_text + TEXT_PADDING // 2, y_text),
                font,
                FONT_SCALE,
                (255, 255, 255),  # Màu trắng để dễ đọc trên nền tối
                TEXT_THICKNESS
            )
        
        return frame
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi vẽ khung bao: {str(e)}")
        return frame


def display_frame(frame, window_name=WINDOW_TITLE):
    """
    Hiển thị khung hình lên cửa sổ.
    
    Tham số:
        frame (numpy array): Khung hình cần hiển thị
        window_name (str): Tên của cửa sổ
    """
    
    try:
        # Hiển thị khung hình
        cv2.imshow(window_name, frame)
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi hiển thị khung hình: {str(e)}")


def add_info_text(frame, text, position=(10, 30), font_scale=0.5, thickness=1, color=(0, 255, 0)):
    """
    Thêm văn bản thông tin vào khung hình.
    
    Hàm này hữu ích để hiển thị:
    - FPS hiện tại
    - Số lượng đối tượng phát hiện
    - Trạng thái hệ thống
    
    Tham số:
        frame (numpy array): Khung hình
        text (str): Văn bản cần thêm
        position (tuple): Vị trí (x, y) để đặt văn bản
        font_scale (float): Kích thước chữ
        thickness (int): Độ dày chữ
        color (tuple): Màu sắc (B, G, R)
        
    Returns:
        frame (numpy array): Khung hình sau khi thêm văn bản
    """
    
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        return frame
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi thêm văn bản: {str(e)}")
        return frame


def create_window(window_name=WINDOW_TITLE):
    """
    Tạo cửa sổ hiển thị.
    
    Tham số:
        window_name (str): Tên của cửa sổ
    """
    
    try:
        # Tạo cửa sổ mới
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print(f"[INFO] Tạo cửa sổ: {window_name}")
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi tạo cửa sổ: {str(e)}")


def close_window(window_name=WINDOW_TITLE):
    """
    Đóng cửa sổ hiển thị.
    
    Tham số:
        window_name (str): Tên của cửa sổ cần đóng
    """
    
    try:
        cv2.destroyWindow(window_name)
        print(f"[INFO] Đã đóng cửa sổ: {window_name}")
    
    except Exception as e:
        print(f"[WARNING] Lỗi khi đóng cửa sổ: {str(e)}")


def close_all_windows():
    """
    Đóng tất cả các cửa sổ OpenCV.
    
    Hàm này được gọi trước khi thoát chương trình để đảm bảo tất cả
    các cửa sổ được đóng sạch sẽ.
    """
    
    try:
        cv2.destroyAllWindows()
        print("[INFO] Đã đóng tất cả cửa sổ")
    
    except Exception as e:
        print(f"[WARNING] Lỗi khi đóng cửa sổ: {str(e)}")


def wait_key(delay=1):
    """
    Chờ đợi sự kiện bàn phím.
    
    Hàm này cần thiết để:
    - Cho phép OpenCV xử lý sự kiện cửa sổ
    - Kiểm tra xem người dùng có nhấn phím 'q' để thoát không
    
    Tham số:
        delay (int): Thời gian chờ (miligiây)
        
    Returns:
        int: Mã phím được nhấn (-1 nếu không có phím nào được nhấn trong thời gian chờ)
    """
    
    try:
        # Chờ đợi và lấy mã phím được nhấn
        key = cv2.waitKey(delay)
        return key
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi chờ đợi phím: {str(e)}")
        return -1


def select_image_file(initial_dir=None):
    """
    Hiển thị hộp thoại để người dùng chọn tệp hình ảnh từ máy tính.
    
    Tham số:
        initial_dir (str): Thư mục mặc định khi mở hộp thoại.
        
    Returns:
        str: Đường dẫn đến tệp đã chọn, hoặc None nếu người dùng hủy.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()  # Ẩn cửa sổ chính của tkinter
        root.update()     # Cập nhật để đảm bảo cửa sổ ẩn hoàn toàn
        root.attributes("-topmost", True)  # Đưa cửa sổ lên trên cùng
        root.lift()                        # Ưu tiên hiển thị
        
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Chọn ảnh rác thải để nhận diện",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print(f"[ERROR] Không thể mở hộp thoại chọn tệp: {str(e)}")
        return None


def is_exit_key_pressed(key):
    """
    Kiểm tra xem phím thoát có được nhấn không.
    
    Tham số:
        key (int): Mã phím được nhấn
        
    Returns:
        bool: True nếu phím thoát được nhấn, False nếu không
    """
    
    # Kiểm tra xem phím nhấn có phải là 'q' hoặc ESC không
    return key == EXIT_KEY_CODE or key == 27  # 27 là mã ASCII của phím ESC


def get_frame_fps(frame):
    """
    Tính toán FPS (Frames Per Second) từ khung hình.
    
    Hàm này có thể mở rộng để theo dõi hiệu suất thực tế.
    
    Returns:
        int: FPS hiện tại (hoặc giá trị mặc định nếu không thể tính toán)
    """
    
    try:
        # Các giá trị này có thể được cập nhật trong main loop
        return 0  # Placeholder - sẽ được cập nhật từ main.py
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi tính FPS: {str(e)}")
        return 0
