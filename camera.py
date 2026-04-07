# ============================================================================
# COS40007 - PHÁT HIỆN RÁC THẢI VEN ĐƯỜNG - NHÓM 10
# Module xử lý camera (CAMERA.PY)
# ============================================================================

"""
MODULE XỬ LÝ CAMERA (CAMERA HANDLING)
=====================================
Module này chịu trách nhiệm:
- Kết nối với camera của laptop
- Quản lý việc ghi hình từ camera
- Xử lý các lỗi liên quan đến camera
"""

import cv2
from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS

# ============================================================================
# Hàm khởi tạo camera
# ============================================================================

def initialize_camera():
    """
    Khởi tạo và cấu hình camera của laptop.
    
    Quy trình:
    1. Mở kết nối đến camera (mã 0 = camera mặc định)
    2. Cấu hình độ phân giải (resolution)
    3. Cấu hình tốc độ khung hình (FPS)
    4. Kiểm tra xem camera có hoạt động bình thường không
    
    Returns:
        cap: Đối tượng VideoCapture đại diện cho camera
        
    Raises:
        RuntimeError: Nếu không thể mở camera
    """
    
    try:
        print(f"[INFO] Đang mở camera (ID: {CAMERA_ID})...")
        
        # Tạo đối tượng VideoCapture để quản lý camera
        cap = cv2.VideoCapture(CAMERA_ID)
        
        # Kiểm tra xem camera đã được mở thành công không
        if not cap.isOpened():
            raise RuntimeError(f"[ERROR] Không thể mở camera ID {CAMERA_ID}")
        
        print("[SUCCESS] Mở camera thành công!")
        
        # Cấu hình độ phân giải (resolution) của camera
        # Thiết lập chiều rộng khung hình
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        
        # Thiết lập chiều cao khung hình
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # Cấu hình tốc độ khung hình (FPS)
        # Thiết lập số khung hình cần chụp mỗi giây
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        # Cấu hình độ trễ (buffer size)
        # Giảm buffer để có ảnh real-time hơn
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"[INFO] Cấu hình camera: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {TARGET_FPS} FPS")
        
        return cap
    
    except Exception as e:
        print(f"[ERROR] Lỗi khởi tạo camera: {str(e)}")
        raise


def capture_frame(cap):
    """
    Ghi một khung hình từ camera.
    
    Parameters:
        cap: Đối tượng VideoCapture của camera
        
    Returns:
        tuple: (success, frame)
            - success (bool): True nếu ghi thành công, False nếu không
            - frame (numpy array): Khung hình được ghi, None nếu thất bại
    """
    
    try:
        # Ghi một khung hình từ camera
        ret, frame = cap.read()
        
        # Kiểm tra xem ghi khung hình có thành công không
        if not ret or frame is None:
            print("[WARNING] Không thể ghi khung hình từ camera")
            return False, None
        
        return True, frame
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi ghi khung hình: {str(e)}")
        return False, None


def release_camera(cap):
    """
    Giải phóng tài nguyên camera.
    
    Hàm này PHẢI được gọi trước khi thoát hệ thống để:
    - Đóng kết nối camera
    - Giải phóng tài nguyên
    - Tránh xung đột khi chạy lại chương trình
    
    Parameters:
        cap: Đối tượng VideoCapture cần giải phóng
    """
    
    try:
        if cap is not None and cap.isOpened():
            cap.release()
            print("[INFO] Đã đóng camera thành công")
    
    except Exception as e:
        print(f"[WARNING] Lỗi khi đóng camera: {str(e)}")


def get_camera_properties(cap):
    """
    Lấy các thông số của camera hiện tại.
    
    Hàm này hữu ích cho việc debug hoặc kiểm tra cấu hình camera.
    
    Parameters:
        cap: Đối tượng VideoCapture
        
    Returns:
        dict: Dictionary chứa các thông tin về camera
    """
    
    try:
        properties = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        return properties
    
    except Exception as e:
        print(f"[WARNING] Lỗi khi lấy thông số camera: {str(e)}")
        return {}
