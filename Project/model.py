# ============================================================================
# COS40007 - PHÁT HIỆN RÁC THẢI VEN ĐƯỜNG - NHÓM 10
# Module tải mô hình (MODEL.PY)
# ============================================================================

"""
MODULE TẢI MÔ HÌNH (MODEL LOADING)
==================================
Module này chịu trách nhiệm tải mô hình YOLOv11 từ tệp hoặc tải bản demo.
Nó cung cấp các hàm để khởi tạo mô hình phát hiện đối tượng.
"""

from ultralytics import YOLO
import pathlib
import platform
import os
from pathlib import Path
from config import MODEL_PATH, DEFAULT_MODEL, CLASSES_TO_DETECT

# ============================================================================
# Hàm tải mô hình
# ============================================================================

def load_model():
    """
    Hàm tải mô hình YOLOv5 từ tệp best.pt hoặc mô hình demo.
    
    Quy trình:
    1. Kiểm tra xem tệp best.pt có tồn tại không
    2. Nếu có: tải mô hình từ tệp best.pt (mô hình đã được huấn luyện)
    3. Nếu không: tải bản mô hình mặc định yolov5s từ PyTorch Hub
    
    Returns:
        model: Mô hình YOLOv5 đã tải, sẵn sàng để sử dụng
        
    Raises:
        Exception: Nếu có lỗi khi tải mô hình
    """
    
    try:
        # Sửa lỗi "cannot instantiate 'PosixPath'" khi chạy trên Windows
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
            
        # Ultralytics tự động nhận diện thiết bị (Cuda/CPU)
        print(f"[INFO] Đang khởi tạo YOLOv11...")
        
        # Kiểm tra xem tệp mô hình best(1).pt có tồn tại không
        if os.path.exists(MODEL_PATH):
            print(f"[INFO] Tìm thấy tệp mô hình: {MODEL_PATH}")
            print("[INFO] Đang tải mô hình từ tệp best.pt...")

            # Tải mô hình YOLOv11
            model = YOLO(MODEL_PATH)
            print("[SUCCESS] Tải mô hình thành công từ best.pt!")
            
        else:
            # Nếu không tìm thấy best.pt, sử dụng mô hình demo
            print(f"[WARNING] Không tìm thấy {MODEL_PATH}")
            print(f"[INFO] Đang tải mô hình demo {DEFAULT_MODEL}...")
            
            # Tải mô hình yolo11n
            model = YOLO(DEFAULT_MODEL)
            print(f"[SUCCESS] Tải mô hình demo {DEFAULT_MODEL} thành công!")
        
        return model
    
    except Exception as e:
        # Nếu có lỗi, in ra thông báo lỗi
        print(f"[ERROR] Lỗi khi tải mô hình: {str(e)}")
        raise


def set_model_classes(model):
    """
    Đặt các lớp mà mô hình cần phát hiện.
    
    Lưu ý: Nếu sử dụng yolov5s (COCO), mô hình sẽ tự động phát hiện tất cả 80 lớp COCO.
    Tham số CLASSES_TO_DETECT chỉ để tham khảo, lọc kết quả có thể được thực hiện
    trong module GUI (gui.py) nếu cần.
    
    Parameters:
        model: Mô hình YOLOv5 đã tải
        
    Returns:
        model: Mô hình sau khi cấu hình lớp
    """
    
    try:
        # Ultralytics YOLOv11 sử dụng một cấu trúc khác để lọc lớp
        # Chúng ta sẽ lưu danh sách ID cần lọc vào một thuộc tính tùy chỉnh
        model.filter_classes = None

        # Kiểm tra nếu là mô hình demo COCO
        if len(model.names) >= 80:
            print("[INFO] Đang dùng mô hình demo COCO. Thiết lập bộ lọc lớp tương ứng...")
            
            # Ánh xạ các lớp từ config sang ID của COCO
            # COCO ID: 57: couch, 59: bed (nệm), 62: tv, 63: laptop, 64: mouse, 65: remote, 67: cell phone, 77: teddy bear (toys)
            coco_mapping = {
                "couch": 57,
                "mattress": 59,
                "electrical_goods": [62, 63],
                "mouse": 64,
                "cell phone": 67,
                "toys": 77
            }
            
            filter_indices = []
            for class_name in CLASSES_TO_DETECT:
                if class_name in coco_mapping:
                    val = coco_mapping[class_name]
                    if isinstance(val, list): filter_indices.extend(val)
                    else: filter_indices.append(val)
            
            # Thêm remote (65) vào bộ lọc vì nó hay bị nhầm với mouse/phone
            filter_indices.append(65) 
            
            model.filter_classes = filter_indices
            print(f"[INFO] Đã kích hoạt bộ lọc COCO cho các ID: {filter_indices}")
        else:
            # Sử dụng tên lớp có sẵn trong mô hình custom (best.pt) để tránh lỗi lệch index
            print(f"[INFO] Đã tải mô hình custom với các lớp: {model.names}")
            
        return model
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi cấu hình lớp: {str(e)}")
        raise


def warmup_model(model, img_size=640):
    """
    Khởi động mô hình bằng cách chạy một lần suy luận trước.
    
    Điều này giúp:
    - Cấp phát bộ nhớ GPU
    - Giảm độ trễ của lần suy luận đầu tiên
    
    Parameters:
        model: Mô hình YOLOv11
        img_size: Kích thước ảnh đầu vào (mặc định 640x640)
    """
    
    try:
        import numpy as np
        
        print("[INFO] Đang khởi động mô hình...")
        
        # Tạo một ảnh dummy (ảnh giả) để khởi động
        import numpy as np
        dummy_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Chạy mô hình một lần
        _ = model.predict(dummy_img, verbose=False)
        
        print("[SUCCESS] Khởi động mô hình hoàn tất!")
    
    except Exception as e:
        print(f"[WARNING] Lỗi trong quá trình khởi động: {str(e)}")
        # Không dừng lại nếu khởi động thất bại
