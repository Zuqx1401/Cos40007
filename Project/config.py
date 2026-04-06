# ============================================================================
# COS40007 - PHÁT HIỆN RÁC THẢI VEN ĐƯỜNG - NHÓM 10
# File cấu hình (CONFIG.PY)
# 
# Thành viên nhóm:
# - Võ Nam Thịnh (103806638)
# - Nguyễn Duy Quang
# - Trần Việt Dũng
# - Lê Nguyễn Thái Sơn
# ============================================================================

"""
MODULE CẤU HÌNH (CONFIG)
========================
Tệp này chứa tất cả các cấu hình cần thiết cho hệ thống phát hiện rác thải.
Các tham số được định nghĩa tại đây để dễ dàng tùy chỉnh mà không cần chỉnh sửa
toàn bộ mã nguồn.
"""

# ============================================================================
# 1. CẤU HÌNH MÔ HÌNH (MODEL CONFIGURATION)
# ============================================================================

# Đường dẫn đến tệp mô hình đã huấn luyện
# Nếu không tìm thấy, hệ thống sẽ tự động tải bản YOLOv5s làm demo
MODEL_PATH = r"D:\COS40007\Project\best (1).pt"

# Bản sao dự phòng - mô hình YOLOv11 mặc định (nếu best.pt không tồn tại)
DEFAULT_MODEL = "yolo11n.pt"

# Ngưỡng độ tin cậy tối thiểu (Confidence threshold)
# Chỉ nhận diện các đối tượng có độ tin cậy >= ngưỡng này
CONFIDENCE_THRESHOLD = 0.3

# Ngưỡng tương tự (IoU - Intersection over Union)
# Dùng để loại bỏ các hộp bao trùng lặp
IOU_THRESHOLD = 0.45

# ============================================================================
# 2. CẤU HÌNH LỚP PHÁT HIỆN (CLASSES CONFIGURATION)
# ============================================================================

# Danh sách các lớp cần phát hiện
# Bao gồm: 4 lớp rác thải + điện thoại + chuột laptop
CLASSES_TO_DETECT = [
    "mattress",           # Nệm giường
    "couch",              # Sofa
    "electrical_goods",   # Đồ điện tử
    "toys",               # Đồ chơi
]

# Số lượng lớp cần phát hiện
NUM_CLASSES = len(CLASSES_TO_DETECT)

# Màu sắc cho mỗi lớp (Format BGR vì OpenCV sử dụng BGR, không phải RGB)
# Sử dụng cho việc vẽ các khung bao (bounding box)
CLASS_COLORS = {
    "mattress": (0, 255, 0),          # Xanh lá
    "couch": (255, 0, 0),             # Xanh dương
    "electrical_goods": (0, 165, 255), # Cam
    "toys": (255, 0, 255)             # Tím
}

# ============================================================================
# 3. CẤU HÌNH CAMERA (CAMERA CONFIGURATION)
# ============================================================================

# ID của camera (0 = camera mặc định/built-in camera của laptop)
CAMERA_ID = 0

# Độ rộng khung hình (frame width) - 1280x720 là cân bằng tốt giữa chất lượng và tốc độ
FRAME_WIDTH = 1280

# Độ cao khung hình (frame height)
FRAME_HEIGHT = 720

# Tốc độ khung hình (FPS - Frames per Second) mục tiêu
TARGET_FPS = 30

# ============================================================================
# 4. CẤU HÌNH GIAO DIỆN (GUI CONFIGURATION)
# ============================================================================

# Tiêu đề cửa sổ hiển thị
WINDOW_TITLE = "Group 10 - Roadside Rubbish Detection System"

# Độ dày đường viền của khung bao (bounding box)
BOUNDING_BOX_THICKNESS = 2

# Kích thước văn bản (label và confidence score)
FONT_SCALE = 0.6

# Độ dày của text
TEXT_THICKNESS = 2

# Khoảng cách giữa văn bản và khung bao (padding)
TEXT_PADDING = 5

# ============================================================================
# 5. CẤU HÌNH PHÍM TẮT (KEYBOARD SHORTCUTS)
# ============================================================================

# Phím để thoát hệ thống an toàn
EXIT_KEY = 'q'
EXIT_KEY_CODE = ord('q')  # Mã ASCII của phím 'q'
UPLOAD_KEY_CODE = ord('u') # Mã ASCII của phím 'u' để tải ảnh

# ============================================================================
# 6. CẤU HÌNH KHÁC (MISCELLANEOUS)
# ============================================================================

# Thời gian chờ trước khi thoát sau khi nhấn phím thoát (miligiây)
EXIT_TIMEOUT = 1

# Thư mục để lưu trữ các ảnh kết quả sau khi nhận diện
RESULTS_DIR = "result"

# Thư mục mặc định khi mở hộp thoại chọn ảnh (Bảng 1)
DEFAULT_INPUT_DIR = None

# Chế độ debug - hiển thị thông tin chi tiết
DEBUG_MODE = False
