🚗 Roadside Rubbish Detection System - Group 10
COS40007 - Artificial Intelligence for Real-world Applications > Swinburne University of Technology | Vietnam Campus

📝 Giới Thiệu Dự Án
Dự án tập trung vào việc giải quyết vấn đề xả rác bừa bãi tại các khu vực đô thị bằng cách sử dụng Deep Learning. Hệ thống tự động nhận diện, phân loại và thống kê các loại rác thải ven đường từ camera thời gian thực hoặc hình ảnh tải lên.

✨ Tính Năng Nổi BậtPhát hiện Đa lớp: 
    Hỗ trợ nhận diện chính xác 18 loại vật thể rác thải khác nhau.
    Dashboard Thông minh: Giao diện Web trực quan (Streamlit) tích hợp biểu đồ thống kê thời gian thực.
    Logic Thống kê Session: Tính toán tỉ lệ "Sạch" của môi trường dựa trên tổng số lượt quét (Total Scans) và số lần không phát hiện rác (Clear/Not Rubbish).
    Hybrid Deployment: Chạy tốt trên cả Local (máy cá nhân) và Cloud (Streamlit Share).

📊 Danh Mục Phát Hiện (18 Classes)
    Hệ thống được huấn luyện đặc biệt để nhận diện các nhóm đối tượng:
    NhómCác Lớp Đối Tượng (Classes):
    Nội thất lớn:
        mattress, couch, wardrobe, chair, cupboard, desk
    Điện tử & Gia dụng:
        tivi, fan, washing_machine, can, box
    Rác thải sinh hoạt:
        rubbish, trash_can, objects, toys, stroller
    Khác:
        tire, people (đối tượng gây ra hành vi xả rác)

🛠️ Yêu Cầu Kỹ Thuật
    Cấu Hình Đề Nghị
    OS: Windows 10/11, Ubuntu 20.04+, hoặc MacOS.
    Hardware: RAM 8GB trở lên. (Hỗ trợ NVIDIA GPU CUDA để đạt FPS cao).
    Môi trường: Python 3.9 - 3.11.

    Thư Viện Chính
   
    tultralytics >= 8.1.0        # YOLOv11 Engine
    streamlit >= 1.32.0         # Dashboard UI
    opencv-python-headless      # Image Processing (Cloud Optimized)
    pandas & matplotlib        # Data Analysis & Charts

📦 Hướng Dẫn Cài Đặt
1. Sao chép Dự án
Bash
git clone https://github.com/Zuqx1401/roadside-rubbish-detection-group10.git
cd roadside-rubbish-detection-group10

2. Thiết lập Môi trườngBash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (macOS/Linux)
source venv/bin/activate

3. Cài đặt Thư viện
Bash
pip install -r requirements.txt

▶️ Khởi Chạy Hệ Thống
    Dự án hỗ trợ 2 chế độ vận hành chính:
    
    Chế độ Dashboard 
    Giao diện hiện đại, đầy đủ biểu đồ và quản lý lịch sử (Logs).
    Bash
    python -m streamlit run app.py

    Chế độ Terminal (Chỉ chạy Camera)
    Dành cho việc kiểm tra nhanh hiệu suất mô hình trên dòng lệnh.
    Bash
    python main.py

📁 Cấu Trúc Mã Nguồn├── .streamlit/          # Cấu hình giao diện Streamlit
├── app.py               # 🚀 File chính: Chạy Dashboard UI
├── config.py            # Cấu hình tham số (Threshold, Colors, Paths)
├── model.py             # Module quản lý việc tải mô hình AI
├── gui.py               # Module xử lý vẽ Bounding Box & Labels
├── best.pt              # Trọng số mô hình YOLOv11 đã huấn luyện
├── requirements.txt     # Danh sách thư viện cần thiết
└── packages.txt         # Thư viện hệ thống (dùng cho Deploy Cloud)

⚙️ Tùy Chỉnh (Configuration)
Người dùng có thể điều chỉnh hệ thống thông qua menu System Settings trên Dashboard hoặc can thiệp trực tiếp vào config.
py:

Confidence Threshold: Điều chỉnh độ nhạy của AI (Mặc định: 0.3).
IOU Threshold: Điều chỉnh độ chồng lấp giữa các khung bao.
Camera ID: Thay đổi nguồn camera (0 cho camera tích hợp, 1 cho webcam rời).

👥 Thành Viên Thực Hiện (Nhóm 10)

