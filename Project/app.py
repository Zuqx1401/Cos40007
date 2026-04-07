import sys
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import streamlit as st
import cv2
import importlib
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
from datetime import datetime
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt 

# Import configurations
from config import (
    MODEL_PATH, IOU_THRESHOLD, 
    CLASS_COLORS, WINDOW_TITLE
)

# ==========================================
# 1. PAGE CONFIG & ENHANCED CSS (Fix Button Visibility)
# ==========================================
st.set_page_config(page_title=WINDOW_TITLE, page_icon="🚗", layout="wide")

st.markdown("""
    <style>
        /* --- Existing Force Light Theme styles --- */
        .stApp { background-color: #F0F2F6 !important; color: #1A233E !important; }
        [data-testid="stSidebar"] { background-color: #1A233E !important; }
        [data-testid="stSidebar"] * { color: white !important; }
        .metric-card {
            background-color: white !important; padding: 20px; border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #6F42C1;
            color: #1A233E !important; margin-bottom: 10px;
        }
        h1, h2, h3, p, span, label { color: #1A233E !important; }
        .alert-item { background-color: white !important; padding: 10px; margin-bottom: 10px; border-radius: 5px; border-bottom: 1px solid #eee; }

       /* --- BUTTON STYLES: Áp dụng cho cả st.button và st.file_uploader --- */
        .stButton button, [data-testid="stFileUploader"] button {
            background-color: #FFFFFF !important; /* Nền trắng */
            color: #1A233E !important;           /* Chữ Navy đậm */
            border: 1px solid #ddd !important;   /* Viền xám nhạt để tách nền */
            border-radius: 5px !important;       
            padding: 10px 15px !important;     
            font-weight: bold !important;        
            width: 100% !important;              
            transition: 0.3s !important;
        }

        /* Hiệu ứng Hover đồng bộ cho cả hai loại nút */
        .stButton button:hover, [data-testid="stFileUploader"] button:hover {
            background-color: #F8F9FA !important; /* Xám cực nhẹ khi di chuột */
            border-color: #bbb !important;       
            color: #1A233E !important;
        }

        /* Ẩn bớt các dòng chữ hướng dẫn rườm rà của Upload để trông sạch hơn */
        [data-testid="stFileUploader"] section > div {
            color: #1A233E !important; 
        }

        .stButton button:hover {
            background-color: #F8F9FA !important; /* Slightly off-white/gray on hover */
            border-color: #bbb !important;       /* Darker border on hover for contrast */
        }
            
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. STATE MANAGEMENT (18 Classes Logic)
# ==========================================
# Danh sách 18 classes theo yêu cầu
RUBBISH_CLASSES = [
    'box', 'can', 'chair', 'couch', 'cupboard', 'desk', 'fan', 'mattress', 
    'objects', 'people', 'rubbish', 'stroller', 'tire', 'tivi', 'toys', 
    'trash_can', 'wardrobe', 'washing_machine'
]

if 'logs' not in st.session_state: st.session_state.logs = []
if 'camera_running' not in st.session_state: st.session_state.camera_running = False
if 'conf_threshold' not in st.session_state: st.session_state.conf_threshold = 0.3

# Khởi tạo bộ đếm cho 18 classes + Total Scans + Not Rubbish
if 'counts' not in st.session_state:
    st.session_state.counts = {cls: 0 for cls in RUBBISH_CLASSES}
    st.session_state.counts['Total_Scans'] = 0
    st.session_state.counts['Not_Rubbish'] = 0

@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_PATH)

model = load_yolo_model()

def update_stats(results, source="System"):
    """Logic: +1 Total cho mọi lượt quét. Nếu không rác -> +1 Not Rubbish"""
    st.session_state.counts['Total_Scans'] += 1 # Luôn cộng 1 cho tổng lượt quét
    
    found_any = False
    detected_items = []
    
    if results and len(results[0].boxes) > 0:
        found_any = True
        for box in results[0].boxes:
            cls_name = results[0].names[int(box.cls[0])].lower()
            if cls_name in st.session_state.counts:
                st.session_state.counts[cls_name] += 1
                detected_items.append(cls_name)
    else:
        # Trường hợp không thấy rác
        st.session_state.counts['Not_Rubbish'] += 1

    # Lưu log chi tiết
    new_log = {
        "Time": datetime.now().strftime("%H:%M:%S, %b %d"),
        "Status": "Rubbish Detected" if found_any else "Clear / Not Rubbish",
        "Items": ", ".join(set(detected_items)) if found_any else "Clean Area",
        "Source": source
    }
    st.session_state.logs.insert(0, new_log)

# ==========================================
# 3. SIDEBAR & NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("###  PROJECT GROUP 10")
    st.markdown("---")
    menu = st.radio("NAVIGATION", ["Overview", "Rubbish Logs", "AI Model Metrics", "System Settings"])
    st.markdown("---")
    st.info("Project")

# ==========================================
# 4. HEADER & METRIC CARDS
# ==========================================
st.title("Roadside Rubbish Detection System")
m1, m2, m3, m4 = st.columns(4)
with m1: st.markdown(f'<div class="metric-card"><p>Total Scans</p><h2>{st.session_state.counts["Total_Scans"]}</h2></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="metric-card" style="border-left-color: #28A745;"><p>Clear (Not Rubbish)</p><h2>{st.session_state.counts["Not_Rubbish"]}</h2></div>', unsafe_allow_html=True)
with m3: 
    # Tính tổng đồ nội thất lớn (Mattress + Couch + Wardrobe...)
    big_items = st.session_state.counts['mattress'] + st.session_state.counts['couch'] + st.session_state.counts['wardrobe']
    st.markdown(f'<div class="metric-card" style="border-left-color: #FD7E14;"><p>Big Furniture</p><h2>{big_items}</h2></div>', unsafe_allow_html=True)
with m4: 
    # Tổng các loại rác khác (Total Scans - Not Rubbish)
    total_rubbish = st.session_state.counts['Total_Scans'] - st.session_state.counts['Not_Rubbish']
    st.markdown(f'<div class="metric-card" style="border-left-color: #DC3545;"><p>Total Rubbish Found</p><h2>{total_rubbish}</h2></div>', unsafe_allow_html=True)

# ==========================================
# 5. MAIN CONTENT
# ==========================================
col_main, col_side = st.columns([2, 1])

with col_main:
    if menu == "Overview":
        st.subheader("Real-time Detection View")
        tab1, tab2 = st.tabs(["Live Webcam", "Image Upload"])

        with tab1:
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                results = model.predict(np.array(image), conf=st.session_state.conf_threshold, iou=IOU_THRESHOLD, verbose=False)
                st.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB), caption="Processed Result", use_container_width=True)
                if st.button("SAVE TO LOGS"):
                    update_stats(results, source=uploaded_file.name)
                    st.success("Analysis complete and recorded!")
        
        with tab2:
            c1, c2 = st.columns(2)
            if c1.button("START CAMERA"): st.session_state.camera_running = True
            if c2.button("STOP CAMERA"): st.session_state.camera_running = False
            
            FRAME_WINDOW = st.empty()
            if st.session_state.camera_running:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                while cap.isOpened() and st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret: break
                    results = model.predict(frame, conf=st.session_state.conf_threshold, iou=IOU_THRESHOLD, verbose=False)
                    FRAME_WINDOW.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
                    # Logic tự động log có thể thêm ở đây nếu muốn
                cap.release()

    

    elif menu == "Rubbish Logs":
        st.subheader("📋 Detection History")
        if st.session_state.logs:
            df = pd.DataFrame(st.session_state.logs)
            st.dataframe(df, use_container_width=True)
        else: st.info("No logs available.")

    elif menu == "AI Model Metrics":
        st.subheader("📊 Statistical Analysis")
        if st.session_state.counts['Total_Scans'] > 0:
            fig, ax = plt.subplots()
            labels = ['Rubbish Detected', 'Clear / Not Rubbish']
            sizes = [st.session_state.counts['Total_Scans'] - st.session_state.counts['Not_Rubbish'], st.session_state.counts['Not_Rubbish']]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#DC3545', '#28A745'])
            st.pyplot(fig)
            
            # Biểu đồ cột cho 18 Classes
            st.write("**Top Detected Objects:**")
            class_df = pd.DataFrame([{"Class": k, "Count": v} for k, v in st.session_state.counts.items() if k in RUBBISH_CLASSES and v > 0])
            if not class_df.empty:
                st.bar_chart(class_df.set_index("Class"))

    elif menu == "System Settings":
        st.subheader("⚙️ AI Configuration")
        conf_val = st.slider("Confidence Threshold", 0.1, 1.0, value=st.session_state.conf_threshold, step=0.05)
        if st.button("APPLY SETTINGS"):
            st.session_state.conf_threshold = conf_val
            st.success(f"System sensitivity set to {conf_val}")

# Alert System (Right side)
with col_side:
    st.subheader("Live Alerts")
    for log in st.session_state.logs[:10]:
        border_color = "#DC3545" if "Rubbish" in log["Status"] else "#28A745"
        st.markdown(f'''
            <div class="alert-item" style="border-left: 5px solid {border_color}">
                <b>🕒 {log["Time"]}</b><br>
                Status: {log["Status"]}<br>
                <small>📦 {log["Items"]}</small>
            </div>
        ''', unsafe_allow_html=True)
