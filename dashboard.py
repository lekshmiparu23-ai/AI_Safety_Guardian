import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import tempfile
from alert_system import play_alarm, send_email_alert, log_alert, get_alert_history, EMAIL_SENDER, EMAIL_PASSWORD
from fall_detection import detect_fall
from fire_detection import detect_fire

# Page Configuration
st.set_page_config(page_title="AI Safety Guardian", page_icon="🛡️", layout="wide")

# Initialize Session State
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'fall_start_time' not in st.session_state:
    st.session_state.fall_start_time = None
if 'alert_triggered' not in st.session_state:
    st.session_state.alert_triggered = False

# Custom Styling
st.markdown("""
    <style>
    /* Main View Container */
    [data-testid="stAppViewContainer"] {
        background-color: #0C1E29 !important;
        color: #FFFFFF !important;
    }
    
    /* Sidebar Container */
    [data-testid="stSidebar"] {
        background-color: #08161E !important;
        border-right: 1px solid rgba(255, 254, 21, 0.15) !important;
    }
    
    /* Header blur effect */
    [data-testid="stHeader"] {
        background-color: rgba(12, 30, 41, 0.8) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Headers & Accent elements */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFE15 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    
    p, span, div, label, li {
        color: #FFFFFF !important;
    }
    
    /* Sidebar Headers */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFE15 !important;
    }
    
    /* Premium Action Buttons */
    .stButton>button {
        background-color: #0C1E29 !important;
        color: #FFFE15 !important;
        border: 1px solid #FFFE15 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        box-shadow: 0 0 5px rgba(255, 254, 21, 0.1) !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        background-color: #FFFE15 !important;
        color: #0C1E29 !important;
        box-shadow: 0 0 15px rgba(255, 254, 21, 0.4) !important;
        border: 1px solid #FFFE15 !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #FFFE15 !important;
        font-size: 38px !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    
    /* Alert cards custom borders */
    .stAlert {
        background-color: #08161E !important;
        border: 1px solid rgba(255, 254, 21, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Info/Success status styling */
    .stInfo, .stSuccess, .stWarning, .stError {
        background-color: #08161E !important;
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ AI Safety Guardian Dashboard")
st.write("Real-time Fall and Fire Detection System with Advanced Escalation")

# Load Models
@st.cache_resource
def load_models():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pose_model = YOLO('yolov8n-pose.pt')
    detect_model = YOLO('yolov8n.pt') 
    pose_model.to(device)
    detect_model.to(device)
    return pose_model, detect_model, device

pose_model, detect_model, device = load_models()

# Sidebar - Settings & History
st.sidebar.header("Settings")
detection_mode = st.sidebar.selectbox("Choose Detection Mode", ["All", "Fall Detection", "Fire Detection"])
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)

# Action Buttons (Highly visible at the top)
st.sidebar.divider()
if st.sidebar.button("Start Monitor"):
    st.session_state.monitoring_active = True
if st.sidebar.button("Stop Monitor"):
    st.session_state.monitoring_active = False
    st.session_state.fall_start_time = None
    st.session_state.alert_triggered = False

st.sidebar.divider()
st.sidebar.header("⚙️ System Configuration")
st.sidebar.info(f"Inference Device: {device.upper()}")

# Check Alarm File
if not os.path.exists("alarm.wav"):
    st.sidebar.warning("⚠️ alarm.wav not found! System will use internal Beep.")
else:
    st.sidebar.success("✅ alarm.wav found.")

# Check Email Config
if EMAIL_SENDER == "yourgmail@gmail.com" or EMAIL_PASSWORD == "16_char_gmail_app_password":
    st.sidebar.error("❌ Email not configured! Setup App Password in alert_system.py.")
else:
    st.sidebar.success("✅ Email configured.")

st.sidebar.divider()
st.sidebar.header("🚨 Alert History")
alert_history_placeholder = st.sidebar.empty()

# Main Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
    col_hdr, col_tgl = st.columns([3, 2])
    with col_hdr:
        st.subheader("📺 Cam 1A // Sector 4")
    with col_tgl:
        input_mode = st.radio(
            "Input Source Mode",
            ["Live Webcam", "Upload Video"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    video_path = None
    if input_mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            tfile.close()
        else:
            st.info("Please upload a video file to begin.")
            
    frame_placeholder = st.empty()

with col2:
    st.subheader("System Status")
    status_placeholder = st.empty()
    
    col_fall, col_fire = st.columns(2)
    with col_fall:
        fall_metric = st.empty()
    with col_fire:
        fire_metric = st.empty()
    
    st.divider()
    st.subheader("Active Alerts")
    alert_placeholder = st.empty()

def draw_cached_fall(frame, cached_boxes, scale_w, scale_h):
    if not cached_boxes:
        return
    for box_info in cached_boxes:
        x1, y1, x2, y2 = box_info["box"]
        conf = box_info["conf"]
        is_fall = box_info["is_fall"]
        
        sx1, sy1, sx2, sy2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
        if is_fall:
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
            cv2.putText(frame, f"Fall Risk Detected! ({conf:.2f})", (sx1, sy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person ({conf:.2f})", (sx1, sy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def draw_cached_fire(frame, cached_boxes, cached_contours, scale_w, scale_h):
    # Draw boxes
    for box_info in cached_boxes:
        x1, y1, x2, y2 = box_info["box"]
        conf = box_info["conf"]
        label = box_info["label"]
        
        sx1, sy1, sx2, sy2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 165, 255), 2)
        cv2.putText(frame, f"{label.capitalize()} ({conf:.2f})", (sx1, sy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    
    # Draw contours
    for contour_info in cached_contours:
        contour = np.array(contour_info["contour"], dtype=np.int32)
        cv = contour_info["flicker"]
        
        scale_array = np.array([scale_w, scale_h])
        scaled_contour = (contour * scale_array).astype(np.int32)
        x, y, w, h = cv2.boundingRect(scaled_contour)
        cv2.drawContours(frame, [scaled_contour], -1, (0, 165, 255), 2)
        cv2.putText(frame, f"Flame (Flicker: {cv:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

# Monitoring Loop
if st.session_state.monitoring_active:
    cap = None
    webcam_failed = False
    
    if input_mode == "Live Webcam":
        try:
            cap = cv2.VideoCapture(0)
            if cap is None or not cap.isOpened():
                webcam_failed = True
        except Exception:
            webcam_failed = True
            
        if webcam_failed:
            frame_placeholder.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 350px; background-color: #08161E; border: 1px dashed rgba(255, 254, 21, 0.4); border-radius: 10px; padding: 20px; text-align: center; color: #FFFFFF; font-family: inherit;">
                    <div style="font-size: 48px; margin-bottom: 15px;">📷</div>
                    <div style="font-weight: 500; font-size: 15px; max-width: 450px; line-height: 1.6;">
                        Live Webcam mode requires local camera access and is designed for on-site/edge deployment. This cloud-hosted demo doesn't have camera access for security reasons — please switch to 'Upload Video' above to test detection with a sample clip.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            status_placeholder.warning("⚠️ Webcam Unavailable")
            st.session_state.monitoring_active = False
            
    elif input_mode == "Upload Video":
        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
            if cap is None or not cap.isOpened():
                st.error("Failed to open uploaded video.")
                st.session_state.monitoring_active = False
        else:
            st.warning("Please upload a video file first.")
            st.session_state.monitoring_active = False

    if cap is not None and st.session_state.monitoring_active:
        frame_idx = 0
        last_fall_res = None
        last_fire_res = None
        scale_w = 1.0
        scale_h = 1.0
        h_new = 0

        while cap.isOpened() and st.session_state.monitoring_active:
            ret, frame = cap.read()
            if not ret:
                if input_mode == "Upload Video":
                    # Loop video by resetting position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read video frame after loop reset.")
                        break
                else:
                    st.error("Failed to access webcam.")
                    break

            if frame_idx % 2 == 0:
                # Downscale each frame to 640px width (maintaining aspect ratio) before running YOLOv8 detection
                h_orig, w_orig = frame.shape[:2]
                scale = 640.0 / w_orig
                h_new = int(h_orig * scale)
                frame_resized = cv2.resize(frame, (640, h_new))
                
                scale_w = w_orig / 640.0
                scale_h = h_orig / h_new

                fall_res = {"fall_detected": False, "confidence": 0.0, "boxes": []}
                fire_res = {"fire_detected": False, "confidence": 0.0, "boxes": [], "contours": []}

                if detection_mode in ["All", "Fall Detection"]:
                    fall_res = detect_fall(frame_resized, pose_model, draw_on=frame)
                
                if detection_mode in ["All", "Fire Detection"]:
                    fire_res = detect_fire(frame_resized, detect_model, draw_on=frame)
                    
                last_fall_res = fall_res
                last_fire_res = fire_res

                # --------------------------------------------------
                # FEATURE 1: 3-SECOND FALL CONFIRMATION LOGIC
                # --------------------------------------------------
                if fall_res["fall_detected"]:
                    if st.session_state.fall_start_time is None:
                        st.session_state.fall_start_time = time.time()
                    elapsed_time = time.time() - st.session_state.fall_start_time
                    if elapsed_time >= 3.0 and not st.session_state.alert_triggered:
                        alert_msg = f"{time.strftime('%H:%M:%S')} - CONFIRMED FALL ({fall_res['confidence']:.2f})"
                        log_alert(alert_msg)
                        play_alarm()
                        send_email_alert("Fall Detected", fall_res['confidence'])
                        st.session_state.alert_triggered = True
                else:
                    st.session_state.fall_start_time = None
                    st.session_state.alert_triggered = False 

                # Update Live Feed every 2 frames
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")

                # Update Metrics every 2 frames
                if show_confidence:
                    fall_metric.metric("Fall Conf", f"{fall_res['confidence']:.2f}")
                    fire_metric.metric("Fire Conf", f"{fire_res['confidence']:.2f}")

                # Update Status & Notifications every 2 frames
                if st.session_state.alert_triggered:
                    status_placeholder.error("🚨 CONFIRMED FALL")
                    alert_placeholder.error("EMERGENCY: Fall detected and confirmed!")
                elif fall_res["fall_detected"]:
                    status_placeholder.warning("⚠️ Potential Fall...")
                    wait_time = 3.0 - (time.time() - st.session_state.fall_start_time)
                    alert_placeholder.warning(f"Confirming fall in {max(0.0, wait_time):.1f}s...")
                elif fire_res["fire_detected"]:
                    status_placeholder.error("🔥 FIRE DETECTED")
                    alert_msg = f"{time.strftime('%H:%M:%S')} - FIRE DETECTED ({fire_res['confidence']:.2f})"
                    log_alert(alert_msg)
                    alert_placeholder.warning("Fire/Smoke risk detected!")
                else:
                    status_placeholder.success("✅ System Monitoring...")
                    alert_placeholder.info("No active threats detected.")

                # Update History in Sidebar every 2 frames
                history = get_alert_history()
                with alert_history_placeholder.container():
                    for h in history[:10]:
                        st.text(h)

                if input_mode == "Upload Video":
                    # Balance sleep duration since we process every 2nd frame
                    time.sleep(0.01)
            else:
                # Skipped Frame (odd frame): Skip re-running inference and skip UI display rendering
                # But still draw cached detections on the frame and run temporal logic to maintain correct alert states
                if last_fall_res is not None:
                    draw_cached_fall(frame, last_fall_res["boxes"], scale_w, scale_h)
                    
                    if last_fall_res["fall_detected"]:
                        if st.session_state.fall_start_time is None:
                            st.session_state.fall_start_time = time.time()
                        elapsed_time = time.time() - st.session_state.fall_start_time
                        if elapsed_time >= 3.0 and not st.session_state.alert_triggered:
                            alert_msg = f"{time.strftime('%H:%M:%S')} - CONFIRMED FALL ({last_fall_res['confidence']:.2f})"
                            log_alert(alert_msg)
                            play_alarm()
                            send_email_alert("Fall Detected", last_fall_res['confidence'])
                            st.session_state.alert_triggered = True
                    else:
                        st.session_state.fall_start_time = None
                        st.session_state.alert_triggered = False

                if last_fire_res is not None:
                    draw_cached_fire(frame, last_fire_res["boxes"], last_fire_res["contours"], scale_w, scale_h)

            frame_idx += 1

        if cap is not None:
            cap.release()
        
    # Cleanup temp video file if created
    if input_mode == "Upload Video" and video_path and os.path.exists(video_path):
        try:
            os.unlink(video_path)
        except Exception:
            pass
else:
    status_placeholder.info("System Offline. Click 'Start Monitor' to begin.")
