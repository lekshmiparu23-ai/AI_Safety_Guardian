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
    .main { background-color: #0e1117; color: white; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ AI Safety Guardian Dashboard")
st.write("Real-time Fall and Fire Detection System with Advanced Escalation")

# Load Models
@st.cache_resource
def load_models():
    pose_model = YOLO('yolov8n-pose.pt')
    detect_model = YOLO('yolov8n.pt') 
    return pose_model, detect_model

pose_model, detect_model = load_models()

# Sidebar - Settings & History
st.sidebar.header("Settings")
detection_mode = st.sidebar.selectbox("Choose Detection Mode", ["All", "Fall Detection", "Fire Detection"])
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)

st.sidebar.divider()
st.sidebar.header("⚙️ System Configuration")

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

# Action Buttons
if st.sidebar.button("Start Monitor"):
    st.session_state.monitoring_active = True
if st.sidebar.button("Stop Monitor"):
    st.session_state.monitoring_active = False
    st.session_state.fall_start_time = None
    st.session_state.alert_triggered = False

# Main Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📺 Cam 1A // Sector 4")
    
    # Input Source Toggle
    input_mode = st.radio(
        "Input Source Mode",
        ["Live Webcam", "Upload Video"],
        horizontal=True
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
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 350px; background-color: #1a1c23; border: 1px dashed #3f444e; border-radius: 10px; padding: 20px; text-align: center; color: #888888; font-family: sans-serif;">
                    <div style="font-size: 48px; margin-bottom: 15px;">📷</div>
                    <div style="font-weight: 500; font-size: 15px; max-width: 450px; line-height: 1.6;">
                        📷 Live Webcam mode requires local camera access and is designed for on-site/edge deployment. This cloud-hosted demo doesn't have camera access for security reasons — please switch to 'Upload Video' above to test detection with a sample clip.
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
        while cap.isOpened() and st.session_state.monitoring_active:
            ret, frame = cap.read()
            if not ret:
                if input_mode == "Upload Video":
                    st.info("Video playback completed.")
                else:
                    st.error("Failed to access webcam.")
                break

            # Process frame based on selected mode
            fall_res = {"fall_detected": False, "confidence": 0.0}
            fire_res = {"fire_detected": False, "confidence": 0.0}

            if detection_mode in ["All", "Fall Detection"]:
                fall_res = detect_fall(frame, pose_model)
            
            if detection_mode in ["All", "Fire Detection"]:
                fire_res = detect_fire(frame, detect_model)

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

            # Update Live Feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # Update Metrics
            if show_confidence:
                fall_metric.metric("Fall Conf", f"{fall_res['confidence']:.2f}")
                fire_metric.metric("Fire Conf", f"{fire_res['confidence']:.2f}")

            # Update Status & Notifications
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

            # Update History in Sidebar
            history = get_alert_history()
            with alert_history_placeholder.container():
                for h in history[:10]:
                    st.text(h)
                    
            # Add a small delay to simulate real-time playback for video file
            if input_mode == "Upload Video":
                time.sleep(0.03)

        cap.release()
        
    # Cleanup temp video file if created
    if input_mode == "Upload Video" and video_path and os.path.exists(video_path):
        try:
            os.unlink(video_path)
        except Exception:
            pass
else:
    status_placeholder.info("System Offline. Click 'Start Monitor' to begin.")
