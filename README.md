<div align="center">
  <img src="assets/banner.png" alt="AI Safety Guardian Banner" width="100%">
</div>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square" alt="YOLOv8/Ultralytics"></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="MIT License"></a>
</p>

<div align="center">
  <h1>AI Safety Guardian</h1>
  <p><strong>Real-Time Fall and Fire Detection System with Advanced Escalation</strong></p>
</div>

---

## 📖 Overview

**AI Safety Guardian** is a computer vision-based safety monitoring system designed to detect human falls and fire or smoke hazards in real time. Leveraging deep learning and classical image processing techniques, it monitors video feeds continuously to detect safety-critical incidents, reducing emergency response latency in high-risk zones such as industrial warehouses, elderly care facilities, and smart homes.

The system uses **YOLOv8-Pose** estimation to analyze skeletal joints and compute body angles to confirm human falls. To reduce false positives, it incorporates a temporal confirmation logic requiring a fall to persist across multiple frames before validating the threat. For fire and smoke, it integrates **YOLOv8** object classification with HSV-based color masking to detect flame flicker and shapes.

It supports two distinct input modes: **Live Webcam** (for local/edge deployment) and **Upload Video** (for cloud-safe demonstration environments). When a safety threat is verified, the system triggers a multi-channel alert escalation—including a localized audio alarm and automatic email notifications via SMTP. A modern, centralized Security Operations Center (SOC) dashboard built with **Streamlit** displays status meters, system health, and an active incident history log.

---

## ✨ Features

* **Real-time fall detection** with confidence scoring and multi-frame confirmation logic
* **Real-time fire/smoke detection** utilizing a combined deep-learning and HSV color mask filtering approach to reduce false positives
* **Dual input modes**: Live Webcam (edge deployment) and Upload Video (cloud-safe testing)
* **Automated email alerts** containing detailed event metadata, confidence scores, and timestamps
* **Audio alarm system** to warn on-site personnel immediately upon threat confirmation
* **Live dashboard** containing system status meters, threat level gauges, and a continuous alert history log
* **Graceful fallback messaging** that handles missing webcam devices in cloud environments automatically

---

## 📷 Dashboard Preview

| | |
| :---: | :---: |
| ![Live Detection View](assets/dashboard_upload.png) | ![Fire & Fall Alert](assets/dashboard_alert.png) |

**_📹 Live Detection View — Real-time person tracking on uploaded warehouse footage._**

**_🚨 Fire & Fall Alert — Simultaneous hazard detection triggering confirmed fall alert and live email notification._**

---

## 📹 Watch Full Demo Video

📹 [Watch Full Demo Video](https://drive.google.com/file/d/1kIysGWD1BfSRI2qp1CuekzNjLl3ybu8x/view?usp=drive_link)

*This demo video showcases the system performing live detections, triggering the local audio alarm sound, and sending real-time SMTP email notifications end-to-end.*

---

## 🛠️ Tech Stack

| Technology | Usage / Role |
| :--- | :--- |
| **Python** | Core backend logic, multi-threaded alert execution, and file handling |
| **YOLOv8 (Ultralytics)** | Pose estimation (skeleton keypoints) and object detection models |
| **OpenCV** | Video capture, real-time image processing, and color masking filters |
| **Streamlit** | Frontend Security Operations Center (SOC) dashboard and live video renderer |
| **SMTP** | Automatic email notifications sent to administrators during safety emergencies |
| **Antigravity AI** | AI assistant utilized for UI engineering and layout refinement |

---

## ⚠️ Note on Deployment

> [!IMPORTANT]
> This system performs real-time video inference using YOLOv8, which is compute-intensive and exceeds the CPU limits of most free-tier cloud platforms (Streamlit Community Cloud, Hugging Face Spaces). The demo video above shows full local execution with live detection, alerting, and email notification working end-to-end. This reflects real-world deployment patterns for edge-AI safety systems, which typically run on local or on-premise hardware rather than shared cloud infrastructure.

---

## 🚀 Installation & Setup

Follow these steps to run the application locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/muhammedshemeer/AI_Safety_Guardian.git
cd AI_Safety_Guardian
```

### 2. Install Requirements
Ensure Python 3.10 or higher is installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Configure Email Credentials
To enable automated email alerts, create a `.env` file in the root directory (based on `.env.example`) and fill in your details:
```env
EMAIL_SENDER="your_gmail@gmail.com"
EMAIL_PASSWORD="your_16_char_app_password"
EMAIL_RECEIVER="recipient_email@gmail.com"
```
> *Note: For Gmail accounts, you will need to generate a 16-character App Password under your Google Account Security settings.*

### 4. Run the Streamlit Application
```bash
streamlit run dashboard.py
```
Open `http://localhost:8501` in your web browser.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Developed by <strong><a href="https://github.com/muhammedshemeer">Mohammed Shemeer</a></strong></p>
</div>
