---
title: AI Safety Guardian
emoji: 🛡️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ AI Safety Guardian

<div align="center">

![AI Safety Guardian](https://img.shields.io/badge/AI%20Safety-Guardian-red?style=for-the-badge&logo=shield&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Real-time AI-powered safety monitoring system for fall detection and fire detection**  
*Built with YOLOv8, OpenCV, and Streamlit — with automated email alerts*

[![GitHub Stars](https://img.shields.io/github/stars/muhammedshemeer/AI_Safety_Guardian?style=for-the-badge)](https://github.com/muhammedshemeer/AI_Safety_Guardian/stargazers)

</div>

---

## 🎬 Demo Video

> See the system detecting falls and fire **in real-time** with live confidence scores and instant alerts!

🎬 [▶️ Watch Live Demo Video](demo.mp4)

---

## 📸 Screenshots

> **Real detections captured live — YOLOv8 running on actual webcam feed**

| 🧍 Fall Detection — CONFIRMED FALL (Conf: 0.74) | 🔥 Fire Detection — FIRE DETECTED (Conf: 0.85) |
|:-:|:-:|
| ![Fall Detection](assets/screenshot1.png) | ![Fire Detection](assets/screenshot2.png) |

> ⬆️ **Left:** System detected a confirmed fall — red bounding box on person, emergency alert triggered, alert logged with timestamp `22:08:36`
>
> ⬆️ **Right:** Fire mask detected in frame — fire confidence 0.85, "Fire/Smoke risk detected!" alert shown, alert logged at `22:08:12`

---

## 🎯 Project Overview

**AI Safety Guardian** is a real-time safety monitoring application built using **Computer Vision** and **Deep Learning**. It uses **YOLOv8** pose estimation and object detection models to continuously monitor live video feeds, detect potential dangers like **human falls** and **fire/smoke**, and instantly trigger **email alerts + audio alarms**.

> 🏢 **Real-world use case:** Hospitals, factories, elderly care homes, and warehouses need 24/7 human monitoring — this system automates that at near-zero cost.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧍 **Fall Detection** | YOLOv8-Pose detects human posture and flags dangerous falls |
| 🔥 **Fire Detection** | YOLOv8 + color masking to identify fire or smoke in real-time |
| ⏱️ **3-Second Confirmation Logic** | Smart fall confirmation system — avoids false alarms by confirming fall persists for 3 seconds |
| 📧 **Email Alerts** | Instant email notification sent to configured receiver on confirmed threat |
| 🔔 **Audio Alarm** | Plays alarm.wav immediately on threat detection |
| 📊 **Live Dashboard** | Real-time Streamlit dashboard showing live feed, confidence scores, and alert history |
| 🗂️ **Alert History** | Timestamped log of all past alerts displayed in sidebar |
| 🎮 **Demo Video Mode** | Upload a video file to test the system without a webcam (cloud-compatible) |

---

## 🏗️ System Architecture

```
AI_Safety_Guardian/
├── dashboard.py         # Main Streamlit application & UI
├── fall_detection.py    # YOLOv8-Pose fall detection logic
├── fire_detection.py    # YOLOv8 + color mask fire detection
├── alert_system.py      # Email alerts, alarm, alert logging
├── alarm.wav            # Alarm sound file
├── yolov8n-pose.pt      # YOLOv8 pose estimation model
├── yolov8n.pt           # YOLOv8 object detection model
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── README.md            # Documentation
```

---

## 🧠 Technical Implementation

### Fall Detection Logic
- Uses **YOLOv8n-pose** to extract 17 body keypoints per person
- Analyzes keypoint positions (shoulder, hip, knee, ankle) to determine body angle
- Implements **3-second persistence check** — fall is only confirmed if it persists, preventing false alarms
- Confidence score displayed in real-time on dashboard

### Fire Detection Logic
- Uses **YOLOv8n** object detector for smoke/fire object classification
- Combined with **HSV color masking** for reliable fire color range detection
- Dual-method approach increases detection accuracy

### Alert Escalation System
```
Threat Detected
    ↓
3-Second Confirmation (Fall) / Instant (Fire)
    ↓
Audio Alarm (alarm.wav) + Email Alert
    ↓
Alert Logged with Timestamp → Dashboard History
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Webcam (for live mode) or video file (for demo mode)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/muhammedshemeer/AI_Safety_Guardian.git
cd AI_Safety_Guardian

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
```

### Configure Email Alerts

Open `.env` and add your Gmail credentials:

```env
EMAIL_SENDER=yourgmail@gmail.com
EMAIL_PASSWORD="your_16_char_app_password"
EMAIL_RECEIVER=receiver_email@gmail.com
```

> 💡 **How to get Gmail App Password:** Go to Google Account → Security → 2-Step Verification → App Passwords → Generate

### Run the App

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

---



## 📦 Dependencies

```txt
streamlit
opencv-python
ultralytics
numpy
python-dotenv
```

---

## 🔮 Future Improvements

- [ ] 📱 SMS alerts via Twilio
- [ ] 🌙 Night vision / low-light enhancement
- [ ] 👤 Multi-person tracking with individual IDs
- [ ] 🔫 Weapon / intrusion detection module
- [ ] 📁 Alert history export to CSV
- [ ] 📡 RTSP IP camera stream support

---

## 🧑‍💻 Author

**Mohammed Shemeer**  
B.Tech AI & ML — Dhanalakshmi Srinivasan University, Trichy  
AI/ML Intern @ V-Dart

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mohammed%20Shemeer-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/mohammed-shemeer-aiml)
[![GitHub](https://img.shields.io/badge/GitHub-muhammedshemeer-black?style=flat-square&logo=github)](https://github.com/muhammedshemeer)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Mohammed Shemeer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

<div align="center">

⭐ **If you found this useful, give it a star!** ⭐

*Built with ❤️ using Python, YOLOv8, and Streamlit*

</div>
