# 🛡 Intelligent Violence Detection System

This project presents a comprehensive approach to **violence detection using deep learning**, combining both **video-based** analysis and **real-time pose-based** detection.

---

##  Project Overview

The system is divided into two major parts:

### 🔹 Part 1: Violence Detection from Videos
- We explored various deep learning architectures for detecting violence from pre-recorded videos.
- Models trained include:
  - CNN + LSTM
  - 3D CNN
  - ConvLSTM
  - MobileNet + BiLSTM
  - VGG + BiLSTM
  - Xception + LSTM
- Each model was trained on the **Real Life Violence Situations** dataset.
- Video frames were extracted (16 per video), converted to NumPy arrays, and fed into the models for classification.

### 🔹 Part 2: Real-time Violence Detection using PoseLSTM
- We built a real-time system called **PoseLSTM** using:
  - **YOLOv8** for human detection
  - **DeepSORT** for person tracking
  - **MediaPipe Pose** for extracting 33 pose landmarks (x, y, z, visibility)
  - **LSTM model** trained on custom-generated pose landmark datasets
- The model achieved **100% accuracy** on our synthetic pose dataset.
- Final predictions are made per person in live video input using a buffer of 20 frames.

---

##  Limitations

- **Computational Expense**: 
  - Running real-time detection with YOLO, MediaPipe, and LSTM prediction per person is resource-intensive.
  - High-end GPU is preferred for smooth performance.

- **Dataset Constraints**: 
  - Publicly available datasets for **pose-based violence detection** are scarce.
  - We generated our own synthetic dataset using controlled pose data, which may limit real-world generalization.
  - Video-based datasets were also limited in diversity and size for robust training.

---

##  Highlights

- ✅ Multi-model experimentation for video classification
- ✅ Custom dataset creation using MediaPipe
- ✅ Real-time human-level tracking and action classification
- ✅ Modular and extensible pipeline for further research or production integration

---

##  Upcoming Work

- We plan to build a **Flask-based web application** where users can:
  - Upload videos for offline detection
  - Use their own **webcam** for real-time violence detection
- **Twilio Integration**: To send alert messages to the user’s **mobile number**
- **Mailtrap Integration**: To send **email alerts** to the user’s registered email ID when violence is detected

---

---

##  Future Improvements

- Integrate advanced pose-based datasets (e.g., NTU RGB+D, PoseTrack)
- Audio cues can be added for more feature extraction
- With more data and without size constraints better model can be developed

---

## 👥 Authors

- **Soumava Das** – Intelligent Violence Detection System (2025)
- **Sarthak Saha** – Intelligent Violence Detection System (2025)
- **Debshankar Dey** – Intelligent Violence Detection System (2025)
- **Avipriya Ghosh** – Intelligent Violence Detection System (2025)

---

For more technical details, refer to the [Part 1 README](./video_violence_detection/README.md) and [Part 2 README](./realtime_violence_detection/README.md).
