# 🧠 Part 1: Violence Detection using Deep Learning Models

This part of the project focuses on **video-based violence detection** using various deep learning architectures. The goal is to explore and evaluate different model architectures for classifying whether a video contains violent behavior or not.

---

## 📁 Dataset Used

We used the **Real Life Violence Situations** dataset, which contains labeled video clips categorized as **Violent** or **Non-Violent**.

---

## ⚙️ Preprocessing & Training Pipeline

The following steps outline the complete preprocessing and training flow:

1. **Frame Extraction**
   - From each video, **16 frames** are captured.
   - Frames are **evenly spaced** across the video to capture meaningful temporal features.

2. **Frame Preprocessing**
   - Extracted frames are resized and normalized.
   - They are converted into **NumPy arrays** for efficient storage and faster access.

3. **Data Structuring**
   - Each video is represented as a NumPy array of shape:  
     `(num_frames, height, width, channels)`
   - Labels are encoded as `Violent` or `Non-Violent`.

4. **Model Input & Training**
   - Preprocessed data is used to train multiple deep learning models.
   - Each model architecture is optimized independently.

---

## 🧪 Models Trained

We evaluated the performance of several deep learning architectures:

| Model Architecture                        | Description                                |
|------------------------------------------|--------------------------------------------|
|  **CNN + LSTM**                         | Extracts spatial features with CNN and temporal features with LSTM |
|  **3D CNN**                            | Captures spatial and temporal features in a unified 3D kernel |
|  **ConvLSTM**                          | Combines convolution and LSTM for better spatio-temporal learning |
|  **MobileNet + BiLSTM**                | Lightweight MobileNet features passed to Bi-directional LSTM |
|  **VGG16 + BiLSTM**                    | Pretrained VGG16 followed by a BiLSTM classifier |
|  **Xception + LSTM**                    | Depthwise separable CNN (Xception) with LSTM for efficient classification |

---

## 📊 Results

Each model’s accuracy and loss metrics were tracked during training. Performance details, training logs, and plots can be found in the corresponding Jupyter notebooks.

---


