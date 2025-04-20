import cv2
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
import mediapipe as mp
import logging

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

class PersonTracker:
    def __init__(self, yolo_weights="yolov8n.pt", deepsort_ckpt="deep_sort/deep/checkpoint/ckpt.t7", buffer_size=20):
        self.current_video_predictions = []
        self.yolo_model = YOLO(yolo_weights)
        self.tracker = DeepSort(model_path=deepsort_ckpt, max_age=70) 
        self.pose = mp.solutions.pose.Pose()
        self.mpDraw = mp.solutions.drawing_utils
        self.person_buffers = defaultdict(lambda: deque(maxlen=buffer_size))
        self.predictions = {} 
        self.buffer_size = buffer_size

    def make_landmark_timestep(self, results):
        return [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z, lm.visibility)]

    def update(self, frame, violence_model=None):
        og_frame = frame.copy()
        detections = self.yolo_model(frame, classes=0, conf=0.65,verbose=False)[0]
        boxes = detections.boxes

        if boxes is not None and len(boxes) > 0:
            xywh = boxes.xywh.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            self.tracker.update(xywh, conf, frame)

            for track in self.tracker.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_tlbr())

                height, width = frame.shape[:2]
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))

                # Skip if bounding box is invalid
                if x2 <= x1 or y2 <= y1:
                    continue

                person_crop = frame[y1:y2, x1:x2]

                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                results = self.pose.process(person_rgb)

                if results.pose_landmarks:

                    landmarks = results.pose_landmarks.landmark
                    for connection in mp.solutions.pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks) and end_idx < len(landmarks):
                            x1_conn = int(landmarks[start_idx].x * (x2 - x1)) + x1
                            y1_conn = int(landmarks[start_idx].y * (y2 - y1)) + y1
                            x2_conn = int(landmarks[end_idx].x * (x2 - x1)) + x1
                            y2_conn = int(landmarks[end_idx].y * (y2 - y1)) + y1
                            cv2.line(og_frame, (x1_conn, y1_conn), (x2_conn, y2_conn), (255, 255, 255), 1)

                    for lm in landmarks:
                        cx = int(lm.x * (x2 - x1)) + x1
                        cy = int(lm.y * (y2 - y1)) + y1
                        cv2.circle(og_frame, (cx, cy), 3, (0, 255, 0), -1)
                    
                    lm = self.make_landmark_timestep(results)
                    self.person_buffers[track_id].append(lm)

                    # Violence prediction after buffer is full
                    if len(self.person_buffers[track_id]) == self.buffer_size and violence_model:
                        sequence = np.array(self.person_buffers[track_id], dtype=np.float32)[np.newaxis, ...]
                        print(f"Sequence shape: {sequence.shape}, dtype: {sequence.dtype}")

                        pred = violence_model.predict(sequence, verbose=0)
                        print(pred)
                        label = "NonViolence" if pred[0][0] > 0.5 else "Violence"
                        self.current_video_predictions.append(label)
                        self.predictions[track_id] = label
                        self.person_buffers[track_id].clear()

                color = (255, 0, 0)
                cv2.rectangle(og_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(og_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                label = self.predictions.get(track_id, "P R O C E S S I N G")
                label_color = (255, 0, 255) if label == "P R O C E S S I N G" else (0, 255, 0) if label == "NonViolence" else (0, 0, 255)
                cv2.putText(og_frame, label, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        return og_frame
