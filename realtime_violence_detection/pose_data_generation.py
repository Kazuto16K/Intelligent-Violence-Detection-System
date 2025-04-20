import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = "violence"
frames_per_batch = 500
total_batches = 9
lm_list = []
frame_counter = 0

def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

for batch_num in range(1, total_batches + 1):
    print(f"\nReady to start Batch {batch_num} of {total_batches}")
    
    # Show OpenCV prompt to wait for key
    prompt_frame = 255 * np.ones((200, 600, 3), dtype=np.uint8)
    cv2.putText(prompt_frame, f"Press any key to start Batch {batch_num}", (30, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow("Prompt", prompt_frame)
    cv2.waitKey(0)
    cv2.destroyWindow("Prompt")

    print(f"Capturing Batch {batch_num}...")

    i = 0
    batch_folder = f"{label}_batch{batch_num}"
    os.makedirs(batch_folder, exist_ok=True)

    while i < frames_per_batch:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)

            frame = draw_landmark_on_image(mpDraw, results, frame)

            frame_filename = os.path.join(batch_folder, f"{i:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            i += 1
            frame_counter += 1

        cv2.imshow("Pose Capture", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    print(f"Completed Batch {batch_num} - Saved {i} frames.")

# Saving all landmarks
df = pd.DataFrame(lm_list)
df.to_csv(f"{label}_landmarks_cctv.csv", index=False)
print(f"\nAll landmark data saved to {label}_landmarks.csv")

cap.release()
cv2.destroyAllWindows()
