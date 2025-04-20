import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from person_tracker import PersonTracker

model = load_model("./models/pose_violence_detection_cctv.keras", custom_objects={'Orthogonal': tf.keras.initializers.Orthogonal})
print("Model loaded. Input shape:", model.input_shape)

cap = cv2.VideoCapture(0)  # or 1 if external webcam

tracker = PersonTracker(buffer_size=20)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = tracker.update(frame, violence_model=model)

    cv2.imshow("Multi-Person Violence Detection", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
