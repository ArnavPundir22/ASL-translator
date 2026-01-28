import os
import cv2
import numpy as np

from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

DATA_PATH = 'MP_Data'
actions = ['hello', 'thanks', 'iloveyou']
no_sequences = 30
sequence_length = 30

# Create folders
for action in actions:
    for seq in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

cap = cv2.VideoCapture(0)
frame_idx = 0

for action in actions:
    for seq in range(no_sequences):

        # ---------- GET READY SCREEN ----------
        for countdown in range(20, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
            cv2.putText(frame, 'GET READY', (180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            cv2.putText(frame, f'Action: {action}', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f'Starting in {countdown}', (400, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Collecting Data', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

      

cap.release()
cv2.destroyAllWindows()

