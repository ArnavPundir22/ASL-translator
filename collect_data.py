import os
import cv2
import numpy as np

from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

DATA_PATH = 'MP_Data'
actions = ['hello', 'thanks', 'iloveyou','Yes']
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

        # ---------- DATA COLLECTION ----------
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_res, hand_res, face_res = mediapipe_detection(frame_rgb, frame_idx)
            draw_styled_landmarks(frame, pose_res, hand_res, face_res)

            keypoints = extract_keypoints(pose_res, hand_res, face_res)
            np.save(
                os.path.join(DATA_PATH, action, str(seq), f"{frame_num}.npy"),
                keypoints
            )

            # ---------- OVERLAY INFO ----------
            cv2.rectangle(frame, (0, 0), (640, 90), (245, 117, 16), -1)

            cv2.putText(frame, f'COLLECTING: {action}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.putText(frame, f'Sequence: {seq + 1}/{no_sequences}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f'Frame: {frame_num + 1}/{sequence_length}', (380, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Collecting Data', frame)
            frame_idx += 1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()

