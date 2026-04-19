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


def _blend_rect(frame, x, y, w, h, color, alpha=0.70):
    x2, y2 = x + w, y + h
    x, y = max(x, 0), max(y, 0)
    x2 = min(x2, frame.shape[1])
    y2 = min(y2, frame.shape[0])
    if x2 <= x or y2 <= y:
        return
    roi = frame[y:y2, x:x2]
    rect = np.full_like(roi, color)
    frame[y:y2, x:x2] = cv2.addWeighted(rect, alpha, roi, 1 - alpha, 0)


def _progress_bar(frame, x, y, w, h, value, total, color):
    _blend_rect(frame, x, y, w, h, (30, 30, 30), alpha=0.80)
    fill = int(w * value / max(total, 1))
    _blend_rect(frame, x, y, fill, h, color, alpha=0.90)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 70), 1)


for action in actions:
    for seq in range(no_sequences):

        # ──────────── GET READY COUNTDOWN ────────────
        for countdown in range(20, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Dark overlay top
            _blend_rect(frame, 0, 0, w, 90, (10, 10, 10), alpha=0.80)

            cv2.putText(frame, 'GET READY', (w // 2 - 110, 44),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 220, 180), 2)
            cv2.putText(frame, f'Action: {action.upper()}', (14, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
            label_text = f'Starting in {countdown}s'
            (lw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
            cv2.putText(frame, label_text, (w - lw - 14, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 100), 1)

            # Bottom bar: overall progress
            _blend_rect(frame, 0, h - 36, w, 36, (10, 10, 10), alpha=0.75)
            action_idx = actions.index(action)
            done = action_idx * no_sequences + seq
            total = len(actions) * no_sequences
            _progress_bar(frame, 8, h - 28, w - 16, 14, done, total, (0, 180, 140))
            cv2.putText(frame, f'{done}/{total} sequences done',
                        (14, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

            cv2.imshow('ASL Data Collector', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # ──────────── DATA COLLECTION ────────────
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_res, hand_res, face_res = mediapipe_detection(frame_rgb, frame_idx)
            draw_styled_landmarks(frame, pose_res, hand_res, face_res)

            keypoints = extract_keypoints(pose_res, hand_res, face_res)
            np.save(
                os.path.join(DATA_PATH, action, str(seq), f"{frame_num}.npy"),
                keypoints
            )

            # Top overlay
            _blend_rect(frame, 0, 0, w, 90, (12, 12, 12), alpha=0.80)
            cv2.putText(frame, f'COLLECTING: {action.upper()}', (14, 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 220, 180), 2)
            cv2.putText(frame, f'Seq {seq + 1}/{no_sequences}', (14, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)

            # Frame progress bar
            _progress_bar(frame, w // 2 - 100, 60, 200, 14, frame_num + 1, sequence_length, (255, 160, 0))
            cv2.putText(frame, f'Frame {frame_num + 1}/{sequence_length}',
                        (w // 2 - 100, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Bottom overall progress bar
            _blend_rect(frame, 0, h - 36, w, 36, (10, 10, 10), alpha=0.75)
            action_idx = actions.index(action)
            done = action_idx * no_sequences + seq
            total = len(actions) * no_sequences
            _progress_bar(frame, 8, h - 28, w - 16, 14, done, total, (0, 180, 140))
            cv2.putText(frame, f'{done}/{total} sequences done',
                        (14, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

            cv2.imshow('ASL Data Collector', frame)
            frame_idx += 1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()

