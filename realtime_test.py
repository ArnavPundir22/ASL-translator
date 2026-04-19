import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────
actions = ['hello', 'thanks', 'iloveyou']
ACTION_COLORS = [
    (0, 230, 180),   # teal
    (255, 180, 0),   # amber
    (220, 60, 180),  # pink
]
THRESHOLD = 0.8
SEQ_LEN = 30
MAX_SENTENCE = 5
STATUS_DOT_SPACING = 90

model = load_model('action.h5', compile=False)

sequence = []
sentence = []
last_action = None
res = None

cap = cv2.VideoCapture(0)
frame_idx = 0
prev_time = time.time()


# ──────────────────────────────────────────────
#  UI helpers
# ──────────────────────────────────────────────

def _blend_rect(frame, x, y, w, h, color, alpha=0.65):
    """Draw a semi-transparent filled rectangle."""
    x2, y2 = x + w, y + h
    x, y = max(x, 0), max(y, 0)
    x2 = min(x2, frame.shape[1])
    y2 = min(y2, frame.shape[0])
    if x2 <= x or y2 <= y:
        return
    roi = frame[y:y2, x:x2]
    rect = np.full_like(roi, color)
    frame[y:y2, x:x2] = cv2.addWeighted(rect, alpha, roi, 1 - alpha, 0)


def _draw_pill(frame, x, y, w, h, color, alpha=0.70):
    """Rounded-looking pill via two rects + two circles."""
    r = h // 2
    _blend_rect(frame, x + r, y, w - 2 * r, h, color, alpha)
    for cx in [x + r, x + w - r]:
        roi_x, roi_y = max(cx - r, 0), max(y, 0)
        roi_x2, roi_y2 = min(cx + r, frame.shape[1]), min(y + h, frame.shape[0])
        roi = frame[roi_y:roi_y2, roi_x:roi_x2]
        mask = np.zeros_like(roi)
        cv2.circle(mask, (cx - roi_x, r), r, color, -1)
        frame[roi_y:roi_y2, roi_x:roi_x2] = cv2.addWeighted(mask, alpha, roi, 1 - alpha, 0)


def _confidence_bar(frame, label, conf, color, x, y, bar_w=150):
    """Draw label + filled confidence bar."""
    _blend_rect(frame, x, y, bar_w, 16, (30, 30, 30), alpha=0.75)
    fill = int(bar_w * conf)
    _blend_rect(frame, x, y, fill, 16, color, alpha=0.90)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + 16), (80, 80, 80), 1)
    cv2.putText(frame, f'{label}  {conf:.0%}',
                (x + bar_w + 8, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


def _status_dot(frame, label, active, cx, cy):
    color = (0, 210, 110) if active else (70, 70, 70)
    cv2.circle(frame, (cx, cy), 5, color, -1)
    cv2.putText(frame, label, (cx + 9, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)


# ──────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror for natural interaction
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # FPS
    curr_time = time.time()
    elapsed = curr_time - prev_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    prev_time = curr_time

    # MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_res, hand_res, face_res = mediapipe_detection(frame_rgb, frame_idx)
    draw_styled_landmarks(frame, pose_res, hand_res, face_res)

    # Keypoints → prediction
    keypoints = extract_keypoints(pose_res, hand_res, face_res)
    sequence.append(keypoints)
    sequence = sequence[-SEQ_LEN:]

    if len(sequence) == SEQ_LEN:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        top_idx = int(np.argmax(res))
        if res[top_idx] > THRESHOLD:
            detected = actions[top_idx]
            # Only append when action changes (deduplicate consecutive repeats)
            if detected != last_action:
                sentence.append(detected)
                sentence = sentence[-MAX_SENTENCE:]
            last_action = detected
        else:
            last_action = None

    # ── TOP BAR ──────────────────────────────
    _blend_rect(frame, 0, 0, w, 52, (12, 12, 12), alpha=0.80)
    cv2.putText(frame, 'ASL Translator', (14, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.90, (0, 220, 180), 2)
    cv2.putText(frame, f'FPS {fps:4.0f}', (w - 88, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (130, 130, 130), 1)

    # Detection status dots
    pose_ok = bool(pose_res.pose_landmarks)
    hand_ok = bool(hand_res.hand_landmarks)
    face_ok = bool(face_res.face_landmarks)
    for i, (lbl, ok) in enumerate([('Pose', pose_ok), ('Hand', hand_ok), ('Face', face_ok)]):
        _status_dot(frame, lbl, ok, 220 + i * STATUS_DOT_SPACING, 30)

    # ── CONFIDENCE PANEL (right) ──────────────
    if res is not None:
        panel_w, panel_h = 236, 110
        px = w - panel_w - 8
        py = 60
        _blend_rect(frame, px, py, panel_w, panel_h, (16, 16, 16), alpha=0.75)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (50, 50, 50), 1)
        cv2.putText(frame, 'CONFIDENCE', (px + 8, py + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
        for i, (action, conf) in enumerate(zip(actions, res)):
            _confidence_bar(frame, action, conf, ACTION_COLORS[i],
                            px + 8, py + 26 + i * 28)

    # ── CURRENT SIGN BADGE ───────────────────
    if last_action is not None:
        badge_label = last_action.upper()
        (tw, th), _ = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        bx, by = 14, 62
        _draw_pill(frame, bx - 8, by - th - 6, tw + 24, th + 16, (0, 180, 140), alpha=0.80)
        cv2.putText(frame, badge_label, (bx + 4, by + 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    # ── BOTTOM SENTENCE BAR ──────────────────
    _blend_rect(frame, 0, h - 62, w, 62, (10, 10, 10), alpha=0.82)
    cv2.line(frame, (0, h - 62), (w, h - 62), (40, 40, 40), 1)
    if sentence:
        disp = ' '.join(sentence)
        cv2.putText(frame, disp, (16, h - 28),
                    cv2.FONT_HERSHEY_DUPLEX, 1.05, (0, 225, 180), 2)
    else:
        cv2.putText(frame, 'Waiting for sign...', (16, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (70, 70, 70), 1)
    cv2.putText(frame, 'Q quit   C clear sentence', (16, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (55, 55, 55), 1)

    cv2.imshow('ASL Translator', frame)
    frame_idx += 1

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence.clear()
        last_action = None

cap.release()
cv2.destroyAllWindows()

