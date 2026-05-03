import base64
import time
import threading

import cv2
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model

from utils import mediapipe_detection, extract_keypoints

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────
ACTIONS = ['hello', 'thanks', 'iloveyou']
THRESHOLD = 0.8
SEQ_LEN = 30
MAX_SENTENCE = 5

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# ──────────────────────────────────────────────
#  App & model
# ──────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=['http://localhost:5000', 'http://127.0.0.1:5000'], async_mode='threading')

model = load_model('action.h5', compile=False)

# Shared MediaPipe landmarkers live in utils.py module-level globals.
# A lock ensures only one thread calls detect_for_video at a time so
# the required monotonically-increasing timestamp constraint holds.
mp_lock = threading.Lock()

# ──────────────────────────────────────────────
#  Per-session state
# ──────────────────────────────────────────────
sessions: dict = {}
sessions_lock = threading.Lock()


def _get_session(sid: str) -> dict:
    with sessions_lock:
        if sid not in sessions:
            sessions[sid] = {
                'sequence': [],
                'sentence': [],
                'last_action': None,
                'confidence': None,
            }
        return sessions[sid]


# ──────────────────────────────────────────────
#  Landmark serialisation helpers
# ──────────────────────────────────────────────

def _build_landmarks(pose_res, hand_res, face_res) -> dict:
    """Return normalised landmark data for browser-side Canvas drawing."""
    data = {'pose': [], 'face': [], 'hands': []}

    if pose_res.pose_landmarks:
        data['pose'] = [[round(p.x, 4), round(p.y, 4)]
                        for p in pose_res.pose_landmarks[0]]

    if face_res.face_landmarks:
        data['face'] = [[round(p.x, 4), round(p.y, 4)]
                        for p in face_res.face_landmarks[0]]

    if hand_res.hand_landmarks:
        for hand in hand_res.hand_landmarks:
            data['hands'].append([[round(p.x, 4), round(p.y, 4)]
                                  for p in hand])

    return data


# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
                           actions=ACTIONS,
                           pose_connections=POSE_CONNECTIONS,
                           hand_connections=HAND_CONNECTIONS)


# ──────────────────────────────────────────────
#  Socket.IO events
# ──────────────────────────────────────────────

@socketio.on('connect')
def handle_connect():
    _get_session(request.sid)


@socketio.on('disconnect')
def handle_disconnect():
    with sessions_lock:
        sessions.pop(request.sid, None)


@socketio.on('clear')
def handle_clear():
    state = _get_session(request.sid)
    state['sentence'] = []
    state['last_action'] = None


@socketio.on('frame')
def handle_frame(data):
    """Receive a base64-encoded JPEG frame, run inference, emit results."""
    sid = request.sid
    state = _get_session(sid)

    # ── Decode ───────────────────────────────
    try:
        img_bytes = base64.b64decode(data['frame'])
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
    except Exception:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ts_ms = int(time.time() * 1000)

    # ── MediaPipe (serialised) ────────────────
    with mp_lock:
        pose_res, hand_res, face_res = mediapipe_detection(frame_rgb, ts_ms)

    # ── Keypoints & prediction ────────────────
    keypoints = extract_keypoints(pose_res, hand_res, face_res)
    state['sequence'].append(keypoints)
    state['sequence'] = state['sequence'][-SEQ_LEN:]

    if len(state['sequence']) == SEQ_LEN:
        raw = model.predict(
            np.expand_dims(state['sequence'], axis=0), verbose=0
        )[0]
        state['confidence'] = {a: float(v) for a, v in zip(ACTIONS, raw)}

        top_idx = int(np.argmax(raw))
        if raw[top_idx] > THRESHOLD:
            detected = ACTIONS[top_idx]
            if detected != state['last_action']:
                state['sentence'].append(detected)
                state['sentence'] = state['sentence'][-MAX_SENTENCE:]
            state['last_action'] = detected
        else:
            state['last_action'] = None

    # ── Emit result ───────────────────────────
    emit('result', {
        'action':     state['last_action'],
        'confidence': state['confidence'],
        'sentence':   state['sentence'].copy(),
        'landmarks':  _build_landmarks(pose_res, hand_res, face_res),
        'pose_ok':    bool(pose_res.pose_landmarks),
        'hand_ok':    bool(hand_res.hand_landmarks),
        'face_ok':    bool(face_res.face_landmarks),
    })


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000, debug=False)
