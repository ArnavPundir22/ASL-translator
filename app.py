import threading
import time
from typing import Optional

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from tensorflow.keras.models import load_model

from utils import draw_styled_landmarks, extract_keypoints

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────
ACTIONS = ['hello', 'thanks', 'iloveyou']
ACTION_COLORS = ['#00dbb4', '#ffb400', '#dc3cb4']
THRESHOLD = 0.8
SEQ_LEN = 30
MAX_SENTENCE = 5

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ──────────────────────────────────────────────
#  Model (cached across Streamlit reruns)
# ──────────────────────────────────────────────

@st.cache_resource
def load_asl_model():
    return load_model('action.h5', compile=False)


# ──────────────────────────────────────────────
#  Video processor
# ──────────────────────────────────────────────

class ASLProcessor(VideoProcessorBase):
    """Per-session WebRTC video processor: MediaPipe detection + LSTM inference."""

    def __init__(self):
        # Per-session MediaPipe landmarkers keep timestamps independent per user.
        self._pose_lm = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
                running_mode=vision.RunningMode.VIDEO,
            )
        )
        self._hand_lm = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )
        self._face_lm = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
                running_mode=vision.RunningMode.VIDEO,
            )
        )

        # Shared state (read by the main thread, written by the recv thread)
        self.lock = threading.Lock()
        self.sequence: list = []
        self.sentence: list = []
        self.last_action: Optional[str] = None
        self.confidence: dict = {a: 0.0 for a in ACTIONS}
        self.pose_ok: bool = False
        self.hand_ok: bool = False
        self.face_ok: bool = False

    def _detect(self, frame_rgb: np.ndarray, ts_ms: int):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        pose_res = self._pose_lm.detect_for_video(mp_image, ts_ms)
        hand_res = self._hand_lm.detect_for_video(mp_image, ts_ms)
        face_res = self._face_lm.detect_for_video(mp_image, ts_ms)
        return pose_res, hand_res, face_res

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        model = load_asl_model()

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ts_ms = int(time.time() * 1000)

        pose_res, hand_res, face_res = self._detect(frame_rgb, ts_ms)
        draw_styled_landmarks(img, pose_res, hand_res, face_res)
        keypoints = extract_keypoints(pose_res, hand_res, face_res)

        with self.lock:
            self.pose_ok = bool(pose_res.pose_landmarks)
            self.hand_ok = bool(hand_res.hand_landmarks)
            self.face_ok = bool(face_res.face_landmarks)

            self.sequence.append(keypoints)
            self.sequence = self.sequence[-SEQ_LEN:]

            if len(self.sequence) == SEQ_LEN:
                raw = model.predict(
                    np.expand_dims(self.sequence, axis=0), verbose=0
                )[0]
                self.confidence = {a: float(v) for a, v in zip(ACTIONS, raw)}
                top_idx = int(np.argmax(raw))
                if raw[top_idx] > THRESHOLD:
                    detected = ACTIONS[top_idx]
                    if detected != self.last_action:
                        self.sentence.append(detected)
                        self.sentence = self.sentence[-MAX_SENTENCE:]
                    self.last_action = detected
                else:
                    self.last_action = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ──────────────────────────────────────────────
#  UI helpers
# ──────────────────────────────────────────────

def _confidence_html(action: str, color: str, value: float) -> str:
    pct = value * 100
    return (
        f"<div style='margin-bottom:8px'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:0.85rem;margin-bottom:3px'>"
        f"<span>{action}</span>"
        f"<span style='color:{color}'>{pct:.0f}%</span></div>"
        f"<div style='background:rgba(255,255,255,0.08);border-radius:4px;height:8px'>"
        f"<div style='background:{color};width:{pct:.1f}%;height:100%;"
        f"border-radius:4px;transition:width 0.25s'></div></div></div>"
    )


def _status_dot(ok: bool) -> str:
    return "🟢" if ok else "⚫"


# ──────────────────────────────────────────────
#  Main app
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="ASL Translator",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        "<h1 style='color:#00dbb4;letter-spacing:2px;margin-bottom:0'>⬡ ASL Translator</h1>"
        "<p style='color:#52525b;margin-top:2px'>Real-time American Sign Language recognition</p>",
        unsafe_allow_html=True,
    )

    col_vid, col_info = st.columns([3, 2], gap="large")

    with col_vid:
        ctx = webrtc_streamer(
            key="asl-translator",
            video_processor_factory=ASLProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_info:
        st.subheader("Confidence")
        conf_placeholder = st.empty()

        st.subheader("Current Sign")
        action_placeholder = st.empty()

        st.subheader("Sentence")
        sentence_placeholder = st.empty()

        st.markdown("---")
        if st.button("🗑️ Clear Sentence") and ctx.video_processor:
            with ctx.video_processor.lock:
                ctx.video_processor.sentence.clear()
                ctx.video_processor.last_action = None

        st.subheader("Detection Status")
        status_cols = st.columns(3)
        pose_ph = status_cols[0].empty()
        hand_ph = status_cols[1].empty()
        face_ph = status_cols[2].empty()

    # ── Read processor state and refresh displays ─────────────────────────────
    proc = ctx.video_processor if ctx.video_processor else None

    if proc:
        with proc.lock:
            confidence = proc.confidence.copy()
            sentence = proc.sentence.copy()
            last_action = proc.last_action
            pose_ok = proc.pose_ok
            hand_ok = proc.hand_ok
            face_ok = proc.face_ok
    else:
        confidence = {a: 0.0 for a in ACTIONS}
        sentence = []
        last_action = None
        pose_ok = hand_ok = face_ok = False

    conf_html = "".join(
        _confidence_html(a, c, confidence.get(a, 0.0))
        for a, c in zip(ACTIONS, ACTION_COLORS)
    )
    conf_placeholder.markdown(conf_html, unsafe_allow_html=True)

    if last_action:
        action_placeholder.markdown(
            f"<p style='font-size:2rem;font-weight:bold;color:#00dbb4;"
            f"background:rgba(0,160,130,0.15);padding:8px 16px;border-radius:8px;"
            f"display:inline-block'>{last_action.upper()}</p>",
            unsafe_allow_html=True,
        )
    else:
        action_placeholder.markdown(
            "<p style='color:#52525b;font-style:italic'>No sign detected</p>",
            unsafe_allow_html=True,
        )

    if sentence:
        sentence_placeholder.markdown(
            f"<p style='font-size:1.5rem;color:#00dbb4;letter-spacing:1px'>"
            f"{' '.join(sentence)}</p>",
            unsafe_allow_html=True,
        )
    else:
        sentence_placeholder.markdown(
            "<p style='color:#52525b;font-style:italic'>Waiting for sign…</p>",
            unsafe_allow_html=True,
        )

    pose_ph.markdown(f"{_status_dot(pose_ok)} Pose")
    hand_ph.markdown(f"{_status_dot(hand_ok)} Hand")
    face_ph.markdown(f"{_status_dot(face_ok)} Face")

    # Rerun every 300 ms while the stream is active to refresh the metrics panel.
    if ctx.state.playing:
        time.sleep(0.3)
        st.rerun()


if __name__ == "__main__":
    main()
