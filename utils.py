import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- LOAD MODELS ---------------- #

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    ),
    running_mode=vision.RunningMode.VIDEO
)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=vision.RunningMode.VIDEO
)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# ---------------- SKELETON CONNECTIONS ---------------- #

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm
]

POSE_CONNECTIONS = [
    (11, 12),                                  # shoulders
    (11, 13), (13, 15),                        # left arm
    (12, 14), (14, 16),                        # right arm
    (11, 23), (12, 24),                        # torso sides
    (23, 24),                                  # hips
    (23, 25), (25, 27),                        # left leg
    (24, 26), (26, 28),                        # right leg
]

# ---------------- MEDIAPIPE DETECTION ---------------- #

def mediapipe_detection(frame, frame_idx):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    pose_results = pose_landmarker.detect_for_video(mp_image, frame_idx)
    hand_results = hand_landmarker.detect_for_video(mp_image, frame_idx)
    face_results = face_landmarker.detect_for_video(mp_image, frame_idx)

    return pose_results, hand_results, face_results

# ---------------- DRAW LANDMARKS ---------------- #

def _get_px(p, image):
    return int(p.x * image.shape[1]), int(p.y * image.shape[0])


def draw_styled_landmarks(image, pose_res, hand_res, face_res):

    # ---- Face (subtle teal dots) ----
    if face_res.face_landmarks:
        for face in face_res.face_landmarks:
            for p in face:
                x, y = _get_px(p, image)
                cv2.circle(image, (x, y), 1, (0, 200, 180), -1)

    # ---- Pose skeleton ----
    if pose_res.pose_landmarks:
        for lm in pose_res.pose_landmarks:
            pts = [_get_px(p, image) for p in lm]
            for a, b in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(image, pts[a], pts[b], (80, 220, 80), 2)
            for pt in pts:
                cv2.circle(image, pt, 3, (0, 255, 100), -1)

    # ---- Hands skeleton (left=cyan, right=orange) ----
    if hand_res.hand_landmarks:
        for idx, hand in enumerate(hand_res.hand_landmarks):
            color = (0, 220, 255) if idx == 0 else (0, 140, 255)
            line_color = (0, 160, 200) if idx == 0 else (0, 100, 200)
            pts = [_get_px(p, image) for p in hand]
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(image, pts[a], pts[b], line_color, 2)
            for pt in pts:
                cv2.circle(image, pt, 4, color, -1)
                cv2.circle(image, pt, 4, (255, 255, 255), 1)


# ---------------- KEYPOINT EXTRACTION ---------------- #
def extract_keypoints(pose_res, hand_res, face_res):

    # -------- POSE (33 x 4 = 132) --------
    if pose_res.pose_landmarks:
        pose = np.array(
            [[p.x, p.y, p.z, p.visibility]
             for p in pose_res.pose_landmarks[0]],
            dtype=np.float32
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    # -------- FACE (FORCE 468 x 3 = 1404) --------
    face = np.zeros(468 * 3, dtype=np.float32)
    if face_res.face_landmarks:
        raw_face = np.array(
            [[p.x, p.y, p.z]
             for p in face_res.face_landmarks[0]],
            dtype=np.float32
        ).flatten()

        face[:min(len(raw_face), 468 * 3)] = raw_face[:468 * 3]

    # -------- HANDS (21 x 3 x 2 = 126) --------
    lh = np.zeros(21 * 3, dtype=np.float32)
    rh = np.zeros(21 * 3, dtype=np.float32)

    if hand_res.hand_landmarks:
        if len(hand_res.hand_landmarks) > 0:
            lh[:] = np.array(
                [[p.x, p.y, p.z]
                 for p in hand_res.hand_landmarks[0]],
                dtype=np.float32
            ).flatten()
        if len(hand_res.hand_landmarks) > 1:
            rh[:] = np.array(
                [[p.x, p.y, p.z]
                 for p in hand_res.hand_landmarks[1]],
                dtype=np.float32
            ).flatten()

    # -------- CONCAT (EXACT HOLISTIC FORMAT) --------
    keypoints = np.concatenate([pose, face, lh, rh])

    # FINAL GUARANTEE
    if keypoints.shape[0] != 1662:
        raise RuntimeError(f"Feature mismatch: {keypoints.shape[0]}")

    return keypoints

