import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

def mediapipe_detection(frame, frame_idx):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    pose_results = pose_landmarker.detect_for_video(mp_image, frame_idx)
    hand_results = hand_landmarker.detect_for_video(mp_image, frame_idx)
    face_results = face_landmarker.detect_for_video(mp_image, frame_idx)

    return pose_results, hand_results, face_results


def draw_styled_landmarks(image, pose_res, hand_res, face_res):

    # Pose
    if pose_res.pose_landmarks:
        for lm in pose_res.pose_landmarks:
            for p in lm:
                x, y = int(p.x * image.shape[1]), int(p.y * image.shape[0])
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # Hands
    if hand_res.hand_landmarks:
        for hand in hand_res.hand_landmarks:
            for p in hand:
                x, y = int(p.x * image.shape[1]), int(p.y * image.shape[0])
                cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

    # Face
    if face_res.face_landmarks:
        for face in face_res.face_landmarks:
            for p in face:
                x, y = int(p.x * image.shape[1]), int(p.y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


def extract_keypoints(pose_res, hand_res, face_res):


    if pose_res.pose_landmarks:
        pose = np.array(
            [[p.x, p.y, p.z, p.visibility]
             for p in pose_res.pose_landmarks[0]],
            dtype=np.float32
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    face = np.zeros(468 * 3, dtype=np.float32)
    if face_res.face_landmarks:
        raw_face = np.array(
            [[p.x, p.y, p.z]
             for p in face_res.face_landmarks[0]],
            dtype=np.float32
        ).flatten()

        face[:min(len(raw_face), 468 * 3)] = raw_face[:468 * 3]

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

    keypoints = np.concatenate([pose, face, lh, rh])

    if keypoints.shape[0] != 1662:
        raise RuntimeError(f"Feature mismatch: {keypoints.shape[0]}")
    return keypoints

