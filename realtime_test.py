import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

actions = ['hello','love','thank you','help','more','please']
model = load_model('action.h5', compile=False)

sequence = []
sentence = []
threshold = 0.7

cap = cv2.VideoCapture(0)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_res, hand_res, face_res = mediapipe_detection(frame_rgb, frame_idx)
    draw_styled_landmarks(frame, pose_res, hand_res, face_res)

    keypoints = extract_keypoints(pose_res, hand_res, face_res)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

        if res[np.argmax(res)] > threshold:
            sentence.append(actions[np.argmax(res)])
            sentence = sentence[-5:]

    cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
    cv2.putText(frame, ' '.join(sentence), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('OpenCV Feed', frame)
    frame_idx += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

