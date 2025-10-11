# realtime_infer.py
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import threading
from collections import deque
from tensorflow.keras.models import load_model
import pyttsx3

#setting 
SEQ_LENGTH = 30
MODEL_PATH = "gesture_model.h5"
CLASS_JSON = "class_names.json"
THRESHOLD = 0.7        # minimum softmax confidence
STABLE_COUNT = 3       # require N consecutive identical predictions
COOLDOWN = 1.5         # seconds between spoken outputs

#this will be used for loading model
model = load_model(MODEL_PATH)
with open(CLASS_JSON, "r") as f:
    class_names = json.load(f)

#this is for detection of hand (media pipe)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
buffer = deque(maxlen=SEQ_LENGTH)
last_pred = None
stable = 0
last_spoken_time = 0.0

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# loop untill user will not press ctr+ c
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    left = [0.0] * 63
    right = [0.0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handed = results.multi_handedness[idx].classification[0].label.lower()
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if handed == "left":
                left = coords
            else:
                right = coords
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    features = np.array(left + right, dtype=np.float32)
    buffer.append(features)

    if len(buffer) == SEQ_LENGTH:
        inp = np.expand_dims(np.array(buffer), axis=0)
        probs = model.predict(inp, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        pred = class_names[idx]


        if pred == last_pred:
            stable += 1
        else:
            stable = 1
            last_pred = pred

        if stable >= STABLE_COUNT and conf >= THRESHOLD and (time.time() - last_spoken_time) > COOLDOWN:
            print(f"Recognized: {pred} ({conf:.2f})")
            threading.Thread(target=speak, args=(pred,), daemon=True).start()
            last_spoken_time = time.time()

        cv2.putText(frame, f"PRED: {pred} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
