import cv2
import mediapipe as mp
import numpy as np
import json
import random
import time
from tensorflow.keras.models import load_model
from collections import deque
import os
import pyttsx3

# setting
SEQ_LENGTH = 30
MODEL_PATH = "gesture_model.h5"
CLASS_JSON = "class_names.json"
IMAGE_DIR = "sign_images"
THRESHOLD = 0.7
TIME_LIMIT = 10  # seconds per sign

# loading model
model = load_model(MODEL_PATH)
with open(CLASS_JSON, "r") as f:
    class_names = json.load(f)

# detecting hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# tts
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

print("Welcome to ISL Test Mode with Countdown + Progress Bar!")
speak("Welcome to Indian Sign Language test mode with countdown.")
input("Press Enter to start...")

cap = cv2.VideoCapture(0)
score = 0
total = 0

while cap.isOpened():
    # pick a random word
    word = random.choice(class_names)
    total += 1

    print(f"\nPerform the sign for '{word}' (10 seconds)")
    speak(f"Perform the sign for {word}")

    # show reference image if available
    img_path = os.path.join(IMAGE_DIR, f"{word}.jpg")
    if os.path.exists(img_path):
        ref = cv2.imread(img_path)
        cv2.imshow("Reference", ref)
    else:
        try:
            if cv2.getWindowProperty("Reference", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Reference")
        except:
            pass

    buffer = deque(maxlen=SEQ_LENGTH)
    correct = False
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        remaining = max(0, TIME_LIMIT - int(elapsed))

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        left = [0.0]*63
        right = [0.0]*63

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
            pred = class_names[idx]
            conf = float(probs[idx])

            if pred == word and conf > THRESHOLD:
                cv2.putText(frame, f"✅ Correct! {pred}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                print(f"✅ Correct! ({conf:.2f})")
                speak("Correct")
                score += 1
                correct = True
                time.sleep(1)
                break
            else:
                cv2.putText(frame, f"Try Again...", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Target and score
        cv2.putText(frame, f"Target: {word}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Score: {score}/{total}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Time left: {remaining}s", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

     
        bar_x, bar_y, bar_w, bar_h = 10, 220, 300, 20
        progress = max(0, (TIME_LIMIT - elapsed) / TIME_LIMIT)
        filled_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 2)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)

        cv2.imshow("Quiz Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nFinal Score: {score}/{total}")
            speak(f"Test ended. Final score {score} out of {total}.")
            exit()

        # stop if time runs out
        if elapsed >= TIME_LIMIT:
            break

    if not correct:
        print(f"⏱️ Time's up for '{word}'!")
        speak(f"Time's up for {word}")

    # close reference window if open
    try:
        if cv2.getWindowProperty("Reference", cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow("Reference")
    except:
        pass

    time.sleep(0.8)  # short pause before next word


cap.release()
cv2.destroyAllWindows()
