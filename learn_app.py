import cv2
import mediapipe as mp
import numpy as np
import json
import os
import random
import time
from tensorflow.keras.models import load_model
from collections import deque
import pyttsx3

#seting
SEQ_LENGTH = 30
MODEL_PATH = "gesture_model.h5"
CLASS_JSON = "class_names.json"
IMAGE_DIR = "sign_images"   # folder containing reference images
THRESHOLD = 0.7

# loading model
model = load_model(MODEL_PATH)
with open(CLASS_JSON, "r") as f:
    class_names = json.load(f)

# mp hand 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# tts 
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# main
print("Welcome to ISL Learning Mode 👋")
speak("Welcome to Indian Sign Language learning mode.")
input("Press Enter to begin learning...")

cap = cv2.VideoCapture(0)
buffer = deque(maxlen=SEQ_LENGTH)

while cap.isOpened():
    # choose a random sign
    word = random.choice(class_names)
    print(f"\nLearn the sign for '{word}'")
    speak(f"Show the sign for {word}")

    # find the correct image regardless of extension 
    img_path = None
    for ext in ["png", "jpg", "jpeg"]:
        path_try = os.path.join(IMAGE_DIR, f"{word}.{ext}")
        if os.path.exists(path_try):
            img_path = path_try
            break

    # show reference if found 
    if img_path is not None:
        ref = cv2.imread(img_path)
        cv2.imshow("Reference", ref)
        ref_found = True
    else:
        ref_found = False
        print(f"⚠️ Reference image not found for '{word}'")

    correct = False

    # Wait until user performs correctly
    while not correct:
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
            pred = class_names[idx]
            conf = float(probs[idx])

            if pred == word and conf > THRESHOLD:
                cv2.putText(frame, f"✅ Correct! {word}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                print(f"✅ Correct! You performed '{word}' correctly.")
                speak(f"Good job. You performed {word} correctly.")
                time.sleep(1)
                correct = True
                break
            else:
                cv2.putText(frame, f"Try again...", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # overlay info 
        cv2.putText(frame, f"Sign: {word}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if not ref_found:
            cv2.putText(frame, "⚠️ No reference image found!", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Learning Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("Exiting Learning Mode...")
            speak("Exiting learning mode.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # close reference window after correct
    try:
        if cv2.getWindowProperty("Reference", cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow("Reference")
    except:
        pass

    time.sleep(1)  # short pause before next word

cap.release()
cv2.destroyAllWindows()
