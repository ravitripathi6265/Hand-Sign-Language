from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import json
import random
import time
import os
from collections import deque
from tensorflow.keras.models import load_model

# ===============================================================
#  INITIAL SETUP
# ===============================================================

app = Flask(__name__)

# Load model and class names
model = load_model('gesture_model.h5')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

SEQ_LENGTH = 30
THRESHOLD = 0.7
buffer = deque(maxlen=SEQ_LENGTH)

current_word = ""  # shared latest prediction

# ===============================================================
#  1️⃣ RECOGNITION MODE
# ===============================================================

def gen_frames():
    global current_word
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=SEQ_LENGTH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        left = [0.0] * 63
        right = [0.0] * 63

        # extract landmarks just like realtime_infer.py
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

        # append features
        features = np.array(left + right, dtype=np.float32)
        buffer.append(features)

        # predict once buffer filled
        if len(buffer) == SEQ_LENGTH:
            inp = np.expand_dims(np.array(buffer), axis=0)
            probs = model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(probs))
            pred = class_names[idx]
            conf = float(probs[idx])

            if conf > THRESHOLD:
                current_word = pred
                cv2.putText(frame, f"{pred.upper()} ({conf:.2f})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Detecting...", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # display current word overlay
        cv2.putText(frame, f"Current: {current_word}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # encode for web stream
        _, buffer_img = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer_img.tobytes() + b'\r\n')

    cap.release()



# ===============================================================
#  2️⃣ TEST MODE
# ===============================================================

def gen_test_frames():
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=SEQ_LENGTH)
    score = 0
    total = 0
    TIME_LIMIT = 10

    while True:
        word = random.choice(class_names)
        total += 1
        start_time = time.time()
        correct = False

        while True:
            success, frame = cap.read()
            if not success:
                break

            elapsed = time.time() - start_time
            remaining = max(0, TIME_LIMIT - int(elapsed))

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            left, right = [0.0] * 63, [0.0] * 63

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

            buffer.append(np.array(left + right, dtype=np.float32))
            if len(buffer) == SEQ_LENGTH:
                inp = np.expand_dims(np.array(buffer), axis=0)
                probs = model.predict(inp, verbose=0)[0]
                idx = int(np.argmax(probs))
                pred = class_names[idx]
                conf = float(probs[idx])

                if pred == word and conf > THRESHOLD:
                    score += 1
                    correct = True
                    cv2.putText(frame, f"✅ Correct! {word}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    time.sleep(1)
                    break
                else:
                    cv2.putText(frame, " ", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.putText(frame, f"Target: {word}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Score: {score}/{total}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Time left: {remaining}s", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')

            if elapsed >= TIME_LIMIT:
                break
        if not correct:
            print(f"⏰ Time up for '{word}'!")

    cap.release()


# ===============================================================
#  3️⃣ LEARN MODE (with reference image)
# ===============================================================

def gen_learn_frames():
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=SEQ_LENGTH)
    IMAGE_DIR = "sign_images"

    while True:
        word = random.choice(class_names)
        print(f"[LEARN] New word: {word}")

        # find reference image
        img_path = None
        for ext in ["png", "jpg", "jpeg"]:
            test_path = os.path.join(IMAGE_DIR, f"{word}.{ext}")
            if os.path.exists(test_path):
                img_path = test_path
                break
        ref_img = cv2.imread(img_path) if img_path else None
        if ref_img is not None:
            ref_img = cv2.resize(ref_img, (160, 160))

        correct = False
        while not correct:
            success, frame = cap.read()
            if not success:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            left, right = [0.0] * 63, [0.0] * 63
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

            buffer.append(np.array(left + right, dtype=np.float32))
            if len(buffer) == SEQ_LENGTH:
                inp = np.expand_dims(np.array(buffer), axis=0)
                probs = model.predict(inp, verbose=0)[0]
                idx = int(np.argmax(probs))
                pred = class_names[idx]
                conf = float(probs[idx])

                if pred == word and conf > THRESHOLD:
                    cv2.putText(frame, f"✅ Correct! {word}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    correct = True
                    time.sleep(1)
                    break
                else:
                    cv2.putText(frame, f"Try: {word}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if ref_img is not None:
                h, w, _ = frame.shape
                frame[10:170, w - 170:w - 10] = ref_img

            cv2.putText(frame, f"Learning: {word}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')

    cap.release()



@app.route('/')
def index():
    return render_template('front_final.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test_feed')
def test_feed():
    return Response(gen_test_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/learn_feed')
def learn_feed():
    return Response(gen_learn_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_prediction')
def current_prediction():
    return jsonify({'word': current_word})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
