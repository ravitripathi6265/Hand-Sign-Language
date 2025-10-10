# collect_data.py
import cv2, time, os, uuid
import mediapipe as mp
import pandas as pd
import numpy as np
# This is the file name:
FILENAME = "gesture_dataset.csv"#csv data will be saved in this file 
SEQ_LENGTH = 30
NUM_SAMPLES = 50
DETECTION_CONF = 0.5

# This is for scanning of hand :
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=DETECTION_CONF,
                       min_tracking_confidence=DETECTION_CONF)
mp_drawing = mp.solutions.drawing_utils
# for coordinates:
cols = ["seq_id", "frame", "label"]
for side in ["left", "right"]:
    for i in range(21):
        for c in ["x", "y", "z"]:
            cols.append(f"{side}_lm{i}_{c}")

if not os.path.exists(FILENAME):
    pd.DataFrame(columns=cols).to_csv(FILENAME, index=False)

cap = cv2.VideoCapture(0)
label = input("Enter label/class name (e.g. hello): ").strip().replace(" ", "_")
num_samples = int(input(f"How many samples for '{label}'? (e.g. 50): ") or NUM_SAMPLES)

print("Starting in 3 seconds… get ready!")
time.sleep(3)
#Loop for all iteration
for sample_i in range(num_samples):
    seq_id = f"{label}_{int(time.time())}_{sample_i}_{uuid.uuid4().hex[:6]}"
    print(f"Recording sample {sample_i+1}/{num_samples}")
    time.sleep(2)

    frame_i = 0
    while frame_i < SEQ_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        left = [0.0] * (21 * 3)
        right = [0.0] * (21 * 3)

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

        row = [seq_id, frame_i, label] + left + right
        pd.DataFrame([row], columns=cols).to_csv(FILENAME, mode='a', header=False, index=False)

        cv2.putText(frame, f"Label:{label}  Sample:{sample_i+1}/{num_samples}  Frame:{frame_i+1}/{SEQ_LENGTH}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Recording (Esc to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release(); cv2.destroyAllWindows(); exit()
        frame_i += 1

    print("Sample saved.\n")

cap.release()
cv2.destroyAllWindows()
print(" Data collection complete.")
