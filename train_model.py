# train_model.py
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical


FILENAME = "gesture_dataset.csv"
SEQ_LENGTH = 30   # must match what you used in data collection
MODEL_OUT = "gesture_model.h5"
CLASS_JSON = "class_names.json"

#reading dataa
print(" Loading dataset...")
df = pd.read_csv(FILENAME)
print("Dataset shape:", df.shape)

feature_cols = [c for c in df.columns if c not in ["seq_id", "frame", "label"]]

# group into sequence
X, y = [], []
for seq_id, group in df.groupby("seq_id"):
    group = group.sort_values("frame")
    arr = group[feature_cols].values
    # pad/truncate to SEQ_LENGTH
    if arr.shape[0] >= SEQ_LENGTH:
        arr = arr[:SEQ_LENGTH]
    else:
        pad = np.zeros((SEQ_LENGTH - arr.shape[0], arr.shape[1]))
        arr = np.vstack([arr, pad])
    X.append(arr)
    y.append(group["label"].iloc[0])

X = np.array(X)
print("X shape:", X.shape)

#label encode
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)
num_classes = len(le.classes_)
print("Classes:", le.classes_)

#data split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.15, random_state=42, stratify=y_cat
)

#buildng model
model = Sequential([
    Masking(mask_value=0., input_shape=(SEQ_LENGTH, X.shape[2])),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

#training
callbacks = [
    ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

print("🚀 Training started...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=80, batch_size=16, callbacks=callbacks)

#save classs
with open(CLASS_JSON, "w") as f:
    json.dump(list(le.classes_), f)

print(f"\n✅ Training complete. Model saved as '{MODEL_OUT}' and labels in '{CLASS_JSON}'.")
