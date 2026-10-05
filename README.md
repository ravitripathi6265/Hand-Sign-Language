# 🤟 Indian Sign Language Detection

A real-time **Indian Sign Language (ISL) hand gesture detection system** built using Computer Vision and Machine Learning. The system uses a webcam to capture hand gestures, processes the input, and predicts the corresponding sign in real time.

The project was developed as a collaborative team project with the objective of using computer vision and machine learning to assist communication for people with hearing and speech disabilities.

## 🎯 Project Objective

The main objective of this project is to build a computer-vision based system capable of recognizing Indian Sign Language hand gestures from live webcam input.

The system follows this pipeline:

```text
Webcam
   ↓
Video Frame Capture
   ↓
Hand / Gesture Processing
   ↓
Feature Extraction
   ↓
Machine Learning Model
   ↓
Gesture Prediction
   ↓
Displayed Output
```

## ✨ Features

- Real-time hand gesture detection
- Webcam-based input
- Indian Sign Language gesture recognition
- Machine-learning based classification
- Custom gesture dataset
- Model training and testing pipeline
- Real-time inference
- Web-based interface
- Gesture class mapping through `class_names.json`

## 🛠️ Technologies Used

### Programming Language

- Python

### Computer Vision

- OpenCV
- MediaPipe

### Machine Learning

- TensorFlow / Keras
- NumPy
- Scikit-learn

### Frontend

- HTML
- CSS
- JavaScript

### Development

- Git
- GitHub

## 📂 Project Structure

```text
Hand-Sign-Language/
│
├── sign_images/
│
├── static/
│
├── templates/
│
├── app_final.py
├── collect_data.py
├── learn_app.py
├── realtime_infer.py
├── test_app.py
├── train_model.py
│
├── class_names.json
├── gesture_dataset.csv
├── gesture_model.h5
├── front_final.html
│
└── README.md
```

## 🔄 Project Workflow

### 1. Data Collection

`collect_data.py` is used for collecting gesture data that can be used for training the recognition model.

The collected data forms the basis for training the system to recognize different hand-sign classes.

### 2. Dataset Preparation

The gesture data is stored in:

```text
gesture_dataset.csv
```

The dataset is processed into a format that can be used by the machine-learning training pipeline.

### 3. Model Training

The model-training process is implemented in:

```text
train_model.py
```

The trained model is saved as:

```text
gesture_model.h5
```

### 4. Class Mapping

The recognized gesture classes are mapped using:

```text
class_names.json
```

This allows the prediction output to be converted from a model class/index into the corresponding gesture label.

### 5. Real-Time Inference

The real-time prediction pipeline is implemented in:

```text
realtime_infer.py
```

The webcam provides live frames, which are processed and passed through the trained model to generate gesture predictions.

### 6. Application

The project also contains:

```text
app_final.py
```

along with:

```text
templates/
static/
front_final.html
```

which provide the application/interface layer for interacting with the recognition system.

## 🧠 Machine Learning Pipeline

The project follows a standard machine-learning workflow:

```text
Gesture Images / Data
        ↓
Data Collection
        ↓
Dataset Preparation
        ↓
Feature Processing
        ↓
Model Training
        ↓
Trained Model
        ↓
Real-Time Webcam Input
        ↓
Prediction
        ↓
Gesture Label
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ravitripathi6265/Hand-Sign-Language.git
```

### 2. Enter the project directory

```bash
cd Hand-Sign-Language
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install dependencies

If the project contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies required by the Python scripts.

## ▶️ Running the Project

To run the main application:

```bash
python app_final.py
```

For real-time inference:

```bash
python realtime_infer.py
```

To train the model:

```bash
python train_model.py
```

To collect additional gesture data:

```bash
python collect_data.py
```

## 🧪 Testing

The repository also contains:

```text
test_app.py
```

which can be used for testing the application and recognition pipeline.

##

## 👨‍💻 My Contribution

### Ravi Narayan Tripathi

I contributed to the core development and implementation of the project, including:

- Development of the gesture-recognition pipeline
- Webcam and real-time inference integration
- Computer-vision processing
- Machine-learning model integration
- Dataset processing and experimentation
- Model training and testing
- Application integration
- Debugging and performance testing

> The contribution list should be adjusted to match the exact modules you personally implemented.

## 🔮 Future Improvements

The project can be extended with:

- Larger ISL gesture vocabulary
- Improved recognition accuracy
- Continuous sentence formation
- Hindi text output
- English text output
- Text-to-speech conversion
- Better recognition under different lighting conditions
- Two-hand gesture support
- Dynamic gesture recognition
- Mobile application
- Cloud deployment

## Project Status

This project was developed as a team-based computer-vision and machine-learning project and can be further extended into a complete Indian Sign Language translation system.

##

##
