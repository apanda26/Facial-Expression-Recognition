import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
MODEL_PATH = "../model/emotion_model.h5"
model = load_model(MODEL_PATH)

# Define emotion labels (must match dataset folders)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI (Region of Interest)
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))  # Resize to match model input
        face = face.astype("float32") / 255.0  # Normalize pixel values
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = np.expand_dims(face, axis=-1)  # Add channel dimension

        # Predict emotion
        predictions = model.predict(face)
        emotion_idx = np.argmax(predictions)
        emotion_text = emotion_labels[emotion_idx]

        # Display the prediction on the video frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the video frame with emotion detection
    cv2.imshow("Emotion Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
