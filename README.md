# Facial-Expression-Recognition
# Facial Expression Recognition 🎭  

This project detects facial expressions using a deep learning model trained on image datasets. It supports real-time emotion detection via webcam.

## Features
✅ Detects emotions such as Happy, Sad, Angry, Neutral, etc.  
✅ Uses CNN for feature extraction  
✅ Supports real-time webcam inference  
✅ Can be deployed as an API  

## Installation  

1️⃣ Clone the Repository  
git clone https://github.com/apanda26/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Model on Test Images
python test.py --image sample.jpg

4️⃣ Run Real-Time Emotion Detection
python real_time_demo.py
Press Q to exit the webcam interface.

Training the Model (Optional)
If you want to train the model from scratch, use:
python train.py
Dependencies
Python 3.8+
TensorFlow/Keras
OpenCV
NumPy
