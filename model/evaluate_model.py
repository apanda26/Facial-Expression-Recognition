from tensorflow.keras.models import load_model
from utils.preprocess import load_dataset

test_dir = "dataset/test"
_, test_generator = load_dataset(None, test_dir)  # Load only test data

model = load_model("model/emotion_model.h5")

loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy:.2f}")
