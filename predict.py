import numpy as np
import pickle

from tensorflow.keras.models import load_model
from feature_extraction import extract_features


# Load trained model
model = load_model("emotion_model.h5")


# Load label encoder
with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)



def predict_emotion(audio_path):

    # Extract MFCC features
    features = extract_features(audio_path)


    # Reshape for CNN-LSTM input
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)


    # Prediction
    prediction = model.predict(features)


    # Get predicted emotion index
    emotion_index = np.argmax(prediction)


    # Convert index to emotion name
    emotion = encoder.inverse_transform(
        [emotion_index]
    )


    return emotion[0]



# Test audio file
audio_file = "test_audio/test.wav"


result = predict_emotion(audio_file)


print("Predicted Emotion:", result)
