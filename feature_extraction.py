import librosa
import numpy as np


def extract_features(file_path):

    # Load audio file
    audio, sample_rate = librosa.load(file_path)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Convert MFCC into single feature vector
    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc


# Test one audio file
# Change this path if your filename is different

file_path = "audio_speech_actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"


features = extract_features(file_path)


print("MFCC Feature Shape:")
print(features.shape)


print("MFCC Values:")
print(features)
