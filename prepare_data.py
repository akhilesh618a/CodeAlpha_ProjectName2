import os
import numpy as np
from feature_extraction import extract_features


data = []
labels = []


# Dataset folder
dataset_path = "audio_speech_actors_01-24"


# Emotion mapping
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# Loop through Actor folders
for actor in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                # Get emotion code from filename
                emotion_code = file.split("-")[2]

                emotion = emotion_map[emotion_code]


                # Full audio path
                file_path = os.path.join(actor_path, file)


                # Extract MFCC features
                features = extract_features(file_path)


                data.append(features)
                labels.append(emotion)


                print("Processed:", file)


# Convert list to numpy arrays
data = np.array(data)
labels = np.array(labels)


# Save features and labels
np.save("features.npy", data)
np.save("labels.npy", labels)


print("\nFeature extraction completed")
print("Feature shape:", data.shape)
print("Labels shape:", labels.shape)
