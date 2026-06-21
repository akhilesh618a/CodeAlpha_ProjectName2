import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D
from tensorflow.keras.utils import to_categorical


# Load extracted features
X = np.load("features.npy")
y = np.load("labels.npy")


print("Original feature shape:", X.shape)


# Convert emotion labels into numbers
encoder = LabelEncoder()

y = encoder.fit_transform(y)

y = to_categorical(y)


# CNN requires 3D input
# (samples, time steps, features)

X = np.expand_dims(X, axis=2)


print("New feature shape:", X.shape)


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create CNN + LSTM model

model = Sequential()


# CNN layer

model.add(
    Conv1D(
        64,
        kernel_size=3,
        activation="relu",
        input_shape=(40,1)
    )
)


model.add(MaxPooling1D(pool_size=2))


# LSTM layer

model.add(
    LSTM(
        64,
        return_sequences=False
    )
)


model.add(Dropout(0.3))


# Output layer

model.add(
    Dense(
        y.shape[1],
        activation="softmax"
    )
)


# Compile model

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# Train model

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# Evaluate

loss, accuracy = model.evaluate(
    X_test,
    y_test
)


print("Test Accuracy:", accuracy)


# Save model

model.save("emotion_model.h5")


# Save label encoder

import pickle

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)


print("Model saved successfully")
