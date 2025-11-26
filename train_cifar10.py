import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load preprocessed data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_folder = "cifar-10-batches-py"

def load_data():
    # Train
    x_train = []
    y_train = []
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_folder, f"data_batch_{i}"))
        x = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1)
        y = batch[b'labels']
        x_train.append(x)
        y_train.extend(y)
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)
    # Test
    test = unpickle(os.path.join(data_folder, "test_batch"))
    x_test = test[b'data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1)
    y_test = np.array(test[b'labels'])
    # Normalize
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)


# Model
# model = keras.Sequential([
#     layers.Conv2D(32, 3, activation='relu', input_shape=(32,32,3)),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

# New model for more accuracy
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=40,  # More epochs, since each epoch is now on augmented data
    validation_data=(x_test, y_test)
)

loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.3f}")

# Save trained model after all training steps
model.save("cifar10_cnn_model.keras")
