import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow import keras
model = keras.models.load_model("cifar10_cnn_model.keras")


# Set the data folder path (relative, since in project directory)
data_folder = "cifar-10-batches-py"
batch_file = os.path.join(data_folder, "data_batch_1")

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load the first batch
batch = unpickle(batch_file)

# Data info
print("Keys in batch:", batch.keys())
print("Data shape:", batch[b'data'].shape)
print("Labels shape:", len(batch[b'labels']))

# CIFAR-10 images are stored flat, need to reshape: (N, 3072) -> (N, 32, 32, 3)
images = batch[b'data']
images = images.reshape(-1, 3, 32, 32).transpose(0,2,3,1)
labels = batch[b'labels']

# Classes from meta
meta = unpickle(os.path.join(data_folder, "batches.meta"))
label_names = [x.decode('utf-8') for x in meta[b'label_names']]

print("Labels:", label_names)

# Show 10 random images
plt.figure(figsize=(10,2))
for i in range(10):
    idx = np.random.randint(0, len(images))
    plt.subplot(1,10,i+1)
    plt.imshow(images[idx])
    plt.title(label_names[labels[idx]])
    plt.axis('off')
plt.tight_layout()
plt.show()
