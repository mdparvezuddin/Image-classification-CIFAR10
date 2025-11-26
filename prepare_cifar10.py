import numpy as np
import pickle
import os

data_folder = "cifar-10-batches-py"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load all training batches
x_train = []
y_train = []
for i in range(1, 6):
    batch = unpickle(os.path.join(data_folder, f"data_batch_{i}"))
    x = batch[b'data']
    x = x.reshape(-1, 3, 32, 32).transpose(0,2,3,1)
    y = batch[b'labels']
    x_train.append(x)
    y_train.extend(y)
x_train = np.concatenate(x_train)
y_train = np.array(y_train)

# Load test batch
test = unpickle(os.path.join(data_folder, "test_batch"))
x_test = test[b'data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1)
y_test = np.array(test[b'labels'])

# Normalize images to [0,1]
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

print("Training data shape:", x_train.shape, y_train.shape)
print("Test data shape:", x_test.shape, y_test.shape)
print("Min, max pixel values (train):", x_train.min(), x_train.max())
print("Min, max pixel values (test):", x_test.min(), x_test.max())
