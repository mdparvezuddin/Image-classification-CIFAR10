import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Reload model and data (or use from previous session)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_folder = "cifar-10-batches-py"

def load_data():
    x_test = []
    y_test = []
    test = unpickle(os.path.join(data_folder, "test_batch"))
    x = test[b'data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1)
    y = test[b'labels']
    x_test = np.array(x)
    y_test = np.array(y)
    x_test = x_test.astype(np.float32) / 255.0
    return x_test, y_test

x_test, y_test = load_data()

# If you saved your model:
# model = keras.models.load_model('your_model_path')
# Otherwise, use the trained model object from train_cifar10.py:
from train_cifar10 import model

# Predict
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('CIFAR-10 Confusion Matrix')
plt.show()

# Print detailed report
print(classification_report(y_test, y_pred_classes, target_names=labels))
