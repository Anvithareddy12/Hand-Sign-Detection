import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_and_preprocess_data

# Load and preprocess the data
data_dir = "Data"  # Modify this to your dataset path
train_images, test_images, train_labels, test_labels = load_and_preprocess_data(data_dir)

# Flatten the images for SVM
train_images_flat = train_images.reshape(train_images.shape[0], -1)  # (num_samples, height*width*channels)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# SVM expects labels as single integers, so we need to convert from one-hot encoding to single labels
train_labels_svm = np.argmax(train_labels, axis=1)
test_labels_svm = np.argmax(test_labels, axis=1)

# Normalize the flattened image data using StandardScaler
scaler = StandardScaler()
train_images_flat = scaler.fit_transform(train_images_flat)
test_images_flat = scaler.transform(test_images_flat)

# Train an SVM classifier
svm_model = svm.SVC(kernel='linear')  # You can also try 'rbf', 'poly' or other kernels
svm_model.fit(train_images_flat, train_labels_svm)

# Predict on the test set
svm_predictions = svm_model.predict(test_images_flat)

# Calculate accuracy
svm_accuracy = accuracy_score(test_labels_svm, svm_predictions)
print(f"SVM Model Accuracy: {svm_accuracy * 100:.2f}%")

# Optional: Display a confusion matrix (to visualize how well the model performs on each label)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(test_labels_svm, svm_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for SVM')
plt.show()