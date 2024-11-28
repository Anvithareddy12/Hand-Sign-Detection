import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_data(data_dir, img_size=300):
    # Initialize lists for images and labels
    images = []
    labels = []

    # Label mapping (A-Z mapped to 0-25)
    label_mapping = {chr(i + 65): i for i in range(26)}  # {'A': 0, 'B': 1, ..., 'Z': 25}

    # Loop through each label folder in the dataset directory
    for label in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label_mapping[label])

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Normalize images
    images = images / 255.0

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=26)

    # Split the dataset
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2,
                                                                            random_state=42)

    return train_images, test_images, train_labels, test_labels


if __name__ == "__main__":
    data_dir = "Data"  # Modify this to your dataset path
    train_images, test_images, train_labels, test_labels = load_and_preprocess_data(data_dir)
    print(f"Training data shape: {train_images.shape}")
    print(f"Testing data shape: {test_images.shape}")