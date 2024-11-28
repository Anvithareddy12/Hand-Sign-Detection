import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data

# Load and preprocess the data
data_dir = "Data"  # Modify this to your dataset path
train_images, test_images, train_labels, test_labels = load_and_preprocess_data(data_dir)

# Reshape the images for RNN (Reshape to (num_samples, time_steps, features))
# Here, time_steps can be the height of the image, and features can be the width * channels
train_images_rnn = train_images.reshape((train_images.shape[0], train_images.shape[1], -1))
test_images_rnn = test_images.reshape((test_images.shape[0], test_images.shape[1], -1))

# Build RNN model
rnn_model = Sequential([
    SimpleRNN(64, input_shape=(train_images_rnn.shape[1], train_images_rnn.shape[2]), activation='relu'),
    Dense(26, activation='softmax')  # 26 classes for A-Z
])

# Compile the model
rnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the RNN model
rnn_model.fit(train_images_rnn, train_labels, epochs=10, batch_size=32)

# Evaluate the RNN model
rnn_loss, rnn_accuracy = rnn_model.evaluate(test_images_rnn, test_labels)
print(f"RNN Model Accuracy: {rnn_accuracy * 100:.2f}%")

# Optional: Predict and visualize results similar to the SVM example
rnn_predictions = rnn_model.predict(test_images_rnn)
rnn_predictions_labels = np.argmax(rnn_predictions, axis=1)

# You can also calculate accuracy similarly
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Convert test_labels back to single labels
test_labels_rnn = np.argmax(test_labels, axis=1)

# Calculate accuracy
rnn_accuracy_score = accuracy_score(test_labels_rnn, rnn_predictions_labels)
print(f"RNN Model Accuracy: {rnn_accuracy_score * 100:.2f}%")

# Display confusion matrix
conf_matrix = confusion_matrix(test_labels_rnn, rnn_predictions_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for RNN')
plt.show()