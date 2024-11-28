from model_training import train_model
from svm_model import svm_accuracy
from rnn_model import rnn_accuracy
import matplotlib.pyplot as plt

print("Model Comparison:")

# Set up your data directory and model save path
data_dir = "Data"
model_save_path = "models/gesture_recognition_model.h5"

# Train the CNN model and capture both accuracies
cnn_train_accuracy, cnn_val_accuracy = train_model(data_dir, model_save_path)

print(f"CNN Training Accuracy: {cnn_train_accuracy * 100:.2f}%")
print(f"CNN Validation Accuracy: {cnn_val_accuracy * 100:.2f}%")
print(f"SVM Model Accuracy: {svm_accuracy * 100:.2f}%")
print(f"RNN Model Accuracy: {rnn_accuracy * 100:.2f}%")

# Prepare data for plotting
models = ['CNN Train', 'CNN Val', 'SVM', 'RNN']
accuracies = [cnn_train_accuracy * 100, cnn_val_accuracy * 100, svm_accuracy * 100, rnn_accuracy * 100]

# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(models, accuracies, marker='o', linestyle='-', color='b', label="Model Accuracies")
# Add labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 100)  # Set the y-axis limit from 0 to 100
plt.grid(axis='y', linestyle='--')

# Show the plot
plt.show()