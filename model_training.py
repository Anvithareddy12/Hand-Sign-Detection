import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import load_and_preprocess_data

# Define CNN model function
def create_model(img_size=300):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation='softmax'))  # 26 classes for A-Z gestures
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(data_dir, model_save_path, epochs=10, batch_size=32):
    # Load and preprocess data
    train_images, test_images, train_labels, test_labels = load_and_preprocess_data(data_dir)

    # Create the model
    model = create_model()

    # Train the model and store the history
    history = model.fit(train_images, train_labels, epochs=epochs, validation_split=0.2, batch_size=batch_size)

    # Save the model
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")

    # Get accuracy values
    train_accuracy = history.history['accuracy'][-1]  # Last training accuracy
    val_accuracy = history.history['val_accuracy'][-1]  # Last validation accuracy

    # Evaluate the model on the test set
    test_predictions = model.predict(test_images)
    predicted_classes = np.argmax(test_predictions, axis=1)  # Convert predictions to class labels
    true_classes = np.argmax(test_labels, axis=1)  # Convert one-hot encoded labels to class labels

    # Calculate the confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    # Visualize the confusion matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), yticklabels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for CNN Model')
    plt.show()

    return train_accuracy, val_accuracy

if __name__ == "__main__":
    data_dir = "Data"
    model_save_path = "models/gesture_recognition_model.h5"
    cnn_train_accuracy, cnn_val_accuracy = train_model(data_dir, model_save_path)

    print(f"CNN Training Accuracy: {cnn_train_accuracy * 100:.2f}%")
    print(f"CNN Validation Accuracy: {cnn_val_accuracy * 100:.2f}%")