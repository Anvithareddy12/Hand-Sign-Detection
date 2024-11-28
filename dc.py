import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize webcam
cap = cv2.VideoCapture(0)

# Hand detector with max 1 hand
detector = HandDetector(maxHands=1)

# Parameters for image processing
offset = 20
imgSize = 300

# Base directory for saving data (this is where images will be saved)
base_folder = "Data"

# Prompt the user to input the label (for example, 'A', 'B', 'C', etc.)
label = input("Enter the gesture/letter you are collecting data for (A-Z): ").upper()

# Create the folder if it doesn't exist
folder = os.path.join(base_folder, label)
if not os.path.exists(folder):
    os.makedirs(folder)

# Initialize a counter to track the number of images saved
counter = 0

# Main loop for capturing and processing hand images
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # Detect hand and return modified image
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Bounding box around the hand
        
            # Boundary checks
        height, width, _ = img.shape
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(width, x + w + offset), min(height, y + h + offset)
        imgCrop = img[y1:y2, x1:x2]




        # Create a white image of size 300x300 for resizing
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Get the shape of the cropped image
        imgCropShape = imgCrop.shape

        # Aspect ratio adjustment
        aspectRatio = h / w

        if aspectRatio > 1:  # If height is greater than width
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize  # Paste into white image
        else:  # If width is greater than height
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Paste into white image

        # Show the cropped and processed images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Display the live webcam feed
    cv2.imshow('Image', img)

    # Press 's' to save the processed image
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        # Save image with timestamp to avoid overwriting
        img_path = os.path.join(folder, f'Image_{time.time()}.jpg')
        cv2.imwrite(img_path, imgWhite)
        print(f"Saved image: {counter} for label: {label}")

    # Press 'q' to quit the program
    if key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()