import cv2
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

# Load the trained model
model = load_model(r"C:\Users\AnvithaReddy\Desktop\Hand Sign Detection\HandSignDetect\models\gesture_recognition_model.h5")

# Function to predict gesture
def predict_gesture(imgWhite, model, img_size=300):
    try:
        imgWhite = cv2.resize(imgWhite, (img_size, img_size))
        imgWhite = np.expand_dims(imgWhite, axis=0)
        imgWhite = imgWhite / 255.0  # Normalize the image

        prediction = model.predict(imgWhite)
        predicted_class = np.argmax(prediction)
        letter = chr(predicted_class + 65)  # Convert to ASCII (A-Z)
        return letter

    except Exception as e:
        print(f"Prediction error: {e}")
        return None

    # return letter


# Real-time webcam prediction with text display
def real_time_prediction():
    cap = cv2.VideoCapture(0)  # Start webcam
    detector = HandDetector(maxHands=1)
    offset = 20
    img_size = 300

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")


            height, width, _ = img.shape
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(width, x + w + offset), min(height, y + h + offset)

            if x1 >= x2 or y1 >= y2:
                print("Invalid cropping dimensions. Skipping frame.")
                continue

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:
                print("Invalid cropped image. Skipping frame.")
                continue

            imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255

            aspect_ratio = h / w




            # imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255
            # imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # aspect_ratio = h / w


        try:
            if aspect_ratio > 1:
                k = img_size / h
                w_cal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (w_cal, img_size))
                wGap = math.ceil((img_size - w_cal) / 2)
                imgWhite[:, wGap:w_cal + wGap] = imgResize
            else:
                k = img_size / w
                h_cal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (img_size, h_cal))
                hGap = math.ceil((img_size - h_cal) / 2)
                imgWhite[hGap:h_cal + hGap, :] = imgResize

            # Predict gesture
            letter = predict_gesture(imgWhite, model)

            if letter:
            # Print predicted letter on the webcam feed
                cv2.putText(img, f'Predicted: {letter}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
                
        except Exception as e:
            print(f"Error in image processing: {e}")
            continue

        # Show the webcam feed
        cv2.imshow("Hand Sign Detection", img)

        if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_prediction()