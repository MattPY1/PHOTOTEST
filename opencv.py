import cv2
import numpy as np

# Load the pre-trained Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_' + 'frontalface_default' + '.xml')
# Open the camera
print(cv2.data.haarcascades)
cap = cv2.VideoCapture(0)
# cap = cv2.imread('cars.png')
if cap.isOpened():
    print('Camera Was opened')

while True:
    ret, frame = cap.read()
    # frame = cap
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 246), 2)

    # Display the frame
    cv2.imshow('Face Overlay', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
