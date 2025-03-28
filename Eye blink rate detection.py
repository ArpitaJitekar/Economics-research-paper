import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model

# Load pre-trained CNN model for eye state classification (Open/Closed)
model = load_model("eye_blink_cnn.h5")  # Replace with your trained model file

# Load dlib face detector & facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define blink threshold
BLINK_THRESHOLD = 3  # Number of consecutive closed-eye frames to count as a blink
blink_count = 0
frame_count = 0  # Counter for consecutive closed eye frames

# Start Video Capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get eye regions (left & right)
        left_eye = landmarks.parts()[36:42]
        right_eye = landmarks.parts()[42:48]

        def get_eye_region(eye_points):
            x1 = min([p.x for p in eye_points])
            x2 = max([p.x for p in eye_points])
            y1 = min([p.y for p in eye_points])
            y2 = max([p.y for p in eye_points])
            return gray[y1:y2, x1:x2]

        left_eye_img = get_eye_region(left_eye)
        right_eye_img = get_eye_region(right_eye)

        # Preprocess eye images for CNN model
        def preprocess_eye(eye_img):
            eye_img = cv2.resize(eye_img, (48, 48))  # Resize to match model input
            eye_img = eye_img / 255.0  # Normalize
            eye_img = np.expand_dims(eye_img, axis=(0, -1))  # Reshape for CNN
            return eye_img

        left_eye_img = preprocess_eye(left_eye_img)
        right_eye_img = preprocess_eye(right_eye_img)

        # Predict eye state (0 = closed, 1 = open)
        left_pred = np.argmax(model.predict(left_eye_img))
        right_pred = np.argmax(model.predict(right_eye_img))

        if left_pred == 0 and right_pred == 0:  # Both eyes closed
            frame_count += 1
        else:
            if frame_count >= BLINK_THRESHOLD:
                blink_count += 1
            frame_count = 0  # Reset counter when eyes open

        # Display Blink Count
        cv2.putText(frame, f'Blinks: {blink_count}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Eye Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
