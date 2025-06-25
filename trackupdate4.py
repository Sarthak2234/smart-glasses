import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Compute Eye Aspect Ratio (EAR)
def compute_ear(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Load dlib's face detector and facial landmarks model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("X:\CSE\gitclones\smart glasses\smart-glasses\shape_predictor_68_face_landmarks.dat")  # Download required model

# Select the right eye (indices for one eye from dlib's 68 landmarks)
RIGHT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41]  # Change to [42, 43, 44, 45, 46, 47] for left eye

# EAR threshold and frame count for blink detection
EAR_THRESHOLD = 0.25  # Adjust based on camera and environment
CONSEC_FRAMES = 3  # Number of frames for a blink
blink_count = 0
frame_counter = 0

# Start video capture (Change 0 to your webcam index if using external camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE_LANDMARKS])

        # Compute EAR for the detected eye
        ear = compute_ear(right_eye)
        
        # Draw eye landmarks
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Blink detection logic
        if ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= CONSEC_FRAMES:
                blink_count += 1
                print(f"Blink detected! Count: {blink_count}")
            frame_counter = 0

        # Display EAR value
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Eye Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
