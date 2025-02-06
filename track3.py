import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Eye landmarks for EAR calculation (six key points)
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Blink detection parameters
BLINK_THRESHOLD = 0.2  # Adjusted threshold for EAR
CONSEC_FRAMES = 3  # Minimum frames eye should be closed to register a blink

blink_counter = 0
frame_counter = 0
is_blinking = False
start_time = time.time()

def eye_aspect_ratio(eye_points):
    """Calculate Eye Aspect Ratio (EAR) using six landmark points."""
    p1, p2, p3, p4, p5, p6 = eye_points

    # Vertical distances
    vertical_1 = abs(p2[1] - p6[1])
    vertical_2 = abs(p3[1] - p5[1])

    # Horizontal distance
    horizontal = abs(p1[0] - p4[0])

    # Compute EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

cap = cv2.VideoCapture(1)  # Use 1 for external camera, 0 for internal camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        right_eye = []
        left_eye = []

        for idx in RIGHT_EYE_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            right_eye.append((x, y))

        for idx in LEFT_EYE_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            left_eye.append((x, y))

        if len(right_eye) == 6 and len(left_eye) == 6:
            ear_right = eye_aspect_ratio(right_eye)
            ear_left = eye_aspect_ratio(left_eye)
            avg_ear = (ear_right + ear_left) / 2.0

            if avg_ear < BLINK_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_counter += 1
                frame_counter = 0

        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            print(f"In the last minute, number of blinks: {blink_counter}")
            blink_counter = 0
            start_time = time.time()

        cv2.putText(frame, f"Blinks: {blink_counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Eye Tracking', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
