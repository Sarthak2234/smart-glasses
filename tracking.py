import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def detect_blinks(frame, landmarks):
    """Detect blinks in the given frame using Eye Aspect Ratio (EAR)."""
    left_eye = landmarks[36:42]  # Left eye landmarks
    right_eye = landmarks[42:48]  # Right eye landmarks

    # Calculate EAR for both eyes
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance
        B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance
        C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
        return (A + B) / (2.0 * C)

    ear_left = eye_aspect_ratio(left_eye)
    ear_right = eye_aspect_ratio(right_eye)
    ear = (ear_left + ear_right) / 2.0

    # Blink detection threshold
    blink_threshold = 0.25
    return ear < blink_threshold  # Return True if blink detected

def measure_eyelid_openness(frame, landmarks):
    """Measure the openness of the eyelid."""
    left_eye = landmarks[36:42]  # Left eye landmarks
    right_eye = landmarks[42:48]  # Right eye landmarks

    # Calculate the vertical distance between eyelids
    left_openness = np.linalg.norm(left_eye[1] - left_eye[5])
    right_openness = np.linalg.norm(right_eye[1] - right_eye[5])
    
    # Normalize to a percentage
    return (left_openness + right_openness) / 2.0

def track_pupil_movement(frame):
    """Track the movement of the pupil."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Hough Circle Transform to detect the pupil
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                param1=50, param2=30, minRadius=5, maxRadius=15)
    if circles is not None:
        return circles[0][0]  # Return the first detected circle
    return (0, 0)  # Return a default value if no pupil is detected

def calculate_alertness(blinked, eyelid_openness, pupil_position):
    """Calculate the alertness level based on the metrics."""
    # Simple alertness calculation logic
    alertness_score = (blinked + eyelid_openness) / 2  # Example calculation
    return alertness_score

def main(camera_index=1):  # Allow camera index to be specified
    cap = cv2.VideoCapture(camera_index)  # Use the specified camera
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])
            blinked = detect_blinks(frame, landmarks)
            eyelid_openness = measure_eyelid_openness(frame, landmarks)
            pupil_position = track_pupil_movement(frame)

            alertness_score = calculate_alertness(blinked, eyelid_openness, pupil_position)
            print(f"Alertness Score: {alertness_score}")

        else:
            print("No landmarks detected.")

        # Display the frame
        cv2.imshow('Eye Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  # Call main with the default camera index