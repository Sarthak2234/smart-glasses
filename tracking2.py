import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def detect_eye_blinks(landmarks):
    """Detect blinks in the given landmarks using Eye Aspect Ratio (EAR) for the right eye."""
    right_eye = landmarks[42:48]  # Right eye landmarks

    # Calculate EAR for the right eye
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance
        B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance
        C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
        return (A + B) / (2.0 * C)

    ear_right = eye_aspect_ratio(right_eye)

    # Blink detection threshold
    blink_threshold = 0.25
    return ear_right < blink_threshold  # Return True if blink detected

def measure_eyelid_openness(landmarks):
    """Measure the openness of the right eyelid."""
    right_eye = landmarks[42:48]  # Right eye landmarks

    # Calculate the vertical distance between eyelids
    right_openness = np.linalg.norm(right_eye[1] - right_eye[5])
    
    # Normalize to a percentage
    return right_openness  # Return the openness value

def main(camera_index=1):  # Allow camera index to be specified
    cap = cv2.VideoCapture(camera_index)  # Use the specified camera
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    blink_count = 0  # Initialize blink count

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])
            blinked = detect_eye_blinks(landmarks)
            if blinked:
                blink_count += 1  # Increment blink count if a blink is detected

            eyelid_openness = measure_eyelid_openness(landmarks)

            print(f"Eyelid Openness: {eyelid_openness}, Blink Count: {blink_count}")

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