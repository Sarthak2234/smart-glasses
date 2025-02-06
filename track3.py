import cv2
import dlib
import numpy as np

# Load Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def main(camera_index=1):  # Allow camera index to be specified
    cap = cv2.VideoCapture(camera_index)  # Use the specified camera
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        rects = detector(gray)

        for rect in rects:
            # Get facial landmarks
            shape = predictor(gray, rect)

            # Access specific eyelid landmark points
            top_eyelid_corner = shape.part(37)  # Top eyelid corner
            bottom_eyelid_corner = shape.part(41)  # Bottom eyelid corner

            # Draw circles on the eyelid corners
            cv2.circle(frame, (top_eyelid_corner.x, top_eyelid_corner.y), 2, (0, 255, 0), -1)
            cv2.circle(frame, (bottom_eyelid_corner.x, bottom_eyelid_corner.y), 2, (0, 0, 255), -1)

            # Calculate eyelid movement or other metrics as needed
            # ...

        # Show the output
        cv2.imshow("Eyelid Tracking", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  # Call main with the default camera index
