import cv2
def test_webcam(camera_index=1):  # Default to 1 for the secondary camera
    cap = cv2.VideoCapture(camera_index)  # Use the specified camera
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Webcam Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()