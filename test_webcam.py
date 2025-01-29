import cv2

def test_webcam():
    cap = cv2.VideoCapture(0)  # Use the first camera
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
