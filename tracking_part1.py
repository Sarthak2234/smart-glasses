import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Eye landmarks indices (MediaPipe 468-point model)
RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173]

def process_eye_region(frame, eye_points):
    """Process eye region for pupil detection"""
    # Create mask and extract eye region
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(eye_points, dtype=np.int32)], 255)
    eye_region = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Get bounding coordinates
    min_x = min(p[0] for p in eye_points)
    max_x = max(p[0] for p in eye_points)
    min_y = min(p[1] for p in eye_points)
    max_y = max(p[1] for p in eye_points)
    
    # Crop and process eye image
    cropped_eye = eye_region[min_y:max_y, min_x:max_x]
    gray_eye = cv2.cvtColor(cropped_eye, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_eye, (9,9), 0)
    
    # Adaptive thresholding for pupil detection
    _, thresholded = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours and detect pupil
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    pupil_position = None
    if len(contours) > 0:
        (x,y), radius = cv2.minEnclosingCircle(contours[0])
        pupil_position = (int(x + min_x), int(y + min_y))
        radius = int(radius)
        
    return pupil_position, thresholded

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        eye_points = []
        
        # Extract right eye landmarks
        for idx in RIGHT_EYE_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            eye_points.append((x,y))
        
        # Process eye region and detect pupil
        pupil_pos, processed_eye = process_eye_region(frame, eye_points)
        
        # Draw results
        if pupil_pos:
            cv2.circle(frame, pupil_pos, 3, (0,0,255), -1)
        
        # Display processed eye
        cv2.imshow('Processed Eye', processed_eye)
    
    cv2.imshow('Eye Tracking', frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()