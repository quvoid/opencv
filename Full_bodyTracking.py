import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect pose
    results = pose.process(rgb_frame)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Optionally: Track specific exercise (e.g., squats)
        # You can implement your logic here to analyze the pose landmarks
        if detect_squat(results.pose_landmarks):
            update_exercise_status("Squat Detected!")
        else:
            update_exercise_status("No Squat Detected")

    # Display the frame
    cv2.imshow("Full Body Tracking", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

def detect_squat(pose_landmarks):
    # Example logic for squat detection
    if pose_landmarks:
        # Extract relevant landmarks (e.g., hip, knee, ankle)
        left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

        # Calculate angles or distances to determine if a squat is happening
        # Example condition for a squat can be added here
        if left_knee.y < left_hip.y and left_knee.y < left_ankle.y:  # Simplified condition
            return True
    return False

def update_exercise_status(status):
    # Update UI or console with the current exercise status
    print(status)
