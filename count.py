import cv2
import mediapipe as mp
import numpy as np
import time

class FullBodyTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize performance metrics
        self.prev_frame_time = 0
        self.new_frame_time = 0

        # Initialize exercise counting variables
        self.squat_count = 0
        self.is_squatting = False

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                 np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def get_landmark_coordinates(self, landmarks, width, height):
        """Convert normalized landmarks to pixel coordinates."""
        coords = {}
        for landmark in self.mp_pose.PoseLandmark:
            landmark_px = self.mp_draw._normalized_to_pixel_coordinates(
                landmarks.landmark[landmark].x,
                landmarks.landmark[landmark].y,
                width,
                height
            )
            if landmark_px:
                coords[landmark] = landmark_px
        return coords

    def count_squats(self, coords):
        """Count the number of squats performed."""
        if all(point in coords for point in [
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]):
            angle = self.calculate_angle(
                coords[self.mp_pose.PoseLandmark.RIGHT_HIP],
                coords[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                coords[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            )
            
            # Define squat thresholds
            if angle < 150:  # Assuming < 150° means in a squat position
                self.is_squatting = True
            elif self.is_squatting and angle > 180:  # Exiting squat position
                self.squat_count += 1
                self.is_squatting = False

    def draw_count_annotations(self, image):
        """Display the squat count on the image."""
        cv2.putText(image, f'Squats: {self.squat_count}', 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def draw_angle_annotations(self, image, coords):
        """Draw angle annotations for key joints."""
        # (Same as your original implementation...)

    def process_frame(self, frame):
        """Process a single frame and return the annotated image."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = self.pose.process(image)
        
        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            # Get pixel coordinates of landmarks
            height, width, _ = image.shape
            coords = self.get_landmark_coordinates(results.pose_landmarks, width, height)
            
            # Draw pose landmarks
            self.mp_draw.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Count squats
            self.count_squats(coords)
            
            # Draw count annotations
            self.draw_count_annotations(image)

            # Draw angle annotations
            self.draw_angle_annotations(image, coords)
        
        # Calculate and display FPS
        self.new_frame_time = time.time()
        fps = 1/(self.new_frame_time-self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image

    def run(self):
        """Run the body tracker using webcam feed."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Failed to read from webcam")
                    break

                # Process frame
                annotated_frame = self.process_frame(frame)

                # Display the frame
                cv2.imshow('Full Body Tracking', annotated_frame)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()

def main():
    tracker = FullBodyTracker()
    tracker.run()

if __name__ == "__main__":
    main()