import cv2
import mediapipe as mp
import numpy as np
import time

class StepTracker:
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
        self.step_count = 0
        self.last_left_knee_y = None
        self.last_right_knee_y = None

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

    def count_steps(self, coords):
        """Count the number of steps performed."""
        if all(point in coords for point in [
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE
        ]):
            left_knee_y = coords[self.mp_pose.PoseLandmark.LEFT_KNEE][1]
            right_knee_y = coords[self.mp_pose.PoseLandmark.RIGHT_KNEE][1]

            # Count step when knees alternate heights
            if self.last_left_knee_y is not None and self.last_right_knee_y is not None:
                if left_knee_y < self.last_left_knee_y and right_knee_y > self.last_right_knee_y:
                    self.step_count += 1

            # Update the last knee positions
            self.last_left_knee_y = left_knee_y
            self.last_right_knee_y = right_knee_y

    def draw_count_annotations(self, image):
        """Display the step count on the image."""
        cv2.putText(image, f'Steps: {self.step_count}', 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
            
            # Count steps
            self.count_steps(coords)
            
            # Draw count annotations
            self.draw_count_annotations(image)
        
        # Calculate and display FPS
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
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
                cv2.imshow('Step Tracker', annotated_frame)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()

def main():
    tracker = StepTracker()
    tracker.run()

if __name__ == "__main__":
    main()
