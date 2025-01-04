    # Initialize performance metrics
    self.prev_frame_time = 0
    self.new_frame_time = 0

    # Initialize exercise counting variables
    self.bicep_curl_count = 0
    self.is_curling = False

def calculate_angle(self, a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
             np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
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

def count_bicep_curls(self, coords):
    """Count the number of bicep curls performed."""
    if all(point in coords for point in [
        self.mp_pose.PoseLandmark.LEFT_SHOULDER,
        self.mp_pose.PoseLandmark.LEFT_ELBOW,
        self.mp_pose.PoseLandmark.LEFT_WRIST
    ]):
        angle = self.calculate_angle(
            coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            coords[self.mp_pose.PoseLandmark.LEFT_ELBOW],
            coords[self.mp_pose.PoseLandmark.LEFT_WRIST]
        )
        
        # Define curl thresholds
        if angle < 30:  # Fully flexed arm
            self.is_curling = True
        elif self.is_curling and angle > 150:  # Returning to start
            self.bicep_curl_count += 1
            self.is_curling = False

def draw_count_annotations(self, image):
    """Display the bicep curl count on the image."""
    cv2.putText(image, f'Bicep Curls: {self.bicep_curl_count}', 
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
        
        # Count bicep curls
        self.count_bicep_curls(coords)
        
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
    """Run the body tracker using IP webcam feed."""
    # Replace this with your IP webcam URL
    ip_webcam_url = "http://192.168.58.82:8080"  # Replace with your IP Webcam URL
    cap = cv2.VideoCapture(ip_webcam_url)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from IP webcam")
                break

            # Process the frame
            annotated_frame = self.process_frame(frame)

            # Display the frame
            cv2.imshow('Bicep Curl Tracker', annotated_frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()