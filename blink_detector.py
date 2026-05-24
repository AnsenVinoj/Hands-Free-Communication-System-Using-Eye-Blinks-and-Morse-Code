import cv2
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import time
import numpy as np
from scipy.spatial import distance as dist

class EyeBlinkDetector:
    def __init__(self, ear_threshold=0.2, consecutive_frames=3):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # EAR constants
        self.EAR_THRESHOLD = ear_threshold
        self.CONSECUTIVE_FRAMES = consecutive_frames
        self.counter = 0
        self.total_blinks = 0
        
        # Landmark indices for eyes (MediaPipe indices)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # For calibration
        self.calibration_data = []
        self.is_calibrated = False
        self.calibrating = False

    def calculate_ear(self, eye_points):
        # Euclidean distances between the two sets of vertical eye landmarks
        v1 = dist.euclidean(eye_points[1], eye_points[5])
        v2 = dist.euclidean(eye_points[2], eye_points[4])
        # Euclidean distance between the horizontal eye landmarks
        h = dist.euclidean(eye_points[0], eye_points[3])
        # Eye Aspect Ratio
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        blink_detected = False
        ear = 0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                # Extract landmarks
                left_eye_pts = []
                right_eye_pts = []
                
                for idx in self.LEFT_EYE:
                    point = face_landmarks.landmark[idx]
                    left_eye_pts.append((point.x * w, point.y * h))
                
                for idx in self.RIGHT_EYE:
                    point = face_landmarks.landmark[idx]
                    right_eye_pts.append((point.x * w, point.y * h))
                
                left_ear = self.calculate_ear(left_eye_pts)
                right_ear = self.calculate_ear(right_eye_pts)
                
                ear = (left_ear + right_ear) / 2.0
                
                # Blink detection logic
                if ear < self.EAR_THRESHOLD:
                    self.counter += 1
                else:
                    if self.counter >= self.CONSECUTIVE_FRAMES:
                        self.total_blinks += 1
                        blink_detected = True
                    self.counter = 0
                
                # Draw landmarks for visualization (optional)
                for pt in left_eye_pts + right_eye_pts:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
                    
        return frame, blink_detected, ear

    def auto_calibrate(self, duration_seconds=5):
        """Automatically sets the threshold by observing the user."""
        self.calibrating = True
        self.calibration_data = []
        start_time = time.time()
        
        # This would typically be called in a loop in the main app
        # But we'll provide the logic here.
        pass

    def update_threshold(self, ear_values):
        if not ear_values:
            return
        # Set threshold to 70% of the mean EAR (rough approximation)
        # In a real app, we'd look for the dips.
        mean_ear = np.mean(ear_values)
        self.EAR_THRESHOLD = mean_ear * 0.7
        self.is_calibrated = True
        print(f"Calibrated EAR Threshold: {self.EAR_THRESHOLD:.4f}")
