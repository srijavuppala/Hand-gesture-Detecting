import cv2
import numpy as np

class HandDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.kernel = np.ones((3,3), np.uint8)
        
    def detect_hand(self, frame):
        # Convert to YCrCb color space for better skin detection
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        
        # Skin color range in YCrCb (adjusted for better detection)
        lower = np.array([0, 135, 85], np.uint8)
        upper = np.array([255, 180, 135], np.uint8)
        
        # Create skin mask
        mask = cv2.inRange(ycrcb, lower, upper)
        
        # Apply morphological operations
        mask = cv2.dilate(mask, self.kernel, iterations=3)
        mask = cv2.erode(mask, self.kernel, iterations=2)
        mask = cv2.medianBlur(mask, 5)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (assumed to be the hand)
            max_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(max_contour) > 1000:  # Minimum area threshold
                # Get convex hull and defects
                hull = cv2.convexHull(max_contour, returnPoints=False)
                defects = cv2.convexityDefects(max_contour, hull)
                
                # Extract landmarks
                landmarks = self._extract_landmarks(max_contour, defects)
                
                return landmarks, mask
            
        return None, mask
    
    def _extract_landmarks(self, contour, defects):
        landmarks = []
        
        # Add contour center
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            landmarks.append([cx, cy])
        
        # Get fingertips and valleys using convexity defects
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                landmarks.extend([start, end, far])
        
        # Ensure we have exactly 36 points (pad if necessary)
        while len(landmarks) < 36:
            landmarks.append([0, 0])
        
        return np.array(landmarks[:36])  # Return exactly 36 points (108 values when flattened)
