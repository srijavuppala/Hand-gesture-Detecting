import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.landmark_history = []
        self.max_history = 30
        
    def draw_landmarks(self, frame, landmarks):
        if landmarks is None:
            return frame
            
        # Draw landmarks
        for point in landmarks:
            cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)
            
        # Draw connections between landmarks
        for i in range(len(landmarks)-1):
            cv2.line(frame, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (255, 0, 0), 2)
                    
        return frame
    
    def draw_gesture_text(self, frame, gesture):
        # Add gesture text to frame
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame
    
    def update_trajectory(self, landmarks):
        if landmarks is not None and len(landmarks) > 0:
            # Track the center point of the hand
            center = np.mean(landmarks, axis=0)
            self.landmark_history.append(center)
            
            # Keep only recent history
            if len(self.landmark_history) > self.max_history:
                self.landmark_history.pop(0)
                
    def draw_trajectory(self):
        self.ax.clear()
        if len(self.landmark_history) > 0:
            history = np.array(self.landmark_history)
            self.ax.plot(history[:, 0], history[:, 1])
            self.ax.set_title("Hand Movement Trajectory")
        plt.pause(0.001)
