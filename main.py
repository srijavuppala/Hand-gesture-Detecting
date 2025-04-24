import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import os
import joblib
import pyttsx3
import statistics as st

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Load the trained model and preprocessing components
model = joblib.load('exports/hand_gesture_model.pkl')
label_encoder = joblib.load('exports/label_encoder.pkl')
scaler = joblib.load('exports/scaler.pkl')
feature_selector = joblib.load('exports/feature_selector.pkl')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White
BG_COLOR = (0, 120, 0)  # Dark Green
RECTANGLE_PADDING = 10
CONFIDENCE_THRESHOLD = 0.7  # Only show predictions above this confidence
HISTORY_LENGTH = 10  # Increased for stability

# Gesture-to-message mapping
GESTURE_MESSAGES = {
    "call": "Help needed",
    "one": "Yes",
    "two": "No",
    "two_up": "No",
    "four": "Four fingers",
    "five": "Five fingers"
}

# Initialize text-to-speech
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
last_spoken = 0  # Timestamp to prevent rapid repeat


def extract_hand_features(landmarks):
    """Extract features from hand landmarks in the same format as training data"""
    features = []
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    return features

def preprocess_features(features):
    """Apply the same preprocessing steps as during training"""
    # Convert to numpy array and reshape
    features_array = np.array(features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    
    # Select best features
    features_selected = feature_selector.transform(features_scaled)
    
    return features_selected

def normalize_hand_features(features):
    """Additional normalization to improve distinction between similar gestures"""
    features_array = np.array(features)
    xs = features_array[0::3] - features_array[0]
    ys = features_array[1::3] - features_array[1]
    y_12 = ys[12] if len(ys) > 12 else 1.0
    if abs(y_12) < 1e-6:
        y_12 = 1.0
    xs = xs / y_12
    ys = ys / y_12
    zs = features_array[2::3]
    return np.concatenate([xs, ys, zs])

def log_gesture(gesture, confidence):
    """Log detected gestures with timestamp"""
    log_dir = "gesture_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - Gesture: {gesture}, Confidence: {confidence:.2f}\n"
    
    with open(os.path.join(log_dir, "gesture_log.txt"), "a") as f:
        f.write(log_entry)

def main():
    prev_time = 0
    last_gestures = {}
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            continue
            
        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(frame_rgb)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Draw FPS on image
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), FONT, 1, TEXT_COLOR, 2)
        
        # Clear predictions for inactive hands
        active_hands = set(range(len(results.multi_hand_landmarks))) if results.multi_hand_landmarks else set()
        last_gestures = {k: v for k, v in last_gestures.items() if k in active_hands}
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get hand position for text placement
                hand_x = int(min(hand_landmarks.landmark[0].x * frame.shape[1], frame.shape[1] - 200))
                hand_y = int(hand_landmarks.landmark[0].y * frame.shape[0])
                
                # Extract and preprocess features
                features = extract_hand_features(hand_landmarks.landmark)
                features_normalized = normalize_hand_features(features)
                features_processed = preprocess_features(features_normalized)
                
                # Predict gesture
                prediction = model.predict_proba(features_processed)[0]
                gesture_idx = np.argmax(prediction)
                confidence = prediction[gesture_idx]
                
                # Update prediction history
                if idx not in last_gestures:
                    last_gestures[idx] = []
                last_gestures[idx].append((gesture_idx, confidence))
                if len(last_gestures[idx]) > HISTORY_LENGTH:
                    last_gestures[idx].pop(0)
                
                # Get smoothed label
                try:
                    labels = [p[0] for p in last_gestures[idx]]
                    confidences = [p[1] for p in last_gestures[idx]]
                    label_idx = st.mode(labels)
                    avg_confidence = np.mean([c for l, c in zip(labels, confidences) if l == label_idx])
                except st.StatisticsError:
                    label_idx, avg_confidence = last_gestures[idx][-1]
                
                # Only show prediction if confidence is above threshold
                if avg_confidence >= CONFIDENCE_THRESHOLD:
                    gesture = label_encoder.inverse_transform([label_idx])[0]
                    
                    # Display text with background
                    display_text = f"Hand {idx+1}: {gesture.upper()} ({avg_confidence:.2f})"
                    text_size = cv2.getTextSize(display_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
                    text_x = max(10, min(hand_x, frame.shape[1] - text_size[0] - 20))
                    text_y = max(text_size[1] + 40, min(hand_y, frame.shape[0] - 20))
                    
                    # Draw background rectangle
                    cv2.rectangle(frame, 
                                (text_x - RECTANGLE_PADDING, text_y - text_size[1] - RECTANGLE_PADDING),
                                (text_x + text_size[0] + RECTANGLE_PADDING, text_y + RECTANGLE_PADDING),
                                BG_COLOR, -1)
                    
                    # Draw text
                    cv2.putText(frame, display_text, (text_x, text_y),
                              FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
                    
                    # Log the gesture
                    log_gesture(gesture, avg_confidence)
                    
                    # Speak message if not too recent
                    if time.time() - last_spoken > 2:  # 2-second cooldown
                        message = GESTURE_MESSAGES.get(gesture, gesture)
                        tts_engine.say(message)
                        tts_engine.runAndWait()
                        last_spoken = time.time()
        
        # Show the image
        cv2.imshow("Hand Gesture Recognition", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
