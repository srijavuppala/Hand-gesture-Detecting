
import joblib
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2
import statistics as st
import pyttsx3
import time
import json
from datetime import datetime

# Font configuration
FONT = cv2.FONT_HERSHEY_TRIPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White for gesture labels
BG_COLOR = (0, 0, 0)  # Black background
NO_HANDS_COLOR = (0, 0, 255)  # Red for "No hands detected"
RECTANGLE_PADDING = 10
CONFIDENCE_THRESHOLD = 0.7  # For gesture reliability
HISTORY_LENGTH = 10  # Increased for stability

# Gesture-to-message mapping
GESTURE_MESSAGES = {
    "call": "Help needed",
    "one": "Yes",
    "two": "No",
    "two_up": "No"
}

# Initialize text-to-speech
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
last_spoken = 0  # Timestamp to prevent rapid repeat

# Load model and encoder
try:
    model = joblib.load('exports/hand_gesture_final_model.pkl')
    encoder = joblib.load('exports/encoder.pkl')
except FileNotFoundError:
    print("Error: Model or encoder file not found.")
    exit(1)

# Load gesture configuration (optional)
try:
    with open('gesture_config.json', 'r') as f:
        GESTURE_MESSAGES.update(json.load(f))
except FileNotFoundError:
    print("Warning: gesture_config.json not found, using default messages.")

# Initialize gesture log
gesture_log = open('gesture_log.txt', 'a')

def log_gesture(label, confidence):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    gesture_log.write(f"{timestamp} - Gesture: {label}, Confidence: {confidence:.2f}\n")
    gesture_log.flush()

def get_tst_points(landmark, order=False):
    """Converts hand_landmarks.landmark into a series of x,y,z"""
    x_ls, y_ls, z_ls = [], [], []
    if order:
        ordered = []
    for p in landmark:
        p = str(p).split('\n')[:-1]
        x = float(p[0].split(' ')[-1])
        y = float(p[1].split(' ')[-1])
        z = float(p[2].split(' ')[-1])
        if order:
            ordered.extend([x, y, z])
        else:
            x_ls.append(x)
            y_ls.append(y)
            z_ls.append(z)
    return pd.Series(ordered if order else [*x_ls, *y_ls, *z_ls])

def normalize_hand(x: pd.Series, with_label=False) -> np.ndarray:
    """Normalizes hand landmarks: (x_i - x_0)/y_12, (y_i - y_0)/y_12"""
    xs = (x[0::3] - x.iloc[0]).values
    ys = (x[1::3] - x.iloc[1]).values
    y_12 = ys[12]
    if abs(y_12) < 1e-6:
        return np.array([])
    xs = xs / y_12
    ys = ys / y_12
    zs = x[2::3].values
    return np.concatenate([xs, ys, zs])

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

last5 = {}  # Now last10 for stability

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Clear predictions for inactive hands
        active_hands = set(range(len(result.multi_hand_landmarks))) if result.multi_hand_landmarks else set()
        last5 = {k: v for k, v in last5.items() if k in active_hands}

        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get hand position
                hand_x = int(min(landmark.x for landmark in hand_landmarks.landmark) * frame.shape[1])
                hand_y = int(min(landmark.y for landmark in hand_landmarks.landmark) * frame.shape[0]) - 20

                # Process and predict
                tst = get_tst_points(hand_landmarks.landmark, order=True)
                tst = normalize_hand(tst)
                if len(tst) != 63:
                    continue

                # Get prediction and confidence
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(tst.reshape(1, 63))[0]
                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx]
                    pred = encoder.inverse_transform([pred_idx])[0]
                else:
                    pred = encoder.inverse_transform(model.predict(tst.reshape(1, 63)))[0]
                    confidence = 1.0  # Assume full confidence if no predict_proba

                # Update prediction history
                if i not in last5:
                    last5[i] = []
                last5[i].append((pred, confidence))
                if len(last5[i]) > HISTORY_LENGTH:
                    last5[i].pop(0)

                # Get smoothed label
                try:
                    labels = [p[0] for p in last5[i]]
                    confidences = [p[1] for p in last5[i]]
                    label = st.mode(labels)
                    avg_confidence = np.mean([c for l, c in zip(labels, confidences) if l == label])
                except st.StatisticsError:
                    label, avg_confidence = last5[i][-1]

                # Display and act on gesture if confident
                display_text = f"{label} ({avg_confidence:.2f})"
                text_size = cv2.getTextSize(display_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
                text_x = max(0, min(hand_x, frame.shape[1] - text_size[0]))
                text_y = max(30, hand_y)
                cv2.rectangle(frame, (text_x - RECTANGLE_PADDING, text_y - text_size[1] - RECTANGLE_PADDING),
                              (text_x + text_size[0] + RECTANGLE_PADDING, text_y + RECTANGLE_PADDING),
                              BG_COLOR, -1)
                cv2.putText(frame, display_text, (text_x, text_y),
                            FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

                if avg_confidence >= CONFIDENCE_THRESHOLD:
                    log_gesture(label, avg_confidence)
                    # Speak message if not too recent
                    if time.time() - last_spoken > 2:  # 2-second cooldown
                        message = GESTURE_MESSAGES.get(label, label)
                        tts_engine.say(message)
                        tts_engine.runAndWait()
                        last_spoken = time.time()
                else:
                    cv2.putText(frame, "Low confidence", (10, 60),
                                FONT, FONT_SCALE, NO_HANDS_COLOR, FONT_THICKNESS)

        else:
            text_size = cv2.getTextSize("No hands detected", FONT, FONT_SCALE, FONT_THICKNESS)[0]
            cv2.putText(frame, "No hands detected", (10, 30),
                        FONT, FONT_SCALE, NO_HANDS_COLOR, FONT_THICKNESS)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
gesture_log.close()
