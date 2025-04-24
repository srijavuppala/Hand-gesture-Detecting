import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

class GestureRecognizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = self._build_model()
        self.gestures = {
            0: "wave",
            1: "thumbs_up",
            2: "thumbs_down",
            3: "peace",
            4: "open_palm",
            5: "closed_palm",
            6: "point"
        }
        
    def _build_model(self):
        model = Sequential([
            Input(shape=(108,)),  # Changed to match the actual input shape
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(7, activation='softmax')  # 7 gesture classes
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def preprocess_landmarks(self, landmarks):
        if landmarks is None:
            return None
            
        # Flatten landmarks to 1D array
        features = landmarks.flatten()
        
        # Normalize
        features = self.scaler.fit_transform(features.reshape(1, -1))
        
        return features
    
    def predict_gesture(self, landmarks):
        features = self.preprocess_landmarks(landmarks)
        if features is None:
            return "no_gesture"
            
        prediction = self.model.predict(features, verbose=0)  # Added verbose=0 to reduce output
        gesture_idx = np.argmax(prediction)
        
        return self.gestures[gesture_idx]
