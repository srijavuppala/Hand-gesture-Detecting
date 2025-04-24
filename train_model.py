import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import os

# Create exports directory if it doesn't exist
if not os.path.exists('exports'):
    os.makedirs('exports')

# Load and preprocess data
print("Loading data...")
df = pd.read_csv('Data/hand_landmarks_data.csv')
X = df.drop('label', axis=1)
y = df['label']

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save label encoder
joblib.dump(label_encoder, 'exports/label_encoder.pkl')

# Feature scaling
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'exports/scaler.pkl')

# Feature selection
print("Selecting best features...")
selector = SelectKBest(score_func=f_classif, k=63)  # Use all features for now
X_selected = selector.fit_transform(X_scaled, y)
joblib.dump(selector, 'exports/feature_selector.pkl')

# Create and train Random Forest with optimized parameters
print("Training model...")
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    class_weight='balanced',
    random_state=42
)

# Train the model
model.fit(X_selected, y)

# Save the model
print("Saving model...")
joblib.dump(model, 'exports/hand_gesture_model.pkl')
print("Model saved as 'exports/hand_gesture_model.pkl'")
