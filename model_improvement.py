import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import speech_recognition as sr
import pyttsx3
import cv2
import mediapipe as mp
import time
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """Load and preprocess the hand landmarks data"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Remove outliers using z-score
    print("Removing outliers...")
    z_scores = zscore(X)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    X = X[filtered_entries]
    y = y[filtered_entries]
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X_scaled, y_encoded, scaler, label_encoder

def create_model():
    """Create an optimized XGBoost classifier"""
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=7,
        min_child_weight=3,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        n_jobs=-1
    )

def evaluate_model(model, X, y):
    """Evaluate model using train-test split and multiple metrics"""
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    print("\nTest set results:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Additional validation split for robustness check
    print("\nPerforming additional validation...")
    X_remain, X_val, y_remain, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=43)
    model.fit(X_remain, y_remain)
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation set accuracy: {val_accuracy:.4f}")
    
    # Final training on all data
    print("\nTraining final model on all data...")
    model.fit(X, y)
    
    return model

def initialize_speech():
    """Initialize speech recognition and synthesis"""
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return recognizer, engine

def main():
    # Load and preprocess data
    X, y, scaler, label_encoder = load_and_preprocess_data('Data/hand_landmarks_data.csv')
    
    # Create and train model
    print("\nCreating model...")
    model = create_model()
    model = evaluate_model(model, X, y)
    
    # Save the model, scaler, and encoder
    print("\nSaving model, scaler, and encoder...")
    joblib.dump(model, 'exports/hand_gesture_improved_model.pkl')
    joblib.dump(scaler, 'exports/scaler.pkl')
    joblib.dump(label_encoder, 'exports/label_encoder.pkl')
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main()
