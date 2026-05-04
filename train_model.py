import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train():
    data_path = r"c:\Users\HP\Desktop\biometric attendance prediction\data\processed_attendance.csv"
    model_dir = r"c:\Users\HP\Desktop\biometric attendance prediction\models"
    os.makedirs(model_dir, exist_ok=True)
    
    print("Loading processed data...")
    df = pd.read_csv(data_path)
    
    # Sort chronologically to prevent data leakage in train/test split
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    
    features = ['day_of_week', 'is_weekend', 'prev_day_attendance', 
                'rolling_7_day_rate', 'cumulative_absences']
    target = 'is_present'
    
    X = df[features]
    y = df[target]
    
    # Split: train on first 3 weeks, test on last week
    # To keep it simple, let's use an 80/20 chronological split
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_path = os.path.join(model_dir, "rf_attendance_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Feature importances
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(importances)

if __name__ == "__main__":
    train()
