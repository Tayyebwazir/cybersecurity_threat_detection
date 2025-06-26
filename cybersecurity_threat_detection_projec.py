# -*- coding: utf-8 -*-
"""Cybersecurity Threat Detection Project

A comprehensive cybersecurity threat detection system using LSTM neural networks
for real-time network traffic analysis and anomaly detection.
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import socket
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_and_preprocess_data(file_path):

    try:
        # Load data with error handling
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        print(f"Dataset loaded successfully. Shape: {df.shape}")

        # Clean column names
        df.rename(columns=lambda x: x.strip(), inplace=True)

        # Remove duplicates and handle missing values
        df.drop_duplicates(inplace=True)

        # Handle missing values more robustly
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"Handling missing values in columns: {missing_cols}")
            for col in missing_cols:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)

        
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Hour'] = df['Timestamp'].dt.hour
            df['Minute'] = df['Timestamp'].dt.minute
            df['Second'] = df['Timestamp'].dt.second
            df.drop('Timestamp', axis=1, inplace=True)

        return df

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def encode_categorical_features(df):
  
    
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'Label':
            categorical_cols.append(col)

    print(f"Categorical columns found: {categorical_cols}")

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders, categorical_cols

def scale_numerical_features(df, categorical_cols):
  
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Label']]
    print(f"Numerical columns: {numerical_cols}")

    scaler = MinMaxScaler()
    if numerical_cols:
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler, numerical_cols

def create_sequences(data, sequence_length):
  
    if len(data) <= sequence_length:
        print(f"Warning: Data length ({len(data)}) is too small for sequence length ({sequence_length})")
        return np.array([]), np.array([])

    sequences = []
    labels = []

    # Drop the Label column for features
    features = data.drop('Label', axis=1).values
    target = data['Label'].values

    for i in range(len(data) - sequence_length):
        seq = features[i:i+sequence_length]
        label = target[i+sequence_length]
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)

# Main data preprocessing pipeline
def preprocess_pipeline(file_path, sequence_length=10):

   
    # Load data
    df = load_and_preprocess_data(file_path)
    if df is None:
        return None

    # Encode categorical features
    df, label_encoders, categorical_cols = encode_categorical_features(df)

    # Scale numerical features
    df, scaler, numerical_cols = scale_numerical_features(df, categorical_cols)

    # Create sequences
    X, y = create_sequences(df, sequence_length)

    if len(X) == 0:
        print("Error: No sequences created. Check your data and sequence length.")
        return None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, label_encoders, categorical_cols, numerical_cols

def build_lstm_model(input_shape, num_classes):
   
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),

        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),

        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001)
    loss_function = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy'])

    return model

def train_model(X_train, X_test, y_train, y_test, model_save_path='best_model.h5'):
  
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))

    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    # Build model
    model = build_lstm_model(input_shape, num_classes)

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
    ]

    # Train model
    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy*100:.2f}%')

    return model, history

def plot_training_history(history):
    """
    Plot training history

    Args:
        history: Training history from model.fit()
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
  
    # Predictions
    y_pred = model.predict(X_test)

    if len(np.unique(y_test)) > 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)

    return {
        'predictions': y_pred,
        'predicted_classes': y_pred_classes,
        'confusion_matrix': cm
    }


def save_preprocessing_objects(scaler, label_encoders, categorical_cols, numerical_cols, sequence_length, save_path='preprocessing.pkl'):
  
    preprocessing = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'sequence_length': sequence_length,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(save_path, 'wb') as f:
        pickle.dump(preprocessing, f)

    print(f"Preprocessing objects saved to {save_path}")
    print(f"Contains: {list(preprocessing.keys())}")

def load_preprocessing_objects(load_path='preprocessing.pkl'):
 
    try:
        with open(load_path, 'rb') as f:
            preprocessing = pickle.load(f)
        print(f"Preprocessing objects loaded from {load_path}")
        return preprocessing
    except FileNotFoundError:
        print(f"Error: {load_path} not found")
        return None
    except Exception as e:
        print(f"Error loading preprocessing objects: {e}")
        return None

class ThreatDetector:
    """
    Real-time threat detection system
    """
    def __init__(self, model_path, preprocessing_path='preprocessing.pkl'):
     
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")

            # Load preprocessing objects
            preprocessing = load_preprocessing_objects(preprocessing_path)
            if preprocessing:
                self.scaler = preprocessing['scaler']
                self.label_encoders = preprocessing['label_encoders']
                self.sequence_length = preprocessing['sequence_length']
                self.categorical_cols = preprocessing['categorical_cols']
                self.numerical_cols = preprocessing['numerical_cols']
            else:
                print("Warning: Could not load preprocessing objects")

        except Exception as e:
            print(f"Error initializing ThreatDetector: {e}")

    def preprocess_new_data(self, data):
       
        try:
            # Convert to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame([data])

            # Apply label encoders to categorical columns
            for col in self.categorical_cols:
                if col in data.columns:
                    # Handle unseen categories
                    try:
                        data[col] = self.label_encoders[col].transform(data[col].astype(str))
                    except ValueError:
                        # Assign a default value for unseen categories
                        data[col] = 0

            # Scale numerical columns
            if self.numerical_cols:
                data[self.numerical_cols] = self.scaler.transform(data[self.numerical_cols])

            return data.values

        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return None

    def detect_threat(self, network_data):
     
        try:
            # Preprocess data
            processed_data = self.preprocess_new_data(network_data)
            if processed_data is None:
                return {"error": "Failed to preprocess data"}

            # Reshape for LSTM (assuming single sample)
            if len(processed_data.shape) == 1:
                processed_data = processed_data.reshape(1, 1, -1)
            elif len(processed_data.shape) == 2:
                processed_data = processed_data.reshape(processed_data.shape[0], 1, processed_data.shape[1])

            # Predict
            prediction = self.model.predict(processed_data, verbose=0)

            # Determine threat probability
            if prediction.shape[1] > 1:  # Multi-class
                threat_prob = np.max(prediction[0])
                predicted_class = np.argmax(prediction[0])
            else:  # Binary
                threat_prob = prediction[0][0]
                predicted_class = 1 if threat_prob > 0.5 else 0

            # Generate alert message
            if threat_prob > 0.7:
                alert_message = f"ALERT: Potential threat detected with confidence {threat_prob*100:.2f}%"
                self.send_alert(alert_message)
            else:
                alert_message = "No significant threat detected"

            return {
                "threat_probability": float(threat_prob),
                "predicted_class": int(predicted_class),
                "alert_message": alert_message,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": f"Detection failed: {e}"}

    def send_alert(self, alert_message):
    
        try:
            
            print(f"ALERT: {alert_message}")

            # Attempt to send to SIEM (modify connection details as needed)
            siem_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            siem_socket.settimeout(5)  # 5 second timeout

           

            alert = {
                'timestamp': datetime.now().isoformat(),
                'message': alert_message,
                'severity': 'high',
                'source': 'ThreatDetector'
            }

            
            print(f"Alert logged: {json.dumps(alert, indent=2)}")

        except Exception as e:
            print(f"Error sending alert to SIEM: {e}")

def train_anomaly_detector(X_train):
  
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    clf.fit(X_train_flat)

    print("Anomaly detector trained successfully")
    return clf

def combined_detection(threat_detector, anomaly_detector, network_data):
   
    try:
        # LSTM prediction
        lstm_result = threat_detector.detect_threat(network_data)

        # Anomaly detection
        processed_data = threat_detector.preprocess_new_data(network_data)
        if processed_data is not None:
            flat_data = processed_data.reshape(1, -1)
            anomaly_score = anomaly_detector.decision_function(flat_data)[0]
            is_anomaly = anomaly_detector.predict(flat_data)[0] == -1
        else:
            anomaly_score = 0
            is_anomaly = False

        # Combined decision
        lstm_threat = lstm_result.get('threat_probability', 0) > 0.7
        combined_threat = lstm_threat or is_anomaly

        return {
            "lstm_result": lstm_result,
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(is_anomaly),
            "combined_threat": bool(combined_threat),
            "final_message": "ALERT: Threat detected" if combined_threat else "Normal traffic",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {"error": f"Combined detection failed: {e}"}


def main():
   
    print("Cybersecurity Threat Detection System")
    print("=" * 50)

    

    print("Pipeline setup complete. Uncomment the main() code to run the full pipeline.")

if __name__ == "__main__":
    main()

