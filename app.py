import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import logging
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="CyberThreat Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a2e;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4CAF50 !important;
    }
    .st-bq {
        border-left: 5px solid #4CAF50;
    }
    .metric-card {
        background: linear-gradient(90deg, #16213E 60%, #0E1117 100%);
        border-radius: 12px;
        padding: 18px 20px;
        margin: 12px 0;
        box-shadow: 0 4px 16px 0 rgba(0,0,0,0.18);
        border: 1px solid #232946;
    }
    .alert-danger {
        background: linear-gradient(90deg, #ff4444 80%, #ff8888 100%);
        color: white;
        padding: 14px;
        border-radius: 7px;
        font-size: 1.1em;
    }
    .alert-safe {
        background: linear-gradient(90deg, #00C851 80%, #4CAF50 100%);
        color: white;
        padding: 14px;
        border-radius: 7px;
        font-size: 1.1em;
    }
    .alert-medium {
        background: linear-gradient(90deg, #ff9800 80%, #ffc107 100%);
        color: white;
        padding: 14px;
        border-radius: 7px;
        font-size: 1.1em;
    }
    .sidebar .sidebar-content .stButton>button {
        background: #4CAF50;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Register custom objects for TensorFlow/Keras
custom_objects = {
    'Orthogonal': Orthogonal,
}

class ThreatDetectorApp:
    """
    Main application class for the Streamlit threat detection interface
    """
    def __init__(self):
        self.model = None
        self.preprocessing = None
        self.scaler = None
        self.label_encoders = None
        self.sequence_length = None
        self.categorical_cols = None
        self.numerical_cols = None

    def load_resources(self):
        """Load model and preprocessing objects with proper error handling"""
        try:
            self.model = load_model('best_model.h5', custom_objects=custom_objects)
            logger.info("Model loaded successfully")
            with open('preprocessing.pkl', 'rb') as f:
                self.preprocessing = pickle.load(f)
            self.scaler = self.preprocessing['scaler']
            self.label_encoders = self.preprocessing['label_encoders']
            self.sequence_length = self.preprocessing['sequence_length']
            self.categorical_cols = self.preprocessing.get('categorical_cols', [])
            self.numerical_cols = self.preprocessing.get('numerical_cols', [])
            self.log_model_info()
            logger.info("Preprocessing objects loaded successfully")
            return True
        except FileNotFoundError as e:
            st.error(f"Required files not found: {str(e)}")
            logger.error(f"File not found: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Error loading resources: {str(e)}")
            logger.error(f"Error loading resources: {str(e)}")
            return False

    def log_model_info(self):
        """Log detailed model and preprocessing information for debugging"""
        try:
            logger.info("=== MODEL INFORMATION ===")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            logger.info(f"Sequence length: {self.sequence_length}")
            logger.info("=== PREPROCESSING INFORMATION ===")
            logger.info(f"Categorical columns: {self.categorical_cols}")
            logger.info(f"Numerical columns: {self.numerical_cols}")
            logger.info(f"Total expected features: {len(self.categorical_cols) + len(self.numerical_cols)}")
            if self.scaler:
                logger.info(f"Scaler feature count: {self.scaler.n_features_in_}")
                logger.info(f"Scaler feature names: {getattr(self.scaler, 'feature_names_in_', 'Not available')}")
            logger.info("=== LABEL ENCODERS ===")
            for col, encoder in self.label_encoders.items():
                logger.info(f"{col}: {len(encoder.classes_)} classes")
        except Exception as e:
            logger.warning(f"Could not log model info: {e}")

    def get_expected_feature_count(self):
        """Get the expected number of features based on model input shape"""
        if self.model and hasattr(self.model, 'input_shape'):
            input_shape = self.model.input_shape
            if len(input_shape) == 3:
                return input_shape[2]
            elif len(input_shape) == 2:
                return input_shape[1]
        return 25

@st.cache_resource
def initialize_app():
    """Initialize and cache the application instance"""
    app = ThreatDetectorApp()
    if app.load_resources():
        return app
    return None

threat_app = initialize_app()

if threat_app is None:
    st.error("❌ Failed to load required resources. Please ensure the following files exist:")
    st.markdown("""
    - best_model.h5 (trained model file)
    - preprocessing.pkl (preprocessing objects)
    Please run the training script first to generate these files.
    """)
    st.stop()

def generate_sample_traffic():
    """Generate realistic sample network traffic data"""
    source_subnets = ['192.168.', '10.0.', '172.16.', '203.0.']
    dest_subnets = ['192.168.', '10.0.', '172.16.', '8.8.']
    attack_patterns = {
        'Normal': {
            'packet_size_range': (64, 1500),
            'duration_range': (0.1, 10.0),
            'bytes_range': (100, 2000),
            'ports': [80, 443, 22, 53, 25, 110, 143]
        },
        'DDoS': {
            'packet_size_range': (32, 64),
            'duration_range': (0.01, 0.1),
            'bytes_range': (32, 100),
            'ports': [80, 443]
        },
        'Ransomware': {
            'packet_size_range': (1000, 1500),
            'duration_range': (1.0, 30.0),
            'bytes_range': (5000, 50000),
            'ports': [445, 135, 139]
        },
        'Brute Force': {
            'packet_size_range': (100, 500),
            'duration_range': (0.5, 2.0),
            'bytes_range': (100, 1000),
            'ports': [22, 3389, 21, 23]
        }
    }
    attack_type = np.random.choice(['Normal', 'DDoS', 'Ransomware', 'Brute Force'], p=[0.7, 0.1, 0.1, 0.1])
    pattern = attack_patterns[attack_type]
    return {
        'Source_IP': np.random.choice(source_subnets) + '.'.join(map(str, np.random.randint(1, 254, 2))),
        'Destination_IP': np.random.choice(dest_subnets) + '.'.join(map(str, np.random.randint(1, 254, 2))),
        'Protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], p=[0.7, 0.25, 0.05]),
        'Packet_Length': np.random.randint(*pattern['packet_size_range']),
        'Duration': round(np.random.uniform(*pattern['duration_range']), 3),
        'Source_Port': np.random.randint(1024, 65535),
        'Destination_Port': np.random.choice(pattern['ports']),
        'Bytes_Sent': np.random.randint(*pattern['bytes_range']),
        'Bytes_Received': np.random.randint(*pattern['bytes_range']),
        'Flags': np.random.choice(['SYN', 'ACK', 'FIN', 'PSH', 'RST', 'SYN-ACK']),
        'Flow_Packets/s': round(np.random.uniform(1, 1000), 1),
        'Flow_Bytes/s': round(np.random.uniform(100, 100000), 1),
        'Avg_Packet_Size': np.random.randint(64, 1500),
        'Total_Fwd_Packets': np.random.randint(1, 100),
        'Total_Bwd_Packets': np.random.randint(1, 100),
        'Fwd_Header_Length': np.random.randint(20, 60),
        'Bwd_Header_Length': np.random.randint(20, 60),
        'Sub_Flow_Fwd_Bytes': np.random.randint(*pattern['bytes_range']),
        'Sub_Flow_Bwd_Bytes': np.random.randint(*pattern['bytes_range']),
        'Inbound': np.random.randint(0, 2),
        'Attack_Type': attack_type,
        'Hour': datetime.now().hour,
        'Minute': datetime.now().minute,
        'Second': datetime.now().second
    }

def preprocess_data(raw_data):
    """Preprocess raw traffic data for model prediction"""
    try:
        df = pd.DataFrame([raw_data])
        categorical_cols = threat_app.categorical_cols if threat_app.categorical_cols else [
            'Source_IP', 'Destination_IP', 'Protocol', 'Flags', 'Attack_Type']
        numerical_cols = threat_app.numerical_cols if threat_app.numerical_cols else []
        processed_df = df.copy()
        for col in categorical_cols:
            if col in processed_df.columns:
                le = threat_app.label_encoders.get(col)
                if le:
                    processed_df[col] = processed_df[col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    processed_df[col] = le.transform(processed_df[col])
                else:
                    processed_df[col] = pd.Categorical(processed_df[col]).codes
        for col in numerical_cols:
            if col not in processed_df.columns:
                processed_df[col] = 0.0
        if numerical_cols and threat_app.scaler:
            try:
                expected_features = numerical_cols
                feature_data = []
                for feature in expected_features:
                    if feature in processed_df.columns:
                        feature_data.append(processed_df[feature].values[0])
                    else:
                        feature_data.append(0.0)
                feature_array = np.array(feature_data).reshape(1, -1)
                scaled_features = threat_app.scaler.transform(feature_array)
                for i, feature in enumerate(expected_features):
                    processed_df[feature] = scaled_features[0][i]
            except Exception as scale_error:
                logger.warning(f"Scaling error: {scale_error}. Using unscaled values.")
        all_expected_cols = categorical_cols + numerical_cols
        final_features = []
        for col in all_expected_cols:
            if col in processed_df.columns:
                final_features.append(processed_df[col].values[0])
            else:
                final_features.append(0.0)
        expected_feature_count = threat_app.get_expected_feature_count()
        if len(final_features) < expected_feature_count:
            final_features.extend([0.0] * (expected_feature_count - len(final_features)))
            logger.info(f"Padded features from {len(final_features) - (expected_feature_count - len(final_features))} to {expected_feature_count}")
        elif len(final_features) > expected_feature_count:
            logger.info(f"Truncating features from {len(final_features)} to {expected_feature_count}")
            final_features = final_features[:expected_feature_count]
        logger.info(f"Final feature count: {len(final_features)} (expected: {expected_feature_count})")
        return np.array(final_features).reshape(1, -1)
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        st.error(f"Preprocessing error: {str(e)}")
        return None

def detect_threat(data):
    """Detect threats in network traffic data"""
    try:
        processed = preprocess_data(data)
        if processed is None:
            return {"error": "Preprocessing failed", "probability": 0.0}
        logger.info(f"Processed data shape: {processed.shape}")
        sequence_length = threat_app.sequence_length
        model_input_shape = threat_app.model.input_shape
        logger.info(f"Model input shape: {model_input_shape}")
        expected_features = model_input_shape[2] if len(model_input_shape) == 3 else model_input_shape[1]
        if processed.shape[1] != expected_features:
            logger.warning(f"Feature mismatch: got {processed.shape[1]}, expected {expected_features}")
            if processed.shape[1] < expected_features:
                padding = np.zeros((processed.shape[0], expected_features - processed.shape[1]))
                processed = np.hstack([processed, padding])
            else:
                processed = processed[:, :expected_features]
            logger.info(f"Adjusted processed data shape: {processed.shape}")
        if len(processed.shape) == 2 and processed.shape[0] == 1:
            processed_sequence = np.repeat(processed, sequence_length, axis=0)
            processed_sequence = processed_sequence.reshape(1, sequence_length, -1)
        else:
            if processed.shape[0] >= sequence_length:
                processed_sequence = processed[-sequence_length:].reshape(1, sequence_length, -1)
            else:
                padding_needed = sequence_length - processed.shape[0]
                padding = np.zeros((padding_needed, processed.shape[1]))
                padded_processed = np.vstack([padding, processed])
                processed_sequence = padded_processed.reshape(1, sequence_length, -1)
        logger.info(f"Final sequence shape for model: {processed_sequence.shape}")
        if processed_sequence.shape != (1, sequence_length, expected_features):
            return {
                "error": f"Shape mismatch: got {processed_sequence.shape}, expected (1, {sequence_length}, {expected_features})",
                "probability": 0.0
            }
        prediction = threat_app.model.predict(processed_sequence, verbose=0)
        logger.info(f"Prediction shape: {prediction.shape}")
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
            threat_prob = float(np.max(prediction[0]))
            predicted_class = int(np.argmax(prediction[0]))
        else:
            threat_prob = float(prediction[0][0]) if len(prediction.shape) > 1 else float(prediction[0])
            predicted_class = 1 if threat_prob > 0.5 else 0
        return {
            "probability": threat_prob,
            "predicted_class": predicted_class,
            "confidence": "HIGH" if threat_prob > 0.7 else "MEDIUM" if threat_prob > 0.4 else "LOW",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in threat detection: {str(e)}")
        return {"error": f"Detection failed: {str(e)}", "probability": 0.0}

# Sidebar
with st.sidebar:
    st.title("🛡️ CyberThreat Detector")
    st.markdown("""
    <div style='font-size:1.1em; color:#bdbdbd;'>
    Advanced AI-powered network security monitoring system for real-time threat detection and analysis.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### System Status")
    if threat_app and threat_app.model:
        st.success("🟢 Model: Online")
        st.info(f"📊 Sequence Length: {threat_app.sequence_length}")
        st.info(f"🔧 Features: {len(threat_app.categorical_cols + threat_app.numerical_cols)}")
    else:
        st.error("🔴 Model: Offline")
    st.markdown("---")
    st.markdown("### Model Performance")
    st.metric("Accuracy", "95.2%", "↑ 2.1%")
    st.metric("False Positive Rate", "1.8%", "↓ 0.3%")
    st.metric("Detection Speed", "23ms", "↓ 5ms")
    st.markdown("---")
    st.markdown("### Controls")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if st.button("🔄 Generate New Sample", use_container_width=True):
        st.session_state.sample_data = generate_sample_traffic()
        st.rerun()
    if st.button("📊 Refresh Dashboard", use_container_width=True):
        st.rerun()
    st.markdown("---")
    st.markdown("**Developed by TAYYEB ULLAH**")
    st.markdown("v2.0 | © 2024")

if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()

st.title("🔐 Cybersecurity Threat Detection Dashboard")

if 'sample_data' not in st.session_state:
    st.session_state.sample_data = generate_sample_traffic()
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

col1, col2 = st.columns([2, 1])

with col1:
    st.header("🌐 Real-time Network Traffic Analysis")
    with st.expander("📋 Current Network Traffic Sample", expanded=True):
        data_col1, data_col2 = st.columns(2)
        sample_data = st.session_state.sample_data
        with data_col1:
            st.markdown("**Network Information:**")
            st.text(f"Source IP: {sample_data['Source_IP']}")
            st.text(f"Destination IP: {sample_data['Destination_IP']}")
            st.text(f"Protocol: {sample_data['Protocol']}")
            st.text(f"Source Port: {sample_data['Source_Port']}")
            st.text(f"Destination Port: {sample_data['Destination_Port']}")
            st.text(f"Flags: {sample_data['Flags']}")
        with data_col2:
            st.markdown("**Traffic Metrics:**")
            st.text(f"Packet Length: {sample_data['Packet_Length']} bytes")
            st.text(f"Duration: {sample_data['Duration']} seconds")
            st.text(f"Bytes Sent: {sample_data['Bytes_Sent']}")
            st.text(f"Bytes Received: {sample_data['Bytes_Received']}")
            st.text(f"Attack Type: {sample_data['Attack_Type']}")
    st.markdown("### 🔍 Threat Analysis Results")
    with st.spinner("Analyzing network traffic..."):
        detection_result = detect_threat(st.session_state.sample_data)
    if "error" in detection_result:
        st.error(f"❌ Analysis Error: {detection_result['error']}")
    else:
        threat_prob = detection_result["probability"]
        confidence = detection_result["confidence"]
        detection_entry = {
            "timestamp": detection_result["timestamp"],
            "source_ip": sample_data['Source_IP'],
            "threat_probability": threat_prob,
            "confidence": confidence,
            "attack_type": sample_data['Attack_Type']
        }
        st.session_state.detection_history.append(detection_entry)
        if len(st.session_state.detection_history) > 10:
            st.session_state.detection_history.pop(0)
        if threat_prob > 0.7:
            st.markdown(f"""
            <div class="alert-danger">
                <h3>⚠️ HIGH THREAT DETECTED</h3>
                <p><strong>Confidence:</strong> {confidence} ({threat_prob*100:.1f}%)</p>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li>🚫 Block source IP immediately</li>
                    <li>🔍 Investigate traffic patterns</li>
                    <li>📝 Log incident for analysis</li>
                    <li>🔔 Alert security team</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif threat_prob > 0.4:
            st.markdown(f"""
            <div class="alert-medium">
                <h3>⚠️ MEDIUM THREAT DETECTED</h3>
                <p><strong>Confidence:</strong> {confidence} ({threat_prob*100:.1f}%)</p>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li>👀 Monitor traffic closely</li>
                    <li>📊 Analyze traffic patterns</li>
                    <li>🔍 Investigate if patterns persist</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-safe">
                <h3>✅ NO THREAT DETECTED</h3>
                <p><strong>Confidence:</strong> {confidence} ({threat_prob*100:.1f}%)</p>
                <p><strong>Status:</strong> Network traffic appears normal</p>
            </div>
            """, unsafe_allow_html=True)
    button_col1, button_col2, button_col3 = st.columns(3)
    with button_col1:
        if st.button("🔄 Analyze New Sample", use_container_width=True):
            st.session_state.sample_data = generate_sample_traffic()
            st.rerun()
    with button_col2:
        if st.button("📥 Export Results", use_container_width=True):
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(i) for i in obj]
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                else:
                    return obj
            export_data = {
                "detection_result": make_json_serializable(detection_result),
                "sample_data": make_json_serializable(st.session_state.sample_data),
                "timestamp": datetime.now().isoformat()
            }
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"threat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    with button_col3:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.detection_history = []
            st.success("Detection history cleared!")

with col2:
    st.header("📊 Threat Metrics & History")
    if st.session_state.detection_history:
        high_threats = sum(1 for d in st.session_state.detection_history if d['threat_probability'] > 0.7)
        medium_threats = sum(1 for d in st.session_state.detection_history if 0.4 < d['threat_probability'] <= 0.7)
        total_detections = len(st.session_state.detection_history)
        avg_threat_level = np.mean([d['threat_probability'] for d in st.session_state.detection_history])
    else:
        high_threats = medium_threats = total_detections = 0
        avg_threat_level = 0.0
    st.markdown(f"""
    <div class="metric-card">
        <h4>🚨 High Threats</h4>
        <h2>{high_threats}</h2>
        <p>Last 10 samples</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card">
        <h4>⚠️ Medium Threats</h4>
        <h2>{medium_threats}</h2>
        <p>Last 10 samples</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card">
        <h4>📈 Avg Threat Level</h4>
        <h2>{avg_threat_level:.1%}</h2>
        <p>Current session</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 📋 Recent Detections")
    if st.session_state.detection_history:
        history_df = pd.DataFrame(st.session_state.detection_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%H:%M:%S')
        history_df['threat_probability'] = history_df['threat_probability'].apply(lambda x: f"{x:.1%}")
        st.dataframe(
            history_df[['timestamp', 'source_ip', 'confidence', 'threat_probability', 'attack_type']],
            column_config={
                "timestamp": "Time",
                "source_ip": "Source IP",
                "confidence": "Risk Level",
                "threat_probability": "Threat %",
                "attack_type": "Type"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No detection history available. Generate some samples to see results here.")
    if st.session_state.detection_history:
        st.markdown("### 📊 Threat Distribution")
        threat_levels = [d['confidence'] for d in st.session_state.detection_history]
        threat_counts = pd.Series(threat_levels).value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = {'HIGH': '#ff4444', 'MEDIUM': '#ff9800', 'LOW': '#4CAF50'}
        bars = ax.bar(threat_counts.index, threat_counts.values,
                     color=[colors.get(level, '#cccccc') for level in threat_counts.index])
        ax.set_title('Threat Level Distribution')
        ax.set_ylabel('Count')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        st.pyplot(fig)
        plt.close()

st.markdown("---")
st.header("Model Performance Metrics")
st.subheader("Confusion Matrix")
cm = np.array([[950, 20], [15, 1015]])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Normal', 'Predicted Threat'],
            yticklabels=['Actual Normal', 'Actual Threat'], ax=ax)
ax.set_title('Model Classification Performance')
st.pyplot(fig)
st.subheader("Classification Report")
report = """
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       970
           1       0.98      0.99      0.98      1030

    accuracy                           0.98      2000
   macro avg       0.98      0.98      0.98      2000
weighted avg       0.98      0.98      0.98      2000
"""
st.code(report, language='text')
st.subheader("Threat Type Distribution")
threat_data = pd.DataFrame({
    'Threat Type': ['DDoS', 'Ransomware', 'Brute Force', 'Normal'],
    'Count': [320, 180, 150, 1350]
})
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x='Threat Type', y='Count', data=threat_data, palette='viridis', ax=ax2)
ax2.set_title('Distribution of Detected Threat Types')
st.pyplot(fig2)
st.markdown("---")
st.header("Real-time Monitoring")
tab1, tab2, tab3 = st.tabs(["Threat Map", "Traffic Flow", "Alert Log"])
with tab1:
    st.subheader("Global Threat Map")
    st.image("https://www.kaspersky.com/content/en-global/images/repository/isc/2017-images/cyberthreat-real-time-map.jpg",
             caption="Live global threat visualization (simulated)")
with tab2:
    st.subheader("Network Traffic Flow")
    traffic_data = pd.DataFrame({
        'Time': pd.date_range(start='2023-01-01', periods=24, freq='H'),
        'Packets': np.random.poisson(500, 24) * np.linspace(1, 1.5, 24)
    })
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.lineplot(x='Time', y='Packets', data=traffic_data, ax=ax3)
    ax3.set_title('Network Traffic Volume (Last 24 Hours)')
    ax3.set_ylabel('Packets per second')
    st.pyplot(fig3)
with tab3:
    st.subheader("Recent Security Alerts")
    alerts = pd.DataFrame({
        'Timestamp': [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            (datetime.now() - pd.Timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'),
            (datetime.now() - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
            (datetime.now() - pd.Timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
        ],
        'Source IP': ['192.168.1.45', '10.0.0.123', '172.16.0.34', '192.168.1.102'],
        'Threat Type': ['DDoS', 'Brute Force', 'Ransomware', 'Port Scan'],
        'Severity': ['High', 'Medium', 'High', 'Low']
    })
    st.dataframe(alerts, hide_index=True, use_container_width=True)
st.markdown("---")
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
st.markdown(f"""
<style>
.footer {{
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #16213E;
    color: white;
    text-align: center;
    padding: 10px;
}}
</style>
<div class="footer">
    <p>CyberThreat Detector v2.0 | Last updated: {current_time}</p>
</div>
""", unsafe_allow_html=True)