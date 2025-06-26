# cybersecurity_threat_detection
📌 Overview:
Modern cyber attacks are fast, adaptive, and often designed to bypass traditional security systems. Static rule-based tools are not enough to detect new and unseen attack patterns in real time. This project addresses this challenge by developing a real-time cybersecurity threat detection system that uses LSTM (Long Short-Term Memory) — a powerful deep learning model — to analyze sequences of network traffic and predict potential threats.

The system includes a Streamlit-based user interface, providing live visual feedback, predictions, and analytics in a browser-based dashboard.

🎯 Project Goals:
Capture and analyze real-time or simulated network traffic.

Use LSTM to detect anomalous patterns that may indicate cybersecurity threats (e.g., intrusions, DoS attacks, brute-force attempts).

Provide an easy-to-use Streamlit UI to monitor, visualize, and interact with the model’s predictions.

Allow for file upload (e.g., CSV logs) or real-time updates.

Display prediction confidence, threat type (optional), and a threat timeline.

🔧 Technologies Used:
Component	Tool / Framework
Language	Python
Deep Learning	LSTM (Keras/TensorFlow)
Data Input	Pre-collected CSV or Live Logs
Visualization	Streamlit, Matplotlib, Plotly
Preprocessing	Pandas, NumPy, Scikit-learn
