# AI-Failure-Radar
Real-time AI Observability &amp; Drift Detection system for mission-critical defense applications. Simulates sensor noise and adversarial jamming to prevent model failure
# 📡 AI Model Failure Radar (Drift Detection System)

### 🛡️ Overview
This project is an **AI Observability & Safety** tool designed for mission-critical systems (e.g., Defense/UAVs). It simulates a "Black-Box" monitoring system that detects **Data Drift** and **Sensor Noise** before the AI model fails.

Instead of just monitoring accuracy, it uses statistical tests (**Kolmogorov-Smirnov**) to detect adversarial attacks or environmental changes in real-time.

### 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Libraries:** Streamlit, Scikit-learn, SciPy, Matplotlib
* **Methodology:** Statistical Drift Detection, Adversarial Perturbation Simulation.

### ⚡ Key Features
* **Electronic Warfare Simulation:** Simulates sensor noise (Jamming) and data drift.
* **Early Warning System:** Triggers alarms based on statistical distribution shifts (KS-Test).
* **Real-time Visualization:** Compares reference (training) data vs. live (production) data distributions.
