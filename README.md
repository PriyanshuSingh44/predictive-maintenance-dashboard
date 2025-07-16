# 🛠 Predictive Maintenance Dashboard

An interactive dashboard built with **Dash + Flask** to visualize and predict **Remaining Useful Life (RUL)** based on sensor inputs using a trained **SVR model**.

## 📦 Features

- Integrated Flask API (`/predict`) for POST-based predictions
- Dash UI for real-time interactive predictions
- Visualization of Temperature, Vibration, Pressure, and Predicted RUL
- SVR model with synthetic sensor data for demo

## 📊 Tech Stack

- Python
- Dash
- Flask
- scikit-learn (SVR)
- Plotly (for interactive graphs)
- NumPy

## 🚀 How to Run

```bash
# Optional: create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate         # On Windows
source .venv/bin/activate      # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
