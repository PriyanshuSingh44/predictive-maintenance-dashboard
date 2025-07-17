import os
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, redirect
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Initialize Flask and Dash
app = Flask(__name__)
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/')

# Generate synthetic sensor data
np.random.seed(42)
n_samples = 442
X = np.zeros((n_samples, 3))
X[:, 0] = np.random.uniform(0, 100, n_samples)  # Temperature: 0–100°C
X[:, 1] = np.random.uniform(0, 10, n_samples)   # Vibration: 0–10 mm/s
X[:, 2] = np.random.uniform(0, 1000, n_samples) # Pressure: 0–1000 kPa
y = 5000 - (30 * X[:, 0] + 100 * X[:, 1]**2 + 1.5 * X[:, 2]) + np.random.normal(0, 200, n_samples)
y = np.clip(y, 0, 5000)  

# Preprocess data
X_normalized = X / np.array([100, 10, 1000])  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_normalized)

# Train SVR model
svr_model = SVR(kernel='rbf', C=10.0, epsilon=50, gamma=0.1)
svr_model.fit(X_scaled, y)

# Validate model
test_inputs = np.array([[20, 2, 300], [70, 8, 800], [50, 5, 500]])
test_inputs_normalized = test_inputs / np.array([100, 10, 1000])
test_inputs_scaled = scaler.transform(test_inputs_normalized)
print("Test RULs:", svr_model.predict(test_inputs_scaled))

# Flask API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    inputs = np.array([[data['param1'], data['param2'], data['param3']]])
    inputs_normalized = inputs / np.array([100, 10, 1000])
    inputs_scaled = scaler.transform(inputs_normalized)
    prediction = svr_model.predict(inputs_scaled)[0]
    return jsonify({'rul': prediction})

@app.route('/')
def home():
    return redirect('/dash/')

# Dash layout
dash_app.layout = html.Div([
    html.H1('Predictive Maintenance Dashboard', style={'textAlign': 'center', 'color': '#333'}),
    html.Div([
        html.Label('Temperature (°C, range 0 to 100):'),
        dcc.Input(id='param1', type='number', value=50.0, min=0, max=100, step=1, style={'width': '100%', 'margin': '10px'}),
        html.Label('Vibration (mm/s, range 0 to 10):'),
        dcc.Input(id='param2', type='number', value=5.0, min=0, max=10, step=0.1, style={'width': '100%', 'margin': '10px'}),
        html.Label('Pressure (kPa, range 0 to 1000):'),
        dcc.Input(id='param3', type='number', value=500.0, min=0, max=1000, step=10, style={'width': '100%', 'margin': '10px'}),
        html.Button('Predict RUL', id='predict-btn', style={'margin': '10px', 'padding': '10px'}),
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Div(id='output', style={'textAlign': 'center', 'marginTop': '20px'}),
    dcc.Graph(id='graph', style={'marginTop': '20px'})
])

# Dash callback for predictions and visualization
@dash_app.callback(
    [Output('output', 'children'), Output('graph', 'figure')],
    [Input('predict-btn', 'n_clicks')],
    [dash.dependencies.State('param1', 'value'),
     dash.dependencies.State('param2', 'value'),
     dash.dependencies.State('param3', 'value')]
)
def update_output(n_clicks, param1, param2, param3):
    if n_clicks is None or param1 is None or param2 is None or param3 is None:
        return "Please enter all parameters.", {}

    inputs = np.array([[param1, param2, param3]])
    inputs_normalized = inputs / np.array([100, 10, 1000])
    inputs_scaled = scaler.transform(inputs_normalized)
    rul = svr_model.predict(inputs_scaled)[0]

    fig = go.Figure(data=[
        go.Scatter(
            x=['Temperature (°C)', 'Vibration (mm/s)', 'Pressure (kPa)', 'Predicted RUL (hours)'],
            y=[param1, param2, param3, rul],
            mode='lines+markers',
            marker=dict(size=10, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']),
            line=dict(color='#9467bd', width=2)
        )
    ])
    fig.update_layout(
        title='Input Parameters and Predicted RUL',
        yaxis_title='Value',
        template='plotly_white',
        showlegend=False
    )

    return f"Predicted Remaining Useful Life: {rul:.2f} hours", fig

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)