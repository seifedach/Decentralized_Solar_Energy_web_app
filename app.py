from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the data
users_data = pd.read_csv('users_dataset.csv')
serie_p = users_data['user1_production']
df = serie_p.values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

time_step = 512

# Load the model
loaded_model = tf.keras.models.load_model('LSTM_user1.keras')

def predict_future(model, data, time_step, forecast_steps):
    temp_input = data[-time_step:].reshape(1, -1).tolist()[0]
    lst_output = []
    
    for _ in range(forecast_steps):
        x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat[0].tolist())
    
    return np.array(lst_output).reshape(-1, 1)

def forecast(days_number, total=0):
    forecast_steps = 24 * days_number
    future_predictions = predict_future(loaded_model, scaled_data, time_step, forecast_steps)
    future_predictions = scaler.inverse_transform(future_predictions)

    if total == 0:
        return future_predictions
    else:
        return np.round(future_predictions.sum(), 3)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast_web():
    days_number = int(request.form['days'])
    action = request.form['action']

    if action == 'Sum':
        total_prediction = forecast(days_number, total=1)
        return jsonify({'result': 'sum', 'value': total_prediction, 'days_number': days_number})
    else:
        future_predictions = forecast(days_number, total=0)
        predictions = future_predictions.flatten().tolist()
        plot_url = create_plot(predictions)
        return jsonify({'result': 'plot', 'value': plot_url, 'days_number': days_number})

def create_plot(predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predicted Values', color='blue')
    plt.title('Forecasted Values')
    plt.xlabel('Time Steps (each step = 1 hour)')
    plt.ylabel('Value (Watt)')
    plt.xticks(ticks=np.arange(0, len(predictions), 12), labels=[f'{i//2}h' for i in range(0, len(predictions), 12)])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

if __name__ == '__main__':
    app.run(debug=True)
