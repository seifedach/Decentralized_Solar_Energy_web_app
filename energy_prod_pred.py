import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Assuming 'serie' is defined and contains your time series data

users_data = pd.read_csv('users_dataset.csv')
serie_p = users_data['user1_production']
df = serie_p.values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Prepare the data for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])  # Get the next step
    return np.array(X), np.array(Y)

time_step = 512 #8*24  # The number of time steps you want to use for each input

# Predict 96 steps ahead
forecast_steps = 24*4



# Load the model
loaded_model = tf.keras.models.load_model('LSTM_user1.keras')



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import ipywidgets as widgets
from ipywidgets import interact

# Assume predict_future, loaded_model, scaled_data, and scaler are defined


# Function to predict next n steps one step at a time
def predict_future(model, data, time_step, forecast_steps):
    temp_input = data[-time_step:].reshape(1, -1).tolist()[0]
    lst_output = []
    
    for _ in range(forecast_steps):
        x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat[0].tolist())
    
    return np.array(lst_output).reshape(-1, 1)


def forecast_and_plot(days_number, scale=1):
    # Predict forecast_steps ahead
    forecast_steps = 24 * days_number
    future_predictions = predict_future(loaded_model, scaled_data[:-forecast_steps], time_step, forecast_steps)

    # Invert predictions to get the original scale
    future_predictions = scaler.inverse_transform(future_predictions)

    # Extract the ground truth for the forecast horizon
    ground_truth = df[-forecast_steps:]

    # Calculate metrics
    mae = mean_absolute_error(ground_truth, np.maximum(0, future_predictions))
    rmse = np.sqrt(mean_squared_error(ground_truth, np.maximum(0, future_predictions)))
    r2 = r2_score(ground_truth, np.maximum(0, future_predictions))

    #print(f"MAE: {mae}")
    #print(f"RMSE: {rmse}")
    print(f"Confidence: {np.round(r2,3)}")

    # Plot the results
    plt.figure(figsize=(14, 7))
    # Plot real values
    plt.plot(np.arange((scale-1)*len(df)//scale, len(df)), df[(scale-1)*len(df)//scale:], label='Real Data', color='blue')
    # Plot predictions
    plt.plot(np.arange(len(df[:-forecast_steps]), len(df)), np.maximum(0, future_predictions), label='Forecast', color='red')

    plt.title(f'Real Data vs Forecast')
    plt.xlabel('Time')
    plt.ylabel('Watt')
    plt.legend()
    plt.show()

# Define the interactive sliders
days_slider = widgets.IntSlider(
    value=3,
    min=1,
    max=10,
    step=1,
    description='Days:',
    continuous_update=False
)

scale_slider = widgets.IntSlider(
    value=1,
    min=1,
    max=15,
    step=1,
    description='Scale:',
    continuous_update=False
)

# Use interact to create an interactive plot
def interactive_dashboard(days_slider, scale_slider):
    interact(forecast_and_plot, days_number=days_slider, scale=scale_slider)


def forecast(days_number, total=0):
    # Predict forecast_steps ahead
    forecast_steps = 24 * days_number
    future_predictions = predict_future(loaded_model, scaled_data[:-forecast_steps], time_step, forecast_steps)

    # Invert predictions to get the original scale
    future_predictions = scaler.inverse_transform(future_predictions)

    if total==0:
        return future_predictions
    else:
        return np.round(future_predictions.sum(),3)


#def main():

if __name__ == '__main__':
    days_number = int(input("Insert the number of days you want to predict : "))

    print(f"The predicted produced energy in the following {days_number} days: {forecast(days_number, total = 1)} Watt")
    
    #interact(forecast_and_plot, days_number=days_slider, scale=scale_slider)
    #main()
