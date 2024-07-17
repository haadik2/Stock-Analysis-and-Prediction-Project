import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

from data_preprocessing import fetch_stock_data, fetch_gdp_data, merge_data
from feature_engineering import add_features
from model import train_model
from visualization import plot_predictions

st.title('Stock Market Analysis and Prediction')

# Short explanation of the project
st.markdown("""
### Project Overview

This project aims to analyze historical stock data and relevant economic indicators to predict future stock prices using a Long Short-Term Memory (LSTM) network. The user can input a stock ticker, select a date range, and adjust the number of epochs for training the LSTM model to see how these changes affect the prediction accuracy.
""")

# User inputs
ticker = st.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.date_input('Start Date', value=pd.to_datetime('2010-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2023-12-31'))

# Explanation section
st.markdown("""
## Explanation of LSTM Model Parameters

**Number of Epochs:**
- The number of epochs defines how many times the learning algorithm will work through the entire training dataset.
- Increasing the number of epochs can improve the model's accuracy but also increases the computation time.
- Too many epochs may lead to overfitting, where the model performs well on training data but poorly on unseen data.

In this application, you can adjust the number of epochs for training the LSTM model using the slider below.
""")

# Model Parameters
n_epochs = st.slider('Number of Epochs for Training the LSTM Model', 10, 200, 50)

if st.button('Fetch and Process Data'):
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    gdp_data = fetch_gdp_data()
    merged_data = merge_data(stock_data, gdp_data)
    
    # Ensure there are enough data points before proceeding
    if len(merged_data) < 60:  # Check if there are at least 60 data points
        st.error("Not enough data to create the required features. Please select a larger date range.")
    else:
        X, y, scaler = add_features(merged_data)
        
        st.write('### Merged and Processed Data')
        st.write(merged_data)
        
        # Split data into training and test sets
        split_index = int(0.8 * len(X))
        X_train, y_train = X[:split_index], y[:split_index]
        X_test, y_test = X[split_index:], y[split_index:]
        
        model = train_model(X_train, y_train, n_epochs=n_epochs)
        
        # Make predictions
        predictions = model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        
        st.write('### Model Performance')
        st.write(f'Mean Squared Error: {mse}')
        
        st.write('### Predictions')
        fig = plot_predictions(merged_data, scaler, predictions, ticker)
        st.pyplot(fig)
        
        # Summary of Results
        latest_actual = scaler.inverse_transform(y_test[-1].reshape(-1, 1)).flatten()[0]
        latest_predicted = scaler.inverse_transform(predictions[-1].reshape(-1, 1)).flatten()[0]
        
        summary_message = (
            f"<div style='border:2px solid #000;padding:10px;border-radius:10px;'>"
            f"<h3>Summary</h3>"
            f"The model predicts a <b>{'upward' if latest_predicted > latest_actual else 'downward'} trend</b> for <b>{ticker}</b> "
            f"with a predicted close price of <b>${latest_predicted:.2f}</b>, "
            f"{'higher' if latest_predicted > latest_actual else 'lower'} than the last actual close price of <b>${latest_actual:.2f}</b>. "
            f"<br><br>{'It might be a good time to consider buying this stock.' if latest_predicted > latest_actual else 'You might want to hold off on buying this stock or consider selling if you already own it.'}"
            f"</div>"
        )
        st.markdown(summary_message, unsafe_allow_html=True)
