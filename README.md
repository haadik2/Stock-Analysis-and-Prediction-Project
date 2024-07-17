Stock Market Analysis and Prediction
Overview
This project aims to analyze historical stock data and relevant economic indicators to predict future stock prices using advanced machine learning techniques. The application is built with a Streamlit front-end for interactive user input and visualization, and uses TensorFlow, Keras, and LightGBM for the back-end machine learning models.

Features
Interactive User Interface: Allows users to input stock tickers, select date ranges, and adjust model parameters.
Real-Time Data Fetching: Fetches up-to-date stock information from Yahoo Finance.
Comprehensive Analysis: Combines historical stock data with economic indicators to enhance prediction accuracy.
Advanced Machine Learning Models: Utilizes LSTM networks and LightGBM for robust time-series forecasting.
Data Visualization: Displays interactive charts and performance metrics.

Installation
Prerequisites
Python 3.7+
Virtual environment (recommended)

Setup
Clone the repository:

git clone https://github.com/yourusername/stock-market-prediction.git
cd stock-market-prediction

Create and activate a virtual environment:

python -m venv myenv
myenv\Scripts\activate   # For Windows
source myenv/bin/activate  # For macOS/Linux

Install required packages:

pip install -r requirements.txt


Run the Streamlit application:

streamlit run app/main.py

Interact with the application:

Enter a stock ticker (e.g., AAPL).
Select a date range.
Adjust the number of epochs for training the LSTM model.
Click "Fetch and Process Data" to generate predictions.

Project Structure

.
├── app
│   ├── data_preprocessing.py   # Functions for fetching and merging data
│   ├── feature_engineering.py  # Functions for creating and scaling features
│   ├── main.py                 # Main Streamlit application
│   ├── model.py                # Machine learning model definitions
│   └── visualization.py        # Functions for plotting predictions
├── myenv                       # Virtual environment directory
├── requirements.txt            # Required Python packages
└── README.md                   # Project README file

Explanation of Key Components

Data Preprocessing:

Utilizes yfinance to fetch historical stock data.
Integrates GDP data from public datasets for economic indicators.
Merges and cleans data to ensure consistency and accuracy.

Feature Engineering:

Generates relevant features such as moving averages (MA50, MA200), lagged close prices, and Relative Strength Index (RSI).
Scales data using StandardScaler to prepare it for machine learning models.

Model Development:

Implements initial models using Random Forest and XGBoost for baseline performance.
Develops a Long Short-Term Memory (LSTM) network using TensorFlow and Keras for more accurate time-series forecasting.
Trains models with various parameters and evaluates their performance using Mean Squared Error (MSE).

Visualization and Analysis:

Creates interactive visualizations of stock price predictions using mplfinance and Matplotlib.
Displays model performance metrics and prediction summaries within the Streamlit application.
Provides a summary and recommendations based on model predictions.
