import numpy as np
from sklearn.preprocessing import StandardScaler

def add_features(data, window_size=60):
    data['Lagged_Close'] = data['Close'].shift(1)
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    def compute_rsi(data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data['RSI'] = compute_rsi(data)
    data.dropna(inplace=True)
    
    # Prepare data for LSTM
    data_scaled = data[['Close']].copy()
    
    # Check if there are enough samples to scale
    if data_scaled.shape[0] < window_size:
        raise ValueError("Not enough data to create the required features for scaling.")
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_scaled)
    
    X, y = [], []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i, 0])
        y.append(data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler
