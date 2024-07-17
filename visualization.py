import mplfinance as mpf
import pandas as pd
import numpy as np

def plot_predictions(data, scaler, predictions, ticker):
    data.index = pd.to_datetime(data.index)
    
    # Prepare data for plotting
    data['Predicted Close'] = np.nan
    data.iloc[-len(predictions):, data.columns.get_loc('Predicted Close')] = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    plot_data = data[['Open', 'High', 'Low', 'Close', 'Predicted Close']].copy()

    add_plot = [
        mpf.make_addplot(plot_data['Predicted Close'], color='orange', secondary_y=False)
    ]

    fig, ax = mpf.plot(
        plot_data, 
        type='candle', 
        volume=False, 
        style='charles', 
        addplot=add_plot,
        title=f'{ticker} Stock Price Prediction',
        ylabel='Price',
        returnfig=True
    )
    
    ax[0].set_xticklabels([])
    
    return fig
