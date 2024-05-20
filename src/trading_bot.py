import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Data fetching function

def fetch_data(ticker, start_date, end_date):
    """
    Fetch historic market data from Yahoo Finance

    Parameters:
    ticker (str): Stock symbol to fetch
    start_date (str): Start date in format 'YYYY-MM-DD'
    end_date (str): End date in format 'YYYY-MM-DD'
    """

    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# V1 trading strategy

def moving_average_crossover(data, short_window=40, long_window=100):
    """
    Simple moving average crossover strategy.
    
    Parameters:
    data (pandas.DataFrame): Dataframe with 'Close' prices
    short_window (int): Period for the short moving average
    long_window (int): Period for the long moving average
    
    Returns:
    pandas.DataFrame: Dataframe with signals
    """
    signals = data.copy()
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    signals['positions'] = signals['signal'].diff()
    return signals

# Backtesting Framework 

def backtest(signals):
    """
    Plot the prices along with buy and sell signals from the strategy.
    
    Parameters:
    signals (pandas.DataFrame): Dataframe with market data and trading signals
    """
    fig, ax = plt.subplots(figsize=(10,5))
    
    # Plot the closing price, the short and long moving averages
    ax.plot(signals.index, signals['Close'], label='Close')
    ax.plot(signals.index, signals['short_mavg'], label='40-Day MA')
    ax.plot(signals.index, signals['long_mavg'], label='100-Day MA')

    # Plot buy signals
    ax.plot(signals[signals.positions == 1].index, signals.short_mavg[signals.positions == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')

    # Plot sell signals
    ax.plot(signals[signals.positions == -1].index, signals.short_mavg[signals.positions == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

    ax.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    data = fetch_data('AMZN', '2020-01-01', '2020-12-31')
    signals = moving_average_crossover(data)
    backtest(signals)
