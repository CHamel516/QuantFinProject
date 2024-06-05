import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas_ta as ta

# Function to get historical data
def get_historical_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    return data

# Moving Average Crossover Strategy
def moving_average_crossover(data, ticker, short_window, long_window):
    signals = data[ticker].copy()
    signals['short_mavg'] = signals['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = signals['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    return signals

# RSI Strategy using pandas_ta
def rsi_strategy(data, ticker, window=14, threshold=30):
    signals = data[ticker].copy()
    signals['RSI'] = ta.rsi(signals['Close'], length=window)
    signals['signal'] = 0
    signals['signal'][window:] = np.where(signals['RSI'][window:] < threshold, 1, 0)
    signals['positions'] = signals['signal'].diff()
    return signals

# Bollinger Bands Strategy
def bollinger_bands_strategy(data, ticker, window=20, no_of_std=2):
    signals = data[ticker].copy()
    signals['middle_band'] = signals['Close'].rolling(window=window).mean()
    signals['std'] = signals['Close'].rolling(window=window).std()
    signals['upper_band'] = signals['middle_band'] + (signals['std'] * no_of_std)
    signals['lower_band'] = signals['middle_band'] - (signals['std'] * no_of_std)
    signals['signal'] = 0
    signals['signal'][window:] = np.where(signals['Close'][window:] < signals['lower_band'][window:], 1, 0)
    signals['positions'] = signals['signal'].diff()
    return signals

# Backtesting function
def backtest_strategy(signals, initial_capital=100000.0, transaction_cost=0.001, stop_loss=0.05):
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['asset'] = 100 * signals['signal']  # 100 shares
    portfolio = positions.multiply(signals['Close'], axis=0)
    pos_diff = positions.diff()

    portfolio['holdings'] = (positions.multiply(signals['Close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(signals['Close'], axis=0)).sum(axis=1).cumsum()
    portfolio['cash'] -= (abs(pos_diff) * signals['Close'] * transaction_cost).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    # Apply stop-loss
    peak = portfolio['total'].cummax()
    drawdown = (portfolio['total'] / peak) - 1
    stop_loss_trigger = drawdown < -stop_loss
    portfolio['total'][stop_loss_trigger] = portfolio['cash'][stop_loss_trigger]

    return portfolio

# Performance metrics function
def calculate_performance_metrics(portfolio):
    returns = portfolio['returns'].dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    end_value = portfolio['total'].iloc[-1]
    start_value = portfolio['total'].iloc[0]
    cagr = (end_value / start_value) ** (1 / (len(returns) / 252)) - 1

    rolling_max = portfolio['total'].cummax()
    drawdown = portfolio['total'] / rolling_max - 1.0
    max_drawdown = drawdown.min()

    downside_returns = returns[returns < 0]
    sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)

    volatility = returns.std() * np.sqrt(252)

    return {
        'Sharpe Ratio': sharpe_ratio,
        'CAGR': cagr,
        'Max Drawdown': max_drawdown,
        'Sortino Ratio': sortino_ratio,
        'Volatility': volatility
    }

# Visualization function
def plot_results(portfolio, signals, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['total'], mode='lines', name='Total Portfolio Value'))

    buy_signals = signals[signals['positions'] == 1.0]
    sell_signals = signals[signals['positions'] == -1.0]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=portfolio['total'][buy_signals.index], mode='markers', name='Buy Signal', marker_symbol='triangle-up', marker=dict(size=10, color='green')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=portfolio['total'][sell_signals.index], mode='markers', name='Sell Signal', marker_symbol='triangle-down', marker=dict(size=10, color='red')))

    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Portfolio Value')
    fig.show()

# MACD Strategy
def macd_strategy(data, ticker, short_window=12, long_window=26, signal_window=9):
    signals = data[ticker].copy()
    signals['short_ema'] = signals['Close'].ewm(span=short_window, adjust=False).mean()
    signals['long_ema'] = signals['Close'].ewm(span=long_window, adjust=False).mean()
    signals['macd'] = signals['short_ema'] - signals['long_ema']
    signals['signal_line'] = signals['macd'].ewm(span=signal_window, adjust=False).mean()
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(signals['macd'][short_window:] > signals['signal_line'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    return signals

# Main script addition for MACD Strategy
if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    data = get_historical_data(tickers, '2020-01-01', '2023-01-01')

    # MACD Strategy
    macd_short_window = int(input("Enter the short window for the MACD Strategy: "))
    macd_long_window = int(input("Enter the long window for the MACD Strategy: "))
    macd_signal_window = int(input("Enter the signal window for the MACD Strategy: "))
    signals_macd = macd_strategy(data, 'AAPL', macd_short_window, macd_long_window, macd_signal_window)
    portfolio_macd = backtest_strategy(signals_macd)
    metrics_macd = calculate_performance_metrics(portfolio_macd)
    plot_results(portfolio_macd, signals_macd, "MACD Strategy")
    print("MACD Strategy Metrics:", metrics_macd)
    
    # Add to Performance Comparison
    comparison_df = pd.DataFrame({
        'Strategy': ['MAC', 'RSI', 'BB', 'MACD'],
        'Sharpe Ratio': [metrics_mac['Sharpe Ratio'], metrics_rsi['Sharpe Ratio'], metrics_bb['Sharpe Ratio'], metrics_macd['Sharpe Ratio']],
        'CAGR': [metrics_mac['CAGR'], metrics_rsi['CAGR'], metrics_bb['CAGR'], metrics_macd['CAGR']],
        'Max Drawdown': [metrics_mac['Max Drawdown'], metrics_rsi['Max Drawdown'], metrics_bb['Max Drawdown'], metrics_macd['Max Drawdown']],
        'Sortino Ratio': [metrics_mac['Sortino Ratio'], metrics_rsi['Sortino Ratio'], metrics_bb['Sortino Ratio'], metrics_macd['Sortino Ratio']],
        'Volatility': [metrics_mac['Volatility'], metrics_rsi['Volatility'], metrics_bb['Volatility'], metrics_macd['Volatility']]
    })

    print("\nPerformance Comparison:\n", comparison_df)
