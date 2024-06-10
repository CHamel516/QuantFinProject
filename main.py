import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas_ta as ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from dash import dcc
from dash import html
import dash
from dash.dependencies import Input, Output
import requests

# Bypass SSL verification
def get_historical_data(tickers, start_date, end_date):
    session = requests.Session()
    session.verify = False
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', session=session)
    return data

# Data Preparation for ML
def prepare_data_for_ml(data, ticker):
    df = data[ticker].copy()
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['Close'].rolling(window=10).std()
    df['momentum'] = df['Close'] - df['Close'].shift(10)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df.dropna(inplace=True)
    
    # Normalization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = df[['returns', 'volatility', 'momentum', 'SMA_20', 'SMA_50', 'MACD']]
    features_scaled = scaler.fit_transform(features)
    df[['returns', 'volatility', 'momentum', 'SMA_20', 'SMA_50', 'MACD']] = features_scaled
    
    return df

# Trading Strategies
def moving_average_crossover(data, ticker, short_window, long_window):
    signals = data[ticker].copy()
    signals['short_mavg'] = signals['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = signals['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'] = 0.0
    signals.iloc[short_window:, signals.columns.get_loc('signal')] = np.where(signals['short_mavg'].iloc[short_window:] > signals['long_mavg'].iloc[short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    return signals

def rsi_strategy(data, ticker, window=14, threshold=30):
    signals = data[ticker].copy()
    signals['RSI'] = ta.rsi(signals['Close'], length=window)
    signals['signal'] = 0
    signals.iloc[window:, signals.columns.get_loc('signal')] = np.where(signals['RSI'].iloc[window:] < threshold, 1, 0)
    signals['positions'] = signals['signal'].diff()
    return signals

def bollinger_bands_strategy(data, ticker, window=20, no_of_std=2):
    signals = data[ticker].copy()
    signals['middle_band'] = signals['Close'].rolling(window=window).mean()
    signals['std'] = signals['Close'].rolling(window=window).std()
    signals['upper_band'] = signals['middle_band'] + (signals['std'] * no_of_std)
    signals['lower_band'] = signals['middle_band'] - (signals['std'] * no_of_std)
    signals['signal'] = 0
    signals.iloc[window:, signals.columns.get_loc('signal')] = np.where(signals['Close'].iloc[window:] < signals['lower_band'].iloc[window:], 1, 0)
    signals['positions'] = signals['signal'].diff()
    return signals

def macd_strategy(data, ticker, short_window=12, long_window=26, signal_window=9):
    signals = data[ticker].copy()
    signals['short_ema'] = signals['Close'].ewm(span=short_window, adjust=False).mean()
    signals['long_ema'] = signals['Close'].ewm(span=long_window, adjust=False).mean()
    signals['macd'] = signals['short_ema'] - signals['long_ema']
    signals['signal_line'] = signals['macd'].ewm(span=signal_window, adjust=False).mean()
    signals['signal'] = 0.0
    signals.iloc[short_window:, signals.columns.get_loc('signal')] = np.where(signals['macd'].iloc[short_window:] > signals['signal_line'].iloc[short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    return signals

# Machine Learning Model
def train_ml_model_with_grid_search(df):
    features = df[['returns', 'volatility', 'momentum', 'SMA_20', 'SMA_50', 'MACD']]
    target = df['Close'].shift(-1).dropna()
    features = features.iloc[:-1]  # Align features and target
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Mean Squared Error: {mse}')
    return best_model

def predict_with_ml_model(model, df):
    features = df[['returns', 'volatility', 'momentum', 'SMA_20', 'SMA_50', 'MACD']]
    df['predicted_close'] = np.nan  # Initialize column with NaNs
    df.loc[features.index, 'predicted_close'] = model.predict(features)
    df['signal'] = np.where(df['predicted_close'] > df['Close'], 1.0, 0.0)
    df['positions'] = df['signal'].diff()
    return df

# Backtesting and Performance Metrics
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

def plot_results(portfolio, signals, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['total'], mode='lines', name='Total Portfolio Value'))

    buy_signals = signals[signals['positions'] == 1.0]
    sell_signals = signals[signals['positions'] == -1.0]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=portfolio['total'][buy_signals.index], mode='markers', name='Buy Signal', marker_symbol='triangle-up', marker=dict(size=10, color='green')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=portfolio['total'][sell_signals.index], mode='markers', name='Sell Signal', marker_symbol='triangle-down', marker=dict(size=10, color='red')))

    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Portfolio Value')
    fig.show()

def create_dashboard(portfolio_mac, portfolio_rsi, portfolio_bb, portfolio_macd, portfolio_ml, signals_mac, signals_rsi, signals_bb, signals_macd, signals_ml):
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Trading Strategy Performance"),
        dcc.Graph(id='portfolio-graph'),
        html.Div([
            html.Label("Select Strategy:"),
            dcc.Dropdown(
                id='strategy-dropdown',
                options=[
                    {'label': 'MAC', 'value': 'MAC'},
                    {'label': 'RSI', 'value': 'RSI'},
                    {'label': 'BB', 'value': 'BB'},
                    {'label': 'MACD', 'value': 'MACD'},
                    {'label': 'ML', 'value': 'ML'}
                ],
                value='ML'
            )
        ])
    ])
    
    @app.callback(
        Output('portfolio-graph', 'figure'),
        [Input('strategy-dropdown', 'value')]
    )
    def update_graph(selected_strategy):
        if selected_strategy == 'MAC':
            plot_data = portfolio_mac
            plot_signals = signals_mac
        elif selected_strategy == 'RSI':
            plot_data = portfolio_rsi
            plot_signals = signals_rsi
        elif selected_strategy == 'BB':
            plot_data = portfolio_bb
            plot_signals = signals_bb
        elif selected_strategy == 'MACD':
            plot_data = portfolio_macd
            plot_signals = signals_macd
        else:
            plot_data = portfolio_ml
            plot_signals = signals_ml
        
        return create_plot(plot_data, plot_signals, f"{selected_strategy} Strategy")
    
    def create_plot(portfolio, signals, title):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['total'], mode='lines', name='Total Portfolio Value'))
        buy_signals = signals[signals['positions'] == 1.0]
        sell_signals = signals[signals['positions'] == -1.0]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=portfolio['total'][buy_signals.index], mode='markers', name='Buy Signal', marker_symbol='triangle-up', marker=dict(size=10, color='green')))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=portfolio['total'][sell_signals.index], mode='markers', name='Sell Signal', marker_symbol='triangle-down', marker=dict(size=10, color='red')))
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Portfolio Value')
        return fig
    
    if __name__ == '__main__':
        app.run_server(debug=True)

# Main script
if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    data = get_historical_data(tickers, '2020-01-01', '2023-01-01')

    # Moving Average Crossover Strategy
    short_window = int(input("Enter the short window for the Moving Average Crossover Strategy: "))
    long_window = int(input("Enter the long window for the Moving Average Crossover Strategy: "))
    signals_mac = moving_average_crossover(data, 'AAPL', short_window, long_window)
    portfolio_mac = backtest_strategy(signals_mac)
    metrics_mac = calculate_performance_metrics(portfolio_mac)
    plot_results(portfolio_mac, signals_mac, "Moving Average Crossover Strategy")
    print("Moving Average Crossover Strategy Metrics:", metrics_mac)

    # RSI Strategy
    rsi_window = int(input("Enter the window for the RSI Strategy: "))
    rsi_threshold = int(input("Enter the RSI threshold for the RSI Strategy: "))
    signals_rsi = rsi_strategy(data, 'AAPL', rsi_window, rsi_threshold)
    portfolio_rsi = backtest_strategy(signals_rsi)
    metrics_rsi = calculate_performance_metrics(portfolio_rsi)
    plot_results(portfolio_rsi, signals_rsi, "RSI Strategy")
    print("RSI Strategy Metrics:", metrics_rsi)

    # Bollinger Bands Strategy
    bb_window = int(input("Enter the window for the Bollinger Bands Strategy: "))
    bb_no_of_std = int(input("Enter the number of standard deviations for the Bollinger Bands Strategy: "))
    signals_bb = bollinger_bands_strategy(data, 'AAPL', bb_window, bb_no_of_std)
    portfolio_bb = backtest_strategy(signals_bb)
    metrics_bb = calculate_performance_metrics(portfolio_bb)
    plot_results(portfolio_bb, signals_bb, "Bollinger Bands Strategy")
    print("Bollinger Bands Strategy Metrics:", metrics_bb)

    # MACD Strategy
    macd_short_window = int(input("Enter the short window for the MACD Strategy: "))
    macd_long_window = int(input("Enter the long window for the MACD Strategy: "))
    macd_signal_window = int(input("Enter the signal window for the MACD Strategy: "))
    signals_macd = macd_strategy(data, 'AAPL', macd_short_window, macd_long_window, macd_signal_window)
    portfolio_macd = backtest_strategy(signals_macd)
    metrics_macd = calculate_performance_metrics(portfolio_macd)
    plot_results(portfolio_macd, signals_macd, "MACD Strategy")
    print("MACD Strategy Metrics:", metrics_macd)

    # Machine Learning Strategy
    df = prepare_data_for_ml(data, 'AAPL')
    model = train_ml_model_with_grid_search(df)
    signals_ml = predict_with_ml_model(model, df)
    portfolio_ml = backtest_strategy(signals_ml)
    metrics_ml = calculate_performance_metrics(portfolio_ml)
    plot_results(portfolio_ml, signals_ml, "Machine Learning Strategy")
    print("Machine Learning Strategy Metrics:", metrics_ml)

    # Performance Comparison
    comparison_df = pd.DataFrame({
        'Strategy': ['MAC', 'RSI', 'BB', 'MACD', 'ML'],
        'Sharpe Ratio': [metrics_mac['Sharpe Ratio'], metrics_rsi['Sharpe Ratio'], metrics_bb['Sharpe Ratio'], metrics_macd['Sharpe Ratio'], metrics_ml['Sharpe Ratio']],
        'CAGR': [metrics_mac['CAGR'], metrics_rsi['CAGR'], metrics_bb['CAGR'], metrics_macd['CAGR'], metrics_ml['CAGR']],
        'Max Drawdown': [metrics_mac['Max Drawdown'], metrics_rsi['Max Drawdown'], metrics_bb['Max Drawdown'], metrics_macd['Max Drawdown'], metrics_ml['Max Drawdown']],
        'Sortino Ratio': [metrics_mac['Sortino Ratio'], metrics_rsi['Sortino Ratio'], metrics_bb['Sortino Ratio'], metrics_macd['Sortino Ratio'], metrics_ml['Sortino Ratio']],
        'Volatility': [metrics_mac['Volatility'], metrics_rsi['Volatility'], metrics_bb['Volatility'], metrics_macd['Volatility'], metrics_ml['Volatility']]
    })

    print("\nPerformance Comparison:\n", comparison_df)

    # Create interactive dashboard
    create_dashboard(portfolio_mac, portfolio_rsi, portfolio_bb, portfolio_macd, portfolio_ml, signals_mac, signals_rsi, signals_bb, signals_macd, signals_ml)
