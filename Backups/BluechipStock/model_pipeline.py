import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import visualkeras
from PIL import ImageFont
from datetime import datetime, timedelta
import config
import os
import joblib
import database

def _prepare_data_and_features(df):
    """Helper function to prepare data and calculate technical indicators."""
    df.set_index('published_date', inplace=True)
    print("Calculating technical indicators...")
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['SMA_5'] = ta.sma(df['close'], length=5)
    df['SMA_10'] = ta.sma(df['close'], length=10)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    bollinger = ta.bbands(df['close'], length=20, std=2)
    df['BB_upper'] = bollinger['BBU_20_2.0_2.0']
    df['BB_lower'] = bollinger['BBL_20_2.0_2.0']
    df.dropna(inplace=True)
    return df

def _visualize_and_save_plot(df, forecast_unscaled, symbol):
    """
    Helper function to generate a visually continuous and appealing forecast plot.
    """
    print("\n🚀 Visualizing the results...")
    
    last_date = df.index.max()
    last_price = df['close'].iloc[-1]
    
    plot_dates = pd.date_range(start=last_date, periods=config.FORECAST_HORIZON + 1)
    plot_prices = np.insert(forecast_unscaled.flatten(), 0, last_price)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 9))

    # Plot historical data
    plt.plot(df.index[-200:], df['close'][-200:], label='Historical Actual Price', color='dodgerblue', linewidth=2)
    
    plt.plot(plot_dates, plot_prices, color='orangered', linestyle='-', marker='o', label=f'{config.FORECAST_HORIZON}-Day Forecast')
    plt.axvspan(plot_dates[0], plot_dates[-1], facecolor='orange', alpha=0.15, label='Forecast Period')
    
    plt.annotate(f'Last Price: {last_price:.2f}',
                 xy=(last_date, last_price),
                 xytext=(-100, -50),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='black'),
                 fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.7))

    plt.title(f'{symbol} - Stock Price Prediction & Forecast', fontsize=20, fontweight='bold')
    plt.suptitle(f'Forecast Generated on {datetime.now().strftime("%Y-%m-%d")}', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price (NPR)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_filename = config.FORECAST_PLOT_TEMPLATE.format(symbol=symbol)
    plt.savefig(plot_filename)
    print(f"\nForecast plot saved as '{plot_filename}'")
    plt.close()

def _visualize_and_save_evaluation_plot(train_data, valid_data, symbol):
    """Helper function to generate a plot showing model performance on validation data."""
    print(f"\nGenerating model evaluation plot for {symbol}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 8))
    plt.title(f'{symbol} - Model Performance: Actual vs. Predicted', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price (NPR)', fontsize=14)
    plt.plot(train_data['close'], label='Training History')
    plt.plot(valid_data['close'], color='yellow', label='Actual Price (Validation)', linewidth=3)
    plt.plot(valid_data['Predictions'], color='black', label='Predicted Price (Validation)', linewidth=0.5)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True)
    plot_filename = config.EVAL_PLOT_TEMPLATE.format(symbol=symbol)
    plt.savefig(plot_filename)
    print(f"Model evaluation plot saved as '{plot_filename}'")
    plt.close()

def run_prediction_pipeline(symbol):
    """FULL PIPELINE: Fetches data from DB, trains a new model, saves it, and generates a forecast."""
    df = database.get_full_history(symbol)
    df = _prepare_data_and_features(df)
    features = ['open', 'high', 'low', 'close', 'traded_quantity', 'RSI', 'SMA_5', 'SMA_10', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower']
    data = df[features]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit_transform(df[['close']])

    X, y = [], []
    for i in range(config.LOOK_BACK_DAYS, len(scaled_data) - config.FORECAST_HORIZON + 1):
        X.append(scaled_data[i-config.LOOK_BACK_DAYS:i, :])
        y.append(scaled_data[i:i+config.FORECAST_HORIZON, 3])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], len(features)))

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = Sequential([
        LSTM(units=300, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=300, return_sequences=False),
        Dropout(0.2),
        Dense(units=config.FORECAST_HORIZON)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=14, validation_data=(X_test, y_test), verbose=1)

    model_path = config.MODEL_PATH_TEMPLATE.format(symbol=symbol)
    main_scaler_path = config.MAIN_SCALER_PATH_TEMPLATE.format(symbol=symbol)
    close_scaler_path = config.CLOSE_SCALER_PATH_TEMPLATE.format(symbol=symbol)
    
    print(f"\nSaving trained model and scalers for {symbol} to disk...")
    model.save(model_path)
    joblib.dump(scaler, main_scaler_path)
    joblib.dump(close_scaler, close_scaler_path)
    print("Model and scalers saved successfully.")

    last_sequence = scaled_data[-config.LOOK_BACK_DAYS:]
    last_sequence_reshaped = np.reshape(last_sequence, (1, config.LOOK_BACK_DAYS, len(features)))
    forecast = model.predict(last_sequence_reshaped)
    forecast_unscaled = close_scaler.inverse_transform(forecast)
    _visualize_and_save_plot(df, forecast_unscaled, symbol)

    predictions = model.predict(X_test)
    predictions_unscaled = close_scaler.inverse_transform(predictions)
    train_len = len(df) - len(y_test)
    train_data = df.iloc[:train_len]
    valid_data = df.iloc[train_len:].copy()
    valid_data['Predictions'] = predictions_unscaled[:,0]
    _visualize_and_save_evaluation_plot(train_data, valid_data, symbol)

def generate_forecast_from_saved_model(symbol):
    """FORECAST-ONLY: Loads a pre-trained model and generates a new forecast."""
    model_path = config.MODEL_PATH_TEMPLATE.format(symbol=symbol)
    main_scaler_path = config.MAIN_SCALER_PATH_TEMPLATE.format(symbol=symbol)
    close_scaler_path = config.CLOSE_SCALER_PATH_TEMPLATE.format(symbol=symbol)
    
    if not all(os.path.exists(f) for f in [model_path, main_scaler_path, close_scaler_path]):
        print(f"Error: Saved model/scaler for {symbol} not found. Cannot generate forecast.")
        return

    print(f"\nLoading pre-trained model and scalers for {symbol} from disk...")
    model = load_model(model_path)
    scaler = joblib.load(main_scaler_path)
    close_scaler = joblib.load(close_scaler_path)
    print("Model and scalers loaded successfully.")
    
    df = database.get_full_history(symbol)
    df = _prepare_data_and_features(df)
    features = ['open', 'high', 'low', 'close', 'traded_quantity', 'RSI', 'SMA_5', 'SMA_10', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower']
    data = df[features]
    scaled_data = scaler.transform(data)
    
    last_sequence = scaled_data[-config.LOOK_BACK_DAYS:]
    last_sequence_reshaped = np.reshape(last_sequence, (1, config.LOOK_BACK_DAYS, len(features)))
    forecast = model.predict(last_sequence_reshaped)
    forecast_unscaled = close_scaler.inverse_transform(forecast)
    
    _visualize_and_save_plot(df, forecast_unscaled, symbol)