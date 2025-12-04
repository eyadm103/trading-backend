# Import all necessary libraries for data retrieval, analysis, and trading.
import pandas as pd
import numpy as np
import os
import joblib
import alpaca_trade_api as tradeapi
import yfinance as yf
from ta.volatility import average_true_range, BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from datetime import datetime, timedelta
import time
import pytz
import warnings
import sys

# Settings to ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. New logging function ---
# A helper function to log all actions with a timestamp.
def log_action(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    # Add encoding='utf-8' to handle all characters correctly
    with open('trading_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"{full_message}\n")
    print(full_message)

# --- 2. Configuration and API Keys ---
# Define global settings and API keys. Keep this information private.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "core", "models", "final_model.pkl")
MODEL_FILENAME = 'final_model.pkl'
GLOBAL_STOCK_TICKER = 'NVDA' # The stock ticker to trade
RISK_PER_TRADE_PCT = 0.01
# Alpaca Paper Trading API keys
API_KEY = "PK54FD3865IGM62HEQ6F"
API_SECRET = "9osd0CTg8XT5XeMcRDDR5SrH7NzN8QsS1ybJyI2C"
BASE_URL = "https://paper-api.alpaca.markets"

# 🚨 New parameters for forced trading mode
# Set to True to guarantee at least one trade per day.
# Set to False for normal, cautious trading.
FORCED_TRADE_MODE = True
# A global variable to track the last trade date
last_trade_date = None

# The 24 feature names used to train the model. This list is crucial.
MODEL_FEATURE_NAMES = [
    'macd_line', 'macd_signal', 'macd_diff', 'bb_hband', 'bb_lband', 'bb_wband',
    'sma_20', 'rsi_14', 'atr_14', 'daily_range_pct', 'volatility_5_std',
    'volatility_20_std', 'year', 'month', 'dayofweek',
    'lag_1_close_return', 'lag_5_close_return', 'lag_20_close_return',
    'gap_open_close_prev_pct',
    'volume', 'volume_change_pct', 'high_low_spread', 'open_close_spread', 'momentum_10d'
]

# --- 3. Load the pre-trained model ---
# Ensure the model file exists and can be loaded.
try:
    log_action("⏳ Loading the model...")
    model_path = os.path.join(BASE_DIR, "core", "models", MODEL_FILENAME)
    final_model = joblib.load(model_path)
    log_action("✅ Model loaded successfully.")
except FileNotFoundError:
    log_action(f"❌ Error: Model file not found at the following path: {model_path}")
    log_action("Please make sure you have trained and saved the model correctly.")
    sys.exit()  # Stop the script if the model is not found


# --- 4. Set up API connection ---
# Connect to the Alpaca Paper Trading API to execute trades.
try:
    log_action("⏳ Connecting to Alpaca API...")
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    account = api.get_account()
    log_action(f"✅ Successfully connected to Alpaca API.")
    log_action(f"📊 Account status: {account.status}")
    log_action(f"💰 Current portfolio value: ${float(account.equity):.2f}")
except Exception as e:
    log_action(f"❌ Error connecting to Alpaca API: {e}")
    sys.exit()

# --- 5. Function to check market open hours ---
# This function ensures the bot only operates during regular market hours.
def is_market_open():
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.now(ny_tz)

    # Check if it's a weekday (Monday=0, Sunday=6)
    if now.weekday() >= 5:
        return False
    
    # Define market open and close times
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Return True if current time is within market hours
    return market_open <= now < market_close

# --- 6. Function to get data and prepare features (with Yahoo Finance) ---
def get_latest_features_with_retry(max_retries=5, delay_seconds=10):
    """
    Fetches stock data from Yahoo Finance and calculates the required features.
    Includes a retry mechanism to handle temporary connection issues.
    """
    for attempt in range(max_retries):
        try:
            # Getting data for the last 5 days to ensure enough data for all indicators
            df = yf.download(tickers=GLOBAL_STOCK_TICKER, period='5d', interval='1m')
            
            # Check if DataFrame is empty or too small
            if df.empty or len(df) < 50:
                 log_action("  - Received empty or insufficient data from Yahoo Finance. Retrying...")
                 continue
            # Reset index to make date/time a column, then rename it explicitly
            df.reset_index(inplace=True)
            
            # Handling column names, including multi-level ones, and ensuring they are lowercase
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Rename the first column, which is now the datetime index, to 'datetime'
            df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
            
            # Set the new 'datetime' column as the index for subsequent calculations
            df.set_index('datetime', inplace=True)
            
            # Calculating the 22 features for the model
            macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd_line'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_hband'] = bollinger.bollinger_hband()
            df['bb_lband'] = bollinger.bollinger_lband()
            df['bb_wband'] = bollinger.bollinger_wband()
            df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
            df['atr_14'] = average_true_range(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['daily_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
            df['volatility_5_std'] = df['close'].rolling(window=5).std()
            df['volatility_20_std'] = df['close'].rolling(window=20).std()
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['dayofweek'] = df.index.dayofweek
            for lag in [1, 5, 20]:
                df[f'lag_{lag}_close_return'] = df['close'].pct_change(lag) * 100
            df['gap_open_close_prev_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
            
            # 🚨 Modified: Using a neutral, non-zero value for volume as requested.
            df['volume'] = 1.0
            df['volume_change_pct'] = 1.0
            
            df['high_low_spread'] = df['high'] - df['low']
            df['open_close_spread'] = df['close'] - df['open']
            df['momentum_10d'] = df['close'].diff(10)
            
            # Drop rows with NaN values resulting from indicator calculations
            df.dropna(inplace=True)
            if df.empty:
                return None, None
            
            # Return the last row (latest data) and the full DataFrame
            return df.iloc[-1], df
            
        except Exception as e:
            log_action(f"❌ Error getting data from Yahoo Finance (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                log_action(f"  - Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                log_action("  - Max retries reached. Giving up for this cycle.")
                return None, None
                
# --- 7. Main execution loop ---
# This loop runs the trading logic at a fixed interval.
log_action("--- Automated trading system started ---")
if FORCED_TRADE_MODE:
    log_action("🚨 WARNING: Forced Trade Mode is ACTIVE. A trade will be initiated if none occur naturally.")
log_action("The code will now check the market every 180 seconds during market hours.")
log_action("To stop the code, press the stop button (red square) in Jupyter or Ctrl+C in the terminal.")
log_action("Starting the main execution loop.")

# 🚨 Initialize last trade date to a past date to allow a trade on the first day
last_trade_date = (datetime.now() - timedelta(days=1)).date()

try:
    while True:
        log_action("⚙️ Checking the market...")
        now = datetime.now(pytz.timezone('America/New_York'))
        
        if not is_market_open():
            log_action("  - Market is currently closed. Will re-check at the next scheduled time.")
            time.sleep(60)
            continue
            
        # 🚨 Reset last_trade_date at the start of a new day
        if now.date() != last_trade_date:
            last_trade_date = now.date()
        
        try:
            # Use the new function with retry mechanism to get data
            latest_data, full_df = get_latest_features_with_retry()
            if latest_data is None or latest_data.isnull().values.any():
                log_action("  - Data quality check failed. Skipping this cycle due to missing values.")
                time.sleep(180)
                continue
            
            current_price = latest_data['close']
            
            # Log all 24 features for full transparency
            log_action(f"  - Current price: ${current_price:.2f}")
            log_action("  - Features for analysis:")
            for feature in MODEL_FEATURE_NAMES:
                value = latest_data[feature]
                log_action(f"    - {feature}: {value:.2f}" if isinstance(value, (int, float)) else f"    - {feature}: {value}")
            
            # Prepare the latest data for the model prediction
            features_for_prediction_df = pd.DataFrame([latest_data])
            # Ensure the features are in the exact order the model expects
            features_for_prediction_df = features_for_prediction_df[MODEL_FEATURE_NAMES]
            prediction = final_model.predict(features_for_prediction_df)[0]
            
            # Get the current open positions from Alpaca
            positions = api.list_positions()
            
            # --- Buy Logic (only if there are no open positions) ---
            if not positions:
                # 🚨 Modified: Check for normal buy signal OR forced trade signal
                is_normal_buy_signal = (prediction == 1) and (current_price > full_df['sma_50'].iloc[-1])
                is_forced_buy_signal = FORCED_TRADE_MODE and (now.date() == last_trade_date) and (now.time() >= now.replace(hour=15, minute=50, second=0).time())
                
                if is_normal_buy_signal or is_forced_buy_signal:
                    log_action(f"✅ Prediction is BUY. Checking risk management...")
                    
                    # Calculate risk per trade based on account equity and ATR
                    sl_multiplier = 2.5
                    stop_loss_amount = latest_data['atr_14'] * sl_multiplier
                    account = api.get_account()
                    risk_per_trade_amount = float(account.equity) * RISK_PER_TRADE_PCT
                    
                    if stop_loss_amount > 0:
                        shares_to_buy = int(risk_per_trade_amount / stop_loss_amount)
                    else:
                        shares_to_buy = 0
                    
                    if shares_to_buy > 0:
                        api.submit_order(
                            symbol=GLOBAL_STOCK_TICKER,
                            qty=shares_to_buy,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        if is_normal_buy_signal:
                            log_action(f"📈 BUY signal! Executing order for {shares_to_buy} shares at ${current_price:.2f}.")
                            log_action(f"  - (Logic: Model recommended a buy and price is above SMA_50)")
                        else:
                            log_action(f"📈 FORCED BUY signal! Executing order for {shares_to_buy} shares at ${current_price:.2f}.")
                            log_action(f"  - (Logic: Forced trade initiated at end of day)")
                    else:
                        log_action("  - Capital or defined risk is too low to make a purchase.")
                else:
                    log_action(f"  - Model does not recommend buying at the moment. Prediction: {prediction}")
                    if FORCED_TRADE_MODE and now.date() == last_trade_date:
                        log_action("  - Forced trade not initiated yet, waiting for 3:50 PM or a natural buy signal.")
            
            # --- Sell Logic (only if there is an open position) ---
            else:
                position = positions[0]
                entry_price = float(position.avg_entry_price)
                
                # Define dynamic Stop-Loss and Take-Profit based on ATR
                sl_multiplier = 2.5
                tp_multiplier = 5.0
                atr_for_exit = latest_data['atr_14']
                stop_loss_price = entry_price - (atr_for_exit * sl_multiplier)
                take_profit_price = entry_price + (atr_for_exit * tp_multiplier)
                
                current_profit = (current_price - entry_price) / entry_price
                
                log_action(f"  - Trade status: Current P/L: {current_profit * 100:.2f}%. SL: ${stop_loss_price:.2f}, TP: ${take_profit_price:.2f}")
                
                # Check all exit conditions
                is_forced_sell = False
                # 🚨 Modified: A forced sell is triggered if a forced buy was made, and a small profit is available.
                if FORCED_TRADE_MODE and current_profit > 0.001 and (now.time() >= now.replace(hour=15, minute=55, second=0).time()):
                    is_forced_sell = True
                    log_action(f"📈 FORCED SELL signal! Closing trade (forced exit) at ${current_price:.2f}. Return: {current_profit * 100:.2f}%")
                    api.close_position(GLOBAL_STOCK_TICKER)
                elif current_price <= stop_loss_price:
                    log_action(f"📉 SELL signal! Closing trade (dynamic stop loss) at ${current_price:.2f}. Return: {current_profit * 100:.2f}%")
                    api.close_position(GLOBAL_STOCK_TICKER)
                elif current_price >= take_profit_price:
                    log_action(f"📈 SELL signal! Closing trade (dynamic take profit) at ${current_price:.2f}. Return: {current_profit * 100:.2f}%")
                    api.close_position(GLOBAL_STOCK_TICKER)
                elif latest_data['macd_line'] < latest_data['macd_signal']:
                    log_action(f"📉 SELL signal! Closing trade (negative MACD crossover) at ${current_price:.2f}. Return: {current_profit * 100:.2f}%")
                    api.close_position(GLOBAL_STOCK_TICKER)
                elif latest_data['rsi_14'] > 70:
                    log_action(f"📉 SELL signal! Closing trade (RSI > 70) at ${current_price:.2f}. Return: {current_profit * 100:.2f}%")
                    api.close_position(GLOBAL_STOCK_TICKER)
        except Exception as e:
            log_action(f"❌ A critical error occurred in the trading loop: {e}")
        finally:
            # Sleep for exactly 180 seconds, regardless of whether an error occurred
            time.sleep(180)
except KeyboardInterrupt:
    log_action("--- Automated trading system stopped ---")