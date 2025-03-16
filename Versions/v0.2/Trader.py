import time
import datetime
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import pytz
import pickle
import math
import talib  # For advanced technical indicators
import psutil
#import warnings
#warnings.filterwarnings("ignore")
# -----------------------
# KONFIGURATION
# -----------------------
TWELVE_DATA_API_KEY = "6438f9035dbf40598b7e588cc74e1e8f"  # Få en gratis nøgle fra https://twelvedata.com

# Minimum confidence for at sende Discord besked (i procent)
MIN_CONFIDENCE_THRESHOLD = 40  

# Add at the top of your script (after imports, before function definitions)
# Initialize global variables
ml_model = None
scaler = None
trade_history = None
active_timeframes_data = {}

# Discord webhooks for de enkelte valuta-par (brug de ønskede webhooks her)
DISCORD_WEBHOOKS_BY_PAIR = {
    "EURUSD": "https://discord.com/api/webhooks/1349494950689247374/1VUmOI93KRYPZ5QujQ-_eEHGb9_IP2JxSUZ_TwxJfJqRNOsIFq9aG6WP-7eTc0zAEIjT",
    "GBPUSD": "https://discord.com/api/webhooks/1349494939448250491/qKgiAidpnbY3sV9ASu2l983NfF3u2W5A5QFVA1y81Gqbv5hRgtY7eJo_oqEz0cVC5A2R",
    "USDJPY": "https://discord.com/api/webhooks/1349494962022121585/fvQ4m1PalnTj0oyiQXNgjiILl9CmHEMPWxvPaZ5X0Uz4FAccqHkwvD1rSnVqXqcvQT_e",
    "AUDUSD": "https://discord.com/api/webhooks/1349495238678282250/H4ALQvbptpmmUaeAPY23SQFMcIgDLqpvqJtVScjf7mkArWNwEnpY0dvBykWJSLRkfAXX",
    "BTCUSD": "https://discord.com/api/webhooks/1349861884421083308/1y3eQuDvSEwIzma4tosyuCyvUNzTOq3724or4oofUXyNaLzEPBB18zZdI7yTo-UD8xUk",
    "ETHUSD": "https://discord.com/api/webhooks/1349861904989945969/Y9aiK0aqlUXhBxy-5pCegE-9WkMvbamLRTpE5XbQWAUzY_1aJx_UMbOsODoAu41jNiT0",
    "XRPUSD": "https://discord.com/api/webhooks/1349862064767893564/ndiDglK9yqDbwUV8Va3uSwvgbjQzZM3Cjqcp9iXJBxR-_l2GoUwobRh736htCJCd3qbd"
}

# Webhook til samlede opdateringer
UPDATE_WEBHOOK_URL = "https://discord.com/api/webhooks/1349729748607176744/4RRBw-yDlJB8hZw_5EGdqkUWxgfZObM9f1E4F3jwxX9H0y5-NGuVUUwEnLr4xtzRHkhe"

# Webhook til machine learning opdateringer
ML_WEBHOOK_URL = "https://discord.com/api/webhooks/1350111866902548581/y2kNJvaqlMgBntUf5Rf9WxOhqRc4_bOyBKZZbGWDzrkWhkPr48A-5BipiNoBff8ddoR3"

# Liste af forex-par (brug fx "EURUSD")
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTCUSD", "ETHUSD", "XRPUSD"]

# Risikobeløb og gearing (basis, kan justeres dynamisk)
BASE_RISK_AMOUNT = 100
MAX_RISK_AMOUNT = 300
BASE_LEVERAGE = 30

# Timeframes for multi-timeframe analysis
TIMEFRAMES = ["15min", "1h", "4h", "1day"]

# Timeout for et tradeforslag (5 timer)
TRADE_TIMEOUT = 8 * 60 * 60

# Filer til logning af trade-historik og win rate
TRADE_LOG_FILE = "trade_history.csv"
WIN_RATE_FILE = "win_rate.txt"

# ML-relaterede filer
ML_MODEL_FILE = "ml_model.pkl"
ML_SCALER_FILE = "ml_scaler.pkl"
ML_FEATURE_IMPORTANCE_FILE = "ml_feature_importance.pkl"
ML_LEARNING_LOG_FILE = "ml_learning_log.csv"
ML_MARKET_CONDITIONS_FILE = "ml_market_conditions.csv"

# Twelve Data gratis API rate limit (ca. 8 kald/minut)
API_CALL_DELAY = 15  # sekunder mellem API-kald

def analyze_and_optimize_strategy(historical_data_dir="historical_data", sample_size=1000):
    """
    Analyzes existing strategy performance across different market conditions
    and suggests optimized parameters.
    """
    print("Starting strategy optimization...")
    results = {}
    
    # Track performance across different conditions
    condition_results = {
        'trending_up': {'trades': 0, 'wins': 0},
        'trending_down': {'trades': 0, 'wins': 0},
        'ranging': {'trades': 0, 'wins': 0},
        'high_volatility': {'trades': 0, 'wins': 0},
        'low_volatility': {'trades': 0, 'wins': 0},
        'overbought': {'trades': 0, 'wins': 0},
        'oversold': {'trades': 0, 'wins': 0}
    }
    
    # Parameter variations to test
    parameter_variations = {
        'rsi_buy_threshold': list(range(20, 41, 5)),         # Default is 30
        'rsi_sell_threshold': list(range(60, 81, 5)),        # Default is 70
        'min_strength_threshold': list(range(20, 51, 10)),   # Default is 30
        'atr_multiplier_trending': [1.5, 2.0, 2.5, 3.0],     # Default range is 2-3
        'atr_multiplier_ranging': [1.0, 1.25, 1.5, 1.75],    # Default range is 1.5-2
        'min_reward_ratio': [1.5, 2.0, 2.5, 3.0]             # Default is 1.5
    }
    
    parameter_performance = {}
    
    # Get list of all forex pairs
    all_pairs = [d for d in os.listdir(historical_data_dir) 
                if os.path.isdir(os.path.join(historical_data_dir, d))]
    
    for pair in all_pairs:
        pair_dir = os.path.join(historical_data_dir, pair)
        
        # Get the 15min data
        tf_file = os.path.join(pair_dir, "15min.csv")
        if not os.path.exists(tf_file):
            continue
            
        print(f"Analyzing strategy for {pair}...")
        df = pd.read_csv(tf_file, parse_dates=["time"])
        df = calculate_indicators_forward_only(df)
        
        # Test different parameter combinations
        best_win_rate = 0
        best_params = {}
        
        # Test original parameters first
        original_results = test_strategy_parameters(df, pair, {})
        print(f"Original strategy performance for {pair}: {original_results['win_rate']:.2f}% win rate")
        
        # Only test a sample of the data for speed
        if len(df) > sample_size:
            indices = np.random.choice(len(df) - 100, sample_size - 100, replace=False)
            indices = sorted(indices) + list(range(len(df) - 100, len(df)))
            test_df = df.iloc[indices].copy()
        else:
            test_df = df
            
        # Test each parameter variation
        for param_name, values in parameter_variations.items():
            for value in values:
                params = {param_name: value}
                
                results = test_strategy_parameters(test_df, pair, params)
                
                # Track parameter performance
                param_key = f"{param_name}_{value}"
                if param_key not in parameter_performance:
                    parameter_performance[param_key] = []
                parameter_performance[param_key].append(results['win_rate'])
                
                if results['win_rate'] > best_win_rate:
                    best_win_rate = results['win_rate']
                    best_params = params
                    
                # Analyze performance in different market conditions
                for condition in results['conditions']:
                    for cond_type, stats in condition.items():
                        if cond_type in condition_results:
                            condition_results[cond_type]['trades'] += stats['trades']
                            condition_results[cond_type]['wins'] += stats['wins']
        
        print(f"Best parameters for {pair}: {best_params}, Win rate: {best_win_rate:.2f}%")
        results[pair] = {
            'original_win_rate': original_results['win_rate'],
            'best_win_rate': best_win_rate,
            'improvement': best_win_rate - original_results['win_rate'],
            'best_params': best_params
        }
    
    # Calculate average parameter performance
    avg_param_performance = {}
    for param_key, win_rates in parameter_performance.items():
        avg_param_performance[param_key] = sum(win_rates) / len(win_rates)
    
    # Calculate condition win rates
    for condition, stats in condition_results.items():
        if stats['trades'] > 0:
            stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
        else:
            stats['win_rate'] = 0
    
    # Generate optimization recommendations
    recommendations = generate_strategy_recommendations(results, avg_param_performance, condition_results)
    
    print("\n==== Strategy Optimization Recommendations ====")
    for rec in recommendations:
        print(f"- {rec}")
    
       # Create JSON-formatted strategy parameter recommendations
    optimized_params = {}
    for param_key, win_rates in sorted(avg_param_performance.items(), key=lambda x: x[1], reverse=True):
        try:
            # Split the key properly to get parameter name and value
            parts = param_key.split('_')
            if len(parts) >= 2:
                # The last part should be the value
                param_value = parts[-1]
                # Everything before the last part is the parameter name
                param_name = '_'.join(parts[:-1])
            
                # Only add if we haven't already added this parameter (to get the best one)
                if param_name not in optimized_params:
                    optimized_params[param_name] = float(param_value) if '.' in param_value else int(param_value)
        except Exception as e:
            print(f"Error processing parameter {param_key}: {e}")
            continue
    
    # Save optimized parameters
    with open("optimized_strategy_params.json", "w") as f:
        json.dump(optimized_params, f, indent=4)
    
    print(f"\nOptimized parameters saved to optimized_strategy_params.json")
    return optimized_params, recommendations, condition_results

def test_strategy_parameters(df, pair, params):
    """
    Tests strategy with specific parameters and returns performance metrics.
    
    Parameters:
    - df: DataFrame with historical price data and indicators
    - pair: Forex pair being tested
    - params: Dictionary of parameters to override
    
    Returns:
    - Dictionary of performance metrics
    """
    # Default parameters
    test_params = {
        'rsi_buy_threshold': 30,
        'rsi_sell_threshold': 70,
        'min_strength_threshold': 30,
        'atr_multiplier_trending': 2.5,
        'atr_multiplier_ranging': 1.5,
        'min_reward_ratio': 1.5
    }
    
    # Override with provided parameters
    for key, value in params.items():
        test_params[key] = value
    
    # Prepare results structure
    results = {
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0,
        'conditions': []
    }
    
    # Condition trackers
    conditions = {
        'trending_up': {'trades': 0, 'wins': 0},
        'trending_down': {'trades': 0, 'wins': 0},
        'ranging': {'trades': 0, 'wins': 0},
        'high_volatility': {'trades': 0, 'wins': 0},
        'low_volatility': {'trades': 0, 'wins': 0},
        'overbought': {'trades': 0, 'wins': 0},
        'oversold': {'trades': 0, 'wins': 0}
    }
    
    # Simulate trades with the given parameters
    for i in range(100, len(df) - 20):
        current_data = df.iloc[:i+1].copy()
        future_data = df.iloc[i+1:i+21].copy()
        
        latest = current_data.iloc[-1]
        
        # Determine market conditions
        is_trending = latest.get('is_trending', 0) == 1
        is_overbought = latest.get('RSI', 50) > 70
        is_oversold = latest.get('RSI', 50) < 30
        is_high_volatility = latest.get('atr_pct', 1) > 2.0
        is_low_volatility = latest.get('atr_pct', 1) < 0.5
        
        # Generate signal based on modified parameters
        signal = None
        if latest['close'] > latest['SMA20'] and latest['RSI'] < test_params['rsi_sell_threshold']:
            signal = "BUY"
            if latest['close'] > latest['SMA20']:
                conditions['trending_up']['trades'] += 1
        elif latest['close'] < latest['SMA20'] and latest['RSI'] > test_params['rsi_buy_threshold']:
            signal = "SELL"
            if latest['close'] < latest['SMA20']:
                conditions['trending_down']['trades'] += 1
        
        # Skip if no signal
        if not signal:
            continue
            
        # Check MACD confirmation with original logic
        macd_hist = latest.get('MACD_hist', 0)
        if signal == "BUY" and macd_hist <= 0:
            continue
        if signal == "SELL" and macd_hist >= 0:
            continue

        # Track market conditions
        if is_trending:
            if signal == "BUY":
                conditions['trending_up']['trades'] += 1
            else:
                conditions['trending_down']['trades'] += 1
        else:
            conditions['ranging']['trades'] += 1
            
        if is_high_volatility:
            conditions['high_volatility']['trades'] += 1
        elif is_low_volatility:
            conditions['low_volatility']['trades'] += 1
            
        if is_overbought:
            conditions['overbought']['trades'] += 1
        elif is_oversold:
            conditions['oversold']['trades'] += 1

        # Entry price
        entry = latest['close']
        
        # Calculate stop loss with custom ATR multiplier
        atr = latest.get('atr', entry * 0.01)
        atr_multiplier = test_params['atr_multiplier_trending'] if is_trending else test_params['atr_multiplier_ranging']
        stop_distance = atr * atr_multiplier
        
        if signal == "BUY":
            stoploss = entry - stop_distance
        else:
            stoploss = entry + stop_distance
        
        # Calculate take profit with custom reward ratio
        risk = abs(entry - stoploss)
        reward_ratio = test_params['min_reward_ratio']
        take_profit = entry + (risk * reward_ratio) if signal == "BUY" else entry - (risk * reward_ratio)
        
        # Simulate trade outcome
        result = "open"
        for _, future_row in future_data.iterrows():
            current_price = future_row["close"]
            
            if signal == "BUY":
                if current_price >= take_profit:
                    result = "win"
                    break
                elif current_price <= stoploss:
                    result = "loss"
                    break
            else:  # SELL
                if current_price <= take_profit:
                    result = "win"
                    break
                elif current_price >= stoploss:
                    result = "loss"
                    break
        
        # If trade remains open after 20 candles, consider it a loss
        if result == "open":
            result = "loss"
        
        # Update condition results
        if is_trending:
            if signal == "BUY":
                if result == "win":
                    conditions['trending_up']['wins'] += 1
            else:
                if result == "win":
                    conditions['trending_down']['wins'] += 1
        else:
            if result == "win":
                conditions['ranging']['wins'] += 1
                
        if is_high_volatility and result == "win":
            conditions['high_volatility']['wins'] += 1
        elif is_low_volatility and result == "win":
            conditions['low_volatility']['wins'] += 1
            
        if is_overbought and result == "win":
            conditions['overbought']['wins'] += 1
        elif is_oversold and result == "win":
            conditions['oversold']['wins'] += 1
        
        # Update overall results
        results['trades'] += 1
        if result == "win":
            results['wins'] += 1
        else:
            results['losses'] += 1
    
    # Calculate win rate
    if results['trades'] > 0:
        results['win_rate'] = (results['wins'] / results['trades']) * 100
    
    # Add conditions to results
    results['conditions'] = [conditions]
    
    return results

def generate_strategy_recommendations(results, param_performance, condition_results):
    """
    Generates strategy recommendations based on parameter testing results.
    """
    recommendations = []
    
    # Find best performing parameters
    best_params = {}
    for param_key, win_rate in sorted(param_performance.items(), key=lambda x: x[1], reverse=True):
        param_name, param_value = param_key.split('_', 1)
        if param_name not in best_params:
            best_params[param_name] = param_value
            
    # Generate parameter recommendations
    recommendations.append(f"Recommended RSI Buy Threshold: {best_params.get('rsi_buy_threshold', 30)}")
    recommendations.append(f"Recommended RSI Sell Threshold: {best_params.get('rsi_sell_threshold', 70)}")
    recommendations.append(f"Recommended Min Strength Threshold: {best_params.get('min_strength_threshold', 30)}")
    recommendations.append(f"Recommended ATR Multiplier (Trending): {best_params.get('atr_multiplier_trending', 2.5)}")
    recommendations.append(f"Recommended ATR Multiplier (Ranging): {best_params.get('atr_multiplier_ranging', 1.5)}")
    recommendations.append(f"Recommended Min Reward Ratio: {best_params.get('min_reward_ratio', 1.5)}")
    
    # Analyze market conditions
    for condition, stats in condition_results.items():
        if stats['trades'] > 10:  # Only consider conditions with enough data
            recommendations.append(f"{condition.replace('_', ' ').title()} markets: {stats['win_rate']:.1f}% win rate ({stats['wins']}/{stats['trades']} trades)")
    
    # Add pair-specific insights
    if results and isinstance(results, dict) and len(results) > 0:
        # Debug output to understand what's in results
        print("Results structure:", type(results))
        valid_pairs = {}
        
        # Check each pair result to make sure it has the required structure
        for pair_name, pair_data in results.items():
            print(f"Checking pair {pair_name}, data type: {type(pair_data)}")
            if isinstance(pair_data, dict) and 'best_win_rate' in pair_data:
                valid_pairs[pair_name] = pair_data
        
        # Only proceed if we have valid pairs
        if valid_pairs:
            best_pair = max(valid_pairs.items(), key=lambda x: x[1]['best_win_rate'])
            worst_pair = min(valid_pairs.items(), key=lambda x: x[1]['best_win_rate'])
            
            recommendations.append(f"Best performing pair: {best_pair[0]} ({best_pair[1]['best_win_rate']:.1f}% win rate)")
            recommendations.append(f"Worst performing pair: {worst_pair[0]} ({worst_pair[1]['best_win_rate']:.1f}% win rate)")
        else:
            recommendations.append("Pair-specific analysis unavailable due to insufficient data")
    else:
        recommendations.append("No pair performance data available")
    
    return recommendations


def train_balanced_ml_model(trade_history, balance_method='smote'):
    """
    Trains a ML model with techniques to handle class imbalance.
    
    Parameters:
    - trade_history: DataFrame with trade history
    - balance_method: Method for handling class imbalance:
      'smote', 'oversample', 'undersample', or 'class_weight'
    
    Returns:
    - trained model, scaler, and report
    """
    if trade_history.empty or len(trade_history) < 20:
        return None, None, "Not enough trades for ML training"
    
    # Ensure timestamp is datetime
    if 'timestamp' in trade_history.columns:
        trade_history['timestamp'] = pd.to_datetime(trade_history['timestamp'])
    else:
        return None, None, "Timestamp column required for time-based training"
    
    # Only use closed trades
    closed_trades = trade_history[trade_history["status"] != "open"].copy()
    if closed_trades.empty or len(closed_trades) < 20:
        return None, None, "Not enough closed trades for training"
    
    # Sort by timestamp
    closed_trades = closed_trades.sort_values('timestamp')
    
    # Extract features and target
    X = []
    y = []
    
    for _, trade in closed_trades.iterrows():
        try:
            # Extract features
            features = extract_features_for_analysis(trade)
            
            # Skip if features are invalid
            if not features:
                continue
                
            # Convert features to list in consistent order
            feature_list = [
                features.get('time_of_day', 0),
                features.get('day_of_week', 0),
                features.get('signal_buy', 0),
                features.get('risk_reward_ratio', 1.0),
                features.get('confidence', 50.0),
                features.get('market_volatility', 0),
                features.get('market_trend_strength', 0),
                features.get('atr_pct', 1.0),
                features.get('adx', 25),
                features.get('is_trending', 0),
                features.get('rsi', 50),
                features.get('macd_hist', 0),
                features.get('bb_width', 0),
                features.get('mtf_strength', 0),
                features.get('sma_score', 5.0),
                features.get('rsi_score', 5.0),
                features.get('macd_score', 5.0),
                features.get('volatility_score', 5.0),
                features.get('regime_score', 5.0),
                features.get('bb_score', 5.0)
            ]
            
            X.append(feature_list)
            y.append(1 if trade['status'] == 'win' else 0)
        except Exception as e:
            print(f"Error extracting features: {e}")
            continue
    
    if len(X) < 20:
        return None, None, f"Only {len(X)} valid trades for training after feature extraction"
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Time-based split - use the last 20% of data chronologically for testing
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Count class distribution in training set
    win_count = np.sum(y_train == 1)
    loss_count = np.sum(y_train == 0)
    class_ratio = win_count / len(y_train)
    
    print(f"Original class distribution: {win_count} wins ({class_ratio:.2%}), {loss_count} losses ({1-class_ratio:.2%})")
    
    # Balance the dataset based on the chosen method
    if balance_method == 'oversample':
        # Oversample the minority class (winning trades)
        from sklearn.utils import resample
        
        # Separate majority and minority classes
        train_data = np.column_stack((X_train, y_train))
        win_samples = train_data[train_data[:, -1] == 1]
        loss_samples = train_data[train_data[:, -1] == 0]
        
        # Oversample minority class
        if len(win_samples) < len(loss_samples):
            win_resampled = resample(
                win_samples,
                replace=True,
                n_samples=len(loss_samples),
                random_state=42
            )
            # Combine majority class with oversampled minority class
            train_resampled = np.vstack([loss_samples, win_resampled])
        else:
            loss_resampled = resample(
                loss_samples,
                replace=True,
                n_samples=len(win_samples),
                random_state=42
            )
            # Combine majority class with oversampled minority class
            train_resampled = np.vstack([win_samples, loss_resampled])
        
        # Split back into features and target
        X_train = train_resampled[:, :-1]
        y_train = train_resampled[:, -1].astype(int)
        
        print(f"After oversampling: {np.sum(y_train == 1)} wins, {np.sum(y_train == 0)} losses")
        
    elif balance_method == 'undersample':
        # Undersample the majority class (losing trades)
        from sklearn.utils import resample
        
        # Separate majority and minority classes
        train_data = np.column_stack((X_train, y_train))
        win_samples = train_data[train_data[:, -1] == 1]
        loss_samples = train_data[train_data[:, -1] == 0]
        
        # Undersample majority class
        if len(win_samples) < len(loss_samples):
            loss_resampled = resample(
                loss_samples,
                replace=False,
                n_samples=len(win_samples),
                random_state=42
            )
            # Combine undersampled majority class with minority class
            train_resampled = np.vstack([win_samples, loss_resampled])
        else:
            win_resampled = resample(
                win_samples,
                replace=False,
                n_samples=len(loss_samples),
                random_state=42
            )
            # Combine undersampled majority class with minority class
            train_resampled = np.vstack([win_resampled, loss_samples])
        
        # Split back into features and target
        X_train = train_resampled[:, :-1]
        y_train = train_resampled[:, -1].astype(int)
        
        print(f"After undersampling: {np.sum(y_train == 1)} wins, {np.sum(y_train == 0)} losses")
        
    elif balance_method == 'smote':
        # Use SMOTE to create synthetic examples of the minority class
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"After SMOTE: {np.sum(y_train == 1)} wins, {np.sum(y_train == 0)} losses")
        except ImportError:
            print("SMOTE requires imbalanced-learn package. Using class weights instead.")
            balance_method = 'class_weight'
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if balance_method == 'class_weight':
        # Calculate class weights based on inverse frequency
        class_weights = {
            0: 1.0 / (np.sum(y_train == 0) / len(y_train)),
            1: 1.0 / (np.sum(y_train == 1) / len(y_train))
        }
        print(f"Using class weights: {class_weights}")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_preds = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_preds)
    
    test_preds = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_preds)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_recall = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    
    # Calculate ROC AUC if possible
    try:
        from sklearn.metrics import roc_auc_score
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)
        auc_result = f"ROC AUC: {test_auc:.4f}"
    except:
        auc_result = "ROC AUC: Not available"
    
    # Check for overfitting
    overfitting_gap = train_accuracy - test_accuracy
    
    # Feature importance
    feature_names = [
        'time_of_day', 'day_of_week', 'signal_buy', 'risk_reward_ratio', 
        'confidence', 'market_volatility', 'market_trend_strength',
        'atr_pct', 'adx', 'is_trending', 'rsi', 'macd_hist', 'bb_width',
        'mtf_strength', 'sma_score', 'rsi_score', 'macd_score',
        'volatility_score', 'regime_score', 'bb_score'
    ]
    
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Find optimal threshold
    optimal_threshold, threshold_metrics = find_optimal_threshold(model, X_test_scaled, y_test)
    
    # Generate result message
    result = (
        f"ML model trained with {balance_method} balancing:\n"
        f"- Train set: {len(X_train)} trades ({train_accuracy:.4f} accuracy)\n"
        f"- Test set: {len(X_test)} trades ({test_accuracy:.4f} accuracy)\n"
        f"- Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}\n"
        f"- {auc_result}\n"
        f"- Confusion Matrix:\n"
        f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}\n"
        f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}\n"
        f"- Optimal Threshold: {optimal_threshold:.4f} (Precision: {threshold_metrics['precision']:.4f}, "
        f"Recall: {threshold_metrics['recall']:.4f}, F1: {threshold_metrics['f1']:.4f})\n"
    )
    
    if overfitting_gap > 0.2:
        result += f"⚠️ Warning: Potential overfitting detected (gap: {overfitting_gap:.4f})\n"
    
    result += "\nTop 5 important features:\n"
    for feat, imp in top_features[:5]:
        result += f"- {feat}: {imp:.4f}\n"
    
    # Save feature importance
    save_feature_importance(importance_dict)
    
    # Save optimal threshold
    with open("ml_optimal_threshold.txt", "w") as f:
        f.write(str(optimal_threshold))
    
    # Log ML training with additional metrics
    log_ml_learning_enhanced(test_accuracy, test_precision, test_recall, top_features[:5], 
                           f1=test_f1, auc=test_auc if 'test_auc' in locals() else None,
                           threshold=optimal_threshold)
    
    return model, scaler, result

def find_optimal_threshold(model, X_test_scaled, y_test):
    """
    Finds the optimal decision threshold for the model based on F1 score.
    """
    # Get predicted probabilities
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    best_metrics = {}
    
    for threshold in thresholds:
        # Apply threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Track best threshold based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    return best_threshold, best_metrics

def log_ml_learning_enhanced(accuracy, precision, recall, top_features, f1=None, auc=None, threshold=0.5):
    """
    Enhanced log function for ML training results with more metrics.
    """
    log_df = load_ml_learning_log()
    
    # Convert top features to string
    top_features_str = ', '.join([f"{feat}: {imp:.4f}" for feat, imp in top_features])
    
    # Prepare new log entry
    log_entry = {
        'timestamp': datetime.datetime.now(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1 if f1 is not None else 0,
        'auc': auc if auc is not None else 0,
        'optimal_threshold': threshold,
        'top_features': top_features_str,
        'train_data_size': len(prepare_ml_data(load_trade_history())[0]) if prepare_ml_data(load_trade_history())[0] is not None else 0
    }
    
    # Add to log
    new_log = pd.DataFrame([log_entry])
    log_df = pd.concat([log_df, new_log], ignore_index=True)
    
    # Save enhanced log
    columns = list(log_df.columns)
    log_df.to_csv(ML_LEARNING_LOG_FILE, index=False, columns=columns)
    
    return log_entry



def calculate_indicators_forward_only(df, current_index=None):
    """
    Calculates technical indicators using ONLY data available up to current_index.
    If current_index is None, calculate for all rows but ensure each calculation
    only uses data available up to that point.
    
    Parameters:
    - df: DataFrame with OHLC data
    - current_index: Optional index up to which data should be considered
    
    Returns:
    - DataFrame with indicators added
    """
    if current_index is not None:
        # Only use data up to current_index
        work_df = df.iloc[:current_index+1].copy()
    else:
        work_df = df.copy()
    
    # Create result DataFrame to avoid in-place modifications
    result_df = work_df.copy()
    
    # For each row, calculate indicators using only past data
    for i in range(len(result_df)):
        # Get data up to current row
        past_df = work_df.iloc[:i+1]
        
        # Skip if we don't have enough data for calculations
        if len(past_df) < 20:  # Minimum data needed for most indicators
            continue
        
        # Calculate Simple Moving Averages
        if len(past_df) >= 20:
            result_df.loc[past_df.index[i], 'SMA20'] = past_df['close'].iloc[-20:].mean()
        
        if len(past_df) >= 50:
            result_df.loc[past_df.index[i], 'SMA50'] = past_df['close'].iloc[-50:].mean()
        
        if len(past_df) >= 200:
            result_df.loc[past_df.index[i], 'SMA200'] = past_df['close'].iloc[-200:].mean()
        
        # Calculate RSI
        if len(past_df) >= 15:  # Need at least 14+1 periods
            close_prices = past_df['close']
            delta = close_prices.diff().iloc[1:]  # Remove first NaN
            
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=14).mean().iloc[-1]
            avg_loss = losses.rolling(window=14).mean().iloc[-1]
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
                
            result_df.loc[past_df.index[i], 'RSI'] = rsi
        
        # Calculate MACD
        if len(past_df) >= 27:  # Need at least 26+1 periods
            ema12 = past_df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = past_df['close'].ewm(span=26, adjust=False).mean().iloc[-1]
            macd = ema12 - ema26
            
            # Signal line is 9-period EMA of MACD
            if i >= 35:  # Need enough data for MACD + signal
                past_macds = []
                for j in range(max(0, i-8), i+1):
                    if j < 26:  # Not enough data for earlier points
                        continue
                    past_j_df = work_df.iloc[:j+1]
                    ema12_j = past_j_df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
                    ema26_j = past_j_df['close'].ewm(span=26, adjust=False).mean().iloc[-1]
                    past_macds.append(ema12_j - ema26_j)
                
                if len(past_macds) > 0:
                    signal = sum(past_macds) / len(past_macds)
                    result_df.loc[past_df.index[i], 'MACD'] = macd
                    result_df.loc[past_df.index[i], 'MACD_signal'] = signal
                    result_df.loc[past_df.index[i], 'MACD_hist'] = macd - signal
        
        # Calculate Support and Resistance
        if len(past_df) >= 20:
            result_df.loc[past_df.index[i], 'Support'] = past_df['low'].rolling(window=20).min().iloc[-1]
            result_df.loc[past_df.index[i], 'Resistance'] = past_df['high'].rolling(window=20).max().iloc[-1]
        
        # Calculate ATR
        if len(past_df) >= 15:  # Need at least 14+1 periods
            high = past_df['high']
            low = past_df['low']
            close = past_df['close']
            
            tr1 = high.iloc[-14:].values - low.iloc[-14:].values
            tr2 = abs(high.iloc[-14:].values - close.iloc[-15:-1].values)
            tr3 = abs(low.iloc[-14:].values - close.iloc[-15:-1].values)
            
            tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
            atr = tr.mean()
            
            result_df.loc[past_df.index[i], 'atr'] = atr
            result_df.loc[past_df.index[i], 'atr_pct'] = (atr / close.iloc[-1]) * 100
        
        # Calculate Bollinger Bands
        if len(past_df) >= 20:
            sma20 = past_df['close'].iloc[-20:].mean()
            std20 = past_df['close'].iloc[-20:].std()
            
            result_df.loc[past_df.index[i], 'bb_middle'] = sma20
            result_df.loc[past_df.index[i], 'bb_upper'] = sma20 + (2 * std20)
            result_df.loc[past_df.index[i], 'bb_lower'] = sma20 - (2 * std20)
            result_df.loc[past_df.index[i], 'bb_width'] = (sma20 + (2 * std20) - (sma20 - (2 * std20))) / sma20
            
            # Calculate percent of price relative to BB
            current_price = past_df['close'].iloc[-1]
            upper = sma20 + (2 * std20)
            lower = sma20 - (2 * std20)
            result_df.loc[past_df.index[i], 'bb_pct'] = (current_price - lower) / (upper - lower)
        
        # Calculate volatility
        if len(past_df) >= 2:
            result_df.loc[past_df.index[i], 'volatility'] = past_df['high'].iloc[-1] - past_df['low'].iloc[-1]
            result_df.loc[past_df.index[i], 'volatility_pct'] = (result_df.loc[past_df.index[i], 'volatility'] / past_df['close'].iloc[-1]) * 100
        
        # Calculate ADX for trend strength
        if len(past_df) >= 25:  # Minimum data for ADX
            try:
                # Simplified ADX calculation
                high = past_df['high'].values
                low = past_df['low'].values
                close = past_df['close'].values
                
                # +DM and -DM
                plus_dm = np.zeros(len(high)-1)
                minus_dm = np.zeros(len(high)-1)
                
                for j in range(1, len(high)):
                    h_diff = high[j] - high[j-1]
                    l_diff = low[j-1] - low[j]
                    
                    if (h_diff > 0) and (h_diff > l_diff):
                        plus_dm[j-1] = h_diff
                    else:
                        plus_dm[j-1] = 0
                        
                    if (l_diff > 0) and (l_diff > h_diff):
                        minus_dm[j-1] = l_diff
                    else:
                        minus_dm[j-1] = 0
                
                # Calculate TR
                tr = np.zeros(len(high)-1)
                for j in range(1, len(high)):
                    tr[j-1] = max(high[j]-low[j], abs(high[j]-close[j-1]), abs(low[j]-close[j-1]))
                
                # Smooth with EMA
                tr_14 = np.mean(tr[-14:])
                plus_di_14 = 100 * np.mean(plus_dm[-14:]) / tr_14
                minus_di_14 = 100 * np.mean(minus_dm[-14:]) / tr_14
                
                # Calculate DX and ADX
                dx = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) if (plus_di_14 + minus_di_14) > 0 else 0
                
                # ADX is 14-period average of DX, but for simplicity, we'll just use current DX
                adx = dx
                
                result_df.loc[past_df.index[i], 'adx'] = adx
                result_df.loc[past_df.index[i], 'is_trending'] = 1 if adx > 25 else 0
                result_df.loc[past_df.index[i], 'market_regime'] = 'trending' if adx > 25 else 'ranging'
            except Exception as e:
                # Fallback if calculation fails
                result_df.loc[past_df.index[i], 'adx'] = 25
                result_df.loc[past_df.index[i], 'is_trending'] = 0
                result_df.loc[past_df.index[i], 'market_regime'] = 'unknown'
        
        # Calculate momentum indicators
        if len(past_df) >= 21:
            result_df.loc[past_df.index[i], 'momentum_5'] = (past_df['close'].iloc[-1] / past_df['close'].iloc[-6] - 1) * 100
            result_df.loc[past_df.index[i], 'momentum_10'] = (past_df['close'].iloc[-1] / past_df['close'].iloc[-11] - 1) * 100
            result_df.loc[past_df.index[i], 'momentum_20'] = (past_df['close'].iloc[-1] / past_df['close'].iloc[-21] - 1) * 100
    
    # Fill NaN values
    result_df = result_df.bfill().ffill().fillna(0)

    
    return result_df

def train_ml_model_time_based(trade_history):
    """
    Trains ML model using time-based splits to prevent data leakage.
    
    Parameters:
    - trade_history: DataFrame of historical trades with timestamps
    
    Returns:
    - trained model, scaler, and training results message
    """
    if trade_history.empty or len(trade_history) < 20:
        return None, None, "Not enough trades for ML training"
    
    # Ensure timestamp is datetime
    if 'timestamp' in trade_history.columns:
        trade_history['timestamp'] = pd.to_datetime(trade_history['timestamp'])
    else:
        return None, None, "Timestamp column required for time-based training"
    
    # Only use closed trades
    closed_trades = trade_history[trade_history["status"] != "open"].copy()
    if closed_trades.empty or len(closed_trades) < 20:
        return None, None, "Not enough closed trades for training"
    
    # Sort by timestamp
    closed_trades = closed_trades.sort_values('timestamp')
    
    # Extract features and target
    X = []
    y = []
    
    for _, trade in closed_trades.iterrows():
        try:
            # Extract features
            features = extract_features_for_analysis(trade)
            
            # Skip if features are invalid
            if not features:
                continue
                
            # Convert features to list in consistent order
            feature_list = [
                features.get('time_of_day', 0),
                features.get('day_of_week', 0),
                features.get('signal_buy', 0),
                features.get('risk_reward_ratio', 1.0),
                features.get('confidence', 50.0),
                features.get('market_volatility', 0),
                features.get('market_trend_strength', 0),
                features.get('atr_pct', 1.0),
                features.get('adx', 25),
                features.get('is_trending', 0),
                features.get('rsi', 50),
                features.get('macd_hist', 0),
                features.get('bb_width', 0),
                features.get('mtf_strength', 0),
                features.get('sma_score', 5.0),
                features.get('rsi_score', 5.0),
                features.get('macd_score', 5.0),
                features.get('volatility_score', 5.0),
                features.get('regime_score', 5.0),
                features.get('bb_score', 5.0)
            ]
            
            X.append(feature_list)
            y.append(1 if trade['status'] == 'win' else 0)
        except Exception as e:
            print(f"Error extracting features: {e}")
            continue
    
    if len(X) < 20:
        return None, None, f"Only {len(X)} valid trades for training after feature extraction"
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Time-based split - use the last 20% of data chronologically for testing
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,  # Limit depth to reduce overfitting
        min_samples_split=5,  # Require more samples per split
        min_samples_leaf=2,  # Require more samples per leaf
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_preds = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_preds)
    
    test_preds = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_recall = recall_score(y_test, test_preds, zero_division=0)
    
    # Check for overfitting
    overfitting_gap = train_accuracy - test_accuracy
    
    # Feature importance
    feature_names = [
        'time_of_day', 'day_of_week', 'signal_buy', 'risk_reward_ratio', 
        'confidence', 'market_volatility', 'market_trend_strength',
        'atr_pct', 'adx', 'is_trending', 'rsi', 'macd_hist', 'bb_width',
        'mtf_strength', 'sma_score', 'rsi_score', 'macd_score',
        'volatility_score', 'regime_score', 'bb_score'
    ]
    
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Generate result message
    result = (
        f"ML model trained with time-based splits:\n"
        f"- Train set: {len(X_train)} trades ({train_accuracy:.2f} accuracy)\n"
        f"- Test set: {len(X_test)} trades ({test_accuracy:.2f} accuracy)\n"
        f"- Precision: {test_precision:.2f}, Recall: {test_recall:.2f}\n"
    )
    
    if overfitting_gap > 0.2:
        result += f"⚠️ Warning: Potential overfitting detected (gap: {overfitting_gap:.2f})\n"
    
    result += "\nTop 5 important features:\n"
    for feat, imp in top_features[:5]:
        result += f"- {feat}: {imp:.3f}\n"
    
    # Save feature importance
    save_feature_importance(importance_dict)
    
    return model, scaler, result

def walk_forward_analysis(pair, start_date, end_date, window_size=30, step_size=7):
    """
    Perform walk-forward analysis for a specific forex pair.
    This function simulates the trading process through time, making
    decisions with only the data available at each point.
    
    Parameters:
    - pair: Forex pair to analyze (e.g., "EURUSD")
    - start_date: Start date for analysis (string or datetime)
    - end_date: End date for analysis (string or datetime)
    - window_size: Size of each training window in days
    - step_size: Number of days to move forward after each window
    
    Returns:
    - DataFrame with trading signals and results
    """
    # Convert dates if they're strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    print(f"Starting walk-forward analysis for {pair} from {start_date.date()} to {end_date.date()}")
    
    # Load historical data
    historical_dir = os.path.join("historical_data", pair)
    if not os.path.exists(historical_dir):
        print(f"No historical data found for {pair}")
        return None
    
    # Load data for each timeframe
    timeframes_data = {}
    for tf in TIMEFRAMES:
        tf_file = os.path.join(historical_dir, f"{tf}.csv")
        if os.path.exists(tf_file):
            df = pd.read_csv(tf_file, parse_dates=["time"])
            timeframes_data[tf] = df
        else:
            print(f"Missing data for {pair} on {tf} timeframe")
            if tf == "15min":  # Primary timeframe is required
                return None
    
    # Filter data by date range
    for tf in timeframes_data:
        timeframes_data[tf] = timeframes_data[tf][
            (timeframes_data[tf]["time"] >= start_date) & 
            (timeframes_data[tf]["time"] <= end_date)
        ]
    
    # Check if we have enough data
    if timeframes_data["15min"].empty:
        print(f"No data available for {pair} in the specified date range")
        return None
    
    # Initialize results storage
    all_results = []
    
    # Initialize trade history for ML training
    trade_history = pd.DataFrame(columns=[
        "timestamp", "instrument", "signal", "entry", "stoploss", 
        "take_profit", "risk_reward", "confidence", "status"
    ])
    
    # Set up initial ML model (None at start)
    ml_model = None
    scaler = None
    
    # Get 15min data (primary timeframe)
    df_15min = timeframes_data["15min"]
    
    # Walk through time in steps
    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + pd.Timedelta(days=step_size)
        print(f"Processing window from {current_date.date()} to {next_date.date()}")
        
        # Get data up to current date for analysis
        window_data = {}
        for tf in timeframes_data:
            window_data[tf] = timeframes_data[tf][
                timeframes_data[tf]["time"] <= current_date
            ].copy()
            
            # Calculate indicators with forward-only approach
            if not window_data[tf].empty:
                window_data[tf] = calculate_indicators_forward_only(window_data[tf])
        
        # Check if we have enough data for analysis
        if any(df.empty for df in window_data.values()):
            print(f"Not enough data for all timeframes at {current_date.date()}")
            current_date = next_date
            continue
        
        # Train/update ML model if we have enough trade history
        closed_trades = trade_history[trade_history["status"] != "open"]
        if len(closed_trades) >= 20 and (ml_model is None or len(closed_trades) % 10 == 0):
            print(f"Training ML model with {len(closed_trades)} closed trades")
            ml_model, scaler, training_result = train_ml_model_time_based(trade_history)
            print(training_result)
        
        # Generate trading signals for the current window
        for day_offset in range(min(step_size, (end_date - current_date).days + 1)):
            analysis_date = current_date + pd.Timedelta(days=day_offset)
            
            # Get data up to this specific day
            day_data = {}
            for tf in timeframes_data:
                day_data[tf] = timeframes_data[tf][
                    timeframes_data[tf]["time"] <= analysis_date
                ].copy()
                
                # Use the already calculated indicators from window_data
                day_data[tf] = window_data[tf].copy()
            
            # Get the latest 15min candle for the day
            day_15min = day_data["15min"]
            if day_15min.empty:
                continue
                
            # Generate trade signal
            try:
                trade = decide_trade_optimized(
                    df=day_15min,
                    multi_timeframe_data=day_data,
                    ml_model=ml_model,
                    scaler=scaler
                )
                
                if trade:
                    trade["instrument"] = pair
                    trade["timestamp"] = analysis_date
                    
                    # Record the trade
                    trade_entry = {
                        "timestamp": analysis_date,
                        "instrument": pair,
                        "signal": trade["signal"],
                        "entry": trade["entry"],
                        "stoploss": trade["stoploss"],
                        "take_profit": trade["take_profit"],
                        "risk_reward": trade["risk_reward"],
                        "confidence": trade["confidence"],
                        "status": "open"
                    }
                    
                    # Add to results
                    all_results.append({
                        "date": analysis_date,
                        "instrument": pair,
                        "signal": trade["signal"],
                        "entry": trade["entry"],
                        "stoploss": trade["stoploss"],
                        "take_profit": trade["take_profit"],
                        "confidence": trade["confidence"],
                        "time_of_day": analysis_date.hour,
                        "day_of_week": analysis_date.dayofweek
                    })
                    
                    # Find out what actually happened (for historical analysis)
                    # First find the index of the current date in our data
                    future_data = df_15min[df_15min["time"] > analysis_date].head(20)
                    
                    # Determine actual outcome
                    result = "open"
                    for _, future_row in future_data.iterrows():
                        current_price = future_row["close"]
                        
                        if trade["signal"] == "BUY":
                            if current_price >= trade["take_profit"]:
                                result = "win"
                                break
                            elif current_price <= trade["stoploss"]:
                                result = "loss"
                                break
                        else:  # SELL
                            if current_price <= trade["take_profit"]:
                                result = "win"
                                break
                            elif current_price >= trade["stoploss"]:
                                result = "loss"
                                break
                    
                    # If still open after 20 candles, consider it a timeout
                    if result == "open":
                        result = "loss"
                    
                    # Update the latest result
                    all_results[-1]["result"] = result
                    
                    # Add to trade history for ML model training
                    trade_entry["status"] = result
                    trade_history = pd.concat([trade_history, pd.DataFrame([trade_entry])], 
                                             ignore_index=True)
                    
                    print(f"Date: {analysis_date}, Signal: {trade['signal']}, Result: {result}, Confidence: {trade['confidence']:.1f}%")
            except Exception as e:
                print(f"Error generating trade for {analysis_date}: {e}")
        
        # Move to next window
        current_date = next_date
    
    # Convert results to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Calculate overall accuracy
        if "result" in results_df.columns and len(results_df) > 0:
            wins = (results_df["result"] == "win").sum()
            total = len(results_df)
            win_rate = (wins / total) * 100
            print(f"Overall walk-forward win rate: {win_rate:.2f}% ({wins}/{total})")
        
        return results_df
    else:
        print("No trading signals generated during walk-forward analysis")
        return None


def evaluate_walk_forward_results(results_df):
    """
    Thoroughly analyze the results of walk-forward testing.
    
    Parameters:
    - results_df: DataFrame with walk-forward testing results
    
    Returns:
    - String with evaluation results
    """
    if results_df is None or len(results_df) == 0 or "result" not in results_df.columns:
        return "No valid results to evaluate"
    
    analysis = []
    
    # Overall performance
    total_trades = len(results_df)
    wins = (results_df["result"] == "win").sum()
    win_rate = (wins / total_trades) * 100
    
    analysis.append(f"=== Overall Performance ===")
    analysis.append(f"Total trades: {total_trades}")
    analysis.append(f"Win rate: {win_rate:.2f}% ({wins}/{total_trades})")
    
    # Performance by signal type
    for signal in results_df["signal"].unique():
        signal_df = results_df[results_df["signal"] == signal]
        signal_wins = (signal_df["result"] == "win").sum()
        signal_win_rate = (signal_wins / len(signal_df)) * 100
        analysis.append(f"{signal} win rate: {signal_win_rate:.2f}% ({signal_wins}/{len(signal_df)})")
    
    # Performance by confidence level
    analysis.append(f"\n=== Performance by Confidence ===")
    confidence_bins = [(0, 50), (50, 70), (70, 85), (85, 100)]
    
    for low, high in confidence_bins:
        bin_df = results_df[(results_df["confidence"] >= low) & (results_df["confidence"] < high)]
        if len(bin_df) > 0:
            bin_wins = (bin_df["result"] == "win").sum()
            bin_win_rate = (bin_wins / len(bin_df)) * 100
            analysis.append(f"Confidence {low}-{high}%: {bin_win_rate:.2f}% win rate ({bin_wins}/{len(bin_df)})")
    
    # Performance by time of day
    analysis.append(f"\n=== Performance by Time of Day ===")
    # Group by 4-hour blocks
    results_df["hour_block"] = (results_df["time_of_day"] // 4) * 4
    for block in sorted(results_df["hour_block"].unique()):
        block_df = results_df[results_df["hour_block"] == block]
        if len(block_df) > 0:
            block_wins = (block_df["result"] == "win").sum()
            block_win_rate = (block_wins / len(block_df)) * 100
            analysis.append(f"Hours {block}-{block+3}: {block_win_rate:.2f}% win rate ({block_wins}/{len(block_df)})")
    
    # Performance by day of week
    analysis.append(f"\n=== Performance by Day of Week ===")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in range(7):
        day_df = results_df[results_df["day_of_week"] == day]
        if len(day_df) > 0:
            day_wins = (day_df["result"] == "win").sum()
            day_win_rate = (day_wins / len(day_df)) * 100
            analysis.append(f"{day_names[day]}: {day_win_rate:.2f}% win rate ({day_wins}/{len(day_df)})")
    
    # Risk/reward analysis
    analysis.append(f"\n=== Risk/Reward Analysis ===")
    avg_tp_distance = ((results_df["take_profit"] - results_df["entry"]).abs() / results_df["entry"]).mean() * 100
    avg_sl_distance = ((results_df["stoploss"] - results_df["entry"]).abs() / results_df["entry"]).mean() * 100
    analysis.append(f"Avg. TP distance: {avg_tp_distance:.2f}%")
    analysis.append(f"Avg. SL distance: {avg_sl_distance:.2f}%")
    analysis.append(f"Avg. Risk/Reward ratio: {avg_tp_distance/avg_sl_distance:.2f}")
    
    # Profit factor calculation
    if "result" in results_df.columns:
        win_df = results_df[results_df["result"] == "win"]
        loss_df = results_df[results_df["result"] == "loss"]
        
        if not win_df.empty and not loss_df.empty:
            # Calculate average profit per winning trade
            avg_win = ((win_df["take_profit"] - win_df["entry"]).abs() / win_df["entry"]).mean() * 100
            # Calculate average loss per losing trade
            avg_loss = ((loss_df["entry"] - loss_df["stoploss"]).abs() / loss_df["entry"]).mean() * 100
            
            if avg_loss > 0:
                profit_factor = (avg_win * wins) / (avg_loss * (total_trades - wins))
                analysis.append(f"Profit factor: {profit_factor:.2f}")
                
                # Calculate expectancy
                expectancy = ((win_rate/100) * avg_win) - ((1 - win_rate/100) * avg_loss)
                analysis.append(f"Expectancy per trade: {expectancy:.2f}%")
    
    # Monthly breakdown
    if "date" in results_df.columns:
        analysis.append(f"\n=== Monthly Performance ===")
        results_df["month"] = results_df["date"].dt.strftime("%Y-%m")
        monthly_results = results_df.groupby("month").apply(
            lambda x: (x["result"] == "win").mean() * 100
        ).reset_index()
        monthly_results.columns = ["month", "win_rate"]
        
        for _, row in monthly_results.iterrows():
            month_df = results_df[results_df["month"] == row["month"]]
            month_trades = len(month_df)
            month_wins = (month_df["result"] == "win").sum()
            analysis.append(f"{row['month']}: {row['win_rate']:.2f}% win rate ({month_wins}/{month_trades})")
    
    return "\n".join(analysis)

# Ensure compatibility between function names
def ensure_functions():
    global decide_trade, decide_trade_optimized
    if 'decide_trade' not in globals() and 'decide_trade_optimized' in globals():
        decide_trade = decide_trade_optimized
    elif 'decide_trade_optimized' not in globals() and 'decide_trade' in globals():
        decide_trade_optimized = decide_trade

# Call this at the beginning of run_out_of_sample_test


def run_out_of_sample_test(pairs=None, start_date="2023-01-01", end_date="2023-06-30"):
    ensure_functions()
    """
    Run a complete out-of-sample test across multiple forex pairs.
    
    Parameters:
    - pairs: List of forex pairs to test (defaults to all pairs in FOREX_PAIRS)
    - start_date: Start date for testing
    - end_date: End date for testing
    
    Returns:
    - Dictionary of results by pair
    """
    if pairs is None:
        pairs = FOREX_PAIRS
    
    results = {}
    summary = []
    
    print(f"Starting out-of-sample testing from {start_date} to {end_date}")
    
    for pair in pairs:
        print(f"\n{'='*50}")
        print(f"Testing {pair}...")
        print(f"{'='*50}")
        
        # Run walk-forward analysis
        pair_results = walk_forward_analysis(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            window_size=30,  # 30 days per window
            step_size=14     # Move forward 2 weeks at a time
        )
        
        if pair_results is not None and len(pair_results) > 0:
            # Save results
            results[pair] = pair_results
            pair_results.to_csv(f"{pair}_walkforward_results.csv", index=False)
            
            # Evaluate results
            evaluation = evaluate_walk_forward_results(pair_results)
            print(f"\nEvaluation for {pair}:\n{evaluation}")
            
            # Generate summary
            total_trades = len(pair_results)
            wins = (pair_results["result"] == "win").sum() if "result" in pair_results.columns else 0
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            
            summary.append(f"{pair}: {win_rate:.2f}% win rate ({wins}/{total_trades})")
            
            # Create visualization
            try:
                plt.figure(figsize=(12, 6))
                
                # Plot equity curve
                trades = []
                balance = 1000  # Starting balance
                equity = [balance]
                
                for _, trade in pair_results.iterrows():
                    if trade["result"] == "win":
                        # Assuming fixed R:R of 1.5
                        gain = balance * 0.02 * 1.5  # 2% risk, 1.5R gain
                        balance += gain
                    else:
                        loss = balance * 0.02  # 2% risk
                        balance -= loss
                    equity.append(balance)
                    trades.append(trade["date"])
                
                plt.plot(trades, equity[1:], label="Equity Curve")
                plt.title(f"{pair} Walk-Forward Test Equity Curve")
                plt.xlabel("Date")
                plt.ylabel("Balance")
                plt.grid(True)
                plt.legend()
                plt.savefig(f"{pair}_equity_curve.png")
                plt.close()
                
                # Plot win rate by month
                if "date" in pair_results.columns:
                    pair_results["month"] = pair_results["date"].dt.strftime("%Y-%m")
                    monthly_results = pair_results.groupby("month").apply(
                        lambda x: (x["result"] == "win").mean() * 100
                    ).reset_index()
                    
                    plt.figure(figsize=(12, 6))
                    plt.bar(monthly_results["month"], monthly_results["win_rate"])
                    plt.title(f"{pair} Monthly Win Rate")
                    plt.xlabel("Month")
                    plt.ylabel("Win Rate (%)")
                    plt.axhline(y=50, color='r', linestyle='--', label="Break-even")
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f"{pair}_monthly_winrate.png")
                    plt.close()
            except Exception as e:
                print(f"Error creating visualizations for {pair}: {e}")
    
    # Print overall summary
    print("\n" + "="*50)
    print("OVERALL SUMMARY")
    print("="*50)
    for line in summary:
        print(line)
    
    return results

# -----------------------
# Hjælpefunktioner
# -----------------------

def get_historical_data(pair, interval="15min", outputsize="100"):
    """
    Henter historiske data fra Twelve Data API.
    """
    symbol = pair[:3] + "/" + pair[3:]
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY
    }
    response = requests.get("https://api.twelvedata.com/time_series", params=params)
    data = response.json()
    if "values" not in data:
        raise ValueError(f"Fejl i API-respons for {pair}: {data.get('message', data)}")
    values = data["values"]
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["open"] = pd.to_numeric(df["open"])
    df["high"] = pd.to_numeric(df["high"])
    df["low"] = pd.to_numeric(df["low"])
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"]) if "volume" in df.columns else 0
    df = df.sort_values("datetime")
    df = df.rename(columns={"datetime": "time"})
    return df

def calculate_rsi(series, period=14):
    """
    Beregner RSI (Relative Strength Index) for en prisserie.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Beregner MACD (Moving Average Convergence Divergence).
    """
    df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['MACD_hist_change'] = df['MACD_hist'].diff()  # Rate of change of histogram
    return df

def calculate_atr(df, period=14):
    """
    Beregner Average True Range (ATR) for volatilitetsmåling.
    """
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR as percentage of price
    return df

def calculate_bollinger_bands(df, window=20, std_dev=2):
    """
    Beregner Bollinger Bands.
    """
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    return df

def calculate_momentum_indicators(df):
    """
    Beregner forskellige momentum-indikatorer.
    """
    # Price momentum over different periods
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(periods=period) * 100
    
    # Rate of change of RSI
    df['rsi_change'] = df['RSI'].diff()
    
    # RSI divergence (price up but RSI down, or vice versa)
    df['price_direction'] = np.sign(df['close'].diff())
    df['rsi_direction'] = np.sign(df['RSI'].diff())
    df['rsi_divergence'] = df.apply(
        lambda x: 1 if (x['price_direction'] > 0 and x['rsi_direction'] < 0) else 
                 -1 if (x['price_direction'] < 0 and x['rsi_direction'] > 0) else 0, 
        axis=1
    )
    
    # Candle size relative to recent candles
    df['candle_size'] = abs(df['close'] - df['open'])
    df['avg_candle_size'] = df['candle_size'].rolling(window=10).mean()
    df['rel_candle_size'] = df['candle_size'] / df['avg_candle_size']
    
    return df

def identify_market_regime(df, window=20):
    """
    Identificerer markedsregime (trending eller ranging).
    """
    # ADX (Average Directional Index) for trend strength
    # High ADX = trending, Low ADX = ranging
    try:
        # Convert to numpy arrays for talib
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate ADX using talib
        if len(close) > window + 10:  # Need enough data for calculation
            df['adx'] = talib.ADX(high, low, close, timeperiod=window)
        else:
            df['adx'] = 25  # Default middle value
    except Exception as e:
        print(f"Error calculating ADX: {e}")
        df['adx'] = 25  # Default if calculation fails
    
    # Classify market regime
    df['market_regime'] = df['adx'].apply(
        lambda x: 'trending' if x > 25 else ('ranging' if x <= 25 else 'unknown')
    )
    
    # Numerical representation for ML
    df['is_trending'] = df['adx'].apply(lambda x: 1 if x > 25 else 0)
    
    return df

def calculate_volume_indicators(df):
    """
    Beregner volume-baserede indikatorer hvis volume er tilgængelig.
    """
    if 'volume' in df.columns and not (df['volume'] == 0).all():
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['rel_volume'] = df['volume'] / df['volume_ma']
        
        # On-balance volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume weighted average price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Price-volume trend
        df['pvt'] = df['close'].pct_change() * df['volume']
        df['pvt'] = df['pvt'].fillna(0).cumsum()
    else:
        # Fill with zeros if volume not available
        df['volume_ma'] = 0
        df['rel_volume'] = 1
        df['obv'] = 0
        df['vwap'] = df['close']
        df['pvt'] = 0
    
    return df

def calculate_indicators(df):
    """
    Beregner alle tekniske indikatorer for et dataframe.
    """
    # Basic indicators
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()
    df['SMA200'] = df['close'].rolling(window=200).mean() if len(df) > 200 else df['close']
    df['RSI'] = calculate_rsi(df['close'], period=14)
    df['Support'] = df['low'].rolling(window=20).min()
    df['Resistance'] = df['high'].rolling(window=20).max()
    
    # Trend indicators
    df = calculate_macd(df)
    
    # Volatility indicators
    df['volatility'] = df['high'] - df['low']
    df['volatility_pct'] = df['volatility'] / df['close'] * 100
    df = calculate_atr(df)
    df = calculate_bollinger_bands(df)
    
    # Momentum indicators
    df = calculate_momentum_indicators(df)
    
    # Market regime
    df = identify_market_regime(df)
    
    # Volume indicators (if available)
    df = calculate_volume_indicators(df)
    
    # Trend strength
    df['trend_strength'] = abs(df['close'].diff(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min())
    
    # Additional derived features
    df['sma_cross'] = (df['SMA20'] > df['SMA50']).astype(int)
    df['price_to_sma_ratio'] = df['close'] / df['SMA20']
    
    # Fill NaN values that might occur during calculations
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df

def get_multi_timeframe_data(pair):
    """
    Henter data for flere timeframes og returnerer dem som en dictionary.
    """
    timeframes_data = {}
    
    for tf in TIMEFRAMES:
        try:
            # Adjust outputsize based on timeframe to get enough history
            if tf == "15min":
                outputsize = "100"
            elif tf == "1h":
                outputsize = "150"
            elif tf == "4h":
                outputsize = "200"
            else:  # 1day
                outputsize = "250"
                
            df = get_historical_data(pair, interval=tf, outputsize=outputsize)
            if not df.empty:
                df = calculate_indicators_forward_only(df)
                timeframes_data[tf] = df
            time.sleep(API_CALL_DELAY)  # Respect API limits
        except Exception as e:
            print(f"Error getting data for {pair} on {tf} timeframe: {e}")
            timeframes_data[tf] = None
    
    return timeframes_data

def analyze_timeframe_confluence(timeframes_data):
    """
    Analyser samstemmighed på tværs af timeframes.
    """
    if not timeframes_data or len(timeframes_data) < 2:
        return {"signal": None, "strength": 0, "reason": "Insufficient timeframe data"}
    
    signals = {}
    reasons = {}
    
    for tf, df in timeframes_data.items():
        if df is None or df.empty:
            signals[tf] = "NEUTRAL"
            reasons[tf] = "No data"
            continue
        
        latest = df.iloc[-1]
        
        # Basic trend analysis
        if latest['close'] > latest['SMA20'] and latest['RSI'] < 70:
            signals[tf] = "BUY"
            reasons[tf] = f"Price above SMA20 ({latest['close']:.5f} > {latest['SMA20']:.5f}) with moderate RSI ({latest['RSI']:.1f})"
        elif latest['close'] < latest['SMA20'] and latest['RSI'] > 30:
            signals[tf] = "SELL"
            reasons[tf] = f"Price below SMA20 ({latest['close']:.5f} < {latest['SMA20']:.5f}) with moderate RSI ({latest['RSI']:.1f})"
        else:
            signals[tf] = "NEUTRAL"
            reasons[tf] = f"No clear signal (RSI: {latest['RSI']:.1f}, Price/SMA20: {latest['close']:.5f}/{latest['SMA20']:.5f})"
    
    # Count signals
    buy_count = sum(1 for s in signals.values() if s == "BUY")
    sell_count = sum(1 for s in signals.values() if s == "SELL")
    neutral_count = sum(1 for s in signals.values() if s == "NEUTRAL")
    
    # Weight timeframes (longer timeframes have more weight)
    weighted_score = 0
    weights = {"15min": 1, "1h": 2, "4h": 3, "1day": 4}
    
    for tf, signal in signals.items():
        if signal == "BUY":
            weighted_score += weights.get(tf, 1)
        elif signal == "SELL":
            weighted_score -= weights.get(tf, 1)
    
    # Determine overall signal and strength
    overall_signal = None
    if weighted_score > 0:
        overall_signal = "BUY"
        strength = min(100, (weighted_score / sum(weights.values())) * 100)
    elif weighted_score < 0:
        overall_signal = "SELL"
        strength = min(100, (abs(weighted_score) / sum(weights.values())) * 100)
    else:
        overall_signal = "NEUTRAL"
        strength = 0
    
    # Format reasons
    signal_summary = ", ".join([f"{tf}: {signals[tf]}" for tf in TIMEFRAMES if tf in signals])
    detailed_reasons = "\n".join([f"{tf}: {reasons[tf]}" for tf in TIMEFRAMES if tf in reasons])
    
    return {
        "signal": overall_signal,
        "strength": strength,
        "reason": f"Multi-timeframe analysis: {signal_summary}",
        "details": detailed_reasons,
        "timeframe_signals": signals
    }

def evaluate_trade_signal(trade, df):
    """
    Evaluerer trade signal baseret på indikatorer og giver et score for hver (1-10).
    Returnerer en samlet "sikkerhedsprocent" (max 95%) og en dict med de enkelte scorer.
    """
    latest = df.iloc[-1]
    price = latest['close']
    sma = latest['SMA20']
    
    # SMA Score: Hvor meget prisen afviger fra SMA20 (for BUY skal prisen være over, for SELL under)
    if trade['signal'] == "BUY":
        diff = price - sma
    else:
        diff = sma - price
    # Hvis afvigelsen er ca. 2% af SMA, så opnås top-score
    sma_score = 5 + 5 * (diff / (sma * 0.02))
    sma_score = max(1, min(sma_score, 10))
    
    # RSI Score: For BUY er lavere RSI bedre (men ikke ekstremt), for SELL omvendt
    rsi = latest['RSI']
    if trade['signal'] == "BUY":
        if rsi < 50:
            rsi_score = 10
        elif rsi > 70:
            rsi_score = 1
        else:
            rsi_score = 10 - (rsi - 50) * (9/20)
    else:
        if rsi > 50:
            rsi_score = 10
        elif rsi < 30:
            rsi_score = 1
        else:
            rsi_score = 10 - (50 - rsi) * (9/20)
    rsi_score = max(1, min(rsi_score, 10))
    
    # MACD Score: For BUY skal MACD-histogrammet være positivt, for SELL negativt.
    macd_hist = latest.get('MACD_hist', 0)
    if trade['signal'] == "BUY":
        if macd_hist <= 0:
            macd_score = 1
        else:
            macd_score = 1 + 9 * (macd_hist / 0.5)
    else:
        if macd_hist >= 0:
            macd_score = 1
        else:
            macd_score = 1 + 9 * ((-macd_hist) / 0.5)
    macd_score = max(1, min(macd_score, 10))
    
    # Risk/Reward Score: Jo højere ratio, jo bedre
    risk_reward = trade['risk_reward']
    if risk_reward < 1:
        rr_score = 1
    else:
        rr_score = 1 + 9 * ((risk_reward - 1) / (3 - 1))
    rr_score = max(1, min(rr_score, 10))
    
    # Volatility Score: Based on ATR
    atr_pct = latest.get('atr_pct', 1)
    # For moderate volatility (1-2%) we want highest score
    if 1 <= atr_pct <= 2:
        vol_score = 10
    elif atr_pct < 0.5 or atr_pct > 4:
        vol_score = 3  # Too low or too high volatility
    elif atr_pct < 1:
        vol_score = 3 + 7 * ((atr_pct - 0.5) / 0.5)  # Scale up from 0.5% to 1%
    else:  # atr_pct > 2
        vol_score = 10 - 7 * ((atr_pct - 2) / 2)  # Scale down from 2% to 4%
    vol_score = max(1, min(vol_score, 10))
    
    # Market Regime Score: Trending markets are better for directional trades
    adx = latest.get('adx', 25)
    if trade['signal'] in ["BUY", "SELL"]:  # Directional trades prefer trending
        regime_score = 5 + (adx - 25) * 0.2  # Higher ADX = better score
    else:  # Range trades prefer non-trending
        regime_score = 5 - (adx - 25) * 0.2  # Lower ADX = better score
    regime_score = max(1, min(regime_score, 10))
    
    # Bollinger Band Score
    bb_pct = latest.get('bb_pct', 0.5)
    if trade['signal'] == "BUY":
        # For buys, we want price near lower band (bb_pct close to 0)
        bb_score = 10 - bb_pct * 9  # 0 = 10, 1 = 1
    else:
        # For sells, we want price near upper band (bb_pct close to 1)
        bb_score = 1 + bb_pct * 9  # 0 = 1, 1 = 10
    bb_score = max(1, min(bb_score, 10))
    
    # Create detailed scores dictionary
    indicator_details = {
        "SMA": round(sma_score, 1),
        "RSI": round(rsi_score, 1),
        "MACD": round(macd_score, 1),
        "Risk/Reward": round(rr_score, 1),
        "Volatility": round(vol_score, 1),
        "Market Regime": round(regime_score, 1),
        "Bollinger Bands": round(bb_score, 1)
    }
    
    # Calculate weighted average score
    weights = {
        "SMA": 1.0,
        "RSI": 1.0,
        "MACD": 1.0,
        "Risk/Reward": 1.5,
        "Volatility": 0.8,
        "Market Regime": 1.2,
        "Bollinger Bands": 0.8
    }
    
    total_weight = sum(weights.values())
    weighted_score = (
        sma_score * weights["SMA"] +
        rsi_score * weights["RSI"] +
        macd_score * weights["MACD"] +
        rr_score * weights["Risk/Reward"] +
        vol_score * weights["Volatility"] +
        regime_score * weights["Market Regime"] +
        bb_score * weights["Bollinger Bands"]
    ) / total_weight
    
    confidence_percent = weighted_score * 10  # Scale to percentage
    confidence_percent = min(confidence_percent, 95)  # Cap at 95%
    
    return confidence_percent, indicator_details

def calculate_adaptive_stop_loss(df, signal, entry_price, base_risk_pct=1.0):
    """
    Beregner adaptiv stop loss baseret på ATR og market regime.
    """
    if df.empty:
        return entry_price * 0.99 if signal == "BUY" else entry_price * 1.01
    
    latest = df.iloc[-1]
    atr = latest.get('atr', entry_price * 0.01)  # Default to 1% if ATR not available
    
    # Adjust ATR multiplier based on market regime
    is_trending = latest.get('is_trending', 0)
    adx = latest.get('adx', 25)
    
    # Higher ATR multiplier for trending markets, lower for ranging
    if is_trending:
        # Scale from 2x to 3x ATR as ADX increases
        atr_multiplier = 2 + (adx - 25) / 25
    else:
        # Scale from 1.5x to 2x ATR
        atr_multiplier = 1.5 + (adx / 50)
    
    atr_multiplier = max(1.5, min(3.0, atr_multiplier))  # Cap between 1.5 and 3.0
    
    # Apply multiplier to ATR
    stop_distance = atr * atr_multiplier
    
    # Ensure minimum risk percentage
    min_distance = entry_price * (base_risk_pct / 100)
    stop_distance = max(stop_distance, min_distance)
    
    # Calculate stop loss level based on signal
    if signal == "BUY":
        stop_loss = entry_price - stop_distance
    else:  # SELL
        stop_loss = entry_price + stop_distance
    
    return stop_loss

def calculate_dynamic_position_size(confidence, volatility, balance=BASE_RISK_AMOUNT, base_risk_pct=1.0, max_risk_pct=3.0):
    """
    Beregner dynamisk positionsstørrelse baseret på confidence og volatilitet.
    """
    # Scale risk percentage based on confidence
    # At 95% confidence, use max_risk_pct. At 50% confidence, use base_risk_pct
    if confidence < 50:
        return 0  # Don't trade with confidence < 50%
    
    confidence_factor = (confidence - 50) / 45  # 0 at 50% confidence, 1 at 95% confidence
    risk_pct = base_risk_pct + confidence_factor * (max_risk_pct - base_risk_pct)
    
    # Adjust for volatility (reduce position size in high volatility)
    # Expect volatility as a percentage value where 1-2% is normal
    if volatility <= 1.0:
        volatility_factor = 1.0  # Normal or low volatility
    else:
        # Decrease position as volatility increases
        volatility_factor = max(0.5, 1 - (volatility - 1) / 3)
    
    # Calculate risk amount
    risk_amount = balance * (risk_pct / 100) * volatility_factor
    
    # Ensure risk amount is within limits
    risk_amount = max(balance * 0.005, min(risk_amount, balance * 0.05))  
    
    return risk_amount

def decide_trade_optimized(df, multi_timeframe_data=None, ml_model=None, scaler=None):
    """
    Optimized version of decide_trade that uses profit-based metrics and optimal threshold.
    """
    if df.empty:
        return None
    
    latest = df.iloc[-1]
    trade_signal = None
    reason = ""
    ml_adjustment = 0
    
    # Load optimal threshold if available
    optimal_threshold = 0.5
    try:
        if os.path.exists("ml_optimal_threshold.txt"):
            with open("ml_optimal_threshold.txt", "r") as f:
                optimal_threshold = float(f.read().strip())
    except:
        pass
    
    # Load optimized strategy parameters if available
    optimized_params = {
        'rsi_buy_threshold': 30,
        'rsi_sell_threshold': 70,
        'min_strength_threshold': 30,
        'atr_multiplier_trending': 2.5,
        'atr_multiplier_ranging': 1.5,
        'min_reward_ratio': 1.5
    }
    
    try:
        if os.path.exists("optimized_strategy_params.json"):
            with open("optimized_strategy_params.json", "r") as f:
                loaded_params = json.load(f)
                optimized_params.update(loaded_params)
    except:
        pass
    
    # If multi-timeframe data is available, use it to confirm signals
    if multi_timeframe_data:
        mtf_analysis = analyze_timeframe_confluence(multi_timeframe_data)
        trade_signal = mtf_analysis['signal']
        reason = mtf_analysis['reason']
        
        # If signal is neutral or weak, don't trade
        if trade_signal == "NEUTRAL" or mtf_analysis['strength'] < optimized_params['min_strength_threshold']:
            return None
    else:
        # Fallback to single timeframe with optimized parameters
        if latest['close'] > latest['SMA20'] and latest['RSI'] < optimized_params['rsi_sell_threshold']:
            trade_signal = "BUY"
            reason = "Pris over SMA20 med moderat RSI indikerer opadgående momentum."
        elif latest['close'] < latest['SMA20'] and latest['RSI'] > optimized_params['rsi_buy_threshold']:
            trade_signal = "SELL"
            reason = "Pris under SMA20 med moderat RSI indikerer nedadgående momentum."
    
    if trade_signal:
        # MACD confirmation
        macd_hist = latest.get('MACD_hist', 0)
        if trade_signal == "BUY" and macd_hist <= 0:
            return None
        if trade_signal == "SELL" and macd_hist >= 0:
            return None

        entry = latest['close']
        
        # Market regime detection for adaptive stop loss
        is_trending = latest.get('is_trending', 0)
        adx = latest.get('adx', 25)
        
        # Use ATR-based stop loss with optimized multipliers
        atr = latest.get('atr', entry * 0.01)
        atr_multiplier = (optimized_params['atr_multiplier_trending'] if is_trending else 
                          optimized_params['atr_multiplier_ranging'])
        
        stop_distance = atr * atr_multiplier
        
        # Calculate stop loss
        if trade_signal == "BUY":
            stoploss = entry - stop_distance
        else:
            stoploss = entry + stop_distance
        
        # Calculate take profit with optimized reward ratio
        risk = abs(entry - stoploss)
        reward_ratio = optimized_params['min_reward_ratio']
        take_profit = entry + (risk * reward_ratio) if trade_signal == "BUY" else entry - (risk * reward_ratio)
        
        risk_reward = reward_ratio
        
        # Dynamic position sizing based on volatility and confidence
        volatility = latest.get('atr_pct', 1.0)
        
        # If we have an ML model, use it to assess probability of success
        ml_success_prob = 0.5
        if ml_model is not None and scaler is not None:
            try:
                # Build feature vector for ML prediction
                features = extract_features_for_prediction(df, trade_signal)
                if features is not None:
                    features_scaled = scaler.transform([features])
                    ml_success_prob = ml_model.predict_proba(features_scaled)[0][1]
                    
                    # Calculate adjustment based on probability difference from 0.5
                    ml_adjustment = (ml_success_prob - 0.5) * 40  # Scale to ±20%
                    
                    # Only proceed if ML probability exceeds optimal threshold
                    if ml_success_prob < optimal_threshold:
                        reason += f" ML-model forudsiger lav succesrate ({ml_success_prob:.2f} < {optimal_threshold:.2f})."
                        return None
                    
                    if ml_success_prob > 0.7:
                        reason += f" ML-model forudser meget høj succesrate ({ml_success_prob:.2f})."
                    elif ml_success_prob > optimal_threshold:
                        reason += f" ML-model forudser god succesrate ({ml_success_prob:.2f})."
            except Exception as e:
                print(f"Fejl ved ML-forudsigelse: {e}")
                ml_adjustment = 0
        
        # Calculate provisional confidence
        provisional_confidence, indicator_details = evaluate_trade_signal({
            'signal': trade_signal,
            'entry': entry,
            'stoploss': stoploss,
            'take_profit': take_profit,
            'risk_reward': risk_reward
        }, df)
        
        # Dynamic position sizing
        risk_amount = calculate_dynamic_position_size(
            confidence=provisional_confidence + ml_adjustment,
            volatility=volatility,
            balance=BASE_RISK_AMOUNT
        )
        
        # Notional value
        notional = risk_amount * BASE_LEVERAGE
        
        # Expected value calculation (win rate * win amount - loss rate * loss amount)
        expected_value = (ml_success_prob * risk_amount * reward_ratio) - ((1 - ml_success_prob) * risk_amount)
        
        # Only proceed if expected value is positive
        if expected_value <= 0:
            reason += f" Negative forventet værdi ({expected_value:.2f})."
            return None
        
        trade = {
            "instrument": None,  # Set by calling function
            "signal": trade_signal,
            "entry": entry,
            "stoploss": stoploss,
            "take_profit": take_profit,
            "risk_reward": risk_reward,
            "notional": notional,
            "risk_amount": risk_amount,
            "expected_value": expected_value,
            "ml_probability": ml_success_prob,
            "reason": reason,
            "timestamp": latest['time'],
            "status": "open",
            "time_of_day": latest['time'].hour,
            "day_of_week": latest['time'].dayofweek,
            "market_volatility": latest.get('volatility_pct', 0),
            "market_trend_strength": latest.get('trend_strength', 0),
            "atr_pct": latest.get('atr_pct', 1.0),
            "adx": latest.get('adx', 25),
            "is_trending": latest.get('is_trending', 0),
            "rsi": latest.get('RSI', 50),
            "macd_hist": latest.get('MACD_hist', 0),
            "bb_width": latest.get('bb_width', 0),
            "ml_adjustment": ml_adjustment
        }
        
        # Add multi-timeframe data if available
        if multi_timeframe_data:
            trade['mtf_strength'] = mtf_analysis.get('strength', 0)
            trade['mtf_signals'] = str(mtf_analysis.get('timeframe_signals', {}))
        
        # Calculate final confidence score
        confidence, indicator_details = evaluate_trade_signal(trade, df)
        
        # Adjust confidence based on ML model
        trade['confidence'] = min(95, max(5, confidence + ml_adjustment))
        trade['indicator_details'] = indicator_details
        
        return trade
    return None


def run_optimized_system():
    """
    Runs the complete optimization and testing workflow.
    """
    print("Starting trading system optimization...")
    
    # Step 1: Optimize the base trading strategy
    print("\n==== STEP 1: Strategy Optimization ====")
    optimized_params, recommendations, condition_results = analyze_and_optimize_strategy()
    
    # Step 2: Perform walk-forward analysis with optimized strategy
    print("\n==== STEP 2: Walk-Forward Analysis ====")
    test_results = run_out_of_sample_test(
        pairs=FOREX_PAIRS,
        start_date=(datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d"),
        end_date=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    )
    
    # Step 3: Train balanced ML model
    print("\n==== STEP 3: Training Balanced ML Model ====")
    trade_history = load_trade_history()
    
    # Try each balancing method and select the best
    methods = ['smote', 'oversample', 'undersample', 'class_weight']
    best_model = None
    best_scaler = None
    best_method = None
    best_f1 = 0
    
    for method in methods:
        print(f"\nTraining with {method} balancing:")
        model, scaler, result = train_balanced_ml_model(trade_history, balance_method=method)
        
        if model is not None:
            # Extract F1 score from the result
            try:
                f1_line = [line for line in result.split('\n') if "F1:" in line][0]
                f1_score = float(f1_line.split('F1:')[1].split(',')[0].strip())
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = model
                    best_scaler = scaler
                    best_method = method
                    print(f"New best model: {method} with F1 score {f1_score:.4f}")
            except:
                continue
    
    if best_model is not None:
        print(f"\nSaving best model ({best_method} with F1 score {best_f1:.4f})")
        save_ml_model(best_model, best_scaler)
        send_ml_notification(
            f"🚀 Optimized ML model trained using {best_method} balancing with F1 score {best_f1:.4f}.\n\n"
            f"Strategy optimization recommendations:\n" +
            "\n".join([f"- {rec}" for rec in recommendations]),
            color=3066993
        )
    
    # Step 4: Summary and recommendations
    print("\n==== STEP 4: System Recommendations ====")
    print("\nStrategy Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")
    
    print("\nML Model Recommendations:")
    print(f"- Best balancing method: {best_method}")
    print(f"- F1 score: {best_f1:.4f}")
    
    # Print performance expectations
    win_rates = []
    for pair, results_df in test_results.items():
        if "result" in results_df.columns:
            win_rate = (results_df["result"] == "win").mean() * 100
            win_rates.append(win_rate)
            print(f"- {pair} expected win rate: {win_rate:.2f}%")
    
    if win_rates:
        avg_win_rate = sum(win_rates) / len(win_rates)
        print(f"\nAverage expected win rate: {avg_win_rate:.2f}%")
        
        # Calculate recommended risk per trade
        avg_reward_ratio = optimized_params.get('min_reward_ratio', 1.5)
        
        # Kelly formula for optimal bet size (simplified)
        win_prob = avg_win_rate / 100
        loss_prob = 1 - win_prob
        kelly_pct = (win_prob * avg_reward_ratio - loss_prob) / avg_reward_ratio
        
        if kelly_pct > 0:
            # Kelly bet typically too aggressive, use half-Kelly
            half_kelly = kelly_pct / 2
            print(f"Recommended risk per trade (half-Kelly): {half_kelly:.2f}% of account")
        else:
            print("Warning: Negative Kelly criterion. Trading system needs further optimization.")
    
    # Save recommendations to file
    recommendations_file = "system_recommendations.txt"
    with open(recommendations_file, "w") as f:
        f.write("===== TRADING SYSTEM OPTIMIZATION RESULTS =====\n\n")
        f.write("Strategy Recommendations:\n")
        for rec in recommendations:
            f.write(f"- {rec}\n")
        
        f.write("\nML Model Recommendations:\n")
        f.write(f"- Best balancing method: {best_method}\n")
        f.write(f"- F1 score: {best_f1:.4f}\n")
        
        f.write("\nPerformance Expectations:\n")
        for pair, results_df in test_results.items():
            if "result" in results_df.columns:
                win_rate = (results_df["result"] == "win").mean() * 100
                f.write(f"- {pair} expected win rate: {win_rate:.2f}%\n")
        
        if win_rates:
            avg_win_rate = sum(win_rates) / len(win_rates)
            f.write(f"\nAverage expected win rate: {avg_win_rate:.2f}%\n")
            
            if kelly_pct > 0:
                f.write(f"Recommended risk per trade (half-Kelly): {half_kelly:.2f}% of account\n")
            else:
                f.write("Warning: Negative Kelly criterion. Trading system needs further optimization.\n")
    
    print(f"\nRecommendations saved to {recommendations_file}")
    
    return optimized_params, best_model, best_scaler

def generate_trade_graph(df, trade, filename="trade_graph.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['close'], label='Lukkepris')
    plt.plot(df['time'], df['SMA20'], label='SMA20')
    
    # Add Bollinger bands if available
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        plt.plot(df['time'], df['bb_upper'], 'g--', alpha=0.3, label='Upper BB')
        plt.plot(df['time'], df['bb_lower'], 'g--', alpha=0.3, label='Lower BB')
        plt.fill_between(df['time'], df['bb_upper'], df['bb_lower'], color='green', alpha=0.05)
    
    # Marker entry, stop loss og take profit med vandrette linjer
    plt.axhline(y=trade['entry'], color='blue', linestyle=':', label='Entry')
    plt.axhline(y=trade['stoploss'], color='red', linestyle='--', label='Stop Loss (ATR-based)')
    plt.axhline(y=trade['take_profit'], color='green', linestyle='--', label='Take Profit')
    
    # Marker det seneste datapunkt
    latest_time = df.iloc[-1]['time']
    latest_price = df.iloc[-1]['close']
    plt.scatter(latest_time, latest_price, color='black')
    
    plt.title(f"{trade['instrument']} 15-min Chart - {trade['signal']} Signal")
    plt.xlabel("Tid")
    plt.ylabel("Pris")
    plt.legend(loc='upper left')
    
    # Tilføj en annotationsboks med de tekniske indikator-scorer
    if 'indicator_details' in trade:
        details = trade['indicator_details']
        textstr = '\n'.join([f"{k}: {v}" for k, v in details.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
    
    # Add ATR and volatility info
    if 'atr_pct' in trade:
        atr_text = (f"ATR: {trade['atr_pct']:.2f}%\n"
                   f"Position Size: {trade['risk_amount']:.2f}\n"
                   f"Notional: {trade['notional']:.2f}")
        plt.text(0.02, 0.4, atr_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Add multi-timeframe info if available
    if 'mtf_strength' in trade:
        mtf_text = f"MTF Strength: {trade['mtf_strength']:.0f}%"
        plt.text(0.02, 0.3, mtf_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def send_discord_notification(message, file_path=None, instrument=None, confidence=None, trade_signal=None):
    # Vælg webhook URL baseret på instrument (valuta-par)
    webhook_url = DISCORD_WEBHOOKS_BY_PAIR.get(instrument, 
                                               "https://discord.com/api/webhooks/1349470597218041926/n3U9-vNHXo2IRUxMKpuvV_h4uFz6XeEGzaTASy6-dD5xG9tPZ-W6HlDz5fY_J1AvHA5K")
    # Vælg embed farve baseret på trade signal
    if trade_signal == "BUY":
        embed_color = 3066993  # grøn
    elif trade_signal == "SELL":
        embed_color = 15158332  # rød
    else:
        embed_color = 3447003  # blå

    data = {
        "embeds": [{
            "title": "Trade Notifikation",
            "description": message,
            "color": embed_color,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }]
    }
    
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "image/png")}
            response = requests.post(webhook_url, data={"payload_json": json.dumps(data)}, files=files)
    else:
        response = requests.post(webhook_url, json=data)
    
    return response

def send_update_notification(message):
    data = {
        "embeds": [{
            "title": "📊 Samlet Opdatering",
            "description": message,
            "color": 3447003,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }]
    }
    response = requests.post(UPDATE_WEBHOOK_URL, json=data)
    return response

def send_ml_notification(message, color=7506394):
    data = {
        "embeds": [{
            "title": "🧠 Machine Learning Opdatering",
            "description": message,
            "color": color,  # Standard er lilla
            "timestamp": datetime.datetime.utcnow().isoformat()
        }]
    }
    response = requests.post(ML_WEBHOOK_URL, json=data)
    return response

def load_trade_history():
    """
    Load trade history with improved error handling and ensuring all required columns exist.
    """
    # Define all required columns
    required_columns = [
        "timestamp", "instrument", "signal", "entry", "stoploss", "take_profit",
        "risk_reward", "confidence", "notional", "status", "time_of_day", 
        "day_of_week", "market_volatility", "market_trend_strength", "atr_pct",
        "adx", "is_trending", "rsi", "macd_hist", "bb_width", "ml_adjustment",
        "mtf_strength"
    ]
    
    if os.path.exists(TRADE_LOG_FILE):
        try:
            df = pd.read_csv(TRADE_LOG_FILE)
            
            # Convert timestamp to datetime if it exists
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            
            # Ensure all required columns exist
            for col in required_columns:
                if col not in df.columns:
                    if col == "timestamp":
                        df[col] = pd.to_datetime('now')
                    elif col in ["confidence", "risk_reward", "atr_pct", "adx", "rsi"]:
                        df[col] = 0.0  # Default numeric values
                    else:
                        df[col] = 0  # Default for other columns
            
            return df
        except Exception as e:
            print(f"Fejl ved indlæsning af trade historik: {e}")
            # Return empty DataFrame with all required columns
            return pd.DataFrame(columns=required_columns)
    else:
        # Return empty DataFrame with all required columns
        return pd.DataFrame(columns=required_columns)

def save_trade_history(df):
    # Opret en kopi for at undgå at ændre den originale dataframe
    save_df = df.copy()
    
    # Konverter indicator_details til string format hvis det er et dictionary
    if 'indicator_details' in save_df.columns:
        for idx, row in save_df.iterrows():
            if isinstance(row['indicator_details'], dict):
                save_df.at[idx, 'indicator_details'] = str(row['indicator_details'])
    
    # Convert mtf_signals to string if it exists
    if 'mtf_signals' in save_df.columns:
        for idx, row in save_df.iterrows():
            if not isinstance(row['mtf_signals'], str):
                save_df.at[idx, 'mtf_signals'] = str(row['mtf_signals'])
    
    save_df.to_csv(TRADE_LOG_FILE, index=False)

def save_win_rate_to_file(win_rate):
    with open(WIN_RATE_FILE, "w") as f:
        f.write(f"{win_rate:.1f}")

def load_win_rate_from_file():
    if os.path.exists(WIN_RATE_FILE):
        try:
            with open(WIN_RATE_FILE, "r") as f:
                return float(f.read().strip())
        except Exception as e:
            print(f"Fejl ved indlæsning af win rate: {e}")
            return None
    return None

def update_trade_status(trade, current_price):
    now = datetime.datetime.utcnow()
    elapsed = (now - pd.to_datetime(trade['timestamp'])).total_seconds()
    if elapsed > TRADE_TIMEOUT:
        return "loss"
    if trade['signal'] == "BUY":
        if current_price >= trade['take_profit']:
            return "win"
        elif current_price <= trade['stoploss']:
            return "loss"
    elif trade['signal'] == "SELL":
        if current_price <= trade['take_profit']:
            return "win"
        elif current_price >= trade['stoploss']:
            return "loss"
    return "open"

def calculate_win_rate(trade_history):
    closed_trades = trade_history[trade_history["status"] != "open"]
    if len(closed_trades) == 0:
        return 0
    wins = len(closed_trades[closed_trades["status"] == "win"])
    win_rate = wins / len(closed_trades) * 100
    return win_rate

# -----------------------
# MACHINE LEARNING FUNKTIONER
# -----------------------

def extract_features_for_analysis(trade_data, market_data=None):
    """
    Udtræk features fra en trade og markedsdata til brug i ML-modellen.
    Ensures all required features are present.
    """
    features = {}
    
    # Generelle trade-features
    features['time_of_day'] = trade_data.get('time_of_day', 0)
    features['day_of_week'] = trade_data.get('day_of_week', 0)
    features['signal_buy'] = 1 if trade_data.get('signal') == 'BUY' else 0
    features['risk_reward_ratio'] = trade_data.get('risk_reward', 1.0)
    features['confidence'] = trade_data.get('confidence', 50.0)
    
    # Markedsspecifikke features
    features['market_volatility'] = trade_data.get('market_volatility', 0)
    features['market_trend_strength'] = trade_data.get('market_trend_strength', 0)
    features['atr_pct'] = trade_data.get('atr_pct', 1.0)
    features['adx'] = trade_data.get('adx', 25)
    features['is_trending'] = trade_data.get('is_trending', 0)
    features['rsi'] = trade_data.get('rsi', 50)
    features['macd_hist'] = trade_data.get('macd_hist', 0)
    features['bb_width'] = trade_data.get('bb_width', 0)
    
    # Multi-timeframe features if available
    features['mtf_strength'] = trade_data.get('mtf_strength', 0)
    
    # Tekniske indikator-scores - med håndtering af string-værdier
    if 'indicator_details' in trade_data:
        details = trade_data['indicator_details']
        
        # Håndter tilfælde hvor indicator_details er gemt som en string
        if isinstance(details, str):
            try:
                # Forsøg at konvertere string til dictionary
                import ast
                details = ast.literal_eval(details)
            except (SyntaxError, ValueError):
                # Hvis konvertering fejler, brug standardværdier
                details = {}
        
        # Hvis details nu er en dictionary, brug den
        if isinstance(details, dict):
            features['sma_score'] = details.get('SMA', 5.0)
            features['rsi_score'] = details.get('RSI', 5.0)
            features['macd_score'] = details.get('MACD', 5.0)
            features['volatility_score'] = details.get('Volatility', 5.0)
            features['regime_score'] = details.get('Market Regime', 5.0)
            features['bb_score'] = details.get('Bollinger Bands', 5.0)
        else:
            # Ellers brug standardværdier
            features['sma_score'] = 5.0
            features['rsi_score'] = 5.0
            features['macd_score'] = 5.0
            features['volatility_score'] = 5.0
            features['regime_score'] = 5.0
            features['bb_score'] = 5.0
    else:
        # Hvis ingen indicator_details, brug standardværdier
        features['sma_score'] = 5.0
        features['rsi_score'] = 5.0
        features['macd_score'] = 5.0
        features['volatility_score'] = 5.0
        features['regime_score'] = 5.0
        features['bb_score'] = 5.0
    
    return features

def extract_features_for_prediction(df, signal):
    """
    Udtræk features fra aktuelle markedsdata til forudsigelse.
    Ensures the same feature count and order as prepare_ml_data.
    """
    if df.empty or df.shape[0] < 20:
        return None
    
    latest = df.iloc[-1]
    
    # Extract features in exactly the same order and structure as in prepare_ml_data
    features = [
        latest['time'].hour,                        # time_of_day
        latest['time'].dayofweek,                   # day_of_week
        1 if signal == 'BUY' else 0,                # signal_buy
        1.5,                                        # risk_reward_ratio (default)
        50.0,                                       # confidence (placeholder)
        latest.get('volatility_pct', 0),            # market_volatility
        latest.get('trend_strength', 0),            # market_trend_strength
        latest.get('atr_pct', 1.0),                 # atr_pct
        latest.get('adx', 25),                      # adx
        latest.get('is_trending', 0),               # is_trending
        latest.get('RSI', 50),                      # rsi
        latest.get('MACD_hist', 0),                 # macd_hist
        latest.get('bb_width', 0),                  # bb_width
        0,                                          # mtf_strength (placeholder)
        5.0,                                        # sma_score (placeholder)
        5.0,                                        # rsi_score (placeholder)
        5.0,                                        # macd_score (placeholder)
        5.0,                                        # volatility_score (placeholder)
        5.0,                                        # regime_score (placeholder)
        5.0                                         # bb_score (placeholder)
    ]
    
    return features

def prepare_ml_data(trade_history):
    """
    Forbereder data til ML-modeltræning.
    Ensures consistent feature extraction for all trades.
    """
    if trade_history.empty or len(trade_history) < 10:
        return None, None
    
    # Filtrer kun afsluttede trades
    closed_trades = trade_history[trade_history["status"] != "open"].copy()
    if closed_trades.empty or len(closed_trades) < 10:
        return None, None
    
    # Opret features og target
    X = []
    y = []
    
    for _, trade in closed_trades.iterrows():
        try:
            # Udtræk features fra trade
            feature_dict = extract_features_for_analysis(trade)
            
            # Ensure we have all required features with defaults if missing
            required_features = [
                'time_of_day', 'day_of_week', 'signal_buy', 'risk_reward_ratio', 
                'confidence', 'market_volatility', 'market_trend_strength',
                'atr_pct', 'adx', 'is_trending', 'rsi', 'macd_hist', 'bb_width',
                'mtf_strength', 'sma_score', 'rsi_score', 'macd_score',
                'volatility_score', 'regime_score', 'bb_score'
            ]
            
            # Check if all required features exist, add defaults if missing
            for feature in required_features:
                if feature not in feature_dict:
                    if feature == 'signal_buy':
                        feature_dict[feature] = 1 if trade.get('signal') == 'BUY' else 0
                    elif feature in ['risk_reward_ratio', 'atr_pct']:
                        feature_dict[feature] = 1.0
                    elif feature == 'confidence':
                        feature_dict[feature] = 50.0
                    elif feature in ['sma_score', 'rsi_score', 'macd_score', 'volatility_score', 'regime_score', 'bb_score']:
                        feature_dict[feature] = 5.0
                    elif feature == 'adx':
                        feature_dict[feature] = 25
                    elif feature == 'rsi':
                        feature_dict[feature] = 50
                    else:
                        feature_dict[feature] = 0
            
            # Extract features in consistent order
            features = [
                feature_dict['time_of_day'],
                feature_dict['day_of_week'],
                feature_dict['signal_buy'],
                feature_dict['risk_reward_ratio'],
                feature_dict['confidence'],
                feature_dict['market_volatility'],
                feature_dict['market_trend_strength'],
                feature_dict['atr_pct'],
                feature_dict['adx'],
                feature_dict['is_trending'],
                feature_dict['rsi'],
                feature_dict['macd_hist'],
                feature_dict['bb_width'],
                feature_dict['mtf_strength'],
                feature_dict['sma_score'],
                feature_dict['rsi_score'],
                feature_dict['macd_score'],
                feature_dict['volatility_score'],
                feature_dict['regime_score'],
                feature_dict['bb_score']
            ]
            
            X.append(features)
            
            # Target: 1 for win, 0 for loss
            y.append(1 if trade['status'] == 'win' else 0)
            
        except Exception as e:
            print(f"Fejl ved forberedelse af ML-data for trade: {e}")
            continue
    
    # Hvis vi har nok data
    if len(X) >= 10:
        return np.array(X), np.array(y)
    else:
        return None, None

def analyze_losing_trades(trade_history):
    """
    Analyser mønstre i tabende trades for at finde forbedringspunkter.
    """
    if trade_history.empty:
        return "Ingen trade-historik tilgængelig til analyse."
    
    losing_trades = trade_history[trade_history["status"] == "loss"]
    if losing_trades.empty:
        return "Ingen tabende trades at analysere."
    
    insights = []
    
    # Tidspunktsanalyse
    time_group = losing_trades.groupby('time_of_day')['status'].count()
    worst_hours = time_group.nlargest(3).index.tolist()
    if worst_hours:
        insights.append(f"Højeste antal tabende trades er i timerne: {', '.join(map(str, worst_hours))}.")
    
    # Ugedag-analyse
    day_group = losing_trades.groupby('day_of_week')['status'].count()
    worst_days = day_group.nlargest(2).index.tolist()
    day_names = ['Mandag', 'Tirsdag', 'Onsdag', 'Torsdag', 'Fredag', 'Lørdag', 'Søndag']
    if worst_days:
        worst_day_names = [day_names[day] for day in worst_days]
        insights.append(f"Dage med flest tabende trades: {', '.join(worst_day_names)}.")
    
    # Instrument-analyse
    instr_group = losing_trades.groupby('instrument')['status'].count()
    worst_instruments = instr_group.nlargest(2).index.tolist()
    if worst_instruments:
        insights.append(f"Højeste andel af tabende trades er i instrumenterne: {', '.join(worst_instruments)}.")
    
    # Volatilitet og trades
    if 'market_volatility' in losing_trades.columns:
        avg_vol = losing_trades['market_volatility'].mean()
        if avg_vol > 0:
            insights.append(f"Gennemsnitlig markedsvolatilitet i tabende trades: {avg_vol:.4f}%.")
    
    # Trend strength analysis
    if 'market_trend_strength' in losing_trades.columns and 'is_trending' in losing_trades.columns:
        avg_trend = losing_trades['market_trend_strength'].mean()
        trending_pct = (losing_trades['is_trending'].sum() / len(losing_trades)) * 100
        insights.append(f"Gennemsnitlig trend-styrke: {avg_trend:.4f}, {trending_pct:.1f}% af tabende trades var i trending markeder.")
    
    # ADX analysis if available
    if 'adx' in losing_trades.columns:
        avg_adx = losing_trades['adx'].mean()
        insights.append(f"Gennemsnit ADX (trend styrke) i tabende trades: {avg_adx:.1f}.")
    
    # Market regime analysis
    if 'is_trending' in losing_trades.columns:
        trending_losses = losing_trades[losing_trades['is_trending'] == 1]
        ranging_losses = losing_trades[losing_trades['is_trending'] == 0]
        
        if not trending_losses.empty and not ranging_losses.empty:
            trending_loss_rate = (len(trending_losses) / len(losing_trades)) * 100
            insights.append(f"{trending_loss_rate:.1f}% af tabende trades var i trending markeder vs. {100-trending_loss_rate:.1f}% i ranging markeder.")
    
    # Multi-timeframe analysis if available
    if 'mtf_strength' in losing_trades.columns:
        avg_mtf_strength = losing_trades['mtf_strength'].mean()
        insights.append(f"Gennemsnitlig multi-timeframe styrke i tabende trades: {avg_mtf_strength:.1f}%.")
    
    # Confidence-niveauer
    avg_conf = losing_trades['confidence'].mean()
    insights.append(f"Gennemsnitlig confidence for tabende trades: {avg_conf:.1f}%.")
    
    # BUY vs SELL signal analyse
    signal_counts = losing_trades['signal'].value_counts()
    total_losing = len(losing_trades)
    for signal, count in signal_counts.items():
        percent = (count / total_losing) * 100
        insights.append(f"{signal} signaler udgør {percent:.1f}% af tabende trades.")
    
    return "\n".join(insights)

def train_ml_model(trade_history):
    """
    Træner en ML-model på trade-historikken for at forudse vindende trades.
    """
    X, y = prepare_ml_data(trade_history)
    
    if X is None or y is None:
        return None, None, "Ikke nok data til at træne modellen endnu."
    
    # Del data op i trænings- og testdata
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Skaler features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Træn RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluer model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    # Feature importance
    feature_names = [
        'time_of_day', 'day_of_week', 'signal_buy', 'risk_reward_ratio', 
        'confidence', 'market_volatility', 'market_trend_strength',
        'atr_pct', 'adx', 'is_trending', 'rsi', 'macd_hist', 'bb_width',
        'mtf_strength', 'sma_score', 'rsi_score', 'macd_score',
        'volatility_score', 'regime_score', 'bb_score'
    ]
    
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Opret resultattekst
    result = (
        f"ML-model trænet med {len(X_train)} eksempler.\n"
        f"Nøjagtighed: {accuracy:.2f}, Præcision: {precision:.2f}, Recall: {recall:.2f}\n"
        f"Top-5 vigtigste faktorer:\n"
    )
    
    for feat, imp in top_features[:5]:
        result += f"- {feat}: {imp:.3f}\n"
    
    # Registrer forbedringer hvis vi har tidligere modelevalueringer
    ml_log = load_ml_learning_log()
    if not ml_log.empty and len(ml_log) > 0:
        prev_accuracy = ml_log.iloc[-1]['accuracy']
        accuracy_change = accuracy - prev_accuracy
        result += f"\nÆndring i nøjagtighed siden sidste model: {accuracy_change:.3f} "
        result += "↑" if accuracy_change > 0 else "↓"
    
    # Gem feature importance
    save_feature_importance(importance_dict)
    
    # Log ML-træning
    log_ml_learning(accuracy, precision, recall, top_features[:5])
    
    return model, scaler, result

def save_ml_model(model, scaler):
    """
    Gemmer ML-modellen og scaler til disk.
    """
    try:
        if model is not None:
            joblib.dump(model, ML_MODEL_FILE)
        if scaler is not None:
            joblib.dump(scaler, ML_SCALER_FILE)
        return True
    except Exception as e:
        print(f"Fejl ved gemning af ML-model: {e}")
        return False

def load_ml_model():
    """
    Indlæser ML-modellen og scaler fra disk.
    """
    model = None
    scaler = None
    
    try:
        if os.path.exists(ML_MODEL_FILE):
            model = joblib.load(ML_MODEL_FILE)
        if os.path.exists(ML_SCALER_FILE):
            scaler = joblib.load(ML_SCALER_FILE)
    except Exception as e:
        print(f"Fejl ved indlæsning af ML-model: {e}")
    
    return model, scaler

def save_feature_importance(importance_dict):
    """
    Gemmer feature importance til disk.
    """
    try:
        with open(ML_FEATURE_IMPORTANCE_FILE, 'wb') as f:
            pickle.dump(importance_dict, f)
        return True
    except Exception as e:
        print(f"Fejl ved gemning af feature importance: {e}")
        return False

def load_feature_importance():
    """
    Indlæser feature importance fra disk.
    """
    try:
        if os.path.exists(ML_FEATURE_IMPORTANCE_FILE):
            with open(ML_FEATURE_IMPORTANCE_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Fejl ved indlæsning af feature importance: {e}")
    
    return {}

def log_ml_learning(accuracy, precision, recall, top_features):
    """
    Logfører ML-træningsresultater.
    """
    log_df = load_ml_learning_log()
    
    # Konverter top features til streng
    top_features_str = ', '.join([f"{feat}: {imp:.3f}" for feat, imp in top_features])
    
    # Opret ny log-række
    new_log = pd.DataFrame([{
        'timestamp': datetime.datetime.now(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'top_features': top_features_str,
        'train_data_size': len(prepare_ml_data(load_trade_history())[0]) if prepare_ml_data(load_trade_history())[0] is not None else 0
    }])
    
    # Tilføj til log
    log_df = pd.concat([log_df, new_log], ignore_index=True)
    
    # Gem log
    log_df.to_csv(ML_LEARNING_LOG_FILE, index=False)

def load_ml_learning_log():
    """
    Indlæser ML-træningslog.
    """
    if os.path.exists(ML_LEARNING_LOG_FILE):
        try:
            return pd.read_csv(ML_LEARNING_LOG_FILE, parse_dates=['timestamp'])
        except Exception as e:
            print(f"Fejl ved indlæsning af ML-læringslog: {e}")
    
    return pd.DataFrame(columns=['timestamp', 'accuracy', 'precision', 'recall', 'top_features', 'train_data_size'])

def analyze_ml_progress():
    """
    Analyser fremskridt i ML-læring.
    """
    log_df = load_ml_learning_log()
    
    if log_df.empty:
        return "Ingen ML-læringshistorik endnu."
    
    results = []
    
    if len(log_df) > 1:
        # Analyser nøjagtighedsfremskridt
        first_acc = log_df.iloc[0]['accuracy']
        last_acc = log_df.iloc[-1]['accuracy']
        acc_change = last_acc - first_acc
        results.append(f"Nøjagtighedsfremgang: {acc_change:.3f} ({first_acc:.3f} → {last_acc:.3f})")
        
        # Precision and recall trends
        first_prec = log_df.iloc[0]['precision']
        last_prec = log_df.iloc[-1]['precision']
        prec_change = last_prec - first_prec
        results.append(f"Præcisionsfremgang: {prec_change:.3f} ({first_prec:.3f} → {last_prec:.3f})")
        
        first_rec = log_df.iloc[0]['recall']
        last_rec = log_df.iloc[-1]['recall']
        rec_change = last_rec - first_rec
        results.append(f"Recall fremgang: {rec_change:.3f} ({first_rec:.3f} → {last_rec:.3f})")
        
        # Analyse af de vigtigste faktorer over tid
        latest_features = log_df.iloc[-1]['top_features'].split(', ')[0].split(':')[0].strip()
        results.append(f"Nuværende vigtigste faktor: {latest_features}")
        
        # Training size growth
        first_size = log_df.iloc[0]['train_data_size']
        last_size = log_df.iloc[-1]['train_data_size']
        size_growth = last_size - first_size
        results.append(f"Træningsdatavækst: {size_growth} samples ({first_size} → {last_size})")
    
    # Seneste læringsresultat
    latest = log_df.iloc[-1]
    results.append(f"Seneste træning: Nøjagtighed {latest['accuracy']:.3f}, Trænet på {latest['train_data_size']} datapunkter")
    
    # Fremtidige forbedringsområder
    importance_dict = load_feature_importance()
    if importance_dict:
        least_imp_feature = min(importance_dict.items(), key=lambda x: x[1])
        most_imp_feature = max(importance_dict.items(), key=lambda x: x[1])
        results.append(f"Mest indflydelsesrig: {most_imp_feature[0]} ({most_imp_feature[1]:.3f})")
        results.append(f"Mindst indflydelsesrig: {least_imp_feature[0]} ({least_imp_feature[1]:.3f})")
    
    return "\n".join(results)

def check_for_ml_retraining(trade_history):
    """
    Tjekker om ML-modellen bør genoptrænes baseret på ny data.
    """
    # Hent seneste ML-log
    log_df = load_ml_learning_log()
    
    # Hvis vi aldrig har trænet før, eller ikke har nogen closed trades, gå direkte til tjek
    if log_df.empty or trade_history[trade_history['status'] != 'open'].empty:
        return True
    
    # Hent seneste træningsinfo
    last_training = log_df.iloc[-1]
    last_train_size = last_training['train_data_size']
    last_training_time = last_training['timestamp']
    
    # Aktuel antal lukkede trades
    current_closed_trades = len(trade_history[trade_history['status'] != 'open'])
    
    # Tjek om vi har nye tabende trades siden sidst
    recent_losses = trade_history[
        (trade_history['status'] == 'loss') & 
        (pd.to_datetime(trade_history['timestamp']) > pd.to_datetime(last_training_time))
    ]
    
    # Beslut om vi skal gentræne
    # 1. Hvis vi har mindst 5 nye lukkede trades
    # 2. Eller hvis vi har mindst 3 nye tabende trades
    # 3. Eller hvis det er mere end 24 timer siden sidste træning og vi har nye trades
    time_since_training = datetime.datetime.now() - pd.to_datetime(last_training_time)
    
    if current_closed_trades - last_train_size >= 5:
        return True
    if len(recent_losses) >= 3:
        return True
    if time_since_training.total_seconds() > 24*60*60 and current_closed_trades > last_train_size:
        return True
    
    return False

def get_market_context(df):
    """
    Udtræk markedskontekst fra aktuelle data.
    """
    if df is None or df.empty:
        return {}
    
    latest = df.iloc[-1] if df is not None and not df.empty else None
    if latest is None:
        return {}
    
    return {
        'instrument': df.get('instrument', 'unknown'),
        'time': latest.get('time', datetime.datetime.now()),
        'volatility': latest.get('volatility_pct', 0),
        'trend_strength': latest.get('trend_strength', 0),
        'rsi': latest.get('RSI', 50),
        'macd': latest.get('MACD', 0),
        'macd_signal': latest.get('MACD_signal', 0),
        'macd_hist': latest.get('MACD_hist', 0),
        'adx': latest.get('adx', 25),
        'is_trending': latest.get('is_trending', 0),
        'atr_pct': latest.get('atr_pct', 1.0),
        'bb_width': latest.get('bb_width', 0),
        'price': latest.get('close', 0),
        'volume': latest.get('volume', 0)
    }

def log_market_context(context_data):
    """
    Log markedskontekst til fil for senere analyse.
    """
    context_df = pd.DataFrame([context_data])
    
    if os.path.exists(ML_MARKET_CONDITIONS_FILE):
        try:
            existing_df = pd.read_csv(ML_MARKET_CONDITIONS_FILE)
            context_df = pd.concat([existing_df, context_df], ignore_index=True)
        except Exception as e:
            print(f"Fejl ved indlæsning af eksisterende markedskontekst: {e}")
    
    context_df.to_csv(ML_MARKET_CONDITIONS_FILE, index=False)




# Performance Metrics Function

def log_performance_metrics():
    """Logs system performance metrics for monitoring"""
    
    try:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent(interval=1)  # Get CPU usage with 1 second interval
        disk_usage = psutil.disk_usage('/').percent
        
        metrics = {
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "memory_mb": round(memory_usage, 2),
            "cpu_percent": cpu_usage,
            "disk_percent": disk_usage,
            "open_trades": len(trade_history[trade_history["status"] == "open"]) if 'trade_history' in globals() else 0,
            "total_trades": len(trade_history) if 'trade_history' in globals() else 0
        }
        
        # Create metrics directory if it doesn't exist
        os.makedirs("metrics", exist_ok=True)
        metrics_file = os.path.join("metrics", "system_metrics.csv")
        
        # Append to metrics log
        file_exists = os.path.exists(metrics_file)
        with open(metrics_file, "a") as f:
            if not file_exists:
                f.write(",".join(metrics.keys()) + "\n")
            f.write(",".join(map(str, metrics.values())) + "\n")
        
        # Log to console
        print(f"METRICS: Memory: {metrics['memory_mb']}MB, CPU: {metrics['cpu_percent']}%, Disk: {metrics['disk_percent']}%")
        
        # If metrics exceed thresholds, send alert
        if metrics['memory_mb'] > 500 or metrics['disk_percent'] > 90:
            try:
                send_ml_notification(
                    f"⚠️ System Resource Warning\n\n"
                    f"• Memory Usage: {metrics['memory_mb']}MB\n"
                    f"• CPU Usage: {metrics['cpu_percent']}%\n"
                    f"• Disk Usage: {metrics['disk_percent']}%\n\n"
                    f"System may need maintenance soon.",
                    color=16747520  # Orange
                )
            except Exception as e:
                print(f"Failed to send resource warning: {e}")
        
        return metrics
    except Exception as e:
        print(f"Error logging performance metrics: {e}")
        return None


# Auto-Restart Mechanism

def check_and_restart_components():
    """Checks critical components and attempts to restart them if needed"""
    try:
        restart_attempts = 0
        components_restarted = []
        
        # 1. Check ML model
        global ml_model, scaler
        # Check if ml_model exists in globals() first
        if ('ml_model' in globals() and ml_model is None and 
            os.path.exists(ML_MODEL_FILE)):
            print("Attempting to reload ML model...")
            try:
                ml_model, scaler = load_ml_model()
                if ml_model is not None:
                    restart_attempts += 1
                    components_restarted.append("ML model")
                    print("Successfully reloaded ML model")
            except Exception as e:
                print(f"Failed to reload ML model: {e}")
        
        # 2. Check trade history file
        global trade_history
        if trade_history is None or trade_history.empty:
            print("Attempting to reload trade history...")
            try:
                trade_history = load_trade_history()
                if not trade_history.empty:
                    restart_attempts += 1
                    components_restarted.append("Trade history")
                    print("Successfully reloaded trade history")
            except Exception as e:
                print(f"Failed to reload trade history: {e}")
        
        # 3. Check for missing directories
        required_dirs = ["grafer", "metrics"]
        for directory in required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                restart_attempts += 1
                components_restarted.append(f"{directory} directory")
                print(f"Created missing directory: {directory}")
        
        # 4. Check API connectivity
        try:
            # Test API with a simple request
            print("Testing API connectivity...")
            test_pair = FOREX_PAIRS[0]
            params = {
                "symbol": test_pair[:3] + "/" + test_pair[3:],
                "interval": "1day",
                "outputsize": "1",
                "apikey": TWELVE_DATA_API_KEY
            }
            response = requests.get("https://api.twelvedata.com/time_series", params=params)
            if response.status_code != 200:
                print(f"API connectivity issue detected: {response.status_code}")
                # Just log the issue, don't count as restart
            else:
                print("API connectivity confirmed")
        except Exception as e:
            print(f"Failed to test API connectivity: {e}")
        
        # Send notification if components were restarted
        if restart_attempts > 0:
            try:
                components_str = ", ".join(components_restarted)
                send_ml_notification(
                    f"🔄 Auto-recovery system activated\n\n"
                    f"Successfully restarted {restart_attempts} component(s):\n"
                    f"• {components_str}\n\n"
                    f"System continuing normal operation.",
                    color=3447003  # Blue
                )
            except Exception as e:
                print(f"Failed to send auto-restart notification: {e}")
        
        return restart_attempts
    except Exception as e:
        print(f"Error in auto-restart mechanism: {e}")
        return 0


def send_detailed_trade_report(trade, result, pair, current_price):
    try:
        # Calculate profit/loss metrics
        if result == "win":
            pips_gained = abs(trade.get('take_profit', 0) - trade.get('entry', 0))
            profit_pct = (pips_gained / trade.get('entry', 1)) * 100 if trade.get('entry', 0) != 0 else 0
            message = (
                f"✅ **Trade Successful - {pair}**\n"
                f"Signal: {trade.get('signal', 'Unknown')}\n"
                f"Entry: {trade.get('entry', 0):.5f}\n"
                f"Close: {trade.get('take_profit', 0):.5f}\n"
                f"Profit: {profit_pct:.2f}%\n"
                f"Risk/Reward: {trade.get('risk_reward', 0):.2f}\n\n"
                f"**Technical indicators at entry:**\n"
                f"RSI: {trade.get('rsi', 'N/A')}\n"
                f"ADX: {trade.get('adx', 'N/A')}\n"
                f"Market Regime: {'Trending' if trade.get('is_trending', 0) == 1 else 'Ranging'}\n"
                f"MTF Strength: {trade.get('mtf_strength', 'N/A')}%\n\n"
                f"ML Confidence: {trade.get('ml_probability', 0.5)*100:.1f}%"
            )
        else:
            pips_lost = abs(trade.get('stoploss', 0) - trade.get('entry', 0))
            loss_pct = (pips_lost / trade.get('entry', 1)) * 100 if trade.get('entry', 0) != 0 else 0
            message = (
                f"❌ **Trade Stopped - {pair}**\n"
                f"Signal: {trade.get('signal', 'Unknown')}\n"
                f"Entry: {trade.get('entry', 0):.5f}\n"
                f"Close: {trade.get('stoploss', 0):.5f}\n"
                f"Loss: {loss_pct:.2f}%\n"
                f"Initial Confidence: {trade.get('confidence', 0):.0f}%\n\n"
                f"**Technical Analysis:**\n"
                f"RSI: {trade.get('rsi', 'N/A')}\n"
                f"ADX: {trade.get('adx', 'N/A')}\n"
                f"Market Regime: {'Trending' if trade.get('is_trending', 0) == 1 else 'Ranging'}\n\n"
                f"**Learning from this trade will improve future decisions.**"
            )
        
        send_discord_notification(message, instrument=pair, confidence=trade.get('confidence', 0), trade_signal=trade.get('signal', ''))
    except Exception as e:
        print(f"Error sending detailed trade report: {e}")

def send_enhanced_ml_training_report(training_result, progress_analysis):
    try:
        # Extract key metrics
        accuracy = None
        precision = None
        recall = None
        feature_importance = []
        
        # Parse training result
        for line in training_result.split('\n'):
            if "accuracy" in line.lower():
                accuracy = line
            if "precision" in line.lower():
                precision = line
            if "recall" in line.lower():
                recall = line
            if "- " in line and ":" in line and len(line) < 30:
                feature_importance.append(line)
        
        message = (
            f"🧠 **ML Model Training Complete**\n\n"
            f"**Performance Metrics:**\n"
            f"{accuracy}\n{precision}\n{recall}\n\n"
            f"**Key Features:**\n"
            f"{chr(10).join(feature_importance[:5])}\n\n"
            f"**Learning Progress:**\n"
            f"{progress_analysis}\n\n"
            f"**Next training will occur after more trades are completed or if 3+ losing trades are detected.**"
        )
        
        send_ml_notification(message, color=3066993)
    except Exception as e:
        print(f"Error sending enhanced ML training report: {e}")

def get_current_market_insights():
    insights = []
    for pair in FOREX_PAIRS:
        try:
            if pair in active_timeframes_data and "15min" in active_timeframes_data[pair]:
                df = active_timeframes_data[pair]["15min"]
                latest = df.iloc[-1]
                
                regime = "Trending" if latest.get('is_trending', 0) == 1 else "Ranging"
                rsi = latest.get('RSI', 0)
                adx = latest.get('adx', 0)
                
                # Determine market condition
                condition = "Neutral"
                if rsi > 70:
                    condition = "Overbought"
                elif rsi < 30:
                    condition = "Oversold"
                elif adx > 25:
                    if latest['close'] > latest['SMA20']:
                        condition = "Uptrend"
                    else:
                        condition = "Downtrend"
                
                insights.append(f"{pair}: {condition} ({regime}, RSI: {rsi:.1f})")
        except Exception:
            continue
    
    return "\n".join(insights) if insights else "No market data available"

def send_performance_report():
    if 'trade_history' not in globals() or trade_history is None or trade_history.empty:
        return
        
    try:
        # Calculate performance metrics
        win_rate = calculate_win_rate(trade_history)
        
        # Performance by instrument
        instrument_performance = []
        for pair in FOREX_PAIRS:
            pair_trades = trade_history[trade_history["instrument"] == pair]
            pair_closed = pair_trades[pair_trades["status"] != "open"]
            if not pair_closed.empty:
                pair_wins = len(pair_closed[pair_closed["status"] == "win"])
                pair_rate = (pair_wins / len(pair_closed)) * 100
                instrument_performance.append(f"{pair}: {pair_rate:.1f}% ({pair_wins}/{len(pair_closed)})")
        
        # Performance by signal type
        buy_trades = trade_history[(trade_history["signal"] == "BUY") & (trade_history["status"] != "open")]
        sell_trades = trade_history[(trade_history["signal"] == "SELL") & (trade_history["status"] != "open")]
        
        buy_win_rate = 0
        if not buy_trades.empty:
            buy_wins = len(buy_trades[buy_trades["status"] == "win"])
            buy_win_rate = (buy_wins / len(buy_trades)) * 100
            
        sell_win_rate = 0
        if not sell_trades.empty:
            sell_wins = len(sell_trades[sell_trades["status"] == "win"])
            sell_win_rate = (sell_wins / len(sell_trades)) * 100
        
        message = (
            f"📊 **Weekly Performance Report**\n\n"
            f"**Overall Win Rate:** {win_rate:.1f}%\n\n"
            f"**By Instrument:**\n{chr(10).join(instrument_performance)}\n\n"
            f"**By Signal Type:**\n"
            f"BUY: {buy_win_rate:.1f}% ({len(buy_trades[buy_trades['status'] == 'win'])}/{len(buy_trades)})\n"
            f"SELL: {sell_win_rate:.1f}% ({len(sell_trades[sell_trades['status'] == 'win'])}/{len(sell_trades)})\n\n"
            f"**System Health:** Good"
        )
        
        send_update_notification(message)
    except Exception as e:
        print(f"Error sending performance report: {e}")
# heatbeat function

def send_enhanced_heartbeat():
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if datetime.datetime.now().hour % 4 == 0 and datetime.datetime.now().minute < 15:
        try:
            if 'trade_history' in globals() and trade_history is not None:
                open_trades = len(trade_history[trade_history["status"] == "open"])
                total_trades = len(trade_history)
                wins = len(trade_history[trade_history["status"] == "win"])
                losses = len(trade_history[trade_history["status"] == "loss"])
                win_rate = calculate_win_rate(trade_history)
                
                # Get current market insights
                market_insights = get_current_market_insights()
                
                send_ml_notification(
                    f"🔍 System Status Report - {current_time}\n\n"
                    f"• System running normally\n"
                    f"• Open trades: {open_trades}\n"
                    f"• Total trades: {total_trades}\n"
                    f"• Win/Loss: {wins}/{losses}\n"
                    f"• Win rate: {win_rate:.1f}%\n\n"
                    f"**Market Insights:**\n{market_insights}\n\n"
                    f"ML model is {'active' if 'ml_model' in globals() and ml_model is not None else 'not loaded'}"
                )
        except Exception as e:
            print(f"Error sending enhanced heartbeat: {e}")




# -----------------------
# ACCELERATED TRAINING
# -----------------------

def collect_historical_data_for_backtesting():
    """Gradually collects historical data for backtesting while respecting API limits"""
    os.makedirs("historical_data", exist_ok=True)
    
    for pair in FOREX_PAIRS:
        pair_dir = os.path.join("historical_data", pair)
        os.makedirs(pair_dir, exist_ok=True)
        
        for tf in TIMEFRAMES:
            tf_file = os.path.join(pair_dir, f"{tf}.csv")
            
            # Check if we already have data for this pair/timeframe
            if os.path.exists(tf_file):
                print(f"Data for {pair} {tf} already exists, skipping")
                continue
                
            print(f"Collecting historical data for {pair} on {tf} timeframe")
            
            # For backtesting, we need more historical data
            # But we need to break it into smaller requests to respect API limits
            
            all_data = []
            
            # For 15min data, we might need multiple API calls to get enough history
            if tf == "15min":
                # We'll make multiple requests with different end dates to build up history
                end_dates = []
                current_date = datetime.datetime.now()
                
                # Create a series of end dates, each 1000 candles apart
                for i in range(5):  # 5 requests would give about 5000 candles
                    end_dates.append(current_date)
                    # Move back by 1000 candles (roughly)
                    if tf == "15min":
                        current_date = current_date - datetime.timedelta(days=10)  # ~1000 15-min candles
                    elif tf == "1h":
                        current_date = current_date - datetime.timedelta(days=42)  # ~1000 1h candles
                    elif tf == "4h":
                        current_date = current_date - datetime.timedelta(days=167)  # ~1000 4h candles
                    elif tf == "1day":
                        current_date = current_date - datetime.timedelta(days=1000)  # 1000 daily candles
                
                # Now make API calls for each end date segment
                for i, end_date in enumerate(end_dates):
                    try:
                        # Format the end date for the API
                        end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
                        
                        params = {
                            "symbol": pair[:3] + "/" + pair[3:],
                            "interval": tf,
                            "outputsize": "1000",  # Max allowed per request
                            "apikey": TWELVE_DATA_API_KEY,
                            "end_date": end_date_str
                        }
                        
                        response = requests.get("https://api.twelvedata.com/time_series", params=params)
                        data = response.json()
                        
                        if "values" not in data:
                            print(f"Error in API response: {data.get('message', data)}")
                            break
                            
                        df = pd.DataFrame(data["values"])
                        
                        # Process the dataframe
                        df["datetime"] = pd.to_datetime(df["datetime"])
                        for col in ["open", "high", "low", "close"]:
                            df[col] = pd.to_numeric(df[col])
                        df["volume"] = pd.to_numeric(df["volume"]) if "volume" in df.columns else 0
                        
                        # Add to our collection
                        all_data.append(df)
                        
                        print(f"Collected {len(df)} candles for {pair} {tf} (batch {i+1}/{len(end_dates)})")
                        
                        # Sleep to respect API limits
                        time.sleep(API_CALL_DELAY)
                        
                    except Exception as e:
                        print(f"Error collecting data for {pair} {tf}: {e}")
                        time.sleep(API_CALL_DELAY)
                        continue
            else:
                # For higher timeframes, we can usually get enough data in one request
                try:
                    params = {
                        "symbol": pair[:3] + "/" + pair[3:],
                        "interval": tf,
                        "outputsize": "5000",  # Request maximum
                        "apikey": TWELVE_DATA_API_KEY
                    }
                    
                    response = requests.get("https://api.twelvedata.com/time_series", params=params)
                    data = response.json()
                    
                    if "values" not in data:
                        print(f"Error in API response: {data.get('message', data)}")
                        continue
                        
                    df = pd.DataFrame(data["values"])
                    
                    # Process the dataframe
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    for col in ["open", "high", "low", "close"]:
                        df[col] = pd.to_numeric(df[col])
                    df["volume"] = pd.to_numeric(df["volume"]) if "volume" in df.columns else 0
                    
                    all_data.append(df)
                    
                    print(f"Collected {len(df)} candles for {pair} {tf}")
                    
                except Exception as e:
                    print(f"Error collecting data for {pair} {tf}: {e}")
                
                time.sleep(API_CALL_DELAY)
            
            # Combine all the data we collected
            if all_data:
                combined_df = pd.concat(all_data)
                
                # Remove duplicates and sort
                combined_df = combined_df.drop_duplicates(subset=["datetime"])
                combined_df = combined_df.sort_values("datetime")
                
                # Rename datetime to time to match the format used in trading code
                combined_df = combined_df.rename(columns={"datetime": "time"})
                
                # Save to CSV
                combined_df.to_csv(tf_file, index=False)
                print(f"Saved {len(combined_df)} candles of historical data for {pair} {tf}")
            else:
                print(f"No data collected for {pair} {tf}")
    
    print("Historical data collection complete")

def run_backtest_simulation():
    """Run a backtest simulation on historical data to generate training examples"""
    print("Starting backtest simulation...")
    
    # Check if we have historical data
    if not os.path.exists("historical_data"):
        print("No historical data available. Please run collect_historical_data_for_backtesting() first.")
        return
    
    # Prepare to store simulated trades
    simulated_trades = []
    
    for pair in FOREX_PAIRS:
        pair_dir = os.path.join("historical_data", pair)
        
        if not os.path.exists(pair_dir):
            print(f"No historical data for {pair}")
            continue
        
        print(f"Running backtest simulation for {pair}...")
        
        # Load data for all timeframes
        timeframes_data = {}
        for tf in TIMEFRAMES:
            tf_file = os.path.join(pair_dir, f"{tf}.csv")
            if os.path.exists(tf_file):
                try:
                    df = pd.read_csv(tf_file, parse_dates=["time"])
                    # Calculate indicators
                    df = calculate_indicators_forward_only(df)
                    timeframes_data[tf] = df
                    print(f"Loaded {len(df)} candles for {pair} {tf}")
                except Exception as e:
                    print(f"Error loading {tf} data for {pair}: {e}")
        
        # If we don't have data for the primary timeframe, skip this pair
        if "15min" not in timeframes_data or timeframes_data["15min"].empty:
            print(f"Missing primary timeframe data for {pair}")
            continue
        
        # Get the 15min data we'll iterate through
        df_15min = timeframes_data["15min"]
        
        # Simulate going through the data candle by candle
        for i in range(100, len(df_15min) - 20):  # Start after enough data for indicators and leave room to check outcomes
            try:
                # Get data up to this point for each timeframe
                current_timeframes_data = {}
                for tf, df in timeframes_data.items():
                    # Filter data up to the current time point
                    current_time = df_15min.iloc[i]["time"]
                    current_df = df[df["time"] <= current_time].copy()
                    current_timeframes_data[tf] = current_df
                
                # Make a trading decision based on the data available at this point
                trade = decide_trade_optimized(
                    df=current_timeframes_data["15min"],
                    multi_timeframe_data=current_timeframes_data
                )
                
                # If we got a valid trade signal
                if trade:
                    trade["instrument"] = pair
                    
                    # Find out what happened after this trade
                    future_data = df_15min.iloc[i+1:i+20]  # Next 20 candles
                    
                    # Simulate the outcome
                    result = "open"
                    entry_time = df_15min.iloc[i]["time"]
                    
                    for j, future_row in future_data.iterrows():
                        current_price = future_row["close"]
                        
                        # Check if trade would have hit take profit or stop loss
                        if trade["signal"] == "BUY":
                            if current_price >= trade["take_profit"]:
                                result = "win"
                                break
                            elif current_price <= trade["stoploss"]:
                                result = "loss"
                                break
                        else:  # SELL
                            if current_price <= trade["take_profit"]:
                                result = "win"
                                break
                            elif current_price >= trade["stoploss"]:
                                result = "loss"
                                break
                    
                    # If trade didn't hit TP or SL within 20 candles, consider it a timeout loss
                    if result == "open":
                        result = "loss"
                    
                    # Add outcome to the trade
                    trade["status"] = result
                    trade["timestamp"] = entry_time
                    
                    # Store the trade
                    simulated_trades.append(trade)
                    
                    if len(simulated_trades) % 100 == 0:
                        print(f"Simulated {len(simulated_trades)} trades so far ({pair})")
            
            except Exception as e:
                print(f"Error during simulation at candle {i} for {pair}: {e}")
                continue
    
    # Convert to DataFrame
    if simulated_trades:
        simulated_df = pd.DataFrame(simulated_trades)
        
        # Save the simulated trades
        os.makedirs("simulated_trades", exist_ok=True)
        simulated_file = os.path.join("simulated_trades", f"backtest_trades_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
        simulated_df.to_csv(simulated_file, index=False)
        
        print(f"Completed backtest simulation with {len(simulated_trades)} trades")
        print(f"Win rate: {(simulated_df['status'] == 'win').mean() * 100:.2f}%")
        
        # Return the simulated trades for ML training
        return simulated_df
    else:
        print("No trades were generated in the simulation")
        return None


def train_ml_model_from_backtest():
    """Train ML model using backtest simulation data"""
    # Load simulated trades if available
    simulated_file = None
    for file in os.listdir("simulated_trades"):
        if file.startswith("backtest_trades_"):
            simulated_file = os.path.join("simulated_trades", file)
            break
    
    if not simulated_file:
        print("No simulated trade data found. Please run backtest first.")
        return None, None
    
    # Load the data
    try:
        simulated_df = pd.read_csv(simulated_file, parse_dates=["timestamp"])
        print(f"Loaded {len(simulated_df)} simulated trades for training")
        
        # Make sure we have enough data
        if len(simulated_df) < 100:
            print("Not enough simulated trades for effective training")
            return None, None
        
        # Ensure consistent data types 
        for col in simulated_df.columns:
            if simulated_df[col].dtype == 'object' and col not in ['instrument', 'signal', 'status', 'timestamp']:
                try:
                    simulated_df[col] = pd.to_numeric(simulated_df[col])
                except:
                    pass
        
        # Convert the timestamp to components used in features
        simulated_df['time_of_day'] = simulated_df['timestamp'].dt.hour
        simulated_df['day_of_week'] = simulated_df['timestamp'].dt.dayofweek
        
        # Train the ML model on this data
        X, y = prepare_ml_data(simulated_df)
        
        if X is None or y is None:
            print("Error preparing data for ML training")
            return None, None
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        print(f"Backtest model trained with {len(X_train)} examples")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Save feature importance
        feature_names = [
            'time_of_day', 'day_of_week', 'signal_buy', 'risk_reward_ratio', 
            'confidence', 'market_volatility', 'market_trend_strength',
            'atr_pct', 'adx', 'is_trending', 'rsi', 'macd_hist', 'bb_width',
            'mtf_strength', 'sma_score', 'rsi_score', 'macd_score',
            'volatility_score', 'regime_score', 'bb_score'
        ]
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        save_feature_importance(importance_dict)
        
        # Save the model with a backtest prefix to distinguish it
        joblib.dump(model, "backtest_ml_model.pkl")
        joblib.dump(scaler, "backtest_ml_scaler.pkl")
        
        # Log training details
        log_ml_learning(accuracy, precision, recall, 
                        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])
        
        print("Backtest model trained and saved")
        
        return model, scaler
        
    except Exception as e:
        print(f"Error training model from backtest data: {e}")
        return None, None


def accelerated_learning_pipeline():
    """Run the complete accelerated learning pipeline"""
    print("Starting accelerated learning pipeline...")
    
    # Step 1: Collect historical data (if needed)
    if not os.path.exists("historical_data") or len(os.listdir("historical_data")) < len(FOREX_PAIRS):
        print("Historical data not complete. Starting collection...")
        collect_historical_data_for_backtesting()
    else:
        print("Historical data already collected")
    
    # Step 2: Run backtest simulation
    simulated_df = run_backtest_simulation()
    
    # Step 3: Train ML model from backtest results
    if simulated_df is not None and not simulated_df.empty:
        model, scaler = train_ml_model_from_backtest()
        
        if model is not None and scaler is not None:
            # Use this model in your live trading system
            print("Backtest model ready for use in live trading")
            
            # You can optionally copy it to the main model files
            if input("Replace main model with backtest model? (y/n): ").lower() == 'y':
                joblib.dump(model, ML_MODEL_FILE)
                joblib.dump(scaler, ML_SCALER_FILE)
                print("Main model replaced with backtest model")
        else:
            print("Failed to create backtest model")
    else:
        print("No simulated trades to train on")
    
    print("Accelerated learning pipeline complete")

# -----------------------
# HOVEDLOOP
# -----------------------

def main_loop():
    global ml_model, scaler, trade_history, active_timeframes_data
    # Opret nødvendige mapper
    os.makedirs("grafer", exist_ok=True)

    
    print("Starting trading system...")

      # Initial heartbeat at startup
    send_enhanced_heartbeat()  # Add this line
    
    try:
        # Indlæs trade history
        trade_history = load_trade_history()
        print(f"Loaded trade history with {len(trade_history)} entries")
        # Log initial performance metrics
        log_performance_metrics()  # Add this line
        
        # Indlæs eller initialiser ML-model
        ml_model, scaler = load_ml_model()
        if ml_model is not None:
            print("ML model loaded successfully")
        else:
            print("No existing ML model found - will train if enough data available")
        
        # Send opstartsbesked
        try:
            send_ml_notification("📊 Automatisk trading-system med ML-analyse startet!\n\nSystemet vil løbende analysere tabende trades og lære fra dem for at forbedre fremtidige beslutninger.")
            print("Startup notification sent")
        except Exception as e:
            print(f"Error sending startup notification: {e}")
        
        # Ved opstart: opdater åbne trades
        for pair in FOREX_PAIRS:
            open_trades = trade_history[(trade_history["instrument"] == pair) & (trade_history["status"] == "open")]
            if not open_trades.empty:
                print(f"Found {len(open_trades)} open trades for {pair}")
                try:
                    # Hent data fra alle timeframes
                    timeframes_data = get_multi_timeframe_data(pair)
                    
                    # Primær timeframe for trading (15min)
                    df = timeframes_data.get('15min', pd.DataFrame())
                    if df.empty:
                        print(f"No data available for {pair} on 15min timeframe")
                        continue
                    
                    current_price = df.iloc[-1]['close']
                    print(f"Current price for {pair}: {current_price}")
                    
                    # Log markedskontekst
                    try:
                        market_context = get_market_context(df)
                        market_context['instrument'] = pair
                        log_market_context(market_context)
                    except Exception as e:
                        print(f"Error logging market context for {pair}: {e}")
                    
                    for idx, trade in open_trades.iterrows():
                        try:
                            new_status = update_trade_status(trade, current_price)
                            if new_status != "open":
                                trade_history.loc[idx, "status"] = new_status
                                print(f"Updated trade for {pair} at startup to {new_status}.")
                                
                                # Hvis en trade lukkes som tab, analyser den
                                if new_status == "loss":
                                    try:
                                        loss_insights = analyze_losing_trades(trade_history[trade_history["instrument"] == pair])
                                        send_ml_notification(
                                            f"❗ Ny tabende trade identificeret for {pair}.\n\n"
                                            f"Analyse af tabende trades for {pair}:\n{loss_insights}", 
                                            color=15158332
                                        )
                                        
                                        # Tjek om vi bør gentræne vores ML-model
                                        if check_for_ml_retraining(trade_history):
                                            print("Retraining ML model after new loss...")
                                            ml_model, scaler, training_result = train_ml_model_time_based(trade_history)
                                            save_ml_model(ml_model, scaler)
                                            send_ml_notification(
                                                f"🔄 ML-model gentrænet efter nyt tab.\n\n{training_result}", 
                                                color=3447003
                                            )
                                    except Exception as e:
                                        print(f"Error processing loss analysis for {pair}: {e}")
                        except Exception as e:
                            print(f"Error updating trade status for {pair} at index {idx}: {e}")
                except Exception as e:
                    print(f"Error processing open trades for {pair}: {e}")
                time.sleep(API_CALL_DELAY)
        
        # Save updated trade history
        save_trade_history(trade_history)
        print("Trade history saved after processing open trades")
        
        # Initial træning af ML-model hvis vi har nok lukkede trades
        closed_trades = trade_history[trade_history["status"] != "open"]
        if len(closed_trades) >= 10 and (ml_model is None or scaler is None):
            print(f"Training initial ML model with {len(closed_trades)} closed trades...")
            try:
                ml_model, scaler, training_result = train_ml_model_time_based(trade_history)
                save_ml_model(ml_model, scaler)
                send_ml_notification(
                    f"🚀 Initial ML-model trænet.\n\n{training_result}",
                    color=3066993
                )
                print("Initial ML model training completed")
            except Exception as e:
                print(f"Error during initial ML model training: {e}")
        
        # Calculate and save win rate
        win_rate = calculate_win_rate(trade_history)
        save_win_rate_to_file(win_rate)
        print(f"Current Win Rate: {win_rate:.1f}%")
        
        # Main processing loop
        while True:
            try:
                # Send heartbeat at the start of each cycle
                send_enhanced_heartbeat()  # Add this line
                # Check and restart components if needed
                check_and_restart_components()  # Add this line
                for pair in FOREX_PAIRS:
                    print(f"Processing {pair}...")
                    try:
                        # Hent data fra alle timeframes
                        timeframes_data = get_multi_timeframe_data(pair)
                        
                        # Hvis vi ikke kunne hente data for alle timeframes, gå videre
                        if not timeframes_data or '15min' not in timeframes_data:
                            print(f"Could not get data for all timeframes for {pair}")
                            time.sleep(API_CALL_DELAY)
                            continue
                        
                        # Primær timeframe for trading (15min)
                        df = timeframes_data['15min']
                        
                        # Log markedskontekst for løbende analyse
                        try:
                            market_context = get_market_context(df)
                            market_context['instrument'] = pair
                            log_market_context(market_context)
                        except Exception as e:
                            print(f"Error logging market context for {pair}: {e}")
                        
                    except Exception as e:
                        print(f"Data error for {pair}: {e}")
                        time.sleep(API_CALL_DELAY)
                        continue

                    # Hent allerede åbne trades for dette par
                    open_trades = trade_history[(trade_history["instrument"] == pair) & (trade_history["status"] == "open")]
                    
                    # Forsøg at finde et nyt tradeforslag
                    try:
                        trade = decide_trade_optimized(
                            df=df, 
                            multi_timeframe_data=timeframes_data, 
                            ml_model=ml_model, 
                            scaler=scaler
                        )
                    except Exception as e:
                        print(f"Error in decide_trade for {pair}: {e}")
                        trade = None
                    
                    if trade:
                        trade["instrument"] = pair
                        # Hvis trade-confidencen er under minimumstærsklen, så ignorer
                        if trade.get('confidence', 0) < MIN_CONFIDENCE_THRESHOLD:
                            print(f"Trade for {pair} with confidence {trade.get('confidence', 0):.0f}% is below minimum threshold ({MIN_CONFIDENCE_THRESHOLD}%), not posting.")
                        else:
                            # Hvis der allerede er åbne trades for parret, skal den nye trade have højere confidence
                            if not open_trades.empty:
                                max_conf = open_trades["confidence"].max()
                                if trade["confidence"] <= max_conf:
                                    print(f"New trade for {pair} has confidence {trade['confidence']:.0f}% which does not exceed the highest open trade ({max_conf:.0f}%), not posting.")
                                else:
                                    try:
                                        graph_file = os.path.join("grafer", f"{pair}_trade_{int(time.time())}.png")
                                        generate_trade_graph(df, trade, filename=graph_file)
                                        win_rate = calculate_win_rate(trade_history)
                                        
                                        # Create trade message and send notification
                                        ml_info = f"\nML-justering: {trade.get('ml_adjustment', 0):.1f}%" if ml_model is not None else ""
                                        risk_info = f"\nVolatilitet (ATR): {trade.get('atr_pct', 0):.2f}%\nDynamisk risiko: {trade.get('risk_amount', 0):.2f}" if 'atr_pct' in trade else ""
                                        mtf_info = f"\nMulti-timeframe styrke: {trade.get('mtf_strength', 0):.0f}%" if 'mtf_strength' in trade else ""
                                        
                                        message = (
                                            f"**Trade Forslag for {pair}**\n"
                                            f"Signal: {trade['signal']}\n"
                                            f"Entry: {trade['entry']:.5f}\n"
                                            f"Stop Loss: {trade['stoploss']:.5f} (adaptiv)\n"
                                            f"Take Profit: {trade['take_profit']:.5f}\n"
                                            f"Risk/Reward Ratio: {trade['risk_reward']:.2f}\n"
                                            f"Confidence: {trade['confidence']:.0f}%{ml_info}{mtf_info}\n"
                                            f"Notional: {trade['notional']:.2f}{risk_info}\n"
                                            f"Grund: {trade['reason']}\n"
                                            f"Win Rate: {win_rate:.1f}%\n"
                                            f"Timestamp: {trade['timestamp']}\n"
                                            "--------------------------"
                                        )
                                        
                                        send_discord_notification(message, file_path=graph_file, instrument=pair, 
                                                                confidence=trade['confidence'], trade_signal=trade['signal'])
                                        
                                        # Add trade to history
                                        trade_history = pd.concat([trade_history, pd.DataFrame([trade])], ignore_index=True)
                                        save_trade_history(trade_history)
                                        print(f"Sent trade proposal for {pair}")
                                    except Exception as e:
                                        print(f"Error posting trade for {pair}: {e}")
                            else:
                                # Ingen åbne trades – post tradeforslag
                                try:
                                    graph_file = os.path.join("grafer", f"{pair}_trade_{int(time.time())}.png")
                                    generate_trade_graph(df, trade, filename=graph_file)
                                    win_rate = calculate_win_rate(trade_history)
                                    
                                    # Create trade message and send notification
                                    ml_info = f"\nML-justering: {trade.get('ml_adjustment', 0):.1f}%" if ml_model is not None else ""
                                    risk_info = f"\nVolatilitet (ATR): {trade.get('atr_pct', 0):.2f}%\nDynamisk risiko: {trade.get('risk_amount', 0):.2f}" if 'atr_pct' in trade else ""
                                    mtf_info = f"\nMulti-timeframe styrke: {trade.get('mtf_strength', 0):.0f}%" if 'mtf_strength' in trade else ""
                                    
                                    message = (
                                        f"**Trade Forslag for {pair}**\n"
                                        f"Signal: {trade['signal']}\n"
                                        f"Entry: {trade['entry']:.5f}\n"
                                        f"Stop Loss: {trade['stoploss']:.5f} (adaptiv)\n"
                                        f"Take Profit: {trade['take_profit']:.5f}\n"
                                        f"Risk/Reward Ratio: {trade['risk_reward']:.2f}\n"
                                        f"Confidence: {trade['confidence']:.0f}%{ml_info}{mtf_info}\n"
                                        f"Notional: {trade['notional']:.2f}{risk_info}\n"
                                        f"Grund: {trade['reason']}\n"
                                        f"Win Rate: {win_rate:.1f}%\n"
                                        f"Timestamp: {trade['timestamp']}\n"
                                        "--------------------------"
                                    )
                                    
                                    send_discord_notification(message, file_path=graph_file, instrument=pair, 
                                                            confidence=trade['confidence'], trade_signal=trade['signal'])
                                    
                                    # Add trade to history
                                    trade_history = pd.concat([trade_history, pd.DataFrame([trade])], ignore_index=True)
                                    save_trade_history(trade_history)
                                    print(f"Sent trade proposal for {pair}")
                                except Exception as e:
                                    print(f"Error posting trade for {pair}: {e}")

                    # Update status of open trades
                    current_price = df.iloc[-1]['close']
                    if not open_trades.empty:
                        for idx, trade in open_trades.iterrows():
                            try:
                                new_status = update_trade_status(trade, current_price)
                                if new_status != "open":
                                    old_status = trade_history.loc[idx, "status"]
                                    trade_history.loc[idx, "status"] = new_status
                                    save_trade_history(trade_history)
                                    
                                    try:
                                        send_detailed_trade_report(trade, new_status, pair, current_price)
                                    except Exception as e:
                                        print(f"Error sending detailed trade report for {pair}: {e}")
                                        
                                    print(f"Updated trade for {pair} from {old_status} to {new_status}.")
                                    
                                    # Hvis en trade lukkes som tab, analyser den og opdater ML-model
                                    if new_status == "loss" and old_status == "open":
                                        try:
                                            loss_insights = analyze_losing_trades(trade_history[trade_history["instrument"] == pair])
                                            send_ml_notification(
                                                f"❗ Ny tabende trade identificeret for {pair}.\n\n"
                                                f"Analyse af tabende trades for {pair}:\n{loss_insights}", 
                                                color=15158332
                                            )
                                            
                                            # Tjek om vi bør gentræne vores ML-model
                                            if check_for_ml_retraining(trade_history):
                                                print("Retraining ML model after new loss...")
                                                ml_model, scaler, training_result = train_ml_model_time_based(trade_history)
                                                save_ml_model(ml_model, scaler)
                                                progress_analysis = analyze_ml_progress()
                                                send_ml_notification(
                                                    f"🔄 ML-model gentrænet efter nyt tab.\n\n{training_result}\n\n"
                                                    f"Læringsanalyse:\n{progress_analysis}", 
                                                    color=3447003
                                                )
                                        except Exception as e:
                                            print(f"Error in loss analysis or ML retraining for {pair}: {e}")
                            except Exception as e:
                                print(f"Error updating trade status for {pair}: {e}")
                                
                    time.sleep(API_CALL_DELAY)
                    
                    # Update win rate
                    try:
                        win_rate = calculate_win_rate(trade_history)
                        save_win_rate_to_file(win_rate)
                    except Exception as e:
                        print(f"Error calculating win rate: {e}")
                        win_rate = 0
                    print(f"Current Win Rate: {win_rate:.1f}%")

                # Add performance logging at the end of each cycle
                log_performance_metrics()  # Add this line
                # Send a consolidated update via webhook after processing all pairs
                try:
                    num_open = len(trade_history[trade_history["status"] == "open"])
                    num_wins = len(trade_history[trade_history["status"] == "win"])
                    num_losses = len(trade_history[trade_history["status"] == "loss"])
                    win_rate = calculate_win_rate(trade_history)
                    update_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Add ML status to update
                    ml_status = "Not trained yet"
                    if ml_model is not None:
                        ml_log = load_ml_learning_log()
                        if not ml_log.empty:
                            last_train = ml_log.iloc[-1]
                            ml_status = f"Trained ({last_train['accuracy']:.2f} accuracy)"
                    
                    update_message = (
                        f"**Open Trades:** `{num_open}`\n"
                        f"**Winning Trades:** `{num_wins}`\n"
                        f"**Losing Trades:** `{num_losses}`\n"
                        f"**Win Rate:** `{win_rate:.1f}%`\n"
                        f"**ML Model Status:** `{ml_status}`\n"
                        f"**Update Time:** `{update_time} UTC`"
                    )
                    send_update_notification(update_message)
                    print("Sent consolidated update.")
                except Exception as e:
                    print(f"Error sending consolidated update: {e}")
                
                # Periodic ML analysis (every 6 hours)
                try:
                    current_hour = datetime.datetime.now().hour
                    if current_hour % 6 == 0 and datetime.datetime.now().minute < 15:
                        print("Running periodic ML analysis...")
                        # Analyze ML progress
                        progress_analysis = analyze_ml_progress()
                        loss_insights = analyze_losing_trades(trade_history)
                        
                        # Feature importance from latest model
                        importance_dict = load_feature_importance()
                        feature_insights = "No feature importance data yet."
                        if importance_dict:
                            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                            feature_insights = "Feature importance:\n" + "\n".join([f"{feat}: {imp:.3f}" for feat, imp in sorted_features[:10]])
                        
                        send_ml_notification(
                            f"🔍 Periodic ML Analysis Report\n\n"
                            f"**Win Rate:** {win_rate:.1f}%\n\n"
                            f"**ML Learning Progress:**\n{progress_analysis}\n\n"
                            f"**Analysis of Losing Trades:**\n{loss_insights}\n\n"
                            f"**{feature_insights}**", 
                            color=7419530
                        )
                        print("Periodic ML analysis completed and notification sent.")
                except Exception as e:
                    print(f"Error in periodic ML analysis: {e}")
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                # Log the full traceback for debugging
                import traceback
                traceback.print_exc()
                # Try to auto-restart after error
                check_and_restart_components()  # Add this line to recover after errors
                
            # Wait before next cycle
            print(f"Waiting {10} minutes before next cycle...")
            time.sleep(10 * 60)
    except Exception as e:
        print(f"Critical error in main loop: {e}")
        import traceback
        traceback.print_exc()
        # Try to send error notification
        try:
            send_ml_notification(
                f"❌ Critical error in trading system:\n\n{str(e)}\n\nSystem needs restart.",
                color=15158332  # red
            )
        except:
            pass

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Forex Trading System')
    parser.add_argument('--mode', type=str, default='trade',
                        help='Operation mode: trade (default), backtest, test, or optimize')
    parser.add_argument('--pairs', type=str, nargs='+',
                        help='Forex pairs to process (default: all pairs)')
    parser.add_argument('--start', type=str, default='2023-01-01',
                        help='Start date for testing (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-06-30',
                        help='End date for testing (YYYY-MM-DD)')
    parser.add_argument('--balance', type=str, default='smote',
                        help='Class balancing method for ML training: smote, oversample, undersample, or class_weight')
    
    args = parser.parse_args()
    
    if args.mode == 'optimize':
        # Run complete optimization process
        run_optimized_system()
    elif args.mode == 'test':
        # Run out-of-sample testing
        run_out_of_sample_test(
            pairs=args.pairs,
            start_date=args.start,
            end_date=args.end
        )
    elif args.mode == 'train':
        # Train balanced ML model only
        trade_history = load_trade_history()
        model, scaler, result = train_balanced_ml_model(trade_history, balance_method=args.balance)
        if model is not None and scaler is not None:
            save_ml_model(model, scaler)
            print(result)
    elif args.mode == 'trade':
        # Run normal trading mode, using optimized decision function if available
        if os.path.exists("optimized_strategy_params.json"):
            print("Using optimized parameters for trading")
        
        main_loop()
    elif args.mode == 'backtest':
        # Run accelerated learning
        accelerated_learning_pipeline()
    else:
        print(f"Unknown mode: {args.mode}")