import time
import datetime
import json  # til korrekt JSON-encoding til Discord
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------
# KONFIGURATION
# -----------------------
TWELVE_DATA_API_KEY = "6438f9035dbf40598b7e588cc74e1e8f"  # Få en gratis nøgle fra https://twelvedata.com

# Discord webhooks for forskellige confidence niveauer
DISCORD_WEBHOOK_URL_DEFAULT = "https://discord.com/api/webhooks/1349470597218041926/n3U9-vNHXo2IRUxMKpuvV_h4uFz6XeEGzaTASy6-dD5xG9tPZ-W6HlDz5fY_J1AvHA5K"
DISCORD_WEBHOOK_URL_0_20 = "https://discord.com/api/webhooks/1349494950689247374/1VUmOI93KRYPZ5QujQ-_eEHGb9_IP2JxSUZ_TwxJfJqRNOsIFq9aG6WP-7eTc0zAEIjT"
DISCORD_WEBHOOK_URL_21_40 = "https://discord.com/api/webhooks/1349494939448250491/qKgiAidpnbY3sV9ASu2l983NfF3u2W5A5QFVA1y81Gqbv5hRgtY7eJo_oqEz0cVC5A2R"
DISCORD_WEBHOOK_URL_41_60 = "https://discord.com/api/webhooks/1349494962022121585/fvQ4m1PalnTj0oyiQXNgjiILl9CmHEMPWxvPaZ5X0Uz4FAccqHkwvD1rSnVqXqcvQT_e"
DISCORD_WEBHOOK_URL_61_80 = "https://discord.com/api/webhooks/1349495238678282250/H4ALQvbptpmmUaeAPY23SQFMcIgDLqpvqJtVScjf7mkArWNwEnpY0dvBykWJSLRkfAXX"
DISCORD_WEBHOOK_URL_81_100 = "https://discord.com/api/webhooks/1349495289895190560/ooV-YFivT9YcLEK2B6c6uWraKfhLbqmk_QkBSko4Uyt8G75eoDLKEclVwUmorP6ugJgt"

# Liste af forex-par (brug fx "EURUSD")
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

# Risikobeløb og gearing
RISK_AMOUNT = 100
LEVERAGE = 30

# Timeout for et tradeforslag (5 timer)
TRADE_TIMEOUT = 5 * 60 * 60  # 

# Filer til logning af trade-historik og win rate
TRADE_LOG_FILE = "trade_history.csv"
WIN_RATE_FILE = "win_rate.txt"

# Twelve Data gratis API rate limit (ca. 8 kald/minut)
API_CALL_DELAY = 15  # sekunder mellem API-kald

# -----------------------
# Hjælpefunktioner
# -----------------------

def get_historical_data(pair, interval="15min", outputsize="100"):
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
    df = df.sort_values("datetime")
    df = df.rename(columns={"datetime": "time"})
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9):
    df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df

def calculate_indicators(df):
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['close'], period=14)
    df['Support'] = df['low'].rolling(window=20).min()
    df['Resistance'] = df['high'].rolling(window=20).max()
    df = calculate_macd(df)
    return df

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
    
    indicator_details = {
        "SMA": round(sma_score, 1),
        "RSI": round(rsi_score, 1),
        "MACD": round(macd_score, 1),
        "Risk/Reward": round(rr_score, 1)
    }
    
    total_score = (sma_score + rsi_score + macd_score + rr_score) / 4
    confidence_percent = total_score * 10  # Skaleret til procent (max 100)
    confidence_percent = min(confidence_percent, 95)  # Vi når aldrig 100%
    
    return confidence_percent, indicator_details

def decide_trade(df):
    latest = df.iloc[-1]
    trade_signal = None
    reason = ""
    # Grundlæggende betingelser – disse kan udvides med flere indikatorer senere
    if latest['close'] > latest['SMA20'] and latest['RSI'] < 70:
        trade_signal = "BUY"
        reason = "Pris over SMA20 med moderat RSI indikerer opadgående momentum."
    elif latest['close'] < latest['SMA20'] and latest['RSI'] > 30:
        trade_signal = "SELL"
        reason = "Pris under SMA20 med moderat RSI indikerer nedadgående momentum."
    
    if trade_signal:
        # MACD-bekræftelse
        macd_hist = latest.get('MACD_hist', 0)
        if trade_signal == "BUY" and macd_hist <= 0:
            return None
        if trade_signal == "SELL" and macd_hist >= 0:
            return None

        entry = latest['close']
        if trade_signal == "BUY":
            stoploss = latest['Support']
            take_profit = latest['Resistance']
        else:
            stoploss = latest['Resistance']
            take_profit = latest['Support']

        risk = abs(entry - stoploss)
        reward = abs(take_profit - entry)
        risk_reward = reward / risk if risk != 0 else np.nan
        # Juster take profit for at sikre mindst 1:1
        if risk != 0 and risk_reward < 1:
            if trade_signal == "BUY":
                take_profit = entry + risk
            else:
                take_profit = entry - risk
            reward = abs(take_profit - entry)
            risk_reward = reward / risk

        trade = {
            "instrument": None,  # skal sættes af kaldende funktion
            "signal": trade_signal,
            "entry": entry,
            "stoploss": stoploss,
            "take_profit": take_profit,
            "risk_reward": risk_reward,
            "notional": RISK_AMOUNT * LEVERAGE,
            "reason": reason,
            "timestamp": latest['time'],
            "status": "open"
        }
        confidence, indicator_details = evaluate_trade_signal(trade, df)
        trade['confidence'] = confidence
        trade['indicator_details'] = indicator_details
        return trade
    return None

def generate_trade_graph(df, trade, filename="trade_graph.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['close'], label='Lukkepris')
    plt.plot(df['time'], df['SMA20'], label='SMA20')
    # Marker entry, stop loss og take profit med vandrette linjer
    plt.axhline(y=trade['entry'], color='blue', linestyle=':', label='Entry')
    plt.axhline(y=trade['stoploss'], color='red', linestyle='--', label='Stop Loss')
    plt.axhline(y=trade['take_profit'], color='green', linestyle='--', label='Take Profit')
    # Marker det seneste datapunkt
    latest_time = df.iloc[-1]['time']
    latest_price = df.iloc[-1]['close']
    plt.scatter(latest_time, latest_price, color='black')
    
    plt.title(f"{trade['instrument']} 15-min Chart - {trade['signal']} Signal")
    plt.xlabel("Tid")
    plt.ylabel("Pris")
    plt.legend()
    
    # Tilføj en annotationsboks med de tekniske indikator-scorer
    if 'indicator_details' in trade:
        details = trade['indicator_details']
        textstr = '\n'.join([f"{k}: {v}" for k, v in details.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def send_discord_notification(message, file_path=None, confidence=None, trade_signal=None):
    # Vælg webhook URL baseret på confidence
    webhook_url = DISCORD_WEBHOOK_URL_DEFAULT
    if confidence is not None:
        if confidence <= 20:
            webhook_url = DISCORD_WEBHOOK_URL_0_20
        elif 21 <= confidence <= 40:
            webhook_url = DISCORD_WEBHOOK_URL_21_40
        elif 41 <= confidence <= 60:
            webhook_url = DISCORD_WEBHOOK_URL_41_60
        elif 61 <= confidence <= 80:
            webhook_url = DISCORD_WEBHOOK_URL_61_80
        elif 81 <= confidence <= 100:
            webhook_url = DISCORD_WEBHOOK_URL_81_100

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

def load_trade_history():
    if os.path.exists(TRADE_LOG_FILE):
        try:
            return pd.read_csv(TRADE_LOG_FILE, parse_dates=["timestamp"])
        except Exception as e:
            print(f"Fejl ved indlæsning af trade historik: {e}")
            return pd.DataFrame(columns=["timestamp", "instrument", "signal", "entry", "stoploss", "take_profit",
                                         "risk_reward", "confidence", "notional", "status"])
    else:
        return pd.DataFrame(columns=["timestamp", "instrument", "signal", "entry", "stoploss", "take_profit",
                                     "risk_reward", "confidence", "notional", "status"])

def save_trade_history(df):
    df.to_csv(TRADE_LOG_FILE, index=False)

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
# HOVEDLOOP
# -----------------------

def main_loop():
    trade_history = load_trade_history()
    
    # Ved opstart: opdater åbne trades
    for pair in FOREX_PAIRS:
        open_trades = trade_history[(trade_history["instrument"] == pair) & (trade_history["status"] == "open")]
        if not open_trades.empty:
            try:
                df = get_historical_data(pair)
                current_price = df.iloc[-1]['close']
                for idx, trade in open_trades.iterrows():
                    new_status = update_trade_status(trade, current_price)
                    if new_status != "open":
                        trade_history.loc[idx, "status"] = new_status
                        print(f"Opdateret trade for {pair} ved opstart til {new_status}.")
            except Exception as e:
                print(f"Fejl ved opdatering af åbne trades for {pair}: {e}")
            time.sleep(API_CALL_DELAY)
    save_trade_history(trade_history)
    
    win_rate = calculate_win_rate(trade_history)
    save_win_rate_to_file(win_rate)
    print(f"Indlæst Win Rate: {win_rate:.1f}%")
    
    while True:
        try:
            for pair in FOREX_PAIRS:
                print(f"Behandler {pair}...")
                try:
                    df = get_historical_data(pair)
                    df = calculate_indicators(df)
                except Exception as e:
                    print(f"Datafejl for {pair}: {e}")
                    time.sleep(API_CALL_DELAY)
                    continue

                open_trades = trade_history[(trade_history["instrument"] == pair) & (trade_history["status"] == "open")]
                if open_trades.empty:
                    trade = decide_trade(df)
                    if trade:
                        trade["instrument"] = pair
                        graph_file = f"{pair}_trade_{int(time.time())}.png"
                        generate_trade_graph(df, trade, filename=graph_file)
                        win_rate = calculate_win_rate(trade_history)
                        message = (
                            f"**Trade Forslag for {pair}**\n"
                            f"Signal: {trade['signal']}\n"
                            f"Entry: {trade['entry']:.5f}\n"
                            f"Stop Loss: {trade['stoploss']:.5f}\n"
                            f"Take Profit: {trade['take_profit']:.5f}\n"
                            f"Risk/Reward Ratio: {trade['risk_reward']:.2f}\n"
                            f"Confidence: {trade['confidence']:.0f}%\n"
                            f"Notional (ved 1:{LEVERAGE}): {trade['notional']}\n"
                            f"Grund: {trade['reason']}\n"
                            f"Win Rate: {win_rate:.1f}%\n"
                            f"Timestamp: {trade['timestamp']}\n"
                            "--------------------------"
                        )
                        send_discord_notification(message, file_path=graph_file, confidence=trade['confidence'], trade_signal=trade['signal'])
                        trade_history = pd.concat([trade_history, pd.DataFrame([trade])], ignore_index=True)
                        save_trade_history(trade_history)
                        print(f"Sendt tradeforslag for {pair}")
                else:
                    print(f"Der er allerede en åben trade for {pair}")

                current_price = df.iloc[-1]['close']
                if not open_trades.empty:
                    for idx, trade in open_trades.iterrows():
                        new_status = update_trade_status(trade, current_price)
                        if new_status != "open":
                            trade_history.loc[idx, "status"] = new_status
                            save_trade_history(trade_history)
                            result_message = (
                                f"**Opdatering for {pair}:** Trade {trade['signal']} fra {trade['timestamp']} er nu {new_status.upper()}.\n"
                                f"Entry: {trade['entry']:.5f} | TP: {trade['take_profit']:.5f} | SL: {trade['stoploss']:.5f}\n"
                                f"Aktuel pris: {current_price:.5f}"
                            )
                            send_discord_notification(result_message, confidence=trade['confidence'], trade_signal=trade['signal'])
                            print(f"Opdateret trade for {pair} som {new_status}.")
                time.sleep(API_CALL_DELAY)
                
                win_rate = calculate_win_rate(trade_history)
                save_win_rate_to_file(win_rate)
                print(f"Aktuel Win Rate: {win_rate:.1f}%")
        except Exception as e:
            print(f"Fejl i hovedloekken: {e}")
        time.sleep(10 * 60)

if __name__ == "__main__":
    main_loop()
