import ccxt
import pandas as pd
import requests
import time
import os
from datetime import datetime

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8635777227:AAHC01rMDifi6m8wpHWO3wPIufZ4AIfiqP0")
CHAT_ID = os.environ.get("CHAT_ID", "7511764144")
MIN_CONFIDENCE = 60
MAX_SIGNALS = 10

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_bbands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower

def get_top_coins():
    exchange = ccxt.binance({"timeout": 15000})
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Загружаю список монет...")
    tickers = exchange.fetch_tickers()
    usdt = {k: v for k, v in tickers.items() if k.endswith("/USDT")}
    result = sorted(usdt, key=lambda x: usdt[x].get("quoteVolume", 0), reverse=True)[:50]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Топ {len(result)} монет загружено")
    return result

def analyze(symbol):
    try:
        exchange = ccxt.binance({"timeout": 10000})
        ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=100)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        rsi = calc_rsi(df["close"])
        macd_line, signal_line = calc_macd(df["close"])
        ema9 = calc_ema(df["close"], 9)
        ema21 = calc_ema(df["close"], 21)
        bb_upper, bb_lower = calc_bbands(df["close"])
        last_rsi = rsi.iloc[-1]
        last_macd = macd_line.iloc[-1]
        last_signal = signal_line.iloc[-1]
        last_ema9 = ema9.iloc[-1]
        last_ema21 = ema21.iloc[-1]
        last_bb_upper = bb_upper.iloc[-1]
        last_bb_lower = bb_lower.iloc[-1]
        close = df["close"].iloc[-1]
        bull = bear = 0
        reasons = []
        if last_rsi < 35: bull += 2; reasons.append("RSI перепродан")
        elif last_rsi > 65: bear += 2; reasons.append("RSI перекуплен")
        if last_macd > last_signal: bull += 2; reasons.append("MACD бычий")
        else: bear += 2; reasons.append("MACD медвежий")
        if last_ema9 > last_ema21 and close > last_ema9: bull += 2; reasons.append("EMA бычье выравнивание")
        elif last_ema9 < last_ema21 and close < last_ema9: bear += 2; reasons.append("EMA медвежье выравнивание")
        if close <= last_bb_lower: bull += 1; reasons.append("Цена у нижней BB")
        elif close >= last_bb_upper: bear += 1; reasons.append("Цена у верхней BB")
        total = bull + bear
        if total == 0: return None
        conf = max(bull, bear) / total * 100
        if conf < MIN_CONFIDENCE: return None
        direction = "LONG" if bull > bear else "SHORT"
        sl = round(close * 0.99 if direction == "LONG" else close * 1.01, 4)
        tp = round(close * 1.03 if direction == "LONG" else close * 0.97, 4)
        return {"symbol": symbol, "direction": direction, "entry": close, "sl": sl, "tp": tp, "conf": round(conf, 1), "reasons": reasons[:3]}
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def scan():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Начинаю сканирование рынка...")
    send_telegram("🔍 Сканирую рынок...")
    try:
        coins = get_top_coins()
    except Exception as e:
        print(f"ОШИБКА загрузки монет: {e}")
        send_telegram(f"❌ Ошибка подключения к Binance: {str(e)[:100]}")
        return
    found = []
    for coin in coins:
        s = analyze(coin)
        if s: found.append(s)
        time.sleep(0.3)
    found.sort(key=lambda x: x["conf"], reverse=True)
    for s in found[:MAX_SIGNALS]:
        emoji = "🟢" if s["direction"] == "LONG" else "🔴"
        msg = f"{emoji} <b>{s['direction']} | {s['symbol']}</b>\n"
        msg += f"📍 Вход: <b>{s['entry']}</b>\n"
        msg += f"🛑 Стоп: <b>{s['sl']}</b> (-1%)\n"
        msg += f"🎯 Цель: <b>{s['tp']}</b> (+3%)\n"
        msg += f"💪 Уверенность: <b>{s['conf']}%</b>\n"
        msg += f"📝 {' | '.join(s['reasons'])}\n"
        msg += f"⏰ {datetime.now().strftime('%H:%M %d.%m.%Y')}"
        send_telegram(msg)
        time.sleep(1)
    if not found:
        send_telegram("😴 Сигналов пока нет — рынок спокойный")

print("=== Криптобот запущен (GitHub Actions) ===")
scan()
print("=== Сканирование завершено ===")
