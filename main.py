import ccxt
import pandas as pd
import requests
import time
import os
import schedule
from datetime import datetime
import pytz

TELEGRAM_TOKEN = "8635777227:AAHC01rMDifi6m8wpHWO3wPIufZ4AIfiqP0"
CHAT_ID = "1364766466"
MIN_FACTORS = 6
TOTAL_FACTORS = 13
TBILISI_TZ = pytz.timezone('Asia/Tbilisi')
TRADE_START = 8
TRADE_END = 20

# ─────────────── TELEGRAM ───────────────
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram ошибка: {e}")

def is_trading_hours():
    now = datetime.now(TBILISI_TZ)
    return TRADE_START <= now.hour < TRADE_END

# ─────────────── УЛУЧШЕНИЕ 4: FEAR & GREED ───────────────
def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        if r.status_code == 200:
            data = r.json()["data"][0]
            value = int(data["value"])
            name = data["value_classification"]
            return value, name
    except:
        pass
    return 50, "Neutral"

# ─────────────── УЛУЧШЕНИЕ 3: ТРЕНД BTC ───────────────
def get_btc_trend(exchange):
    try:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "4h", limit=50)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        ema21 = df["close"].ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        close = df["close"].iloc[-1]
        if close > ema21 and ema21 > ema50:
            return "bull"
        elif close < ema21 and ema21 < ema50:
            return "bear"
        return "neutral"
    except:
        return "neutral"

# ─────────────── ИНДИКАТОРЫ ───────────────
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
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_bbands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + std_dev * std, sma - std_dev * std

def calc_adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    dm_pos = high.diff()
    dm_neg = -low.diff()
    dm_pos = dm_pos.where((dm_pos > dm_neg) & (dm_pos > 0), 0.0)
    dm_neg = dm_neg.where((dm_neg > dm_pos) & (dm_neg > 0), 0.0)
    atr = tr.ewm(span=period, adjust=False).mean()
    sm_pos = dm_pos.ewm(span=period, adjust=False).mean()
    sm_neg = dm_neg.ewm(span=period, adjust=False).mean()
    di_pos = 100 * sm_pos / atr
    di_neg = 100 * sm_neg / atr
    dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-9)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, di_pos, di_neg

def calc_stochastic(df, k=14, d=3):
    low_min = df["low"].rolling(window=k).min()
    high_max = df["high"].rolling(window=k).max()
    stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
    stoch_d = stoch_k.rolling(window=d).mean()
    return stoch_k, stoch_d

def calc_obv(df):
    direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * df["volume"]).cumsum()

def calc_mfi(df, period=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(window=period).sum()
    neg_sum = neg.rolling(window=period).sum()
    return 100 - (100 / (1 + pos_sum / (neg_sum + 1e-9)))

def calc_cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-9)

# ─────────────── УЛУЧШЕНИЕ 2: ATR-СТОП ───────────────
def calc_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def detect_candles(df):
    bull = bear = 0
    c = df.iloc[-1]
    p = df.iloc[-2]
    body_c = abs(c["close"] - c["open"])
    rng_c = c["high"] - c["low"] + 1e-9
    if p["close"] < p["open"] and c["close"] > c["open"] and c["close"] > p["open"] and c["open"] < p["close"]:
        bull += 1
    if p["close"] > p["open"] and c["close"] < c["open"] and c["close"] < p["open"] and c["open"] > p["close"]:
        bear += 1
    low_shadow = min(c["open"], c["close"]) - c["low"]
    up_shadow  = c["high"] - max(c["open"], c["close"])
    if low_shadow >= 2 * body_c and up_shadow < 0.3 * rng_c and c["close"] >= c["open"]:
        bull += 1
    if up_shadow >= 2 * body_c and low_shadow < 0.3 * rng_c and c["close"] <= c["open"]:
        bear += 1
    if body_c < 0.1 * rng_c:
        if c["close"] < df["close"].tail(10).mean(): bull += 1
        else:                                        bear += 1
    return bull, bear

def get_news_sentiment(symbol):
    coin = symbol.replace("/USDT", "")
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?currencies={coin}&kind=news&public=true"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            results = r.json().get("results", [])[:5]
            if results:
                pos = sum(1 for n in results if n.get("votes", {}).get("positive", 0) > n.get("votes", {}).get("negative", 0))
                neg = sum(1 for n in results if n.get("votes", {}).get("negative", 0) > n.get("votes", {}).get("positive", 0))
                if pos > neg:   return "📈 позитивные", 1
                elif neg > pos: return "📉 негативные", -1
    except:
        pass
    return "➡️ нейтральные", 0

def detect_vol_spike(df, period=20):
    avg   = df["volume"].tail(period).mean()
    last  = df["volume"].iloc[-1]
    ratio = last / (avg + 1e-9)
    return ratio > 1.5, ratio

# ─────────────── АНАЛИЗ ОДНОГО ТАЙМФРЕЙМА ───────────────
def score_timeframe(df):
    if len(df) < 60:
        return None, None, None, None

    close  = df["close"].iloc[-1]
    rsi_s          = calc_rsi(df["close"]).iloc[-1]
    macd_s, sig_s  = calc_macd(df["close"])
    macd_s = macd_s.iloc[-1]; sig_s = sig_s.iloc[-1]
    ema9   = calc_ema(df["close"], 9).iloc[-1]
    ema21  = calc_ema(df["close"], 21).iloc[-1]
    ema50  = calc_ema(df["close"], 50).iloc[-1]
    ema200 = calc_ema(df["close"], 200).iloc[-1]
    bb_up, bb_lo   = calc_bbands(df["close"])
    bb_up = bb_up.iloc[-1]; bb_lo = bb_lo.iloc[-1]
    adx_s, di_p, di_n = calc_adx(df)
    adx_s = adx_s.iloc[-1]; di_p = di_p.iloc[-1]; di_n = di_n.iloc[-1]
    sk, sd         = calc_stochastic(df)
    sk = sk.iloc[-1]; sd = sd.iloc[-1]
    obv_s          = calc_obv(df)
    obv_now = obv_s.iloc[-1]; obv_prev = obv_s.iloc[-6]
    mfi_s  = calc_mfi(df).iloc[-1]
    cci_s  = calc_cci(df).iloc[-1]
    atr_s  = calc_atr(df).iloc[-1]
    vol_spike, vol_ratio = detect_vol_spike(df)
    c_bull, c_bear = detect_candles(df)

    bull = bear = 0
    reasons = []

    if rsi_s < 30:   bull += 1; reasons.append(f"RSI перепродан ({rsi_s:.0f})")
    elif rsi_s > 70: bear += 1; reasons.append(f"RSI перекуплен ({rsi_s:.0f})")
    else:
        if rsi_s < 50: bull += 0.5
        else:          bear += 0.5

    if macd_s > sig_s and macd_s > 0:   bull += 1; reasons.append("MACD бычий")
    elif macd_s < sig_s and macd_s < 0: bear += 1; reasons.append("MACD медвежий")

    if ema9 > ema21 and close > ema9:   bull += 1; reasons.append("EMA 9>21 ↑")
    elif ema9 < ema21 and close < ema9: bear += 1; reasons.append("EMA 9<21 ↓")

    if close > ema50 and close > ema200:   bull += 1; reasons.append("Выше EMA 50/200")
    elif close < ema50 and close < ema200: bear += 1; reasons.append("Ниже EMA 50/200")

    if close <= bb_lo:   bull += 1; reasons.append("Нижняя полоса BB ↑")
    elif close >= bb_up: bear += 1; reasons.append("Верхняя полоса BB ↓")

    if adx_s > 25:
        if di_p > di_n: bull += 1; reasons.append(f"ADX тренд ↑ ({adx_s:.0f})")
        else:           bear += 1; reasons.append(f"ADX тренд ↓ ({adx_s:.0f})")

    if sk < 20 and sk > sd:   bull += 1; reasons.append(f"Stoch перепродан ({sk:.0f})")
    elif sk > 80 and sk < sd: bear += 1; reasons.append(f"Stoch перекуплен ({sk:.0f})")

    if obv_now > obv_prev * 1.02:   bull += 1; reasons.append("OBV растёт ↑")
    elif obv_now < obv_prev * 0.98: bear += 1; reasons.append("OBV падает ↓")

    if mfi_s < 20:   bull += 1; reasons.append(f"MFI перепродан ({mfi_s:.0f})")
    elif mfi_s > 80: bear += 1; reasons.append(f"MFI перекуплен ({mfi_s:.0f})")

    if cci_s < -100:  bull += 1; reasons.append(f"CCI перепродан ({cci_s:.0f})")
    elif cci_s > 100: bear += 1; reasons.append(f"CCI перекуплен ({cci_s:.0f})")

    if vol_spike:
        if close > df["close"].iloc[-2]: bull += 1; reasons.append(f"Объём ×{vol_ratio:.1f} бычий")
        else:                            bear += 1; reasons.append(f"Объём ×{vol_ratio:.1f} медвежий")

    if c_bull > c_bear:   bull += 1; reasons.append("Бычий паттерн свечей")
    elif c_bear > c_bull: bear += 1; reasons.append("Медвежий паттерн свечей")

    return bull, bear, atr_s, reasons

# ─────────────── АНАЛИЗ МОНЕТЫ ───────────────
def analyze(symbol, exchange, btc_trend, fg_value):
    try:
        # 1h данные
        ohlcv_1h = exchange.fetch_ohlcv(symbol, "1h", limit=220)
        df_1h = pd.DataFrame(ohlcv_1h, columns=["ts", "open", "high", "low", "close", "volume"])
        if len(df_1h) < 60:
            return None

        # УЛУЧШЕНИЕ 1: 4h данные для подтверждения
        ohlcv_4h = exchange.fetch_ohlcv(symbol, "4h", limit=100)
        df_4h = pd.DataFrame(ohlcv_4h, columns=["ts", "open", "high", "low", "close", "volume"])

        bull_1h, bear_1h, atr_1h, reasons_1h = score_timeframe(df_1h)
        bull_4h, bear_4h, atr_4h, reasons_4h = score_timeframe(df_4h)

        if bull_1h is None or bull_4h is None:
            return None

        # Направление по 1H (основной таймфрейм)
        dir_1h = "LONG" if bull_1h > bear_1h else "SHORT"
        dir_4h = "LONG" if bull_4h > bear_4h else "SHORT"
        direction = dir_1h

        # 4H совпадает — даём бонус +20% к весу
        if dir_1h == dir_4h:
            bull_total = bull_1h * 0.6 + bull_4h * 0.4
            bear_total = bear_1h * 0.6 + bear_4h * 0.4
        else:
            # 4H не совпадает — берём только 1H, но требуем выше порог
            bull_total = bull_1h
            bear_total = bear_1h

        winning = max(bull_total, bear_total)

        if winning < MIN_FACTORS * 0.6:
            return None

        conf = round(winning / (TOTAL_FACTORS * 0.6) * 100, 1)
        if conf > 99: conf = 99.0

        # BTC тренд и F&G — только информационно, не блокируем сигналы

        close = df_1h["close"].iloc[-1]

        # УЛУЧШЕНИЕ 2: ATR-стоп вместо фиксированного 1%
        atr_pct = atr_1h / close
        sl_pct = min(max(atr_pct * 1.5, 0.008), 0.03)  # от 0.8% до 3%
        tp_pct = sl_pct * 3  # всегда 1:3

        if direction == "LONG":
            sl = round(close * (1 - sl_pct), 8)
            tp = round(close * (1 + tp_pct), 8)
        else:
            sl = round(close * (1 + sl_pct), 8)
            tp = round(close * (1 - tp_pct), 8)

        # Новости
        news_text, news_score = get_news_sentiment(symbol)

        # Топ 5 причин из обоих таймфреймов
        all_reasons = reasons_1h[:3] + [f"4H: {r}" for r in reasons_4h[:2]]

        return {
            "symbol": symbol,
            "direction": direction,
            "entry": close,
            "sl": sl,
            "tp": tp,
            "sl_pct": round(sl_pct * 100, 2),
            "tp_pct": round(tp_pct * 100, 2),
            "conf": conf,
            "reasons": all_reasons[:5],
            "news": news_text,
            "btc_trend": btc_trend,
            "fg": fg_value
        }
    except Exception as e:
        print(f"Ошибка {symbol}: {e}")
        return None

# ─────────────── СКАНИРОВАНИЕ ───────────────
def scan():
    now_tbilisi = datetime.now(TBILISI_TZ)
    if not is_trading_hours():
        print(f"[{now_tbilisi.strftime('%H:%M')} Тбилиси] Нерабочее время, жду 08:00")
        return

    print(f"[{now_tbilisi.strftime('%H:%M')} Тбилиси] Начинаю сканирование...")
    send_telegram("🔍 Сканирую рынок (v3.0 — мультитаймфрейм + ATR + BTC фильтр)...")

    exchange = ccxt.bybit({"timeout": 15000})

    # Получаем общий контекст рынка
    fg_value, fg_name = get_fear_greed()
    btc_trend = get_btc_trend(exchange)
    btc_emoji = "📈" if btc_trend == "bull" else ("📉" if btc_trend == "bear" else "➡️")
    print(f"Fear & Greed: {fg_value} ({fg_name}) | BTC тренд: {btc_trend}")

    try:
        tickers = exchange.fetch_tickers()
        usdt = {k: v for k, v in tickers.items() if k.endswith("/USDT")}
        coins = sorted(usdt, key=lambda x: usdt[x].get("quoteVolume", 0), reverse=True)[:50]
        print(f"Загружено {len(coins)} монет")
    except Exception as e:
        send_telegram(f"❌ Ошибка Binance: {str(e)[:100]}")
        return

    found = []
    for coin in coins:
        s = analyze(coin, exchange, btc_trend, fg_value)
        if s:
            found.append(s)
            print(f"✅ {coin} {s['direction']} {s['conf']}%")
        time.sleep(0.5)

    found.sort(key=lambda x: x["conf"], reverse=True)

    for s in found:
        emoji  = "🟢" if s["direction"] == "LONG" else "🔴"
        dir_ru = "ЛОНГ" if s["direction"] == "LONG" else "ШОРТ"
        t = datetime.now(TBILISI_TZ).strftime("%H:%M %d.%m.%Y")
        msg = (
            f"{emoji} <b>{dir_ru} | {s['symbol']}</b>\n"
            f"📍 Вход: <b>{s['entry']}</b>\n"
            f"🛑 Стоп: <b>{s['sl']}</b> (-{s['sl_pct']}% ATR)\n"
            f"🎯 Цель: <b>{s['tp']}</b> (+{s['tp_pct']}% | 1:3)\n"
            f"💪 Уверенность: <b>{s['conf']}%</b> (1H+4H)\n"
            f"📰 Новости: {s['news']}\n"
            f"📝 Причины: {' | '.join(s['reasons'])}\n"
            f"🌡 Рынок: F&G {s['fg']} | BTC {btc_emoji}\n"
            f"⏰ {t}"
        )
        send_telegram(msg)
        time.sleep(1)

    if not found:
        print("Сигналов нет")
        send_telegram(
            f"😴 Сигналов нет\n"
            f"🌡 F&G: {fg_value} ({fg_name}) | BTC: {btc_emoji}\n"
            f"Следующий скан через 30 мин."
        )
    else:
        print(f"Отправлено {len(found)} сигналов")

# ─────────────── СТАРТ ───────────────
print("=" * 50)
print("  🤖 КриптоБот v3.0 — Professional Edition")
print("=" * 50)
print(f"  ✅ Мультитаймфрейм (1H + 4H)")
print(f"  ✅ ATR-стоп (адаптивный, всегда 1:3)")
print(f"  ✅ Фильтр тренда BTC")
print(f"  ✅ Fear & Greed Index")
print(f"  Порог: {MIN_FACTORS}/{TOTAL_FACTORS} факторов")
print(f"  Время: {TRADE_START}:00 — {TRADE_END}:00 (Тбилиси UTC+4)")
print(f"  Сканирование: каждые 30 минут")
print("=" * 50)

import traceback
import sys

schedule.every(30).minutes.do(scan)
try:
    scan()
except Exception as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА при запуске: {e}", flush=True)
    traceback.print_exc()
    send_telegram(f"❌ Бот упал при запуске: {str(e)[:200]}")
    sys.exit(1)

while True:
    try:
        schedule.run_pending()
    except Exception as e:
        print(f"Ошибка в цикле: {e}", flush=True)
        traceback.print_exc()
    time.sleep(60)
