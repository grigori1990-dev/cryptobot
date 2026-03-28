#!/usr/bin/env python3
"""
КриптоБот v4.0 — DigitalOcean Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ asyncio + ccxt.async_support — параллельный fetch (8 монет одновременно)
✅ numpy вместо pandas — в 3-5х меньше RAM
✅ ВСЕ 50 монет каждые 5 минут (было: 25 монет каждые 10 мин)
✅ Скор ≥ 60% обязателен (было: ~46%)
✅ gc.collect() после каждой монеты и каждого цикла
✅ Bybit API (Binance заблокирован с EU-серверов Frankfurt)
"""

import asyncio
import ccxt.async_support as ccxt_async
import numpy as np
import requests
import gc
import os
import time
import traceback
from datetime import datetime, timedelta
import pytz

# ════════════════════════════════════════════════════════
#  КОНФИГУРАЦИЯ
# ════════════════════════════════════════════════════════
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID        = os.getenv("CHAT_ID", "")
MIN_CONF       = 60.0    # Минимальный скор (%) — СТРОГО
SCAN_INTERVAL  = 300     # 5 минут между циклами
MAX_CONCURRENT = 8       # Параллельных запросов (не перегружаем Bybit)
TRADE_START    = 8       # Начало торговли (UTC+4 Тбилиси)
TRADE_END      = 20      # Конец торговли
TBILISI_TZ     = pytz.timezone('Asia/Tbilisi')
TOTAL_FACTORS  = 12.0    # Всего факторов скоринга

TOP_50 = [
    "BTC/USDT",  "ETH/USDT",   "SOL/USDT",   "BNB/USDT",   "XRP/USDT",
    "DOGE/USDT", "ADA/USDT",   "AVAX/USDT",  "TRX/USDT",   "DOT/USDT",
    "LINK/USDT", "TON/USDT",   "LTC/USDT",   "BCH/USDT",   "UNI/USDT",
    "ATOM/USDT", "XLM/USDT",   "ETC/USDT",   "APT/USDT",   "NEAR/USDT",
    "OP/USDT",   "ARB/USDT",   "SUI/USDT",   "INJ/USDT",   "FTM/USDT",
    "MATIC/USDT","SHIB/USDT",  "FIL/USDT",   "VET/USDT",   "SAND/USDT",
    "MANA/USDT", "GALA/USDT",  "AXS/USDT",   "CHZ/USDT",   "FLOW/USDT",
    "ALGO/USDT", "ICP/USDT",   "THETA/USDT", "AAVE/USDT",  "MKR/USDT",
    "SNX/USDT",  "CRV/USDT",   "COMP/USDT",  "YFI/USDT",   "SUSHI/USDT",
    "1INCH/USDT","ZEC/USDT",   "DASH/USDT",  "XMR/USDT",   "ENJ/USDT",
]

# ════════════════════════════════════════════════════════
#  NUMPY ИНДИКАТОРЫ (без pandas — экономим RAM)
# ════════════════════════════════════════════════════════

def _ewm(arr: np.ndarray, span: int) -> np.ndarray:
    """EWM — аналог pandas .ewm(span=span, adjust=False).mean()"""
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def calc_rsi(close: np.ndarray, period: int = 14) -> float:
    delta    = np.diff(close)
    gain     = np.where(delta > 0, delta, 0.0)
    loss     = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ewm(gain, period)[-1]
    avg_loss = _ewm(loss, period)[-1]
    if avg_loss < 1e-10:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def calc_macd(close: np.ndarray, fast=12, slow=26, sig=9):
    macd   = _ewm(close, fast) - _ewm(close, slow)
    signal = _ewm(macd, sig)
    return macd[-1], signal[-1]


def calc_ema_last(close: np.ndarray, period: int) -> float:
    return _ewm(close, period)[-1]


def calc_bbands(close: np.ndarray, period=20, std_dev=2):
    wins = np.lib.stride_tricks.sliding_window_view(close, period)
    sma  = wins[-1].mean()
    std  = wins[-1].std()
    return sma + std_dev * std, sma - std_dev * std


def calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14):
    if len(close) < period + 5:
        return 25.0, 50.0, 50.0
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    dm_p_raw = high[1:] - high[:-1]
    dm_n_raw = low[:-1]  - low[1:]
    dm_pos = np.where((dm_p_raw > dm_n_raw) & (dm_p_raw > 0), dm_p_raw, 0.0)
    dm_neg = np.where((dm_n_raw > dm_p_raw) & (dm_n_raw > 0), dm_n_raw, 0.0)
    atr_e  = _ewm(tr, period)
    smp    = _ewm(dm_pos, period)
    smn    = _ewm(dm_neg, period)
    di_pos = 100.0 * smp / (atr_e + 1e-9)
    di_neg = 100.0 * smn / (atr_e + 1e-9)
    dx_arr = 100.0 * np.abs(di_pos - di_neg) / (di_pos + di_neg + 1e-9)
    adx    = _ewm(dx_arr, period)[-1]
    return adx, di_pos[-1], di_neg[-1]


def calc_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k=14, d=3):
    if len(close) < k + d:
        return 50.0, 50.0
    wins_lo = np.lib.stride_tricks.sliding_window_view(low,  k)
    wins_hi = np.lib.stride_tricks.sliding_window_view(high, k)
    lo_min  = wins_lo.min(axis=1)
    hi_max  = wins_hi.max(axis=1)
    stoch_k = 100.0 * (close[k - 1:] - lo_min) / (hi_max - lo_min + 1e-9)
    if len(stoch_k) < d:
        return stoch_k[-1], stoch_k[-1]
    stoch_d = np.lib.stride_tricks.sliding_window_view(stoch_k, d).mean(axis=1)
    return stoch_k[-1], stoch_d[-1]


def calc_obv_delta(close: np.ndarray, volume: np.ndarray, lookback=6):
    direction = np.sign(np.diff(close))
    obv = np.concatenate([[0.0], np.cumsum(direction * volume[1:])])
    prev = obv[-1 - lookback] if len(obv) > lookback else obv[0]
    return obv[-1], prev


def calc_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             volume: np.ndarray, period=14) -> float:
    if len(close) < period + 1:
        return 50.0
    tp   = (high + low + close) / 3.0
    mf   = tp * volume
    diff = np.diff(tp)
    pos  = np.where(diff > 0, mf[1:], 0.0)
    neg  = np.where(diff < 0, mf[1:], 0.0)
    ps   = pos[-period:].sum()
    ns   = neg[-period:].sum()
    return 100.0 - 100.0 / (1.0 + ps / (ns + 1e-9))


def calc_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=20) -> float:
    if len(close) < period:
        return 0.0
    tp   = (high + low + close) / 3.0
    tp_w = tp[-period:]
    sma  = tp_w.mean()
    mad  = np.abs(tp_w - sma).mean()
    return (tp[-1] - sma) / (0.015 * mad + 1e-9)


def calc_atr_last(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> float:
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    return _ewm(tr, period)[-1]


def detect_vol_spike(volume: np.ndarray, period=20):
    if len(volume) < period + 1:
        return False, 1.0
    avg   = volume[-period - 1:-1].mean()
    ratio = volume[-1] / (avg + 1e-9)
    return ratio > 1.5, ratio


def detect_candles(open_: np.ndarray, high_: np.ndarray,
                   low_: np.ndarray, close_: np.ndarray):
    bull = bear = 0
    co, ch, cl, cc = open_[-1], high_[-1], low_[-1], close_[-1]
    po, _,  _,  pc = open_[-2], high_[-2], low_[-2], close_[-2]
    body  = abs(cc - co)
    rng   = ch - cl + 1e-9
    # Engulfing
    if pc < po and cc > co and cc > po and co < pc:
        bull += 1
    if pc > po and cc < co and cc < po and co > pc:
        bear += 1
    # Hammer / Shooting star
    lo_sh = min(co, cc) - cl
    up_sh = ch - max(co, cc)
    if lo_sh >= 2 * body and up_sh < 0.3 * rng and cc >= co:
        bull += 1
    if up_sh >= 2 * body and lo_sh < 0.3 * rng and cc <= co:
        bear += 1
    # Doji
    if body < 0.1 * rng:
        if cc < close_[-10:].mean():
            bull += 1
        else:
            bear += 1
    return bull, bear


# ════════════════════════════════════════════════════════
#  СКОРИНГ ОДНОГО ТАЙМФРЕЙМА
# ════════════════════════════════════════════════════════
def score_timeframe(ohlcv_raw: list):
    """
    Принимает сырой список [[ts, open, high, low, close, volume], ...]
    Возвращает (bull, bear, atr, reasons) или (None, None, None, None)
    """
    if len(ohlcv_raw) < 60:
        return None, None, None, None

    data   = np.array(ohlcv_raw, dtype=np.float64)
    open_  = data[:, 1]
    high_  = data[:, 2]
    low_   = data[:, 3]
    close_ = data[:, 4]
    vol_   = data[:, 5]

    rsi_v            = calc_rsi(close_)
    macd_v, sig_v    = calc_macd(close_)
    ema9             = calc_ema_last(close_, 9)
    ema21            = calc_ema_last(close_, 21)
    ema50            = calc_ema_last(close_, 50)
    ema200           = calc_ema_last(close_, min(200, len(close_)))
    bb_up, bb_lo     = calc_bbands(close_)
    adx_v, dip, din  = calc_adx(high_, low_, close_)
    sk, sd           = calc_stochastic(high_, low_, close_)
    obv_now, obv_pr  = calc_obv_delta(close_, vol_)
    mfi_v            = calc_mfi(high_, low_, close_, vol_)
    cci_v            = calc_cci(high_, low_, close_)
    atr_v            = calc_atr_last(high_, low_, close_)
    vol_spike, vrat  = detect_vol_spike(vol_)
    c_bull, c_bear   = detect_candles(open_, high_, low_, close_)
    close_now        = close_[-1]
    close_prev       = close_[-2]

    # Немедленно освобождаем numpy-массивы
    del data, open_, high_, low_, close_, vol_

    bull = 0.0
    bear = 0.0
    reasons = []

    # 1. RSI
    if rsi_v < 30:
        bull += 1; reasons.append(f"RSI перепродан ({rsi_v:.0f})")
    elif rsi_v > 70:
        bear += 1; reasons.append(f"RSI перекуплен ({rsi_v:.0f})")
    else:
        if rsi_v < 50: bull += 0.5
        else:          bear += 0.5

    # 2. MACD
    if macd_v > sig_v and macd_v > 0:
        bull += 1; reasons.append("MACD бычий")
    elif macd_v < sig_v and macd_v < 0:
        bear += 1; reasons.append("MACD медвежий")

    # 3. EMA 9/21
    if ema9 > ema21 and close_now > ema9:
        bull += 1; reasons.append("EMA 9>21 ↑")
    elif ema9 < ema21 and close_now < ema9:
        bear += 1; reasons.append("EMA 9<21 ↓")

    # 4. EMA 50/200
    if close_now > ema50 and close_now > ema200:
        bull += 1; reasons.append("Выше EMA 50/200")
    elif close_now < ema50 and close_now < ema200:
        bear += 1; reasons.append("Ниже EMA 50/200")

    # 5. Bollinger Bands
    if close_now <= bb_lo:
        bull += 1; reasons.append("Нижняя BB ↑")
    elif close_now >= bb_up:
        bear += 1; reasons.append("Верхняя BB ↓")

    # 6. ADX
    if adx_v > 25:
        if dip > din: bull += 1; reasons.append(f"ADX тренд ↑ ({adx_v:.0f})")
        else:         bear += 1; reasons.append(f"ADX тренд ↓ ({adx_v:.0f})")

    # 7. Stochastic
    if sk < 20 and sk > sd:
        bull += 1; reasons.append(f"Stoch перепродан ({sk:.0f})")
    elif sk > 80 and sk < sd:
        bear += 1; reasons.append(f"Stoch перекуплен ({sk:.0f})")

    # 8. OBV
    if obv_now > obv_pr * 1.02:
        bull += 1; reasons.append("OBV растёт ↑")
    elif obv_now < obv_pr * 0.98:
        bear += 1; reasons.append("OBV падает ↓")

    # 9. MFI
    if mfi_v < 20:
        bull += 1; reasons.append(f"MFI перепродан ({mfi_v:.0f})")
    elif mfi_v > 80:
        bear += 1; reasons.append(f"MFI перекуплен ({mfi_v:.0f})")

    # 10. CCI
    if cci_v < -100:
        bull += 1; reasons.append(f"CCI перепродан ({cci_v:.0f})")
    elif cci_v > 100:
        bear += 1; reasons.append(f"CCI перекуплен ({cci_v:.0f})")

    # 11. Объём-спайк
    if vol_spike:
        if close_now > close_prev:
            bull += 1; reasons.append(f"Объём ×{vrat:.1f} бычий")
        else:
            bear += 1; reasons.append(f"Объём ×{vrat:.1f} медвежий")

    # 12. Свечные паттерны
    if c_bull > c_bear:
        bull += 1; reasons.append("Бычий паттерн свечей")
    elif c_bear > c_bull:
        bear += 1; reasons.append("Медвежий паттерн свечей")

    return bull, bear, atr_v, reasons


# ════════════════════════════════════════════════════════
#  TELEGRAM
# ════════════════════════════════════════════════════════
def send_telegram(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print(f"Telegram ошибка: {e}", flush=True)


# ════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ
# ════════════════════════════════════════════════════════
def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        if r.status_code == 200:
            d = r.json()["data"][0]
            return int(d["value"]), d["value_classification"]
    except Exception:
        pass
    return 50, "Neutral"


def is_trading_hours() -> bool:
    now = datetime.now(TBILISI_TZ)
    return TRADE_START <= now.hour < TRADE_END


# ════════════════════════════════════════════════════════
#  ASYNC: BTC ТРЕНД
# ════════════════════════════════════════════════════════
async def get_btc_trend(exchange) -> str:
    try:
        ohlcv = await exchange.fetch_ohlcv("BTC/USDT", "4h", limit=55)
        data  = np.array(ohlcv, dtype=np.float64)
        close = data[:, 4]
        ema21 = _ewm(close, 21)[-1]
        ema50 = _ewm(close, 50)[-1]
        last  = close[-1]
        del data, close, ohlcv
        gc.collect()
        if last > ema21 and ema21 > ema50: return "bull"
        if last < ema21 and ema21 < ema50: return "bear"
        return "neutral"
    except Exception:
        return "neutral"


# ════════════════════════════════════════════════════════
#  ASYNC: АНАЛИЗ ОДНОЙ МОНЕТЫ
# ════════════════════════════════════════════════════════
async def fetch_and_analyze(
    symbol: str,
    exchange,
    semaphore: asyncio.Semaphore,
    btc_trend: str,
    fg_value: int
):
    ohlcv_1h = ohlcv_4h = None
    async with semaphore:
        try:
            # Параллельный fetch обоих таймфреймов за один раунд
            ohlcv_1h, ohlcv_4h = await asyncio.gather(
                exchange.fetch_ohlcv(symbol, "1h", limit=210),
                exchange.fetch_ohlcv(symbol, "4h", limit=110),
            )

            if len(ohlcv_1h) < 60:
                return None

            # Сохраняем скаляры ДО удаления данных
            last_close = float(ohlcv_1h[-1][4])
            prev_close = float(ohlcv_1h[-2][4])

            bull_1h, bear_1h, atr_1h, reasons_1h = score_timeframe(ohlcv_1h)
            bull_4h, bear_4h, _,      reasons_4h = score_timeframe(ohlcv_4h)

            # Освобождаем сырые данные немедленно (None вместо del — finally не упадёт)
            ohlcv_1h = None
            ohlcv_4h = None
            gc.collect()

            if bull_1h is None or bull_4h is None:
                return None

            dir_1h    = "LONG" if bull_1h > bear_1h else "SHORT"
            dir_4h    = "LONG" if bull_4h > bear_4h else "SHORT"
            direction = dir_1h

            # Взвешенный скор: 1H 60% + 4H 40%
            if dir_1h == dir_4h:
                bull_t = bull_1h * 0.6 + bull_4h * 0.4
                bear_t = bear_1h * 0.6 + bear_4h * 0.4
            else:
                bull_t = float(bull_1h)
                bear_t = float(bear_1h)

            winning = max(bull_t, bear_t)
            conf    = min(round(winning / TOTAL_FACTORS * 100.0, 1), 99.0)

            # ★ СТРОГИЙ ФИЛЬТР — ниже 60% не отправляем
            if conf < MIN_CONF:
                return None

            # Адаптивный SL/TP через ATR, соотношение 1:3
            atr_pct = atr_1h / (last_close + 1e-10)
            sl_pct  = min(max(atr_pct * 1.5, 0.008), 0.03)
            tp_pct  = sl_pct * 3.0

            if direction == "LONG":
                sl = round(last_close * (1.0 - sl_pct), 8)
                tp = round(last_close * (1.0 + tp_pct), 8)
            else:
                sl = round(last_close * (1.0 + sl_pct), 8)
                tp = round(last_close * (1.0 - tp_pct), 8)

            all_reasons = (reasons_1h or [])[:3] + \
                          [f"4H: {r}" for r in (reasons_4h or [])[:2]]

            return {
                "symbol":    symbol,
                "direction": direction,
                "entry":     last_close,
                "sl":        sl,
                "tp":        tp,
                "sl_pct":    round(sl_pct * 100, 2),
                "tp_pct":    round(tp_pct * 100, 2),
                "conf":      conf,
                "reasons":   all_reasons[:5],
                "btc_trend": btc_trend,
                "fg":        fg_value,
            }

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"  ⚠ {symbol}: {e}", flush=True)
            return None
        finally:
            ohlcv_1h = None
            ohlcv_4h = None
            gc.collect()


# ════════════════════════════════════════════════════════
#  ASYNC: ОСНОВНОЙ СКАН
# ════════════════════════════════════════════════════════
async def scan() -> None:
    now_tbs = datetime.now(TBILISI_TZ)

    # ── Нерабочее время ─────────────────────────────────
    if not is_trading_hours():
        next_open = now_tbs.replace(hour=TRADE_START, minute=0, second=0, microsecond=0)
        if now_tbs.hour >= TRADE_END:
            next_open = (now_tbs + timedelta(days=1)).replace(
                hour=TRADE_START, minute=0, second=0, microsecond=0)
        sleep_sec = min(max(0, int((next_open - now_tbs).total_seconds())), 3600)
        print(
            f"[{now_tbs.strftime('%H:%M')} Тбилиси] "
            f"Нерабочее время. Сплю {sleep_sec // 60} мин",
            flush=True
        )
        await asyncio.sleep(sleep_sec)
        return

    # ── Рабочее время ───────────────────────────────────
    print(f"\n[{now_tbs.strftime('%H:%M')} Тбилиси] 🔍 Сканирую 50 монет...", flush=True)

    exchange  = ccxt_async.bybit({"timeout": 15000, "enableRateLimit": True})
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    try:
        fg_value, fg_name = get_fear_greed()
        btc_trend = await get_btc_trend(exchange)
        btc_emoji = "📈" if btc_trend == "bull" else ("📉" if btc_trend == "bear" else "➡️")
        print(f"  F&G: {fg_value} ({fg_name}) | BTC: {btc_trend}", flush=True)

        # Параллельный анализ всех 50 монет
        tasks   = [
            fetch_and_analyze(sym, exchange, semaphore, btc_trend, fg_value)
            for sym in TOP_50
        ]
        results = await asyncio.gather(*tasks)
        found   = [r for r in results if r is not None]
        found.sort(key=lambda x: x["conf"], reverse=True)
        del results, tasks
        gc.collect()

        # ── Отправка сигналов ───────────────────────────
        for s in found:
            emoji  = "🟢" if s["direction"] == "LONG" else "🔴"
            dir_ru = "ЛОНГ" if s["direction"] == "LONG" else "ШОРТ"
            t = datetime.now(TBILISI_TZ).strftime("%H:%M %d.%m.%Y")
            msg = (
                f"{emoji} <b>{dir_ru} | {s['symbol']}</b>\n"
                f"📍 Вход:  <b>{s['entry']}</b>\n"
                f"🛑 Стоп:  <b>{s['sl']}</b>  (-{s['sl_pct']}% ATR)\n"
                f"🎯 Цель:  <b>{s['tp']}</b>  (+{s['tp_pct']}% | 1:3)\n"
                f"💪 Скор:  <b>{s['conf']}%</b>  (≥60% | 1H+4H)\n"
                f"📝 {' | '.join(s['reasons'])}\n"
                f"🌡 F&amp;G: {s['fg']} ({fg_name}) | BTC {btc_emoji}\n"
                f"⏰ {t}"
            )
            send_telegram(msg)
            print(f"  ✅ {s['symbol']} {s['direction']} {s['conf']}%", flush=True)
            await asyncio.sleep(0.5)

        if not found:
            print(f"  Сигналов нет (порог ≥{MIN_CONF}%)", flush=True)
            send_telegram(
                f"😴 Сигналов нет  [≥{MIN_CONF}%]\n"
                f"🌡 F&amp;G: {fg_value} ({fg_name}) | BTC: {btc_emoji}\n"
                f"⏰ Следующий скан через 5 мин"
            )
        else:
            print(f"  → Отправлено: {len(found)} сигнал(ов)", flush=True)

    finally:
        await exchange.close()
        gc.collect()


# ════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ════════════════════════════════════════════════════════
async def main() -> None:
    print("=" * 54, flush=True)
    print("  🤖 КриптоБот v4.0 — DigitalOcean Edition", flush=True)
    print("=" * 54, flush=True)
    print(f"  ✅ Async ({MAX_CONCURRENT} монет параллельно)", flush=True)
    print(f"  ✅ numpy — pandas убран (RAM ↓3-5x)", flush=True)
    print(f"  ✅ Все 50 монет каждые 5 минут", flush=True)
    print(f"  ✅ Скор ≥ {MIN_CONF}%  |  ATR-стоп  |  1:3", flush=True)
    print(f"  ✅ gc.collect() после каждой монеты", flush=True)
    print(f"  Торговые часы: {TRADE_START}:00–{TRADE_END}:00 (UTC+4 Тбилиси)", flush=True)
    print("=" * 54, flush=True)

    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("❌ Не заданы TELEGRAM_TOKEN / CHAT_ID!", flush=True)
        exit(1)

    send_telegram(
        f"🚀 <b>КриптоБот v4.0 запущен!</b>\n"
        f"⏰ Торговые часы: {TRADE_START}:00–{TRADE_END}:00 (Тбилиси)\n"
        f"🔄 Сканирование каждые 5 минут\n"
        f"📊 Скор ≥ {MIN_CONF}% | Топ-50 монет\n"
        f"⚡ Async: {MAX_CONCURRENT} потоков | numpy RAM-safe"
    )

    while True:
        t_start = time.monotonic()
        try:
            await scan()
        except Exception as e:
            print(f"❌ Ошибка цикла: {e}", flush=True)
            traceback.print_exc()
            send_telegram(f"⚠️ Ошибка бота:\n{str(e)[:300]}\n\nПерезапуск через 60 сек...")
            await asyncio.sleep(60)
            continue

        elapsed = time.monotonic() - t_start
        wait    = max(0.0, SCAN_INTERVAL - elapsed)
        print(
            f"  Цикл: {elapsed:.0f}с | "
            f"Следующий через {int(wait // 60)}м {int(wait % 60)}с",
            flush=True
        )
        gc.collect()
        await asyncio.sleep(wait)


if __name__ == "__main__":
    asyncio.run(main())
