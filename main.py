#!/usr/bin/env python3
"""
КриптоБот v5.4 — Production Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Взвешенный скоринг: EMA 30% / RSI 25% / MACD 25% / Volume 20%
• BTC-фильтр: bull→только LONG, bear→только SHORT, flat→−20%
• Anti-fake: мин. 3 индикатора + растущий объём
• Duplicate filter: 30 мин cooldown на монету
• Volatility filter: пропуск если движение <0.5%
• Blacklist низколиквидных монет
• Liquidity Grab v2 (15m): 15-свечное окно, быстрый возврат,
  улучш. volume, фильтр "не поздно" (>0.5% → игнор)
• Трекер исходов + дневной Win Rate отчёт в 20:00
• Retry (3×) при ECONNRESET/timeout → стабильность без падений
• asyncio + numpy — оптимизация под 512MB RAM
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

# ════════════════════════════════════════════════════════════
#  КОНФИГУРАЦИЯ
# ════════════════════════════════════════════════════════════
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID         = os.getenv("CHAT_ID", "")

# --- Скоринг ---
MIN_CONF            = 60.0      # Минимальный итоговый скор (%)
MIN_INDICATORS      = 3         # Мин. индикаторов в одном направлении
WEIGHT_TREND        = 0.30      # EMA тренд
WEIGHT_RSI          = 0.25      # RSI
WEIGHT_MACD         = 0.25      # MACD
WEIGHT_VOLUME       = 0.20      # Объём

# --- Тайминг ---
SCAN_INTERVAL       = 300       # 5 минут между циклами
MIN_SCAN_GAP        = 60        # Не чаще 1 раза в минуту
DUPLICATE_COOLDOWN  = 1800      # 30 мин: не дублировать сигнал по монете

# --- Фильтры ---
VOLATILITY_MIN      = 0.005     # 0.5% минимальное движение
BTC_FLAT_PENALTY    = 0.80      # Снижение уверенности при флэте BTC

# --- Сервер ---
MAX_CONCURRENT      = 6         # параллельных запросов (снижено для стабильности)
TRADE_START         = 8
TRADE_END           = 20
TBILISI_TZ          = pytz.timezone('Asia/Tbilisi')

# --- Сетевая устойчивость ---
RETRY_ATTEMPTS      = 3         # попыток при ECONNRESET / timeout
RETRY_DELAY         = 1.5       # базовая задержка между попытками (сек)
API_TIMEOUT         = 10000     # ms — таймаут каждого запроса

# --- Монеты ---
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

# Blacklist: мусорные / низколиквидные монеты (пополнять по мере выявления)
BLACKLIST = {
    "FLOW/USDT", "CHZ/USDT", "SAND/USDT", "MANA/USDT",
    "GALA/USDT", "AXS/USDT", "ENJ/USDT",
}

# ════════════════════════════════════════════════════════════
#  RUNTIME STATE (очищается каждый цикл)
# ════════════════════════════════════════════════════════════
last_prices: dict = {}          # symbol → last_close (для детекции ≥1%)
last_signal_ts: dict = {}       # symbol → timestamp последнего сигнала
last_scan_time: float = 0.0     # monotonic time последнего скана

# — Трекер исходов сигналов (в памяти, без БД) —
open_signals: dict = {}         # symbol → {entry,sl,tp,direction,conf,ts,day}
daily_stats: dict = {}          # "YYYY-MM-DD" → {WIN,LOSS,EXPIRED,signals:[]}
last_report_day: str = ""       # чтобы не слать отчёт дважды в один день


# ════════════════════════════════════════════════════════════
#  NUMPY ИНДИКАТОРЫ
# ════════════════════════════════════════════════════════════
def _ewm(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def calc_rsi(close: np.ndarray, period: int = 14) -> float:
    delta = np.diff(close)
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    ag = _ewm(gain, period)[-1]
    al = _ewm(loss, period)[-1]
    if al < 1e-10:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def calc_macd(close: np.ndarray, fast=12, slow=26, sig=9):
    macd_line = _ewm(close, fast) - _ewm(close, slow)
    signal    = _ewm(macd_line, sig)
    return macd_line[-1], signal[-1], macd_line[-2], signal[-2]


def calc_ema_last(close: np.ndarray, period: int) -> float:
    return _ewm(close, period)[-1]


def calc_atr_last(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> float:
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    return _ewm(tr, period)[-1]


def calc_volatility(close: np.ndarray, period=20) -> float:
    """Процентная волатильность за последние period свечей"""
    if len(close) < period:
        return 0.0
    segment = close[-period:]
    return (segment.max() - segment.min()) / (segment.mean() + 1e-10)


def volume_trend(volume: np.ndarray, period=10) -> float:
    """Отношение последнего объёма к среднему за period. >1 = рост"""
    if len(volume) < period + 1:
        return 1.0
    avg = volume[-period - 1:-1].mean()
    return volume[-1] / (avg + 1e-10)


# ════════════════════════════════════════════════════════════
#  LIQUIDITY GRAB — детектор ложных пробоев
# ════════════════════════════════════════════════════════════
def detect_liquidity_grab(ohlcv_15m: list, direction: str):
    """
    Детектирует паттерн "снятия ликвидности" на 15m свечах. v2.

    Base  : 15 свечей (3.75 ч) — достаточный локальный уровень без устаревших данных.
    Pattern: последние 3 свечи (45 мин) — окно sweep + возврат.

    LONG : пробой локального LOW ↓ → возврат выше уровня → бычья свеча
    SHORT: пробой локального HIGH ↑ → возврат ниже уровня → медвежья свеча

    Быстрый возврат    : sweep + return в рамках 3 свечей (45 мин) — встроено.
    Не поздно входить  : close[-1] не ушёл более чем на 0.5% от уровня пробоя.
    Volume (улучшено)  : ≥ среднего за 20 свечей ИЛИ ≥ ×1.5 среднего за базу.
    Минимальный sweep  : ≥ 0.3% (защита от шума).

    Возвращает (detected: bool, reason: str).
    """
    if len(ohlcv_15m) < 20:
        return False, ""

    # Сегмент 18 свечей: base = первые 15, pattern = последние 3
    seg    = np.array(ohlcv_15m[-18:], dtype=np.float64)
    open_  = seg[:, 1]
    high_  = seg[:, 2]
    low_   = seg[:, 3]
    close_ = seg[:, 4]
    vol_   = seg[:, 5]

    base_high    = high_[:15].max()    # локальный HIGH за 15 свечей (~3.75ч)
    base_low     = low_[:15].min()     # локальный LOW
    avg_vol_base = vol_[:15].mean()    # средний объём за базу

    # Volume за 20 свечей (для более мягкого порога)
    vol_20   = np.array(ohlcv_15m[-20:], dtype=np.float64)[:, 5]
    avg_vol_20 = vol_20.mean()
    vol_20   = None  # освобождаем

    # Volume OK: ≥ среднего за 20 свечей ИЛИ ≥ ×1.5 базового среднего
    vol_ok = (vol_[-1] >= avg_vol_20) or (vol_[-1] >= avg_vol_base * 1.5)

    del seg

    if direction == "LONG":
        # 1. Sweep: хотя бы одна из 3 pattern-свечей пробила base_low
        swept = min(low_[-3], low_[-2], low_[-1])
        if swept >= base_low:
            return False, ""

        # 2. Размер sweep ≥ 0.3% (отсев шума)
        move_pct = (base_low - swept) / (base_low + 1e-10)
        if move_pct < 0.003:
            return False, ""

        # 3. Быстрый возврат: close[-1] вернулся выше base_low
        if close_[-1] <= base_low:
            return False, ""

        # 4. Подтверждение: последняя свеча бычья
        if close_[-1] <= open_[-1]:
            return False, ""

        # 5. Объём подтверждает вход крупного игрока
        if not vol_ok:
            return False, ""

        # 6. Не поздно: close[-1] не ушёл >0.5% выше уровня пробоя (base_low)
        late_pct = (close_[-1] - base_low) / (base_low + 1e-10)
        if late_pct > 0.005:
            return False, ""

        return True, f"LiqGrab ↓{move_pct * 100:.1f}% возврат+{late_pct * 100:.2f}% (15m)"

    elif direction == "SHORT":
        # 1. Sweep: хотя бы одна из 3 pattern-свечей пробила base_high
        spiked_to = max(high_[-3], high_[-2], high_[-1])
        if spiked_to <= base_high:
            return False, ""

        # 2. Размер sweep ≥ 0.3%
        move_pct = (spiked_to - base_high) / (base_high + 1e-10)
        if move_pct < 0.003:
            return False, ""

        # 3. Быстрый возврат: close[-1] вернулся ниже base_high
        if close_[-1] >= base_high:
            return False, ""

        # 4. Подтверждение: последняя свеча медвежья
        if close_[-1] >= open_[-1]:
            return False, ""

        # 5. Объём подтверждает вход крупного игрока
        if not vol_ok:
            return False, ""

        # 6. Не поздно: close[-1] не ушёл >0.5% ниже уровня пробоя (base_high)
        late_pct = (base_high - close_[-1]) / (base_high + 1e-10)
        if late_pct > 0.005:
            return False, ""

        return True, f"LiqGrab ↑{move_pct * 100:.1f}% возврат+{late_pct * 100:.2f}% (15m)"

    return False, ""


# ════════════════════════════════════════════════════════════
#  СЕТЕВЫЕ УТИЛИТЫ — retry + exchange factory
# ════════════════════════════════════════════════════════════
# Ключевые слова для retryable сетевых ошибок
_RETRY_KEYWORDS = (
    "econnreset", "timeout", "timed out", "connection",
    "network", "ssl", "reset", "502", "503", "504",
)


async def _retry(fn, *args, attempts: int = RETRY_ATTEMPTS,
                 delay: float = RETRY_DELAY, **kwargs):
    """
    Оборачивает любой async-вызов ccxt в retry-логику.
    При сетевой ошибке повторяет до `attempts` раз с экспоненциальной задержкой.
    CancelledError пробрасывается немедленно.
    """
    last_exc: Exception = Exception("unknown")
    for attempt in range(attempts):
        try:
            return await fn(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                err = str(exc).lower()
                if any(k in err for k in _RETRY_KEYWORDS):
                    await asyncio.sleep(delay * (attempt + 1))  # 1.5 → 3.0 → 4.5
                    continue
            break
    raise last_exc


def make_exchange(timeout: int = API_TIMEOUT) -> ccxt_async.bybit:
    """
    Фабрика Bybit exchange с оптимальными настройками стабильности.
    Единая точка конфигурации — менять только здесь.
    """
    return ccxt_async.bybit({
        "timeout":         timeout,
        "enableRateLimit": True,
    })


# ════════════════════════════════════════════════════════════
#  ВЗВЕШЕННЫЙ СКОРИНГ (PRO)
# ════════════════════════════════════════════════════════════
def weighted_score(ohlcv_raw: list):
    """
    Возвращает dict:
      direction: "LONG" | "SHORT" | None
      conf: float (0-100)
      reasons: list[str]
      atr: float
      indicators_agree: int (сколько индикаторов в направлении сигнала)
      vol_rising: bool
    Или None если данных мало.
    """
    if len(ohlcv_raw) < 60:
        return None

    data   = np.array(ohlcv_raw, dtype=np.float64)
    high_  = data[:, 2]
    low_   = data[:, 3]
    close_ = data[:, 4]
    vol_   = data[:, 5]
    close_now = close_[-1]

    # ── 1. TREND (EMA) — вес 30% ──────────────────────────
    ema9  = calc_ema_last(close_, 9)
    ema21 = calc_ema_last(close_, 21)
    ema50 = calc_ema_last(close_, 50)

    trend_score = 0.0
    trend_reason = ""

    if ema9 > ema21 > ema50 and close_now > ema9:
        trend_score = 1.0
        trend_reason = "Сильный восходящий тренд (EMA 9>21>50)"
    elif ema9 > ema21 and close_now > ema21:
        trend_score = 0.6
        trend_reason = "Восходящий тренд (EMA 9>21)"
    elif ema9 < ema21 < ema50 and close_now < ema9:
        trend_score = -1.0
        trend_reason = "Сильный нисходящий тренд (EMA 9<21<50)"
    elif ema9 < ema21 and close_now < ema21:
        trend_score = -0.6
        trend_reason = "Нисходящий тренд (EMA 9<21)"
    else:
        trend_score = 0.0
        trend_reason = ""

    # ── 2. RSI — вес 25% ──────────────────────────────────
    rsi_val = calc_rsi(close_)
    rsi_score = 0.0
    rsi_reason = ""

    if rsi_val < 25:
        rsi_score = 1.0;  rsi_reason = f"RSI сильно перепродан ({rsi_val:.0f})"
    elif rsi_val < 35:
        rsi_score = 0.6;  rsi_reason = f"RSI перепродан ({rsi_val:.0f})"
    elif rsi_val > 75:
        rsi_score = -1.0; rsi_reason = f"RSI сильно перекуплен ({rsi_val:.0f})"
    elif rsi_val > 65:
        rsi_score = -0.6; rsi_reason = f"RSI перекуплен ({rsi_val:.0f})"

    # ── 3. MACD — вес 25% ─────────────────────────────────
    macd_now, sig_now, macd_prev, sig_prev = calc_macd(close_)
    macd_score = 0.0
    macd_reason = ""

    # Crossover detection
    cross_up   = macd_prev <= sig_prev and macd_now > sig_now
    cross_down = macd_prev >= sig_prev and macd_now < sig_now

    if cross_up:
        macd_score = 1.0;  macd_reason = "MACD crossover ↑"
    elif macd_now > sig_now and macd_now > 0:
        macd_score = 0.7;  macd_reason = "MACD бычий"
    elif cross_down:
        macd_score = -1.0; macd_reason = "MACD crossover ↓"
    elif macd_now < sig_now and macd_now < 0:
        macd_score = -0.7; macd_reason = "MACD медвежий"

    # ── 4. VOLUME — вес 20% ───────────────────────────────
    vt = volume_trend(vol_)
    vol_score = 0.0
    vol_reason = ""
    vol_rising = vt > 1.2

    if vt > 2.0:
        # Сильный спайк — направление по цене
        if close_now > close_[-2]:
            vol_score = 1.0;  vol_reason = f"Объём ×{vt:.1f} + рост цены"
        else:
            vol_score = -1.0; vol_reason = f"Объём ×{vt:.1f} + падение цены"
    elif vt > 1.5:
        if close_now > close_[-2]:
            vol_score = 0.6;  vol_reason = f"Объём ×{vt:.1f} растёт"
        else:
            vol_score = -0.6; vol_reason = f"Объём ×{vt:.1f} с давлением вниз"
    elif vt < 0.6:
        vol_score = 0.0; vol_reason = "Объём низкий — слабый сигнал"
        vol_rising = False

    # ── ИТОГОВЫЙ ВЗВЕШЕННЫЙ СКОР ──────────────────────────
    total = (
        WEIGHT_TREND  * trend_score +
        WEIGHT_RSI    * rsi_score   +
        WEIGHT_MACD   * macd_score  +
        WEIGHT_VOLUME * vol_score
    )

    # Считаем сколько индикаторов согласны
    direction = "LONG" if total > 0 else "SHORT"
    scores = [trend_score, rsi_score, macd_score, vol_score]

    if direction == "LONG":
        agree = sum(1 for s in scores if s > 0.1)
    else:
        agree = sum(1 for s in scores if s < -0.1)

    conf = min(abs(total) * 100.0, 99.0)

    # Собираем причины (только значимые)
    reasons = []
    for r in [trend_reason, rsi_reason, macd_reason, vol_reason]:
        if r:
            reasons.append(r)

    atr_val = calc_atr_last(high_, low_, close_)

    # Очистка
    del data, high_, low_, close_, vol_

    if conf < 5.0:
        return None

    return {
        "direction":        direction,
        "conf":             round(conf, 1),
        "reasons":          reasons,
        "atr":              atr_val,
        "indicators_agree": agree,
        "vol_rising":       vol_rising,
        "close":            close_now,
    }


# ════════════════════════════════════════════════════════════
#  TELEGRAM (только сигналы)
# ════════════════════════════════════════════════════════════
def send_telegram(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception:
        pass  # Не спамим логами — бот продолжает работать


def send_error(msg: str) -> None:
    """Отправить только критическую ошибку"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": CHAT_ID, "text": f"⚠️ {msg}", "parse_mode": "HTML"},
            timeout=10
        )
    except Exception:
        pass


# ════════════════════════════════════════════════════════════
#  ТРЕКЕР ИСХОДОВ СИГНАЛОВ
# ════════════════════════════════════════════════════════════
def record_signal(signal: dict) -> None:
    """Записываем новый сигнал в трекер при отправке."""
    global open_signals, daily_stats
    symbol = signal["symbol"]
    today  = datetime.now(TBILISI_TZ).strftime("%Y-%m-%d")

    # Если по этой монете уже есть открытый сигнал — перезаписываем
    open_signals[symbol] = {
        "entry":     signal["entry"],
        "sl":        signal["sl"],
        "tp":        signal["tp"],
        "direction": signal["direction"],
        "conf":      signal["conf"],
        "ts":        time.time(),
        "day":       today,
    }

    if today not in daily_stats:
        daily_stats[today] = {"WIN": 0, "LOSS": 0, "EXPIRED": 0, "signals": []}

    daily_stats[today]["signals"].append({
        "symbol":    symbol,
        "direction": signal["direction"],
        "conf":      signal["conf"],
        "entry":     signal["entry"],
        "result":    None,
    })


def close_signal(symbol: str, result: str, price: float) -> None:
    """Закрываем сигнал, обновляем статистику и шлём уведомление в Telegram."""
    global open_signals, daily_stats
    sig = open_signals.pop(symbol, None)
    if not sig:
        return

    day = sig["day"]
    if day not in daily_stats:
        daily_stats[day] = {"WIN": 0, "LOSS": 0, "EXPIRED": 0, "signals": []}
    daily_stats[day][result] += 1

    # Проставляем результат в запись дня
    for s in daily_stats[day]["signals"]:
        if s["symbol"] == symbol and s["result"] is None:
            s["result"] = result
            break

    # Формируем уведомление
    entry_price = sig["entry"]
    pnl_pct = abs(price - entry_price) / (entry_price + 1e-10) * 100
    dir_label = "LONG" if sig["direction"] == "LONG" else "SHORT"

    if result == "WIN":
        emoji  = "✅"
        detail = f"+{pnl_pct:.2f}%"
    elif result == "LOSS":
        emoji  = "❌"
        detail = f"−{pnl_pct:.2f}%"
    else:  # EXPIRED
        emoji  = "⏱"
        detail = "истёк лимит 24ч"

    msg = (
        f"{emoji} <b>{symbol} — {dir_label} → {result}</b>\n"
        f"Вход: {entry_price}  →  Текущая: {round(price, 8)}\n"
        f"Результат: {detail}\n"
        f"Уверенность была: {sig['conf']}%"
    )
    send_telegram(msg)


async def check_outcomes(exchange) -> None:
    """Проверяем открытые сигналы на достижение TP/SL или истечение 24ч."""
    if not open_signals:
        return

    symbols = list(open_signals.keys())
    try:
        tickers = await _retry(exchange.fetch_tickers, symbols)
    except Exception:
        return

    now = time.time()

    for symbol in list(open_signals.keys()):
        sig = open_signals.get(symbol)
        if not sig:
            continue

        # Таймаут 24 часа → EXPIRED
        if now - sig["ts"] > 86400:
            tick  = tickers.get(symbol, {})
            price = tick.get("last") or sig["entry"]
            close_signal(symbol, "EXPIRED", price)
            continue

        tick  = tickers.get(symbol, {})
        price = tick.get("last")
        if not price:
            continue

        if sig["direction"] == "LONG":
            if price >= sig["tp"]:
                close_signal(symbol, "WIN",  price)
            elif price <= sig["sl"]:
                close_signal(symbol, "LOSS", price)
        else:  # SHORT
            if price <= sig["tp"]:
                close_signal(symbol, "WIN",  price)
            elif price >= sig["sl"]:
                close_signal(symbol, "LOSS", price)


def send_daily_report() -> None:
    """Отправляем дневной отчёт в Telegram (вызывается в 20:00 по Тбилиси)."""
    global last_report_day
    today = datetime.now(TBILISI_TZ).strftime("%Y-%m-%d")
    last_report_day = today

    stats   = daily_stats.get(today, {"WIN": 0, "LOSS": 0, "EXPIRED": 0, "signals": []})
    wins    = stats["WIN"]
    losses  = stats["LOSS"]
    expired = stats["EXPIRED"]
    total   = wins + losses + expired

    if total == 0:
        send_telegram(
            f"📊 <b>Дневной отчёт — {today}</b>\n"
            "Сигналов сегодня не было."
        )
        return

    closed   = wins + losses
    win_rate = round(wins / closed * 100, 1) if closed > 0 else 0.0

    signals_today = [s for s in stats["signals"] if s["result"]]
    best  = max(signals_today, key=lambda x: x["conf"], default=None) if signals_today else None
    worst_list = [s for s in signals_today if s["result"] == "LOSS"]
    worst = worst_list[0] if worst_list else None

    lines = [
        f"📊 <b>Дневной отчёт — {today}</b>",
        f"Сигналов: {total}  |  ✅ {wins}  ❌ {losses}  ⏱ {expired}",
    ]
    if closed > 0:
        lines.append(f"Win Rate: <b>{win_rate}%</b>")
    if best:
        lines.append(
            f"Лучший: {best['symbol']} {best['direction']} "
            f"({best['conf']}% → {best['result']})"
        )
    if worst:
        lines.append(f"Проигрыш: {worst['symbol']} {worst['direction']}")

    send_telegram("\n".join(lines))


# ════════════════════════════════════════════════════════════
#  BTC ТРЕНД-ФИЛЬТР
# ════════════════════════════════════════════════════════════
async def get_btc_trend(exchange) -> str:
    """Возвращает 'bull', 'bear', или 'flat'. Retry при сетевых ошибках."""
    try:
        ohlcv = await _retry(exchange.fetch_ohlcv, "BTC/USDT", "4h", limit=55)
        data  = np.array(ohlcv, dtype=np.float64)
        close = data[:, 4]
        ema21 = _ewm(close, 21)[-1]
        ema50 = _ewm(close, 50)[-1]
        last  = close[-1]
        del data, close, ohlcv
        if last > ema21 and ema21 > ema50: return "bull"
        if last < ema21 and ema21 < ema50: return "bear"
        return "flat"
    except Exception:
        return "flat"


# ════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ
# ════════════════════════════════════════════════════════════
def is_trading_hours() -> bool:
    return TRADE_START <= datetime.now(TBILISI_TZ).hour < TRADE_END


def is_duplicate(symbol: str) -> bool:
    """True если сигнал по этой монете уже отправлялся <30 мин назад"""
    ts = last_signal_ts.get(symbol, 0)
    return (time.time() - ts) < DUPLICATE_COOLDOWN


def price_changed_enough(symbol: str, current_price: float) -> bool:
    """True если цена изменилась ≥1% с прошлого скана"""
    prev = last_prices.get(symbol)
    if prev is None:
        return False
    return abs(current_price - prev) / (prev + 1e-10) >= 0.01


# ════════════════════════════════════════════════════════════
#  ASYNC: АНАЛИЗ ОДНОЙ МОНЕТЫ
# ════════════════════════════════════════════════════════════
async def analyze_coin(
    symbol: str,
    exchange,
    semaphore: asyncio.Semaphore,
    btc_trend: str,
):
    if symbol in BLACKLIST:
        return None

    ohlcv_1h = ohlcv_4h = ohlcv_15m = None
    async with semaphore:
        try:
            # Три таймфрейма параллельно (каждый с retry при ECONNRESET/timeout)
            ohlcv_1h, ohlcv_4h, ohlcv_15m = await asyncio.gather(
                _retry(exchange.fetch_ohlcv, symbol, "1h",  limit=210),
                _retry(exchange.fetch_ohlcv, symbol, "4h",  limit=110),
                _retry(exchange.fetch_ohlcv, symbol, "15m", limit=60),
            )

            if len(ohlcv_1h) < 60:
                return None

            last_close = float(ohlcv_1h[-1][4])

            # ── Volatility filter ─────────────────────────
            data_tmp = np.array(ohlcv_1h, dtype=np.float64)
            volatility = calc_volatility(data_tmp[:, 4])
            del data_tmp

            if volatility < VOLATILITY_MIN:
                last_prices[symbol] = last_close
                return None

            # ── Скоринг 1H (основной) ────────────────────
            result_1h = weighted_score(ohlcv_1h)

            # ── Liquidity Grab на 15m ─────────────────────
            # 15m: 3 свечи = 45 мин — реальное окно для stop hunt
            # Запускаем ДО очистки данных; результат — просто bool + строка
            _dir_tmp = result_1h["direction"] if result_1h else "LONG"
            lg_detected, lg_reason = detect_liquidity_grab(ohlcv_15m, _dir_tmp)
            ohlcv_15m = None          # сразу освобождаем

            # ── Скоринг 4H (подтверждение) ───────────────
            result_4h = weighted_score(ohlcv_4h)

            ohlcv_1h = None
            ohlcv_4h = None

            if result_1h is None:
                last_prices[symbol] = last_close
                return None

            direction = result_1h["direction"]
            conf      = result_1h["conf"]
            reasons   = result_1h["reasons"]
            atr       = result_1h["atr"]
            agree     = result_1h["indicators_agree"]
            vol_ok    = result_1h["vol_rising"]

            # ── 4H подтверждение: бонус если совпадает ────
            if result_4h and result_4h["direction"] == direction:
                conf = conf * 0.7 + result_4h["conf"] * 0.3
                conf = min(conf, 99.0)

            # ── Liquidity Grab бонус: +12% к скору ───────
            # Не обязателен — сигнал работает и без него
            if lg_detected and lg_reason:
                conf = min(conf + 12.0, 99.0)
                reasons = [lg_reason] + reasons  # LG — первым в причинах

            # ── ФИЛЬТР 1: Anti-fake — мин. 3 индикатора ──
            if agree < MIN_INDICATORS:
                last_prices[symbol] = last_close
                return None

            # ── ФИЛЬТР 2: Объём должен расти ─────────────
            if not vol_ok:
                last_prices[symbol] = last_close
                return None

            # ── ФИЛЬТР 3: BTC-фильтр ─────────────────────
            if btc_trend == "bull" and direction == "SHORT":
                last_prices[symbol] = last_close
                return None
            if btc_trend == "bear" and direction == "LONG":
                last_prices[symbol] = last_close
                return None
            if btc_trend == "flat":
                conf *= BTC_FLAT_PENALTY

            # ── ФИЛЬТР 4: Минимальный скор ───────────────
            if conf < MIN_CONF:
                last_prices[symbol] = last_close
                return None

            # ── ФИЛЬТР 5: Дубликат <30 мин ───────────────
            if is_duplicate(symbol):
                last_prices[symbol] = last_close
                return None

            # ── SL/TP через ATR (1:3) ────────────────────
            atr_pct = atr / (last_close + 1e-10)
            sl_pct  = min(max(atr_pct * 1.5, 0.008), 0.03)
            tp_pct  = sl_pct * 3.0

            if direction == "LONG":
                sl = round(last_close * (1.0 - sl_pct), 8)
                tp = round(last_close * (1.0 + tp_pct), 8)
            else:
                sl = round(last_close * (1.0 + sl_pct), 8)
                tp = round(last_close * (1.0 - tp_pct), 8)

            # Обновляем state
            last_prices[symbol] = last_close

            return {
                "symbol":    symbol,
                "direction": direction,
                "entry":     last_close,
                "sl":        sl,
                "tp":        tp,
                "sl_pct":    round(sl_pct * 100, 2),
                "tp_pct":    round(tp_pct * 100, 2),
                "conf":      round(conf, 1),
                "reasons":   reasons,
            }

        except asyncio.CancelledError:
            raise
        except Exception:
            return None
        finally:
            ohlcv_1h  = None
            ohlcv_4h  = None
            ohlcv_15m = None
            gc.collect()


# ════════════════════════════════════════════════════════════
#  ASYNC: ОСНОВНОЙ СКАН
# ════════════════════════════════════════════════════════════
async def scan() -> None:
    global last_scan_time

    if not is_trading_hours():
        now_tbs = datetime.now(TBILISI_TZ)
        next_open = now_tbs.replace(hour=TRADE_START, minute=0, second=0, microsecond=0)
        if now_tbs.hour >= TRADE_END:
            next_open = (now_tbs + timedelta(days=1)).replace(
                hour=TRADE_START, minute=0, second=0, microsecond=0)
        sleep_sec = min(max(0, int((next_open - now_tbs).total_seconds())), 3600)
        await asyncio.sleep(sleep_sec)
        return

    # Rate limit — не чаще 1 раза в минуту
    now_mono = time.monotonic()
    if now_mono - last_scan_time < MIN_SCAN_GAP:
        await asyncio.sleep(MIN_SCAN_GAP - (now_mono - last_scan_time))
    last_scan_time = time.monotonic()

    exchange  = make_exchange()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    try:
        btc_trend = await get_btc_trend(exchange)

        # Отфильтровать монеты: исключить blacklist
        coins = [c for c in TOP_50 if c not in BLACKLIST]

        tasks   = [analyze_coin(sym, exchange, semaphore, btc_trend) for sym in coins]
        results = await asyncio.gather(*tasks)
        found   = [r for r in results if r is not None]
        found.sort(key=lambda x: x["conf"], reverse=True)

        del results, tasks

        for s in found:
            dir_label = "LONG" if s["direction"] == "LONG" else "SHORT"
            t = datetime.now(TBILISI_TZ).strftime("%H:%M %d.%m.%Y")
            msg = (
                f"<b>{s['symbol']} — {dir_label}</b>\n"
                f"Цена: {s['entry']}\n"
                f"Стоп: {s['sl']}  (−{s['sl_pct']}%)\n"
                f"Цель: {s['tp']}  (+{s['tp_pct']}%)\n"
                f"Причина: {' + '.join(s['reasons'])}\n"
                f"Уверенность: {s['conf']}%\n"
                f"{t}"
            )
            send_telegram(msg)
            last_signal_ts[s["symbol"]] = time.time()
            record_signal(s)
            await asyncio.sleep(0.3)

    finally:
        await exchange.close()
        gc.collect()


# ════════════════════════════════════════════════════════════
#  ДЕТЕКЦИЯ РЕЗКИХ ДВИЖЕНИЙ (≥1%)
# ════════════════════════════════════════════════════════════
async def quick_price_check(exchange) -> list:
    """Быстрая проверка: какие монеты двинулись ≥1% с прошлого скана"""
    changed = []
    try:
        tickers = await _retry(
            exchange.fetch_tickers,
            [c for c in TOP_50 if c not in BLACKLIST]
        )
        for sym, tick in tickers.items():
            price = tick.get("last")
            if price and sym in last_prices:
                if price_changed_enough(sym, price):
                    changed.append(sym)
            if price:
                last_prices[sym] = price
    except Exception:
        pass
    return changed


# ════════════════════════════════════════════════════════════
#  ГЛАВНЫЙ ЦИКЛ
# ════════════════════════════════════════════════════════════
async def main() -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("TELEGRAM_TOKEN / CHAT_ID не заданы", flush=True)
        exit(1)

    send_telegram(
        "<b>КриптоБот v5.4</b> запущен\n"
        f"Часы: {TRADE_START}:00–{TRADE_END}:00 UTC+4\n"
        f"Скор ≥{MIN_CONF}% | LiqGrab v2 | Retry 3× | Win Rate отчёт"
    )

    while True:
        t_start = time.monotonic()
        try:
            await scan()
        except Exception as e:
            send_error(str(e)[:300])
            await asyncio.sleep(60)
            continue

        # ── Между основными сканами: проверяем резкие движения ──
        elapsed = time.monotonic() - t_start
        remaining = max(0.0, SCAN_INTERVAL - elapsed)

        # Проверяем цены каждые 60 секунд в паузе
        while remaining > 0:
            wait_chunk = min(remaining, 60.0)
            await asyncio.sleep(wait_chunk)
            remaining -= wait_chunk

            if remaining > 30 and is_trading_hours():
                try:
                    exchange = make_exchange()
                    changed = await quick_price_check(exchange)
                    await check_outcomes(exchange)   # проверяем TP/SL открытых сигналов
                    await exchange.close()

                    if changed:
                        # Внеплановый скан только для изменившихся монет
                        exchange = make_exchange()
                        sem = asyncio.Semaphore(MAX_CONCURRENT)
                        btc_t = await get_btc_trend(exchange)
                        tasks = [analyze_coin(s, exchange, sem, btc_t) for s in changed]
                        results = await asyncio.gather(*tasks)
                        for r in results:
                            if r:
                                t = datetime.now(TBILISI_TZ).strftime("%H:%M %d.%m.%Y")
                                dl = "LONG" if r["direction"] == "LONG" else "SHORT"
                                msg = (
                                    f"⚡ <b>{r['symbol']} — {dl}</b>  (движение ≥1%)\n"
                                    f"Цена: {r['entry']}\n"
                                    f"Стоп: {r['sl']}  (−{r['sl_pct']}%)\n"
                                    f"Цель: {r['tp']}  (+{r['tp_pct']}%)\n"
                                    f"Причина: {' + '.join(r['reasons'])}\n"
                                    f"Уверенность: {r['conf']}%\n"
                                    f"{t}"
                                )
                                send_telegram(msg)
                                last_signal_ts[r["symbol"]] = time.time()
                                record_signal(r)
                        del results, tasks
                        await exchange.close()
                        gc.collect()
                except Exception:
                    pass

        # ── Дневной отчёт в конце торгового дня ──────────────────
        now_tbs = datetime.now(TBILISI_TZ)
        today   = now_tbs.strftime("%Y-%m-%d")
        if now_tbs.hour >= TRADE_END and last_report_day != today:
            send_daily_report()

        gc.collect()


if __name__ == "__main__":
    asyncio.run(main())
