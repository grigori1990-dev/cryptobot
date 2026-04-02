"""
signal.py — Сигнальный модуль
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ответственность:
  Получить массивы OHLCV закрытых баров → вернуть SignalResult или None.

Что НЕ делает:
  - Ничего не знает о балансе, позициях, Bybit API, Telegram.
  - Не читает и не пишет состояние.
  - Одинаковая логика для всех символов — никакой индивидуальной настройки.

Стратегия: RSI + Bollinger Bands (Hypothesis B, backtested)
  LONG:  RSI14 < 30  AND  close ≤ BB_lower(20, 2σ)
  SHORT: RSI14 > 70  AND  close ≥ BB_upper(20, 2σ)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Literal

# ─── CONFIG ────────────────────────────────────────────────────────────────────
RSI_PERIOD = 14
RSI_OS     = 30       # oversold  → LONG
RSI_OB     = 70       # overbought→ SHORT
BB_PERIOD  = 20
BB_STD     = 2.0
ATR_PERIOD = 14


# ─── OUTPUT ────────────────────────────────────────────────────────────────────
@dataclass
class SignalResult:
    symbol:       str
    direction:    Literal["LONG", "SHORT"]
    signal_price: float          # close последнего закрытого бара
    rsi:          float
    bb_lower:     float
    bb_upper:     float
    atr:          float          # ATR14 — нужен Risk для расчёта стопа и зоны входа
    bar_ts:       int            # timestamp закрытого бара (мс) — State использует для дедупликации


# ─── INDICATORS (all causal, no lookahead) ─────────────────────────────────────
def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI — только закрытые данные, без lookahead."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    d  = np.diff(close)
    g  = np.where(d > 0, d, 0.0)
    lo = np.where(d < 0, -d, 0.0)
    ag = g[:period].mean()
    al = lo[:period].mean()
    out[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(period, n - 1):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + lo[i]) / period
        out[i + 1] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def _bb(close: np.ndarray, period: int = 20, std: float = 2.0):
    """Bollinger Bands — rolling window, без lookahead."""
    n  = len(close)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    for i in range(period - 1, n):
        w    = close[i - period + 1: i + 1]
        m    = w.mean()
        s    = w.std(ddof=0)
        lo[i] = m - std * s
        hi[i] = m + std * s
    return lo, hi


def _atr(high: np.ndarray, low: np.ndarray,
         close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's ATR."""
    n  = len(high)
    tr = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    atr = np.full(n, np.nan)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ─── PUBLIC API ────────────────────────────────────────────────────────────────
def check(
    symbol: str,
    highs:  np.ndarray,
    lows:   np.ndarray,
    closes: np.ndarray,
    bar_ts: int,
) -> SignalResult | None:
    """
    Проверить сигнал на последнем закрытом баре.

    Параметры:
        symbol  — торговая пара (например "BTC/USDT:USDT")
        highs   — массив максимумов закрытых баров (индекс -1 = последний закрытый)
        lows    — массив минимумов закрытых баров
        closes  — массив цен закрытия закрытых баров
        bar_ts  — timestamp последнего закрытого бара (мс)

    Возвращает SignalResult или None (нет сигнала).
    """
    min_bars = max(RSI_PERIOD, BB_PERIOD, ATR_PERIOD) + 5
    if len(closes) < min_bars:
        return None

    rsi14         = _rsi(closes, RSI_PERIOD)
    bb_lo, bb_hi  = _bb(closes, BB_PERIOD, BB_STD)
    atr14         = _atr(highs, lows, closes, ATR_PERIOD)

    r   = rsi14[-1]
    cl  = closes[-1]
    blo = bb_lo[-1]
    bhi = bb_hi[-1]
    a   = atr14[-1]

    if math.isnan(r) or math.isnan(blo) or math.isnan(a):
        return None

    direction: Literal["LONG", "SHORT"] | None = None
    if r <= RSI_OS and cl <= blo:
        direction = "LONG"
    elif r >= RSI_OB and cl >= bhi:
        direction = "SHORT"

    if direction is None:
        return None

    return SignalResult(
        symbol       = symbol,
        direction    = direction,
        signal_price = cl,
        rsi          = round(r, 2),
        bb_lower     = round(blo, 4),
        bb_upper     = round(bhi, 4),
        atr          = round(a, 4),
        bar_ts       = bar_ts,
    )


def bb_overshoot(sig: SignalResult) -> float:
    """
    Метрика для ранжирования сигналов внутри одного цикла.
    Насколько цена статистически "выбита" за полосу Боллинджера.
    Используется orchestrator'ом для выбора лучших кандидатов.

    LONG:  (bb_lower - signal_price) / bb_lower   → чем больше, тем дальше ниже полосы
    SHORT: (signal_price - bb_upper) / bb_upper   → чем больше, тем дальше выше полосы
    """
    if sig.direction == "LONG":
        return (sig.bb_lower - sig.signal_price) / sig.bb_lower
    else:
        return (sig.signal_price - sig.bb_upper) / sig.bb_upper
