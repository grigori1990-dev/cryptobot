"""
risk.py — Модуль управления риском
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ответственность:
  Из SignalResult + параметры счёта → рассчитать TradeParams или вернуть отказ.

Что НЕ делает:
  - Не обращается к бирже.
  - Не читает/пишет State напрямую — получает готовые данные от orchestrator.
  - Не знает про Telegram.
  - Не округляет qty под требования биржи (это notifier знает о форматировании).

Ключевые решения:
  - Зона входа рассчитана через ATR (адаптивно к волатильности).
  - Все параметры считаются по WORST-CASE краю зоны:
      LONG  → entry_worst = entry_high (самая дорогая цена в зоне)
      SHORT → entry_worst = entry_low  (самая дешёвая цена при шорте)
    Это гарантирует, что реальный риск не превысит расчётный,
    даже если пользователь войдёт по самой невыгодной цене в зоне.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal

from bot.signal import SignalResult

# ─── CONFIG ────────────────────────────────────────────────────────────────────
RISK_PCT              = 0.01    # 1% баланса на сделку
RR                    = 3.0     # Risk:Reward
ATR_STOP_MULT         = 1.5     # stop = ATR × ATR_STOP_MULT
ATR_TOLERANCE_MULT    = 0.15    # entry_zone = signal_price ± ATR × ATR_TOLERANCE_MULT
MIN_NOTIONAL_USDT     = 5.0     # минимальный размер позиции (игнорируем мелкие сигналы)


# ─── OUTPUT ────────────────────────────────────────────────────────────────────
@dataclass
class TradeParams:
    symbol:         str
    direction:      Literal["LONG", "SHORT"]
    # Зона входа
    entry_low:      float          # нижняя граница зоны
    entry_high:     float          # верхняя граница зоны
    entry_worst:    float          # worst-case цена (основа расчётов)
    # Уровни
    stop:           float
    target:         float
    rr:             float
    # Метрики
    stop_dist_pct:  float          # % риска от entry_worst
    tp_dist_pct:    float          # % прибыли от entry_worst
    # Позиция
    size_usdt:      float          # сколько USDT открывать
    risk_usdt:      float          # сколько теряем при стопе
    # Для Notifier
    atr:            float
    signal_price:   float


@dataclass
class RejectionResult:
    reason: str


# ─── PUBLIC API ────────────────────────────────────────────────────────────────
def calculate(
    sig:          SignalResult,
    balance_usdt: float,
    risk_pct:     float = RISK_PCT,
) -> TradeParams | RejectionResult:
    """
    Рассчитать параметры сделки из сигнала.

    Параметры:
        sig          — результат Signal.check()
        balance_usdt — свободный баланс на счёте (USDT)
        risk_pct     — доля баланса, которую рискуем (по умолчанию 1%)

    Возвращает TradeParams или RejectionResult с причиной отказа.
    """
    if balance_usdt <= 0:
        return RejectionResult("balance_zero")

    atr       = sig.atr
    sp        = sig.signal_price
    tolerance = atr * ATR_TOLERANCE_MULT
    stop_dist = atr * ATR_STOP_MULT

    if stop_dist <= 0 or math.isnan(stop_dist):
        return RejectionResult("atr_invalid")

    # Зона входа
    entry_low  = round(sp - tolerance, 4)
    entry_high = round(sp + tolerance, 4)

    # Worst-case и уровни
    if sig.direction == "LONG":
        entry_worst = entry_high                              # самый дорогой вход
        stop        = round(entry_worst - stop_dist, 4)
        target      = round(entry_worst + stop_dist * RR, 4)
    else:
        entry_worst = entry_low                               # самый дешёвый вход при шорте
        stop        = round(entry_worst + stop_dist, 4)
        target      = round(entry_worst - stop_dist * RR, 4)

    stop_dist_pct = round(stop_dist / entry_worst * 100, 3)
    tp_dist_pct   = round(stop_dist * RR / entry_worst * 100, 3)

    # Размер позиции
    risk_usdt = round(balance_usdt * risk_pct, 2)
    size_usdt = round(risk_usdt / (stop_dist_pct / 100), 2)

    if size_usdt < MIN_NOTIONAL_USDT:
        return RejectionResult(
            f"size_too_small: {size_usdt:.2f} USDT < min {MIN_NOTIONAL_USDT}"
        )

    return TradeParams(
        symbol        = sig.symbol,
        direction     = sig.direction,
        entry_low     = entry_low,
        entry_high    = entry_high,
        entry_worst   = entry_worst,
        stop          = stop,
        target        = target,
        rr            = RR,
        stop_dist_pct = stop_dist_pct,
        tp_dist_pct   = tp_dist_pct,
        size_usdt     = size_usdt,
        risk_usdt     = risk_usdt,
        atr           = atr,
        signal_price  = sp,
    )
