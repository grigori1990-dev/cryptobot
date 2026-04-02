"""
state.py — Модуль состояния
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ответственность:
  Единственный источник правды о текущем состоянии бота.
  Атомарное чтение/запись state.json.
  Восстановление после рестарта.

Что НЕ делает:
  - Не принимает торговых решений.
  - Не обращается к бирже или Telegram.
  - Не знает про индикаторы.

Структура state.json:
  {
    "global": {
      "signals_today": 2,
      "signals_today_date": "2026-04-02",
      "circuit_breaker": false
    },
    "symbols": {
      "BTC/USDT:USDT": {
        "last_bar_ts": 1712000000000,
        "cooldown_until": 1712014400000,
        "signal_history": [...]
      }
    }
  }

Атомарность:
  Запись через temp-файл + os.replace() — POSIX-гарантия атомарности.
  Если процесс упадёт во время записи — старый файл останется нетронутым.
"""

from __future__ import annotations
import json
import os
import time
from datetime import date, datetime, timezone
from typing import Any

STATE_PATH   = "state.json"
COOLDOWN_SEC = 4 * 3600   # 1 бар 4H = 4 часа


# ─── DEFAULTS ──────────────────────────────────────────────────────────────────
def _default_symbol() -> dict:
    return {
        "last_bar_ts":   0,
        "cooldown_until": 0,
        "signal_history": [],
    }


def _default_state(symbols: list[str]) -> dict:
    return {
        "global": {
            "signals_today":      0,
            "signals_today_date": str(date.today()),
            "circuit_breaker":    False,
        },
        "symbols": {s: _default_symbol() for s in symbols},
    }


# ─── IO ────────────────────────────────────────────────────────────────────────
def load(symbols: list[str]) -> dict:
    """Загрузить state.json. Если файла нет или он повреждён — вернуть default."""
    if not os.path.exists(STATE_PATH):
        return _default_state(symbols)
    try:
        with open(STATE_PATH) as f:
            state = json.load(f)
        # Убедиться что все символы присутствуют (могли добавиться новые)
        for s in symbols:
            if s not in state.get("symbols", {}):
                state.setdefault("symbols", {})[s] = _default_symbol()
        return state
    except Exception:
        return _default_state(symbols)


def save(state: dict) -> None:
    """Атомарная запись — через temp-файл + os.replace()."""
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


# ─── QUERIES ───────────────────────────────────────────────────────────────────
def is_daily_limit_reached(state: dict, max_signals: int = 3) -> bool:
    """Проверить, не исчерпан ли дневной лимит сигналов."""
    g = state["global"]
    # Сброс счётчика если наступил новый день
    if g["signals_today_date"] != str(date.today()):
        g["signals_today"]      = 0
        g["signals_today_date"] = str(date.today())
    return g["signals_today"] >= max_signals


def is_circuit_breaker(state: dict) -> bool:
    return state["global"].get("circuit_breaker", False)


def is_bar_processed(state: dict, symbol: str, bar_ts: int) -> bool:
    """Был ли этот бар уже обработан? Защита от двойного сигнала."""
    return state["symbols"].get(symbol, {}).get("last_bar_ts", 0) == bar_ts


def is_in_cooldown(state: dict, symbol: str) -> bool:
    """Символ в кулдауне после последнего сигнала?"""
    until = state["symbols"].get(symbol, {}).get("cooldown_until", 0)
    return time.time() < until


# ─── MUTATIONS ─────────────────────────────────────────────────────────────────
def mark_bar_processed(state: dict, symbol: str, bar_ts: int) -> None:
    """Отметить бар как обработанный (даже если сигнала не было)."""
    state["symbols"].setdefault(symbol, _default_symbol())["last_bar_ts"] = bar_ts


def record_signal_sent(state: dict, symbol: str, bar_ts: int,
                        direction: str, signal_price: float) -> None:
    """Записать факт отправки сигнала. Вызывается ПОСЛЕ успешной отправки в Telegram."""
    sym = state["symbols"].setdefault(symbol, _default_symbol())
    sym["last_bar_ts"]   = bar_ts
    sym["cooldown_until"] = time.time() + COOLDOWN_SEC
    sym["signal_history"].append({
        "ts":        int(time.time()),
        "bar_ts":    bar_ts,
        "direction": direction,
        "price":     signal_price,
    })
    # Хранить только последние 50 сигналов на символ
    if len(sym["signal_history"]) > 50:
        sym["signal_history"] = sym["signal_history"][-50:]

    g = state["global"]
    if g.get("signals_today_date") != str(date.today()):
        g["signals_today"]      = 0
        g["signals_today_date"] = str(date.today())
    g["signals_today"] = g.get("signals_today", 0) + 1


def set_circuit_breaker(state: dict, active: bool, reason: str = "") -> None:
    """Включить/выключить circuit breaker. Вызывается из analytics."""
    state["global"]["circuit_breaker"] = active
    if active:
        state["global"]["circuit_breaker_reason"] = reason
    else:
        state["global"].pop("circuit_breaker_reason", None)


# ─── CONSISTENCY CHECK (при старте) ────────────────────────────────────────────
def validate_on_startup(state: dict) -> list[str]:
    """
    Проверить консистентность state.json при старте.
    Возвращает список предупреждений.
    """
    warnings = []
    for symbol, sym_state in state.get("symbols", {}).items():
        until = sym_state.get("cooldown_until", 0)
        if until > 0 and until < time.time():
            # Кулдаун истёк пока бот не работал — сброс
            sym_state["cooldown_until"] = 0
            warnings.append(f"{symbol}: expired cooldown cleared")
    return warnings
