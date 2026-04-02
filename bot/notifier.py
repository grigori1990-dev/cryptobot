"""
notifier.py — Модуль уведомлений
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ответственность:
  1. Drift-проверка: текущая цена должна быть в зоне входа (ДО отправки).
  2. Форматирование Telegram-сообщений.
  3. Отправка в Telegram.

Что НЕ делает:
  - Не принимает торговых решений (только форматирует и валидирует актуальность).
  - Не пишет в State (это делает orchestrator после успешной отправки).
  - Не обращается к бирже за данными.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pytz
import requests

from bot.risk import TradeParams

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID        = os.getenv("CHAT_ID", "")
TBILISI_TZ     = pytz.timezone("Asia/Tbilisi")


# ─── OUTPUT ────────────────────────────────────────────────────────────────────
@dataclass
class NotifyResult:
    sent:   bool
    reason: str = ""   # "ok" | "drift" | "telegram_error"


# ─── VALIDATION ────────────────────────────────────────────────────────────────
def _in_entry_zone(current_price: float, params: TradeParams) -> bool:
    """
    Сигнал актуален только если текущая цена внутри зоны входа.
    Зона: [entry_low, entry_high] рассчитана Risk-модулем через ATR × tolerance.
    """
    return params.entry_low <= current_price <= params.entry_high


# ─── FORMATTING ────────────────────────────────────────────────────────────────
def _format(params: TradeParams, current_price: float) -> str:
    t    = datetime.now(TBILISI_TZ).strftime("%H:%M  %d.%m.%Y")
    emj  = "🟢" if params.direction == "LONG" else "🔴"
    dire = "LONG 📈" if params.direction == "LONG" else "SHORT 📉"

    # Название монеты из символа (BTC/USDT:USDT → BTC)
    coin = params.symbol.split("/")[0]

    # Форматировать числа: >= 100 → 1 decimal, < 1 → 4 decimals
    def fmt(v: float) -> str:
        if v >= 100:   return f"{v:,.1f}"
        if v >= 1:     return f"{v:.3f}"
        return f"{v:.5f}"

    # RR отображаем как "1:3"
    rr_str = f"1:{int(params.rr)}" if params.rr == int(params.rr) else f"1:{params.rr}"

    return (
        f"🚨 <b>СИГНАЛ: {coin} — {dire}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 Зона входа:    <b>{fmt(params.entry_low)} – {fmt(params.entry_high)}</b>\n"
        f"{emj} Сейчас:         <b>{fmt(current_price)}</b>\n"
        f"🛑 Стоп-лосс:    <b>{fmt(params.stop)}</b>  (−{params.stop_dist_pct}%)\n"
        f"🎯 Тейк-профит: <b>{fmt(params.target)}</b>  (+{params.tp_dist_pct}%)\n"
        f"⚖️ RR:              <b>{rr_str}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Размер позиции: ~${params.size_usdt:,.0f} при риске 1%\n"
        f"   (потери при стопе: ~${params.risk_usdt:.1f})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 RSI14: —  |  ATR: {fmt(params.atr)}\n"
        f"⚠️ Вход ТОЛЬКО в зоне {fmt(params.entry_low)}–{fmt(params.entry_high)}\n"
        f"   Если цена вышла за границы — сигнал недействителен.\n"
        f"🕐 {t}"
    )


def _send_raw(text: str) -> bool:
    """Отправить текст в Telegram. Возвращает True при успехе."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.ok
    except Exception:
        return False


# ─── PUBLIC API ────────────────────────────────────────────────────────────────
def validate_and_notify(params: TradeParams, current_price: float) -> NotifyResult:
    """
    Валидировать актуальность сигнала и отправить в Telegram.

    Drift-проверка выполняется ДО отправки:
      - Если цена вне зоны входа → NotifyResult(sent=False, reason="drift")
      - Если Telegram недоступен → NotifyResult(sent=False, reason="telegram_error")
      - Если успешно → NotifyResult(sent=True, reason="ok")
    """
    # Шаг 1: drift-проверка
    if not _in_entry_zone(current_price, params):
        drift_pct = abs(current_price - params.signal_price) / params.signal_price * 100
        return NotifyResult(
            sent   = False,
            reason = f"drift: price={current_price:.4f} "
                     f"zone=[{params.entry_low:.4f},{params.entry_high:.4f}] "
                     f"drift={drift_pct:.2f}%",
        )

    # Шаг 2: форматирование и отправка
    msg = _format(params, current_price)
    ok  = _send_raw(msg)

    return NotifyResult(sent=ok, reason="ok" if ok else "telegram_error")


def send_text(text: str) -> bool:
    """Отправить произвольный текст (статус, старт, алерты)."""
    return _send_raw(text)
