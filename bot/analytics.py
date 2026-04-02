"""
analytics.py — Модуль аналитики
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ответственность:
  - Читать историю сигналов из State.
  - Формировать метрики (WR, Expectancy, consecutive signals).
  - Отправлять периодические отчёты в Telegram.
  - ЕДИНСТВЕННОЕ исключение из правила "только читает":
    если consecutive_signals_without_result >= CIRCUIT_BREAKER_THRESHOLD
    → выставить circuit_breaker=True в State.

Что НЕ делает:
  - Не принимает торговых решений (кроме circuit_breaker).
  - Не знает про Bybit API.
  - Не рассчитывает P&L (бот не знает фактических цен входа/выхода,
    только рекомендовал зону — торговал пользователь вручную).

Примечание о метриках:
  Поскольку бот сигнальный, мы не знаем:
    - По какой цене в зоне вошёл пользователь
    - Соблюдал ли он стоп и тейк
  Поэтому analytics считает только то, что знает точно:
    - Количество отправленных сигналов
    - Распределение по символам и направлениям
    - Частоту (сигналов в день/неделю)
  Все P&L данные пользователь добавляет вручную (будущий функционал).
"""

from __future__ import annotations
import json
import os
import time
from datetime import datetime, date, timezone
from collections import Counter

import pytz

from bot import notifier, state as state_module

ANALYTICS_PATH            = "analytics.json"
CIRCUIT_BREAKER_THRESHOLD = 0     # резерв — пока отключён (нет P&L данных)
HOURLY_STATUS_INTERVAL    = 3600  # статус раз в час

TBILISI_TZ = pytz.timezone("Asia/Tbilisi")

# ─── STATE ─────────────────────────────────────────────────────────────────────
_last_hourly_ts: float = 0.0


# ─── ANALYTICS STORAGE ─────────────────────────────────────────────────────────
def _load_analytics() -> dict:
    if not os.path.exists(ANALYTICS_PATH):
        return {"total_signals": 0, "by_symbol": {}, "by_direction": {}, "daily": {}}
    try:
        with open(ANALYTICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {"total_signals": 0, "by_symbol": {}, "by_direction": {}, "daily": {}}


def _save_analytics(data: dict) -> None:
    tmp = ANALYTICS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, ANALYTICS_PATH)


# ─── PUBLIC API ────────────────────────────────────────────────────────────────
def record_signal(symbol: str, direction: str) -> None:
    """Записать факт отправленного сигнала. Вызывается orchestrator'ом."""
    data  = _load_analytics()
    today = str(date.today())

    data["total_signals"] = data.get("total_signals", 0) + 1
    data.setdefault("by_symbol", {}).setdefault(symbol, 0)
    data["by_symbol"][symbol] += 1
    data.setdefault("by_direction", {}).setdefault(direction, 0)
    data["by_direction"][direction] += 1
    data.setdefault("daily", {}).setdefault(today, 0)
    data["daily"][today] += 1

    _save_analytics(data)


def maybe_send_hourly_status(state: dict, symbols: list[str]) -> None:
    """
    Отправить часовой статус-отчёт если прошёл час.
    Показывает текущий RSI по каждому символу из State.
    Вызывается orchestrator'ом в конце каждого цикла.
    """
    global _last_hourly_ts
    if time.time() - _last_hourly_ts < HOURLY_STATUS_INTERVAL:
        return
    _last_hourly_ts = time.time()

    data      = _load_analytics()
    t         = datetime.now(TBILISI_TZ).strftime("%H:%M  %d.%m.%Y")
    today_cnt = data.get("daily", {}).get(str(date.today()), 0)
    total_cnt = data.get("total_signals", 0)
    cb        = "🔴 ВКЛЮЧЁН" if state_module.is_circuit_breaker(state) else "🟢 активен"

    lines = [
        f"📊 <b>Статус-отчёт</b> | {t}",
        f"Бот: {cb}",
        f"Сигналов сегодня: {today_cnt}/3  |  Всего: {total_cnt}",
        f"",
        f"Ожидаем следующего сигнала...",
    ]

    # Кулдауны по символам
    in_cooldown = []
    for sym in symbols:
        if state_module.is_in_cooldown(state, sym):
            until = state["symbols"].get(sym, {}).get("cooldown_until", 0)
            mins  = max(0, int((until - time.time()) / 60))
            coin  = sym.split("/")[0]
            in_cooldown.append(f"  {coin}: кулдаун ещё {mins} мин")

    if in_cooldown:
        lines.append("⏳ В кулдауне:")
        lines.extend(in_cooldown)

    notifier.send_text("\n".join(lines))


def send_weekly_report() -> None:
    """Еженедельный отчёт по отправленным сигналам."""
    data = _load_analytics()
    t    = datetime.now(TBILISI_TZ).strftime("%d.%m.%Y")

    by_sym = data.get("by_symbol", {})
    top    = sorted(by_sym.items(), key=lambda x: x[1], reverse=True)[:5]

    lines  = [
        f"📅 <b>Недельный отчёт</b> | {t}",
        f"Всего сигналов: {data.get('total_signals', 0)}",
        f"",
        f"Топ-5 активных монет:",
    ]
    for sym, cnt in top:
        coin = sym.split("/")[0]
        lines.append(f"  {coin}: {cnt}")

    by_dir = data.get("by_direction", {})
    longs  = by_dir.get("LONG", 0)
    shorts = by_dir.get("SHORT", 0)
    total  = longs + shorts
    if total:
        lines.append(f"")
        lines.append(f"LONG:  {longs} ({longs/total*100:.0f}%)")
        lines.append(f"SHORT: {shorts} ({shorts/total*100:.0f}%)")

    notifier.send_text("\n".join(lines))
