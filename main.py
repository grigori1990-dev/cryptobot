#!/usr/bin/env python3
"""
КриптоБот v7.0 — Модульная архитектура
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Стратегия  : Hypothesis B — RSI14 + Bollinger Bands (Mean Reversion)
Таймфрейм  : 4H
Инструменты: топ-12 по ликвидности (конфиг ниже)
Сигналы    : только в Telegram, вход вручную

Модули:
  bot/signal.py    — RSI+BB индикаторы и детекция сигнала
  bot/risk.py      — зона входа, стоп, тейк, размер позиции
  bot/notifier.py  — drift-проверка + Telegram
  bot/state.py     — атомарный state.json, кулдауны, лимиты
  bot/analytics.py — метрики, часовой статус, circuit breaker

Flow одного цикла (60 сек):
  1. Глобальные предусловия (лимит, circuit breaker)
  2. Сбор кандидатов по 12 символам (батчи по 4, semaphore)
  3. Noise filter: >5 сигналов → skip all
  4. Сортировка по BB-overshoot, обрезка до дневного лимита
  5. Для каждого: Risk → drift → Telegram → State.save()
  6. Аналитика (hourly status)
"""

import asyncio
import os
import time

import numpy as np
import ccxt.async_support as ccxt

from bot import signal as sig_mod
from bot import risk   as risk_mod
from bot import state  as state_mod
from bot import notifier
from bot import analytics

# ─── КОНФИГ ────────────────────────────────────────────────────────────────────
SYMBOLS: list[str] = [
    # Tier 1 — максимальная ликвидность
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    # Tier 2 — высокая ликвидность
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "LTC/USDT:USDT",
    "DOT/USDT:USDT",
    "MATIC/USDT:USDT",
]

TIMEFRAME       = "4h"
CANDLES         = 250          # баров для прогрева индикаторов
LOOP_INTERVAL   = 60           # пауза между циклами (сек)
BATCH_SIZE      = 4            # символов в одном батче API
BATCH_DELAY     = 0.2          # сек между батчами
MAX_SIGNALS_DAY = 3            # дневной лимит сигналов
NOISE_THRESHOLD = 5            # >N сигналов в цикле → skip all
BALANCE_USDT    = float(os.getenv("BALANCE_USDT", "500"))  # баланс для Risk
VERSION         = "v7.0"


# ─── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ───────────────────────────────────────────────────
async def _retry(coro, *args, attempts: int = 3):
    """Повторить async-вызов до attempts раз с экспоненциальной паузой."""
    import random
    for i in range(attempts):
        try:
            return await coro(*args)
        except Exception as e:
            if i == attempts - 1:
                raise
            await asyncio.sleep((i + 1) + random.uniform(-0.2, 0.2))


async def _fetch_symbol(
    symbol: str,
    exchange: ccxt.bybit,
    semaphore: asyncio.Semaphore,
    state: dict,
) -> sig_mod.SignalResult | None:
    """
    Получить OHLCV для символа и проверить сигнал.
    Возвращает SignalResult или None.
    Ошибки одного символа не прерывают обработку остальных.
    """
    async with semaphore:
        try:
            ohlcv = await _retry(
                exchange.fetch_ohlcv, symbol, TIMEFRAME, None, CANDLES
            )
        except Exception as e:
            print(f"[FETCH ERR] {symbol}: {e}", flush=True)
            return None

    if len(ohlcv) < 50:
        return None

    # ohlcv[-1] = текущий ОТКРЫТЫЙ бар → исключаем
    # ohlcv[-2] = последний ЗАКРЫТЫЙ бар
    closed  = ohlcv[:-1]
    bar_ts  = closed[-1][0]   # timestamp последнего закрытого бара

    # Пропустить если бар уже обработан или символ в кулдауне
    if state_mod.is_bar_processed(state, symbol, bar_ts):
        return None
    if state_mod.is_in_cooldown(state, symbol):
        coin = symbol.split("/")[0]
        print(f"[COOLDOWN] {coin}", flush=True)
        state_mod.mark_bar_processed(state, symbol, bar_ts)
        return None

    highs  = np.array([c[2] for c in closed])
    lows   = np.array([c[3] for c in closed])
    closes = np.array([c[4] for c in closed])

    result = sig_mod.check(symbol, highs, lows, closes, bar_ts)
    # Всегда помечаем бар как обработанный (чтобы не перепроверять)
    state_mod.mark_bar_processed(state, symbol, bar_ts)

    if result:
        coin = symbol.split("/")[0]
        print(f"[SIGNAL] {coin} {result.direction}  "
              f"RSI={result.rsi}  "
              f"overshoot={sig_mod.bb_overshoot(result)*100:.2f}%", flush=True)

    return result


async def _get_current_price(symbol: str, exchange: ccxt.bybit) -> float | None:
    """Лёгкий запрос текущей цены (bid/ask mid)."""
    try:
        ticker = await _retry(exchange.fetch_ticker, symbol)
        bid    = ticker.get("bid") or ticker.get("last")
        ask    = ticker.get("ask") or ticker.get("last")
        if bid and ask:
            return (bid + ask) / 2
        return ticker.get("last")
    except Exception as e:
        print(f"[TICKER ERR] {symbol}: {e}", flush=True)
        return None


# ─── ОСНОВНОЙ ЦИКЛ ─────────────────────────────────────────────────────────────
async def main_loop(exchange: ccxt.bybit):
    state = state_mod.load(SYMBOLS)

    # Проверка консистентности при старте
    warnings = state_mod.validate_on_startup(state)
    for w in warnings:
        print(f"[STARTUP] {w}", flush=True)
    state_mod.save(state)

    notifier.send_text(
        f"🟢 <b>КриптоБот {VERSION} запущен</b>\n"
        f"Стратегия: RSI14 + Bollinger Bands  4H\n"
        f"Инструменты: {len(SYMBOLS)} монет\n"
        f"Лимит: {MAX_SIGNALS_DAY} сигнала/день  |  Кулдаун: 4ч\n"
        f"Noise filter: >{NOISE_THRESHOLD} сигналов в цикле → пропуск\n"
        f"Ожидаю закрытия 4H баров..."
    )
    print(f"КриптоБот {VERSION} запущен. {len(SYMBOLS)} символов.", flush=True)

    semaphore = asyncio.Semaphore(BATCH_SIZE)

    while True:
        cycle_start = time.time()

        # ── Шаг 0: Глобальные предусловия ─────────────────────────────────────
        state = state_mod.load(SYMBOLS)   # перечитываем на каждом цикле

        if state_mod.is_circuit_breaker(state):
            print("[CIRCUIT BREAKER] торговля остановлена", flush=True)
            await asyncio.sleep(LOOP_INTERVAL)
            continue

        if state_mod.is_daily_limit_reached(state, MAX_SIGNALS_DAY):
            print("[LIMIT] дневной лимит сигналов исчерпан", flush=True)
            analytics.maybe_send_hourly_status(state, SYMBOLS)
            await asyncio.sleep(LOOP_INTERVAL)
            continue

        # ── Шаг 1: Сбор кандидатов ────────────────────────────────────────────
        tasks      = [_fetch_symbol(s, exchange, semaphore, state) for s in SYMBOLS]
        results    = await asyncio.gather(*tasks, return_exceptions=False)
        candidates = [r for r in results if r is not None]

        # Сохраняем last_bar_ts для всех обработанных символов
        state_mod.save(state)

        if not candidates:
            print("[OK] нет сигналов в этом цикле", flush=True)
            analytics.maybe_send_hourly_status(state, SYMBOLS)
            await asyncio.sleep(LOOP_INTERVAL)
            continue

        # ── Шаг 2: Noise filter ───────────────────────────────────────────────
        if len(candidates) > NOISE_THRESHOLD:
            print(
                f"[NOISE FILTER] {len(candidates)} сигналов в цикле → skip all. "
                f"Рынок коррелирует, пропускаем.", flush=True
            )
            analytics.maybe_send_hourly_status(state, SYMBOLS)
            await asyncio.sleep(LOOP_INTERVAL)
            continue

        # ── Шаг 3: Ранжирование по BB-overshoot ──────────────────────────────
        candidates.sort(key=sig_mod.bb_overshoot, reverse=True)

        slots = MAX_SIGNALS_DAY - state["global"].get("signals_today", 0)
        selected = candidates[:slots]

        print(f"[CANDIDATES] {len(candidates)} сигналов, "
              f"выбираем {len(selected)} (slots={slots})", flush=True)

        # ── Шаг 4: Risk → Notifier → State ───────────────────────────────────
        for sig in selected:
            coin = sig.symbol.split("/")[0]

            # Risk
            params = risk_mod.calculate(sig, BALANCE_USDT)
            if isinstance(params, risk_mod.RejectionResult):
                print(f"[RISK REJECT] {coin}: {params.reason}", flush=True)
                continue

            # Текущая цена (для drift-проверки)
            await asyncio.sleep(BATCH_DELAY)
            cur_price = await _get_current_price(sig.symbol, exchange)
            if cur_price is None:
                print(f"[PRICE ERR] {coin}: не удалось получить цену", flush=True)
                continue

            # Notifier (drift-проверка + Telegram)
            result = notifier.validate_and_notify(params, cur_price)

            if result.sent:
                # Атомарно записываем ПОСЛЕ успешной отправки
                state_mod.record_signal_sent(
                    state, sig.symbol, sig.bar_ts,
                    sig.direction, sig.signal_price
                )
                state_mod.save(state)
                analytics.record_signal(sig.symbol, sig.direction)
                print(f"[SENT] {coin} {sig.direction}  "
                      f"zone=[{params.entry_low:.4f},{params.entry_high:.4f}]  "
                      f"SL={params.stop}  TP={params.target}", flush=True)

                # Проверить дневной лимит после отправки
                if state_mod.is_daily_limit_reached(state, MAX_SIGNALS_DAY):
                    print("[LIMIT] дневной лимит достигнут", flush=True)
                    break
            else:
                print(f"[SKIP] {coin}: {result.reason}", flush=True)

        # ── Шаг 5: Аналитика ─────────────────────────────────────────────────
        analytics.maybe_send_hourly_status(state, SYMBOLS)

        elapsed = time.time() - cycle_start
        sleep_s = max(0, LOOP_INTERVAL - elapsed)
        await asyncio.sleep(sleep_s)


# ─── ЗАПУСК ────────────────────────────────────────────────────────────────────
async def main():
    exchange = ccxt.bybit({
        "options":         {"defaultType": "linear"},
        "enableRateLimit": True,
    })
    try:
        await main_loop(exchange)
    finally:
        await exchange.close()


if __name__ == "__main__":
    import requests as _req

    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    CHAT_ID        = os.getenv("CHAT_ID", "")

    print(f"КриптоБот {VERSION} — запуск...", flush=True)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Остановлен вручную.", flush=True)
    except Exception as _err:
        import traceback
        traceback.print_exc()
        if TELEGRAM_TOKEN and CHAT_ID:
            try:
                _req.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    json={"chat_id": CHAT_ID,
                          "text": f"💀 Бот упал:\n{str(_err)[:400]}",
                          "parse_mode": "HTML"},
                    timeout=10,
                )
            except Exception:
                pass
        raise
