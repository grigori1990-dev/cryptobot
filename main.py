#!/usr/bin/env python3
"""
КриптоБот v6.0 — Hypothesis B: RSI + Bollinger Bands (Mean Reversion)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Инструмент : BTC/USDT:USDT (перп. фьючерс Bybit)
Таймфрейм  : 4H
Сигнал LONG:  RSI14 < 30  И  close ≤ BB_lower(20, 2σ)
Сигнал SHORT: RSI14 > 70  И  close ≥ BB_upper(20, 2σ)
Вход      : открытие следующего 4H бара (рыночный ордер)
Стоп      : ATR14 × 1.5 от цены входа
Тейк      : стоп-дистанция × 3  (RR = 1:3)
Бэктест   : E = +0.38R IS, WF 3/3 окна зелёные. OOS +0.07R.
"""

import asyncio
import os
import time
import sqlite3
import math
import numpy as np
import requests
import pytz
import ccxt.async_support as ccxt
from datetime import datetime

# ─── КОНФИГ ────────────────────────────────────────────────────────────────────
SYMBOL          = "BTC/USDT:USDT"
TIMEFRAME       = "4h"
CANDLES         = 250          # для прогрева RSI(14)+BB(20)+ATR(14)

RSI_PERIOD      = 14
RSI_OS          = 30           # oversold → LONG
RSI_OB          = 70           # overbought → SHORT
BB_PERIOD       = 20
BB_STD          = 2.0
ATR_PERIOD      = 14
ATR_MULT        = 1.5          # stop = ATR × ATR_MULT
RR              = 3.0          # take profit = stop_dist × RR

SIGNAL_COOLDOWN = 4 * 3600     # минимум 4 часа между сигналами (1 бар)
LOOP_INTERVAL   = 60           # проверка каждые 60 сек

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID         = os.getenv("CHAT_ID", "")
TBILISI_TZ      = pytz.timezone("Asia/Tbilisi")
DB_PATH         = "signals.db"

VERSION         = "v6.0"

# ─── TELEGRAM ──────────────────────────────────────────────────────────────────

def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"[TG ERR] {e}", flush=True)

# ─── БАЗА ДАННЫХ ────────────────────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        INTEGER,
            direction TEXT,
            entry_est REAL,
            stop      REAL,
            target    REAL,
            rsi       REAL,
            atr       REAL
        )
    """)
    con.commit()
    con.close()


def save_signal(sig: dict):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO signals (ts,direction,entry_est,stop,target,rsi,atr) "
        "VALUES (?,?,?,?,?,?,?)",
        (int(time.time()), sig["direction"], sig["entry_est"],
         sig["stop"], sig["target"], sig["rsi"], sig["atr"]),
    )
    con.commit()
    con.close()


def last_signal_time() -> float:
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT ts FROM signals ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return float(row[0]) if row else 0.0

# ─── ИНДИКАТОРЫ ────────────────────────────────────────────────────────────────

def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI — каузальный, без lookahead."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    d = np.diff(close)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = g[:period].mean()
    al = l[:period].mean()
    out[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(period, n - 1):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
        out[i + 1] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def calc_bb(close: np.ndarray, period: int = 20, std: float = 2.0):
    """Bollinger Bands — каузальный."""
    n = len(close)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = close[i - period + 1: i + 1]
        m = w.mean()
        s = w.std(ddof=0)
        lo[i] = m - std * s
        hi[i] = m + std * s
    return lo, hi


def calc_atr(high: np.ndarray, low: np.ndarray,
             close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's ATR."""
    n = len(high)
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

# ─── ЛОГИКА СИГНАЛА ────────────────────────────────────────────────────────────

def check_signal(highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray) -> dict | None:
    """
    Проверяем последний ЗАКРЫТЫЙ бар (index -1).
    Решение принимается только на закрытом баре — без lookahead.
    """
    rsi14     = calc_rsi(closes, RSI_PERIOD)
    bb_lo, bb_hi = calc_bb(closes, BB_PERIOD, BB_STD)
    atr14     = calc_atr(highs, lows, closes, ATR_PERIOD)

    r   = rsi14[-1]
    cl  = closes[-1]
    blo = bb_lo[-1]
    bhi = bb_hi[-1]
    a   = atr14[-1]

    if math.isnan(r) or math.isnan(blo) or math.isnan(a):
        return None

    direction = None
    if r <= RSI_OS and cl <= blo:
        direction = "LONG"
    elif r >= RSI_OB and cl >= bhi:
        direction = "SHORT"

    if direction is None:
        return None

    # Оценка входа: текущий close (пользователь входит на открытии следующего бара)
    entry_est = cl
    stop_dist = a * ATR_MULT

    if direction == "LONG":
        stop   = round(entry_est - stop_dist, 1)
        target = round(entry_est + stop_dist * RR, 1)
    else:
        stop   = round(entry_est + stop_dist, 1)
        target = round(entry_est - stop_dist * RR, 1)

    sl_pct = round(stop_dist / entry_est * 100, 2)
    tp_pct = round(stop_dist * RR / entry_est * 100, 2)

    return {
        "direction": direction,
        "entry_est": round(entry_est, 1),
        "stop":      stop,
        "target":    target,
        "sl_pct":    sl_pct,
        "tp_pct":    tp_pct,
        "rsi":       round(r, 1),
        "atr":       round(a, 1),
    }


def format_signal(sig: dict) -> str:
    t    = datetime.now(TBILISI_TZ).strftime("%H:%M  %d.%m.%Y")
    dire = "LONG 📈" if sig["direction"] == "LONG" else "SHORT 📉"
    emj  = "🟢" if sig["direction"] == "LONG" else "🔴"
    return (
        f"🚨 <b>СИГНАЛ: BTC — {dire}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{emj} Вход (откр. след. 4H бара): <b>{sig['entry_est']}</b>\n"
        f"🛑 Стоп-лосс: <b>{sig['stop']}</b>  (−{sig['sl_pct']}%)\n"
        f"🎯 Тейк-профит: <b>{sig['target']}</b>  (+{sig['tp_pct']}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 RSI14: {sig['rsi']}  |  ATR: {sig['atr']}\n"
        f"⚡ Стратегия: RSI+BB Mean Reversion  RR 1:3\n"
        f"🕐 {t}"
    )

# ─── RETRY-ОБЁРТКА ─────────────────────────────────────────────────────────────

async def _retry(coro_fn, *args, attempts=3):
    import random
    for i in range(attempts):
        try:
            return await coro_fn(*args)
        except Exception as e:
            if i == attempts - 1:
                raise
            delay = (i + 1) + random.uniform(-0.3, 0.3)
            await asyncio.sleep(delay)

# ─── ОСНОВНОЙ ЦИКЛ ─────────────────────────────────────────────────────────────

async def main():
    init_db()

    exchange = ccxt.bybit({
        "options":      {"defaultType": "linear"},
        "enableRateLimit": True,
    })

    send_telegram(
        f"🟢 <b>КриптоБот {VERSION} запущен</b>\n"
        f"Стратегия: RSI+BB Mean Reversion  4H\n"
        f"Инструмент: BTC/USDT:USDT\n"
        f"RR: 1:{int(RR)}  |  Stop: {ATR_MULT}×ATR\n"
        f"Жду закрытия 4H баров..."
    )
    print(f"КриптоБот {VERSION} запущен.", flush=True)

    last_bar_ts  = 0   # timestamp последнего обработанного бара
    last_sig_log = 0   # последний stat-лог (раз в час)

    try:
        while True:
            try:
                # Получаем 4H свечи (последние CANDLES штук)
                ohlcv = await _retry(
                    exchange.fetch_ohlcv, SYMBOL, TIMEFRAME, None, CANDLES
                )

                if len(ohlcv) < 50:
                    await asyncio.sleep(LOOP_INTERVAL)
                    continue

                # ohlcv[-1] = текущий ОТКРЫТЫЙ бар
                # ohlcv[-2] = последний ЗАКРЫТЫЙ бар
                closed_bar_ts = ohlcv[-2][0]  # миллисекунды

                if closed_bar_ts != last_bar_ts:
                    last_bar_ts = closed_bar_ts

                    # Берём только закрытые бары (исключаем текущий открытый)
                    bars = ohlcv[:-1]
                    highs  = np.array([c[2] for c in bars])
                    lows   = np.array([c[3] for c in bars])
                    closes = np.array([c[4] for c in bars])

                    sig = check_signal(highs, lows, closes)

                    if sig:
                        # Кулдаун: не больше одного сигнала за 4 часа
                        elapsed = time.time() - last_signal_time()
                        if elapsed >= SIGNAL_COOLDOWN:
                            msg = format_signal(sig)
                            send_telegram(msg)
                            save_signal(sig)
                            print(f"[СИГНАЛ] {sig['direction']}  RSI={sig['rsi']}  "
                                  f"Entry≈{sig['entry_est']}  SL={sig['stop']}  "
                                  f"TP={sig['target']}", flush=True)
                        else:
                            print(f"[КУЛДАУН] сигнал есть, но прошло только "
                                  f"{elapsed/3600:.1f}ч (нужно 4ч)", flush=True)
                    else:
                        bar_dt = datetime.fromtimestamp(
                            closed_bar_ts / 1000, tz=TBILISI_TZ
                        ).strftime("%H:%M %d.%m")
                        print(f"[ОК] Новый бар {bar_dt} — "
                              f"RSI={round(calc_rsi(closes)[-1], 1)}  "
                              f"нет сигнала", flush=True)

                # Статус-лог раз в час
                now = time.time()
                if now - last_sig_log >= 3600:
                    last_sig_log = now
                    rsi_val = round(calc_rsi(
                        np.array([c[4] for c in ohlcv[:-1]])
                    )[-1], 1)
                    t = datetime.now(TBILISI_TZ).strftime("%H:%M")
                    send_telegram(
                        f"📊 Статус | {t}\n"
                        f"BTC RSI14: {rsi_val}\n"
                        f"{'🔥 Близко к сигналу!' if rsi_val < 35 or rsi_val > 65 else '⏳ Ждём экстремума'}"
                    )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[ERR] {e}", flush=True)
                await asyncio.sleep(10)

            await asyncio.sleep(LOOP_INTERVAL)

    finally:
        await exchange.close()


# ─── ЗАПУСК ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"КриптоБот {VERSION} — запуск...", flush=True)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Остановлен вручную.", flush=True)
    except Exception as _err:
        import traceback as _tb
        _tb.print_exc()
        if TELEGRAM_TOKEN and CHAT_ID:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    json={"chat_id": CHAT_ID,
                          "text": f"💀 Бот упал:\n{str(_err)[:300]}",
                          "parse_mode": "HTML"},
                    timeout=10,
                )
            except Exception:
                pass
        raise
