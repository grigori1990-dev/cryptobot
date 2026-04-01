#!/usr/bin/env python3
"""
Backtester — Hypothesis B: Mean Reversion (RSI + Bollinger Bands)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal:
  LONG  → RSI[i] < RSI_OVERSOLD  AND  close[i] <= BB_lower[i]
  SHORT → RSI[i] > RSI_OVERBOUGHT AND close[i] >= BB_upper[i]

Entry:  open of bar i+1  (closed-bar decision, no lookahead)
Stop:   entry ± ATR14 * ATR_MULT  (set at entry, never moved)
Target: entry ± stop_distance * RR

Anti-whipsaw: no re-entry while in same RSI zone (must reset to neutral first)

Costs:  commission 0.055% × 2 = 0.11%,  slippage 0.02%
        total per trade = 0.13%

Validation:
  In-sample   : first 70% of bars
  Out-of-sample: last 30% of bars
  Walk-forward: 3 windows on in-sample
"""

import time
import json
import ssl
import numpy as np
import urllib.request
from datetime import datetime, timezone

# ─── SSL (Python 3.14 on Mac) ─────────────────────────────────────────────────
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SYMBOLS       = ["BTCUSDT", "ETHUSDT"]
INTERVAL      = 240           # 4H in minutes
YEARS         = 2
COMMISSION    = 0.00055 * 2   # 0.11%
SLIPPAGE      = 0.0002        # 0.02%
TOTAL_COST    = COMMISSION + SLIPPAGE  # 0.13%

# Indicator params
RSI_PERIOD    = 14
RSI_OS        = 30            # oversold threshold (LONG signal)
RSI_OB        = 70            # overbought threshold (SHORT signal)
BB_PERIOD     = 20            # Bollinger period
BB_STD        = 2.0           # Bollinger std multiplier
ATR_PERIOD    = 14
ATR_MULT      = 1.5           # stop = ATR * ATR_MULT

RR_LIST       = [2.0, 3.0]
SPLIT_IS      = 0.70          # in-sample fraction

# ─── DATA DOWNLOAD ─────────────────────────────────────────────────────────────

def bybit_klines(symbol: str, interval: int, start_ms: int, end_ms: int) -> list:
    url_base = "https://api.bybit.com/v5/market/kline"
    all_candles = []
    cur_end = end_ms

    while True:
        params = (f"category=linear&symbol={symbol}"
                  f"&interval={interval}&limit=200"
                  f"&start={start_ms}&end={cur_end}")
        url = f"{url_base}?{params}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=20, context=_SSL_CTX) as resp:
                    data = json.loads(resp.read())
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

        rows = data.get("result", {}).get("list", [])
        if not rows:
            break

        for r in rows:
            ts = int(r[0])
            if start_ms <= ts <= end_ms:
                all_candles.append({"ts": ts, "open": float(r[1]),
                                    "high": float(r[2]), "low": float(r[3]),
                                    "close": float(r[4]), "volume": float(r[5])})

        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms:
            break
        cur_end = oldest_ts - 1
        time.sleep(0.15)

    all_candles.sort(key=lambda x: x["ts"])
    seen, unique = set(), []
    for c in all_candles:
        if c["ts"] not in seen:
            seen.add(c["ts"])
            unique.append(c)
    return unique


def download_symbol(symbol: str) -> list:
    now_ms   = int(time.time() * 1000)
    start_ms = now_ms - int(YEARS * 365.25 * 24 * 3600 * 1000)
    print(f"  Downloading {symbol}  {YEARS}y × 4H ...", flush=True)
    candles = bybit_klines(symbol, INTERVAL, start_ms, now_ms)
    print(f"  → {len(candles)} bars", flush=True)
    return candles


# ─── INDICATORS (all causal, no lookahead) ─────────────────────────────────────

def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI — causal."""
    n   = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    deltas = np.diff(close)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed: simple average of first `period` changes
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    # Seed value at bar `period`
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = gains[:period].mean() / losses[:period].mean() if losses[:period].mean() else float('inf')
        rsi[period] = 100.0 - 100.0 / (1.0 + rs) if losses[:period].mean() else 100.0

    return rsi


def calc_bb(close: np.ndarray, period: int = 20, std_mult: float = 2.0):
    """Bollinger Bands — causal (rolling window)."""
    n    = len(close)
    mid  = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close[i - period + 1: i + 1]
        m = window.mean()
        s = window.std(ddof=0)
        mid[i]   = m
        upper[i] = m + std_mult * s
        lower[i] = m - std_mult * s
    return lower, mid, upper


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
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


# ─── BACKTEST ──────────────────────────────────────────────────────────────────

def backtest(candles: list, rr: float, label: str = "") -> dict:
    n  = len(candles)
    op = np.array([c["open"]  for c in candles])
    hi = np.array([c["high"]  for c in candles])
    lo = np.array([c["low"]   for c in candles])
    cl = np.array([c["close"] for c in candles])

    rsi14       = calc_rsi(cl, RSI_PERIOD)
    bb_lo, _, bb_hi = calc_bb(cl, BB_PERIOD, BB_STD)
    atr14       = calc_atr(hi, lo, cl, ATR_PERIOD)

    trades = []
    in_trade_until = -1
    # Anti-whipsaw: track last RSI zone to require reset before re-entry
    last_rsi_zone = 0  # 0=neutral, 1=oversold-fired, -1=overbought-fired

    min_bar = max(RSI_PERIOD, BB_PERIOD, ATR_PERIOD) + 1

    for i in range(min_bar, n - 1):
        if i <= in_trade_until:
            continue

        if np.isnan(rsi14[i]) or np.isnan(bb_lo[i]) or np.isnan(atr14[i]):
            continue

        r = rsi14[i]

        # Reset zone when RSI returns to neutral band
        if last_rsi_zone == 1 and r > 45:
            last_rsi_zone = 0
        elif last_rsi_zone == -1 and r < 55:
            last_rsi_zone = 0

        direction = None
        if r <= RSI_OS and cl[i] <= bb_lo[i] and last_rsi_zone != 1:
            direction     = 1   # LONG
            last_rsi_zone = 1
        elif r >= RSI_OB and cl[i] >= bb_hi[i] and last_rsi_zone != -1:
            direction     = -1  # SHORT
            last_rsi_zone = -1

        if direction is None:
            continue

        entry_idx = i + 1
        if entry_idx >= n:
            break

        entry_raw = op[entry_idx]
        stop_dist = atr14[i] * ATR_MULT
        if stop_dist <= 0:
            continue

        if direction == 1:
            stop_price   = entry_raw - stop_dist
            target_price = entry_raw + stop_dist * rr
        else:
            stop_price   = entry_raw + stop_dist
            target_price = entry_raw - stop_dist * rr

        result   = None
        exit_idx = n - 1

        for j in range(entry_idx, n):
            lo_j, hi_j = lo[j], hi[j]
            if direction == 1:
                if lo_j <= stop_price:
                    result, exit_idx = "loss", j
                    raw_pnl_pct = -(stop_dist / entry_raw) - TOTAL_COST
                    break
                if hi_j >= target_price:
                    result, exit_idx = "win", j
                    raw_pnl_pct = (stop_dist * rr / entry_raw) - TOTAL_COST
                    break
            else:
                if hi_j >= stop_price:
                    result, exit_idx = "loss", j
                    raw_pnl_pct = -(stop_dist / entry_raw) - TOTAL_COST
                    break
                if lo_j <= target_price:
                    result, exit_idx = "win", j
                    raw_pnl_pct = (stop_dist * rr / entry_raw) - TOTAL_COST
                    break

        if result is None:
            continue  # trade still open — skip

        one_r_pct = stop_dist / entry_raw
        pnl_r     = raw_pnl_pct / one_r_pct

        trades.append({
            "signal_bar": i, "entry_bar": entry_idx, "exit_bar": exit_idx,
            "direction":  "LONG" if direction == 1 else "SHORT",
            "entry_price": entry_raw, "stop": stop_price, "target": target_price,
            "result":     result, "pnl_r": pnl_r, "stop_pct": one_r_pct,
            "rsi_at_signal": r,
            "ts_entry": candles[entry_idx]["ts"], "ts_exit": candles[exit_idx]["ts"],
        })
        in_trade_until = exit_idx

    return compute_metrics(trades, rr, label)


def compute_metrics(trades: list, rr: float, label: str) -> dict:
    if not trades:
        return {"label": label, "rr": rr, "n_trades": 0, "error": "no trades"}

    results  = [t["result"] for t in trades]
    pnl_list = [t["pnl_r"]  for t in trades]

    n      = len(trades)
    wins   = results.count("win")
    losses = results.count("loss")
    wr     = wins / n

    avg_win_r  = np.mean([p for p in pnl_list if p > 0]) if wins   else 0.0
    avg_loss_r = np.mean([p for p in pnl_list if p < 0]) if losses else 0.0
    expectancy = wr * avg_win_r + (1 - wr) * avg_loss_r

    max_cl, cur_cl = 0, 0
    for r in results:
        if r == "loss":
            cur_cl += 1; max_cl = max(max_cl, cur_cl)
        else:
            cur_cl = 0

    cum_r = np.cumsum(pnl_list)
    peak  = np.maximum.accumulate(cum_r)
    max_dd = (cum_r - peak).min()

    longs  = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]

    return {
        "label": label, "rr": rr, "n_trades": n,
        "n_wins": wins, "n_losses": losses, "win_rate": wr,
        "expectancy_r": expectancy, "avg_win_r": avg_win_r, "avg_loss_r": avg_loss_r,
        "total_r": sum(pnl_list), "max_consec_loss": max_cl, "max_drawdown_r": max_dd,
        "n_longs":  len(longs),
        "n_shorts": len(shorts),
        "wr_longs":  sum(1 for t in longs  if t["result"]=="win") / len(longs)  if longs  else 0,
        "wr_shorts": sum(1 for t in shorts if t["result"]=="win") / len(shorts) if shorts else 0,
        "trades": trades,
    }


def walk_forward(candles: list, rr: float, n_windows: int = 3) -> list:
    is_end     = int(len(candles) * SPLIT_IS)
    is_candles = candles[:is_end]
    wf_size    = len(is_candles) // n_windows
    results    = []
    for k in range(n_windows):
        start = k * wf_size
        end   = start + wf_size if k < n_windows - 1 else len(is_candles)
        results.append(backtest(is_candles[start:end], rr, label=f"WF-window-{k+1}"))
    return results


# ─── REPORT ────────────────────────────────────────────────────────────────────

def print_report(sym: str, candles: list):
    print(f"\n{'━'*62}")
    print(f"  SYMBOL: {sym}   bars: {len(candles)}")
    dt_f = datetime.fromtimestamp(candles[0]["ts"]  / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    dt_t = datetime.fromtimestamp(candles[-1]["ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"  Period: {dt_f} → {dt_t}")
    print(f"{'━'*62}")

    n_is = int(len(candles) * SPLIT_IS)
    is_c  = candles[:n_is]
    oos_c = candles[n_is:]

    for rr in RR_LIST:
        print(f"\n  ┌─ RR 1:{int(rr) if rr == int(rr) else rr} {'─'*48}┐")
        m_is  = backtest(is_c,  rr, "In-sample")
        m_oos = backtest(oos_c, rr, "Out-of-sample")

        for m, tag in [(m_is, "IN-SAMPLE "), (m_oos, "OUT-OF-SAMPLE")]:
            if "error" in m:
                print(f"  │  {tag}: NO TRADES")
                continue
            verdict = ("✅ TRADEABLE" if m["expectancy_r"] >= 0.2
                       else "⚠️  WEAK" if m["expectancy_r"] >= 0.1 else "❌ NO EDGE")
            print(f"  │  {tag}  n={m['n_trades']:>3}  "
                  f"WR={m['win_rate']*100:.1f}%  "
                  f"E={m['expectancy_r']:+.3f}R  "
                  f"TotalR={m['total_r']:+.2f}  "
                  f"MaxDD={m['max_drawdown_r']:.2f}R  "
                  f"MaxLoss={m['max_consec_loss']}  {verdict}")
            print(f"  │            Longs {m['n_longs']} WR={m['wr_longs']*100:.1f}%  "
                  f"Shorts {m['n_shorts']} WR={m['wr_shorts']*100:.1f}%")

        wf = walk_forward(candles, rr)
        print(f"  │  WALK-FORWARD ({len(wf)} windows):")
        for w in wf:
            if "error" in w:
                print(f"  │    {w['label']}: NO TRADES")
            else:
                v = ("✅" if w["expectancy_r"] >= 0.2
                     else "⚠️" if w["expectancy_r"] >= 0.1 else "❌")
                print(f"  │    {w['label']}: n={w['n_trades']:>3}  "
                      f"WR={w['win_rate']*100:.1f}%  E={w['expectancy_r']:+.3f}R  {v}")
        print(f"  └{'─'*57}┘")

    print(f"\n  DECISION THRESHOLDS:")
    print(f"    E ≥ 0.20R + stable OOS + MaxLoss ≤ 15 → BUILD LIVE BOT")
    print(f"    E  0.10–0.19R                          → OPTIMIZE")
    print(f"    E < 0.10R                              → DISCARD → try Hypothesis C")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  HYPOTHESIS B — Mean Reversion (RSI + Bollinger Bands)")
    print(f"  RSI period={RSI_PERIOD}  OS<{RSI_OS}  OB>{RSI_OB}")
    print(f"  BB period={BB_PERIOD}  std×{BB_STD}")
    print(f"  ATR mult={ATR_MULT}×  Commission={COMMISSION*100:.3f}%  "
          f"Slippage={SLIPPAGE*100:.2f}%")
    print(f"  In-sample: {int(SPLIT_IS*100)}%")
    print("=" * 62)

    for sym in SYMBOLS:
        candles = download_symbol(sym)
        if len(candles) < max(RSI_PERIOD, BB_PERIOD, ATR_PERIOD) + 50:
            print(f"  ⚠️  Not enough data for {sym}")
            continue
        print_report(sym, candles)

    print("\nDone.")


if __name__ == "__main__":
    main()
