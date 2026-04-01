#!/usr/bin/env python3
"""
Backtester — Hypothesis A: Trend-following (EMA200 + MACD crossover)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal:
  LONG  → close[i] > EMA200[i]  AND  MACD crosses ABOVE signal at bar i
  SHORT → close[i] < EMA200[i]  AND  MACD crosses BELOW signal at bar i

Entry:  open of bar i+1  (closed-bar decision, no lookahead)
Stop:   entry ± ATR14 * ATR_MULT  (set at entry, never moved)
Target: entry ± stop_distance * RR

Costs:  commission 0.055% × 2 = 0.11%,  slippage 0.02%
        total per trade = 0.13%

Validation:
  In-sample   : first 70% of bars
  Out-of-sample: last 30% of bars
"""

import time
import json
import math
import ssl
import numpy as np
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

# SSL context: disable verification for public market data APIs
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SYMBOLS      = ["BTCUSDT", "ETHUSDT"]
INTERVAL     = 240          # 4H in minutes
YEARS        = 2
COMMISSION   = 0.00055 * 2  # 0.11%
SLIPPAGE     = 0.0002       # 0.02%
TOTAL_COST   = COMMISSION + SLIPPAGE  # 0.13%

EMA_PERIOD   = 200
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIG     = 9
ATR_PERIOD   = 14
ATR_MULT     = 1.5          # stop = ATR * ATR_MULT

RR_LIST      = [2.0, 3.0]

SPLIT_IS     = 0.70         # in-sample fraction

# ─── DATA DOWNLOAD ─────────────────────────────────────────────────────────────

def bybit_klines(symbol: str, interval: int, start_ms: int, end_ms: int) -> list:
    """Download all 4H candles between start_ms and end_ms from Bybit v5 REST API."""
    url_base = "https://api.bybit.com/v5/market/kline"
    all_candles = []
    cur_end = end_ms

    while True:
        params = (
            f"category=linear&symbol={symbol}"
            f"&interval={interval}&limit=200"
            f"&start={start_ms}&end={cur_end}"
        )
        url = f"{url_base}?{params}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=20, context=_SSL_CTX) as resp:
                    data = json.loads(resp.read())
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

        rows = data.get("result", {}).get("list", [])
        if not rows:
            break

        # rows: [[ts, open, high, low, close, volume, turnover], ...]  descending
        for r in rows:
            ts = int(r[0])
            if ts < start_ms or ts > end_ms:
                continue
            all_candles.append({
                "ts":     ts,
                "open":   float(r[1]),
                "high":   float(r[2]),
                "low":    float(r[3]),
                "close":  float(r[4]),
                "volume": float(r[5]),
            })

        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms:
            break
        cur_end = oldest_ts - 1
        time.sleep(0.15)  # be nice to API

    # Sort ascending by timestamp
    all_candles.sort(key=lambda x: x["ts"])
    # Deduplicate
    seen = set()
    unique = []
    for c in all_candles:
        if c["ts"] not in seen:
            seen.add(c["ts"])
            unique.append(c)
    return unique


def download_symbol(symbol: str) -> dict:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(YEARS * 365.25 * 24 * 3600 * 1000)
    print(f"  Downloading {symbol}  {YEARS}y × 4H ...", flush=True)
    candles = bybit_klines(symbol, INTERVAL, start_ms, now_ms)
    print(f"  → {len(candles)} bars", flush=True)
    return candles


# ─── INDICATORS ────────────────────────────────────────────────────────────────

def calc_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA — computed forward, strictly causal (no lookahead)."""
    out = np.full(len(arr), np.nan)
    # Seed with SMA of first `period` values
    if len(arr) < period:
        return out
    out[period - 1] = arr[:period].mean()
    k = 2.0 / (period + 1)
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def calc_macd(close: np.ndarray, fast=12, slow=26, sig=9):
    """MACD line, signal line, histogram — all causal."""
    ema_fast   = calc_ema(close, fast)
    ema_slow   = calc_ema(close, slow)
    macd_line  = ema_fast - ema_slow          # NaN until bar slow-1
    signal_line = calc_ema(macd_line[~np.isnan(macd_line)], sig)
    # Re-align signal_line into full-length array
    full_signal = np.full(len(close), np.nan)
    valid_idx   = np.where(~np.isnan(macd_line))[0]
    sig_start   = valid_idx[0] + sig - 1
    if sig_start < len(close):
        full_signal[sig_start:] = signal_line[sig - 1:]
    return macd_line, full_signal


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> np.ndarray:
    """Wilder's ATR."""
    n = len(high)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        hl  = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i]  - close[i - 1])
        tr[i] = max(hl, hpc, lpc)
    tr[0] = high[0] - low[0]

    atr = np.full(n, np.nan)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ─── BACKTEST ──────────────────────────────────────────────────────────────────

def backtest(candles: list, rr: float, label: str = "") -> dict:
    """
    Run Hypothesis A backtest on given candles.
    Returns dict with metrics.
    """
    n = len(candles)
    op  = np.array([c["open"]  for c in candles])
    hi  = np.array([c["high"]  for c in candles])
    lo  = np.array([c["low"]   for c in candles])
    cl  = np.array([c["close"] for c in candles])

    # Indicators (fully causal — computed over all bars, but each bar[i]
    # only uses data up to i inclusive)
    ema200      = calc_ema(cl, EMA_PERIOD)
    macd_line, macd_sig = calc_macd(cl, MACD_FAST, MACD_SLOW, MACD_SIG)
    atr14       = calc_atr(hi, lo, cl, ATR_PERIOD)

    trades = []
    in_trade_until = -1   # bar index when current trade closes (no overlap)

    min_bar = max(EMA_PERIOD, MACD_SLOW + MACD_SIG) + 1  # warm-up

    for i in range(min_bar, n - 1):
        if i <= in_trade_until:
            continue  # still in previous trade

        # Valid indicator data check
        if np.isnan(ema200[i]) or np.isnan(macd_line[i]) or np.isnan(macd_sig[i]):
            continue
        if np.isnan(macd_line[i - 1]) or np.isnan(macd_sig[i - 1]):
            continue
        if np.isnan(atr14[i]):
            continue

        # Signal: MACD crossover at bar i (closed)
        macd_cross_up   = (macd_line[i] > macd_sig[i]) and (macd_line[i - 1] <= macd_sig[i - 1])
        macd_cross_down = (macd_line[i] < macd_sig[i]) and (macd_line[i - 1] >= macd_sig[i - 1])

        direction = None
        if cl[i] > ema200[i] and macd_cross_up:
            direction = 1   # LONG
        elif cl[i] < ema200[i] and macd_cross_down:
            direction = -1  # SHORT

        if direction is None:
            continue

        # Entry at open of next bar + slippage (always costs slippage)
        entry_idx = i + 1
        if entry_idx >= n:
            break

        entry_raw = op[entry_idx]
        # Slippage: we always pay it (add to entry cost, not to price)
        stop_dist = atr14[i] * ATR_MULT
        if stop_dist <= 0:
            continue

        if direction == 1:
            stop_price   = entry_raw - stop_dist
            target_price = entry_raw + stop_dist * rr
        else:
            stop_price   = entry_raw + stop_dist
            target_price = entry_raw - stop_dist * rr

        # Scan forward for stop or target hit
        result    = None
        exit_idx  = n - 1
        pnl_r     = None

        for j in range(entry_idx, n):
            lo_j = lo[j]
            hi_j = hi[j]

            if direction == 1:
                # Conservative: on same-bar conflict, assume stop hit first
                stop_hit   = lo_j <= stop_price
                target_hit = hi_j >= target_price
                if stop_hit:
                    result   = "loss"
                    exit_idx = j
                    raw_pnl_pct = -(stop_dist / entry_raw) - TOTAL_COST
                    break
                elif target_hit:
                    result   = "win"
                    exit_idx = j
                    raw_pnl_pct = (stop_dist * rr / entry_raw) - TOTAL_COST
                    break
            else:
                stop_hit   = hi_j >= stop_price
                target_hit = lo_j <= target_price
                if stop_hit:
                    result   = "loss"
                    exit_idx = j
                    raw_pnl_pct = -(stop_dist / entry_raw) - TOTAL_COST
                    break
                elif target_hit:
                    result   = "win"
                    exit_idx = j
                    raw_pnl_pct = (stop_dist * rr / entry_raw) - TOTAL_COST
                    break

        if result is None:
            # Trade still open at end of data — skip (incomplete trade)
            continue

        # Convert raw pnl % to R
        one_r_pct = stop_dist / entry_raw
        pnl_r     = raw_pnl_pct / one_r_pct

        ts_entry = candles[entry_idx]["ts"]
        ts_exit  = candles[exit_idx]["ts"]

        trades.append({
            "signal_bar":  i,
            "entry_bar":   entry_idx,
            "exit_bar":    exit_idx,
            "direction":   "LONG" if direction == 1 else "SHORT",
            "entry_price": entry_raw,
            "stop":        stop_price,
            "target":      target_price,
            "result":      result,
            "pnl_r":       pnl_r,
            "stop_pct":    one_r_pct,
            "ts_entry":    ts_entry,
            "ts_exit":     ts_exit,
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
    wr     = wins / n if n else 0

    avg_win_r  = np.mean([p for p in pnl_list if p > 0]) if wins  else 0
    avg_loss_r = np.mean([p for p in pnl_list if p < 0]) if losses else 0
    expectancy = wr * avg_win_r + (1 - wr) * avg_loss_r  # avg_loss_r is negative

    # Max consecutive losses
    max_consec_loss = 0
    cur_consec_loss = 0
    for r in results:
        if r == "loss":
            cur_consec_loss += 1
            max_consec_loss  = max(max_consec_loss, cur_consec_loss)
        else:
            cur_consec_loss = 0

    # Max drawdown in R
    cum_r  = np.cumsum(pnl_list)
    peak   = np.maximum.accumulate(cum_r)
    dd     = cum_r - peak
    max_dd = dd.min()

    # Direction breakdown
    longs  = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]

    return {
        "label":           label,
        "rr":              rr,
        "n_trades":        n,
        "n_wins":          wins,
        "n_losses":        losses,
        "win_rate":        wr,
        "expectancy_r":    expectancy,
        "avg_win_r":       avg_win_r,
        "avg_loss_r":      avg_loss_r,
        "total_r":         sum(pnl_list),
        "max_consec_loss": max_consec_loss,
        "max_drawdown_r":  max_dd,
        "n_longs":         len(longs),
        "n_shorts":        len(shorts),
        "wr_longs":        sum(1 for t in longs  if t["result"] == "win") / len(longs)  if longs  else 0,
        "wr_shorts":       sum(1 for t in shorts if t["result"] == "win") / len(shorts) if shorts else 0,
        "trades":          trades,
    }


# ─── WALK-FORWARD ──────────────────────────────────────────────────────────────

def walk_forward(candles: list, rr: float, n_windows: int = 3) -> list:
    """
    Walk-forward test: split in-sample into n_windows chunks.
    Each window: train on first 70%, test on last 30%.
    """
    is_end = int(len(candles) * SPLIT_IS)
    is_candles = candles[:is_end]
    wf_size = len(is_candles) // n_windows
    results = []
    for k in range(n_windows):
        start = k * wf_size
        end   = start + wf_size if k < n_windows - 1 else len(is_candles)
        chunk = is_candles[start:end]
        m = backtest(chunk, rr, label=f"WF-window-{k+1}")
        results.append(m)
    return results


# ─── PRINT REPORT ──────────────────────────────────────────────────────────────

def fmt(x, digits=3):
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def print_report(sym: str, candles: list):
    print(f"\n{'━'*62}")
    print(f"  SYMBOL: {sym}   bars: {len(candles)}")
    dt_from = datetime.fromtimestamp(candles[0]["ts"]  / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    dt_to   = datetime.fromtimestamp(candles[-1]["ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"  Period: {dt_from} → {dt_to}")
    print(f"{'━'*62}")

    n_is  = int(len(candles) * SPLIT_IS)
    n_oos = len(candles) - n_is
    is_candles  = candles[:n_is]
    oos_candles = candles[n_is:]

    for rr in RR_LIST:
        print(f"\n  ┌─ RR 1:{int(rr) if rr == int(rr) else rr} {'─'*48}┐")

        m_is  = backtest(is_candles,  rr, label="In-sample")
        m_oos = backtest(oos_candles, rr, label="Out-of-sample")

        for m, tag in [(m_is, "IN-SAMPLE "), (m_oos, "OUT-OF-SAMPLE")]:
            if "error" in m:
                print(f"  │  {tag}: NO TRADES")
                continue
            verdict = ""
            if m["expectancy_r"] >= 0.2:
                verdict = "✅ TRADEABLE"
            elif m["expectancy_r"] >= 0.1:
                verdict = "⚠️  WEAK"
            else:
                verdict = "❌ NO EDGE"

            print(f"  │  {tag}  n={m['n_trades']:>3}  "
                  f"WR={m['win_rate']*100:.1f}%  "
                  f"E={m['expectancy_r']:+.3f}R  "
                  f"TotalR={m['total_r']:+.2f}  "
                  f"MaxDD={m['max_drawdown_r']:.2f}R  "
                  f"MaxLoss={m['max_consec_loss']}  "
                  f"{verdict}")
            print(f"  │            Longs {m['n_longs']} WR={m['wr_longs']*100:.1f}%  "
                  f"Shorts {m['n_shorts']} WR={m['wr_shorts']*100:.1f}%")

        # Walk-forward
        wf_results = walk_forward(candles, rr, n_windows=3)
        print(f"  │  WALK-FORWARD ({len(wf_results)} windows):")
        for w in wf_results:
            if "error" in w:
                print(f"  │    {w['label']}: NO TRADES")
            else:
                v = "✅" if w["expectancy_r"] >= 0.2 else ("⚠️" if w["expectancy_r"] >= 0.1 else "❌")
                print(f"  │    {w['label']}: n={w['n_trades']:>3}  WR={w['win_rate']*100:.1f}%  "
                      f"E={w['expectancy_r']:+.3f}R  {v}")
        print(f"  └{'─'*57}┘")

    # Decision summary
    print(f"\n  DECISION THRESHOLDS:")
    print(f"    E ≥ 0.20R + stable OOS + MaxLoss ≤ 15 → BUILD LIVE BOT")
    print(f"    E  0.10–0.19R                          → OPTIMIZE (tighter filters)")
    print(f"    E < 0.10R                              → DISCARD, try next hypothesis")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  HYPOTHESIS A — EMA200 + MACD Trend-Following Backtest")
    print(f"  Commission: {COMMISSION*100:.3f}%  Slippage: {SLIPPAGE*100:.2f}%  "
          f"Total cost: {TOTAL_COST*100:.3f}%")
    print(f"  ATR period: {ATR_PERIOD}  ATR mult: {ATR_MULT}×  "
          f"In-sample split: {int(SPLIT_IS*100)}%")
    print("=" * 62)

    for sym in SYMBOLS:
        candles = download_symbol(sym)
        if len(candles) < EMA_PERIOD + MACD_SLOW + MACD_SIG + 50:
            print(f"  ⚠️  Not enough data for {sym}: {len(candles)} bars")
            continue
        print_report(sym, candles)

    print("\nDone.")


if __name__ == "__main__":
    main()
