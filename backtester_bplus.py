#!/usr/bin/env python3
"""
Backtester — Hypothesis B+: Mean Reversion с фильтром тренда EMA200
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal:
  LONG  → RSI[i] < 30  AND  close[i] <= BB_lower[i]  AND  close[i] > EMA200[i]
  SHORT → RSI[i] > 70  AND  close[i] >= BB_upper[i]  AND  close[i] < EMA200[i]

Логика: mean reversion только ПО ТРЕНДУ — не ловим падающие ножи в нисходящем тренде.
EMA200 — граница: выше = бычий рынок (только лонги), ниже = медвежий (только шорты).

Entry:  open[i+1]  (решение на закрытом баре, без lookahead)
Stop:   entry ± ATR14 * 1.5  (не двигается)
Target: entry ± stop * RR

Costs:  0.055% × 2 + 0.02% slippage = 0.13% итого
"""

import time, json, ssl, numpy as np, urllib.request
from datetime import datetime, timezone

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SYMBOLS     = ["BTCUSDT", "ETHUSDT"]
INTERVAL    = 240
YEARS       = 2
COMMISSION  = 0.00055 * 2
SLIPPAGE    = 0.0002
TOTAL_COST  = COMMISSION + SLIPPAGE   # 0.13%

RSI_PERIOD  = 14;  RSI_OS = 30;  RSI_OB = 70
BB_PERIOD   = 20;  BB_STD  = 2.0
EMA_PERIOD  = 200
ATR_PERIOD  = 14;  ATR_MULT = 1.5
RR_LIST     = [2.0, 3.0]
SPLIT_IS    = 0.70

# ── DOWNLOAD ───────────────────────────────────────────────────────────────────
def bybit_klines(symbol, interval, start_ms, end_ms):
    url_base, all_c, cur_end = "https://api.bybit.com/v5/market/kline", [], end_ms
    while True:
        url = (f"{url_base}?category=linear&symbol={symbol}"
               f"&interval={interval}&limit=200&start={start_ms}&end={cur_end}")
        for a in range(4):
            try:
                with urllib.request.urlopen(url, timeout=20, context=_SSL_CTX) as r:
                    data = json.loads(r.read()); break
            except Exception:
                if a == 3: raise
                time.sleep(2**a)
        rows = data.get("result", {}).get("list", [])
        if not rows: break
        for r in rows:
            ts = int(r[0])
            if start_ms <= ts <= end_ms:
                all_c.append({"ts": ts, "open": float(r[1]), "high": float(r[2]),
                              "low": float(r[3]), "close": float(r[4])})
        oldest = int(rows[-1][0])
        if oldest <= start_ms: break
        cur_end = oldest - 1
        time.sleep(0.15)
    all_c.sort(key=lambda x: x["ts"])
    seen, u = set(), []
    for c in all_c:
        if c["ts"] not in seen: seen.add(c["ts"]); u.append(c)
    return u

def download(symbol):
    now = int(time.time()*1000)
    start = now - int(YEARS*365.25*24*3600*1000)
    print(f"  Скачиваю {symbol} {YEARS}y×4H...", flush=True)
    c = bybit_klines(symbol, INTERVAL, start, now)
    print(f"  → {len(c)} баров", flush=True)
    return c

# ── INDICATORS ─────────────────────────────────────────────────────────────────
def ema(arr, p):
    out = np.full(len(arr), np.nan)
    if len(arr) < p: return out
    out[p-1] = arr[:p].mean(); k = 2/(p+1)
    for i in range(p, len(arr)): out[i] = arr[i]*k + out[i-1]*(1-k)
    return out

def rsi(close, p=14):
    n = len(close); out = np.full(n, np.nan)
    if n < p+1: return out
    d = np.diff(close)
    g = np.where(d>0,d,0.); l = np.where(d<0,-d,0.)
    ag = g[:p].mean(); al = l[:p].mean()
    out[p] = 100. if al==0 else 100.-100./(1.+ag/al)
    for i in range(p, n-1):
        ag = (ag*(p-1)+g[i])/p; al = (al*(p-1)+l[i])/p
        out[i+1] = 100. if al==0 else 100.-100./(1.+ag/al)
    return out

def bb(close, p=20, s=2.0):
    n = len(close); lo = np.full(n,np.nan); hi = np.full(n,np.nan)
    for i in range(p-1, n):
        w = close[i-p+1:i+1]; m = w.mean(); sd = w.std(ddof=0)
        lo[i] = m - s*sd; hi[i] = m + s*sd
    return lo, hi

def atr(high, low, close, p=14):
    n = len(high); tr = np.full(n, np.nan); tr[0] = high[0]-low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    out = np.full(n, np.nan); out[p-1] = tr[:p].mean()
    for i in range(p, n): out[i] = (out[i-1]*(p-1)+tr[i])/p
    return out

# ── BACKTEST ───────────────────────────────────────────────────────────────────
def backtest(candles, rr, label=""):
    n  = len(candles)
    op = np.array([c["open"]  for c in candles])
    hi = np.array([c["high"]  for c in candles])
    lo = np.array([c["low"]   for c in candles])
    cl = np.array([c["close"] for c in candles])

    rsi14  = rsi(cl, RSI_PERIOD)
    bb_lo, bb_hi = bb(cl, BB_PERIOD, BB_STD)
    ema200 = ema(cl, EMA_PERIOD)
    atr14  = atr(hi, lo, cl, ATR_PERIOD)

    trades = []; in_until = -1; last_zone = 0
    min_bar = max(RSI_PERIOD, BB_PERIOD, EMA_PERIOD, ATR_PERIOD) + 1

    for i in range(min_bar, n-1):
        if i <= in_until: continue
        if any(np.isnan(v) for v in [rsi14[i], bb_lo[i], ema200[i], atr14[i]]): continue

        r = rsi14[i]
        # Reset zone
        if last_zone == 1 and r > 45:  last_zone = 0
        if last_zone == -1 and r < 55: last_zone = 0

        direction = None
        # LONG: oversold + below BB + price above EMA200 (bullish context)
        if r <= RSI_OS and cl[i] <= bb_lo[i] and cl[i] > ema200[i] and last_zone != 1:
            direction = 1; last_zone = 1
        # SHORT: overbought + above BB + price below EMA200 (bearish context)
        elif r >= RSI_OB and cl[i] >= bb_hi[i] and cl[i] < ema200[i] and last_zone != -1:
            direction = -1; last_zone = -1

        if direction is None: continue

        ei = i + 1
        if ei >= n: break
        ep  = op[ei]
        sd  = atr14[i] * ATR_MULT
        if sd <= 0: continue

        sp = ep - sd if direction == 1 else ep + sd
        tp = ep + sd*rr if direction == 1 else ep - sd*rr

        result = None; eidx = n-1
        for j in range(ei, n):
            lj, hj = lo[j], hi[j]
            if direction == 1:
                if lj <= sp:  result="loss"; eidx=j; raw=-(sd/ep)-TOTAL_COST; break
                if hj >= tp:  result="win";  eidx=j; raw=(sd*rr/ep)-TOTAL_COST; break
            else:
                if hj >= sp:  result="loss"; eidx=j; raw=-(sd/ep)-TOTAL_COST; break
                if lj <= tp:  result="win";  eidx=j; raw=(sd*rr/ep)-TOTAL_COST; break

        if result is None: continue
        one_r = sd/ep
        trades.append({"direction":"LONG" if direction==1 else "SHORT",
                        "result":result, "pnl_r":raw/one_r,
                        "rsi":r, "ts_e":candles[ei]["ts"], "ts_x":candles[eidx]["ts"]})
        in_until = eidx

    return metrics(trades, rr, label)

def metrics(trades, rr, label):
    if not trades: return {"label":label,"rr":rr,"n_trades":0,"error":"no trades"}
    res = [t["result"] for t in trades]; pnl = [t["pnl_r"] for t in trades]
    n = len(trades); w = res.count("win"); wr = w/n
    aw = np.mean([p for p in pnl if p>0]) if w else 0.
    al = np.mean([p for p in pnl if p<0]) if n-w else 0.
    exp = wr*aw + (1-wr)*al
    mc=cc=0
    for r in res:
        cc = cc+1 if r=="loss" else 0; mc = max(mc,cc)
    cum = np.cumsum(pnl); peak = np.maximum.accumulate(cum); mdd = (cum-peak).min()
    lo_t=[t for t in trades if t["direction"]=="LONG"]
    sh_t=[t for t in trades if t["direction"]=="SHORT"]
    return {"label":label,"rr":rr,"n_trades":n,"n_wins":w,"win_rate":wr,
            "expectancy_r":exp,"total_r":sum(pnl),"max_consec_loss":mc,
            "max_drawdown_r":mdd,
            "n_longs":len(lo_t),"n_shorts":len(sh_t),
            "wr_longs": sum(1 for t in lo_t if t["result"]=="win")/len(lo_t) if lo_t else 0,
            "wr_shorts":sum(1 for t in sh_t if t["result"]=="win")/len(sh_t) if sh_t else 0,
            "trades":trades}

def wf(candles, rr, nw=3):
    ise = int(len(candles)*SPLIT_IS); isc = candles[:ise]
    sz = len(isc)//nw; res=[]
    for k in range(nw):
        s=k*sz; e=s+sz if k<nw-1 else len(isc)
        res.append(backtest(isc[s:e], rr, f"WF-{k+1}"))
    return res

# ── REPORT ─────────────────────────────────────────────────────────────────────
def report(sym, candles):
    print(f"\n{'━'*62}")
    print(f"  {sym}  {len(candles)} баров")
    df = datetime.fromtimestamp(candles[0]["ts"]/1000,tz=timezone.utc).strftime("%Y-%m-%d")
    dt = datetime.fromtimestamp(candles[-1]["ts"]/1000,tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"  {df} → {dt}")
    print(f"{'━'*62}")

    nis = int(len(candles)*SPLIT_IS)
    isc = candles[:nis]; oos = candles[nis:]

    for rr in RR_LIST:
        print(f"\n  ┌─ RR 1:{int(rr)} {'─'*49}┐")
        mis  = backtest(isc, rr, "In-sample")
        moos = backtest(oos, rr, "Out-of-sample")

        for m,tag in [(mis,"IN-SAMPLE  "),(moos,"OUT-OF-SAMPLE")]:
            if "error" in m: print(f"  │  {tag}: НЕТ СДЕЛОК"); continue
            v = "✅ TRADEABLE" if m["expectancy_r"]>=0.2 else "⚠️  WEAK" if m["expectancy_r"]>=0.1 else "❌ NO EDGE"
            print(f"  │  {tag}  n={m['n_trades']:>3}  WR={m['win_rate']*100:.1f}%  "
                  f"E={m['expectancy_r']:+.3f}R  TotalR={m['total_r']:+.2f}  "
                  f"MaxDD={m['max_drawdown_r']:.2f}R  MaxLoss={m['max_consec_loss']}  {v}")
            print(f"  │             Longs {m['n_longs']} WR={m['wr_longs']*100:.1f}%  "
                  f"Shorts {m['n_shorts']} WR={m['wr_shorts']*100:.1f}%")

        wfr = wf(candles, rr)
        print(f"  │  WALK-FORWARD:")
        for w_ in wfr:
            if "error" in w_: print(f"  │    {w_['label']}: НЕТ СДЕЛОК"); continue
            v = "✅" if w_["expectancy_r"]>=0.2 else "⚠️" if w_["expectancy_r"]>=0.1 else "❌"
            print(f"  │    {w_['label']}: n={w_['n_trades']:>3}  WR={w_['win_rate']*100:.1f}%  "
                  f"E={w_['expectancy_r']:+.3f}R  {v}")
        print(f"  └{'─'*57}┘")

    print(f"\n  ПОРОГИ: E≥0.20R + стаб.OOS + MaxLoss≤15 → СТРОИМ БОТ")
    print(f"          E<0.10R → Hypothesis C (Volume Breakout)")

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("="*62)
    print("  HYPOTHESIS B+ — RSI + Bollinger + EMA200 (тренд-фильтр)")
    print(f"  Лонг только выше EMA200, шорт только ниже EMA200")
    print(f"  RSI<{RSI_OS}/>{RSI_OB}  BB{BB_PERIOD}×{BB_STD}  EMA{EMA_PERIOD}  ATR{ATR_PERIOD}×{ATR_MULT}")
    print(f"  Комиссия={COMMISSION*100:.3f}% + Проскальзывание={SLIPPAGE*100:.2f}% = {TOTAL_COST*100:.3f}%")
    print("="*62)

    for sym in SYMBOLS:
        c = download(sym)
        if len(c) < max(RSI_PERIOD,BB_PERIOD,EMA_PERIOD,ATR_PERIOD)+50:
            print(f"  ⚠ Мало данных: {sym}"); continue
        report(sym, c)

    print("\nГотово.")

if __name__ == "__main__": main()
