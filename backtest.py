"""
Backtester for the EMA crossover + ATR stop strategy used in ema_bot.py.

It only needs pandas and numpy. Three ways to feed it data:

    python backtest.py --demo                          # synthetic data, proves the code runs
    python backtest.py --csv EURUSD_H1.csv             # columns: time,open,high,low,close
    python backtest.py --oanda EUR_USD --years 2       # downloads from your OANDA practice
                                                       # account (needs OANDA_TOKEN env var
                                                       # and: pip install oandapyV20)

What it does
    1. Splits data 70/30. Picks the best parameters on the first 70% (in-sample),
       then reports how they do on the unseen 30% (out-of-sample).
    2. Walk-forward test: re-optimizes on a rolling window and trades the next window,
       stitching all the unseen windows into one result. This is the number to trust.

Realism choices (so results are not flattering)
    - Signals use closed bars; entries happen at the NEXT bar's open (no look-ahead).
    - Spread is charged on every entry and exit. Stops are triggered on bid/ask.
    - Gaps through a stop fill at the gap price, not at the stop.
    - Position size risks a fixed % of current equity per trade.

Limitations
    - P&L is in the quote currency, so it is right for XXX/USD pairs on a USD account.
    - No swap/rollover, commissions, or news-time spread widening.
    - Good backtest results do NOT guarantee live profit. Treat them as a filter for
      bad ideas, not proof of a good one.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for alt in ("datetime", "date", "timestamp"):
        if "time" not in df.columns and alt in df.columns:
            df = df.rename(columns={alt: "time"})
    missing = {"time", "open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return (df[["time", "open", "high", "low", "close"]]
            .sort_values("time").drop_duplicates("time").reset_index(drop=True))


def make_demo_data(n: int = 15000, seed: int = 7) -> pd.DataFrame:
    """Synthetic hourly data with slowly changing drift. For testing code only."""
    rng = np.random.default_rng(seed)
    vol = 0.0004
    drift = np.repeat(rng.normal(0, 0.00004, n // 500 + 1), 500)[:n]
    close = 1.10 * np.exp(np.cumsum(drift + rng.normal(0, vol, n)))
    open_ = np.concatenate([[1.10], close[:-1]])
    wick = np.abs(rng.normal(0, vol * 0.6, n)) * close
    return pd.DataFrame({
        "time": pd.date_range("2023-01-02", periods=n, freq="h", tz="UTC"),
        "open": open_,
        "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick,
        "close": close,
    })


def download_oanda(instrument: str, granularity: str, years: float) -> pd.DataFrame:
    from oandapyV20 import API
    import oandapyV20.endpoints.instruments as ins

    client = API(access_token=os.environ["OANDA_TOKEN"], environment="practice")
    cursor = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=int(years), months=int((years % 1) * 12))
    rows = []
    while True:
        r = ins.InstrumentsCandles(instrument=instrument, params={
            "granularity": granularity, "price": "M", "count": 5000,
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        })
        client.request(r)
        raw = r.response["candles"]
        done = [c for c in raw if c["complete"]]
        rows += [{"time": c["time"], "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
                  "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"])} for c in done]
        if len(raw) < 5000 or not done:
            break
        cursor = pd.Timestamp(done[-1]["time"]) + pd.Timedelta(microseconds=1)
        print(f"  downloaded {len(rows)} candles...")
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates("time").reset_index(drop=True)
    df.to_csv(f"{instrument}_{granularity}.csv", index=False)
    print(f"Saved {len(df)} candles to {instrument}_{granularity}.csv")
    return df


# ------------------------------------------------------------ indicators
def indicators(df: pd.DataFrame, fast: int, slow: int, atr_period: int):
    c = df["close"]
    ef = c.ewm(span=fast, adjust=False).mean().to_numpy()
    es = c.ewm(span=slow, adjust=False).mean().to_numpy()
    pc = c.shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean().to_numpy()
    return ef, es, atr


# ------------------------------------------------------------- simulator
def run_backtest(df, fast=20, slow=50, atr_period=14, atr_mult=2.0, risk=0.005,
                 spread=0.00012, equity0=10_000.0, max_units=100_000, start=None):
    """Bar-by-bar simulation. `start` = first bar index allowed to trade."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    times = df["time"].to_numpy()
    ef, es, atr = indicators(df, fast, slow, atr_period)
    n = len(df)
    warm = start if start is not None else slow * 2
    half = spread / 2

    equity = equity0
    eq_curve = np.full(n, equity0)
    trades, pos = [], None
    pending, pending_atr = 0, 0.0

    def close_trade(p, price, i, reason):
        nonlocal equity
        pnl = p["side"] * (price - p["entry"]) * p["units"]
        equity += pnl
        trades.append({"entry_time": p["time"], "exit_time": times[i],
                       "side": "long" if p["side"] == 1 else "short",
                       "units": p["units"], "entry": p["entry"], "exit": price,
                       "pnl": pnl, "reason": reason})

    for i in range(n):
        # 1) execute the order decided at the previous bar's close, at this open
        if pending != 0 and i >= warm:
            if pos is not None and pos["side"] != pending:
                price = o[i] - half if pos["side"] == 1 else o[i] + half
                close_trade(pos, price, i, "opposite_cross")
                pos = None
            if pos is None and equity > 0:
                stop_dist = pending_atr * atr_mult
                units = int(min(equity * risk / stop_dist, max_units)) if stop_dist > 0 else 0
                if units >= 1:
                    entry = o[i] + half if pending == 1 else o[i] - half
                    pos = {"side": pending, "units": units, "entry": entry,
                           "stop": entry - pending * stop_dist, "time": times[i]}
        pending = 0

        # 2) stop-loss check inside this bar (bid for longs, ask for shorts)
        if pos is not None:
            if pos["side"] == 1 and l[i] - half <= pos["stop"]:
                close_trade(pos, min(o[i] - half, pos["stop"]), i, "stop")
                pos = None
            elif pos["side"] == -1 and h[i] + half >= pos["stop"]:
                close_trade(pos, max(o[i] + half, pos["stop"]), i, "stop")
                pos = None

        # 3) mark to market at the close
        unreal = 0.0
        if pos is not None:
            unreal = pos["side"] * (c[i] - pos["side"] * half - pos["entry"]) * pos["units"]
        eq_curve[i] = equity + unreal
        if eq_curve[i] <= 0:
            eq_curve[i:] = 0
            break

        # 4) look for a crossover on this closed bar; act on the next open
        if i >= max(warm, 1):
            up = ef[i] > es[i] and ef[i - 1] <= es[i - 1]
            down = ef[i] < es[i] and ef[i - 1] >= es[i - 1]
            sig = 1 if up else (-1 if down else 0)
            if sig != 0 and (pos is None or pos["side"] != sig):
                pending, pending_atr = sig, atr[i]

    # close anything still open at the end so P&L is fully counted
    if pos is not None:
        last = c[-1] - half if pos["side"] == 1 else c[-1] + half
        close_trade(pos, last, n - 1, "end_of_data")
        eq_curve[-1] = equity

    return pd.DataFrame(trades), eq_curve[warm:] if warm < n else eq_curve


def metrics(trades: pd.DataFrame, eq: np.ndarray, equity0: float) -> dict:
    if len(eq) == 0:
        return {"return": 0.0, "max_dd": 0.0, "trades": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_trade": 0.0, "final_equity": equity0}
    peak = np.maximum.accumulate(eq)
    dd = float((eq / np.where(peak > 0, peak, 1) - 1).min())
    if trades.empty:
        return {"return": eq[-1] / equity0 - 1, "max_dd": dd, "trades": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_trade": 0.0, "final_equity": float(eq[-1])}
    wins = trades.loc[trades.pnl > 0, "pnl"].sum()
    losses = -trades.loc[trades.pnl < 0, "pnl"].sum()
    return {"return": float(eq[-1] / equity0 - 1), "max_dd": dd, "trades": len(trades),
            "win_rate": float((trades.pnl > 0).mean()),
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            "avg_trade": float(trades.pnl.mean()), "final_equity": float(eq[-1])}


def fmt(m: dict) -> str:
    pf = "inf" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
    return (f"return {m['return']*100:7.2f}% | max drawdown {m['max_dd']*100:6.2f}% | "
            f"trades {m['trades']:4d} | win rate {m['win_rate']*100:5.1f}% | "
            f"profit factor {pf} | avg trade {m['avg_trade']:8.2f}")


# ------------------------------------------------------- validation tools
GRID = {"fast": [8, 12, 20, 30], "slow": [40, 50, 80, 120], "atr_mult": [1.5, 2.0, 3.0]}
WARMUP = 300  # extra bars before a test window so indicators are "warmed up"


def score(m: dict) -> float:
    """Return divided by drawdown, ignoring parameter sets with too few trades."""
    if m["trades"] < 15:
        return -np.inf
    return m["return"] / max(abs(m["max_dd"]), 0.01)


def optimize(df: pd.DataFrame, base: dict):
    best, best_score, best_m = dict(base), -np.inf, None
    for fast, slow, mult in itertools.product(GRID["fast"], GRID["slow"], GRID["atr_mult"]):
        if fast >= slow:
            continue
        p = {**base, "fast": fast, "slow": slow, "atr_mult": mult}
        tr, eq = run_backtest(df, **p)
        m = metrics(tr, eq, base["equity0"])
        s = score(m)
        if s > best_score:
            best, best_score, best_m = p, s, m
    if best_m is None:  # nothing qualified; fall back to the defaults
        tr, eq = run_backtest(df, **base)
        best_m = metrics(tr, eq, base["equity0"])
    return best, best_m


def split_test(df: pd.DataFrame, base: dict, ratio: float = 0.7):
    cut = int(len(df) * ratio)
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut - WARMUP:].reset_index(drop=True)
    best, m_in = optimize(train, base)
    tr, eq = run_backtest(test, **{**best, "start": WARMUP})
    m_out = metrics(tr, eq, base["equity0"])
    print("\n=== In-sample / out-of-sample split (70/30) ===")
    print(f"Best parameters on in-sample: fast={best['fast']} slow={best['slow']} "
          f"atr_mult={best['atr_mult']}")
    print("In-sample  :", fmt(m_in))
    print("Out-of-samp:", fmt(m_out), "  <-- the honest one")
    return tr


def walk_forward(df: pd.DataFrame, base: dict, train_bars: int, test_bars: int):
    print("\n=== Walk-forward (re-optimize, then trade the next unseen window) ===")
    equity = base["equity0"]
    all_trades, curves, i, k = [], [], 0, 1
    while i + train_bars + test_bars <= len(df):
        train = df.iloc[i:i + train_bars].reset_index(drop=True)
        test = df.iloc[i + train_bars - WARMUP:i + train_bars + test_bars].reset_index(drop=True)
        best, _ = optimize(train, base)
        tr, eq = run_backtest(test, **{**best, "start": WARMUP, "equity0": equity})
        m = metrics(tr, eq, equity)
        print(f"Window {k:2d}: fast={best['fast']:>3} slow={best['slow']:>3} "
              f"mult={best['atr_mult']:.1f} -> {fmt(m)}")
        equity = float(eq[-1])
        all_trades.append(tr)
        curves.append(eq)
        i += test_bars
        k += 1
    if not curves:
        print("Not enough data for the chosen window sizes.")
        return
    trades = pd.concat(all_trades, ignore_index=True) if any(len(t) for t in all_trades) else pd.DataFrame()
    m = metrics(trades, np.concatenate(curves), base["equity0"])
    print("\nStitched out-of-sample result:")
    print(fmt(m))
    if not trades.empty:
        trades.to_csv("walkforward_trades.csv", index=False)
        print("Trade list saved to walkforward_trades.csv")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="EMA/ATR strategy backtester")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="use synthetic data")
    src.add_argument("--csv", help="CSV with time,open,high,low,close")
    src.add_argument("--oanda", help="instrument to download, e.g. EUR_USD")
    ap.add_argument("--granularity", default="H1")
    ap.add_argument("--years", type=float, default=2)
    ap.add_argument("--spread-pips", type=float, default=1.2, help="round-trip cost per side is half of this")
    ap.add_argument("--pip-size", type=float, default=0.0001, help="0.01 for JPY pairs")
    ap.add_argument("--risk", type=float, default=0.005, help="fraction of equity risked per trade")
    ap.add_argument("--train-bars", type=int, default=4000)
    ap.add_argument("--test-bars", type=int, default=1000)
    args = ap.parse_args()

    if args.demo:
        df = make_demo_data()
        print("Using SYNTHETIC demo data. These results mean nothing about real markets.")
    elif args.csv:
        df = load_csv(args.csv)
    else:
        df = download_oanda(args.oanda, args.granularity, args.years)
    print(f"{len(df)} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    base = {"fast": 20, "slow": 50, "atr_period": 14, "atr_mult": 2.0, "risk": args.risk,
            "spread": args.spread_pips * args.pip_size, "equity0": 10_000.0, "max_units": 100_000}

    tr, eq = run_backtest(df, **base)
    print("\n=== Default parameters (20/50 EMA, 2x ATR stop), whole dataset ===")
    print(fmt(metrics(tr, eq, base["equity0"])))
    print("(This one is in-sample by definition. Do not trust it on its own.)")

    split_test(df, base)
    walk_forward(df, base, args.train_bars, args.test_bars)


if __name__ == "__main__":
    main()
