#!/usr/bin/env python3
"""
Gap and Go Backtester
Tests the scanner's strategy against 1 year of historical daily data.

How it works:
  - For each trading day, checks if the stock gapped up 2–5% at the open
  - Simulates entry at open price
  - Stop loss  = open − 1× ATR(14)
  - Target     = open + 1.5× ATR(14)
  - Uses daily High/Low to determine which was hit first
  - Hard exit  at close if neither stop nor target triggered

Run:
  python backtest.py
  python backtest.py --tickers AAPL MSFT NVDA
  python backtest.py --months 6
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gap_scanner import DEFAULT_WATCHLIST, MIN_GAP_PCT, MAX_GAP_PCT, ATR_PERIOD

# ── Config ─────────────────────────────────────────────────────────────────────
ACCOUNT_SIZE   = 10_000
RISK_PCT       = 1.0        # % of account risked per trade
LOOKBACK_MONTHS = 12        # how far back to test
MIN_VOLUME_RATIO = 2.0      # same as scanner


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class Trade:
    ticker:     str
    date:       str
    gap_pct:    float
    entry:      float
    stop:       float
    target:     float
    exit_price: float
    exit_reason: str        # "target", "stop", "close"
    pnl:        float
    pnl_pct:    float
    shares:     int
    won:        bool


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"].shift(1)
    tr    = pd.concat([
        high - low,
        (high - close).abs(),
        (low  - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["Close"].rolling(period).mean()


def compute_volume_ratio(df: pd.DataFrame, period: int = 10) -> pd.Series:
    avg_vol = df["Volume"].rolling(period).mean().shift(1)
    return df["Volume"] / avg_vol.replace(0, np.nan)


# ── Core backtest for one ticker ───────────────────────────────────────────────
def backtest_ticker(ticker: str, months: int) -> list[Trade]:
    start = (datetime.today() - timedelta(days=months * 31)).strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    except Exception:
        return []

    if df.empty or len(df) < ATR_PERIOD + 20:
        return []

    df = df.copy()

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["PrevClose"]    = df["Close"].shift(1)
    df["GapPct"]       = (df["Open"] - df["PrevClose"]) / df["PrevClose"] * 100
    df["ATR"]          = compute_atr(df).shift(1)   # use previous day's ATR
    df["RSI"]          = compute_rsi(df).shift(1)
    df["SMA20"]        = compute_sma(df).shift(1)
    df["VolRatio"]     = compute_volume_ratio(df)
    df = df.dropna()

    trades = []
    dollar_risk = ACCOUNT_SIZE * (RISK_PCT / 100)

    for idx, row in df.iterrows():
        gap   = row["GapPct"]
        entry = row["Open"]
        atr   = row["ATR"]
        rsi   = row["RSI"]
        sma   = row["SMA20"]
        prev  = row["PrevClose"]
        vol_r = row["VolRatio"]
        high  = row["High"]
        low   = row["Low"]
        close = row["Close"]

        # ── Apply all scanner filters ─────────────────────────────────────────
        if not (MIN_GAP_PCT <= gap <= MAX_GAP_PCT):
            continue
        if vol_r < MIN_VOLUME_RATIO:
            continue
        if rsi > 70 or rsi < 30:
            continue
        if prev < sma:
            continue
        if atr <= 0:
            continue
        gap_atr = (entry - prev) / atr
        if not (0.5 <= gap_atr <= 2.0):
            continue

        # ── Trade levels ──────────────────────────────────────────────────────
        stop   = entry - atr
        target = entry + (atr * 1.5)
        shares = max(int(dollar_risk / atr), 1)

        # ── Simulate outcome using High/Low ───────────────────────────────────
        # Assume worst case: stop can be hit before target on same bar
        if low <= stop and high >= target:
            # Both levels hit same day — assume stop hit first (conservative)
            exit_price  = stop
            exit_reason = "stop"
        elif high >= target:
            exit_price  = target
            exit_reason = "target"
        elif low <= stop:
            exit_price  = stop
            exit_reason = "stop"
        else:
            exit_price  = close
            exit_reason = "close"

        pnl     = (exit_price - entry) * shares
        pnl_pct = (exit_price - entry) / entry * 100

        trades.append(Trade(
            ticker=ticker,
            date=idx.strftime("%Y-%m-%d"),
            gap_pct=round(gap, 2),
            entry=round(entry, 2),
            stop=round(stop, 2),
            target=round(target, 2),
            exit_price=round(exit_price, 2),
            exit_reason=exit_reason,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            shares=shares,
            won=exit_price >= target,
        ))

    return trades


# ── Summary stats ──────────────────────────────────────────────────────────────
def print_summary(all_trades: list[Trade]) -> None:
    if not all_trades:
        print("\n  No trades generated — filters may be too strict for historical data.")
        return

    df = pd.DataFrame([t.__dict__ for t in all_trades])
    wins   = df[df["won"]]
    losses = df[~df["won"]]

    total_pnl    = df["pnl"].sum()
    win_rate     = len(wins) / len(df) * 100
    avg_win      = wins["pnl"].mean()   if not wins.empty   else 0
    avg_loss     = losses["pnl"].mean() if not losses.empty else 0
    profit_factor = wins["pnl"].sum() / abs(losses["pnl"].sum()) if not losses.empty and losses["pnl"].sum() != 0 else float("inf")

    bar = "=" * 54
    print(f"\n{bar}")
    print(f"  BACKTEST RESULTS  —  {len(all_trades)} trades")
    print(bar)
    print(f"  Win Rate:       {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Total P&L:      ${total_pnl:,.2f}")
    print(f"  Avg Win:        ${avg_win:,.2f}")
    print(f"  Avg Loss:       ${avg_loss:,.2f}")
    print(f"  Profit Factor:  {profit_factor:.2f}x")
    print(f"  Best Trade:     ${df['pnl'].max():,.2f}  ({df.loc[df['pnl'].idxmax(), 'ticker']} {df.loc[df['pnl'].idxmax(), 'date']})")
    print(f"  Worst Trade:    ${df['pnl'].min():,.2f}  ({df.loc[df['pnl'].idxmin(), 'ticker']} {df.loc[df['pnl'].idxmin(), 'date']})")
    print(bar)

    print(f"\n  By ticker:")
    by_ticker = df.groupby("ticker").agg(
        trades=("pnl", "count"),
        wins=("won", "sum"),
        total_pnl=("pnl", "sum"),
    ).sort_values("total_pnl", ascending=False)
    for t, row in by_ticker.iterrows():
        wr = row["wins"] / row["trades"] * 100
        print(f"    {t:<6}  {row['trades']:>3} trades  {wr:>5.0f}% win  ${row['total_pnl']:>8,.2f}")

    print(f"\n  Exit breakdown:")
    for reason, count in df["exit_reason"].value_counts().items():
        print(f"    {reason:<8} {count:>4} trades  ({count/len(df)*100:.0f}%)")


# ── Chart ──────────────────────────────────────────────────────────────────────
def plot_results(all_trades: list[Trade]) -> None:
    if not all_trades:
        return

    df = pd.DataFrame([t.__dict__ for t in all_trades])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["cumulative_pnl"] = df["pnl"].cumsum()

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#0f0f0f")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Equity curve ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0f0f0f")
    ax1.plot(df["date"], df["cumulative_pnl"], color="#4fc3f7", linewidth=2)
    ax1.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax1.fill_between(df["date"], df["cumulative_pnl"], 0,
                     where=df["cumulative_pnl"] >= 0, alpha=0.15, color="#66bb6a")
    ax1.fill_between(df["date"], df["cumulative_pnl"], 0,
                     where=df["cumulative_pnl"] < 0,  alpha=0.15, color="#ef5350")
    ax1.set_title("Cumulative P&L", color="white", fontsize=11)
    ax1.tick_params(colors="#aaaaaa")
    ax1.set_ylabel("$ P&L", color="#aaaaaa")
    for sp in ax1.spines.values(): sp.set_edgecolor("#333333")
    ax1.yaxis.label.set_color("#aaaaaa")
    ax1.grid(axis="y", color="#222222", linewidth=0.5)

    # ── Win/Loss bar chart per ticker ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0f0f0f")
    by_t = df.groupby("ticker")["pnl"].sum().sort_values()
    colors = ["#ef5350" if v < 0 else "#66bb6a" for v in by_t.values]
    ax2.barh(by_t.index, by_t.values, color=colors)
    ax2.axvline(0, color="#555555", linewidth=0.8)
    ax2.set_title("P&L by Ticker", color="white", fontsize=11)
    ax2.tick_params(colors="#aaaaaa")
    for sp in ax2.spines.values(): sp.set_edgecolor("#333333")
    ax2.grid(axis="x", color="#222222", linewidth=0.5)

    # ── Trade outcome pie ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0f0f0f")
    counts = df["exit_reason"].value_counts()
    pie_colors = {"target": "#66bb6a", "stop": "#ef5350", "close": "#ffb74d"}
    ax3.pie(counts.values,
            labels=counts.index,
            colors=[pie_colors.get(k, "#aaaaaa") for k in counts.index],
            autopct="%1.0f%%",
            textprops={"color": "white"})
    ax3.set_title("Exit Breakdown", color="white", fontsize=11)

    plt.suptitle("Gap & Go Backtest Results", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Gap and Go Backtester")
    parser.add_argument("--tickers", nargs="*", help="Tickers to test (default: full watchlist)")
    parser.add_argument("--months",  type=int, default=LOOKBACK_MONTHS, help="Months of history to test")
    args = parser.parse_args()

    watchlist = [t.upper() for t in args.tickers] if args.tickers else DEFAULT_WATCHLIST
    months    = args.months

    print(f"\nGap & Go Backtester")
    print(f"Period:   last {months} months")
    print(f"Tickers:  {len(watchlist)}")
    print(f"Account:  ${ACCOUNT_SIZE:,}  |  Risk: {RISK_PCT}%/trade")
    print(f"\nDownloading data and running simulation...\n")

    all_trades: list[Trade] = []
    for ticker in watchlist:
        trades = backtest_ticker(ticker, months)
        if trades:
            print(f"  {ticker:<6}  {len(trades):>3} signal(s) found")
        else:
            print(f"  {ticker:<6}  no signals")
        all_trades.extend(trades)

    print_summary(all_trades)
    plot_results(all_trades)


if __name__ == "__main__":
    main()
