#!/usr/bin/env python3
"""
Gap and Go Scanner
Scans a watchlist for pre-market gap-up setups before market open.
Run between 8:00 AM – 9:25 AM ET for best results.
"""

import sys
import argparse
from dataclasses import dataclass, field
from datetime import datetime, date, time
from typing import List, Optional

import pandas as pd
import pytz
import yfinance as yf

# ── Configuration ──────────────────────────────────────────────────────────────
ACCOUNT_SIZE       = 10_000   # Your account size ($)
RISK_PCT           = 1.0      # Max risk per trade as % of account
MIN_GAP_PCT        = 2.0      # Minimum gap % to qualify
CAUTION_GAP_PCT    = 10.0     # Gap > this is a caution flag
MIN_VOLUME_RATIO   = 2.0      # Pre-market volume must be Nx avg
VOLUME_LOOKBACK    = 5        # Trading days to average pre-mkt vol over

DEFAULT_WATCHLIST = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "NFLX", "JPM", "BAC", "COIN", "PLTR", "SOFI", "MARA", "SMCI",
]

ET = pytz.timezone("America/New_York")

PREMARKET_START = time(4, 0)
MARKET_OPEN     = time(9, 30)

# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class TradeSetup:
    ticker:           str
    prev_close:       float
    premarket_price:  float
    gap_pct:          float
    volume_ratio:     float
    entry_price:      float
    stop_loss:        float
    target_price:     float
    risk_per_share:   float
    reward_per_share: float
    rr_ratio:         float
    shares:           int
    dollar_risk:      float
    valid:            bool
    failures:         List[str] = field(default_factory=list)
    warnings:         List[str] = field(default_factory=list)


# ── Data fetchers ──────────────────────────────────────────────────────────────
def _ticker_history(ticker: str, **kwargs) -> pd.DataFrame:
    """Fetch history via yf.Ticker — returns simple column DataFrame."""
    t = yf.Ticker(ticker)
    return t.history(**kwargs)


def get_prev_close(ticker: str) -> Optional[float]:
    """Previous trading day closing price."""
    try:
        hist = _ticker_history(ticker, period="5d", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 2:
            return None
        return float(hist["Close"].iloc[-2])
    except Exception:
        return None


def get_premarket_price(ticker: str) -> Optional[float]:
    """Latest pre-market (or most recent available) price for today."""
    try:
        hist = _ticker_history(ticker, period="2d", interval="1m", prepost=True)
        if hist.empty:
            return None

        if hist.index.tz is None:
            hist.index = hist.index.tz_localize("UTC")
        hist.index = hist.index.tz_convert(ET)

        today = datetime.now(ET).date()
        pm = hist[
            (hist.index.date == today)
            & (hist.index.time >= PREMARKET_START)
            & (hist.index.time < MARKET_OPEN)
        ]

        if not pm.empty:
            return float(pm["Close"].iloc[-1])

        # Outside pre-market hours — use last available price as proxy
        day_rows = hist[hist.index.date == today]
        if day_rows.empty:
            return None
        return float(day_rows["Close"].iloc[-1])
    except Exception:
        return None


def get_volume_ratio(ticker: str) -> float:
    """
    Today's volume vs 10-day average volume.
    yfinance does not expose pre-market volume, so we use the full-day
    traded volume from fast_info and compare it to the 10-day average.
    Returns 0.0 if data is unavailable.
    """
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        today_vol = fi.last_volume
        avg_vol   = fi.ten_day_average_volume
        if not avg_vol or avg_vol == 0:
            return 0.0
        return round(today_vol / avg_vol, 2)
    except Exception:
        return 0.0


# ── Setup calculation ──────────────────────────────────────────────────────────
def build_setup(
    ticker: str,
    prev_close: float,
    pm_price: float,
    volume_ratio: float,
    account_size: float = ACCOUNT_SIZE,
    risk_pct: float = RISK_PCT,
) -> TradeSetup:

    gap_pct = (pm_price - prev_close) / prev_close * 100

    entry          = pm_price
    stop           = prev_close
    risk_per_share = max(entry - stop, 0.01)   # floor to avoid div/0
    reward         = risk_per_share * 2         # 2:1 R:R
    target         = entry + reward
    rr_ratio       = reward / risk_per_share

    dollar_risk = account_size * (risk_pct / 100)
    shares      = max(int(dollar_risk / risk_per_share), 0)

    failures: List[str] = []
    warnings: List[str] = []

    if gap_pct < MIN_GAP_PCT:
        failures.append(f"Gap {gap_pct:.2f}% < minimum {MIN_GAP_PCT}%")
    if volume_ratio < MIN_VOLUME_RATIO:
        failures.append(f"Volume ratio {volume_ratio:.1f}x < minimum {MIN_VOLUME_RATIO}x")
    if gap_pct > CAUTION_GAP_PCT:
        warnings.append(
            f"Gap > {CAUTION_GAP_PCT}% — momentum may already be exhausted, size down"
        )

    # Always require manual catalyst check
    warnings.append("Confirm news catalyst manually before entering")

    return TradeSetup(
        ticker=ticker,
        prev_close=round(prev_close, 2),
        premarket_price=round(pm_price, 2),
        gap_pct=round(gap_pct, 2),
        volume_ratio=round(volume_ratio, 2),
        entry_price=round(entry, 2),
        stop_loss=round(stop, 2),
        target_price=round(target, 2),
        risk_per_share=round(risk_per_share, 2),
        reward_per_share=round(reward, 2),
        rr_ratio=round(rr_ratio, 2),
        shares=shares,
        dollar_risk=round(dollar_risk, 2),
        valid=len(failures) == 0,
        failures=failures,
        warnings=warnings,
    )


# ── Display ────────────────────────────────────────────────────────────────────
def display_setup(s: TradeSetup, show_trade_levels: bool = True) -> None:
    bar   = "=" * 54
    label = "VALID SETUP" if s.valid else "FILTERED OUT"
    print(f"\n{bar}")
    print(f"  {s.ticker:<8}  |  {label}")
    print(bar)
    print(f"  Gap:            {s.gap_pct:+.2f}%")
    print(f"  Prev Close:     ${s.prev_close:.2f}")
    print(f"  Pre-mkt Price:  ${s.premarket_price:.2f}")
    print(f"  Volume Ratio:   {s.volume_ratio:.1f}x avg pre-market volume")

    if not s.valid:
        for f in s.failures:
            print(f"  FAIL: {f}")

    if show_trade_levels:
        print(f"  {'-' * 44}")
        print(f"  Entry:          ${s.entry_price:.2f}  (limit ~0.1% above ask after 9:35 AM)")
        print(f"  CLOSE AT LOSS:  ${s.stop_loss:.2f}  — stop-loss at yesterday's close")
        print(f"  CLOSE AT PROFIT:${s.target_price:.2f}  — 2:1 target ({s.rr_ratio:.0f}x your risk)")
        print(f"  Risk / share:   ${s.risk_per_share:.2f}")
        print(f"  Reward / share: ${s.reward_per_share:.2f}")
        print(f"  Position Size:  {s.shares} shares  (${s.dollar_risk:.0f} max loss)")
        print(f"  Hard exit by:   11:00 AM ET regardless of P&L")

    for w in s.warnings:
        print(f"  NOTE: {w}")


# ── Main scanner ───────────────────────────────────────────────────────────────
def scan(
    watchlist: List[str] = DEFAULT_WATCHLIST,
    account_size: float = ACCOUNT_SIZE,
    risk_pct: float = RISK_PCT,
) -> List[TradeSetup]:

    now_et = datetime.now(ET)
    print(f"\nGap and Go Scanner  —  {now_et.strftime('%Y-%m-%d %H:%M %Z')}")

    if now_et.weekday() == 0:   # Monday
        print("  WARNING: Monday gaps are less reliable — trade cautiously today.")

    print(f"Account: ${account_size:,.0f}  |  Risk: {risk_pct}%/trade  |  Min gap: {MIN_GAP_PCT}%")
    print(f"Scanning {len(watchlist)} tickers (this may take ~60 seconds)...\n")

    all_setups: List[TradeSetup] = []

    for ticker in watchlist:
        print(f"  {ticker:<6} ...", end=" ", flush=True)

        prev_close = get_prev_close(ticker)
        if prev_close is None:
            print("skip  (no close data)")
            continue

        pm_price = get_premarket_price(ticker)
        if pm_price is None:
            print("skip  (no pre-market data)")
            continue

        volume_ratio = get_volume_ratio(ticker)

        setup = build_setup(ticker, prev_close, pm_price, volume_ratio, account_size, risk_pct)
        all_setups.append(setup)

        status = "OK" if setup.valid else "filtered"
        print(f"gap {setup.gap_pct:+.2f}%  vol {volume_ratio:.1f}x  [{status}]")

    # ── Summary ────────────────────────────────────────────────────────────────
    valid = [s for s in all_setups if s.valid]
    bar   = "=" * 54

    print(f"\n{bar}")
    print(f"  SCAN COMPLETE  —  {len(valid)} valid setup(s) of {len(all_setups)} scanned")
    print(bar)

    if valid:
        ranked = sorted(valid, key=lambda x: x.gap_pct, reverse=True)
        for s in ranked:
            display_setup(s)
    else:
        print("\n  No qualifying gap setups found right now.")
        gapped = sorted(
            [s for s in all_setups if s.gap_pct > 0],
            key=lambda x: x.gap_pct, reverse=True
        )[:5]
        if gapped:
            print("\n  Closest gap-up candidates (did not pass all filters — trade levels shown for reference):")
            for s in gapped:
                display_setup(s, show_trade_levels=True)

    print()
    return valid


# ── CLI entry point ────────────────────────────────────────────────────────────
def main() -> None:
    global MIN_GAP_PCT
    parser = argparse.ArgumentParser(
        description="Gap and Go pre-market scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gap_scanner.py                          # scan default watchlist
  python gap_scanner.py AAPL NVDA TSLA           # scan specific tickers
  python gap_scanner.py --account 25000 --risk 2 # larger account, 2% risk
  python gap_scanner.py --min-gap 3.0            # stricter gap filter
        """,
    )
    parser.add_argument("tickers",    nargs="*",        help="Tickers to scan (default: built-in watchlist)")
    parser.add_argument("--account",  type=float,       default=ACCOUNT_SIZE,    metavar="$",  help="Account size in dollars")
    parser.add_argument("--risk",     type=float,       default=RISK_PCT,        metavar="%%", help="Risk per trade as %% of account")
    parser.add_argument("--min-gap",  type=float,       default=MIN_GAP_PCT,     metavar="%%", help="Minimum gap %% to qualify")

    args = parser.parse_args()

    watchlist = [t.upper() for t in args.tickers] if args.tickers else DEFAULT_WATCHLIST

    MIN_GAP_PCT = args.min_gap

    scan(watchlist, account_size=args.account, risk_pct=args.risk)


if __name__ == "__main__":
    main()

