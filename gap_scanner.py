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

import urllib.request
import urllib.parse
import json
import pandas as pd
import pytz
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ──────────────────────────────────────────────────────────────
ACCOUNT_SIZE       = 10_000   # Your account size ($)
RISK_PCT           = 1.0      # Max risk per trade as % of account
MIN_GAP_PCT        = 2.0      # Minimum gap % to qualify
MAX_GAP_PCT        = 5.0      # Maximum gap % to qualify
CAUTION_GAP_PCT    = 10.0     # Gap > this is a caution flag
MIN_VOLUME_RATIO   = 2.0      # Pre-market volume must be Nx avg
VOLUME_LOOKBACK    = 5        # Trading days to average pre-mkt vol over
MIN_MARKET_CAP     = 5_000_000_000  # Minimum market cap ($5 billion)

NEWSAPI_KEY  = ""             # Get a free key at https://newsapi.org/register
POLYGON_KEY  = "fPmjWyQMbRNvocDUPYoVgCO7o3GV8yWj"

# Keywords per catalyst type — matched against headline + description
CATALYST_KEYWORDS: dict[str, list[str]] = {
    "Earnings Beat":    ["earnings beat", "eps beat", "beat estimates", "beat expectations",
                         "quarterly results", "revenue beat", "profit surge", "raised guidance",
                         "earnings surprise"],
    "Acquisition/M&A":  ["acquisition", "merger", "acquire", "buyout", "takeover",
                         "to buy ", "buys ", "acquired by", "merge with"],
    "Major Contract":   ["contract", "awarded", "partnership", "collaboration", "licensing deal",
                         "supply agreement", "multi-year deal", "government contract"],
    "Analyst Upgrade":  ["upgrade", "price target raised", "outperform", "buy rating",
                         "overweight", "strong buy", "initiated coverage"],
    "Geopolitical":     ["war", "conflict", "sanctions", "military", "stimulus package",
                         "defense spending", "opec", "oil supply", "crude oil",
                         "tariff", "trade war", "geopolitical"],
    "Macro / Fed":      ["interest rate", "federal reserve", "fed rate", "rate hike",
                         "rate cut", "fomc", "inflation data", "cpi report",
                         "gdp growth", "monetary policy", "treasury yield"],
}

DEFAULT_WATCHLIST = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "ORCL", "CRM", "ADBE", "INTC", "QCOM", "TXN", "NOW",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP", "SCHW",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO",
    # Consumer
    "WMT", "COST", "HD", "NKE", "MCD", "SBUX", "TGT",
    # Media / Comms
    "NFLX", "DIS", "CMCSA",
    # High-volatility / momentum
    "COIN", "PLTR", "SOFI", "HOOD", "UBER", "RBLX", "MARA", "SMCI",
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
    atr:              float     = 0.0
    rsi:              float     = 0.0
    sma20:            float     = 0.0
    valid:            bool      = False
    catalyst_type:    str       = ""
    catalyst_headline: str      = ""
    failures:         List[str] = field(default_factory=list)
    warnings:         List[str] = field(default_factory=list)


ATR_PERIOD = 14   # days used to calculate Average True Range

# ── Data fetchers ──────────────────────────────────────────────────────────────
def _ticker_history(ticker: str, **kwargs) -> pd.DataFrame:
    """Fetch history via yf.Ticker — returns simple column DataFrame."""
    t = yf.Ticker(ticker)
    return t.history(**kwargs)


def _polygon_snapshot(ticker: str) -> Optional[dict]:
    """Fetch the Polygon snapshot for a ticker — single API call for all price data."""
    try:
        url = (f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
               f"/{ticker}?apiKey={POLYGON_KEY}")
        req = urllib.request.Request(url, headers={"User-Agent": "gap-scanner/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return data.get("ticker")
    except Exception:
        return None


def get_prev_close(ticker: str) -> Optional[float]:
    """Previous trading day closing price — yfinance primary, Polygon fallback."""
    try:
        hist = _ticker_history(ticker, period="5d", interval="1d", auto_adjust=True)
        if not hist.empty and len(hist) >= 2:
            return float(hist["Close"].iloc[-2])
    except Exception:
        pass
    # Polygon fallback
    snap = _polygon_snapshot(ticker)
    if snap:
        c = snap.get("prevDay", {}).get("c")
        if c:
            return float(c)
    return None


def get_premarket_price(ticker: str) -> Optional[float]:
    """
    Latest pre-market price.
    yfinance primary (1m bars with prepost), Polygon snapshot fallback.
    """
    try:
        hist = _ticker_history(ticker, period="2d", interval="1m", prepost=True)
        if not hist.empty:
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
            day_rows = hist[hist.index.date == today]
            if not day_rows.empty:
                return float(day_rows["Close"].iloc[-1])
    except Exception:
        pass
    # Polygon fallback
    snap = _polygon_snapshot(ticker)
    if snap:
        for key in ("fmv", ):
            val = snap.get(key)
            if val:
                return float(val)
        last = snap.get("lastTrade", {}).get("p")
        if last:
            return float(last)
    return None


def get_market_cap(ticker: str) -> Optional[float]:
    """Return market cap in dollars — yfinance primary, Polygon fallback."""
    try:
        return float(yf.Ticker(ticker).fast_info.market_cap)
    except Exception:
        pass
    snap = _polygon_snapshot(ticker)
    if snap:
        mc = snap.get("market_cap") or snap.get("marketCap")
        if mc:
            return float(mc)
    return None


def get_volume_ratio(ticker: str) -> float:
    """
    Today's volume vs 10-day average — yfinance primary, Polygon fallback.
    Returns 0.0 if data is unavailable.
    """
    try:
        fi        = yf.Ticker(ticker).fast_info
        today_vol = fi.last_volume
        avg_vol   = fi.ten_day_average_volume
        if avg_vol and avg_vol > 0:
            return round(today_vol / avg_vol, 2)
    except Exception:
        pass
    # Polygon fallback
    snap = _polygon_snapshot(ticker)
    if snap:
        today_vol = snap.get("day", {}).get("v", 0)
        prev_vol  = snap.get("prevDay", {}).get("v", 0)
        if prev_vol and prev_vol > 0:
            return round(today_vol / prev_vol, 2)
    return 0.0


def get_rsi(ticker: str, period: int = 14) -> Optional[float]:
    """14-day RSI. Returns None if data unavailable."""
    try:
        hist = _ticker_history(ticker, period=f"{period + 10}d", interval="1d", auto_adjust=True)
        if len(hist) < period + 1:
            return None
        delta = hist["Close"].diff()
        gain  = delta.clip(lower=0).tail(period).mean()
        loss  = (-delta.clip(upper=0)).tail(period).mean()
        if loss == 0:
            return 100.0
        rs = gain / loss
        return float(100 - (100 / (1 + rs)))
    except Exception:
        return None


def get_sma(ticker: str, period: int = 20) -> Optional[float]:
    """20-day simple moving average of closing price."""
    try:
        hist = _ticker_history(ticker, period=f"{period + 5}d", interval="1d", auto_adjust=True)
        if len(hist) < period:
            return None
        return float(hist["Close"].tail(period).mean())
    except Exception:
        return None


def get_atr(ticker: str, period: int = ATR_PERIOD) -> Optional[float]:
    """
    14-day Average True Range — measures how much the stock normally moves per day.
    Used to set volatility-adaptive stop and target levels.
    """
    try:
        hist = _ticker_history(ticker, period=f"{period + 5}d", interval="1d", auto_adjust=True)
        if len(hist) < period:
            return None
        high  = hist["High"]
        low   = hist["Low"]
        close = hist["Close"].shift(1)
        tr    = pd.concat([
            high - low,
            (high - close).abs(),
            (low  - close).abs(),
        ], axis=1).max(axis=1)
        return float(tr.tail(period).mean())
    except Exception:
        return None


# ── Market condition ──────────────────────────────────────────────────────────
def get_spy_change() -> float:
    """
    Return SPY % change based on pre-market price vs previous close.
    This reflects market sentiment before the open, which is when the scanner runs.
    Falls back to regular session change if pre-market data is unavailable.
    """
    try:
        prev_close = get_prev_close("SPY")
        if not prev_close:
            return 0.0
        pm_price = get_premarket_price("SPY")
        if not pm_price:
            return 0.0
        return float((pm_price - prev_close) / prev_close * 100)
    except Exception:
        return 0.0


def market_multiplier(spy_chg: float) -> float:
    """
    Widen stops/targets in strong up markets, tighten in down markets.
    SPY > +0.5%  → 1.2x  (more room to run)
    SPY < -0.5%  → 0.8x  (reduce exposure in weak market)
    Otherwise    → 1.0x
    """
    if spy_chg > 0.5:
        return 1.2
    if spy_chg < -0.5:
        return 0.8
    return 1.0


# ── News & catalyst detection ─────────────────────────────────────────────────
def _classify(text: str) -> tuple[str, str]:
    """Return (catalyst_type, matched_keyword) or ('', '') if no match."""
    lower = text.lower()
    for catalyst, keywords in CATALYST_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return catalyst, kw
    return "", ""


def fetch_yahoo_news(ticker: str) -> list[dict]:
    """Return list of {title, summary} from Yahoo Finance via yfinance."""
    try:
        t = yf.Ticker(ticker)
        items = t.news or []
        results = []
        for item in items[:10]:
            content = item.get("content", {})
            title   = content.get("title", "") or item.get("title", "")
            summary = content.get("summary", "") or ""
            if title:
                results.append({"title": title, "summary": summary})
        return results
    except Exception:
        return []


def fetch_newsapi_news(ticker: str, company_name: str) -> list[dict]:
    """Return list of {title, summary} from NewsAPI (requires NEWSAPI_KEY)."""
    if not NEWSAPI_KEY:
        return []
    try:
        query   = urllib.parse.quote(f"{ticker} OR {company_name}")
        url     = (f"https://newsapi.org/v2/everything?q={query}"
                   f"&sortBy=publishedAt&pageSize=10&language=en"
                   f"&apiKey={NEWSAPI_KEY}")
        req  = urllib.request.Request(url, headers={"User-Agent": "gap-scanner/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return [
            {"title": a.get("title", ""), "summary": a.get("description", "") or ""}
            for a in data.get("articles", [])
            if a.get("title")
        ]
    except Exception:
        return []


def detect_catalyst(ticker: str) -> tuple[str, str]:
    """
    Scan Yahoo Finance + NewsAPI headlines for a known catalyst.
    Returns (catalyst_type, headline) — both empty strings if nothing found.
    """
    t    = yf.Ticker(ticker)
    name = ""
    try:
        name = (t.fast_info.get("longName", "") or
                t.info.get("shortName", "") or ticker)
    except Exception:
        name = ticker

    articles = fetch_yahoo_news(ticker) + fetch_newsapi_news(ticker, name)

    for article in articles:
        combined = article["title"] + " " + article["summary"]
        cat, _   = _classify(combined)
        if cat:
            headline = article["title"][:120]
            return cat, headline

    return "", ""


# ── Setup calculation ──────────────────────────────────────────────────────────
def build_setup(
    ticker: str,
    prev_close: float,
    pm_price: float,
    volume_ratio: float,
    catalyst_type: str = "",
    catalyst_headline: str = "",
    mkt_mult: float = 1.0,
    atr: Optional[float] = None,
    rsi: Optional[float] = None,
    sma20: Optional[float] = None,
    spy_gap_pct: float = 0.0,
    account_size: float = ACCOUNT_SIZE,
    risk_pct: float = RISK_PCT,
) -> TradeSetup:

    gap_pct = (pm_price - prev_close) / prev_close * 100

    entry = pm_price

    if atr and atr > 0:
        # Volatility-adaptive: stop = 1× ATR below entry, target = 1.5× ATR above (× market mult)
        stop_dist = atr * mkt_mult
        target_dist = atr * 1.5 * mkt_mult
    else:
        # Fallback: 50% of gap for stop, 1.5:1 R:R for target
        stop_dist   = (pm_price - prev_close) * 0.5 * mkt_mult
        target_dist = stop_dist * 1.5

    stop           = entry - stop_dist
    target         = entry + target_dist
    risk_per_share = max(stop_dist, 0.01)
    reward         = target_dist
    rr_ratio       = reward / risk_per_share

    dollar_risk = account_size * (risk_pct / 100)
    shares      = max(int(dollar_risk / risk_per_share), 0)

    failures: List[str] = []
    warnings: List[str] = []

    # ── Core filters ───────────────────────────────────────────────────────────
    if gap_pct < MIN_GAP_PCT:
        failures.append(f"Gap {gap_pct:.2f}% below minimum {MIN_GAP_PCT}%")
    if gap_pct > MAX_GAP_PCT:
        failures.append(f"Gap {gap_pct:.2f}% above maximum {MAX_GAP_PCT}% — too extended")
    if volume_ratio < MIN_VOLUME_RATIO:
        failures.append(f"Volume {volume_ratio:.1f}x below minimum {MIN_VOLUME_RATIO}x")

    # ── Catalyst (hard requirement) ─────────────────────────────────────────────
    if not catalyst_type:
        failures.append("No news catalyst detected — skip until confirmed")

    # ── RSI: not overbought, not oversold ──────────────────────────────────────
    if rsi is not None:
        if rsi > 70:
            failures.append(f"RSI {rsi:.0f} — overbought, gap likely to fade")
        elif rsi < 30:
            failures.append(f"RSI {rsi:.0f} — oversold, avoid gap-up in weak trend")

    # ── Trend: price above 20-day SMA ──────────────────────────────────────────
    if sma20 is not None and prev_close < sma20:
        failures.append(f"Price ${prev_close:.2f} below 20-day SMA ${sma20:.2f} — downtrend")

    # ── Relative strength: stock must gap more than SPY ────────────────────────
    if spy_gap_pct > 0 and gap_pct <= spy_gap_pct:
        failures.append(f"Gap {gap_pct:.2f}% not exceeding SPY {spy_gap_pct:.2f}% — no relative strength")

    # ── ATR ratio: gap meaningful but not exhausted ────────────────────────────
    if atr and atr > 0:
        gap_atr_ratio = (pm_price - prev_close) / atr
        if gap_atr_ratio < 0.5:
            failures.append(f"Gap only {gap_atr_ratio:.1f}× ATR — too small to be meaningful")
        elif gap_atr_ratio > 2.0:
            failures.append(f"Gap {gap_atr_ratio:.1f}× ATR — move likely exhausted pre-market")

    if gap_pct > CAUTION_GAP_PCT:
        warnings.append(f"Gap > {CAUTION_GAP_PCT}% — momentum may already be played out")

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
        atr=round(atr, 2) if atr else 0.0,
        rsi=round(rsi, 1) if rsi else 0.0,
        sma20=round(sma20, 2) if sma20 else 0.0,
        valid=len(failures) == 0,
        catalyst_type=catalyst_type,
        catalyst_headline=catalyst_headline,
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
    if s.rsi:
        rsi_label = "overbought" if s.rsi > 70 else ("oversold" if s.rsi < 30 else "healthy")
        print(f"  RSI (14):       {s.rsi:.0f}  [{rsi_label}]")
    if s.sma20:
        trend = "above" if s.prev_close >= s.sma20 else "BELOW"
        print(f"  20-day SMA:     ${s.sma20:.2f}  [price {trend} SMA]")
    if s.atr:
        print(f"  ATR (14):       ${s.atr:.2f}  [daily volatility range]")
    if s.catalyst_type:
        print(f"  Catalyst:       [{s.catalyst_type}]")
        print(f"  Headline:       {s.catalyst_headline}")
    else:
        print(f"  Catalyst:       None detected")

    if not s.valid:
        for f in s.failures:
            print(f"  FAIL: {f}")

    if show_trade_levels:
        print(f"  {'-' * 44}")
        print(f"  Entry:          ${s.entry_price:.2f}  (limit ~0.1% above ask after 9:35 AM)")
        print(f"  CLOSE AT LOSS:  ${s.stop_loss:.2f}  — stop-loss at 50% of pre-market gap")
        print(f"  CLOSE AT PROFIT:${s.target_price:.2f}  — 1.5:1 target ({s.rr_ratio:.1f}x your risk)")
        print(f"  Risk / share:   ${s.risk_per_share:.2f}")
        print(f"  Reward / share: ${s.reward_per_share:.2f}")
        position_value = s.shares * s.entry_price
        pct_of_account = (position_value / ACCOUNT_SIZE) * 100
        print(f"  Position Size:  {s.shares} shares")
        print(f"  Capital Needed: ${position_value:,.0f}  ({pct_of_account:.1f}% of account)")
        print(f"  Max Loss:       ${s.dollar_risk:.0f}  ({RISK_PCT}% of account)")
        print(f"  Max Gain:       ${s.shares * s.reward_per_share:,.0f}")
        print(f"  Hard exit by:   11:00 AM ET regardless of P&L")

    for w in s.warnings:
        print(f"  NOTE: {w}")


# ── Chart ─────────────────────────────────────────────────────────────────────
def plot_setup(s: TradeSetup) -> None:
    """Show a price chart with the last 20 daily closes + trade levels."""
    try:
        hist = _ticker_history(s.ticker, period="30d", interval="1d", auto_adjust=True)
        if hist.empty:
            print(f"  [chart] No price history for {s.ticker}")
            return
        hist = hist.tail(20)
    except Exception:
        print(f"  [chart] Could not fetch history for {s.ticker}")
        return

    dates  = list(range(len(hist)))
    closes = hist["Close"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    # Price line
    ax.plot(dates, closes, color="#4fc3f7", linewidth=2, label="Close price")
    ax.plot(dates[-1], closes[-1], "o", color="#4fc3f7", markersize=6)

    # Pre-market price dot (today's gap)
    ax.plot(len(dates), s.premarket_price, "^", color="#ffffff", markersize=9,
            label=f"Pre-mkt ${s.premarket_price:.2f} (gap {s.gap_pct:+.2f}%)", zorder=5)

    x_end = len(dates) + 1  # extend lines slightly past today

    # Stop-loss line
    ax.hlines(s.stop_loss,  xmin=0, xmax=x_end, colors="#ef5350", linewidths=1.5,
              linestyles="--", label=f"STOP LOSS  ${s.stop_loss:.2f}")

    # Entry line
    ax.hlines(s.entry_price, xmin=0, xmax=x_end, colors="#ffffff", linewidths=1.5,
              linestyles="--", label=f"ENTRY      ${s.entry_price:.2f}")

    # Target line
    ax.hlines(s.target_price, xmin=0, xmax=x_end, colors="#66bb6a", linewidths=1.5,
              linestyles="--", label=f"TARGET     ${s.target_price:.2f}")

    # Shaded risk / reward zones
    ax.axhspan(s.stop_loss,  s.entry_price,  alpha=0.12, color="#ef5350")
    ax.axhspan(s.entry_price, s.target_price, alpha=0.12, color="#66bb6a")

    # Labels on the right edge
    right = x_end + 0.1
    ax.text(right, s.stop_loss,   f" ${s.stop_loss:.2f}",   color="#ef5350", va="center", fontsize=8)
    ax.text(right, s.entry_price,  f" ${s.entry_price:.2f}",  color="#ffffff", va="center", fontsize=8)
    ax.text(right, s.target_price, f" ${s.target_price:.2f}", color="#66bb6a", va="center", fontsize=8)

    # Axes styling
    ax.set_xlim(-0.5, x_end + 2)
    ax.set_title(f"{s.ticker}  —  Gap & Go Setup  |  R:R {s.rr_ratio:.0f}:1  |  "
                 f"{s.shares} shares  (${s.dollar_risk:.0f} at risk)",
                 color="white", fontsize=12, pad=10)
    ax.set_xlabel("Trading days (most recent = right)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Price ($)", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white", fontsize=8)
    ax.grid(axis="y", color="#222222", linewidth=0.5)

    plt.tight_layout()
    plt.show()


# ── Main scanner ───────────────────────────────────────────────────────────────
def scan(
    watchlist: List[str] = DEFAULT_WATCHLIST,
    account_size: float = ACCOUNT_SIZE,
    risk_pct: float = RISK_PCT,
) -> List[TradeSetup]:

    now_et  = datetime.now(ET)
    spy_chg = get_spy_change()
    mult    = market_multiplier(spy_chg)

    if spy_chg > 0.5:
        mkt_label = f"SPY {spy_chg:+.2f}%  — BULL DAY  (stops & targets widened 20%)"
    elif spy_chg < -0.5:
        mkt_label = f"SPY {spy_chg:+.2f}%  — BEAR DAY  (stops & targets tightened 20%)"
    else:
        mkt_label = f"SPY {spy_chg:+.2f}%  — NEUTRAL"

    print(f"\nGap and Go Scanner  —  {now_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"Market:  {mkt_label}")

    if now_et.weekday() == 0:
        print("  WARNING: Monday gaps are less reliable — trade cautiously today.")

    print(f"Account: ${account_size:,.0f}  |  Risk: {risk_pct}%/trade  |  Min gap: {MIN_GAP_PCT}%")
    print(f"Scanning {len(watchlist)} tickers (this may take ~2 minutes)...\n")

    all_setups: List[TradeSetup] = []

    for ticker in watchlist:
        print(f"  {ticker:<6} ...", end=" ", flush=True)

        mkt_cap = get_market_cap(ticker)
        if mkt_cap is None or mkt_cap < MIN_MARKET_CAP:
            cap_str = f"${mkt_cap/1e9:.1f}B" if mkt_cap else "unknown"
            print(f"skip  (market cap {cap_str} < $5B)")
            continue

        prev_close = get_prev_close(ticker)
        if prev_close is None:
            print("skip  (no close data)")
            continue

        pm_price = get_premarket_price(ticker)
        if pm_price is None:
            print("skip  (no pre-market data)")
            continue

        volume_ratio = get_volume_ratio(ticker)
        atr          = get_atr(ticker)
        rsi          = get_rsi(ticker)
        sma20        = get_sma(ticker)
        catalyst_type, catalyst_headline = detect_catalyst(ticker)

        setup = build_setup(ticker, prev_close, pm_price, volume_ratio,
                            catalyst_type, catalyst_headline, mult, atr,
                            rsi, sma20, spy_chg, account_size, risk_pct)
        all_setups.append(setup)

        status    = "OK" if setup.valid else "filtered"
        cat_label = f"  catalyst: {catalyst_type}" if catalyst_type else "  no catalyst"
        print(f"gap {setup.gap_pct:+.2f}%  vol {volume_ratio:.1f}x  [{status}]{cat_label}")

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
            plot_setup(s)
    else:
        print("\n  No qualifying gap setups found right now.")
        gapped = sorted(
            [s for s in all_setups if s.gap_pct > 0],
            key=lambda x: x.gap_pct, reverse=True
        )[:5]
        if gapped:
            print("\n  Closest gap-up candidates (did not pass all filters):")
            for s in gapped:
                gap_failed = any("maximum" in f or "minimum" in f and "Gap" in f for f in s.failures)
                display_setup(s, show_trade_levels=not gap_failed)
                if not gap_failed:
                    plot_setup(s)

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

