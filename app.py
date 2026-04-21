#!/usr/bin/env python3
"""
Gap & Go Web Dashboard
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify
from datetime import datetime
import pytz

from gap_scanner import (
    DEFAULT_WATCHLIST, ACCOUNT_SIZE, RISK_PCT, MIN_GAP_PCT, MAX_GAP_PCT,
    MIN_MARKET_CAP, MIN_VOLUME_RATIO, ET,
    get_spy_change, market_multiplier,
    get_prev_close, get_premarket_price, get_market_cap,
    get_volume_ratio, get_atr, get_rsi, get_sma,
    detect_catalyst, build_setup, _ticker_history,
)

app = Flask(__name__)


def get_history(ticker: str) -> list[dict]:
    """Last 20 daily closes for the chart."""
    try:
        hist = _ticker_history(ticker, period="30d", interval="1d", auto_adjust=True)
        hist = hist.tail(20)
        return [
            {"date": str(idx.date()), "close": round(float(row["Close"]), 2)}
            for idx, row in hist.iterrows()
        ]
    except Exception:
        return []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/spy")
def spy():
    spy_chg = get_spy_change()
    mult    = market_multiplier(spy_chg)
    if spy_chg > 0.5:
        label = "BULL DAY"
    elif spy_chg < -0.5:
        label = "BEAR DAY"
    else:
        label = "NEUTRAL"
    return jsonify({"spy_chg": round(spy_chg, 2), "mult": round(mult, 2), "label": label})


@app.route("/scan")
def scan():
    now_et  = datetime.now(ET)
    spy_chg = get_spy_change()
    mult    = market_multiplier(spy_chg)

    all_setups = []
    scanned    = 0

    for ticker in DEFAULT_WATCHLIST:
        mkt_cap = get_market_cap(ticker)
        if mkt_cap is None or mkt_cap < MIN_MARKET_CAP:
            continue

        prev_close = get_prev_close(ticker)
        if prev_close is None:
            continue

        pm_price = get_premarket_price(ticker)
        if pm_price is None:
            continue

        scanned += 1

        volume_ratio           = get_volume_ratio(ticker)
        atr                    = get_atr(ticker)
        rsi                    = get_rsi(ticker)
        sma20                  = get_sma(ticker)
        catalyst_type, cat_hl  = detect_catalyst(ticker)
        history                = get_history(ticker)

        setup = build_setup(
            ticker, prev_close, pm_price, volume_ratio,
            catalyst_type, cat_hl, mult, atr, rsi, sma20, spy_chg
        )

        all_setups.append({
            "ticker":           setup.ticker,
            "prev_close":       setup.prev_close,
            "premarket_price":  setup.premarket_price,
            "gap_pct":          setup.gap_pct,
            "volume_ratio":     setup.volume_ratio,
            "entry_price":      setup.entry_price,
            "stop_loss":        setup.stop_loss,
            "target_price":     setup.target_price,
            "risk_per_share":   setup.risk_per_share,
            "reward_per_share": setup.reward_per_share,
            "rr_ratio":         setup.rr_ratio,
            "shares":           setup.shares,
            "dollar_risk":      setup.dollar_risk,
            "atr":              setup.atr,
            "rsi":              setup.rsi,
            "sma20":            setup.sma20,
            "valid":            setup.valid,
            "catalyst_type":    setup.catalyst_type,
            "catalyst_headline": setup.catalyst_headline,
            "failures":         setup.failures,
            "warnings":         setup.warnings,
            "history":          history,
        })

    valid      = [s for s in all_setups if s["valid"]]
    candidates = sorted(
        [s for s in all_setups if not s["valid"] and s["gap_pct"] > 0],
        key=lambda x: x["gap_pct"], reverse=True
    )[:5]

    return jsonify({
        "valid":      valid,
        "candidates": candidates,
        "total":      scanned,
        "spy_chg":    round(spy_chg, 2),
        "mult":       round(mult, 2),
        "timestamp":  now_et.strftime("%Y-%m-%d %H:%M %Z"),
    })


if __name__ == "__main__":
    print("\nGap & Go Dashboard running at http://localhost:5000\n")
    app.run(debug=False, port=5000)
