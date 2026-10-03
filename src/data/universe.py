"""Static metadata for the expanded equity research universe.

The stock list is intentionally sector balanced and restricted to securities
with long public histories. It is an exploratory universe when sourced from
Yahoo Finance because that source does not provide delisting-complete,
point-in-time index membership data.
"""

from __future__ import annotations

import json
from pathlib import Path


EXPANDED_EQUITY_SECTORS: dict[str, str] = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "CSCO": "Information Technology",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "VZ": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "WMT": "Consumer Staples",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "JNJ": "Health Care",
    "MRK": "Health Care",
    "UNH": "Health Care",
    "CAT": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "APD": "Materials",
    "NEM": "Materials",
    "FCX": "Materials",
}

DEFENSIVE_ASSET_SECTORS: dict[str, str] = {
    "SPY": "Broad Equity",
    "IEF": "Treasury Bonds",
    "GLD": "Gold",
    "BIL": "Cash Equivalent",
}

EXPANDED_TICKERS = [
    *EXPANDED_EQUITY_SECTORS,
    *DEFENSIVE_ASSET_SECTORS,
]
EXPANDED_SECTORS = {
    **EXPANDED_EQUITY_SECTORS,
    **DEFENSIVE_ASSET_SECTORS,
}


def write_universe_manifest(path: Path) -> Path:
    """Persist the universe assumptions beside generated experiment data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tickers": EXPANDED_TICKERS,
        "sectors": EXPANDED_SECTORS,
        "equity_count": len(EXPANDED_EQUITY_SECTORS),
        "defensive_asset_count": len(DEFENSIVE_ASSET_SECTORS),
        "selection_rule": (
            "Long-history, sector-balanced exploratory universe fixed before "
            "model evaluation. Three equities per sector plus four common "
            "defensive/reference assets."
        ),
        "survivorship_warning": (
            "Yahoo Finance does not provide point-in-time index membership or "
            "complete delisting returns. Results from this universe must be "
            "labelled exploratory unless replaced with point-in-time data."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
