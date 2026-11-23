"""
data.py — Data loading for FinSocial-Lite.

Fetches OHLCV data via yfinance (free, no API key).
Also provides a synthetic data fallback for fully offline use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Default tickers: a diverse mix of large-cap US equities.
# (US tickers are the most reliably supported by yfinance across versions.)
DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "JPM",
    "JNJ",
    "V",
    "WMT",
    "XOM",
]

SAMPLE_CSV_PATH = "data/sample_ohlcv.csv"


def fetch_yfinance(
    tickers: list[str] | None = None,
    period: str = "5y",
) -> pd.DataFrame:
    """Download OHLCV history for *tickers* via yfinance.

    Returns a long-format DataFrame with columns:
        date, ticker, open, high, low, close, volume
    """
    import yfinance as yf

    tickers = tickers or DEFAULT_TICKERS
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if data.empty:
                print(f"  ⚠ No data for {ticker}, skipping.")
                continue
            data = data.reset_index()
            data = data.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
            })
            data["ticker"] = ticker
            data = data[["date", "ticker", "open", "high", "low", "close", "volume"]]
            frames.append(data)
            print(f"  ✓ {ticker}: {len(data)} rows")
        except Exception as e:
            print(f"  ✗ {ticker} failed: {e}")

    if not frames:
        raise RuntimeError("Could not fetch data for any ticker.")
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined


def make_synthetic(n_per_ticker: int = 2000, n_tickers: int = 5) -> pd.DataFrame:
    """Generate synthetic OHLCV data for offline / CI use.

    Returns the same long-format DataFrame as fetch_yfinance.
    """
    rng = np.random.RandomState(42)
    frames: list[pd.DataFrame] = []
    dates = pd.bdate_range("2018-01-01", periods=n_per_ticker)

    for i in range(n_tickers):
        base_price = 500 + i * 300
        prices = base_price + np.cumsum(rng.randn(n_per_ticker) * (base_price * 0.01))
        prices = np.maximum(prices, 10.0)  # floor at ₹10
        noise = rng.randn(n_per_ticker)
        df = pd.DataFrame({
            "date": dates[:n_per_ticker],
            "ticker": f"SYNTH{i+1}.NS",
            "open":   prices * (1 + noise * 0.003),
            "high":   prices * (1 + np.abs(rng.randn(n_per_ticker)) * 0.012),
            "low":    prices * (1 - np.abs(rng.randn(n_per_ticker)) * 0.012),
            "close":  prices,
            "volume": rng.randint(100_000, 5_000_000, n_per_ticker).astype(float),
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_sample_csv(path: str = SAMPLE_CSV_PATH) -> pd.DataFrame:
    """Load a previously saved CSV dataset."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def save_sample_csv(df: pd.DataFrame, path: str = SAMPLE_CSV_PATH) -> None:
    """Persist a DataFrame to CSV for reproducible offline use."""
    df.to_csv(path, index=False)
    print(f"  💾 Saved {len(df)} rows to {path}")
