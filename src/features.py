"""
features.py — Feature engineering for the XGBoost Buy/Sell/Hold classifier.

Ported directly from FinSocial ml-service/ml_features.py.
All 20 features are scale-free (ratios / percentages) — no scaler needed.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

# ── Label parameters ──────────────────────────────────────────────────────────
# 5-day forward return with ±2.5% threshold filters market micro-noise and
# produces high-conviction BUY / SELL / HOLD regimes.
LABEL_HORIZON_DAYS = int(os.environ.get("ML_LABEL_HORIZON_DAYS", "5"))
LABEL_THRESHOLD = float(os.environ.get("ML_LABEL_THRESHOLD", "0.025"))
LABEL_SELL_THRESHOLD = float(os.environ.get("ML_LABEL_SELL_THRESHOLD", "-0.025"))

# ── Feature list (must match training order) ──────────────────────────────────
FEATURE_COLUMNS = [
    # Returns / momentum
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "ret_1d_lag1", "ret_1d_lag2", "roc_10",
    # Volume
    "vol_ratio", "vol_5d_ratio",
    # Oscillators
    "rsi", "stoch_k",
    # MACD (scale-free normalized as % of price)
    "macd", "macd_signal", "macd_hist_slope",
    # Trend
    "adx", "dist_sma20", "dist_sma50", "sma_spread",
    # Volatility / Bollinger / Channels
    "bb_position", "bb_width_ratio", "atr_ratio",
    "volatility_20d", "donchian_pos", "hl_range",
]

CLASS_NAMES = ["SELL", "HOLD", "BUY"]   # index 0, 1, 2
NUM_CLASSES = 3


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase open/high/low/close/volume."""
    out = df.copy()
    mapping = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume", "Date": "date",
    }
    out = out.rename(columns={k: v for k, v in mapping.items() if k in out.columns})
    if "date" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        if out.columns[0] != "date":
            out = out.rename(columns={out.columns[0]: "date"})
    return out


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, Bollinger Bands, and Stochastic %K to OHLCV frame."""
    df = normalize_ohlcv(df).copy()
    close = df["close"].astype(float)

    # Simple moving averages
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()

    # RSI (Wilder's EMA, alpha=1/14 — matches TradingView)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9) normalized by price to remain scale-free across all tickers
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ((ema12 - ema26) / close.replace(0, np.nan)) * 100
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20-period, 2 std)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    # Stochastic %K (14-period)
    lowest_low = df["low"].rolling(14).min()
    highest_high = df["high"].rolling(14).max()
    stoch_range = (highest_high - lowest_low).replace(0, np.nan)
    df["stoch_k"] = (100 * (close - lowest_low) / stoch_range).fillna(50.0)

    return df


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 20 scale-free features for XGBoost."""
    df = calculate_technical_indicators(df)
    close = df["close"].astype(float)
    vol = df["volume"].astype(float).clip(lower=0)
    vol_ma = vol.rolling(20).mean()

    # Returns
    df["ret_1d"]      = close.pct_change(1)
    df["ret_5d"]      = close.pct_change(5)
    df["ret_10d"]     = close.pct_change(10)
    df["ret_20d"]     = close.pct_change(20)
    df["ret_1d_lag1"] = df["ret_1d"].shift(1)
    df["ret_1d_lag2"] = df["ret_1d"].shift(2)
    df["roc_10"]      = (close - close.shift(10)) / close.shift(10).replace(0, np.nan)

    # Volume ratios
    df["vol_ratio"]    = np.where(vol_ma > 0, vol / vol_ma, 1.0)
    vol_5d_ma          = vol.rolling(5).mean()
    df["vol_5d_ratio"] = np.where(vol_ma > 0, vol_5d_ma / vol_ma, 1.0)

    # MACD histogram slope (momentum acceleration)
    df["macd_hist"]       = df["macd"] - df["macd_signal"]
    df["macd_hist_slope"] = df["macd_hist"].diff(1)

    # Distance from moving averages (fractional)
    df["dist_sma20"] = (close - df["sma_20"]) / df["sma_20"].replace(0, np.nan)
    df["dist_sma50"] = (close - df["sma_50"]) / df["sma_50"].replace(0, np.nan)
    df["sma_spread"] = (df["sma_20"] - df["sma_50"]) / df["sma_50"].replace(0, np.nan)

    # Bollinger position & width
    bb_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_position"]    = ((close - df["bb_lower"]) / bb_width).fillna(0.5)
    df["bb_width_ratio"] = (bb_width / df["sma_20"].replace(0, np.nan)).fillna(0.0)

    # Volatility and range features
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()
    high_p = df["high"].astype(float)
    low_p = df["low"].astype(float)
    high_20 = high_p.rolling(20).max()
    low_20 = low_p.rolling(20).min()
    donchian_range = (high_20 - low_20).replace(0, np.nan)
    df["donchian_pos"] = ((close - low_20) / donchian_range).fillna(0.5)
    df["hl_range"] = (high_p - low_p) / close.replace(0, np.nan)

    # ATR ratio (14-day exponential true range as fraction of price)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    df["atr_ratio"] = atr14 / close.replace(0, np.nan)

    # ADX (Average Directional Index — trend strength, not direction)
    high_p  = df["high"].astype(float)
    low_p   = df["low"].astype(float)
    up_move   = high_p.diff()
    down_move = -low_p.diff()
    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    safe_atr = atr14.replace(0, np.nan)
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / safe_atr
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / safe_atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(span=14, adjust=False).mean().fillna(20.0)

    return df


def add_target(
    df: pd.DataFrame,
    horizon: int = LABEL_HORIZON_DAYS,
    threshold: float = LABEL_THRESHOLD,
    sell_threshold: float = LABEL_SELL_THRESHOLD,
) -> pd.DataFrame:
    """Add 3-class target: 0=SELL, 1=HOLD, 2=BUY based on forward return."""
    df = df.copy()
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    df["target"] = np.select(
        [fwd > threshold, fwd < sell_threshold],
        [2, 0],
        default=1,
    ).astype(int)
    return df


def build_training_frame(
    raw_df: pd.DataFrame,
    horizon: int = LABEL_HORIZON_DAYS,
    threshold: float = LABEL_THRESHOLD,
    sell_threshold: float = LABEL_SELL_THRESHOLD,
) -> pd.DataFrame:
    """Per-ticker feature rows with date, target, and FEATURE_COLUMNS.

    Drops the last `horizon` rows per ticker (they have NaN targets).
    Returns a combined frame ready for train/test splitting.
    """
    raw_df = normalize_ohlcv(raw_df)
    parts: list[pd.DataFrame] = []

    groups = (
        raw_df.groupby("ticker") if "ticker" in raw_df.columns
        else [(None, raw_df)]
    )
    for _ticker, grp in groups:
        grp = grp.sort_values("date") if "date" in grp.columns else grp.reset_index(drop=True)
        grp = add_ml_features(grp)
        grp = add_target(grp, horizon=horizon, threshold=threshold, sell_threshold=sell_threshold)
        grp = grp.iloc[:-horizon]  # drop last N rows (no known future)
        if _ticker is not None:
            grp["ticker"] = _ticker
        parts.append(grp)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.dropna(subset=FEATURE_COLUMNS + ["target"])
    return combined


def latest_feature_row(df: pd.DataFrame) -> pd.Series | None:
    """Extract the most recent valid feature row from OHLCV history.

    Used for single-ticker inference (prediction on today's data).
    """
    df = normalize_ohlcv(df)
    if df.empty or len(df) < 60:
        return None
    enriched = add_ml_features(df)
    if enriched.empty:
        return None

    # Walk back from latest bar in case the last day has NaNs
    for i in range(1, min(8, len(enriched)) + 1):
        row = enriched.iloc[-i]
        subset = row[FEATURE_COLUMNS]
        if not subset.isna().any():
            return row  # return full row (has close, date, etc. for context)
    return None
