"""
predict.py — Inference helper for the trained XGBoost model.

Loads the saved model bundle and provides a clean predict() API.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, CLASS_NAMES, latest_feature_row, add_ml_features, normalize_ohlcv

MODEL_PATH = "models/xgboost_model.pkl"


def load_model(path: str = MODEL_PATH) -> dict:
    """Load the model bundle (dict with model + metadata)."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Invalid model bundle — expected dict with 'model' key.")
    return bundle


def predict_single(
    ohlcv_df: pd.DataFrame,
    bundle: dict | None = None,
) -> dict:
    """Run inference on OHLCV history for a single ticker.

    Args:
        ohlcv_df: DataFrame with date, open, high, low, close, volume columns.
        bundle:   Pre-loaded model bundle. Loaded from disk if None.

    Returns:
        Dictionary with verdict, probabilities, confidence, and feature values.
    """
    if bundle is None:
        bundle = load_model()

    model = bundle["model"]
    feat_cols = bundle.get("feature_columns", FEATURE_COLUMNS)

    feat_row = latest_feature_row(ohlcv_df)
    if feat_row is None:
        raise ValueError(
            "Could not compute features — need at least 60 trading days of OHLCV data."
        )

    features = pd.DataFrame([feat_row[feat_cols].astype(float)])
    probs = model.predict_proba(features)[0]

    sell_prob = float(probs[0])
    hold_prob = float(probs[1])
    buy_prob  = float(probs[2])

    pred_idx = int(np.argmax(probs))
    verdict  = CLASS_NAMES[pred_idx]

    # Confidence: 50 (random, 1/3) → 95 (certain, 1.0)
    pred_prob  = float(probs[pred_idx])
    confidence = int(50 + min(45, (pred_prob - 1/3) / (2/3) * 45))

    # Feature values for explainability
    feature_values = {col: round(float(feat_row[col]), 6) for col in feat_cols}

    # Additional context from the row
    context = {}
    for col in ["close", "rsi", "macd", "macd_signal", "adx", "sma_20", "sma_50"]:
        if col in feat_row.index and not pd.isna(feat_row[col]):
            context[col] = round(float(feat_row[col]), 4)

    return {
        "verdict":     verdict,
        "confidence":  confidence,
        "probabilities": {
            "SELL": round(sell_prob, 4),
            "HOLD": round(hold_prob, 4),
            "BUY":  round(buy_prob, 4),
        },
        "feature_values": feature_values,
        "context":        context,
    }


def get_feature_importances(bundle: dict) -> dict[str, float]:
    """Extract feature importances from the model bundle."""
    if "feature_importances" in bundle:
        return bundle["feature_importances"]
    model = bundle["model"]
    cols = bundle.get("feature_columns", FEATURE_COLUMNS)
    return dict(zip(cols, model.feature_importances_))
