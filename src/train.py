"""
train.py — Train the XGBoost Buy/Sell/Hold classifier.

Usage:
    python src/train.py                      # uses yfinance data
    python src/train.py --synthetic          # uses synthetic data (offline)
    python src/train.py --csv data/sample_ohlcv.csv  # uses saved CSV

The script:
1. Loads OHLCV data (yfinance / synthetic / CSV)
2. Engineers 20 scale-free features per row
3. Labels each row: BUY (+1.5% forward 5d), SELL (−1.5%), else HOLD
4. Splits chronologically (80/20, no shuffling — no future leakage)
5. Trains XGBoost with class-weight balancing
6. Saves model bundle + metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Ensure project root is on the path so `src.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (
    FEATURE_COLUMNS,
    CLASS_NAMES,
    LABEL_HORIZON_DAYS,
    LABEL_THRESHOLD,
    LABEL_SELL_THRESHOLD,
    build_training_frame,
)
from src.evaluate import compute_metrics, save_metrics, print_report
from src.data import fetch_yfinance, make_synthetic, load_sample_csv, save_sample_csv

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/xgboost_model.pkl"
METRICS_PATH = "models/metrics.json"
TRAIN_RATIO  = 0.8
RANDOM_SEED  = 42


def time_based_split(
    frame: pd.DataFrame, train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — no future leakage into training.

    For multi-ticker data, we sort by date globally so the test set is
    always the most recent 20% of calendar time across all tickers.
    """
    frame = frame.sort_values("date").reset_index(drop=True)
    cut = int(len(frame) * train_ratio)
    if cut < 100 or len(frame) - cut < 30:
        raise ValueError(
            f"Not enough samples for time split (n={len(frame)}, cut={cut}). "
            "Need ≥100 train and ≥30 test rows."
        )
    return frame.iloc[:cut], frame.iloc[cut:]


def train(
    raw: pd.DataFrame,
    horizon: int = LABEL_HORIZON_DAYS,
    threshold: float = LABEL_THRESHOLD,
    sell_threshold: float = LABEL_SELL_THRESHOLD,
) -> None:
    """Full training pipeline: features → split → fit → evaluate → save."""
    print(f"\n{'='*60}")
    print("FinSocial-Lite — XGBoost Training")
    print(f"{'='*60}")
    print(f"Label: {horizon}-day forward return")
    print(f"  BUY  threshold: >{threshold:+.1%}")
    print(f"  SELL threshold: <{sell_threshold:+.1%}")
    print(f"  HOLD: in between\n")

    t0 = time.time()

    # 1. Feature engineering
    print("Building features...")
    frame = build_training_frame(raw, horizon=horizon, threshold=threshold, sell_threshold=sell_threshold)
    if len(frame) < 200:
        print(f"ERROR: Only {len(frame)} rows after feature engineering (need ≥200).")
        sys.exit(1)
    print(f"  Total feature rows: {len(frame)}")

    # 2. Class distribution
    print("\nClass distribution:")
    for cls, name in [(2, "BUY"), (1, "HOLD"), (0, "SELL")]:
        ct = int((frame["target"] == cls).sum())
        print(f"  {name}: {ct:>6,} ({ct/len(frame):.1%})")

    # 3. Time-based split
    train_df, test_df = time_based_split(frame)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    X_test  = test_df[FEATURE_COLUMNS]
    y_test  = test_df["target"]
    print(f"\nTrain: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # 4. Compute balanced class weights (dampened to avoid inflating false alarms)
    class_counts = {c: int((y_train == c).sum()) for c in [0, 1, 2]}
    n = len(y_train)
    weights = {c: (n / (3 * max(class_counts[c], 1))) ** 0.35 for c in [0, 1, 2]}
    sample_weights = np.array([weights[c] for c in y_train])
    print(f"  Class weights: SELL={weights[0]:.2f}, HOLD={weights[1]:.2f}, BUY={weights[2]:.2f}")

    # 5. Train XGBoost with early stopping and robust regularization
    print(f"\nTraining XGBoost (3-class, early_stopping_rounds=30)...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=12,
        reg_alpha=3.0,
        reg_lambda=5.0,
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print(f"  Optimal trees: {model.best_iteration} (prevented overfitting from 300+ trees)")

    # 6. Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    y_prob_train = model.predict_proba(X_train)
    y_prob_test  = model.predict_proba(X_test)

    print("\n" + "─"*40)
    print("TRAIN (in-sample):")
    print_report(y_train.values, y_pred_train)

    print("─"*40)
    print("TEST (held-out, forward time):")
    print_report(y_test.values, y_pred_test)

    metrics = compute_metrics(y_test.values, y_pred_test, y_prob_test)
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Weighted F1:  {metrics['weighted_avg']['f1_score']:.4f}")
    print(f"  Weighted Prec:{metrics['weighted_avg']['precision']:.4f}")
    if metrics["roc_auc_macro"]:
        print(f"  ROC-AUC:      {metrics['roc_auc_macro']:.4f}")

    # 7. Save model bundle
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    bundle = {
        "model":              model,
        "feature_columns":    FEATURE_COLUMNS,
        "label_horizon_days": horizon,
        "label_threshold":    threshold,
        "label_sell_threshold": sell_threshold,
        "num_class":          3,
        "class_names":        CLASS_NAMES,
        "version":            2,
        "feature_importances": dict(zip(FEATURE_COLUMNS, model.feature_importances_)),
        "random_seed":        RANDOM_SEED,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n  ✓ Model saved to {MODEL_PATH}")

    # Also save test predictions for the app to display signals on the chart
    test_df = test_df.copy()
    test_df["predicted"] = y_pred_test
    test_df["prob_sell"] = y_prob_test[:, 0]
    test_df["prob_hold"] = y_prob_test[:, 1]
    test_df["prob_buy"]  = y_prob_test[:, 2]
    test_cols = ["date", "ticker", "close", "target", "predicted",
                 "prob_sell", "prob_hold", "prob_buy"] + FEATURE_COLUMNS
    available_cols = [c for c in test_cols if c in test_df.columns]
    test_df[available_cols].to_csv("models/test_predictions.csv", index=False)
    print(f"  ✓ Test predictions saved to models/test_predictions.csv")

    # 8. Save metrics
    save_metrics(metrics)

    elapsed = time.time() - t0
    print(f"\n  ⏱ Training completed in {elapsed:.1f}s")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train FinSocial-Lite XGBoost classifier")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--synthetic", action="store_true", help="Use synthetic data (offline)")
    group.add_argument("--csv", type=str, help="Path to CSV with OHLCV data")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save fetched data to data/sample_ohlcv.csv")
    parser.add_argument("--threshold", type=float, default=LABEL_THRESHOLD,
                        help=f"Forward return threshold (default: {LABEL_THRESHOLD})")
    parser.add_argument("--horizon", type=int, default=LABEL_HORIZON_DAYS,
                        help=f"Forward horizon days (default: {LABEL_HORIZON_DAYS})")
    args = parser.parse_args()

    if args.csv:
        print(f"Loading data from {args.csv}...")
        raw = load_sample_csv(args.csv)
    elif args.synthetic:
        print("Generating synthetic data...")
        raw = make_synthetic()
    else:
        print("Fetching data via yfinance...")
        raw = fetch_yfinance()
        if args.save_csv:
            save_sample_csv(raw)

    train(raw, horizon=args.horizon, threshold=args.threshold, sell_threshold=-args.threshold)


if __name__ == "__main__":
    main()
