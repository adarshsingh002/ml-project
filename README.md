# FinSocial-Lite

> **A toned-down, standalone XGBoost Buy/Sell/Hold stock classifier** extracted from
> the full [FinSocial](https://github.com/adarshsingh002/FinSocial) project.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/frontend-Streamlit-red)

---

## What It Does

Predicts whether a stock is a **Buy**, **Sell**, or **Hold** over the next **5 trading days**
using 24 technical indicators derived purely from OHLCV (Open/High/Low/Close/Volume) price data.

| Label | Condition |
|-------|-----------|
| **BUY** | 5-day forward return > +2.5% |
| **SELL** | 5-day forward return < −2.5% |
| **HOLD** | Between −2.5% and +2.5% |

*(Thresholds can be customized via `--threshold` and `--horizon` flags)*

## Features Used (24 total)

| Category | Features |
|----------|----------|
| Returns / Momentum | ret_1d, ret_5d, ret_10d, ret_20d, ret_1d_lag1, ret_1d_lag2, roc_10 |
| Volume | vol_ratio, vol_5d_ratio |
| Oscillators | rsi, stoch_k |
| MACD (normalized) | macd (% of price), macd_signal (% of price), macd_hist_slope |
| Trend | adx, dist_sma20, dist_sma50, sma_spread ((sma20 - sma50)/sma50) |
| Volatility & Channels | bb_position, bb_width_ratio, atr_ratio, volatility_20d, donchian_pos, hl_range |

All features are **scale-free** (ratios, percentages) — ensuring cross-asset comparability between different priced stocks ($50 vs $500).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (fetches ~5 years of data via yfinance)
python src/train.py

# 3. Launch the dashboard
streamlit run app.py
```

For **offline / CI** use with synthetic data:
```bash
python src/train.py --synthetic
```

## Project Structure

```
finsocial-lite/
├── app.py                  # Streamlit frontend (3 tabs)
├── src/
│   ├── data.py             # Data loading (yfinance / synthetic / CSV)
│   ├── features.py         # Feature engineering (ported from FinSocial)
│   ├── train.py            # Trains XGBoost, saves model + metrics
│   ├── evaluate.py         # Metrics and confusion matrix helpers
│   └── predict.py          # Inference helper
├── models/                 # Saved model + metrics (created by train.py)
├── data/                   # Sample CSV (optional)
├── requirements.txt        # Pinned dependencies
└── README.md               # This file
```

## Model Details & Improvements

- **Algorithm**: XGBoost (`multi:softprob`, 3 classes) with early stopping
- **Early Stopping**: Halts tree growth when validation loss stops improving (`early_stopping_rounds=30`), avoiding the severe overfitting of training hundreds of trees on financial noise.
- **Hyperparameters**: `max_depth=4`, `learning_rate=0.03`, `subsample=0.8`,
  `colsample_bytree=0.8`, `min_child_weight=12`, `reg_alpha=3.0`, `reg_lambda=5.0`
- **Class Balancing**: Uses balanced square-root frequency sample weighting:
  `weight_c = (n / (3 × count_c)) ** 0.35`. This prevents high false positive rates for minority classes while preserving high recall.
- **Train/test split**: Chronological (time-ordered, 80/20) — no shuffling,
  no future data leaks into training.
- **Random seed**: 42 (fixed for reproducibility)

## Performance Benchmark (Held-out Test Set)

| Metric | Original Baseline | Optimized Model | Gain |
|--------|-------------------|-----------------|------|
| **Accuracy** | 36.93% | **49.71%** | **+12.78%** |
| **Weighted Precision** | 37.07% | **45.25%** | **+8.18%** |
| **Weighted F1** | 36.94% | **45.18%** | **+8.24%** |
| **ROC-AUC (macro)** | 0.5397 | **0.5775** | **+0.0378** |
| **Optimal Trees** | 500 (overfit) | 144 (early stopped) | -356 trees |

## Dashboard Tabs

1. **🔮 Predict**: Enter a ticker → see Buy/Sell/Hold verdict with probability bars,
   top contributing features, and a SHAP waterfall explanation.
2. **📊 Model Performance**: Accuracy, precision/recall/F1 per class, confusion matrix
   heatmap, class distribution, and imbalance warnings.
3. **📉 Visualisations**: Price chart with signal overlays, feature importance,
   label distribution, SHAP summary plot, and probability distributions.

## Deviations from Original FinSocial

| Aspect | Original | Lite |
|--------|----------|------|
| Data source | PostgreSQL (StockHistory table) | yfinance (free, no API key) |
| Inference API | Flask REST API | Streamlit frontend |
| Sentiment features | FinBERT NLP model | Removed (not used by XGBoost) |
| Portfolio optimizer | PyPortfolioOpt max-Sharpe | Removed |
| BUY bias | 4% bias for platform UX | Removed (pure model output) |
| Social features | User auth, feeds, tribes | Removed |
| Docker/cloud | Docker Compose + Render | None (runs locally) |
| Feature engineering | Identical | ✅ Identical |
| XGBoost hyperparams | Identical | ✅ Identical |
| Label logic | Identical | ✅ Identical |
| Class balancing | Identical | ✅ Identical |

## Limitations

- **Educational demo only** — this model is not suitable for real trading decisions.
- Trained on historical data; past performance does not predict future returns.
- Technical indicators alone cannot capture fundamental value, news, or sentiment.
- The 5-day horizon and ±1.5% thresholds are arbitrary and may not suit all markets.
- HOLD class typically dominates (~40-60%), which inflates raw accuracy.
- No transaction costs, slippage, or market impact are modeled.

---

> ⚠️ **DISCLAIMER**: This project is for **educational and research purposes only**.
> It does not constitute financial advice, and no investment decisions should be made
> based on its outputs. The authors are not responsible for any financial losses.
# ml-project
