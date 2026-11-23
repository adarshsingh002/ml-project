"""
app.py — FinSocial-Lite Streamlit Dashboard

A clean, educational demo of a Buy/Sell/Hold stock classifier
using XGBoost on 20 technical indicators derived from OHLCV data.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import (
    FEATURE_COLUMNS,
    CLASS_NAMES,
    LABEL_HORIZON_DAYS,
    LABEL_THRESHOLD,
    LABEL_SELL_THRESHOLD,
    add_ml_features,
    normalize_ohlcv,
    latest_feature_row,
)
from src.predict import load_model, predict_single, get_feature_importances
from src.evaluate import load_metrics

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSocial-Lite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {"BUY": "#00C853", "HOLD": "#FFA000", "SELL": "#FF1744"}

# ── Load model & metrics (cached) ────────────────────────────────────────────
@st.cache_resource
def _load_bundle():
    try:
        return load_model()
    except Exception as e:
        return None

@st.cache_data
def _load_metrics():
    try:
        return load_metrics()
    except Exception:
        return None

@st.cache_data
def _load_test_predictions():
    try:
        df = pd.read_csv("models/test_predictions.csv", parse_dates=["date"])
        return df
    except Exception:
        return None

bundle = _load_bundle()
metrics = _load_metrics()
test_preds = _load_test_predictions()

_horizon = bundle.get("label_horizon_days", LABEL_HORIZON_DAYS) if bundle else LABEL_HORIZON_DAYS
_thresh = bundle.get("label_threshold", LABEL_THRESHOLD) if bundle else LABEL_THRESHOLD
_sell_thresh = bundle.get("label_sell_threshold", LABEL_SELL_THRESHOLD) if bundle else LABEL_SELL_THRESHOLD
_n_feats = len(bundle.get("feature_columns", FEATURE_COLUMNS)) if bundle else len(FEATURE_COLUMNS)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 FinSocial-Lite")
    st.caption("XGBoost Buy / Sell / Hold Classifier")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        f"""
        1. **Data**: OHLCV prices fetched via *yfinance*
        2. **Features**: {_n_feats} scale-free technical indicators
           (returns, RSI, normalized MACD, ADX, Bollinger Bands, ATR, Donchian, volatility…)
        3. **Label**: {_horizon}-day forward return
           - **BUY**: > {_thresh:+.1%}
           - **SELL**: < {_sell_thresh:+.1%}
           - **HOLD**: in between
        4. **Model**: Regularized XGBoost (`multi:softprob`)
           with early stopping and class-weight balancing
        5. **Split**: Chronological 80/20 — no future leakage
        """
    )

    st.markdown("---")
    st.markdown(
        "⚠️ **Disclaimer**: This is an educational demo. "
        "It is **not financial advice**. Past performance does "
        "not guarantee future results."
    )
metrics = _load_metrics()
test_preds = _load_test_predictions()

if bundle is None:
    st.error(
        "⚠️ Model not found. Run `python src/train.py` first to train the model."
    )
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_perf, tab_viz = st.tabs([
    "🔮 Predict",
    "📊 Model Performance",
    "📉 Visualisations",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Live Prediction")
    st.markdown(
        "Pick a ticker and date range. The model predicts whether the stock is a "
        f"**Buy**, **Sell**, or **Hold** based on the last {_horizon}-day "
        "outlook."
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        ticker = st.text_input("Ticker (Yahoo Finance)", value="RELIANCE.NS")
    with col2:
        period = st.selectbox("History period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    with col3:
        st.write("")
        st.write("")
        predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                import yfinance as yf
                data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if data.empty:
                    st.error(f"No data returned for `{ticker}`. Check the ticker symbol.")
                    st.stop()
                data = data.reset_index()
                data = data.rename(columns={
                    "Date": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                })
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
                st.stop()

        with st.spinner("Running prediction..."):
            result = predict_single(data, bundle)

        # ── Verdict card ──
        verdict = result["verdict"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        ctx = result["context"]

        v_col, p_col = st.columns([1, 2])
        with v_col:
            st.markdown(
                f"<div style='text-align:center; padding:1.5rem; border-radius:12px; "
                f"background:{COLORS[verdict]}22; border:2px solid {COLORS[verdict]}'>"
                f"<h1 style='color:{COLORS[verdict]}; margin:0'>{verdict}</h1>"
                f"<p style='font-size:1.2rem; margin:0'>Confidence: {confidence}%</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if ctx:
                st.markdown("##### Current Indicators")
                ind_cols = st.columns(3)
                for i, (k, v) in enumerate(ctx.items()):
                    with ind_cols[i % 3]:
                        st.metric(k.upper(), f"{v:.2f}")

        with p_col:
            st.markdown("##### Class Probabilities")
            prob_df = pd.DataFrame({
                "Class": list(probs.keys()),
                "Probability": list(probs.values()),
            })
            fig_prob = px.bar(
                prob_df, x="Class", y="Probability",
                color="Class",
                color_discrete_map=COLORS,
                text_auto=".1%",
            )
            fig_prob.update_layout(
                yaxis_range=[0, 1],
                showlegend=False,
                height=300,
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # ── Top contributing features ──
        st.markdown("##### Top Contributing Features (by importance × value)")
        importances = get_feature_importances(bundle)
        feat_vals = result["feature_values"]
        model_feats = bundle.get("feature_columns", FEATURE_COLUMNS)
        contrib = {
            f: abs(importances.get(f, 0)) * abs(feat_vals.get(f, 0))
            for f in model_feats
        }
        top_feats = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:8]
        explanation_parts = []
        for feat_name, _ in top_feats:
            val = feat_vals.get(feat_name, 0)
            imp = importances.get(feat_name, 0)
            explanation_parts.append(f"**{feat_name}** = {val:.4f} (importance: {imp:.3f})")
        st.markdown(" · ".join(explanation_parts))

        # ── SHAP explanation (per-prediction) ──
        st.markdown("##### SHAP Waterfall (this prediction)")
        try:
            import shap
            explainer = shap.TreeExplainer(bundle["model"])
            feat_df = pd.DataFrame(
                [feat_vals], columns=model_feats,
            ).astype(float)
            shap_values = explainer.shap_values(feat_df)

            pred_class = CLASS_NAMES.index(verdict)
            if isinstance(shap_values, list):
                sv = shap_values[pred_class][0]
            elif shap_values.ndim == 3:
                sv = shap_values[0, :, pred_class]
            else:
                sv = shap_values[0]

            shap_df = pd.DataFrame({
                "Feature": model_feats,
                "SHAP Value": sv,
            }).sort_values("SHAP Value", key=abs, ascending=True).tail(12)

            fig_shap = px.bar(
                shap_df, x="SHAP Value", y="Feature", orientation="h",
                color="SHAP Value",
                color_continuous_scale=["#FF1744", "#BDBDBD", "#00C853"],
                color_continuous_midpoint=0,
            )
            fig_shap.update_layout(height=400, margin=dict(l=10, t=10))
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP not available: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.header("Model Performance (Held-Out Test Set)")

    if metrics is None:
        st.warning("Metrics not found. Run `python src/train.py` first.")
    else:
        # ── Metric cards ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        m2.metric("Macro F1", f"{metrics['macro_avg']['f1_score']:.2%}")
        m3.metric("Weighted F1", f"{metrics['weighted_avg']['f1_score']:.2%}")
        m4.metric(
            "ROC-AUC (macro)",
            f"{metrics['roc_auc_macro']:.4f}" if metrics.get("roc_auc_macro") else "N/A",
        )

        st.markdown("---")

        col_cm, col_report = st.columns(2)

        # ── Confusion matrix heatmap ──
        with col_cm:
            st.subheader("Confusion Matrix")
            cm = np.array(metrics["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                x=CLASS_NAMES, y=CLASS_NAMES,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                text_auto=True,
                color_continuous_scale="Blues",
                aspect="equal",
            )
            fig_cm.update_layout(height=400, margin=dict(t=30))
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── Classification report table ──
        with col_report:
            st.subheader("Classification Report")
            rows = []
            for cls in CLASS_NAMES:
                pc = metrics["per_class"][cls]
                rows.append({
                    "Class":     cls,
                    "Precision": f"{pc['precision']:.3f}",
                    "Recall":    f"{pc['recall']:.3f}",
                    "F1-Score":  f"{pc['f1_score']:.3f}",
                    "Support":   pc["support"],
                })
            rows.append({
                "Class": "**macro avg**",
                "Precision": f"{metrics['macro_avg']['precision']:.3f}",
                "Recall":    f"{metrics['macro_avg']['recall']:.3f}",
                "F1-Score":  f"{metrics['macro_avg']['f1_score']:.3f}",
                "Support":   metrics["total_samples"],
            })
            rows.append({
                "Class": "**weighted avg**",
                "Precision": f"{metrics['weighted_avg']['precision']:.3f}",
                "Recall":    f"{metrics['weighted_avg']['recall']:.3f}",
                "F1-Score":  f"{metrics['weighted_avg']['f1_score']:.3f}",
                "Support":   metrics["total_samples"],
            })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Class distribution ──
        st.subheader("Class Distribution (Test Set)")
        dist = metrics.get("class_distribution", {})
        total = sum(dist.values())
        dist_df = pd.DataFrame([
            {"Class": k, "Count": v, "Percentage": f"{v/total:.1%}"}
            for k, v in dist.items()
        ])
        dc1, dc2 = st.columns([1, 2])
        with dc1:
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        with dc2:
            fig_dist = px.pie(
                dist_df, names="Class", values="Count",
                color="Class",
                color_discrete_map=COLORS,
                hole=0.4,
            )
            fig_dist.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig_dist, use_container_width=True)

        if total > 0 and dist:
            dominant = max(dist, key=dist.get)
            dom_pct = dist[dominant] / total
            if dom_pct > 0.5:
                st.info(
                    f"⚠️ **Class imbalance detected**: {dominant} accounts for "
                    f"{dom_pct:.0%} of test samples. Accuracy alone can be misleading — "
                    f"check per-class F1 scores above. The model uses sample-weight "
                    f"balancing during training to mitigate this."
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_viz:
    st.header("Visualisations")

    if test_preds is None or test_preds.empty:
        st.warning("No test predictions found. Run `python src/train.py` first.")
    else:
        # Let user pick a ticker from the test data
        available_tickers = sorted(test_preds["ticker"].unique()) if "ticker" in test_preds.columns else ["ALL"]
        selected_ticker = st.selectbox(
            "Select ticker for visualisations",
            available_tickers,
            index=0,
        )
        if selected_ticker != "ALL" and "ticker" in test_preds.columns:
            viz_df = test_preds[test_preds["ticker"] == selected_ticker].copy()
        else:
            viz_df = test_preds.copy()

        viz_df = viz_df.sort_values("date").reset_index(drop=True)

        # ── 1. Price chart with Buy/Sell/Hold signals ──
        st.subheader("1. Price Chart with Signals")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=viz_df["date"], y=viz_df["close"],
            mode="lines", name="Close",
            line=dict(color="#616161", width=1.5),
        ))

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            mask = viz_df["predicted"] == cls_idx
            if mask.sum() > 0:
                subset = viz_df[mask]
                fig_price.add_trace(go.Scatter(
                    x=subset["date"], y=subset["close"],
                    mode="markers", name=cls_name,
                    marker=dict(
                        color=COLORS[cls_name],
                        size=6,
                        symbol="triangle-up" if cls_name == "BUY" else (
                            "triangle-down" if cls_name == "SELL" else "circle"
                        ),
                    ),
                ))

        fig_price.update_layout(
            height=450,
            xaxis_title="Date", yaxis_title="Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # ── 2. Feature Importance ──
        st.subheader("2. Feature Importance")
        importances = get_feature_importances(bundle)
        imp_df = pd.DataFrame(
            sorted(importances.items(), key=lambda x: x[1], reverse=True),
            columns=["Feature", "Importance"],
        )
        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig_imp.update_layout(
            height=500, yaxis=dict(autorange="reversed"),
            margin=dict(l=10, t=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # ── 3. Class Distribution (full dataset labels) ──
        st.subheader("3. Label Distribution (Test Set)")
        label_counts = viz_df["target"].value_counts().sort_index()
        label_df = pd.DataFrame({
            "Class": [CLASS_NAMES[i] for i in label_counts.index],
            "Count": label_counts.values,
        })
        fig_label = px.bar(
            label_df, x="Class", y="Count",
            color="Class", color_discrete_map=COLORS,
            text_auto=True,
        )
        fig_label.update_layout(
            height=350, showlegend=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_label, use_container_width=True)

        # ── 4. SHAP Summary Plot ──
        st.subheader("4. SHAP Feature Summary")
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            model_feats = bundle.get("feature_columns", FEATURE_COLUMNS)
            explainer = shap.TreeExplainer(bundle["model"])
            # Use a sample for speed
            sample_size = min(500, len(viz_df))
            sample_df = viz_df.sample(n=sample_size, random_state=42)
            X_sample = sample_df[model_feats].astype(float)
            shap_values = explainer.shap_values(X_sample)

            # For 3-class, shap_values can be list[3] or ndarray(n, feat, 3)
            if isinstance(shap_values, list):
                sv_buy = shap_values[2]
            elif shap_values.ndim == 3:
                sv_buy = shap_values[:, :, 2]
            else:
                sv_buy = shap_values

            st.caption("SHAP values shown for the **BUY** class (class 2)")
            fig_shap_summary, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                sv_buy, X_sample,
                feature_names=model_feats,
                show=False, max_display=15,
            )
            st.pyplot(fig_shap_summary)
            plt.close(fig_shap_summary)
        except Exception as e:
            st.info(f"SHAP summary not available: {e}")

        # ── 5. (Bonus) Probability Distribution by Class ──
        st.subheader("5. Predicted Probability Distribution by True Class")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            prob_col = f"prob_{cls_name.lower()}"
            if prob_col in viz_df.columns:
                pass  # column exists
            else:
                continue

        if all(f"prob_{c.lower()}" in viz_df.columns for c in CLASS_NAMES):
            viz_long = []
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                subset = viz_df[viz_df["target"] == cls_idx].copy()
                for pred_cls in CLASS_NAMES:
                    prob_col = f"prob_{pred_cls.lower()}"
                    for _, row in subset.iterrows():
                        viz_long.append({
                            "True Class": cls_name,
                            "Predicted Class Prob": pred_cls,
                            "Probability": row[prob_col],
                        })

            if viz_long:
                viz_long_df = pd.DataFrame(viz_long)
                fig_probdist = px.box(
                    viz_long_df,
                    x="True Class", y="Probability", color="Predicted Class Prob",
                    color_discrete_map=COLORS,
                )
                fig_probdist.update_layout(
                    height=400,
                    margin=dict(t=10, b=10),
                )
                st.plotly_chart(fig_probdist, use_container_width=True)
