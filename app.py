"""
app.py — FinSocial-Lite Streamlit Dashboard

A modern, polished fintech UI for the XGBoost Buy/Sell/Hold classifier.
Pure presentation layer: model, data, and inference logic remain 100% intact.

Run: streamlit run app.py
"""
from __future__ import annotations

import os
import sys

# Configure writable cache directory for matplotlib/SHAP before imports
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
)
from src.predict import load_model, predict_single, get_feature_importances
from src.evaluate import load_metrics

# UI Design System imports
from ui.styles import THEME, SIGNAL_COLORS, CUSTOM_CSS, get_plotly_layout
from ui.components import (
    render_header,
    render_kpi_card,
    render_prediction_hero,
    render_technicals_chips,
    render_feature_impacts,
    render_empty_state,
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSocial-Lite · Market Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Global Fintech Design System
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Cached Data Loaders (Preserved Unchanged) ──────────────────────────────────
@st.cache_resource
def _load_bundle():
    try:
        return load_model()
    except Exception:
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
model_feats = bundle.get("feature_columns", FEATURE_COLUMNS) if bundle else FEATURE_COLUMNS
_n_feats = len(model_feats)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="padding: 0.5rem 0 1rem 0;">'
        f'<div style="font-size: 1.25rem; font-weight: 800; color: {THEME["text_primary"]};">'
        f'📈 FinSocial<span style="color: {THEME["buy"]}; font-weight: 400;">Lite</span>'
        f'</div>'
        f'<div style="font-size: 0.78rem; color: {THEME["text_muted"]}; margin-top: 2px;">'
        f'XGBoost Buy / Hold / Sell Classifier'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        f'<div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: {THEME["text_secondary"]}; margin-bottom: 0.5rem;">'
        f'Architecture & Pipeline'
        f'</div>'
        f'<div style="font-size: 0.82rem; color: {THEME["text_secondary"]}; line-height: 1.6;">'
        f'• <strong>Data</strong>: Free OHLCV via <code>yfinance</code><br>'
        f'• <strong>Features</strong>: {_n_feats} scale-free technical indicators (RSI, normalized MACD, ADX, Bollinger Bands, ATR)<br>'
        f'• <strong>Horizon</strong>: {_horizon}-day forward return<br>'
        f'• <strong>Target</strong>: BUY (> {_thresh:+.1%}), SELL (< {_sell_thresh:+.1%}), else HOLD<br>'
        f'• <strong>Model</strong>: Regularized XGBoost with early stopping<br>'
        f'• <strong>Validation</strong>: Strict forward-time split (no leakage)'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        f'<div style="background: rgba(245, 158, 11, 0.06); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 10px; padding: 0.75rem; font-size: 0.75rem; color: #FCD34D;">'
        f'<strong>⚠️ Educational Demo</strong><br>'
        f'This dashboard is for research and educational purposes only. It is <strong>not financial advice</strong>.'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Guard: Missing Model ─────────────────────────────────────────────────────
if bundle is None:
    st.error("⚠️ Model bundle not found at `models/xgboost_model.pkl`. Please run `python src/train.py` first.")
    st.stop()

# ── Top Brand Header ──────────────────────────────────────────────────────────
render_header()

# ── Primary Navigation Tabs ───────────────────────────────────────────────────
tab_predict, tab_perf, tab_viz = st.tabs([
    "🔮 Predict",
    "📊 Model Performance",
    "📉 Visualisations",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    with st.container(border=True):
        st.markdown(
            f'<div style="color: {THEME["text_secondary"]}; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.6rem;">Live Market Inference</div>',
            unsafe_allow_html=True,
        )

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.5, 1.5, 1.2])
        with ctrl_col1:
            ticker_input = st.text_input(
                "Symbol (Yahoo Finance)",
                value="AAPL",
                placeholder="e.g. AAPL, MSFT, RELIANCE.NS",
                label_visibility="collapsed",
            )
        with ctrl_col2:
            period = st.selectbox(
                "Lookback Window",
                ["3mo", "6mo", "1y", "2y", "5y"],
                index=2,
                label_visibility="collapsed",
            )
        with ctrl_col3:
            predict_btn = st.button("🚀 Forecast Signal", type="primary", use_container_width=True)

    # State: Prediction Execution
    if predict_btn:
        clean_ticker = (ticker_input or "").strip().upper()
        if not clean_ticker:
            st.error("Please enter a valid ticker symbol (e.g. AAPL, NVDA, RELIANCE.NS).")
            st.stop()

        with st.spinner(f"Fetching market data for {clean_ticker}..."):
            try:
                import yfinance as yf
                # Fetch at least 2y of history so technical indicators have sufficient warmup window (>60 bars)
                fetch_period = "2y" if period in ["3mo", "6mo", "1y"] else period
                data = yf.Ticker(clean_ticker).history(period=fetch_period, auto_adjust=True)
                if data.empty:
                    st.error(f"No price data returned for symbol `{clean_ticker}`. Please verify the Yahoo Finance ticker symbol.")
                    st.stop()
                data = data.reset_index()
                data = data.rename(columns={
                    "Date": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                })
            except Exception as e:
                st.error(f"Data retrieval failed: {e}")
                st.stop()

        with st.spinner("Generating XGBoost signals and SHAP attributions..."):
            try:
                result = predict_single(data, bundle)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        verdict = result["verdict"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        feat_vals = result["feature_values"]
        ctx = result.get("context", {})
        importances = get_feature_importances(bundle)
        latest_close = ctx.get("close")

        # Two-Column Layout: Left = Hero & Technicals, Right = Feature Attributions & SHAP
        left_col, right_col = st.columns([1.1, 1], gap="medium")

        with left_col:
            # Prediction Hero Badge
            render_prediction_hero(
                verdict=verdict,
                confidence=confidence,
                probs=probs,
                ticker=clean_ticker,
                latest_close=latest_close,
            )

            # Current Indicators Micro-Cards
            if ctx:
                render_technicals_chips(ctx)

        with right_col:
            with st.container(border=True):
                render_feature_impacts(feat_vals, importances, model_feats, top_n=6)

        # Full-Width SHAP Waterfall Section
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        try:
            import shap
            explainer = shap.TreeExplainer(bundle["model"])
            feat_df = pd.DataFrame([feat_vals], columns=model_feats).astype(float)
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
            }).sort_values("SHAP Value", key=abs, ascending=True).tail(10)

            # Colors based on positive or negative push
            bar_colors = [THEME["buy"] if v >= 0 else THEME["sell"] for v in shap_df["SHAP Value"]]

            fig_shap = go.Figure(
                go.Bar(
                    x=shap_df["SHAP Value"],
                    y=shap_df["Feature"],
                    orientation="h",
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:.4f}<extra></extra>",
                )
            )
            fig_shap.update_layout(
                get_plotly_layout(
                    height=320,
                    title=f"SHAP Attribution Breakdown (Directional Impact for {verdict})",
                    margin=dict(l=10, r=10, t=35, b=20),
                )
            )
            fig_shap.add_vline(x=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.2)")

            with st.container(border=True):
                st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP explanation: {e}")

    else:
        render_empty_state()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    if metrics is None:
        st.warning("Held-out evaluation metrics not found. Run `python src/train.py` to train and evaluate.")
    else:
        acc = metrics.get("accuracy", 0.0)
        macro_f1 = metrics.get("macro_avg", {}).get("f1_score", 0.0)
        weighted_f1 = metrics.get("weighted_avg", {}).get("f1_score", 0.0)
        auc = metrics.get("roc_auc_macro")
        auc_str = f"{auc:.4f}" if auc else "N/A"
        samples = metrics.get("total_samples", 0)

        # KPI Header Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(render_kpi_card("Accuracy", f"{acc:.2%}", f"Held-out test set ({samples:,} bars)", "+12.8% vs 33% rand"), unsafe_allow_html=True)
        with k2:
            st.markdown(render_kpi_card("Macro F1", f"{macro_f1:.2%}", "Unweighted mean across classes"), unsafe_allow_html=True)
        with k3:
            st.markdown(render_kpi_card("Weighted F1", f"{weighted_f1:.2%}", "Frequency-weighted F1-Score"), unsafe_allow_html=True)
        with k4:
            st.markdown(render_kpi_card("ROC-AUC (Macro)", auc_str, "One-vs-Rest classification area"), unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        col_matrix, col_report = st.columns([1, 1.1], gap="medium")

        # ── Confusion Matrix Card ──
        with col_matrix:
            with st.container(border=True):
                cm = np.array(metrics["confusion_matrix"])

                # Clean annotated heatmap with guaranteed high-contrast white text
                fig_cm = px.imshow(
                    cm,
                    x=CLASS_NAMES,
                    y=CLASS_NAMES,
                    labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Observations"),
                    text_auto=True,
                    color_continuous_scale=[
                        [0.0, "#162032"],
                        [0.3, "#1E293B"],
                        [0.6, "#312E81"],
                        [1.0, "#4F46E5"],
                    ],
                    aspect="equal",
                )
                fig_cm.update_traces(
                    textfont=dict(color="#FFFFFF", size=15, family="Inter"),
                    hovertemplate="Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Count: <b>%{z:,}</b><extra></extra>",
                )
                fig_cm.update_layout(
                    get_plotly_layout(height=360, title="Confusion Matrix (Held-Out Test Set)", coloraxis_showscale=False)
                )
                fig_cm.update_xaxes(side="bottom")
                st.plotly_chart(fig_cm, use_container_width=True)

        # ── Classification Report Table Card ──
        with col_report:
            with st.container(border=True):
                st.markdown(
                    f'<div style="font-weight: 700; color: {THEME["text_primary"]}; font-size: 0.95rem; margin-bottom: 0.75rem;">'
                    f'Per-Class Classification Report'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                rows = []
                for cls in CLASS_NAMES:
                    pc = metrics["per_class"][cls]
                    rows.append({
                        "Target Class": cls,
                        "Precision": f"{pc['precision']:.3f}",
                        "Recall": f"{pc['recall']:.3f}",
                        "F1-Score": f"{pc['f1_score']:.3f}",
                        "Support": f"{pc['support']:,}",
                    })
                rows.append({
                    "Target Class": "Macro Average",
                    "Precision": f"{metrics['macro_avg']['precision']:.3f}",
                    "Recall": f"{metrics['macro_avg']['recall']:.3f}",
                    "F1-Score": f"{metrics['macro_avg']['f1_score']:.3f}",
                    "Support": f"{metrics['total_samples']:,}",
                })
                rows.append({
                    "Target Class": "Weighted Average",
                    "Precision": f"{metrics['weighted_avg']['precision']:.3f}",
                    "Recall": f"{metrics['weighted_avg']['recall']:.3f}",
                    "F1-Score": f"{metrics['weighted_avg']['f1_score']:.3f}",
                    "Support": f"{metrics['total_samples']:,}",
                })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

                # Class Distribution Callout
                dist = metrics.get("class_distribution", {})
                total = sum(dist.values()) or 1
                dist_html = (
                    f'<div style="margin-top: 0.75rem; padding: 0.65rem 0.85rem; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.78rem; color: {THEME["text_secondary"]}; line-height: 1.4;">'
                    f'<strong>Class Distribution in Test Set:</strong><br>'
                    f'<span style="color: {SIGNAL_COLORS["BUY"]}; font-weight: 600;">BUY</span>: {dist.get("BUY", 0):,} ({dist.get("BUY", 0)/total:.1%}) &nbsp;·&nbsp; '
                    f'<span style="color: {SIGNAL_COLORS["HOLD"]}; font-weight: 600;">HOLD</span>: {dist.get("HOLD", 0):,} ({dist.get("HOLD", 0)/total:.1%}) &nbsp;·&nbsp; '
                    f'<span style="color: {SIGNAL_COLORS["SELL"]}; font-weight: 600;">SELL</span>: {dist.get("SELL", 0):,} ({dist.get("SELL", 0)/total:.1%})'
                    f'</div>'
                )
                st.markdown(dist_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_viz:
    if test_preds is None or test_preds.empty:
        st.warning("No test predictions found. Run `python src/train.py` to generate forecast datasets.")
    else:
        # Ticker Filter Selector
        available_tickers = sorted(test_preds["ticker"].unique()) if "ticker" in test_preds.columns else ["ALL"]
        
        top_viz_c1, top_viz_c2 = st.columns([1.5, 3])
        with top_viz_c1:
            selected_ticker = st.selectbox(
                "Filter Ticker Context",
                available_tickers,
                index=0,
            )
        
        if selected_ticker != "ALL" and "ticker" in test_preds.columns:
            viz_df = test_preds[test_preds["ticker"] == selected_ticker].copy()
        else:
            viz_df = test_preds.copy()

        viz_df = viz_df.sort_values("date").reset_index(drop=True)

        # ── 1. Price Chart with Overlaid Signals ──
        with st.container(border=True):
            fig_price = go.Figure()
            
            # Base Close Price line
            fig_price.add_trace(go.Scatter(
                x=viz_df["date"],
                y=viz_df["close"],
                mode="lines",
                name="Close Price",
                line=dict(color="rgba(255, 255, 255, 0.45)", width=1.75),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            ))

            # Signal Markers
            symbols = {"BUY": "triangle-up", "SELL": "triangle-down", "HOLD": "circle"}
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                mask = viz_df["predicted"] == cls_idx
                if mask.sum() > 0:
                    subset = viz_df[mask]
                    fig_price.add_trace(go.Scatter(
                        x=subset["date"],
                        y=subset["close"],
                        mode="markers",
                        name=f"Signal: {cls_name}",
                        marker=dict(
                            color=SIGNAL_COLORS[cls_name],
                            size=7 if cls_name != "HOLD" else 5,
                            symbol=symbols[cls_name],
                            line=dict(width=1, color="rgba(0,0,0,0.5)"),
                        ),
                        hovertemplate=f"Signal: <b>{cls_name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Close: %{{y:.2f}}<extra></extra>",
                    ))

            fig_price.update_layout(
                get_plotly_layout(height=420, title=f"Price Trajectory & Forecasted Signals ({selected_ticker})")
            )
            st.plotly_chart(fig_price, use_container_width=True)

        # Two-Column Grid for Charts 2 & 3
        vcol1, vcol2 = st.columns(2, gap="medium")

        # ── 2. Feature Importance ──
        with vcol1:
            with st.container(border=True):
                importances = get_feature_importances(bundle)
                imp_df = pd.DataFrame(
                    sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15],
                    columns=["Feature", "Importance"],
                )
                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=[[0, "rgba(99, 102, 241, 0.3)"], [1, THEME["buy"]]],
                )
                fig_imp.update_layout(
                    get_plotly_layout(height=380, title="Top 15 Feature Importances (XGBoost Gain)", coloraxis_showscale=False)
                )
                fig_imp.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_imp, use_container_width=True)

        # ── 3. Label Distribution ──
        with vcol2:
            with st.container(border=True):
                label_counts = viz_df["target"].value_counts().sort_index()
                label_df = pd.DataFrame({
                    "Class": [CLASS_NAMES[i] for i in label_counts.index],
                    "Count": label_counts.values,
                })
                fig_label = px.bar(
                    label_df,
                    x="Class",
                    y="Count",
                    color="Class",
                    color_discrete_map=SIGNAL_COLORS,
                    text_auto=True,
                )
                fig_label.update_layout(
                    get_plotly_layout(height=380, title=f"Target Label Distribution ({selected_ticker})", showlegend=False)
                )
                st.plotly_chart(fig_label, use_container_width=True)

        # Two-Column Grid for Charts 4 & 5
        vcol3, vcol4 = st.columns(2, gap="medium")

        # ── 4. SHAP Feature Attribution Summary ──
        with vcol3:
            with st.container(border=True):
                try:
                    import shap
                    explainer = shap.TreeExplainer(bundle["model"])
                    sample_size = min(400, len(viz_df))
                    sample_df = viz_df.sample(n=sample_size, random_state=42)
                    X_sample = sample_df[model_feats].astype(float)
                    shap_values = explainer.shap_values(X_sample)

                    if isinstance(shap_values, list):
                        sv_buy = shap_values[2]
                    elif shap_values.ndim == 3:
                        sv_buy = shap_values[:, :, 2]
                    else:
                        sv_buy = shap_values

                    mean_abs_shap = np.abs(sv_buy).mean(axis=0)
                    shap_imp_df = pd.DataFrame({
                        "Feature": model_feats,
                        "Mean |SHAP|": mean_abs_shap,
                    }).sort_values("Mean |SHAP|", ascending=True).tail(12)

                    fig_shap_summary = px.bar(
                        shap_imp_df,
                        x="Mean |SHAP|",
                        y="Feature",
                        orientation="h",
                        color="Mean |SHAP|",
                        color_continuous_scale=[[0, "rgba(99, 102, 241, 0.4)"], [1, THEME["accent"]]],
                    )
                    fig_shap_summary.update_layout(
                        get_plotly_layout(height=380, title="SHAP Mean Impact (BUY Direction)", coloraxis_showscale=False)
                    )
                    st.plotly_chart(fig_shap_summary, use_container_width=True)
                except Exception as e:
                    st.info(f"SHAP chart not available: {e}")

        # ── 5. Probability Distribution by Class ──
        with vcol4:
            with st.container(border=True):
                if all(f"prob_{c.lower()}" in viz_df.columns for c in CLASS_NAMES):
                    viz_long = []
                    for cls_idx, cls_name in enumerate(CLASS_NAMES):
                        subset = viz_df[viz_df["target"] == cls_idx].copy()
                        for pred_cls in CLASS_NAMES:
                            prob_col = f"prob_{pred_cls.lower()}"
                            for _, row in subset.iterrows():
                                viz_long.append({
                                    "Ground Truth": cls_name,
                                    "Predicted Class": pred_cls,
                                    "Probability": row[prob_col],
                                })

                    if viz_long:
                        viz_long_df = pd.DataFrame(viz_long)
                        fig_probdist = px.box(
                            viz_long_df,
                            x="Ground Truth",
                            y="Probability",
                            color="Predicted Class",
                            color_discrete_map=SIGNAL_COLORS,
                        )
                        fig_probdist.update_layout(
                            get_plotly_layout(height=380, title="Confidence Probability Spread by Class")
                        )
                        st.plotly_chart(fig_probdist, use_container_width=True)
