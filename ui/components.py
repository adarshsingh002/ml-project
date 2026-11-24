"""
ui/components.py — Polished reusable UI components for FinSocial-Lite.

Clean presentation helpers: cards, badges, probability bars, and layout wrappers.
All HTML outputs avoid leading indentation to prevent CommonMark code block conversion.
"""
from __future__ import annotations

import streamlit as st
from ui.styles import THEME, SIGNAL_COLORS


def render_header():
    """Render top brand bar with app title, badge, and educational disclaimer."""
    header_html = (
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem; flex-wrap: wrap; gap: 0.75rem;">'
        f'<div>'
        f'<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">'
        f'<span style="font-size: 1.5rem;">📈</span>'
        f'<h2 style="margin: 0; font-size: 1.5rem; font-weight: 800; color: {THEME["text_primary"]}; letter-spacing: -0.02em;">'
        f'FinSocial<span style="color: {THEME["buy"]}; font-weight: 400;">Lite</span>'
        f'</h2>'
        f'<span class="header-badge">XGBoost 3-Class ML</span>'
        f'</div>'
        f'<div style="color: {THEME["text_secondary"]}; font-size: 0.85rem;">'
        f'Scale-free technical indicator forecasting · 5-day Buy/Hold/Sell classifier'
        f'</div>'
        f'</div>'
        f'<div style="display: flex; align-items: center; gap: 8px;">'
        f'<span class="disclaimer-pill">⚠️ Educational Demo · Not Financial Advice</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, subtext: str = "", delta: str | None = None) -> str:
    """Render a single modern metric card with tabular numerals.
    
    Returns clean single-line HTML without indentation to ensure
    Streamlit/CommonMark never interprets lines as indented code blocks.
    """
    delta_html = f'<span style="color: {THEME["buy"]}; font-size: 0.75rem; font-weight: 600;">{delta}</span>' if delta else ""
    return (
        f'<div class="kpi-container">'
        f'<div class="kpi-label"><span>{label}</span>{delta_html}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-subtext">{subtext}</div>'
        f'</div>'
    )


def render_prediction_hero(
    verdict: str,
    confidence: int,
    probs: dict[str, float],
    ticker: str = "TICKER",
    latest_close: float | None = None,
):
    """Render the hero prediction badge with probability breakdown."""
    color = SIGNAL_COLORS.get(verdict, THEME["hold"])
    badge_class = f"badge-{verdict.lower()}"
    
    # Safe currency formatting
    if latest_close is not None:
        sym = "₹" if (".NS" in ticker or ".BO" in ticker) else "$"
        price_str = f" · {sym}{latest_close:,.2f}"
    else:
        price_str = ""

    # Generate probability bars
    prob_bars_html = ""
    for cls in ["BUY", "HOLD", "SELL"]:
        p_val = probs.get(cls, 0.0)
        p_pct = f"{p_val * 100:.1f}%"
        bar_color = SIGNAL_COLORS[cls]
        prob_bars_html += (
            f'<div class="prob-bar-container">'
            f'<span class="prob-label">{cls}</span>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width: {p_val * 100}%; background-color: {bar_color};"></div>'
            f'</div>'
            f'<span class="prob-val" style="color: {bar_color};">{p_pct}</span>'
            f'</div>'
        )

    hero_html = (
        f'<div style="background: linear-gradient(145deg, {THEME["bg_surface"]} 0%, {THEME["bg_surface_alt"]} 100%); '
        f'border: 1px solid {THEME["border_subtle"]}; border-left: 4px solid {color}; border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.25rem;">'
        f'<div>'
        f'<div style="color: {THEME["text_muted"]}; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">'
        f'Forecast Horizon (5-Day Return)'
        f'</div>'
        f'<div style="color: {THEME["text_primary"]}; font-size: 1.2rem; font-weight: 700;">'
        f'{ticker}<span style="color: {THEME["text_secondary"]}; font-weight: 400;">{price_str}</span>'
        f'</div>'
        f'</div>'
        f'<div style="text-align: right;">'
        f'<div style="color: {THEME["text_muted"]}; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 2px;">'
        f'Confidence'
        f'</div>'
        f'<div style="color: {color}; font-size: 1.35rem; font-weight: 800; font-variant-numeric: tabular-nums;">'
        f'{confidence}%'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div style="display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap;">'
        f'<div><span class="{badge_class}">{verdict}</span></div>'
        f'<div style="flex: 1; min-width: 200px;">{prob_bars_html}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)


def render_technicals_chips(ctx: dict):
    """Render current technical indicators as sleek micro-cards."""
    if not ctx:
        return

    items = []
    labels = {
        "close": "Close",
        "rsi": "RSI (14)",
        "macd": "MACD %",
        "macd_signal": "Signal %",
        "adx": "ADX",
        "sma_20": "SMA 20",
        "sma_50": "SMA 50",
    }
    for k, v in ctx.items():
        if v is None:
            continue
        name = labels.get(k, k.upper())
        formatted_val = f"{v:,.2f}" if k in ["close", "sma_20", "sma_50"] else f"{v:.2f}"
        items.append(
            f'<div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 6px 12px; text-align: center;">'
            f'<div style="color: {THEME["text_muted"]}; font-size: 0.7rem; font-weight: 500; text-transform: uppercase;">{name}</div>'
            f'<div style="color: {THEME["text_primary"]}; font-size: 0.95rem; font-weight: 600; font-variant-numeric: tabular-nums;">{formatted_val}</div>'
            f'</div>'
        )

    grid_html = "".join(items)
    st.markdown(
        f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(90px, 1fr)); gap: 8px; margin-top: 0.75rem;">{grid_html}</div>',
        unsafe_allow_html=True,
    )


def render_feature_impacts(feat_vals: dict, importances: dict, model_feats: list[str], top_n: int = 6):
    """Render top contributing features with mini visual progress meters."""
    contrib = {
        f: abs(float(importances.get(f, 0.0) or 0.0)) * abs(float(feat_vals.get(f, 0.0) or 0.0))
        for f in model_feats
    }
    sorted_feats = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:top_n]
    max_c = max([c for _, c in sorted_feats]) if sorted_feats else 1.0

    chips_html = ""
    for feat_name, c_val in sorted_feats:
        val = feat_vals.get(feat_name, 0.0)
        val = 0.0 if val is None else float(val)
        imp = importances.get(feat_name, 0.0)
        imp = 0.0 if imp is None else float(imp)
        pct = min(100, int((c_val / (max_c or 1.0)) * 100))
        chips_html += (
            f'<div class="feat-chip">'
            f'<div style="display: flex; flex-direction: column; width: 45%;">'
            f'<span class="feat-name">{feat_name}</span>'
            f'<span style="color: {THEME["text_muted"]}; font-size: 0.72rem;">Importance {imp:.3f}</span>'
            f'</div>'
            f'<div style="flex: 1; height: 5px; background: rgba(255,255,255,0.06); border-radius: 99px; margin: 0 10px; overflow: hidden;">'
            f'<div style="height: 100%; width: {pct}%; background: {THEME["accent"]}; border-radius: 99px;"></div>'
            f'</div>'
            f'<div class="feat-val" style="width: 25%; text-align: right; font-weight: 600;">{val:.4f}</div>'
            f'</div>'
        )

    card_html = (
        f'<div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">'
        f'<span style="font-weight: 700; color: {THEME["text_primary"]}; font-size: 0.95rem;">Key Feature Attributions</span>'
        f'<span style="font-size: 0.72rem; color: {THEME["text_muted"]};">Top Impact Factors</span>'
        f'</div>'
        f'{chips_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_empty_state():
    """Render a clean welcoming placeholder with quick-action ticker presets."""
    empty_html = (
        f'<div class="empty-state-box">'
        f'<div style="font-size: 2.2rem; margin-bottom: 0.5rem;">⚡</div>'
        f'<h4 style="color: {THEME["text_primary"]}; margin-bottom: 0.35rem; font-weight: 700;">Ready for Forecast</h4>'
        f'<p style="color: {THEME["text_secondary"]}; font-size: 0.88rem; max-width: 480px; margin: 0 auto 1.25rem auto;">'
        f'Select a ticker and lookback period above, then click <strong>Forecast Signal</strong> to generate real-time technical indicators and SHAP attributions.'
        f'</p>'
        f'<div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">'
        f'<span style="color: {THEME["text_muted"]}; font-size: 0.8rem; align-self: center;">Popular presets:</span>'
        f'<code style="background: rgba(255,255,255,0.06); padding: 3px 8px; border-radius: 6px; color: {THEME["text_primary"]};">AAPL</code>'
        f'<code style="background: rgba(255,255,255,0.06); padding: 3px 8px; border-radius: 6px; color: {THEME["text_primary"]};">MSFT</code>'
        f'<code style="background: rgba(255,255,255,0.06); padding: 3px 8px; border-radius: 6px; color: {THEME["text_primary"]};">TSLA</code>'
        f'<code style="background: rgba(255,255,255,0.06); padding: 3px 8px; border-radius: 6px; color: {THEME["text_primary"]};">NVDA</code>'
        f'<code style="background: rgba(255,255,255,0.06); padding: 3px 8px; border-radius: 6px; color: {THEME["text_primary"]};">RELIANCE.NS</code>'
        f'</div>'
        f'</div>'
    )
    st.markdown(empty_html, unsafe_allow_html=True)
