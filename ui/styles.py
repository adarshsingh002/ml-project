"""
ui/styles.py — Global design tokens, CSS styling, and Plotly theme.

Implements a modern, calm, data-first fintech aesthetic (Linear / Stripe / Robinhood).
"""
from __future__ import annotations

# ── Semantic Design Tokens ────────────────────────────────────────────────────
THEME = {
    # Core Surfaces
    "bg_primary": "#0B0F19",       # Deep canvas
    "bg_surface": "#111827",       # Elevated card background
    "bg_surface_alt": "#162032",   # Hover / contrast card
    "bg_glass": "rgba(17, 24, 39, 0.75)",

    # Borders & Dividers
    "border_subtle": "rgba(255, 255, 255, 0.08)",
    "border_card": "rgba(255, 255, 255, 0.14)",
    "border_focus": "rgba(99, 102, 241, 0.5)",

    # Semantic Signals (WCAG AA Compliant)
    "buy": "#10B981",              # Emerald 500
    "buy_glow": "rgba(16, 185, 129, 0.16)",
    "buy_border": "rgba(16, 185, 129, 0.35)",

    "sell": "#F43F5E",             # Rose 500
    "sell_glow": "rgba(244, 63, 94, 0.16)",
    "sell_border": "rgba(244, 63, 94, 0.35)",

    "hold": "#F59E0B",             # Amber 500
    "hold_glow": "rgba(245, 158, 11, 0.16)",
    "hold_border": "rgba(245, 158, 11, 0.35)",

    # Accents & Neutrals
    "accent": "#6366F1",           # Indigo
    "text_primary": "#F9FAFB",     # Gray 50
    "text_secondary": "#9CA3AF",   # Gray 400
    "text_muted": "#6B7280",       # Gray 500
}

SIGNAL_COLORS = {
    "BUY": THEME["buy"],
    "HOLD": THEME["hold"],
    "SELL": THEME["sell"],
}

# ── Global CSS Stylesheet ────────────────────────────────────────────────────
CUSTOM_CSS = f"""
<style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    letter-spacing: -0.011em;
}}

/* Tabular numerals for numbers and financial metrics */
.tabular-nums, [data-testid="stMetricValue"], .metric-val {{
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}}

/* Clean up Streamlit header clutter */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{background: transparent !important;}}

/* Smooth scrollbar */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: {THEME["bg_primary"]};
}}
::-webkit-scrollbar-thumb {{
    background: rgba(255, 255, 255, 0.15);
    border-radius: 999px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: rgba(255, 255, 255, 0.25);
}}

/* Style Streamlit's native container(border=True) as fintech cards */
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background-color: {THEME["bg_surface"]} !important;
    border-color: {THEME["border_subtle"]} !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    transition: border-color 0.2s ease;
}}
[data-testid="stVerticalBlockBorderWrapper"] > div:hover {{
    border-color: {THEME["border_card"]} !important;
}}

/* Top Hero Header Badge */
.header-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 9999px;
    color: #A5B4FC;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* Disclaimer pill */
.disclaimer-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 8px;
    color: #FCD34D;
    font-size: 0.72rem;
    font-weight: 500;
}}

/* KPI Metric Cards */
.kpi-container {{
    background: {THEME["bg_surface"]};
    border: 1px solid {THEME["border_subtle"]};
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 105px;
    box-shadow: 0 4px 15px -2px rgba(0, 0, 0, 0.2);
    transition: transform 0.15s ease, border-color 0.15s ease;
}}
.kpi-container:hover {{
    border-color: {THEME["border_card"]};
    transform: translateY(-1px);
}}
.kpi-label {{
    color: {THEME["text_secondary"]};
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.kpi-value {{
    color: {THEME["text_primary"]};
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0.35rem 0 0.15rem 0;
    font-variant-numeric: tabular-nums;
}}
.kpi-subtext {{
    color: {THEME["text_muted"]};
    font-size: 0.75rem;
}}

/* Signal Badges */
.badge-buy {{
    background: {THEME["buy_glow"]};
    color: {THEME["buy"]};
    border: 1px solid {THEME["buy_border"]};
    padding: 6px 18px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.4rem;
    display: inline-block;
    letter-spacing: 0.03em;
}}
.badge-sell {{
    background: {THEME["sell_glow"]};
    color: {THEME["sell"]};
    border: 1px solid {THEME["sell_border"]};
    padding: 6px 18px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.4rem;
    display: inline-block;
    letter-spacing: 0.03em;
}}
.badge-hold {{
    background: {THEME["hold_glow"]};
    color: {THEME["hold"]};
    border: 1px solid {THEME["hold_border"]};
    padding: 6px 18px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.4rem;
    display: inline-block;
    letter-spacing: 0.03em;
}}

/* Probability mini-meter */
.prob-bar-container {{
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
    gap: 0.75rem;
}}
.prob-label {{
    width: 48px;
    font-size: 0.8rem;
    font-weight: 600;
    color: {THEME["text_secondary"]};
}}
.prob-track {{
    flex: 1;
    height: 8px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 999px;
    overflow: hidden;
}}
.prob-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}}
.prob-val {{
    width: 45px;
    text-align: right;
    font-size: 0.82rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}}

/* Feature Impact Chip */
.feat-chip {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    padding: 6px 12px;
    margin-bottom: 6px;
    font-size: 0.82rem;
}}
.feat-name {{
    color: {THEME["text_primary"]};
    font-weight: 500;
}}
.feat-val {{
    color: {THEME["text_secondary"]};
    font-variant-numeric: tabular-nums;
}}

/* Streamlit Tabs Navigation Polish */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: {THEME["bg_surface"]};
    padding: 6px;
    border-radius: 12px;
    border: 1px solid {THEME["border_subtle"]};
}}
.stTabs [data-baseweb="tab"] {{
    height: 40px;
    border-radius: 8px;
    color: {THEME["text_secondary"]};
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0 16px;
    transition: all 0.15s ease;
    border: none !important;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(255, 255, 255, 0.08) !important;
    color: {THEME["text_primary"]} !important;
    font-weight: 600;
}}

/* Streamlit Input & Button Styling */
div[data-baseweb="input"] {{
    background-color: {THEME["bg_surface"]} !important;
    border-color: {THEME["border_subtle"]} !important;
    border-radius: 10px !important;
}}
div[data-baseweb="select"] > div {{
    background-color: {THEME["bg_surface"]} !important;
    border-color: {THEME["border_subtle"]} !important;
    border-radius: 10px !important;
}}
.stButton > button {{
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}}

/* Empty / Ready State Box */
.empty-state-box {{
    text-align: center;
    padding: 3rem 1.5rem;
    border: 1px dashed {THEME["border_card"]};
    border-radius: 16px;
    background: rgba(17, 24, 39, 0.35);
}}
</style>
"""


def get_plotly_layout(height: int = 400, title: str | None = None, **kwargs) -> dict:
    """Returns a unified dark fintech Plotly layout configuration with arbitrary overrides."""
    layout = dict(
        height=height,
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(size=14, color=THEME["text_primary"], family="Inter"),
            x=0.01,
            y=0.96,
        ) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=16, t=40 if title else 20, b=20),
        font=dict(family="Inter, sans-serif", color=THEME["text_secondary"], size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=THEME["text_secondary"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            zeroline=False,
            color=THEME["text_muted"],
            tickfont=dict(size=11, color=THEME["text_muted"]),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            zeroline=False,
            color=THEME["text_muted"],
            tickfont=dict(size=11, color=THEME["text_muted"]),
        ),
        hoverlabel=dict(
            bgcolor=THEME["bg_surface"],
            bordercolor=THEME["border_card"],
            font=dict(family="Inter", color=THEME["text_primary"], size=12),
        ),
    )
    # Merge any custom overrides safely
    layout.update(kwargs)
    return layout
