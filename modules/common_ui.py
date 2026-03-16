import html

import streamlit as st


BLUE_DEEP = "#406093"
BLUE_BRIGHT = "#4C8CE4"
GREEN_ACCENT = "#91D06C"
YELLOW_ACCENT = "#FFF799"
TEXT_PRIMARY = BLUE_DEEP
TEXT_MUTED = "rgba(64, 96, 147, 0.72)"
SURFACE_PRIMARY = "rgba(255, 255, 255, 0.88)"
SURFACE_SECONDARY = "rgba(255, 255, 255, 0.72)"
SURFACE_HIGHLIGHT = "rgba(255, 247, 153, 0.24)"
BORDER_COLOR = "rgba(64, 96, 147, 0.18)"
PLOT_BACKGROUND = "#F7FBFF"


def apply_base_theme(top5_font_size: int = 18):
    """Apply app-wide Streamlit theme and reusable card styles."""
    st.markdown(
        f"""
<style>
:root {{
    --blue-deep: {BLUE_DEEP};
    --blue-bright: {BLUE_BRIGHT};
    --green-accent: {GREEN_ACCENT};
    --yellow-accent: {YELLOW_ACCENT};
    --text-primary: {TEXT_PRIMARY};
    --text-muted: {TEXT_MUTED};
    --surface-primary: {SURFACE_PRIMARY};
    --surface-secondary: {SURFACE_SECONDARY};
    --surface-highlight: {SURFACE_HIGHLIGHT};
    --border-color: {BORDER_COLOR};
}}

header {{visibility: visible;}}
footer {{visibility: hidden;}}

[data-testid="stHeader"] {{
    background: rgba(255, 255, 255, 0.7) !important;
    border-bottom: 1px solid var(--border-color);
    backdrop-filter: blur(16px);
}}

[data-testid="stToolbar"] button {{
    color: var(--text-primary) !important;
}}

.stApp {{
    background:
        radial-gradient(circle at top left, rgba(255, 247, 153, 0.52), transparent 30%),
        radial-gradient(circle at top right, rgba(145, 208, 108, 0.22), transparent 28%),
        linear-gradient(180deg, #f9fcff 0%, #eef5ff 100%);
    color: var(--text-primary);
}}

[data-testid="stSidebar"] {{
    background: rgba(255, 255, 255, 0.76);
    border-right: 1px solid var(--border-color);
}}

[data-testid="stSidebar"] * {{
    color: var(--text-primary);
}}

[data-testid="stMarkdownContainer"],
label,
.stCaption,
.stRadio,
.stSelectbox,
.stMultiSelect,
.stTextInput,
.stTextArea,
.stNumberInput {{
    color: var(--text-primary);
}}

[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
textarea {{
    background: rgba(255, 255, 255, 0.92) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}}

[data-baseweb="tag"] {{
    background: rgba(76, 140, 228, 0.12) !important;
    border: 1px solid rgba(76, 140, 228, 0.22) !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid transparent;
    border-radius: 999px;
    color: var(--text-primary);
}}

.stTabs [aria-selected="true"] {{
    background: rgba(76, 140, 228, 0.16);
    border-color: rgba(76, 140, 228, 0.32);
}}

h1 {{
    text-align: center;
    font-size: 2.6rem;
    color: var(--text-primary);
    margin-bottom: 18px;
    letter-spacing: -0.03em;
}}

.card {{
    background: var(--surface-primary);
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid var(--border-color);
    box-shadow: 0 18px 36px rgba(64, 96, 147, 0.08);
    text-align: center;
    font-size: 20px;
    white-space: pre;
}}

.top5-card {{
    background: var(--surface-secondary);
    padding: 8px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    font-size: {top5_font_size}px;
    white-space: pre;
    min-height: 72px;
    border: 1px solid var(--border-color);
}}

.top5-card-changed {{
    border: 1px solid rgba(255, 247, 153, 0.7);
    background: rgba(255, 247, 153, 0.34);
}}

.top5-card-new {{
    border: 1px solid rgba(145, 208, 108, 0.7);
    background: rgba(145, 208, 108, 0.22);
}}

div[data-testid="stButton"] > button {{
    width: 100%;
    color: white;
    font-weight: 700;
    border-radius: 999px;
    height: 46px;
    background: var(--blue-bright);
    border: 1px solid rgba(64, 96, 147, 0.14);
    box-shadow: 0 12px 24px rgba(76, 140, 228, 0.2);
    transition: all 0.2s ease;
}}

div[data-testid="stButton"] > button:hover {{
    background: var(--blue-deep);
    color: white;
    transform: translateY(-1px);
}}

div[data-testid="stAlert"] {{
    background-color: rgba(255, 247, 153, 0.22) !important;
    border: 1px solid rgba(64, 96, 147, 0.18) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 13px !important;
}}

div[data-testid="stAlert"] p {{
    color: var(--text-primary) !important;
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid var(--border-color);
    border-radius: 14px;
    overflow: hidden;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_title(title: str):
    st.markdown(f"<h1>{html.escape(title)}</h1>", unsafe_allow_html=True)


def visualize_token(token: str) -> str:
    """Replace whitespace with visible symbols for easier token inspection."""
    return "".join("␣" if c.isspace() else c for c in token)


def render_token_card(label: str, token: str, subtext: str | None = None):
    subline = (
        f"<br><span style='font-size:14px;color:{TEXT_MUTED}'>{html.escape(subtext)}</span>" if subtext else ""
    )
    st.markdown(
        f"<div class='card'><b>{html.escape(label)}</b><br><span style='font-size:22px;'>{html.escape(visualize_token(token))}</span>{subline}</div>",
        unsafe_allow_html=True,
    )


def render_value_card(label: str, value: str):
    st.markdown(
        f"<div class='card'><b>{html.escape(label)}</b><br><span style='font-size:22px;'>{html.escape(value)}</span></div>",
        unsafe_allow_html=True,
    )


def render_top5_cards(tokens: list[str], probs: list[float]):
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.markdown(
                f"""
                <div class='top5-card'>
                    {html.escape(visualize_token(tokens[i]))}<br>
                    <span style='font-size:14px; color:{TEXT_MUTED};'>{probs[i]:.2%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_top5_diff_cards(
    tokens: list[str],
    probs: list[float],
    baseline_tokens: list[str],
    top1_changed: bool,
):
    """Render Top-5 tokens and color-code changes vs baseline.

    - Red: Top-1 token replaced baseline Top-1.
    - Green: Token newly entered Top-5.
    """
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            status_class = ""
            if top1_changed and i == 0:
                status_class = "top5-card-changed"
            elif tokens[i] not in baseline_tokens:
                status_class = "top5-card-new"

            st.markdown(
                f"""
                <div class='top5-card {status_class}'>
                    {html.escape(visualize_token(tokens[i]))}<br>
                    <span style='font-size:14px; color:{TEXT_MUTED};'>{probs[i]:.2%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
