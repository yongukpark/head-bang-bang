import html

import streamlit as st


def apply_base_theme(top5_font_size: int = 18):
    """Apply app-wide Streamlit theme and reusable card styles."""
    st.markdown(
        f"""
<style>
header {{visibility: hidden;}}
footer {{visibility: hidden;}}

.stApp {{
    background: #0f1117;
    color: white;
}}

h1 {{
    text-align: center;
    font-size: 2.6rem;
    background: linear-gradient(90deg,#ff00cc,#7928ca,#00f2ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}}

.card {{
    background: #1a1d25;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid rgba(255,255,255,0.05);
    text-align: center;
    font-size: 20px;
    white-space: pre;
}}

.top5-card {{
    background: #20242f;
    padding: 8px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    font-size: {top5_font_size}px;
    white-space: pre;
    min-height: 72px;
}}

div[data-testid="stButton"] > button {{
    width: 100%;
    color: black;
    font-weight: 700;
    border-radius: 50px;
    height: 46px;
    background: white;
    transition: all 0.2s ease;
}}

div[data-testid="stButton"] > button:hover {{
    background: linear-gradient(90deg,#ff00cc,#7928ca);
    color: white;
    transform: scale(1.02);
}}

div[data-testid="stAlert"] {{
    background-color: #151821 !important;
    border: 1px solid #2f3542 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 13px !important;
}}

div[data-testid="stAlert"] p {{
    color: #e6e6e6 !important;
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
    subline = f"<br><span style='font-size:14px;color:#aaaaaa'>{html.escape(subtext)}</span>" if subtext else ""
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
                    <span style='font-size:14px; color:#aaaaaa;'>{probs[i]:.2%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
