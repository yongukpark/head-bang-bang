import streamlit as st

from modules.common_ui import apply_base_theme, render_title

st.set_page_config(page_title="Head Bang Bang Launcher", layout="wide")
apply_base_theme()

pg = st.navigation(
    {
        "Current": [
            st.Page(
                "pages/avo_patching.py",
                title="A*V*W_O Patching",
                icon=":material/polyline:",
                default=True,
            ),
            st.Page(
                "pages/multihead_resampling.py",
                title="Multi-Head Resampling",
                icon=":material/shuffle:",
            ),
        ],
        "Zero-ablation": [
            st.Page(
                "legacy/zero_ablation.py",
                title="Visualizing Head Impact Map (Zero Ablation)",
                icon=":material/history:",
            ),
            st.Page(
                "legacy/multihead_abulation.py",
                title="Multi-Head Deactivation",
                icon=":material/block:",
            ),
            st.Page(
                "legacy/multihead_addition.py",
                title="Multi-Head Activate-Only",
                icon=":material/bolt:",
            ),
            st.Page(
                "legacy/multiprompt.py",
                title="Multi-Prompt Heatmap",
                icon=":material/grid_view:",
            ),
        ],
    }
)

render_title("🧠 Head Bang Bang")
pg.run()
