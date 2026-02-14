import streamlit as st

from modules.common_ui import apply_base_theme, render_title

st.set_page_config(page_title="Head Bang Bang Launcher", layout="wide")
apply_base_theme()

render_title("🧠 Head Bang Bang")

pages = [
    ("Visualizing Head Impact Map", "pages/headbangbang.py", "Given one prompt, deactivate each attention head one at a time and visual the results"),
    ("Multi-Head Deactivation", "pages/multihead_abulation.py", "Given multiple prompts, deactivate the selected specific heads simultaneously"),
    ("Multi-Head Activate-Only", "pages/multihead_addition.py", "Given multiple prompts, only activate the selected specific heads simultaneously"),
    ("Multi-Prompt Heatmap", "pages/multiprompt.py", "Given multiple prompts, deactivate each attention head one at a time"),
]

for title, page_path, description in pages:
    if st.button(f"{title}", use_container_width=True):
        st.switch_page(page_path)
    st.caption(f": {description}")

