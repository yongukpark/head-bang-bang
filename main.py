import streamlit as st

from modules.common_ui import apply_base_theme, render_title


st.set_page_config(page_title="Head Bang Bang Launcher", layout="wide")
apply_base_theme()

pg = st.navigation(
    {
        "Current": [
            st.Page(
                "pages/home_guide.py",
                title="Usage Guide",
                icon=":material/home:",
                default=True,
            ),
            st.Page(
                "pages/avo_patching.py",
                title="Head Intervention Lab",
                icon=":material/polyline:",
            ),
            st.Page(
                "pages/prompt_repository.py",
                title="Prompt Repository",
                icon=":material/bookmarks:",
            ),  
            st.Page(
                "pages/stable_head_mining.py",
                title="Stable Head Mining",
                icon=":material/query_stats:",
            ),
            st.Page(
                "pages/head_mlp_logit_lens.py",
                title="Architecture Lens Explorer",
                icon=":material/visibility:",
            ),
            st.Page(
                "pages/multi_prompt_head_logit_lens.py",
                title="Multi-Prompt Head Lens",
                icon=":material/view_list:",
            ),
            st.Page(
                "pages/multi_prompt_numeric_head_lens.py",
                title="Numeric Subset Head Lens",
                icon=":material/pin:",
            ),
            st.Page(
                "pages/multihead_resampling.py",
                title="Multi-Head Transfer Lab",
                icon=":material/shuffle:",
            ),
            st.Page(
                "pages/head_note_board.py",
                title="Head Knowledge Base",
                icon=":material/dashboard:",
            ),
        ],
        "Zero-ablation": [
            st.Page(
                "legacy/zero_ablation.py",
                title="Head Impact Map (Legacy)",
                icon=":material/history:",
            ),
            st.Page(
                "legacy/multihead_abulation.py",
                title="Multi-Head Deactivation (Legacy)",
                icon=":material/block:",
            ),
            st.Page(
                "legacy/multihead_addition.py",
                title="Multi-Head Activate-Only (Legacy)",
                icon=":material/bolt:",
            ),
            st.Page(
                "legacy/multiprompt.py",
                title="Multi-Prompt Heatmap (Legacy)",
                icon=":material/grid_view:",
            ),
        ],
    }
)

render_title("🧠 Head Bang Bang")
pg.run()
