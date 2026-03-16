import streamlit as st

from modules.common_app import APP_NAME, build_page_title
from modules.common_ui import apply_base_theme, render_title


st.set_page_config(page_title=build_page_title("Workspace"), layout="wide")
apply_base_theme()

pg = st.navigation(
    {
        "Workspace": [
            st.Page(
                "pages/overview.py",
                title="Overview",
                icon=":material/home:",
                default=True,
            ),
            st.Page(
                "pages/intervention_lab.py",
                title="Intervention Lab",
                icon=":material/polyline:",
            ),
            st.Page(
                "pages/prompt_sets.py",
                title="Prompt Sets",
                icon=":material/bookmarks:",
            ),
            st.Page(
                "pages/stable_head_mining.py",
                title="Stable Head Mining",
                icon=":material/query_stats:",
            ),
            st.Page(
                "pages/architecture_lens.py",
                title="Architecture Lens",
                icon=":material/visibility:",
            ),
            st.Page(
                "pages/head_logit_lens.py",
                title="Head Logit Lens",
                icon=":material/view_list:",
            ),
            st.Page(
                "pages/multi_head_transfer.py",
                title="Multi-Head Transfer",
                icon=":material/shuffle:",
            ),
        ],
    }
)

render_title(f"🧠 {APP_NAME}")
pg.run()
