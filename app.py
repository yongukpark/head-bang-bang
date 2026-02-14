import streamlit as st

from modules.common_ui import apply_base_theme, render_title

st.set_page_config(page_title="Head Bang Bang Launcher", layout="wide")
apply_base_theme()

render_title("🧠 Head Bang Bang - Main")
st.markdown("원하는 분석 페이지를 선택하세요.")

pages = [
    ("Head Impact Map", "pages/headbangbang.py", "전체 헤드를 하나씩 제거하며 영향도를 시각화합니다."),
    ("Multi-Head Ablation", "pages/multihead_abulation.py", "선택한 여러 헤드를 동시에 비활성화합니다."),
    ("Multi-Head Keep-Only", "pages/multihead_addition.py", "선택한 헤드만 통과시키고 나머지는 차단합니다."),
    ("Multi-Prompt Heatmap", "pages/multiprompt.py", "여러 프롬프트에서 헤드 중요도 순위를 비교합니다."),
]

for title, page_path, description in pages:
    st.markdown(f"### {title}")
    st.caption(description)
    if st.button(f"Open {title}", use_container_width=True):
        st.switch_page(page_path)
