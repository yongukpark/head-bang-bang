import streamlit as st

from modules.common_ui import apply_base_theme, render_title


st.set_page_config(page_title="Head Bang Bang Guide", layout="wide")
apply_base_theme()
render_title("🧭 Head Bang Bang Usage Guide")

st.markdown("### 1) 데이터 준비")
st.markdown(
    """
- `Prompt Repository`: 분석할 프롬프트 세트를 저장합니다.
- 한 실험 주제(예: 수도 질문, 수학 질문)별로 세트를 나눠 관리하세요.
"""
)

st.markdown("### 2) 헤드 탐색")
st.markdown(
    """
- `Stable Head Mining`: 핵심 페이지입니다.
- 여러 프롬프트에 대해 헤드별 평균 영향도를 측정합니다.('개별만' 옵션)
- 하나의 헤드가 완전한 영향을 미치는 경우는 적음. 따라서 여러 헤드의 조합에 대한 영향도 측정('멀티만' 옵션)
- 주요 지표:
  - `mean_drop_prob`: 개입 후 baseline top1 확률 평균 하락량
  - `degrade_rate`: 출력 token의 확률이 내려간 비율
  - `escape_rate`: baseline top1이 top-20 밖으로 밀린 비율
  - `change_rate`: 최종 top1 자체가 바뀐 비율
  - `break_score`: 위 지표를 종합한 붕괴 점수
"""
)

st.markdown("### 3) 원인/기여 해석")
st.markdown(
    """
- `Architecture Lens Explorer`: 레이어/헤드의 내부 토큰 기여를 구조적으로 확인합니다.
- `Head Intervention Lab`: 특정 헤드를 donor 정보로 치환했을 때 모델의 출력이 어떻게 변하는지 구체적으로 확인합니다.
- `Multi-Head Transfer Lab`: 여러 헤드를 동시에 교체해 모델의 출력이 어떻게 변화하는지 확인합니다.
"""
)

st.markdown("### 4) 지식 정리")
st.markdown(
    """
- `Head Knowledge Base`: 찾은 헤드 특성을 모델별로 기록/관리합니다.
- 실험에서 반복적으로 확인된 패턴만 남겨 최종 의사결정 근거를 축적합니다.
"""
)

st.markdown("## 추천 파이프라인")
st.markdown(
    """
의사결정 순서:
1. `Head Intervention Lab`에서 실험에 사용할 주제 및 프롬프트 구조를 선정
2. 해당 주제 및 프롬프트들을 모아 `Prompt Repository`에 저장
3. 해당 프롬프트들을 토대로 `Stable Head Mining (개별만)`에서 영향력 있는 헤드 후보 추출
4. `Architecture Lens Explorer`로 실제 모델의 영향력이 어떻게 변하는지 측정
5. 여러 헤드의 영향력을 파악하기 위해 `Multi-Head Transfer Lab`에서 여러 헤드의 조합으로 모델이 망가지는 모습 확인
6. `Stable Head Mining (멀티만)`으로 실제 조합 안정성 검증
7. `Head Knowledge Base`에 근거 기록 후 최종 채택/제외 결정
"""
)

st.info("팁: 한 번의 결과보다, 여러 프롬프트 세트에서 반복 재현되는 헤드를 우선 채택하세요.")
