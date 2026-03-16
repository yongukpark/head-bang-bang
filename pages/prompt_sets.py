from __future__ import annotations

import json
from datetime import datetime, timezone

import streamlit as st

from modules.common_app import PROMPT_LIBRARY_FILE, SAVED_PROMPTS_DIR, build_page_title
from modules.common_ui import apply_base_theme, render_title


SAVE_DIR = SAVED_PROMPTS_DIR
SAVE_FILE = PROMPT_LIBRARY_FILE


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_repo() -> dict:
    if not SAVE_FILE.exists():
        return {"sets": []}
    try:
        payload = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"sets": []}
    if not isinstance(payload, dict) or not isinstance(payload.get("sets"), list):
        return {"sets": []}
    return payload


def _save_repo(repo: dict) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_FILE.write_text(json.dumps(repo, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_prompts(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def _upsert_set(repo: dict, name: str, description: str, prompts: list[str]) -> None:
    sets = repo["sets"]
    for idx, item in enumerate(sets):
        if item.get("name") == name:
            sets[idx] = {
                "name": name,
                "description": description,
                "prompts": prompts,
                "updated_at_utc": _now_utc_iso(),
            }
            return
    sets.append(
        {
            "name": name,
            "description": description,
            "prompts": prompts,
            "updated_at_utc": _now_utc_iso(),
        }
    )


def _delete_set(repo: dict, name: str) -> bool:
    before = len(repo["sets"])
    repo["sets"] = [item for item in repo["sets"] if item.get("name") != name]
    return len(repo["sets"]) < before


def _load_set_into_editor(name: str, description: str, prompts: list[str]) -> None:
    st.session_state["repo_edit_name"] = name
    st.session_state["repo_edit_desc"] = description
    st.session_state["repo_edit_prompts"] = "\n".join(prompts)


st.set_page_config(page_title=build_page_title("Prompt Sets"), layout="wide")
apply_base_theme()
render_title("🗂️ Prompt Sets")

repo = _load_repo()
all_sets = repo["sets"]
set_names = [item.get("name", "") for item in all_sets if item.get("name")]

if "repo_edit_name" not in st.session_state:
    st.session_state["repo_edit_name"] = ""
if "repo_edit_desc" not in st.session_state:
    st.session_state["repo_edit_desc"] = ""
if "repo_edit_prompts" not in st.session_state:
    st.session_state["repo_edit_prompts"] = ""

st.caption(f"저장 파일: {SAVE_FILE}")
tab_add, tab_load = st.tabs(["세트 추가/편집", "세트 불러오기/삭제"])

with tab_add:
    st.markdown("### 세트 추가/편집")
    name = st.text_input("세트 이름", key="repo_edit_name", placeholder="예: capital-city-batch")
    description = st.text_area("설명", key="repo_edit_desc", height=80, placeholder="세트 용도/실험 목적")
    prompts_text = st.text_area(
        "프롬프트 목록 (한 줄 = 1개)",
        key="repo_edit_prompts",
        height=260,
        placeholder="What is the capital of France? Answer:\nWhat is the capital of Germany? Answer:",
    )

    if st.button("저장/업데이트"):
        normalized_name = name.strip()
        prompts = _normalize_prompts(prompts_text)
        if not normalized_name:
            st.warning("세트 이름을 입력하세요.")
        elif not prompts:
            st.warning("프롬프트를 하나 이상 입력하세요.")
        else:
            _upsert_set(repo, normalized_name, description.strip(), prompts)
            _save_repo(repo)
            st.success(f"저장 완료: {normalized_name} ({len(prompts)}개)")
            st.rerun()

with tab_load:
    st.markdown("### 세트 불러오기/삭제")
    if set_names:
        selected_name = st.selectbox("세트 선택", options=set_names, key="repo_selected_name")
        selected = next(item for item in all_sets if item["name"] == selected_name)
        selected_prompts = selected.get("prompts", [])

        st.caption(f"설명: {selected.get('description', '')}")
        st.caption(f"프롬프트 {len(selected_prompts)}개")
        for idx, prompt in enumerate(selected_prompts[:5]):
            st.write(f"{idx + 1}. {prompt}")
        if len(selected_prompts) > 5:
            st.caption(f"... 외 {len(selected_prompts) - 5}개")

        c1, c2 = st.columns(2)
        with c1:
            st.button(
                "편집 폼으로 불러오기",
                on_click=_load_set_into_editor,
                args=(selected["name"], selected.get("description", ""), selected_prompts),
            )
        with c2:
            if st.button("선택 세트 삭제"):
                deleted = _delete_set(repo, selected_name)
                if deleted:
                    _save_repo(repo)
                    st.success(f"삭제 완료: {selected_name}")
                    st.rerun()
                else:
                    st.warning("삭제할 세트를 찾지 못했습니다.")
    else:
        st.info("저장된 프롬프트 세트가 없습니다.")

st.markdown("### 세트 목록")
if not all_sets:
    st.caption("저장된 항목이 없습니다.")
else:
    for item in sorted(all_sets, key=lambda x: x.get("updated_at_utc", ""), reverse=True):
        prompts = item.get("prompts", [])
        st.markdown(f"**{item.get('name', '')}**")
        st.caption(f"{item.get('description', '')}")
        st.caption(f"프롬프트 {len(prompts)}개 | updated: {item.get('updated_at_utc', '-')}")
