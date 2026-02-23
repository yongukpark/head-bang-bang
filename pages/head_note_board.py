from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from transformers import AutoConfig

from modules.common_model import get_selected_model_name
from modules.common_ui import apply_base_theme, render_title


SAVE_DIR = Path("/home/head-bang-bang/saved_heads")
FALLBACK_MODEL_DIMS = {
    "EleutherAI/pythia-1.4b": (24, 16),
}


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def _head_id(layer_idx: int, head_idx: int) -> str:
    return f"L{layer_idx}.H{head_idx}"


def _inject_css() -> None:
    st.markdown(
        """
<style>
.head-card {
    border-radius: 10px;
    padding: 10px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.14);
    margin-bottom: 8px;
    position: relative;
}
.head-empty {
    background: #1a1f2b;
}
.head-filled {
    background: #1f3b2a;
    border-color: #39d98a;
}
.head-title {
    font-size: 13px;
    font-weight: 700;
    color: #f4f6ff;
}
.head-meta {
    font-size: 12px;
    color: #aeb7cc;
    margin-top: 4px;
}
.head-wrap {
    position: relative;
}
.head-tooltip {
    display: none;
    position: absolute;
    left: 50%;
    bottom: calc(100% + 6px);
    transform: translateX(-50%);
    min-width: 220px;
    max-width: 320px;
    padding: 8px 10px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.2);
    background: #0f1420;
    color: #eef3ff;
    font-size: 12px;
    line-height: 1.35;
    white-space: pre-wrap;
    z-index: 9999;
    pointer-events: none;
}
.head-wrap:hover .head-tooltip {
    display: block;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _sync_heads(heads: dict[str, list[str]], n_layers: int, n_heads: int) -> dict[str, list[str]]:
    synced: dict[str, list[str]] = {}
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            hid = _head_id(layer_idx, head_idx)
            notes = heads.get(hid, [])
            synced[hid] = notes if isinstance(notes, list) else []
    return synced


def _load_model_state(model_name: str) -> dict:
    save_path = SAVE_DIR / f"{_model_slug(model_name)}.json"
    if not save_path.exists():
        return {
            "n_layers": 12,
            "n_heads_per_layer": 12,
            "heads": {},
            "loaded_from": None,
        }
    try:
        payload = json.loads(save_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "n_layers": 12,
            "n_heads_per_layer": 12,
            "heads": {},
            "loaded_from": None,
        }
    return {
        "n_layers": int(payload.get("n_layers", 12)),
        "n_heads_per_layer": int(payload.get("n_heads_per_layer", 12)),
        "heads": payload.get("heads", {}) if isinstance(payload.get("heads"), dict) else {},
        "loaded_from": str(save_path),
    }


def _save_model_state(model_name: str, n_layers: int, n_heads: int, heads: dict[str, list[str]]) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "n_layers": n_layers,
        "n_heads_per_layer": n_heads,
        "heads": heads,
    }
    out_path = SAVE_DIR / f"{_model_slug(model_name)}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _build_head_tooltip(head_id: str, notes: list[str]) -> str:
    if not notes:
        return f"{head_id}\n저장된 정보 없음"
    preview = notes[:3]
    lines = [f"{i + 1}. {txt}" for i, txt in enumerate(preview)]
    if len(notes) > 3:
        lines.append(f"... 외 {len(notes) - 3}개")
    return f"{head_id}\n" + "\n".join(lines)


@st.cache_data
def _get_model_dims(model_name: str) -> tuple[int, int]:
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        return int(cfg.num_hidden_layers), int(cfg.num_attention_heads)
    except Exception:
        return FALLBACK_MODEL_DIMS.get(model_name, (12, 12))


st.set_page_config(page_title="Head Note Board", layout="wide")
apply_base_theme()
_inject_css()
render_title("🧩 Head Note Board")

selected_model_name = get_selected_model_name()
st.caption(f"현재 선택 모델: {selected_model_name}")

if "head_notes_by_model" not in st.session_state:
    st.session_state["head_notes_by_model"] = {}

by_model: dict[str, dict] = st.session_state["head_notes_by_model"]
active_model = st.session_state.get("head_note_active_model")
if selected_model_name not in by_model or active_model != selected_model_name:
    by_model[selected_model_name] = _load_model_state(selected_model_name)
    st.session_state["head_note_active_model"] = selected_model_name

state = by_model[selected_model_name]
model_key = _model_slug(selected_model_name)
model_layers, model_heads = _get_model_dims(selected_model_name)

left, right = st.columns([1, 2])
with left:
    st.markdown("### 설정")
    loaded_from = state.get("loaded_from")
    if loaded_from:
        st.caption(f"불러온 파일: {loaded_from}")
    else:
        st.caption("불러온 파일이 없습니다. 새 데이터로 시작합니다.")

    if st.button("저장 파일 다시 불러오기", key=f"reload_{model_key}"):
        by_model[selected_model_name] = _load_model_state(selected_model_name)
        st.rerun()

    st.caption(f"레이어 수: {model_layers} (모델 최대)")
    st.caption(f"레이어당 헤드 수: {model_heads} (모델 최대)")

state["n_layers"] = model_layers
state["n_heads_per_layer"] = model_heads
state["heads"] = _sync_heads(state.get("heads", {}), n_layers=model_layers, n_heads=model_heads)

with left:
    st.markdown("### 정보 입력")
    all_head_ids = list(state["heads"].keys())
    batch_heads = st.multiselect(
        "동일 정보 일괄 추가 대상",
        options=all_head_ids,
        default=[],
        key=f"batch_heads_{model_key}",
    )
    new_note = st.text_area(
        "추가할 정보",
        height=100,
        placeholder="예: 이 헤드는 숫자 토큰에 반응함",
        key=f"new_note_{model_key}",
    )

    if st.button("선택 헤드들에 동일 정보 추가", key=f"add_batch_note_{model_key}"):
        text = new_note.strip()
        if not text:
            st.warning("추가할 정보를 입력해 주세요.")
        elif not batch_heads:
            st.warning("대상 헤드를 하나 이상 선택해 주세요.")
        else:
            for hid in batch_heads:
                state["heads"][hid].append(text)
            st.success(f"{len(batch_heads)}개 헤드에 동일 정보를 추가했습니다.")

    st.markdown("### 선택 헤드 정보 목록")
    if not batch_heads:
        st.caption("헤드를 선택하면 해당 헤드들의 정보를 여기에서 확인/삭제할 수 있습니다.")
    else:
        for hid in batch_heads:
            st.markdown(f"#### {hid}")
            notes = state["heads"][hid]
            if not notes:
                st.caption("아직 입력된 정보가 없습니다.")
                continue
            for idx, note in enumerate(notes):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.write(f"{idx + 1}. {note}")
                with c2:
                    if st.button("삭제", key=f"del_{model_key}_{hid}_{idx}"):
                        del state["heads"][hid][idx]
                        st.rerun()

    st.markdown("### JSON 저장")
    if st.button("현재 모델 저장", key=f"save_{model_key}"):
        saved = _save_model_state(
            model_name=selected_model_name,
            n_layers=model_layers,
            n_heads=model_heads,
            heads=state["heads"],
        )
        state["loaded_from"] = str(saved)
        st.success(f"저장 완료: {saved}")

with right:
    st.markdown("### 헤드 상태")
    st.caption("헤드 카드에 마우스를 올리면 저장된 정보 요약이 보입니다.")

    for layer_idx in range(model_layers):
        st.markdown(f"#### Layer {layer_idx}")
        cols = st.columns(model_heads)
        for head_idx in range(model_heads):
            hid = _head_id(layer_idx, head_idx)
            note_count = len(state["heads"][hid])
            css_class = "head-filled" if note_count > 0 else "head-empty"
            tooltip = html.escape(_build_head_tooltip(hid, state["heads"][hid]))
            cols[head_idx].markdown(
                f"""
<div class="head-wrap">
  <div class="head-card {css_class}">
    <div class="head-title">{hid}</div>
    <div class="head-meta">정보 {note_count}개</div>
  </div>
  <div class="head-tooltip">{tooltip}</div>
</div>
""",
                unsafe_allow_html=True,
            )
