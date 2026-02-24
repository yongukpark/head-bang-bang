from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from transformers import AutoConfig

from modules.common_model import get_selected_model_name
from modules.common_ui import apply_base_theme, render_title


ROOT_DIR = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT_DIR / "saved_heads"
FALLBACK_MODEL_DIMS = {
    "EleutherAI/pythia-1.4b": (24, 16),
    "EleutherAI/pythia-410m": (24, 12),
}
LEGACY_CATEGORY = "legacy_note"


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def _head_id(layer_idx: int, head_idx: int) -> str:
    return f"L{layer_idx}.H{head_idx}"


def _sanitize_categories(raw: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _sync_heads_index(n_layers: int, n_heads: int) -> list[str]:
    return [_head_id(layer_idx, head_idx) for layer_idx in range(n_layers) for head_idx in range(n_heads)]


def _sync_head_categories(
    raw: dict[str, list[str]],
    valid_heads: list[str],
    valid_categories: list[str],
) -> dict[str, list[str]]:
    cat_set = set(valid_categories)
    out: dict[str, list[str]] = {}
    for hid in valid_heads:
        assigned = raw.get(hid, [])
        if not isinstance(assigned, list):
            assigned = []
        clean: list[str] = []
        seen: set[str] = set()
        for item in assigned:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if not name or name in seen or name not in cat_set:
                continue
            seen.add(name)
            clean.append(name)
        out[hid] = clean
    return out


def _sync_head_category_details(
    raw: dict[str, dict[str, str]],
    valid_heads: list[str],
    valid_categories: list[str],
) -> dict[str, dict[str, str]]:
    cat_set = set(valid_categories)
    out: dict[str, dict[str, str]] = {}
    for hid in valid_heads:
        by_cat = raw.get(hid, {})
        if not isinstance(by_cat, dict):
            by_cat = {}
        clean: dict[str, str] = {}
        for cat, detail in by_cat.items():
            if not isinstance(cat, str):
                continue
            name = cat.strip()
            if not name or name not in cat_set:
                continue
            if isinstance(detail, str):
                clean[name] = detail
            else:
                clean[name] = ""
        out[hid] = clean
    return out


def _migrate_legacy_heads(
    legacy_heads: dict[str, list[str]],
    valid_heads: list[str],
) -> tuple[list[str], dict[str, list[str]], dict[str, dict[str, str]]]:
    categories: list[str] = []
    head_categories = {hid: [] for hid in valid_heads}
    details = {hid: {} for hid in valid_heads}

    has_legacy = False
    for hid in valid_heads:
        notes = legacy_heads.get(hid, [])
        if not isinstance(notes, list):
            continue
        texts = [n.strip() for n in notes if isinstance(n, str) and n.strip()]
        if not texts:
            continue
        has_legacy = True
        head_categories[hid] = [LEGACY_CATEGORY]
        details[hid] = {LEGACY_CATEGORY: "\n".join(f"- {txt}" for txt in texts)}

    if has_legacy:
        categories.append(LEGACY_CATEGORY)
    return categories, head_categories, details


def _load_model_state(model_name: str) -> dict:
    save_path = SAVE_DIR / f"{_model_slug(model_name)}.json"
    if not save_path.exists():
        return {
            "n_layers": 12,
            "n_heads_per_layer": 12,
            "categories": [],
            "head_categories": {},
            "head_category_details": {},
            "loaded_from": None,
        }

    try:
        payload = json.loads(save_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "n_layers": 12,
            "n_heads_per_layer": 12,
            "categories": [],
            "head_categories": {},
            "head_category_details": {},
            "loaded_from": None,
        }

    n_layers = int(payload.get("n_layers", 12))
    n_heads = int(payload.get("n_heads_per_layer", 12))
    valid_heads = _sync_heads_index(n_layers, n_heads)

    categories = _sanitize_categories(payload.get("categories", []))
    head_categories_raw = payload.get("head_categories", {})
    details_raw = payload.get("head_category_details", {})

    if isinstance(head_categories_raw, dict) and isinstance(details_raw, dict):
        head_categories = _sync_head_categories(head_categories_raw, valid_heads, categories)
        details = _sync_head_category_details(details_raw, valid_heads, categories)
    else:
        legacy_heads = payload.get("heads", {}) if isinstance(payload.get("heads"), dict) else {}
        categories, head_categories, details = _migrate_legacy_heads(legacy_heads, valid_heads)

    return {
        "n_layers": n_layers,
        "n_heads_per_layer": n_heads,
        "categories": categories,
        "head_categories": head_categories,
        "head_category_details": details,
        "loaded_from": str(save_path),
    }


def _save_model_state(
    model_name: str,
    n_layers: int,
    n_heads: int,
    categories: list[str],
    head_categories: dict[str, list[str]],
    head_category_details: dict[str, dict[str, str]],
) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "n_layers": n_layers,
        "n_heads_per_layer": n_heads,
        "categories": categories,
        "head_categories": head_categories,
        "head_category_details": head_category_details,
    }
    out_path = SAVE_DIR / f"{_model_slug(model_name)}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _tooltip_text(head_id: str, categories: list[str], details: dict[str, str]) -> str:
    if not categories:
        return f"{head_id} | 카테고리 없음"
    lines = [f"{head_id}"]
    for cat in categories[:4]:
        detail = details.get(cat, "").strip()
        preview = detail.splitlines()[0][:40] if detail else "상세 없음"
        lines.append(f"- {cat}: {preview}")
    if len(categories) > 4:
        lines.append(f"... 외 {len(categories) - 4}개")
    return "\n".join(lines)


@st.cache_data
def _get_model_dims(model_name: str) -> tuple[int, int]:
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        return int(cfg.num_hidden_layers), int(cfg.num_attention_heads)
    except Exception:
        return FALLBACK_MODEL_DIMS.get(model_name, (12, 12))


st.set_page_config(page_title="Head Knowledge Base", layout="wide")
apply_base_theme()
render_title("🧩 Head Knowledge Base")

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

state["n_layers"] = model_layers
state["n_heads_per_layer"] = model_heads
valid_heads = _sync_heads_index(model_layers, model_heads)
state["categories"] = _sanitize_categories(state.get("categories", []))
state["head_categories"] = _sync_head_categories(state.get("head_categories", {}), valid_heads, state["categories"])
state["head_category_details"] = _sync_head_category_details(
    state.get("head_category_details", {}),
    valid_heads,
    state["categories"],
)

selected_head_key = f"selected_head_{model_key}"
if selected_head_key not in st.session_state or st.session_state[selected_head_key] not in valid_heads:
    st.session_state[selected_head_key] = valid_heads[0]
selected_head = st.session_state[selected_head_key]

category_filter_key = f"category_filter_{model_key}"
all_filter = "__ALL__"
valid_filters = [all_filter] + state["categories"]
if category_filter_key not in st.session_state or st.session_state[category_filter_key] not in valid_filters:
    st.session_state[category_filter_key] = all_filter
selected_category_filter = st.session_state[category_filter_key]

left, right = st.columns([1.2, 1.8])
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

    st.caption(f"레이어 수: {model_layers}")
    st.caption(f"레이어당 헤드 수: {model_heads}")

    st.markdown("### 카테고리 관리")
    new_category = st.text_input(
        "새 카테고리",
        value="",
        placeholder="예: antonym, induction, number, syntax",
        key=f"new_category_{model_key}",
    )
    if st.button("카테고리 추가", key=f"add_category_{model_key}"):
        cat = new_category.strip()
        if not cat:
            st.warning("카테고리 이름을 입력해 주세요.")
        elif cat in state["categories"]:
            st.warning("이미 존재하는 카테고리입니다.")
        else:
            state["categories"].append(cat)
            st.success(f"카테고리 추가: {cat}")

    if state["categories"]:
        remove_target = st.selectbox(
            "삭제할 카테고리",
            options=["(선택 안 함)"] + state["categories"],
            index=0,
            key=f"remove_category_{model_key}",
        )
        if st.button("카테고리 삭제", key=f"delete_category_btn_{model_key}"):
            if remove_target == "(선택 안 함)":
                st.warning("삭제할 카테고리를 선택해 주세요.")
            else:
                state["categories"] = [c for c in state["categories"] if c != remove_target]
                for hid in valid_heads:
                    state["head_categories"][hid] = [c for c in state["head_categories"][hid] if c != remove_target]
                    state["head_category_details"][hid].pop(remove_target, None)
                st.success(f"카테고리 삭제: {remove_target}")
    else:
        st.caption("등록된 카테고리가 없습니다.")

    st.markdown("### 일괄 카테고리 매핑")
    bulk_heads = st.multiselect(
        "대상 헤드",
        options=valid_heads,
        default=[],
        key=f"bulk_heads_{model_key}",
    )
    bulk_categories = st.multiselect(
        "적용할 카테고리",
        options=state["categories"],
        default=[],
        key=f"bulk_categories_{model_key}",
    )
    bulk_mode = st.radio(
        "적용 방식",
        options=["추가", "덮어쓰기"],
        horizontal=True,
        key=f"bulk_mode_{model_key}",
    )
    if st.button("일괄 적용", key=f"bulk_apply_{model_key}"):
        if not bulk_heads:
            st.warning("대상 헤드를 하나 이상 선택해 주세요.")
        elif not bulk_categories:
            st.warning("적용할 카테고리를 하나 이상 선택해 주세요.")
        else:
            for hid in bulk_heads:
                before = state["head_categories"][hid]
                if bulk_mode == "덮어쓰기":
                    state["head_categories"][hid] = list(bulk_categories)
                else:
                    merged = list(before)
                    for cat in bulk_categories:
                        if cat not in merged:
                            merged.append(cat)
                    state["head_categories"][hid] = merged
            st.success(f"{len(bulk_heads)}개 헤드에 카테고리를 적용했습니다.")

    st.markdown("### 선택 헤드 편집")
    st.caption(f"현재 선택: {selected_head}")
    assigned = state["head_categories"][selected_head]
    selected_for_head = st.multiselect(
        "이 헤드의 카테고리",
        options=state["categories"],
        default=assigned,
        key=f"categories_for_head_{model_key}_{selected_head}",
    )
    if st.button("헤드 카테고리 저장", key=f"save_head_categories_{model_key}_{selected_head}"):
        state["head_categories"][selected_head] = list(selected_for_head)
        keep = set(selected_for_head)
        state["head_category_details"][selected_head] = {
            cat: txt for cat, txt in state["head_category_details"][selected_head].items() if cat in keep
        }
        st.success(f"{selected_head} 카테고리를 저장했습니다.")

    if state["head_categories"][selected_head]:
        st.caption("카테고리별 상세 정보를 입력하세요.")
        pending_updates: dict[str, str] = {}
        for cat in state["head_categories"][selected_head]:
            existing = state["head_category_details"][selected_head].get(cat, "")
            pending_updates[cat] = st.text_area(
                f"[{cat}] 상세 정보",
                value=existing,
                height=110,
                key=f"detail_{model_key}_{selected_head}_{cat}",
            )
        if st.button("상세 정보 저장", key=f"save_details_{model_key}_{selected_head}"):
            for cat, text in pending_updates.items():
                state["head_category_details"][selected_head][cat] = text.strip()
            st.success(f"{selected_head} 상세 정보를 저장했습니다.")
    else:
        st.caption("먼저 이 헤드에 카테고리를 지정해 주세요.")

    st.markdown("### JSON 저장")
    if st.button("현재 모델 저장", key=f"save_{model_key}"):
        saved = _save_model_state(
            model_name=selected_model_name,
            n_layers=model_layers,
            n_heads=model_heads,
            categories=state["categories"],
            head_categories=state["head_categories"],
            head_category_details=state["head_category_details"],
        )
        state["loaded_from"] = str(saved)
        st.success(f"저장 완료: {saved}")

with right:
    st.markdown("### 카테고리 정보")
    if state["categories"]:
        counts = {
            cat: sum(1 for hid in valid_heads if cat in state["head_categories"][hid])
            for cat in state["categories"]
        }
        active_filter_label = "전체" if selected_category_filter == all_filter else selected_category_filter
        st.caption(f"현재 필터: {active_filter_label}")

        if st.button("전체 보기", key=f"filter_all_{model_key}", use_container_width=True):
            st.session_state[category_filter_key] = all_filter
            st.rerun()

        cat_cols = st.columns(3)
        for idx, cat in enumerate(state["categories"]):
            label = f"{cat} ({counts[cat]})"
            if cat_cols[idx % 3].button(label, key=f"filter_{model_key}_{cat}", use_container_width=True):
                st.session_state[category_filter_key] = cat
                st.rerun()
    else:
        st.caption("등록된 카테고리가 없습니다.")

    st.markdown("### 헤드 선택")
    st.caption("헤드를 누르면 왼쪽에서 카테고리/상세 정보를 편집할 수 있습니다. 필터가 켜지면 해당 카테고리 헤드만 선택됩니다.")

    for layer_idx in range(model_layers):
        st.markdown(f"#### Layer {layer_idx}")
        cols = st.columns(model_heads)
        for head_idx in range(model_heads):
            hid = _head_id(layer_idx, head_idx)
            is_match = selected_category_filter == all_filter or selected_category_filter in state["head_categories"][hid]
            cat_count = len(state["head_categories"][hid])
            label = f"H{head_idx} ({cat_count})"
            if cols[head_idx].button(
                label,
                key=f"pick_{model_key}_{hid}",
                use_container_width=True,
                disabled=not is_match,
            ):
                st.session_state[selected_head_key] = hid
                st.rerun()
            if hid == selected_head and is_match:
                cols[head_idx].caption("선택됨")
            categories_text = ", ".join(state["head_categories"][hid][:3]) if state["head_categories"][hid] else "-"
            cols[head_idx].caption(f"카테고리: {categories_text}")
            tooltip = _tooltip_text(
                hid,
                state["head_categories"][hid],
                state["head_category_details"][hid],
            )
            cols[head_idx].caption(tooltip)
