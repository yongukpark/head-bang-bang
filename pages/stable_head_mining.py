import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from modules.common_inference import encode_prompt, forward_last_token, summarize_prediction
from modules.common_model import get_device, get_selected_model_name, load_model
from modules.common_ui import apply_base_theme, render_title


PROMPT_REPO_FILE = Path("/home/head-bang-bang/saved_prompts/prompt_library.json")


def _head_id(layer_idx: int, head_idx: int) -> str:
    return f"L{layer_idx}.H{head_idx}"


def _replace_last_token_head_hook(head_idx: int, n_heads: int, donor_head_vec: torch.Tensor):
    def hook(module, inputs):
        hidden = inputs[0]
        bsz, seq_len, hidden_dim = hidden.shape
        head_dim = hidden_dim // n_heads
        start = head_idx * head_dim
        end = start + head_dim
        patched = hidden.clone()
        patched[:, seq_len - 1, start:end] = donor_head_vec.to(hidden.device)
        return (patched,)

    return hook


def _replace_last_token_heads_hook(
    head_indices: list[int], n_heads: int, donor_head_vec_by_head: dict[int, torch.Tensor]
):
    def hook(module, inputs):
        hidden = inputs[0]
        _, seq_len, hidden_dim = hidden.shape
        head_dim = hidden_dim // n_heads
        patched = hidden.clone()
        for head_idx in head_indices:
            vec = donor_head_vec_by_head.get(head_idx)
            if vec is None:
                continue
            start = head_idx * head_dim
            end = start + head_dim
            patched[:, seq_len - 1, start:end] = vec.to(hidden.device)
        return (patched,)

    return hook


def _capture_attn_pre_dense_last(model, input_ids: torch.Tensor, n_layers: int) -> dict[int, torch.Tensor]:
    cached: dict[int, torch.Tensor] = {}
    handles = []

    def build_hook(layer_idx: int):
        def hook(module, inputs):
            cached[layer_idx] = inputs[0].detach().clone()
            return inputs

        return hook

    for layer_idx in range(n_layers):
        handle = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(build_hook(layer_idx))
        handles.append(handle)

    with torch.no_grad():
        _ = model(input_ids)

    for handle in handles:
        handle.remove()

    last_token_by_layer: dict[int, torch.Tensor] = {}
    for layer_idx, hidden in cached.items():
        last_token_by_layer[layer_idx] = hidden[0, -1].cpu()
    return last_token_by_layer


def _rank_heads(results: list[dict], top_n: int) -> tuple[list[dict], go.Figure]:
    sorted_rows = sorted(results, key=lambda row: row["break_score"], reverse=True)
    top_rows = sorted_rows[:top_n]

    xs = [row["degrade_rate"] for row in sorted_rows]
    ys = [row["std_delta"] for row in sorted_rows]
    sizes = [8 + 36 * row["change_rate"] for row in sorted_rows]
    colors = [row["layer"] for row in sorted_rows]
    hover = [
        (
            f"{row['head_id']}<br>"
            f"degrade_rate: {row['degrade_rate']:.2%}<br>"
            f"std_delta: {row['std_delta']:.5f}<br>"
            f"change_rate: {row['change_rate']:.2%}<br>"
            f"before_top5_rate: {row['before_top5_rate']:.2%}<br>"
            f"after_top5_rate: {row['after_top5_rate']:.2%}<br>"
            f"before_top20_rate: {row['before_top20_rate']:.2%}<br>"
            f"after_top20_rate: {row['after_top20_rate']:.2%}<br>"
            f"break_score: {row['break_score']:.5f}"
        )
        for row in sorted_rows
    ]

    fig = go.Figure(
        data=go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="Layer"),
                opacity=0.85,
                line=dict(width=0.5, color="rgba(255,255,255,0.35)"),
            ),
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Intervention Break Map",
        xaxis_title="Degrade Rate",
        yaxis_title="Std Delta",
        height=520,
        margin=dict(l=16, r=16, t=56, b=16),
    )
    return top_rows, fig


def _load_prompt_sets() -> list[dict]:
    if not PROMPT_REPO_FILE.exists():
        return []
    try:
        payload = json.loads(PROMPT_REPO_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    sets = payload.get("sets", []) if isinstance(payload, dict) else []
    return [item for item in sets if isinstance(item, dict) and item.get("name") and isinstance(item.get("prompts"), list)]


st.set_page_config(page_title="Stable Head Mining", layout="wide")
apply_base_theme()
render_title("⛏️ Stable Head Mining")

device = get_device()
selected_model_name = get_selected_model_name()
model, tokenizer = load_model(selected_model_name, str(device))
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
total_heads = n_layers * n_heads

st.caption(f"Model: {selected_model_name}")
st.caption(f"Device: {device}")
st.caption(f"Layers: {n_layers}, Heads per layer: {n_heads}, Total heads: {total_heads}")

prompt_sets = _load_prompt_sets()
input_mode = st.radio("프롬프트 입력 방식", options=["직접 입력", "저장소 세트 선택"], horizontal=True)
selected_set_prompts: list[str] = []

if input_mode == "저장소 세트 선택":
    if not prompt_sets:
        st.warning("저장소 세트가 없습니다. Prompt Repository 페이지에서 먼저 세트를 저장하세요.")
    else:
        names = [item["name"] for item in prompt_sets]
        selected_name = st.selectbox("세트 선택", options=names)
        selected = next(item for item in prompt_sets if item["name"] == selected_name)
        selected_set_prompts = [str(p).strip() for p in selected.get("prompts", []) if str(p).strip()]
        st.markdown("### 선택 세트 프롬프트")
        for idx, prompt in enumerate(selected_set_prompts):
            st.write(f"{idx + 1}. {prompt}")
else:
    prompt_text = st.text_area(
        "프롬프트를 여러 줄로 입력하세요 (한 줄 = 1 프롬프트)",
        """What is the capital of France? Answer:
What is the capital of Germany? Answer:
What is the capital of Korea? Answer:""",
        height=180,
    )

top_n = int(st.number_input("Top-N stable heads", min_value=5, max_value=200, value=30, step=5))
intervention_mode = st.radio(
    "Intervention 방식",
    options=["resampling", "mean"],
    format_func=lambda m: "Resampling (다른 프롬프트 head 값 주입)" if m == "resampling" else "Mean (프롬프트 평균 head 값 주입)",
)
analysis_scope = st.radio("분석 대상", options=["개별만", "멀티만"], horizontal=True)
all_head_ids = [_head_id(layer, head) for layer in range(n_layers) for head in range(n_heads)]
selected_multi_heads = st.multiselect(
    "동시 개입할 헤드 세트",
    options=all_head_ids,
    disabled=analysis_scope != "멀티만",
)
run = st.button("🚀 Analyze Stable Heads")

if run:
    if input_mode == "저장소 세트 선택":
        prompts = selected_set_prompts
    else:
        prompts = [line.strip() for line in prompt_text.splitlines() if line.strip()]

    if not prompts:
        st.warning("프롬프트를 한 줄 이상 입력하세요.")
        st.stop()

    if intervention_mode == "resampling" and len(prompts) < 2:
        st.warning("Resampling 방식은 프롬프트가 최소 2개 필요합니다.")
        st.stop()

    do_individual = analysis_scope == "개별만"
    do_multi = analysis_scope == "멀티만"

    if do_multi and not selected_multi_heads:
        st.warning("멀티만 모드에서는 동시 개입할 헤드를 하나 이상 선택하세요.")
        st.stop()

    base_steps = len(prompts)
    individual_steps = len(prompts) * total_heads if do_individual else 0
    multi_steps = len(prompts) if do_multi else 0
    total_steps = base_steps + individual_steps + multi_steps
    progress = st.progress(0.0)
    step = 0

    deltas_by_head: dict[tuple[int, int], list[float]] = {
        (layer, head): [] for layer in range(n_layers) for head in range(n_heads)
    }
    changed_by_head: dict[tuple[int, int], int] = {
        (layer, head): 0 for layer in range(n_layers) for head in range(n_heads)
    }
    before_top5_hits_by_head: dict[tuple[int, int], int] = {
        (layer, head): 0 for layer in range(n_layers) for head in range(n_heads)
    }
    before_top20_hits_by_head: dict[tuple[int, int], int] = {
        (layer, head): 0 for layer in range(n_layers) for head in range(n_heads)
    }
    after_top5_hits_by_head: dict[tuple[int, int], int] = {
        (layer, head): 0 for layer in range(n_layers) for head in range(n_heads)
    }
    after_top20_hits_by_head: dict[tuple[int, int], int] = {
        (layer, head): 0 for layer in range(n_layers) for head in range(n_heads)
    }
    baseline_items: list[dict] = []
    captured_last_by_prompt: list[dict[int, torch.Tensor]] = []

    for prompt in prompts:
        input_ids = encode_prompt(tokenizer, prompt, device)
        baseline_last, baseline_probs = forward_last_token(model, input_ids)
        baseline = summarize_prediction(tokenizer, baseline_last, baseline_probs)
        baseline_target_prob = baseline_probs[baseline.top1_id].item()
        captured_last_by_prompt.append(_capture_attn_pre_dense_last(model, input_ids, n_layers=n_layers))
        baseline_items.append(
            {
                "input_ids": input_ids,
                "baseline": baseline,
                "baseline_probs": baseline_probs,
                "baseline_target_prob": baseline_target_prob,
            }
        )
        step += 1
        progress.progress(step / total_steps)

    head_dim = model.config.hidden_size // n_heads
    mean_head_vec_by_layer_head: dict[tuple[int, int], torch.Tensor] = {}
    if intervention_mode == "mean":
        for layer in range(n_layers):
            stacked = torch.stack([captured[layer] for captured in captured_last_by_prompt], dim=0)
            for head in range(n_heads):
                start = head * head_dim
                end = start + head_dim
                mean_head_vec_by_layer_head[(layer, head)] = stacked[:, start:end].mean(dim=0).cpu()

    prompt_count = len(prompts)
    if do_individual:
        for prompt_idx in range(prompt_count):
            input_ids = baseline_items[prompt_idx]["input_ids"]
            baseline = baseline_items[prompt_idx]["baseline"]
            baseline_probs = baseline_items[prompt_idx]["baseline_probs"]
            baseline_target_prob = baseline_items[prompt_idx]["baseline_target_prob"]
            donor_idx = (prompt_idx + 1) % prompt_count
            donor_hidden_by_layer = captured_last_by_prompt[donor_idx]
            before_top5_ids = torch.topk(baseline_probs, k=min(5, baseline_probs.shape[-1])).indices.tolist()
            before_top20_ids = torch.topk(baseline_probs, k=min(20, baseline_probs.shape[-1])).indices.tolist()
            before_hit5 = baseline.top1_id in before_top5_ids
            before_hit20 = baseline.top1_id in before_top20_ids

            for layer in range(n_layers):
                for head in range(n_heads):
                    if intervention_mode == "resampling":
                        start = head * head_dim
                        end = start + head_dim
                        donor_head_vec = donor_hidden_by_layer[layer][start:end]
                    else:
                        donor_head_vec = mean_head_vec_by_layer_head[(layer, head)]

                    handle = model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(
                        _replace_last_token_head_hook(head, n_heads, donor_head_vec)
                    )
                    modified_last, modified_probs = forward_last_token(model, input_ids)
                    modified = summarize_prediction(tokenizer, modified_last, modified_probs)
                    handle.remove()

                    delta = modified_probs[baseline.top1_id].item() - baseline_target_prob
                    key = (layer, head)
                    deltas_by_head[key].append(delta)
                    if modified.top1_id != baseline.top1_id:
                        changed_by_head[key] += 1
                    if before_hit5:
                        before_top5_hits_by_head[key] += 1
                    if before_hit20:
                        before_top20_hits_by_head[key] += 1
                    after_top5_ids = torch.topk(modified_probs, k=min(5, modified_probs.shape[-1])).indices.tolist()
                    after_top20_ids = torch.topk(modified_probs, k=min(20, modified_probs.shape[-1])).indices.tolist()
                    if baseline.top1_id in after_top5_ids:
                        after_top5_hits_by_head[key] += 1
                    if baseline.top1_id in after_top20_ids:
                        after_top20_hits_by_head[key] += 1

                    step += 1
                    progress.progress(step / total_steps)

        rows: list[dict] = []
        for layer in range(n_layers):
            for head in range(n_heads):
                key = (layer, head)
                arr = np.array(deltas_by_head[key], dtype=np.float64)
                mean_delta = float(arr.mean())
                std_delta = float(arr.std())
                change_rate = changed_by_head[key] / prompt_count
                degrade_rate = float((arr < 0).sum() / max(1, arr.size))
                before_top5_rate = before_top5_hits_by_head[key] / prompt_count
                before_top20_rate = before_top20_hits_by_head[key] / prompt_count
                after_top5_rate = after_top5_hits_by_head[key] / prompt_count
                after_top20_rate = after_top20_hits_by_head[key] / prompt_count
                top5_drop = max(0.0, before_top5_rate - after_top5_rate)
                top20_drop = max(0.0, before_top20_rate - after_top20_rate)

                break_score = (
                    (0.5 + 0.5 * degrade_rate)
                    * (0.5 + 0.5 * change_rate)
                    * (1.0 + top5_drop + top20_drop)
                )
                rows.append(
                    {
                        "head_id": _head_id(layer, head),
                        "layer": layer,
                        "head": head,
                        "mean_delta": mean_delta,
                        "std_delta": std_delta,
                        "change_rate": change_rate,
                        "degrade_rate": degrade_rate,
                        "before_top5_rate": before_top5_rate,
                        "after_top5_rate": after_top5_rate,
                        "before_top20_rate": before_top20_rate,
                        "after_top20_rate": after_top20_rate,
                        "break_score": break_score,
                    }
                )

        top_rows, fig = _rank_heads(rows, top_n=top_n)

        st.markdown("### Top Consistently Breaking Heads")
        st.dataframe(top_rows, use_container_width=True, hide_index=True)
        st.plotly_chart(fig, use_container_width=True)

    if do_multi and selected_multi_heads:
        selected_pairs: list[tuple[int, int]] = []
        for label in selected_multi_heads:
            layer_str, head_str = label.split(".H")
            selected_pairs.append((int(layer_str[1:]), int(head_str)))

        multi_deltas: list[float] = []
        multi_changed = 0
        multi_before_top5_hits = 0
        multi_before_top20_hits = 0
        multi_after_top5_hits = 0
        multi_after_top20_hits = 0
        for prompt_idx in range(prompt_count):
            input_ids = baseline_items[prompt_idx]["input_ids"]
            baseline = baseline_items[prompt_idx]["baseline"]
            baseline_probs = baseline_items[prompt_idx]["baseline_probs"]
            baseline_target_prob = baseline_items[prompt_idx]["baseline_target_prob"]
            before_top5_ids = torch.topk(baseline_probs, k=min(5, baseline_probs.shape[-1])).indices.tolist()
            before_top20_ids = torch.topk(baseline_probs, k=min(20, baseline_probs.shape[-1])).indices.tolist()
            donor_idx = (prompt_idx + 1) % prompt_count
            donor_hidden_by_layer = captured_last_by_prompt[donor_idx]

            heads_by_layer: dict[int, list[int]] = {}
            for layer_idx, head_idx in selected_pairs:
                heads_by_layer.setdefault(layer_idx, []).append(head_idx)

            handles = []
            for layer_idx, head_list in heads_by_layer.items():
                donor_vecs: dict[int, torch.Tensor] = {}
                for head_idx in head_list:
                    start = head_idx * head_dim
                    end = start + head_dim
                    if intervention_mode == "resampling":
                        donor_vecs[head_idx] = donor_hidden_by_layer[layer_idx][start:end]
                    else:
                        donor_vecs[head_idx] = mean_head_vec_by_layer_head[(layer_idx, head_idx)]
                handle = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(
                    _replace_last_token_heads_hook(head_list, n_heads, donor_vecs)
                )
                handles.append(handle)

            modified_last, modified_probs = forward_last_token(model, input_ids)
            modified = summarize_prediction(tokenizer, modified_last, modified_probs)
            for handle in handles:
                handle.remove()

            delta = modified_probs[baseline.top1_id].item() - baseline_target_prob
            multi_deltas.append(delta)
            if modified.top1_id != baseline.top1_id:
                multi_changed += 1
            if baseline.top1_id in before_top5_ids:
                multi_before_top5_hits += 1
            if baseline.top1_id in before_top20_ids:
                multi_before_top20_hits += 1
            after_top5_ids = torch.topk(modified_probs, k=min(5, modified_probs.shape[-1])).indices.tolist()
            after_top20_ids = torch.topk(modified_probs, k=min(20, modified_probs.shape[-1])).indices.tolist()
            if baseline.top1_id in after_top5_ids:
                multi_after_top5_hits += 1
            if baseline.top1_id in after_top20_ids:
                multi_after_top20_hits += 1
            step += 1
            progress.progress(step / total_steps)

        multi_arr = np.array(multi_deltas, dtype=np.float64)
        multi_mean_delta = float(multi_arr.mean()) if multi_arr.size else 0.0
        multi_std = float(multi_arr.std()) if multi_arr.size else 0.0
        multi_degrade = float((multi_arr < 0).sum() / max(1, multi_arr.size))
        multi_change_rate = multi_changed / max(1, prompt_count)
        multi_before_top5_rate = multi_before_top5_hits / max(1, prompt_count)
        multi_before_top20_rate = multi_before_top20_hits / max(1, prompt_count)
        multi_after_top5_rate = multi_after_top5_hits / max(1, prompt_count)
        multi_after_top20_rate = multi_after_top20_hits / max(1, prompt_count)
        multi_top5_drop = max(0.0, multi_before_top5_rate - multi_after_top5_rate)
        multi_top20_drop = max(0.0, multi_before_top20_rate - multi_after_top20_rate)
        multi_break_score = (
            (0.5 + 0.5 * multi_degrade)
            * (0.5 + 0.5 * multi_change_rate)
            * (1.0 + multi_top5_drop + multi_top20_drop)
        )

        st.markdown("### Multi-Head Set Evaluation")
        st.caption(f"선택 헤드 수: {len(selected_multi_heads)}")
        st.dataframe(
            [
                {
                    "head_set": ", ".join(selected_multi_heads),
                    "mean_delta": multi_mean_delta,
                    "std_delta": multi_std,
                    "degrade_rate": multi_degrade,
                    "change_rate": multi_change_rate,
                    "before_top5_rate": multi_before_top5_rate,
                    "after_top5_rate": multi_after_top5_rate,
                    "before_top20_rate": multi_before_top20_rate,
                    "after_top20_rate": multi_after_top20_rate,
                    "break_score": multi_break_score,
                }
            ],
            use_container_width=True,
            hide_index=True,
        )

    progress.empty()

    st.caption(
        f"개입 방식: {intervention_mode}. "
        "지표 설명: degrade_rate(하락 비율), change_rate(top1 변경 비율), "
        "before/after top5, top20 rate(개입 전/후 비율), break_score(붕괴 종합 점수)."
    )
