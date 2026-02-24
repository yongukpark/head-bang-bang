from __future__ import annotations

import re

import streamlit as st
import torch
from transformers import AutoTokenizer

from modules.common_inference import encode_prompt
from modules.common_model import get_device, get_selected_model_name, load_model
from modules.common_ui import apply_base_theme, render_title, visualize_token


def _parse_prompts(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _capture_attn_pre_last_token(model, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    o_proj = model.gpt_neox.layers[layer_idx].attention.dense
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, inputs):
        captured["attn_pre"] = inputs[0].detach()
        return inputs

    handle = o_proj.register_forward_pre_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        handle.remove()

    if "attn_pre" not in captured:
        raise RuntimeError("Failed to capture attention pre-W_O activation.")
    return captured["attn_pre"][0, -1]


def _head_contribution_after_wo(model, layer_idx: int, head_idx: int, attn_pre_last: torch.Tensor) -> torch.Tensor:
    n_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // n_heads

    heads = attn_pre_last.view(n_heads, head_dim)
    dense_weight = model.gpt_neox.layers[layer_idx].attention.dense.weight.detach()
    w_by_head = dense_weight.view(hidden_size, n_heads, head_dim).permute(1, 0, 2).contiguous()
    return torch.einsum("d,od->o", heads[head_idx], w_by_head[head_idx]).detach()


@st.cache_data(show_spinner=False)
def _numeric_subset_ids(
    tokenizer_name: str,
    vocab_size: int,
    allow_sign: bool,
    allow_decimal: bool,
) -> tuple[list[int], list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if allow_decimal:
        pattern = r"^[+-]?\d+(?:[.,]\d+)?$" if allow_sign else r"^\d+(?:[.,]\d+)?$"
    else:
        pattern = r"^[+-]?\d+$" if allow_sign else r"^\d+$"
    regex = re.compile(pattern)

    ids: list[int] = []
    previews: list[str] = []
    for token_id in range(vocab_size):
        try:
            tok = tokenizer.decode([token_id])
        except Exception:
            continue
        stripped = tok.strip()
        if not stripped:
            continue
        if regex.fullmatch(stripped):
            ids.append(token_id)
            if len(previews) < 20:
                previews.append(visualize_token(tok))
    return ids, previews


def _run_numeric_head_lens(
    model,
    tokenizer,
    head_vec: torch.Tensor,
    topk: int,
    apply_final_ln: bool,
    numeric_ids: list[int],
) -> tuple[list[str], list[int], list[float], list[float]]:
    final_ln = model.gpt_neox.final_layer_norm
    lm_head = model.embed_out if hasattr(model, "embed_out") else model.lm_head
    device = next(model.parameters()).device

    vec = head_vec.to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        if apply_final_ln:
            vec = final_ln(vec)
        logits = lm_head(vec).squeeze(0).squeeze(0)
        probs_global = torch.softmax(logits, dim=-1)

    subset_ids_tensor = torch.tensor(numeric_ids, device=device, dtype=torch.long)
    subset_logits = logits.index_select(0, subset_ids_tensor)
    subset_probs = torch.softmax(subset_logits, dim=-1)

    k = min(topk, subset_logits.numel())
    top_vals, top_pos = torch.topk(subset_logits, k=k, dim=-1)
    chosen_ids = subset_ids_tensor[top_pos]
    top_subset_probs = subset_probs[top_pos]
    top_global_probs = probs_global[chosen_ids]

    top_tokens = [visualize_token(tokenizer.decode([idx.item()])) for idx in chosen_ids]
    return (
        top_tokens,
        [int(idx.item()) for idx in chosen_ids],
        top_vals.tolist(),
        top_subset_probs.tolist(),
        top_global_probs.tolist(),
    )


st.set_page_config(page_title="Multi-Prompt Numeric Head Lens", layout="wide")
apply_base_theme()
render_title("🔢 Multi-Prompt Numeric Head Lens")

selected_model_name = get_selected_model_name()
device = get_device()

try:
    model, tokenizer = load_model(selected_model_name, device.type)
except RuntimeError as exc:
    st.error(f"모델 로딩 실패: {exc}")
    st.stop()

n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
vocab_size = model.config.vocab_size

st.caption(f"모델: {selected_model_name}")
st.caption(f"레이어 수: {n_layers}, 레이어당 헤드 수: {n_heads}, vocab: {vocab_size}")

left, right = st.columns([1, 1])
with left:
    layer_idx = int(st.number_input("Layer", min_value=0, max_value=n_layers - 1, value=0, step=1))
    head_idx = int(st.number_input("Head", min_value=0, max_value=n_heads - 1, value=0, step=1))
    topk = int(st.slider("Top-k (numeric subset)", min_value=1, max_value=50, value=10))
with right:
    apply_final_ln = st.toggle("Apply final layer norm before LM head", value=True)
    allow_sign = st.toggle("Allow signed numbers (+/-)", value=False)
    allow_decimal = st.toggle("Allow decimals (.,)", value=False)

numeric_ids, preview_tokens = _numeric_subset_ids(
    tokenizer_name=selected_model_name,
    vocab_size=vocab_size,
    allow_sign=allow_sign,
    allow_decimal=allow_decimal,
)
st.caption(f"Numeric subset size: {len(numeric_ids)}")
if preview_tokens:
    st.caption("예시 토큰: " + ", ".join(preview_tokens[:10]))

default_prompts = "\n".join(
    [
        "2 + 2 =",
        "In 2024, the population was",
        "The answer is",
    ]
)
raw_prompts = st.text_area(
    "Prompts (한 줄에 하나씩)",
    value=default_prompts,
    height=180,
    placeholder="여기에 여러 프롬프트를 줄바꿈으로 입력하세요.",
)

if st.button("Run Numeric Subset Lens", use_container_width=True):
    prompts = _parse_prompts(raw_prompts)
    if not prompts:
        st.warning("최소 1개 이상의 프롬프트를 입력해 주세요.")
        st.stop()
    if not numeric_ids:
        st.warning("현재 설정에서 숫자 subset 토큰이 없습니다.")
        st.stop()

    rows: list[dict] = []
    progress = st.progress(0.0)

    for idx, prompt in enumerate(prompts):
        try:
            input_ids = encode_prompt(tokenizer, prompt, device)
            attn_pre_last = _capture_attn_pre_last_token(model, input_ids, layer_idx)
            head_vec = _head_contribution_after_wo(model, layer_idx, head_idx, attn_pre_last)
            tokens, ids, logits, subset_probs, global_probs = _run_numeric_head_lens(
                model=model,
                tokenizer=tokenizer,
                head_vec=head_vec,
                topk=topk,
                apply_final_ln=apply_final_ln,
                numeric_ids=numeric_ids,
            )
            rows.append(
                {
                    "prompt": prompt,
                    "top1_numeric_token": tokens[0],
                    "top1_numeric_id": ids[0],
                    "topk_numeric_tokens": " | ".join(tokens),
                    "topk_numeric_ids": " | ".join(str(v) for v in ids),
                    "topk_logits": " | ".join(f"{v:.3f}" for v in logits),
                    "topk_subset_probs": " | ".join(f"{v:.3f}" for v in subset_probs),
                    "topk_global_probs": " | ".join(f"{v:.3f}" for v in global_probs),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "prompt": prompt,
                    "top1_numeric_token": f"[ERROR] {exc}",
                    "top1_numeric_id": "",
                    "topk_numeric_tokens": "",
                    "topk_numeric_ids": "",
                    "topk_logits": "",
                    "topk_subset_probs": "",
                    "topk_global_probs": "",
                }
            )
        progress.progress((idx + 1) / len(prompts))

    st.markdown(f"### Result: L{layer_idx}.H{head_idx} (numeric subset, n={len(prompts)})")
    st.dataframe(rows, use_container_width=True)
