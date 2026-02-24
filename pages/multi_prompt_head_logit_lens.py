from __future__ import annotations

import streamlit as st
import torch

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


def _run_head_lens(
    model,
    tokenizer,
    head_vec: torch.Tensor,
    topk: int,
    apply_final_ln: bool,
) -> tuple[list[str], list[float], list[float]]:
    final_ln = model.gpt_neox.final_layer_norm
    lm_head = model.embed_out if hasattr(model, "embed_out") else model.lm_head
    device = next(model.parameters()).device

    vec = head_vec.to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        if apply_final_ln:
            vec = final_ln(vec)
        logits = lm_head(vec).squeeze(0).squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_ids = torch.topk(logits, k=topk, dim=-1)
        top_probs = probs[top_ids]

    top_tokens = [visualize_token(tokenizer.decode([idx.item()])) for idx in top_ids]
    return top_tokens, top_vals.tolist(), top_probs.tolist()


st.set_page_config(page_title="Multi-Prompt Head Logit Lens", layout="wide")
apply_base_theme()
render_title("🧪 Multi-Prompt Head Logit Lens")

selected_model_name = get_selected_model_name()
device = get_device()

try:
    model, tokenizer = load_model(selected_model_name, device.type)
except RuntimeError as exc:
    st.error(f"모델 로딩 실패: {exc}")
    st.stop()

n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads

st.caption(f"모델: {selected_model_name}")
st.caption(f"레이어 수: {n_layers}, 레이어당 헤드 수: {n_heads}")

left, right = st.columns([1, 1])
with left:
    layer_idx = int(st.number_input("Layer", min_value=0, max_value=n_layers - 1, value=0, step=1))
    head_idx = int(st.number_input("Head", min_value=0, max_value=n_heads - 1, value=0, step=1))
    topk = int(st.slider("Top-k", min_value=1, max_value=20, value=10))
with right:
    apply_final_ln = st.toggle("Apply final layer norm before LM head", value=True)
    show_logits = st.toggle("Show logits/probabilities", value=True)

default_prompts = "\n".join(
    [
        "The capital of France is",
        "The opposite of hot is",
        "2 + 2 =",
    ]
)
raw_prompts = st.text_area(
    "Prompts (한 줄에 하나씩)",
    value=default_prompts,
    height=180,
    placeholder="여기에 여러 프롬프트를 줄바꿈으로 입력하세요.",
)

if st.button("Run Multi-Prompt Head Lens", use_container_width=True):
    prompts = _parse_prompts(raw_prompts)
    if not prompts:
        st.warning("최소 1개 이상의 프롬프트를 입력해 주세요.")
        st.stop()

    rows: list[dict] = []
    progress = st.progress(0.0)

    for idx, prompt in enumerate(prompts):
        try:
            input_ids = encode_prompt(tokenizer, prompt, device)
            attn_pre_last = _capture_attn_pre_last_token(model, input_ids, layer_idx)
            head_vec = _head_contribution_after_wo(model, layer_idx, head_idx, attn_pre_last)
            tokens, logits, probs = _run_head_lens(
                model=model,
                tokenizer=tokenizer,
                head_vec=head_vec,
                topk=topk,
                apply_final_ln=apply_final_ln,
            )
            row = {
                "prompt": prompt,
                "top1_token": tokens[0],
                "topk_tokens": " | ".join(tokens),
            }
            if show_logits:
                row["top1_logit"] = logits[0]
                row["top1_prob"] = probs[0]
                row["topk_logits"] = " | ".join(f"{v:.3f}" for v in logits)
                row["topk_probs"] = " | ".join(f"{v:.3f}" for v in probs)
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "prompt": prompt,
                    "top1_token": f"[ERROR] {exc}",
                    "topk_tokens": "",
                }
            )
        progress.progress((idx + 1) / len(prompts))

    st.markdown(f"### Result: L{layer_idx}.H{head_idx} (n={len(prompts)})")
    st.dataframe(rows, use_container_width=True)
