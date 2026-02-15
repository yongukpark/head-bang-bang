import streamlit as st

from modules.common_heads import keep_only_selected_heads
from modules.common_inference import (
    build_head_labels,
    encode_prompt,
    forward_last_token,
    heads_by_layer,
    parse_head_labels,
    summarize_prediction,
)
from modules.common_model import get_device, get_selected_model_name, load_model
from modules.common_ui import (
    apply_base_theme,
    render_title,
    render_token_card,
    render_top5_cards,
    render_top5_diff_cards,
)

# =============================
# Page Config
# =============================
st.set_page_config(page_title="Interactive Head Addition Lab", layout="wide")
apply_base_theme()


# =============================
# Load Model
# =============================
device = get_device()
selected_model_name = get_selected_model_name()
model, tokenizer = load_model(selected_model_name, str(device))
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads

render_title("🧠 Interactive Head Keep-Only Lab")

head_labels = build_head_labels(n_layers, n_heads)
selected_heads = st.multiselect("Select Heads to Keep", options=head_labels)
selected_head_indices = parse_head_labels(selected_heads)
selected_heads_map = heads_by_layer(selected_head_indices, n_layers)
st.caption(f"Selected heads: {len(selected_head_indices)}")

prompt_text = st.text_area(
    "Enter multiple prompts (one per line)",
    """What is the capital of France? Answer:
What is the capital of Germany? Answer:""",
)

run = st.button("🚀 Run")


# =============================
# Keep-only Hook
# =============================
def keep_only_hook(head_indices: list[int]):
    """Pass only selected heads and block all others in the layer."""

    def hook(module, input):
        hidden = input[0]
        hidden = keep_only_selected_heads(hidden, head_indices, n_heads)
        return (hidden,)

    return hook


# =============================
# Run Logic
# =============================
if run:
    prompts = [p.strip() for p in prompt_text.split("\n") if p.strip()]

    if not selected_head_indices:
        st.warning("하나 이상의 헤드를 선택하세요.")
        st.stop()

    for prompt in prompts:
        st.markdown("---")
        st.markdown("### 🔹 Prompt")
        st.info(prompt)

        input_ids = encode_prompt(tokenizer, prompt, device)
        baseline_last, baseline_probs = forward_last_token(model, input_ids)
        baseline = summarize_prediction(tokenizer, baseline_last, baseline_probs)

        handles = []
        for layer in range(n_layers):
            layer_heads = selected_heads_map[layer]
            handle = model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(
                keep_only_hook(layer_heads)
            )
            handles.append(handle)

        modified_last, modified_probs = forward_last_token(model, input_ids)

        for handle in handles:
            handle.remove()

        kept_only = summarize_prediction(tokenizer, modified_last, modified_probs)
        top1_changed = kept_only.top1_id != baseline.top1_id

        st.markdown("### 📊 Result")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Baseline Top-1")
            render_token_card("Token", baseline.top1_token, f"{baseline.top1_prob:.2%}")

        with right:
            st.markdown("#### Baseline Top-5")
            render_top5_cards(baseline.top5_tokens, baseline.top5_probs)

        colb, cola = st.columns(2)
        with colb:
            st.markdown("#### Keep-Only Top-1")
            render_token_card(
                "Token",
                kept_only.top1_token,
                f"Δ {kept_only.top1_prob - baseline.top1_prob:+.2%}",
            )

        with cola:
            st.markdown("#### Keep-Only Top-5")
            render_top5_diff_cards(
                kept_only.top5_tokens,
                kept_only.top5_probs,
                baseline.top5_tokens,
                top1_changed,
            )
