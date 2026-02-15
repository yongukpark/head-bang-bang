import streamlit as st

from modules.common_heads import replace_selected_heads_from_donor
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

MAX_PROMPTS = 12


def _render_prompt_inputs(
    first_label: str,
    first_default: str,
    item_label_prefix: str,
    second_default: str,
    count: int,
) -> list[str]:
    first_value = st.text_area(first_label, first_default)
    values = [first_value]
    for idx in range(2, count + 1):
        values.append(
            st.text_input(
                f"{item_label_prefix} {idx}",
                value=second_default if idx == 2 else "",
            )
        )
    return values


def _normalize_non_empty(items: list[str]) -> list[str]:
    return [item.strip() for item in items if item.strip()]


# =============================
# Page Config
# =============================
st.set_page_config(page_title="Interactive Multi-Head Resampling Lab", layout="wide")
apply_base_theme()


# =============================
# Load Model
# =============================
device = get_device()
selected_model_name = get_selected_model_name()
model, tokenizer = load_model(selected_model_name, str(device))
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads

render_title("🧠 Interactive Multi-Head Resampling Lab")

head_labels = build_head_labels(n_layers, n_heads)
selected_heads = st.multiselect("Select Heads to Resample", options=head_labels)
selected_head_indices = parse_head_labels(selected_heads)
selected_heads_map = heads_by_layer(selected_head_indices, n_layers)
st.caption(f"Selected heads: {len(selected_head_indices)}")
st.caption("Resampling target: A*V values from donor prompt (selected heads).")

num_prompts = int(st.number_input("Number of prompts", min_value=1, max_value=MAX_PROMPTS, value=2, step=1))
target_prompt_inputs = _render_prompt_inputs(
    first_label="Enter Prompt 1",
    first_default="What is the capital of France? Answer:",
    item_label_prefix="Enter Prompt",
    second_default="What is the capital of Germany? Answer:",
    count=num_prompts,
)
donor_prompt_inputs = _render_prompt_inputs(
    first_label="Enter Donor Prompt 1",
    first_default="What is the capital of Germany? Answer:",
    item_label_prefix="Enter Donor Prompt",
    second_default="What is the capital of France? Answer:",
    count=num_prompts,
)

run = st.button("🚀 Run")


def _capture_donor_av(model, donor_ids, n_layers):
    donor_av_by_layer = {}
    handles = []

    def build_capture_hook(layer_idx):
        def capture_hook(module, input):
            donor_av_by_layer[layer_idx] = input[0].detach().clone()
            return input

        return capture_hook

    for layer in range(n_layers):
        handle = model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(
            build_capture_hook(layer)
        )
        handles.append(handle)

    _ = forward_last_token(model, donor_ids)
    for handle in handles:
        handle.remove()

    return donor_av_by_layer


if run:
    prompts = _normalize_non_empty(target_prompt_inputs)
    donor_prompts = _normalize_non_empty(donor_prompt_inputs)

    if not selected_head_indices:
        st.warning("하나 이상의 헤드를 선택하세요.")
        st.stop()
    if not prompts:
        st.warning("프롬프트를 하나 이상 입력하세요.")
        st.stop()
    if len(donor_prompts) != len(prompts):
        st.warning("Donor 프롬프트 개수는 Target 프롬프트 개수와 같아야 합니다.")
        st.stop()

    for idx, prompt in enumerate(prompts):
        st.markdown("---")
        st.markdown("### 🔹 Prompt")
        st.info(prompt)
        st.caption(f"Donor: {donor_prompts[idx]}")

        input_ids = encode_prompt(tokenizer, prompt, device)
        baseline_last, baseline_probs = forward_last_token(model, input_ids)
        baseline = summarize_prediction(tokenizer, baseline_last, baseline_probs)
        donor_ids = encode_prompt(tokenizer, donor_prompts[idx], device)
        donor_av_by_layer = _capture_donor_av(model, donor_ids, n_layers)

        handles = []
        for layer in range(n_layers):
            layer_heads = selected_heads_map[layer]
            if layer_heads:
                donor_av = donor_av_by_layer[layer]

                def donor_swap_hook(module, input, current_heads=layer_heads, dh=donor_av):
                    hidden = input[0]
                    hidden = replace_selected_heads_from_donor(hidden, dh, current_heads, n_heads)
                    return (hidden,)

                handle = model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(
                    donor_swap_hook
                )
                handles.append(handle)

        modified_last, modified_probs = forward_last_token(model, input_ids)

        for handle in handles:
            handle.remove()

        resampled = summarize_prediction(tokenizer, modified_last, modified_probs)
        top1_changed = resampled.top1_id != baseline.top1_id

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
            st.markdown("#### Resampled Top-1")
            render_token_card(
                "Token",
                resampled.top1_token,
                f"Δ {resampled.top1_prob - baseline.top1_prob:+.2%}",
            )

        with cola:
            st.markdown("#### Resampled Top-5")
            render_top5_diff_cards(
                resampled.top5_tokens,
                resampled.top5_probs,
                baseline.top5_tokens,
                top1_changed,
            )
