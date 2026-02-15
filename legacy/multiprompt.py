import numpy as np
import plotly.graph_objects as go
import streamlit as st

from modules.common_heads import apply_single_head_scale
from modules.common_inference import (
    build_head_labels,
    encode_prompt,
    forward_last_token,
    summarize_prediction,
)
from modules.common_model import get_device, get_selected_model_name, load_model
from modules.common_ui import apply_base_theme, render_title, render_token_card

# =============================
# Page Config
# =============================
st.set_page_config(page_title="Multi Prompt Rank Heatmap", layout="wide")
apply_base_theme(top5_font_size=16)


# =============================
# Load Model
# =============================
device = get_device()
selected_model_name = get_selected_model_name()
model, tokenizer = load_model(selected_model_name, str(device))
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
total_heads = n_layers * n_heads

render_title("🔥 Multi-Prompt Head Ranking Heatmap")

prompt_text = st.text_area(
    "Enter multiple prompts (one per line)",
    """What is the capital of France? Answer:
What is the capital of Germany? Answer:
What is the capital of Korea? Answer:
What is the capital of China? Answer:
What is the capital of United State? Answer:""",
)

run = st.button("🚀 Run Analysis")


# =============================
# Ablation Hook
# =============================
def ablation_hook(head_idx: int):
    """Disable one head for the current layer."""

    def hook(module, input):
        hidden = input[0]
        hidden = apply_single_head_scale(hidden, head_idx, n_heads, scale=0.0)
        return (hidden,)

    return hook


# =============================
# Run Logic
# =============================
if run:
    prompts = [p.strip() for p in prompt_text.split("\n") if p.strip()]
    num_prompts = len(prompts)

    if num_prompts == 0:
        st.warning("프롬프트를 한 줄 이상 입력하세요.")
        st.stop()

    head_importance = []
    baseline_answers = []
    baseline_confidences = []

    progress = st.progress(0)
    total_steps = num_prompts * total_heads
    step_count = 0

    for prompt in prompts:
        input_ids = encode_prompt(tokenizer, prompt, device)
        baseline_last, baseline_probs = forward_last_token(model, input_ids)
        baseline = summarize_prediction(tokenizer, baseline_last, baseline_probs)

        baseline_answers.append(baseline.top1_token)
        baseline_confidences.append(baseline.top1_prob)

        prompt_scores = []
        for layer in range(n_layers):
            for head in range(n_heads):
                handle = model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(
                    ablation_hook(head)
                )

                ablated_last, ablated_probs = forward_last_token(model, input_ids)
                delta = ablated_probs[baseline.top1_id].item() - baseline.top1_prob
                prompt_scores.append(delta)

                handle.remove()

                step_count += 1
                progress.progress(step_count / total_steps)

        head_importance.append(prompt_scores)

    progress.empty()

    rank_matrix = np.array([np.argsort(np.argsort(scores)) for scores in head_importance])
    heatmap_data = rank_matrix.T

    st.markdown("### 📌 Baseline Predictions")
    cols = st.columns(num_prompts)
    for i, col in enumerate(cols):
        with col:
            render_token_card(f"P{i + 1}", baseline_answers[i], f"{baseline_confidences[i]:.2%}")

    threshold = 10
    display_matrix = np.where(heatmap_data < threshold, heatmap_data + 1, np.nan)

    fig = go.Figure(
        data=go.Heatmap(
            z=display_matrix,
            colorscale="Reds_R",
            zmin=1,
            zmax=threshold,
            reversescale=False,
            showscale=False,
            xgap=2,
            ygap=2,
            hoverongaps=False,
        )
    )

    fig.update_layout(
        height=total_heads * 12,
        plot_bgcolor="#11141c",
        paper_bgcolor="#0f1117",
        font=dict(color="#e6e6e6"),
        margin=dict(l=80, r=20, t=40, b=60),
    )

    fig.update_xaxes(
        title_text="Prompt",
        tickmode="array",
        tickvals=list(range(num_prompts)),
        ticktext=[f"P{i+1}" for i in range(num_prompts)],
        gridcolor="rgba(255,255,255,0.08)",
    )

    fig.update_yaxes(
        title_text="Attention Head (Layer-Head)",
        tickmode="array",
        tickvals=list(range(total_heads)),
        ticktext=build_head_labels(n_layers, n_heads),
        automargin=True,
        gridcolor="rgba(255,255,255,0.08)",
    )

    st.plotly_chart(fig, use_container_width=True)
