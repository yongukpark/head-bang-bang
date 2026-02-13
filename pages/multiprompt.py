import streamlit as st
import torch
import numpy as np
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================
# Page Config
# =============================
st.set_page_config(page_title="Multi Prompt Rank Heatmap", layout="wide")

# =============================
# Load Model
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model_name = "EleutherAI/pythia-410m"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
total_heads = n_layers * n_heads

st.title("🔥 Multi-Prompt Head Ranking Heatmap")

# =============================
# Prompt Input
# =============================
prompt_text = st.text_area(
    "Enter multiple prompts (one per line)",
    """What is the capital of France? Answer:
What is the capital of Germany? Answer:
What is the capital of Korea? Answer:
What is the capital of China? Answer:
What is the capital of United State? Answer:"""
)

run = st.button("🚀 Run Analysis")

# =============================
# Ablation Hook
# =============================
def ablation_hook(head_idx):
    def hook(module, input):
        hidden = input[0]
        b, s, d = hidden.shape
        head_dim = d // n_heads
        hidden = hidden.view(b, s, n_heads, head_dim)
        hidden[:, :, head_idx, :] = 0.0
        hidden = hidden.view(b, s, d)
        return (hidden,)
    return hook

# =============================
# Run Logic
# =============================
if run:
    prompts = [p.strip() for p in prompt_text.split("\n") if p.strip()]
    num_prompts = len(prompts)

    head_importance = []
    baseline_answers = []
    baseline_confidences = []

    progress = st.progress(0)
    total_steps = num_prompts * total_heads
    step_count = 0

    for prompt in prompts:

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            baseline_logits = model(input_ids).logits

        baseline_last = baseline_logits[0, -1]
        baseline_probs = torch.softmax(baseline_last, dim=-1)

        baseline_top1_id = torch.argmax(baseline_last).item()
        baseline_top1_prob = baseline_probs[baseline_top1_id].item()
        
        baseline_token = tokenizer.decode([baseline_top1_id]).strip()

        baseline_answers.append(baseline_token)
        baseline_confidences.append(baseline_top1_prob)
        
        prompt_scores = []

        for l in range(n_layers):
            for h in range(n_heads):

                handle = model.gpt_neox.layers[l].attention.dense.register_forward_pre_hook(
                    ablation_hook(h)
                )

                with torch.no_grad():
                    ablated_logits = model(input_ids).logits

                ablated_last = ablated_logits[0, -1]
                ablated_probs = torch.softmax(ablated_last, dim=-1)

                delta = ablated_probs[baseline_top1_id].item() - baseline_top1_prob
                prompt_scores.append(delta)

                handle.remove()

                step_count += 1
                progress.progress(step_count / total_steps)

        head_importance.append(prompt_scores)

    progress.empty()

    head_importance = np.array(head_importance)  # (num_prompts, total_heads)

    # =============================
    # Rank 계산
    # =============================
    rank_matrix = []

    for scores in head_importance:
        ranks = np.argsort(np.argsort(scores))
        rank_matrix.append(ranks)

    rank_matrix = np.array(rank_matrix)  # (num_prompts, total_heads)
    heatmap_data = rank_matrix.T  # (total_heads, num_prompts)

    # =============================
    # Baseline 출력
    # =============================
    st.markdown("### 📌 Baseline Predictions")

    cols = st.columns(num_prompts)
    for i in range(num_prompts):
        with cols[i]:
            st.metric(
                label=f"P{i+1}",
                value=baseline_answers[i],
                delta=f"{baseline_confidences[i]:.2%}"
            )

    # =============================
    # Plot
    # =============================
    import plotly.graph_objects as go

    threshold = 10

    display_matrix = np.where(
        heatmap_data < threshold,  # 🔥 <= 가 아니라 <
        heatmap_data+1,
        np.nan
    )

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
            hoverongaps=False
        )
    )

    fig.update_layout(
        height=total_heads * 12,
        width=1400,
        plot_bgcolor="#e5e5e5",      # 🔥 검은 배경 제거
        paper_bgcolor="white",
        margin=dict(l=80, r=20, t=40, b=60),
    )

    fig.update_xaxes(
        title_text="Prompt",
        tickmode="array",
        tickvals=list(range(num_prompts)),
        ticktext=[f"P{i+1}" for i in range(num_prompts)]
    )

    head_labels = [
        f"L{idx // n_heads}H{idx % n_heads}"
        for idx in range(total_heads)
    ]

    fig.update_yaxes(
        title_text="Attention Head (Layer-Head)",
        tickmode="array",
        tickvals=list(range(total_heads)),
        ticktext=head_labels,
        automargin=True
    )

    st.plotly_chart(fig, use_container_width=True)
