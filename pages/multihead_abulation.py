import streamlit as st
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================
# Page Config
# =============================
st.set_page_config(page_title="Interactive Head Disable Lab", layout="wide")

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

st.title("🧠 Interactive Head Disable Lab")

# =============================
# Head Selection
# =============================

head_labels = [
    f"L{idx // n_heads}H{idx % n_heads}"
    for idx in range(total_heads)
]

selected_heads = st.multiselect(
    "Select Heads to Disable",
    options=head_labels
)

# index 변환
selected_head_indices = []
for label in selected_heads:
    l = int(label.split("H")[0][1:])
    h = int(label.split("H")[1])
    selected_head_indices.append((l, h))

# =============================
# Prompt Input
# =============================
prompt_text = st.text_area(
    "Enter multiple prompts (one per line)",
    """What is the capital of France? Answer:
What is the capital of Germany? Answer:"""
)

run = st.button("🚀 Run")

# =============================
# Ablation Hook
# =============================
def multi_head_ablation(layer_idx, head_indices):
    def hook(module, input):
        hidden = input[0]
        b, s, d = hidden.shape
        head_dim = d // n_heads

        hidden = hidden.view(b, s, n_heads, head_dim)

        for h in head_indices:
            hidden[:, :, h, :] = 0.0

        hidden = hidden.view(b, s, d)
        return (hidden,)
    return hook

# =============================
# Run Logic
# =============================
if run:

    prompts = [p.strip() for p in prompt_text.split("\n") if p.strip()]

    for prompt in prompts:

        st.markdown("---")
        st.markdown(f"## 🔹 Prompt\n{prompt}")

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # -----------------
        # Baseline
        # -----------------
        with torch.no_grad():
            baseline_logits = model(input_ids).logits

        baseline_last = baseline_logits[0, -1]
        baseline_probs = torch.softmax(baseline_last, dim=-1)

        baseline_top1_id = torch.argmax(baseline_last).item()
        baseline_top1_token = tokenizer.decode([baseline_top1_id]).strip()
        baseline_conf = baseline_probs[baseline_top1_id].item()

        # -----------------
        # Ablation
        # -----------------
        handles = []

        # layer별로 head disable
        for l in range(n_layers):
            heads_in_layer = [
                h for (layer, h) in selected_head_indices if layer == l
            ]
            if len(heads_in_layer) > 0:
                handle = model.gpt_neox.layers[l].attention.dense.register_forward_pre_hook(
                    multi_head_ablation(l, heads_in_layer)
                )
                handles.append(handle)

        with torch.no_grad():
            ablated_logits = model(input_ids).logits

        # hook 제거
        for h in handles:
            h.remove()

        ablated_last = ablated_logits[0, -1]
        ablated_probs = torch.softmax(ablated_last, dim=-1)

        ablated_top1_id = torch.argmax(ablated_last).item()
        ablated_top1_token = tokenizer.decode([ablated_top1_id]).strip()
        ablated_conf = ablated_probs[ablated_top1_id].item()

        # =============================
        # 결과 출력
        # =============================
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟢 Baseline")
            st.metric(
                label="Top-1",
                value=baseline_top1_token,
                delta=f"{baseline_conf:.2%}"
            )

        with col2:
            st.markdown("### 🔴 After Ablation")
            st.metric(
                label="Top-1",
                value=ablated_top1_token,
                delta=f"{ablated_conf - baseline_conf:.2%}"
            )

        # Top-5 비교
        st.markdown("### 📊 Top-5 Comparison")

        base_top5_ids = torch.topk(baseline_last, 5).indices
        abl_top5_ids = torch.topk(ablated_last, 5).indices

        base_tokens = [tokenizer.decode([i]).strip() for i in base_top5_ids]
        abl_tokens = [tokenizer.decode([i]).strip() for i in abl_top5_ids]

        colb, cola = st.columns(2)

        with colb:
            st.write("Baseline Top-5")
            for t in base_tokens:
                st.write(t)

        with cola:
            st.write("Ablated Top-5")
            for t in abl_tokens:
                st.write(t)
