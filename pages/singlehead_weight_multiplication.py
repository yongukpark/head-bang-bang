import streamlit as st
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================
# Page Config
# =============================
st.set_page_config(page_title="Independent Head Control Lab", layout="wide")

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

st.title("🧠 Independent Multi-Head Control Lab")

# =============================
# Prompt
# =============================
prompt = st.text_input(
    "Enter Prompt",
    "What is the capital of France? Answer:"
)

# =============================
# Head Selection UI
# =============================
st.markdown("## 🎛 Select Heads & Set Alpha")

num_controls = st.number_input("Number of Heads to Control", 1, 10, 1)

head_controls = []

for i in range(num_controls):
    st.markdown(f"### Head Control #{i+1}")
    col1, col2, col3 = st.columns(3)

    with col1:
        layer = st.selectbox(
            f"Layer {i+1}",
            list(range(n_layers)),
            key=f"layer_{i}"
        )

    with col2:
        head = st.selectbox(
            f"Head {i+1}",
            list(range(n_heads)),
            key=f"head_{i}"
        )

    with col3:
        alpha = st.slider(
            f"Alpha {i+1}",
            0.0, 10.0, 1.0, 0.5,
            key=f"alpha_{i}"
        )

    head_controls.append((layer, head, alpha))

run = st.button("🚀 Run Experiment")

# =============================
# Multi Head Hook
# =============================
def multi_head_hook(control_list):
    def hook(module, input):
        hidden = input[0]  # (B, S, D)
        b, s, d = hidden.shape
        head_dim = d // n_heads

        hidden = hidden.view(b, s, n_heads, head_dim)
        hidden = hidden.clone()

        for layer_idx, head_idx, alpha in control_list:
            hidden[:, :, head_idx, :] *= alpha

        hidden = hidden.view(b, s, d)
        return (hidden,)
    return hook

# =============================
# Run Logic
# =============================
if run:

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # =============================
    # Baseline
    # =============================
    with torch.no_grad():
        baseline_logits = model(input_ids).logits

    baseline_last = baseline_logits[0, -1]
    baseline_probs = torch.softmax(baseline_last, dim=-1)

    baseline_topk_vals, baseline_topk_ids = torch.topk(baseline_last, 5)

    st.markdown("## 🟢 Baseline Prediction")
    for i in range(5):
        tok = tokenizer.decode([baseline_topk_ids[i]])
        prob = baseline_probs[baseline_topk_ids[i]].item()
        st.write(f"{i+1}. {tok} — {prob:.2%}")

    baseline_top1_id = baseline_topk_ids[0]
    baseline_top1_token = tokenizer.decode([baseline_top1_id])

    # =============================
    # Apply Multi Head Control
    # =============================
    handles = []

    for layer_idx, head_idx, alpha in head_controls:
        handle = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(
            multi_head_hook([(layer_idx, head_idx, alpha)])
        )
        handles.append(handle)

    with torch.no_grad():
        modified_logits = model(input_ids).logits

    for h in handles:
        h.remove()

    modified_last = modified_logits[0, -1]
    modified_probs = torch.softmax(modified_last, dim=-1)
    modified_topk_vals, modified_topk_ids = torch.topk(modified_last, 5)

    st.markdown("## 🔴 Modified Prediction")

    for i in range(5):
        tok = tokenizer.decode([modified_topk_ids[i]])
        prob = modified_probs[modified_topk_ids[i]].item()
        st.write(f"{i+1}. {tok} — {prob:.2%}")

    modified_top1_token = tokenizer.decode([modified_topk_ids[0]])

    # =============================
    # Summary
    # =============================
    st.markdown("## ⚡ Effect Summary")

    delta = modified_probs[baseline_top1_id].item() - baseline_probs[baseline_top1_id].item()

    st.write(f"Baseline Top-1: **{baseline_top1_token}**")
    st.write(f"Modified Top-1: **{modified_top1_token}**")
    st.write(f"Δ Probability of Baseline Token: **{delta:.4f}**")

    st.markdown("### 🎛 Applied Head Modifications")
    for layer_idx, head_idx, alpha in head_controls:
        st.write(f"L{layer_idx} H{head_idx} → α = {alpha}")
