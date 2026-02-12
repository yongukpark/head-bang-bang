import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit.components.v1 as components
import html

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Head Bang Bang",
    layout="wide"
)

# =============================
# Custom CSS
# =============================
st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}

.stApp {
    background: #0f1117;
    color: white;
}

h1 {
    text-align:center;
    font-size:2.6rem;
    background: linear-gradient(90deg,#ff00cc,#7928ca,#00f2ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-bottom: 20px;
}

.card {
    background: #1a1d25;
    padding: 14px;
    border-radius: 12px;
    margin-bottom:10px;
    border:1px solid rgba(255,255,255,0.05);
    text-align: center;
    font-size: 20px;
    white-space: pre;
}

.top5-card {
    background:#20242f;
    padding:8px;
    border-radius:8px;
    text-align:center;
    font-weight: bold;
    font-size: 20px;
    white-space: pre;
}

div[data-testid="stButton"] > button {
    width: 100%;
    color: black;
    font-weight: 700;
    border-radius: 50px;
    height: 46px;
    background: white;
    transition: all 0.2s ease;
}

div[data-testid="stButton"] > button:hover {
    background: linear-gradient(90deg,#ff00cc,#7928ca);
    color: white;
    transform: scale(1.02);
}
div[data-testid="stAlert"] {
    background-color: #151821 !important;
    border: 1px solid #2f3542 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 13px !important;
}

div[data-testid="stAlert"] svg {
    display: none; 
}

div[data-testid="stAlert"] p {
    color: #e6e6e6 !important;
}
</style>
""", unsafe_allow_html=True)

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

def visualize_token(tok):
    # 모든 whitespace를 ␣ 로 표시
    return "".join("␣" if c.isspace() else c for c in tok)

model, tokenizer = load_model()
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads

# =============================
# Header
# =============================
st.markdown("<h1>🧠 Head Bang Bang</h1>", unsafe_allow_html=True)

with st.container():
    prompt = st.text_input(
        "Enter Prompt",
        "What is the capital of France? Answer:",
        help="모델이 다음 토큰을 예측할 문장을 입력하세요."
    )
    run_button = st.button("🚀 Run Ablation Analysis")

# =============================
# Run Logic
# =============================
if run_button:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        baseline_logits = model(input_ids).logits

    last_pos = -1
    baseline_last = baseline_logits[0, last_pos]
    baseline_probs = torch.softmax(baseline_last, dim=-1)

    baseline_topk_vals, baseline_topk_ids = torch.topk(baseline_last, 5)

    baseline_topk_tokens = []
    baseline_topk_probs = []

    for idx in baseline_topk_ids:
        token = tokenizer.decode([idx])
        prob = baseline_probs[idx].item()
        baseline_topk_tokens.append(token)
        baseline_topk_probs.append(prob)

    baseline_top1_id = baseline_topk_ids[0].item()
    baseline_top1_token = baseline_topk_tokens[0]
    baseline_top1_prob = baseline_probs[baseline_top1_id].item()

    # =============================
    # Baseline UI
    # =============================
    st.markdown("### 🔍 Baseline Prediction")
    c1, c2 = st.columns(2)

    with c1:
        display_top1 = visualize_token(baseline_top1_token)
        st.markdown(
            f"<div class='card'><b>Top-1 Token</b><br>"
            f"<span style='font-size:22px;'>{html.escape(display_top1)}</span></div>",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"<div class='card'><b>Confidence</b><br>"
            f"<span style='font-size:22px;'>{baseline_top1_prob:.2%}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("#### Top-5 Tokens")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            display_tok = visualize_token(baseline_topk_tokens[i])
            prob = baseline_topk_probs[i]

            st.markdown(
                f"""
                <div class='top5-card'>
                    {html.escape(display_tok)}<br>
                    <span style='font-size:14px; color:#aaaaaa;'>{prob:.2%}</span>
                </div>
                """,
                unsafe_allow_html=True
            )


    # =============================
    # Ablation
    # =============================
    xs, ys, colors, sizes, symbols, hover_texts = [], [], [], [], [], []
    impact_data = []

    progress_bar = st.progress(0)
    total_steps = n_layers * n_heads
    step_count = 0

    def ablation_hook(layer_idx, head_idx):
        def hook(module, input):
            # input is tuple: (hidden_states,)
            hidden = input[0]  # (B, S, D)
            b, s, d = hidden.shape
            head_dim = d // n_heads

            hidden = hidden.view(b, s, n_heads, head_dim)
            hidden[:, :, head_idx, :] = 0.0
            hidden = hidden.view(b, s, d)

            return (hidden,)
        return hook

    for l in range(n_layers):
        for h in range(n_heads):
            handle = model.gpt_neox.layers[l].attention.dense.register_forward_pre_hook(
                ablation_hook(l, h)
            )

            with torch.no_grad():
                ablated_logits = model(input_ids).logits

            ablated_last = ablated_logits[0, last_pos]
            ablated_probs = torch.softmax(ablated_last, dim=-1)

            ablated_topk_vals, ablated_topk_ids = torch.topk(ablated_last, 5)

            ablated_topk_tokens = []
            ablated_topk_probs = []

            for idx in ablated_topk_ids:
                token = tokenizer.decode([idx])
                prob = ablated_probs[idx].item()
                ablated_topk_tokens.append(token)
                ablated_topk_probs.append(prob)

            ablated_top1_id = ablated_topk_ids[0].item()
            top1_changed = ablated_top1_id != baseline_top1_id
            delta = ablated_probs[baseline_top1_id].item() - baseline_top1_prob

            xs.append(h)
            ys.append(l)
            colors.append("#ff4d6d" if delta < 0 else "#00f2ff")
            sizes.append(8 + abs(delta) * 450)
            symbols.append("diamond" if top1_changed else "circle")
            hover_texts.append(
                f"L{l} H{h}<br>Δ Prob: {delta:.4f}<br>Changed: {top1_changed}"
            )

            impact_data.append({
                "layer": l,
                "head": h,
                "delta": delta,
                "changed": top1_changed,
                "top5": ablated_topk_tokens,
                "top5_probs": ablated_topk_probs
            })


            handle.remove()
            step_count += 1
            progress_bar.progress(step_count / total_steps)

    progress_bar.empty()

    # =============================
    # Layout
    # =============================
    left, right = st.columns([3, 0.9])

    with left:
        st.markdown("### 🗺️ Head Impact Map")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                symbol=symbols,
                line=dict(width=1, color="rgba(255,255,255,0.3)")
            ),
            text=hover_texts,
            hoverinfo="text"
        ))

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#11141c",
            paper_bgcolor="#0f1117",
            xaxis=dict(title="Head"),
            yaxis=dict(title="Layer", autorange="reversed"),
            height=700,
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### 📋 Detail")

        html_content = """
        <style>
        body { font-family: sans-serif; background: transparent; margin:0; padding:0; }
        .scroll-container { height:640px; overflow-y:auto; }
        .head-card {
            background:#171a21;
            border-radius:6px;
            padding:8px 10px;
            margin-bottom:6px;
            font-size:16px;
            line-height:1.2;
        }
        .header-line {
            display:flex;
            justify-content:space-between;
            margin-bottom:4px;
            color: white;
        }
        .delta-pos { color:#00f2ff; font-weight:600; }
        .delta-neg { color:#ff4d6d; font-weight:600; }
        .token-tag {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 40px;
            padding: 4px 8px;
            font-size: 15px;
            color: white;
            border-radius:4px;
            background:#242833;
            white-space: pre;
            font-family: monospace;
        }

        .tag-changed { border:1px solid #ff4d6d; background:#321a1a; }
        .tag-new { border:1px solid #00f2aa; background:#1a2e26;}
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-thumb { background:#444; border-radius:10px; }
        </style>
        <div class="scroll-container">
        """

        for item in impact_data:
            d_class = "delta-neg" if item['delta'] < 0 else "delta-pos"

            html_content += f"""
            <div class="head-card">
                <div class="header-line">
                    <span>L{item['layer']} · H{item['head']}</span>
                    <span class="{d_class}">Δ {item['delta']:.4f}</span>
                </div>
            """

            for i, tok in enumerate(item["top5"]):
                prob = item["top5_probs"][i]
                display_tok = "".join("␣" if c.isspace() else c for c in tok)
                safe_tok = html.escape(display_tok)
                status_class = ""
                if item["changed"] and i == 0:
                    status_class = "tag-changed"
                elif tok not in baseline_topk_tokens:
                    status_class = "tag-new"

                html_content += f'''
                <div style="display:inline-block; margin-right:6px; text-align:center;">
                    <span class="token-tag {status_class}">{safe_tok}</span><br>
                    <span style="font-size:12px; color:#aaaaaa;">{prob:.2%}</span>
                </div>
                '''


            html_content += "</div>"

        html_content += "</div>"

        components.html(html_content, height=650)

st.markdown("""
<div style="
background:#151821;
border:1px solid #2f3542;
border-radius:10px;
padding:14px 18px;
font-size:15px;
color:#e6e6e6;
margin-bottom:12px;
line-height:1.5;
">

<b style="font-size:18px;">🗺️ Head Impact Map</b><br><br>

- Each dot represents an attention head<br>
- Dot size = magnitude of Δ probability 
    - Blue = probability increase 
    - Red = probability decrease<br>
- ♦ indicates the Top-1 prediction changed from baseline

</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="
background:#151821;
border:1px solid #2f3542;
border-radius:10px;
padding:14px 18px;
font-size:15px;
color:#e6e6e6;
margin-bottom:12px;
line-height:1.5;
">

<b style="font-size:18px;">📋 Detail</b><br><br>

- 🔴 Red border = Replaced Top-1 token<br>
- 🟢 Green border = Newly entered Top-5 token

</div>
""", unsafe_allow_html=True)
