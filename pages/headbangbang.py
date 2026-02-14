import html

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from modules.common_heads import apply_single_head_scale
from modules.common_inference import encode_prompt, forward_last_token, summarize_prediction
from modules.common_model import get_device, get_selected_model_name, load_model
from modules.common_ui import apply_base_theme, render_title, visualize_token

# =============================
# Page Config
# =============================
st.set_page_config(page_title="Head Bang Bang", layout="wide")
apply_base_theme(top5_font_size=20)
st.markdown(
    """
<style>
div[data-testid="stAlert"] svg {
    display: none;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================
# Load Model
# =============================
device = get_device()
selected_model_name = get_selected_model_name()
model, tokenizer = load_model(selected_model_name, str(device))
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads


# =============================
# Shared Render Helpers
# =============================
def render_baseline_cards(baseline):
    st.markdown("### 🔍 Baseline Prediction")
    c1, c2 = st.columns(2)

    with c1:
        display_top1 = visualize_token(baseline.top1_token)
        st.markdown(
            f"<div class='card'><b>Top-1 Token</b><br>"
            f"<span style='font-size:22px;'>{html.escape(display_top1)}</span></div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"<div class='card'><b>Confidence</b><br>"
            f"<span style='font-size:22px;'>{baseline.top1_prob:.2%}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Top-5 Tokens")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            display_tok = visualize_token(baseline.top5_tokens[i])
            prob = baseline.top5_probs[i]
            st.markdown(
                f"""
                <div class='top5-card'>
                    {html.escape(display_tok)}<br>
                    <span style='font-size:14px; color:#aaaaaa;'>{prob:.2%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_details_panel(impact_data, baseline_top5_tokens):
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
    .tag-new { border:1px solid #00f2aa; background:#1a2e26; }
    ::-webkit-scrollbar { width:4px; }
    ::-webkit-scrollbar-thumb { background:#444; border-radius:10px; }
    </style>
    <div class="scroll-container">
    """

    for item in sorted(impact_data, key=lambda x: x["delta"]):
        d_class = "delta-neg" if item["delta"] < 0 else "delta-pos"
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
            elif tok not in baseline_top5_tokens:
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


# =============================
# Header
# =============================
render_title("🧠 Head Bang Bang")

with st.container():
    prompt = st.text_input(
        "Enter Prompt",
        "What is the capital of France? Answer:",
        help="모델이 다음 토큰을 예측할 문장을 입력하세요.",
    )
    run_button = st.button("🚀 Run Ablation Analysis")


# =============================
# Run Logic
# =============================
if run_button:
    input_ids = encode_prompt(tokenizer, prompt, device)
    baseline_last, baseline_probs = forward_last_token(model, input_ids)
    baseline = summarize_prediction(tokenizer, baseline_last, baseline_probs)

    render_baseline_cards(baseline)

    xs, ys, colors, sizes, symbols, hover_texts = [], [], [], [], [], []
    impact_data = []

    progress_bar = st.progress(0)
    total_steps = n_layers * n_heads
    step_count = 0

    # Evaluate every head independently by ablation and collect impact metrics.
    for layer in range(n_layers):
        for head in range(n_heads):
            def ablation_hook(module, input, current_head=head):
                hidden = input[0]
                hidden = apply_single_head_scale(hidden, current_head, n_heads, scale=0.0)
                return (hidden,)

            handle = model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(
                ablation_hook
            )

            ablated_last, ablated_probs = forward_last_token(model, input_ids)
            ablated = summarize_prediction(tokenizer, ablated_last, ablated_probs)

            top1_changed = ablated.top1_id != baseline.top1_id
            delta = ablated_probs[baseline.top1_id].item() - baseline.top1_prob

            xs.append(head)
            ys.append(layer)
            colors.append("#ff4d6d" if delta < 0 else "#00f2ff")
            sizes.append(8 + abs(delta) * 450)
            symbols.append("diamond" if top1_changed else "circle")
            hover_texts.append(f"L{layer} H{head}<br>Δ Prob: {delta:.4f}<br>Changed: {top1_changed}")

            impact_data.append(
                {
                    "layer": layer,
                    "head": head,
                    "delta": delta,
                    "changed": top1_changed,
                    "top5": ablated.top5_tokens,
                    "top5_probs": ablated.top5_probs,
                }
            )

            handle.remove()
            step_count += 1
            progress_bar.progress(step_count / total_steps)

    progress_bar.empty()

    left, right = st.columns([3, 0.9])

    with left:
        st.markdown("### 🗺️ Head Impact Map")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=colors,
                    symbol=symbols,
                    line=dict(width=1, color="rgba(255,255,255,0.3)"),
                ),
                text=hover_texts,
                hoverinfo="text",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#11141c",
            paper_bgcolor="#0f1117",
            xaxis=dict(title="Head"),
            yaxis=dict(title="Layer", autorange="reversed"),
            height=700,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        render_details_panel(impact_data, baseline.top5_tokens)

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)
