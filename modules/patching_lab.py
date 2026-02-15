import html

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch

from modules.common_inference import encode_prompt, forward_last_token, summarize_prediction
from modules.common_model import get_device, get_selected_model_name, load_model
from modules.common_ui import apply_base_theme, render_title, visualize_token


MODE_META = {
    "v_only": {
        "title": "🧠 Value-Only Patching Lab",
        "button": "🚀 Run Value-Only Patching",
        "map_title": "### 🗺️ Value-Only Patching Impact Map",
        "help_line": "Each run replaces only V on full prefix (min_len tokens).",
    },
    "qk_only": {
        "title": "🧠 Q/K-Only Patching Lab",
        "button": "🚀 Run Q/K-Only Patching",
        "map_title": "### 🗺️ Q/K-Only Patching Impact Map",
        "help_line": "Each run replaces only Q/K on full prefix (min_len tokens).",
    },
    "qkv": {
        "title": "🧠 Q/K/V Patching Lab",
        "button": "🚀 Run Q/K/V Patching",
        "map_title": "### 🗺️ Q/K/V Patching Impact Map",
        "help_line": "Each run replaces Q/K/V on full prefix (min_len tokens).",
    },
    "av": {
        "title": "🧠 A*V Patching Lab",
        "button": "🚀 Run A*V Patching",
        "map_title": "### 🗺️ A*V Patching Impact Map",
        "help_line": "Each run replaces one head's A*V vector before W_O on full prefix.",
    },
    "avo": {
        "title": "🧠 A*V*W_O Patching Lab",
        "button": "🚀 Run A*V*W_O Patching",
        "map_title": "### 🗺️ A*V*W_O Patching Impact Map",
        "help_line": "Each run swaps one head's post-W_O contribution on full prefix.",
    },
}


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
            safe_tok = html.escape("".join("␣" if c.isspace() else c for c in tok))
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


def render_aggregate_details_panel(impact_data, pair_count):
    st.markdown("### 📋 Aggregate Detail")
    rows = []
    for item in sorted(impact_data, key=lambda x: x["delta"]):
        rows.append(
            {
                "Layer": item["layer"],
                "Head": item["head"],
                "Avg ΔProb": round(item["delta"], 6),
                "Changed %": round(item["changed_rate"] * 100, 2),
            }
        )
    st.dataframe(rows, use_container_width=True, height=650)
    st.caption(f"Aggregated across {pair_count} target/donor pair(s).")


def render_impact_scatter(xs, ys, sizes, colors, symbols, hover_texts):
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


def _capture_donor_qkv(model, donor_ids, n_layers):
    donor_qkv_by_layer = {}
    handles = []

    def build_capture_hook(layer_idx):
        def capture_hook(module, input, output):
            donor_qkv_by_layer[layer_idx] = output.detach().clone()
            return output

        return capture_hook

    for layer in range(n_layers):
        handle = model.gpt_neox.layers[layer].attention.query_key_value.register_forward_hook(
            build_capture_hook(layer)
        )
        handles.append(handle)
    _ = forward_last_token(model, donor_ids)
    for handle in handles:
        handle.remove()
    return donor_qkv_by_layer


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


def run_patching_page(mode: str):
    if mode not in MODE_META:
        st.error(f"Unsupported mode: {mode}")
        return

    meta = MODE_META[mode]
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

    device = get_device()
    selected_model_name = get_selected_model_name()
    model, tokenizer = load_model(selected_model_name, str(device))
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    render_title(meta["title"])

    with st.container():
        multi_mode = False
        if mode == "avo":
            multi_mode = st.toggle("Multi-input mode", value=False)

        if mode == "avo" and multi_mode:
            target_prompts_text = st.text_area(
                "Target Prompts (one per line)",
                """What is the capital of France? Answer:
What is the capital of Germany? Answer:
What is the capital of Korea? Answer:""",
            )
            donor_prompts_text = st.text_area(
                "Donor Prompts (one per line, same count)",
                """What is the capital of Germany? Answer:
What is the capital of France? Answer:
What is the capital of Japan? Answer:""",
            )
        else:
            target_prompt = st.text_input("Target Prompt", "What is the capital of France? Answer:")
            donor_prompt = st.text_input("Donor Prompt", "What is the capital of Germany? Answer:")

        run_button = st.button(meta["button"])

    if not run_button:
        return

    if mode == "avo" and multi_mode:
        target_prompts = [p.strip() for p in target_prompts_text.splitlines() if p.strip()]
        donor_prompts = [p.strip() for p in donor_prompts_text.splitlines() if p.strip()]

        if not target_prompts or not donor_prompts:
            st.warning("Target/Donor 프롬프트를 한 줄 이상 입력하세요.")
            return
        if len(target_prompts) != len(donor_prompts):
            st.warning("Target 줄 수와 Donor 줄 수를 동일하게 맞춰주세요.")
            return

        pair_count = len(target_prompts)
        aggregate = {}
        baseline_summaries = []
        per_pair_results = []

        progress_bar = st.progress(0)
        total_steps = pair_count * n_layers * n_heads
        step_count = 0

        for pair_idx, (target_prompt_i, donor_prompt_i) in enumerate(
            zip(target_prompts, donor_prompts), start=1
        ):
            target_ids = encode_prompt(tokenizer, target_prompt_i, device)
            donor_ids = encode_prompt(tokenizer, donor_prompt_i, device)

            baseline_last, baseline_probs = forward_last_token(model, target_ids)
            baseline = summarize_prediction(tokenizer, baseline_last, baseline_probs)
            baseline_summaries.append(
                {
                    "pair": pair_idx,
                    "top1": baseline.top1_token,
                    "prob": baseline.top1_prob,
                    "top5": baseline.top5_tokens,
                    "target": target_prompt_i,
                    "donor": donor_prompt_i,
                }
            )

            donor_av_by_layer = _capture_donor_av(model, donor_ids, n_layers)
            pair_xs, pair_ys, pair_colors, pair_sizes, pair_symbols, pair_hover_texts = [], [], [], [], [], []
            pair_impact_data = []

            for layer in range(n_layers):
                donor_av = donor_av_by_layer[layer]
                for head in range(n_heads):

                    def avo_hook(module, input, output, dh=donor_av, current_head=head):
                        hidden = input[0]
                        patch_len = min(hidden.shape[1], dh.shape[1])
                        hidden_dim = hidden.shape[-1]
                        head_dim = hidden_dim // n_heads
                        start = current_head * head_dim
                        end = start + head_dim
                        weight_slice = module.weight[:, start:end]
                        target_part = hidden[:, :patch_len, start:end]
                        donor_part = dh[:, :patch_len, start:end]
                        target_contrib = torch.matmul(target_part, weight_slice.t())
                        donor_contrib = torch.matmul(donor_part, weight_slice.t())
                        patched = output.clone()
                        patched[:, :patch_len, :] = patched[:, :patch_len, :] + donor_contrib - target_contrib
                        return patched

                    handle = model.gpt_neox.layers[layer].attention.dense.register_forward_hook(avo_hook)
                    patched_last, patched_probs = forward_last_token(model, target_ids)
                    patched = summarize_prediction(tokenizer, patched_last, patched_probs)
                    handle.remove()

                    top1_changed = patched.top1_id != baseline.top1_id
                    delta = patched_probs[baseline.top1_id].item() - baseline.top1_prob
                    key = (layer, head)
                    if key not in aggregate:
                        aggregate[key] = {"delta_sum": 0.0, "changed_count": 0}
                    aggregate[key]["delta_sum"] += delta
                    aggregate[key]["changed_count"] += 1 if top1_changed else 0
                    pair_xs.append(head)
                    pair_ys.append(layer)
                    pair_colors.append("#ff4d6d" if delta < 0 else "#00f2ff")
                    pair_sizes.append(8 + abs(delta) * 450)
                    pair_symbols.append("diamond" if top1_changed else "circle")
                    pair_hover_texts.append(
                        f"L{layer} H{head}<br>Δ Prob: {delta:.4f}<br>Changed: {top1_changed}"
                    )
                    pair_impact_data.append(
                        {
                            "layer": layer,
                            "head": head,
                            "delta": delta,
                            "changed": top1_changed,
                            "top5": patched.top5_tokens,
                            "top5_probs": patched.top5_probs,
                        }
                    )

                    step_count += 1
                    progress_bar.progress(step_count / total_steps)

            per_pair_results.append(
                {
                    "pair": pair_idx,
                    "target": target_prompt_i,
                    "donor": donor_prompt_i,
                    "baseline_top1": baseline.top1_token,
                    "baseline_prob": baseline.top1_prob,
                    "baseline_top5": baseline.top5_tokens,
                    "xs": pair_xs,
                    "ys": pair_ys,
                    "colors": pair_colors,
                    "sizes": pair_sizes,
                    "symbols": pair_symbols,
                    "hover_texts": pair_hover_texts,
                    "impact_data": pair_impact_data,
                }
            )

        progress_bar.empty()

        st.markdown("### 🔍 Baseline Prediction (Per Pair)")
        with st.expander("Show per-pair baselines", expanded=False):
            for summary in baseline_summaries:
                display_top1 = html.escape(visualize_token(summary["top1"]))
                st.markdown(
                    f"P{summary['pair']}: **{display_top1}** ({summary['prob']:.2%})  \n"
                    f"- Target: `{summary['target']}`  \n"
                    f"- Donor: `{summary['donor']}`"
                )

        xs, ys, colors, sizes, symbols, hover_texts = [], [], [], [], [], []
        impact_data = []
        for layer in range(n_layers):
            for head in range(n_heads):
                stats = aggregate[(layer, head)]
                avg_delta = stats["delta_sum"] / pair_count
                changed_rate = stats["changed_count"] / pair_count
                xs.append(head)
                ys.append(layer)
                colors.append("#ff4d6d" if avg_delta < 0 else "#00f2ff")
                sizes.append(8 + abs(avg_delta) * 450)
                symbols.append("diamond" if changed_rate >= 0.5 else "circle")
                hover_texts.append(
                    f"L{layer} H{head}<br>Avg Δ Prob: {avg_delta:.4f}<br>Changed rate: {changed_rate:.2%}"
                )
                impact_data.append(
                    {
                        "layer": layer,
                        "head": head,
                        "delta": avg_delta,
                        "changed_rate": changed_rate,
                    }
                )

        tabs = st.tabs(["Average"] + [f"P{item['pair']}" for item in per_pair_results])
        with tabs[0]:
            left, right = st.columns([3, 0.9])
            with left:
                st.markdown(f'{meta["map_title"]} (Average over {pair_count} pairs)')
                render_impact_scatter(xs, ys, sizes, colors, symbols, hover_texts)
            with right:
                render_aggregate_details_panel(impact_data, pair_count)

            st.markdown(
                f"""
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
<b style="font-size:18px;">Info</b><br><br>
- Each dot represents one (layer, head)<br>
- Dot size = |Average Δ probability| on each pair's baseline top-1 token<br>
- Blue = average probability increase, Red = average probability decrease<br>
- Diamond = top-1 changed in at least 50% of pairs<br>
- {meta["help_line"]}
</div>
""",
                unsafe_allow_html=True,
            )

        for tab, pair_result in zip(tabs[1:], per_pair_results):
            with tab:
                st.markdown(
                    f"### Pair P{pair_result['pair']} · Baseline: "
                    f"`{html.escape(visualize_token(pair_result['baseline_top1']))}` "
                    f"({pair_result['baseline_prob']:.2%})"
                )
                st.markdown(
                    f"- Target: `{pair_result['target']}`  \n"
                    f"- Donor: `{pair_result['donor']}`"
                )

                left, right = st.columns([3, 0.9])
                with left:
                    st.markdown(f'{meta["map_title"]} (P{pair_result["pair"]})')
                    render_impact_scatter(
                        pair_result["xs"],
                        pair_result["ys"],
                        pair_result["sizes"],
                        pair_result["colors"],
                        pair_result["symbols"],
                        pair_result["hover_texts"],
                    )
                with right:
                    render_details_panel(pair_result["impact_data"], pair_result["baseline_top5"])
        return

    target_ids = encode_prompt(tokenizer, target_prompt, device)
    donor_ids = encode_prompt(tokenizer, donor_prompt, device)
    baseline_last, baseline_probs = forward_last_token(model, target_ids)
    baseline = summarize_prediction(tokenizer, baseline_last, baseline_probs)
    render_baseline_cards(baseline)

    donor_qkv_by_layer = _capture_donor_qkv(model, donor_ids, n_layers) if mode in {"v_only", "qk_only", "qkv"} else {}
    donor_av_by_layer = _capture_donor_av(model, donor_ids, n_layers) if mode in {"av", "avo"} else {}

    xs, ys, colors, sizes, symbols, hover_texts = [], [], [], [], [], []
    impact_data = []

    progress_bar = st.progress(0)
    total_steps = n_layers * n_heads
    step_count = 0

    for layer in range(n_layers):
        for head in range(n_heads):
            if mode in {"v_only", "qk_only", "qkv"}:
                donor_qkv = donor_qkv_by_layer[layer]
                head_dim = donor_qkv.shape[-1] // (3 * n_heads)
                base = head * (3 * head_dim)
                q_slice = (base, base + head_dim)
                k_slice = (base + head_dim, base + 2 * head_dim)
                v_slice = (base + 2 * head_dim, base + 3 * head_dim)

                def qkv_hook(module, input, output, qs=q_slice, ks=k_slice, vs=v_slice, dq=donor_qkv):
                    patched = output.clone()
                    patch_len = min(patched.shape[1], dq.shape[1])
                    if mode in {"qk_only", "qkv"}:
                        patched[:, :patch_len, qs[0] : qs[1]] = dq[:, :patch_len, qs[0] : qs[1]]
                        patched[:, :patch_len, ks[0] : ks[1]] = dq[:, :patch_len, ks[0] : ks[1]]
                    if mode in {"v_only", "qkv"}:
                        patched[:, :patch_len, vs[0] : vs[1]] = dq[:, :patch_len, vs[0] : vs[1]]
                    return patched

                handle = model.gpt_neox.layers[layer].attention.query_key_value.register_forward_hook(
                    qkv_hook
                )
            elif mode == "av":
                donor_av = donor_av_by_layer[layer]

                def av_hook(module, input, dh=donor_av, current_head=head):
                    hidden = input[0]
                    bsz, seq_len, hidden_dim = hidden.shape
                    head_dim = hidden_dim // n_heads
                    patch_len = min(seq_len, dh.shape[1])
                    target_heads = hidden.view(bsz, seq_len, n_heads, head_dim).clone()
                    donor_heads = dh.view(1, dh.shape[1], n_heads, head_dim)
                    target_heads[:, :patch_len, current_head, :] = donor_heads[:, :patch_len, current_head, :]
                    return (target_heads.view(bsz, seq_len, hidden_dim),)

                handle = model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(av_hook)
            else:
                donor_av = donor_av_by_layer[layer]

                def avo_hook(module, input, output, dh=donor_av, current_head=head):
                    hidden = input[0]
                    patch_len = min(hidden.shape[1], dh.shape[1])
                    hidden_dim = hidden.shape[-1]
                    head_dim = hidden_dim // n_heads
                    start = current_head * head_dim
                    end = start + head_dim
                    weight_slice = module.weight[:, start:end]
                    target_part = hidden[:, :patch_len, start:end]
                    donor_part = dh[:, :patch_len, start:end]
                    target_contrib = torch.matmul(target_part, weight_slice.t())
                    donor_contrib = torch.matmul(donor_part, weight_slice.t())
                    patched = output.clone()
                    patched[:, :patch_len, :] = patched[:, :patch_len, :] + donor_contrib - target_contrib
                    return patched

                handle = model.gpt_neox.layers[layer].attention.dense.register_forward_hook(avo_hook)

            patched_last, patched_probs = forward_last_token(model, target_ids)
            patched = summarize_prediction(tokenizer, patched_last, patched_probs)
            handle.remove()

            top1_changed = patched.top1_id != baseline.top1_id
            delta = patched_probs[baseline.top1_id].item() - baseline.top1_prob

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
                    "top5": patched.top5_tokens,
                    "top5_probs": patched.top5_probs,
                }
            )

            step_count += 1
            progress_bar.progress(step_count / total_steps)

    progress_bar.empty()

    left, right = st.columns([3, 0.9])
    with left:
        st.markdown(meta["map_title"])
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
        f"""
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
<b style="font-size:18px;">Info</b><br><br>
- Each dot represents one (layer, head)<br>
- Dot size = |Δ probability| on baseline top-1 token<br>
- Blue = probability increase, Red = probability decrease<br>
- {meta["help_line"]}
</div>
""",
        unsafe_allow_html=True,
    )
