from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st
import torch

from modules.common_inference import encode_prompt, forward_last_token, summarize_prediction
from modules.common_model import get_device, get_selected_model_name, load_model
from modules.common_ui import apply_base_theme, render_title, visualize_token


def _capture_states(model, input_ids: torch.Tensor):
    layer_input: dict[int, torch.Tensor] = {}
    layer_output: dict[int, torch.Tensor] = {}
    attn_add: dict[int, torch.Tensor] = {}
    mlp_add: dict[int, torch.Tensor] = {}
    attn_pre_dense: dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        def build_layer_pre(idx):
            def _hook(module, inputs):
                layer_input[idx] = inputs[0].detach()
                return inputs

            return _hook

        def build_layer_out(idx):
            def _hook(module, inputs, output):
                layer_output[idx] = output[0].detach() if isinstance(output, tuple) else output.detach()
                return output

            return _hook

        def build_attn_add(idx):
            def _hook(module, inputs, output):
                attn_add[idx] = output.detach()
                return output

            return _hook

        def build_mlp_add(idx):
            def _hook(module, inputs, output):
                mlp_add[idx] = output.detach()
                return output

            return _hook

        def build_attn_pre(idx):
            def _hook(module, inputs):
                attn_pre_dense[idx] = inputs[0].detach()
                return inputs

            return _hook

        handles.append(layer.register_forward_pre_hook(build_layer_pre(layer_idx)))
        handles.append(layer.register_forward_hook(build_layer_out(layer_idx)))
        handles.append(layer.post_attention_dropout.register_forward_hook(build_attn_add(layer_idx)))
        handles.append(layer.post_mlp_dropout.register_forward_hook(build_mlp_add(layer_idx)))
        handles.append(layer.attention.dense.register_forward_pre_hook(build_attn_pre(layer_idx)))

    with torch.no_grad():
        _ = model(input_ids)

    for handle in handles:
        handle.remove()

    return layer_input, layer_output, attn_add, mlp_add, attn_pre_dense


def _build_probe_vectors(model, layer_input, layer_output, attn_add, mlp_add, attn_pre_dense):
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // n_heads

    probes: list[dict] = []
    for layer_idx in range(n_layers):
        x_in = layer_input[layer_idx][0, -1].detach().cpu()
        a_add = attn_add[layer_idx][0, -1].detach().cpu()
        m_add = mlp_add[layer_idx][0, -1].detach().cpu()
        x_out = layer_output[layer_idx][0, -1].detach().cpu()
        x_after_attn = x_in + a_add

        probes.extend(
            [
                {"id": f"L{layer_idx}.res_in", "layer": layer_idx, "kind": "arch", "title": "Residual In", "vec": x_in},
                {
                    "id": f"L{layer_idx}.attn_add",
                    "layer": layer_idx,
                    "kind": "arch",
                    "title": "Attention Add",
                    "vec": a_add,
                },
                {
                    "id": f"L{layer_idx}.res_after_attn",
                    "layer": layer_idx,
                    "kind": "arch",
                    "title": "Residual After Attn",
                    "vec": x_after_attn,
                },
                {"id": f"L{layer_idx}.mlp_add", "layer": layer_idx, "kind": "arch", "title": "MLP Add", "vec": m_add},
                {"id": f"L{layer_idx}.res_out", "layer": layer_idx, "kind": "arch", "title": "Residual Out", "vec": x_out},
            ]
        )

        dense_weight = model.gpt_neox.layers[layer_idx].attention.dense.weight.detach()
        attn_cat = attn_pre_dense[layer_idx][0, -1]
        heads = attn_cat.view(n_heads, head_dim)
        w_by_head = dense_weight.view(hidden_size, n_heads, head_dim).permute(1, 0, 2).contiguous()
        head_contribs = torch.einsum("hd,hod->ho", heads, w_by_head)
        for head_idx in range(n_heads):
            probes.append(
                {
                    "id": f"L{layer_idx}.H{head_idx}",
                    "layer": layer_idx,
                    "kind": "head",
                    "title": f"H{head_idx}",
                    "vec": head_contribs[head_idx].detach().cpu(),
                }
            )

    return probes


def _run_lens(
    model,
    tokenizer,
    probes: list[dict],
    topk_display: int,
    apply_final_ln: bool,
    chunk_size: int,
    target_token_id: int | None,
    target_token_text: str | None,
) -> dict[str, dict]:
    if not probes:
        return {}
    device = next(model.parameters()).device
    final_ln = model.gpt_neox.final_layer_norm
    lm_head = model.embed_out
    vectors = torch.stack([probe["vec"] for probe in probes], dim=0)
    out: dict[str, dict] = {}

    vocab_size = model.config.vocab_size
    topn_for_color = min(100, vocab_size)
    topk_display = min(topk_display, vocab_size)
    normalized_target = target_token_text.lstrip() if target_token_text is not None else None

    for start in range(0, vectors.size(0), chunk_size):
        end = min(start + chunk_size, vectors.size(0))
        batch = vectors[start:end].to(device).unsqueeze(1)
        with torch.no_grad():
            if apply_final_ln:
                batch = final_ln(batch)
            logits = lm_head(batch).squeeze(1)
            top_vals, top_ids = torch.topk(logits, k=topn_for_color, dim=-1)

        for i in range(end - start):
            probe = probes[start + i]
            ids_100 = top_ids[i].tolist()
            vals_100 = top_vals[i].tolist()
            ids = ids_100[:topk_display]
            vals = vals_100[:topk_display]
            decoded_100 = [tokenizer.decode([tok]) for tok in ids_100]
            target_rank_candidates: list[int] = []
            if target_token_id is not None and target_token_id in ids_100:
                target_rank_candidates.append(ids_100.index(target_token_id) + 1)
            if normalized_target is not None:
                for idx, decoded in enumerate(decoded_100):
                    if decoded.lstrip() == normalized_target:
                        target_rank_candidates.append(idx + 1)
            target_rank = min(target_rank_candidates) if target_rank_candidates else None
            out[probe["id"]] = {
                "id": probe["id"],
                "title": probe["title"],
                "layer": probe["layer"],
                "kind": probe["kind"],
                "norm": float(torch.linalg.norm(vectors[start + i]).item()),
                "topk_ids": ids,
                "topk_tokens": [visualize_token(tok) for tok in decoded_100[:topk_display]],
                "topk_logits": vals,
                "top1_token": visualize_token(decoded_100[0]),
                "top1_logit": vals[0],
                "target_rank": target_rank,
            }
    return out


def _inject_css():
    st.markdown(
        """
<style>
.arch-card {
    background: #171b24;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 8px 10px;
    margin-bottom: 8px;
}
.arch-title {
    font-size: 13px;
    color: #e5ecff;
    font-weight: 700;
}
.arch-sub {
    font-size: 11px;
    color: #9fb0d9;
}
.arch-dot-on { color: #00f2aa; font-weight: 800; }
.arch-dot-off { color: #5d6c8e; font-weight: 800; }
.probe-card {
    background: #171b24;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 10px;
}
.probe-top1 {
    font-family: monospace;
    font-size: 24px;
    color: #f2f6ff;
}
.probe-sub {
    color: #9fb0d9;
    font-size: 12px;
    margin-top: 4px;
}
.token-hit {
    border: 1px solid #00f2aa;
    background: #132a23;
}
.token-nohit {
    border: 1px solid rgba(255,255,255,0.07);
    background: #20242f;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _build_layer_arch_figure(
    layer_idx: int,
    lens: dict[str, dict],
    target_label: str,
    n_heads: int,
    use_top5: bool,
    use_top20: bool,
    use_top100: bool,
):
    nodes = [
        ("Residual In", f"L{layer_idx}.res_in", 0.0, 10.5, True),
        ("Attention Add", f"L{layer_idx}.attn_add", 2.1, 7.8, True),
        ("Residual After Attn", f"L{layer_idx}.res_after_attn", 0.0, 7.1, True),
        ("MLP Add", f"L{layer_idx}.mlp_add", 2.1, 4.3, True),
        ("Residual Out", f"L{layer_idx}.res_out", 0.0, 3.6, True),
    ]

    edge_pairs = [
        (0, 2),  # residual backbone: res_in -> res_after_attn
        (0, 1),  # residual stream enters attention branch
        (1, 2),  # attention add merges into residual
        (2, 4),  # residual backbone: res_after_attn -> res_out
        (2, 3),  # residual stream enters mlp branch
        (3, 4),  # mlp add merges into residual
    ]

    head_start_idx = len(nodes)
    base_x = -2.4
    span = 2.1
    denom = max(n_heads - 1, 1)
    for head_idx in range(n_heads):
        hx = base_x + span * (head_idx / denom)
        nodes.append((f"H{head_idx}", f"L{layer_idx}.H{head_idx}", hx, 8.2, True))
        edge_pairs.append((head_start_idx + head_idx, 1))  # each head contributes to attention add

    fig = go.Figure()

    for src_idx, dst_idx in edge_pairs:
        x0, y0 = nodes[src_idx][2], nodes[src_idx][3]
        x1, y1 = nodes[dst_idx][2], nodes[dst_idx][3]
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.4,
            arrowcolor="rgba(180,200,255,0.45)",
        )

    x_vals = []
    y_vals = []
    labels = []
    colors = []
    sizes = []
    customdata = []
    hover = []

    for title, probe_id, x, y, clickable in nodes:
        x_vals.append(x)
        y_vals.append(y)
        labels.append(title)
        customdata.append(probe_id if clickable else "")
        sizes.append(42 if clickable else 36)
        row = lens.get(probe_id)
        rank = row.get("target_rank") if row else None
        if rank is not None and use_top5 and rank <= 5:
            node_color = "#ff3b3b"
        elif rank is not None and use_top20 and 6 <= rank <= 20:
            node_color = "#3f7cff"
        elif rank is not None and use_top100 and 21 <= rank <= 100:
            node_color = "#00f2aa"
        else:
            node_color = "#39527d"
        colors.append(node_color)
        if row is None:
            hover.append(f"{title}<br>no lens data")
        else:
            selected_labels = []
            if use_top5:
                selected_labels.append("TOP-5")
            if use_top20:
                selected_labels.append("TOP-20")
            if use_top100:
                selected_labels.append("TOP-100")
            cutoff_label = ", ".join(selected_labels) if selected_labels else "none"
            if rank is None:
                rank_text = "outside top-100"
                bucket_text = "none"
            elif rank <= 5:
                rank_text = f"#{rank}"
                bucket_text = "TOP-5 (RED)"
            elif rank <= 20:
                rank_text = f"#{rank}"
                bucket_text = "TOP-20 (BLUE)"
            else:
                rank_text = f"#{rank}"
                bucket_text = "TOP-100 (GREEN)"
            if row["id"].endswith(".res_after_attn"):
                combine_hint = "combine: Residual In + Attention Add"
            elif row["id"].endswith(".res_out"):
                combine_hint = "combine: Residual After Attn + MLP Add"
            else:
                combine_hint = "branch contribution"
            topk_rows = "<br>".join(
                f"{i+1}. {html.escape(row['topk_tokens'][i])} ({row['topk_logits'][i]:.2f})"
                for i in range(len(row["topk_tokens"]))
            )
            hover.append(
                f"{html.escape(row['id'])}"
                f"<br>{combine_hint}"
                f"<br>top1: {html.escape(row['top1_token'])} ({row['top1_logit']:.2f})"
                f"<br>||v||={row['norm']:.3f}"
                f"<br>target: {html.escape(target_label)}"
                f"<br>target rank: {rank_text}"
                f"<br>selected cutoff: {cutoff_label}"
                f"<br>bucket: {bucket_text}"
                f"<br>top-k:<br>{topk_rows}"
            )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text",
            text=labels,
            textposition="middle center",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1, color="rgba(255,255,255,0.35)"),
                symbol="square",
            ),
            customdata=customdata,
            hovertext=hover,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        height=520,
        margin=dict(l=10, r=10, t=8, b=8),
        xaxis=dict(visible=False, range=[-2.8, 2.8]),
        yaxis=dict(visible=False, range=[2.8, 11.2]),
    )
    return fig


def _is_cuda_runtime_issue(exc: RuntimeError) -> bool:
    msg = str(exc).upper()
    return "CUDA" in msg or "CUBLAS_STATUS_NOT_INITIALIZED" in msg or "OUT OF MEMORY" in msg


def _compute_all(model, tokenizer, prompt_text, device):
    input_ids = encode_prompt(tokenizer, prompt_text, device)
    layer_input, layer_output, attn_add, mlp_add, attn_pre_dense = _capture_states(model, input_ids)
    probes = _build_probe_vectors(model, layer_input, layer_output, attn_add, mlp_add, attn_pre_dense)
    final_logits, final_probs = forward_last_token(model, input_ids)
    final_summary = summarize_prediction(tokenizer, final_logits, final_probs)
    return probes, final_summary, input_ids


def _render_arch_node(row: dict, is_hit: bool):
    dot_class = "arch-dot-on" if is_hit else "arch-dot-off"
    dot_text = "ON" if is_hit else "OFF"
    st.markdown(
        f"""
<div class="arch-card">
  <div class="arch-title">{html.escape(row["title"])}</div>
  <div class="arch-sub"><span class="{dot_class}">● {dot_text}</span> contains final token in top-k</div>
</div>
""",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Head + MLP Logit Lens", layout="wide")
apply_base_theme(top5_font_size=16)
_inject_css()

if "head_mlp_cache" not in st.session_state:
    st.session_state["head_mlp_cache"] = None

device = get_device()
selected_model_name = get_selected_model_name()
model, tokenizer = load_model(selected_model_name, str(device))
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads

render_title("🔬 Head + MLP Architecture Logit Lens")

prompt = st.text_area("Prompt", "The capital city of France is", height=120)
chunk_size = 64
topk_display = 5
apply_final_ln = True
cutoff_cols = st.columns(3)
with cutoff_cols[0]:
    use_top5 = st.checkbox("Top 5", value=True)
with cutoff_cols[1]:
    use_top20 = st.checkbox("Top 20", value=True)
with cutoff_cols[2]:
    use_top100 = st.checkbox("Top 100", value=True)

run = st.button("🚀 Analyze Architecture")

if run:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        st.warning("프롬프트를 입력하세요.")
        st.stop()
    try:
        used_device = device
        used_model = model
        used_tokenizer = tokenizer
        probes, final_summary, input_ids = _compute_all(
            model=model,
            tokenizer=tokenizer,
            prompt_text=clean_prompt,
            device=device,
        )
    except RuntimeError as exc:
        if device.type == "cuda" and _is_cuda_runtime_issue(exc):
            torch.cuda.empty_cache()
            used_device = torch.device("cpu")
            used_model, used_tokenizer = load_model(selected_model_name, "cpu")
            st.warning("CUDA 실패로 CPU 재시도했습니다.")
            probes, final_summary, input_ids = _compute_all(
                model=used_model,
                tokenizer=used_tokenizer,
                prompt_text=clean_prompt,
                device=used_device,
            )
        else:
            raise

    target_token_id = final_summary.top1_id
    target_label = visualize_token(final_summary.top1_token)

    lens = _run_lens(
        used_model,
        used_tokenizer,
        probes,
        topk_display,
        apply_final_ln,
        chunk_size,
        target_token_id,
        final_summary.top1_token,
    )

    st.session_state["head_mlp_cache"] = {
        "prompt": clean_prompt,
        "probes": probes,
        "lens": lens,
        "final_summary": final_summary,
        "target_token_id": target_token_id,
        "target_label": target_label,
        "input_len": int(input_ids.shape[-1]),
        "device": str(used_device),
        "model_name": selected_model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
    }

cache = st.session_state.get("head_mlp_cache")
if cache:
    lens: dict[str, dict] = cache["lens"]
    final_summary = cache["final_summary"]
    target_label = cache.get("target_label", visualize_token(final_summary.top1_token))

    st.caption(f"Model: {cache['model_name']}")
    st.caption(f"Device used: {cache['device']}")
    st.caption(f"Prompt length: {cache['input_len']} tokens")
    st.caption("색상 기준: Top 5 빨강 / Top 6~20 파랑 / Top 21~100 초록 (체크한 구간만 표시)")
    st.caption(f"Target: {target_label}")

    st.markdown("### Final Output")
    a, b = st.columns(2)
    with a:
        st.markdown(
            f"<div class='probe-card'><b>Top-1 Token</b><div class='probe-top1'>{html.escape(visualize_token(final_summary.top1_token))}</div></div>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f"<div class='probe-card'><b>Top-1 Prob</b><div class='probe-top1'>{final_summary.top1_prob:.2%}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Transformer Architecture View")
    st.caption("세로 backbone이 residual 흐름입니다. Attention Add, MLP Add가 backbone으로 합쳐지는 구조로 표시됩니다.")
    st.caption("노드 hover 시 상세를 확인할 수 있고, 색상 기준은 선택한 Top cutoff(5/20/100)입니다.")
    for layer_idx in range(cache["n_layers"]):
        st.markdown(f"#### Layer {layer_idx}")
        fig = _build_layer_arch_figure(
            layer_idx=layer_idx,
            lens=lens,
            target_label=target_label,
            n_heads=cache["n_heads"],
            use_top5=use_top5,
            use_top20=use_top20,
            use_top100=use_top100,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"arch_fig_{layer_idx}",
            config={"displayModeBar": False},
        )
