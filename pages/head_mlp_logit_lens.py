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


def _run_lens(model, tokenizer, probes: list[dict], topk: int, apply_final_ln: bool, chunk_size: int) -> dict[str, dict]:
    if not probes:
        return {}
    device = next(model.parameters()).device
    final_ln = model.gpt_neox.final_layer_norm
    lm_head = model.embed_out
    vectors = torch.stack([probe["vec"] for probe in probes], dim=0)
    out: dict[str, dict] = {}

    for start in range(0, vectors.size(0), chunk_size):
        end = min(start + chunk_size, vectors.size(0))
        batch = vectors[start:end].to(device).unsqueeze(1)
        with torch.no_grad():
            if apply_final_ln:
                batch = final_ln(batch)
            logits = lm_head(batch).squeeze(1)
            top_vals, top_ids = torch.topk(logits, k=topk, dim=-1)

        for i in range(end - start):
            probe = probes[start + i]
            ids = top_ids[i].tolist()
            vals = top_vals[i].tolist()
            out[probe["id"]] = {
                "id": probe["id"],
                "title": probe["title"],
                "layer": probe["layer"],
                "kind": probe["kind"],
                "norm": float(torch.linalg.norm(vectors[start + i]).item()),
                "topk_ids": ids,
                "topk_tokens": [visualize_token(tokenizer.decode([tok])) for tok in ids],
                "topk_logits": vals,
                "top1_token": visualize_token(tokenizer.decode([ids[0]])),
                "top1_logit": vals[0],
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
    final_token_id: int,
    n_heads: int,
    show_head_direct: bool,
):
    nodes = [
        ("Residual In", f"L{layer_idx}.res_in", 0.0, 2.0, True),
        ("QKV", None, 1.0, 3.0, False),
        ("Score", None, 2.0, 3.0, False),
        ("Softmax", None, 3.0, 3.0, False),
        ("A*V", None, 4.0, 3.0, False),
        ("W_O", None, 5.0, 3.0, False),
        ("Attention Add", f"L{layer_idx}.attn_add", 4.0, 2.0, True),
        ("Residual After Attn", f"L{layer_idx}.res_after_attn", 5.2, 2.0, True),
        ("MLP Add", f"L{layer_idx}.mlp_add", 4.0, 1.0, True),
        ("Residual Out", f"L{layer_idx}.res_out", 6.4, 2.0, True),
    ]

    edge_pairs = [
        (0, 6),  # res in -> attn add branch target
        (6, 7),  # attn add -> post attn
        (7, 9),  # post attn -> out
        (7, 8),  # post attn -> mlp add
        (8, 9),  # mlp add -> out
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
    ]

    if show_head_direct:
        base_x = 1.0
        span = 4.6
        denom = max(n_heads - 1, 1)
        for head_idx in range(n_heads):
            hx = base_x + span * (head_idx / denom)
            nodes.append((f"H{head_idx}", f"L{layer_idx}.H{head_idx}", hx, 0.2, True))

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
        if not clickable:
            colors.append("#5d6c8e")
            hover.append(f"{title}<br>decorative node")
        else:
            row = lens.get(probe_id)
            hit = row is not None and final_token_id in row["topk_ids"]
            colors.append("#00f2aa" if hit else "#39527d")
            hover.append(f"{title}<br>click to inspect")

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
        height=310 if show_head_direct else 260,
        margin=dict(l=10, r=10, t=8, b=8),
        xaxis=dict(visible=False, range=[-0.5, 7.0]),
        yaxis=dict(visible=False, range=[-0.4, 3.5]),
    )
    return fig


def _is_cuda_runtime_issue(exc: RuntimeError) -> bool:
    msg = str(exc).upper()
    return "CUDA" in msg or "CUBLAS_STATUS_NOT_INITIALIZED" in msg or "OUT OF MEMORY" in msg


def _compute_all(model, tokenizer, prompt_text, device, topk, apply_final_ln, chunk_size):
    input_ids = encode_prompt(tokenizer, prompt_text, device)
    layer_input, layer_output, attn_add, mlp_add, attn_pre_dense = _capture_states(model, input_ids)
    probes = _build_probe_vectors(model, layer_input, layer_output, attn_add, mlp_add, attn_pre_dense)
    lens = _run_lens(model, tokenizer, probes, topk, apply_final_ln, chunk_size)
    final_logits, final_probs = forward_last_token(model, input_ids)
    final_summary = summarize_prediction(tokenizer, final_logits, final_probs)
    return probes, lens, final_summary, input_ids


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

if "head_mlp_force_cpu" not in st.session_state:
    st.session_state["head_mlp_force_cpu"] = False
if "head_mlp_force_cpu_pending" not in st.session_state:
    st.session_state["head_mlp_force_cpu_pending"] = False
if "head_mlp_selected_probe" not in st.session_state:
    st.session_state["head_mlp_selected_probe"] = None
if "head_mlp_cache" not in st.session_state:
    st.session_state["head_mlp_cache"] = None

# Defer force_cpu widget value changes to before widget instantiation.
if st.session_state["head_mlp_force_cpu_pending"]:
    st.session_state["head_mlp_force_cpu"] = True
    st.session_state["head_mlp_force_cpu_pending"] = False

force_cpu = st.sidebar.checkbox("Force CPU (Head+MLP Lens)", key="head_mlp_force_cpu")
show_head_direct = st.sidebar.checkbox("Show head probes", value=True)

device = torch.device("cpu") if force_cpu else get_device()
selected_model_name = get_selected_model_name()
model, tokenizer = load_model(selected_model_name, str(device))
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads

render_title("🔬 Head + MLP Architecture Logit Lens")

prompt = st.text_area("Prompt", "The capital city of France is", height=120)
cols = st.columns(3)
with cols[0]:
    topk = int(st.slider("Top-k", min_value=1, max_value=10, value=5, step=1))
with cols[1]:
    apply_final_ln = st.checkbox("Apply final layer norm before LM head", value=True)
with cols[2]:
    chunk_size = int(st.selectbox("Chunk size", options=[32, 64, 96, 128], index=1))

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
        probes, lens, final_summary, input_ids = _compute_all(
            model=model,
            tokenizer=tokenizer,
            prompt_text=clean_prompt,
            device=device,
            topk=topk,
            apply_final_ln=apply_final_ln,
            chunk_size=chunk_size,
        )
    except RuntimeError as exc:
        if device.type == "cuda" and _is_cuda_runtime_issue(exc):
            torch.cuda.empty_cache()
            used_device = torch.device("cpu")
            used_model, used_tokenizer = load_model(selected_model_name, "cpu")
            st.session_state["head_mlp_force_cpu_pending"] = True
            st.warning("CUDA 실패로 CPU 재시도했습니다.")
            probes, lens, final_summary, input_ids = _compute_all(
                model=used_model,
                tokenizer=used_tokenizer,
                prompt_text=clean_prompt,
                device=used_device,
                topk=topk,
                apply_final_ln=apply_final_ln,
                chunk_size=chunk_size,
            )
        else:
            raise

    st.session_state["head_mlp_cache"] = {
        "probes": probes,
        "lens": lens,
        "final_summary": final_summary,
        "input_len": int(input_ids.shape[-1]),
        "device": str(used_device),
        "model_name": selected_model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
    }
    st.session_state["head_mlp_selected_probe"] = "L0.res_in"

cache = st.session_state.get("head_mlp_cache")
if cache:
    lens: dict[str, dict] = cache["lens"]
    final_summary = cache["final_summary"]
    final_token_id = final_summary.top1_id

    st.caption(f"Model: {cache['model_name']}")
    st.caption(f"Device used: {cache['device']}")
    st.caption(f"Prompt length: {cache['input_len']} tokens")
    st.caption("아키텍처 박스의 불(ON)은 해당 지점 top-k에 최종 출력 토큰이 포함된다는 뜻입니다.")

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
    st.caption("노드를 클릭하면 해당 위치의 logit lens 결과가 아래에 표시됩니다. 초록색은 final token이 top-k에 포함된 노드입니다.")
    for layer_idx in range(cache["n_layers"]):
        with st.expander(f"Layer {layer_idx}", expanded=layer_idx < 2):
            fig = _build_layer_arch_figure(
                layer_idx=layer_idx,
                lens=lens,
                final_token_id=final_token_id,
                n_heads=cache["n_heads"],
                show_head_direct=show_head_direct,
            )
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"arch_fig_{layer_idx}",
                on_select="rerun",
                selection_mode="points",
                config={"displayModeBar": False},
            )
            if event and event.get("selection") and event["selection"].get("points"):
                point = event["selection"]["points"][0]
                if point.get("customdata"):
                    st.session_state["head_mlp_selected_probe"] = point["customdata"]

    selected_probe = st.session_state.get("head_mlp_selected_probe")
    if selected_probe and selected_probe in lens:
        row = lens[selected_probe]
        is_hit = final_token_id in row["topk_ids"]
        st.markdown("### Probe Result")
        st.markdown(
            f"<div class='probe-card'><b>{html.escape(row['id'])}</b><div class='probe-top1'>{html.escape(row['top1_token'])}</div>"
            f"<div class='probe-sub'>top1 logit: {row['top1_logit']:.3f} | ||v||={row['norm']:.3f}</div></div>",
            unsafe_allow_html=True,
        )
        if is_hit:
            st.success("최종 출력 토큰이 이 지점의 top-k에 포함됩니다.")
        else:
            st.warning("최종 출력 토큰이 이 지점의 top-k에 포함되지 않습니다.")

        token_cols = st.columns(len(row["topk_tokens"]))
        for i, col in enumerate(token_cols):
            tok = row["topk_tokens"][i]
            logit = row["topk_logits"][i]
            tok_id = row["topk_ids"][i]
            css_class = "token-hit" if tok_id == final_token_id else "token-nohit"
            with col:
                st.markdown(
                    f"<div class='top5-card {css_class}'>{html.escape(tok)}<br><span style='font-size:13px;color:#aab6d4'>{logit:.2f}</span></div>",
                    unsafe_allow_html=True,
                )
