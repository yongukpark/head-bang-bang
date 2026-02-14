from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class PredictionSnapshot:
    """Structured prediction result for the last token position."""

    top1_id: int
    top1_token: str
    top1_prob: float
    top5_ids: list[int]
    top5_tokens: list[str]
    top5_probs: list[float]


def parse_head_labels(labels: Iterable[str]) -> list[tuple[int, int]]:
    """Convert L{layer}H{head} labels into (layer, head) tuples."""
    parsed: list[tuple[int, int]] = []
    for label in labels:
        layer = int(label.split("H")[0][1:])
        head = int(label.split("H")[1])
        parsed.append((layer, head))
    return parsed


def build_head_labels(n_layers: int, n_heads: int) -> list[str]:
    return [f"L{idx // n_heads}H{idx % n_heads}" for idx in range(n_layers * n_heads)]


def heads_by_layer(selected: Iterable[tuple[int, int]], n_layers: int) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {layer: [] for layer in range(n_layers)}
    for layer, head in selected:
        grouped[layer].append(head)
    return grouped


def encode_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def forward_last_token(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model once and return (last_logits, last_probs)."""
    with torch.no_grad():
        logits = model(input_ids).logits
    last_logits = logits[0, -1]
    last_probs = torch.softmax(last_logits, dim=-1)
    return last_logits, last_probs


def summarize_prediction(tokenizer, last_logits: torch.Tensor, last_probs: torch.Tensor) -> PredictionSnapshot:
    top5_ids_tensor = torch.topk(last_logits, 5).indices
    top5_ids = [idx.item() for idx in top5_ids_tensor]
    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
    top5_probs = [last_probs[idx].item() for idx in top5_ids]

    return PredictionSnapshot(
        top1_id=top5_ids[0],
        top1_token=top5_tokens[0],
        top1_prob=top5_probs[0],
        top5_ids=top5_ids,
        top5_tokens=top5_tokens,
        top5_probs=top5_probs,
    )
