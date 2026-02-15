from typing import Iterable

import torch


def apply_single_head_scale(
    hidden: torch.Tensor, head_idx: int, n_heads: int, scale: float = 0.0
) -> torch.Tensor:
    """Scale one attention head in the dense input tensor.

    `scale=0.0` disables the head, while values like `0.1` keep partial effect.
    """
    bsz, seq_len, hidden_dim = hidden.shape
    head_dim = hidden_dim // n_heads

    heads = hidden.view(bsz, seq_len, n_heads, head_dim).clone()
    heads[:, :, head_idx, :] *= scale
    return heads.view(bsz, seq_len, hidden_dim)


def apply_multi_head_scale_by_repetition(
    hidden: torch.Tensor, head_indices: Iterable[int], n_heads: int, scale: float = 0.0
) -> torch.Tensor:
    """Apply the single-head scaling algorithm repeatedly for multiple heads."""
    updated = hidden
    for head_idx in head_indices:
        updated = apply_single_head_scale(updated, head_idx, n_heads, scale)
    return updated


def keep_only_selected_heads(
    hidden: torch.Tensor, selected_head_indices: Iterable[int], n_heads: int
) -> torch.Tensor:
    """Keep only selected heads and zero-out every other head."""
    bsz, seq_len, hidden_dim = hidden.shape
    head_dim = hidden_dim // n_heads

    heads = hidden.view(bsz, seq_len, n_heads, head_dim)
    kept = torch.zeros_like(heads)
    for head_idx in selected_head_indices:
        kept[:, :, head_idx, :] = heads[:, :, head_idx, :]
    return kept.view(bsz, seq_len, hidden_dim)


def replace_selected_heads_from_donor(
    hidden: torch.Tensor,
    donor_hidden: torch.Tensor,
    selected_head_indices: Iterable[int],
    n_heads: int,
) -> torch.Tensor:
    """Replace selected heads using donor prompt activations (overlapping prefix length)."""
    bsz, seq_len, hidden_dim = hidden.shape
    head_dim = hidden_dim // n_heads
    patch_len = min(seq_len, donor_hidden.shape[1])

    target_heads = hidden.view(bsz, seq_len, n_heads, head_dim).clone()
    donor_heads = donor_hidden.view(1, donor_hidden.shape[1], n_heads, head_dim)
    for head_idx in selected_head_indices:
        target_heads[:, :patch_len, head_idx, :] = donor_heads[:, :patch_len, head_idx, :]
    return target_heads.view(bsz, seq_len, hidden_dim)
