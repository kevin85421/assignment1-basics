import math

import torch

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention (Vaswani et al., 2017, section 3.2.1).

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    For each query, compute a similarity score against every key, scale by
    1/sqrt(d_k) to keep the softmax gradients well-behaved, optionally mask out
    disallowed positions, softmax over the keys to get attention weights, then
    take the weighted sum of the value vectors.

    Args:
        Q (torch.Tensor): Queries of shape (..., queries, d_k). Any number of
            leading batch/head dimensions is allowed.
        K (torch.Tensor): Keys of shape (..., keys, d_k). Same d_k as Q.
        V (torch.Tensor): Values of shape (..., keys, d_v). Same number of keys as K.
        mask (torch.Tensor | None): Optional boolean mask of shape (..., queries, keys).
            True means "attend to this key"; False positions are set to -inf before
            the softmax so they receive ~zero weight. If None, all keys are attended.

    Returns:
        torch.Tensor: Attention output of shape (..., queries, d_v).

    Hints:
        - d_k = Q.shape[-1]; scale scores by 1 / sqrt(d_k).
        - scores = Q @ K.transpose(-2, -1)  -> shape (..., queries, keys).
        - Apply the mask with scores.masked_fill(~mask, float("-inf")).
        - Softmax over the last dim (keys), then multiply by V.
    """
    # Shapes: Q (..., n_q, d_k), K (..., n_k, d_k), V (..., n_k, d_v)
    d_k = Q.shape[-1]
    # K.transpose(-2, -1): (..., n_k, d_k) -> (..., d_k, n_k)
    # Q @ K^T:            (..., n_q, d_k) @ (..., d_k, n_k) -> (..., n_q, n_k)
    # scores[..., i, j] = similarity between query i and key j.
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # mask: (..., n_q, n_k), True = attend. Set disallowed positions
        # (mask == False) to -inf so that after softmax exp(-inf) = 0, giving
        # them zero attention weight (e.g. a causal mask hides future tokens).
        scores = scores.masked_fill(~mask, float("-inf"))
    # softmax over dim=-1 (the keys axis n_k) -> weights (..., n_q, n_k)
    # weights @ V: (..., n_q, n_k) @ (..., n_k, d_v) -> output (..., n_q, d_v)
    return softmax(scores, dim=-1) @ V
