import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Rotary Position Embedding (RoPE), Su et al. [2021].

        RoPE injects position information by *rotating* pairs of embedding
        dimensions. The d_k-dim query/key vector is split into d_k/2 pairs; the
        k-th pair of a token at position i is rotated (as a 2D vector) by angle

            theta_{i,k} = i / theta ** ((2k - 2) / d_k),   k in {1, ..., d_k/2}

        Here `i` is the *absolute token position* and `k` is the *dimension-pair
        index* -- theta_{i,k} is NOT a relation between two tokens. The relative
        nature only shows up in attention: for a query at i and a key at j,
        R_i^T R_j = R_{j-i}, so their dot product depends only on (j - i).

        This layer has NO learnable parameters: the rotation angles are fully
        determined by `theta` and position, so cos/sin can be precomputed once
        and reused across batches and layers (store them via
        register_buffer(..., persistent=False), not nn.Parameter).

        Args:
            theta (float): The constant Theta in the angle formula (base for the
                per-dimension-pair frequencies). Larger Theta -> lower frequencies.
            d_k (int): Dimension of the query/key vectors. Must be even so it can
                be split into d_k/2 rotated pairs.
            max_seq_len (int): Largest sequence length we will ever see; the cos/sin
                buffer is precomputed for positions 0 .. max_seq_len - 1.
            device (torch.device | None, optional): Device to store the buffers on.
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # Suggested precompute (in __init__), reused every forward:
        #   freqs[k]        = theta ** (-(2k) / d_k)          shape (d_k/2,)
        #   angles[i, k]    = i * freqs[k]                    shape (max_seq_len, d_k/2)
        #   self.cos = cos(angles), self.sin = sin(angles)    -> register_buffer(persistent=False)
        num_pairs = d_k // 2
        freqs = theta ** (-(torch.arange(num_pairs, device=device) * 2 / d_k))
        # Outer product i * freqs[k] via broadcasting:
        #   [:, None] makes positions a column  (max_seq_len, 1)
        #   [None, :] makes freqs a row         (1, num_pairs)
        # multiplying broadcasts to angles[i, k] = i * freqs[k]  (max_seq_len, num_pairs)
        angles = torch.arange(max_seq_len, device=device)[:, None] * freqs[None, :]
        # register_buffer stores a tensor as part of the module's state (so it moves
        # with .to(device)/.cuda() and shows up in state_dict) but WITHOUT making it a
        # trainable nn.Parameter -- RoPE's cos/sin have no gradients to learn. Accessed
        # later as self.cos / self.sin. persistent=False keeps them out of state_dict
        # since they're fully recomputable from theta/d_k/max_seq_len.
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to `x` (a query or key tensor).

        Args:
            x (torch.Tensor): Shape (..., seq_len, d_k). Any number of leading
                batch dimensions is allowed.
            token_positions (torch.Tensor): Shape (..., seq_len). Integer tensor
                giving the absolute position `i` of each token along the sequence
                dimension -- this is the `i` in theta_{i,k}. Usually 0..seq_len-1,
                but may differ (e.g. KV-cache decoding, padded/packed sequences).
                Used to slice the precomputed cos/sin along the sequence axis.

        Returns:
            torch.Tensor: Same shape as `x`, with each dimension-pair rotated.

        Rotation per pair (x1, x2) using cos = cos(theta_{i,k}), sin = sin(theta_{i,k}):
            x1' = x1 * cos - x2 * sin
            x2' = x1 * sin + x2 * cos
        i.e. the 2x2 rotation R(theta_{i,k}) applied to each of the d_k/2 pairs.
        """
        # Gather the precomputed cos/sin for each token's position.
        # token_positions: (..., seq_len) -> indexes rows of the (max_seq_len, num_pairs)
        # buffers, giving cos/sin of shape (..., seq_len, num_pairs).
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # Split the d_k dims into adjacent pairs: pair k is (x[..., 2k], x[..., 2k+1]).
        # x1, x2 each have shape (..., seq_len, num_pairs).
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # Apply the 2x2 rotation to every pair.
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Re-interleave the rotated pairs back into the original layout:
        # stack -> (..., seq_len, num_pairs, 2), then flatten the last two dims
        # back to d_k, giving [out1_0, out2_0, out1_1, out2_1, ...].
        return torch.stack((out1, out2), dim=-1).flatten(-2)