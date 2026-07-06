import torch
from torch import nn
from einops import rearrange

from cs336_basics.linear import Linear
from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Causal multi-head self-attention (Vaswani et al., 2017, section 3.2.2).

        Projects the input into num_heads independent (Q, K, V) subspaces, runs
        scaled dot-product attention per head with a causal mask, concatenates
        the heads, and applies a final output projection.

        Args:
            d_model (int): Model / embedding dimension. Must be divisible by num_heads.
            num_heads (int): Number of attention heads. Each head has size
                d_k = d_v = d_model // num_heads.
            rope (RotaryPositionalEmbedding | None): Optional RoPE module applied to
                Q and K before attention. If None, no positional rotation is applied.
            device (torch.device | None): Device for the projection weights.
            dtype (torch.dtype | None): Dtype for the projection weights.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        # Q/K/V projections map d_model -> d_model (= num_heads * head_dim); the
        # adapter copies reference weights into these. output_proj maps the
        # concatenated heads (d_model) back to d_model.
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply causal multi-head self-attention to `x`.

        Args:
            x (torch.Tensor): Input of shape (..., seq_len, d_model).
            token_positions (torch.Tensor | None): Positions of shape (..., seq_len),
                used by RoPE. If None and RoPE is set, default to 0..seq_len-1.

        Returns:
            torch.Tensor: Output of shape (..., seq_len, d_model).

        Hints:
            - Project x -> Q, K, V, each (..., seq_len, d_model).
            - Reshape/split into heads: (..., num_heads, seq_len, head_dim)
              (e.g. via einops.rearrange with "... s (h d) -> ... h s d").
            - If self.rope is not None, apply it to Q and K (per head).
            - Build a causal mask (seq_len x seq_len) that is True on/below the
              diagonal so each query only attends to itself and earlier keys.
            - Run scaled_dot_product_attention(Q, K, V, mask) per head.
            - Merge heads back to (..., seq_len, d_model) and apply output_proj.
        """
        seq_len = x.shape[-2]

        # Project into Q, K, V, each (..., seq_len, d_model).
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split the d_model dim into heads: (..., seq_len, num_heads * head_dim)
        # -> (..., num_heads, seq_len, head_dim), so each head attends independently.
        q = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)

        # Rotate Q and K with RoPE (per head, on the head_dim axis) if enabled.
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Causal mask (seq_len, seq_len): True on/below the diagonal, so query i
        # can attend to keys j <= i only. Broadcasts over batch/head dims.
        # Example for seq_len=4 (rows = query i, cols = key j; True = can attend):
        #        k0     k1     k2     k3
        #   q0 [ True  False  False  False ]   query 0 sees only key 0
        #   q1 [ True  True   False  False ]   query 1 sees keys 0..1
        #   q2 [ True  True   True   False ]   query 2 sees keys 0..2
        #   q3 [ True  True   True   True  ]   query 3 sees keys 0..3
        # The False (upper-triangle) entries are future keys and get -inf in SDPA.
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Per-head scaled dot-product attention -> (..., num_heads, seq_len, head_dim).
        attn = scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads back to (..., seq_len, d_model), then output projection.
        attn = rearrange(attn, "... h s d -> ... s (h d)")
        return self.output_proj(attn)
