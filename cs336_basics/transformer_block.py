import torch
from torch import nn

from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Pre-norm Transformer block (assignment section 3.6, Figure 2).

        The block has two 'sublayers'. In each sublayer we first apply RMSNorm,
        then the main operation (multi-head self-attention / feed-forward
        network), and finally add the residual connection:

            y = x + MultiHeadSelfAttention(RMSNorm(x))    # sublayer 1, Eq. (15)
            z = y + FFN(RMSNorm(y))                       # sublayer 2

        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
            rope (RotaryPositionalEmbedding | None): Optional RoPE module passed
                down to the attention sublayer. The same instance can be shared
                across all blocks of a model, since RoPE has no learnable state.
            device (torch.device | None): Device for the block's weights.
            dtype (torch.dtype | None): Dtype for the block's weights.
        """
        super().__init__()
        # Sublayer 1: RMSNorm -> causal multi-head self-attention (+ residual).
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        # Sublayer 2: RMSNorm -> SwiGLU feed-forward network (+ residual).
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply the pre-norm Transformer block to `x`.

        Args:
            x (torch.Tensor): Input of shape (..., seq_len, d_model).
            token_positions (torch.Tensor | None): Positions of shape (..., seq_len),
                forwarded to the attention sublayer for RoPE. If None, attention
                defaults to 0..seq_len-1.

        Returns:
            torch.Tensor: Output of shape (..., seq_len, d_model).

        """
        # Sublayer 1: y = x + MHA(RMSNorm(x)). Pre-norm: only the attention
        # input is normalized; the residual adds the unnormalized x.
        y = x + self.attn(self.ln1(x), token_positions)
        # Sublayer 2: z = y + FFN(RMSNorm(y)).
        return y + self.ffn(self.ln2(y))
