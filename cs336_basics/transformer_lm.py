import torch
from torch import nn

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Transformer language model (assignment section 3.6, Figure 1).

        Pipeline: token ids -> token embeddings -> num_layers Transformer
        blocks -> final RMSNorm -> lm_head projection -> unnormalized logits
        over the vocabulary. (No softmax here; the loss / decoder applies it.)

        Args:
            vocab_size (int): Size of the vocabulary; determines the token
                embedding matrix and the lm_head output dimensionality.
            context_length (int): Maximum context length; determines how many
                positions RoPE pre-caches.
            d_model (int): Dimensionality of the model embeddings and sublayer outputs.
            num_layers (int): Number of Transformer blocks.
            num_heads (int): Number of attention heads per block. `d_model` must be
                evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer.
            rope_theta (float): The RoPE Theta parameter.
            device (torch.device | None): Device for the model's weights.
            dtype (torch.dtype | None): Dtype for the model's weights.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # RoPE operates per head, so its dimension is head_dim (= d_k), not
        # d_model. One instance is shared by all blocks: it has no learnable
        # parameters and the cos/sin cache depends only on (theta, d_k,
        # context_length).
        rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)

        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass of the language model.

        Args:
            in_indices (torch.Tensor): Int tensor of token ids with shape
                (batch_size, seq_len), where seq_len <= context_length.

        Returns:
            torch.Tensor: Unnormalized next-token logits of shape
                (batch_size, seq_len, vocab_size).

        Hints:
            - Embed the token ids: (batch, seq_len) -> (batch, seq_len, d_model).
            - Pass through each block in self.layers in order (each block
              defaults token_positions to 0..seq_len-1 for RoPE).
            - Apply self.ln_final, then self.lm_head to get logits.
        """
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
