import torch
import math
from einops import einsum

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Initialize the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model.
            eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-5.
            device (torch.device | None, optional): The device to use for the RMSNorm weights. Defaults to None.
            dtype (torch.dtype | None, optional): The dtype to use for the RMSNorm weights. Defaults to None.
        """
        
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        # Initialize the gain parameter. The adapter will set the gain to the weights.
        self.gain = torch.nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward method should apply RMSNorm to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, d_model).
        """
        in_dtype = x.dtype
        # Upcast to float32 to avoid overflows when squaring the input.
        x = x.to(torch.float32)

        # RMSNorm is token-wise, so we calculate the RMS and normalize each token independently.
        #
        # RMS = sqrt(1/d_model * sum(a^2) + eps)
        result = torch.zeros_like(x)
        for batch_idx in range(x.shape[0]):
            for seq_idx in range(x.shape[1]):
                rms = torch.sqrt(torch.mean(x[batch_idx, seq_idx] ** 2) + self.eps)
                result[batch_idx, seq_idx] = (x[batch_idx, seq_idx] / rms) * self.gain
        return result.to(in_dtype)
