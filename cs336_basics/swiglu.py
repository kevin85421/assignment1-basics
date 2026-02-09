import torch

def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU is a smooth activation function that is similar to ReLU, but has a smoother gradient.
    """
    return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # FFN(x) = SwiGLU(x, w1, w2, w3) = (silu(x @ w1^T) * (x @ w3^T)) @ w2^T
        #
        # Note: The transpose is just to be consistent with the shape of the weights defined in the adapter.
        #
        # x: (1, d_model)
        # w1: (d_ff, d_model)
        # w2: (d_model, d_ff)
        # w3: (d_ff, d_model)
        self.w1 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T
