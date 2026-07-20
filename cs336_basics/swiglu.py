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

        # Unlike the traditional FFN, which has only two matrices
        # (FFN(x) = ReLU(x @ W1^T) @ W2^T: up-project, activation, down-project),
        # GLU variants like SwiGLU use THREE matrices: w1 and w3 both up-project
        # x to d_ff (one goes through the activation, the other stays linear and
        # acts as a gate), and w2 down-projects back to d_model. The extra matrix
        # adds ~50% more FFN parameters, which is why SwiGLU models conventionally
        # use d_ff ~= (8/3) * d_model instead of 4 * d_model to keep the total
        # parameter count comparable.
        #
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
        # Two parallel up-projections of x, combined by element-wise multiply (*):
        #   a = silu(x @ w1.T)   (..., d_ff)  "content" path, with nonlinearity
        #   b = x @ w3.T         (..., d_ff)  "gate" path, purely linear
        #   a * b                (..., d_ff)  gating: b[i] scales a[i] per element
        #
        # Example (d_ff = 4): each b[i] acts like an independent valve on a[i]:
        #   a     = [0.8, -0.1, 2.3,  1.5]   content
        #   b     = [1.0,  0.0, 0.3, -2.0]   gate
        #   a * b = [0.8,  0.0, 0.69, -3.0]  pass through / block / attenuate / flip+amplify
        #
        # Each of the d_ff gate values is computed from x itself, so the input
        # decides how much of each feature passes through. w2 projects back to d_model.
        return (silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T
