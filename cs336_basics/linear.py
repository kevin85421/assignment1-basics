import torch
import math
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Parameter Initialization: 
        #
        # "construct and store your parameter as W (not W ⊤) for memory ordering reasons, putting it in an nn.Parameter"
        # See section 3.3.1 and "Problem (linear): Implementing the linear module" for more details.
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        # Linear weights: N(µ = 0, σ^2 = 2 / (din + dout)) truncated at [-3σ, 3σ]. See section 3.4.1 for more details.
        std = math.sqrt(2 / (in_features + out_features))
        w = torch.nn.init.trunc_normal_(w, mean=0, std=std, a=-3 * std, b=3 * std)
        self.w = torch.nn.Parameter(w)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.w, x, "d_out d_in, batch seq_len d_in -> batch seq_len d_out")
