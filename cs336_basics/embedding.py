import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        w = torch.nn.init.trunc_normal_(w, mean=0, std=1, a=-3, b=3)
        self.w = torch.nn.Parameter(w)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(token_ids.shape[0], token_ids.shape[1], self.w.shape[1], device=self.w.device, dtype=self.w.dtype)
        for row in range(token_ids.shape[0]):
            for col in range(token_ids.shape[1]):
                output[row][col] = self.w[token_ids[row][col]]
        return output
