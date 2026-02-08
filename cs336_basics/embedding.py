import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Initialize the Embedding module.

        Args:
            num_embeddings (int): The size of the vocabulary.
            embedding_dim (int): The size of the embedding vectors, i.e. `d_model`.
            device (torch.device | None, optional): The device to use for the embedding matrix. Defaults to None.
            dtype (torch.dtype | None, optional): The dtype to use for the embedding matrix. Defaults to None.
        """
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        w = torch.nn.init.trunc_normal_(w, mean=0, std=1, a=-3, b=3)
        self.w = torch.nn.Parameter(w)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        The forward method should select the embedding
        vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a
        torch.LongTensor of token IDs with shape (batch_size, sequence_length).

        Example:

        # weights: d_model = 3, vocab_size = 5
        self.w = [
            [0.1, 0.2, 0.3], # token_id 0
            [0.4, 0.5, 0.6], # token_id 1
            [0.7, 0.8, 0.9], # token_id 2
            [1.0, 1.1, 1.2], # token_id 3
            [1.3, 1.4, 1.5], # token_id 4
        ]

        # batch_size = 2, sequence_length = 2
        token_ids = [[2, 0], [4, 1]]

        # mapping token_ids to embeddings
        output = [
            [
                [0.7, 0.8, 0.9], # token_id 2
                [0.1, 0.2, 0.3], # token_id 0
            ],
            [
                [1.3, 1.4, 1.5], # token_id 4
                [0.4, 0.5, 0.6], # token_id 1
            ],
        ]
        """
        output = torch.zeros(token_ids.shape[0], token_ids.shape[1], self.w.shape[1], device=self.w.device, dtype=self.w.dtype)
        for row in range(token_ids.shape[0]):
            for col in range(token_ids.shape[1]):
                output[row][col] = self.w[token_ids[row][col]]
        return output
