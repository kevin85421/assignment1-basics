import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply the softmax operation to a tensor along a given dimension.

    softmax(x)_i = exp(x_i) / sum_j exp(x_j), computed over `dim`.

    For numerical stability, subtract the max along `dim` before exponentiating
    (this does not change the result, since softmax is invariant to a constant
    shift of its inputs).

    Args:
        x (torch.Tensor): Input tensor of arbitrary shape.
        dim (int): The dimension along which to normalize.

    Returns:
        torch.Tensor: Tensor of the same shape as `x`, where the values along
        `dim` form a probability distribution (non-negative, sum to 1).
    """
    # Running example: x has shape (2, 3) and dim=1, so we normalize each row.
    #   x = [[1., 2., 3.],
    #        [1., 1., 1.]]

    # Subtract the per-slice max for numerical stability (softmax is shift-invariant).
    #   x.max(dim=1, keepdim=True).values = [[3.],   <- max of each row, shape (2, 1)
    #                                        [1.]]
    #   x - max = [[-2., -1., 0.],   <- broadcasts (2,1) across the row
    #              [ 0.,  0., 0.]]
    x = x - x.max(dim=dim, keepdim=True).values

    # Exponentiate every element (shape unchanged).
    #   exp_x = [[0.135, 0.368, 1.   ],
    #            [1.,    1.,    1.   ]]
    exp_x = torch.exp(x)

    # Sum the exponentials along `dim` to get each slice's normalizer.
    # keepdim=True keeps that axis as size 1 ((2, 3) -> (2, 1)) so it
    # broadcasts back against exp_x in the division below.
    #   denom = [[1.503],   <- 0.135 + 0.368 + 1.
    #            [3.   ]]    <- 1. + 1. + 1.
    denom = exp_x.sum(dim=dim, keepdim=True)

    # Divide each element by its slice's normalizer; broadcasting stretches the
    # size-1 `dim` of denom so all elements along `dim` share the same denominator.
    #   result = [[0.090, 0.245, 0.665],   <- each row now sums to 1
    #             [0.333, 0.333, 0.333]]
    return exp_x / denom
