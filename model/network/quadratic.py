import torch
import torch.nn as nn

from typing import Tuple

from .mlp import MLP


class QuadraticMLP(nn.Module):
    """
    Quadratic multi-layer perceptron: y = NN(x) + x^T * Q * x. Note that output dim is 1.
    Q and NN are trainable.

    Parameters
    ----------
    in_dim: int
    hidden_layers: tuple
    hidden_activation: nn.Module
    output_activation: nn.Module
    init: bool
        init the MLP to be orthogonal
    gain: float
    """
    def __init__(self, in_dim: int, hidden_layers: tuple,
                 hidden_activation: nn.Module = nn.ReLU(), output_activation: nn.Module = None,
                 init: bool = True, gain: float = 1.):
        super(QuadraticMLP, self).__init__()
        self.nn = MLP(
            in_dim, 1, hidden_layers, hidden_activation, output_activation, init, gain
        )
        self.Q = nn.Parameter(torch.eye(in_dim) + 0.1 * torch.ones((in_dim, in_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x) + self.quadratic(x)

    def quadratic(self, x: torch.Tensor) -> torch.Tensor:
        def symmetric(Q: torch.Tensor):
            return 0.5 * (Q.t() + Q)
        return torch.bmm(torch.matmul(x, symmetric(self.quadratic_matrix)).unsqueeze(1), x.unsqueeze(2)).squeeze(2)

    @property
    def quadratic_matrix(self) -> torch.Tensor:
        def symmetric(Q: torch.Tensor):
            return 0.5 * (Q.t() + Q)
        return symmetric(self.Q)

    def forward_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the forward value and the jacobian"""
        eye = torch.eye(x.shape[1]).type_as(x)
        jacobian = eye.repeat(x.shape[0], 1, 1)

        # compute layer by layer
        y = x
        for layer in self.nn.net:
            y = layer(y)

            if isinstance(layer, nn.Linear):
                jacobian = torch.matmul(layer.weight, jacobian)
            elif isinstance(layer, nn.Tanh):
                jacobian = torch.matmul(torch.diag_embed(1 - y ** 2), jacobian)
            elif isinstance(layer, nn.ReLU):
                jacobian = torch.matmul(torch.diag_embed(torch.sign(y)), jacobian)

        # add the quadratic part
        y += self.quadratic(x)
        jacobian += 2 * torch.matmul(x, self.quadratic_matrix).unsqueeze(1)

        return y, jacobian

    def disable_grad(self):
        for i in self.parameters():
            i.requires_grad = False
