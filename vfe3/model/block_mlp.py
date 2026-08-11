import torch
from torch import nn


_ACTIVATIONS = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}


def block_mlp_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown block MLP activation {name!r}") from exc


class BlockMLP(nn.Module):
    def __init__(self, embed_dim: int, expansion: int, activation: str, dropout: float) -> None:
        super().__init__()
        hidden_dim = embed_dim * expansion
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = block_mlp_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return mu + self.dropout(self.fc2(self.activation(self.fc1(mu))))
