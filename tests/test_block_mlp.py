import torch

from vfe3.model.block_mlp import BlockMLP


def test_block_mlp_known_residual_and_shape():
    """Identity linear maps make ReLU's residual output hand-checkable."""
    mlp = BlockMLP(embed_dim=2, expansion=1, activation="relu", dropout=0.0)
    with torch.no_grad():
        mlp.fc1.weight.copy_(torch.eye(2))
        mlp.fc1.bias.zero_()
        mlp.fc2.weight.copy_(torch.eye(2))
        mlp.fc2.bias.zero_()

    mu = torch.tensor([[-1.0, 2.0]])

    output = mlp(mu)

    assert output.shape == mu.shape
    assert torch.equal(output, torch.tensor([[-1.0, 4.0]]))
