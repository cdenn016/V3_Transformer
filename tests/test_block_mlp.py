import pytest
import torch
from torch import nn

from vfe3.config import VFE3Config
from vfe3.model.block import vfe_block
from vfe3.model.block_mlp import BlockMLP
from vfe3.model.model import VFEModel
from vfe3.model.stack import vfe_stack


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

@pytest.mark.parametrize("activation", ["gelu", "silu", "relu"])
def test_block_mlp_supported_activations_forward_finite(activation):
    mlp = BlockMLP(embed_dim=3, expansion=2, activation=activation, dropout=0.0)
    mu = torch.tensor([[-1.0, 0.0, 2.0]])
    output = mlp(mu)

    assert output.shape == mu.shape
    assert torch.isfinite(output).all()


def _tiny_cfg(**overrides):
    config = dict(
        vocab_size=17,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_q_mu_lr=0.1,
        e_q_sigma_lr=0.02,
        e_phi_lr=0.0,
        use_prior_bank=True,
        norm_type_block="none",
    )
    config.update(overrides)
    return VFE3Config(**config)


def test_off_mode_registers_no_mlp_state_or_parameters():
    model = VFEModel(_tiny_cfg(use_block_mlp=False))

    assert model.block_mlps is None
    assert not any("block_mlps" in key for key in model.state_dict())
    assert not any("block_mlps" in name for name, _ in model.named_parameters())


def test_active_mlp_is_untied_per_layer():
    model = VFEModel(_tiny_cfg(use_block_mlp=True, n_layers=3))

    assert len(model.block_mlps) == 3
    assert len({id(module) for module in model.block_mlps}) == 3
    assert len({id(parameter) for module in model.block_mlps for parameter in module.parameters()}) == 12


class _Add(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return mu + self.amount


def test_block_applies_norm_then_mlp_before_handoff():
    cfg = _tiny_cfg(n_layers=2, prior_handoff_rho=1.0)
    model = VFEModel(cfg)
    tokens = torch.tensor([[1, 2, 3, 4]])
    belief = model.prior_bank.encode(tokens)
    block_mlps = nn.ModuleList([_Add(3.0), _Add(5.0)])
    diagnostic = {}

    vfe_stack(
        belief,
        belief.mu,
        belief.sigma,
        model.group,
        cfg,
        block_norm=lambda mu, sigma: mu + 2.0,
        block_mlps=block_mlps,
        capture={"diagnostic": diagnostic},
    )

    first_converged = diagnostic["layer_converged"][0]
    first_output = diagnostic["layer_outputs"][0]
    expected = block_mlps[0](first_converged.mu + 2.0)
    assert torch.equal(first_output.mu, expected)
    assert torch.equal(diagnostic["layer_priors"][1][0], first_output.mu)


def test_coordinate_mlp_preserves_covariance_and_other_belief_fields():
    for family, decode_mode in (("gaussian_diagonal", "diagonal_chunked"),
                                ("gaussian_full", "full_chunked")):
        cfg = _tiny_cfg(family=family, decode_mode=decode_mode, n_heads=1, embed_dim=2)
        model = VFEModel(cfg)
        belief = model.prior_bank.encode(torch.tensor([[1, 2]]))
        capture = {}

        out = vfe_block(
            belief,
            belief.mu,
            belief.sigma,
            model.group,
            cfg,
            block_norm=lambda mu, sigma: mu + 2.0,
            block_mlp=_Add(3.0),
            capture=capture,
        )

        converged = capture["converged"]
        assert torch.equal(out.mu, _Add(3.0)(converged.mu + 2.0))
        assert torch.equal(out.sigma, converged.sigma)
        assert out.phi is converged.phi
        assert out.s is converged.s
        assert out.r is converged.r
        assert out.omega is converged.omega
        assert out.reflection is converged.reflection
        assert out.right_phi is converged.right_phi


def test_active_mlp_gradients_are_finite_for_supported_estimators():
    tokens = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    for estimator in ("unroll", "straight_through"):
        model = VFEModel(_tiny_cfg(use_block_mlp=True, e_step_gradient=estimator))

        _, loss, _ = model(tokens, targets)
        loss.backward()

        for parameter in model.block_mlps[0].parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
