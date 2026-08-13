from typing import NamedTuple, Optional, Sequence

import torch
from torch import nn

from vfe3.contracts import CanonicalFrameContext


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


def _activation_derivative(name: str, value: torch.Tensor) -> torch.Tensor:
    if name == "relu":
        return (value > 0).to(dtype=value.dtype)
    if name == "silu":
        sigmoid = torch.sigmoid(value)
        return sigmoid * (1.0 + value * (1.0 - sigmoid))
    if name == "gelu":
        inv_sqrt_two = value.new_tensor(2.0).rsqrt()
        inv_sqrt_two_pi = value.new_tensor(2.0 * torch.pi).rsqrt()
        return 0.5 * (1.0 + torch.erf(value * inv_sqrt_two)) \
            + value * torch.exp(-0.5 * value.square()) * inv_sqrt_two_pi
    raise RuntimeError(f"unsupported block MLP activation {name!r}")


def _dropout_scale(dropout: nn.Dropout, reference: torch.Tensor) -> torch.Tensor:
    if not dropout.training or dropout.p == 0.0:
        return torch.ones_like(reference)
    return torch.nn.functional.dropout(
        torch.ones_like(reference), p=dropout.p, training=True)


class BlockMLPMomentResult(NamedTuple):
    """Post-MLP belief moments and, when requested, the mean-map Jacobian."""

    mu: torch.Tensor
    sigma: torch.Tensor
    jacobian: Optional[torch.Tensor]


def _delta_covariance(
    jacobian: torch.Tensor,
    sigma: torch.Tensor,
    covariance_floor: float,
) -> torch.Tensor:
    js = torch.einsum("...ij,...jk->...ik", jacobian, sigma)
    jsj = torch.einsum("...ik,...lk->...il", js, jacobian)
    pushed = jsj + covariance_floor * sigma
    return 0.5 * (pushed + pushed.transpose(-1, -2))


class BlockMLP(nn.Module):
    """Ordinary coordinate residual MLP, retained as the backward-compatible default."""

    def __init__(
        self,
        embed_dim: int,
        expansion: int,
        activation: str,
        dropout: float,
        *,
        covariance_contract: str = "passthrough",
        covariance_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        hidden_dim = embed_dim * expansion
        self.activation_name = activation
        self.covariance_contract = covariance_contract
        self.covariance_floor = float(covariance_floor)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = block_mlp_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return mu + self.dropout(self.fc2(self.activation(self.fc1(mu))))

    def _coordinate_update_with_jacobian(
        self,
        mu: torch.Tensor,
    ) -> 'tuple[torch.Tensor, torch.Tensor]':
        hidden_pre = self.fc1(mu)
        residual = self.fc2(self.activation(hidden_pre))
        dropout_scale = _dropout_scale(self.dropout, residual)
        mu_out = mu + dropout_scale * residual
        residual_jacobian = torch.einsum(
            "oh,...h,hi->...oi",
            self.fc2.weight,
            _activation_derivative(self.activation_name, hidden_pre),
            self.fc1.weight,
        )
        residual_jacobian = dropout_scale.unsqueeze(-1) * residual_jacobian
        identity = torch.eye(mu.shape[-1], dtype=mu.dtype, device=mu.device)
        return mu_out, identity + residual_jacobian

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame: Optional[CanonicalFrameContext] = None,
    ) -> BlockMLPMomentResult:
        del frame
        if self.covariance_contract == "passthrough":
            return BlockMLPMomentResult(self.forward(mu), sigma, None)
        mu_out, jacobian = self._coordinate_update_with_jacobian(mu)
        return BlockMLPMomentResult(
            mu_out, _delta_covariance(jacobian, sigma, self.covariance_floor), jacobian)


class GaugeGateBlockMLP(nn.Module):
    """Gauge-equivariant residual gate built from Mahalanobis invariants.

    The deterministic/evaluation path and every fixed realization of the scalar per-block
    dropout mask are exactly equivariant. Independent dropout draws are equivariant in
    distribution, not identical sample by sample.

    For every declared irrep block h, s_h = mu_h^T Sigma_hh^-1 mu_h is invariant under the
    corresponding invertible frame action. A scalar MLP maps the invariant vector s to one
    gate per block, and that scalar multiplies every coordinate in the block.
    """

    def __init__(
        self,
        irrep_dims: Sequence[int],
        expansion: int,
        activation: str,
        dropout: float,
        *,
        covariance_contract: str = "passthrough",
        covariance_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.irrep_dims = tuple(int(dim) for dim in irrep_dims)
        if not self.irrep_dims or any(dim < 1 for dim in self.irrep_dims):
            raise ValueError(f"irrep_dims must contain positive dimensions, got {irrep_dims!r}")
        n_blocks = len(self.irrep_dims)
        hidden_dim = n_blocks * expansion
        self.activation_name = activation
        self.covariance_contract = covariance_contract
        self.fc1 = nn.Linear(n_blocks, hidden_dim)
        self.covariance_floor = float(covariance_floor)
        self.activation = block_mlp_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, n_blocks)
        self.dropout = nn.Dropout(dropout)
        # Identity at step zero keeps this opt-in transform from perturbing a baseline before its
        # invariant gate has learned evidence for a nonzero update.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _invariants(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> 'tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]':
        mu_blocks = []
        expected_dim = sum(self.irrep_dims)
        if mu.shape[-1] != expected_dim:
            raise ValueError(
                "irrep_dims must sum to the mean dimension, got "
                f"sum(irrep_dims)={expected_dim} and mean dimension={mu.shape[-1]}"
            )
        if sigma.shape[-2:] != (expected_dim, expected_dim):
            raise ValueError(
                "gauge gate covariance axes must match sum(irrep_dims), got "
                f"{tuple(sigma.shape[-2:])} and expected {(expected_dim, expected_dim)}"
            )
        sigma_solutions = []
        invariants = []
        start = 0
        for dim in self.irrep_dims:
            stop = start + dim
            mu_h = mu[..., start:stop]
            sigma_hh = sigma[..., start:stop, start:stop]
            solved = torch.linalg.solve(sigma_hh, mu_h.unsqueeze(-1)).squeeze(-1)
            mu_blocks.append(mu_h)
            sigma_solutions.append(solved)
            invariants.append((mu_h * solved).sum(dim=-1))
            start = stop
        return torch.stack(invariants, dim=-1), mu_blocks, sigma_solutions

    def _gate_values(
        self,
        invariants: torch.Tensor,
        *,
        need_jacobian: bool,
    ) -> 'tuple[torch.Tensor, Optional[torch.Tensor]]':
        hidden_pre = self.fc1(invariants)
        raw_gates = self.fc2(self.activation(hidden_pre))
        if not need_jacobian:
            return self.dropout(raw_gates), None
        dropout_scale = _dropout_scale(self.dropout, raw_gates)
        dgate_ds = torch.einsum(
            "oh,...h,hi->...oi",
            self.fc2.weight,
            _activation_derivative(self.activation_name, hidden_pre),
            self.fc1.weight,
        )
        return dropout_scale * raw_gates, dropout_scale.unsqueeze(-1) * dgate_ds

    def mean_update(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        invariants, mu_blocks, _ = self._invariants(mu, sigma)
        gates, _ = self._gate_values(invariants, need_jacobian=False)
        return torch.cat([
            (1.0 + gates[..., index:index + 1]) * mu_h
            for index, mu_h in enumerate(mu_blocks)
        ], dim=-1)

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame: Optional[CanonicalFrameContext] = None,
    ) -> BlockMLPMomentResult:
        del frame
        invariants, mu_blocks, sigma_solutions = self._invariants(mu, sigma)
        need_jacobian = self.covariance_contract == "delta_full"
        gates, dgate_ds = self._gate_values(invariants, need_jacobian=need_jacobian)
        mu_out = torch.cat([
            (1.0 + gates[..., index:index + 1]) * mu_h
            for index, mu_h in enumerate(mu_blocks)
        ], dim=-1)
        if not need_jacobian:
            return BlockMLPMomentResult(mu_out, sigma, None)

        K = mu.shape[-1]
        jacobian = torch.zeros(*mu.shape[:-1], K, K, dtype=mu.dtype, device=mu.device)
        row_start = 0
        for out_index, (out_dim, mu_h) in enumerate(zip(self.irrep_dims, mu_blocks)):
            row_stop = row_start + out_dim
            jacobian[..., row_start:row_stop, row_start:row_stop] = \
                (1.0 + gates[..., out_index]).unsqueeze(-1).unsqueeze(-1) \
                * torch.eye(out_dim, dtype=mu.dtype, device=mu.device)
            col_start = 0
            for in_index, (in_dim, solved) in enumerate(
                    zip(self.irrep_dims, sigma_solutions)):
                col_stop = col_start + in_dim
                cross = torch.einsum(
                    "...i,...,...j->...ij",
                    mu_h,
                    dgate_ds[..., out_index, in_index],
                    2.0 * solved,
                )
                jacobian[..., row_start:row_stop, col_start:col_stop] = \
                    jacobian[..., row_start:row_stop, col_start:col_stop] + cross
                col_start = col_stop
            row_start = row_stop
        return BlockMLPMomentResult(
            mu_out, _delta_covariance(jacobian, sigma, self.covariance_floor), jacobian)


class CanonicalFrameBlockMLP(BlockMLP):
    """Ordinary residual MLP in a realized, frame-fixed vertex coordinate system."""

    def mean_update(
        self,
        mu: torch.Tensor,
        frame: CanonicalFrameContext,
    ) -> torch.Tensor:
        canonical_mu = torch.einsum("...ij,...j->...i", frame.inverse, mu)
        canonical_out = super().forward(canonical_mu)
        return torch.einsum("...ij,...j->...i", frame.forward, canonical_out)

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame: Optional[CanonicalFrameContext] = None,
    ) -> BlockMLPMomentResult:
        if frame is None:
            raise ValueError("canonical_frame block MLP requires a realized canonical frame")
        canonical_mu = torch.einsum("...ij,...j->...i", frame.inverse, mu)
        if self.covariance_contract == "passthrough":
            canonical_out = super().forward(canonical_mu)
            mu_out = torch.einsum("...ij,...j->...i", frame.forward, canonical_out)
            return BlockMLPMomentResult(mu_out, sigma, None)
        canonical_out, canonical_jacobian = self._coordinate_update_with_jacobian(canonical_mu)
        mu_out = torch.einsum("...ij,...j->...i", frame.forward, canonical_out)
        realized_jacobian = frame.forward @ canonical_jacobian @ frame.inverse
        return BlockMLPMomentResult(
            mu_out,
            _delta_covariance(realized_jacobian, sigma, self.covariance_floor),
            realized_jacobian,
        )


def build_block_mlp(
    mode: str,
    *,
    embed_dim: int,
    irrep_dims: Sequence[int],
    expansion: int,
    activation: str,
    dropout: float,
    covariance_contract: str,
    covariance_floor: float,
) -> nn.Module:
    if mode == "coordinate":
        return BlockMLP(
            embed_dim, expansion, activation, dropout,
            covariance_contract=covariance_contract,
            covariance_floor=covariance_floor,
        )
    if mode == "gauge_gate":
        return GaugeGateBlockMLP(
            irrep_dims, expansion, activation, dropout,
            covariance_contract=covariance_contract,
            covariance_floor=covariance_floor,
        )
    if mode == "canonical_frame":
        return CanonicalFrameBlockMLP(
            embed_dim, expansion, activation, dropout,
            covariance_contract=covariance_contract,
            covariance_floor=covariance_floor,
        )
    raise ValueError(f"unknown block MLP mode {mode!r}")
