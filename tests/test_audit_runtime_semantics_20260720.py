"""Runtime-boundary regressions for audit findings M5, M6, and M7."""

import math

import pytest
import torch
import torch.nn.functional as F

import vfe3.model.prior_bank as prior_bank_module
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import e_step
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank, get_decode_registration
from vfe3.train import evaluate


_FUSED_MODES = tuple(sorted(
    name for name, registration in prior_bank_module._DECODERS.items()
    if registration.fused_ce is not None
))


def _fused_case(
    mode: str,
) -> 'tuple[PriorBank, torch.Tensor, torch.Tensor]':
    V, K = 7, 3
    pb = PriorBank(
        V,
        K,
        4,
        use_prior_bank=mode != "linear",
        decode_mode=mode,
        decode_chunk_size=3,
    )
    mu_q = torch.randn(2, 3, K)
    sigma_q = torch.rand(2, 3, K) + 0.5
    if mode == "full_chunked":
        sigma_q = torch.diag_embed(sigma_q)
    return pb, mu_q, sigma_q


def _dense_ce(
    pb:      PriorBank,
    mu_q:    torch.Tensor,
    sigma_q: torch.Tensor,
    targets: torch.Tensor,

    *,
    mode:         str,
    ignore_index: int = -100,
) -> torch.Tensor:
    registration = get_decode_registration(mode)
    logits = registration.callable(pb, mu_q, sigma_q, pb._tau_eff())
    return F.cross_entropy(
        logits.reshape(-1, pb.vocab_size),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def _fused_ce(
    pb:      PriorBank,
    mu_q:    torch.Tensor,
    sigma_q: torch.Tensor,
    targets: torch.Tensor,

    *,
    mode:         str,
    ignore_index: int = -100,
) -> torch.Tensor:
    fused_ce = get_decode_registration(mode).fused_ce
    assert fused_ce is not None
    if mode == "linear":
        return fused_ce(
            pb,
            mu_q,
            targets,
            chunk_size=3,
            ignore_index=ignore_index,
        )
    return fused_ce(
        pb,
        mu_q,
        sigma_q,
        targets,
        chunk_size=3,
        ignore_index=ignore_index,
    )


@pytest.mark.parametrize("mode", _FUSED_MODES)
@pytest.mark.parametrize(
    "target_case",
    [
        pytest.param("below", id="target-minus-one"),
        pytest.param("above", id="target-vocab-size"),
        pytest.param("first", id="target-zero"),
        pytest.param("last", id="target-vocab-size-minus-one"),
        pytest.param("ignored", id="ignore-index-control"),
    ],
)
def test_registered_fused_ce_matches_dense_target_boundary(
    mode:        str,
    target_case: str,
) -> None:
    torch.manual_seed(0)
    pb, mu_q, sigma_q = _fused_case(mode)
    V = pb.vocab_size
    target_value = {
        "below":   -1,
        "above":   V,
        "first":   0,
        "last":    V - 1,
        "ignored": 0,
    }[target_case]
    targets = torch.full(mu_q.shape[:-1], target_value, dtype=torch.long)
    if target_case == "ignored":
        targets[0, 0] = -100

    if target_case in {"below", "above"}:
        with pytest.raises(Exception) as dense_error:
            _dense_ce(pb, mu_q, sigma_q, targets, mode=mode)
        with pytest.raises(type(dense_error.value)):
            _fused_ce(pb, mu_q, sigma_q, targets, mode=mode)
        return

    dense = _dense_ce(pb, mu_q, sigma_q, targets, mode=mode)
    fused = _fused_ce(pb, mu_q, sigma_q, targets, mode=mode)
    assert torch.allclose(fused, dense, atol=1e-3, rtol=0.0)


def _evaluation_model() -> VFEModel:
    cfg = VFE3Config(
        vocab_size=7,
        embed_dim=2,
        n_heads=1,
        max_seq_len=3,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
    )
    return VFEModel(cfg)


@pytest.mark.parametrize(
    "loader",
    [
        pytest.param([], id="empty-loader"),
        pytest.param(
            [(torch.zeros((2, 3), dtype=torch.long),
              torch.full((2, 3), -100, dtype=torch.long))],
            id="all-targets-ignored",
        ),
    ],
)
def test_evaluate_rejects_undefined_zero_token_metrics(
    loader: 'list[tuple[torch.Tensor, torch.Tensor]]',
) -> None:
    model = _evaluation_model()
    model.train()

    with pytest.raises(ValueError, match="no non-ignored target tokens.*undefined"):
        evaluate(model, loader, tokens_per_char=1.0)

    assert model.training


def _e_step_inputs() -> 'tuple[BeliefState, torch.Tensor, torch.Tensor, object]':
    group = get_group("glk")(2)
    generator = torch.Generator().manual_seed(7)
    belief = BeliefState(
        mu=torch.randn(3, 2, generator=generator),
        sigma=torch.rand(3, 2, generator=generator) + 0.5,
        phi=0.05 * torch.randn(3, group.generators.shape[0], generator=generator),
    )
    mu_p = torch.randn(3, 2, generator=generator)
    sigma_p = torch.rand(3, 2, generator=generator) + 0.5
    return belief, mu_p, sigma_p, group


@pytest.mark.parametrize(
    "malformed_depth",
    [
        pytest.param(0.5, id="fractional"),
        pytest.param(True, id="boolean"),
    ],
)
def test_e_step_rejects_non_plain_integer_backprop_depth(
    malformed_depth: object,
) -> None:
    belief, mu_p, sigma_p, group = _e_step_inputs()

    with pytest.raises(
        ValueError,
        match="e_steps_backprop_last must be a non-negative plain int",
    ):
        e_step(
            belief,
            mu_p,
            sigma_p,
            group,
            n_iter=2,
            e_phi_lr=0.0,
            e_steps_backprop_last=malformed_depth,
            training=True,
        )


@pytest.mark.parametrize("backprop_last", [0, 1])
def test_e_step_plain_integer_gradient_controls_remain_live(
    backprop_last: int,
) -> None:
    belief, mu_p, sigma_p, group = _e_step_inputs()
    mu_p = mu_p.requires_grad_(True)
    out = e_step(
        belief,
        mu_p,
        sigma_p,
        group,
        n_iter=2,
        e_phi_lr=0.0,
        e_steps_backprop_last=backprop_last,
        training=True,
    )
    out.mu.sum().backward()

    assert mu_p.grad is not None
    assert torch.isfinite(mu_p.grad).all()
    assert not math.isclose(float(mu_p.grad.abs().sum()), 0.0)
