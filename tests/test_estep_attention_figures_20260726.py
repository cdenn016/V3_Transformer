r"""The E-step attention figure set (audit 2026-07-26 R-3).

The shipped ``attention/`` heatmaps score beta at each block's OUTPUT belief -- after the E-step has
moved it and after head_mixer/block_norm -- so they are a post-hoc re-score, not the pattern the
E-step aggregated with. Measured max row difference on a trained checkpoint: 1.0. The new
``attention_estep/`` set scores at the ENTRY belief and is written BESIDE the existing one; the
property under test throughout is that the old set is untouched and the new one measures a
different, correctly-located quantity.

Every model here is TINY (K = 4, single-digit everything) and CPU-bound per CLAUDE.md.
"""

import json

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts


def _model(**overrides) -> VFEModel:
    cfg = VFE3Config(
        embed_dim=4, n_heads=2, vocab_size=11, max_seq_len=6, batch_size=2,
        n_e_steps=2, max_steps=1, lambda_alpha_mode="constant",
        **{"n_layers": 1, **overrides},
    )
    torch.manual_seed(0)
    return VFEModel(cfg)


def _tokens() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 11, (1, 6))


# ------------------------------------------------------------------ the scoring point


def test_every_scoring_point_returns_a_row_stochastic_map() -> None:
    model, tokens = _model(), _tokens()

    for score_at in ("converged", "entry", "prior"):
        with torch.no_grad():
            beta = model.attention_maps(tokens, score_at=score_at)
        assert beta.shape == (1, 2, 6, 6), score_at
        assert torch.allclose(beta.sum(-1), torch.ones_like(beta.sum(-1)), atol=1e-5), score_at


def test_entry_and_converged_are_different_quantities() -> None:
    r"""If these ever agree the new figure set is measuring nothing (audit 2026-07-26 R-3)."""
    model, tokens = _model(), _tokens()

    with torch.no_grad():
        entry = model.attention_maps(tokens, score_at="entry")
        converged = model.attention_maps(tokens, score_at="converged")

    assert not torch.equal(entry, converged)


def test_converged_is_byte_identical_to_the_shipped_default() -> None:
    r"""The default path must be untouched: score_at is additive, not a redefinition."""
    model, tokens = _model(), _tokens()

    with torch.no_grad():
        default = model.attention_maps(tokens)
        explicit = model.attention_maps(tokens, score_at="converged")

    assert torch.equal(default, explicit)


def test_prior_arm_is_independent_of_the_belief_content() -> None:
    r"""Zeroing the coupling energy must leave a map that no token choice can move.

    Pinned because the prior arm is the reference the profile panel reads 'how much of this map is
    content' against; if it tracked content at all that comparison would be meaningless.
    """
    model = _model(gamma_as_beta_prior=False, precision_weighted_attention=False)
    torch.manual_seed(1)
    first, second = torch.randint(0, 11, (1, 6)), torch.randint(0, 11, (1, 6))
    assert not torch.equal(first, second)

    with torch.no_grad():
        a = model.attention_maps(first, score_at="prior")
        b = model.attention_maps(second, score_at="prior")

    assert torch.equal(a, b)


def test_scoring_point_rejects_an_unknown_value() -> None:
    model, tokens = _model(), _tokens()

    with pytest.raises(ValueError, match="score_at"):
        model.attention_maps(tokens, score_at="postblock")


def test_snapshot_refuses_the_non_converged_arms() -> None:
    r"""A snapshot stores ONLY the converged maps; silently returning them for score_at='entry'
    would republish the exact defect this set exists to correct."""
    model, tokens = _model(), _tokens()
    with torch.no_grad():
        snapshot = model.build_diagnostic_snapshot(tokens)

    with pytest.raises(ValueError, match="snapshot"):
        model.attention_maps(tokens, score_at="entry", snapshot=snapshot)


# ------------------------------------------------------------------ the artifact


def _stacked(model: VFEModel, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack([model.attention_maps(tokens, score_at=s)
                            for s in ("entry", "converged", "prior")])


def test_figures_land_in_their_own_directory_and_leave_attention_alone(tmp_path) -> None:
    model, tokens = _model(), _tokens()
    artifacts = RunArtifacts(tmp_path / "run", model.cfg, model)

    paths = artifacts.save_estep_attention_maps(0, _stacked(model, tokens))

    assert paths is not None, "figure worker failed"
    written = sorted(p.name for p in (tmp_path / "run" / "attention_estep").iterdir())
    assert written == ["step_0_layer0_head0.png", "step_0_layer0_head1.png", "step_0_profile.png"]
    assert not (tmp_path / "run" / "attention").exists()   # the shipped set is never touched
    for path in paths:
        assert path.is_file() and path.stat().st_size > 0


def test_none_maps_is_a_noop(tmp_path) -> None:
    model = _model()
    artifacts = RunArtifacts(tmp_path / "run", model.cfg, model)

    assert artifacts.save_estep_attention_maps(0, None) is None
    assert not (tmp_path / "run" / "attention_estep").exists()


def test_wrong_rank_is_rejected_before_a_worker_is_spawned(tmp_path) -> None:
    r"""The (3, L, H, N, N) contract is what pairs the three arms; a rank slip must not render."""
    model, tokens = _model(), _tokens()
    artifacts = RunArtifacts(tmp_path / "run", model.cfg, model)

    with torch.no_grad():
        four_d = model.attention_maps(tokens)              # (L, H, N, N), missing the arm axis

    assert artifacts.save_estep_attention_maps(0, four_d) is None
    assert not (tmp_path / "run" / "attention_estep").exists()


# ------------------------------------------------------------------ the distance profile


def test_distance_profile_averages_each_lag_over_its_own_support() -> None:
    r"""Lag d is present in N-d rows; dividing by N instead would damp the tail and hide exactly
    the long-range revivals the panel exists to surface."""
    from vfe3.viz.figure_worker import _distance_profile

    n = 5
    beta = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            beta[i, j] = 1.0                                # every causal entry equal

    profile = _distance_profile(beta)

    assert profile.shape == (n,)
    # constant on the causal band -> every well-defined lag averages to exactly 1.0
    assert torch.allclose(profile, torch.ones(n, dtype=profile.dtype))


def test_distance_profile_ignores_the_acausal_upper_triangle() -> None:
    from vfe3.viz.figure_worker import _distance_profile

    n = 6
    beta = torch.zeros(n, n)
    beta[0, -1] = 5.0                                       # a future key: negative lag
    beta[3, 1] = 2.0                                        # lag 2, one of four rows having it

    profile = _distance_profile(beta)

    assert float(profile[2]) == pytest.approx(2.0 / 4.0)
    assert float(profile[0]) == 0.0                         # the acausal mass never enters
