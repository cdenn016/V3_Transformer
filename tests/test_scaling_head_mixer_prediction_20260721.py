"""Exact scaling-predictor coverage for active HeadMixer capacity."""

import warnings

import pytest

import scaling
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _cfg(
    gauge_group: str,
    use_head_mixer: bool,

    *,
    embed_dim: int = 6,
    n_heads: int = 3,
    group_n: int | None = None,
    irrep_spec: list[tuple[str, int]] | None = None,
) -> VFE3Config:
    return VFE3Config(
        vocab_size=11,
        embed_dim=embed_dim,
        n_heads=n_heads,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        gauge_group=gauge_group,
        group_n=group_n,
        irrep_spec=irrep_spec,
        phi_precond_mode="none",
        pos_phi="none",
        prior_source="token",
        use_prior_bank=False,
        decode_bias=False,
        lambda_h=0.0,
        lambda_gamma=0.0,
        s_e_step=False,
        use_head_mixer=use_head_mixer,
    )


def _realized_count(cfg: VFE3Config) -> int:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="HeadMixer:.*", category=UserWarning)
        model = VFEModel(cfg)
    return sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize("gauge_group", ["block_glk", "tied_block_glk"])
def test_predict_n_params_counts_equal_block_head_mixer_exactly(gauge_group: str) -> None:
    off = _cfg(gauge_group, False)
    on = _cfg(gauge_group, True)

    assert scaling.predict_n_params(off)[0] == _realized_count(off)
    assert scaling.predict_n_params(on)[0] == _realized_count(on)
    assert scaling.predict_n_params(on)[0] - scaling.predict_n_params(off)[0] == on.n_heads ** 2


@pytest.mark.parametrize(
    ("irrep_spec", "embed_dim", "expected_mixer_params"),
    [
        ([("l0", 1), ("l1", 2), ("l2", 1)], 12, 6),
        ([("l0", 1), ("l1", 1), ("l2", 1)], 9, 3),
        ([("l1", 1), ("l2", 1), ("l1", 1)], 11, 3),
    ],
    ids=["contiguous-multiplicity", "singleton-labels", "noncontiguous-label"],
)
def test_predict_n_params_counts_labeled_tower_mixer_runs_exactly(
    irrep_spec: list[tuple[str, int]],
    embed_dim: int,
    expected_mixer_params: int,
) -> None:
    off = _cfg(
        "so_n",
        False,
        embed_dim=embed_dim,
        n_heads=1,
        group_n=3,
        irrep_spec=irrep_spec,
    )
    on = _cfg(
        "so_n",
        True,
        embed_dim=embed_dim,
        n_heads=1,
        group_n=3,
        irrep_spec=irrep_spec,
    )

    assert scaling.predict_n_params(off)[0] == _realized_count(off)
    assert scaling.predict_n_params(on)[0] == _realized_count(on)
    assert scaling.predict_n_params(on)[0] - scaling.predict_n_params(off)[0] == expected_mixer_params


def test_predict_n_params_respects_single_block_auto_disable() -> None:
    with pytest.warns(UserWarning, match="auto-disabling use_head_mixer"):
        cfg = _cfg("block_glk", True, embed_dim=4, n_heads=1)

    assert cfg.use_head_mixer is False
    assert scaling.predict_n_params(cfg)[0] == _realized_count(cfg)


def test_current_scaling_baseline_prediction_matches_realized_model() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="decode_tau=.* is inert when use_prior_bank=False", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message="e_step_update='mm_exact'.*", category=UserWarning)
        cfg = VFE3Config(**scaling.BASELINE)

    assert cfg.use_head_mixer is True
    assert scaling.predict_n_params(cfg)[0] == _realized_count(cfg)
