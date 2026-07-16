import pytest
import torch

from vfe3.config import VFE3Config
import vfe3.gauge_optim as gauge_optim
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.model.model import VFEModel


def _four_phi_table_model() -> VFEModel:
    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=1,
        max_seq_len=4,
        n_layers=1,
        prior_source="model_channel",
        s_frame_mode="phi_tilde",
        s_e_step=True,
        lambda_h=1.0,
        lambda_gamma=1.0,
        pos_phi="learned",
    )
    return VFEModel(cfg)


def _phi_tables(model: VFEModel) -> list[torch.Tensor]:
    return [
        model.prior_bank.phi_embed,
        model.prior_bank.s_phi_embed,
        model.pos_phi_free,
        model.s_pos_phi_free,
    ]


def _dense_norm(phi: torch.Tensor, generators: torch.Tensor) -> torch.Tensor:
    embedded = torch.einsum("...a,aij->...ij", phi, generators)
    return torch.linalg.matrix_norm(embedded, ord="fro", dim=(-2, -1))


@pytest.mark.parametrize(
    "name, kwargs, expected_uniform",
    [
        ("glk", {}, 1.0),
        ("block_glk", {"n_heads": 2}, 1.0),
        (
            "block_glk",
            {"n_heads": 2, "cross_couplings": [(0, 1)]},
            1.0,
        ),
        ("tied_block_glk", {"n_heads": 2}, 2.0),
        ("so_k", {}, 2.0),
        ("sp", {}, None),
    ],
)
def test_certified_gram_diagonal_matches_dense_gram(
    name:             str,
    kwargs:           dict,
    expected_uniform: float | None,
) -> None:
    group = get_group(name)(K=4, dtype=torch.float64, **kwargs)
    gram = torch.einsum("aij,bij->ab", group.generators, group.generators)
    diagonal = gram.diagonal()

    torch.testing.assert_close(group.gram_diagonal(), diagonal)
    torch.testing.assert_close(
        gram - torch.diag(diagonal),
        torch.zeros_like(gram),
        rtol=0.0,
        atol=0.0,
    )
    assert group.gram_diagonal_uniform() == expected_uniform
    assert group.phi_norm_route() == "diagonal_gram"


def test_closed_basis_fails_closed_to_dense_route() -> None:
    group = get_group("block_glk")(
        K=4,
        n_heads=2,
        cross_couplings=[(0, 1)],
        close_basis=True,
    )

    assert group.gram_diagonal() is None
    assert group.gram_diagonal_uniform() is None
    assert group.phi_norm_route() == "dense_fallback"


def test_custom_group_defaults_to_dense_route() -> None:
    generators = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ]
    )
    group = GaugeGroup(
        name="custom_nonorthogonal",
        generators=generators,
        irrep_dims=[2],
        skew_symmetric=False,
    )

    assert group.gram_diagonal() is None
    assert group.gram_diagonal_uniform() is None
    assert group.phi_norm_route() == "dense_fallback"


def test_gram_diagonal_cache_refreshes_after_in_place_generator_change() -> None:
    group = get_group("glk")(K=2, dtype=torch.float64)
    first = group.gram_diagonal()
    assert first is group.gram_diagonal()

    group.generators.mul_(2.0)
    second = group.gram_diagonal()

    assert second is not first
    torch.testing.assert_close(second, torch.full_like(second, 4.0))
    assert group.gram_diagonal_uniform() == 4.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_diagonal_norm_kernel_matches_dense_embedding(dtype: torch.dtype) -> None:
    assert hasattr(gauge_optim, "embedded_phi_frobenius_norm")
    group = get_group("block_glk")(K=8, n_heads=2, dtype=dtype)
    generator = torch.Generator().manual_seed(17)
    phi = torch.randn(13, group.generators.shape[0], dtype=dtype, generator=generator)
    expected = _dense_norm(phi, group.generators)

    actual = gauge_optim.embedded_phi_frobenius_norm(phi, group)

    torch.testing.assert_close(actual, expected)


def test_nonuniform_diagonal_norm_kernel_matches_dense_embedding() -> None:
    assert hasattr(gauge_optim, "embedded_phi_frobenius_norm")
    group = get_group("sp")(K=4, dtype=torch.float64)
    phi = torch.randn(11, group.generators.shape[0], dtype=torch.float64)

    torch.testing.assert_close(
        gauge_optim.embedded_phi_frobenius_norm(phi, group),
        _dense_norm(phi, group.generators),
    )


def test_uncertified_nonorthogonal_group_uses_exact_dense_fallback() -> None:
    assert hasattr(gauge_optim, "embedded_phi_frobenius_norm")
    generators = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ]
    )
    group = GaugeGroup(
        name="custom_nonorthogonal",
        generators=generators,
        irrep_dims=[2],
        skew_symmetric=False,
    )
    phi = torch.randn(7, 2)

    with pytest.warns(RuntimeWarning, match="dense_fallback"):
        actual = gauge_optim.embedded_phi_frobenius_norm(phi, group)

    torch.testing.assert_close(actual, _dense_norm(phi, generators))


def test_silent_projection_matches_dense_oracle_and_returns_no_stats() -> None:
    model = _four_phi_table_model()
    generator = torch.Generator().manual_seed(23)
    expected_tables = []
    with torch.no_grad():
        for table in _phi_tables(model):
            table.copy_(torch.randn(table.shape, generator=generator) * 3.0)
            expected = table.detach().clone()
            norm = _dense_norm(expected, model.group.generators)
            scale = (2.0 / norm.clamp(min=1e-12)).clamp(max=1.0)
            expected.mul_(scale.unsqueeze(-1))
            expected_tables.append(expected)

    stats = gauge_optim.project_phi_parameter_rows_(
        model,
        2.0,
        chunk_rows=3,
        collect_stats=False,
    )

    assert stats == {}
    for table, expected in zip(_phi_tables(model), expected_tables):
        torch.testing.assert_close(table, expected)


def test_projection_metrics_match_dense_oracle() -> None:
    model = _four_phi_table_model()
    with torch.no_grad():
        for table in _phi_tables(model):
            table.fill_(5.0)
    preproject_max = max(
        float(_dense_norm(table, model.group.generators).max().detach())
        for table in _phi_tables(model)
    )
    total_rows = sum(table.numel() // table.shape[-1] for table in _phi_tables(model))

    stats = gauge_optim.project_phi_parameter_rows_(
        model,
        2.0,
        chunk_rows=3,
        collect_stats=True,
    )

    assert stats["phi_chart_projected_rows"] == total_rows
    assert stats["phi_chart_total_rows"] == total_rows
    assert stats["phi_chart_projected_fraction"] == 1.0
    assert stats["phi_chart_preproject_max"] == pytest.approx(preproject_max)
    assert 0.0 < stats["phi_chart_projection_scale_min"] < 1.0


def test_automatic_projection_chunks_preserve_below_bound_rows_exactly() -> None:
    model = _four_phi_table_model()
    with torch.no_grad():
        for table in _phi_tables(model):
            table.fill_(0.1)
    before = [table.detach().clone() for table in _phi_tables(model)]

    gauge_optim.project_phi_parameter_rows_(model, 2.0, collect_stats=False)

    for table, expected in zip(_phi_tables(model), before):
        torch.testing.assert_close(table, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("radius", [0.0, -1.0, float("inf"), float("nan")])
def test_projection_rejects_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="max_matrix_norm"):
        gauge_optim.project_phi_parameter_rows_(_four_phi_table_model(), radius)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_rows": 0},
        {"chunk_rows": 1.5},
        {"temporary_bytes": 0},
        {"temporary_bytes": True},
    ],
)
def test_projection_rejects_invalid_chunk_controls(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        gauge_optim.project_phi_parameter_rows_(_four_phi_table_model(), 2.0, **kwargs)


def test_projection_deduplicates_aliased_phi_tables() -> None:
    model = _four_phi_table_model()
    model.pos_phi_free = model.prior_bank.phi_embed
    unique_tables = {id(table): table for table in _phi_tables(model)}
    expected_rows = sum(table.numel() // table.shape[-1] for table in unique_tables.values())

    stats = gauge_optim.project_phi_parameter_rows_(model, 2.0)

    assert stats["phi_chart_total_rows"] == expected_rows
