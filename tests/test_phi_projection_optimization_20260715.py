import pytest
import torch

from vfe3.geometry.groups import GaugeGroup, get_group


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
