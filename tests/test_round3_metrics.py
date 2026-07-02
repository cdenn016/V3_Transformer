"""Round-3 audit pins for vfe3.metrics (2026-07-01): fail-closed metric registry and the
unequal-irrep-tower Wilson decomposition (``irrep_dims``) of holonomy_wilson_sampled."""

import pytest
import torch

from vfe3.metrics import get_metric, holonomy_wilson_sampled, register_metric


# --- registry guard ---------------------------------------------------------

def test_register_metric_duplicate_fails_closed_and_override_replaces():
    original = get_metric("attention_entropy")
    try:
        with pytest.raises(KeyError):
            @register_metric("attention_entropy")
            def _dup(**kw):                                        # noqa: unused -- registration must fail
                return 0.0
        assert get_metric("attention_entropy") is original         # failed registration left registry intact

        @register_metric("attention_entropy", override=True)
        def _replacement(**kw):
            return -1.0
        assert get_metric("attention_entropy") is _replacement     # override replaces deliberately
    finally:
        register_metric("attention_entropy", override=True)(original)
    assert get_metric("attention_entropy") is original


# --- Wilson per-block decomposition on an unequal irrep tower ---------------

def _constant_diag_omega(d: torch.Tensor, n: int) -> torch.Tensor:
    # Omega_ij = diag(d) for EVERY pair, so every sampled triangle has the SAME holonomy
    # H = diag(d)^3 = diag(d**3) -- the sampled estimate is deterministic regardless of triples.
    return torch.diag_embed(d).expand(n, n, d.shape[0], d.shape[0]).contiguous()


def test_holonomy_wilson_unequal_tower_per_block_traces():
    # K=8 tower with blocks [3, 5] (so_n/sp_n-style direct sum); distinct diagonal entries so the
    # equal-chunk 4/4 split would mis-slice and give DIFFERENT numbers than the true blocks.
    d = torch.tensor([1.1, 1.0, 0.9, 1.2, 0.8, 1.05, 0.95, 1.0])
    omega = _constant_diag_omega(d, n=5)
    out = holonomy_wilson_sampled(omega, n_heads=2, n_triples=32, seed=0, irrep_dims=[3, 5])
    h_diag = d ** 3                                                # diagonal of H = Omega^3
    expected = torch.stack([h_diag[:3].mean(), h_diag[3:].mean()]) # per-block Tr(H_b)/d_b
    assert out["per_head"].shape == (2,)
    assert torch.allclose(out["per_head"], expected, atol=1e-5)
    # d_b-weighted mean of per-block values recovers the full W/K
    weighted = (3.0 * out["per_head"][0] + 5.0 * out["per_head"][1]) / 8.0
    assert torch.allclose(weighted, out["wilson_mean"], atol=1e-5)
    # the equal-chunk 4/4 decomposition mis-slices this tower (different numbers)
    out_eq = holonomy_wilson_sampled(omega, n_heads=2, n_triples=32, seed=0)
    assert not torch.allclose(out_eq["per_head"], expected, atol=1e-3)


def test_holonomy_wilson_equal_tower_matches_default_path():
    # irrep_dims=None keeps the old equal-chunk numbers; an EQUAL tower [4, 4] reproduces them.
    d = torch.tensor([1.1, 1.0, 0.9, 1.2, 0.8, 1.05, 0.95, 1.0])
    omega = _constant_diag_omega(d, n=5)
    out_none = holonomy_wilson_sampled(omega, n_heads=2, n_triples=32, seed=0)
    h_diag = d ** 3
    expected = torch.stack([h_diag[:4].mean(), h_diag[4:].mean()]) # old equal-chunk Tr(H_h)/d_k
    assert torch.allclose(out_none["per_head"], expected, atol=1e-5)
    out_44 = holonomy_wilson_sampled(omega, n_heads=2, n_triples=32, seed=0, irrep_dims=[4, 4])
    assert torch.allclose(out_44["per_head"], out_none["per_head"], atol=1e-6)
    assert torch.allclose(out_44["wilson_mean"], out_none["wilson_mean"], atol=1e-6)


def test_holonomy_wilson_irrep_dims_validation():
    omega = torch.eye(8).expand(5, 5, 8, 8).contiguous()
    with pytest.raises(ValueError):
        holonomy_wilson_sampled(omega, n_heads=2, irrep_dims=[3, 4])       # sum 7 != K=8
    with pytest.raises(ValueError):
        holonomy_wilson_sampled(omega, n_heads=3, irrep_dims=[3, 5])       # len 2 != n_heads=3
