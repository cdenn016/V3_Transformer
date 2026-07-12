r"""Audit finding V2: silent out-of-span truncation on a non-bracket-closed gauge basis.

When ``cross_couplings`` chains span 3+ distinct heads, the raw chain generator basis
is NOT closed under the Lie bracket: the (0->1) x (1->2) bracket produces an out-of-span
E_{0->2} block. Two consumers silently truncate that component:
  - ``_structure_constants`` (phi_preconditioner) projects [G_a,G_b] onto the span, so the
    pullback / pullback_per_block natural-grad metric is built on truncated constants;
  - ``compose_bch`` (lie_ops) returns ``extract_phi(Z)``, a least-squares projection that
    discards the out-of-span BCH commutator terms.
These tests pin the new SILENT-ON-CLOSED-BASIS diagnostic guards: a non-closed basis warns
from both consumers, the SAME chain with ``close_basis=True`` (genuinely bracket-closed) is
silent, and the DEFAULT direct-sum group (cross_couplings=None) is silent.
"""
import gc
import importlib
import warnings
import weakref

import pytest
import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.lie_ops import compose_bch
from vfe3.geometry.phi_preconditioner import _structure_constants


lie_ops_module = importlib.import_module("vfe3.geometry.lie_ops")

# Distinctive, stable substrings emitted by the two guards (match against these so the
# "no warning" assertions do not false-fail on unrelated torch/numpy warnings).
_STRUCT_MSG  = "not closed under the Lie bracket"
_BCH_MSG     = "not closed under the Lie bracket"

# Small dims: d_head=2, n_heads=3 -> embed_dim K=6. A 3-head chain (0,1)+(1,2).
_D_HEAD  = 2
_N_HEADS = 3
_K       = _D_HEAD * _N_HEADS
_CHAIN   = [(0, 1), (1, 2)]


def _build(close_basis: bool, cross):
    return get_group("block_glk")(
        _K, _N_HEADS, close_basis=close_basis, cross_couplings=cross, dtype=torch.float64
    )


def _bch_inputs(n_gen: int) -> tuple:
    """Two phi vectors with O(1) weight on every coord so both couplings are excited."""
    torch.manual_seed(0)
    phi1 = torch.randn(n_gen, dtype=torch.float64)
    phi2 = torch.randn(n_gen, dtype=torch.float64)
    return phi1, phi2


def _clear_bch_closure_caches() -> None:
    lie_ops_module._BRACKET_CLOSURE_RES.clear()
    lie_ops_module._BRACKET_CLOSURE_WARNED.clear()
    if hasattr(lie_ops_module, "_BRACKET_CLOSURE_IDENTITIES"):
        lie_ops_module._BRACKET_CLOSURE_IDENTITIES.clear()


def test_bch_closure_identity_cache_hashes_once_per_identity_and_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_bch_closure_caches()
    generators = _build(close_basis=False, cross=None).generators.clone()
    signature_calls = 0
    original = lie_ops_module._basis_value_signature

    def _signature_spy(basis: torch.Tensor) -> tuple:
        nonlocal signature_calls
        signature_calls += 1
        return original(basis)

    monkeypatch.setattr(lie_ops_module, "_basis_value_signature", _signature_spy)
    lie_ops_module.warn_if_basis_not_closed(generators, where="identity-cache")
    lie_ops_module.warn_if_basis_not_closed(generators, where="identity-cache")
    assert signature_calls == 1
    assert len(lie_ops_module._BRACKET_CLOSURE_RES) == 1

    equal_value_copy = generators.clone()
    lie_ops_module.warn_if_basis_not_closed(equal_value_copy, where="identity-cache")
    assert signature_calls == 2
    assert len(lie_ops_module._BRACKET_CLOSURE_RES) == 1

    generators[0, 0, 0].add_(0.125)
    lie_ops_module.warn_if_basis_not_closed(generators, where="identity-cache-mutated")
    assert signature_calls == 3
    assert len(lie_ops_module._BRACKET_CLOSURE_RES) == 2


def test_bch_closure_identity_cache_is_bounded_weak_and_checks_exact_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_bch_closure_caches()
    template = _build(close_basis=False, cross=None).generators
    live_bases = [template.clone() for _ in range(40)]
    for basis in live_bases:
        lie_ops_module.warn_if_basis_not_closed(basis, where="identity-cache-bound")

    identity_cache = lie_ops_module._BRACKET_CLOSURE_IDENTITIES
    assert len(identity_cache) == 32

    generators = template.clone().requires_grad_()
    other = template.clone()
    identity_key = (
        id(generators),
        generators._version,
        tuple(generators.shape),
        generators.dtype,
        generators.device,
    )
    bogus_signature = (tuple(generators.shape), generators.dtype, b"wrong-caller")
    identity_cache[identity_key] = (weakref.ref(other), bogus_signature)
    signature_calls = 0
    original = lie_ops_module._basis_value_signature

    def _signature_spy(basis: torch.Tensor) -> tuple:
        nonlocal signature_calls
        signature_calls += 1
        return original(basis)

    monkeypatch.setattr(lie_ops_module, "_basis_value_signature", _signature_spy)
    lie_ops_module.warn_if_basis_not_closed(generators, where="identity-cache-exact")
    assert signature_calls == 1

    cached_ref, cached_signature = identity_cache[identity_key]
    assert isinstance(cached_ref, weakref.ReferenceType)
    assert cached_ref() is generators
    assert all(not isinstance(item, torch.Tensor) for item in cached_signature)
    generators_ref = weakref.ref(generators)
    del generators
    gc.collect()
    assert generators_ref() is None
    assert identity_key not in identity_cache


def _max_offspan_bracket_residual(G: torch.Tensor) -> float:
    r"""max_{a,b} ||[G_a,G_b] - P_span([G_a,G_b])||_F / (||[G_a,G_b]||_F + eps)."""
    G = G.double()
    n, K, _ = G.shape
    flat = G.reshape(n, K * K)
    # Orthonormal span projector P = Q Q^T from a thin SVD of the basis.
    U, S, _ = torch.linalg.svd(flat.transpose(-1, -2), full_matrices=False)
    Q = U[:, S > 1e-9 * S[0]]
    worst = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            C = (G[a] @ G[b] - G[b] @ G[a]).reshape(-1)
            cn = float(C.norm())
            if cn < 1e-30:
                continue
            C_perp = C - Q @ (Q.transpose(-1, -2) @ C)
            worst = max(worst, float(C_perp.norm()) / (cn + 1e-12))
    return worst


# ---------------------------------------------------------------------------
# (1) Non-closed 3-head chain: BOTH consumers must warn.
# ---------------------------------------------------------------------------
def test_nonclosed_chain_structure_constants_warns():
    grp = _build(close_basis=False, cross=_CHAIN)
    # Confirm the basis really is non-closed (the residual the guard measures).
    assert _max_offspan_bracket_residual(grp.generators) > 1e-4
    with pytest.warns(UserWarning, match=_STRUCT_MSG):
        _structure_constants(grp.generators)


def test_nonclosed_chain_compose_bch_warns():
    grp = _build(close_basis=False, cross=_CHAIN)
    phi1, phi2 = _bch_inputs(grp.generators.shape[0])
    with pytest.warns(UserWarning, match=_BCH_MSG):
        compose_bch(phi1, phi2, grp.generators)


# ---------------------------------------------------------------------------
# (2) SAME chain, close_basis=True: bracket-closed basis, NO warning from either.
# ---------------------------------------------------------------------------
def test_closed_chain_is_bracket_closed_and_silent():
    grp = _build(close_basis=True, cross=_CHAIN)
    # The closed basis is genuinely bracket-closed: max off-span residual ~0.
    assert _max_offspan_bracket_residual(grp.generators) < 1e-6
    # Input generators are preserved verbatim as the first n_gen rows (closure contract),
    # and the closed basis strictly extends the open one.
    assert grp.generators.shape[0] > _build(False, _CHAIN).generators.shape[0]

    phi1, phi2 = _bch_inputs(grp.generators.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # any UserWarning -> test failure
        _structure_constants(grp.generators)
        compose_bch(phi1, phi2, grp.generators)


# ---------------------------------------------------------------------------
# (3) DEFAULT direct-sum group (cross_couplings=None): the silent default.
# ---------------------------------------------------------------------------
def test_default_direct_sum_is_silent():
    grp = _build(close_basis=False, cross=None)
    assert grp.irrep_dims == [_D_HEAD] * _N_HEADS        # genuine block-diagonal default
    phi1, phi2 = _bch_inputs(grp.generators.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # any UserWarning -> test failure
        _structure_constants(grp.generators)
        compose_bch(phi1, phi2, grp.generators)
