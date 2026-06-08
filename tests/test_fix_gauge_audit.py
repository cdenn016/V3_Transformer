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
import warnings

import pytest
import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.lie_ops import compose_bch
from vfe3.geometry.phi_preconditioner import _structure_constants

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
