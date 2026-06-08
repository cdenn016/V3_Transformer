r"""Audit pins for routing LIVE SPD inverse / eigenvalue-floor sites through the centralized
``safe_spd_inverse`` / ``floor_eigenvalues`` helpers in ``vfe3/numerics.py``.

OUTCOME OF THE AUDIT: ZERO live sites could be routed without changing the numerical result.
Every candidate inverse/floor in the four owned files
(numerics.py, geometry/retraction.py, families/gaussian.py, geometry/transport.py) either uses a
divergent algorithm, needs a ridge-free / factor-reusing variant the helper does not provide, needs
the gap-damped backward + a ceiling the helper lacks, or is not a matrix operation at all. This file
pins the two non-obvious LEAVE decisions as TESTED FACTS (not comments), so the reviewer sees they
were audited empirically. NO LIVE NUMERICAL RESULT CHANGED in this audit.

LEFT UNCHANGED, with reasons:

  safe_spd_inverse (SPD inverse policy: eps=1e-6 ridge -> cholesky_inverse -> pinv):
    - gaussian.py FullGaussian.natural (~L182): prec = solve(Sigma + 1e-6 I, I). Same 1e-6 ridge,
      but the helper's symmetrize + cholesky_inverse DIVERGES from this LU solve by ~1.5e-4 once
      cond(Sigma) >= 1e3 -- and natural() realistically sees cond up to ~5e6 (eps=1e-6 variance floor
      / sigma_max=5 ceiling). Routing would change the live forward. Pinned below
      (test_natural_inverse_not_routed_solve_vs_safe_spd_diverges); the well-conditioned-agreement
      and divergence-onset are both asserted so the leave-decision is reproducible.
    - gaussian.py FullGaussian.log_partition_at (~L190-191): cholesky inverse with NO ridge, and the
      Cholesky factor L is REUSED for _logdet_chol. safe_spd_inverse adds a 1e-6 ridge (changes the
      value) and discards L (would force a second factorization). Not routable.
    - gaussian.py renyi_closed_form (~L222-223): Sigma + eps I feeds safe_cholesky / solve_triangular,
      not a matrix inverse -- not a safe_spd_inverse call shape.

  floor_eigenvalues (eigenvalue/variance floor: symmetrize -> stock eigh -> clamp(min=floor)):
    - retraction.py retract_spd_diagonal / log_euclidean diagonal arm / natural_gradient and
      gaussian.py (~L50,63,79,81,96,122,124,137): DIAGONAL VECTOR clamps on (...,K) variances, NOT a
      (...,K,K) matrix eigh -- a vector clamp is a different operation.
    - retraction.py retract_spd_full / retract_logeuclidean_full input floors (eigenvalues.clamp(min=eps)
      at ~L148,247): a clamp on ALREADY-DECOMPOSED eigenvalues fed straight to sqrt/log; routing would
      re-run eigh and discard the eigenvectors in hand.
    - retraction.py FINAL output floors (~L168, ~L264): floor AND an upper cap sigma_max the helper
      lacks, AND they use the gap-damped ``_eigh_damped`` so the backward is finite at the degenerate
      Sigma = I init. floor_eigenvalues uses stock torch.linalg.eigh, whose backward is NaN there.
      Pinned below (test_floor_eigenvalues_stock_eigh_backward_is_nan_at_isotropic).

  transport.py stable_matrix_exp_pair (~L230): .clamp(min=1e-8) on a Frobenius NORM (a scalar rescale
    guard), not an SPD operation.
"""

import torch

from vfe3.families.gaussian import FullGaussian
from vfe3.numerics import floor_eigenvalues, safe_spd_inverse


def _old_natural_precision(sigma: torch.Tensor) -> torch.Tensor:
    r"""The live inline formula for the precision in ``FullGaussian.natural`` (kept, NOT routed):
    prec = solve(Sigma + 1e-6 I, I) = (Sigma + 1e-6 I)^{-1}."""
    eye = torch.eye(sigma.shape[-1], device=sigma.device, dtype=sigma.dtype)
    return torch.linalg.solve(sigma + 1e-6 * eye, eye.expand_as(sigma))


def _spd(batch: tuple, K: int, *, cond: float, seed: int) -> torch.Tensor:
    r"""A representative SPD batch with target condition number ``cond`` (eigenvalues 1..cond on a
    random orthonormal frame)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(*batch, K, K, generator=g)
    Q, _ = torch.linalg.qr(A)
    evals = torch.linspace(1.0, cond, K).expand(*batch, K)
    return (Q * evals.unsqueeze(-2)) @ Q.transpose(-1, -2)


def test_natural_inverse_well_conditioned_would_agree():
    r"""Sanity: on a WELL-conditioned SPD Sigma (cond ~ 10) the centralized safe_spd_inverse and the
    live ``solve(Sigma + 1e-6 I, I)`` agree to float32 tolerance -- so the leave-decision is driven by
    conditioning, not by a gross policy mismatch."""
    sigma = _spd((3, 4), 5, cond=10.0, seed=0)
    routed = safe_spd_inverse(sigma, eps=1e-6)
    live = _old_natural_precision(sigma)
    assert torch.allclose(routed, live, atol=1e-4, rtol=1e-4)


def test_natural_inverse_not_routed_solve_vs_safe_spd_diverges():
    r"""Why FullGaussian.natural's inverse is LEFT inline, not routed through safe_spd_inverse: the
    helper's cholesky_inverse diverges from the live LU solve beyond 1e-5 once cond(Sigma) >= 1e3,
    and natural() realistically sees cond up to ~5e6 (eps=1e-6 floor / sigma_max=5 cap). Pinning the
    divergence makes the leave-decision a tested fact: centralizing here WOULD change the live result."""
    sigma = _spd((2, 6), 5, cond=1.0e4, seed=1)
    routed = safe_spd_inverse(sigma, eps=1e-6)
    live = _old_natural_precision(sigma)
    assert not torch.allclose(routed, live, atol=1e-5, rtol=1e-5)
    # ... and both are merely float32 round-off around the SAME true inverse (float64 reference):
    eye = torch.eye(5, dtype=torch.float64)
    ref = torch.linalg.inv(sigma.double() + 1e-6 * eye).float()
    assert (routed - ref).abs().max() < 1e-2 and (live - ref).abs().max() < 1e-2


def test_full_gaussian_natural_unchanged_by_audit():
    r"""The live call site is UNCHANGED by this audit: FullGaussian.natural() still equals the inline
    solve construction byte-for-byte (the route was reverted)."""
    g = torch.Generator().manual_seed(2)
    sigma = _spd((2, 3), 4, cond=50.0, seed=3)
    mu = torch.randn(2, 3, 4, generator=g)
    t1, t2 = FullGaussian(mu, sigma).natural()

    prec = _old_natural_precision(sigma)
    t1_live = (prec @ mu.unsqueeze(-1)).squeeze(-1)
    t2_live = -0.5 * prec
    assert torch.equal(t1, t1_live)
    assert torch.equal(t2, t2_live)


def test_floor_eigenvalues_stock_eigh_backward_is_nan_at_isotropic():
    r"""Why the FINAL SPD-retraction floors are LEFT on ``_eigh_damped`` and NOT routed through
    floor_eigenvalues: the helper uses stock torch.linalg.eigh, whose backward is NaN at the
    degenerate Sigma = I init (the gaussian_full default prior init) -- exactly the failure
    retract_spd_full's gap-damped eigh avoids. Asserting the helper's backward is non-finite there
    makes the leave-decision a tested fact, not a comment."""
    A = torch.eye(4, requires_grad=True)
    floor_eigenvalues(A, floor=1e-6).sum().backward()
    assert not torch.isfinite(A.grad).all(), (
        "stock-eigh floor_eigenvalues was finite at Sigma=I; if torch changed this, re-evaluate "
        "whether the retraction final floors can now route through it"
    )
