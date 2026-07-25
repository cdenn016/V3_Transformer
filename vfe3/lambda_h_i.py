r"""Hyper-prior coupling coefficient lambda_h_i: the weight on the model-channel term KL(s_i || r).

The model-fiber analogue of the belief-fiber self-coupling alpha (``vfe3/alpha_i.py``). The free
energy carries ``alpha_i KL(q_i||p_i)`` on the state fiber and ``lambda_h KL(s_i||r_i)`` on the
model fiber as structurally identical positive-precision-weighted forward-KL self-terms; the
manuscript writes them side by side and names the state-dependent lambda_h "a parallel extension
not developed here" (Participatory_it_from_bit.tex, appendix free energy). A config-selected
registry of forms:

  constant                  lambda_h = value (default; the bare scalar cfg.lambda_h); no regularizer.
  state_dependent           lambda_h*_i = c0_h / (b0_h + KL(s_i||r)), the stationary point of
                            lambda_h*KL(s_i||r) + R_h(lambda_h), R_h = b0_h*lambda_h - c0_h*log lambda_h
                            -- the SAME Gamma-prior-on-precision envelope as alpha (structural, NOT exact
                            Normal-Gamma conjugacy, per Participatory_it_from_bit.tex:1344). The caller
                            MUST add R_h to F (and to the s E-step) so the envelope cancellation holds.
  state_dependent_per_coord per-coordinate lambda_h^(k)* = c0_h^(k) / (b0_h^(k) + KL_k(s_i||r)) on the
                            UNSUMMED per-coordinate divergence (..., N, K), with R_h^(k) summed over k by
                            the caller. The model-fiber mirror of alpha's per-coord form: each model
                            coordinate is shrunk toward r by its own envelope weight (coordinates far from
                            the prior shrunk differently from near ones). Needs a coordinate-decomposable
                            family/divergence (diagonal Gaussian + renyi/KL/...); enforced at config.

The forms reuse alpha's verified envelope math: this module delegates to
``vfe3.alpha_i.self_coupling_alpha`` so lambda_h*, R_h, and the per-coordinate form have a single
source of truth (the shared alpha registry). KL(s_i||r) decomposes coordinate-wise on the diagonal
s/r tables, so the per-coordinate form is the same construction as alpha's, fed the unsummed divergence.
"""

from typing import Tuple

import torch

from vfe3.alpha_i import _ALPHAS, alpha_is_per_coord, self_coupling_alpha


def lambda_h_modes() -> Tuple[str, ...]:
    r"""The admissible ``lambda_h_mode`` keys: exactly the registered alpha forms.

    This seam delegates its whole implementation to :func:`vfe3.alpha_i.self_coupling_alpha`, so its
    admissible keys ARE the alpha registry's keys. It previously carried a hard-coded tuple, which
    made the write-and-register contract hold for the belief fiber (``lambda_alpha_mode``, validated
    against ``_ALPHAS``) and silently fail for its mirror: a newly registered alpha form was accepted
    by ``lambda_alpha_mode`` and rejected by both this module's dispatcher and ``config.py``'s
    validator, so exposing it on the model fiber required editing two call sites (audit 2026-07-25 F9).
    """
    return tuple(sorted(_ALPHAS))


def hyper_prior_lambda_h(
    kl:           torch.Tensor,          # (..., N) per-token KL(s_i||r); (..., N, K) per-coord (state_dependent_per_coord)

    *,
    mode:         str   = "constant",
    value:        float = 0.0,
    b0_h:         'float | torch.Tensor' = 1.0,    # scalar, or (K,) per-coord for state_dependent_per_coord
    c0_h:         'float | torch.Tensor' = 1.0,    # scalar, or (K,) per-coord
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return ``(lambda_h_i, R_h_i)``: the hyper-prior weight and its regularizer (per-token, or
    per-coordinate of shape ``kl`` when ``mode='state_dependent_per_coord'``).

    The model-fiber mirror of :func:`vfe3.alpha_i.self_coupling_alpha`, to which it delegates so the
    envelope ``lambda_h* = c0_h/(b0_h + KL)``, its per-coordinate form, and the regularizer ``R_h``
    share alpha's single verified implementation. ``mode`` in :func:`lambda_h_modes`; ``b0_h``/``c0_h``
    are the hyper-prior's OWN precision-shape hyperparameters (distinct from alpha's ``b0``/``c0``;
    scalar, or ``(K,)`` per-coordinate for the per-coord form). The returned regularizer is zero for
    ``constant`` and ``R_h`` for the two state-dependent forms (the caller adds it to F,
    and SUMS over coordinates for the per-coord form, so the stationary-point/envelope cancellation holds).
    """
    modes = lambda_h_modes()
    if mode not in modes:
        raise KeyError(f"no lambda_h_mode {mode!r}; available: {modes}")
    return self_coupling_alpha(kl, mode=mode, value=value, b0=b0_h, c0=c0_h)


def lambda_h_is_per_coord(mode: str) -> bool:
    r"""Whether lambda_h_mode ``mode`` consumes a per-coordinate (unsummed) hyper-prior divergence
    of shape ``(..., N, K)`` rather than the per-token summed ``(..., N)``.

    Validates membership in :func:`lambda_h_modes`, then defers to :func:`vfe3.alpha_i.alpha_is_per_coord`
    so the ``per_coord`` flag has one source of truth (lambda_h delegates to the shared alpha registry).
    The diagnostics / s-E-step routing reads this flag to supply the correctly-shaped divergence.
    """
    modes = lambda_h_modes()
    if mode not in modes:
        raise KeyError(f"no lambda_h_mode {mode!r}; available: {modes}")
    return alpha_is_per_coord(mode)
