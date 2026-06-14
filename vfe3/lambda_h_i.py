r"""Hyper-prior coupling coefficient lambda_h_i: the weight on the model-channel term KL(s_i || r).

The model-fiber analogue of the belief-fiber self-coupling alpha (``vfe3/alpha_i.py``). The free
energy carries ``alpha_i KL(q_i||p_i)`` on the state fiber and ``lambda_h KL(s_i||r_i)`` on the
model fiber as structurally identical positive-precision-weighted forward-KL self-terms; the
manuscript writes them side by side and names the state-dependent lambda_h "a parallel extension
not developed here" (Participatory_it_from_bit.tex, appendix free energy). A config-selected
registry of forms:

  constant         lambda_h = value (default; the bare scalar cfg.lambda_h); no regularizer.
  state_dependent  lambda_h*_i = c0_h / (b0_h + KL(s_i||r)), the stationary point of
                   lambda_h*KL(s_i||r) + R_h(lambda_h), R_h = b0_h*lambda_h - c0_h*log lambda_h --
                   the SAME Gamma-prior-on-precision envelope as alpha (structural, NOT exact
                   Normal-Gamma conjugacy, per Participatory_it_from_bit.tex:1344). The caller MUST
                   add R_h to F (and to the s E-step) so the envelope cancellation holds.
  learnable        NEURAL-NETWORK EXCEPTION (sanctioned, default-off): lambda_h = exp(log_lambda_h),
                   a model-owned scalar nn.Parameter trained by backprop (sibling of log_alpha /
                   log_lambda_beta); R = 0.

The three forms reuse alpha's verified envelope math: this module delegates to
``vfe3.alpha_i.self_coupling_alpha`` so lambda_h* and R_h have a single source of truth. The
hyper-prior divergence KL(s_i||r) is per-token (summed over coordinates), so there is no
per-coordinate form (unlike alpha's ``state_dependent_per_coord``).
"""

from typing import Optional, Tuple

import torch

from vfe3.alpha_i import self_coupling_alpha

_LAMBDA_H_MODES: Tuple[str, ...] = ("constant", "state_dependent", "learnable")


def hyper_prior_lambda_h(
    kl:           torch.Tensor,          # (..., N) per-token hyper-prior divergence KL(s_i||r)

    *,
    mode:         str   = "constant",
    value:        float = 0.0,
    b0_h:         float = 1.0,
    c0_h:         float = 1.0,
    log_lambda_h: Optional[torch.Tensor] = None,   # learned scalar (lambda_h=exp(log_lambda_h)); 'learnable' only
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return ``(lambda_h_i, R_h_i)``: the per-token hyper-prior weight and its regularizer.

    The model-fiber mirror of :func:`vfe3.alpha_i.self_coupling_alpha`, to which it delegates so the
    envelope ``lambda_h* = c0_h/(b0_h + KL)`` and the regularizer ``R_h`` share alpha's single
    verified implementation. ``mode`` in :data:`_LAMBDA_H_MODES`; ``b0_h``/``c0_h`` are the
    hyper-prior's OWN precision-shape hyperparameters (distinct from alpha's ``b0``/``c0``). The
    returned regularizer is zero for ``constant``/``learnable`` and ``R_h`` for ``state_dependent``
    (the caller adds it to F so the stationary-point/envelope cancellation holds).
    """
    if mode not in _LAMBDA_H_MODES:
        raise KeyError(f"no lambda_h_mode {mode!r}; available: {_LAMBDA_H_MODES}")
    return self_coupling_alpha(kl, mode=mode, value=value, b0=b0_h, c0=c0_h, log_alpha=log_lambda_h)
