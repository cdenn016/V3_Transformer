r"""Categorical emission factor for the belief's Markov blanket (Bohning majorization).

VFE 4.0 ``eq:state-model-markov-blanket-potentials``
(``vfe4_whitepaper/06_elbo_coordinate_updates.tex:406``) puts the emission inside the potential for
the continuous latent,

.. math::
    \Phi_{z_s} = \log K^z_{\theta,sa_s}(z_s \mid z_{a_s}, m_s, x_{<s}, \Gamma)
               + \log L_{\theta,s}(x_s \mid z_s, m_s, \Gamma) + \ldots,

so an observation-conditioned update of the belief includes the emission. V3's E-step descends
``alpha KL(q||p) + Sum_j beta KL(q||Omega q) + entropy`` with no data term at all, which is why the
pairwise energy measures belief AGREEMENT rather than predictive relevance.

The factor reads ``x_s``, the CURRENT token, so it is prefix-measurable: it is exactly the
non-leaky half of the observation channel. Conditioning the belief at ``t`` on the held-out target
``x_{t+1}`` would be a genuine leak, because evaluation runs the same recognition path.

No finite Gaussian ``(dJ, dh)`` identity exists for a categorical softmax
(``06_elbo_coordinate_updates.tex:777`` names the admissible routes as a declared bound, quadrature,
gradient optimization, or an accepted generalized-EM proposal). This module takes the declared-bound
route with the Bohning quadratic majorizer, whose curvature is CONSTANT in the expansion point:

.. math::
    -\log \mathrm{softmax}(Wz)_{x} \le
        \tfrac12 (z-z_0)^\top W^\top H W (z-z_0) - (z-z_0)^\top W^\top(e_x - p_0) + \mathrm{const},
    \qquad H = \tfrac12\left(I_V - \tfrac{\mathbf 1\mathbf 1^\top}{V}\right),

with ``p_0 = softmax(W z_0)``. Bohning (1992), "Multinomial logistic regression algorithm",
Ann. Inst. Statist. Math. 44, 197-200. Because ``H`` dominates the Hessian of the log-partition
uniformly in the Loewner order, the bound holds for every ``z`` with the FULL curvature
``W^T H W``.

**The diagonal restriction is an approximation, not a majorizer.** Under
``family='gaussian_diagonal'`` this module consumes only ``diag(W^T H W)``, and for a general
direction ``v`` there is no inequality between ``v^T diag(W^T H W) v`` and ``v^T W^T H W v`` --
the off-diagonal curvature is dropped, not bounded. So the emission step is majorize-minimize
only along the coordinate axes, and the strict global-upper-bound guarantee does NOT survive the
restriction. This is the same class of approximation the family already makes everywhere else
(``transport_covariance`` returns only the diagonal of ``Omega Sigma Omega^T``), and it is stated
here rather than papered over. A strictly majorizing variant would need the full ``(K, K)``
curvature and a full-covariance family. ``tests/test_emission_factor_20260726.py`` pins the full
curvature as a true majorizer and pins the diagonal as its exact diagonal, so the gap is measured
rather than assumed.

The consumed diagonal is

.. math::
    d_k = \tfrac12\left(\sum_v W_{vk}^2 - \tfrac{1}{V}\Big(\sum_v W_{vk}\Big)^2\right),

one ``O(VK)`` reduction per forward rather than a ``(K, K)`` product. It is non-negative for any
``W``, so the fused precision stays positive.

Expanding at ``z_0`` and dropping constants leaves a diagonal precision ``d`` and a linear term
``g = W^\top(e_x - p_0)``, which enter the closed-form fusion as

    ``prec      += emission_weight * d``
    ``numerator += emission_weight * (d * z_0 + g)``.

**The expansion point travels with the terms.** ``z_0`` is RETURNED as the third element rather
than re-derived at the consumer, because the consumer's ``mu_p`` is not ``z_0`` once the stack is
deeper than one layer: ``vfe_stack`` advances the block prior (``mu_p <- (1-rho) mu_p + rho mu``)
every layer while the Bohning pair is built once, so a consumer that substituted its own ``mu_p``
would center the quadratic on a point where ``g`` was never evaluated and minimize a surrogate that
majorizes nothing. Measured before the fix: the beta-frozen surrogate's gradient residual at the
returned optimum was ``1.2e-07`` at ``n_layers=1`` (anchors coincide) and ``5.5e-01`` beyond it
(audit 2026-07-27, duel-free finding confirmed independently by two agents).
"""

from typing import Optional, Tuple

import torch


VOCAB_CHUNK = 8192          # vocab tile for the streaming softmax; bounds the (B, N, V) logit spike


def bohning_curvature_diagonal(
    weight: torch.Tensor,                     # (V, K) readout table

    *,
    eps:    float = 1e-12,
) -> torch.Tensor:                            # (K,) diag(W^T H W), >= 0
    r"""``d_k = (1/2)(sum_v W_vk^2 - (sum_v W_vk)^2 / V)``, the diagonal of the Bohning curvature.

    This is ``V/2`` times the per-coordinate variance of the column of ``W`` across the vocabulary,
    so it is non-negative for any ``W`` and vanishes only on a column that is constant across every
    vocabulary row. It does not depend on the expansion point, hence one reduction per forward.

    Computed in the TWO-PASS centered form. The algebraically identical one-pass identity
    ``sum_v W^2 - (sum_v W)^2 / V`` is the textbook cancellation trap (Higham 2002, section 1.9):
    at the live ``V = 50257`` with columns of std 0.02 the fp32 relative error grows from 2.4e-07 at
    zero column mean to 8.7e-04 at unit mean, because it differs two O(V) quantities that cancel to
    O(V * variance). The centered form is non-negative by construction and its error no longer grows
    with the column MEAN.

    It is not exact at zero: a genuinely constant column still leaves fp32 accumulation residue
    (~4e-12 at V = 512), so the ``eps`` clamp is a backstop rather than a value the result actually
    reaches. What matters for the fusion is that ``d_k`` is strictly positive and negligible against
    any real curvature, which both forms satisfy -- the one-pass form's failure was accuracy on
    LARGE ``d_k``, not the floor.
    """
    centered = weight - weight.mean(dim=0, keepdim=True)           # (V, K) column-centered readout
    return (0.5 * (centered * centered).sum(dim=0)).clamp(min=eps)  # (K,) SPD guard for the fusion


def bohning_emission_terms(
    mu_p:        torch.Tensor,                # (..., N, K) prior means; the MM expansion point
    weight:      torch.Tensor,                # (V, K) readout table
    token_ids:   torch.Tensor,                # (..., N) CURRENT-token ids x_t (never the target)

    *,
    bias:        Optional[torch.Tensor] = None,   # (V,) additive logit bias, or None
    vocab_chunk: int   = VOCAB_CHUNK,             # vocabulary tile size
    eps:         float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # (K,) curvature, (..., N, K) linear, (..., N, K) z_0
    r"""Return ``(d, g, z_0)``: ``d = diag(W^T H W)``, ``g = W^T(e_{x_t} - softmax(W z_0 + b))``.

    The expansion-point probabilities ``p_0`` are DETACHED, which is the majorize-minimize
    convention: the surrogate's coefficients are frozen at the current iterate and only the
    surrogate is minimized. ``W`` stays live in ``d`` and in the CONTRACTION ``sum_v p_v W_v``, so
    the readout table still receives a gradient from the E-step -- that coupling is the entire point
    of the "shared" mode, which ties the belief's inference objective to the decoder that scores the
    prediction. Freezing ``p_0`` does not weaken that coupling: ``W`` enters ``g`` linearly through
    the contraction and through the observed row ``W_{x_t}`` regardless.

    Keeping ``p_0`` differentiable would also defeat the streaming: every vocabulary tile of all
    three passes would be retained for backward. Measured at ``V=50257, K=64, B=8, N=128`` on the
    5090, the live-``p_0`` loop peaked at 682 MiB against 425 MiB for the naive one-shot dense
    softmax it exists to avoid -- 1.6x WORSE. With ``p_0`` frozen the same call peaks at 161 MiB.

    The softmax is streamed over vocabulary tiles in two passes (running max, then normalize and
    accumulate), so the ``(..., N, V)`` logit block is never materialized in full. At V = 50257 and
    a (64, 128) batch that block would be about 1.6 GB in fp32.
    """
    # fp32 island, matching ``belief_gradients`` / ``mm_exact_update`` / ``stable_matrix_exp_pair``.
    # The vocabulary reduction is a logsumexp over V = O(1e4) terms feeding a precision; in bf16 the
    # returned g drifted by 6.0e-03 relative (audit 2026-07-27).
    if torch.is_autocast_enabled(mu_p.device.type):
        with torch.autocast(device_type=mu_p.device.type, enabled=False):
            return bohning_emission_terms(
                mu_p.float(), weight.float(), token_ids,
                bias=(bias.float() if bias is not None else None),
                vocab_chunk=vocab_chunk,
                eps=eps,
            )

    vocab_size, _ = weight.shape
    expansion = mu_p.detach()                                      # MM: coefficients frozen at z_0

    # Passes 1 and 2 build p_0 only. They run under no_grad so the frozen-coefficient contract is
    # the code's behavior and not merely its docstring, and so no vocabulary tile is retained.
    with torch.no_grad():
        w_const = weight.detach()
        b_const = bias.detach() if bias is not None else None

        running_max = None                                         # (..., N)
        for start in range(0, vocab_size, vocab_chunk):
            tile = w_const[start:start + vocab_chunk]               # (C, K)
            logit_tile = expansion @ tile.transpose(-1, -2)         # (..., N, C)
            if b_const is not None:
                logit_tile = logit_tile + b_const[start:start + vocab_chunk]
            tile_max = logit_tile.amax(dim=-1)                      # (..., N)
            running_max = tile_max if running_max is None else torch.maximum(running_max, tile_max)

        exp_sum = torch.zeros_like(running_max)                     # (..., N)
        for start in range(0, vocab_size, vocab_chunk):
            tile = w_const[start:start + vocab_chunk]
            logit_tile = expansion @ tile.transpose(-1, -2)
            if b_const is not None:
                logit_tile = logit_tile + b_const[start:start + vocab_chunk]
            exp_sum = exp_sum + (logit_tile - running_max.unsqueeze(-1)).exp().sum(dim=-1)

        log_norm = running_max + exp_sum.clamp(min=eps).log()        # (..., N) logsumexp

    # Pass 3 -- accumulate g = -sum_v p_v W_v with p_0 frozen and W LIVE in the contraction only.
    g = torch.zeros_like(mu_p)                                     # (..., N, K)
    for start in range(0, vocab_size, vocab_chunk):
        tile = weight[start:start + vocab_chunk]                   # (C, K) live
        with torch.no_grad():
            logit_tile = expansion @ w_const[start:start + vocab_chunk].transpose(-1, -2)
            if b_const is not None:
                logit_tile = logit_tile + b_const[start:start + vocab_chunk]
            prob_tile = (logit_tile - log_norm.unsqueeze(-1)).exp()  # (..., N, C) frozen p_0
        g = g - prob_tile @ tile                                   # -sum_v p_v W_v, W live
    g = g + weight[token_ids]                                      # + W_{x_t}: the observed row

    return bohning_curvature_diagonal(weight, eps=eps), g, expansion
