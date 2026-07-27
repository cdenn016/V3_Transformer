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

Expanding at ``z_0 = mu_p`` and dropping constants leaves a diagonal precision ``d`` and a linear
term ``g = W^\top(e_x - p_0)``, which enter the closed-form fusion as

    ``prec      += emission_weight * d``
    ``numerator += emission_weight * (d * mu_p + g)``.
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
    """
    vocab_size = weight.shape[0]
    sum_sq = (weight * weight).sum(dim=0)                          # (K,) sum_v W_vk^2
    sq_sum = weight.sum(dim=0) ** 2                                # (K,) (sum_v W_vk)^2
    return (0.5 * (sum_sq - sq_sum / vocab_size)).clamp(min=eps)   # (K,) SPD guard for the fusion


def bohning_emission_terms(
    mu_p:        torch.Tensor,                # (..., N, K) prior means; the MM expansion point
    weight:      torch.Tensor,                # (V, K) readout table
    token_ids:   torch.Tensor,                # (..., N) CURRENT-token ids x_t (never the target)

    *,
    bias:        Optional[torch.Tensor] = None,   # (V,) additive logit bias, or None
    vocab_chunk: int   = VOCAB_CHUNK,             # vocabulary tile size
    eps:         float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:           # (K,) curvature, (..., N, K) linear term
    r"""Return ``(d, g)`` with ``d = diag(W^T H W)`` and ``g = W^T(e_{x_t} - softmax(W mu_p + b))``.

    The expansion-point probabilities ``p_0`` are DETACHED, which is the majorize-minimize
    convention: the surrogate's coefficients are frozen at the current iterate and only the
    surrogate is minimized. ``W`` itself stays live in both ``d`` and ``g``, so the readout table
    still receives a gradient from the E-step -- that coupling is the entire point of the "shared"
    mode, which ties the belief's inference objective to the decoder that scores the prediction.

    The softmax is streamed over vocabulary tiles in two passes (running max, then normalize and
    accumulate), so the ``(..., N, V)`` logit block is never materialized in full. At V = 50257 and
    a (64, 128) batch that block would be about 1.6 GB in fp32.
    """
    vocab_size, _ = weight.shape
    expansion = mu_p.detach()                                      # MM: coefficients frozen at z_0

    # Pass 1 -- running max and the exponential sum, both in the expansion point only.
    running_max = None                                             # (..., N)
    for start in range(0, vocab_size, vocab_chunk):
        tile = weight[start:start + vocab_chunk]                   # (C, K)
        logit_tile = expansion @ tile.transpose(-1, -2)            # (..., N, C)
        if bias is not None:
            logit_tile = logit_tile + bias[start:start + vocab_chunk]
        tile_max = logit_tile.amax(dim=-1)                         # (..., N)
        running_max = tile_max if running_max is None else torch.maximum(running_max, tile_max)

    exp_sum = torch.zeros_like(running_max)                        # (..., N)
    for start in range(0, vocab_size, vocab_chunk):
        tile = weight[start:start + vocab_chunk]
        logit_tile = expansion @ tile.transpose(-1, -2)
        if bias is not None:
            logit_tile = logit_tile + bias[start:start + vocab_chunk]
        exp_sum = exp_sum + (logit_tile - running_max.unsqueeze(-1)).exp().sum(dim=-1)

    # Pass 2 -- normalize each tile and accumulate g = -sum_v p_v W_v, then add the observed row.
    log_norm = running_max + exp_sum.clamp(min=eps).log()          # (..., N) logsumexp
    g = torch.zeros_like(mu_p)                                     # (..., N, K)
    for start in range(0, vocab_size, vocab_chunk):
        tile = weight[start:start + vocab_chunk]                   # (C, K)
        logit_tile = expansion @ tile.transpose(-1, -2)
        if bias is not None:
            logit_tile = logit_tile + bias[start:start + vocab_chunk]
        prob_tile = (logit_tile - log_norm.unsqueeze(-1)).exp()    # (..., N, C) detached by construction
        g = g - prob_tile @ tile                                   # -sum_v p_v W_v, W live
    g = g + weight[token_ids]                                      # + W_{x_t}: the observed row

    return bohning_curvature_diagonal(weight, eps=eps), g
