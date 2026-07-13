r"""Packed strict-lower Cholesky storage for full-covariance model-channel prior tables (PB-11).

The model-channel s/r tables become full-covariance Gaussians under ``family="gaussian_full"``.
Rather than store a dense (K, K) covariance per token (of which K(K-1)/2 lower entries plus K
diagonal entries are free), each covariance is parameterized by its Cholesky factor L: the K log-
variances feed the diagonal ``L_ii = sqrt(bounded_variance_from_log(log_diag_i))`` (reusing the
SAME bounded-variance policy the diagonal tables use, so there is no second max-log clamp) and a
packed ``K*(K-1)//2`` vector holds the strict-lower entries ``L_ij`` (i>j). The covariance is then
the SPD product ``L L^T``. A zero packed vector yields a diagonal Cholesky, so a zero-init table is
exactly the diagonal covariance the diagonal family stores -- the pure-path guarantee.
"""

import torch

from vfe3.numerics import bounded_variance_from_log, safe_cholesky


def packed_strict_lower_size(K: int) -> int:
    r"""Number of strict-lower-triangle entries of a ``K x K`` matrix: ``K*(K-1)//2``."""
    return K * (K - 1) // 2


def covariance_from_packed(
    log_diag:     torch.Tensor,          # (..., K) log-variances (bounded before sqrt -> Cholesky diagonal)
    packed_lower: torch.Tensor,          # (..., K*(K-1)//2) strict-lower Cholesky entries

    *,
    eps:          float = 1e-6,
) -> torch.Tensor:                       # (..., K, K) SPD covariance L L^T
    r"""Assemble an SPD covariance ``L L^T`` from a log-variance diagonal and packed strict-lower
    Cholesky entries.

    The diagonal variance is formed by :func:`bounded_variance_from_log` BEFORE the square root
    (the SAME max-log policy the diagonal tables follow), so ``log_diag=100`` stays finite in
    float32 and emits the same overflow warning/clamp as an ordinary diagonal table -- there is no
    second clamp policy here.
    """
    k = log_diag.shape[-1]
    row, col = torch.tril_indices(k, k, offset=-1, device=log_diag.device)
    chol = log_diag.new_zeros(*log_diag.shape[:-1], k, k)
    chol[..., row, col] = packed_lower
    diagonal_variance = bounded_variance_from_log(log_diag, eps=eps)
    chol.diagonal(dim1=-2, dim2=-1).copy_(torch.sqrt(diagonal_variance))
    return chol @ chol.transpose(-1, -2)


def packed_from_covariance(
    covariance: torch.Tensor,            # (..., K, K) SPD covariance

    *,
    eps:        float = 1e-6,
) -> 'tuple[torch.Tensor, torch.Tensor]':   # (log_diag (..., K), packed_lower (..., K*(K-1)//2))
    r"""Invert :func:`covariance_from_packed`: factor an SPD covariance into a log-variance diagonal
    and packed strict-lower Cholesky entries.

    ``L = chol(covariance)`` gives the Cholesky factor whose diagonal ``L_ii`` satisfies
    ``L_ii^2 = variance_i``, so ``log_diag_i = log(L_ii^2)`` (clamped at ``eps`` for a numerically
    non-positive diagonal) recovers the value :func:`bounded_variance_from_log` maps back to
    ``L_ii^2`` in the normal range. The strict-lower entries are read off in the same
    ``tril_indices`` order the forward map writes, so the round trip is exact for an SPD input.
    ``safe_cholesky`` (jittered, never raises) hardens a numerically non-PD input.
    """
    k = covariance.shape[-1]
    row, col = torch.tril_indices(k, k, offset=-1, device=covariance.device)
    chol, _ = safe_cholesky(covariance, eps=eps, rounds=5)
    diag = torch.diagonal(chol, dim1=-2, dim2=-1)                       # (..., K) = sqrt(variance)
    log_diag = torch.log((diag ** 2).clamp(min=eps))                   # (..., K)
    packed_lower = chol[..., row, col]                                 # (..., K*(K-1)//2)
    return log_diag, packed_lower
