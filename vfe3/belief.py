r"""The Gaussian belief tuple for VFE_3.0."""

from typing import NamedTuple

import torch


class BeliefState(NamedTuple):
    """A per-token Gaussian belief q_i = N(mu_i, Sigma_i) with gauge frame phi_i."""

    mu:    torch.Tensor             # (..., N, K) means
    sigma: torch.Tensor             # (..., N, K) diagonal variances (or (..., N, K, K) full)
    phi:   torch.Tensor             # (..., N, n_gen) gauge-frame coordinates
