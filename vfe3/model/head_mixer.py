r"""Schur-commutant head mixer for VFE_3.0 (opt-in; VFE_2.0 ``VFEHeadMixer`` parity).

Mixes the equal-size gauge-irrep blocks of a belief with one learned matrix
:math:`A = I + \Delta \in R^{n \times n}` embedded as :math:`\mathrm{kron}(A, I_d)`, where
:math:`n` is the number of blocks and :math:`d` the (shared) block dimension. Under
``block_glk`` the blocks are the ``n_heads`` heads, so the mixer mixes heads. Applied
symmetrically to the mean and covariance:

.. math::
    M    = \mathrm{kron}(A, I_d) \in R^{nd \times nd}, \qquad
    \mu' = M\,\mu, \qquad
    \Sigma' = M\,\Sigma\,M^{\top},

with the diagonal-covariance closed form (the diagonal-of-sandwich approximation already used
throughout V3 when ``diagonal_covariance=True``)

.. math::
    \sigma'[m, c] = \sum_n A[m, n]^2\, \sigma[n, c].

Initialization is exactly the identity (:math:`\Delta = 0`, stored as the delta-from-identity
so the init is bit-exact), so a model with the mixer enabled is bitwise indistinguishable from
the mixer-disabled path at step 0.

Gauge equivariance: :math:`\mathrm{kron}(A, I_d)` commutes with a block-diagonal gauge
:math:`\mathrm{diag}(h_1, \ldots, h_n)` ONLY when the gauge is TIED (:math:`h_k = h_0` for all
:math:`k`). V3's ``block_glk`` generators (``generate_glk_multihead``) give each head its OWN
independent ``gl(d_head)`` sub-algebra -- an UNTIED gauge -- so the mixer does NOT commute with
the per-head gauge action and breaks strict gauge equivariance there. The deviation is zero at
the identity init and grows as :math:`A` drifts from :math:`I` during training. This is an
accepted, opt-in departure (the no-mixer path is the default and stays equivariant); a future
tied-gauge group (one shared ``gl(d)`` replicated across heads) would restore exact equivariance.
"""

from typing import List, Tuple

import torch
from torch import nn


class HeadMixer(nn.Module):
    r"""Per-irrep-block mixer over ``n`` equal-size blocks: :math:`A = I + \Delta`."""

    def __init__(
        self,
        irrep_dims: List[int],           # gauge block sizes (must be all equal, length >= 2)
    ) -> None:
        super().__init__()
        if len(irrep_dims) < 2:
            raise ValueError(
                f"HeadMixer needs >= 2 blocks to mix, got irrep_dims={irrep_dims}; a single-block "
                f"group (glk / so_k) has nothing to mix. Use block_glk (n_heads >= 2)."
            )
        if len(set(irrep_dims)) != 1:
            raise ValueError(
                f"HeadMixer needs equal-size blocks for kron(A, I_d), got irrep_dims={irrep_dims}."
            )
        self.n_blocks = len(irrep_dims)
        self.d_block = irrep_dims[0]
        # Store the delta-from-identity (zeros) so the identity init is bit-exact and the
        # bitwise equivalence to the no-mixer path at step 0 is obvious.
        self.mixer_delta = nn.Parameter(torch.zeros(self.n_blocks, self.n_blocks))

    def _A(self) -> torch.Tensor:
        r"""The mixing matrix :math:`A = I + \Delta` (device/dtype follow the parameter)."""
        eye = torch.eye(self.n_blocks, device=self.mixer_delta.device, dtype=self.mixer_delta.dtype)
        return eye + self.mixer_delta

    def is_identity(self) -> bool:
        r"""True iff :math:`A = I` exactly (``mixer_delta == 0``)."""
        return bool((self.mixer_delta.detach() == 0).all().item())

    def forward(
        self,
        mu:    torch.Tensor,             # (..., K) belief means
        sigma: torch.Tensor,             # (..., K) diagonal variances OR (..., K, K) full covariance
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Apply :math:`\mu \mapsto M\mu`, :math:`\Sigma \mapsto M\Sigma M^{\top}`.

        Diagonal (``sigma.dim() == mu.dim()``) uses the closed form
        :math:`\sigma'[m] = \sum_n A[m,n]^2 \sigma[n]`; full (``sigma.dim() == mu.dim() + 1``)
        applies the exact sandwich on the reshaped block axes. The block layout is contiguous
        (V3 ``irrep_dims`` are contiguous equal blocks), so the reshape ``(..., K) -> (..., n, d)``
        splits heads directly with no permutation.
        """
        A = self._A()
        n, d = self.n_blocks, self.d_block

        mu_blocks = mu.reshape(*mu.shape[:-1], n, d)                       # (..., n, d)
        mu_out = torch.einsum("mn,...nd->...md", A, mu_blocks).reshape(mu.shape)

        if sigma.dim() == mu.dim():                                        # diagonal variances
            sigma_blocks = sigma.reshape(*sigma.shape[:-1], n, d)          # (..., n, d)
            sigma_out = torch.einsum("mn,...nd->...md", A * A, sigma_blocks).reshape(sigma.shape)
        else:                                                              # full covariance (..., K, K)
            block = sigma.reshape(*sigma.shape[:-2], n, d, n, d)           # (..., n, d, n, d)
            block = torch.einsum("mp,...pcqe->...mcqe", A, block)          # A on the first block axis
            block = torch.einsum("nq,...mcqe->...mcne", A, block)          # A on the second block axis
            sigma_out = block.reshape(sigma.shape)
        return mu_out, sigma_out
