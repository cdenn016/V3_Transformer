r"""Schur-commutant head mixer for VFE_3.0 (opt-in; VFE_2.0 ``VFEHeadMixer`` parity).

Without labels (the legacy equal-dims form) mixes ``n`` equal-size gauge-irrep blocks with one
learned matrix :math:`A = I + \Delta \in \mathbb{R}^{n \times n}` embedded as
:math:`\mathrm{kron}(A, I_d)`, where :math:`n` is the number of blocks and :math:`d` the
(shared) block dimension. With ``irrep_labels`` the class builds one :math:`A_t = I + \Delta_t`
per maximal run of equal-labeled blocks -- the full linear commutant of a mixed-irrep tower for
real-type irreps -- so unequal block dimensions no longer raise a dim-collision hazard provided
they carry different labels. Under ``block_glk`` the blocks are the ``n_heads`` heads, so the
mixer mixes heads. Applied symmetrically to the mean and covariance:

.. math::
    M    = \mathrm{blockdiag}_t(A_t \otimes I_{d_t}) \in \mathbb{R}^{K \times K}, \qquad
    \mu' = M\,\mu, \qquad
    \Sigma' = M\,\Sigma\,M^{\top},

with the diagonal-covariance closed form (the diagonal-of-sandwich approximation already used
throughout V3 when ``diagonal_covariance=True``)

.. math::
    \sigma'[m, c] = \sum_n A_t[m, n]^2\, \sigma[n, c]

applied block-by-block for each component :math:`t`.

Initialization is exactly the identity (:math:`\Delta_t = 0`, stored as the delta-from-identity
so the init is bit-exact), so a model with the mixer enabled is bitwise indistinguishable from
the mixer-disabled path at step 0.

Gauge equivariance: :math:`\mathrm{kron}(A, I_d)` commutes with a block-diagonal gauge
:math:`\mathrm{diag}(h_1, \ldots, h_n)` ONLY when the gauge is TIED (:math:`h_k = h_0` for all
:math:`k`). V3's ``block_glk`` generators (``generate_glk_multihead``) give each head its OWN
independent ``gl(d_head)`` sub-algebra -- an UNTIED gauge -- so the mixer does NOT commute with
the per-head gauge action and breaks strict gauge equivariance there. The deviation is zero at
the identity init and grows as :math:`A` drifts from :math:`I` during training. This is an
accepted, opt-in departure (the no-mixer path is the default and stays equivariant).

The ``tied_block_glk`` group (generators ``kron(I_n, gl(d))``, one shared frame across heads)
restores exact equivariance: under a tied gauge :math:`\Omega = \mathrm{kron}(I_n, h)`,
:math:`M = \mathrm{kron}(A, I_d)` commutes with :math:`\Omega`, so the FULL-COVARIANCE mixer is
exactly equivariant -- :math:`\mathrm{mix}(\Omega\mu, \Omega\Sigma\Omega^\top) = (\Omega M\mu,
\Omega M\Sigma M^\top \Omega^\top)` (pinned by ``test_head_mixer_equivariant_under_tied_gauge_full_cov``).
This is a statement about the MIXER OPERATION, not a claim that the whole model is gauge-equivariant.
CAVEAT: the diagonal closed form :math:`\sigma'[m] = \sum_n A[m,n]^2 \sigma[n]` is equivariant only
under DIAGONAL gauges (the diagonal-of-sandwich approximation V3 already uses when
``diagonal_covariance=True``), not under a general tied gauge.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn


class HeadMixer(nn.Module):
    r"""Isotypic per-component mixer: one :math:`A_t = I + \Delta_t` per maximal run of
    equal-labeled blocks, embedded as :math:`\mathrm{blockdiag}_t(A_t \otimes I_{d_t})` --
    the full linear commutant of the tower for real-type irreps. Without labels the whole
    group must be one equal-dims component (the legacy behavior, byte-identical)."""

    def __init__(
        self,
        irrep_dims:   List[int],                      # gauge block sizes
        irrep_labels: Optional[List[str]] = None,     # per-block labels; None -> legacy equal-dims
    ) -> None:
        super().__init__()
        if len(irrep_dims) < 2:
            raise ValueError(
                f"HeadMixer needs >= 2 blocks to mix, got irrep_dims={irrep_dims}; a single-block "
                f"group (glk / so_k) has nothing to mix. Use block_glk (n_heads >= 2)."
            )
        if irrep_labels is None:
            if len(set(irrep_dims)) != 1:
                raise ValueError(
                    f"HeadMixer needs equal-size blocks for kron(A, I_d), got "
                    f"irrep_dims={irrep_dims}. A labeled irrep tower (so_n/sp_n) mixes per "
                    f"isotypic component instead."
                )
            runs = [(0, len(irrep_dims))]                       # one component: all blocks
        else:
            runs, i = [], 0                                     # maximal runs of equal labels
            while i < len(irrep_dims):
                j = i
                while j < len(irrep_dims) and irrep_labels[j] == irrep_labels[i]:
                    j += 1
                runs.append((i, j))
                i = j
        # components: (coordinate start, copies m, block dim d); spec layout makes runs contiguous
        starts = [0]
        for d in irrep_dims:
            starts.append(starts[-1] + d)
        self.components = [(starts[i], j - i, irrep_dims[i]) for i, j in runs]
        self.mixer_deltas = nn.ParameterList(
            nn.Parameter(torch.zeros(m, m)) for _, m, _ in self.components
        )

    @property
    def mixer_delta(self) -> nn.Parameter:
        r"""Back-compat accessor for the single-component (legacy equal-dims) mixer."""
        if len(self.mixer_deltas) != 1:
            raise AttributeError("mixer_delta is single-component only; use mixer_deltas")
        return self.mixer_deltas[0]

    def _A(self, t: int) -> torch.Tensor:
        d = self.mixer_deltas[t]
        return torch.eye(d.shape[0], device=d.device, dtype=d.dtype) + d

    def is_identity(self) -> bool:
        return all(bool((d.detach() == 0).all().item()) for d in self.mixer_deltas)

    def forward(
        self,
        mu:    torch.Tensor,             # (..., K) belief means
        sigma: torch.Tensor,             # (..., K) diagonal variances OR (..., K, K) full covariance
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_parts, sig_parts = [], []
        for t, (s, m, d) in enumerate(self.components):
            A = self._A(t)
            blk = mu[..., s:s + m * d].reshape(*mu.shape[:-1], m, d)
            mu_parts.append(torch.einsum("mn,...nd->...md", A, blk)
                            .reshape(*mu.shape[:-1], m * d))
            if sigma.dim() == mu.dim():                          # diagonal closed form
                sblk = sigma[..., s:s + m * d].reshape(*sigma.shape[:-1], m, d)
                sig_parts.append(torch.einsum("mn,...nd->...md", A * A, sblk)
                                 .reshape(*sigma.shape[:-1], m * d))
        mu_out = torch.cat(mu_parts, dim=-1)
        if sigma.dim() == mu.dim():
            return mu_out, torch.cat(sig_parts, dim=-1)
        # full covariance: exact sandwich M Sigma M^T with the block-diagonal commutant M
        M = self._dense_m(sigma.device, sigma.dtype)             # (K, K)
        return mu_out, M @ sigma @ M.transpose(-1, -2)

    def _dense_m(self, device, dtype) -> torch.Tensor:
        r"""blockdiag_t(A_t kron I_d) materialized once per call (K x K, full-cov path only)."""
        K = sum(m * d for _, m, d in self.components)
        M = torch.zeros(K, K, device=device, dtype=dtype)
        for t, (s, m, d) in enumerate(self.components):
            M[s:s + m * d, s:s + m * d] = torch.kron(
                self._A(t).to(device=device, dtype=dtype), torch.eye(d, device=device, dtype=dtype))
        return M
