r"""Clebsch-Gordan between-block coupling for irrep towers (opt-in; default off).

The only exactly-equivariant cross-type information flow: linear equivariant maps between
inequivalent irreps are zero (Schur), so the coupling is BILINEAR through the numerically
solved CG intertwiners,

    mu'^(c,r) = mu^(c,r) + sum_p w_p C_p( mu^(a,i), mu^(b,j) ),

one learned scalar w_p per path (source copy pair x target copy x multiplicity slot),
zero-initialized so step 0 is byte-identical to the coupling-off path. Equivariance holds
for ANY weights because the weights multiply intertwiners. Covariance is MEANS-ONLY in
this phase: sigma passes through untouched (a bilinear map of Gaussians has no closed-form
pushforward; the honest sigma treatment belongs to the deferred F-term phase -- see the
2026-06-09 design spec). NEURAL-NETWORK EXCEPTION (sanctioned, default-off), the
use_head_mixer family.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from vfe3.geometry.cg import cg_intertwiners, cg_selection


class CGCoupling(nn.Module):
    r"""Bilinear CG coupling over the blocks of a labeled irrep tower."""

    def __init__(
        self,
        group_n:      int,                       # N of SO(N) / 2m of Sp(2m)
        algebra:      str,                       # 'so' | 'sp'
        irrep_dims:   List[int],                 # per-block dims
        irrep_labels: Optional[List[str]],       # per-block labels (REQUIRED non-None)
    ) -> None:
        super().__init__()
        if irrep_labels is None:
            raise ValueError(
                "CGCoupling requires a labeled irrep tower (gauge_group 'so_n'/'sp_n')"
            )
        starts = [0]
        for d in irrep_dims:
            starts.append(starts[-1] + d)
        blocks = list(zip(irrep_labels, starts[:-1], irrep_dims))     # (label, start, d)

        # one stacked intertwiner buffer per admissible type triple
        triples = cg_selection(group_n, algebra=algebra, labels=irrep_labels)
        self._triple_index = {}
        for t, (a, b, c, _n) in enumerate(triples):
            C = cg_intertwiners(group_n, algebra=algebra,
                                label_a=a, label_b=b, label_c=c)      # (n_mult, dc, da*db)
            # Stored float64 (the solver's construction precision). A model-wide .float()/
            # .double() converts buffers like everything else; forward re-casts to mu's dtype
            # from the cast cache either way, so numerics follow the RUNTIME dtype -- float64
            # storage just avoids a lossy fp32 round-trip before the cast.
            self.register_buffer(f"cg_{t}", C)
            self._triple_index[(a, b, c)] = t

        # paths: source copy pair (i <= j for equal labels) x target copy x multiplicity slot
        self.paths: List[Tuple[int, int, int, int, int, int, int, int]] = []
        self.path_types: List[Tuple[str, str, str]] = []
        for (a, b, c, n_mult) in triples:
            t = self._triple_index[(a, b, c)]
            srcs_a = [(s, d) for lab, s, d in blocks if lab == a]
            srcs_b = [(s, d) for lab, s, d in blocks if lab == b]
            tgts_c = [(s, d) for lab, s, d in blocks if lab == c]
            for ia, (sa, da) in enumerate(srcs_a):
                for jb, (sb, db) in enumerate(srcs_b):
                    if a == b and jb < ia:                            # unordered copies
                        continue
                    for (sc, dc) in tgts_c:
                        for m in range(n_mult):
                            self.paths.append((sa, da, sb, db, sc, dc, t, m))
                            self.path_types.append((a, b, c))
        if not self.paths:
            raise ValueError(
                f"CGCoupling found no admissible CG paths for labels {irrep_labels} "
                f"(algebra {algebra!r}, N={group_n}); disable use_cg_coupling"
            )
        self.path_weights = nn.Parameter(torch.zeros(len(self.paths)))
        self._cast_cache: Dict[Tuple[torch.dtype, torch.device], List[torch.Tensor]] = {}
        # maps (dtype, device) -> list of cast intertwiner stacks, lazily built, cleared on _apply

    def _apply(self, fn, recurse=True):
        # .to()/.float()/.cuda() move or convert the buffers; drop the stale cast cache.
        self._cast_cache.clear()
        return super()._apply(fn, recurse)

    def _cast_buffers(self, dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
        key = (dtype, device)
        cached = self._cast_cache.get(key)
        if cached is None:
            n_triples = len(self._triple_index)
            cached = [getattr(self, f"cg_{t}").to(dtype=dtype, device=device)
                      for t in range(n_triples)]
            self._cast_cache[key] = cached
        return cached

    def is_identity(self) -> bool:
        return bool((self.path_weights.detach() == 0).all().item())

    def forward(
        self,
        mu:    torch.Tensor,             # (..., K) belief means
        sigma: torch.Tensor,             # (..., K) or (..., K, K); passes through UNTOUCHED
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cg = self._cast_buffers(mu.dtype, mu.device)              # per-(dtype, device), cached
        w = self.path_weights.to(dtype=mu.dtype)                  # one cast, differentiable
        delta = torch.zeros_like(mu)
        for p, (sa, da, sb, db, sc, dc, t, m) in enumerate(self.paths):
            x = mu[..., sa:sa + da]
            y = mu[..., sb:sb + db]
            xy = (x.unsqueeze(-1) * y.unsqueeze(-2)).reshape(*x.shape[:-1], da * db)
            delta[..., sc:sc + dc] = delta[..., sc:sc + dc] \
                + w[p] * torch.einsum("cd,...d->...c", cg[t][m], xy)
        return mu + delta, sigma
