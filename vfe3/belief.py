r"""The Gaussian belief tuple for VFE_3.0.

Form choice (roadmap M3): ``BeliefState`` stays a ``typing.NamedTuple`` and gains
trailing optional channel fields (``s``, ``r``) defaulting to ``None``, rather than
converting to a dataclass. A codebase-wide audit found every construction uses
keyword arguments (``BeliefState(mu=, sigma=, phi=)``) and every read uses attribute
access (``.mu/.sigma/.phi``); no site relies on a NamedTuple-only behavior that
trailing defaulted fields would break — no 3-way positional unpack of a belief, no
indexing, no iteration, no ``_replace``/``_asdict``. Trailing fields with ``None``
defaults preserve construction, attribute access, ``_replace``, indexing, and
iteration; only an N-way positional unpack at the OLD arity would change, and none
exists. This is the lowest-surface extensible form: a second belief channel (the
future hyper-prior ``s_i``/``r_i``, natural params, etc.) can be carried without a
signature sweep, and the 3-field default is byte-identical in behavior.
"""

from typing import NamedTuple, Optional

import torch
from vfe3.geometry.lie_ops import CompactBlockElement


class BeliefState(NamedTuple):
    """A per-token Gaussian belief q_i = N(mu_i, Sigma_i) with gauge frame phi_i."""

    mu:    torch.Tensor                       # (..., N, K) means
    sigma: torch.Tensor                       # (..., N, K) diagonal variances (or (..., N, K, K) full)
    phi:   torch.Tensor                       # (..., N, n_gen) gauge-frame coordinates

    s:     Optional[torch.Tensor] = None      # optional future hyper-prior channel s_i (None by default)
    r:     Optional[torch.Tensor] = None      # optional future hyper-prior channel r_i (None by default)
    omega: 'torch.Tensor | CompactBlockElement | None' = None  # optional stored GL(K) frame, dense or compact
    reflection: Optional[torch.Tensor] = None # (..., N) per-token sign +1/-1; set only on the phi path under phi_reflection
