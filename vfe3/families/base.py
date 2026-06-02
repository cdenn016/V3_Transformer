r"""The exponential-family parameter layer for VFE_3.0.

A ``BeliefParams`` is a batched parameter container plus the family's math
(natural<->moment, log-partition A(theta), entropy, divergences). The divergence
functional ``renyi`` (KL = alpha 1) dispatches on the parameter object: a family with a
``renyi_closed_form`` method uses it (the pinned Gaussian moment forms); a family that
defines only ``log_partition_at`` (and ``natural``/``expected_statistic``) gets the
generic Bregman/Renyi-from-A divergence for free. This is the seam a new exponential
family slots in behind -- by writing-and-registering a subclass, never editing call sites.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Dict, Tuple, Type

import torch


def safe_kl_clamp(
    kl:     torch.Tensor,

    *,
    kl_max: float = 100.0,
) -> torch.Tensor:
    r"""Clamp to [0, kl_max]; map NaN/+inf -> kl_max, -inf -> 0."""
    kl = kl.clamp(min=0.0, max=kl_max)
    return kl.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)


def _warn_alpha_gt_one(alpha: float, family: str) -> None:
    r"""Warn that alpha > 1 leaves the convex regime of the Renyi blend."""
    warnings.warn(
        f"renyi: alpha={alpha} > 1 (family={family!r}) leaves the convex regime; "
        f"the blend (1-alpha)*Sigma_q + alpha*Sigma_t may be non-positive-definite "
        f"(diagonal clamps; full may fail Cholesky and return NaN).",
        RuntimeWarning,
        stacklevel=3,
    )


def _logdet_chol(L: torch.Tensor) -> torch.Tensor:
    r"""log|Sigma| for SPD Sigma = L Lᵀ from its Cholesky factor L."""
    return 2.0 * torch.log(
        torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-12)
    ).sum(dim=-1)


class BeliefParams(ABC):
    r"""Batched parameters of an exponential family, with the family's behavior.

    Concrete subclasses hold the family's tensors (with arbitrary leading batch dims and a
    trailing coordinate structure) and implement the interface below. ``cov_kind`` is the
    single source of truth for the covariance structure (replacing name sniffing).

    Override tiers (what a new family must provide):
      - ALWAYS required (abstract): ``coordinate_dim``, ``block``, ``broadcast_over_keys``,
        ``natural``, ``log_partition_at``, ``entropy``, and the ``cov_kind`` class attribute.
      - Required ONLY to use the generic divergence path: ``expected_statistic`` (= gradA, the
        mean of the sufficient statistics), consumed by the generic KL (alpha = 1). A family
        that supplies its own ``renyi_closed_form`` never hits the generic path and need not
        override it; the base raises a clear error if the generic path needs it and it is absent.
      - Optional hooks: ``renyi_closed_form(other, *, alpha, kl_max, eps)`` (a pinned closed form
        that bypasses the generic A-path) and ``renyi_per_coord(other, ...)`` (the unsummed
        per-coordinate divergence, defined only for families whose divergence decomposes).
    """

    cov_kind: ClassVar[str]

    @abstractmethod
    def coordinate_dim(self) -> int:
        r"""K, the number of belief coordinates."""

    @abstractmethod
    def block(self, start: int, end: int) -> "BeliefParams":
        r"""The parameters restricted to coordinate block [start, end) (per-irrep slice)."""

    @abstractmethod
    def broadcast_over_keys(self) -> "BeliefParams":
        r"""Insert a singleton key axis so a query (..., N, K) broadcasts against keys
        (..., N, N, K) in the pairwise energy."""

    @abstractmethod
    def natural(self) -> Tuple[torch.Tensor, ...]:
        r"""Natural parameters theta from these (moment) parameters."""

    @classmethod
    @abstractmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""Log-partition A(theta) at arbitrary natural coordinates theta."""

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        r"""Differential entropy H of this distribution."""

    def expected_statistic(self) -> Tuple[torch.Tensor, ...]:
        r"""E_q[T] = gradA(theta), the mean of the sufficient statistics, aligned with
        ``natural()``. Consumed by the generic KL (alpha = 1) Bregman form. Not abstract:
        a family that supplies ``renyi_closed_form`` never reaches the generic path and need
        not override this. Overriding is required only to use the generic KL path."""
        raise NotImplementedError(
            f"{type(self).__name__} has no renyi_closed_form and does not override "
            f"expected_statistic, which the generic KL (alpha=1) path requires (it is gradA, "
            f"the mean of the sufficient statistics). Provide either method."
        )


_FAMILIES: Dict[str, Type[BeliefParams]] = {}


def register_family(name: str) -> Callable:
    r"""Register a ``BeliefParams`` subclass under ``name`` (the config ``family`` value)."""
    def _wrap(cls: Type[BeliefParams]) -> Type[BeliefParams]:
        _FAMILIES[name] = cls
        return cls
    return _wrap


def get_family(name: str) -> Type[BeliefParams]:
    r"""The registered ``BeliefParams`` subclass for ``name`` (KeyError if absent)."""
    if name not in _FAMILIES:
        raise KeyError(f"no family registered under {name!r}; available: {sorted(_FAMILIES)}")
    return _FAMILIES[name]


def family_cov_kind(name: str) -> str:
    r"""Covariance structure ("diagonal" | "full") of family ``name``, from its subclass."""
    return get_family(name).cov_kind


def divergence_families() -> Tuple[str, ...]:
    r"""Registered family names (the valid ``family`` config values)."""
    return tuple(sorted(_FAMILIES))


_FUNCTIONALS: Dict[str, Callable] = {}


def register_functional(name: str) -> Callable:
    r"""Register a divergence functional (renyi, ...) under ``name`` (the ``divergence_family``)."""
    def _wrap(fn: Callable) -> Callable:
        _FUNCTIONALS[name] = fn
        return fn
    return _wrap


def get_functional(name: str) -> Callable:
    r"""The registered divergence functional for ``name`` (KeyError if absent)."""
    if name not in _FUNCTIONALS:
        raise KeyError(f"no functional registered under {name!r}; available: {sorted(_FUNCTIONALS)}")
    return _FUNCTIONALS[name]


def _renyi_from_log_partition(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    alpha:   float,
    kl_max:  float,
    eps:     float,
) -> torch.Tensor:
    r"""Generic Renyi/KL from the log-partition (for families with no closed form).

    alpha != 1:  R = 1/(alpha-1) [ A(alpha*tq + (1-alpha)*tp) - alpha*A(tq) - (1-alpha)*A(tp) ].
    alpha == 1:  KL = A(tp) - A(tq) - <gradA(tq), tp - tq>, gradA(tq) = E_q[T] (expected_statistic).

    ``eps`` is accepted for signature symmetry with the closed-form path and is intentionally
    unused here: the A-form is evaluated directly from the natural parameters with no clamp.
    """
    cls = type(q)
    tq = q.natural()
    tp = p.natural()
    if abs(alpha - 1.0) < 1e-6:
        grad = q.expected_statistic()                       # E_q[T] = gradA(theta_q)
        A_q = cls.log_partition_at(tq)                       # batch-shaped (parameter axes summed)
        batch_ndim = A_q.dim()
        # Bregman KL = A(tp) - A(tq) - sum_c <gradA_c, (tp - tq)_c>, where each natural-parameter
        # component c is contracted over ITS parameter axes (the trailing dims beyond the batch).
        # A vector statistic (..., K) sums the last axis; a matrix statistic (..., K, K) is
        # Frobenius-contracted over the last two -- so a matrix-parameter family (e.g. the full
        # Gaussian's t2) works through the generic path, not only vector-parameter families.
        inner = 0.0
        for g, a, b in zip(grad, tq, tp):
            term = g * (b - a)
            param_axes = tuple(range(batch_ndim, term.dim()))
            inner = inner + (term.sum(dim=param_axes) if param_axes else term)
        div = cls.log_partition_at(tp) - A_q - inner
    else:
        blend = tuple(alpha * a + (1.0 - alpha) * b for a, b in zip(tq, tp))
        div = (cls.log_partition_at(blend)
               - alpha * cls.log_partition_at(tq)
               - (1.0 - alpha) * cls.log_partition_at(tp)) / (alpha - 1.0)
    return safe_kl_clamp(div, kl_max=kl_max)


def renyi(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
) -> torch.Tensor:
    r"""Renyi alpha-divergence D_alpha(q || p) between two parameter objects (KL at alpha=1).

    Uses ``q.renyi_closed_form`` when the family provides one (the pinned Gaussian moment
    form); otherwise the generic Bregman/Renyi-from-A path.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if alpha > 1.0:
        _warn_alpha_gt_one(alpha, type(q).__name__)
    closed = getattr(q, "renyi_closed_form", None)
    if closed is not None:
        return closed(p, alpha=alpha, kl_max=kl_max, eps=eps)
    return _renyi_from_log_partition(q, p, alpha=alpha, kl_max=kl_max, eps=eps)


def kl(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
) -> torch.Tensor:
    r"""KL(q || p) = Renyi at alpha = 1."""
    return renyi(q, p, alpha=1.0, kl_max=kl_max, eps=eps)


register_functional("renyi")(renyi)
