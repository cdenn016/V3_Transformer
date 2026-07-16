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
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Type

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

    cov_kind:                 ClassVar[str]
    dispersion_is_covariance: ClassVar[bool] = False

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

    def natural_gradient(
        self,
        grad_mu:    torch.Tensor,                    # (..., K) Euclidean grad wrt mu (location)
        grad_sigma: Optional[torch.Tensor],          # None freezes sigma; else Euclidean grad wrt sigma

        *,
        eps:        float = 1e-6,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:   # (nat_mu, nat_sigma) = Fisher^{-1} grad
        r"""Fisher preconditioner: Euclidean -> natural gradient for THIS family, ``g^{-1} grad``
        with ``g`` the family's Fisher-Rao metric on its (location, scale) coordinates. The single
        seam the E-step's preconditioner step dispatches through (``e_step_iteration``), so a family
        is descended in ITS OWN information geometry rather than a hardcoded Gaussian Fisher (which
        is the wrong DIRECTION on a non-Gaussian product manifold, not a rescalable learning rate).
        Not abstract -- existing families need no edit until they opt in; the base raises because the
        correct metric is family-dependent. ``grad_sigma=None`` requests the family-aware mean
        block only and must return ``nat_sigma=None`` without evaluating the sigma block."""
        raise NotImplementedError(
            f"{type(self).__name__} does not override natural_gradient, the family's Fisher "
            f"preconditioner (Euclidean -> natural gradient). Provide it so the E-step descends in "
            f"this family's information geometry rather than the hardcoded Gaussian Fisher."
        )

    @classmethod
    def covariance_diagonal(
        cls,
        dispersion: torch.Tensor,                    # (..., K) diagonal or (..., K, K) full parameter

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:                               # (..., K) marginal covariance diagonal
        r"""Map the stored family dispersion to ``diag(Cov[X])``.

        The belief state's ``sigma`` slot is deliberately family-generic: it stores a Gaussian
        variance/covariance but a Laplace scale. Consumers that need uncertainty, rather than the
        raw parameter, dispatch through this hook. A family must override it because covariance
        semantics cannot be inferred from ``cov_kind`` alone.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not provide covariance_diagonal for its dispersion parameter"
        )

    @classmethod
    def mean_fisher_precision(
        cls,
        dispersion: torch.Tensor,                    # (..., K) diagonal or (..., K, K) full parameter

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:                               # (..., K) diagonal or (..., K, K) precision
        r"""Mean-block Fisher information in the family's stored coordinates."""
        raise NotImplementedError(
            f"{cls.__name__} does not provide mean_fisher_precision"
        )

    @classmethod
    def trust_region_scale(
        cls,
        dispersion: torch.Tensor,                    # (..., K) diagonal or (..., K, K) full parameter

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:                               # (..., K) scale or (..., K, K) whitening factor
        r"""Factor ``L`` used to whiten a mean perturbation in this family's Fisher geometry."""
        raise NotImplementedError(
            f"{cls.__name__} does not provide trust_region_scale"
        )

    @classmethod
    def mix_dispersion(
        cls,
        dispersion: torch.Tensor,                    # (..., n, d) or (..., K, K) family parameter
        mixing:    torch.Tensor,                     # (m, n) component map or (K, K) dense map
    ) -> torch.Tensor:
        r"""Push the stored dispersion through an independent-component linear mixer."""
        raise NotImplementedError(
            f"{cls.__name__} does not provide mix_dispersion"
        )

    @classmethod
    def diagnostic_labels(cls) -> Dict[str, str]:
        r"""Human-readable labels distinguishing stored dispersion from covariance statistics."""
        raise NotImplementedError(
            f"{cls.__name__} does not provide diagnostic_labels"
        )

    @classmethod
    def diagnostic_statistics(
        cls,
        dispersion: torch.Tensor,                    # (..., K) diagonal or (..., K, K) full parameter

        *,
        eps:        float = 1e-12,
    ) -> Dict[str, torch.Tensor]:
        r"""Family-aware covariance spectrum and trace without computing Fisher inverses."""
        covariance_diagonal = cls.covariance_diagonal(dispersion, eps=eps)
        if cls.cov_kind == "full":
            covariance = 0.5 * (dispersion + dispersion.transpose(-1, -2))
            covariance_spectrum = torch.linalg.eigvalsh(covariance)
        else:
            covariance_spectrum = covariance_diagonal
        return {
            "covariance_diagonal": covariance_diagonal,
            "covariance_spectrum": covariance_spectrum,
            "covariance_trace":    covariance_diagonal.sum(dim=-1),
        }

    @classmethod
    def stack(cls, parts: List["BeliefParams"], *, dim: int = 0) -> "BeliefParams":
        r"""Stack a list of same-family parts into one ``BeliefParams`` along a NEW axis ``dim``,
        stacking each underlying tensor of the parts. Family-agnostic batching primitive: a single
        functional call over the stacked axis then computes every part's divergence at once (used by
        ``pairwise_energy`` to batch the per-irrep-block loop). Not abstract -- a family that never
        takes the batched path need not override it. Overriding is required only to batch."""
        raise NotImplementedError(
            f"{cls.__name__} does not override stack, the family-agnostic batching primitive "
            f"(stack each part's underlying tensor along a new axis). Provide it to batch the "
            f"per-block loop."
        )

    @classmethod
    def transport_dispersion(
        cls,
        dispersion: torch.Tensor,         # (..., N, K) diagonal or (..., N, K, K) full parameter
        omega:      object,               # dense/factored/direct-link/RoPE transport container

        *,
        diagonal_out: Optional[bool] = None,
    ) -> torch.Tensor:
        r"""Transport the family's dispersion parameter under ``omega``.

        Covariance-parameterized families inherit the congruence action. A location-scale family
        overrides this seam when its stored parameter has a different homogeneity degree. The lazy
        import avoids a module cycle while keeping family selection in the family registry rather
        than hard-coding family names into geometry or inference call sites.
        """
        from vfe3.geometry.transport import transport_covariance
        return transport_covariance(omega, dispersion, diagonal_out=diagonal_out)


_FAMILIES: Dict[str, Type[BeliefParams]] = {}


def register_family(name: str, *, override: bool = False) -> Callable[[Type[BeliefParams]], Type[BeliefParams]]:
    r"""Register a ``BeliefParams`` subclass under ``name`` (the config ``family`` value).

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(cls: Type[BeliefParams]) -> Type[BeliefParams]:
        if name in _FAMILIES and not override:
            raise KeyError(f"family {name!r} already registered; pass override=True to replace")
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


_FUNCTIONALS: Dict[str, Callable[..., torch.Tensor]] = {}


def register_functional(name: str, *, override: bool = False) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    r"""Register a divergence functional (renyi, ...) under ``name`` (the ``divergence_family``).

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        if name in _FUNCTIONALS and not override:
            raise KeyError(f"divergence functional {name!r} already registered; pass override=True to replace")
        _FUNCTIONALS[name] = fn
        return fn
    return _wrap


def get_functional(name: str) -> Callable[..., torch.Tensor]:
    r"""The registered divergence functional for ``name`` (KeyError if absent)."""
    if name not in _FUNCTIONALS:
        raise KeyError(f"no functional registered under {name!r}; available: {sorted(_FUNCTIONALS)}")
    return _FUNCTIONALS[name]


def divergence_functionals() -> Tuple[str, ...]:
    r"""Registered functional names (the valid ``divergence_family`` config values)."""
    return tuple(sorted(_FUNCTIONALS))


# Per-COORDINATE divergence registry (the unsummed (..., K) form consumed by the
# state_dependent_per_coord alpha). A divergence has a per-coordinate form ONLY when it decomposes
# as a sum over the diagonal-Gaussian coordinates: Renyi/KL does (renyi_per_coord), and so do the
# divergences that are AFFINE in the Renyi divergence -- Bhattacharyya (0.5 D_{1/2}) and Jeffreys
# (KL + KL_rev). squared_hellinger is deliberately ABSENT: H^2 = 1 - exp(-D_{1/2}/2) is a nonlinear
# transform of the SUMMED divergence and does not split coordinate-wise. A functional with no member
# here is rejected by ``free_energy.self_divergence_per_coord`` (and at config construction).
_FUNCTIONALS_PER_COORD: Dict[str, Callable[..., torch.Tensor]] = {}


def register_functional_per_coord(name: str, *, override: bool = False) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    r"""Register a PER-COORDINATE divergence functional under ``name`` (the ``divergence_family``).

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        if name in _FUNCTIONALS_PER_COORD and not override:
            raise KeyError(f"per-coordinate divergence functional {name!r} already registered; pass override=True to replace")
        _FUNCTIONALS_PER_COORD[name] = fn
        return fn
    return _wrap


def get_functional_per_coord(name: str) -> Callable[..., torch.Tensor]:
    r"""The registered per-coordinate functional for ``name`` (KeyError if absent)."""
    if name not in _FUNCTIONALS_PER_COORD:
        raise KeyError(
            f"no per-coordinate functional registered under {name!r}; available: "
            f"{sorted(_FUNCTIONALS_PER_COORD)}"
        )
    return _FUNCTIONALS_PER_COORD[name]


def has_per_coord_functional(name: str) -> bool:
    r"""Whether divergence ``name`` has a registered per-coordinate (coordinate-decomposing) form."""
    return name in _FUNCTIONALS_PER_COORD


def divergence_functionals_per_coord() -> Tuple[str, ...]:
    r"""Registered per-coordinate functional names (the divergences that decompose coordinate-wise)."""
    return tuple(sorted(_FUNCTIONALS_PER_COORD))


# fp32 catastrophic-cancellation band around the alpha->1 (KL) limit of the generic Renyi A-form:
# outside the |alpha-1| < 1e-6 KL switch but inside this band the three nearly-equal log-partition
# values cancel before the /(alpha-1) divide, losing ~1% accuracy in float32 out to ~|alpha-1| ~ 1e-3.
# Inside the band the A-form is evaluated in float64 and cast back. Kept as a separate constant from
# gaussian._RENYI_KL_BAND because base.py cannot import gaussian (gaussian imports base).
_RENYI_KL_BAND: float = 1e-2


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
        inner: 'torch.Tensor | float' = 0.0          # float seed; becomes a Tensor on the first term
        for g, a, b in zip(grad, tq, tp):
            term = g * (b - a)
            param_axes = tuple(range(batch_ndim, term.dim()))
            inner = inner + (term.sum(dim=param_axes) if param_axes else term)
        div = cls.log_partition_at(tp) - A_q - inner
    elif abs(alpha - 1.0) < _RENYI_KL_BAND:
        # fp32 cancellation band: evaluate the three log-partitions in float64, then cast back
        # (mirrors the closed-form gaussian._RENYI_KL_BAND float64 island).
        tq64    = tuple(t.double() for t in tq)
        tp64    = tuple(t.double() for t in tp)
        blend64 = tuple(alpha * a + (1.0 - alpha) * b for a, b in zip(tq64, tp64))
        div = ((cls.log_partition_at(blend64)
                - alpha * cls.log_partition_at(tq64)
                - (1.0 - alpha) * cls.log_partition_at(tp64)) / (alpha - 1.0)).to(tq[0].dtype)
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
    **kwargs,
) -> torch.Tensor:
    r"""Renyi alpha-divergence D_alpha(q || p) between two parameter objects (KL at alpha=1).

    Uses ``q.renyi_closed_form`` when the family provides one (the pinned Gaussian moment
    form); otherwise the generic Bregman/Renyi-from-A path.

    The trailing ``**kwargs`` is the permissive functional contract every divergence-registry
    member shares (a member ignores params it does not use); ``renyi`` consumes ``alpha``.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if alpha > 1.0 + 1e-6:                              # only when the closed form takes the blend
        _warn_alpha_gt_one(alpha, type(q).__name__)     # branch; alpha in (1, 1+1e-6] is plain KL
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


def squared_hellinger(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    **kwargs,
) -> torch.Tensor:                             # (...) squared Hellinger H^2(q||p) in [0, 1]
    r"""Squared Hellinger f-divergence H^2(q || p) = 1 - BC(q || p).

    For Gaussians the Bhattacharyya coefficient is BC = exp(-D_{1/2}(q||p)/2), where D_{1/2}
    is the Renyi-1/2 divergence the pinned ``renyi`` kernel already computes, so

        H^2(q || p) = 1 - exp( -D_{1/2}(q || p) / 2 ),   D_{1/2} = renyi(q, p, alpha=0.5).

    This member ignores any ``alpha`` the call sites forward (Hellinger has no order); it is
    absorbed by ``**kwargs`` and never reaches ``renyi`` (the inner call always uses alpha=0.5,
    so the alpha>1 blend warning cannot fire). ``kl_max`` IS forwarded so the inner D_{1/2}
    stays bounded in [0, kl_max]; the H^2 output is then naturally in [0, 1] without a second
    clamp (mathematically the range is the half-open [0, 1), but in float32 ``1 - exp(-0.5 d_half)``
    saturates to exactly 1.0 for d_half >= ~35, so at the default kl_max=100 a clamped
    D_{1/2}=kl_max maps to H^2 = 1.0, the maximal-Hellinger limit, which composes correctly).
    """
    d_half = renyi(q, p, alpha=0.5, kl_max=kl_max, eps=eps)
    return 1.0 - torch.exp(-0.5 * d_half)


def bhattacharyya(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    **kwargs,
) -> torch.Tensor:                             # (...) Bhattacharyya distance D_B(q||p) >= 0
    r"""Bhattacharyya distance D_B(q || p) = -log BC(q || p), BC the Bhattacharyya coefficient.

    For Gaussians BC = exp(-D_{1/2}(q||p)/2) (the same coefficient squared-Hellinger uses), so

        D_B(q || p) = D_{1/2}(q || p) / 2,   D_{1/2} = renyi(q, p, alpha=0.5).

    SYMMETRIC (the Renyi-1/2 divergence is) and zero iff q == p. Reuses the pinned Gaussian renyi
    closed form at alpha=0.5; ``kl_max`` bounds the inner D_{1/2}. A forwarded ``alpha`` is absorbed
    by ``**kwargs`` (Bhattacharyya has no order) and never reaches the inner renyi call (always 0.5).
    """
    return 0.5 * renyi(q, p, alpha=0.5, kl_max=kl_max, eps=eps)


def jeffreys(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    **kwargs,
) -> torch.Tensor:                             # (...) Jeffreys (symmetrized KL) J(q||p) >= 0
    r"""Jeffreys divergence J(q || p) = KL(q || p) + KL(p || q), the symmetrized KL.

    Each term is renyi at alpha=1 (the pinned Gaussian KL closed form):

        J(q || p) = renyi(q, p, alpha=1) + renyi(p, q, alpha=1).

    SYMMETRIC by construction and zero iff q == p. Each KL is clamped to ``kl_max``, so J is bounded
    in [0, 2*kl_max]. A forwarded ``alpha`` is absorbed by ``**kwargs`` (Jeffreys has no order); the
    two inner calls always use alpha=1.
    """
    return (renyi(q, p, alpha=1.0, kl_max=kl_max, eps=eps)
            + renyi(p, q, alpha=1.0, kl_max=kl_max, eps=eps))


def renyi_per_coord(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    **kwargs,
) -> torch.Tensor:                             # (..., K) per-coordinate Renyi/KL D^(k)
    r"""Per-coordinate Renyi/KL: the unsummed diagonal coordinate terms (family hook).

    Defined only for a family exposing ``renyi_per_coord`` (the diagonal Gaussian); the caller
    (``free_energy.self_divergence_per_coord``) guards ``cov_kind == 'diagonal'`` first.
    """
    return q.renyi_per_coord(p, alpha=alpha, kl_max=kl_max, eps=eps)


def bhattacharyya_per_coord(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    **kwargs,
) -> torch.Tensor:                             # (..., K) per-coordinate Bhattacharyya D_B^(k)
    r"""Per-coordinate Bhattacharyya D_B^(k) = 0.5 D_{1/2}^(k); sum_k recovers ``bhattacharyya``.

    Bhattacharyya is AFFINE in the Renyi-1/2 divergence (D_B = 0.5 D_{1/2}), so it decomposes
    coordinate-wise as 0.5 times the per-coordinate Renyi-1/2. A forwarded ``alpha`` is absorbed
    (Bhattacharyya has no order; the inner per-coord call always uses alpha=0.5).
    """
    return 0.5 * q.renyi_per_coord(p, alpha=0.5, kl_max=kl_max, eps=eps)


def jeffreys_per_coord(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    **kwargs,
) -> torch.Tensor:                             # (..., K) per-coordinate Jeffreys J^(k)
    r"""Per-coordinate Jeffreys J^(k) = KL^(k)(q||p) + KL^(k)(p||q); sum_k recovers ``jeffreys``.

    Jeffreys is a SUM of two KLs, each of which decomposes coordinate-wise (per-coord Renyi at
    alpha=1), so the symmetrized divergence decomposes too. Both q and p are diagonal Gaussians
    (the belief and its prior). A forwarded ``alpha`` is absorbed (the inner calls always use 1).
    """
    return (q.renyi_per_coord(p, alpha=1.0, kl_max=kl_max, eps=eps)
            + p.renyi_per_coord(q, alpha=1.0, kl_max=kl_max, eps=eps))


register_functional("renyi")(renyi)
register_functional("squared_hellinger")(squared_hellinger)
register_functional("bhattacharyya")(bhattacharyya)
register_functional("jeffreys")(jeffreys)

# Per-coordinate forms for the divergences that decompose coordinate-wise (see _FUNCTIONALS_PER_COORD).
# squared_hellinger is intentionally NOT registered here (non-additive outer transform).
register_functional_per_coord("renyi")(renyi_per_coord)
register_functional_per_coord("bhattacharyya")(bhattacharyya_per_coord)
register_functional_per_coord("jeffreys")(jeffreys_per_coord)
