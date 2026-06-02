# Families / ExponentialFamily seam + parameter-object divergence signature — Design Spec

Date: 2026-06-01
Status: APPROVED (architecture) — awaiting spec review before the implementation plan
Roadmap: `docs/2026-06-01-buildout-roadmap.md` punch-list item 1 (family seam) + addendum M2
Builds on: the `cov_kind` seam shipped 2026-06-01 (commit 71623aa)

## 1. Motivation

The spec `2026-05-29-vfe3-clean-room-design.md` section 4.1 names `families/` as the seam a
non-Gaussian exponential family lives behind: an `ExponentialFamily` interface exposing
natural-to-moment conversion, `log_partition(theta)`, and `entropy`, with `divergence.py`
dispatching through it. That layer does not exist. Two coupled gaps block it:

The divergence kernels are written directly in Gaussian moment coordinates (`divergence.py`),
and every divergence caller passes the fixed four-tensor signature `(mu_q, sigma_q, mu_t,
sigma_t)` — a mean tensor and a single (co)variance tensor per distribution. A family whose
parameters are not a `(mean, single (co)variance)` pair (a categorical, a gamma, a mixture)
cannot be passed through that signature at all, independent of whether a closed form exists
(buildout roadmap addendum M2). The covariance structure was, until the `cov_kind` seam,
sniffed from a name substring (`"diagonal" in family`); `cov_kind` fixed the dispatch but the
parameters themselves are still raw moment tensors.

This design generalizes the divergence interface to a family-typed parameter object and houses
the exponential-family abstraction behind it. The purpose is expandability: a future family is
added by writing-and-registering a `BeliefParams` subclass, never by editing the divergence
call sites. The current model is Gaussian throughout, so this ships with zero behavior change;
the payoff is that the interface stops assuming `(mean, covariance)`.

## 2. Goals and non-goals

Goals. A `families/` package with an `ExponentialFamily` / `BeliefParams` interface (natural and
moment parameters, `log_partition`, `entropy`, and divergences) plus `DiagonalGaussian` and
`FullGaussian` concrete families carrying the existing closed forms verbatim. A divergence layer
whose public functions (`renyi`, `kl`, `pairwise_energy`, `self_divergence*`) take and return
parameter objects. A generic Bregman / Renyi-from-`A(theta)` divergence any family inherits from
defining only `log_partition` and the natural-parameter map. Validation by a test-only toy
exponential family that defines only `A(theta)` and the natural map and flows end-to-end through
the generic path, proving the interface admits a non-`(mean, covariance)` family.

Non-goals. No new family ships in production (the toy family is test-only). The gauge / transport
layer and the SPD retraction stay Gaussian and tensor-based (spec section 3 walls them to the
location-scale structure); the parameter object is the interface at the divergence boundary, not
inside transport. The hand-derived analytic gradient kernel (`gradients/kernels.py`) stays
tensor-based and is not reached by this change. Mixture families remain out of scope (they are not
exponential families; their divergences have no closed form). A concrete categorical
observation-likelihood family and the gauge action on non-Gaussian families (Phase 5) are separate
roadmap items, not this one.

## 3. The interface

### 3.1 `families/base.py`

`BeliefParams` is an abstract dataclass: a batched parameter container plus the family behavior.
It carries the family's tensors (subclass fields) with arbitrary leading batch dims and a trailing
coordinate structure. Interface:

```
class BeliefParams(ABC):
    cov_kind: ClassVar[str]                      # "diagonal" | "full" | ...

    # --- layout operations the energy path needs ---
    def coordinate_dim(self) -> int              # K
    def block(self, start: int, end: int) -> "BeliefParams"     # per-irrep-block slice
    def broadcast_over_keys(self) -> "BeliefParams"             # query unsqueeze vs the key axis

    # --- exponential-family math ---
    def natural(self) -> Tuple[torch.Tensor, ...]              # theta from these moments
    @classmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, ...]) -> torch.Tensor   # A(theta)
    def entropy(self) -> torch.Tensor
```

The functional registry lives here as free functions on parameter objects:

```
def renyi(q: BeliefParams, p: BeliefParams, *, alpha=1.0, kl_max=100.0, eps=1e-6) -> Tensor
def kl(q, p, *, kl_max=100.0, eps=1e-6) -> Tensor          # renyi at alpha=1
```

`renyi` dispatches by checking the parameter object for an optional `renyi_closed_form(self, other,
*, alpha, kl_max, eps)` method: present (the Gaussian families) it is used (the pinned moment closed
form); absent (the toy family, future families that define only `A`) the generic identity

```
R_alpha(q || p) = 1/(alpha - 1) [ A(alpha*theta_q + (1-alpha)*theta_p)
                                  - alpha*A(theta_q) - (1-alpha)*A(theta_p) ]
```

is evaluated from `natural()` and `log_partition_at`. `safe_kl_clamp` and the `alpha > 1` warning
(the blended natural parameter leaving the domain) move here unchanged. `register_family(name)`
registers a `BeliefParams` subclass; `family_cov_kind(name)` derives `cov_kind` from the registered
subclass (subsuming the string registry shipped in 71623aa); `divergence_families()` returns the
registered names. The `config.py` validation added in 71623aa is unchanged (it still calls
`family_cov_kind` / `divergence_families`).

A per-coordinate divergence is an optional capability: families whose divergence decomposes
coordinate-wise (diagonal) provide `renyi_per_coord(q, p, ...) -> (..., K)`; others raise (matching
today's `self_divergence_per_coord` guard, now dispatched on the family).

### 3.2 `families/gaussian.py`

`DiagonalGaussian(BeliefParams)` — fields `mu (..., K)`, `sigma (..., K)` (variances);
`cov_kind = "diagonal"`. Carries the existing `_gaussian_diagonal_renyi` and
`gaussian_diagonal_renyi_per_coord` closed forms ported verbatim (byte-identical numerics).
`block` slices the coordinate axis; `broadcast_over_keys` does `unsqueeze(-2)` on `mu` and `sigma`.

`FullGaussian(BeliefParams)` — fields `mu (..., K)`, `sigma (..., K, K)`; `cov_kind = "full"`.
Carries the Cholesky `_gaussian_full_renyi` closed form verbatim; `block` slices `mu[..., a:b]` and
the marginal `sigma[..., a:b, a:b]`; `broadcast_over_keys` does `unsqueeze(-2)` on `mu` and
`unsqueeze(-3)` on `sigma`; `renyi_per_coord` raises (full covariance does not decompose).

Natural-parameter conventions (so the generic `A`-form is concrete and testable):

- Diagonal: moment `(mu_k, s_k)` with `s_k` the variance; natural `theta = (theta1, theta2)` with
  `theta1_k = mu_k / s_k`, `theta2_k = -1/(2 s_k)`. Log-partition
  `A(theta) = sum_k [ -theta1_k^2 / (4 theta2_k) - 1/2 log(-2 theta2_k) ]`. Entropy
  `H = 1/2 sum_k log(2 pi e s_k)`.
- Full: `theta1 = Sigma^{-1} mu`, `theta2 = -1/2 Sigma^{-1}`; `A(theta) = -1/4 theta1^T theta2^{-1}
  theta1 - 1/2 log|-2 theta2|`. The blended natural `alpha*theta_q + (1-alpha)*theta_p` must keep
  `theta2` negative-definite; this is exactly the `alpha > 1` domain boundary today's code clamps.

The generic Renyi-from-`A` for a Gaussian equals these moment closed forms; this identity is
verified symbolically (sympy) and pinned by a test before the generic path is relied on.

### 3.3 `divergence.py` after the change

A thin façade: `renyi`/`kl` re-export the `families` functionals; `safe_kl_clamp` stays; the
`_gaussian_*_renyi` kernels move into `gaussian.py` as the families' closed forms. The
`_FUNCTIONALS` registry (the `divergence_family = "renyi"` functional seam) is preserved as the
free-function functional layer, now operating on parameter objects, so a future f-divergence
(Hellinger, Jensen-Shannon) registers as another functional that consumes the same family `A` /
natural maps — orthogonal to the family axis.

## 4. Consumer conversion (the divergence boundary only)

Parameter objects are constructed from `(mu, sigma)` tensors at the divergence call boundary; the
gauge / transport internals stay tensor-based and Gaussian.

- `free_energy.py` — `pairwise_energy(q: BeliefParams, transported_key: BeliefParams, *, alpha,
  ..., irrep_dims)` builds the query via `broadcast_over_keys()`, slices each irrep block with
  `block()`, and calls `renyi`. `self_divergence`, `self_divergence_per_coord`,
  `self_divergence_for_alpha` take parameter objects. `free_energy()` (the scalar assembler over
  already-reduced `self_div` / `energy` tensors) is unchanged.
- `inference/e_step.py`, `gradients/oracle.py` — build parameter objects from the belief and the
  transported key moments at the energy / self-divergence calls.
- `model/prior_bank.py` — `decode` and `reference_decode` build parameter objects for the KL-to-prior
  readout.
- `model/model.py` — `diagnostics()` builds parameter objects for its energy / self-divergence recompute.
- `gradients/kernels.py` — the Gaussian-diagonal hand kernel stays tensor-based (its analytic
  `(grad_mu, grad_sigma)` math is unchanged and does not call `renyi`); the has-this-kernel guard
  (`family == "gaussian_diagonal"`, a kernel-availability check) stays. This file is essentially
  untouched.

Data flow is conceptually identical: transport produces moment tensors, which are wrapped as
`DiagonalGaussian` / `FullGaussian` at the energy boundary, reduced via `renyi`, and fed to the
softmax and `F`. The Gaussian numerics are byte-identical because the closed forms are reused.

## 5. Validation

A test-only `ToyExponential(BeliefParams)` defines only `natural()` and `log_partition_at` (no
moment closed form) for a simple one- or two-parameter exponential family. Tests:

- The generic path: `renyi(toy_q, toy_p, alpha)` and `kl(toy_q, toy_p)` compute via `A(theta)`
  end-to-end, against an independently hand-derived expected value for that family.
- Generic-equals-moment pin: for a Gaussian, the generic Renyi-from-`A` equals the moment closed
  form to tolerance (the correctness proof for the generic path), at several `alpha`.
- Interface equivalence: `block`/`broadcast_over_keys` reproduce the prior tensor slicing /
  unsqueeze exactly (`pairwise_energy` per-head and full-covariance results unchanged).
- Regression gate: the full existing suite (259 tests) stays green with Gaussian numerics
  byte-identical; the golden tolerances are the equivalence gate.

TDD throughout: each new behavior gets a failing test first.

## 6. Risks

The conversion spans roughly six files but is mechanical at each call site. The numerically
delicate code (the hand gradient kernel, the Cholesky full-covariance path) is reused verbatim or
untouched, and the 259-test suite plus the VFE_2.0 golden tolerances pin equivalence. The one
genuinely new numerical path is the generic Renyi-from-`A`, used only by the toy family and the
pin test; it is verified symbolically before use. The `alpha > 1` domain boundary is unchanged
(the diagonal closed form clamps; the full Cholesky may NaN — the existing behavior, with the
hardening deferred as its own roadmap item).

## 7. Out of scope (explicit)

Mixture families; a concrete categorical observation-likelihood family; the gauge action on
non-Gaussian families (Phase 5); the `BeliefState` container redesign (addendum M3); `register_transport`
/ Regime-II; and the alpha>1 Cholesky hardening. Each is a separate roadmap item.
