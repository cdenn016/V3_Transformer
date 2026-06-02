# F-Divergence Beyond Renyi for the Functional Registry — Design Spec (2026-06-01)

## Motivation

The buildout roadmap names the f-divergence functional seam as punch-list item 11 and Tier-B finding "f-divergence functional seam beyond Renyi" (`docs/2026-06-01-buildout-roadmap.md:36,118`): the functional axis is a genuine registry (`register_functional`/`get_functional`, `vfe3/families/base.py:138,146`) but carries exactly one member, with `_VALID_DIVERGENCE_FUNCTIONALS = ("renyi",)` (`vfe3/config.py:12`), so the de-facto interface is the single `renyi(...)` signature. The clean-room spec lists "the divergence (`divergence.py`: KL, Renyi, future divergences)" as the first named swap point (`docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md:44`) and states the divergence-value generalization explicitly: for any exponential family with log-partition `A(theta)`, KL is the Bregman divergence of `A` and Renyi has the generic A-form (spec sec 3 line 34). CLAUDE.md's modularity constraint requires that "we should be able to slot in ... different f-divergences." The roadmap is precise that this item "depends on #1" (the now-shipped family seam) and warns the build hinges on whether adding a second f-divergence forces "fitting the `renyi(...)` signature (alpha-parameterized) or editing the energy call sites that assume an alpha argument" (`docs/2026-06-01-buildout-roadmap.md:36`). This spec resolves exactly that signature question and picks the first member to build.

## What the code actually constrains (judged from executable source)

The functional contract is fixed by three facts read from source, not docstrings:

1. The registry stores bare callables; `get_functional(name)` returns one (`vfe3/families/base.py:135-150`). Only `renyi` is registered, at `base.py:232`.

2. Every consumer invokes the functional through one of two `free_energy` wrappers, and both forward `alpha=...` as a keyword:
   - `pairwise_energy` calls `functional(q_b, key, alpha=alpha, kl_max=kl_max, eps=eps)` (`vfe3/free_energy.py:61,65,71-74`).
   - `self_divergence` calls `get_functional(divergence_family)(q, p, alpha=alpha, kl_max=kl_max, eps=eps)` (`vfe3/free_energy.py:92-94`).

3. The value plumbed into `alpha` is the config scalar `alpha_div` (default 1.0), threaded identically through `inference/e_step.py:106-110,149-150`, `gradients/oracle.py:70-74`, and `gradients/kernels.py:188-190`. The config field is `alpha_div: float = 1.0` (`vfe3/config.py:42`); `divergence_family` is the separate functional-registry key (`vfe3/config.py:41`).

The consequence: a registered functional MUST accept `alpha`, `kl_max`, `eps` as keyword arguments today, or the call sites break. A non-alpha f-divergence (Hellinger, Jensen-Shannon) has no `alpha` order. This is the signature lock-in the roadmap flagged, and it is the one architectural decision this spec turns on.

## Math — closed forms available for the shipped Gaussian families

The shipped `DiagonalGaussian`/`FullGaussian` expose `natural()`, `log_partition_at()`, `entropy()`, `expected_statistic()`, and a pinned `renyi_closed_form()` (`vfe3/families/gaussian.py:41-56,156-172,174-223`). What follows uses only those.

**Squared Hellinger (recommended first member).** For densities `q,p`, `H^2(q,p) = 1 - BC(q,p)` where `BC = integral sqrt(q p) dx` is the Bhattacharyya coefficient. For Gaussians the Bhattacharyya coefficient is `BC = exp(-D_{1/2}(q||p)/2)`, where `D_{1/2}` is the Renyi 1/2-divergence the code already computes. I verified symbolically (sympy, exact `diff = 0`) that for a 1D/diagonal Gaussian

```
D_{1/2}(q||p) = (m_q - m_p)^2 / (2(s_q + s_p)) - 1/2 log s_q - 1/2 log s_p + log(s_q + s_p) - log 2,
```

summed over coordinates, equals `-2 log BC`, and that this matches the shipped Renyi kernel `renyi_closed_form(alpha=0.5)` term-for-term (the same `sigma_blend` form at `alpha=1/2`). Therefore

```
H^2(q || p) = 1 - exp( -D_{1/2}(q || p) / 2 ),   D_{1/2} = q.renyi_closed_form(p, alpha=0.5, ...),
```

is a closed form for BOTH Gaussian families (the multivariate Bhattacharyya identity holds identically; the full kernel's `alpha=1/2` branch is the multivariate `-2 log BC`). `H^2` lies in `[0, 1]`, is symmetric, and is `0` at `q=p`. Building it is a thin wrapper over machinery already pinned by golden tests; no new family-specific Cholesky/blend math is written. This is the decisive reason to build Hellinger first.

**Forward/reverse KL as explicit non-limiting members.** `kl(q,p) = renyi(q,p,alpha=1)` already exists (`vfe3/families/base.py:220-229`); a reverse-KL member is `kl(p,q)` (argument swap). These are trivial registrations that exercise the new non-alpha registry path without any math risk, useful as the first regression rung.

**Jensen-Shannon (NOT closed form — research/approximation).** `JS(q,p) = 1/2 KL(q || m) + 1/2 KL(p || m)`, `m = (q+p)/2`. The mixture `m` is not Gaussian, so `KL(q||m)` has no closed form for Gaussians (the roadmap's mixture caveat, `docs/2026-06-01-buildout-roadmap.md:40`: a mixture has no single natural parameter and its KL does not separate). JS therefore needs either a numerical-integration evaluation or a moment-matched Gaussian-`m` approximation, neither of which is closed-form pinnable. Recommendation: do NOT build JS in this pass; document it as needing the approximate/Monte-Carlo divergence abstraction the roadmap already flags as a separate larger item.

**Chi-squared / Amari-in-another-convention.** Pearson `chi^2(q||p) = integral (q-p)^2/p dx` has a Gaussian closed form but is unbounded and dominated numerically by the existing `safe_kl_clamp`; it offers little the Renyi family at `alpha=2` does not already reach (`D_2` is a monotone transform of `chi^2`). Defer. The Amari-alpha "different convention" is a reparameterization of the same Renyi member already shipped and adds no new capability; treat as documentation, not a new functional.

## Interface / Architecture

The design generalizes the functional signature once, registers squared Hellinger (plus forward/reverse-KL members as regression rungs), and edits no call site.

**Signature generalization (the central decision, recommended form).** Make the functional contract accept a permissive keyword set: every functional takes `(q, p, *, alpha=1.0, kl_max=100.0, eps=1e-6, **kwargs)`. The two wrappers in `free_energy.py` already pass exactly `alpha`, `kl_max`, `eps` by keyword, so an `alpha`-free functional simply ignores `alpha` (it is accepted-and-unused, the same pattern `eps` already follows in the generic Renyi-from-A path, `vfe3/families/base.py:166-168`). This is a strictly additive change: `renyi` keeps its current signature unchanged; new members add `**kwargs` and ignore `alpha`. No call site is touched, satisfying "never edit call sites."

This is preferred over the alternative of dropping/optionalizing `alpha` at the call sites, which would require editing `pairwise_energy`/`self_divergence` and every consumer in `e_step.py`/`oracle.py`/`kernels.py` — the exact edit the constraint forbids.

**Functional-specific parameters** (e.g. a future divergence needing its own order) ride in via `**kwargs`, mirroring how the `alpha_i` forms already take form-specific params; the wrappers forward only `alpha/kl_max/eps`, so any extra parameter is supplied by a config-driven partial at registration time (see below) rather than threaded through the call sites.

**New code (no new files required).** Add to `vfe3/families/base.py` (the home of `renyi`/`kl` and the registry):

```
def squared_hellinger(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    alpha:   float = 1.0,                  # accepted-and-ignored (no order); kept for signature parity
    kl_max:  float = 100.0,                # forwarded to the inner Renyi-1/2 call
    eps:     float = 1e-6,
    **kwargs,
) -> torch.Tensor:                         # (...) squared Hellinger H^2(q||p) in [0,1]
    r"""H^2(q||p) = 1 - BC, BC = exp(-D_{1/2}(q||p)/2); D_{1/2} from the pinned Renyi-1/2 kernel."""
    d_half = renyi(q, p, alpha=0.5, kl_max=kl_max, eps=eps)
    return (1.0 - torch.exp(-0.5 * d_half)).clamp(min=0.0, max=1.0)

register_functional("squared_hellinger")(squared_hellinger)

def reverse_kl(q, p, *, alpha=1.0, kl_max=100.0, eps=1e-6, **kwargs):
    return renyi(p, q, alpha=1.0, kl_max=kl_max, eps=eps)
register_functional("reverse_kl")(reverse_kl)
```

`renyi` itself gains a trailing `**kwargs` for contract uniformity (additive, harmless). Re-export the new names from `vfe3/divergence.py` `__all__` alongside `renyi`/`kl` (`vfe3/divergence.py:17-40`) so callers keep the single import surface.

**Config.** Widen one tuple: `_VALID_DIVERGENCE_FUNCTIONALS = ("renyi", "squared_hellinger", "reverse_kl")` (`vfe3/config.py:12`). No other config field changes; `divergence_family` already selects the member (`vfe3/config.py:41`). Note in the field comment that `alpha_div` is ignored for non-alpha functionals.

**Per-coord interaction (must be guarded).** `self_divergence_per_coord` raises unless `divergence_family == "renyi"` (`vfe3/free_energy.py:121-125`), and `state_dependent_per_coord` alpha routes through it. The existing guard already rejects a non-Renyi functional on the per-coord path with a clear error, so squared Hellinger composes safely with `alpha_mode="constant"` (the default) and correctly refuses the per-coord alpha form. Document this as a known incompatibility rather than building a per-coord Hellinger (it would not match any alpha-form derivation).

## Phased TDD Implementation outline

Buildable-once-decided (no research) for squared Hellinger and the forward/reverse-KL members. JS and chi-squared are research/deferred.

**Phase 0 — signature generalization + regression rung.** Add `**kwargs` to `renyi`; register `reverse_kl` and a forward-`kl` alias as functionals. KEY TEST: a `divergence_family="reverse_kl"` run of `self_divergence` returns exactly `renyi(p,q,alpha=1)` and a full forward/E-step pass executes with no call-site change. ORACLE: the already-pinned `renyi` kernel with arguments swapped (algebraic identity, `atol=0`); proves the registry path accepts and dispatches a non-default member end-to-end.

**Phase 1 — squared Hellinger member.** Register `squared_hellinger`. KEY TEST: for random diagonal and full Gaussian pairs, `squared_hellinger(q,p)` equals `1 - exp(-0.5 * renyi(q,p,alpha=0.5))` to machine precision, AND equals an INDEPENDENT reference. ORACLE (the proof): a numerical-integration Bhattacharyya `BC = integral sqrt(q p) dx` via dense 1D Gauss-Hermite quadrature per coordinate for the diagonal family (product over k), and a fresh sympy-derived `BC = exp(-(m_q-m_p)^2/(4(s_q+s_p))) * prod sqrt(2 sqrt(s_q s_p)/(s_q+s_p))` evaluated in float64 — both independent of the code path under test (the sympy form is verified above with exact `diff=0`). Assert `H^2 in [0,1]`, `H^2(q,q)=0`, and symmetry `H^2(q,p)=H^2(p,q)` to `atol`.

**Phase 2 — energy/E-step wiring smoke test.** KEY TEST: a `VFEModel` forward with `divergence_family="squared_hellinger"`, `alpha_mode="constant"` runs, F is finite, and `pairwise_energy` returns the per-head `(..., H, N, N)` shape unchanged. ORACLE: shape and finiteness assertions plus a confirmation that `alpha_div` perturbation does NOT change the Hellinger energy (proving `alpha` is correctly ignored). Also assert the per-coord guard: `alpha_mode="state_dependent_per_coord"` with `divergence_family="squared_hellinger"` raises the existing `ValueError` (`vfe3/free_energy.py:121-125`).

## Risks

The squared-Hellinger member inherits whatever `safe_kl_clamp` did to `D_{1/2}`: a clamped `D_{1/2}=kl_max` maps to `H^2 = 1 - exp(-kl_max/2) ≈ 1`, which is correct (maximal Hellinger), so the clamp composes benignly; this should be asserted, not assumed. The full-covariance `alpha=0.5` Renyi path uses a bare `torch.linalg.cholesky` on `sigma_blend` (`vfe3/families/gaussian.py:208-210`); at `alpha=1/2` the blend `0.5(sigma_q+sigma_t)` is always SPD for SPD inputs, so Hellinger does not hit the `alpha>1` indefinite-blend hazard the roadmap flags (`docs/2026-06-01-buildout-roadmap.md:94`) — but this safety is specific to `alpha=1/2` and must be stated. The M-step gradient on a non-Renyi functional falls back to the autograd oracle (the hand kernel guards `divergence_family == "renyi"`, `vfe3/gradients/kernels.py:169`); this is automatic and correct (roadmap confirms no call-site edit needed, `docs/2026-06-01-buildout-roadmap.md:147`) but means Hellinger trains through the oracle path, which the surrogate/end-to-end tests should cover. The gauge-invariance admissibility of Hellinger under the GL(K) congruence action is mathematically true (Hellinger is an f-divergence, invariant under common pushforward, the same Theorem `glk_invariance` basis as Renyi) but is not executably checked today (roadmap item 17, the admissibility verifier); building that verifier first would let the new member be validated rather than asserted.

## DECISION NEEDED FROM USER

1. **Signature generalization: optional-alpha-with-`**kwargs` vs dropping alpha at call sites.** RECOMMENDATION: keep the additive `(q, p, *, alpha=1.0, kl_max, eps, **kwargs)` contract; non-alpha functionals accept-and-ignore `alpha`, exactly as the generic Renyi-from-A path already accepts-and-ignores `eps`. This touches zero call sites and is reversible. The alternative (optionalizing `alpha` out of `pairwise_energy`/`self_divergence`) edits every consumer and violates "never edit call sites." Buildable once decided.

2. **First member to build.** RECOMMENDATION: squared Hellinger, because it is a closed form over the already-pinned Renyi-1/2 kernel (`H^2 = 1 - exp(-D_{1/2}/2)`, verified) and so needs no new family-specific math and no new numerical hazard at `alpha=1/2`; ship forward/reverse-KL members alongside it purely as the registry-path regression rung. Buildable once decided.

3. **Jensen-Shannon scope.** RECOMMENDATION: do NOT build now. JS requires `KL(q||mixture)`, and the mixture is non-Gaussian with no closed form (the roadmap's mixture caveat); it needs the approximate/Monte-Carlo divergence abstraction listed as a separate larger item. Mark as research, not buildable in this pass. If the user wants JS regardless, the only honest options are Gauss-Hermite quadrature (diagonal only, slow) or a moment-matched Gaussian-`m` approximation (a documented non-pure path) — needs a research decision on which, and neither is golden-pinnable.

4. **Chi-squared / Amari-other-convention.** RECOMMENDATION: defer chi-squared (redundant with Renyi `alpha=2` up to a monotone transform, and unbounded), and treat the Amari convention as documentation of the existing Renyi member, not a new functional. Decide whether either is wanted as an explicit named member for completeness.

Files that will change once decided: `vfe3/families/base.py` (add `squared_hellinger`/`reverse_kl`, register them, add `**kwargs` to `renyi`), `vfe3/divergence.py` (re-export), `vfe3/config.py:12` (widen `_VALID_DIVERGENCE_FUNCTIONALS`), and a new test module for the Hellinger-vs-quadrature/sympy oracle. No edits to `free_energy.py`, `inference/e_step.py`, `gradients/`, or `geometry/`.