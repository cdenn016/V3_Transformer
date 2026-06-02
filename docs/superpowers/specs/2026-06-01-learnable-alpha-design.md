# Design Spec: Learnable and Fully-Bayesian Self-Coupling Alpha

Status: forward-looking design for overnight review. Item 19 / "learnable & fully-Bayesian alpha" on the build-next punch list (`docs/2026-06-01-buildout-roadmap.md:56,126`). One sub-form (learnable `nn.Parameter`) is **blocked pending your explicit sanction** (it is a third learned-parameter exception). The other two sub-forms (empirical-Bayes `b0`/`c0`; fully-Bayesian posterior alpha) are buildable-once-decided and need NO new learned-parameter exception. Read the DECISION NEEDED section first.

## Motivation

The self-coupling axis is a live registry — `register_alpha`/`get_alpha` with a `per_coord` declaration (`vfe3/alpha_i.py:19,35,42`) — carrying `constant`, `state_dependent`, and `state_dependent_per_coord` (`alpha_i.py:60,72,87`). The latter two are the Gamma-MAP closed form `alpha* = c0/(b0 + D(q||p))`, derived in `GL(K)_attention.tex:949-964` (`eq:state_dependent_alpha`) as the stationary point of the augmented free energy `alpha_i D(q_i||p_i) + R(alpha_i)` with `R(alpha_i) = b0 alpha_i - c0 log alpha_i` (`eq:precision_regularizer`, line 939), the negative log-density of `Gamma(alpha_i; c0+1, b0)` — the conjugate prior for a Gaussian precision. The roadmap names two absent members: a learnable alpha (a raw parameter trained by backprop) and a fully-Bayesian alpha that propagates the Gamma posterior rather than collapsing to its MAP point. The manuscript itself sanctions exactly one learnable variant: "Both hyper-parameters [b0, c0] may be learned via empirical Bayes by introducing two scalar parameters" (`GL(K)_attention.tex:968`, restated `Participatory_it_from_bit.tex:1317`). It does NOT derive a raw learnable alpha, and it does NOT derive a posterior-propagating alpha; both are extensions beyond the published theory. That gap is the design decision.

## The seam as it exists (verified against executable code)

Every alpha consumer obtains alpha through one dispatcher, `self_coupling_alpha(kl, *, mode, **kwargs)` (`alpha_i.py:135-150`), which forwards `**kwargs` verbatim to the registered form. The forms read only the divergence tensor `kl` and scalar/`(K,)`-tensor kwargs (`value`, `b0`, `c0`); none touches module state. The six consumer sites all call it (or its envelope sibling `alpha_gradient_coefficient`) with config scalars:

- `inference/e_step.py:108` (`free_energy_value`) and `:196` (`e_step_iteration` via `free_energy_value`)
- `gradients/oracle.py:72` (autograd-of-F oracle)
- `gradients/kernels.py:193` (analytic filtering kernel, via `alpha_gradient_coefficient`)
- `model/model.py:247` (diagnostics)

The shape the form receives is routed by `free_energy.self_divergence_for_alpha(..., alpha_mode=...)` (`free_energy.py:129-154`), which reads `alpha_is_per_coord(alpha_mode)` and supplies either the per-position summed divergence `(..., N)` or the per-coordinate `(..., N, K)`. Config validates `alpha_mode` against `_VALID_ALPHA_MODES` (`config.py:18,189`) and rejects a per-coordinate form on a non-diagonal family (`config.py:196-201`).

The load-bearing consequence: **the existing seam can carry a learnable `b0`/`c0` with zero call-site edits**, because `b0`/`c0` are already kwargs and `state_dependent_per_coord` already accepts `(K,)`-tensor `b0`/`c0` (`alpha_i.py:92-93`; pinned by `test_alpha_i.py:24-29`). It **cannot** carry a raw learnable per-coordinate alpha through the same mechanism, because the forms are pure functions of `kl` and the parameter would have to enter as a new tensor kwarg threaded through all six call sites. That asymmetry decides the architecture below.

## The math

### Gamma-MAP recap (the shipped `state_dependent` form)

`alpha_i D + R(alpha_i)`, `R = b0 alpha_i - c0 log alpha_i`, is the augmented per-agent free energy. `R` is `-log Gamma(alpha_i; shape = c0+1, rate = b0)` up to a constant (`GL(K)_attention.tex:939-948`). The self-coupling term `alpha_i D` adds `D` to the rate, so the per-agent objective in alpha is the negative log of an UN-normalized Gamma with shape `c0+1` and rate `b0+D`. Its stationary point (`eq:state_dependent_alpha`):

```
alpha_i*  =  c0 / (b0 + D_i),       D_i = D(q_i || p_i).
```

This is the MAP (mode) of `Gamma(shape = c0+1, rate = b0+D_i)`, whose mode is `(shape-1)/rate = c0/(b0+D_i)`. The envelope theorem makes the explicit `d alpha/d theta` term vanish from the reduced-F gradient at `alpha*`, so the kernel coefficient is `alpha*` itself (`alpha_i.py:118-132`; `GL(K)_attention.tex:971-983`).

### (b) Fully-Bayesian alpha — propagate the Gamma posterior

The exact posterior over the precision alpha_i, given the belief, is

```
p(alpha_i | q_i, p_i)  =  Gamma(alpha_i;  shape a_i = c0 + 1,  rate b_i = b0 + D_i).
```

The MAP collapses this to its mode `c0/(b0+D_i)`. The fully-Bayesian form does NOT collapse; it reports a posterior summary. The two natural choices differ from the MAP by exactly one count:

```
posterior MEAN     E[alpha_i]   = a_i / b_i          = (c0 + 1) / (b0 + D_i),
posterior MODE/MAP                                   =  c0      / (b0 + D_i),
posterior VARIANCE Var[alpha_i] = a_i / b_i^2        = (c0 + 1) / (b0 + D_i)^2.
```

So the minimal, closed-form fully-Bayesian variant is the **posterior-mean** alpha, `alpha_bayes_i = (c0+1)/(b0+D_i)`, a one-line change from MAP (numerator `c0` → `c0+1`). It is closed-form in F in the same sense the MAP is: it is a deterministic function of `D_i` and the hyperparameters, so it slots into the existing routing with the same `(alpha, regularizer)` return contract.

The genuinely Bayesian (not just point-summary) treatment marginalizes alpha out of the self-coupling term. The variational free energy contribution from the self-coupling block, integrating the precision against its Gamma posterior, is

```
F_self,i  =  E_{Gamma(a_i, b_i)}[ alpha_i D_i + R(alpha_i) ]
          =  (a_i / b_i) D_i  +  b0 (a_i / b_i)  -  c0 ( psi(a_i) - log b_i ),
```

using `E[alpha] = a/b` and `E[log alpha] = psi(a) - log b` (digamma `psi`). With `a_i = c0+1`, `b_i = b0+D_i` this is closed-form (no sampling), and its gradient w.r.t. the belief flows through `D_i` and `b_i`. Reporting `(c0+1)/(b0+D_i)` as the effective coefficient and the `b0 E[alpha] - c0 E[log alpha]` expression as the regularizer makes the fully-Bayesian path drop into the existing `(alpha, regularizer)` contract exactly as the MAP path does — the regularizer term just carries the `psi` correction. **This is the one substantive math claim to pin with the `sympy`/`pymc` oracle (see TDD), because the manuscript does not derive it.**

Whether the envelope cancellation (no product-rule correction at `alpha*`) still holds for the posterior-mean coefficient is a SEPARATE check: the posterior mean is NOT the stationary point of `alpha D + R(alpha)` (the mode is), so the envelope argument does not transfer unchanged. The honest position is that the posterior-mean alpha is a different estimator, not a stationary point, and its belief-gradient should be taken by autograd-of-F (the oracle path), not by assuming `alpha_gradient_coefficient == alpha` as the MAP form does. This is a real subtlety, flagged in Risks.

### (a) Learnable alpha — a raw trained parameter

A learnable alpha makes alpha a free `nn.Parameter` trained by backprop, not a function of `D_i`. Two granularities:

- per-coordinate: `alpha_raw` shape `(K,)`, `alpha = softplus(alpha_raw)` (or `exp`) to enforce positivity; broadcast over `(..., N, K)`.
- per-block (per irrep/head): `alpha_raw` shape `(H,)`, expanded to `(K,)` by `group.irrep_dims`.

There is no regularizer from the Gamma prior in the pure learnable form (alpha is no longer a precision posterior summary); `R = 0`, matching the `constant` form's contract. Optionally a weight-decay-like `R = b0 alpha - c0 log alpha` can be retained as a prior on the learned alpha (this IS the empirical-Bayes-on-alpha reading), but that is a config choice, not forced.

The narrower, manuscript-sanctioned learnable variant learns `b0`, `c0` (not alpha) by empirical Bayes (`GL(K)_attention.tex:968`). alpha stays the closed form `c0/(b0+D)`; only the two `(K,)` or scalar hyperparameters are `nn.Parameter`s with positivity reparameterization. This is strictly weaker (two scalars, or `2K` per-coordinate) and is the form the published theory authorizes.

## Interface / architecture

Three sub-forms, in increasing order of what they cost. None edits a consumer call site for its core wiring; the seam was built for this.

### Path (b1): fully-Bayesian posterior-mean alpha — NO new exception, buildable now

New `register_alpha` members in `vfe3/alpha_i.py`, pure functions of `kl` exactly like the shipped forms:

```python
@register_alpha("bayesian_posterior")
def alpha_bayesian_posterior(
    kl:   torch.Tensor,             # (..., N) per-position self-divergence D
    *,
    b0:   'float | torch.Tensor' = 1.0,
    c0:   'float | torch.Tensor' = 1.0,
    eps:  float = 1e-12,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Posterior-mean precision under Gamma(c0+1, b0+D): alpha = (c0+1)/(b0+D).
    Regularizer R = b0*E[alpha] - c0*(psi(c0+1) - log(b0+D)), the expectation of
    R(alpha) over the Gamma posterior (digamma psi)."""
    rate  = (b0 + kl).clamp(min=eps)
    alpha = (c0 + 1.0) / rate
    reg   = b0 * alpha - c0 * (torch.digamma(torch.as_tensor(c0 + 1.0)) - torch.log(rate))
    return alpha, reg

@register_alpha("bayesian_posterior_per_coord", per_coord=True)
def alpha_bayesian_posterior_per_coord(kl, *, b0=1.0, c0=1.0, eps=1e-12, **kwargs):
    # identical body; kl is (..., N, K)
    ...
```

Config: add `"bayesian_posterior"`, `"bayesian_posterior_per_coord"` to `_VALID_ALPHA_MODES` (`config.py:18`). No other config field. The existing per-coord-needs-diagonal-family guard (`config.py:196-201`) already covers the `_per_coord` variant because it reads `alpha_is_per_coord`, which the `per_coord=True` registration sets. The existing `b0`/`c0` config fields feed it. Zero consumer edits. The `alpha_gradient_coefficient` envelope sibling must NOT be used for this form (it assumes the MAP stationary point); the oracle/autograd path (`oracle.py`) is the correct gradient and is already the fallback for any non-`constant`/non-MAP form. Confirm `kernels.py:193` (the hand kernel) is only reached for the diagonal-KL filtering path with the MAP/constant forms — the analytic kernel's envelope assumption does not hold for posterior-mean alpha, so this form must route to the oracle. This is the one wiring subtlety: gate the hand kernel's availability so `bayesian_posterior*` falls back to the oracle (the existing `gradient_mode`/family availability guard at `kernels.py` is where this check lives).

### Path (b2): full marginalized self-coupling — NO new exception, buildable but needs the F term

Same `register_alpha` member returning `(alpha = (c0+1)/(b0+D), reg = b0*E[alpha] - c0*E[log alpha])`. The difference from (b1) is purely that the regularizer carries the digamma term, so `F_self = alpha*D + reg` equals the marginalized expectation derived above. (b1) and (b2) are the same code; (b2) is the claim that the returned `reg` makes `alpha*D + reg` the exact posterior expectation of the self block. Pin that identity (TDD below).

### Path (a): learnable `nn.Parameter` alpha — REQUIRES YOUR SANCTION (third exception)

The parameter cannot live in a `register_alpha` form (forms are pure functions called with no module). It lives in the `PriorBank` (`model/prior_bank.py:80`), the existing home for all learned tables (`mu_embed`, `sigma_log_embed`, `phi_embed`, and the blessed `output_proj_weight`), guarded by a config flag exactly like `output_proj_weight` is guarded by `use_prior_bank=False` (`prior_bank.py:90`):

```python
# in PriorBank.__init__, default OFF:
self.alpha_raw: Optional[nn.Parameter] = (
    nn.Parameter(torch.zeros(K if learn_alpha_per_coord else n_heads))
    if learn_alpha else None
)   # alpha = softplus(alpha_raw); zeros -> softplus(0)=log2 ~ 0.69 init (or shift so init==1.0)
```

The seam routing then needs the parameter to reach `self_coupling_alpha`. The cleanest no-call-site-edit option: a `register_alpha("learnable")` form that reads alpha from a kwarg the model already forwards. The model passes `cfg.alpha`/`cfg.b0`/`cfg.c0` down as scalar kwargs today (`e_step.py`, `oracle.py`, `block.py:39`); add ONE kwarg `alpha_param: Optional[torch.Tensor] = None` to the shared knob bag and have the `learnable` form consume it:

```python
@register_alpha("learnable")
def alpha_learnable(kl, *, alpha_param=None, **kwargs):
    if alpha_param is None:
        raise ValueError("alpha_mode='learnable' requires alpha_param (the nn.Parameter)")
    a = torch.nn.functional.softplus(alpha_param).expand_as(kl)  # (...) broadcast (K,) or (H,)->(K,)
    return a, torch.zeros_like(kl)   # no Gamma regularizer in the pure learnable form
```

This adds `alpha_param` to the `**kwargs` bag the dispatcher already forwards (`alpha_i.py:140`), so the form picks it up by registration; the only call-site touch is the model passing `self.prior_bank.alpha_raw` into the knob bag at the four E-step/oracle/diagnostics sites (a one-kwarg addition, not a logic edit). The per-block expansion uses `group.irrep_dims`, mirroring the head mixer's block handling (`model.py:80`).

Config additions (only if sanctioned): `learn_alpha: bool = False`, `learn_alpha_per_coord: bool = True`, and `"learnable"` in `_VALID_ALPHA_MODES`. A `__post_init__` guard rejects `alpha_mode="learnable"` without `learn_alpha=True` and vice versa, and warns under `detach_e_step=True` that the detached E-step severs `alpha_raw` from the loss (mirroring the existing `use_prior_bank=False` + `detach_e_step` footgun warning, `model.py:81-92`).

The empirical-Bayes variant (the manuscript-sanctioned one) is the SAME `nn.Parameter` mechanism but with the parameters being `b0`/`c0`, fed to the EXISTING `state_dependent` form (which already accepts tensor `b0`/`c0`). It still introduces `nn.Parameter`s, so it still needs sanction — but it is the variant the published theory authorizes, and it requires no new alpha form at all (just learnable `b0_raw`/`c0_raw` in the PriorBank forwarded as the existing `b0`/`c0` kwargs).

### Pure path preserved

All three paths default OFF. `alpha_mode="constant"` (`config.py:70`) and the shipped `state_dependent*` forms are untouched; the bit-identical pure path is the default. New forms are additive registrations.

## Phased TDD implementation

Build order: (b1) first (no exception, smallest, validates the math machinery), then the empirical-Bayes `b0`/`c0` (smallest sanctioned learnable), then raw learnable alpha only if sanctioned.

**Phase 1 — fully-Bayesian posterior-mean form (b1/b2).**
- Task: register `bayesian_posterior` and `bayesian_posterior_per_coord`; add to `_VALID_ALPHA_MODES`; gate the analytic kernel so these forms fall back to the oracle.
- KEY test (closed form): `alpha == (c0+1)/(b0+kl)` and `reg == b0*alpha - c0*(digamma(c0+1) - log(b0+kl))` on a fixed `kl` grid, atol 1e-6. ORACLE: the Gamma posterior is `Gamma(c0+1, b0+D)`; its mean `(c0+1)/(b0+D)` and `E[log alpha] = psi(c0+1) - log(b0+D)` are standard, independently re-derivable with `sympy` (digamma of the shape) and cross-checked numerically by `pymc` posterior sampling of `Gamma(c0+1, b0+D)` (sample mean ≈ form's alpha; sample `mean(log alpha)` ≈ the digamma expression). This is what proves the form is the genuine posterior, not a guessed numerator bump.
- KEY test (F identity, b2): `alpha*D + reg` equals `E_{Gamma(c0+1,b0+D)}[alpha*D + b0*alpha - c0*log alpha]` to Monte-Carlo tolerance. ORACLE: `pymc`/`torch` MC average of the integrand over posterior draws.
- Routing test: `self_divergence_for_alpha(..., alpha_mode="bayesian_posterior_per_coord")` returns `(..., N, K)` and its `.sum(-1)` reduces to the per-position form (reuses the existing `test_self_divergence_for_alpha_routes_by_declared_reduction` pattern, `test_free_energy.py:285`).
- Gradient test: finite-difference of `free_energy_value` w.r.t. `(mu, sigma)` matches the oracle autograd under this alpha_mode (the project's standard FD-vs-autograd-of-F check, the convention in `tests/test_gradients_kernels.py`). ORACLE: `torch.autograd.gradcheck`-style central difference on the scalar F.

**Phase 2 — empirical-Bayes learnable b0/c0 (sanctioned-learnable; needs the parameter-exception nod, but is the manuscript path).**
- Task: `learn_b0_c0: bool` flag; `b0_raw`/`c0_raw` `nn.Parameter`s in PriorBank with softplus positivity; forward `softplus(b0_raw)`/`softplus(c0_raw)` as the existing `b0`/`c0` kwargs at the four sites; reuse the shipped `state_dependent*` form unchanged.
- KEY test: with `learn_b0_c0=True`, a single optimizer step changes `b0_raw.grad`/`c0_raw.grad` to nonzero and leaves `alpha_mode` form math identical to `state_dependent` evaluated at the current `b0`/`c0`. ORACLE: the form is the same shipped function; the only new claim is that grads flow to `b0_raw`/`c0_raw`, checked by a `loss.backward()` and `assert b0_raw.grad is not None and not torch.allclose(b0_raw.grad, 0)`. This proves the learnable hyperparameters are in the loss graph (guards the `detach_e_step` footgun).

**Phase 3 — raw learnable alpha (ONLY if sanctioned).**
- Task: `learn_alpha`/`learn_alpha_per_coord` flags; `alpha_raw` `nn.Parameter` in PriorBank; `register_alpha("learnable")` reading `alpha_param` from the knob bag; per-block expansion via `group.irrep_dims`; config guard + detach warning.
- KEY test (init parity): with `alpha_raw` initialized so `softplus(alpha_raw)==1.0`, the model forward is bit-identical to `alpha_mode="constant", alpha=1.0` at step 0. ORACLE: the `constant` form output, `atol=0` (mirrors the head-mixer identity-init parity test convention).
- KEY test (learns): one optimizer step gives `alpha_raw.grad != 0`; per-block expansion maps `(H,)` to the right `(K,)` slices per `irrep_dims`. ORACLE: hand-built block-expansion of a known `(H,)` vector.
- Modularity test: `alpha_mode="learnable"` selectable by config alone, no consumer call-site edit beyond the one `alpha_param` kwarg in the knob bag (reuses `test_new_form_with_novel_kwarg_reachable_without_editing_dispatcher`, `test_alpha_i.py:32`).

## Risks

The posterior-mean alpha is not the stationary point of `alpha D + R(alpha)` (the MODE is), so the envelope cancellation that justifies `alpha_gradient_coefficient == alpha` for the MAP form (`alpha_i.py:118-132`) does NOT transfer. The fully-Bayesian forms must take their belief-gradient by autograd-of-F (the oracle), and the analytic filtering kernel (`kernels.py`) must be gated to fall back to the oracle for these modes, or the M-step gradient is silently wrong. This gating is the single correctness-critical wiring step.

The marginalized self-coupling F (b2) uses a closed form (`b0 E[alpha] - c0 E[log alpha]`) the manuscript does not derive; the `digamma` term is a new theory claim and must be MC-pinned, not assumed. If the user wants strict manuscript fidelity, only the MAP form is published; the posterior-mean is an extension and should be documented as such in the form's docstring (LaTeX of the Gamma posterior).

A raw learnable per-coordinate alpha decouples alpha from the precision-posterior interpretation entirely — it is no longer "the precision of the self-coupling," just a free weight. That is a genuine departure from the variational-free-energy story; it may improve fit but erases the Bayesian reading the `state_dependent` form earns. The empirical-Bayes `b0`/`c0` variant keeps the Bayesian reading and is the conservative choice.

Under `detach_e_step=True`, the E-step is wrapped in `no_grad` (`model.py:143`), so any alpha parameter consumed only inside the E-step receives no gradient. The learnable-alpha and learnable-`b0`/`c0` paths need the same footgun warning the linear decode has (`model.py:81-92`), and Phase-2/3 tests must assert grad flow with `detach_e_step=False`.

## DECISION NEEDED FROM USER

1. **Do you sanction a raw learnable `nn.Parameter` alpha as a third blessed learned-parameter exception (beyond linear decode and head mixer)? (yes / no, one line.)** Recommendation: **no** to a raw learnable alpha; **yes** to the manuscript-sanctioned empirical-Bayes learnable `b0`/`c0` if you want any learning on this axis. Rationale: `GL(K)_attention.tex:968` authorizes learning `b0`/`c0`, not alpha; the empirical-Bayes path keeps the precision-posterior interpretation, is two scalars (or `2K`), and reuses the shipped `state_dependent` form unchanged — strictly less invasive than a free alpha and theory-faithful. A raw learnable alpha is a larger conceptual departure for marginal modeling gain.

2. **Which is the better next step: fully-Bayesian posterior alpha, or learnable alpha?** Recommendation: **fully-Bayesian posterior-mean alpha (Path b1)** first. It needs no new learned-parameter exception, is a one-numerator-bump closed form (`c0` → `c0+1`), slots into the existing seam with zero call-site edits, and is the natural Bayesian completion of the MAP form already shipped. Build it, pin the Gamma-posterior math with the `sympy`/`pymc` oracle, then revisit learnable alpha only if you sanction the exception.

3. **For the fully-Bayesian form: posterior-MEAN point summary (b1), or the full marginalized self-coupling F with the digamma regularizer (b2)?** Recommendation: **b1** (posterior mean as the coefficient, MAP-style regularizer optional) as the shippable default, with **b2** as a documented opt-in once the digamma F-identity test passes. b1 is the minimal, robust change; b2 is the more honest Bayesian object but rests on an unpublished closed form that must be MC-verified first.

4. **Strict manuscript fidelity, or sanctioned extension?** The posterior-propagating alpha is NOT in the manuscripts (only the MAP collapse is). Recommendation: build it as a clearly-documented forward-looking extension (docstring carries the Gamma-posterior LaTeX and the "extends `eq:state_dependent_alpha`" note), keep the MAP `state_dependent` as the canonical/pure path, and add a one-line `verified.md` entry recording that the posterior-mean/digamma forms were math-checked against `sympy`/`pymc`, not against a manuscript equation.

## Buildable-once-decided vs needs research

- Fully-Bayesian posterior-mean alpha (b1) and the marginalized-F form (b2): **buildable once decided** (decisions 2-4). The only new theory is the `digamma` F-identity, which is standard and pin-able with `sympy`/`pymc` — not open research.
- Empirical-Bayes learnable `b0`/`c0`: **buildable once decision 1 grants the parameter-exception nod**; it is manuscript-sanctioned and reuses the shipped form.
- Raw learnable `nn.Parameter` alpha (Path a): **blocked on decision 1**; mechanically buildable (the seam supports it via one knob-bag kwarg), but it is the disfavored option and a genuine theory departure.