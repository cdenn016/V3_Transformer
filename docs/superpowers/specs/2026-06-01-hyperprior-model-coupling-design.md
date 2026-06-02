# lambda_h=0, gamma=0` default-path docstring at free_energy.py:223-232. The `self_divergence` and `pairwise_energy`/`reduced_free_energy` primitives are already channel-agnostic.
5. The E-step iterates the belief; the model channel needs a slow s-update.
6. `lambda_h=0` and `gamma_ij=0` in the simulation regime (manuscript line 1279, 1296) — so this is a buildout target, identity at zero.

I'll write the complete design doc as my final message.

# Design Spec: Hyper-Prior `lambda_h KL(s||r)` + Model-Coupling `gamma KL(s_i||Omega s_j)` Two-Tier Channel

Status: **buildable once decisions below are fixed** (no open research; the math is closed-form Gaussian, the seams already exist). Roadmap punch-list item **#6 (XL)**. Gates the entire Regime-II meta-bundle hierarchy.

## Motivation

The VFE_3.0 free energy as shipped is the single-tier belief functional: `free_energy()` (`vfe3/free_energy.py:209-252`) assembles `alpha*D(q||p) + sum_j beta E + entropy - loglik` and its own docstring (`free_energy.py:223,229-231`) names `lambda_h KL(s||r)` and `gamma KL(s_i||Omega s_j)` as absent extension points. `BeliefState` (`vfe3/belief.py:8-13`) carries only `(mu, sigma, phi)`; `config.py` has no `lambda_h`/`gamma`. The full two-tier canonical free energy is `Participatory_it_from_bit.tex` eq:pointwise_free_energy (lines 1241-1249) and the boxed eq:free_energy_functional_final (1265-1277), with the `h -> s -> p -> q` hierarchy, the model field `s_i` on the model fiber `B_model` and hyper-prior `r_i` (lines 192, 514), restated compactly in `GL(K)_supplementary.tex` eq:free_energy_full_supp (1069-1079). The manuscript sets `lambda_h=0` in the active-inference frozen-slow-subsystem regime and `gamma_ij=0` in the reported simulations (`Participatory_it_from_bit.tex:1279,1296`), so this is a **buildout target, not a fidelity fix**: it is the formal content of meta-cognitive consensus (line 1296) and the precondition for the meta-agent / cross-scale hierarchy of Section sec:meta_agent_rg. Roadmap item #6 (`docs/2026-06-01-buildout-roadmap.md:48,113`) prescribes building it "as a second `BeliefState` that reuses the existing `pairwise_energy`/`reduced_free_energy` machinery."

## Architecture: a second channel, not a fork

The governing design principle, verified against the live seams, is that the model channel reuses every value-layer seam by being **a second instance of the same abstractions**, not a parallel copy. The math is identical to the belief channel with `(q,p,Omega,beta,pi) -> (s,r,Omega_tilde,gamma,pi_s)`; the manuscript states this explicitly ("the model channel has an identical structure with coupling weights `gamma_ij` replacing `beta_ij` and model-fiber transport `Omega_tilde_ij` replacing belief-fiber transport `Omega_ij`", `Participatory_it_from_bit.tex:1235`). Both fibers are Gaussian (`Participatory_it_from_bit.tex:518-521`), so:

- `s_i`, `r_i` are `BeliefParams` of the **same registered families** (`gaussian_diagonal`/`gaussian_full`, `vfe3/families/gaussian.py:18,126`). The divergence functional (`renyi`/`kl`), the family covariance kernels, the `register_functional`/`register_family` registries, and the `GaugeGroup` transport (`compute_transport_operators`) are all consumed unchanged.
- `KL(s_i||r_i)` is `self_divergence(s_params, r_params, ...)` — the existing primitive (`free_energy.py:80-94`), called a second time.
- The `gamma` block `sum_j gamma_ij KL(s_i || Omega_tilde_ij s_j)` is structurally the same softmax-over-KL object as the belief block: build the per-pair energy with `pairwise_energy(s_i, s_t, ...)` (`free_energy.py:33-77`) and reduce with `reduced_free_energy(energy_s, tau=tau, log_prior=log_prior_s)` (`free_energy.py:196-206`). Nothing in those functions is belief-specific; they take `BeliefParams` and a generic energy tensor.

A bolt-on (a separate `model_channel.py` with its own KL and softmax) would fork the family/divergence/group seams and diverge the moment a new family or functional is registered. Building it as a second channel keeps all four seams (`divergence_family`, `family`, `gauge_group`, `attention_prior`) **shared across both tiers** by construction. This couples to roadmap finding **M3** (`docs/2026-06-01-buildout-roadmap.md:140`): `BeliefState`'s positional-3-tuple shape is the one obstruction, and adding the channel is the forcing function to convert it to a dataclass with an optional fourth/fifth field.

### New files

- `vfe3/model_channel.py` (new). Holds the `ModelBank` table (analogue of `PriorBank` for `s`/`r`) and the model-channel free-energy assembler. Kept separate from `free_energy.py` so the belief tier stays untouched and the two-tier total is a thin sum. *Decision A below* may instead place `r_i` inside `PriorBank` as extra tables; if so this file holds only the assembler + the s-update.

### Changed files

- `vfe3/belief.py` — `BeliefState` 3-tuple -> small dataclass with optional `s_mu, s_sigma, s_phi` (or a nested `model: Optional[ModelState]`). Default `None` keeps the single-tier path bit-identical (this is roadmap M3, and a hard precondition).
- `vfe3/free_energy.py` — `free_energy()` gains `*, lambda_h: float = 0.0, model_self_div=None, model_energy=None, gamma_log_prior=None, tau_gamma=None`. When `lambda_h==0` and `model_energy is None` (the default), the function is byte-identical to today (the `+= lambda_h*... + gamma_block` terms are gated and dead at zero). This preserves the pure single-tier path.
- `vfe3/config.py` — new fields (below); validation block extended.
- `vfe3/model/prior_bank.py` — adds the `r_i` tables (Decision A) or a sibling `ModelBank` is built in `model.py`.
- `vfe3/inference/e_step.py` — `e_step_iteration` gains a slow-channel `s`-update behind an `enable_model_channel` flag; `e_step` threads the model state.
- `vfe3/model/{block,stack,model}.py` — thread the model state through, decode unchanged (decode reads the belief `q` only).

### New config fields

```
# two-tier model channel (default OFF -> single-tier pure path, bit-identical)
enable_model_channel:   bool  = False   # master gate; False -> lambda_h=gamma=0, no s-state
lambda_h:               float = 0.0     # hyper-prior weight KL(s_i||r_i)  (manuscript lambda_h)
gamma_coupling:         float = 0.0     # model-coupling scale on the gamma block
gamma_attention_prior:  str   = "causal"   # pi^(s)_ij seam for the model channel (own prior)
kappa_gamma:            float = 1.0     # model-channel temperature tau_gamma = kappa_gamma*sqrt(d_head)
model_transport:        str   = "tied"  # "tied" (Omega_tilde = Omega) | "independent" (own s_phi)
s_lr:                   float = 0.05    # slow-channel s-update step (s evolves slower than q)
n_s_steps:              int   = 1       # model-channel inner iterations per block
```

`tau_gamma` is computed as a `VFE3Config` property mirroring `tau` (`config.py:273-284`): `kappa_gamma * sqrt(d_head)`. Validation: `lambda_h, gamma_coupling, s_lr >= 0`; `kappa_gamma > 0`; `gamma_attention_prior in _VALID_ATTENTION_PRIORS`; `model_transport in ("tied","independent")`; and a guard that `enable_model_channel=False` forces the channel inert (warn, don't silently honor, if `lambda_h>0` while disabled). When `model_transport=="independent"` the belief carries a second gauge frame `s_phi`, so the s-channel transport is `Omega_tilde_ij = exp(s_phi_i) exp(-s_phi_j)` via the same `compute_transport_operators`; when `"tied"`, `Omega_tilde = Omega` (reuse the belief transport, zero extra cost).

### No new registry, two registry reuses

The channel adds **no new registry**: the `gamma` attention prior reuses `attention_prior.py`'s `_PRIORS` (a second `attention_log_prior(cfg.gamma_attention_prior, ...)` call), and `Omega_tilde` reuses the `GaugeGroup` + `_TRANSPORTS` machinery (which roadmap #7 is separately building). This is the payoff of the second-channel design: the only genuinely new code is (i) the `r_i`/`s_i` storage, (ii) the two extra F terms, (iii) the slow s-update. Everything else is a second call into an existing seam.

## Math

Per-token two-tier point free energy (`Participatory_it_from_bit.tex:1241-1249`, model-channel-only `GL(K)_supplementary.tex:1069-1079`):

```
F = sum_i  alpha_i KL(q_i || p_i)                                   (belief self, shipped)
  + lambda_h sum_i KL(s_i || r_i)                                   (NEW: hyper-prior)
  + sum_ij [ beta_ij  KL(q_i || Omega_ij q_j)
             + tau   beta_ij  log(beta_ij /pi_ij) ]                 (belief coupling, shipped)
  + sum_ij [ gamma_ij KL(s_i || Omega_tilde_ij s_j)
             + tau_g gamma_ij log(gamma_ij/pi^s_ij) ]               (NEW: model coupling + meta-entropy)
  - sum_i E_q[log p(o|k_i)]                                         (likelihood, shipped seam)
```

Both `s_i, r_i` are Gaussians on `B_model` (`Participatory_it_from_bit.tex:518`), so every NEW term is an existing Gaussian closed form:

- **Hyper-prior term.** `lambda_h * KL(s_i || r_i)` = `lambda_h * self_divergence(s_params, r_params, alpha=1, divergence_family=...)` summed over `i`. Identical kernel to the belief self term; the manuscript shows it with unit weight `lambda_h` (the belief self uses the state-dependent `alpha_i`, `Participatory_it_from_bit.tex:1279`).

- **Model-coupling block.** The optimal `gamma*_ij` is the same softmax (`Participatory_it_from_bit.tex:1294`): `gamma*_ij = softmax_j(log pi^s_ij - E^s_ij/tau_g)` with `E^s_ij = KL(s_i || Omega_tilde_ij s_j)`. At the optimum the block equals `-tau_g log Z^s_i` (the envelope/reduced form), so it is assembled exactly by `reduced_free_energy(E_s, tau=tau_gamma, log_prior=gamma_log_prior)` — the same envelope identity already proven for `beta` (`free_energy.py:178-206`, manuscript line 1251). The `gamma_coupling` scale multiplies this reduced block (a global weight; at `gamma_coupling=1` it is the canonical functional).

The total assembled `F` is `F_belief + lambda_h*sum_i KL(s_i||r_i) + gamma_coupling*reduced_free_energy(E_s,...)`. At `lambda_h=0, gamma_coupling=0` (defaults) the two added terms vanish identically — the manuscript's frozen-slow-subsystem regime and the bit-identical pure single-tier path.

### The slow s-update (E-step)

The belief E-step descends `F` in `(mu,sigma,phi)` by Fisher natural gradient (`e_step.py:189-232`). The model channel descends the same `F` in `(s_mu,s_sigma[,s_phi])`. Because `s_i` couples to `q_i` only through the shared likelihood and (in the meta-hierarchy) through `p_i(k_i|m_i)`, and the reported regime freezes the slow subsystem, the canonical and simplest realization is a **separate, slower natural-gradient block** reusing `belief_gradients`/`natural_gradient`/`retract_spd_*` on the model-channel terms:

```
grad_s_mu, grad_s_sigma  <- gradient of [ lambda_h KL(s||r) + gamma_coupling reduced_free_energy(E_s) ]
nat_s                    <- natural_gradient(grad_s..., s_sigma)
s_mu, s_sigma            <- retract with step s_lr   (s_lr < e_mu_lr: "slow channel")
s_phi                    <- (independent transport only) phi-style autograd block
```

This mirrors `e_step_iteration` exactly with the channel's own energy. The "slow" character is the smaller `s_lr` and `n_s_steps`, not a different update rule. **First increment freezes `s_phi` (tied transport)** so there is no second autograd phi block to build.

## Phased TDD implementation

The decisive recommendation: **build incrementally, hyper-prior first**, because the hyper-prior term has a closed-form oracle (it is just a second KL) while the gamma block requires the full transport + softmax + envelope path. Scope the first increment to the smallest thing that exercises the second channel end-to-end.

**Phase 0 — `BeliefState` dataclass (roadmap M3).** Convert `belief.py:8-13` to a dataclass with the three required fields plus optional `s_mu=s_sigma=s_phi=None`. Touch every `BeliefState(...)` construction (it is a `NamedTuple` today, so positional unpacking `mu, sigma, phi = belief` must be audited; grep shows constructions in `prior_bank.py`, `e_step.py`, `block.py`, `model.py`). **Key test:** every existing test passes unchanged (the optional fields default `None`); add one test that `BeliefState(mu,sigma,phi).s_mu is None`. **Oracle:** the full existing suite is the oracle — Phase 0 is behavior-preserving refactor, proven by zero test deltas.

**Phase 1 — hyper-prior term in `free_energy()`.** Add `lambda_h`, `model_self_div` args; `F += lambda_h * model_self_div.sum()`. Build `r_i` storage (Decision A). **Key test:** `test_free_energy_hyperprior_equals_two_kls` — construct a tiny belief + model state, set `lambda_h=0.5`, assert `free_energy(..., lambda_h=0.5, model_self_div=sd_s) == free_energy(...) + 0.5*sd_s.sum()` to `atol=0`. **Oracle:** the existing single-tier `free_energy` value plus a hand-summed `0.5*KL(s||r)` computed independently via `families.base.kl(s_params, r_params)`. A second test pins `lambda_h=0 -> bit-identical to the single-tier F` (the pure-path guard).

**Phase 2 — slow s-update (hyper-prior only).** Add the `s`-update to `e_step_iteration` driving `s` toward `r` under `lambda_h` only (gamma still 0). **Key test:** `test_s_update_descends_hyperprior` — with `gamma_coupling=0`, run N s-iterations and assert `KL(s_t||r)` is non-increasing and `s -> r` (the unique minimizer of `lambda_h KL(s||r)` alone is `s=r`). **Oracle:** the analytic fixed point `s*=r` for the isolated hyper-prior term, plus a finite-difference gradient check of the s-update direction against `autograd.grad(lambda_h*KL(s||r), s)` (the project's standard FD-vs-autograd oracle, `gradients/oracle.py` pattern).

**Phase 3 — gamma model-coupling block.** Add `model_energy`, `gamma_log_prior`, `tau_gamma`; assemble `gamma_coupling * reduced_free_energy(model_energy, tau=tau_gamma, log_prior=gamma_log_prior)`. Wire `Omega_tilde` (tied first). **Key test:** `test_gamma_block_matches_belief_envelope` — feed the gamma block the **same** energies/prior the belief block uses and assert `gamma_block == belief_coupling_block` (the two are the identical softmax-over-KL object; this proves the reuse claim numerically). A second test pins the envelope identity for the gamma channel: `sum_j gamma*_ij E^s_ij + tau_g sum_j gamma*_ij log(gamma*/pi^s) == -tau_g log Z^s_i` to machine precision (reuse the existing belief-channel envelope test as the template). **Oracle:** the belief channel's already-pinned `reduced_free_energy` and `attention_weights` (`free_energy.py`), reused with model-channel inputs.

**Phase 4 — model-channel gradient + independent transport (optional).** Add the gamma gradient to the s-update and (if Decision B selects it) the `s_phi` autograd block mirroring the belief phi step (`e_step.py:212-230`). **Key test:** FD-vs-autograd on the full model-channel `F` w.r.t. `(s_mu,s_sigma,s_phi)`. **Oracle:** `autograd.grad` of the assembled two-tier `F`.

**Phase 5 — integration + diagnostics.** Thread through `block`/`stack`/`model`; surface `gamma`, `KL(s||r)`, and a model-channel holonomy in `diagnostics()` (`model.py:179-262`) reusing the belief diagnostics machinery. **Key test:** `test_model_channel_off_is_bit_identical` — a full `VFEModel` forward with `enable_model_channel=False` produces logits/loss byte-identical to the current model on the same seed (the pure-path regression guard). **Oracle:** a saved logits tensor from the pre-change model.

## Risks

- **`BeliefState` refactor blast radius (Phase 0).** `NamedTuple` allows positional unpacking and indexing that a dataclass does not; any `mu, sigma, phi = belief` or `belief[0]` site breaks. Mitigation: grep all constructions/destructurings first; keep field order `(mu, sigma, phi, ...)`; a dataclass with `__iter__` or a `NamedTuple` subclass with optional fields can preserve unpacking if needed. This is the single largest mechanical risk and is why it is Phase 0, gated on zero test deltas.
- **`B_model` dimension (Decision C).** The manuscript says `B_model` may differ in dimension from `B_state` (`Participatory_it_from_bit.tex:514`). If `s_i` lives in a `K_model != K` space, it cannot reuse the belief group/transport directly. The frozen-slow-subsystem regime sidesteps this; the first increment must fix `K_model = K` (model fiber = state fiber) to reuse the seams, and flag genuine `K_model != K` as out of scope.
- **Slow-channel coupling correctness.** In the full hierarchy `p_i(k_i|m_i)` makes the belief prior depend on the model state (`GL(K)_supplementary.tex:1083`), coupling the two channels. The first increment treats `r_i` as a fixed table and `s` as decoupled from `q` (frozen-slow regime, `lambda_h` reading at line 1279) — correct for the reported simulations but **not** the full cross-scale shadow (`Participatory_it_from_bit.tex:1233`). This must be documented as a scoped simplification, not the general theory.
- **Test-suite cost.** XL with ~5 phases; each phase must keep the full suite green (read pass counts from `--junitxml`, not memory, per CLAUDE.md).
- **No oracle for the meta-hierarchy.** The terms have closed-form oracles; the *purpose* (meta-agent emergence) does not, and validating that is research beyond this buildout.

## DECISION NEEDED FROM USER

**A. `r_i` / `s_i` representation — new `PriorBank` tables vs. derived vs. a sibling `ModelBank`.**
Options: (1) add `r_mu_embed/r_sigma_log_embed[/r_phi_embed]` tables to `PriorBank` (per-token model hyper-priors, parallel to the belief prior tables); (2) a separate `ModelBank` `nn.Module` (clean separation, same table structure); (3) derive `r_i` as a cross-scale shadow of a meta-belief (`Participatory_it_from_bit.tex:1233`) — the full theory but no meta-belief object exists yet. **Recommendation: (2) a sibling `ModelBank` reusing the table layout, with `s_i` initialized `s=r` at encode (mirroring `q=p`, `prior_bank.py:143`).** It keeps the belief `PriorBank` untouched (smaller blast radius, the pure path is provably unchanged), makes the model channel a self-contained module, and the same `nn.Parameter` tables satisfy the no-NN rule. Option 3 is the right long-term target but needs the meta-belief object that this item is the precondition for, so it is deferred.

**B. Transport for the model channel — tied (`Omega_tilde = Omega`) vs. independent (own `s_phi`).**
The manuscript writes `Omega_tilde` as a *distinct* model-fiber transport (`Participatory_it_from_bit.tex:1235,1294`). Tied reuses the belief `Omega` (zero extra geometry, no second phi block); independent gives the model channel its own gauge frame `s_phi` (the general theory, doubles the gauge state and adds a second autograd phi step). **Recommendation: ship tied as the first increment and the default (`model_transport="tied"`), register independent as the opt-in.** Tied is exact when the two fibers share a frame (the natural starting point under `K_model=K`) and lets Phases 1-3 land without touching the phi machinery; independent is Phase 4, behind the toggle, so the pure general path still exists.

**C. Scope of the first increment.**
**Recommendation: Phases 0-2 only — `BeliefState` dataclass + the `lambda_h KL(s||r)` hyper-prior term + its slow s-update, with `gamma_coupling=0`, `model_transport="tied"`, `s_phi` frozen, `K_model=K`.** This delivers a genuine, tested second channel end-to-end (a working `s -> r` regularizer descending the two-tier `F`) with closed-form oracles at every step, while deferring the gamma block (Phase 3, needs the full transport+softmax) and the independent transport (Phase 4) to a follow-on. Rationale: the hyper-prior term is the half with an exact analytic fixed point (`s*=r`) and no new transport/softmax surface, so it proves the second-channel architecture (and forces the M3 dataclass refactor) at the lowest risk; the gamma block is then a pure reuse of the already-pinned belief envelope and lands as a small follow-up.

**D. `gamma` attention prior — own seam vs. shared with `beta`.**
The manuscript gives the model channel its own `pi^(s)_ij` (`Participatory_it_from_bit.tex:1245,1294`). **Recommendation: own field `gamma_attention_prior` (default mirrors `attention_prior`) reusing the existing `attention_prior.py` registry** — a second `attention_log_prior(...)` call, no new code, but a distinct config knob so the two channels can diverge (the manuscript's intent) without forking the seam.

**Buildable once A-D are fixed.** No open research; all four NEW terms are existing Gaussian closed forms with FD/analytic oracles, and the only non-mechanical decisions are the four above. The single non-trivial engineering risk is the Phase-0 `BeliefState` refactor, which is behavior-preserving and gated on zero test deltas.