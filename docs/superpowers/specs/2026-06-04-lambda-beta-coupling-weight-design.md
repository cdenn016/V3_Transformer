# lambda_beta: belief-coupling weight (VFE_2.0 `lambda_align` parity)

Date: 2026-06-04

## Motivation

VFE_2.0 exposes `lambda_align`, a scalar multiplier on the belief-coupling
(attention) block of the free energy, threaded into the gradients. V3 has no such
knob: the belief-coupling block is fixed at unit weight relative to the
self-coupling term. This adds `lambda_beta`, the V3 name for the same multiplier,
as a config field (constant path) and as an optional learned scalar (`log_lambda_beta`,
mirroring `log_alpha`). VFE_2.0's separate `lambda_soft`/`lambda_softmax` knob (the
entropy-suppressed surrogate's softmax-coupling term) is deliberately NOT ported.

## What it multiplies

`lambda_beta` (symbol λ_β) is a uniform weight on the entire belief-coupling block
of the per-position free energy:

```
F = Σ_i [ α_i·D(q_i‖p_i) (+ R(α_i))
        + λ_β·( Σ_j β_ij·E_ij  +  τ·Σ_j β_ij·log(β_ij/π_ij) )
        − ℓ_i ],
    E_ij = D(q_i ‖ Ω_ij q_j),   β_ij = softmax_j(log π_ij − E_ij/τ).
```

At `λ_β = 1.0` this is byte-identical to the current canonical/pure F. `λ_β ≠ 1`
reweights the attention/coupling contribution against the `α·D(q‖p)` self-term and
the likelihood. It scales the belief (q) channel only; the model-channel (`gamma`)
block keeps its own `gamma_coupling` weight and is untouched.

## Correctness invariant (load-bearing)

`λ_β` scales BOTH `coupling` and `entropy` by the SAME factor, and β itself stays
λ-free (the softmax keeps `−E/τ`, with no λ inside it). These two facts are jointly
what keeps the analytic kernel in agreement with the autograd oracle:

Scaling the whole block by λ_β does not move its argmin over the simplex, so
β* = softmax(−E/τ) remains stationary for `λ_β·(coupling + entropy)`. The envelope
identity therefore still gives

```
∂/∂θ [ λ_β·(coupling + entropy) ] = λ_β · Σ_j β*_ij ∂E_ij/∂θ   at β*,
```

for θ ∈ {μ, σ, φ}, which is exactly the kernel's `pair` term scaled by λ_β. The
oracle (which differentiates `λ_β·F_red` through β) and the kernel (which computes
`λ_β·Σβ ∂E/∂θ` directly) thus agree. Two failure modes this rules out:

- scaling only `coupling` (entropy left at unit weight) would make β no longer
  stationary for the scaled objective, breaking the envelope cancellation, and the
  kernel would silently disagree with the oracle;
- pushing λ_β inside the softmax (`E → λ_β·E`) would build a temperature knob, not
  `lambda_align`.

The chosen plan does neither: it scales the post-softmax `(β·E).sum()` and the
`entropy` term in `free_energy`, and the `pair_mu`/`pair_sig` terms in the kernel.

## Phase 1 — constant `lambda_beta` (rides cfg like `mass_phi`)

Config: add `lambda_beta: float = 1.0` to `VFE3Config`, with a `>= 0` validation in
`__post_init__`. No model parameter is created on this path.

Insertion sites (the float is threaded from `cfg` through the existing knob bag):

1. `vfe3/free_energy.py::free_energy` — new `lambda_beta: float | torch.Tensor = 1.0`
   parameter; `F = self_total + λ_β·coupling [+ λ_β·entropy] − ℓ`. This single change
   serves the autograd oracle gradient, the scalar-F monitor, and model diagnostics.
2. `vfe3/gradients/kernels.py` — thread `lambda_beta` into `belief_gradients` and the
   registered `_diag_kl_filtering_kernel`; scale `pair_mu` and `pair_sig` by λ_β, leaving
   `self_mu`/`self_sig` (the α·∂D/∂θ self terms) unscaled. Design choice: pass λ_β into
   the kernel signature (kept symmetric with the `alpha_coef` argument it already takes),
   rather than returning `(self, pair)` separately.
3. `vfe3/gradients/oracle.py::belief_gradients_autograd` — accept `lambda_beta` and
   forward it to `free_energy` (the oracle differentiates that scaled F).
4. `vfe3/inference/e_step.py` — `phi_alignment_loss` multiplies the coupling block
   (the `reduced_free_energy` envelope value on the canonical branch, the `(β·E).sum()`
   on the surrogate branch) by λ_β, but NOT the `mass_phi` penalty; `free_energy_value`
   forwards λ_β to `free_energy`; `e_step_iteration` and `e_step` thread it through.
5. `vfe3/model/block.py::vfe_block` — pass `lambda_beta=cfg.lambda_beta` into `e_step`.

Note: λ_β multiplies the φ-step loss, so the effective φ update is `e_phi_lr·λ_β·∇`;
λ_β and `e_phi_lr` interact (this matches VFE_2.0, where `lambda_align` scaled the
whole alignment loss including ∇_φ). A φ-LR sweep at λ_β ≠ 1 must be read with this in
mind.

Diagnostics: the monitored total F reflects λ_β (the "scaled F", as VFE_2.0 documents);
the per-component `belief_coupling` diagnostic stays the raw `Σβ·E` so the unweighted
energy remains observable.

## Phase 2 — learnable `lambda_beta` (mirrors `log_alpha` end-to-end)

Config: add `learnable_lambda_beta: bool = False`. When True, `VFEModel.__init__`
creates `self.log_lambda_beta = nn.Parameter(torch.zeros(()))` (init 0 →
λ_β = exp(0) = 1.0, byte-identical to the constant 1.0 path at step 0), and emits the
same `detach_e_step` footgun warning `log_alpha` uses (the parameter enters the loss
only through the E-step belief updates, which a detached/no_grad E-step severs).

Threading: `log_lambda_beta` flows as a live tensor through
`vfe_stack → vfe_block → e_step → e_step_iteration → belief_gradients`
(and the oracle / `phi_alignment_loss`) exactly as `log_alpha` does, via
`getattr(self, "log_lambda_beta", None)` in `model.py`. Consumers use
`λ_β = exp(log_lambda_beta)` when the tensor is supplied, else `cfg.lambda_beta`.
Because λ_β multiplies the live `pair` terms, the M-step CE backpropagates to
`log_lambda_beta` through the unrolled E-step — on the μ/σ path only, since grad_φ is
detached (`create_graph=False`), exactly the signal path `log_alpha` already has.

Optimizer: add `log_lambda_beta` to a param group in `vfe3/train.py::build_optimizer`
(at `m_phi_lr`, a coupling/gauge-scale group, or `m_mu_lr`; decide during
implementation) so the exact-coverage guard is satisfied and it actually trains.

## Tests (gate)

In `tests/test_gradients_kernels.py` / `test_gradients_oracle.py` (or a new
`test_lambda_beta.py`):

1. Kernel == oracle at `λ_β ∈ {0.5, 2.0}` (the primary correctness gate — stronger
   than FD-vs-autograd here).
2. β is bit-identical across `λ_β` at a fixed belief (catches an accidental
   λ-into-softmax leak that test 1 might not localize).
3. `λ_β = 1.0` is byte-identical to the pre-change path (pure-path preservation).
4. Learnable: `log_lambda_beta.grad is not None` after a backward with
   `detach_e_step=False`, and `is None`/frozen with `detach_e_step=True`; init gives
   λ_β = 1.0.

## Sweeps (`ablation.py`)

- Numeric single-field sweep `lambda_beta`: values `[0.25, 0.5, 1.0, 2.0, 4.0]`
  (symmetric log-spaced around the pure 1.0 baseline).
- Multi-arm `learnable_lambda_beta` sweep: `constant` (`learnable_lambda_beta=False`)
  vs `learnable` (`learnable_lambda_beta=True`).
- Add `lambda_beta` to `BASELINE_CONFIG` (= 1.0) and optionally to `SWEEP_ORDER`.

## Pure-path preservation

The default config (`lambda_beta=1.0`, `learnable_lambda_beta=False`) creates no new
parameter and is numerically identical to the current canonical F path. Both new knobs
are opt-in deviations, satisfying the project's "a theoretically pure path always
exists under appropriate toggles" constraint.
