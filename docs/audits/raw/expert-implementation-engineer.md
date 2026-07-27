# Expert — audit-implementation-engineer (config wiring, pure-path existence, propagation)

Returned 2026-07-27 ~09:40 CDT (retry after the 09:00 rate-limit kill). Verbatim; NOT yet verified.
Bands: **no critical**, 2 high, 3 medium, 2 low.

---

### 1. `family='gaussian_frame_diagonal'` with `e_phi_lr > 0` hard-crashes the forward; `__post_init__` accepts it
**Location:** vfe3/inference/e_step.py:1192 (validation gap at vfe3/config.py:2417)
**Severity:** high
**Evidence:** The phi sub-step differentiates the alignment loss w.r.t. a fresh leaf with no
`allow_unused`:
```python
1165        with torch.enable_grad():
1166            phi_g = belief.phi.detach().clone().requires_grad_(True)
1167            L = phi_alignment_loss(
...
1192            grad_phi = torch.autograd.grad(L, phi_g)[0]
```
Under `gaussian_frame_diagonal` the relative frame cancels out of the pair energy, so `phi_g` never
enters `L` and the call raises. Tiny-model probe (K=4, N=5, V=7, `use_prior_bank=True`,
`decode_mode='family'`, `e_step_update='gradient'`):
```
FAIL frame_ephi RuntimeError One of the differentiated Tensors appears to not have been used in the graph.
     vfe3 frames: ['\vfe3\model\stack.py:133 vfe_stack', '\vfe3\model\block.py:132 vfe_block',
                   '\vfe3\inference\e_step.py:1503 e_step', '\vfe3\inference\e_step.py:1192 e_step_iteration']
OK   frame_ephi_massphi 1.946104884147644      # mass_phi=0.1 only "fixes" it by descending the penalty alone
OK   diag_ephi(control) 1.9464759826660126
```
Fails identically with `oracle_unroll_grad` True or False and with `pos_phi='none'`.
`grep -n "gaussian_frame_diagonal" vfe3/config.py` returns nothing — the string does not appear in
the validator, while the exactly analogous inert-frame case IS guarded:
```python
2417        if self.transport_mode == "regime_ii_link" and self.e_phi_lr > 0.0:
2418            raise ValueError(
2419                f"transport_mode='regime_ii_link' is edge-owned and independent of the vertex frame "
2420                f"phi; set e_phi_lr=0.0 (got {self.e_phi_lr}), ...
```
**Fix:** Add the mirror guard rejecting `e_phi_lr > 0` for families whose coupling energy is
frame-independent (register the property on `BeliefParams` so it is a registry query, not a name
literal).

### 2. The emission Bohning terms are built once at the layer-0 prior and reused at every layer, while the fusion pairs them with each block's drifted `mu_p`
**Location:** vfe3/model/model.py:1126 -> vfe3/model/stack.py:142 -> vfe3/gradients/kernels.py:701
**Severity:** high
**Evidence:** *(INDEPENDENT CONFIRMATION of the variational expert's finding #2, different probe —
runtime object-identity spy rather than a gradient residual.)* `forward_beliefs` builds `(d, g)`
once, outside `vfe_stack` (model.py:1126-1129); `vfe_stack` forwards that same object inside its
per-layer loop and then advances the prior (stack.py:142, :151
`mu_p = (1.0 - rho) * mu_p + rho * belief.mu`); the kernel assumes the expansion point IS the `mu_p`
it receives (kernels.py:699-701). `emission_pull = g = W^T(e_x - softmax(W mu_p^{(0)}))` is computed
at the layer-0 prior (emission.py:110 `expansion = mu_p.detach()`), so from layer 2 on the term is
`d*mu_p^{(L)} + g^{(0)}` — two different expansion points. Runtime probe (`n_layers=3`,
`emission_mode='separate'`, `emission_weight=1.0`, `prior_handoff_rho=1.0`,
`e_step_update='mm_exact'`), spying on `vfe_block`:
```
layer 0: emission_id=1262427856768 mean|mu_p|=0.45455 mean|g|=0.253378301858902
layer 1: emission_id=1262427856768 mean|mu_p|=0.40544 mean|g|=0.253378301858902
layer 2: emission_id=1262427856768 mean|mu_p|=0.40909 mean|g|=0.253378301858902
emission object identical across layers: True
```
`config.py:1640-1670` validates `emission_mode` against `e_step_update`, `use_prior_bank` and
`emission_weight`, but places no constraint on `n_layers`/`prior_handoff_rho`.
**Fix:** Rebuild the Bohning `(d, g)` pair per layer from that layer's `mu_p` inside `vfe_stack`,
or pass the expansion point alongside `(d, g)` and use it in place of `mu_p` at kernels.py:701.

### 3. Under the live `e_step_update='mm_exact'`, every E-step step-size / preconditioner / trust knob is unread on BOTH channels, and the inert-config detector is silent
**Location:** vfe3/inference/e_step.py:1001-1058 (mm branch) vs :1059-1150 (gradient branch); detector at vfe3/config.py:2956-3032
**Severity:** medium
**Evidence:** `e_q_mu_lr` / `e_q_sigma_lr` / `e_step_mu_precond` / `e_mu_q_trust` /
`mu_trust_mode` / `e_sigma_q_trust` are referenced ONLY in the `else` (gradient) branch —
`delta_mu = e_q_mu_lr * mu_grad` at 1125, `-e_q_sigma_lr * nat_sigma` at 1148,
`apply_mu_trust_region(...)` at 1135. The `mm_exact` branch (1001-1058) uses `mm_damping` alone.
`_refine_s` forwards `e_step_update=cfg.e_step_update` (model.py:924), so `e_s_mu_lr` /
`e_s_sigma_lr` die with them. Measured on tiny models (`s_e_step=True`,
`prior_source='model_channel'`), A = `(e_q_mu_lr=0.9, e_q_sigma_lr=0.001, e_s_mu_lr=0.85,
e_s_sigma_lr=0.1, precond='fisher', e_sigma_q_trust=10.0)` vs B = `(0.05, 0.9, 0.01, 0.9, 'raw',
0.1)`:
```
e_step_update=mm_exact: loss(A)=2.4700307846 loss(B)=2.4700307846 identical=True
   inert warnings for A: []
e_step_update=gradient: loss(A)=2.5144736767 loss(B)=2.6721525192 identical=False
```
**The live `train_vfe3.py` sets `e_step_update="mm_exact"` with `e_q_mu_lr=0.9,
e_q_sigma_lr=0.001, e_s_mu_lr=0.85, e_s_sigma_lr=0.1, e_step_mu_precond="fisher"`.** The detector
already reports the exact mirror condition — config.py:2957:
`if canonical_e_step_update != "mm_exact" and _changed("mm_damping")` — but has no rule for the
reverse direction.
**Fix:** Add the reverse rule to the `_inert` block: under `e_step_update='mm_exact'`, report
changed `e_q_mu_lr`/`e_q_sigma_lr`/`e_s_mu_lr`/`e_s_sigma_lr`/`e_step_mu_precond`/`e_mu_q_trust`/
`e_sigma_q_trust` as unread.

### 4. `exp_fp64_mode` / `exp_fp64_norm_threshold` never reach the vertex-factor matrix exp on any non-flat transport
**Location:** vfe3/geometry/transport.py:903-913 and :1072-1090
**Severity:** medium
**Evidence:** The registry forwards both keys to every builder (transport.py:480-481 lists them in
`_TRANSPORT_BUILDER_RESERVED_STATE_KEYS`, and `e_step._transport` passes them at 155 and 174), but
the non-flat builders declare them only through `**kwargs` and never pass them on:
```python
903    if connection_W is None or cocycle_relaxation == 0.0:
904        return compute_transport_operators(
905            phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor,
906            validity_max_norm=validity_max_norm)
...
911    fac = build_factored_transport(
912        phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor,
913        validity_max_norm=validity_max_norm)
```
`_build_regime_ii_covariant` repeats the pattern at 1076 and 1088; the flat builder DOES forward
them (transport.py:809-810). `stable_matrix_exp_pair` therefore always uses the `'dim'` rule on
`regime_ii` / `regime_ii_covariant` / the link modes, so setting `exp_fp64_mode='norm'` changes
nothing there. Additionally `e_step_iteration`'s belief-dependent rebuild closure omits them:
```python
938        def _omega_builder(mu_q, sigma_q, mu_k, sigma_k):
939            return build_belief_transport(
940                belief.phi, group, transport_mode=transport_mode,
...   # no exp_fp64_mode / exp_fp64_norm_threshold / transport_mean_per_head
```
**Fix:** Thread `exp_fp64_mode`/`exp_fp64_norm_threshold` from the non-flat builders into their
`compute_transport_operators` / `build_factored_transport` calls, and add them to `_omega_builder`.

### 5. The run's purity ledger records neither the emission factor nor the norm seam, so `on_pure_path: True` is published for a learned non-equivariant affine on the belief mean
**Location:** vfe3/run_artifacts.py:4116-4134 (`pure_flags`), :4138-4146 (`gauge_flags`), :4152-4211 (`config_toggles`)
**Severity:** medium
**Evidence:** `grep -n "emission\|norm_type\|layernorm" vfe3/run_artifacts.py` returns **no matches**
in the entire file. `pure_flags` covers `no_head_mixer`, `unweighted_attention`,
`full_sigma_update`, `no_twohop_coupling`, `gradient_e_step_update`, and `config_toggles` records
`detached_query_adaptive_tau`, `fixed_covariance_surrogate`, `state_dependent_alpha_majorizer` — but
no key for `emission_mode`, `emission_weight`, `norm_type_block`, `norm_type_final`, or
`layernorm_affine`. Direct call of `_pure_path_report`:
```
layernorm_affine: on_pure_path=True on_gauge_pure_path=False
   emission keys present: [] | norm keys: [] | unigram keys: []
```
with `norm_type_block='layernorm'`, `norm_type_final='layernorm'`, `layernorm_affine=True` — a
learned per-feature `gamma`/`beta` applied to the belief mean inside the stack.
`emission_separate_w0.5` reports `on_pure_path=False`, but only incidentally, via
`gradient_e_step_update` flipping on the `mm_exact` that emission requires; the manifest still
carries no record of which non-pure component ran.
**Fix:** Add `emission_off` and `gauge_pure_norm_seam` to the flag sets and echo
`emission_mode`/`emission_weight`/`norm_type_block`/`norm_type_final`/`layernorm_affine` in
`config_toggles`.

### 6. `across_layer_belief_trace` replays the stack at the config-init temperature, ignoring `learnable_kappa_beta` and `query_adaptive_tau`
**Location:** vfe3/viz/extract.py:902-908
**Severity:** low
**Evidence:**
```python
902            belief = vfe_block(
903                belief, mu_p, sigma_p, model.group, cfg, log_prior=log_prior,
904                block_norm=model.block_norm, head_mixer=model.head_mixer,
905                cg_coupling=model.cg_coupling,                       # replay the trained model
906                lambda_beta=cfg.lambda_beta, transport_state=model.transport_state,
907                rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
908                gauge_parameterization=cfg.gauge_parameterization, rope_insertion=cfg.rope_insertion)
```
No `tau=` and no `kappa_beta_override=`, so `vfe_block` falls back to
`tau = attention_tau(_as_coeff(cfg.kappa_beta, ...), group.irrep_dims)` (block.py:124-125) — the
init value, not `exp(log_kappa_beta)`. Its two sibling replays in the same file DO pass it
(extract.py:531 and :668 both carry
`kappa_beta_override=model.effective_kappa_beta(device),   # learned tau, not init (audit M1)`, and
:1058 likewise). The layer-depth diagnostics this feeds (`d_ai`, `effective_rank`,
`rank_one_residual`, extract.py:933-934) are computed off-objective under `learnable_kappa_beta=True`.
**Fix:** Pass `tau=model._beta_tau(belief.sigma, belief.mu,
attention_tau(model.effective_kappa_beta(device), model.group.irrep_dims))`, matching model.py:3303
and :3408.

### 7. `randomize_e_steps` samples an independent depth for the model channel, so `s_e_step_n_iter=None` ("follow `n_e_steps`") does not hold
**Location:** vfe3/model/model.py:926-929 -> vfe3/inference/e_step.py:1433-1436
**Severity:** low
**Evidence:** `_refine_s` forwards the randomization knobs to its own `e_step` call (model.py:926-929),
and `e_step` draws its depth per call from the global RNG
(`n_total = int(torch.randint(e_steps_min, e_steps_max + 1, (1,)).item())`, e_step.py:1436).
`forward_beliefs` calls `_refine_s` (model.py:1082) then `vfe_stack` (model.py:1127), so a training
forward under `s_e_step=True` performs TWO independent draws: the s-channel and belief-channel
depths differ within the same forward, and the s call consumes extra global RNG that an
`s_e_step=False` run does not. `n_iter=(cfg.n_e_steps if cfg.s_e_step_n_iter is None else
cfg.s_e_step_n_iter)` at model.py:870 is overridden by the sampled `n_total`.
**Fix:** Sample the depth once per forward in `forward_beliefs` and pass it as an explicit `n_iter`
to both `_refine_s` and `vfe_stack`, leaving `randomize_e_steps` off at the inner `e_step` calls.
