# Audit open obligations — closure record, 2026-07-25

Closes the open obligations carried by `audit-2026-07-24.md`, `audit-2026-07-25-second-deeper.md`
(PR #178), and `merged-punch-list-2026-07-25.md`. Every closure below is backed by a measurement, and
the primary evidence is revision-bound by the run artifact's own content hash rather than inferred.

## The evidence run

`vfe3_runs/20260725-125058_wikitext-103_K20_block_glk_linear_mix_s6`

| Field | Value |
|---|---|
| device | **cuda** (RTX 5090) |
| `amp_dtype` | **`'bf16'`** |
| `code_identity_sha256` | `59b2e6c5113812aa7e9df6bb046b7d8f793d2018cce3c5d3f063e1ad06b31c2d` |
| `max_steps` / evals logged | 15000 / 25 |
| `n_params` | 13,142,722 |
| `phi_mstep_max_matrix_norm` | **5** |
| `pos_phi_compose` | `'group_product'` (exact composition) |
| `phi_retract_mode` | `'bch'` |
| `e_step_update` | `'mm_exact'` |
| `lambda_alpha_mode` | `'state_dependent_per_coord'` |
| `s_e_step` | `True` |
| `mstep_self_coupling_weight` | 0.1 |
| seed | 6 |

**Revision binding.** `_package_code_identity()` (`vfe3/run_artifacts.py:121`) hashes `vfe3/**/*.py`
plus the root drivers, excluding generated files, tests, docs and git metadata. Recomputing it against
the working tree reproduces the bundle's stored digest **exactly**
(`59b2e6c5113812aa7e9df6bb046b7d8f793d2018cce3c5d3f063e1ad06b31c2d`), while the same function at
`HEAD = f7da10b` alone yields `cd07a379d48ffd9862ead1b3af857a25fb6c0c0c1211407cc29694998bb86e47`
because the tree additionally carries uncommitted configuration edits. The run therefore executed
`f7da10b` — every remediation through PR #182, including F1, F2 and F3 — plus those config values.
This matters: the closures below would be invalid if the run predated F2, and the hash rules that out
without relying on timestamps.

## Obligation 1 — no bf16 training run had ever been logged. CLOSED.

Both adversarial agents in the F2 duel named this as the one thing they could not resolve: every
archived run used `amp_dtype=None`, so the bf16-trained `||phi||` equilibrium and the in-force
conditioning were unmeasured, leaving F2's severity genuinely undecided (recorded INCONCLUSIVE).

Measured over the 25 logged evaluations of this run, against the archived fp32 runs the audit used:

| Metric | Archived fp32 runs (unbounded `phi`) | This bf16 + CUDA run (bounded at 5) |
|---|---|---|
| `phi_matrix_norm_median` | 7.0 – 7.8 | 2.17 → 4.92 |
| `phi_matrix_norm_max` | 15.8 – 18.1 | **5.000** at every eval |
| `vertex_cond_median` | 35 – 89 | **32.3** (max 38.0) |
| `vertex_cond_p99` | 500 – 3800 | 113.7 (max 172.4) |
| `vertex_cond_max` | 2088 / 2760 / **4050** | **289.6** |

`cond(U)` peaked at 290, an order of magnitude below the ~3e3 at which the audit measured the bf16
group-inverse residual reaching O(1). The configuration never entered the failure regime.

**The reason is the chart bound, and the run shows it working.** `phi_matrix_norm_max` is pinned at
exactly 5.000 across all 25 evals — the projection is active and binding, not incidental.
`phi_chart_projected_fraction` runs 0 → 2.4% of 50,380 rows, `phi_chart_preproject_max` 4.95 – 5.06
clipped back to 5, `phi_chart_projection_scale_min` 0.988, at 0.36 – 0.58 ms per application. Without
that bound the archived trajectories reached `vertex_cond_max` 4050, so this closure is contingent on
`phi_mstep_max_matrix_norm` remaining set — see the standing condition below.

Two collateral confirmations from the same run:

- `phi_exp_clamp_frac = 0` and `pos_phi_exp_clamp_frac = 0` at every eval, so the matrix-exponential
  Frobenius clamp never fired. The surrogate-operator path F2c makes non-silent was never entered, and
  the ~98%-relative-error regime measured at `||M||_F = 30` was never approached.
- `guard_energy_klmax_frac = guard_selfdiv_klmax_frac = 0` at every eval, so no divergence saturated
  `kl_max`. The F3 failure mode — a cancellation-induced negative variance floored to `eps`, inverting
  a key's precision weight and pushing `E_ij` to `kl_max` — had no opportunity to trigger.

**Scope limit, stated explicitly.** `cocycle_residual` (1.4e-05) and `holonomy_deviation` (3.8e-05)
are NOT evidence for F2. Diagnostics rebuild the transport outside autocast, which was precisely the
blind spot F2 identified: those columns read fp32-clean whether or not the forward was bf16. They are
reported here because, since F2 stores factors at fp32, they now describe the same operator the
forward used — but they cannot themselves discriminate. The discriminating evidence is
`vertex_cond_max` together with the revision binding above.

## Obligation 2 — no CUDA path had been exercised. CLOSED.

Both audits ran CPU-only: the 2026-07-24 environment is `torch 2.11.0+cpu`
(`torch.cuda.is_available() == False`), and PR #178's container had no driver. Every measurement in
both reports was therefore CPU.

This run's `device` is `cuda`, for 15000 steps under bf16 autocast. The conditioning, flatness, clamp
and saturation figures above are consequently CUDA-side measurements of the production path, not CPU
proxies. Note the asymmetry that remains: the *test suite* is still CPU-only in this environment, so
CUDA coverage is empirical from training runs rather than from pytest.

## Obligation 3 — PR #178's two torch-version-sensitive goldens. CLOSED.

PR #178 reported 8 failures in a container that began with no Python packages installed and therefore
pulled `torch 2.13.0+cu130` rather than the pinned build; it recorded re-running the two
version-sensitive goldens on the pinned environment as an obligation.

On the pinned build: `pytest tests/ -k golden` → **2 passed**, 4216 deselected. The failures were
environmental, as PR #178 suspected.

## Obligation 4 — the `merge_legacy_transport_state` conflict. CLOSED.

The merged punch list recorded a genuine disagreement: the 2026-07-24 audit called
`transport.py:485` live because its call sites pass the named kwargs, while PR #178's AST scan
concluded the legacy kwargs were production-dead and the duplicate-provision guard unreachable.

Settled by direct inspection, and the answer is a third thing neither stated exactly. The seven
internal sites (`e_step.py:118,346,522,720,873`, `block.py:116`, `stack.py:72`) do pass
`connection_W=connection_W` — so the first audit was right that the kwargs are passed — but they
forward a *function parameter*, and the only production caller is `model.py`, which at eight sites
passes `transport_state=self.transport_state` and never supplies `connection_W`. The legacy kwargs are
therefore always `None` in production, the duplicate-provision `ValueError` is unreachable, and PR
#178's conclusion holds for the reason it gave. Per `CLAUDE.md` this is pre-existing dead code to
report, not delete.

## Standing conditions on obligation 1

These are not open obligations but they bound the closure, and both would invalidate it:

1. **`phi_mstep_max_matrix_norm` must stay set.** With it `None` under `m_phi_update_mode='adamw'`
   and `e_phi_lr=0.0`, nothing bounds `||phi||`: the chart projection is gated on that field
   (`train.py:745-749`), the `adamw` phi policy registers no manifold hook, and `retract_phi`'s own
   cap is reachable only from the E-step phi substep. `TRANSPORT_CLAMP_MAX_NORM` bounds the
   exponentiated operator, not the parameter. The archived unbounded runs reached
   `vertex_cond_max` 4050, where the bf16 inverse residual is O(1). Audit F4 documents this on the
   field; it is deliberately not a construction warning, because the condition is true for a bare
   `VFE3Config()` and this project keeps a clean config warning-free.
2. **The closure is per-configuration, not universal.** A larger `embed_dim`, a different group, or a
   higher `m_phi_lr` changes the `||phi||` trajectory and hence the conditioning. Re-check
   `vertex_cond_max` on any materially different configuration before assuming the same headroom.

## Not closed

- **CUDA coverage in the test suite.** Exercised empirically by training, not by pytest, in this
  environment. `VFE3_TEST_DEVICE=cuda` on a CUDA-enabled build remains unrun here.
- **Punch-list F18 tail.** Two code items: the kernel/oracle divergence below the `eps` floor
  (`kernels.py:152`, the kernel retains the sigma derivative where the oracle zeroes it) and the
  gamma-prior fold's `log(1e-12)` floor (`model.py:2367-2370`, inert at the current prior sharpness).
- **Punch-list F22.** Report-only by `CLAUDE.md`'s instruction to mention pre-existing dead code
  rather than delete it; the inventory is in the merged punch list.

## Incidental result worth recording

With `evaluate_zero_e_steps_counterfactual` enabled, this run reports
`CE@0 − CE@1 = 1.6404` nats on held-out test (test CE@1 = 4.9318, PPL 138.64):

```
CE@0 = 6.5722  ->  PPL 714.94
CE@1 = 4.9318  ->  PPL 138.63      ratio 5.16x, 2.367 bits/token, 25.0% of CE@0
```

At `n_e_steps = 0` the beliefs remain at their encoded per-token prior and never couple across
positions, so no contextual information reaches the linear decode. `CE@0` is therefore this model's
context-free floor, and the 1.64 nats is not a marginal refinement gain but the entire contextual
contribution of the architecture — the measurement most directly probative of the project's premise
that capacity comes from iterative VFE minimization rather than learned layers. Generalization is
clean: val CE exceeds test by 0.0069 nats (val PPL 139.59 versus test 138.64).

`Converged final E-step F/token = 33.0770` is the E-step's own target-blind functional and is not
comparable to CE: it carries the self-coupling and hyper-prior terms and excludes the data term.
