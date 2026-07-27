# Verifier B — base-wave medium/low findings + negative-result spot checks

Returned 2026-07-27 ~09:57 CDT. Independent `general-purpose` verifier, no investigator reasoning
shown to it. Probes run under `C:/anaconda/python.exe` (torch 2.10.0.dev+cu128, CUDA available).

## Verdict table

| # | Source | Finding (short) | Verdict | Reachable? | Evidence line |
|---|---|---|---|---|---|
| 1 | base-1 | Three drivers use raw `subprocess.run`, no timeout / no containment | **CONFIRMED** | Only when those standalone drivers run; not on the `train_vfe3.py` path | `scaling_analysis.py:47`, `multiseed_analysis.py:48`, `compare_vocab_figures.py:36` vs `process_utils.py:176` |
| 2 | base-1 | Sample decoder dispatches on hardcoded `vocab_size` ranges | **CONFIRMED (latent)** | Reachable; benign today (`vocab_size=50257` + wikitext-103 -> gpt2, correct) | `train.py:494-499`; `datasets.py:421-425` |
| 3 | base-1 | 43 functions violate keyword-only ordering | **CONFIRMED, count corrected to 41** | N/A (convention) | `emission.py:93-95`; `run_artifacts.py:1770-1772` |
| 4 | base-1 | `except Exception: pass` drops silhouette diagnostic | **CONFIRMED** | Reachable (`generate_figures=True`; `english_linguistic_diagnostics` defaults True) | `figures.py:2611-2612` |
| 5 | base-1 | `UMAPWorker.close()` unbounded post-kill `proc.wait()` | **CONFIRMED** | Reachable via `report.py:898` `with figs.UMAPWorker()` | `figures.py:368-372`, `:391` |
| 6 | base-2 | Diagonal `ball` trust region NaNs coord, zeros the rest | **CONFIRMED (reproduced)** | **UNREACHABLE under live config** (`e_mu_q_trust=None`) | `numerics.py:143-145`; `config.py:465,468` |
| 7 | base-2 | `reduced_free_energy` lacks the length-1 tau collapse | **CONFIRMED for `(N,N)`; the `(B,N)` half REFUTED** | Unreachable (equal irrep dims + scalar kappa -> float tau) | `free_energy.py:376-382` vs `:38-51` |
| 8 | base-3 | Copy-pasted `i`/`j`/mask boilerplate, 7 attention priors | **CONFIRMED (one nit)** | Reachable | `attention_prior.py:100-104, 124-128, 148-152, 178-184, 207-213, 237-241, 262-267` |
| 9 | base-3 | Orphaned `DiagonalGaussian` import | **CONFIRMED (pyflakes)** | Dead line | `viz/extract.py:1234` |
| 10 | base-3 | Unused local `gram_factor` | **CONFIRMED (pyflakes)** | Dead line | `phi_preconditioner.py:527` |
| 11 | base-3 | Unused local `n_transport_blocks` | **CONFIRMED (pyflakes)** | Dead line | `exact_congruence.py:144` |
| 12 | base-4 | `generate()` reruns full pipeline per token | **CONFIRMED** | Reachable (`generate_samples` defaults True, called each eval) | `model.py:2249-2253`; `train.py:1641-1646` |
| 13 | base-4 | Per-parameter grad-norm host syncs + duplicate norm | **CONFIRMED (gate is :669, not :678)** | Reachable on logged steps | `train.py:669, 681-683, 688, 730` |
| 14 | base-4 | E-step trajectory `.item()` per inner iteration | **CONFIRMED** | **Unreachable from production**; only tests set it | `e_step.py:1421-1422`, `:1272`, `:1518` |
| 15 | base-5 | `renyi_per_coord` capability check is a proxy | **CONFIRMED** | Path is **LIVE**; works today because `DiagonalGaussian` defines the hook | `config.py:1784-1805`; `base.py:703` unguarded vs `:597` guarded |
| 16 | base-5 | `BeliefParams` ABC declares no `__init__` | **CONFIRMED** | Latent | `base.py:48, 318, 401, 411-422` |
| 17 | base-5 | `omega: object` on three family hooks | **CONFIRMED** | Annotation-only | `base.py:324, 346, 368` |
| 18 | base-5 | `fold_rope_into_frame` omits `DirectLinkTransport` | **CONFIRMED** | Annotation-only; branch unreachable live (`pos_rotation='none'`, `transport_mode='flat'`) | `transport.py:345-348, 368-380` |

## Corrections the verifier made to the investigators

- **Finding 3 count is wrong.** Verifier's own AST sweep found **41**, not 43. Also
  `load_checkpoint`'s first Optional-before-defined-scalar is `max_step`
  (`run_artifacts.py:1771`), not `map_location` — the violation is real, the naming was loose.
- **Finding 7 is half wrong.** The `(N,N)` headless case genuinely produces a spurious leading
  axis (measured `torch.Size([1,3])` vs `(3,)`), but the claimed `(B,N)` failure does NOT occur —
  with a `(2,3,3)` energy the `(1,1)` reshape broadcasts correctly to `(2,3)`.
- **Finding 13's cited gate line is off by nine.** The actual gate is `if metrics_out is not None:`
  at `train.py:669`.
- **Finding 8 nit:** `prior_alibi` (`:148-152`) has no `masked_fill` at all, so the "ALiBi trio"
  characterization covers only two of three.
- **Finding 6 reachability is narrower than implied.** Reproduced exactly
  (`[1., inf, 2.] -> [0., nan, 0.]` in `ball`, `-> [1., 5., 2.]` in `box`), but `train_vfe3.py`
  sets `e_mu_q_trust = None` and the mode is consulted only when trust is not None, so it is dead
  under the live config.
- **Finding 12 nuance:** the self-instrumented 2 GiB warning is live code, not a comment, but is
  inert at the live scale (~26 MB). The cost claim stands independently of it.

## NEGATIVE-RESULT SPOT CHECKS — both survive

### base-3: "all ~171 config fields have a real consumption site"
Field count is **exact: 171** annotated fields by AST. Verifier picked **14 obscure fields itself**
and grepped for reads outside `config.py`: `bch_residual_max` (`model.py:738,771`, `train.py:398`),
`omega_reorth_every` (`train.py:404` -> `gauge_optim.py:539`), `pos_phi_project_slk`
(`model.py:735,751,773`, `run_artifacts.py:3801`), `parameter_motion_rel_tol` (`train.py:1944`, via
`getattr` — a real read), `prior_handoff_rho` (`model.py:2765,3251`), `cg_covariance_mode`
(`model.py:349` -> `cg_coupling.py:83,195`), `unigram_kappa` (`model.py:267` ->
`prior_bank.py:448,920`), `phi_mstep_max_matrix_norm` (`train.py:393-394,760`,
`run_artifacts.py:356`), `s_frame_mode` (`model.py:260,449,761,764`), `mstep_self_coupling_weight`
(`model.py:1534,1620`), `eval_stride` (`train_vfe3.py:551`), `untie_decode_bank` (`model.py:270` ->
`prior_bank.py:455,603`), `gamma_prior_weight` (`model.py:2378`), `omega_metropolis_every`
(`train.py:826`).
**Every one has a genuine `cfg.<field>` read on an executable path. No dead field found.**
Verifier's own caveat: 14 of 171 is a sample, so this corroborates rather than proves the universal
claim.

### base-5: "the shared E-step kwargs bag is fully synchronized"
`e_step_shared_kwargs` (`model/block.py:29-70`) returns **32** keys. Verifier checked each against
both signatures by AST: **all 32 are explicit named parameters on `e_step_iteration`
(`e_step.py:818`) AND on `free_energy_value` (`e_step.py:447`), and neither declares a `**kwargs`
sink.** The only sink in the trio is on the `e_step` carrier (`e_step.py:1246`), which forwards to
two sink-free destinations, so a misspelled knob still raises `TypeError`.
**Confirmed exactly as stated.**

### Incidental (unassigned, checked cheaply): the `torch.load` security claim
Holds. Every `torch.load` in `vfe3/` passes `weights_only=True` (`datasets.py:325,391`;
`run_artifacts.py:919,1705,1834`; `figure_worker.py:325,566`; `run_loading.py:47,73`;
`sweep_adapters.py:238`); the single `weights_only=False` at `run_artifacts.py:1847` is reached only
after the safe load raises AND `trust_resume_checkpoint` is True (`:1837-1843`).
