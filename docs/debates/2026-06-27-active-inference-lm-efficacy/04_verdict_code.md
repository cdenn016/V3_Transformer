# Verdict (code-truth) - active-inference-lm-efficacy

## Verdict

Outcome: BLUE_WINS

The code supports Blue's narrow implementation claim: V3 has explicit belief tensors and a no-grad generation-time decision surface where an opt-in candidate scorer can be added without replacing training. Red is correct that the current code does not already implement an active-inference policy layer, a public belief-rollout API, or a live train-time EFE objective. That limitation does not defeat the claim as stated, because the claim argues for a first opt-in inference-time scorer and treats train-time replacement as premature.

## Decisive code evidence

Decisive evidence: `vfe3/model/model.py:1164-1221` defines `generate()` under `@torch.no_grad()`, calls `self.forward(context)` at `vfe3/model/model.py:1201`, reads the last-position logits at `vfe3/model/model.py:1202`, and then performs greedy, temperature, top-k, top-p, or multinomial token selection at `vfe3/model/model.py:1203-1220`. `vfe3/model/model.py:644-794` shows that `forward(..., targets=None)` encodes beliefs, runs `vfe_stack`, decodes final beliefs, and returns logits rather than a belief-policy object.

Executed check in the copied worktree:

```text
cfg.seed 122
forward_type Tensor
forward_is_tensor True
forward_shape (1, 4, 50257)
generate_type Tensor
generate_shape (1, 5)
generate_prefix_equal True
```

This confirms the public inference path is logits-out and no-grad generation appends tokens after forward. It also confirms the natural insertion point for an opt-in inference-time scorer.

## Reasoning

### My re-traced active config

Active entry point: `train_vfe3.py`. The config dict is declared at `train_vfe3.py:68-313`, `kl_max` is set from width at `train_vfe3.py:321`, and each run builds `VFE3Config(**{**config, "seed": seed})` at `train_vfe3.py:462`. `main()` uses `SEEDS[:NUM_RUNS]` when `SEEDS` is nonempty at `train_vfe3.py:543-555`, so the resolved run seed is `122`, not the literal `config["seed"] = 6`.

There is no `BlockConfig` or `TrainingConfig` in this copied tree. `rg` finds only `VFE3Config.__post_init__` in `vfe3/config.py` and `GaugeGroup.__post_init__` in `vfe3/geometry/groups.py`; the active config surface is the single `VFE3Config` dataclass. The relevant `VFE3Config.__post_init__` checks preserve the active settings: `embed_dim=10`, `n_heads=2`, `max_seq_len=128`, `batch_size=64`, `max_steps=15000`, `n_layers=1`, `n_e_steps=1`, `vocab_size=50257`, `kl_max=80`, `divergence_family="renyi"`, `renyi_order=1.0`, `family="gaussian_diagonal"`, `gauge_transport="on"`, `gauge_group="block_glk"`, `transport_mode="flat"`, `pos_phi="learned"`, `pos_rotation="none"`, `use_prior_bank=False`, `decode_mode="diagonal_chunked"`, `decode_precision_scaled=False`, `lambda_alpha_mode="state_dependent"`, `lambda_h=0.25`, `lambda_gamma=0.75`, `prior_source="model_channel"`, `s_e_step=True`, `learnable_r=True`, `beta_attention_prior="causal_alibi"`, `gamma_attention_prior="causal_alibi"`, `precision_weighted_attention=True`, `include_attention_entropy=True`, `gradient_mode="filtering"`, `e_step_gradient="unroll"`, `oracle_unroll_grad=False`, `m_phi_natural_grad=True`, and `use_head_mixer=True`.

This differs from the Research wiki's current-program summary that mentions a live `embed_dim=70` run. The debate worktree's active entry point resolves to `embed_dim=10`, `n_heads=2`. The line claims under review are structural and remain reachable under the copied active config.

### Reachability verification

| path:line | Cited by | Reachable under active config? | Notes |
|---|---|---|---|
| `train_vfe3.py:68-313` | Evidence pack | Yes | Active config dict. |
| `train_vfe3.py:321` | My trace | Yes | Resolves `kl_max = 8 * embed_dim = 80`. |
| `train_vfe3.py:462` | My trace | Yes | Builds `VFE3Config` for each run. |
| `vfe3/config.py:563-1605` | My trace | Yes | Single config validation/coercion path; no `BlockConfig` or `TrainingConfig` exists. |
| `vfe3/belief.py:22-30` | Both | Yes | `BeliefState` has `mu`, `sigma`, `phi`, optional `s`, `r`. |
| `vfe3/model/model.py:644-661` | Both | Yes | `forward()` starts, encodes token ids, applies positional phi. |
| `vfe3/model/model.py:707-725` | Both | Yes | Active config has `s_e_step=True`; then `vfe_stack(...)` runs. |
| `vfe3/model/model.py:739-794` | Both | Yes | Final belief tensors are decoded; inference branch returns logits when `targets is None`. |
| `vfe3/model/model.py:1164-1221` | Both | Yes | `generate()` is no-grad, calls `forward(context)`, then selects tokens. Executed check confirmed. |
| `vfe3/model/prior_bank.py:223-228` | Blue | Yes | `PriorBank.encode()` is called by `forward()`. |
| `vfe3/model/prior_bank.py:312-328` | Blue | Yes | `decode()` is called by `forward()`; active `use_prior_bank=False` routes to `linear`. |
| `vfe3/model/prior_bank.py:812-833` | My trace / Red relevance | Yes | Active linear decode discards `sigma_q` unless `decode_precision_scaled=True`, which is false. |
| `vfe3/free_energy.py:268-279` | Evidence pack | Conditionally | Defines softmax attention weights; this is library code, not an EFE policy layer. |
| `vfe3/free_energy.py:307-324` | Evidence pack | Yes | `reduced_free_energy()` is used by active E-step/model-channel reductions. |
| `vfe3/free_energy.py:327-342`, `401-402` | Red | Function reachable, observation term inert | `log_likelihood` is optional and only subtracted when supplied. No production caller passes it. |
| `vfe3/inference/e_step.py:279-284` | My trace | Yes for diagnostics/oracle path | Calls `free_energy(...)` without `log_likelihood`. |
| `vfe3/inference/e_step.py:345` | Red | Yes | Active reduced/kernel route returns `lambda_beta * reduced_free_energy(...)`. |
| `vfe3/model/model.py:553-623` | Red | Yes | Active `s_e_step=True` calls `_refine_s`; it explicitly forces `transport_mode="flat"` at `vfe3/model/model.py:618`. |
| `vfe3/model/model.py:1021-1062` | Both | Yes for diagnostics/loss when called | `_gamma_energy()` builds model-channel transport with `transport_mode="flat"` at `vfe3/model/model.py:1047`. |
| `vfe3/model/model.py:898-929` | My trace | Gated off in forward loss | With active `s_e_step=True`, loss-level hyper-prior/gamma terms are not added; the `s` refinement path is the live channel. |
| `vfe3/inference/e_step.py:423-435` | Evidence pack / Red | Not active under `transport_mode="flat"` | This covariance-sensitive builder is only for mu/sigma-dependent non-flat transports. It matters for future non-flat rollout claims, not the active flat run. |
| `rg "belief_rollout|Expected|EFE|policy" vfe3` | Red / my trace | No public API found | Hits are comments or visualization candidates, not a public EFE or belief-rollout policy layer. |

### Evidence audit

| Side | path:line verified | Test output | Unverified path-line | Comment/docstring as behavior | Canon-cop strikes |
|---|---:|---:|---:|---:|---:|
| Blue | 4 core verified code claims: `BeliefState`, `forward`, `generate`, `PriorBank.encode/decode` | 1 executed smoke check confirms logits-out and generate-out behavior | 0 material | 0 counted | 0 |
| Red | 4 core verified limitation claims: no public belief object from `forward`, inert `log_likelihood`, flat model-channel transport, no EFE/rollout API | 1 search/execution set supports missing public API and logits-out path | 0 material | 0 counted | 0 |

### Concessions made

Red conceded that V3 has a plausible no-grad generation hook and enough belief-state material to justify an experiment.

Blue conceded that V3 does not currently expose a public active-inference policy layer, does not return a reusable final-belief object from `forward()`, and limits model-channel or agent-set claims to flat transport unless new code is built.

### My weighted scores

Blue weighted total: 15. Verified path-line evidence plus executed output supports the code half of the narrow claim: an opt-in inference-time scorer is implementable at the no-grad generation boundary, and training replacement is not the current path.

Red weighted total: 12. Verified path-line evidence supports real implementation limits: no public belief-rollout API, no live observation-likelihood training objective, flat model-channel transport, and no current EFE policy layer. These facts constrain the future implementation but do not show the narrow claim fails.

## Strength of each side

Blue is strongest on reachability. `BeliefState` carries the relevant tensors, `forward()` actually runs the E-step and returns logits, and `generate()` is no-grad with a clean post-forward selection boundary. The code therefore supports a default-off inference-time experiment and supports keeping training unchanged.

Red is strongest on scope control. The current implementation is not already active inference over policies. The observation likelihood is a guarded stub without production callers, active decode uses the linear head and discards `sigma_q` at the output under the current config, the model channel uses flat transport, and there is no public belief-rollout or EFE policy API.

## Action

Treat the code verdict as: proceed only with an opt-in inference-time experiment. First add a public no-grad helper that returns final beliefs plus logits for explicit candidate continuations. Then add a default-off scorer that logs risk, ambiguity or epistemic proxy, raw log probability, preference object, candidate prior, horizon, and compute. Do not call it a train-time EFE replacement, and do not make non-flat model-channel or agent-set claims until a non-flat covariant rollout path is implemented and tested.
