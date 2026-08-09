# Independent Head-Mixer Learning Rates

**Date:** 2026-08-09
**Status:** Approved design
**Scope:** Optimizer/configuration controls for the PriorBank head-evidence mixer and the legacy
post-belief HeadMixer

## 1. Goal

Give the two semantically different mixers independent M-step learning rates without changing the
behavior of existing serialized configurations. The PriorBank mixer calibrates decoder-side
per-block KL evidence, while the legacy HeadMixer transforms the converged belief representation;
neither parameter family should be forced to use the prior-mean table learning rate.

## 2. Considered approaches

1. **Optional absolute learning rates with inheritance (selected).** Add one optional field for
   each mixer. `None` inherits `m_p_mu_lr`, exactly preserving existing behavior; launchers can set
   explicit independent values.
2. **Fixed new global defaults.** Give each mixer a default such as `0.001`. This is simpler but
   silently changes every old mixer-enabled configuration.
3. **Multipliers of `m_p_mu_lr`.** Add two scalar ratios. This preserves coupling to the prior-mean
   schedule and makes run provenance less direct, so it does not provide the requested independent
   absolute controls.

## 3. Configuration contract

Add these fields to `VFE3Config`:

```python
m_head_evidence_lr: Optional[float] = None
m_head_mixer_lr: Optional[float] = None
```

- `m_head_evidence_lr` controls `PriorBank.head_evidence_logits` only.
- `m_head_mixer_lr` controls all parameters owned by the legacy post-belief `HeadMixer` only.
- `None` resolves to `m_p_mu_lr` at optimizer construction.
- Explicit values must be finite and nonnegative. Zero deliberately freezes that mixer; the shared
  absolute LR floor must not resurrect a zero-base group.
- Both fields remain serialized as configured so artifacts distinguish inheritance from an explicit
  coincident value.

Set both fields explicitly to `0.001` in the editable configuration dictionaries in
`train_vfe3.py` and `ablation.py`. Do not alter either mixer's enable/disable toggle or add an
ablation sweep in this change.

## 4. Optimizer and scheduling

Keep the existing optimizer parameter-group count and order. Replace only the two groups' base
learning-rate selection:

- head-evidence group: `m_head_evidence_lr` when explicit, otherwise `m_p_mu_lr`;
- legacy HeadMixer group: `m_head_mixer_lr` when explicit, otherwise `m_p_mu_lr`.

The head-evidence group retains its explicit `weight_decay=0.0`; the legacy HeadMixer retains its
existing inherited `cfg.weight_decay`. Both retain `role="mu"`, preserving gradient clipping and
optimizer state layout. The existing per-group warmup, cosine decay, proportional floor, and
zero-base freeze semantics apply unchanged.

## 5. Reporting and provenance

Add scheduled metrics `lr_head_evidence` and `lr_head_mixer` without changing the existing required
`lr_mu`, `lr_sigma`, and `lr_phi` reporting contract. Each optimizer group receives an auxiliary,
unique reporting label. At a metrics row:

- an active mixer reports that group's current scheduled learning rate;
- an inactive mixer reports `NaN`, which renders as a blank CSV cell;
- inherited and explicit base settings remain distinguishable in the serialized config.

Update optimizer/configuration documentation and the launcher's M-step LR summary so users can see
both configured values and their effective inheritance behavior.

## 6. Compatibility and errors

- Old configuration JSON lacking the fields loads with `None` and retains `m_p_mu_lr` behavior.
- Existing mixer checkpoints retain the same model state keys and optimizer group order.
- Explicit negative, NaN, or infinite mixer learning rates fail during config construction.
- The fields do not enable either mixer and do not allocate parameters when its mixer is disabled.
- The two mixer learning rates remain independent from each other and from `m_p_mu_lr` after
  construction.

## 7. Test strategy

Use test-first red/green cycles covering:

1. Config defaults are `None`; explicit valid values survive serialization/migration.
2. Invalid explicit values reject with field-specific messages.
3. Each mixer inherits `m_p_mu_lr` when its field is `None`.
4. Simultaneously enabled mixers receive distinct explicit base learning rates exactly once each,
   retain their pre-existing weight-decay policies, and preserve parameter-group order.
5. The scheduler scales each mixer independently and preserves a zero explicit base at zero.
6. Metrics report current scheduled mixer LRs and emit `NaN` for inactive mixers without changing
   the three required role metrics.
7. Both click-to-run configuration dictionaries expose `0.001` for both controls.
8. Focused HeadMixer, PriorBank evidence, config, scheduler, checkpoint/resume, metrics, and run-
   artifact regressions pass using the CUDA-capable interpreter where Torch is imported.

## 8. Non-goals

- Changing either mixer architecture, initialization, diagnostics, or enablement.
- Selecting an empirically optimal learning rate; `0.001` is an explicit starting configuration.
- Adding learning-rate sweeps or changing `m_p_mu_lr`.
- Combining the two mixers or renaming their existing parameters.
