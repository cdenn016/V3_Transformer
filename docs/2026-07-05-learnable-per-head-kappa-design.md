# Design: Learnable per-head kappa (2026-07-05)

Source idea: `docs/2026-07-05-improvement-ideas.md`, Tier 3 —

> Learnable per-head kappa. tau's Vaswani calibration is a dot-product-statistics
> argument that does not obviously transfer to KL energies whose scale is set by
> `sigma_init` and guards; an `(H,)` kappa parameter (init 1.0, step-0 identical,
> t5-exception family, no gauge contact) lets heads pick sharp-copy vs
> diffuse-context regimes. One optimizer group.

## Goal

Promote the two fixed softmax temperatures from scalars to per-irrep-block **learned**
temperatures, so each head (or irrep block) can independently choose a sharp-copy regime
(small kappa, low tau, near-argmax attention) or a diffuse-context regime (large kappa,
high tau, near-uniform attention). Both channels are covered:

- belief channel: `tau_beta = kappa_beta * sqrt(d_block)` (the softmax temperature that
  shapes the output logits and cross-entropy);
- model channel: `tau_gamma = kappa_gamma * sqrt(d_block)` (the s-coupling temperature in
  the gamma model-consensus term).

Two default-OFF toggles, step-0 byte-identical, no gauge contact. This is a clean member
of the "t5-exception family" (`t5_learnable_bias`, `learnable_r`, `pos_phi='learned'`):
a sanctioned learned-scalar/table exception to the no-neural-networks constraint, with the
pure param-free path preserved as the default.

## Background: how tau flows today

`free_energy.attention_tau(kappa, irrep_dims)` (`vfe3/free_energy.py:43-78`) computes
`tau = kappa * sqrt(d_block)`, where `d_block` is the gauge-irrep block size read off
`irrep_dims` (sum == K). For an equal-block group (`block_glk`, the production default) it
returns a scalar when `kappa` is scalar, and an `(H,)` vector when `kappa` is itself an
`(H,)` tensor. tau is consumed un-detached through three differentiable paths:

- `attention_weights`: `logits = -energy / _broadcast_tau(tau, energy)` then softmax
  (`free_energy.py:313`);
- `log_partition` / `reduced_free_energy`: `-tau * logsumexp(...)` (`free_energy.py:335, 360-366`);
- the attention-entropy term `tau * beta * log(beta / pi)` inside `free_energy`
  (`free_energy.py:443-445`).

`_broadcast_tau` (`free_energy.py:25-40`) reshapes a 1-d `(H,)` tau to `(H,1,1)` for the
`(...,H,N,N)` energy; `reduced_free_energy` has its own `(H,)->(H,1)` reshape. Because tau
is never detached, a learned kappa receives a real gradient from the loss.

Today kappa can already be *per-head* — a `kappa_beta` list becomes an `(H,)` float32
constant via `_as_coeff` (`vfe3/model/block.py:21`) — but it is never a learned tensor.
No `nn.Parameter` kappa exists anywhere in the codebase.

Consumption sites relevant to this design:

- belief channel: `vfe_stack` at `stack.py:66` (the real E-step temperature, hoisted once
  per stack and threaded to every block), reached from `model.py` via three `vfe_stack`
  calls (`:730` forward, `:1219` gamma-attention-maps, `:1540` diagnostics);
- model channel: `_refine_s` at `model.py:524` (the gamma E-step) and `_gamma_energy` at
  `model.py:1133` (the gamma coupling term in the forward loss). Both are `VFEModel`
  methods and read `self` directly, so they need no argument threading.

`stack.py:66` and `block.py:63` are stateless free functions that read `cfg.kappa_beta`;
they never see the model, so the learned belief-channel kappa must be threaded in as an
argument.

## Parameterization (constraint-forced)

Store `log_kappa_beta` and `log_kappa_gamma` as `nn.Parameter`s and use
`kappa = exp(log_kappa)`. Log-space is forced, not a preference:

1. **Positivity is load-bearing.** tau is a divisor in `softmax(-E/tau)`; a raw parameter
   crossing zero divides by zero and flips the softmax sign. `exp(.)` keeps tau strictly
   positive for any parameter value. This matches the codebase's existing convention for
   every positive quantity (`sigma_log_embed`, `r_sigma_log`).
2. **Exact step-0 identity.** `log_kappa = log(kappa_config)`; for the scalar default
   `kappa=1.0`, `log(1.0) = 0` exactly and `exp(0) = 1.0` exactly, so the learnable model
   is byte-identical to the config-scalar path at construction. Softplus cannot reproduce
   `kappa=1` exactly at init (`inverse_softplus(1)` roundtrips with ULP error).

**Shape** is `(len(group.irrep_dims),)`, **not** `cfg.n_heads`. That is precisely the
length `attention_tau` validates: `n_heads` for `block_glk`/`tied_block_glk`, the
irrep-block count for `so_n`/`sp_n`, and `1` under `cross_couplings` (single `[K]` block).
Sizing by `len(irrep_dims)` makes the parameter automatically correct for every group.

**Init** reads `cfg.kappa_beta` / `cfg.kappa_gamma`: a scalar broadcasts to a full
`(H,)` vector; a per-head/per-block list (already validated to length `len(irrep_dims)` by
`__post_init__`) is used elementwise. Init draws zero global RNG (deterministic), so it is
byte-safe regardless of placement, but the parameter is still created **last and
conditionally** so the default (toggle-OFF) `state_dict` stays parameter-free.

## The five seams (mirroring `t5_learnable_bias`)

### 1. Config (`vfe3/config.py`)

Add two plain dataclass fields with the sanctioned-exception inline comment convention:

- `learnable_kappa_beta: bool = False` immediately after `kappa_beta` (`:207`);
- `learnable_kappa_gamma: bool = False` immediately after `kappa_gamma` (`:288`).

In `__post_init__`, add an inert-warning (not an error) when a toggle is ON but the active
group has a single irrep block: per-head learning is then vacuous (one scalar temperature),
which is harmless and still valid. No new hard validation is required — the existing kappa
positivity/length checks (`:1007-1029`) govern the config values, and the learned parameter
is kept positive by `exp`.

### 2. Parameter creation (`vfe3/model/model.py`, `VFEModel.__init__`)

After the existing t5/pos_phi/connection parameter blocks (created last), conditionally:

```python
if cfg.learnable_kappa_beta:
    H = len(self.group.irrep_dims)
    k0 = cfg.kappa_beta
    k0_vec = (torch.tensor(k0, dtype=torch.float32) if isinstance(k0, (list, tuple))
              else torch.full((H,), float(k0)))
    self.log_kappa_beta = nn.Parameter(torch.log(k0_vec))
    # freeze warning under detach / straight_through (see seam 5)
```

and symmetrically `log_kappa_gamma` from `cfg.kappa_gamma`. Both draw zero RNG.

### 3. Consumption

Two helper methods on `VFEModel`:

```python
def effective_kappa_beta(self, device):
    p = getattr(self, "log_kappa_beta", None)
    return torch.exp(p).to(device) if p is not None else _as_coeff(self.cfg.kappa_beta, device)
# effective_kappa_gamma analogous
```

- **Belief channel:** add one optional kwarg `kappa_beta_override: Optional[torch.Tensor] = None`
  to `vfe_stack`; at `stack.py:66` feed `kappa_beta_override` to `attention_tau` when it is
  not None, else the current `_as_coeff(cfg.kappa_beta, device)`. Pass
  `kappa_beta_override=self.effective_kappa_beta(dev)` at all three `vfe_stack` calls
  (`model.py:730, 1219, 1540`). `attention_tau` then validates the length and returns the
  `(H,)` tau, which the existing `_broadcast_tau` / `query_adaptive_tau` machinery already
  handles.
- **Model channel:** at `_refine_s:524` and `_gamma_energy:1133`, replace
  `_as_coeff(cfg.kappa_gamma, dev)` with `self.effective_kappa_gamma(dev)`. No threading —
  both are `self` methods.

### 4. Optimizer (`vfe3/train.py`, `build_optimizer`)

Two conditional groups, mirroring `t5_bias` (`train.py:210`):

```python
if getattr(model, "log_kappa_beta", None) is not None:
    groups.append({"params": [model.log_kappa_beta], "lr": cfg.m_p_mu_lr,
                   "weight_decay": 0.0, "role": "mu"})
# log_kappa_gamma analogous
```

`role='mu'` (the documented catch-all for learned non-variance/non-gauge tables), `lr`
= the mean LR `m_p_mu_lr`, `weight_decay=0.0` (a temperature decayed toward 0 biases the
softmax — the same exemption `t5_bias`/`output_proj_bias`/`r` carry), and **no** `gauge`
flag (no gauge contact, so it rides as a plain group even under `GaugeNaturalGradAdamW`).
This group is mandatory: the exact-coverage guard (`train.py:232-238`) raises
`AssertionError` for any ungrouped trainable parameter.

### 5. Freeze guards

Both kappas enter the loss only through E-step-coupled softmax temperatures, so the
detached / straight-through estimators can sever their gradient (the family's
`detach_e_step` footgun). Add:

- a model-level `warnings.warn` at parameter creation, gated on
  `cfg.effective_e_step_gradient in ("detach", "straight_through")`, mirroring
  `model.py:337-354` for `t5_bias`;
- an extension of the config oracle-route freeze predicate (`config.py:~1906`) to include
  `self.learnable_kappa_beta or self.learnable_kappa_gamma`.

The **exact** estimators under which each kappa freezes are verified by the gradient test
below rather than asserted from the docstrings; the predicate is set to match the observed
behavior.

## Scope boundaries

**In scope (load-bearing):** the training/validation forward path (`model.forward` ->
`vfe_stack`, `_refine_s`, `_gamma_energy`) and the belief-channel diagnostics
(`vfe_stack` at `:1219`, `:1540`). Validation cross-entropy flows through `model.forward`
and therefore always reflects the learned temperature.

**Deferred (documented limitation):** the external `attention_tau` sites read
`cfg.kappa_*` and will show the *config* temperature — `inference/belief_cache.py:176`
(autoregressive `generate()`), `metrics.py:1045`, `viz/extract.py`, `train.py:1139`, and
the `cfg.tau` / `cfg.tau_gamma` logging properties. Only generation fidelity via
`belief_cache` could matter; it is out of scope for this build and can be threaded later.

**Must verify during implementation:** confirm the validation-CE path routes through
`model.forward` (so the learned kappa reaches it) and not an independent `attention_tau`
in `train.py:1139`; thread whichever computes the evaluated loss.

## Equivariance

Like `t5_bias`, a per-block scalar temperature multiplies the already-gauge-invariant
per-block energy inside the softmax and touches no gauge transport, so a learnable kappa
does **not** break gauge equivariance. It is the cleanest exception class (unlike
`connection_W` / the head mixer, which deviate off identity init). This will be recorded in
the `CLAUDE.md` documented-exceptions list.

## Testing (TDD; tiny K < 6, CPU-bound)

1. **Step-0 byte-identity:** `learnable_kappa_beta=True` (and `_gamma`) vs the config-scalar
   path produce identical logits and free energy at construction (`kappa = exp(0) = 1.0`).
2. **Gradient flow:** under `e_step_gradient='unroll'`, `log_kappa_beta.grad` and
   `log_kappa_gamma.grad` are non-None and nonzero after one backward pass; under `'detach'`
   / `'straight_through'` they are None. This pins the freeze-warning predicate to real
   behavior.
3. **Shape / group coverage:** parameter shape is `(len(group.irrep_dims),)` for
   `block_glk`, an irrep tower (`so_n`), and a single-block/cross_couplings group; the
   single-block case emits the vacuity warning and learns one scalar.
4. **Optimizer wiring:** `build_optimizer` places both parameters in a `role='mu'`,
   `weight_decay=0.0`, non-`gauge` group, and the exact-coverage guard passes.

All tests instantiate single-digit models (K, N, heads, layers, vocab) and run on CPU in
well under a second, per the project testing rules. No production-scale model is built in
any test.

## Files touched (anticipated)

- `vfe3/config.py` — two fields + inert warning + oracle-route predicate.
- `vfe3/model/model.py` — parameter creation, two helper methods, freeze warnings, three
  `vfe_stack` threadings, two gamma-site edits.
- `vfe3/model/stack.py` — one optional `kappa_beta_override` kwarg + its use at `:66`.
- `vfe3/train.py` — two conditional optimizer groups.
- `tests/` — a new `test_learnable_kappa.py` covering the four cases above.
- `CLAUDE.md` — add the toggle to the documented-exceptions family list.
- A dated post-edit note per the project's Post-Edit Policy.
