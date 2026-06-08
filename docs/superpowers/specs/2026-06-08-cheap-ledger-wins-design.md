# Design spec: three cheap ledger wins (M6 b0/c0 sequence, T3 per-head ALiBi, T1 per-head kappa)

**Date:** 2026-06-08
**Status:** approved design, pre-implementation
**Roadmap:** ledger items M6, T3, T1 (`docs/2026-06-07-buildout-roadmap-status.md`)
**Scope:** three INDEPENDENT small expandability seams; each is opt-in / default byte-identical.

These generalize three scalar config knobs into per-coordinate / per-head forms. The kernels mostly
already support the generalized type; the work is the config leg plus correct broadcasting. The pure
default path stays bitwise-unchanged for all three.

## M6 — `b0` / `c0` accept a sequence (cheapest)

The state-dependent-alpha kernel already accepts `b0`/`c0` as `float | torch.Tensor` and is `(K,)`-capable
(`alpha_i.py:53,77,121,143`). Only the config leg is scalar.

- **Config:** `cfg.b0`, `cfg.c0` become `float | List[float]`. Validation: a list must have
  `len == embed_dim` and all entries `> 0` (the regularizer `R(alpha)=b0·alpha − c0·log alpha` needs
  positive constants). A `List[float]` round-trips through `asdict`→`config.json` (the roadmap's deferral
  worried about a *tensor* field; a list does not break json).
- **Conversion:** the list -> `(K,)` float32 tensor on the model device happens at the consumption
  boundary. `VFEModel.__init__` builds `self._b0` / `self._c0` (the float unchanged, or a `(K,)` tensor
  registered as a non-persistent buffer so it follows `.to(device)`), and these are passed everywhere
  `cfg.b0`/`cfg.c0` currently flow (`block.py:52` into `e_step`; `model.py` M-step self-coupling). The
  per-coord alpha consumes `b0`/`c0` only under `alpha_mode='state_dependent_per_coord'`; a `(K,)` tensor
  broadcasts against the `(..., K)` per-coordinate divergence.
- **Default (scalar):** `self._b0`/`self._c0` are the float; every call site is byte-identical.
- **Files:** `vfe3/config.py` (type + validation), `vfe3/model/model.py` (build + thread `_b0`/`_c0`).
- **Tests:** default scalar byte-identical; a `List[float]` of length `K` threads a `(K,)` tensor into
  the per-coord alpha and changes `alpha_i^(k)` per coordinate; wrong-length list rejected; non-positive
  entry rejected; `asdict(cfg)` is json-serializable with a list `b0`.

## T3 — per-head ALiBi slope (Press schedule)

`prior_alibi` / `prior_causal_alibi` (`attention_prior.py:69,86`) take a single `slope=1.0` (pinned,
unreachable from config) and return `(N, N)`. Press et al. use a per-head geometric slope.

- **Schedule:** `slope_h = alibi_slope · 2^(−8(h+1)/H)` for `h = 0..H−1` (`H = n_heads`); at
  `alibi_slope=1` this is the standard Press geometric schedule. The two priors return `(H, N, N)`
  (per-head bias) which broadcasts against the `(B, H, N, N)` block_glk energy.
- **Config:** a new `alibi_slope: float = 1.0` field (the base; makes the slope config-reachable, closing
  the audit's "slope unreachable" note). `n_heads` reaches the prior through its call site.
- **Call site:** `VFEModel._attention_log_prior` must pass `n_heads` (and `alibi_slope`) into the prior;
  the returned `(H, N, N)` flows as the attention `log_prior`. For single-block groups (no head axis) the
  prior collapses to `(N, N)` with `H=1` — i.e. the single Press slope `alibi_slope·2^(−8)`.
- **Default config is UNCHANGED:** the default `attention_prior='causal'` (not alibi), so no default run
  touches this. This changes the behavior of the `alibi`/`causal_alibi` variants only (making them the
  correct per-head ALiBi). The pure path (causal `−inf` mask) is untouched.
- **Files:** `vfe3/attention_prior.py` (per-head slopes), `vfe3/config.py` (`alibi_slope`),
  `vfe3/model/model.py` (`_attention_log_prior` passes `n_heads`/`alibi_slope`).
- **Tests:** `prior_alibi` with `n_heads=H` returns `(H, N, N)` with the Press geometric slopes (head 0
  steepest decay); the bias is symmetric `−slope·|i−j|`; `causal_alibi` keeps the upper-triangular `−inf`
  mask per head; default config (causal) is byte-identical (alibi path not exercised).

## T1 — per-head `kappa` (most threading)

`attention_tau(kappa, irrep_dims) = kappa · sqrt(irrep_dims[0])` (`free_energy.py:19-32`) already accepts
`kappa: float | torch.Tensor`; a `(H,)` kappa yields a `(H,)` tau. The work is broadcasting that `(H,)`
tau correctly through every softmax/energy site.

- **Config:** `cfg.kappa`, `cfg.kappa_gamma` become `float | List[float]`. A list requires the active
  group to be block_glk with `len == n_heads` (per-head only makes sense with equal irrep blocks); a
  single-block group requires a scalar (validation). At construction the list -> `(H,)` float32 tensor.
- **Broadcasting:** the energy is `(B, H, N, N)` (block_glk). Every `−energy/tau` and `tau·(…)` site must
  reshape a `(H,)` tau to `(H, 1, 1)` so it aligns with the head axis. Sites to cover (the implementer
  greps for `tau` to confirm the full set): `free_energy.attention_weights` (`:211`),
  `free_energy.log_partition` (`:233`), `free_energy.free_energy` entropy term (`:314`),
  `free_energy.reduced_free_energy` (`:252`), and the closed-form belief-gradient kernel in
  `vfe3/gradients/kernels.py` (which also forms `beta = softmax(−E/tau)`), plus
  `e_step.phi_alignment_loss`. A scalar tau leaves all of these byte-identical (no reshape path taken).
- **Helper:** add one small `_broadcast_tau(tau, energy_ndim)` in `free_energy.py` that returns the
  scalar unchanged or reshapes a `(H,)` tensor to `(H,) + (1,)*(energy_ndim-3)` aligned to the head axis;
  call it at each consumption site. This keeps the broadcasting logic in one place.
- **Default (scalar):** `_broadcast_tau` returns the scalar; bitwise-unchanged.
- **Files:** `vfe3/config.py` (type + validation), `vfe3/free_energy.py` (the helper + reshape at the tau
  sites), `vfe3/gradients/kernels.py` (reshape at the kernel's tau site), `vfe3/model/model.py` /
  `vfe3/model/block.py` (build the `(H,)` kappa tensor and pass it to `attention_tau`).
- **Tests:** default scalar byte-identical (logits + loss); a per-head `List[float]` kappa changes the
  attention weights per head (a head with larger kappa = sharper softmax); single-block group rejects a
  list kappa; the per-head and scalar paths agree when all per-head entries are equal.

## Build structure

One plan, **three independent tasks** built cheapest-first (M6 -> T3 -> T1), each TDD with a
default-byte-identity test plus a positive per-coord/per-head test. No shared state. T1 is the most
threading (multiple tau consumption sites incl. the kernel); its byte-identity test is the guard that no
site was missed.

## Out of scope

- Learnable per-head kappa (T2) — stays a fixed scalar/vector; not part of this.
- Non-geometric / learned ALiBi slope schedules — only the Press geometric schedule (base-scaled).
- Tensor-valued (as opposed to list-valued) config fields — lists only, to keep `config.json`
  serializable.
