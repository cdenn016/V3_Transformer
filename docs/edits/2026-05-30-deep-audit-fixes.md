# Deep-audit fix pass — 2026-05-30

Branch `audit-fixes-2026-05-30` (fresh from `main`). Implements the confirmed punch list
from `docs/audits/audit-2026-05-30.md` (six-investigator deep audit + verifier). The user
authorized "fix all" and selected the maximal option on all four forks: implement the
full-covariance path end-to-end, wire all five dead config seams, switch to per-head
temperature, and do the golden-test-gated transport/batch perf rewrite. Work proceeds in
five dependency-ordered groups, each committed separately so banked work stays green.

This doc is a companion to `2026-05-30-diagnostics-tier.md` (which documents the earlier,
already-merged diagnostics build); kept separate because it is a distinct work stream on an
unmerged branch.

## Group 1 — safe hygiene + safe perf (no dynamics change)

Fixes that cannot change any numeric output on the default path; the suite stays at 182/182.

### Files changed
- `vfe3/model/model.py`
- `vfe3/viz/figures.py`
- `vfe3/metrics.py`
- `vfe3/inference/e_step.py`
- `vfe3/free_energy.py`

### Changes
- **`model.py`**: replaced the hand-rolled `_nullcontext` class with `contextlib.nullcontext`
  (finding 5f); typed `_apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool
  = True) -> "VFEModel"` (finding 5d). Behavior identical.
- **`viz/figures.py`**: narrowed `except Exception` to `except ImportError` around the optional
  seaborn palette so a real seaborn runtime error no longer gets swallowed (finding 1e).
- **`metrics.py`**: the four registered metrics now take their OWN context key as a REQUIRED
  keyword (no `None` default), so a missing/mis-keyed context raises `TypeError` at the call
  rather than an `AttributeError` deep inside `effective_rank(None)` (finding 5b); the trailing
  `**kw` is kept deliberately because `compute_metrics` floods the full context to every metric,
  so each must absorb its siblings' keys. `holonomy_deviation` now evaluates the first 512
  triangles as one batched `(T,K,K)` matmul instead of 512 Python-dispatched `(K,K)` matmuls —
  same triangles, same value, one kernel launch (finding 4i).
- **`e_step.py`**: the diagnostic `return_trajectory` free-energy evaluations now run under
  `torch.no_grad()` and use `.item()` (explicit host sync) so the logged scalar never enters the
  training graph (finding 4g). Default path (`return_trajectory=False`) is untouched.
- **`free_energy.py`**: the entropy-term log floor is now a parameter `log_eps: float = 1e-12`
  (was a hardcoded literal), default unchanged so values are bit-identical (finding 1d).

### Deliberately deferred / kept (with rationale)
- **4j** (redundant `.float()` casts in `divergence.py`): KEPT. They are dtype-safety guards on
  the path that also serves float64 numerical islands; conditionalizing them adds branch noise to
  the hottest kernel for negligible gain at default sizes. Documented, not changed.
- **5c / 5e / 5g** (pure static-typing ergonomics: `@overload` on `e_step`, `Protocol` registry
  types, `TypeVar`-preserving decorators): DEFERRED. They have no runtime effect and there is no
  type-checker in this repo's CI (pytest only), so the boilerplate trades against the project's
  simplicity mandate. Re-open if a mypy gate is added.
- **2d / 4f** (norm re-instantiated per forward): folded into Group 4, which restructures the
  model forward / stack for batch vectorization and will own the norm-instance caching.

### Verification
- `pytest -q` (JUnit XML count): tests=182, failures=0, errors=0 after Group 1.

## Group 2 — wire / enforce the five config seams (+ self-contained plumbing)

The user chose "wire all five" over "delete redundant." Following the advisor's
"enforce, don't invent" rule: the two seams with a ready implementation are genuinely
wired; the three under-defined ones are made LIVE and ENFORCED (reject contradictory or
unsupported values) rather than having fictional FEP paths invented for them.

### Files changed
- `vfe3/config.py`, `vfe3/train.py`, `vfe3/inference/e_step.py`, `vfe3/model/block.py`,
  `vfe3/alpha_i.py`, `tests/test_config.py`, `tests/test_e_step.py`

### Changes
- **`seed`** (3d): `run_training` now calls `torch.manual_seed(cfg.seed)` before building the
  model/loader — reproducible prior-table init and data order. Genuinely wired.
- **`use_prior_bank`** (3b): `__post_init__` raises `NotImplementedError` for
  `use_prior_bank=False` — the PriorBank is the only encode/decode boundary and no alternative
  is specified, so the knob is live and rejects the unsupported value (no invented path).
- **`divergence_family`** (2a) and **`diagonal_covariance`** (3c): `__post_init__` enforces both
  consistent with `family` (the single source of truth) — a contradictory pair (e.g.
  `divergence_family='gaussian_full'` with `family='gaussian_diagonal'`, or
  `diagonal_covariance=False` with a diagonal family) now raises `ValueError` instead of being
  silently ignored.
- **`gauge_parameterization`** (2c/3a): the actual transport dispatch (phi vs omega_direct) is
  wired in Group 4, which owns `_transport`; deferred here to avoid editing the E-step plumbing
  twice (Group 4 rewrites it).
- **kwargs sink** (1c/5a): `free_energy_value`'s blanket `**kwargs` sink is replaced by explicit
  accept-and-ignore iteration knobs (`gradient_mode`, `phi_precond_mode`, `phi_retract_mode`,
  `sigma_max`, `e_sigma_q_trust`). A misspelled real parameter now raises `TypeError` instead of
  being swallowed; `e_step_iteration` already had no sink, so both knob-bag consumers now reject
  typos. New test `test_free_energy_value_rejects_misspelled_kwarg`.
- **`state_dependent_per_coord`** (2e/3e): emits a `RuntimeWarning` (deduped) that the mode
  currently receives the summed per-position divergence and silently degrades to per-position
  alpha; points users to `state_dependent`.
- **`compose_bch`** (3j): new config field `phi_retract_mode` ("euclidean" | "bch", validated),
  threaded `block -> e_step -> e_step_iteration -> retract_phi(mode=...)`, so the registered BCH
  chart correction is now config-selectable (was registered but unreachable).

### New tests (tests/test_config.py, tests/test_e_step.py)
- `use_prior_bank=False` -> NotImplementedError; `divergence_family != family` -> ValueError;
  `diagonal_covariance` must agree with family (and the consistent full triple is accepted);
  `phi_retract_mode` validated and "bch" accepted; `seed` field present; `free_energy_value`
  rejects a misspelled kwarg but accepts the iteration-only knobs.

### Pre-existing issues noted (not changed, per surgical-changes policy)
- `tests/test_config.py` defines `test_invalid_divergence_family_raises` twice (the second
  shadows the first).
- `tests/test_config.py` lines 11 and 31 pin `tau == kappa*sqrt(embed_dim)` — these are the
  formula-pinning assertions Group 3 updates to `sqrt(d_head)`.

### Verification
- `pytest -q` (JUnit XML count): tests=188 (+6 new), failures=0, errors=0 after Group 2.
