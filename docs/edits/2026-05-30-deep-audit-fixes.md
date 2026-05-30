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
