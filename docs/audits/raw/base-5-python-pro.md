# Base investigator 5 — python-pro (type safety, contracts)

Returned 2026-07-27 ~08:33 CDT. Verbatim findings; NOT yet verified.
Tooling actually run: `mypy 1.11.2` via `C:/anaconda/python.exe -m mypy --ignore-missing-imports`
against `families/base.py`, `geometry/transport.py`, `geometry/groups.py`, `geometry/retraction.py`,
`alpha_i.py`, `lambda_h_i.py`, `attention_prior.py`, `contracts.py`.

## Negative results (load-bearing)

No critical or high findings. **The kwargs-bag hazard flagged in the brief — the pattern that
historically caused a 47-test failure — was traced key-by-key and is NOT present.** Path traced:
`model/block.py:e_step_shared_kwargs` -> `inference/e_step.py` (`e_step`, `e_step_iteration`,
`free_energy_value`, `phi_alignment_loss`) -> `viz/extract.py` (`_iter_kwargs`, `_fe_kwargs`).
Every key in the shared bag is declared as an explicit accept-and-ignore parameter — never a bare
`**kwargs` sink — on both `e_step_iteration` and `free_energy_value`, so a misspelled knob still
raises `TypeError` rather than being silently swallowed. The `audit 2026-07-12 N5` fix is intact.

No violation of the signature-alignment or mutable-default-argument conventions found in the
reviewed registries.

---

### Registry capability check is a proxy, not the real capability — a new diagonal family can pass config validation and still crash on first use
**Location:** vfe3/config.py:1784-1798 (validation), vfe3/free_energy.py:276-290 (runtime guard), vfe3/families/base.py:688-703 (unguarded call)
**Severity:** medium
**Evidence:** `config.py` only checks `family_is_diagonal` and
`has_per_coord_functional(self.divergence_family)` before allowing
`lambda_alpha_mode='state_dependent_per_coord'`. `free_energy.py:276` mirrors this:
`if q.cov_kind != "diagonal": raise ValueError(...)`. Neither checks whether the concrete family
class actually implements `renyi_per_coord`. The consumer calls it unconditionally —
`families/base.py:703`: `return q.renyi_per_coord(p, alpha=alpha, kl_max=kl_max, eps=eps)` — with no
`getattr`/`hasattr` guard, in contrast to the sibling hook at `base.py:597`,
`closed = getattr(q, "renyi_closed_form", None)`, which *is* guarded. `BeliefParams` declares no
`renyi_per_coord` stub (mypy: `"BeliefParams" has no attribute "renyi_per_coord"` at base.py:703,
721, 739, 740), unlike other optional hooks (`expected_statistic`, `natural_gradient`,
`covariance_diagonal`) which raise a clear `NotImplementedError`. Today's diagonal families
(`DiagonalGaussian`, `DiagonalLaplace`, `ExactCongruenceDiagonalGaussian`, `FrameDiagonalGaussian`)
happen to inherit it, so no shipped config is broken. But `base.py:66-71`'s own docstring calls out
this exact gap class — "CAPABILITY DECLARATIONS (audit 2026-07-26 E-03/E-04) ... so
`VFE3Config.__post_init__` can reject an unsupported pairing at construction instead of letting the
family raise mid-forward" — and that pattern (`transport_requirement`, `requires_kl_divergence`,
`gaussian_pointwise_algebra`) was never extended to per-coordinate divergence support.
**Fix:** Add a `ClassVar[bool] supports_per_coord_divergence` to `BeliefParams` defaulting `False`,
set `True` on `DiagonalGaussian`/`DiagonalLaplace`, and validate it in `config.py` alongside the
existing `cov_kind` check.

### `BeliefParams` ABC has no declared constructor, so its own classmethods call an unchecked `cls(mu, dispersion)`
**Location:** vfe3/families/base.py:318, vfe3/families/base.py:401
**Severity:** medium
**Evidence:** `base.py:318`: `return cls(mu, dispersion)` inside `from_transported`; `base.py:401`:
`cls(mu_q, dispersion_q)` inside `coupling_energy`. `BeliefParams` (`class BeliefParams(ABC):`,
base.py:48) declares no `__init__` anywhere in the file (`grep def __init__` in base.py returns
nothing), so mypy reports `Too many arguments for "BeliefParams"` at both call sites — there is no
enforced contract that a registered family's constructor accepts exactly `(mu, dispersion)`
positionally. Every shipped family matches it, but `register_family` (base.py:411-422) checks no
constructor signature, so an incompatible new registration fails only with a runtime `TypeError`
raised from deep inside `from_transported`/`coupling_energy`, far from the registration mistake.
**Fix:** Declare an abstract or default `__init__(self, mu: torch.Tensor, dispersion: torch.Tensor)
-> None` on `BeliefParams`, or have `register_family` validate the constructor signature at
registration time.

### `omega` typed as bare `object` on the family transport/coupling hooks defeats type checking
**Location:** vfe3/families/base.py:324, :346, :368
**Severity:** low
**Evidence:** `transport_location` (321-324), `transport_dispersion` (343-346) and
`coupling_energy` (362-368) all declare `omega: object`, then call `transport_mean(omega, mu)`
(340) and `transport_covariance(omega, dispersion, diagonal_out=diagonal_out)` (359). mypy:
`Argument 1 to "transport_mean" has incompatible type "object"; expected "Tensor |
CompactFactoredTransport | DirectLinkTransport | FactoredTransport | RopeTransport"` (base.py:340)
and the analogue at 359. Every other signature carrying this parameter (`belief_gradients`,
`_transport`, `build_belief_transport` in `inference/e_step.py`) spells out the precise union; only
these three public, subclass-overridable hooks degrade it to an `Any`-equivalent.
**Fix:** Replace `omega: object` with the precise union used elsewhere.

### `fold_rope_into_frame`'s declared type omits the `DirectLinkTransport` case it actually handles
**Location:** vfe3/geometry/transport.py:345-348, :368-380
**Severity:** low
**Evidence:** Signature: `def fold_rope_into_frame(base: 'FactoredTransport |
CompactFactoredTransport', rope: torch.Tensor) -> 'FactoredTransport | CompactFactoredTransport':`
(345-348). Body: `if isinstance(base, DirectLinkTransport): ... return
DirectLinkTransport(exp_link=base.exp_link, exp_phi=base.exp_phi @ rope_t, exp_neg_phi=rope @
base.exp_neg_phi)` (368-380). mypy: `Incompatible return value type (got "DirectLinkTransport",
expected "FactoredTransport | CompactFactoredTransport")` at 376. Reachable:
`RopeTransport.score_operator` (250-273) explicitly allows `self.base` to be a
`DirectLinkTransport` (264-265) before calling `fold_rope_into_frame(self.base, self.rope)` (271).
**Fix:** Add `DirectLinkTransport` to both the parameter and return annotations.
