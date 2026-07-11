# Design: Ablation Sweep Route Compatibility

Date: 2026-07-11
Status: approved for implementation
Branch: `codex/ablation-sweep-route-20260711`
Base: `origin/main` at `ecc128966087cbba6471c028cfe2076d78d73b84`

## Problem

The complete CPU suite on the rebased P6 branch reported four ablation sweep-construction
failures. A detached worktree at the exact `origin/main` base reproduced all four failures without
any P6 files present. The failing cells inherit `e_step_update="mm_exact"` from
`ablation.py`'s active `BASELINE_CONFIG`, but their scientific purpose requires an autograd-oracle
belief-gradient route. `VFE3Config` correctly rejects MM-exact on that route because the closed-form
MM update is available only to the eligible diagonal-Gaussian filtering kernel.

The affected sweeps are `attention_entropy`, whose entropy-suppressed arms disable a kernel
eligibility condition; `gauge_equivariance`, whose full-covariance arms require the oracle;
`pos_extrapolation`, whose RoPE arm uses the decoupled-value oracle route; and `regime_ii`, whose
non-flat connection requires the oracle.

## Immutable active settings

This repair will not change any value in `ablation.py`'s `BASELINE_CONFIG` or in
`train_vfe3.py`'s `config` dictionary. In particular, the ablation baseline remains
`e_step_update="mm_exact"` with `mm_damping=0.75`, and the training entry point remains
`e_step_update="gradient"` with `mm_damping=1.0`. Cadences, data seeds, model choices, optimization
settings, and all other active toggles remain untouched.

Tests will pin these four values before the implementation is added. Final diff inspection will
also require `train_vfe3.py` to be absent from the prerequisite branch and will reject any change
to the `BASELINE_CONFIG` block.

## Selected repair

Each of the four affected sweep definitions will declare the legal update rule as a local
requirement. `attention_entropy`, `gauge_equivariance`, and `regime_ii` will add
`"requires": {"e_step_update": "gradient"}`. `pos_extrapolation` will add the same key to its
existing `requires` mapping while retaining its existing oracle and sequence-length requirements.

This changes only the generated cells for experiments whose own arms necessarily leave the
MM-exact kernel domain. It does not change the baseline used by ordinary ablation cells, the
explicit `mm_damping` sweep, or the main training configuration. The saved cell configuration will
state `e_step_update="gradient"` directly; there will be no hidden coercion.

## Rejected alternatives

A generic fallback in `_cell_cfg_dict` would silently replace an inherited MM-exact setting after
the user selected it and would duplicate route-eligibility logic already owned by `VFE3Config`.
Relaxing the `VFE3Config` guard would permit an update rule without a valid closed-form derivation.
Changing tests to expect construction failures would leave four advertised sweeps unusable. None
of these alternatives is acceptable.

## Test-first verification

The RED test will assert that the active ablation and training config values retain their approved
values, then require every generated arm of the four affected sweeps to carry the local gradient
requirement and build as a real `VFE3Config` plus `VFEModel`. Before implementation it must fail on
the missing local requirement while proving the active dictionaries already contain the protected
values.

After the minimal sweep-contract edit, the new regression and the original four failing nodes must
pass. The prerequisite branch will then run the relevant ablation/config test set and the complete
default CPU suite with JUnit XML. After review and merge, P6 will rebase onto the repaired
`origin/main`, rerun its focused CPU and RTX 5090 checks, and rerun the complete default CPU suite
before P6 is pushed and merged.

## Acceptance criteria

The repair is accepted when all four sweeps construct under their scientifically required oracle
route, both active config dictionaries retain the approved values, no hidden fallback or validator
relaxation exists, the prerequisite diff is limited to sweep contracts, regression coverage, and
the dated documentation, and the complete suite is green on both the prerequisite branch and the
rebased P6 branch.
