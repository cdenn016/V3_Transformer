# Block MLP Sweep Design

## Goal

Expose independent, opt-in ablation sweeps for the block MLP toggle, expansion ratio, activation, dropout, and optimizer learning rate without changing the active sweep order or any existing local experiment setting.

## Sweep contracts

- `block_mlp`: compare `use_block_mlp=False` with `use_block_mlp=True`.
- `block_mlp_expansion`: values `[1, 2, 4, 8]` with `use_block_mlp=True`.
- `block_mlp_activation`: values `["gelu", "silu", "relu"]` with `use_block_mlp=True`.
- `block_mlp_dropout`: values `[0.0, 0.01, 0.05, 0.1]` with `use_block_mlp=True`.
- `m_block_mlp_lr`: values `[0.001, 0.002, 0.004, 0.008]` with `use_block_mlp=True`.

Each definition follows the existing `SWEEPS` schema. None of these sweep names is added to `SWEEP_ORDER`, so defining them cannot start an experiment automatically. The grids are one-dimensional rather than a 192-cell Cartesian product, allowing the effect of each control to be isolated.

## Preservation boundary

The current contents of `ablation.py`, `train_vfe3.py`, and `vfe3/config.py` are authoritative local settings. Implementation may only add the four missing MLP hyperparameter sweep declarations and their tests. `zzzzz.py` remains untracked and byte-identical.

## Verification

Behavioral tests must call `make_run_overrides` and `validate_sweeps` for every MLP sweep, assert the exact emitted override dictionaries, and assert that no MLP sweep is present in `SWEEP_ORDER`. Syntax parsing and construction of both click-to-run configuration dictionaries must remain successful.
