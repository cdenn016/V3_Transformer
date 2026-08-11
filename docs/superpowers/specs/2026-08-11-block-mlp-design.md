# Default-Off Per-Block MLP Design

Date: 2026-08-11
Status: approved for implementation
Branch: `codex/block-mlp-20260811`

## Objective

Add a conventional Transformer-style position-wise MLP to the VFE Transformer as an explicit, default-off structural augmentation. The MLP is an experimental comparison arm; it is not derived from the canonical variational free-energy objective and is not claimed to be gauge equivariant.

The disabled configuration must instantiate no MLP module, add no parameters or state-dictionary keys, and preserve the existing forward path exactly.

## Architectural placement

Each VFE layer currently performs:

1. target-blind E-step refinement and convergence capture;
2. optional head mixing;
3. optional Clebsch-Gordan coupling;
4. optional block normalization;
5. prior handoff to the next layer.

The MLP belongs between steps 4 and 5. Consequently, `capture["converged"]` remains the pre-augmentation VFE belief, while decode and later-layer prior handoff consume the augmented state.

Every layer owns a distinct MLP in an `nn.ModuleList`; weights are not shared across depth.

## Moment contract

For post-block mean `mu`, the active coordinate MLP computes

```text
hidden = activation(W1(mu) + b1)
delta  = dropout(W2(hidden) + b2)
mu_out = mu + delta
```

The hidden width is `block_mlp_expansion * embed_dim`. The operation is position-wise over the last dimension and therefore accepts both batched and unbatched token axes supported by the current model.

The covariance is passed through unchanged for both diagonal and full-Gaussian families. This is a mean-stream structural edit, not a nonlinear Gaussian pushforward or delta-method covariance approximation. All other `BeliefState` fields are preserved.

The dense coordinate maps and pointwise activation do not intertwine the active untied `GL(d_head)^H` action. An enabled MLP must therefore be reported as a non-gauge-pure structural augmentation.

## Configuration

Add these `VFE3Config` fields:

```python
use_block_mlp: bool = False
block_mlp_expansion: int = 4
block_mlp_activation: str = "gelu"
block_mlp_dropout: float = 0.0
m_block_mlp_lr: Optional[float] = None
```

`m_block_mlp_lr=None` inherits `m_p_mu_lr`. Supported activations are `gelu`, `silu`, and `relu`. Expansion must be a positive exact integer. Dropout must be finite and lie in `[0, 1)`. A supplied MLP learning rate must be finite and nonnegative.

The five fields must be written explicitly in the editable dictionaries in `train_vfe3.py` and `ablation.py`, with `use_block_mlp=False`.

When the MLP is disabled, nondefault MLP-only controls are inert and should use the repository's existing inert-setting notice mechanism.

## Gradient and optimizer contract

`effective_e_step_gradient="detach"` wraps the whole VFE stack in `torch.no_grad()`. An active in-stack MLP would consequently freeze. Configuration construction must reject the active-MLP/detach combination rather than permit silent nontraining. Both `unroll` and `straight_through` remain supported.

All active MLP parameters form one optimizer group with role `mu`, ordinary configured weight decay, and learning rate `m_block_mlp_lr` or inherited `m_p_mu_lr`. The optimizer's exact-coverage assertion must include every MLP parameter exactly once.

## Serialization, reporting, and accounting

Configuration serialization and checkpoint model-state persistence use existing generic dataclass and `state_dict` paths. Active and inactive model topologies are intentionally distinct; incompatible checkpoint topology must fail closed through the existing state/optimizer validation.

Reporting must include:

- the active toggle and MLP hyperparameters;
- resolved MLP learning rate when active;
- a `no_block_mlp` pure-path flag;
- a structural classification identifying the active route as a coordinate, mean-only, covariance-passthrough, nonintertwining augmentation;
- MLP parameters and FLOPs in scaling/accounting outputs.

An opt-in MLP ablation may be registered, but it must not be inserted into the default `SWEEP_ORDER`.

## Verification strategy

Implementation follows red-green TDD. The initial failing tests must establish:

1. default-off configuration and validation;
2. no disabled modules, parameters, or state keys;
3. disabled-path forward parity;
4. one untied module per layer when active;
5. exact placement after block normalization and before handoff;
6. residual mean update with bitwise covariance passthrough;
7. active detach rejection and trainable unroll/straight-through paths;
8. optimizer coverage and resolved learning rate;
9. serialization, checkpoint, reporting, replay, parameter, and FLOP accounting;
10. explicit default-off entries in both click-to-run dictionaries.

CPU tests use `C:\Python314\python.exe` with `CUDA_VISIBLE_DEVICES=-1` and `VFE3_TEST_DEVICE=cpu`. No CUDA claim is required for this device-agnostic module. The pre-implementation focused baseline is 336 tests, 335 passing and one inherited failure in `tests/test_config.py::test_config_model_defaults`: it expects legacy `decode_mode="diagonal"`, while current executable configuration defaults to `"diagonal_chunked"`.

## Deferred alternatives

The following are out of scope for v1:

- Jacobian covariance propagation `J Sigma J^T`, which is only a local delta-method approximation and is materially more expensive;
- canonical-frame MLPs, which require an authoritative realized token-frame object across every transport/reflection/RoPE route;
- invariant scalar gating or representation-typed equivariant MLPs;
- a learned unary potential integrated into the VFE objective and every oracle/diagnostic path;
- redesign of attention-side and FFN-side normalization into a fully conventional Pre-LN or Post-LN Transformer block.

These remain separate research extensions rather than hidden behavior under the first comparison toggle.
