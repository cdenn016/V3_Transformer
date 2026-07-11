# Omega-Tilde Model-Channel Frame Design

Date: 2026-07-11

Status: implemented and verified on 2026-07-11

## 1. Decision

The transformer will gain an opt-in model-channel frame mode named `phi_tilde`. In that mode the
model distributions $s_i$ are transported by a separately learned vertex cocycle

$$
\widetilde{\Omega}_{ij}
= \exp(\widetilde{\phi}_i)\exp(-\widetilde{\phi}_j),
$$

while the belief channel continues to use

$$
\Omega_{ij}=\exp(\phi_i)\exp(-\phi_j).
$$

The shipped behavior remains `s_frame_mode="tied"`. The tied mode creates no new parameters,
draws no additional random numbers, performs no additional transport construction, and retains the
current state-dictionary and forward-value contract. The new mode is exposed but disabled in both
click-to-run entry points, `train_vfe3.py` and `ablation.py`.

The implementation is restricted to the exponential `phi` parameterization. It does not touch,
extend, or route through `omega_direct`.

## 2. Mathematical scope

The canonical model-channel energy remains

$$
\mathcal{F}_{s}
= \lambda_h\sum_i \operatorname{KL}(s_i\|r_i)
+ \sum_{ij}\left[
\gamma_{ij}\operatorname{KL}
\left(s_i\|\widetilde{\Omega}_{ij}s_j\right)
+ \tau_\gamma\gamma_{ij}
\log\frac{\gamma_{ij}}{\pi^{(s)}_{ij}}
\right],
$$

with $\gamma_i$ a row-normalized variational distribution and
$\tau_\gamma=\kappa_\gamma\sqrt{K_m}$. This build keeps $K_m=K_q=K$ and reuses the existing
registered gauge group, generators, Gaussian transport kernels, entropy-retaining reduced free
energy, and model E-step. It changes only the frame that supplies
$\widetilde{\Omega}_{ij}$.

The phrase `phi_tilde` means parameter independence, not an independently acting principal bundle.
The current transformer identifies the model and state coordinate types through
$q_i^{(0)}=p_i=s_i^{(1)}$. If the two fibers carried independent local gauge actions $g_i$ and
$h_i$, this identity handoff would not be covariant without a typed bridge

$$
C_i' = g_i C_i h_i^{-1},
\qquad
q_i^{(0)}=p_i=(C_i)_\#s_i^{(1)}.
$$

No such bridge exists in the transformer or in this design. Both frame fields therefore remain in
the same $K$-dimensional representation and transform under the same declared coordinate action,
while their learned vertex values and optimizer histories are separate. A product-gauge
implementation with a learned cross-fiber bridge is a different feature and is out of scope.

The model cocycle is flat by construction:

$$
\widetilde{\Omega}_{ij}\widetilde{\Omega}_{jk}
=\widetilde{\Omega}_{ik}.
$$

This identity does not prevent transported consensus or collapse. It certifies only the Regime-I
cocycle. The diagonal Gaussian model family also remains a projected approximation under general
congruence, exactly as on the existing tied path.

## 3. Configuration contract

`VFE3Config` will expose the following fields.

| Field | Default | Contract |
|---|---:|---|
| `s_frame_mode` | `"tied"` | Registry key: `"tied"` or `"phi_tilde"`. The default is the existing shared-frame path. |
| `m_s_phi_lr` | `0.015` | M-step learning rate for model-frame parameters. It is inert when `s_frame_mode="tied"`. |
| `share_refine_s_transport` | existing default | Must be `False` under `phi_tilde`, because the model and belief transports have distinct parameter graphs. |

The click-run dictionaries will add `s_frame_mode="tied"` and `m_s_phi_lr=0.016`. This matches the
belief-frame rate in `train_vfe3.py`; `ablation.py` retains its pre-existing `m_phi_lr=0.010`
unchanged and exposes the model-frame clock separately. The separate field permits later controlled
sweeps. `ablation.py` will expose a two-cell `s_frame_mode` arm and an `m_s_phi_lr` sweep, but neither
will be added to the active `SWEEP_ORDER` list by this feature.

`phi_tilde` requires `gauge_parameterization="phi"`, an active `s_e_step=True` model channel,
`prior_source="model_channel"`, and `share_refine_s_transport=False`. It rejects
`phi_reflection!="off"` and `pos_rotation!="none"` in this first build. Those features contribute
additional frame factors; silently sharing them would create a hybrid frame, while independently
owning their discrete or rotary states is a separate design. The existing learned, frozen, and
absent `pos_phi` modes are supported as specified below.

An effective detached or straight-through E-step severs the only training path from cross-entropy
through `_refine_s` into the model frame. Configuration validation will warn when `phi_tilde` is
combined with a severing estimator or a zero `m_s_phi_lr`; it will not invent a direct scored-gamma
gradient to compensate.

## 4. State and initialization

`PriorBank` will create `s_phi_embed` only when `s_frame_mode="phi_tilde"`. Its shape is
`(vocab_size, n_gen)`, matching `phi_embed`. Initialization is a detached clone of `phi_embed`, not
zero. This creates independent storage without drawing random numbers and makes the token-frame
values equal at construction.

When `pos_phi="learned"`, `VFEModel` will create `s_pos_phi_free` as an independently owned detached
clone of `pos_phi_free`. The model frame applies the same registered positional composition mode,
BCH order, scale, and trace projection as the belief frame, but reads the model-owned positional
table. When `pos_phi="none"`, neither channel adds a positional frame. When `pos_phi="frozen"`, both
channels apply the same deterministic positional rule; no learned state is shared because no learned
positional table exists.

Cloning both learned components gives exact initial equality of the effective frames under the same
composition path. It does not tie their gradients or optimizer state. Zero-initializing an absolute
`s_phi_embed` would instead produce identity or positional-only model transport and would not be a
nested control, because the existing belief token frame is nonzero.

The model-frame parameters use the existing `phi_weight_decay` policy. On the geometric natural
gradient path they receive the same zero-decay treatment as the belief frame. No `mass_s_phi` term is
added: a chart norm is gauge-breaking regularization and was not requested.

## 5. Registry and component boundaries

A focused model-frame registry will live in `vfe3/model/model_frame.py`. Its records select how the
effective model frame is obtained without teaching gamma, diagnostics, or the E-step about storage.
The `tied` builder returns the already composed belief frame. The `phi_tilde` builder looks up
`s_phi_embed`, composes the configured positional model frame, and returns the resulting
`\widetilde{\phi}` tensor.

`PriorBank.encode_s` will retain its two-value `(s_mu, s_sigma)` API. A dedicated
`PriorBank.s_phi(token_ids)` lookup will expose the model token frame. This avoids widening every
model-state caller and keeps frame construction explicit at the orchestration layer.

The generic `BeliefState` and `e_step` APIs need no new field. `_refine_s` already uses a temporary
`BeliefState` as a single-channel carrier; it will receive the chosen model frame in the existing
`phi` slot. The transport and Gaussian kernels therefore remain unchanged.

## 6. Executable data flow

`forward_beliefs` will encode and position-compose the belief frame as it does now, then resolve the
model frame once through the registry. In tied mode the existing shared transport optimization
remains available. In `phi_tilde` mode the model transport is built separately from
`\widetilde{\phi}` and may be hoisted across the model E-step, but it is never reused by the belief
stack.

The explicit model frame or prebuilt model transport will be threaded through `_refine_s`,
`_gamma_energy`, `_fold_gamma_prior`, `_gamma_coupling_term`, split gamma diagnostics,
`gamma_attention_maps`, diagnostic snapshots, and visualization replay. `_gamma_energy` will not
perform a hidden live parameter lookup. Frame-explicit calls preserve snapshot immutability,
gradient boundaries, and exact replay semantics.

The belief stack continues to consume `beliefs.phi` and its own transport. Perturbing
`s_phi_embed` must change gamma energy and refined $s$ while leaving the belief transport unchanged.
Perturbing `phi_embed` after holding the explicit model frame fixed must change the belief transport
without changing the model transport.

## 7. Gradient and objective boundaries

With `s_e_step=True`, cross-entropy trains `s_phi_embed` and `s_pos_phi_free` only through the
attached `_refine_s` trajectory. The model E-step continues to realize the hyper-prior and gamma
forces internally, so their scored outer terms remain suppressed to avoid double counting.

The existing `s_e_step=False` scored gamma block deliberately detaches the frame so its gradient
trains the model Gaussian tables but not the gauge frame. The new explicit frame threading will
preserve that rule. It will not silently add a direct gamma-to-`s_phi_embed` gradient merely because
the frame moved into its own table.

`build_optimizer` will create a distinct model-frame parameter group at `m_s_phi_lr`. The group will
contain `s_phi_embed` and, when present, `s_pos_phi_free`; it will carry the existing gauge optimizer
metadata and the `phi` diagnostic role. Exact parameter coverage remains mandatory. Training logs
and the initialization banner will report `m_s_phi_lr` separately so the new clock is observable.

## 8. Diagnostics and checkpoint behavior

`DiagnosticSnapshot` will retain a detached clone of the effective model frame used by the captured
forward. Snapshot-based gamma and model-channel diagnostics will consume that clone and must not
recompute from later parameter values. Standalone diagnostics will resolve the current model frame
through the registry.

The frame-clamp monitor will inspect both learned token-frame tables and both learned positional
tables where present, reporting the model and belief channels separately. Checkpoint serialization
needs no custom tensor path because the new tensors are ordinary registered parameters. Strict
loading remains strict: a tied checkpoint does not load into a `phi_tilde` model without the
existing configuration-bound migration rules, and a `phi_tilde` checkpoint records the new mode and
learning rate in its embedded configuration.

## 9. Testing contract

The implementation will follow red-green TDD. The first test will prove that tied mode creates no
new parameter, state-dictionary key, random draw, or output change, while `phi_tilde` creates the two
independent tables required by the active positional configuration. A second test will prove copied
token and positional frames produce equal effective transports at construction while occupying
different storage.

The load-bearing behavior test will hold the belief frame and model Gaussians fixed, perturb only
`s_phi_embed`, and require both `_gamma_energy` and `_refine_s` to move. Its converse will hold the
explicit model frame fixed, perturb the belief frame, and require model-channel outputs to remain
unchanged. This is a direct frame-use certificate rather than a covariance-only proxy.

Geometry tests will verify model-frame cocycle closure and model-channel pairwise-KL invariance under
a common shared-coordinate pushforward. They will not claim invariance under two independent local
gauge actions, because this build has no cross-fiber bridge. The diagonal-family projection caveat
will remain explicit.

Optimizer tests will prove exact parameter coverage, the configured `m_s_phi_lr`, a nonzero
`s_phi_embed` gradient through an attached model E-step, and actual parameter motion after an
optimizer step. Configuration tests will pin every rejection and warning, including the
`omega_direct`, shared-transport, reflection, RoPE, inactive-channel, and severed-gradient cases.

Forward and diagnostic tests will cover gamma-prior folding, snapshot stability, standalone replay,
checkpoint round-trip, and both click-run configuration dictionaries. Focused CPU tests will run
first, followed by the complete machine-readable suite. A small `VFE3_TEST_DEVICE=cuda` smoke on the
RTX 5090 will exercise the new frame construction, gamma energy, backward pass, and optimizer step.

## 10. Files and ownership

The configuration surface belongs in `vfe3/config.py`, `train_vfe3.py`, and `ablation.py`. Frame
selection belongs in the new `vfe3/model/model_frame.py`. Token-frame storage belongs in
`vfe3/model/prior_bank.py`; learned positional model-frame storage and orchestration belong in
`vfe3/model/model.py`. Optimizer grouping, monitoring, and learning-rate reporting belong in
`vfe3/train.py`; the duplicate click-run initialization banner in `train_vfe3.py` must report the
same model-frame learning rate. Visualization consumers change only where they currently replay
gamma from the belief frame. Tests will be concentrated in a new
`tests/test_omega_tilde_model_frame.py`, with surgical additions to existing optimizer, snapshot,
configuration, and click-config tests where their public contracts are already pinned.

The daily post-edit record remains `docs/2026-07-11-edits.md`. No Research-vault or manuscript file
will be modified during this build without separate user approval.

## 11. Completion criteria

The feature is complete when `s_frame_mode="tied"` is proven value- and state-compatible with the
pre-feature baseline; `phi_tilde` owns complete learned token and positional frame state, controls
every executable model-channel transport consumer, and trains through the intended attached model
E-step; the belief and model frame graphs are demonstrably separate; all new configuration fields
are visible in both click-run entry points and in inactive ablation arms; focused, full CPU, and CUDA
verification are machine-readable; the dated edit record is current; and the branch completes the
repository's commit, push, merge, and cleanup lifecycle.

No performance improvement, slower emergent timescale, independent product gauge, cross-scale
hyper-prior transport, or manuscript-level empirical claim is part of this implementation result.
