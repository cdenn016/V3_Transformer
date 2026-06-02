# Positional Encodings for VFE_3.0: gauge-RoPE and BCH positional encoding

Status: approved design (2026-06-02). Author session: positional-encoding buildout.
Guide (not a port source): `C:\Users\chris and christine\Desktop\VFE_2.0`. Theory authority:
`Manuscripts-Theory/GL(K)_attention.tex` for gauge-RoPE (the paragraph "Identification with
rotary positional structure", eq:gauge_qk); `Manuscripts-Theory/GL(K)_supplementary.tex` and
`vfe3/geometry/lie_ops.py::compose_bch` for the BCH composition chart.

## Motivation

VFE_3.0 currently carries position only through the additive attention prior `B_ij` (the
`attention_prior` registry: uniform, causal, ALiBi). That seam injects position as a softmax
log-bias and cannot express rotary or frame-composed positional structure. Two positional
schemes are missing, both natural in the gauge-theoretic framework and both present in VFE_2.0:
a rotary scheme acting on the transport operator (gauge-RoPE) and a Baker-Campbell-Hausdorff
scheme acting on the gauge frame (BCH positional encoding). This document specifies building
both from scratch into VFE_3.0's own seams.

Both schemes are independent, default-off toggles, so the pure no-positional path (causal mask
only) remains the default and always exists, per the project's pure-path constraint.

## Theory grounding

The manuscript identifies the per-token frame `U_i` of the gauge transport with the rotary
positional frame: `U_i in O(d_k)` is a block-diagonal rotation depending on token position, so
`Omega_ij = U_i U_j^{-1} = U_i U_j^T`, and the mean carving reduces to `Q_i = U_i^T mu_i`,
`K_j = U_j^T mu_j` (eq:gauge_qk). The cross term `x_i^T k_j` is the only coupling between query
and key through the belief means, which is why a means-only rotary scheme already captures the
central rotary effect. For an orthogonal frame the covariance closure `Sigma_j = U_j C U_j^T`
holds automatically for any `C` commuting with the block-rotation structure; the means-only path
therefore leaves the covariance transport untouched, while a complete means-and-covariance
variant rotates the covariance sandwich as well and (because the diagonal covariance
approximation cannot carry the intra-block off-diagonal correlations the rotation generates)
requires full covariance.

BCH positional encoding has no dedicated manuscript section; its primitive is the BCH chart
correction already implemented as `compose_bch` (the symmetric Dynkin series, order knob), which
the supplementary backs as the composition chart on the Lie algebra. The scheme itself, a learned
per-position Lie-algebra element composed into the frame, follows VFE_2.0 as the guide.

## Seam A: gauge-RoPE (transport-operator rotation)

### A.1 The rotation builder

A new module `vfe3/geometry/rope.py` holds a registry (`register_pos_rotation` /
`get_pos_rotation`) keyed by `pos_rotation` config, with `"none"` and `"rope"` registered. The
`"rope"` builder produces a per-position block-diagonal rotation on the group's `irrep_dims`:

```
build_rope_rotation(
    positions:   torch.Tensor,   # (N,) integer token positions
    irrep_dims:  List[int],      # block sizes; sum == K
    *,
    base:        float = 100.0,
    device, dtype,
) -> torch.Tensor                # (N, K, K) block-diagonal orthogonal rotation
```

Within each block of size `d` at offset `s`, coordinate pairs `(s+2k, s+2k+1)` for
`k = 0..floor(d/2)-1` rotate by angle `theta_{n,k} = n * base^{-2k/d}` (block-local frequency,
using the block dimension `d` so each head-block carries its own rotary spectrum). A leftover odd
coordinate stays identity. The block-diagonal structure is mandatory: it keeps `R` orthogonal
(`R R^T = I`) and preserves block-diagonality so the `_can_fuse_flat` fast path is not destroyed.
`R` carries no learned parameters.

The combined frame is `U_i = R(theta_i) exp(phi_i)`, hence
`Omega_ij^RoPE = R(theta_i) exp(phi_i) exp(-phi_j) R(theta_j)^T`. At the trivial gauge
(`phi = 0`, `Omega = I`) this is `R(theta_i - theta_j)`, giving the relative-position property.
Once the learned gauge is non-trivial the dependence is on absolute `i, j` (the attention-gauge
versus value-gauge factorization); the spec states this rather than implying clean relativity.
gauge-RoPE breaks gauge equivariance by construction (an external fixed rotation, not conjugated
by the gauge transform); this is accepted and documented, matching VFE_2.0.

### A.2 Means-only versus means-and-covariance

Two regimes, selected by `rope_full_gauge`:

Means-only (`rope_full_gauge = False`, the default when RoPE is on) applies `R` to the mean
transport only. `transport_mean` computes `mu_t[i,j] = R(theta_i) exp(phi_i) exp(-phi_j)
R(theta_j)^T mu_j`; `transport_covariance` uses the un-rotated `Omega`. Compatible with
`diagonal_covariance = True` (the active run's setting).

Means-and-covariance (`rope_full_gauge = True`) feeds the rotated factors to both the mean and
the covariance sandwich (`Sigma_t[i,j] = Omega^RoPE Sigma_j (Omega^RoPE)^T`). Because the
rotation generates intra-block off-diagonal covariance that the diagonal approximation cannot
propagate, this regime is gated to `diagonal_covariance = False` (config validation raises
otherwise), mirroring VFE_2.0's `rope_full_gauge` constraint.

### A.3 Mechanism (how R reaches the transport)

`R` is folded into the transport factors. `FactoredTransport` gains one optional field
`rope: Optional[torch.Tensor]` carrying the `(N, K, K)` rotation that applies to means only.

Means-only: the factored `exp_phi` / `exp_neg_phi` stay plain; `rope` holds `R`. `transport_mean`,
when `rope` is set, applies `R(theta_j)^T` to `mu_j` before `exp(-phi_j)` and `R(theta_i)` after
`exp(phi_i)`, staying on the fused fast path. `transport_covariance` ignores `rope` and uses the
plain factors, so the covariance sandwich is numerically identical to the no-RoPE build.

Means-and-covariance: `R` is pre-folded into the factors (`exp_phi[i] <- R(theta_i) exp(phi_i)`,
`exp_neg_phi[j] <- exp(-phi_j) R(theta_j)^T`) and `rope` is left `None`; both `transport_mean` and
`transport_covariance` then see the rotated factors automatically. The dense (non-fused) path
builds `Omega^RoPE` from the rotated factors the same way.

### A.4 Integration surface

`R` must reach every transport consumer, or diagnostics and the gradient oracle will disagree with
the forward. The model precomputes and caches `R` on `(N, device, dtype)`, exactly like
`_attention_log_prior`. The rotation threads through the transport chokepoints:
`vfe3/inference/e_step.py::_transport`, `build_belief_transport`, `_transport_qk`; the
FD/autograd oracle inside `belief_gradients` (so finite-difference gradient checks pass);
`vfe3/model/model.py::diagnostics`; and `vfe3/model/model.py::attention_maps`. The builders
already accept `**kwargs`, so the rotation flows as an optional keyword without breaking call
shapes.

## Seam B: BCH positional encoding (gauge-frame composition)

### B.1 The pos-phi seam

A new module `vfe3/model/positional_phi.py` holds a registry (`register_pos_phi` /
`get_pos_phi`) keyed by `pos_phi` config, with `"none"`, `"learned"`, and `"frozen"` registered.
Each returns the per-position Lie-algebra coordinates `pos_phi_coords: (N, n_gen)`:

`"none"` returns `None` (no composition; the frame is unchanged and the model is byte-identical to
the no-PE build). `"learned"` slices a raw `(max_seq_len, n_gen)` parameter `pos_phi_free` (held on
the model, initialized at scale `pos_phi_scale`) to the first `N` rows. `"frozen"` returns
`i * direction` for a fixed unit `direction` in `n_gen` (the first generator coordinate by default),
a Lie-algebra analogue of ALiBi with zero learned parameters.

`pos_phi_free` is a raw `nn.Parameter` (like `connection_W` and `log_alpha`), not a neural network,
so the no-NN constraint holds; it is opt-in and default-off, so the pure path is preserved.

### B.2 Composition into the frame

After `prior_bank.encode` produces `belief.phi`, and before the block loop, the frame is composed
with the positional element:

```
phi_i <- compose_phi(phi_token_i, pos_phi_i, generators,
                     mode = pos_phi_compose,   # "bch" (default) or "euclidean"
                     order = bch_pe_order)
```

`mode = "bch"` is the point of the scheme; `"euclidean"` (additive) is available as an O(1)
ablation that is exact only when `[phi_token, pos_phi] = 0`. With `pos_phi_project_slk = True` the
positional element (or the composed frame) is projected to `sl(K)` per block via
`project_phi_to_slk`, preserving `det(Omega_h) = 1`.

The composition modifies the off-diagonal transport `Omega_ij` symmetrically; the self-transport
`Omega_ii = exp(phi_i) exp(-phi_i) = I` is unaffected regardless of `pos_phi` (correct and
expected). The positional element enters transport only; the self-coupling `KL(q_i || p_i)` reads
the prior `p_i`, which is untouched.

### B.3 Integration surface

One call site in `VFEModel.forward`, mirrored in `diagnostics` and `attention_maps` (which replay
the same frame handoff). The learned `pos_phi_free` parameter is added to the optimizer parameter
groups (`build_optimizer`) so the M-step trains it by backprop on the free energy through
transport.

## Configuration

New fields on `VFE3Config`, all default-off so the pure path is the default:

```
pos_rotation:         str   = "none"     # "none" | "rope"
rope_base:            float = 100.0      # rotary frequency base
rope_full_gauge:      bool  = False      # False: means-only; True: means+covariance (needs full cov)

pos_phi:              str   = "none"     # "none" | "learned" | "frozen"
pos_phi_compose:      str   = "bch"      # "bch" | "euclidean"
bch_pe_order:         int   = 4          # BCH Dynkin truncation order
pos_phi_scale:        float = 0.02       # init scale for the learned pos_phi_free
pos_phi_project_slk:  bool  = False      # per-block trace projection (det preservation)
```

Validation: `pos_rotation` against the rope registry; `pos_phi` against the pos-phi registry;
`pos_phi_compose` against the compose registry; and `rope_full_gauge = True` requires
`diagonal_covariance = False` (raise with a clear message otherwise).

## Testing (property-based, not VFE_2.0 byte-parity)

The project's golden byte-equivalence discipline is against the pinned VFE_2.0 checkout and does
not apply here (the user's instruction is to build from scratch against different seams). Tests are
property-based.

gauge-RoPE (`tests/test_rope.py`):
- `R` is block-diagonal on `irrep_dims` and orthogonal (`R R^T = I`).
- `R` preserves the `_can_fuse_flat` precondition (block-diagonal with equal blocks stays so).
- Relative-position: at `phi = 0` (`Omega = I`) the attention logits over a constant-mean sequence
  are Toeplitz (depend only on `i - j`).
- Means-only leaves the covariance transport unchanged: `transport_covariance` is identical with
  and without RoPE when `rope_full_gauge = False`.
- FD gradient check of `F` through the means-only rope transport.
- A second FD oracle for the means-and-covariance (full-gauge) path under `diagonal_covariance =
  False`.
- Config validation: `rope_full_gauge = True` with `diagonal_covariance = True` raises.

BCH positional encoding (`tests/test_positional_phi.py`):
- `"none"` is identity: the model forward is byte-identical to the no-PE build.
- `"learned"`: `pos_phi_free` is a registered parameter, appears in the optimizer parameter groups,
  and `dF/dpos_phi` is non-zero (autograd or FD).
- `"frozen"`: deterministic, parameter-free, position-dependent.
- BCH versus euclidean composition differ when `[phi_token, pos_phi] != 0`, and agree when the
  bracket vanishes.
- `pos_phi_project_slk = True` yields per-block traceless composed frames (`det Omega_h = 1`).
- `Omega_ii = I` regardless of `pos_phi`.
- `pos_phi` does not change the self-coupling `KL(q_i || p_i)`.

Config-validation tests live alongside the existing config tests.

## Files

New: `vfe3/geometry/rope.py`, `vfe3/model/positional_phi.py`, `tests/test_rope.py`,
`tests/test_positional_phi.py`.

Modified: `vfe3/config.py` (fields and validation), `vfe3/inference/e_step.py` (thread `R` through
`_transport` / `build_belief_transport` / `_transport_qk` and the oracle),
`vfe3/geometry/transport.py` (`FactoredTransport.rope` field; `transport_mean` /
`transport_covariance` rope handling; rope-folded dense path),
`vfe3/model/model.py` (apply pos-phi after encode; precompute and cache `R`; thread both through
`diagnostics` and `attention_maps`; hold `pos_phi_free`), optimizer wiring in `vfe3/train.py`, and
the config-validation tests.

## Out of scope

Learned rotary frequencies, per-block rotary base overrides, a rotary attention-prior bias
variant, and any Regime-II-specific RoPE interaction beyond the orthogonal-composition default are
out of scope for this build.
