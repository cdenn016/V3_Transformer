# GL(K) Attention Code Audit - 2026-05-30

## Scope

This audit covers the executable Python code under `vfe3/` and `tests/` against the theory claims in `Manuscripts-Theory/GL(K)_attention.tex`. I ignored config defaults as evidence of intended behavior. A finding is included only when a selected or advertised path is missing, unreachable, or mathematically different from the GL(K) attention path it purports to implement.

This was a local single-agent audit (the codex GL(K) pass). It is the task input that the 2026-05-31 multi-agent audit verifies and fixes; see `docs/audits/audit-2026-05-31.md` for the resolutions.

## Theory Anchors

The manuscript defines gauge action on Gaussian beliefs as `(mu, Sigma) -> (Omega mu, Omega Sigma Omega.T)` and states KL invariance for invertible `Omega` in `Manuscripts-Theory/GL(K)_attention.tex:441` and `Manuscripts-Theory/GL(K)_attention.tex:522`. It derives `E_ij = KL(q_i || Omega_ij q_j)`, the entropy-regularized row objective, and the softmax stationary point in `Manuscripts-Theory/GL(K)_attention.tex:697`, `:721`, and `:752`. It then gives the temperature-scaled canonical alignment term and reduced free energy `-tau log Z_i` in `:766` and `:843`. The language-model algorithm presents per-head divergences and per-head attention weights in `:2060`.

## Findings (all CONFIRMED and fixed on 2026-05-31)

### 1. Multi-head GL(K) attention collapses to a single shared beta (high)

The manuscript algorithm computes per-head divergences `D_ij^(h)` and per-head weights `beta_ij^(h)` (`:2062`-`:2064`). The implementation built block generators (`irrep_dims = [d_head] * n_heads`) but the attention energy had no head axis: `pairwise_energy` returned one `(..., N, N)` energy and `attention_weights` one `(..., N, N)` beta, so toggling `n_heads` changed the block structure of `Omega` but produced a single attention distribution, not per-head GL(K) attention.

### 2. The full-covariance pure path is not runnable through the model or E-step (high)

The GL(K) invariance theorem requires the full covariance push-forward `Omega Sigma Omega.T`. The integrated path was diagonal-only: the prior bank created diagonal variance parameters, the E-step called `retract_spd_diagonal` unconditionally, and `decode_mode="full"` raised `NotImplementedError`. The exact GL(K) covariance path existed only as isolated kernels and tests.

### 3. `gauge_parameterization="omega_direct"` is a dead toggle in the model path (high)

`omega_direct` was validated but never dispatched: `compute_transport_operators_direct` was never called by the model/E-step, which always used the phi/exp transport.

### 4. `divergence_family` is validated but unused by the executable model path (medium)

`divergence_family` was validated in config but only `cfg.family` was threaded into the E-step. A user toggling `divergence_family` silently kept using `cfg.family` (a dead seam).

### 5. The phi mass term is an M-step loss regularizer, not part of the phi E-step (medium)

The manuscript includes the gauge-frame penalty inside the phi gradient objective (`:2077`-`:2079`). `mass_phi` was added only to the outer training loss, not to `phi_alignment_loss`, so the phi E-step did not descend the penalized objective when `e_phi_lr > 0`.

### 6. The filtered free-energy diagnostic freezes query frames when `keys` is supplied (low)

`free_energy_value(..., keys=...)` built the entire `Omega_ij` from `keys.phi`, freezing both source and query frames. A filtered objective with current query frames and frozen key beliefs could not be represented.

## Note

Resolutions, verifier verdicts, and the temperature reconciliation are recorded in `docs/audits/audit-2026-05-31.md`.
