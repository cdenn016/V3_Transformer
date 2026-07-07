# Design: `gauge_parameterization = "omega_direct"` — direct GL(K) group-element frames

Date: 2026-07-07
Status: design, pending approval
Branch: `feat/omega-direct-gl-k`
Scope of first delivery: `glk` and `block_glk`; other groups documented but phased.

## One-line summary

Add a second chart for the per-token gauge frame: instead of storing the Lie-algebra
coordinate `phi_i` and exponentiating it every forward pass (the `"phi"` path), store the
group element `U_i` itself and optimize it on the group manifold. This reaches the full
structure group `G` (both the non-exponential interior of the identity component and the
`det < 0` component for `GL(K)`/`O(K)`), which `phi` structurally cannot, and it removes the
matrix exponential from the forward transport build. The pure `phi`/exp path stays the
default; `omega_direct` is a guarded, default-OFF opt-in.

## 1. Motivation and goal

The transport is the flat cocycle `Omega_ij = U_i U_j^{-1}`, gauge-covariant under a frame
change `U_i -> g_i U_i` as `Omega_ij -> g_i Omega_ij g_j^{-1}`. The `"phi"` chart sets
`U_i = exp(phi_i . G)` with `phi_i` the belief's Lie-algebra coordinate vector
(`belief.py:22`, field `phi: (..., N, n_gen)`), assembled at `transport.py:960-983`.

Two facts make the direct-element chart strictly more expressive than `phi`, on independent
axes:

1. The exponential is not surjective onto the identity component. `exp: gl(K) -> GL+(K)`
   misses elements for `K >= 2` (Culver's criterion). The manuscript's own witness
   `diag(-2, -1/2)` has positive determinant yet no real logarithm
   (`meta_entropy.tex:198-208`). So `{exp(phi)}` is a proper subset of `GL+(K)`.
2. `det exp(M) = e^{tr M} > 0` always. A single exponential, and the two-exp cocycle
   `exp(A) exp(-B)` (still `det > 0`), can never leave `GL+(K)`. The `det < 0` sheet is a
   disconnected component, topologically unreachable by `exp` (`GL(K)_attention.tex:1143`).

A directly stored `U_i` escapes both limits. This is the primary goal: cover the full
`GL(K)` (and, per group, the full structure group).

The secondary goal is compute. Storing `U_i` removes the per-token matrix exponential from the
forward transport build. The honest scope of that win:

- The exp does not disappear from the update; the principled retraction still exponentiates a
  small tangent step (Section 3.3). But that exp is of a well-conditioned near-identity step in
  the update only, not of the full accumulated coordinate every forward pass. The forward hot
  path is genuinely exp-free.
- The forward build trades `exp` for an inverse `U_j^{-1}`. `exp(M)` never fails and is
  perfectly conditioned; `U^{-1}` degrades as `det U -> 0`. The free-energy barrier (Section
  3.4) keeps a trained `U` away from singular, but an fp64 inversion island guards init and
  transients.
- The compute win is concentrated on the non-compact `GL(K)`. For the compact `so_k`/`so_n`
  the `phi` path already gets `exp(-M) = exp(M)^T` for free and perfectly conditioned, so
  `omega_direct` is a conditioning wash-to-loss there; its value on the compact groups is the
  reflection reach, not speed.

Admissibility is not a blocker. The Gaussian congruence `mu -> g mu, Sigma -> g Sigma g^T` is
divergence-invariant for every `g in GL(K)`, reflections included; this is already verified to
machine precision including `det < 0` (`groups.py:241-250`). The entire cost is in sourcing and
optimizing the element, never in the divergence/family layer, which already accepts it.

## 2. What exists today

- `gauge_parameterization` is notional at runtime. It is a config field (`config.py:66`), a
  membership check against `_VALID_GAUGE_PARAM = ("phi", "omega_direct")` (`config.py:22`,
  validated at `config.py:893`), and a hard second-gate `NotImplementedError`
  (`config.py:907-912`). No code reads it to change a computed value.
- The transport funnel `build_belief_transport` (`e_step.py:117`) takes `phi` as its first
  argument and dispatches only on `transport_mode` (the connection regime, orthogonal to this
  axis) through `get_transport` (`transport.py:159`). It has no `gauge_parameterization`
  parameter; the dispatch seam must be built.
- The belief tuple `BeliefState(mu, sigma, phi, s=None, r=None)` (`belief.py:22`) is a
  `NamedTuple` explicitly designed to accept trailing optional fields without a signature
  sweep.
- The belief-gradient machinery (`gradients/oracle.py`, `gradients/kernels.py`) reads the
  transport through its `Omega` / `exp_phi` / `exp_neg_phi` tensors opaquely; it does not read
  `phi` for the belief gradient. So supplying `U` and `U^{-1}` in those slots flows the E-step
  belief update through unchanged. Only the frame's own gradient `dF/dU` and its retraction are
  new.
- The retraction registry (`retraction.py`) is SPD-only for covariance and coordinate-only for
  the frame; `_COMPOSE` (`lie_ops.py`) composes algebra coordinates. There is no
  group-element retraction slot.
- The gauge optimizer `GaugeNaturalGradAdamW` (`gauge_optim.py:127-216`) preconditions a
  coordinate gradient in the identity chart; there is no base-pointed (at `U`) natural gradient.

## 3. Design

### 3.1 The object and the transport

The per-token frame becomes a group element `U_i in G`. The transport is the same flat cocycle,

    Omega_ij = U_i U_j^{-1},

so `omega_direct` changes only the chart on the frame: `"phi"` stores the logarithm
`phi_i in g` and exponentiates; `"omega_direct"` stores the element `U_i in G` and inverts.
Because it is the identical cocycle, it preserves strict gauge equivariance
`Omega_ij -> g_i Omega_ij g_j^{-1}` and vanishing holonomy (flat Regime-I purity) exactly,
unlike the head-mixer / `regime_ii` / `connection_W` exceptions that break equivariance away
from zero-init. `omega_direct` is a pure-path-preserving alternative chart, not a sanctioned
NN exception.

The stored element respects the group's block structure, so its free-parameter count equals
the algebra dimension `n_gen` (there is no storage blowup versus `phi`):

- `glk`: a dense `(..., N, K, K)` element.
- `block_glk`: a block-diagonal element carried as `(..., N, H, d, d)` (H heads of `d x d`).
- compact `so_*`: an orthogonal element, kept orthogonal by the retraction.

### 3.2 Expressivity (restated as the design contract)

The chart must be able to represent any element of `G` at init, and to move continuously
within the identity component `G^0` during optimization. It cannot move between components
(Section 3.4). Concretely the contract is: `omega_direct` at identity init is byte-identical to
a trivial gauge; seeded away from identity it reaches `GL+` interior points and `det < 0`
points that `phi` cannot.

### 3.3 Manifold geometry and the principled optimizer

`G` is a Lie group; `G^0` is its identity component. On `G^0` the principled first-order
optimization is Riemannian gradient descent with a base-pointed metric and a Lie-group
retraction.

- Metric / natural gradient. The Euclidean frame gradient `E = dF/dU` is converted to an
  algebra tangent ` xi in g` under a left-invariant metric induced by the trace inner product
  at the identity (the Killing form on the semisimple part) — the same metric family the
  existing `phi_preconditioner` uses in the identity chart, now left-translated to the base
  point `U`. For the left-invariant trace metric the Riemannian gradient's algebra tangent is
  `xi = Gram^{-1} proj_g(U^T E)` (the translated gradient is `U^T E`, not `U^{-1} E`; they
  coincide only for orthogonal `U`), where `Gram^{-1}` is the (block) Killing/Gram inverse the
  preconditioner already computes and `proj_g` projects onto the group's algebra.
- Retraction (principled default). The Lie-group exponential retraction,

      U <- U exp(-eta . xi),        xi in g,

  follows one-parameter subgroups, stays in `G^0` (`det(U exp xi) = det(U) e^{tr xi}`, same
  sign), and exponentiates only the small near-identity step `eta . xi` in the update. The
  forward pass reads `U` directly with no exp. `GL(K, R)` admits no bi-invariant metric
  (non-compact), so there is no unique geodesic; the Lie-exp retraction is the canonical choice
  among the left-invariant ones.
- Retraction (exp-free opt-in). The Cayley retraction `U <- U cay(eta . xi)`,
  `cay(A) = (I - A/2)^{-1}(I + A/2)`, is a valid second-order retraction on `GL(K)` and is
  exactly structure-preserving on the quadratic groups (`so`, `sp`), where it is the classical
  exp-free parameterization. Offered as a toggle for a strictly-no-exp path; it also stays in
  `G^0`.

Both retractions register into a new group-retraction slot (Section 5). The belief E-step is
unchanged: `U` and `U^{-1}` fill the `exp_phi` / `exp_neg_phi` slots, so `oracle.py` /
`kernels.py` flow the mean/covariance belief gradient through untouched.

### 3.4 The discrete component (det-sign): the one non-geometric piece

For `GL(K)` and `O(K)`, `pi_0(G) = Z/2`: two disconnected sheets separated by the singular set
`det U = 0`. No continuous or Riemannian update can cross it. The obstruction is a free-energy
barrier, not merely topology: as `det U -> 0` the congruence `Sigma -> U Sigma U^T` collapses a
covariance direction, and every divergence reading `Sigma^{-1}` or `log det Sigma` diverges (in
code, clamps to `kl_max`, a flat plateau with vanishing gradient). A numerical demonstration
along `U(t) = diag(1, 1 - 2t)` (from `I` to a reflection) shows `KL(q || U q)` climbing through
`10^2, 10^3, 10^5` and to the `kl_max` cap at `t = 1/2` (`det = 0`), with low energy on both
sides — the two basins are walled off from each other. So the gradient never points across, and
gradient-based VFE descent is confined to whichever sheet a token is initialized in.

The principled decomposition is therefore `U = R . U^0`, with `U^0 in G^0` handled by Section
3.3 and `R` a discrete component representative (a reflection, `det R = -1`) handled
combinatorially:

- Default: init-only. `R` is fixed per token at initialization (all identity by default, with
  optional per-token seeding into `det < 0`). This alone is the element-valued field that
  reaches `det < 0` as a representable set (the "option 2" behavior).
- Opt-in (the "option 3" capability the user selected): a straight-through estimator proposes
  per-token sign flips, accepted through a straight-through gradient (or a Delta-F-gated
  Metropolis flip). This makes the det-sign learnable rather than fixed. This is inherently the
  least geometric piece — it optimizes over the discrete quotient `pi_0(G)`, not a manifold —
  and is documented as such.

The manuscript proposes exactly this straight-through treatment of reflections
(`GL(K)_attention.tex:1146-1157`, `eq:ok_transport`; reflections may alternatively be absorbed
into the mean as sign flips `mu_i -> s_i (.) mu_i`, `eq:ok_mahalanobis`).

### 3.5 Inverse and conditioning

The forward needs `U_j^{-1}`. Per group: `glk` a dense inverse; `block_glk` a per-block `d x d`
inverse; compact `so_*` the transpose `U^T` (free, orthogonality preserved by the retraction).
The inverse runs in an fp64 island (mirroring the existing `transport_covariance` M4 upcast and
`stable_matrix_exp_pair`'s float64 island). The F-barrier keeps trained `U` well-conditioned;
the island guards init and transients. No soft det-floor is imposed (it would bias the frame);
the barrier is the intended regularizer.

### 3.6 Per-group reach (the "other groups" semantics)

- `glk`, `block_glk`: full reach — non-exp interior of `GL+` (continuous) and `det < 0` (init /
  STE). The primary target.
- `tied_block_glk`: same as `block_glk` but one shared element across heads.
- `so_k`, `so_n`: the continuous part is already exp-surjective (compact connected), so the
  element store adds nothing continuous; the new capacity is the reflection into `O(K)` /
  `rho(O(N))`. For the towers, the reachable component is `rho(O(N))`, not the full `O(K)` on
  the embedded space — documented, not silently implied.
- `sp`, `sp_n`: connected with `det ≡ +1`, so reflections are vacuous; the only new capacity is
  the non-exp interior of the identity component. STE is a no-op here.

No group is rejected; each carries a documented reach.

## 4. Config surface

New fields (names provisional, all default-OFF / pure-path-preserving):

- `gauge_parameterization: "phi" | "omega_direct"` — already present; remove the reject.
- `omega_retraction: "lie_exp" | "cayley"` — principled default `"lie_exp"`.
- `omega_init: "identity" | "phi_match" | ...` — `"identity"` gives step-0 parity; `"phi_match"`
  seeds `U = exp(phi_scale . randn)` to match the `phi`-path init distribution.
- `omega_reflection: "off" | "init_seed" | "ste"` — `"off"` (pure `det > 0` element field),
  `"init_seed"` (per-token `det < 0` at init, fixed), `"ste"` (learnable det-sign; the option-3
  capability).
- Validation: keep `_require` against `_VALID_GAUGE_PARAM`; replace the `NotImplementedError`
  with per-group cross-checks (reflection modes are no-ops / rejected for `sp`/`sp_n`; `so_n`
  reflection reaches only `rho(O(N))`).

Default `gauge_parameterization = "phi"` is unchanged, so the pure exp path remains the default
and the mandatory theoretically-pure path is preserved.

## 5. Implementation inventory (ordered checklist)

1. `config.py`: delete the `NotImplementedError` (`:907-912`) and its comment (`:901-906`); add
   the fields in Section 4 and per-group validation beside the existing group gates
   (`:887-892`, `:919+`). Refresh stale entry-point comments (`train_vfe3.py:128`,
   `scaling.py:148`, `ablation.py:139`).
2. `belief.py`: add one trailing `omega: Optional[torch.Tensor] = None` field (sanctioned by
   the tuple's extensible design, `belief.py:1-14`).
3. `prior_bank.py`: add a gated per-token `omega_embed` source table, created only when
   `omega_direct` is active (idiom at `:179-181`, `:191-196`, so the pure `state_dict` stays
   byte-identical). Extend the per-token encode to look up the element and stash it in the
   belief; init per `omega_init` and `omega_reflection`.
4. `transport.py`: a new element build path (registered via `register_transport` at `:119`,
   selected by the threaded `gauge_parameterization`) that fills the `FactoredTransport` slots
   `exp_phi := U_i`, `exp_neg_phi := U_j^{-1}` directly, reusing the assembly einsum (`:982`)
   and every downstream factored / RoPE contraction (`:1152-1278`) unchanged. Per-block inverse
   with an fp64 island; compact groups use the transpose.
5. `e_step.py`: thread `gauge_parameterization` and the `omega` field through
   `build_belief_transport` (`:117`) and `_transport` (`:41-98`), gated on a new
   `_TRANSPORT_NEEDS_OMEGA` set parallel to `_TRANSPORT_NEEDS_MU/_SIGMA` (`:180-183`). The
   continuous `phi` sub-step is untouched; add the element retraction sub-step.
6. `retraction.py` + `lie_ops.py`: register the group-element retractions (`lie_exp`,
   `cayley`) into a new retraction slot; add the left-translated natural-gradient map
   (`U^{-1} E` pullback + Killing/Gram precondition reusing `phi_preconditioner`).
7. `gauge_optim.py`: a matrix-parameter param group whose step is retraction-based (the
   coordinate-only `GaugeNaturalGradAdamW` cannot step an element); the STE discrete update for
   `omega_reflection = "ste"`.
8. Off-funnel threading so `det < 0` is visible at generation / eval: the KV-cache builder
   (`belief_cache.py`), the holonomy diagnostic `_transport` (`model.py`), and `viz/extract.py`,
   which rebuild `Omega` from `phi` today.
9. `tests/`: invert the rejection test (`test_config.py:224-229`); add
   `tests/test_gauge_parameterization.py` — identity-init byte-parity with `phi`; a `det < 0`
   reach property; a non-identity-frame end-to-end gauge-invariance test; the cocycle identity
   for the element transport; a Lie-exp-vs-Cayley agreement-to-second-order check; per-group
   sizing including `sp`/`sp_n` semantics. All CPU tests use `K < 6`, single-digit dims.
10. `ablation.py`: remove `"gauge_parameterization"` from `NON_SWEPT_FIELDS` (`:1043`), rewrite
    the comment (`:1033`), add a categorical sweep arm with a `requires` clause.

## 6. Testing plan

Golden and property tests pin the new path exactly where the `phi` path is pinned. The
load-bearing properties: (a) identity-init parity (`omega_direct` at `U = I` equals the
trivial-gauge transport byte-for-byte); (b) equivalence to `phi` when `U` is seeded as
`exp(phi)` for a `det > 0` frame (to fp32 tolerance, the exp/log round-trip); (c) `det < 0`
reachability yielding a valid `mu -> Omega mu`, `Sigma -> Omega Sigma Omega^T`; (d) strict
gauge equivariance of the full model under a non-identity frame (the current gauge test zeroes
the frame and is vacuous for a transport claim); (e) the retraction stays in the identity
component (det sign preserved) and, for compact groups, preserves orthogonality; (f) the STE
flip changes the component and the sign is learnable. All tiny-`K` CPU tests.

## 7. Risks and open questions

- The STE is the least-principled and least-tested piece. If evidence does not show a learnable
  det-sign matters, ship with `omega_reflection in {"off", "init_seed"}` and defer STE.
- `U^{-1}` conditioning near init: mitigated by the fp64 island and the barrier; watch for
  early-training near-singular frames.
- Freeze footgun: any learnable frame field enters `F` only through `build_belief_transport`
  and inherits the `detach_e_step` freeze family (freezes at init under detached / straight-
  through E-steps). The retraction/STE updates must run on a path that keeps `U` in the graph,
  the same discipline as `connection_W`.
- The natural-gradient metric for `U` reuses the `phi` preconditioner's Killing/Gram machinery;
  confirm the left-translation `U^{-1} E` composes with the per-block preconditioner for
  `block_glk` (the per-block Killing form assumes generators that partition per block).

## 8. Phasing

Phase 1 (this spec): `glk` + `block_glk`, `omega_retraction in {lie_exp, cayley}`,
`omega_reflection in {off, init_seed}` (continuous element field + init-time `det < 0`), full
dispatch / belief / transport / retraction / tests / ablation. This delivers full `GL(K)` as a
representable set with the principled optimizer.

Phase 2: `omega_reflection = "ste"` (learnable det-sign, the option-3 capability), and the
compact / symplectic tower semantics (`so_n`, `sp_n`).
