# Regime-II Non-Flat (Edge-Relaxed) Connection as a `register_transport` Variant — Design Spec

Status: seam refactor is **buildable-once-decided**; the δ-parameterization physics is **spec-only, needs a user decision** (no oracle for which connection is "right", and the *useful* forms require a learned parameter that is not yet a blessed no-NN exception). Date 2026-06-01. Author: overnight design agent.

## 1. Motivation

The spec designs the connection regime as a registry-backed modular axis on equal footing with the structure group: "two orthogonal, registry-backed modular axes: the structure group and the connection regime ... config-selected so new variants are added by writing-and-registering, never by editing call sites" (`docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md:54`), and names the exact builder: "Regime II is the edge-relaxed cocycle `Omega_ij = exp(phi_i) exp(delta_ij . G) exp(-phi_j)` with `delta_ij` an edge-local connection (non-trivial holonomy, Yang-Mills field strength), recovering Regime I at `delta_ij = 0`. The regime is a toggle" (`spec:60`). The manuscript formalizes the discrete realization at `Participatory_it_from_bit.tex:837-844` (`eq:edge_relaxed_omega`), the two-regime statement at `:806`, the homotopy at `:876-880` (`cocycle_relaxation`), the Wilson observable and penalty at `:862-873`, and the two candidate δ parameterizations at `:884` (free per-edge vs the learnable bilinear `delta_ij^a = mu_i^T W^a mu_j`). The roadmap ranks this punch-list item 7 (`docs/2026-06-01-buildout-roadmap.md:50,114`): it is "the sole geometry axis the spec promises as registry-backed that has no registry", and the already-shipped `holonomy_deviation` diagnostic (`vfe3/metrics.py:50-91`) is identically ~0 for the flat cocycle and exists precisely to exercise the Regime-II transport that was never built. Today `transport.py` exposes only the two flat (Regime I) cocycle builders (`compute_transport_operators` at `transport.py:114`, `compute_transport_operators_direct` at `transport.py:156`), and the E-step imports `compute_transport_operators` directly (`inference/e_step.py:24,39,51`) with no `register_transport`/`get_transport` indirection.

## 2. The central decision: how is δ parameterized in a no-NN belief

This is the load-bearing decision and it resolves to a purity dichotomy, not a smooth expressivity tradeoff. Working CLAUDE.md's no-NN constraint against each candidate:

**(a) δ ≡ 0 — the flat cocycle.** This is Regime I exactly. Pure, always available, and does nothing (`holonomy_deviation` stays ~0). It is the default registered entry and the preserved pure path.

**(b) δ as an E-step variational field** (a per-forward tensor over edges, initialized at 0, optimized by the same natural-gradient inner loop as φ). This is the *only* genuinely no-NN non-flat path: like φ, it is a belief-side latent with no learned weights. It is also functionally degenerate for the default causal LM, and the spec must say why. A free per-edge `delta_ij` is an independent group element on each edge, so the belief-coupling block `sum_ij beta_ij D(q_i || Omega_ij q_j)` can drive each `D(q_i || exp(phi_i) exp(delta_ij.G) exp(-phi_j) q_j) -> 0` independently by choosing `delta_ij` to align the transported `q_j` onto `q_i`. The only thing restraining that collapse is the Wilson closed-loop penalty (`Participatory:866-873`), which is degenerate on the causal DAG (Section 5). So the pure variational path is theoretically clean but, on the default causal stream, collapses belief-coupling toward uniform attention and learns nothing — a genuine, documentable limitation, not a bug.

**(c) Bilinear `delta_ij^a = mu_i^T W^a mu_j`, an FFN on concatenated means, or a learned per-edge/per-scalar table.** All carry a *learned weight* (`W^a`, the FFN, or the table entries) trained by backprop on CE. None is in CLAUDE.md's enumerated exceptions (the `use_prior_bank=False` linear decode and `use_head_mixer`). The weight-sharing is exactly what makes them useful — it stops the per-edge collapse of (b) and lets cross-entropy train δ for a real predictive job — and it is exactly what places them outside the pure path. The bilinear is the leanest: `W` is a raw `(n_gen, K, K)` `nn.Parameter` consumed by one einsum, no activation, no bias, directly analogous to the already-blessed linear-decode `W` (a raw `(V, K)` `nn.Parameter`, not an `nn.Linear`). The manuscript names the bilinear and the FFN as the two implementation choices (`Participatory:884`) and notes both "see the inputs in the embedding ambient frame rather than in a per-vertex rotated frame," which is the fixed-internal-frame reading consistent with the gauge-covariance law below.

**Honest headline for the user:** a *useful* Regime-II connection requires a learned parameter that is not currently a blessed no-NN exception. The pure no-NN non-flat option (b) exists but is functionally vacuous on the causal default. This is the decision only the user can make (Section 8).

## 3. Interface / architecture (no call-site edits)

### 3.1 New registry in `vfe3/geometry/transport.py`

Mirror `register_group`/`get_group` (`groups.py:53-70`) and `register_alpha` (`alpha_i.py:19-39`). Add to `transport.py`:

```python
_TRANSPORTS: Dict[str, Callable[..., TransportDict]] = {}

def register_transport(name: str) -> Callable:
    """Decorator registering a transport builder under ``name``."""
    def _wrap(fn: Callable[..., TransportDict]) -> Callable[..., TransportDict]:
        _TRANSPORTS[name] = fn
        return fn
    return _wrap

def get_transport(name: str) -> Callable[..., TransportDict]:
    """Return the registered transport builder for ``name`` (KeyError if absent)."""
    if name not in _TRANSPORTS:
        raise KeyError(f"no transport registered under {name!r}; available: {sorted(_TRANSPORTS)}")
    return _TRANSPORTS[name]
```

### 3.2 Uniform builder signature (the wrinkle to settle)

The flat builder is a pure function of φ (`transport.py:114`). A non-flat builder is stateful: form (b) needs a variational δ channel, form (c) needs a model-owned `W`. So `register_transport` cannot use `register_group`'s pure-builder shape. The uniform signature carries the extras as keyword-only, so the flat entry ignores them and the call sites never change:

```python
def build(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    group:      GaugeGroup,

    *,
    gauge_mode: str                    = "learned",   # 'learned' | 'trivial' (unchanged)
    mu:         Optional[torch.Tensor] = None,         # (B, N, K) means; bilinear δ reads these
    connection: Optional[torch.Tensor] = None,         # (B, N, N, n_gen) precomputed δ_ij (variational form)
    **kw,
) -> TransportDict:
    ...
```

The two existing flat builders are wrapped in registered adapters that ignore `mu`/`connection`:

```python
@register_transport("flat")           # Regime I, the default and pure path
def _build_flat(phi, group, *, gauge_mode="learned", **kw):
    return compute_transport_operators(phi, group, gauge_mode=gauge_mode)

@register_transport("regime_ii")       # edge-relaxed cocycle (Section 4)
def _build_regime_ii(phi, group, *, gauge_mode="learned", mu=None, connection=None,
                     cocycle_relaxation=1.0, **kw):
    ...
```

`compute_transport_operators` and `compute_transport_operators_direct` stay exactly as they are (the `flat` adapter delegates to the first; `omega_direct` is config-rejected today at `config.py:156-158` so it needs no transport-registry entry until that path is built).

### 3.3 Routing through the seam

`inference/e_step.py:_transport` (`:28-40`) and `_transport_qk` (`:43-53`) are the two live consumers in the inner loop; `model/model.py:231` is the diagnostics consumer. The minimal, call-site-stable route is to make `_transport`/`_transport_qk` look up `get_transport(transport_mode)` instead of calling `compute_transport_operators` by name, threading `transport_mode` (and `mu`, `connection`) down from `e_step_iteration` / `phi_alignment_loss` the same way `family`, `divergence_family`, `phi_precond_mode` already thread (`e_step.py:181-185`). The default `transport_mode="flat"` makes this a behavior-preserving refactor: the flat adapter delegates to the identical `compute_transport_operators`, so `test_transport_matches_frozen_oracle` (`tests/test_perf_equivalence.py:77`) and `test_phi_path_cocycle_identity` (`tests/test_transport.py:107`) stay green untouched.

### 3.4 New config fields (`vfe3/config.py`)

```python
_VALID_TRANSPORT = ("flat", "regime_ii")
transport_mode:          str   = "flat"            # connection regime
connection_param:        str   = "bilinear"        # 'variational' | 'bilinear' | 'free_edge'  (regime_ii only)
cocycle_relaxation:      float = 1.0               # homotopy α ∈ [0,1]; 0 == flat, 1 == fully relaxed
holonomy_penalty_weight: float = 0.0               # Wilson regularizer coefficient (Section 6); 0 == off
```

Validation, in `Config.__post_init__` (`config.py:148`): `_require(self.transport_mode, _VALID_TRANSPORT, "transport_mode")`. If `transport_mode != "flat"` and `connection_param == "bilinear"` (or `"free_edge"` table), require an explicit opt-in flag (the third blessed NN-exception, Section 8) — refuse construction otherwise, mirroring how `omega_direct` is rejected (`config.py:156`). For the default causal `attention_prior`, emit a config-time note that the Wilson penalty and belief-level holonomy are degenerate on the DAG (Section 5) and `holonomy_penalty_weight>0` will not constrain the active connection.

### 3.5 Where the learned weight lives (form (c))

`W: (n_gen, K, K)` is an `nn.Parameter` registered on `VFEModel` (next to the head-mixer's `mixer_delta`, `head_mixer.py:70`), gated by the opt-in flag, initialized at zero so `delta_ij ≡ 0` at step 0 and the model is bit-flat (Regime I) initially. It is passed through the forward into `e_step`'s `connection` slot already-evaluated, so the transport builder stays a pure tensor function and `W`'s gradient flows through the einsum. For form (b) the variational δ is a new per-forward latent — which collides with roadmap finding M3 (`BeliefState` is a rigid positional 3-field `NamedTuple`, `belief.py:8-13`); adding a δ channel needs the M3 dataclass refactor first. State this dependency; do not bolt a fourth positional field onto the tuple.

## 4. The math

### 4.1 Edge-relaxed cocycle (`Participatory:838-844`, `eq:edge_relaxed_omega`)

For an edge-local connection `delta_ij ∈ R^{n_gen}` valued in the structure group's own generator coordinates,

```
Omega_ij = exp(phi_i . G) exp(delta_ij . G) exp(-phi_j . G),
delta_ij . G = sum_{a=1}^{n_gen} delta_ij^a G_a  ∈  g,
```

with `{G_a}` the group's generator basis (`group.generators`). Setting `delta_ij = 0` recovers `Omega_ij = exp(phi_i)exp(-phi_j)`, the flat cocycle, and the vanishing-holonomy theorem. Because δ is expressed on the group's own basis, `exp(delta_ij.G) ∈ G` by construction: block structure, `irrep_dims`, and `(family, group)` admissibility are preserved automatically. For `block_glk` (`irrep_dims = [d_head]*H`), δ partitions per head and one gets the per-head decomposition `H_ijk = direct-sum_h H_ijk^(h)`, `W_ijk = sum_h W_ijk^(h)` (`Participatory:886`) for free. This is the clean part of the design.

### 4.2 Homotopy (`Participatory:876-880`)

```
Omega_ij(alpha) = exp(phi_i.G) exp(alpha delta_ij.G) exp(-phi_j.G),   alpha ∈ [0,1],
```

interpolating Regime I (`alpha=0`) to fully relaxed Regime II (`alpha=1`). Exposed as `cocycle_relaxation`. Gauge covariance holds for every α.

### 4.3 Bilinear connection (form (c), `Participatory:884`)

```
delta_ij^a = mu_i^T W^a mu_j,      a = 1..n_gen,      W : (n_gen, K, K),
```

one einsum: `delta = torch.einsum("bik,akl,bjl->bija", mu, W, mu)`, giving `(B, N, N, n_gen)`, then `delta_mat = torch.einsum("bija,akl->bijkl", delta, G)` and `exp(delta_mat)` via the existing `stable_matrix_exp_pair` / `_blockwise_matrix_exp` (`transport.py:19,76`), then the triple product `exp(phi_i) @ exp_delta @ exp(-phi_j)`. No activation, no bias.

### 4.4 Gauge covariance (`Participatory:846-851`, `eq:omega_gauge_law`)

Under a vertex-local gauge transformation `U_i -> g_i U_i`,

```
Omega_ij  ->  g_i Omega_ij g_j^{-1},
```

with the connection coefficients `delta_ij` invariant under the chosen lift (absorbed into the vertex factors). This is the standard lattice-gauge link-variable law (Wilson 1974). It holds for the edge-relaxed cocycle and for every α in the homotopy.

### 4.5 Wilson observable and holonomy (`Participatory:854-873`)

Around a 3-cycle `i->j->k->i` the vertex factors cancel in the interior:

```
H_ijk = Omega_ij Omega_jk Omega_ki = U_i [ exp(delta_ij.G) exp(delta_jk.G) exp(delta_ki.G) ] U_i^{-1},
W_ijk = Re Tr( exp(delta_ij.G) exp(delta_jk.G) exp(delta_ki.G) )   (gauge-invariant),
S_Wilson[delta] = beta sum_{(i,j,k)} (1 - W_ijk / K),   beta -> inf  ==>  H_ijk -> I.
```

The squared-Frobenius variant `sum_ijk ||H_ijk - I||_F^2` is equivalent up to bounded reparameterization and is exactly the form the shipped diagnostic computes (`metrics.py:91`).

## 5. The causal-DAG nuance (must be in the doc)

Two distinct objects behave differently under the causal mask, and the difference drives both the config warning and the test design:

- The **`holonomy_deviation` diagnostic** enumerates all `(i,j,k)` triples over the index space and ignores the attention mask (`metrics.py:68-91`). So with `delta != 0` it reads non-zero even on causal data — it measures δ's holonomy "as a tensor on the index space rather than ... the actual message-passing pattern" (`Participatory:882`). The premise that the diagnostic "exists to exercise" Regime II holds for causal runs.
- The **Wilson penalty as a regularizer** is degenerate on the causal DAG (`Participatory:874,882`): every 3-cycle mixes a causal edge (`beta>0`, seen by the forward) with an anti-causal edge (`beta=0`, invisible), so penalizing the loop neither constrains the active connection nor corresponds to real path-dependence of information flow. Meaningful Wilson regularization, and belief-level holonomy (transport a belief around a loop and observe it returns rotated), live in **bidirectional / encoder** attention where the graph admits closed loops.

Consequence: the δ=0 and gauge-covariance tests are mask-independent, but the non-trivial-holonomy and Wilson tests need a small bidirectional fixture (`attention_prior="uniform"`, a 3-token toy) or they are vacuous.

## 6. Optional Wilson holonomy-penalty regularizer

The diagnostic at `metrics.py:50-91` already computes `mean ||H_ijk - I||_F`. The regularizer reuses it: at the converged belief in the loss path (next to the optional `mass_phi` gauge penalty referenced in roadmap item 2), add `holonomy_penalty_weight * holonomy_deviation(omega)` (or its squared-Frobenius variant, which matches `S_Wilson` up to bounded reparameterization). Gated to 0 by default. Document that it is degenerate on the causal default (Section 5) and is intended for the bidirectional research branch.

## 7. Phased TDD implementation outline

Each phase names the key test and the oracle that proves correctness. The oracles are ranked decisive-first.

**Phase 0 — the seam refactor (buildable-once-decided, no physics).** Add `_TRANSPORTS`/`register_transport`/`get_transport`; wrap the existing flat builders in `flat`/`omega_direct` adapters; route `_transport`/`_transport_qk` (`e_step.py:28-53`) and `model.py:231` through `get_transport("flat")`; add `transport_mode` config + validation. KEY TEST: the existing VFE_2.0 golden pins `test_transport_matches_frozen_oracle` (`tests/test_perf_equivalence.py:77`) and `test_phi_path_cocycle_identity` (`tests/test_transport.py:107`) stay green with `transport_mode="flat"` as the dispatched path. ORACLE: bit-for-bit identity to the pre-refactor `compute_transport_operators` output — the unambiguous correctness pin and the reason Phase 0 is safe to ship tonight if the user later blesses it.

**Phase 1 — the `regime_ii` builder, δ supplied externally.** Implement the triple product `exp(phi_i) exp(alpha delta.G) exp(-phi_j)` reusing `stable_matrix_exp_pair`/`_blockwise_matrix_exp`. KEY TEST 1 (decisive): `delta_ij ≡ 0` (or `cocycle_relaxation=0`) reproduces `compute_transport_operators(phi, group)["Omega"]` to `atol=0`. ORACLE: the flat builder itself — same function, δ-branch zeroed. KEY TEST 2: gauge covariance — draw `g_i = exp(xi_i.G)`, push `U_i -> g_i U_i`, assert `Omega_ij -> g_i Omega_ij g_j^{-1}` to `atol=1e-5` (`Participatory:846-848`). ORACLE: the analytic link-variable law. KEY TEST 3 (bidirectional fixture): a hand-built 3-token toy with chosen `delta_ij, delta_jk, delta_ki`, assert the diagnostic's `H_ijk` matches the analytic `Re Tr(exp(delta_ij.G)exp(delta_jk.G)exp(delta_ki.G))` (`eq:wilson_observable`, `Participatory:862-865`) and that `holonomy_deviation > 0` (mirroring the shipped `test_holonomy_deviation_positive_for_non_cocycle`, `tests/test_metrics.py:52`). ORACLE: closed-form `Re Tr` of three matrix exponentials computed independently.

**Phase 2 — the chosen δ parameterization (after the user decision).** For the bilinear: `W:(n_gen,K,K)` Parameter zero-init on the model; KEY TEST: `W=0 => delta=0 =>` Phase-1 KEY-TEST-1 flatness (model is bit-flat at init); plus a finite-difference gradient check of CE w.r.t. `W` against autograd. ORACLE: the δ=0 flat pin for the init, autograd-of-CE for the gradient. For the variational form: blocked on the M3 `BeliefState` dataclass refactor (`belief.py:8-13`); spec the δ-channel update as a fourth E-step latent, init 0.

**Phase 3 — optional Wilson penalty.** KEY TEST (bidirectional fixture): loss with `holonomy_penalty_weight>0` exceeds the unpenalized loss by exactly `weight * ||H-I||_F^2` at a configuration with known holonomy, and a gradient step reduces `holonomy_deviation`. ORACLE: the independently-computed penalty value; assert degeneracy (no constraint) on the causal fixture.

## 8. DECISION NEEDED FROM USER

1. **Which δ parameterization, and is it blessed as a third no-NN exception?** This is the only decision that touches CLAUDE.md's hard constraint. Recommendation: register all three, defaulting to **`flat` (δ≡0, pure path)**; offer **`regime_ii` + `bilinear` `delta_ij^a = mu_i^T W^a mu_j`** as an opt-in toggle the user explicitly blesses as the third documented NN-exception (alongside `use_prior_bank=False` and `use_head_mixer`). Rationale: the bilinear is the manuscript's named choice (`Participatory:884`), the leanest (one raw `(n_gen,K,K)` Parameter + one einsum, no activation, directly analogous to the blessed linear-decode `W`), and the only useful form that does not collapse. The pure no-NN non-flat option (variational δ field) is offered for completeness but flagged functionally vacuous on the causal default (Section 2b). Do **not** present the bilinear as satisfying the no-NN path — it does not.

2. **Causal vs bidirectional intent for Regime II.** The Wilson penalty and belief-level holonomy are degenerate on the causal DAG (Section 5); only the index-space diagnostic is non-trivial there. Recommendation: ship the transport and the diagnostic for the causal LM (the diagnostic is meaningful), but document the Wilson regularizer as a bidirectional-research-branch feature, defaulted off. Decide whether a bidirectional fixture/config is in scope tonight or deferred.

3. **`BeliefState` refactor ordering.** The variational-δ form (2b) requires the M3 dataclass refactor (`belief.py:8-13`, roadmap M3) before a δ channel can be added without a positional-signature sweep. The bilinear form does not (its `W` lives on the model). Recommendation: if going bilinear, build it now and leave the variational form to land after M3.

4. **Homotopy knob exposure.** Whether to expose `cocycle_relaxation` (α) as a config field now (`spec:60` calls the regime "a toggle"; the manuscript exposes α only in the research branch, `Participatory:880`). Recommendation: include it (one float, defaulting 1.0; α=0 reduces to flat and is already covered by `transport_mode="flat"`) since it is free once `regime_ii` exists and is the documented Regime-I↔II interpolation.

## 9. Risks

- **No-NN purity drift.** The useful δ forms carry a learned weight outside the current blessed exceptions. Mitigated by defaulting `flat`, gating `regime_ii+bilinear` behind an explicit opt-in flag, and refusing construction without it (mirroring the `omega_direct` rejection, `config.py:156`). The pure path (`flat`, δ≡0) always exists.
- **Per-edge collapse of the free variational form** (Section 2b): drives belief-coupling to uniform attention. Mitigated only by the Wilson penalty, itself degenerate on the causal DAG. Honest limitation; do not ship the free variational form as a default.
- **Builder-signature creep.** The stateful builder needs `mu`/`connection` extras the pure flat builder ignores. Mitigated by the uniform keyword-only signature (Section 3.2) so call sites never branch.
- **`stable_matrix_exp_pair` Frobenius clamp** (`transport.py:43-45`, `max_norm=15`) now sees three stacked exponentials' worth of argument; a large bilinear δ could trip the clamp and silently rescale. Mitigated by zero-init `W` and the monitor the roadmap already proposes for the clamp (Tier D); document that `max_norm` must exceed the combined `phi+delta` norm budget.
- **No ground-truth oracle for the physics.** Which connection is "correct" is a research question, not a pinnable fact — hence the seam is buildable tonight but the δ choice is spec-only and user-gated.

## 10. Verdict

The `register_transport`/`get_transport` seam, the `flat` default adapter, and the `regime_ii` builder with the **δ=0 / α=0 flatness pin** are **buildable-once-decided** — the flat-recovery oracle is unambiguous and bit-exact against the existing VFE_2.0 golden. The δ parameterization is **needs-user-decision and needs-no-research-but-needs-blessing**: the candidate forms are fully specified by the manuscript, but the useful ones require a learned parameter the user must bless as a third no-NN exception, and there is no oracle for which connection physics is right. Correctly spec-only tonight.