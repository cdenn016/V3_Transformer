# Dense Regime II Direct-Link Specification

Date: 2026-06-29

Branch: `docs/regime-ii-link-spec-20260629`

Status: design specification; no production code is changed by this document.

This specification records the recommended Regime II target for `V3_Transformer` after a four-lens review across gauge theory, variational free energy, implementation wiring, and numerical analysis. The user constraint is binding: Regime II should work more or less the way Regime I works at the behavioral level. A run sets a sequence length, the model constructs pairwise transports for all configured token pairs, and the existing attention prior decides which pairs carry attention mass. The v1 design is therefore dense all-pairs over `max_seq_len`, not a sparse, windowed, or local-link connection. Any future chunking or factorization is a memory optimization only; it must not change the all-pairs graph.

## Decision

The canonical Regime II object should be a direct, group-valued link variable

$$
L_{ij}\in G,
$$

attached to every ordered pair of token positions \(i,j\) in the configured sequence domain. The link \(L_{ij}\) is the transport map from the fiber over \(j\) to the fiber over \(i\). It is not a scalar score, not an attention weight, and not merely a perturbation of the vertex frames. For Gaussian beliefs it acts by

$$
\mu_j\mapsto L_{ij}\mu_j,\qquad
\Sigma_j\mapsto L_{ij}\Sigma_j L_{ij}^{\top}.
$$

The belief coupling remains the existing gauge-transported KL energy,

$$
E_{ij}=D_{\mathrm{KL}}\left(q_i\middle\|L_{ij}q_j\right),
$$

and the attention block remains the canonical entropy-regularized variational block,

$$
\sum_j \beta_{ij}E_{ij}
+\tau \sum_j \beta_{ij}\log\frac{\beta_{ij}}{\pi_{ij}}.
$$

With the row constraint \(\sum_j\beta_{ij}=1\), this block gives

$$
\beta_{ij}
=
\frac{\pi_{ij}\exp(-E_{ij}/\tau)}
{\sum_k \pi_{ik}\exp(-E_{ik}/\tau)}.
$$

Dropping the entropy term changes the variational problem; it turns the softmax into a heuristic surrogate rather than the stationary point of the row Lagrangian. Regime II must not be used as a reason to relax this requirement.

## Relation to the existing Regime I path

Regime I is the flat cocycle

$$
\Omega^{(0)}_{ij}=U_iU_j^{-1},\qquad U_i=\exp(\phi_i),
$$

so every triangle closes:

$$
\Omega^{(0)}_{ij}\Omega^{(0)}_{jk}\Omega^{(0)}_{ki}=I.
$$

The existing `regime_ii` and `regime_ii_covariant` builders insert an edge factor between vertex frames,

$$
\Omega_{ij}=U_i\exp(\delta_{ij})U_j^{-1}.
$$

That is a valid coordinate chart for a link, but it should not be treated as the ontology of Regime II. The ontology is the edge transport \(L_{ij}\). If a future implementation chooses to represent \(L_{ij}\) as \(U_iV_{ij}U_j^{-1}\), that sandwich is a parameterization choice. It is not the mathematical reason the connection is curved.

The direct-link mode specified here should be named `transport_mode="regime_ii_link"`. In that mode, the transport builder should produce a pairwise link tensor from an all-pairs learned algebra table. The clean v1 formulation is

$$
A_{ij}\in\mathfrak g,\qquad
L_{ij}=\exp(\alpha_{\mathrm{link}} A_{ij}),
$$

with \(A_{ii}=0\) before exponentiation, so \(L_{ii}=I\). The parameter table should be indexed by sequence positions and sliced to the active length:

$$
A \in \mathbb R^{\mathrm{max\_seq\_len}\times\mathrm{max\_seq\_len}\times n_{\mathrm{gen}}},
\qquad
A_{0:N,0:N,:}\text{ for a length }N\text{ forward pass}.
$$

This is dense all-pairs. Causal masking, ALiBi, T5-style priors, or other attention priors may assign zero or small mass to some pairs, but the connection itself exists for all ordered pairs in the configured domain. A sparse/windowed Regime II may be useful later as an experiment, but it is not this specification.

## Gauge covariance requirement

Under a local change of frame \(g_i\in G\), beliefs transform as

$$
\mu_i\mapsto g_i\mu_i,\qquad
\Sigma_i\mapsto g_i\Sigma_i g_i^{\top}.
$$

A direct link must transform as

$$
L_{ij}\mapsto g_iL_{ij}g_j^{-1}.
$$

Then the transported key belief transforms in the query frame:

$$
L'_{ij}\mu'_j=g_iL_{ij}\mu_j,
$$

and

$$
L'_{ij}\Sigma'_j(L'_{ij})^{\top}
=
g_iL_{ij}\Sigma_jL_{ij}^{\top}g_i^{\top}.
$$

The two arguments of \(D_{\mathrm{KL}}(q_i\|L_{ij}q_j)\) are therefore pushed forward by the same \(g_i\), so the Gaussian KL is invariant. This is the gauge principle the new mode must satisfy.

In the current codebase, the raw bilinear `regime_ii` route does not satisfy this law once `connection_W` is nonzero. The test `tests/test_regime_ii.py::test_regime_ii_edge_factor_breaks_gauge_invariance_for_nonzero_W` records that behavior. The current `regime_ii_covariant` route is better: it builds the middle factor from invariant KL features and pins the end-to-end law \(\Omega_{ij}\mapsto g_i\Omega_{ij}g_j^{-1}\). The direct-link design should be judged by the same end-to-end covariance standard.

Because a learned parameter table lives in one gauge during ordinary training, the covariance test should be active rather than passive. A small full-covariance test should transform the beliefs and the link table together, then verify that the pairwise KL energy matrix is unchanged to tolerance. A second test may check that the assembled link tensor itself obeys the expected conjugation law.

## Curvature and holonomy

A non-identity link is not by itself curvature. Curvature is measured by a closed loop:

$$
H_C=L_{i_0i_1}L_{i_1i_2}\cdots L_{i_mi_0}.
$$

Under a gauge transformation this loop transforms by conjugation at the base point,

$$
H_C\mapsto g_{i_0}H_Cg_{i_0}^{-1}.
$$

The curvature certificate should therefore be a conjugacy-invariant observable: the spectrum of \(H_C\), the characteristic polynomial, a Wilson-loop trace, or a stable function of eigenvalues. The existing Frobenius diagnostic \(\|H-I\|_F\) is useful as a fixed-gauge training signal, but it is not gauge invariant for full noncompact `GL(K)`. The spec should not call that quantity the final gauge-invariant curvature observable.

For compact or orthogonal groups, the Wilson density

$$
1-\frac{1}{K}\operatorname{Re}\operatorname{tr}(H_C)
$$

is the natural lattice-gauge diagnostic. For noncompact `GL(K)`, a naive Wilson trace is not a positive bounded Yang-Mills density. The v1 diagnostic should report both the existing fixed-gauge Frobenius holonomy and at least one conjugacy-invariant loop statistic, such as eigenvalue log-distance from one on small sampled triangles. A later Yang-Mills regularizer must use a nonnegative noncompact-safe density or restrict the link group to a compact/skew subgroup.

## Variational status

The v1 direct-link mode is a learned model connection coupled to beliefs through the transported KL block. It is ELBO-clean only if the link variable is treated as a model parameter with a stated prior/regularizer, or as a variational latent with its own KL term. The first implementation should take the simpler route: \(A_{ij}\) is a model parameter, zero-initialized, optimized in the M-step through the outer loss and the unrolled E-step. It should be described as a learned connection with a connection prior or weight decay, not as an automatically derived posterior over links.

The E-step must remain target-blind. Regime II links may alter the geometry of belief coupling, but they must not read the next-token target, cross-entropy loss, or decode error during belief inference. The only permitted route for link learning is the M-step gradient through the differentiable forward computation, exactly as current model-owned parameters train.

The current non-flat modes auto-enable `oracle_unroll_grad=True` because their learned connections enter the loss through the E-step trajectory. For a direct link that is independent of the current beliefs, the closed-form kernel can remain valid as long as the transport is treated as fixed during the belief-gradient derivation. If the implementation later makes links belief-conditioned, the mode must route through the oracle and must state the resulting approximation. Matter-conditioned links are an amortized connection, not the canonical Regime II specified here.

The model-channel \(\gamma\) coupling is out of scope for v1. The current code intentionally keeps that channel on flat transport even when the belief channel uses a Regime II mode. This document specifies belief-channel Regime II only. Extending model-channel transport requires a separate design because it changes the \(s\)-fiber coupling, its attention prior, and the interpretation of \(\lambda_\gamma\).

## Implementation contract

The target mode should be registered by adding a builder to `vfe3/geometry/transport.py`:

```python
@register_transport("regime_ii_link")
def _build_regime_ii_link(
    phi:                torch.Tensor,
    group:              GaugeGroup,
    *,
    gauge_mode:         str                    = "learned",
    link_alpha:         float                  = 1.0,
    link_soft_cap:      float                  = 6.0,
    connection_L:       Optional[torch.Tensor] = None,
    **kwargs,
) -> TransportDict:
```

The returned dictionary should initially match the existing transport interface:

```python
{"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}
```

For the direct-link mode, `Omega` is the pairwise link tensor \(L_{ij}\). `exp_phi` and `exp_neg_phi` may be returned for interface compatibility and diagnostics, but they should not be multiplied into the direct link unless the implementation explicitly selects a charted variant. The direct-link mode must avoid accidental double transport by not applying both \(L_{ij}\) and \(U_iU_j^{-1}\) as independent physical transports.

The model should create `connection_L` only when `cfg.transport_mode == "regime_ii_link"`. The parameter shape should be

```python
(cfg.max_seq_len, cfg.max_seq_len, n_gen)
```

where `n_gen = model.group.generators.shape[0]`. The table should be zero-initialized. The self-edge should be forced to zero at build time rather than trusted to stay zero under optimization. A config scalar `link_alpha` should provide the homotopy \(L_{ij}=\exp(\alpha_{\mathrm{link}}A_{ij})\), with `link_alpha=0.0` giving the identity link table. A separate `link_soft_cap` should smooth-cap the embedded algebra matrix \(A_{ij}\cdot G\), not the raw coordinate vector.

The dense all-pairs design means no `needs_mu` or `needs_sigma` registration is required for the canonical direct-link builder. The link table is independent of current belief state. That keeps the existing kernel route available for fixed-transport belief gradients, subject to direct verification. If later variants compute \(A_{ij}\) from \(\mu,\Sigma\), they must be registered as state-dependent and must use the oracle route.

The optimizer should group `connection_L` at `cfg.m_phi_lr`, role `"phi"`, with `cfg.connection_weight_decay` when that field is set. This mirrors `connection_W` and `connection_M`. The exact-coverage guard in `build_optimizer` should catch any missed parameter. Diagnostics should log `connection_l_norm`, `link_cap_frac`, and at least one link-conditioning statistic. The reporting path should forward `connection_l_norm` into metrics CSV and generated reports so it does not repeat the partial visibility gap that can occur when a diagnostic key is emitted but not surfaced downstream.

The belief-prefix cache should remain unsupported for `regime_ii_link` until proven otherwise. Even though a direct link is belief-independent, the cached rollout proof was written for flat causal transport. The first implementation should make `cache_supported(cfg)` continue to reject every non-flat transport mode.

## Dense all-pairs performance rule

The v1 graph is dense all-pairs. The implementation may still chunk over queries, heads, or blocks to control memory, but the result must be exactly the same as computing every \(L_{ij}\) for \(0\le i,j<N\) and feeding the full pairwise energy matrix to attention. A chunked implementation must be value-equivalent to the dense reference on small cases, including gradients to selected link coordinates.

Dense all-pairs full \(K\times K\) link materialization is expensive. For \(B=64\), \(N=128\), and \(K=64\), a batched `(B,N,N,K,K)` fp32 transport tensor is about 16 GiB before autograd, optimizer state, logits, or E-step intermediates. The correctness implementation may materialize dense tensors for small tests, but production use should prefer exact all-pairs chunking or a block-wise transport container. That is a performance implementation detail, not a change to the mathematical graph.

The initial supported operating point should be `family="gaussian_diagonal"` with `block_glk` equal-head blocks. Full covariance with noncompact links should warn or reject until it has a float64 sandwich path and a hard non-PD failure policy. Compact/skew link groups are the safer full-covariance path because their condition number stays near one.

No link exponentials, Cholesky solves, log-determinants, or covariance sandwiches should run in bf16 or fp16. Autocast may be used elsewhere, but link matrix functions and SPD-sensitive operations need fp32 or a documented float64 island. The current Regime II covariant path already shows why: congruence sandwiches square the condition number of the transport.

## Required tests

The implementation should add `tests/test_regime_ii_link.py` and extend the relevant routing tests. The minimum acceptance suite is as follows.

First, registry and config tests should show that `get_transport("regime_ii_link")` succeeds, `VFE3Config(transport_mode="regime_ii_link")` validates, `link_alpha` is finite and in `[0,1]`, and `gauge_transport="off"` still rejects the mode unless the meaning of that meta-toggle is redesigned. The default `transport_mode="flat"` path must create no `connection_L` attribute.

Second, flat-limit tests should show that `connection_L=None` and `link_alpha=0.0` return identity direct links, and that a matched identity-link baseline agrees with an identity flat baseline. If the implementation also supports a charted compatibility mode \(U_iV_{ij}U_j^{-1}\), that mode must separately prove zero-link equality to current Regime I. The direct-link mode should not silently claim zero-link equality to a nonzero learned \(\phi\) cocycle; identity gauge and pure-gauge cocycle are gauge-equivalent, but they are not byte-identical coordinates.

Third, self-edge tests should verify \(L_{ii}=I\) after slicing, capping, exponentiation, and any optimizer step. The diagonal should be masked in the builder so the model cannot learn a spurious self-transport.

Fourth, curvature tests should show that sampled triangle holonomy is near zero for the identity-link table and nonzero for a seeded nonzero table. The fixed-gauge Frobenius statistic may be used for sensitivity, but at least one conjugacy-invariant loop statistic must also be tested on a small example.

Fifth, gauge-covariance tests should transform small full-covariance beliefs and the link table by \(g_iL_{ij}g_j^{-1}\), then verify that the pairwise KL energy is invariant. This is the test that separates a principled direct link from the old bilinear impurity.

Sixth, gradient tests should compare `dF/dconnection_L` to central finite differences on a small deterministic problem. If the kernel route remains enabled, a test should also verify that the served belief gradient matches the autograd reference for fixed direct links. If any state-dependent link variant is later added, it must have separate oracle-gradient tests.

Seventh, model-wiring tests should show that `VFEModel` creates `connection_L` only in direct-link mode, that the optimizer groups it exactly once, that a nonzero table changes the forward loss, and that `loss.backward()` populates a finite nonzero gradient under the supported estimator.

Eighth, diagnostics and artifacts tests should show that `model.diagnostics()`, `viz.extract.converged_state`, per-layer diagnostics, metrics CSV, and run summaries all see the active direct-link transport. The tests should include `connection_l_norm`, holonomy, cocycle residual, link conditioning, and cap-saturation keys where applicable.

Ninth, cache tests should show that `cache_supported(cfg)` rejects `transport_mode="regime_ii_link"` until a dedicated cache proof is written.

Tenth, a memory guard should assert that production all-pairs code does not allocate avoidable full batched transport tensors at the intended large operating point. This should be a CUDA test guarded by availability and may record `torch.cuda.max_memory_allocated()` rather than asserting an exact number.

## Rejected alternatives

The raw bilinear `regime_ii` route should remain an exploratory impurity and regression oracle, not the principled Regime II. It produces nontrivial holonomy, but it does not satisfy the gauge law for nonzero `connection_W`.

The current `regime_ii_covariant` route should remain a stronger covariant middle-factor experiment. It is a useful correctness reference for the law \(\Omega_{ij}\mapsto g_i\Omega_{ij}g_j^{-1}\), but it still builds the edge from current belief features and therefore should be described as an amortized or belief-conditioned connection unless a generative link model is added.

Sparse or windowed links are rejected for this v1 specification because the required behavior is all-pairs attention over the configured sequence length. They may be added later as a scalability experiment, but their results should not be compared to Regime I as if the attention graph were unchanged.

A pure vertex-frame path \(\exp(\phi_i)\exp(-\phi_j)\) is Regime I. It is the flat baseline, not a Regime II implementation, no matter how nontrivial the individual frames become.

Diagonal or Cartan-only direct links are acceptable as debug baselines, but they are too commutative to serve as the main non-flat gauge connection. They should be labeled as an ablation, not as the canonical Regime II.

## References and source anchors

The external mathematical template is standard lattice gauge theory: Wilson's link-variable formulation of confinement, Phys. Rev. D 10, 2445 (1974), https://doi.org/10.1103/PhysRevD.10.2445, and Kogut and Susskind's Hamiltonian formulation, Phys. Rev. D 11, 395 (1975), https://doi.org/10.1103/PhysRevD.11.395. For neural local-gauge covariance, Cohen, Weiler, Kicanaoglu, and Welling give the gauge-equivariant CNN construction in arXiv:1902.04615, https://arxiv.org/abs/1902.04615, and Weiler et al. formulate coordinate-independent gauge-equivariant convolutions in fiber-bundle language in arXiv:2106.06020, https://arxiv.org/abs/2106.06020. Textbook geometry anchors remain Nakahara, Baez-Muniain, and Frankel for connections and holonomy. The in-repo anchors are `vfe3/geometry/transport.py` for the transport registry and current non-flat builders, `vfe3/free_energy.py` for the canonical attention block, `vfe3/inference/e_step.py` for transport routing and gradient mode, `vfe3/model/model.py` and `vfe3/train.py` for parameter creation and optimizer grouping, `vfe3/metrics.py` for holonomy diagnostics, `tests/test_regime_ii.py` for the bilinear impurity characterization, and `tests/test_regime_ii_covariant.py` for the current covariant Route B law.

## Summary

The most principled Regime II for `V3_Transformer` is a dense all-pairs direct link connection \(L_{ij}\) over the configured sequence domain. It should enter only through gauge-transported Gaussian KL belief coupling, keep the canonical attention entropy term, learn as a model-owned connection in the M-step, preserve a default flat pure path by remaining opt-in, and certify non-flatness through closed-loop holonomy rather than nonzero edge factors. The vertex-frame sandwich remains a useful chart, but it is no longer the conceptual target.
