# Memo — debate-expert-gauge-theorist — BLUE — round 1 — blocks-k48-gauge-block-scaling-axis

## Lens
Gauge theory — Lie groups, principal bundles, irreps, holonomy, equivariance, gauge fixing.

## Steelman of the opposing position
The 3.62x total-parameter growth is an unstructured `V x n_gen` lookup table dressed in Lie-algebra notation; a flat connection adds no holonomy, so nothing gauge-theoretic is doing work that a plain learned embedding of matched size would not.

## Verdict
The gauge structure supports the narrow, true core: block enlargement is a *structured* capacity axis distinct from width, because it genuinely enlarges the local gauge group per block (`GL(3)^16 -> GL(24)^2`) and the added parameters are gauge-covariant connection coordinates, not an arbitrary dense blob. Gauge theory does NOT by itself certify the compound qualifier "parameter-efficient / publishable as-is" — concede that, because these runs run off the strictly equivariant path and no non-gauge control isolates structured capacity from raw capacity.

## Argument

**Vector 1 — the block partition genuinely enlarges the local gauge group; the added params are structured.** `block_glk` builds `GL(d_head)^n_heads` (`vfe3/geometry/groups.py:144-152`); `generate_glk_multihead` returns `(n_heads * d_head^2, K, K)` (`vfe3/geometry/generators.py:96,103`). At fixed `K=48`, block width `g = d_head`, `n_heads = 48/g`, so `n_gen = n_heads * g^2 = 48*g` — confirmed 144 (GL3) to 1152 (GL24). Each block carries the *full* `gl(g)` Lie algebra (`g^2` generators), so enlarging `g` from 3 to 24 raises the per-block fiber-symmetry dimension 9 -> 576: a change of *structure group*, not a wider dense layer. Those `phi_embed` rows (`vfe3/model/prior_bank.py:167,682`) are Lie-algebra coordinates driving covariant transport `Omega_ij = exp(phi_i . G) exp(-phi_j . G)` acting by the sandwich `mu -> Omega mu`, `Sigma -> Omega Sigma Omega^T` (`vfe3/geometry/transport.py:4,6,183`). Nakahara §10.1-10.3 fixes that the sandwich congruence is the correct structure-group action on associated-bundle sections, so the extra DOF are gauge-covariant transport, not free weights.
- **Cohen & Welling 2016, "Group Equivariant Convolutional Networks," ICML 2016** — enlarging the symmetry group (p4 -> p4m) raises accuracy at *fixed* parameter count via structured weight sharing; the group is a capacity axis orthogonal to channel width. Directly supports "structured, not raw."
- **Kondor & Trivedi 2018, "On the Generalization of Equivariance and Convolution... to the Action of Compact Groups," ICML 2018** — an equivariant linear map is *exactly* a group convolution; the group determines the admissible structure. Changing the group is a structural change, not added free capacity.
- **Nakahara 2003, §10.1-10.5** — connection/holonomy: `Omega_ij = g_i g_j^{-1}` is a pure-gauge (coboundary) transition; for a triangle `Omega_ij Omega_jk Omega_ki = I`, so the flat (Regime-I) connection has trivial holonomy (Ambrose-Singer with `F=0`). Honest scope of the mechanism: block enlargement enriches *flat-frame* belief transport, not curvature.

**Vector 2 — group choice is a recognized design axis distinct from width, so a complementary knob is a legitimate contribution.**
- **Weiler & Cesa 2019, "General E(2)-Equivariant Steerable CNNs," NeurIPS 2019** — "implement a wide range of ... equivariant network architectures and extensively compare their performances," varying the symmetry group and field-representation type as the design dimension (kernel constraints for arbitrary reps reduce to irrep constraints). Establishes group/representation choice as a benchmarked axis separate from width.
- **Cohen, Weiler, Kicanaoglu & Welling 2019, "Gauge Equivariant CNNs and the Icosahedral CNN," ICML 2019** — extends equivariance "beyond global symmetries to local gauge transformations"; a gauge is an arbitrary choice of fiber frame and the network is equivariant to changing it. This is the closest published analog to the `GL(g)`-per-block *internal* gauge here, grounding "local gauge group" as a first-class structural object.

**Vector 3 — the equivariance-breaking component shrinks as blocks enlarge, so the gain is not learned-mixer capacity.** The head mixer is one `A_t = I + Delta_t` per equal-label run, i.e. `H x H = (48/g)^2` params, shrinking 256 (GL3) -> 4 (GL24) (`vfe3/model/head_mixer.py:54-70`). PPL improves 124.6 -> 92.2 as this learned-linear, non-equivariant block shrinks 64x, so the improvement co-moves *opposite* to mixer capacity — it cannot be the driver.

## Concessions / limits
1. These runs are off the strictly gauge-equivariant path: `use_head_mixer=True` under `block_glk`'s *untied* per-head gauge breaks strict equivariance off identity init (`vfe3/model/head_mixer.py:29-35`), and `use_prior_bank=False` is the learned linear decode. "Gauge structure does the work" is scoped to the block partition's *structured capacity*, not to exact equivariance of the trained model.
2. No non-gauge control was trained (a matched-size unstructured `V x m` table, or a matched-param dense head-mix), so "structured gauge capacity per se vs raw capacity" is theoretically motivated but empirically unisolated.
3. The axis co-varies three knobs: enlarging `g` grows the local group but *shrinks* the number of independent gauge fibers `n_heads` (16 -> 2). Gauge theory calls this a tradeoff (richer per-block group, fewer blocks), not pure enrichment.
4. Flat connection: `transport_mode='flat'` => trivial holonomy. The added DOF are pure-gauge frame coordinates; they enrich flat transport, not curvature/holonomy. The claim must not be sold as "richer holonomy."

## Falsification conditions
This position is wrong on gauge-theoretic grounds if: (X) a matched-size unstructured `V x m` learned table (`m = n_gen`, no group action) reproduces the 124.6 -> 92.2 curve — then the gain is raw table capacity, not structured gauge capacity; (Y) a `tied_block_glk` run (exactly equivariant, `n_gen = g^2` not `48g`) matches the untied curve at far fewer params — then per-block untied richness is not the mechanism; or (Z) the improvement vanishes when the head mixer is turned OFF on the pure equivariant path — then the gain was mixer-linked despite the shrink argument.

## Newly-discovered canon (for 01b_extended_evidence.md)
- Cohen, Weiler, Kicanaoglu & Welling 2019, "Gauge Equivariant Convolutional Networks and the Icosahedral CNN," ICML 2019, arXiv:1902.04615 (https://proceedings.mlr.press/v97/cohen19d.html) — equivariance "extended beyond global symmetries to local gauge transformations"; closest analog to `GL(g)`-per-block internal gauge.
- Weiler & Cesa 2019, "General E(2)-Equivariant Steerable CNNs," NeurIPS 2019, arXiv:1911.08251 (https://proceedings.neurips.cc/paper/2019/hash/45d6637b718d0f24a237069fe41b0db4-Abstract.html) — benchmarks group/representation choice as a design axis separate from width; kernel constraints reduce to irrep constraints.
- Kondor & Trivedi 2018, "On the Generalization of Equivariance and Convolution... to the Action of Compact Groups," ICML 2018, arXiv:1802.03690 — equivariant linear = group convolution; group choice is structural, not free capacity.
- Bronstein, Bruna, Cohen & Velickovic 2021, "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges," arXiv:2104.13478 — the symmetry group is the primary design axis (the "5 Gs"), gauge symmetry included; frames the block-group knob as first-class.

## Confidence
MEDIUM — HIGH that block enlargement is structured (not raw) gauge capacity distinct from width; LOW on the compound "parameter-efficient / publishable as-is." A trained matched-size non-gauge control (condition X) or a `tied_block_glk` match (condition Y) would shift me toward red.
