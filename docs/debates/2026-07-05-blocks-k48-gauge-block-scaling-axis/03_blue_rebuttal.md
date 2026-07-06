# Blue Rebuttal — blocks-k48-gauge-block-scaling-axis

Side: BLUE (defend by steelmanning). Mode: implementation. Round: rebuttal. Panel:
philosophy-of-science, implementation-engineer, ml-engineer, gauge-theorist, numerical-analyst.

Red's opening is two-thirds correct and one move too aggressive. It proves that the efficiency
*metric* is definitionally flat, that the fitted *exponent* is unidentified, and that the causal
*label* "gauge structure per se" is confounded. Blue grants all three. Where red overreaches is the
rhetorical slide from those three results toward the disposition that the compound "fails" with, in
effect, nothing left but a bare monotone curve. There is a real, seed-robust, single-budget effect
that survives every one of red's three attacks untouched, and there is a well-defined structural axis
underneath it. The correct disposition is REMAND to a *real effect* whose causal decomposition and
efficiency framing are not yet identified — not defeat with an empty record.

## Concession (honest, and load-bearing)

Blue concedes the compound claim as written. Specifically:

1. **The efficiency proxy is a definitional identity, not a measurement.** Under this config
   `vfe3/run_artifacts.py:616-620` resolves to `active = 5·V·K + 2·K + n_gen`; with `V=50257` and
   `K=48` fixed, only `+n_gen` varies (`+1,008`, `+0.008%` across GL3→GL24). "Essentially constant
   ~12.06M" is what the formula must return, not an efficiency finding. Red's section 1 is right.

2. **There is no reportable scaling exponent.** Blue's own numerical-analyst reproduced red's result:
   the offset law `CE = E + A·N^{-α}` gives `α = 0.929` vs `n_params` and `0.180` vs `n_gen` (a 5.15×
   swing on identical CE data), with `corr(E, α) = +0.976 / +0.998` (floor and exponent not separately
   estimable) and a log-log design condition number of 306, above the Belsley-Kuh-Welsch threshold of
   100. The dynamic range is under one decade, short of the two-decade floor Stumpf & Porter (2012)
   require of a credible power law. Blue defends no exponent and no power law.

3. **The cross-sweep "complementary to width-scaling" comparison is a Chinchilla D-slice confound.**
   `blocks_K48` (245.76M tokens) versus `grow_K_GL10` (491.52M) is a 2× token gap; under Hoffmann et
   al.'s (2022) `L(N,D) = E + A/N^α + B/D^β`, every cross-sweep reading — matched-active, matched-params,
   "GL24 sits on the grow frontier on half the tokens" — sits on a different D-slice and is not
   identified. Blue withdraws it.

4. **The efficiency conjunct is unproven on real compute, and the causal conjunct is uncontrolled.**
   Shazeer et al. (2017) and Fedus, Zoph & Shazeer (2021) license the total-vs-active axis only against a
   flat compute axis ("a constant computational cost"); here wall-time is U-shaped with GL24 — the
   best-loss point — the slowest run, so "parameter-efficient" is not earned as a *compute* claim. And no
   discriminating control was trained: neither a plain `V × m` learned table of matched size nor a
   non-gauge multi-head baseline swept at fixed head geometry. The share of the gain attributable to the
   generator algebra versus the raw `phi_embed` scalars that grow in lockstep is unmeasured.

What blue does *not* concede is that these four concessions empty the record. They scope to the proxy,
the exponent, the cross-sweep comparison, and the causal label. None of them reaches the effect itself.

## Core attack: red conflates "the causal label is not isolated" with "there is no effect"

An artifact would not produce a large, strictly monotone, three-seed-robust 26% perplexity reduction
(PPL 124.57 → 92.15) at a **single fixed token budget** with **no internal D-confound**. Red's own
steelman concedes the CE decrease "is real, monotone, and seed-robust" (`02_red_opening.md`, l.22).
Blue holds red to that concession and shows that each of red's three attacks lands on the interpretive
overlay, not on the phenomenon.

**The effect is a phenomenon, not a metric.** Bogen & Woodward (1988, *Philosophical Review*
97:303-352, pp. 306, 314) separate *data* — the raw CSV integers — from *phenomena* — the stable,
inferred regularities that theories explain. The monotone PPL 124.6 → 92.2 across seeds {6, 23, 64} at
per-label std ≤ 1.08 is a detected phenomenon. Red's proxy-flatness, exponent-instability, and
confound arguments are all criticisms of the *theory that would explain or quantify* the phenomenon,
not evidence that the phenomenon is absent. Hacking's (1983) point that experiment "has a life of its
own, independent of theory" is the same separation stated from the laboratory side: the seed-robust
regularity does not dissolve when the efficiency proxy is shown to be theory-laden.

**A confound underdetermines the causal label, not the effect.** Red invokes Duhem (1906, p. 187) —
"the experiment does not designate which one should be changed" — to attack the attribution to gauge
structure. Grant it in full. But Duhem presupposes that *something changed* and denies only that we can
say *which* co-varying knob deserves credit. Pearl (2009, *Statistics Surveys* 3:96-146, p. 99) makes
the boundary exact: red's confound defeats the interventional query `P(CE | do(gauge structure))`; it
leaves the observed association `P(CE | block size)` intact and robust. Absence of the clean
intervention is a demand for a control experiment, not a refutation of the measured regularity.

**The effect is a detection result, mathematically independent of the estimation red refutes.** The
condition number that dooms the exponent fit bounds only the *inverse* map — recovering a slope from
the data (Hansen 1998, *Rank-Deficient and Discrete Ill-Posed Problems*); it says nothing about a
well-posed forward functional of the data. Blue's numerical-analyst computed that functional. The
per-step PPL drops are 12.16, 6.21, 6.43, 7.62; the *smallest* step (GL6→GL8) clears the seed-noise
floor by 4.07σ (conservative single-observation SE) to 7.04σ (seed-mean SE). Distribution-free, under
exchangeability of the five labels within a seed, all three seeds independently realizing the exact
strictly-decreasing order has probability `(1/5!)^3 = (1/120)^3 = 5.8e-7`; Spearman ρ and Kendall τ are
both −1.000. This is the ordered-alternatives detection problem of Jonckheere (1954) and Page (1963),
and it presumes no fitted α, no x-axis choice, and no parametric model. Red proved the slope parameter
is unrecoverable. That is not evidence the effect is absent.

**The within-sweep budget is fixed, so red's strongest citation does not reach the effect.** Blue's
implementation-engineer verified from all five `blocks_K48` configs that each carries identical
`max_steps=60000`, `batch_size=32`, `max_seq_len=128`, `grad_accum_steps=1` and the same `data_sha256`,
giving exactly `245,760,000` tokens per cell. The 2× cross-sweep gap is mechanically a `batch_size`
difference (grow uses 64). Under Hoffmann's `L(N,D)`, the `B/D^β` floor term is byte-identical across
all five blocks cells and cannot generate a within-sweep ordering; Kaplan (2020) and Hoffmann (2022)
constrain comparisons across *different* D, which the within-sweep effect is not. Red imported the
D-slice confound from the cross-sweep comparison and let it color the within-sweep effect; it does not
transfer.

## Defense: the sparse-capacity bookkeeping is standard, and the block axis is a real structural knob

**`phi_embed` is genuine embedding-table capacity, and excluding it from the per-token count is the
prescribed convention, not a trick.** At `vfe3/model/prior_bank.py:167`, `phi_embed` is a learned
`(V, n_gen)` parameter table; at `:682`, `phi = pb.phi_embed[token_ids]` is a token-indexed row lookup
returning one row of width `n_gen` per token — reached under the active `encode_mode='per_token'` path,
structurally the GPT-2 `[50257 × 768]` token-embedding matrix with `n_gen` in place of `d_model`. This
is the total-vs-active signature by construction: total capacity grows in the table (99.7% of the 3.62×
growth) while each token reads one row. Kaplan et al. (2020, §2.1) define their scaling variable N as
"the number of *non-embedding* parameters," and Figure 6 shows that only when embedding parameters are
excluded do models "converge to a single trend." Counting `phi_embed`'s `V·n_gen` bulk *out* of the
per-token working set is the Kaplan convention applied to a Kaplan-shaped object, reinforced by Shazeer
(2017) and Fedus, Zoph & Shazeer (2021) on total-vs-active capacity. A low per-token working set at high
total capacity is a genuine, citable property. It is not by itself an efficiency proof — blue conceded
that the compute axis is adverse — but it is not the metric artifact red frames it as either.

**The block enlargement is a genuine change of structure group, which is what makes "distinct axis"
defensible as a design coordinate.** From code, `block_glk` builds `GL(g)^(48/g)` (`groups.py:144-152`,
`generators.py:96,103`), a strictly nested tower `GL(3)^16 ⊂ GL(6)^8 ⊂ GL(12)^4 ⊂ GL(24)^2` whose
per-block fiber-symmetry dimension rises from `gl(3)=9` to `gl(24)=576` generators, with `n_gen = 48·g`
equal to the dimension of the structure group. Red's "Vaswani `d_k = d_model/h` run backward" is a
correct but incomplete description: standard multi-head attention gives each head a dense subspace with
no internal Lie group and no congruence action on a per-token covariance, whereas here each block
carries the full `gl(g)` algebra acting by the sandwich `μ → Ωμ`, `Σ → ΩΣΩᵀ` (`transport.py`). Blue's
gauge-theorist verified that `Ω_ij = g_i g_j^{-1}` is a coboundary satisfying the cocycle condition and
inducing a flat, trivial-holonomy connection (Nakahara 2003, Ch. 9-10), matching `transport_mode='flat'`
— so the transport is exactly what the configuration claims. Cohen & Welling (2016) and Kondor & Trivedi
(2018) establish enlarging the group as a capacity axis orthogonal to width: the two sweeps are
orthogonal coordinates in `(K, g)` structure-group space (grow_K fixes `g=10` and grows `K`; blocks_K48
fixes `K=48` and grows `g`), and that orthogonality is a structural fact that holds regardless of whether
either fitted exponent is identifiable. "Distinct scaling axis" is defensible as a *design* axis even
though the *causal share* attributable to the group structure versus the co-growing scalar table is not
yet measured.

## Falsification conditions (blue's own)

The residual true claim is: *at fixed `embed_dim=48`, enlarging the GL gauge block GL3 → GL24 lowers
cross-entropy strictly monotonically (PPL 124.6 → 92.2) at a 245.76M-token budget, three-seed-robust — a
genuinely new phenomenon absent from either manuscript, along a well-defined structure-group design
axis, whose causal decomposition and efficiency framing remain open.* This is sharply falsifiable, and
blue abandons it if any of the following holds:

1. **The effect is a small-N / data-limited artifact.** A `blocks_K48` run at the matched 491.52M-token
   budget erases the GL3 → GL24 CE improvement. Then it was a D-slice artifact, not a held-D capacity
   effect.
2. **The gain is head-geometry recovery, not structured capacity.** A non-gauge multi-head baseline at
   `d_model=48`, swept 16 heads × `d_k=3` → 2 heads × `d_k=24` with no gauge machinery and no `phi_embed`
   growth, reproduces the monotone CE drop on its own. Then the gain is recovery from a pathological
   `d_k=3` regime plus head-count reduction (a largely free knob per Michel et al. 2019; Voita et al.
   2019), and the gauge framing is wrong.
3. **The gain is raw table capacity, not the generator algebra.** A plain `V × m` learned table of
   matched parameter size, at fixed head geometry, reproduces the GL24 gain. Then the effect is scalar
   capacity in group-shaped coordinates, and "gauge structure" earns no credit.
4. **The seed-noise floor is materially larger than reported.** If the true per-label PPL std is ~4-6
   rather than ≤ 1.08, the smallest 6.21-PPL step falls below two sigma and the detection result weakens.
   (Blue's numerical-analyst flags that the raw per-seed table was not inspected; the ≤ 1.08 figure is
   from `01_evidence.md:3,48`.)

## The single experiment that converts REMAND into a win

Blue and red name the same decisive artifact from opposite sides: a **matched 491.52M-token `blocks_K48`
run** (mechanically, `batch_size=64` at the same `max_steps`), which removes the entire cross-sweep
D-confound, **paired with a non-gauge matched-parameter `V × m` table control** at fixed head geometry,
which isolates the structured gauge capacity from raw table capacity. If the GL3 → GL24 improvement
survives at matched D *and* the plain table fails to reproduce the GL24 gain *and* the non-gauge
multi-head sweep does not reproduce it, then the phenomenon is a gauge-attributable, held-D capacity
effect, and only then is the efficiency/axis framing adjudicable against a compute axis. Absent that
battery, the honest disposition is REMAND to the real, falsifiable, seed-robust sub-claim above.

## Circularity check

Blue's case derives entirely from external canon (Bogen & Woodward, Pearl, Kaplan, Hoffmann, Shazeer,
Fedus, Cohen & Welling, Kondor & Trivedi, Nakahara, Jonckheere, Page, Hansen) and from code path:line
behavior. No step relies on `GL(K)_attention.tex`, `PIFB.tex`, `CLAUDE.md`, or any manuscript as
authority; those artifacts are the claim under evaluation, and "the manuscript derives gauge structure as
a capacity axis" would be circular. The geometric-deep-learning canon establishes only the *plausibility*
that group choice is a capacity axis; it does not discharge the missing control, and blue does not treat
it as corroboration of *this* effect's gauge-specificity.

## Expert attribution

All five consultants are used. The philosophy-of-science memo supplies the data-vs-phenomena and
association-vs-causation frame (Core attack) and polices blue's own circularity. The
implementation-engineer supplies the `phi_embed:167,682` code-truth and the decisive fixed-budget
verification across all five configs (Core attack, Defense). The ml-engineer supplies the Kaplan
non-embedding-N convention, the held-D Hoffmann reading, and the honest efficiency/exponent concessions
(Concession, Defense). The gauge-theorist supplies the change-of-structure-group case, the coboundary /
flat-holonomy verification, and the `(K, g)` orthogonality (Defense). The numerical-analyst supplies the
estimation-vs-detection separation and the reproduced detection statistics (Concession, Core attack). No
memo is discounted.
