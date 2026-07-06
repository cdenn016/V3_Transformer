# Red Opening — blocks-k48-gauge-block-scaling-axis

Side: RED (falsify). Mode: implementation. Round: opening. Panel: philosophy-of-science,
ml-engineer, implementation-engineer, transformer-ml, numerical-analyst.

## Steelman

The strongest honest version of the claim is this. The `blocks_K48` sweep records a monotone,
three-seed-robust cross-entropy decrease (PPL 124.6 at GL3 to 92.2 at GL24) as the GL gauge block
widens at a fixed `embed_dim=48`. That improvement is empirically real and is not in either
manuscript, so it is a genuinely new direction for the program. The per-token active-parameter
working set holds near `12.06M` across the whole sweep while total parameters grow `3.62x`, which is
exactly the total-vs-active signature of conditional-computation models: a large table contributes one
active row per token, so counting only active parameters is the standard and defensible bookkeeping
(Shazeer et al. 2017; Fedus, Zoph & Shazeer 2021). On that reading, `blocks_K48` buys accuracy while
the honest per-token working set stays flat, and it reaches the width-scaling frontier's cross-entropy
on half the token budget, which if anything under-reports its efficiency. A defender concludes that
this is a real, parameter-efficient scaling axis complementary to width-scaling, worth reporting as
such.

We accept the empirical core of that steelman: the CE decrease is real, monotone, and seed-robust. We
reject everything the claim builds on top of it.

## Position

The claim is a five-fold conjunction — genuine AND publishable AND parameter-efficient AND a distinct
scaling axis AND not an artifact-plus-confound — and it fails at three independent joints, any one of
which is sufficient to defeat the compound proposition. First, the "essentially constant `12.06M`
active working set" is a definitional property of a closed-form proxy at fixed vocabulary and width, not
a measured efficiency; the one compute axis that would license the efficiency claim under the very
literature the claim invokes (Shazeer/Fedus: hold FLOPs per example constant) is adverse, because real
transport cost grows `64x` and wall-time is U-shaped with the best-loss point being the slowest run.
Second, "distinct scaling axis" is not a well-posed claim: on identical CE data the fitted exponent
swings `5x` depending on the analyst's free choice of x-axis (0.93 vs `n_params`, 0.18 vs `n_gen`,
degenerate vs FLOPs), with a `n_params` confidence interval of [0.07, 1.73] that spans from "no scaling"
to "super-linear," because the fit is ill-conditioned over a compressed `3.62x` dynamic range. A
monotone trend with no axis-invariant exponent is not a scaling law in the Kaplan/Hoffmann sense. Third,
the causal attribution to "gauge structure per se" is confounded by construction: enlarging the block at
fixed `embed_dim` is, by the code's own definitions, the Vaswani multi-head partition `d_k = d_model/h`
run backward — it simultaneously moves head width (`3 → 24`), head count (`16 → 2`), total capacity
(`3.62x`, 99.7% of it inside a token-indexed lookup table), and the softmax temperature (`~ sqrt(g)`),
and no non-gauge control was ever trained. What survives is the narrow sub-claim the clause itself names
as its fallback: at fixed `embed_dim=48`, block enlargement lowers CE monotonically at a 245.76M-token
budget. That is a REMAND, not the compound claim.

## Evidence

### 1. The flat working set is definitional, and the compute axis that would license it is adverse

The implementation-engineer traced the proxy to source. At `vfe3/run_artifacts.py:616-620`, under this
config (`use_prior_bank=False` fires the `+= V·K` branch at `:617-618`; `model_channel=True`, resolved
at `:612-613` because `lambda_h=0.25>0`, fires the `+= 2·V·K` branch at `:619-620`), the active count
resolves to `active = 5·V·K + 2·K + n_gen`, reached at the call site `:757`. With `V=50257` and `K=48`
held fixed sweep-wide, the only varying term is `+ n_gen`. The base `5·50257·48 + 2·48 = 12,061,776`
matches the recorded CSV to the integer, and the total variation across GL3 to GL24 is `+n_gen = +1,008`
(+0.008%). The "essentially constant working set" is therefore what the formula must return when `V` and
`K` are fixed; it is an algebraic identity, not an efficiency measurement. The `50M` of growth lives
almost entirely in `phi_embed`, a `(V, n_gen)` table at `vfe3/model/prior_bank.py:167` read one row per
token at `:682` — 99.7% of the `3.62x` total-parameter growth, which the proxy excludes by design.

The total-vs-active distinction is legitimate, but the literature the claim relies on attaches a
condition to it. Fedus, Zoph & Shazeer (2021) define the axis as "increase the parameter count while
keeping the floating point operations (FLOPs) per example constant," and their comparisons are
FLOP-matched; Shazeer et al. (2017) license conditional computation because it increases capacity
"without a proportional increase in computation." An active-parameter efficiency claim is warranted only
when paired with a genuinely flat compute axis. Here it is not. The implementation-engineer and
numerical-analyst independently recomputed the FLOP accounting from `vfe3/run_artifacts.py:611,625-627`:
the transport sub-term `2·N·d_head² = 2·N·g²` (with `d_head = g` at fixed `K=48`) grows `64x`
(2,304 → 147,456 FLOP/token), yet `est_flops_analytic` moves only `1.030x` because the fixed decode term
`fpt_decode = 2·V·K = 4,824,672` FLOP/token swamps it. (The evidence pack's line 31 gloss of
"~12M/token" for decode conflates the FLOP proxy, 4.82M, with `active_params_per_token`, 12.06M; the
decode-dominance conclusion holds under either number.) The analytic flatness is a decode-dominance
artifact of the proxy, blind to the axis being scaled. The empirical ground-truth compute, `wall_time_s`,
is U-shaped and non-monotone (seed-6: GL3 = 6168.6, GL8 = 4634.2 minimum, GL24 = 11070.6 seconds), and
GL24 — the best-loss point — is the slowest run, `2.39x` the GL8 minimum. On the honest compute axis the
efficiency reverses.

### 2. "Distinct scaling axis" is not well-posed: no axis-invariant exponent, and the fit is ill-conditioned

A scaling axis in the Kaplan/Hoffmann sense carries an exponent that is a stable property of the
loss-versus-resource law, not of the plot. On identical `blocks_K48` CE data the offset-law exponent is
0.929 versus `n_params`, 0.181 versus `n_gen`, and degenerate (R² = 0.17) versus analytic FLOPs
(`01_evidence.md`, line 61). The numerical-analyst reproduced this with scipy: fitting
`CE = E + A·N^{-α}` gives α = 0.929 vs `n_params` and α = 0.180 vs `n_gen` — a `5.15x` swing on the same
y-values — with the floor–exponent correlation `corr(E, α)` at +0.976 and +0.998, near +1, meaning the
floor `E` and the exponent `α` are not separately estimable. The dynamic range is 0.559 decades for
`n_params` and 0.903 decades for `n_gen`, both under one order of magnitude. Clauset, Shalizi & Newman
(2009) warn that least-squares fits on the logarithm "generate significant systematic errors" and that
"the corresponding error estimate gives no warning of the bias"; Stumpf & Porter (2012) hold that a
credible empirical power law should span at least two orders of magnitude. `blocks_K48` offers a third to
a half of one. The reported bootstrap CI of [0.07, 1.73] — a factor-25 span crossing α = 1 — is the
honest width of an undetermined exponent. By contrast, `grow_K_GL10` spans 1.08 decades and its exponent
is axis-robust (0.558 / 0.555 / 0.569), which is what a real scaling axis looks like.

The cross-sweep half of "distinct axis complementary to width-scaling" fails on a second, independent
ground. The ml-engineer shows that under Hoffmann et al.'s (2022) fitted law `L(N,D) = E + A/N^α + B/D^β`,
loss depends jointly on parameters `N` and tokens `D`, so the exact 2.0x token-budget gap between
`blocks_K48` (245.76M) and `grow_K_GL10` (491.52M, same `data_sha256`) places the two sweeps on different
D-slices of the loss surface. Every cross-sweep reading in the evidence pack — matched-active, matched-
params, "GL24 sits on the grow frontier" — compares points across D-slices and is non-identified, exactly
what Kaplan et al. (2020) warn against when they state that performance "enters a regime of diminishing
returns if either N or D is held fixed while the other increases." The matched-budget run that would
remove the confound, `blocks_K48` at 491.52M tokens, was never done (`01_evidence.md`, line 79). We do not
argue that the token gap inflates or deflates the blocks loss in a known direction; we argue that it makes
the "complementary" comparison unidentifiable, which is enough.

### 3. The causal attribution to "gauge structure" is confounded: the multi-head trade-off run backward, with no control

The transformer-ml consultant establishes from code that "enlarging the GL gauge block at fixed
`embed_dim`" is not an isolated gauge manipulation. At `vfe3/run_artifacts.py:611`, `d_head = K/n_blocks`
with `n_blocks = n_heads`, so at fixed `K=48` the block width `g` equals `d_head` and sets `n_heads = 48/g`.
GL3 to GL24 is therefore exactly 16 heads of `d_k=3` moving to 2 heads of `d_k=24` at fixed `d_model=48` —
the Vaswani et al. (2017, §3.2.2) partition `d_k = d_model/h` traversed in reverse. The project's own
softmax confirms the identification: `vfe3/model/free_energy.py:42-54` returns `tau = kappa·sqrt(d_energy)`
with `d_energy = d_head`, and its docstring reads "kappa=1 -> Vaswani recovery," so `g` is literally the
`d_k` of the `1/sqrt(d_k)` construction, and block enlargement silently re-tunes the softmax temperature by
`sqrt(g)`. The GL3 to GL24 gain thus co-moves with four bundled changes: recovery from a pathologically
narrow `d_k=3` toward a reasonable head width (Vaswani's own `d_k=64` is the reference), a reduction in head
count into a regime the literature shows is largely free (Michel et al. 2019: heads "can be removed at test
time without significantly impacting performance ... some layers can even be reduced to a single head";
Voita et al. 2019: "pruning 38 out of 48 encoder heads results in a drop of only 0.15 BLEU"), a `3.62x`
total-parameter inflation that is 99.7% a sparse embedding table, and a softmax re-tuning.

The philosophy-of-science consultant supplies the frame. The design co-varies three knobs at once —
`n_params` (3.62x), `n_heads` (16 → 2), block width `g` — and no discriminating control was trained: no
plain `V × m` learned table of matched size, no matched-parameter dense head-mix (`01_evidence.md`, lines
51, 80). Attributing the gain to gauge structure while several manipulations move together is the
Duhem-Quine problem stated verbatim: "the experiment does not designate which one should be changed"
(Duhem 1906, p. 187). The head mixer, an on-toggle here, moves the wrong way — it shrinks 16×16 to 2×2 as
blocks enlarge (`vfe3/model/head_mixer.py:105-106`), opposite to the gain — which rules it out as the driver
but confirms that uncontrolled learned-linear capacity co-moves with the sweep. And "complementary to
width-scaling" exports a regularity produced inside one specific machine (245.76M tokens, single layer,
single E-step, this toggle stack) to a different machine (491.52M tokens) where it was never run, which
Cartwright (1999, p. 50) identifies as inferential overreach beyond the nomological machine that grounds
the law. Finally, an "axis" whose exponent takes any value in [0.07, 1.73] depending on the chosen x-axis
forbids almost no observation, and by Popper's (1963) criterion — "irrefutability is not a virtue of a
theory but a vice" — that is scientific weakness, not strength.

### Expert attribution

All five memos are used. The implementation-engineer supplies the code-truth of the proxy formula and the
`phi_embed` table (section 1). The ml-engineer supplies the Shazeer/Fedus FLOP-matched standard, the
Chinchilla D-slice non-identification, and the data-limited-slice reading (sections 1 and 2). The
numerical-analyst supplies the reproduced ill-conditioning of the exponent fit and the FLOP-masking
arithmetic (sections 1 and 2). The transformer-ml consultant supplies the `d_k = d_model/h` head-geometry
identification and the head-redundancy literature (section 3). The philosophy-of-science consultant supplies
the compound-conjunction frame, the Duhem-Quine confound, the Cartwright over-export, and the Popper
falsifiability verdict (section 3 and Position). No memo is discounted.

## Falsification conditions

We concede the compound claim, and withdraw to accepting a genuine parameter-efficient gauge-block scaling
axis, if any of the following is shown.

1. **Matched-budget survival plus a determined exponent.** A `blocks_K48` run at the matched 491.52M-token
   budget preserves the GL3 to GL24 CE improvement AND the fitted exponent tightens to an axis-invariant
   value (CI width under 2x, agreeing across `n_params` / `n_gen` / a compute axis to within its CI). Then it
   is a power law, not a monotone trend, and the D-slice confound is removed.

2. **A flat honest compute axis.** An empirical compute measure — wall-clock GPU-seconds or a
   transport-inclusive profiler FLOP count — is shown flat or monotone-decreasing across GL3 to GL24, rather
   than U-shaped with GL24 the slowest run. Then the Shazeer/Fedus efficiency warrant is met on real compute,
   not on a decode-dominated proxy.

3. **A non-gauge control that fails, and a gauge control that succeeds.** A standard multi-head baseline at
   `d_model=48` swept from 16 heads × `d_k=3` to 2 heads × `d_k=24`, without gauge machinery or `phi_embed`
   growth, shows no monotone CE improvement (ruling out narrow-head recovery and head-count reduction as the
   driver); AND a plain `V × m` learned table of matched parameter size fails to reproduce the GL24 gain at
   fixed head geometry (isolating the generator algebra from raw table capacity). Then the gain is
   gauge-specific, and the causal conjunct is corroborated rather than confounded.

Absent all three, the correct disposition is REMAND to the narrow, well-formed, locally corroborated
sub-claim: at fixed `embed_dim=48`, enlarging the GL gauge block lowers cross-entropy monotonically across
GL3 to GL24 on a 245.76M-token budget, robust across three seeds — a gauge-structure ablation at fixed
width, not a demonstrated parameter-efficient scaling axis complementary to width-scaling.
