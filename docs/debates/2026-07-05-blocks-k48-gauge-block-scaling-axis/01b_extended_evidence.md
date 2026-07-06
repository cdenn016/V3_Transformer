# Extended Evidence — blocks-k48-gauge-block-scaling-axis

Canon harvested by the debate panels beyond the neutral pack in `01_evidence.md`. Sources are external literature (source of truth) and in-repo code (canonical for code behavior). The user's manuscripts are under evaluation, not authority.

## Phase 2 — Blue-discovered canon

### Scaling laws and conditional computation (ml-engineer, transformer-ml, implementation-engineer)

- **Shazeer et al. 2017, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," ICLR 2017 (arXiv:1701.06538).** MoE up to 137B params; conditional computation increases capacity "without a proportional increase in computation," reporting ">1000x improvements in model capacity with only minor losses in computational efficiency." Anchors the total-vs-active legitimacy of `phi_embed`'s one-row-per-token read.
- **Fedus, Zoph & Shazeer 2021, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," JMLR 2022 (arXiv:2101.03961).** Frames "increase the parameter count while keeping the FLOPs per example constant" as "a separately important axis on which to scale"; "an outrageous number of parameters -- but a constant computational cost"; 7x pretraining speedup at fixed compute. The canonical statement that total-vs-active is a real axis, but only when paired with a compute (FLOP/wall-clock) axis.
- **Kaplan et al. 2020, "Scaling Laws for Neural Language Models" (arXiv:2001.08361).** The clean parameter power law is fit on the *non-embedding* parameter count, precisely because vocab/embedding tables do not scale like compute; data-/range-limited fits are slices, not infinite-data exponents. Supports both "active/compute is the right x-axis" and "a compressed-range exponent is not physical."
- **Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (Chinchilla, arXiv:2203.15556).** `L(N,D) = E + A/N^alpha + B/D^beta`; compute-optimal approx 20 tokens/param; Chinchilla 70B on 4x data beats Gopher 280B. A 2x difference in D between the two sweeps is a first-order confound for any cross-sweep absolute-loss or floor comparison.
- **Williams, Waterman & Patterson 2009, "Roofline: An Insightful Visual Performance Model," CACM.** Decode over the full vocabulary is memory-bandwidth / arithmetic-intensity bound; explains why the analytic FLOP proxy saturates on `2VK` and why wall-time, not the FLOP proxy, is the ground-truth compute signal.

### Transformer architecture — embedding tables and head partition (transformer-ml)

- **Vaswani et al. 2017, "Attention Is All You Need," NeurIPS, §3.2.2 and §3.4 (arXiv:1706.03762).** §3.2.2 fixes `d_model` and splits it into `h` heads of dim `d_model/h` at "similar" total cost — the exact head-count/head-width partition the `blocks_K48` sweep moves. §3.4 defines the learned input-embedding table (one active row per token).
- **Radford et al. 2019, "Language Models are Unsupervised Multitask Learners" (GPT-2).** Token-embedding matrix `[50257 x 768]` used as a lookup table, one active row per token; `V = 50257` byte-level BPE is exactly the project's `vocab_size`, anchoring the `phi_embed` embedding-table analogy.

### Equivariance / gauge structure as a design axis (gauge-theorist)

- **Cohen & Welling 2016, "Group Equivariant Convolutional Networks," ICML 2016 (arXiv:1602.07576).** Enlarging the symmetry group (p4 -> p4m) raises accuracy at fixed parameter count via structured weight sharing; the group is a capacity axis orthogonal to channel width. Supports "structured, not raw" capacity.
- **Kondor & Trivedi 2018, "On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups," ICML 2018 (arXiv:1802.03690).** An equivariant linear map is exactly a group convolution; the group determines admissible structure, so changing the group is a structural change, not added free capacity.
- **Weiler & Cesa 2019, "General E(2)-Equivariant Steerable CNNs," NeurIPS 2019 (arXiv:1911.08251).** Benchmarks group/representation choice as a design axis separate from width; kernel constraints for arbitrary reps reduce to irrep constraints.
- **Cohen, Weiler, Kicanaoglu & Welling 2019, "Gauge Equivariant Convolutional Networks and the Icosahedral CNN," ICML 2019 (arXiv:1902.04615).** Extends equivariance "beyond global symmetries to local gauge transformations"; the closest published analog to the `GL(g)`-per-block internal gauge here.
- **Bronstein, Bruna, Cohen & Velickovic 2021, "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" (arXiv:2104.13478).** The symmetry group is a primary design axis (the "5 Gs"), gauge symmetry included — frames the block-group knob as first-class.

### Philosophy of science — observation vs. interpretation, progressive problemshift (philosophy-of-science)

- **Lakatos, *The Methodology of Scientific Research Programmes* (1978).** A progressive problemshift must have "excess empirical content" predicting novel facts, some of which is corroborated. The operational test for whether the block-axis is a real increment or a re-labeled artifact.
- **Cartwright, *How the Laws of Physics Lie* (1983).** The generality-vs-truth tradeoff; realism about phenomenological laws, instrumentalism about fundamental laws. Licenses "reportable regularity" while denying "publishable exponent."
- **Hacking, *Representing and Intervening* (1983), "The life of experiment."** Experiment "has a life of its own, independent of theory"; robust observation is separable from a theory-laden metric — separates the CE effect (robust) from the efficiency proxy (metric-bound).
- **Popper, *Conjectures and Refutations* (1963).** "Irrefutability is not a virtue of a theory but a vice." The demarcation warrant for treating the falsifiers as the claim's scientific credential.

### In-repo code facts newly surfaced by the panel (implementation-engineer, gauge-theorist)

- **`vfe3/geometry/groups.py:144-152`, `vfe3/geometry/generators.py:96,103`** — `block_glk` builds `GL(d_head)^n_heads`; `generate_glk_multihead` returns `(n_heads * d_head^2, K, K)` generators, so `n_gen = n_heads * g^2 = 48*g` at fixed `K=48`. Each block carries the full `gl(g)` algebra (`g^2` generators): enlarging `g` raises per-block fiber-symmetry dimension 9 -> 576, a change of structure group, not a wider dense layer.
- **`vfe3/geometry/transport.py:4,6,183`** — the `phi_embed` rows are Lie-algebra coordinates driving covariant transport `Omega_ij = exp(phi_i . G) exp(-phi_j . G)` acting by the sandwich `mu -> Omega mu`, `Sigma -> Omega Sigma Omega^T`.
- **`vfe3/model/head_mixer.py:105-107`** — the mixer is `nn.Parameter(torch.zeros(m, m))` with `m = n_heads` for `block_glk`; it shrinks 16x16=256 (GL3) -> 2x2=4 (GL24) as blocks enlarge, opposite to the loss gain. Reached via `model.py:172` -> `block.py:96-97` (applied after E-step). `detach_e_step=false`, so the freeze footgun is not triggered.
- **`vfe3/model/prior_bank.py:181,625`** — under `use_prior_bank=False`, decode is the learned linear projection `logits = (M mu) @ W^T` with `output_proj_weight = nn.Parameter(torch.empty(V,K))`; the mixer composes with decode as `logits = mu @ (W M)^T`, a learned linear map co-moving with the sweep.
- **Convention correction (implementation-engineer):** `01_evidence.md:31` labels `fpt_decode approx 12M/token`; the code value is `fpt_decode = 2*V*K = 4,824,672 approx 4.82M/token` (`run_artifacts.py:625`). The approx 12M figure is `active_params_per_token`, a different quantity. Decode still dominates `fpt_estep` (approx 0.015-0.16M/token), so `est_flops_analytic` moves only 1.03x across the sweep; the decode-dominance conclusion is unaffected.
- **Co-moving learned tables beyond `phi_embed` (for any "gauge-structure per se" isolation):** `pos_phi="learned"` (a second learned gauge/positional table), `decode_bias=true` (learned per-vocab bias `(V,)` under linear decode, `prior_bank.py:183-189`), and `learnable_r=true` with `r_update_mode=gradient` (learned hyper-prior centroid) are all active in this config.

## Phase 2 — Red-discovered canon

Harvested by the 5 red consultants (philosophy-of-science, ml-engineer, implementation-engineer,
transformer-ml, numerical-analyst). Items that also appear in the blue section above are marked
"(reinforced)"; the rest are new to the record. None is from the project's own manuscript.

### Transformer head-geometry confound (new — not in the blue section)

- **Vaswani et al. 2017, "Attention Is All You Need," NeurIPS 2017, §3.2.2.**
  URL: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf —
  "d_k = d_v = d_model/h = 64 ... Due to the reduced dimension of each head, the total computational
  cost is similar to that of single-head attention with full dimensionality." Head count and head
  dimension are two faces of one `d_model = h·d_k` axis. §3.2.1 footnote gives the `1/sqrt(d_k)`
  temperature from the variance argument. (Blue cited Vaswani for the embedding table; red cites the
  multi-head identity that the block sweep traverses.)
- **Michel, Levy, Neubig 2019, "Are Sixteen Heads Really Better than One?" NeurIPS 2019,
  arXiv:1905.10650.** URL: https://arxiv.org/abs/1905.10650 — "a large percentage of attention heads
  can be removed at test time without significantly impacting performance ... some layers can even be
  reduced to a single head." Head count is a redundancy-laden knob.
- **Voita, Talbot, Moiseev, Sennrich, Titov 2019, "Analyzing Multi-Head Self-Attention," ACL 2019,
  arXiv:1905.09418.** URL: https://arxiv.org/abs/1905.09418 — "pruning 38 out of 48 encoder heads
  results in a drop of only 0.15 BLEU." Quantifies the head-count / head-width trade-off.

### Power-law fit-conditioning canon (new — the exponent-not-identified argument)

- **Clauset, Shalizi & Newman 2009, "Power-Law Distributions in Empirical Data," SIAM Review
  51(4):661-703; arXiv:0706.1062, §3 + Appendix A.** URL: https://epubs.siam.org/doi/10.1137/070710111
  — "least-squares linear regression on the logarithm ... generate[s] significant systematic errors"
  and "in each case where the estimate is biased, the corresponding error estimate gives no warning of
  the bias." Canonical warning against reading a power-law exponent off a log-log line.
- **Stumpf & Porter 2012, "Critical Truths About Power Laws," Science 335(6069):665-666.**
  URL: https://www.science.org/doi/10.1126/science.1216142 — a credible empirical power law should span
  at least two orders of magnitude. (Quoted as reported via secondary summary.)
- **Belsley, Kuh & Welsch 1980, "Regression Diagnostics," Wiley.** Condition-index / variance-
  decomposition diagnostics: a scaled condition index > 100 signals estimates carrying substantial
  error; near-collinear predictors yield unstable, high-variance coefficients. (Threshold language as
  standardly attributed.)

### Philosophy-of-science canon (new additions beyond the blue Lakatos/Cartwright/Hacking/Popper set)

- **Duhem, P. 1906, "The Aim and Structure of Physical Theory," p. 187** (Eng. tr. Wiener 1954). Via SEP
  "Underdetermination": https://plato.stanford.edu/entries/scientific-underdetermination/ — "the
  physicist can never subject an isolated hypothesis to experimental test, but only a whole group of
  hypotheses ... the experiment does not designate which one should be changed." The canonical
  confounded-manipulation statement for the 3-knob co-variation.
- **Quine, W. V. O. 1951, "Two Dogmas of Empiricism," The Philosophical Review 60(1), p. 42.** Via SEP
  "Underdetermination" — "The unit of empirical significance is the whole of science." Holist
  underdetermination of the entangled toggle stack.
- **Worrall, J. 2014, "Prediction and accommodation revisited," Stud. Hist. Phil. Sci. 45, 54-61.**
  DOI: 10.1016/j.shpsa.2013.10.001 — the "no double use" principle: one body of evidence cannot both fix
  a theory's free content and count as independent support for the theory so fixed. Disciplines reading
  the exponent off the same CE data it is meant to characterize.
- **Popper 1963, Lakatos 1978, Cartwright 1999** — (reinforced) already in the blue section; red anchors
  the observer-dependent-exponent falsifiability attack (Popper), the progressive-vs-degenerating audit
  (Lakatos), and the nomological-machine over-export argument (Cartwright, *Dappled World* p. 50).

### Scaling-law / conditional-computation canon (reinforced; red-specific excerpts)

- **Fedus, Zoph, Shazeer 2021, "Switch Transformers," arXiv:2101.03961, §1** — (reinforced) red's
  operative excerpt: "we investigate a fourth axis: increase the parameter count while keeping the
  floating point operations (FLOPs) per example constant." A total-vs-active efficiency claim is licensed
  only when a genuine compute axis is held fixed.
- **Kaplan et al. 2020, arXiv:2001.08361, §1.2/§4** — (reinforced) red's operative excerpt: performance
  "enters a regime of diminishing returns if either N or D is held fixed while the other increases," and
  the finite-data law `L(D) = (Dc/D)^{α_D}`. A fixed-token sweep is a D-conditioned slice.
- **Hoffmann et al. 2022, arXiv:2203.15556, Approach 3** — (reinforced) `L̂(N,D) = E + A/N^α + B/D^β`,
  E=1.69, A=406.4, B=410.7, α=0.34, β=0.28; the additive-in-D floor makes any cross-sweep floor-E
  comparison across the 245.76M vs 491.52M token budgets non-identified.

### Red-discovered in-repo code facts (code-truth, for the panel)

- `vfe3/run_artifacts.py:616-620` — under this config (`use_prior_bank=False`, `model_channel=True` at
  `:612-613`) the active formula resolves to `active = 5·V·K + 2·K + n_gen`; reached via the call at
  `:757`. Base `5·50257·48 + 2·48 = 12,061,776` matches the CSV to the integer; total variation across
  the sweep is `+n_gen = +1,008` (+0.008%). The flatness is definitional at fixed `V,K`.
- `vfe3/run_artifacts.py:611` sets `d_head = K/n_blocks` with `n_blocks = n_heads`; at fixed `K=48`,
  block width `g = d_head` and `n_heads = 48/g`. `vfe3/model/free_energy.py:42-54` `attention_tau` returns
  `tau = kappa·sqrt(d_energy)` with `d_energy = d_head`, docstring "kappa=1 -> Vaswani recovery" — the
  project's own softmax names `g` as `d_k`.
- `vfe3/run_artifacts.py:625-627` — `fpt_decode = 2·V·K = 4,824,672` (fixed) dominates
  `fpt_estep = L·T·(2·N·K + 2·N·g²)` (14,592 → 159,744), so `est_flops_analytic` moves 1.030× while its
  transport sub-term `2·N·g²` grows 64×. Decode-dominance masks the scaled axis. Empirical `wall_time_s`
  is U-shaped (seed-6: GL3=6168.6, GL8=4634.2 min, GL24=11070.6; GL24 slowest, 2.39× the GL8 minimum).
- Reproduced conditioning test (numpy/scipy): offset-law fit `CE = E + A·N^{-α}` gives α = 0.929 vs
  n_params (corr(E,α)=+0.976) and α = 0.180 vs n_gen (corr(E,α)=+0.998) on identical CE data — a 5.15×
  swing, with the E/α correlation near +1 (floor and exponent not separately estimable). Dynamic range:
  n_params 0.559 decades, n_gen 0.903 decades — both below the ≥2-decade credibility floor.

## Phase 3 — Red-discovered canon (rebuttal)

Harvested by the 5 red rebuttal consultants (philosophy-of-science, gauge-theorist, ml-engineer,
implementation-engineer, transformer-ml). New items sharpen or extend the Phase 2 record with
exact-page verbatim excerpts; none is from the project's own manuscript.

### Confounded-manipulation / causal-identification canon (philosophy-of-science, new)

- **Mill, J. S. 1843 [1882], *A System of Logic*, bk. III ch. 8, p. 483.** Method of Difference —
  two instances alike in "every circumstance … save one," that one being the effect, cause, or an
  indispensable part of the cause. The canonical single-difference requirement: a manipulation that
  moves three variables at once (n_params 3.62×, n_heads 16→2, block width g) identifies none of them.
- **Woodward, J. 2003, *Making Things Happen*, ch. 2 (interventionist causation).** Via SEP
  "Causation and Manipulability" (https://plato.stanford.edu/entries/causation-mani/) — a genuine
  intervention on X sets X "and only" X; a fat-handed intervention that also moves Z leaves the X→Y
  claim unidentified. The modern formalization of Mill; the un-run non-gauge control is the missing
  clean intervention.
- **Lakatos, I. 1978, *The Methodology of Scientific Research Programmes*, pp. 33–34.** The two-part
  progressive test verbatim: a problemshift is progressive only if it is both theoretically
  progressive (predicts *novel* facts) and empirically progressive (some novel prediction is
  *corroborated*). A discovered-then-narrated regularity (accommodation) whose "excess content" is
  equally consistent with "any added table" is not corroborated excess content *of the gauge hard core*.

### Flat-connection / structured-vs-raw-capacity canon (gauge-theorist, new)

- **Cohen & Welling 2016, "Group Equivariant Convolutional Networks," ICML (arXiv:1602.07576),
  verbatim.** "G-convolutions increase the expressive capacity of the network without increasing the
  number of parameters," via "a substantially higher degree of weight sharing." The equivariance
  precedent for group-as-capacity-axis is explicitly a *fixed-parameter, weight-sharing* phenomenon —
  the opposite of `blocks_K48`, whose params grow 3.62× (99.7% in `phi_embed`) as the group enlarges.
- **Nakahara 2003, *Geometry, Topology and Physics* (2nd ed.), §10.2 (Holonomy) / §10.3 (Curvature).**
  Flat connection ⟺ zero curvature ⟺ trivial contractible-loop holonomy. Under `transport_mode='flat'`,
  `Omega_ij = exp(phi_i·G)exp(-phi_j·G)` is a coboundary of the 0-cochain `g_i = exp(phi_i·G)`: it
  satisfies `Omega_ij Omega_jk = Omega_ik` automatically, holonomy is the identity, curvature is
  identically zero. The connection's only content is a per-token learned frame `exp(phi_i·G) ∈ GL⁺(48)`,
  not rich gauge-field structure.
- **Bleecker 1981, *Gauge Theory and Variational Principles*, Ch. 3.** A coboundary connection has
  trivial holonomy; physical (gauge-invariant) content lives in the curvature, which vanishes on the
  flat path exercised here.
- **Weiler, Forré, Verlinde & Welling 2021, "Coordinate Independent Convolutional Networks"
  (arXiv:2106.06020), verbatim.** "A simultaneous demand for coordinate independence and weight
  sharing is shown to result in a requirement on the network to be equivariant under local gauge
  transformations." Ties gauge equivariance itself to weight sharing (fewer parameters), reinforcing
  that the genuine gauge-structural axis is a fixed/reduced-parameter one.

### Scaling-law confound canon (ml-engineer, sharpened verbatim)

- **Kaplan et al. 2020, arXiv:2001.08361, §1.3 + §3.1 + §3.2 (verbatim).** "N — the number of model
  parameters, excluding all vocabulary and positional embeddings." §3.1: "Transformer performance
  depends very weakly on the shape parameters n_layer, n_heads, and d_ff when we hold the total
  non-embedding parameter count N fixed." Two-pronged: `phi_embed (V, n_gen)` is the excluded
  vocab/embedding table (carrying 99.7% of the sweep's param growth), and `n_heads` (16→2) is a *weak*
  knob at fixed N — so the clean-axis power law removes exactly the axis the claim plots on.
- **Hoffmann et al. 2022, Chinchilla, arXiv:2203.15556, Eq. 2 (verbatim constants).**
  `L̂(N,D) = E + A/N^α + B/D^β`, E=1.69, A=406.4, B=410.7, α=0.34, β=0.28; "for every doubling of
  model size the number of training tokens should also be doubled." The additive `B/D^β` term makes any
  cross-sweep comparison across 245.76M (blocks) vs 491.52M (width, exactly 2×) tokens non-identified —
  including the "complementary to width" conjunct and blue's own "suggestive" GL24-vs-K50 point.
- **Fedus, Zoph & Shazeer 2021, Switch Transformers, arXiv:2101.03961 (abstract, verbatim).**
  "outrageous numbers of parameters — but a constant computational cost." The total-vs-active axis is
  legitimate only at *constant compute*; `blocks_K48` buys its best loss (GL24) at the *highest*
  wall-clock (slowest run) with transport compute up 64×, so the antecedent fails.

### Multi-head partition / embedding-scale confound canon (transformer-ml, sharpened verbatim)

- **Vaswani et al. 2017, §3.2.2 (verbatim).** "d_k = d_v = d_model/h = 64 … Due to the reduced
  dimension of each head, the total computational cost is similar to that of single-head attention with
  full dimensionality." The `blocks_K48` sweep slides along this `d_model = h·d_k` partition of a fixed
  `d_model=48` (16 heads at d_k=3 → 2 heads at d_k=24), a known architectural knob co-moving with the
  gauge block.
- **Michel, Levy & Neubig 2019, arXiv:1905.10650 (verbatim).** "a large percentage of attention heads
  can be removed at test time without significantly impacting performance … some layers can even be
  reduced to a single head." Head count is redundancy-laden.
- **Voita et al. 2019, ACL P19-1580 (verbatim).** "pruning 38 out of 48 encoder heads results in a
  drop of only 0.15 BLEU." Quantifies the head-count/head-width trade the sweep traverses.

### Red-discovered in-repo code facts (rebuttal; code-truth)

- **`vfe3/model/model.py:375-376`** — `pos_phi_free` is a *second* size-co-moving learned table under
  `pos_phi='learned'`, shape `(max_seq_len=128, n_gen)`, growing 18,432 → 147,456 params across
  GL3→GL24 alongside `phi_embed`. With `detach_e_step=False` and `e_step_gradient='unroll'` it is not
  frozen — it trains. So the sweep co-moves at least two learned tables plus fixed-size
  `output_proj_weight`/`output_proj_bias`/`learnable_r`; "gauge structure per se" is not isolated in code.
- **`vfe3/run_artifacts.py:616-620` (re-verified under active config)** — `use_prior_bank=False` +
  `model_channel=True` branches both fire (`lambda_h=0.25>0`), resolving `active = 5·V·K + 2·K + n_gen`.
  Base `5·50257·48 + 2·48 = 12,061,776`; total sweep variation `+n_gen = +1,008` (+0.0084%). The
  flatness is an algebraic identity at fixed `V,K`, not an efficiency finding.
- **`vfe3/run_artifacts.py:611,626` (re-verified)** — `d_head = K/n_blocks`, `n_blocks = n_heads`, so
  at fixed `K=48`, `d_head = g`; transport sub-term `2·N·g²` runs 2,304 → 147,456 (g²: 9 → 576, 64×).
  `fpt_decode = 2·V·K = 4,824,672` (fixed) dominates `fpt_estep` (14,592 → 159,744), so
  `est_flops_analytic` moves only 1.030× while transport moves 64×. Convention correction extends:
  `01_evidence.md:31` mislabels ~12M as `fpt_decode` (code value 4.82M; 12M is
  `active_params_per_token`), and its stated `fpt_estep ~0.1–1.3M` is also wrong (code: 0.015–0.16M).
- **Williams, Waterman & Patterson 2009, "Roofline," CACM** (reinforced) — full-vocab decode is
  memory-bandwidth / arithmetic-intensity bound, explaining why the analytic FLOP proxy saturates on
  `2VK` and why empirical wall-time (U-shaped, GL24 slowest) is the ground-truth compute signal.

## Phase 3 — Blue-discovered canon (rebuttal)

Harvested by the 5 blue rebuttal consultants (philosophy-of-science, implementation-engineer,
ml-engineer, gauge-theorist, numerical-analyst). None is from the project's own manuscript.

### Data-vs-phenomena and association-vs-causation (philosophy-of-science)

- **Bogen, J. & Woodward, J. 1988, "Saving the Phenomena," The Philosophical Review 97(3):303-352.**
  URL: https://www.jstor.org/stable/2185445 — pp. 306, 314: "Phenomena are detected through the use of
  data, but in most cases are not observable in any interesting sense of that term"; "we need to
  distinguish what theories explain (phenomena …) from what is uncontroversially observable (data)."
  Separates the effect (a detected phenomenon) from the efficiency proxy / fitted exponent (the
  theory-laden overlay red refutes).
- **Pearl, J. 2009, "Causal Inference in Statistics: An Overview," Statistics Surveys 3:96-146, p. 99.**
  URL: https://projecteuclid.org/journals/statistics-surveys/volume-3/issue-none/Causal-inference-in-statistics-An-overview/10.1214/09-SS057.full
  — "an associational concept is any relationship that can be defined in terms of a joint distribution
  of observed variables, and a causal concept is any relationship that cannot be defined from the
  distribution alone." Red's Duhem-Quine confound defeats the interventional query P(CE | do(gauge
  structure)); it leaves the observed association P(CE | block size) intact.

### Scaling-law conventions (ml-engineer)

- **Kaplan et al. 2020, arXiv:2001.08361, §2.1 + Fig. 6 (verbatim, upgrading the Phase-2 paraphrase).**
  §2.1: "We use N to denote the model size, which we define as the number of non-embedding parameters."
  Fig. 6: "When we exclude embedding parameters, the performance of models with different depths
  converge to a single trend." Excluding the (V, n_gen) `phi_embed` table from the per-token working set
  is the prescribed convention, not a dodge.
- **Besiroglu et al. 2024, "Reconciling Kaplan and Chinchilla Scaling Laws," arXiv:2406.12907.**
  URL: https://arxiv.org/abs/2406.12907 — attributes much of the Kaplan/Chinchilla exponent gap to
  Kaplan's exclusion of embedding parameters. Cuts both ways: confirms excluding embedding tables is a
  real, load-bearing convention (supports blue's bookkeeping) and that include-vs-exclude materially
  moves the fitted exponent (supports blue's concession that the blocks exponent is convention-dependent).

### Structure-group / gauge design axis (gauge-theorist)

- **Cohen & Welling 2016, arXiv:1602.07576, abstract (verbatim):** "G-convolutions increase the
  expressive capacity of the network without increasing the number of parameters." Enlarging the
  symmetry group is a capacity axis via structured weight sharing, distinct from channel width — the
  analog of enlarging g at fixed embed_dim. (Blue reads this as licensing the *structural-axis* framing;
  red reads the fixed-parameter clause as adverse to a params-growing sweep. The clause is on the record
  for both.)
- **Nakahara, Geometry, Topology and Physics, 2nd ed. (2003), Ch. 9 (structure group, transition-function
  cocycle t_ij t_jk = t_ik) and Ch. 10 (connections, congruence gauge action, flat = trivial holonomy).**
  Primary-source anchor: `Omega_ij = g_i g_j^{-1}` is a coboundary satisfying the cocycle condition and
  inducing a flat connection (holonomy `Omega_ij Omega_jk Omega_ki = I`), matching `transport_mode='flat'`;
  enlarging `GL(g)^(48/g)` (fiber dim `48·g`: 144 → 1152) is a change of structure group in the bundle sense.

### Estimation-vs-detection and ordered-alternatives statistics (numerical-analyst)

- **Jonckheere, A. R. 1954, "A distribution-free k-sample test against ordered alternatives," Biometrika
  41(1-2):133-145.** URL: https://academic.oup.com/biomet/article-abstract/41/1-2/133/456667 — canonical
  distribution-free test of H_0 (identical distributions) against a monotone/ordered alternative;
  detection of an ordering with no parametric exponent.
- **Page, E. B. 1963, "Ordered Hypotheses for Multiple Treatments," JASA 58(301):216-230.** URL:
  https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500843 — rank statistic for a monotone
  relationship across ordered treatments assuming "data of only ordinal strength."
- **Hansen, P. C. 1998, Rank-Deficient and Discrete Ill-Posed Problems, SIAM.** The condition number
  governs amplification of data perturbations into the recovered inverse-problem parameters; a
  rank-deficient forward map leaves the solution undetermined while leaving well-posed functionals of the
  data (the ordering) unaffected — the numerical statement of the estimation-vs-detection split.

### Blue-surfaced code / statistical facts (implementation-engineer, numerical-analyst)

- **Fixed within-sweep budget (all five blocks_K48 cells).** Every s6 config carries identical
  `max_steps=60000`, `batch_size=32`, `max_seq_len=128`, `grad_accum_steps=1`, `dataset=wikitext-103`,
  giving `60000·32·128 = 245,760,000` tokens per cell (verified by arithmetic on all five configs). The
  2× cross-sweep gap is mechanically a `batch_size` difference: `grow_K_GL10` uses `batch_size=64` →
  491.52M at the same `max_steps` and `data_sha256`. Hoffmann's `B/D^β` floor term is therefore
  byte-identical across the blocks sweep and cannot generate the within-sweep CE ordering; the D-slice
  confound is strictly cross-sweep. The matched-D blocks run (`batch_size=64`, 491.52M) is the single
  missing artifact.
- **Reproduced detection statistics (numpy/scipy) on seed-averaged PPL 124.57 → 112.41 → 106.20 → 99.77
  → 92.15, per-label std ≤ 1.08.** Per-step drops 12.16, 6.21, 6.43, 7.62 PPL; smallest step (GL6→GL8)
  is 4.07 seed-σ (conservative single-obs SE) to 7.04 σ (seed-mean SE). Exact permutation probability of
  all three seeds independently realizing the strict decreasing order: `(1/5!)^3 = (1/120)^3 = 5.8e-7`.
  Spearman ρ = −1.000, Kendall τ = −1.000. Log-log design condition number 306 vs n_params (> Belsley-
  Kuh-Welsch threshold 100); both offset-law fits return R² ≈ 0.998, so R² cannot discriminate the two
  incompatible exponents. Caveat: the raw per-seed PPL table was not inspected; the ≤ 1.08 std and
  "seed-robust" are from `01_evidence.md:3,48`.
