# Red Rebuttal — blocks-k48-gauge-block-scaling-axis (Phase 3)

Side: RED (falsifying). Panel: philosophy-of-science, gauge-theorist, ml-engineer,
implementation-engineer, transformer-ml. All five memos are cited below; none discounted.

Blue's opening is a partial defense that already surrenders two of the four conjuncts. Red's
position is simple: on a claim written as a conjunction, surrendering conjuncts is losing, not
winning. What Blue defends is real but is a *different, narrower* proposition than the one on
trial. The full claim — "genuine AND publishable AND parameter-efficient AND a distinct scaling
axis, NOT artifact+confound" — fails as stated. The correct verdict is REMAND to the narrow
sub-claim, which is a refutation of the compound claim, not a BLUE_WINS.

## Concession

Red concedes, without reservation, everything Blue actually proved, and stops there.

1. **The within-sweep effect is real.** PPL 124.57 / 112.41 / 106.20 / 99.77 / 92.15 across
   GL3→GL24 is strictly monotone, large, and seed-robust (per-label std ≤ 1.08, far below every
   adjacent gap; independently recomputed on seeds {6,23,64}). All five cells share the same
   245.76M-token budget and `data_sha256`, so the *internal* axis carries no D-confound. Red does
   not dispute this and does not need to. Following Hacking (*Representing and Intervening*, 1983),
   this is a robust experimental regularity with "a life of its own." A phenomenon, granted.

2. **`phi_embed` is a genuine token-indexed embedding-table access pattern.** At runtime a token
   gathers exactly one width-`n_gen` row (`prior_bank.py:167`, shape `(V, n_gen)`; `:682`,
   `phi = pb.phi_embed[token_ids]`, one row per token; both re-verified against the active GL24
   config by the implementation-engineer). This is structurally the token-embedding table of Vaswani
   et al. 2017 §3.4 and GPT-2 (Radford et al. 2019). Blue's access-pattern reading is correct.

3. **The head mixer is not the driver.** It is `nn.Parameter(torch.zeros(m, m))` with `m = n_heads`
   (`head_mixer.py:105-107`), shrinking 16×16 → 2×2 as blocks enlarge — capacity moving *opposite*
   to the loss gain (implementation-engineer, transformer-ml). Red does not run the "the toggle
   smuggled the gain in via head-mixer capacity" attack; it fails on monotonicity, as Blue showed.

These concessions cover Blue's Pillar 1, the access-pattern half of Pillar 2, and the head-mixer
half of Pillar 3. None of them establishes the claim under trial.

## Core attack

### 1. "Structurally new" is not "efficient scaling axis" — the conjunction is already dead

The claim document fixes the proposition as a conjunction and pre-commits: "if the panel finds the
empirical improvement real but the 'parameter-efficient / publishable as-is' qualifier unsupported,
that is a REMAND." Blue conceded the publishable-exponent conjunct (axis-dependent fit, 0.929 vs
`n_params` with CI [0.07, 1.73], 0.181 vs `n_gen`, R² = 0.17 vs FLOPs, over a compressed 3.62×
range) and the parameter-efficient-as-dominance conjunct (at matched total `n_params`, blocks is
+4.8 to +12.8 PPL *worse* than width). A conjunction with a false conjunct is false. A REMAND on
this claim *is* the refutation of the claim as written (philosophy-of-science memo).

What survives — Blue's Pillar 3 — is a statement about the *literature* (fixed-`embed_dim`,
vary-block scaling appears in neither manuscript) and about *structure* (a change of block group).
Novelty and structural distinctness are not efficiency and are not a scaling law. Blue's own
conceded evidence shows the surviving axis is *less* efficient than width: wall-time is U-shaped
with GL24 the slowest run (11049s, 2.37× the GL8 minimum), and real per-token transport compute
grows 64× (`g²`, 9 → 576). A knob can be new and inefficient at once; the compound claim launders
novelty into efficiency by juxtaposition. Turning Blue's own Hacking cite against it: the same
observation/interpretation cut that shields the *effect* denies the *interpretation* — a robust
theory-free regularity licenses no theory-laden reading ("efficient," "scaling axis"). It demotes
the claim to exactly the REMAND target (philosophy-of-science memo).

### 2. The mechanism is unidentified — no non-gauge matched-size control exists

The sweep is a fat-handed manipulation. As `g` grows it co-varies three quantities in lockstep:
total `n_params` (3.62×), `n_heads` (16 → 2), and block width `g` driving `phi_embed` (144 → 1152).
Mill's Method of Difference (*A System of Logic*, 1843, bk. III ch. 8, p. 483) and Woodward's
interventionism (*Making Things Happen*, 2003, ch. 2) both require the cause be varied *alone*; a
manipulation that also moves two other knobs identifies none of them. Blue's head-mixer rebuttal
validly kills one alternative and leaves the dominant one — raw `phi_embed` table capacity —
uncontrolled (philosophy-of-science memo).

The decisive missing experiment is Blue's own Falsification Condition 2: a plain non-gauge `V × m`
learned table with `m = n_gen`, read one row per token. It has not been run. And Blue's Pillar 2
supplies the reason to expect it would reproduce the curve: if `phi_embed` behaves like a
token-embedding table, then a plain non-gauge table of matched width delivers the same per-token
capacity with the same flat `active/token` (transformer-ml memo). The embedding-table analogy Blue
uses to *legitimize* the metric simultaneously shows the gain is *not* attributable to gauge
structure. A larger embedding table lowering loss is unsurprising and well-known; it does not make
enlarging the gauge block an efficient axis.

Blue's own gauge canon inverts on inspection. Cohen & Welling 2016 (verbatim): "G-convolutions
increase the expressive capacity of the network *without increasing the number of parameters*," via
"a substantially higher degree of weight sharing." Kondor & Trivedi 2018 and Weiler & Cesa 2019
make the group a *constraint* that reduces the admissible-map space, not free capacity; Weiler,
Forré, Verlinde & Welling 2021 tie gauge equivariance itself to weight sharing (arXiv:2106.06020).
The equivariance precedent for "group is a capacity axis orthogonal to width" is a *fixed-parameter,
weight-sharing* phenomenon. `blocks_K48` does the opposite — params grow 3.62× (99.7% in
`phi_embed`) as the group enlarges. Invoking these theorems to license *added* capacity gets them
backwards; a genuine gauge constraint should *match* the loss at *fewer* parameters, which is
exactly the un-run tied control (gauge-theorist memo).

The identification is weaker still on the flat path actually run. With `transport_mode='flat'`,
`Omega_ij = exp(phi_i·G)exp(-phi_j·G)` is a coboundary of the 0-cochain `g_i = exp(phi_i·G)`: its
holonomy is the identity and its curvature is identically zero (Nakahara 2003, §10.2–10.3; Bleecker
1981, ch. 3). A flat, zero-holonomy connection is gauge-equivalent to the trivial one; its only
content is a per-token learned frame `exp(phi_i·G) ∈ GL⁺(48)` — a learned linear reparametrization,
not rich gauge-field structure (gauge-theorist memo). And the runs are not even on the strictly
equivariant path: `use_head_mixer=True` and `use_prior_bank=False` both break strict gauge
equivariance off identity-init (per CLAUDE.md's documented exceptions), so decode composes as
`logits = mu @ (WM)^T`, a learned linear map co-moving with the sweep. Code confirms the mechanism
is not isolated: `pos_phi_free` is a *second* size-co-moving learned table (`model.py:375-376`,
shape `(128, n_gen)`, 18,432 → 147,456 params), and with `detach_e_step=False` it trains alongside
`phi_embed`, `decode_bias`, and `learnable_r` (implementation-engineer memo). "Gauge structure did
it" is unproven in the literature sense, in the differential-geometry sense, and in the code.

### 3. "Complementary to width" is a cross-sweep claim, and the 2× token confound is un-removed

The claim's operative phrase is "a distinct scaling knob *complementary to width-scaling*
(`grow_K_GL10`)." That is inherently a *cross-sweep* comparison, and it compares blocks_K48
(245.76M tokens) against grow_K_GL10 (491.52M tokens, exactly 2×). Under Hoffmann et al. 2022,
`L̂(N,D) = E + A/N^α + B/D^β` (E=1.69, A=406.4, B=410.7, α=0.34, β=0.28) is additive in D, and
"for every doubling of model size the number of training tokens should also be doubled." A 2×
difference in D is a first-order confound on any cross-sweep loss or floor comparison, including the
"complementary axis" reading and Blue's own "suggestive" GL24 (CE 4.523) vs grow K50 (CE 4.544)
point — which Blue itself flags is "on half the tokens," surrendering the only cross-sweep evidence
the conjunct could rest on. The matched-budget (491.52M-token) blocks run does not exist
(ml-engineer memo).

A "distinct efficient *scaling axis*" is a claim about a trend, not a single-budget monotone curve.
Kaplan et al. 2020 §1.3 defines the clean axis as "N — the number of model parameters, excluding
all vocabulary and positional embeddings," and §3.1 reports performance "depends very weakly on the
shape parameters n_layer, n_heads, and d_ff when we hold the total non-embedding parameter count N
fixed." Two consequences for this claim: `phi_embed (V, n_gen)` is exactly the excluded
vocab/embedding table (carrying 99.7% of the sweep's param growth), so plotting CE vs `n_params` is
the wrong x-axis; and `n_heads` (16 → 2) is a *weak* knob at fixed N, so the head-count traversal is
not the mechanism either. The sweep also slides along Vaswani §3.2.2's `d_model = h·d_k` partition
(16 heads at d_k=3 → 2 heads at d_k=24), a known architectural knob, with head count itself
redundancy-laden (Michel et al. 2019; Voita et al. 2019: "pruning 38 out of 48 encoder heads
results in a drop of only 0.15 BLEU") (transformer-ml memo). A monotone descent at one budget along
a compressed, multiply-confounded axis is an ablation data point, not a scaling axis.

## Defense

Blue conceded Red's code-truth spine; it stands intact and unopposed.

1. **The `active/token` flatness is an algebraic identity, not an efficiency finding.** Under this
   config (`use_prior_bank=False` + `model_channel=True`, both branches firing), `run_artifacts.py:616-620`
   resolves to `active = 5·V·K + 2·K + n_gen`; the base `5·50257·48 + 2·48 = 12,061,776` matches the
   CSV to the integer, and the *only* sweep-varying term is `+n_gen` (+1,008 total = +0.0084%). A
   metric defined to hold the decode readout fixed cannot be evidence that the axis is efficient —
   that is circular, and Blue conceded the circularity explicitly (implementation-engineer memo;
   Blue Pillar 2 / conceded point 5).

2. **Real compute rises with block size; the constant-compute precondition fails.** `d_head = g` at
   fixed K=48 (`run_artifacts.py:611`), so the transport sub-term `2·N·g²` grows 64× (9 → 576;
   `run_artifacts.py:626`). `fpt_decode = 2·V·K = 4,824,672` (fixed) dominates `fpt_estep`, so
   `est_flops_analytic` moves only 1.030× and the analytic-FLOP axis is decode-saturated — Williams
   et al. 2009 (roofline) explains why full-vocab decode is bandwidth-bound and why wall-time is the
   ground-truth signal. Empirical wall-time is U-shaped, GL24 the slowest run. Fedus, Zoph & Shazeer
   2021 license the total-vs-active axis only at "a constant computational cost"; blocks buys its
   best loss at the *highest* wall-clock, so the antecedent is not met (implementation-engineer,
   ml-engineer memos). Blue conceded the mixed compute picture in full.

3. **The convention correction stands and extends.** `01_evidence.md:31` mislabels ~12M as
   `fpt_decode`; the code value is 4.82M (12M is `active_params_per_token`, a different quantity), a
   correction Blue itself carried forward against its own interest (implementation-engineer memo).

The net: Red concedes the phenomenon, the access pattern, and the head-mixer exoneration. On the
claim actually on trial, Blue's surviving Pillar 3 proves novelty and structural distinctness, not
efficiency and not a scaling axis; the mechanism is unidentified for want of the non-gauge
matched-size control and the matched-token-budget run, both of which Blue lists as falsifiers and
neither of which exists. The compound claim fails as written. Verdict sought: REMAND to the narrow
sub-claim (a robust, monotone, seed-stable CE improvement along a genuinely new fixed-`embed_dim`
block axis — reportable as a structural ablation curve, not as a parameter-efficient scaling axis
complementary to width).
