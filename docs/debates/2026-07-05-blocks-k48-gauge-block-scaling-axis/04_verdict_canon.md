# Verdict (canon-strict) — blocks-k48-gauge-block-scaling-axis

## Evidence audit

Verification note: canon here is `embedded`, so there is no `external_bibliography.md` to
grep against. In its place both canon-cop passes (`02_canoncop_*`, `03_canoncop_*`) verified
every load-bearing external citation as real and in-domain, with several re-verified verbatim
against arXiv (Voita 2019, Cohen & Welling 2016, Weiler-Forré-Verlinde-Welling 2021, the
Chinchilla Approach-3 coefficients). I treat canon-cop-verified citations as verified (weight 3).

| Side | External citations (verified) | External citations (unverified) | sympy/FD | path:line | Canon-cop strikes |
|------|------------------------------|--------------------------------|----------|-----------|-------------------|
| Red  | ~20: Shazeer 2017, Fedus/Zoph/Shazeer 2021 (constant-FLOP precondition), Kaplan 2020 (N excludes embeddings; shape weak at fixed N), Hoffmann 2022 (additive-in-D floor, verbatim coefficients), Clauset-Shalizi-Newman 2009, Stumpf & Porter 2012, Vaswani 2017 §3.2.2, Michel 2019, Voita 2019 (verbatim), Duhem 1906, Mill 1843, Woodward 2003, Cartwright 1999, Popper 1963, Quine 1951, Worrall 2014, Lakatos 1978, Cohen & Welling 2016 (verbatim, inverted), Kondor-Trivedi 2018 / Weiler-Cesa 2019, Weiler-Forré-Verlinde-Welling 2021 (verbatim), Nakahara 2003 / Bleecker 1981, Williams 2009, Belsley-Kuh-Welsch 1980 | 0 | 2 (reproduced scipy: 5.15× exponent swing α=0.929/0.180 + corr(E,α)≈+1; log-log condition number 306) | ~10 (run_artifacts.py:611,616-620,625-627; prior_bank.py:167,682; free_energy.py:42-54; head_mixer.py:105-107; model.py:375-376) | 0 |
| Blue | ~21: Vaswani 2017 §3.4, Radford 2019 (GPT-2), Shazeer 2017, Fedus/Zoph/Shazeer 2021, Kaplan 2020 §2.1/Fig.6, Besiroglu 2024, Cohen & Welling 2016, Kondor-Trivedi 2018, Cohen-Weiler-Kicanaoglu-Welling 2019, Bronstein 2021, Williams 2009, Hoffmann 2022 (conceded), Hacking 1983, Lakatos 1978, Popper 1963, Bogen & Woodward 1988, Pearl 2009, Hansen 1998, Jonckheere 1954, Page 1963, Nakahara 2003, Stumpf & Porter 2012 (conceded) | 0 | 2-3 (reproduced scipy: permutation p=(1/120)^3=5.8e-7; Spearman ρ / Kendall τ = -1.000; smallest-step 4.07-7.04σ) | ~7 (prior_bank.py:167,682; run_artifacts.py:616-620; groups.py:144-152; generators.py:96,103; transport.py:183; plus the five-config fixed-budget arithmetic) | 0 |

Weighted (verified ext ×3, path:line ×1, sympy/scipy ×2, strikes ×-1):
- Red: ~20·3 + ~10·1 + 2·2 = ~74
- Blue: ~21·3 + ~7·1 + ~2.5·2 = ~75

Near-tie. Neither side cited `Attention/*.tex`, `GL(K)_attention.tex`, `PIFB.tex`, or `CLAUDE.md`
as canonical authority. Red's single `CLAUDE.md` reference (`03_red_rebuttal.md:98`) invokes the
documented-exceptions block adversarially, as code-behavior documentation turned against the claim
(the runs are on the equivariance-breaking `use_head_mixer=True`/`use_prior_bank=False` path), not
as authority for a canonical form; canon-cop scored it 0. Blue filed an explicit manuscript-authority
disclaimer (`03_blue_rebuttal.md:166-172`). No −2 authority strikes apply to either side.

## Concessions made
- Red conceded: the within-sweep CE effect is real, monotone, and seed-robust (PPL 124.57 → 92.15, std ≤ 1.08), with no internal D-confound (all five cells share 245.76M tokens and `data_sha256`); `phi_embed` is a genuine token-indexed embedding-table access pattern (Vaswani §3.4, GPT-2); the head mixer is not the driver (it shrinks 16×16 → 2×2, opposite to the gain). Red seeks REMAND, not a claim that no effect exists.
- Blue conceded: the compound claim "as written" fails; the flat ~12.06M active/token is a definitional identity (`5·V·K + 2·K + n_gen`), not a measurement; there is no reportable scaling exponent (5.15× axis-dependent swing, CI [0.07, 1.73], < 1 decade of range); the cross-sweep "complementary to width-scaling" comparison is a Chinchilla D-slice confound and is withdrawn; the "parameter-efficient" conjunct is unearned on real compute (wall-time U-shaped, GL24 slowest, transport 64×); and the causal share attributable to the generator algebra versus the co-growing scalar table is unmeasured for want of the non-gauge control.

## Decisive evidence
Hoffmann et al. 2022 (Chinchilla), `L̂(N,D) = E + A/N^α + B/D^β`, Approach-3 coefficients
E=1.69, A=406.4, B=410.7, α=0.34, β=0.28 — verified verbatim by both Phase-2.5 and Phase-3.5
canon-cop passes. This single verified external entry is the tie-hinge because it does two opposite
things at once. Its additive-in-D floor refutes the "distinct scaling axis complementary to
width-scaling" conjunct as stated: `blocks_K48` (245.76M tokens) versus `grow_K_GL10` (491.52M,
exactly 2×, same `data_sha256`) sit on different D-slices, so every cross-sweep reading is
non-identified. Yet the same `B/D^β` term is byte-identical across all five `blocks_K48` cells
(identical `max_steps=60000`, `batch_size=32`, `max_seq_len=128`), so it cannot generate the
within-sweep ordering — the genuine sub-claim survives it untouched. Verified canon that refutes one
conjunct while preserving the sub-claim is the textbook REMAND signature.

## My weighted scores
- Red weighted total: ~74
- Blue weighted total: ~75

## Outcome (this judge)
REMAND

Focused follow-up question: does the strictly monotone GL3 → GL24 CE improvement survive (a) a
matched 491.52M-token `blocks_K48` run (mechanically `batch_size=64` at the same `max_steps`, which
removes the entire cross-sweep D-confound under Hoffmann's `B/D^β` floor), AND (b) a non-gauge
matched-parameter `V × m` learned-table control at fixed head geometry (which discharges the
Duhem/Mill/Woodward confound and isolates the `gl(g)` generator algebra from raw `phi_embed` table
capacity)? Absent both artifacts, the reportable proposition is the narrow sub-claim, not the compound.

## Per-conjunct assessment (which of the four survive the canon)

1. Genuine — SURVIVES. Verified external canon supports the effect as a detected phenomenon, not a plot artifact: Pearl 2009 p.99 (the confound defeats `P(CE | do(gauge structure))` but leaves the association `P(CE | block size)` intact), Bogen & Woodward 1988 pp.306,314 (data vs phenomena), and the ordered-alternatives detection canon Jonckheere 1954 / Page 1963 (reproduced permutation p = 5.8e-7, ρ = τ = -1.000). Red concedes it. The within-sweep budget is fixed, so Hoffmann/Kaplan constrain only the cross-sweep, not this ordering.

2. Publishable — FAILS as a scaling exponent/law; survives only as a structural ablation curve. Contradicted by verified canon: Clauset-Shalizi-Newman 2009 (least-squares on the log "generate significant systematic errors" with "no warning of the bias") and Stumpf & Porter 2012 (a credible power law spans ≥ 2 orders of magnitude; `blocks_K48` spans 0.559 decades vs n_params, 0.903 vs n_gen). The exponent is not identified (5.15× swing, CI crossing α=1). Blue concedes.

3. Parameter-efficient — FAILS as stated. Contradicted by verified canon: Fedus/Zoph/Shazeer 2021 and Shazeer 2017 license the total-vs-active axis only at "a constant computational cost," and that precondition is not met — real per-token transport grows 64× (`2·N·g²`), analytic FLOPs are decode-saturated (only 1.030× on the fixed `2·V·K`), and empirical wall-time is U-shaped with GL24 (best loss) the slowest run. At matched total n_params, blocks is +4.8 to +12.8 PPL worse than width. Kaplan 2020 §2.1 (N = non-embedding params) legitimizes the flat-active bookkeeping but, as Blue concedes, "is not by itself an efficiency proof." Blue concedes.

4. Distinct scaling axis — SPLIT: fails as a scaling axis, survives as a design/structural coordinate with an unidentified causal share. Cohen & Welling 2016, Kondor-Trivedi 2018, and Bronstein 2021 support "change of structure group is a design axis orthogonal to width." But Red's verbatim inversion is on the record and adverse: Cohen & Welling's own words are that G-convolutions raise capacity "without increasing the number of parameters" via weight sharing — a fixed-parameter phenomenon, whereas `blocks_K48` grows params 3.62× (99.7% in `phi_embed`) as the group enlarges, and Weiler-Forré-Verlinde-Welling 2021 ties gauge equivariance itself to weight sharing. So the gauge canon licenses a design coordinate, not an efficient params-growing scaling axis; and the causal attribution to "gauge structure per se" is confounded (Duhem 1906, Mill 1843, Woodward 2003: three knobs — n_params 3.62×, n_heads 16→2, block width g — move together, no control run), sharpened by Nakahara 2003 §10.2-10.3 / Bleecker 1981 (the flat-path `Ω_ij = exp(φ_i·G)exp(-φ_j·G)` is a coboundary: zero curvature, trivial holonomy, only a per-token learned frame).

Compound verdict: three of the four conjuncts fail as stated (publishable, parameter-efficient,
distinct scaling axis), and "NOT artifact+confound" is itself split — not a pure metric artifact
(Kaplan bookkeeping, Pearl association) but genuinely causally confounded (Duhem/Mill/Woodward). The
conjunction is false as written; the narrow sub-claim is true and canon-supported.

## Reasoning
Both sides brought large, near-equal, canon-cop-clean external-canon sets (weighted ~74 vs ~75),
and — this is the operative fact for a canon-strict reading — their verified citations land on
different conjuncts. Red's verified canon (Fedus/Shazeer constant-FLOP precondition, Clauset and
Stumpf & Porter on unidentified exponents over sub-decade range, Hoffmann additive-in-D floor on the
cross-sweep) refutes the efficiency, publishable-exponent, and complementary-axis conjuncts. Blue's
verified canon (Pearl association-vs-causation, Bogen & Woodward data-vs-phenomena, Jonckheere/Page
detection, Kaplan non-embedding-N) supports the genuine-effect conjunct and the metric-legitimacy of
the bookkeeping, all of which Red concedes. Neither side cited a manuscript or `CLAUDE.md` as
authority, so no −2 strikes and no forbidden circularity discount apply. My rubric forbids splitting
the difference only when one side has external canon and the other does not; here both do, on
different parts of a compound claim, at a near-tie total — which is precisely the stated REMAND
condition. The decisive Hoffmann citation is the hinge: it refutes the cross-sweep conjunct while
leaving the fixed-budget within-sweep effect intact, so the canon supports not the compound as
written but a narrower true sub-claim. For the chief's accountability record I state the asymmetry
plainly: on a reading of the compound as a single indivisible proposition, the verified external
canon refutes it (three conjuncts fail), so this REMAND leans Red on the efficiency/scaling-law
substance and preserves only Blue's genuine-effect core.

## Recommended action
REMAND to the narrow, canon-supported sub-claim: at fixed `embed_dim=48`, enlarging the GL gauge
block GL3 → GL24 lowers cross-entropy strictly monotonically (PPL 124.6 → 92.2) at a 245.76M-token
budget, three-seed-robust — a genuinely new, previously unpublished fixed-`embed_dim` block-partition
phenomenon along a well-defined structure-group design coordinate, reportable as a structural
ablation curve, whose causal decomposition and efficiency framing remain open. Strike from the
reportable claim, per verified external canon, the words "publishable [exponent/scaling law],"
"parameter-efficient," and "distinct scaling axis complementary to width-scaling." To convert the
REMAND into a compound win, run the paired battery both sides named: the matched 491.52M-token
`blocks_K48` run (removes the Hoffmann D-slice confound) and the non-gauge matched-parameter `V × m`
table control at fixed head geometry (discharges the Duhem/Mill/Woodward confound); adjudicate
efficiency only afterward, against wall-clock or a transport-inclusive FLOP axis, never against the
definitionally flat `active_params_per_token`.
