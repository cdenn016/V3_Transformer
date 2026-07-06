# Blue Opening — blocks-k48-gauge-block-scaling-axis (Phase 2)

Side: BLUE (defending, by steelmanning). Panel: ml-engineer, transformer-ml, implementation-engineer, gauge-theorist, philosophy-of-science. Every memo is cited below.

## Steelman

Stated at its strongest, the claim is this. Hold `embed_dim = 48` fixed and vary only the gauge partition, `GL(3)^16 -> GL(24)^2`. Cross-entropy then falls strictly and monotonically, PPL 124.6 -> 112.4 -> 106.2 -> 99.8 -> 92.2, seed-stable at per-label std <= 1.08 (far below every adjacent gap, independently recomputed on seeds {6,23,64}). The 3.62x growth in total parameters that accompanies this improvement lives almost entirely (99.7%) in `phi_embed`, a `(V, n_gen)` table read one row per token; the decode-bound per-token working set moves by +1,008 params total (+0.008%). So the improvement buys real accuracy while touching essentially no additional parameters per token. The direction is genuinely new: fixed-`embed_dim`, vary-block scaling appears in neither manuscript, which have only ever grown width at fixed block. Read charitably, the claim is that this is a distinct, complementary, parameter-lean scaling knob for the VFE transformer, not a mirage produced by choosing `active_params_per_token` as the x-axis compounded by a half-token budget.

## Position

Blue defends the robust core of this claim and concedes, with reasons, the two conjuncts the evidence cannot carry. This is a partial defense that resolves to a **REMAND** — the outcome the claim document itself anticipated ("if the panel finds the empirical improvement real but the 'parameter-efficient / publishable as-is' qualifier unsupported, that is a REMAND toward a narrower true sub-claim").

What Blue defends as true:

1. **The within-sweep empirical effect is real, large, monotonic, and seed-robust — and, unlike the cross-sweep comparison, it carries no token-budget confound.** All five `blocks_K48` cells train on the *same* 245.76M tokens (ml-engineer memo). The half-token gap versus `grow_K_GL10` bites only the cross-sweep frontier comparison, not the internal block axis. A monotone descent this far above seed noise is a phenomenon, not an artifact of any plot choice (philosophy-of-science memo).

2. **The small per-token working set is a genuine runtime access-pattern property, not merely a self-serving metric.** At runtime a token gathers exactly one width-`n_gen` row of `phi_embed` (`prior_bank.py:682`, verified by implementation-engineer); the `V x n_gen` bulk is never contracted per token. This is structurally the token-embedding-table pattern every GPT-class model uses (transformer-ml memo), and the recognized total-vs-active decoupling of sparse / conditional-computation models (ml-engineer memo).

3. **The block-partition axis is genuinely new and structurally distinct from the published width axis.** It varies the local gauge group per block — a change of structure group, not a wider dense layer (gauge-theorist memo). Group/representation choice is a recognized design axis separate from channel width across the equivariance literature.

What Blue concedes and remands:

4. **"Publishable scaling *exponent*" fails.** The fitted exponent is axis-dependent on identical CE data (0.929 vs `n_params` with CI [0.07, 1.73], 0.181 vs `n_gen`, degenerate R^2 = 0.17 vs analytic-FLOPs) over a compressed 3.62x x-range. A power law with a sign-ambiguous CI and no cross-axis invariant is not a reportable exponent (all five memos concur; philosophy-of-science and ml-engineer are most explicit). Report a *curve/ablation*, not an exponent.

5. **"Parameter-efficient" as *dominance over width* fails as stated, and partially self-falsifies on existing evidence.** At matched *total* `n_params`, `blocks_K48` is *worse* than `grow_K_GL10` by +4.8 to +12.8 PPL at all five matched points. Blue does not lean on the flatness of `active_params_per_token` as proof of efficiency — that metric is defined to hold the working set flat (`run_artifacts.py:616-620`), and using its flatness as evidence of efficiency would be circular (philosophy-of-science memo, mandatory flag). The honest efficiency question is settled on wall-time / FLOPs, where the picture is mixed: wall-time is U-shaped (6366s at GL3 -> 4657s at GL8 -> 11049s at GL24, GL24 the slowest run), and real per-token transport compute grows 64x (`d_head^2 = g^2`, 9 -> 576).

The defensible narrower claim: **a robust, monotonic, seed-stable cross-entropy improvement along a genuinely new, previously-unpublished fixed-`embed_dim` gauge-block-partition axis, whose extra capacity is structured gauge-covariant transport read one row per token — reportable as a complementary structural knob, not as a standalone parameter-efficiency scaling law.**

## Evidence

### Pillar 1 — the effect is real and confound-free within the sweep

Empirical (evidence pack, independently confirmed): PPL 124.57 / 112.41 / 106.20 / 99.77 / 92.15 across GL3/6/8/12/24; per-label std <= 1.08; monotone in block width `g`. All five cells share the same 245.76M-token budget and `data_sha256`, so the internal axis carries no D-confound. The GL3 -> GL8 segment is Pareto-improving on the empirical frontier: PPL drops 124.6 -> 106.2 while wall-time *falls* 6366s -> 4657s (ml-engineer memo). Better loss at lower wall-clock on equal data is an efficiency signal on its own terms.

Philosophy of science separates the robust observation from the contested interpretation. Following Hacking (*Representing and Intervening*, 1983), a robust experimental regularity "has a life of its own, independent of theory" — the loss-vs-`g` descent is monotone however one plots the parameter axis, so the "vertical column" objection, which is itself frame-laden in `active_params/token`, cannot demote the *effect*; it demotes only the efficiency *reading* (philosophy-of-science memo). By Lakatos's criterion (*Methodology of Scientific Research Programmes*, 1978), a monotone, seed-corroborated phenomenon absent from and not derivable by the published width-axis work is exactly the "excess empirical content" a progressive problemshift requires.

### Pillar 2 — the low per-token working set is a real access pattern, and total-vs-active is legitimate

Code truth (implementation-engineer, each line self-verified under the resolved GL24 config): `phi_embed = nn.Parameter(... (vocab_size, n_gen))` (`prior_bank.py:167`); read as `phi = pb.phi_embed[token_ids]` returning `(B, N, n_gen)` — one row per token (`prior_bank.py:682`). The closed form `active_params_per_token = 5*V*K + 2*K + n_gen` reproduces all five CSV cells to the integer; the only sweep-varying term is `+n_gen`, +1,008 total.

This is the canonical token-embedding table. Vaswani et al. 2017 (§3.4) define the learned input-embedding table used one row per token; GPT-2 (Radford et al. 2019) realizes it as a `[50257 x 768]` lookup, and the project's `vocab_size = 50257` is exactly that byte-level BPE vocabulary (transformer-ml memo). A table's total count `V x d` is large while its per-token working set is one row — so a flat active/token at growing `n_gen` is the expected, well-understood property, not a metric pathology. The sparse-model canon names this axis directly: Shazeer et al. 2017 obtained ">1000x improvements in model capacity with only minor losses in computational efficiency" by activating a per-example parameter subset, and Fedus, Zoph & Shazeer 2021 (Switch Transformer) frame "increase the parameter count while keeping the FLOPs per example constant" as "a separately important axis on which to scale." Kaplan et al. 2020 reinforce that the clean parameter power law is fit on the *non-embedding* count precisely because vocab tables do not scale like compute — so `phi_embed (V, n_gen)` belongs in the excluded-from-N bucket, and `n_params` is the wrong x-axis for a headline efficiency claim here (ml-engineer memo).

The Blue defense is scoped honestly: Fedus's own framing requires pairing the active-param axis with a compute axis. On matched active-compute the best block point sits favorably — GL24 (CE 4.523, active 12.06M) versus grow K50 (CE 4.544, active 12.56M) — but on half the tokens, so this comparison is offered as suggestive, not decisive, and is deferred to the falsification conditions.

### Pillar 3 — the added capacity is structured gauge capacity, and the head mixer is not the driver

Code truth (gauge-theorist + implementation-engineer): `block_glk` builds `GL(d_head)^n_heads` (`groups.py:144-152`); `generate_glk_multihead` returns `(n_heads * d_head^2, K, K)` generators, so `n_gen = n_heads * g^2 = 48*g` (144 -> 1152 confirmed). Each block carries the full `gl(g)` Lie algebra; enlarging `g` from 3 to 24 raises the per-block fiber-symmetry dimension 9 -> 576. The `phi_embed` rows are Lie-algebra coordinates driving covariant transport `Omega_ij = exp(phi_i . G) exp(-phi_j . G)` by the sandwich action (`transport.py:183`). This is a change of *structure group*, not a wider dense layer.

The equivariance literature establishes group richness as a capacity axis orthogonal to width: Cohen & Welling 2016 raise accuracy at *fixed* parameter count by enlarging the symmetry group (p4 -> p4m); Kondor & Trivedi 2018 show an equivariant linear map is exactly a group convolution, so changing the group is a structural change rather than added free capacity; Cohen, Weiler, Kicanaoglu & Welling 2019 extend this to *local gauge* transformations, the closest published analog to the `GL(g)`-per-block internal gauge (gauge-theorist memo).

The head-mixer confound is ruled out by monotonicity. The mixer is `nn.Parameter(torch.zeros(m, m))` with `m = n_heads` (`head_mixer.py:105-107`), so it shrinks 16x16 = 256 (GL3) -> 2x2 = 4 (GL24) as blocks enlarge — capacity moving *opposite* to the loss gain (transformer-ml, implementation-engineer, gauge-theorist all confirm). A "the toggle smuggled the gain in via head-mixer capacity" attack fails on these grounds.

### Accuracy note carried forward

The implementation-engineer corrected the evidence pack: `fpt_decode = 2*V*K = 4.82M/token` (`run_artifacts.py:625`), not the ~12M quoted at `01_evidence.md:31` (that figure is `active_params_per_token`, a different quantity). Decode still dominates `fpt_estep`, so `est_flops_analytic` moves only 1.03x and the analytic-FLOP axis is decode-saturated — Williams et al. 2009 (roofline) explains why full-vocab decode is bandwidth-bound and why wall-time, not the analytic proxy, is the ground-truth compute signal. Blue reports this correction against its own interest, in service of the record.

## Falsification conditions

This claim — even the narrower defensible version — is *not* defensible if any of the following holds. Blue states them explicitly because falsifiability is the claim's credential, not a weakness (Popper, *Conjectures and Refutations*, 1963).

1. **Matched-token-budget run.** Retrain `blocks_K48` at 491.52M tokens. If the monotone GL3 -> GL24 gain collapses into seed noise once D matches `grow_K_GL10`, the "genuine new axis" reading is refuted as a data-budget / optimization-transient artifact. Under Hoffmann et al. 2022 (`L(N,D) = E + A/N^a + B/D^b`, ~20 tokens/param), a 2x difference in D is a first-order confound; this run is the decisive test and does not yet exist.

2. **Non-gauge matched-size control.** Train a plain `V x m` learned table (or a matched-param dense head-mix) with `m = n_gen`, no gauge-block action. If it reproduces the 124.6 -> 92.2 curve at matched added params, the gain is raw table capacity, not gauge structure — collapsing the "structured / distinct axis" conjunct. No such control exists (all five memos name this gap).

3. **tied_block_glk match.** If an exactly-equivariant `tied_block_glk` run (`n_gen = g^2`, not `48*g`) matches the untied curve at far fewer parameters, per-block untied richness is not the mechanism (gauge-theorist condition Y).

4. **Compute-frontier reversal at equal tokens.** On a calibrated wall-clock / FLOP frontier at matched 491.52M tokens, if GL24 (the slowest run) sits *above* the grow frontier, the "efficient on compute" reading fails. This conjunct already partially self-falsifies: at matched *total* `n_params`, blocks is +4.8 to +12.8 PPL worse than width.

5. **Credit-reassignment under ablation.** If zeroing or freezing the learned linear decode `W` and the head mixer at fixed `g` removes the monotone improvement, or if rescaling `phi_scale ~ 1/sqrt(n_gen)` to hold the phi-row init variance fixed collapses the gain, then credit moves off the gauge block onto learned linear co-moving capacity or init conditioning (implementation-engineer flagged the co-moving tables `pos_phi=learned`, `decode_bias`, `learnable_r`; ml-engineer flagged the init-variance test).

Discounted memos: none. All five panelists are cited above; all five converge on the same partition — defend the empirical + new-axis + structured-capacity core, concede/remand the "publishable exponent" and "parameter-efficient-as-dominance" conjuncts.
