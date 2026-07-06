# Verdict (code-truth) — blocks-k48-gauge-block-scaling-axis

## My re-traced active config

Traced directly from `vfe3_scaling_results/blocks_K48/K48_GL24/s6/config.json` (nested under the
`config` key) and cross-checked against all five cells' `config.json` plus `scaling_points.csv`.

| key | value | source |
|---|---|---|
| `vocab_size` V | 50257 | config.json |
| `embed_dim` K | 48 | config.json (fixed sweep-wide) |
| `n_heads` H | 2 / 4 / 6 / 8 / 16 for GL24/12/8/6/3 | config.json (= 48/g) |
| `gauge_group` | `block_glk` | config.json |
| `transport_mode` | `flat` | config.json |
| `use_prior_bank` | False | config.json |
| `use_head_mixer` | True | config.json |
| `prior_source` | `model_channel` | config.json |
| `s_e_step` | True | config.json |
| `lambda_h`, `lambda_gamma` | 0.25, 0.75 | config.json |
| `n_layers`, `n_e_steps` | 1, 1 | config.json |
| `max_seq_len` N | 128 | config.json |
| `batch_size` | 32 | config.json |
| `max_steps` | 60000 | config.json |
| `grad_accum_steps` | 1 | config.json |
| `encode_mode` | `per_token` | config.json |
| `decode_bias`, `pos_phi`, `learnable_r` | True, `learned`, True | config.json |
| `detach_e_step`, `e_step_gradient` | False, `unroll` | config.json |
| tokens/cell | 60000·32·128 = **245,760,000** | recomputed |

No disagreement with the openings' trace. The dispatch checklist is confirmed to the item:
`vocab_size=50257`, `embed_dim=48`, `use_prior_bank=False`, `use_head_mixer=True`,
`prior_source=model_channel`, `s_e_step=True`, `transport_mode=flat`, `n_layers=1`, `n_e_steps=1`,
`max_seq_len=128`. All five blocks_K48 cells are byte-identical except `n_heads` (16/8/6/4/2), and
the CSV carries one shared `data_sha256` (`d2a72d0...`) across every blocks row, so the within-sweep
budget and data are fixed. `model_channel` resolves True at `run_artifacts.py:612-613` because
`lambda_h=0.25>0`. The grow_K_GL10 comparison sweep uses `batch_size=64` → 491,520,000 tokens =
**exactly 2×** the blocks budget (recomputed from `grow_K_GL10/K50_GL10/s6/config.json`).

## Reachability verification

| path:line | Cited by | Reachable under active config? | Notes |
|-----------|----------|--------------------------------|-------|
| `run_artifacts.py:616` `active = 2VK + (2K + n_gen)` | both | YES — reached via `finalize_run` → `:757` `_cost_model_fields` | base term |
| `run_artifacts.py:617-618` `+= V*K` | both | YES — `use_prior_bank=False` fires | linear-decode readout |
| `run_artifacts.py:619-620` `+= 2*V*K` | both | YES — `model_channel=True` (`:612-613`, `lambda_h>0`) | s tables |
| `run_artifacts.py:611` `d_head = K/n_blocks` | both | YES — `n_blocks = len(irrep_dims) = n_heads`; at K=48, d_head = g | representative block dim, exact for equal blocks |
| `run_artifacts.py:625` `fpt_decode = 2*V*K` | both | YES | fixed = 4,824,672 |
| `run_artifacts.py:626` `fpt_estep`, transport `2N·d_head²` | both | YES | = 2N·g² |
| `prior_bank.py:167` `phi_embed = nn.Parameter(...(V, n_gen))` | both | YES — constructed unconditionally in `PriorBank.__init__` | shape (V, n_gen) |
| `prior_bank.py:682` `phi = pb.phi_embed[token_ids]` | both | YES — `encode_mode='per_token'` path (`_encode_per_token`) | one row per token, (B,N,n_gen) |
| `prior_bank.py:181` `output_proj_weight = nn.Parameter((V,K))` | both | YES — created under `use_prior_bank=False` | linear decode weight |
| `head_mixer.py:105-107` `mixer_deltas` `zeros(m,m)`, m=n_heads | both | YES — one (n_heads×n_heads) block; 16²→2² as g grows | shrinks opposite to the gain |
| `groups.py:144-152` `block_glk` = GL(d_head)^n_heads | blue | YES — no cross_couplings → `generate_glk_multihead`, irrep_dims=[d_head]·n_heads | n_gen = 48·g |
| `model.py:375-376` `pos_phi_free` (128, n_gen) | red | YES — `pos_phi='learned'` active | second co-moving learned table |

Every cited line is reached under the active config. No side is running a different code path; no
citation is invalid for this debate.

## Independent recomputation (executed against the CSV and configs)

| label | g | n_gen | active/tok | PPL (seed-avg) | CE (seed-avg) | wall_s (avg) | flops_analytic | n_params |
|---|---|---|---|---|---|---|---|---|
| GL3 | 3 | 144 | 12,061,920 | 124.574 | 4.8249 | 6366.3 | 1.1893e15 | 19,367,730 |
| GL6 | 6 | 288 | 12,062,064 | 112.413 | 4.7222 | 5010.5 | 1.1910e15 | 26,622,978 |
| GL8 | 8 | 384 | 12,062,160 | 106.195 | 4.6653 | **4657.4** (min) | 1.1928e15 | 31,459,910 |
| GL12 | 12 | 576 | 12,062,352 | 99.772 | 4.6029 | 4795.8 | 1.1978e15 | 41,133,810 |
| GL24 | 24 | 1152 | 12,062,928 | 92.150 | 4.5234 | **11049.1** (slowest) | 1.2250e15 | 70,155,558 |

Confirmed to the integer / to reported precision:
- `active_params_per_token` base `5·50257·48 + 2·48 = 12,061,776`; each cell = base + n_gen; total
  range GL3→GL24 = **+1,008** (= Δn_gen), the only sweep-varying term at fixed V,K.
- PPL strictly monotone **124.6 → 92.2**; CE strictly monotone 4.8249 → 4.5234.
- `n_params` growth 70,155,558 / 19,367,730 = **3.622×**.
- Transport sub-term `2N·g²`: g² 9 → 576 (**64×**); `est_flops_analytic` ratio 1.2250e15 / 1.1893e15
  = **1.030×** (decode-dominated: `fpt_decode = 2VK = 4,824,672` swamps `fpt_estep` 14,592 → 159,744).
- Wall-time **U-shaped**, min at GL8 (4657s), GL24 slowest (11049s = 2.37× the GL8 min).
- Token budgets: blocks 245.76M vs grow 491.52M = **exactly 2×** (mechanically batch 32 vs 64 at
  identical max_steps).

The evidence pack's line-31 label "`fpt_decode ≈ 12M/token`" is a code error: the code value is
`2VK = 4,824,672 ≈ 4.82M/token`; ~12M is `active_params_per_token`. Both implementation-engineers
flagged and corrected it. Verified here; decode-dominance holds under the correct number.

## Evidence audit

| Side | path:line (verified) | path:line (unverified) | Test/CSV outputs | External citations | Comment/docstring cites |
|------|----------------------|------------------------|------------------|--------------------|--------------------------|
| Red  | run_artifacts.py:611,616-620,625,626 (active identity, d_head=g, transport 64×, decode-dominance); head_mixer.py:105-107; model.py:375-376 | none material | scaling_points.csv (wall-time U-shaped, GL24 slowest); configs (2× token budget) | Shazeer, Fedus, Kaplan, Hoffmann, Vaswani §3.2.2, Michel, Voita, Clauset, Stumpf-Porter, Mill, Woodward, Duhem, Nakahara, Bleecker (all in-domain, canon-cop clean) | none used as authority; one CLAUDE.md ref used adversarially (0 strikes) |
| Blue | prior_bank.py:167,682,181; run_artifacts.py:616-620 (conceded); groups.py:144-152; generators.py:96,103 | none material | scaling_points.csv (monotone CE, within-budget); configs (identical within-sweep budget + data_sha256) | Bogen-Woodward, Pearl, Hansen, Jonckheere, Page, Kaplan §2.1, Hoffmann, Cohen-Welling, Kondor-Trivedi, Nakahara (in-domain, canon-cop clean) | none used as authority (explicit circularity disclaimer) |

Canon-cop: 0 strikes both sides, both rounds. No penalty either way. Every load-bearing code fact
either side cites was re-verified by me and is TRUE. The two implementation-engineer memos agree on
every code value; there is no code-fact dispute between the sides.

## Per-conjunct code-truth assessment

The claim is a conjunction. Assessed against what the code and CSV actually produce:

1. **"lowers CE substantially, PPL 124.6 → 92.2"** — **TRUE.** Verified CSV, seed-averaged 124.57 →
   92.15, strictly monotone, seed-stable, on a fixed within-sweep 245.76M-token budget with one
   shared `data_sha256`. No internal D-confound. Both sides concede this.

2. **"per-token active-parameter working set stays essentially constant (~12.06M)"** — **TRUE AS A
   NUMBER, DEFINITIONAL AS EVIDENCE.** `active = 5VK + 2K + n_gen` (run_artifacts.py:616-620) forces
   this at fixed V,K; only `+n_gen` (+1,008) moves. Base 12,061,776 matches the CSV to the integer.
   It is an algebraic identity, not a measured efficiency, so it carries no independent weight for
   the "parameter-efficient" conjunct.

3. **"parameter-efficient scaling axis"** — **CODE/DATA ADVERSE (FALSE as a compute-efficiency
   claim).** Real per-token transport grows 64× (`2N·g²`, run_artifacts.py:626, verified 9→576);
   `est_flops_analytic` is flat (1.03×) only because fixed decode `2VK` dominates; empirical
   wall-time is U-shaped with the best-loss point (GL24) the slowest run (11049s, 2.37× the GL8
   minimum). Under the Shazeer/Fedus constant-compute standard the antecedent fails.

4. **"publishable scaling exponent"** — **UNSUPPORTED.** The fit range is a verified 3.62× in
   n_params (`log10 3.622 = 0.559` decades); the exponent is axis-dependent. The numerical
   reproductions (α = 0.93 vs n_params, 0.18 vs n_gen) are the numerical-analyst's, not re-run by me;
   the compressed input range that drives the instability is code-verified. This conjunct is mainly
   the canon-strict judge's domain; on the code side the x-range is confirmed too short to carry an
   exponent. Both sides concede it.

5. **"distinct scaling knob complementary to width-scaling (grow_K_GL10)"** — **CONFOUNDED /
   UNIDENTIFIED.** This is inherently cross-sweep and compares blocks (245.76M tokens) to grow
   (491.52M, verified exactly 2×). The cross-sweep readings are on different D-slices. Blue withdrew
   it; the code confirms the 2× gap is real.

6. **"rather than a metric artifact of active_params_per_token plus a half-token confound"** —
   **PARTLY WRONG AS STATED.** The CE effect itself is not a metric artifact (it lives in `test_ce`,
   independent of the proxy). But the "parameter-efficient" reading DOES rest on the definitional
   active/tok identity (conjunct 2), and the half-token confound IS real for the "complementary"
   cross-sweep comparison (conjunct 5). The claim's denial of these two is contradicted by the code.

**Mechanism attribution (gauge structure vs raw table capacity)** — **UNIDENTIFIED in code.** Only
`block_glk` cells exist under `blocks_K48/`; no non-gauge `V × m` matched-size control and no
matched-budget (491.52M) blocks run were trained. `phi_embed` is a genuine `(V, n_gen)`
token-indexed lookup (prior_bank.py:167,682) — the same access pattern a plain embedding table would
have, so the embedding-table analogy that legitimizes the flat active/token metric simultaneously
prevents attributing the gain to gauge structure. Two further learned tables co-move with the sweep
and are not frozen (`pos_phi_free (128, n_gen)` at model.py:375-376; `output_proj_weight (V,K)` +
`decode_bias`), verified reachable, so the causal share is not isolated in code.

## Concessions made
- Red conceded: the within-sweep CE effect is real, monotone, seed-robust and carries no internal
  D-confound; `phi_embed` is a genuine token-indexed embedding-table access pattern; the head mixer
  shrinks (16²→2²) and is not the driver.
- Blue conceded: `active/tok` flatness is a definitional identity, not a measurement; there is no
  reportable scaling exponent; the cross-sweep "complementary to width" comparison is a Chinchilla
  D-slice confound (withdrawn); "parameter-efficient" is unearned on real (wall-time/FLOP) compute;
  the gauge-vs-table mechanism is uncontrolled. Blue conceded the compound claim as written.

## Decisive evidence
`vfe3/run_artifacts.py:616-620` (reached via `:757`): `active = 5·V·K + 2·K + n_gen`, base
`5·50257·48 + 2·48 = 12,061,776` matching the CSV to the integer with only `+n_gen` (+1,008)
varying — the flat working set is an algebraic identity, not efficiency — taken together with the
CSV `wall_time_s` (U-shaped, GL24 = 11049s slowest) and the configs (245.76M vs 491.52M = exactly
2×). These three verified artifacts jointly refute the "parameter-efficient / complementary scaling
axis" conjuncts, while the CSV `test_ppl` (124.57 → 92.15, fixed within-sweep budget) verifies the
core effect. Both bodies of verified evidence are true and land on different conjuncts.

## My weighted scores
- Red weighted total (code/data): ~24 (six verified path:line ×3 = 18; two verified CSV/config
  outputs ×3 = 6), plus in-domain external citations (~+6). All facts re-verified TRUE.
- Blue weighted total (code/data): ~21 (five verified path:line ×3 = 15; two verified CSV/config
  outputs ×3 = 6), plus in-domain external citations (~+5). All facts re-verified TRUE.

The totals are close because both sides' verified facts are correct and non-conflicting; they
establish different conjuncts. Red's ledger refutes the efficiency/axis/complementary conjuncts;
Blue's ledger establishes the real within-budget effect. Neither side holds a verified code/data
fact that the other disputes.

## Outcome (this judge)
REMAND

## Reasoning
On a strict reading the compound claim fails: Red's verified path:line evidence
(run_artifacts.py:616-620 active identity, :626 transport 64×, the U-shaped wall-time, the exact 2×
token gap) shows the "parameter-efficient," "publishable scaling exponent," and "complementary to
width-scaling" conjuncts are unsupported or adverse on the code and data, so this is not a
vindication of the claim as written. But it is not a clean code-truth win for Red either, because
Blue's verified evidence (CSV `test_ppl` 124.57 → 92.15, identical within-sweep budget and
`data_sha256`, the genuine `(V, n_gen)` token-indexed lookup at prior_bank.py:167,682) establishes a
real, monotone, seed-stable, D-confound-free effect that survives all three of Red's attacks
untouched. Both sides cite verified path:line on different parts of the code path, all of it true and
uncontested between them, which is the textbook REMAND signature. The one question the code cannot
answer — is the gain gauge structure or raw table capacity — is unresolved because the discriminating
artifacts (a matched-budget 491.52M blocks run and a non-gauge `V × m` matched-size control) were
never run; only `block_glk` cells exist. The claim document itself pre-commits this exact scenario
("empirical improvement real but the parameter-efficient / publishable-as-is qualifier unsupported")
to REMAND, and both sides requested it. REMAND to the narrow, code-true sub-claim: at fixed
`embed_dim=48`, enlarging the GL gauge block GL3 → GL24 lowers cross-entropy strictly monotonically
(PPL 124.6 → 92.2) on a 245.76M-token budget, three-seed-robust — a genuinely new within-budget
gauge-block ablation, not a demonstrated parameter-efficient scaling axis complementary to width.

## Recommended action
Run the two decisive experiments both sides name before any "efficient scaling axis" framing is
published: (1) a matched 491.52M-token blocks_K48 run (`batch_size=64`, same `max_steps`) to remove
the cross-sweep D-confound; (2) a non-gauge `V × m` learned-table control at matched added parameters
and fixed head geometry to isolate the generator algebra from raw table capacity. Report the existing
result as a within-budget structural ablation curve on the wall-time / FLOP axis (not on
`active_params_per_token`, which is definitionally flat, and not as a fitted exponent over a 3.62×
range). Fix the evidence-pack line-31 `fpt_decode` label (4.82M, not 12M).
