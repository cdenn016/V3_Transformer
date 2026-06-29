# Deep Peer Review Pass 18: GL(K) Manuscript Pair

Review date: 2026-06-27.

Reviewed files: `C:\Users\chris and christine\Desktop\Research\manuscripts\GL(K)_attention.tex`,
`C:\Users\chris and christine\Desktop\Research\manuscripts\GL(K)_supplementary.tex`, and the vault
`references.bib`. The local repo mirror under `Manuscripts-Theory/` was used only as a comparison
point because the project instructions identify the Research vault as the freshest WIP.

Recommendation: major revision before submission. I did not find a new equation-level defect in the
ledger-settled analytic core. The GL(K) invariance theorem, softmax/KKT construction, envelope versus
surrogate distinction, covariance fixed point, SPD retraction, and KL uniqueness appendix should not
be re-litigated on the evidence in this pass. The current blockers are empirical provenance,
reproducibility, and claim-status calibration introduced or exposed by the latest ablation integration.

## Method

I read the verified ledger in `Research/manuscripts/verified-ledger.md`, the Research wiki pages for
the VFE Transformer Program, GL(K) gauge-equivariant attention, the GL(K) manuscript note, the
2026-06-27 ablation suite, and the attention/gauge-equivalence themes. The peer-review skill calls
for a parallel expert panel for deep reviews; the available sub-agent tool is restricted unless the
user explicitly asks for delegation, so I ran the same lenses locally: gauge/math consistency,
information geometry, variational/free-energy, transformer/ML, empirical reporting, citation and
notation hygiene, and an adversarial check against the verified ledger.

Static integrity checks on the vault WIP:

| Check | Result |
| --- | --- |
| Source size | `GL(K)_attention.tex`: 2495 lines; `GL(K)_supplementary.tex`: 1495 lines |
| Labels | 209 labels, 0 duplicates |
| Cross-references | 343 refs, 0 missing targets |
| Citations | 104 distinct cite keys, 184 cite instances, 1 missing key |
| Missing cite | `Neal1998` at `GL(K)_attention.tex:2121`; absent from vault `references.bib` |
| Placeholders | 15 `TBD` entries, all in `tab:glk_hyperparams` |
| Banned LaTeX spacing macros | 0 instances of `\,`, `\;`, `\!` |
| UK spelling from project rules | 6 instances of `analogue` |

No LaTeX build was run for the vault WIP because the static BibTeX check already found an unresolved
key in the vault bibliography, and this review intentionally leaves the manuscript sources untouched.

## Claim-Status Table

| Claim | Current status | Review judgment |
| --- | --- | --- |
| KL attention softmax follows from entropy-regularized source assignment | D/P | Settled by ledger; no new issue. |
| Gaussian KL is GL(K)-invariant under transported pushforwards | P | Settled by ledger; no new issue. |
| Standard QK attention is recovered in the isotropic, flat, key-norm-constant limit | D-sharp/S mix | Current abstract/table mostly preserve the needed caveats. |
| Main-text K sweep reaches PPL 64.9 at K=120 | E | Not currently submission-ready; conflicts with Appendix J's live scaling table. |
| Learned gauge transport explains the internal ablation gap | E | Strong learned-versus-frozen result, but the causal sentence over-isolates what the controls prove. |
| Canonical attention entropy matters at low kappa and vanishes at high temperature | E | Plausible and well framed; keep single-seed/noise-floor caveats. |
| Reported phi dynamics use Killing/natural-gradient preconditioning | E/D mix | Stale or ambiguous relative to the ablation operating point; must be separated by run family. |
| Per-head temperature dispersion carries model information | E/S | Suggestive only from five architectures; one later paragraph overstates it. |

## Major Comments

1. The main-text K-sweep numbers conflict with the new Appendix J scaling table.

The abstract states that a GL+(10) embedding-dimension sweep reaches test PPL 64.9 at K=120
(`GL(K)_attention.tex:48`). The methods table repeats the sweep as K=10--120 with a 64.9 best
(`GL(K)_attention.tex:2078`) and says it improves from 194.5 at K=10 (`GL(K)_attention.tex:2081`).
The results paragraph lists the full old sequence 194.5, 127.9, 104.1, 92.7, 85.8, 80.5, 68.2,
64.9 for K=10,20,30,40,50,60,100,120 (`GL(K)_attention.tex:2281`). The table caption says this is
reproducible from `publication_outputs/scaling_analysis/aggregated_K_sweep.csv`
(`GL(K)_attention.tex:2274`).

Appendix J, however, now reports the V3 reference implementation scaling sweep as K=10..70 with
PPL 219.0, 135.7, 113.1, 101.4, 94.1, 88.9, 83.9 (`GL(K)_supplementary.tex:1460-1480`). The wiki
ablation note and `docs/2026-06-27-ablation-manuscript-digest.md` agree with the Appendix J numbers
and explicitly describe K=70 as single-seed. These cannot all be the same released/reproducible
sweep. A reviewer cannot tell whether the abstract headline number is an older best-validation
checkpoint, a different implementation, a different token budget, or stale text.

This is essential for acceptance because the abstract uses 64.9 as a headline empirical result. The
fix is to make one authoritative scaling table and propagate it consistently through the abstract,
Table `tab:glk_spec`, Table `tab:glk_results`, the results paragraph, and Appendix J. If the 64.9
curve is a separate older artifact, label it as such and state the checkpoint-selection policy; if
Appendix J supersedes it, remove 64.9 from the abstract until the K=120 value is rerun under the same
protocol.

2. The reproducibility package is not submission-ready while the hyperparameter table is still
`TBD` and the vault bibliography has an unresolved citation.

The training hyperparameter table still contains `TBD` for optimizer, schedule, batch size, token
budget, gradient clipping, weight decay, kappa, mu/sigma/phi learning rates, alpha, inner iterations,
seeds, checkpoint policy, and hardware (`GL(K)_attention.tex:2094-2112`). The Code Availability
section then says all result-section experiments can be reproduced from configuration files and
documented random seeds (`GL(K)_attention.tex:2483`). Those statements are not compatible in a
submission manuscript. The archived configs may exist, but the paper does not yet expose enough
information for a referee to reproduce or even audit the reported runs without project-local
knowledge.

There is also one live BibTeX break in the vault WIP: `GL(K)_attention.tex:2121` cites `Neal1998`,
but `Research/manuscripts/references.bib` does not contain that key. The repo mirror's
`Manuscripts-Theory/references.bib` does contain it, which means the vault and mirror bibliography
state has diverged. Since the project instructions route manuscript work to the vault, the vault
bibliography is the one that matters for this review.

The fix is straightforward: fill `tab:glk_hyperparams` from the actual archived configs, include
per-run values where the Japanese, K=90, old sweep, and Appendix J runs differ, state validation/test
selection policy, and add the `Neal1998` BibTeX entry to the vault bibliography.

3. The manuscript overgeneralizes "natural-gradient dynamics" and "reported runs use Killing" across
run families.

The abstract and methods describe the trained gauge transformers as using natural-gradient dynamics
(`GL(K)_attention.tex:48`, `GL(K)_attention.tex:2063`). The main text later gives a better caveat:
the natural-gradient label is exact for the belief channel, while the gauge frame and M-step optimizer
need separate qualifications (`GL(K)_attention.tex:1582`). The supplement still says the reported
runs use the block-diagonal Killing-form variant (`GL(K)_supplementary.tex:560`,
`GL(K)_supplementary.tex:578`, `GL(K)_supplementary.tex:673`).

That supplement statement no longer matches the ablation operating point that Appendix J reports as
the released V3 reference implementation. In `ablation.py`, the baseline used for the K=20 ablation
suite has `m_phi_natural_grad = False` and `phi_precond_mode = "killing_per_block"`, meaning phi is
stepped by AdamW, not by the Killing natural-gradient branch. The June 27 digest flags the same
tension: AdamW-on-phi gives PPL 144.5, while the natural-gradient variants give 252.3 and 271.8 in
the available sweep; a later analysis argues the natural-gradient hypothesis is refuted at that
operating point, with caveats about external validity.

This does not invalidate the analytic derivation, but it does make the empirical prose too broad.
The fix is to separate channels and run families explicitly: mu/sigma E-step uses the Gaussian
Fisher natural gradient; phi M-step is AdamW for the V3 ablation suite unless a run config says
`m_phi_natural_grad=True`; older K=90 or Japanese runs should name their actual optimizer from the
archived configs. The supplement should not say "the reported runs use" a Killing variant unless the
claim is scoped to the exact run family that did.

4. The gauge-transport ablation is strong, but the final causal interpretation is one step too broad.

The new main-text ablation paragraph reports learned transport at PPL 154, frozen random transport at
279, and identity transport at 267 (`GL(K)_attention.tex:2294`). This is a strong result. The
learned-versus-frozen contrast is parameter-matched and shows that learning the frame is load-bearing
rather than merely allocating parameters to phi. The paragraph also discloses that the identity arm
removes the learned positional gauge and is therefore confounded.

The last sentence then says the ablation "locates the advantage in the gauge-transport geometry
rather than in the diagonal-covariance beliefs or the KL-divergence attention alone"
(`GL(K)_attention.tex:2294`). That is stronger than the controls prove. Frozen-random versus learned
tests trainable transport against a harmful fixed random frame. Identity versus learned tests
transport plus positional-gauge changes unless an identity-with-matched-positional-parameters control
exists. The fair conclusion is: learned GL(K) frame transport is load-bearing and cannot be reduced
to parameter count; the comparison strongly supports the transport-geometry claim, but a clean
identity-with-position control is still needed to isolate "transport" from all positional-gauge
effects and to rule out every "KL attention alone" interpretation.

5. The per-head temperature claim is promoted above its evidence in the architectural implications.

The supplement is careful: with only five architectures, the association between temperature
dispersion and alpha-beta correlation is exploratory rather than a dominant explanatory variable
(`GL(K)_supplementary.tex:853`, `GL(K)_supplementary.tex:877`). Later the main paper says the RoBERTa
median/std "demonstrates that this covariance structure carries real information that a single global
temperature discards" (`GL(K)_attention.tex:2341`). That verb is too strong for an n=5
cross-architecture association and a single-temperature approximation analysis. Recast it as
"suggests" or "is consistent with" and keep the useful falsifiable claim: per-head temperature
heterogeneity should predict where a single-temperature KL approximation fails, to be tested on more
architectures and with direct per-head optimal-temperature distributions.

## Minor Comments

The Supplementary Material paragraph still lists only Appendices A through I
(`GL(K)_attention.tex:2490`), and the supplement opening summary also stops at Appendix I
(`GL(K)_supplementary.tex:44`). Appendix J now exists and should be named in both places.

`GL(K)_attention.tex:2274` says "The K = 10 VFE results are forthcoming," but the manuscript now
reports K=10 values in both the older main-text sweep and Appendix J. Delete or update this sentence
as part of the K-sweep reconciliation.

The main text contains six instances of `analogue` (`GL(K)_attention.tex:931`, `1610`, `1655`,
`2012`, `2336`; `GL(K)_supplementary.tex:130`). The project style requires American English, so use
`analog` in touched prose.

`GL(K)_attention.tex:2298` has a sentence splice: "rather than count-based smoothing suggesting..."
should be split or punctuated.

The vault WIP and repo mirror are no longer byte-identical. Hashes differ for both GL(K) source files,
and the repo mirror bibliography is a superset containing `Neal1998`. This is not a manuscript defect
by itself, but it increases the risk of applying fixes to the wrong copy. Continue treating
`Research/manuscripts` as the manuscript source for reviews and edits.

## Questions For The Author

Which scaling curve should be authoritative in the abstract: the older K=10..120 curve ending at
64.9, or the new V3 K=10..70 Appendix J curve ending at 83.9? If both remain, what exact differences
in implementation, seed count, token budget, checkpoint policy, and test evaluation justify reporting
both?

For each headline empirical row, what optimizer actually updates phi: AdamW, Killing preconditioned
natural-gradient, pullback natural-gradient, or the newer Adam-on-preconditioned-gradient hybrid?

Is the identity gauge-transport arm intended to remove positional phi by design, or should there be a
matched identity-transport control that preserves the positional parameter allocation while forcing
Omega_ij = I?

Should Appendix J's single-seed ablations be reported in the main text with exact seed values and
validation-selection policy, or kept in the supplement until the three-seed repeats are complete?

## Bottom Line

The paper's mathematical center remains strong and heavily verified. The current submission risk is
that the empirical story now contains two incompatible K-scaling narratives, a still-empty
hyperparameter table, a broken vault bibliography key, and optimizer prose that no longer cleanly
matches the V3 ablation runs. Fix those before sending the manuscript to reviewers; otherwise a
referee can reject on reproducibility and result provenance without touching the gauge-theoretic
derivations.
