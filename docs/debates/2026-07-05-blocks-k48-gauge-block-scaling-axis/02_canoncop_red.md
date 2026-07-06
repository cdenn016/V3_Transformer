# Canon-cop report — blocks-k48-gauge-block-scaling-axis — Phase 2.5 (opening) — RED

STRIKES=0 STATUS=RECORD

```json
{
  "debate": "blocks-k48-gauge-block-scaling-axis",
  "phase": "2.5",
  "round": "opening",
  "side": "red",
  "target": "02_red_opening.md",
  "grep_pass": "SKIPPED (embedded canon, no external_bibliography.md / canon-dir; validator requires both args)",
  "total_strikes": 0,
  "status": "RECORD",
  "strikes": []
}
```

## Grep pass

Skipped. `canon_cop_validator.py` hard-requires `--bibliography` and `--canon-dir`; this
debate's canon is embedded (`00_claim.md` line 9: "Canon location: embedded"), so no
`external_bibliography.md` or canon-dir exists. The LLM pass below is the primary and only
adjudication, per dispatch instructions ("if it errors on the missing args, SKIP it and note").

## LLM pass — per rule

| Rule | Finding | Strikes |
|---|---|---|
| (a) manuscript-as-authority | None. RED cites only in-repo CODE at path:line (`run_artifacts.py:616-620`, `prior_bank.py:167,682`, `free_energy.py:42-54`, `head_mixer.py:105-106`) for what the code DOES — allowed and not a strike. It never cites `GL(K)_attention.tex` / `CLAUDE.md` / `user_theory_summary.md` as the standard; it explicitly notes the result "is not in either manuscript." | 0 |
| (b) reasoning-by-construction circularity | None. RED does the opposite of committing this fallacy: it *exposes* the claim's circularity — the "flat ~12.06M working set" is shown to be "an algebraic identity, not an efficiency measurement" (lines 48-59). | 0 |
| (c) hand-wave-with-citation / wrong-domain | None. Every external cite lands in-domain and supports its sentence (see verification below). | 0 |
| (d) fabricated / unverifiable canonical cites | None. All twelve external sources are real and say what is claimed (see verification below). | 0 |
| (e) banned phrases | None. Grep for the full banned list returned no matches. | 0 |

## Citation verification detail

- **Shazeer et al. 2017** ("without a proportional increase in computation") — real (Sparsely-Gated MoE); accurate paraphrase of the conditional-computation motivation. OK.
- **Fedus, Zoph & Shazeer 2021 / Switch Transformer** ("increase the parameter count while keeping the FLOPs per example constant") — verified via arXiv:2101.03961; abstract confirms "sparsely-activated model ... but a constant computational cost." Accurate framing, in-domain. OK.
- **Kaplan et al. 2020** ("enters a regime of diminishing returns if either N or D is held fixed while the other increases") — genuine verbatim quote from the Scaling Laws paper. OK.
- **Hoffmann et al. 2022 / Chinchilla** (`L(N,D) = E + A/N^α + B/D^β`) — correct statement of the Chinchilla parametric loss (their Approach 3). The dispatcher's specific "mis-cite Chinchilla's L(N,D)" watch-item is NOT triggered; the form and the joint-N,D dependence are represented correctly. OK.
- **Clauset, Shalizi & Newman 2009** (LS on the log "generate significant systematic errors"; "error estimate gives no warning of the bias") — verified via arXiv:0706.1062; abstract confirms least-squares "can produce substantially inaccurate estimates" and methods "give no indication of whether the data obey a power law." Substance matches; in-domain (power-law-fit critique). Note the dispatcher's "Clauset power-law credibility criteria" watch-item: RED does NOT attribute the two-orders-of-magnitude span rule to Clauset — it correctly attributes that to Stumpf & Porter 2012 and gives Clauset only the LS-bias critique. Correct attribution split; no strike.
- **Stumpf & Porter 2012** ("credible empirical power law should span at least two orders of magnitude") — genuine ("Critical Truths About Power Laws," Science 2012). OK.
- **Vaswani et al. 2017, §3.2.2** (`d_k = d_model/h`) — §3.2.2 is exactly "Multi-Head Attention"; the partition is correct. Section number is right. OK.
- **Michel et al. 2019** (heads removable at test time; some layers reducible to one head) — genuine ("Are Sixteen Heads Really Better than One?"). OK.
- **Voita et al. 2019** ("pruning 38 out of 48 encoder heads results in a drop of only 0.15 BLEU") — verified via arXiv:1905.09418; VERBATIM match. OK.
- **Duhem 1906, p. 187** ("the experiment does not designate which one should be changed") — genuine Duhem-Quine thesis; substance correct, page plausible (Wiener translation). OK.
- **Cartwright 1999, p. 50** (nomological machine / over-export) — genuine ("The Dappled World"); the concept and its application are represented faithfully. OK.
- **Popper 1963** ("irrefutability is not a virtue of a theory but a vice") — genuine verbatim quote (Conjectures and Refutations). OK.

## Internal-consistency spot checks (not strikes, diligence notes)

RED's arithmetic ties to the evidence pack: base active `5·50257·48 + 2·48 = 12,061,776`
(exact); transport `2·N·g²` at N=128 gives `2304 → 147456` (64x, exact); `fpt_decode =
2·V·K = 4,824,672` (exact); dynamic ranges `log10(3.62)=0.559` and `log10(8)=0.903` (exact).
RED also self-corrects an evidence-pack error (line-31 "~12M/token" for decode conflates the
4.82M FLOP proxy with the 12.06M active-param count) and notes the conclusion survives either
number — diligence, not overreach.

## Summary

Zero strikes. RED is a clean opening from a canon-cop standpoint. It attacks the claim, so
manuscript-as-authority risk is low a priori, and it does not slip: it grounds every mechanism
in cited in-repo code (allowed) and every normative standard in verified external literature
(Shazeer/Fedus for the FLOP-matched conditional-computation warrant, Kaplan/Hoffmann for
D-slice non-identification, Clauset/Stumpf-Porter for ill-conditioned power-law fitting,
Vaswani/Michel/Voita for the multi-head-partition confound, Duhem/Cartwright/Popper for the
philosophy-of-science frame). All twelve external citations are real, in-domain, and quoted
accurately (two verified verbatim against arXiv this pass). No banned phrases. No
reasoning-by-construction circularity — RED in fact turns that fallacy against the claim.
Action: RECORD. Debate continues; no rewrite triggered.
