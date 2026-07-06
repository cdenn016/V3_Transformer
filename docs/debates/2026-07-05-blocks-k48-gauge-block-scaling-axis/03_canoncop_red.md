# Canon-cop report — blocks-k48-gauge-block-scaling-axis — Phase 3.5 (rebuttal) — RED

STRIKES=0 STATUS=RECORD

```json
{
  "debate": "blocks-k48-gauge-block-scaling-axis",
  "phase": "3.5",
  "round": "rebuttal",
  "side": "red",
  "target": "03_red_rebuttal.md",
  "grep_pass": "SKIPPED (embedded canon, no external_bibliography.md / canon-dir; validator hard-requires both args)",
  "total_strikes": 0,
  "status": "RECORD",
  "strikes": []
}
```

## Grep pass

Skipped. `canon_cop_validator.py` hard-requires `--bibliography` and `--canon-dir`; this debate's
canon is embedded (`00_claim.md:9` "Canon location: embedded"), so no `external_bibliography.md`
or canon-dir exists. Substituted a direct Grep sweep (manuscript-as-authority patterns +
banned-phrase list). Results feed the LLM pass below, which is the primary adjudication per
dispatch ("if it errors on the missing args, SKIP it and note").

Direct Grep results on the target:
- Banned-phrase list (`key insight`, `crucially`, `critically`, `notably`, `importantly`,
  `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`,
  `underscores`, `perhaps`, `it could be argued`, `one might suggest`, `both sides have a
  point`): **no matches**.
- Manuscript-family patterns (`Attention/`, `GL(K)`, `.tex`, `PIFB`, `user_theory_summary`,
  `as shown in`, `as established`, `our framework`, `by construction`, `CLAUDE.md`): **one hit**
  — line 98, `CLAUDE.md`. Adjudicated below as 0 strikes (code-behavior documentation used
  adversarially, not canonical authority).

## LLM pass — per rule

| Rule | Finding | Strikes |
|---|---|---|
| (a) manuscript-as-authority | None. RED cites only in-repo CODE at path:line (`prior_bank.py:167,682`, `head_mixer.py:105-107`, `run_artifacts.py:611,616-620,626`, `model.py:375-376`) for what the code DOES — allowed, not a strike. The one `CLAUDE.md` reference (L98) invokes its "documented exceptions" for the factual claim that `use_head_mixer=True`/`use_prior_bank=False` break strict gauge equivariance off identity-init; this is code-behavior documentation turned AGAINST the claim (the runs are not on the equivariant path), not authority for a canonical form. RED explicitly treats the manuscripts as the *claim under evaluation*, noting the axis "appears in neither manuscript" (L52). | 0 |
| (b) reasoning-by-construction circularity | None. The opposite: RED *exposes* circularity in the claim — the `active/token` flatness "is an algebraic identity, not an efficiency finding … that is circular" (L135-141), and the coboundary triviality of `Omega_ij` is used to deny (not assert) gauge content, sourced to Nakahara/Bleecker (L92-94). | 0 |
| (c) hand-wave-with-citation / wrong-domain | None. Every external cite lands on a sentence it actually supports; all in-domain (see detail). | 0 |
| (d) fabricated / unverifiable canonical cites | None. All external sources are real and say what is claimed; two verbatim gauge quotes re-verified against arXiv this pass. | 0 |
| (e) banned phrases | None (grep clean). | 0 |

## Citation verification detail

- **Hacking 1983, *Representing and Intervening*** ("a life of its own") — genuine; Hacking's canonical phrase for robust experimental phenomena. RED both concedes it (L21-22) and turns it against Blue (the observation/interpretation cut that shields the effect denies the theory-laden reading, L56-59) — legitimate, in-domain. OK.
- **Vaswani et al. 2017 §3.4** (token-embedding table) — §3.4 is "Embeddings and Softmax"; correct section. **§3.2.2** (`d_model = h·d_k`, L126) — §3.2.2 is "Multi-Head Attention"; correct. **Radford et al. 2019** (GPT-2) — genuine. OK.
- **Mill 1843, *A System of Logic*, bk. III ch. 8** (Method of Difference) — Book III Ch. 8 is "Of the Four Methods of Experimental Inquiry"; correct locus. **Woodward 2003, *Making Things Happen*, ch. 2** (interventionism / vary-the-cause-alone) — genuine, correct domain. OK.
- **Cohen & Welling 2016** (verbatim, L81-82) — RE-VERIFIED against arXiv:1602.07576 abstract: "G-convolutions increase the expressive capacity of the network without increasing the number of parameters" and "substantially higher degree of weight sharing" are BOTH verbatim. The dispatch watch-item "Cohen & Welling fixed-parameter weight-sharing" is satisfied; RED's inversion ("blocks_K48 does the opposite — params grow 3.62×") is a legitimate contrast, not a misattribution. OK.
- **Kondor & Trivedi 2018 / Weiler & Cesa 2019** (group as constraint that reduces admissible-map space, L83-84) — genuine (equivariance-as-constraint is the correct reading of both). OK.
- **Weiler, Forré, Verlinde & Welling 2021 (arXiv:2106.06020)** — RE-VERIFIED: title "Coordinate Independent Convolutional Networks — Isometry and Gauge Equivariant Convolutions on Riemannian Manifolds," authors correct; abstract does tie coordinate-independence + weight-sharing to a requirement of local-gauge equivariance. arXiv number correct, in-domain. OK.
- **Nakahara 2003, §10.2–10.3; Bleecker 1981, ch. 3** (flat = zero curvature = trivial holonomy; coboundary is gauge-equivalent to trivial, L92-94) — correct domain (differential geometry of connections on fiber bundles). This is NOT the skill's "Nakahara wrong-domain" pattern (that pattern is Nakahara mis-attached to variational inference); here the cite lands exactly on holonomy/curvature. Substance textbook-correct. OK.
- **Hoffmann et al. 2022 / Chinchilla** (`L̂(N,D) = E + A/N^α + B/D^β`, E=1.69, A=406.4, B=410.7, α=0.34, β=0.28, additive in D; "doubling model size → double tokens," L109-116) — coefficients match the published Approach-3 parametric fit; the additive-in-D floor and the 2×-token-confound reading are correct. Dispatch watch-item "Chinchilla L(N,D) additive-in-D floor" satisfied. OK.
- **Kaplan et al. 2020 §1.3, §3.1** ("N … excluding all vocabulary and positional embeddings"; shape params weak at fixed N, L119-124) — the N-definition is verbatim Kaplan; the weak-shape-dependence finding is genuine. Correct use to argue `phi_embed` is the excluded embedding table and `n_heads` is a weak knob. OK.
- **Michel et al. 2019; Voita et al. 2019** ("pruning 38 out of 48 encoder heads results in a drop of only 0.15 BLEU," L128) — genuine; the Voita figure was verified verbatim against arXiv:1905.09418 in the Phase-2.5 pass. OK.
- **Williams et al. 2009 (roofline)** (full-vocab decode bandwidth-bound; wall-time is ground truth, L146-148) — genuine CACM roofline model; correct domain. OK.
- **Fedus, Zoph & Shazeer 2021 / Switch** ("a constant computational cost," L148-150) — verified prior (arXiv:2101.03961). Dispatch watch-item "Switch constant-FLOP" satisfied; RED uses the precondition correctly (blocks buys its best loss at highest wall-clock, so the constant-compute antecedent is not met). OK.
- Clauset/Stumpf-Porter — NOT cited in this rebuttal (they appeared in the Phase-2 opening); nothing to check here.

## Internal-consistency spot checks (diligence, not strikes)

RED's arithmetic remains consistent with the evidence pack: base active
`5·50257·48 + 2·48 = 12,061,776` (exact); the only sweep-varying active term `+n_gen`
(+1,008 = +0.0084%); transport `2·N·g²` at N=128 grows 9·256-fold i.e. 64× (9 → 576, L54;
147,456 param form in the memo); `fpt_decode = 2·V·K = 4,824,672` (fixed). Ranges: total params
3.62×, `n_heads` 16 → 2, `n_gen` 144 → 1152 (untied `48·g`), wall-time GL24 11049s ≈ 2.37× the
GL8 minimum. The L153-155 convention correction (evidence `01_evidence.md:31` mislabels ~12M as
`fpt_decode`; code value 4.82M) is carried against RED's own interest — diligence.

## Summary

Zero strikes. RED's rebuttal is clean from a canon-cop standpoint. It attacks the compound claim,
so manuscript-as-authority risk is low a priori, and it does not slip: mechanism is grounded in
cited in-repo code (allowed), the normative gauge/scaling standards are all external literature,
and the two load-bearing "verbatim" gauge quotes (Cohen & Welling 2016; Weiler-Forré-Verlinde-
Welling 2021 / arXiv:2106.06020) were re-verified against arXiv this pass and are exact and
in-domain. The single `CLAUDE.md` reference (L98) is code-behavior documentation used adversarially
against the claim, not canonical authority — not a strike per the "in-repo behavior allowed" and
"claim-under-evaluation" carve-outs. No banned phrases. No reasoning-by-construction circularity —
RED in fact turns that fallacy against the claim (the `active/token` identity and the coboundary
triviality). All external citations real, in-domain, accurately quoted. Action: RECORD. Debate
continues; no rewrite triggered.
