# Verdict (canon-strict) - active-inference-lm-efficacy

## Verdict

REMAND

## Evidence audit

Verification note: canon location is `embedded`; no `external_bibliography.md` or `external_canon_*.md` file exists in this debate folder. I count a citation as verified when it is named in `01_evidence.md` or `01b_extended_evidence.md` and accepted by the canon-cop source-domain checks.

| Side | External citations (verified) | External citations (unverified) | sympy/FD | path:line | Canon-cop strikes |
|------|-------------------------------|---------------------------------|----------|-----------|-------------------|
| Red | 20: Friston 2016; Friston 2017; Smith 2022; Heins 2022; Sajid 2021; Buckley 2017; Vaswani 2017; Holtzman 2019; Finlayson 2023; Malekzadeh and Plataniotis 2022; Popper 1959; Lakatos 1978; Guo 2017; Kuleshov 2018; Rainforth 2023; Meister et al. 2022 locally typical sampling; Meister et al. 2022 probability-quality; Meister et al. 2023; Kaplan 2020; Hoffmann 2022 | 0 | 0 | 10 | 0 |
| Blue | 16: Friston 2016; Friston 2017; Smith 2022; Heins 2022; Sajid 2021; Buckley 2017; Vaswani 2017; Radford 2019; Kaplan 2020; Hoffmann 2022; Popper 1959; Lakatos 1978; Holtzman 2019; Finlayson 2023; Neal 1998; Malekzadeh and Plataniotis 2022 | 2: Cartwright 1999; Parr, Pezzulo, and Friston 2022 | 0 | 10 | 0 |

## Concessions made

- Red conceded: finite candidate continuations or agent sets can be treated as a policy set in active inference if each candidate induces predicted future outcomes through explicit likelihood, transition, preference, prior, and policy-prior objects; an opt-in no-grad inference-time scorer is more defensible than a train-time EFE replacement.
- Blue conceded: an arbitrary scalar reranker with active-inference vocabulary is not active inference; V3 does not currently expose a public final-belief or policy-rollout API; the model-channel agent-set path is flat; train-time EFE replacement is premature.

## Decisive evidence

Smith, Friston, and Whyte 2022, as summarized in `01_evidence.md` and `01b_extended_evidence.md`, supplies the operational active-inference template: likelihood `A`, transitions `B`, prior preferences `C`, initial-state prior `D`, policy prior `E`, plus risk-and-ambiguity EFE over predicted outcomes. That citation supports Blue's conditional finite-policy defense and Red's objection to any present scorer that has not instantiated those objects.

## My weighted scores

- Red weighted total: 64
- Blue weighted total: 65

These are relevance-weighted scores, not raw citation counts: repeated uses of the same source for the same proposition are counted once, V3 path references receive 1x only for implementation behavior, and canon-cop strikes are zero for both sides.

## Outcome (this judge)

REMAND

## Reasoning

The external canon does not reject finite policy sets. Friston 2017, Smith 2022, and Heins 2022 give Blue a verified canon path for a no-grad finite candidate scorer if it declares policies, predicted outcomes, likelihood, transitions, preferences, priors, policy precision, and risk-plus-ambiguity diagnostics. The same canon gives Red the decisive objection to the stronger reading: V3's current hook, by itself, returns logits and lacks a public belief-rollout object and live outcome-likelihood policy model, so calling the current unspecified scorer "theoretically legitimate active inference" outruns the record. Because the claim can mean either "legitimate if implemented with Smith-style objects" or "legitimate now as proposed from existing V3 hooks," the totals are near-tied across different subclaims. The dispatch instruction says to choose REMAND under this equivocation, so I do.

## Strength of each side

Red is strongest on canonical sufficiency. Its Friston/Smith/Sajid line shows that EFE is not a generic uncertainty penalty or log-probability reranker, and its Guo/Kuleshov plus Holtzman/Meister line shows why sigma and decoding gains require calibration, ablations, and matched baselines before they carry active-inference weight.

Blue is strongest on conditional legitimacy. It correctly anchors finite candidate scoring in discrete active-inference policy selection and keeps the proposal no-grad and inference-time first, which Friston 2016 and Smith 2022 support better than a train-time objective replacement.

## Action

Remand the exact subclaim: "A V3 no-grad generation-time scorer that explicitly implements Smith-style `A/B/C/D/E` analogs, a policy-conditioned belief rollout, predicted outcome distribution `q(o|pi)`, prior preferences `p(o|C)`, an ambiguity or epistemic term tied to outcome likelihood entropy or validated information gain, fixed horizon, fixed candidate generator, and policy precision is a theoretically legitimate active-inference policy-selection layer over finite candidate continuations or agent sets."

If that is the claim, Blue should win the canon question unless implementation evidence defeats it. If the claim is instead that the current V3 code/report already supplies a canonical active-inference scorer without those objects, Red should win.
