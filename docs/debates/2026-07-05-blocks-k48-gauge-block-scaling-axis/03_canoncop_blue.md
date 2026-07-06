# Canon-cop report — blocks-k48-gauge-block-scaling-axis — Phase 3.5 — BLUE

STRIKES=0 STATUS=RECORD

## Strike list

None. (0 strikes — below the soft cap; RECORD, no rewrite.)

```json
{
  "debate": "blocks-k48-gauge-block-scaling-axis",
  "phase": "3.5",
  "side": "blue",
  "target": "03_blue_rebuttal.md",
  "grep_pass": "skipped (embedded canon, no external_bibliography.md / canon-dir)",
  "total_strikes": 0,
  "status": "RECORD",
  "strikes": []
}
```

## Grep pass (canon_cop_validator.py)

Grep pass skipped (embedded canon). The validator hard-requires `--bibliography`
and `--canon-dir` and errors without them (`error: the following arguments are
required: --bibliography, --canon-dir`); this debate carries no
`external_bibliography.md`. Per dispatch the phase is not failed over this. A
direct Grep sweep was substituted against the target for both authority patterns
and banned phrases:

- Authority / circularity patterns (`Attention/`, `CLAUDE.md`,
  `user_theory_summary`, `.tex`, `as shown in`, `as established`, `by
  construction`, `our framework`, `GL(K)_attention`, `PIFB`, `as derived in`,
  `the manuscript shows/proves/establishes/derives`): two mechanical hits, both
  adjudicated non-strikes below (L102 code-architecture description; L168-169
  explicit manuscript-as-authority DISCLAIMER — the forbidden-to-strike case).
- Banned phrases (`key insight`, `crucially`, `critically`, `notably`,
  `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in
  particular`, `leverages`, `underscores`, `perhaps`, `it could be argued`, `one
  might suggest`, `both sides have a point`): **no matches**.

## LLM pass — subtle patterns

| Pattern | Line | Strikes | Note |
|---------|------|---------|------|
| manuscript-as-authority (a) — BLUE signature failure, double-weight | 131,134,168-169 | 0 | Every manuscript reference is either a NOVELTY claim ("absent from either manuscript", "a genuinely new phenomenon absent from either manuscript") or the explicit CIRCULARITY DISCLAIMER at L166-172 ("No step relies on `GL(K)_attention.tex`, `PIFB.tex`, `CLAUDE.md`... those artifacts are the claim under evaluation"). Manuscript is the object under evaluation, never the standard. Forbidden to strike per dispatch. |
| in-repo CODE path:line for behavior (allowed) | 21,98-99,113,118 | 0 | `run_artifacts.py:616-620`, `prior_bank.py:167/682`, `groups.py:144-152`, `generators.py:96,103`, `transport.py` are all framed as what the code DOES ("resolves to", "is a learned (V,n_gen) table", "builds GL(g)^(48/g)", "acting by the sandwich"). Explicitly permitted. |
| reasoning-by-construction circularity (b) — "flat active/token ⇒ efficient" | 19-22,45-46,108-109 | 0 | The signature circular move is affirmatively REFUSED. L19-22 concedes the proxy is "a definitional identity, not a measurement"; L108-109 states the low working set "is not by itself an efficiency proof — blue conceded that the compute axis is adverse." The flat-⇒-efficient inference is NOT reintroduced. |
| "by construction" (mechanical hit) | 102 | 0 | Describes a code-STRUCTURAL fact (the `phi_embed` (V,n_gen) table + per-token row lookup mechanically yields the total-vs-active pattern), immediately hedged as "not by itself an efficiency proof." Not a circular justification of the debate claim. |
| gauge coboundary/cocycle claim | 118-120 | 0 | "Ω_ij = g_i g_j⁻¹ is a coboundary satisfying the cocycle condition" is a genuine math fact (g_i g_j⁻¹·g_j g_k⁻¹ = g_i g_k⁻¹), used to verify code transport matches `transport_mode='flat'`. Not the "we defined it so it holds" fallacy against the debate claim. |
| self-computed detection statistics (method stated) | 77-82,148-151 | 0 | Per-step σ (4.07/7.04), permutation p=(1/120)³=5.8e-7, Spearman ρ / Kendall τ = −1.000 are self-computed with stated assumptions (exchangeability within seed, seed independence, conservative vs seed-mean SE). Arithmetic checks (120³=1,728,000; 1/1,728,000=5.79e-7). A sound, stated self-computed statistic is not a citation strike. |
| hand-wave / wrong-domain / fabricated citation (c) | all external cites | 0 | All external authorities real, in-domain, faithfully represented (verification below). |
| banned phrases (d) | — | 0 | None present. |

### External citation verification

- **Bogen & Woodward 1988, *Philosophical Review* 97:303-352** — "Saving the
  Phenomena"; the data-vs-phenomena distinction is exactly this paper's thesis.
  Volume/pages correct. In-domain, correct use (L56-59).
- **Hacking 1983** — *Representing and Intervening*; "experiment has a life of
  its own, independent of theory" is a genuine Hacking thesis. Correct (L61-63).
- **Duhem 1906, p.187** — *The Aim and Structure of Physical Theory*; underde-
  termination of which hypothesis to revise. Correct, in-domain (L65-67).
- **Pearl 2009, *Statistics Surveys* 3:96-146, p.99** — "Causal inference in
  statistics: An overview." Journal, volume, and page range all correct; the
  P(y|do(x)) vs P(y|x) association-vs-intervention boundary is the paper's core.
  Correct, in-domain (L68-71).
- **Hansen 1998, *Rank-Deficient and Discrete Ill-Posed Problems*** (SIAM) —
  condition number bounds the inverse (recovery) map. Correct, in-domain
  numerical-analysis use (L74-76).
- **Jonckheere 1954 / Page 1963** — distribution-free tests against ordered
  alternatives (Jonckheere-Terpstra; Page's trend test). Correct, in-domain use
  for the ordered-detection framing (L82).
- **Kaplan et al. 2020, §2.1** — N = "the number of non-embedding parameters";
  embedding exclusion yields a cleaner single trend. Correct convention, in-domain
  (L102-104). Application to `phi_embed (V,n_gen)` is appropriately hedged.
- **Hoffmann et al. 2022 (Chinchilla)** — L(N,D)=E+A/N^α+B/D^β; used as the
  D-slice confound formalism. Correct functional form (L33-34,89-90).
- **Shazeer et al. 2017 / Fedus, Zoph & Shazeer 2021** — total-vs-active capacity
  axis against a flat compute axis ("a constant computational cost"). Correct,
  in-domain (L39,106-107).
- **Stumpf & Porter 2012** — "Critical Truths About Power Laws" (*Science* 335);
  the ≥2-decade dynamic-range requirement for a credible power law. Correct,
  in-domain (L28-29).
- **Belsley-Kuh-Welsch (condition-number/collinearity diagnostic, threshold 100)**
  — real diagnostic source; threshold within the range BKW discuss for severe
  collinearity. A stated numerical diagnostic, not a fabrication (L27).
- **Cohen & Welling 2016 / Kondor & Trivedi 2018** — group enlargement as a
  capacity axis (p4→p4m gains at fixed params) and equivariant-map = group
  convolution. In-domain; the load-bearing "group-as-capacity" claim is directly
  supported. The "orthogonal to width" rider is blue's own (K,g) structural
  argument, not attributed to the papers — fair extension, not a mis-cite (L122-126).
- **Nakahara 2003, Ch.9-10** — fibre bundles (Ch.9) and connections/holonomy
  (Ch.10); flat, trivial-holonomy connection. CORRECT chapters and RIGHT domain
  (differential geometry of connections), not the wrong-domain trap (L120).
- **Michel et al. 2019 / Voita et al. 2019** — head pruning; head-count reduction
  a largely free knob. Correct, in-domain, correctly used in falsifier 2 (L143-144).

### Intra-debate cite

- **L52 cites `02_red_opening.md` l.22** ("the CE decrease 'is real, monotone,
  and seed-robust'"). Verified against the red opening (L21-22: "We accept the
  empirical core of that steelman: the CE decrease is real, monotone, and
  seed-robust"). Faithful quotation of red's own concession; an intra-debate
  reference, not a canon cite. No strike.

## Prose summary

Zero strikes. The Blue rebuttal is clean on the source-of-truth precedence rule
and, more consequentially, on Blue's signature failure modes.

Manuscript-as-authority (the double-weighted trap): never committed. Every
mention of `GL(K)_attention.tex` / `PIFB.tex` / `CLAUDE.md` sits inside either a
novelty claim (the effect is "absent from either manuscript") or the explicit
self-policing circularity disclaimer in the "Circularity check" section, which
correctly frames those artifacts as the claim under evaluation and names the
would-be circular move ("the manuscript derives gauge structure as a capacity
axis") in order to refuse it. That is the forbidden-to-strike posture, not a
violation. All in-repo references are path:line behavior cites, which are
permitted.

Reasoning-by-construction circularity: the "active/token is flat ⇒ efficient"
inference is not reintroduced. Blue concedes up front that the proxy is a
definitional identity, and where it defends the low per-token working set as a
genuine access-pattern property it immediately hedges that this "is not by itself
an efficiency proof." The one mechanical "by construction" (L102) describes the
`phi_embed` table's architecture producing the total-vs-active signature — a code
fact — not a circular justification of the debate claim. The gauge coboundary /
cocycle statement (L118-120) is a genuine mathematical verification that the code
transport matches `transport_mode='flat'`, not a defined-so-it-holds fallacy.

Citations: every external authority — Bogen & Woodward, Hacking, Duhem, Pearl,
Hansen, Jonckheere, Page, Kaplan, Hoffmann, Shazeer, Fedus-Zoph-Shazeer, Stumpf &
Porter, Belsley-Kuh-Welsch, Cohen & Welling, Kondor & Trivedi, Nakahara, Michel,
Voita — is real, in-domain, and faithfully represented. Pearl 2009 and Bogen &
Woodward 1988 carry correct journal/volume/page detail; Nakahara Ch.9-10 is the
right domain (fibre bundles and connections), not the wrong-domain trap. The
self-computed detection statistics (σ margins, the (1/120)³ = 5.8e-7 permutation
probability, ρ/τ = −1.000) are method-stated with explicit assumptions and check
arithmetically; per dispatch a sound stated self-computed statistic is not a
citation strike. No banned phrases.

Action: RECORD. Below the soft cap; debate continues. Judges may note that Blue's
canon discipline remains clean into the rebuttal and that both the
manuscript-authority and flat-⇒-efficient guardrails were honored.
