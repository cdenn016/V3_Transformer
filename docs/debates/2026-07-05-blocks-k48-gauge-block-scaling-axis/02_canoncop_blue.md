# Canon-cop report — blocks-k48-gauge-block-scaling-axis — Phase 2.5 — BLUE

STRIKES=0 STATUS=RECORD

```json
{
  "debate": "blocks-k48-gauge-block-scaling-axis",
  "phase": "2.5",
  "side": "blue",
  "target": "02_blue_opening.md",
  "grep_pass": "skipped (embedded canon, no external_bibliography.md / canon-dir)",
  "total_strikes": 0,
  "status": "RECORD",
  "strikes": []
}
```

## Grep pass (canon_cop_validator.py)

Skipped. Canon is `embedded`; the validator hard-requires `--bibliography` and
`--canon-dir` and errors without them (`error: the following arguments are
required: --bibliography, --canon-dir`). Per dispatch instructions the phase is
not failed over this. Mechanical patterns were instead checked directly with
Grep against the target: `Attention/`, `CLAUDE.md`, `user_theory_summary`,
`.tex`, `as shown in`, `as established`, `by construction`, `our framework`,
`GL(K)_attention`, `PIFB` — **no matches**. Banned-phrase sweep (`key insight`,
`crucially`, `critically`, `notably`, `importantly`, `it's worth noting`,
`interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`,
`perhaps`, `it could be argued`, `one might suggest`, `both sides have a point`)
— **no matches**.

## LLM pass — subtle patterns

| Pattern | Line | Strikes | Note |
|---------|------|---------|------|
| manuscript-as-authority (a) | — | 0 | The only manuscript references (L7: "appears in neither manuscript, which have only ever grown width at fixed block") describe what the manuscripts contain to support a *novelty* claim. The manuscript is the object under evaluation, not the standard. Correct treatment, not a strike. |
| in-repo code cites | 39,47,51,55 etc. | 0 | `prior_bank.py:167/682`, `run_artifacts.py:616-620/625`, `groups.py:144-152`, `transport.py:183`, `head_mixer.py:105-107` all cite what the code DOES. Explicitly allowed, not strikes. |
| reasoning-by-construction circularity (b) | 25 | 0 | Blue's signature failure mode is affirmatively refused: "Blue does not lean on the flatness of `active_params_per_token` as proof of efficiency ... using its flatness as evidence of efficiency would be circular (philosophy-of-science memo, mandatory flag)." Efficiency is grounded in wall-time/FLOPs (L25,33,55), and the compute picture is conceded mixed. Compliant with the brief. |
| metric-legitimacy vs efficiency-proof | 42 | 0 | L42 ("a flat active/token at growing `n_gen` is the expected ... property, not a metric pathology") defends the metric's *legitimacy as an access pattern*, not "flat therefore efficient." The two are separated at L25. Not circular. |
| hand-wave / wrong-domain citation (c) | 41,49,55,61 | 0 | Every external cite supports its sentence and sits in-domain (see verification below). |
| fabricated / unverifiable canon (d) | 35,41,49,55,59,61 | 0 | All sanity-checked; all real and correctly represented (see below). |
| banned phrases (e) | — | 0 | None present. |

### External citation verification

- **Vaswani et al. 2017 §3.4** — "Embeddings and Softmax"; learned input-embedding
  table used one row per token. Correct section, correct claim.
- **Radford et al. 2019 (GPT-2)** — 50257-vocab BPE, `[50257 x 768]` small-model
  embedding. Correct; project `vocab_size = 50257` matches.
- **Shazeer et al. 2017** — "Outrageously Large Neural Networks" (Sparsely-Gated
  MoE); ">1000x improvements in model capacity with only minor losses in
  computational efficiency" is the paper's own abstract wording. Correct.
- **Fedus, Zoph & Shazeer 2021 (Switch Transformer)** — parameter count as a
  scaling axis at constant FLOPs-per-example is the paper's central framing.
  Faithful representation, in-domain.
- **Kaplan et al. 2020** — clean power law fit on *non-embedding* parameter count
  N; embeddings excluded because vocab tables do not scale like compute. Correct;
  the application to `phi_embed (V, n_gen)` is an appropriately-hedged analogy.
- **Cohen & Welling 2016 (G-CNNs)** — accuracy gain at fixed parameter count via
  larger symmetry group (p4 -> p4m). Correct.
- **Kondor & Trivedi 2018** — equivariant linear map is exactly a group
  convolution. Correct theorem, correct use (group change = structural change).
- **Cohen, Weiler, Kicanaoglu & Welling 2019** — gauge-equivariant CNN / local
  gauge transformations; the closest published analog to per-block internal
  gauge. In-domain, correct.
- **Williams et al. 2009 (roofline)** — memory-bandwidth vs compute-bound model;
  applied to argue full-vocab decode is bandwidth-bound. In-domain, correct use.
- **Hoffmann et al. 2022 (Chinchilla)** — `L(N,D) = E + A/N^a + B/D^b`, ~20
  tokens/param. Correct functional form and ratio; correctly deployed as the
  half-token D-confound formalism.
- **Hacking 1983, Lakatos 1978, Cartwright(implied)/Popper 1963** —
  cross-checked against `memo_blue_philosophy-of-science.md` (which cites the SEP
  entries). "Has a life of its own," "excess empirical content," "progressive
  problemshift," "irrefutability is a vice" are all genuine, correctly-attributed
  terms. In-domain (philosophy of experiment / research programmes).

## Prose summary

Zero strikes. The Blue opening is exemplary on the source-of-truth precedence
rule. It never cites the user's GL(K)/PIFB manuscript, `CLAUDE.md`, or any
in-repo derivation as the *standard*; the sole manuscript reference treats the
manuscripts as the object of a novelty claim ("this axis appears in neither
manuscript"), which is the correct posture. Every load-bearing authority in the
memo is drawn from external canon — the transformer/scaling literature (Vaswani,
Radford, Shazeer, Fedus-Zoph-Shazeer, Kaplan, Hoffmann), the equivariance
literature (Cohen & Welling, Kondor & Trivedi, Cohen-Weiler-Kicanaoglu-Welling),
systems (Williams roofline), and philosophy of science (Hacking, Lakatos, Popper)
— and each citation is real and faithfully represents its source. In-repo cites
are all path:line references to what the code DOES, which are permitted.

Most consequentially for Blue's signature failure mode: the opening does **not**
commit the "active_params_per_token is flat, therefore efficient" circular move.
It explicitly refuses it (L25), flags it as circular per the mandatory
philosophy-of-science flag, and relocates the efficiency question onto wall-time
and FLOPs — where it honestly concedes a mixed, U-shaped, partially
self-falsifying picture. The metric-legitimacy argument in Pillar 2 defends the
access-pattern reading of the flat working set (grounded in the token-embedding
analogy and the sparse-model canon) without smuggling it back in as efficiency
proof. No banned phrases. No fabricated or wrong-domain citations.

Action: RECORD. Debate continues; judges may note that Blue's canon discipline is
clean and that the circularity guardrail was honored.
