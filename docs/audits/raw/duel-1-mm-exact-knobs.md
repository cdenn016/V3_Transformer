# Duel 1 — `mm_exact` silently ignores the E-step knobs

## SKEPTIC (attack) — returned 10:26 CDT

**Verdict argued: DOWNGRADE to medium.** Concedes the mechanics entirely ("I cannot refute its
mechanics"); attacks the severity framing and the knob enumeration.

### Steelman it accepted
Live config takes the `mm_exact` branch on BOTH channels — belief (`block.py:65` forwards
`e_step_update`) and model (`model.py:924`, live because `train_vfe3.py:338` sets `s_e_step=True`).
The branch at `e_step.py:1001-1058` reads only `mm_damping`/`eps`/`sigma_max`/
`skip_belief_sigma_update`. `config.py:2957` implements only the forward inert rule.

### Attacks that LANDED

1. **Blast radius is zero numerical error.** The mm branch is not missing a step size — it HAS one.
   `e_step.py:1046-1056` performs a damped natural-coordinate blend with `eta = mm_damping`, live at
   0.75 (`train_vfe3.py:411`), covered by the forward inert rule (`config.py:2957-2958`) and by a
   properly guarded sweep (`ablation.py:1566-1570`, `"requires": {"e_step_update": "mm_exact"}`).
   Every loss/PPL a live run produces is a correct evaluation of the configured objective. The pure
   path with all six knobs live is one field away (`e_step_update="gradient"`).
   **This is config/experiment hygiene, not correctness.**

2. **The enumeration is padded by half.** The detector's own contract reports only fields CHANGED
   from the dataclass default (`config.py:2946-2954`, `_changed`). Four of the six named knobs sit
   AT their defaults on the live config and could not be reported even by a perfect reverse rule:
   `e_step_mu_precond` default `"fisher"` (`config.py:474`) == live (`train_vfe3.py:143`);
   `e_mu_q_trust` default `None` (`config.py:465`) == live (`:385`); `mu_trust_mode` default
   `"box"` (`config.py:468`), never set in `train_vfe3.py` at all; `e_s_sigma_lr` default `0.1`
   (`config.py:445`) == live (`:341`). Worse, `mu_trust_mode` is dead under `gradient` too whenever
   `e_mu_q_trust is None` (`e_step.py:1130` gates the whole trust block on it) — so listing it as an
   mm_exact casualty is double-counting.
   **The live, changed, genuinely-dead set is FOUR:** `e_q_mu_lr` (0.9 vs default 0.5),
   `e_q_sigma_lr` (0.001 vs 0.015), `e_sigma_q_trust` (10.0 vs 5.0), `e_s_mu_lr` (0.85 vs 0.1).

3. **"No warning emitted" is not strictly true on the live config.** Executed probe (config
   construction only, K=4) returned **2 warnings**, including `config.py:2801-2811`:
   > `e_step_update='mm_exact' with lambda_alpha_mode='state_dependent_per_coord' computes the
   > frozen-alpha minimizer of the strict-pair-masked surrogate ...; the iteration takes a step
   > toward it using mm_damping=0.75`

   It fires on the live shape because `train_vfe3.py:267` sets
   `lambda_alpha_mode="state_dependent_per_coord"`, and it names `mm_damping` as *the* step the mm
   iteration takes. It does NOT say the LRs are dead and is conditional on state-dependent alpha —
   so this mitigates rather than refutes, but "no warning" is overstated for the config being run.

### Attacks the skeptic TRIED AND FAILED (reported because they are the finding's load-bearing supports)

- *Indirect reach / misdiagnosis*: **none found.** `mm_exact_update`'s call site
  `e_step.py:1021-1030` passes no LR, precond, or trust; the blend `:1046-1058` clamps with
  `eps`/`sigma_max` only, never `e_sigma_q_trust`; the phi substep `:1153-1205` consumes only
  `e_phi_lr`/`phi_precond_mode`/`phi_retract_mode`; the outer loop `:1448-1517` forwards
  `e_q_mu_lr`/`e_q_sigma_lr` into both call sites but nothing reads them off the mm branch;
  `skip_belief_sigma_update` comes straight from cfg (`block.py:67`), not derived from
  `e_q_sigma_lr`; `free_energy_value` declares the whole bag accept-and-ignore (`e_step.py:463-471`).
- *`n_e_steps=1` moots step sizes*: **no.** At depth 1 the gradient branch still scales its single
  step by `e_q_mu_lr` (`e_step.py:1125`) — which is exactly why the two agents saw divergent loss
  under `gradient`.
- *s-channel not live*: **no.** `train_vfe3.py:338` `s_e_step=True`; `model.py:924-925` forwards
  both `e_step_update` and `mm_damping`.
- *Pinned as intended design*: **no.** No test pins LR-independence under mm_exact;
  `tests/test_tier12_estep.py` covers stationarity, monotone descent, oracle rejection, damping
  validation, sigma-skip — nothing about the dead knobs.

### AGGRAVATOR the skeptic disclosed against its own verdict

**The live `ablation.py` enqueues `"e_q_mu_lr"` and `"e_s_mu_lr"` in `SWEEP_ORDER`
(`ablation.py:1703-1704`) against a baseline that sets `e_step_update="mm_exact"`
(`ablation.py:543`), and those sweep definitions (`ablation.py:1461-1464`, `1448-1451`, `1466-1469`,
`1453-1456`, `1312-1315`, `1292-1301`) carry NO `"requires": {"e_step_update": "gradient"}` while
sibling sweeps in the same registry DO (`ablation.py:898, 916, 1092, 1189`).**

So the defect's real teeth are **a null-result generator in the experiment driver** — four identical
PPLs plotted as an LR curve — not a wrong number in training. Medium-grade experiment-hygiene
hazard; cheapest fix is a `requires` entry per sweep plus the reverse `_inert` rule. Loud in the
artifacts (byte-identical cells) rather than silent.

### Skeptic's decisive evidence
`e_step.py:1046-1056` (mm HAS live step control) + `config.py:2946-2954` with `:465,468,474,445`
(four of six knobs at defaults, unreportable under the detector's own contract), against the
aggravator `ablation.py:1703-1704` + `ablation.py:543`.

## DEFENDER — returned 10:44 CDT

**Verdict argued: UPHELD as a silent-configuration / experiment-integrity defect (High), NOT a
numerics defect.**

### Source proof
`e_step.py:1001` opens `if e_step_update == "mm_exact":`; body `:1002-1058` reads exactly five things
— `mm_damping` (:1046), `eps` (:1053), `sigma_max` (:1058), `skip_belief_sigma_update` (:1026,
:1047), `e_step_gradient` (:1035). `:1059` is the `else`. Every contested read is inside it:
`e_step_mu_precond` (:1097, :1121), `e_q_mu_lr` (:1125), `e_mu_q_trust` (:1130), `mu_trust_mode`
(:1136), `e_q_sigma_lr` (:1148), `e_sigma_q_trust` (:1149) — **plus `spd_retract_mode` (:1147), a
SEVENTH knob the finding missed.** `mm_exact_update`'s full signature (`kernels.py:509-538`) has no
step-size, preconditioner or trust parameter, so the callee cannot rescue them.

### Measured (tiny model, K=4, n_heads=2, vocab=8, N=4, 1 layer)
Arm A = live values; arm B = opposite settings of the six, nothing else:

| `e_step_update` | loss A | loss B | identical |
|---|---|---|---|
| `mm_exact` | 2.079459667206 | 2.079459667206 | yes, `torch.equal(logits)=True`, max abs delta = 0 |
| `gradient` | 2.079218149185 | 2.074122428894 | no, max abs delta logits = 3.077e-02 |

Per-knob isolation under `gradient` (each perturbed alone): `e_q_mu_lr` 3.01e-03, `e_q_sigma_lr`
1.49e-04, `e_mu_q_trust=0.05` 2.97e-03, `e_step_mu_precond="raw"` 1.27e-04, `e_s_mu_lr` 3.08e-02,
`e_s_sigma_lr` 5.72e-04, `e_sigma_q_trust` 7.53e-04. **All seven are exactly 0.000e+00 under
`mm_exact`.**

### Detector enumeration
Twenty rules at `config.py:2956-3032`. **Only `:2957` keys on `e_step_update`, forward direction
only.** Rule `:2987` already models "`e_s_*_lr` can be inert" — but only for `s_e_step=False`, which
is not the live condition. No reverse rule exists.

### THE DECISIVE NEW EVIDENCE — recorded experiments are contaminated
`ablation.py:543` pins the baseline to `mm_exact`. `make_run_overrides` (`ablation.py:2239-2249`)
folds `requires` onto it. The defender RAN it: **seven sweeps carry no `e_step_update` in
`requires`, so all their cells run `mm_exact`:**
```
e_q_mu_lr 4 cells   e_q_sigma_lr 3   e_s_mu_lr 6   e_s_sigma_lr 5
e_mu_q_trust 3      e_mu_q_trust_ball 3            e_sigma_q_trust 2      TOTAL 26 cells
```
Instantiating the REAL sweep cells at tiny size (`ablation._cell_cfg_dict` + overrides): every
`e_q_mu_lr` and `e_s_mu_lr` cell is **byte-identical to its sibling under `mm_exact`** and all differ
under `gradient`.

**Worse — `fisher_mu_precond` (`ablation.py:1046-1058`, the B3/EXP-14 mean-arm ablation, 6 cells x 3
seeds = 18 runs) has `requires: {"e_phi_lr": 0.0}` and no route pin.** Matched-`T` measurement:
`fisher_T1` vs `raw_T1`, `fisher_T3` vs `raw_T3`, `fisher_T5` vs `raw_T5` are **all
`identical=True` under `mm_exact`** and differ by 6.06e-02 / 4.60e-02 / 5.92e-02 under `gradient`.
**The experiment that exists to isolate the Fisher-vs-Euclidean mean arm compares an arm against
itself.** The `requires`-pinned-to-`gradient` pattern IS used correctly in eleven other sweeps
(`ablation.py:820, 822, 830, 898, 916, 1092, 1146, 1179, 1189, 1194, 1217, 1277-1282`) — the
mechanism was available and not applied here.

**`scaling.py:422` also runs `mm_exact`, and `scaling.py:534` defines the muP arm as
`{"e_q_mu_lr": base_eqmu * w, "m_p_mu_lr": base_mpmu * w, "mu_init_std": base_init * w**0.5}`. One
of the three muP factors is silently discarded, so the `grow_K_mup` route is not the muP
parameterization it reports.**

`run_artifacts.py:4189-4190` records `e_step_update` and `mm_damping`, and `config.json` records
`e_q_mu_lr=0.9` beside them, with no indication one is fiction.

### Defender's concessions
1. **Three of the six carry no live tuning intent** — `e_mu_q_trust=None`, `mu_trust_mode="box"`,
   `e_step_mu_precond="fisher"`, and `e_s_sigma_lr=0.1` all match defaults. Only `e_q_mu_lr` (0.9 vs
   0.5), `e_q_sigma_lr` (0.001 vs 0.015), `e_sigma_q_trust` (10.0 vs 5.0), `e_s_mu_lr` (0.85 vs 0.1)
   are changed-from-default and dead. **Agrees with the skeptic.**
2. **`mm_exact` is not the sole cause for two of them on the belief channel.**
   `train_vfe3.py:445` sets `skip_belief_sigma_update=True` and `e_step.py:1141-1143` passes sigma
   through untouched, so `e_q_sigma_lr`/`e_sigma_q_trust` are already dead there under `gradient`
   too (measured: max abs delta = 0). They die only via `mm_exact` on the s channel.
3. **The trust sweeps may have been degenerate anyway.** At the real radii (1.0/2.0/5.0 and 10/15)
   the cells are identical under `gradient` as well — the clamps do not bind
   (`ablation.py:1292-1297` says as much). So for 8 of the 26 cells the misattribution charge is
   weaker.
4. **Step-size control is not wholly lost** — `mm_damping=0.75` is a real shared eta. The user has
   one shared eta instead of four per-channel, per-moment rates. Mitigates "no control", not "the
   knobs you set do nothing".
5. `n_e_steps=1` moots nothing — `fisher_T1` vs `raw_T1` differ under `gradient`.

## ADJUDICATION — **UPHELD at HIGH, reclassified as experiment-integrity, not correctness**

Both sides agree the mechanics are real and neither could break them. The skeptic's two substantive
corrections are ACCEPTED and the defender independently conceded both: the enumeration is padded
(the live changed-and-dead set is **four**, not six), and a warning does fire on the live config
(though it names `mm_damping`, not the dead LRs). `mm_damping` is a genuine live step control, so
"the E-step has no step size" would be wrong.

The tie breaks on cited source, not compromise. **The skeptic argued medium but disclosed the
aggravator itself; the defender then quantified it and it is worse than either first stated.** The
damage is not in training — every loss the live run produces is a correct evaluation of the
configured objective — it is in **recorded experiments**: 26 ablation cells whose only varied field
is unread, the 18-run B3/EXP-14 Fisher-vs-Euclidean ablation comparing an arm against itself
(`identical=True` at every matched `T`), and `scaling.py`'s muP route silently dropping one of its
three muP factors. That is high.

The `requires`-pin mechanism is used correctly in eleven sibling sweeps, so this is an omission with
a known, cheap fix, not a design gap.

**Punch-list framing:** four changed-and-dead knobs + a missing reverse `_inert` rule + missing
`requires` pins on seven ablation sweeps + the `scaling.py` muP arm. Not a numerics bug. Any prior
LR/precond ablation result from those cells should be treated as void, and the byte-identical cells
are the signature to check for.
