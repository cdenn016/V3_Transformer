# Duel 2 — ALiBi log-prior floored at -27.6 nats by the gamma fold

## DEFENDER — returned 11:06 CDT

**Verdict argued: DOWNGRADE — from HIGH to MEDIUM.** The defender downgraded its OWN finding after
measuring the caveat it was asked to close. This is the honest outcome the brief invited.

### Reachability CONFIRMED — runs on every forward under the live config
`train_vfe3.py:431 gamma_as_beta_prior=True`, `:433 gamma_prior_weight=0.5`,
`:307/:308 beta/gamma_attention_prior="causal_alibi_noself"`, `:310 alibi_slope=1`,
`:291 lambda_gamma=0.75`, `:82 embed_dim=210`, `:83 n_heads=7`, `:85 max_seq_len=128`,
`:299 kappa_beta=1` -> `tau = sqrt(210/7) = 5.4772`, `:314 precision_weighted_attention=False`.
Path: `model.py:1105 -> :2445` (precision fold early-returns at `:2329-2330`) `-> :2446` True
`-> :2449 _fold_gamma_prior -> :2396-2397`, the floor. Replayed by every diagnostic consumer
(`model.py:2149, 2731, 3244, 3370`; `viz/extract.py:204, 522, 659`).

Reproduction matches the investigator to four decimals:
```
slopes: [0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125, 0.03125]
causal entries: 56903   min bias: -63.5
floored entries: 2954 / 56903  (5.19%)   max nats added: 36.3017
  head 4 slope=0.500000: 2701/8129 = 33.23%  max=36.302
  head 0 slope=0.250000:  253/8129 =  3.11%  max= 5.378
  heads 1,2,3,5,6: 0/8129
```
`56903 = 7 * (127*128/2 + 1)` is exactly the `causal_alibi_noself` support.

### The STRUCTURAL claim is upheld — a hard measurement
Head 4, query row 127, walking outward from distance 1 to 127:
```
exact  : strictly-decreasing steps 126, FLAT (tied) 0,  increasing 0   (127 distinct values)
floored: strictly-decreasing steps  54, FLAT (tied) 72, increasing 0   ( 55 distinct values)
floored plateau value = -27.6310 nats;  exact spans -63.933 .. -0.933
```
**72 of 126 steps are ties at exactly `log(1e-12)`. Monotonicity DOES break**, and the effective
prior is not the prior the config names. ALiBi's only content is monotone distance decay.

### The DOWNSTREAM claim does NOT survive measurement at the operating point
Both folds pushed through `attention_weights` (`free_energy.py:329-332`) at `tau = 5.4772`:
```
energy regime                       max|dbeta|   max rowKL   max TVmass
E = 0 (uniform energy)               1.000e-12   0.000e+00    3.556e-11
E ~ |N(0,1)|*5   (iid)               2.897e-12   0.000e+00    4.287e-11
E ~ |N(0,1)|*20  (iid)               1.625e-11   0.000e+00    1.240e-10
E ~ |N(0,1)|*150 (iid)               5.960e-08   1.163e-07    6.296e-08
```
`max row TV(pi_floored, pi_exact) = 3.556e-11`, mean `1.532e-12`, `max row KL = 0.0` — **below fp32
eps `1.192e-07`.** The reason is in the code ORDER: **`model.py:2396` normalizes `pi` BEFORE
`model.py:2397` clamps the log**, so the floor corrupts only the LOGARITHM of a region carrying
`1.88e-12` of the row's exact mass.

### It IS a cliff — but the cliff is far from the operating point
A floored entry matters only when `(min_k E_ik) - E_ij` exceeds roughly
`tau * (27.631 - 0.933) ~ 146` nats. Measured with a copy/induction pattern (query 127, deep key
`j=0` at energy 0, nearer keys at energy `X`):
```
X (nats)   beta_exact[j=0]   beta_floored[j=0]   TV mass
     50      1.581e-24          9.216e-09        6.5e-08
    100      1.457e-20          8.493e-05        8.5e-05
    146      6.469e-17          2.738e-01        2.7e-01
    200      1.238e-12          9.999e-01        1.0e+00
```
Catastrophic when it fires — 27% then 99.99% of a head's mass on a key the exact prior forbids — but
it needs a ~146-nat gap favoring a distant key. Against that: `train_vfe3.py:137 mu_init_std=0.065`
with `sigma_init=1.0` (`prior_bank.py:348, 457`) and `K=210` gives at-init energy
`E = 0.5|dmu|^2/sigma^2` of **mean 0.89, max 1.16 nats** — two orders short. And the project's own
recorded measurement agrees: `docs/2026-07-25-state-of-knowledge.md:22, :128-131` — flattening
`log_prior` costs 0.612 nats, ablating the coupling energy costs 0.210, "bounded above by ~0.21 nats
as currently constituted." **A router that weak does not produce 146-nat gaps.**

### The defender CONCEDED the precedent argument — the parallel was overstated
> "The two are the same *idiom* and the same number — `free_energy.py:456-457` names it — but not
> the same *defect*. There the floor lived inside `reduced_free_energy` only, while `beta` was built
> from the raw `log_prior` at `free_energy.py:434`, so `F` and `-tau logZ` disagreed. **Here the
> floor is applied at construction**, so `attention_weights` (`:331`), `log_partition` (`:353`), and
> the entropy term (`:460`) all receive the identical floored tensor."

Measured envelope identity `|sum_j b E + tau sum_j b log(b/pi) + tau logZ|`: **floored 2.861e-06,
exact 3.099e-06** — indistinguishable fp32 noise. **No F/oracle inconsistency exists here.** The
investigator's "already removed this exact failure mode" overstates the parallel.

No discrete amplifier exists either: `pair_mask` (`kernels.py:126-131, 162`) gates on KL-clamp
saturation, not on the prior, and `guard_energy_klmax_frac` is 0.0 in recorded runs. No top-k or
threshold sparsifier.

### Falsification conditions — ALREADY MET, and they falsify HIGH
Met: max `|dbeta| < 1e-6` at the operating energy scale — measured `1.0e-12` flat, `1.6e-11` at
`E ~ |N(0,1)|*20`; max row TV below fp32 eps. **"Decisive" is not supported.**
Would restore HIGH: a logged `max_j E_ij - min_j E_ij` from a live trained run exceeding ~146 nats
on any head-4 row (`kl_max = 1680` permits 11x that), or any run with
`guard_energy_klmax_frac > 0`.
Would take it to DROP: a live diagnostic showing coupling-energy spread stays under ~50 nats.

### OPEN OBLIGATION the defender could not close
**Nobody logs the coupling-energy spread.** The nearest recorded runs (`K=300, H=10, d_head=30`,
same `tau`) log only `val_attn_entropy` 2.07-2.27 nats mean — diffuse and consistent with a
near-field positional router, but it does not measure the gap.

## SKEPTIC (attack) — returned 11:20 CDT

**Verdict argued: DOWNGRADE to LOW** (concedes MEDIUM as the ceiling, on config fragility alone).

### THE INVERSION — the floor cannot touch any head that carries long-range attention
`_press_slopes` (`attention_prior.py:24-48`) at `n_heads=7`, `alibi_slope=1` gives
`[0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125, 0.03125]`. The 2,954 floored entries split as
head4 (slope 0.5) 2,701, head0 (slope 0.25) 253, and **heads 1, 2, 3, 5, 6: ZERO** — the five
shallow-slope heads have no floored entry at any distance up to N=128.

**Inverting the live run's own saved attention maps**
(`vfe3_runs/20260726-225843_.../attention/step_100000_layer0_head*.png`, log-scale magma clipped at
1e-4): **the model's long-range attention lives entirely on the untouched heads.** head6 reaches
`d=122` with beta up to 1.00 at `d>=55`; heads 1/2/3/5 reach `d=113..121` with beta up to 0.96. The
two heads where the floor binds stop far short of it — **head4's deepest entry above the clip is
`d=28` (forward) / `d=36` (E-step) against a floor onset at `d=55`**; head0's is `d=55/47` against
an onset at `d=106`.

### The "306 nats" premise is an unattained bound
`metrics.py:896` defines `energy_klmax_frac`. The live run reports
**`guard_energy_klmax_frac = 0.0` and `val_guard_energy_klmax_frac = 0.0`** at step 115000.
`kl_max/tau = 306` is never realized.

Measured from the run's own maps via `adv(d) = log beta(d) - log beta(1) + slope*(d-1) =
(E_1 - E_d)/tau`: max **5.10 nats** (head4 forward), **8.36** (head4 E-step), **5.68/5.97** (head0).
The handicap a floored entry must clear to reach parity with the previous token is **26.70 nats**
(head4) and **26.12** (head0).

Granting the advantage to EVERY floored entry:
| advantage | max\|dbeta\| | max spurious row mass |
|---|---|---|
| 0 (prior only) | 1.0e-12 | 7.3e-11 (row-KL exactly 0) |
| 5.10 | 2.98e-08 | 1.20e-08 |
| **8.36 (measured max)** | **1.49e-07** | **3.12e-07** |
| 12.0 | 4.53e-06 | 1.19e-05 |
| 20.0 | 1.31e-02 | 3.42e-02 |
| 26.70 | 2.13e-01 | 9.66e-01 |

**At the measured energy contrast the floor perturbs beta at 1.5e-7 — fp32 eps is 1.19e-7.** The
proposed fix moves numerical noise at the config the finding was raised against.

### The precedent analogy is a category error (independently reaching the defender's conclusion)
`free_energy.py:456-460` removed an INTERNAL INCONSISTENCY: beta used the raw bias (`:329-331`)
while F's entropy term used a separately floored `log(softmax(...).clamp())`, so F deviated from
`-tau logZ`. Here there is no split — `_fold_gamma_prior` returns ONE tensor, `model.py:1105`
installs it as *the* `log_prior`, and both consumers read it. Measured
`max |log_softmax(out) - out|` on the support is **9.5e-7**, i.e. the returned prior is already
row-normalized. The floor changes which prior the model HAS, not whether F is consistent with beta.

### Attack lines the skeptic tried and reported FAILED
The sum-clamp does not bind (row sums 0.9999998..1.0000002, so `pi.clamp` is the binding clamp
exactly as claimed); the path is fully live; the downstream softmax does not undo it (the distortion
is entry-dependent, 0 to 36.3 nats, not row-constant); the counts are exact (56,903 support, 2,954
floored, 36.301727 max — reproduced to the digit).

### HONEST CAVEAT the skeptic raised against its own case
> "the margin is config-dependent. `vfe3_runs/140.48_alibi-slope=2/` shows `alibi_slope=2` **has been
> run**; there the steepest head slope is 1.0 and the floor onset moves in to `d~29`, **adjacent to
> head4's measured reach of `d=28-36`**. That is an argument for making the fix and pinning it — not
> for calling it high at the live `alibi_slope=1`."

## ADJUDICATION — **UPHELD, DOWNGRADED high -> MEDIUM**

Both sides independently converged that HIGH is unsupported, and the **defender downgraded its own
finding** before seeing the skeptic — the strongest possible signal. Neither could be accused of
motivated reasoning: the skeptic argued low but disclosed the slope-2 fragility that raises it, and
the defender argued medium but retracted its own precedent argument.

**The structural claim stands and is a hard measurement**: 72 of 126 steps on head 4 become ties at
exactly `log(1e-12)`, so ALiBi's monotone distance decay — its only content — genuinely breaks on a
third of that head's causal entries. The effective prior is not the prior the config names.

**The severity claim is refuted at the live config.** Two independent routes agree: normalize-then-
clamp-the-log means the floor corrupts only the LOGARITHM of a region carrying `1.88e-12` of the row
mass, and the trained model's long-range attention lives entirely on the five heads with zero
floored entries. Measured `max|dbeta| = 1.0e-12` at prior-only and `1.5e-7` at the measured energy
contrast — at or below fp32 eps. `guard_energy_klmax_frac = 0.0` confirms the 306-nat range is never
realized; the measured advantage is 5-8 nats against a 26.7-nat handicap.

**Both sides also retracted the `free_energy.py` precedent independently**, by the same argument:
that fix cured an `F != -tau logZ` split; this floor is applied at construction so every consumer
sees the same tensor. Measured envelope identity is fp32 noise either way. My earlier framing of
this as "the same failure mode the project already removed" was wrong.

Held at MEDIUM rather than LOW on one cited fact: **`alibi_slope=2` has actually been run**
(`vfe3_runs/140.48_alibi-slope=2/`), and at that slope the floor onset moves to `d~29`, adjacent to
head4's measured reach of `d=28-36`. The cliff is genuine — at a 26.7-nat gap it puts 97% of a row's
mass on keys the exact prior forbids — and the config that approaches it is already in the run
history. That is a latent correctness cliff with a free one-line fix, not a live numerical error.

**Punch-list framing:** fix and pin it, motivated by the slope-2 configuration, not by the live one.
The open obligation both sides named: **nobody logs the coupling-energy spread**, so the margin is
inferred rather than monitored.
