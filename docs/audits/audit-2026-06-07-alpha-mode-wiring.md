# Audit — `alpha_mode` self-coupling wiring (2026-06-07)

Question (user): "deeply audit the alpha_mode paths. my experiments suggest they might
not be wired up correctly."

Provenance: this report recovers and completes a multi-agent audit launched as a
background Workflow (`alpha-mode-audit`, run `wf_37fc5e7a-caf`) that crashed with its
parent session before returning. The four probe agents had finished and their structured
findings were recovered from the run journal; the three adversarial verifiers had been
dispatched but produced no verdict before the crash. They were re-dispatched fresh
(`audit-skeptic`) in this session against the recovered high findings. Every claim below
traces to a `path:line` read or a printed number from an agent that RAN code on CPU
(torch 2.11.0+cpu) at an operating point faithful to `train_vfe3.py`.

## Bottom line

`alpha_mode` is wired correctly and is numerically correct. The hand kernel and the
autograd-of-`F` oracle agree to float32 epsilon (`~2e-7`) for all four forms — `constant`,
`state_dependent`, `state_dependent_per_coord`, `learnable` — so nothing is disconnected
or mathematically wrong. The user's symptom ("experiments suggest it does nothing") is an
operating-point property of the active configuration, not a wiring defect, and at the
configuration currently in the working tree it is a literal no-op for a precise and
benign reason.

The self-coupling `alpha * KL(q_i || p_i)` reaches the trained model through two distinct
channels, and at the active configuration both are quiet:

The E-step channel is inert at `n_e_steps = 1` by construction. The belief is initialized
at the prior, `q(0) = p` (`prior_bank` encode returns the per-token prior table as the
initial belief, and with `n_layers = 1` and `prior_handoff_rho = 0` the same tensor is
passed as `mu_p`), so on the single E-step iteration the self-coupling gradient is
`grad KL(q || p)` evaluated at `q = p`, which is exactly zero. Both factors of the self
leg vanish independently: `mu_q - mu_p = 0` and the variance term `1/sigma_p - 1/sigma_q = 0`
(`gradients/kernels.py:132,136`). The saturation mask `1[0 < D < kl_max]` is not even
load-bearing here — the leg is zeroed by the gradient identity itself, not by the clamp.
A verifier confirmed this empirically (`_raw_diag_kl(enc, enc) = 0`, self-mask nonzero
fraction `= 0`, and `d|mu| = dCE = 0` across all modes and `b0 in {1e-2, 1, 1e2}` at
`n_e_steps = 1`), and confirmed the wiring is genuinely live with a control at
`n_e_steps = 4` (`state_dependent_per_coord`, `b0 = 0.01`: `d|mu| = 9.02`, `dCE = +1.68`).
This is correct mathematics, not a defect.

The M-step channel is the only path that carries `alpha` at `n_e_steps = 1`, because it
recomputes the self-divergence on the converged belief `out.mu` (which the unmasked
pair-coupling moves off the prior on iteration 1) versus the prior
(`model.py:447-460`), with `alpha` detached (`model.py:456`) and `D(q*||p)` gradient-connected.
It is gated by `mstep_self_coupling_weight`, and the current working tree sets that weight
to `0.00` (`train_vfe3.py:120`; the committed value was `0.05`).

The two facts compose into the decisive answer: at the working-tree configuration
(`n_e_steps = 1`, `mstep_self_coupling_weight = 0.00`) the E-step self leg is zero by the
`q = p` identity and the only channel that could otherwise carry `alpha` is switched off,
so `alpha_mode`, `b0`, and `c0` are no-ops for every alpha form. The ablation sweeps
correctly read "no effect" because there is, at this operating point, no effect to read.
Restoring `mstep_self_coupling_weight = 0.05` reintroduces the channel, but at the active
`b0 = 1.0` its effect is negligible: because `D(q*||p) << b0` keeps
`alpha* = c0/(b0 + D) ~ 1.0`, the across-mode difference in the M-step term is about
`1.85e-5` and the per-forward CE does not move at all. The channel becomes consequential
only when `b0` is lowered toward the `D` scale, quantified in "Making `alpha` bite" below.

## Verified findings

The probe severities were adjudicated by the adversarial verifiers; the right-hand
columns are the verifiers' verdicts, not the probes' self-assessments.

| Finding | Probe sev. | Verdict | Adjudicated sev. |
|---|---|---|---|
| `alpha` math is numerically correct: hand kernel == autograd-`F` oracle to `~2e-7` for all four modes (`kernels.py`, `oracle.py`, `alpha_i.py`) | info | confirmed | info |
| E-step self leg is zero at `n_e_steps=1` because `q(0)=p` (`grad KL(q‖p)|_{q=p}=0`); mask not load-bearing (`kernels.py:132,136`; `prior_bank` encode; `e_step.py:345-358`) | high | partial — mechanism confirmed, "cannot affect CE/PPL" headline refuted | info / low (correct-by-construction) |
| M-step self-coupling (`model.py:447-460`) is the only `alpha` channel at `n_e_steps=1`; grad-connected but `D(q*‖p) << b0` so `alpha* ~ 1.0` (small effect); OFF at the working tree (`weight=0.00`) | medium / info | confirmed | low (operating-point + toggle) |
| b0/c0 ablation sweeps force `alpha_mode='state_dependent'` (per-position, summed over `K`) while the active config trains per-coord (`ablation.py:371/375` vs `:126`; routing `free_energy.py:192`) | medium | refuted as the symptom's cause — at faithful LR both paths sit at `D~=0` and are equally `b0`-sensitive | low (config-data note) |
| `learnable` alpha is silently frozen at `n_e_steps=1` (`log_alpha.grad=0`, no warning); guard covers only `detach_e_step`/`straight_through` (`model.py:125-139`; `config.py:625-645`) | medium | confirmed (real, reachable, correctly scoped) | low (silent no-op back to the pure `alpha=1.0` default) |
| Stale memory: the "RuntimeWarning about silent per-coord -> per-position degradation" is no longer in the code (`alpha_i.py` has no such warn; routing is a hard dispatch) | info | confirmed | info |

Two probe corrections matter enough to state plainly. First, the E-step finding's headline
("`alpha` cannot affect the CE/PPL") is too strong: with the M-step term on it does affect
the training gradients to the learned priors and therefore CE/PPL across training, even
though it never moves a single frozen forward's CE. Second, the per-coord-versus-per-position
sweep mismatch is a real fact about the ablation registry but is not the explanation of the
symptom: at the faithful learning rate (`e_mu_lr = 0.7`) the per-position summed `D` has mean
`0.0017` and max `0.0043`, three to four orders of magnitude below the `b0 = 10` saturation
the finding's mechanism required, and `renyi_per_coord` summed over `k` exactly recovers the
per-position `D` (`gaussian.py:120`, agreement `1.5e-6`), so both routings sit at `D ~= 0`
where both are maximally and equally `b0`-sensitive. The asymmetry the probe claimed does not
exist at the trained operating point.

## The one genuine code gap (proposed fix, not applied)

Selecting `alpha_mode = 'learnable'` together with `n_e_steps = 1` freezes `log_alpha` at its
initialization with no diagnostic. The parameter reaches the loss only through the E-step
(the M-step term detaches it), and at `n_e_steps = 1` the E-step gradient sits at `q = p`
where `dD/dq = 0`, so `log_alpha.grad = 0.0` and the optimizer leaves it at `exp(0) = 1.0`,
byte-identical to the pure `constant` `alpha = 1.0` default. A verifier reproduced this
(`log_alpha.grad = 0.0`, optimizer `|delta| = 0` at `n_e_steps = 1`; `grad = -1.7e-5`,
`|delta| = 2.5e-2` at `n_e_steps = 2`), and confirmed the scoping is correct: `log_lambda_beta`
is not frozen (`grad = 3.4e-5`) because it scales the coupling term, which is live at `q = p`;
only `alpha` scales the self term that vanishes there. The project already warns this exact
freeze (alpha stuck at 1.0) on two sibling routes — `detach_e_step` (`model.py` constructor)
and `e_step_gradient in {straight_through, detach}` (`config.py` post-init) — so the missing
warning on the `learnable` + `n_e_steps == 1` route is a genuine consistency gap, of low
severity (it silently no-ops back to the pure path rather than corrupting anything, and is
reachable only when the user opts into `learnable`).

Proposed fix (deferred to user confirmation, since the task was to investigate): add a
non-breaking `warnings.warn` in `VFE3Config.__post_init__`, beside the existing
`e_step_gradient` freeze warning (`config.py` ~`:631`), firing when
`alpha_mode == 'learnable' and n_e_steps == 1`. That is the right condition (alpha-only) and
the right location (`n_e_steps` is pure config, and the sibling freeze warning already lives
there). No code was changed by this audit.

## Making `alpha` bite (if that is the intent)

To give the self-coupling a live effect at the active operating point, raise `n_e_steps` to
`2` or more so the E-step evaluates the self term at a moved belief (`D > 0`, unmasked) on
iteration two onward; and/or restore `mstep_self_coupling_weight` to `0.05` or higher and
lower `b0` toward the actual `D` scale (order `0.01`) so that `c0/(b0 + D)` varies meaningfully
rather than collapsing onto `1.0`. The size of this lever is concrete: at `b0 = 0.01`, where
`alpha* = 1/(0.01 + D) ~ 90`, the gradient norm on `prior_bank.mu_embed` moves from `1.92e-2`
(`constant`) to `8.43e-2` (`state_dependent_per_coord`), about 4.4x, at `n_e_steps = 1`; the
same M-step `dloss` sweep falls from `+8e-3` at `b0 = 0.01` to `~1e-4` at `b0 = 10`, which is
why the active `b0 = 1.0` sits in the inert tail. For the b0/c0 ablation to be discriminative it must run in
a `D` regime below `kl_max` with `n_e_steps >= 2`; a per-coord sweep arm
(`requires = {'alpha_mode': 'state_dependent_per_coord'}`) is expressible and probes the path
the active config actually trains, though at the faithful LR it sits at the same `D ~= 0`
operating point as the per-position arm.

## Caveats

The audit reads and runs code at small dimensions; it does not run a full training to
convergence on a real corpus, so the M-step channel's effect at a non-trivial trained `D`
(where `D` could approach `b0`) is bounded by argument, not measured. The decisive
no-op conclusion is specific to the working-tree toggles at the time of writing
(`n_e_steps = 1`, `mstep_self_coupling_weight = 0.00`); the user changes toggles freely, and
flipping either one re-activates a channel. Two of the original verifiers noted the advisor
channel was rate-limited during the crashed run; the re-dispatched verifiers in this session
reached their verdicts on primary-source code and printed numbers, quoted above.
