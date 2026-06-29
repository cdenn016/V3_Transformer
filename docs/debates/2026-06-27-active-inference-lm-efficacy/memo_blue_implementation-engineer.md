# Simulated Blue Expert Memo - implementation-engineer

## Position

The code supports a narrow no-grad inference-time experiment after one missing abstraction is added: a way to expose final beliefs or run belief rollouts for explicit candidates. The code does not support a train-time EFE replacement today.

## Code-grounded analysis

V3 already has the state object an EFE scorer would need to start from. `vfe3/belief.py:22-30` defines `BeliefState` with `mu`, `sigma`, `phi`, plus optional `s` and `r` channels. `VFEModel.forward()` encodes token ids into beliefs at `vfe3/model/model.py:660`, applies positional gauge frames at `vfe3/model/model.py:661`, runs the E-step stack at `vfe3/model/model.py:725-734`, obtains `mu_final` and `sigma_final` at `vfe3/model/model.py:739-740`, and decodes them to logits at `vfe3/model/model.py:791-794` when `targets is None`. That is close to what a rollout scorer needs, but the public return is logits, not the belief tuple.

The natural hook is generation. `generate()` loops over tokens, calls `self.forward(context)` at `vfe3/model/model.py:1201`, reads last-position logits at `vfe3/model/model.py:1202`, and then chooses by greedy, top-k, top-p, temperature, or multinomial sampling at `vfe3/model/model.py:1203-1220`. An opt-in scorer can be placed between candidate construction and `next_token` selection without touching training. This also honors the codebase's registry-and-toggle discipline because the scorer can be a generation option with a default-off path.

The blockers are real. `vfe3/free_energy.py:327-402` assembles a scalar free energy, but the observation likelihood argument is optional and subtracted only if supplied; the evidence pack reports no production caller. `vfe3/model/model.py:1021-1062` builds model-channel gamma energy under tied flat transport, so a full community or agent-set EFE claim must either accept flat model-channel semantics or defer non-flat claims. `vfe3/inference/e_step.py:423-435` builds covariance-sensitive transport using live `mu` leaves but binds sigma with a detached key slot, which is a poor base for train-through-policy claims about covariance-sensitive non-flat rollouts.

External active-inference canon agrees with this architecture-first caution. Smith et al.'s POMDP template requires explicit likelihood, transition, preference, initial-state, and policy-prior components [Smith et al. 2022]. Friston et al. separate inference, policy selection, and learning [Friston et al. 2016]. Vaswani et al. and Radford et al. anchor the existing LM path in ordinary transformer decoding and autoregressive training [Vaswani et al. 2017; Radford et al. 2019].

## Newly-discovered canon

- [Smith et al. 2022] An EFE implementation needs named generative-model components, not only a scalar score.
- [Friston et al. 2016] Fast policy selection can be studied without replacing slower parameter learning.
- [Radford et al. 2019] Autoregressive generation supplies the operational baseline for candidate scoring.

## Recommendation

Add no code in this debate. If implemented later, start with a default-off generation scorer that consumes explicit candidate continuations, calls a belief-rollout helper, and returns diagnostics showing risk, ambiguity, and raw log-probability separately.
