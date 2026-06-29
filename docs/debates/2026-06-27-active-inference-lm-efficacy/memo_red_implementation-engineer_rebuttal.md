# Red Rebuttal Memo - implementation-engineer

## Newly-discovered canon

- Heins et al. (2022), "pymdp: A Python library for active inference in discrete state spaces," arXiv:2201.03904, https://arxiv.org/abs/2201.03904. Use: active-inference implementations expose POMDP components rather than only a scalar post-hoc score.
- Smith, Friston, and Whyte (2022), Journal of Mathematical Psychology, DOI 10.1016/j.jmp.2021.102632. Use: the executable active-inference template has named `A`, `B`, `C`, `D`, and `E` objects and a policy posterior over expected free energy.

## Expert memo

Blue's implementation concession is sound: V3 has a plausible hook for a no-grad inference-time scorer. `BeliefState` exists with `mu`, `sigma`, `phi`, plus optional `s` and `r` at `vfe3/belief.py:22-30`. `forward()` encodes tokens to beliefs at `vfe3/model/model.py:660`, applies positional gauge frames at `vfe3/model/model.py:661`, and runs the E-step stack at `vfe3/model/model.py:725-734`. `generate()` is explicitly no-grad at `vfe3/model/model.py:1164-1165`.

The same code defeats the stronger blue move from "hook" to "legitimate active inference." The inference branch does not return a belief object. It decodes `mu_final` and `sigma_final` to logits at `vfe3/model/model.py:791-792` and returns logits directly at `vfe3/model/model.py:793-794`. Any EFE layer that needs candidate-conditioned future beliefs has to add an API that the current public path does not provide.

The observation likelihood is also not live in the scalar free-energy function. `vfe3/free_energy.py:341` defines `log_likelihood` as an optional argument, and `vfe3/free_energy.py:401-402` subtracts it only if it is non-None. The source search found production uses of `reduced_free_energy(...)` in `vfe3/inference/e_step.py:345`, `vfe3/model/model.py:1081`, and `vfe3/model/model.py:1118`, but no production caller passing `log_likelihood=`.

The agent-set variant has a separate limitation. Model-channel coupling constructs `omega` with `transport_mode="flat"` at `vfe3/model/model.py:1047` before transporting `s_mu` and `s_sigma` at `vfe3/model/model.py:1048-1049`. A policy over agent sets can still be tested as a flat semantic route selector, but the code does not support a full non-flat, outcome-likelihood active-inference community policy without new design work.

My red verdict is code-level: V3 has components that could support an experiment, but the implementation currently supplies a logits-returning generator and an inert observation-likelihood hook, not an active-inference POMDP or rollout model.
