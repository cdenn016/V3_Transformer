# Simulated Expert Memo - Red Implementation-Engineer

Side: red. Round: opening. Memo path: memo_red_implementation-engineer.md.

## Source-of-truth rule

For code behavior, the V3 source is canonical. For theory labels, the external active-inference literature is canonical. Comments are not treated as proof when executable paths settle the behavior.

## Newly-discovered canon

- Heins et al. (2022), arXiv:2201.03904: practical active-inference agents are built around explicit POMDP components and policy evaluation, not only around a scalar energy score.
- Smith, Friston, and Whyte (2022), DOI 10.1016/j.jmp.2021.102632: A/B/C/D/E style generative-model objects are the implementation-facing template for active inference.
- Friston et al. (2017), DOI 10.1162/NECO_a_00912: policy posterior updates combine free energy accrued so far and expected free energy over future outcomes.

## Memo

The code supports the report's narrow hook but not the stronger active-inference semantics. The belief carrier exists: `BeliefState` has `mu`, `sigma`, and `phi` at `vfe3/belief.py:22-28`. `VFEModel.forward()` encodes tokens into beliefs at `vfe3/model/model.py:660`, applies the E-step stack at `vfe3/model/model.py:725-734`, decodes final beliefs to logits at `vfe3/model/model.py:790-794`, and returns logits when `targets is None` at `vfe3/model/model.py:793-794`. It does not return a public final belief object from the inference path.

The generation hook is real but conventional. `generate()` loops over positions, slices the current context, calls `self.forward(context)`, selects last-position logits, and applies greedy, temperature, top-k, top-p, or multinomial sampling at `vfe3/model/model.py:1198-1221`. That is a viable insertion point for a no-grad scorer, but only as a post-logit decoding layer until a belief-rollout API exists.

The active-inference objects are not wired as production policy machinery. `free_energy()` accepts an optional `log_likelihood` argument at `vfe3/free_energy.py:327-342` and subtracts it only inside `if log_likelihood is not None` at `vfe3/free_energy.py:401-402`; an `rg` scan found no `log_likelihood=` caller under `vfe3`. The model-channel gamma energy uses flat transport by calling `build_belief_transport(..., transport_mode="flat")` at `vfe3/model/model.py:1045-1049`, so community or agent-set policy claims cannot assume non-flat covariant model-channel rollouts.

The strongest implementation attack is therefore not "this cannot be implemented." It can. The attack is that the first implementation would be a scorer bolted to `generate()`, while active inference canon requires explicit transition, likelihood, preference, and policy-prior objects [Smith et al. 2022; Friston et al. 2017]. The current code lacks those as public, reusable entities.

Direct falsifier from this lens: implement the scorer with a fixed interface that logs its A/B/C/D/E analogues. If any logged component is missing, constant, or only a copy of logits, the implementation is a reranker rather than an active-inference policy scorer.
