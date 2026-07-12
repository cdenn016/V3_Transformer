# Expert Architecture README Design

## Purpose

Replace the current `README.md` with a professional GitHub landing page for readers who already
understand machine learning, language models, variational inference, information geometry, or
geometric deep learning. The README must explain the repository's executable architecture without
collapsing four different scopes into one: the reusable inference engine, the preserved pure
configuration, the checked-in click-to-run experiment, and the broader manuscript program.

The document is an architecture guide, not a manuscript, benchmark card, configuration reference,
or historical changelog. It should let an expert answer three questions quickly: what computation
the repository performs, which parts are mathematically exact versus approximate or interpretive,
and where each subsystem lives in the code.

## Evidence base

Every implementation claim will be grounded in `origin/main` at commit `01204cfb7374`. The primary
runtime sources are `train_vfe3.py`, `vfe3/model/model.py`, `vfe3/model/stack.py`,
`vfe3/model/block.py`, `vfe3/inference/e_step.py`, `vfe3/free_energy.py`,
`vfe3/model/prior_bank.py`, `vfe3/geometry/`, `vfe3/train.py`, and
`vfe3/run_artifacts.py`.

The theory wording will follow the current Research-vault synthesis pages `VFE Transformer
Program`, `GL(K) gauge-equivariant attention`, `GL(K) gauge group`, and `Variational free energy`,
with the July 9 manuscript review/revision record taking precedence over older summaries. Public
links will point only to repository-contained files under `Manuscripts-Theory/`; the private vault
will not appear as a GitHub path or be described as a public dependency.

## Chosen presentation

The README will use an architecture-first expert narrative. A theory-first version would obscure
how the checked-in code executes, while a developer-only reference would understate the
mathematical construction. The chosen structure introduces the executable graph first, places the
minimum derivation beside the mechanism it justifies, and follows with configuration, operations,
and theory scope.

The document will use one Mermaid diagram, compact comparison tables, short prose sections, and at
most two displayed equations. It will not use unverified badges, a generated hero image, benchmark
marketing, a manually maintained pass count, or a complete catalog of every configuration field.

## Information architecture

The final README will use this order:

1. **Title and scope.** A two-paragraph summary will identify V3_Transformer as an experimental,
   registry-driven sequence model that performs finite target-blind refinement over Gaussian token
   states and trains a next-token readout through the unrolled computation. It will state that the
   preserved pure configuration has no learned Q/K/V projections, MLP, or pointwise activation,
   while optional learned extensions and the current click-run profile must be considered
   separately.
2. **Architecture at a glance.** A Mermaid flowchart will show token and positional inputs, optional
   model-channel refinement, the belief-refinement stack, selectable post-refinement transforms,
   selectable decode boundaries, outer cross-entropy, optimizer updates, and run artifacts.
3. **What attention means here.** The entropy-regularized row objective and its Gibbs solution will
   define the exact sense in which attention is derived from fixed pair energies and priors.
4. **Execution profiles.** A comparison table will distinguish the architecture engine,
   `VFE3Config()` defaults, the checked-in `train_vfe3.py` profile, the preserved pure profile, and
   opt-in experiments.
5. **End-to-end data flow.** Short subsections will cover state construction, model-channel
   refinement, belief refinement, stack handoff, decode, and outer training.
6. **Geometry and mathematical scope.** This section will state the exact full-Gaussian gauge
   result, the diagonal-family projection boundary, flat Regime-I holonomy, and the status of
   nonflat transports and frame metrics.
7. **Modularity.** A source-linked table will map belief families, divergences, self-coupling,
   groups, transports, position, inference updates, decode, policy, metrics, and visualization to
   their owning modules. It will say that the code is registry-heavy rather than claiming that
   every dispatch is a registry.
8. **Status ledger.** Implemented core, preserved pure profile, opt-in experiments, partial
   implementations, deliberate stubs, interpretive readings, and future manuscript work will be
   separated explicitly.
9. **Run the repository.** Installation, cached-corpus requirements, the click-to-run workflow,
   main entrypoints, artifacts, figure regeneration, and test commands will be concise and
   executable.
10. **Repository map and theory.** A compact tree will route readers to source, tests, documents,
    and working manuscript copies. The GL(K) manuscript and supplement will be the primary theory
    links; PIFB will be described as a broader companion framework rather than as fully implemented
    architecture.

## Architecture diagram

The Mermaid graph will represent the generic executable system, not only one experiment. Its main
path will be:

`token_ids -> PriorBank.encode -> position/frame construction -> optional refine_s -> attention
prior construction -> L blocks x T refinement iterations -> optional mixer/coupling/norm -> decode
selection -> next-token cross-entropy -> AdamW parameter update`.

Within a refinement iteration, a nested subgraph will show `transport -> pair energy -> Gibbs
weights -> gradient or damped-MM belief update`. Dashed or labeled edges will identify optional
branches. The diagram will not imply that model-channel refinement, covariance retraction, frame
retraction, KL decode, or a nonflat connection is active in every configuration.

The artifact path will branch from training into `config.json`, `metrics.csv`, checkpoints,
provenance, the pure-path report, summaries, and figures. This makes the persistence/reporting layer
visible without mixing it into the mathematical model.

## Mathematical claim policy

The attention derivation will use

$$
\mathcal F_i(\beta_i) = \sum_j \beta_{ij} E_{ij}
+ \tau \sum_j \beta_{ij} \log\frac{\beta_{ij}}{\pi_{ij}},
\qquad
\beta_{ij}^{*} = \frac{\pi_{ij}\exp(-E_{ij}/\tau)}
{\sum_k \pi_{ik}\exp(-E_{ik}/\tau)}.
$$

The prose will say that this is the unique row-wise minimizer on the active support for fixed
energies, transports, and prior. It will not claim that every registry divergence creates an ELBO,
that the full training loop optimizes one scalar free energy, or that finite filtering steps
converge. The inner refinements are target-blind; the next-token target enters the separate outer
cross-entropy.

Gauge claims will distinguish three levels. Common invertible pushforward leaves full-Gaussian KL
invariant. Local frame changes preserve pair scores when transport transforms by the induced
conjugation law. The diagonal covariance family is not closed under a general GL(K) congruence, so
the diagonal realization is projected or approximate outside monomial transports. The flat
vertex cocycle has identity loop holonomy; nonzero holonomy belongs only to opt-in edge-relaxed
paths.

Interpretive language such as tokens as agents, consensus, predictive coding, or learning as
symmetry breaking will be labeled as interpretation rather than executable machinery.

## Execution-profile policy

The README will not call any single configuration simply "the default." It will use explicit names:

- **Preserved pure profile:** token prior, single belief channel, flat phi cocycle, canonical
  attention entropy, constant self-coupling, belief-covariance updates enabled, no mixer, no detached
  precision prior, and KL-to-prior decode.
- **Dataclass defaults:** the values produced by `VFE3Config()`; these include learned positional
  phi and linear decode and therefore are not identical to the pure profile.
- **Checked-in click-run profile:** the current `train_vfe3.py` experiment. At the source commit it
  uses Wikitext-103, K=20, two heads, one layer, one refinement step, a diagonal Gaussian family,
  order-one Renyi/KL pair energies, block-GL flat phi transport, learned BCH position, active
  model-channel refinement, state-dependent self-coupling, detached precision and gamma prior
  folds, damped `mm_exact` updates, a skipped belief-covariance update, zero in-E-step frame rate, a
  head mixer, biased linear decode, and outer cross-entropy/AdamW training.
- **Opt-in experiments:** full covariance, Laplace beliefs, alternate groups, omega-direct frames,
  reflection sampling, nonflat transports, CG coupling, RoPE/T5 position, alternate decoders,
  randomized refinement depth, and policy scoring.

The checked-in profile will be labeled a mutable experiment snapshot so future config changes do
not redefine the architecture.

## Component boundaries

The README will link each responsibility to its owner:

| Responsibility | Source owner |
|---|---|
| Configuration and compatibility guards | `vfe3/config.py` |
| Belief state and distribution families | `vfe3/belief.py`, `vfe3/families/` |
| Pair energies and attention objective | `vfe3/divergence.py`, `vfe3/free_energy.py` |
| Gauge groups, frames, transport, and retractions | `vfe3/geometry/` |
| Iterative refinement | `vfe3/inference/e_step.py`, `vfe3/gradients/` |
| Blocks, stack, model channel, and forward path | `vfe3/model/` |
| Encode and decode boundaries | `vfe3/model/prior_bank.py` |
| Outer optimization | `vfe3/train.py` |
| Generation and policy scoring | `vfe3/inference/`, `generate_efe.py`, `efe_ring_experiment.py` |
| Metrics, artifacts, and figures | `vfe3/metrics.py`, `vfe3/run_artifacts.py`, `vfe3/viz/` |

The status text will identify registered-but-raising seams such as `gauge_fixed` and inactive
estimators such as `sigma_mc`. It will describe the same-scale model channel as implemented but
restricted to a diagonal Gaussian, flat model transport, and a global centroid; it will not call
the full multiscale PIFB hierarchy implemented.

## Operational content

The quickstart will require Python 3.10 or newer and use
`python -m pip install -e ".[dev,data,viz]"`. It will explain that real-corpus entrypoints read
pre-tokenized `.pt` or `.bin` streams under `~/.cache/tokenized_cache` and intentionally do not
download, tokenize, or substitute synthetic data. The primary command will be
`python train_vfe3.py` after editing its configuration dictionary.

The entrypoint list will include `train_vfe3.py`, `ablation.py`, `scaling.py`, `make_figures.py`,
`generate_efe.py`, and `efe_ring_experiment.py`, each with one sentence of scope. Artifact
documentation will name the durable files produced by `RunArtifacts` rather than enumerate every
possible figure.

Testing instructions in the README may document plain pytest, `--runslow`, and
`VFE3_TEST_DEVICE=cuda`. For this documentation task itself, pytest will not run because the user
explicitly waived the suite. Verification will instead cover Markdown structure, relative links,
source claims, formatting, and the final diff.

## Files and change scope

The implementation phase will modify `README.md` and append the same-day post-edit record to
`docs/2026-07-11-edits.md`. This specification is the only additional permanent file. No source,
configuration, manuscript, test, run artifact, or Research-vault file will change.

## Verification

Documentation verification will perform all of the following without running pytest:

1. Check every repository-relative link and referenced path for existence.
2. Check balanced Markdown fences and the single Mermaid block's structural syntax by inspection.
3. Run `git diff --check` for whitespace errors.
4. Search the README for superseded statements such as "only learnable objects," "converged
   belief," "single authoritative scalar" for the full training loop, "every seam is a registry,"
   and "canonical vault copies."
5. Recheck the click-run profile against `train_vfe3.py` and the data flow against
   `VFEModel.forward_beliefs`, `VFEModel.forward`, `vfe_stack`, `vfe_block`, and `e_step`.
6. Inspect `git status --short` and the staged diff before each commit.

The final Git lifecycle will follow the repository requirement: commit and push the task branch,
merge it into `main`, push `main`, fast-forward the user's local checkout only if its WIP remains
safe, remove the temporary worktree and local task branch, and report the resulting SHAs and actual
status.
