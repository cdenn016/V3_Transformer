# README Mathematical Derivations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the architecture-first README into a derivation-level technical landing page whose equations reconstruct the checked-in VFE transformer path without overstating its theoretical guarantees.

**Architecture:** Keep the existing execution graph and operational sections, but insert a mathematical spine immediately after the architecture overview. Pack the derivation into exactly 12 GitHub-rendered display blocks and label every result as an ambient exact theorem, a reusable-engine formula, or the checked-in specialization at commit `fc6fecd38ef56f3851c6a522088364a5fe72f78d`.

**Tech Stack:** GitHub-flavored Markdown, GitHub math rendering, Mermaid, Python/PyTorch source links, and PowerShell documentation checks.

## Global Constraints

- Modify only `README.md` and `docs/2026-07-11-edits.md`; retain this specification and plan as the only additional permanent files.
- Do not modify source, configuration, tests, manuscripts, run artifacts, or the Research vault.
- Do not run pytest; the user explicitly waived tests for this documentation-only task.
- Use American English and academic prose. Do not use horizontal rules, LaTeX spacing macros, private vault paths, badges, pass counts, benchmark marketing, or the repository's banned prose patterns.
- Preserve the existing Mermaid graph as the sole diagram and retain all currently valid source and manuscript links.
- Produce exactly 12 display-math blocks, delimited by 24 standalone `$$` lines.
- Distinguish ambient full-Gaussian theorems, reusable registered routes, and the exact checked-in experiment in surrounding prose.
- Do not claim that the production path optimizes one ELBO, that its finite MM schedule converges, that projected diagonal covariance is unrestricted `GL(K)`-equivariant, or that the complete model is gauge invariant.
- Preserve the mandatory isolated Git lifecycle, including branch push, merge to `main`, remote verification, safe handling of live WIP, and task-worktree cleanup.

## File Structure

- `README.md`: public architecture, derivation, execution-profile, operation, and theory landing page.
- `docs/2026-07-11-edits.md`: same-day durable record of the derivation expansion and its documentation-only verification.
- `docs/superpowers/specs/2026-07-11-readme-mathematical-derivations-design.md`: approved mathematical and editorial contract; do not revise during implementation unless a source contradiction is found.
- `docs/superpowers/plans/2026-07-11-readme-mathematical-derivations.md`: this executable checklist.

### Task 1: Add the mathematical state, transport, gauge, and attention derivation

**Files:**
- Modify: `README.md`, after `## Architecture at a glance` and through `## Attention as variational source selection`

**Interfaces:**
- Consumes: the notation and scope contract in the approved design and the current source behavior in `vfe3/belief.py`, `vfe3/families/gaussian.py`, `vfe3/geometry/transport.py`, `vfe3/geometry/retraction.py`, and `vfe3/attention.py`.
- Produces: the first five display blocks and the definitions used by every later objective and update equation.

- [ ] **Step 1: Insert the notation and scope preamble**

Add `## Mathematical state and transport` after the architecture graph. Define `q`, `p`, `s`, `r`, `beta`, `gamma`, `phi`, `U`, `Omega`, `K`, `H`, and `d_h` in one compact Markdown table. State the three scopes in prose: ambient exact result, reusable-engine formula, and checked-in specialization.

- [ ] **Step 2: Write display block 1 for beliefs and frames**

In one display block, define the Gaussian state, Lie-algebra chart, group element, relative transport, and block group:

```text
q_i=N(mu_i,Sigma_i), mu_i in R^K, Sigma_i in SPD(K);
A_i=sum_a phi_i^a G_a;
Ahat_i=A_i when ||A_i||_F<=20, else Ahat_i=20 A_i/||A_i||_F;
U_i=exp(Ahat_i), Omega_ij=U_i U_j^{-1};
G_block=product_{h=1}^H GL(d_h), d_h=K/H,
U_i=diag(U_i^(1),...,U_i^(H)).
```

Immediately specialize to `K=20`, `H=2`, and `d_h=10`, and describe the checked-in frame as two `GL^+(10)` blocks. State that the non-skew path applies the whole-frame norm-20 clamp before exponentiation. Explain that the ordinary exponential parameterization applies without modification only inside the unclamped region; beyond it, radial magnitudes collapse onto the clamp boundary, and the effective joint frame image is a restricted subset of the unclamped block-group exponential image. Also state that each real matrix exponential stays in the positive-determinant component but that the blockwise exponential map is not a surjection onto that component. Require a gradient-boundary sentence: the piecewise clamp gives forward values, while the implementation computes the scale under `torch.no_grad()` and autograd treats it as constant, so outer frame gradients are not the derivative of the displayed radial normalization.

- [ ] **Step 3: Write display blocks 2 and 3 for transport and comparison energy**

Block 2 must place the ambient pushforward beside the implemented diagonal projection:

```text
mu_tilde_ij=Omega_ij mu_j,
Sigma_tilde_ij=Omega_ij Sigma_j Omega_ij^T,
sigma_tilde_ij=diag(Omega_ij Diag(sigma_j) Omega_ij^T).
```

State that unrestricted congruence is exact for full covariance, while the diagonal cone is preserved for every diagonal input only by monomial transformations.

Block 3 must contain the full forward Gaussian KL and the literal checked-in per-head diagonal energy over index set `I_h`, including the safety map `C_[0,K_max]`. Define the variance floor, `K_max=8K=160`, and the order-one Renyi selection as forward KL.

- [ ] **Step 4: Write display block 4 for gauge covariance and flatness**

Put the local transformation law, invariant ambient score, cocycle identity, and loop identity in one block:

```text
q_i'=(h_i)_*q_i, q_j'=(h_j)_*q_j,
Omega_ij'=h_i Omega_ij h_j^{-1};
D_KL(q_i' || (Omega_ij')_*q_j')=D_KL(q_i || (Omega_ij)_*q_j);
Omega_ij Omega_jk=Omega_ik,
Omega_ij Omega_jk Omega_ki=I.
```

Limit exact invariance to ambient full-Gaussian pushforwards. Explicitly exclude the projected diagonal route, linear readout, and untied mixer from that theorem, and describe flatness as an operator identity rather than path independence for repeated projection.

- [ ] **Step 5: Expand attention into display block 5**

Retain `## Attention as variational source selection`, but replace the compact derivation with one block containing the fixed-row objective, `tau_h=kappa_beta sqrt(d_h)`, and the normalized Gibbs solution with prior `pi_ij`. State uniqueness only on the active support with beliefs, transports, energies, and prior fixed. Record `tau_beta=tau_gamma=sqrt(10)` for the checked-in profile.

- [ ] **Step 6: Inspect the first-stage structure**

Run:

```powershell
rg -n "^## |^### " README.md
(Select-String -Path README.md -Pattern '^\$\$$').Count
```

Expected: `## Mathematical state and transport` appears between architecture and attention; the interim delimiter count is `10` for five display blocks.

- [ ] **Step 7: Commit the first mathematical stage**

```powershell
git add README.md
git diff --cached --check
git commit -m "docs: derive belief transport and attention"
```

Expected: the staged whitespace check exits `0`, and the commit records only `README.md`.

### Task 2: Derive the inner objectives and executable updates

**Files:**
- Modify: `README.md`, new `## Inner objectives and executable updates` section

**Interfaces:**
- Consumes: notation and transported energies from Task 1, plus `vfe3/free_energy.py`, `vfe3/alpha_i.py`, `vfe3/gradients/kernels.py`, `vfe3/inference/e_step.py`, `vfe3/model/model.py`, and `vfe3/model/prior_bank.py`.
- Produces: display blocks 6 through 11 and the exact boundary between structural objectives and the active finite update schedule.

- [ ] **Step 1: Write display block 6 for the q objective and profiled self-coupling**

Add the target-blind belief objective with the beta-weighted one-hop energy and entropy term. In the same block define

```text
R(a)=b_0 a-c_0 log(a),
a_i^*=c_0/(b_0+D(q_i||p_i)).
```

State that the optional two-hop coefficient is zero in the checked-in profile. With `b_0=c_0=1` and `q^(0)=p`, explain that the only q iteration begins at zero self-divergence and therefore evaluates `a_i^*=1`; do not generalize this value beyond that initialization.

- [ ] **Step 2: Write display block 7 for the s objective and handoff**

Show the same-scale model objective with hyperprior and gamma consensus terms, then include the committed coefficients and handoff in the same block:

```text
lambda_h=0.25, lambda_gamma=0.75, tau_gamma=sqrt(10),
q_i^(0)=p_i=s_i^(1).
```

State that the token-independent global `r` is frozen because `learnable_r=False`. Describe this as an implemented same-scale hierarchy, not the full multiscale PIFB theory and not evidence of a slower learned model timescale.

- [ ] **Step 3: Write display block 8 for the detached hierarchical attention prior**

Put reliability and the probability-space gamma fold in one block:

```text
rho_j=-log(2+tr Sigma_(s,j)^(1)),
pi_hij^b=softmax_j(B_hij^q+rho_j),
pi_bar_hij^q=(1-w)pi_hij^b+w stopgrad(gamma_hij), w=0.5.
```

Distinguish the prior-mixture weight `w=0.5` from the model-coupling coefficient `lambda_gamma=0.75`, and state that this detached fold alters the q forward update without carrying a gradient into s through that edge.

- [ ] **Step 4: Write display block 9 for the mask-selected `mm_exact` target**

Define the strict pair mask `m_ij^(h) = 1{0 < E_ij^(h) < K_max}`, the nonnegative effective weights `c_ik` and `w_ijk`, precision `P_ik`, precision-weighted mean target `mu_ik^*`, and variance target `sigma_ik^*`. With attention, transported keys, coefficients, and masks fixed, label the formulas as closed-form minimizers over the enabled, nondegenerate coordinates of the mask-selected diagonal-KL surrogate implemented by `mm_exact`. State that exact-zero pair energies are omitted, so the surrogate is not a majorizer of the canonical frozen-attention objective, and deny majorization, descent, and exact self-consistent-argmin guarantees.

- [ ] **Step 5: Write display block 10 for damping and the checked-in q specialization**

Place natural-coordinate Gaussian damping beside the active q E-step:

```text
Lambda^+=(1-eta)Lambda+eta Lambda^*,
(Lambda mu)^+=(1-eta)Lambda mu+eta Lambda^* mu^*;
mu^(1)=0.25 mu^(0)+0.75 mu^*,
sigma^(1)=sigma^(0), phi^(1)=phi^(0).
```

State that the s step uses covariance-enabled `mm_exact` fusion with `eta=0.75`, while the q E-step freezes covariance and frames. Clarify that `e_phi_lr=0` does not freeze outer frame learning: AdamW uses `m_phi_lr=0.010`. Also clarify that the upstream s covariance and active mixer can still change covariance outside the q E-step.

- [ ] **Step 6: Write display block 11 for the alternate Fisher-gradient route**

Show the full-covariance and diagonal Gaussian natural gradients:

```text
natgrad_mu F=Sigma grad_mu F,
natgrad_Sigma F=2 Sigma sym(grad_Sigma F) Sigma;
natgrad_mu F=sigma odot grad_mu F,
natgrad_sigma F=2 sigma^2 odot grad_sigma F.
```

Label these as a reusable registered route that is inactive under the checked-in `mm_exact` selection. Do not apply these equations to the frame AdamW update or infer convergence for the finite schedule.

- [ ] **Step 7: State the objective boundary**

End the section with a prominent prose paragraph stating that `F_s` and `F_q` are target-blind structural objectives, the outer next-token loss is separate, and each inner stage takes one damped step toward a mask-selected `mm_exact` fusion target. Explicitly deny majorization, surrogate-descent, exact self-consistent-argmin, one-ELBO, shared-functional EM monotonicity, evidence-ascent, convergence, and global-free-energy-descent claims for the production forward pass.

- [ ] **Step 8: Inspect the second-stage structure**

Run:

```powershell
(Select-String -Path README.md -Pattern '^\$\$$').Count
rg -n "one ELBO|frozen-attention|mm_exact|m_phi_lr|learnable_r" README.md
```

Expected: the delimiter count is `22` for 11 display blocks, and every boundary term appears in explanatory prose.

- [ ] **Step 9: Commit the executable-update stage**

```powershell
git add README.md
git diff --cached --check
git commit -m "docs: derive VFE inner updates"
```

Expected: the staged whitespace check exits `0`, and the commit records only `README.md`.

### Task 3: Complete decode mathematics and integrate the document

**Files:**
- Modify: `README.md`
- Modify: `docs/2026-07-11-edits.md`

**Interfaces:**
- Consumes: the 11 blocks from Tasks 1 and 2 and the decoder/outer-loss paths in `vfe3/model/model.py`, `vfe3/model/prior_bank.py`, `vfe3/train.py`, and `train_vfe3.py`.
- Produces: the twelfth display block, the final H2 order, and a durable implementation record.

- [ ] **Step 1: Write display block 12 for decode and supervision**

Expand `### Decode and outer objective` so one display block contains the checked-in linear logits, optional KL-to-prior logits, and shifted-token cross-entropy:

```text
z_iv=m_i^T W_v+b_v,
z_iv=-D_KL(q_i^*||p_v^decode)/tau_decode,
L_CE=-(1/M) sum_(b,i:y_bi!=-100) log softmax(z_bi)_(y_bi).
```

State that the first and second logit equations are alternative registered decode boundaries. The checked-in linear path has `use_prior_bank=False`, so covariance and decode temperature do not enter its logits. Record `x_i=t_i`, `y_i=t_(i+1)`.

- [ ] **Step 2: Pin the exact outer scalar**

Explain that `mass_phi=0`, `mstep_self_coupling_weight=0`, `z_loss_weight=0`, and `s_e_step=True` gates the explicit hyperprior/gamma outer blocks. Therefore the checked-in backpropagated scalar is exactly next-token cross-entropy through the unrolled s and q paths. Do not imply that this makes cross-entropy identical to either inner structural objective.

- [ ] **Step 3: Integrate the final section order**

Ensure the H2 order is exactly:

```text
Architecture at a glance
Mathematical state and transport
Attention as variational source selection
Inner objectives and executable updates
Execution profiles
End-to-end model flow
Geometry and mathematical scope
Registry-driven extension points
Implementation status
Running the repository
Outputs and diagnostics
Repository map
Theory and manuscripts
```

Remove duplicated equations or claims displaced by the new mathematical spine, but retain operational instructions, the component maps, status tables, repository map, and public manuscript links.

- [ ] **Step 4: Append the same-day edit record**

Append one subsection named `## README derivation-level mathematical expansion` to `docs/2026-07-11-edits.md`. Record the 12-block derivation, exact/reusable/checked-in labels, the objective-boundary correction, source seams reviewed, the explicit pytest waiver, and documentation-only verification. Do not add pass counts or claim runtime validation.

- [ ] **Step 5: Commit the integrated README**

```powershell
git add README.md docs/2026-07-11-edits.md
git diff --cached --check
git commit -m "docs: complete README mathematical architecture"
```

Expected: the staged whitespace check exits `0`, and the commit contains the integrated README plus the dated record.

### Task 4: Conduct independent mathematical and architecture reviews

**Files:**
- Review: `README.md`
- Review: `train_vfe3.py`
- Review: `vfe3/gradients/kernels.py`
- Review: `vfe3/inference/e_step.py`
- Review: `vfe3/model/model.py`
- Review: `vfe3/families/gaussian.py`
- Review: `vfe3/geometry/transport.py`
- Review: repository manuscripts linked from `README.md`

**Interfaces:**
- Consumes: the integrated README.
- Produces: two independent review reports delivered to the coordinating agent: one code-to-equation trace and one theory-scope audit.

- [ ] **Step 1: Dispatch the runtime-mathematics review**

Ask a fresh reviewer to check every checked-in constant, active branch, detach edge, update schedule, decoder selection, and outer-loss gate against executable source. Require file-and-line evidence for every correction.

- [ ] **Step 2: Dispatch the theory-scope review**

Ask a different reviewer to check KL orientation, Gaussian pushforwards, diagonal projection limits, gauge covariance, cocycle flatness, Gibbs stationarity, MM assumptions, Fisher geometry, and ELBO/convergence boundaries. Require the reviewer to distinguish mathematical error from editorial preference.

- [ ] **Step 3: Apply only evidence-backed corrections**

Edit `README.md` and the dated record for every confirmed defect. Reject stylistic expansion that would duplicate the manuscripts or weaken GitHub readability.

- [ ] **Step 4: Commit review corrections if needed**

```powershell
git add README.md docs/2026-07-11-edits.md
git diff --cached --check
git diff --cached --stat
git commit -m "docs: correct README derivation review findings"
```

Expected when corrections exist: one correction commit records only the two authorized documentation files. When the staged stat is empty, skip the commit command.

### Task 5: Perform documentation-only verification and complete Git lifecycle

**Files:**
- Verify: `README.md`
- Verify: `docs/2026-07-11-edits.md`
- Verify: the two permanent specification/plan files

**Interfaces:**
- Consumes: the reviewed documentation branch.
- Produces: verified `origin/main`, a safely handled live checkout, and no remaining task-owned temporary worktree or local task branch.

- [ ] **Step 1: Verify structure and display packing**

```powershell
rg -n "^## |^### " README.md
(Select-String -Path README.md -Pattern '^\$\$$').Count
(Select-String -Path README.md -Pattern '^```').Count
git diff origin/main...HEAD --check
```

Expected: the 13 H2 headings have the specified order, the math delimiter count is `24`, the fence count is even, and the diff check exits `0`.

- [ ] **Step 2: Scan forbidden and stale language**

```powershell
rg -n -i "colour|behaviour|normalise|optimise|factorise|centre|modelling|fibre|key insight|crucially|critically|notably|importantly|worth noting|fundamentally|leverages|underscores" README.md docs/2026-07-11-edits.md
rg -n -F -e '\;' -e '\,' -e '\!' README.md docs/2026-07-11-edits.md
rg -n "Research\\|Users\\|one ELBO|global free-energy descent|whole-model gauge" README.md
```

Expected: the first two scans return no matches. Any objective-boundary matches from the third scan must be explicit denials, and no private path may appear.

- [ ] **Step 3: Resolve repository-relative Markdown links**

Use a PowerShell link scan that extracts non-HTTP, non-anchor Markdown targets from `README.md`, strips optional line fragments, and passes each path to `Test-Path` relative to the repository root. Expected: zero missing targets.

- [ ] **Step 4: Reconcile the final diff and skip pytest**

```powershell
git status --short
git diff origin/main...HEAD --stat
git diff origin/main...HEAD -- README.md docs/2026-07-11-edits.md docs/superpowers/specs/2026-07-11-readme-mathematical-derivations-design.md docs/superpowers/plans/2026-07-11-readme-mathematical-derivations.md
```

Expected: only the four authorized documentation files differ from `origin/main`. Do not invoke pytest; report the waiver exactly.

- [ ] **Step 5: Push the task branch**

```powershell
git push -u origin codex/readme-mathematical-derivations-20260711
git ls-remote --heads origin codex/readme-mathematical-derivations-20260711
```

Expected: the remote task branch resolves to the verified branch tip.

- [ ] **Step 6: Merge and push main from an isolated clean integration worktree**

Fetch `origin`, confirm no unexpected divergence, merge the task branch into an isolated `main` worktree without rewriting history, push `main`, fetch again, and inspect `git log -3 origin/main`. Expected: `origin/main` contains every documentation commit.

- [ ] **Step 7: Protect the live checkout and clean up task-owned Git state**

Inspect the live checkout branch and `git status --short`. Fast-forward it only if that operation cannot alter or overwrite user WIP. Remove the temporary task worktree and delete the local task branch only after the merge and remote verification. Never stash, restore, reset, clean, or edit the pre-existing live changes.

- [ ] **Step 8: Record the completion receipt**

Report the task branch, documentation commit SHA, resulting `origin/main` SHA, documentation-check results, explicit pytest skip, remote task-branch state, worktree removal, local task-branch deletion, and the actual final live `git status --short` with remaining dirty files identified as user-owned WIP.
