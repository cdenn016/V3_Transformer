# README Mathematical Derivations Design

## Purpose

Expand `README.md` from an architecture-first overview into a derivation-level technical landing
page for readers who already understand machine learning, language models, variational inference,
information geometry, or geometric deep learning. The expansion must expose enough mathematics to
reconstruct the checked-in forward path without turning the README into a duplicate manuscript.

This specification supersedes the earlier design's limit of two displayed equations. It preserves
the earlier architecture, execution-profile, operational, and epistemic boundaries. The intended
result contains 12 thematic display blocks and remains readable as a GitHub README. The
specification shows constituent formulas separately for mathematical review; implementation will
consolidate related formulas into those 12 blocks.

## Evidence base

Implementation claims are grounded in committed `origin/main` at
`fc6fecd38ef56f3851c6a522088364a5fe72f78d`. The primary source seams are
`train_vfe3.py`, `vfe3/belief.py`, `vfe3/families/gaussian.py`,
`vfe3/free_energy.py`, `vfe3/alpha_i.py`, `vfe3/geometry/transport.py`,
`vfe3/geometry/retraction.py`, `vfe3/gradients/kernels.py`,
`vfe3/inference/e_step.py`, `vfe3/model/model.py`, `vfe3/model/prior_bank.py`,
and `vfe3/train.py`.

Theory wording follows the current Research-vault synthesis pages `VFE Transformer Program`,
`GL(K) gauge-equivariant attention`, `GL(K) gauge group`, `Variational free energy`, and `Natural
gradient`, together with the July 9 review/revision record. Public README links continue to point
only to repository-contained manuscripts. The Research vault remains read-only and does not appear
as a public dependency.

## Chosen presentation

The equations will form an integrated mathematical core near the top of the README. Distributing
them across operational sections would fragment the derivation, while a collapsible appendix would
hide the model's defining computation. The selected layout adds two top-level sections and expands
the existing attention, geometry, and decode subsections.

The revised top-level order will be:

1. `Architecture at a glance`
2. `Mathematical state and transport`
3. `Attention as variational source selection`
4. `Inner objectives and executable updates`
5. `Execution profiles`
6. `End-to-end model flow`
7. `Geometry and mathematical scope`
8. `Registry-driven extension points`
9. `Implementation status`
10. `Running the repository`
11. `Outputs and diagnostics`
12. `Repository map`
13. `Theory and manuscripts`

The existing Mermaid graph remains the only diagram. The derivation will use full prose, compact
notation, and displayed equations. It will not introduce empirical claims, historical pass counts,
benchmark marketing, private paths, or an exhaustive configuration reference.

## Scope labels

Each mathematical block must identify one of three scopes in its surrounding prose:

- **Ambient exact result:** a theorem for unrestricted full Gaussians and invertible pushforwards.
- **Reusable-engine formula:** an implemented route whose assumptions are stated explicitly.
- **Checked-in specialization:** the exact values and branches selected by committed
  `train_vfe3.py`.

No equation may silently move between these scopes. In particular, the full-Gaussian invariance
theorem does not certify the projected diagonal route, the registered Fisher-gradient formula does
not describe the active MM update, and the inner structural objectives do not become the outer
next-token loss.

## Notation and belief geometry

The mathematical core will begin with a compact notation table defining `q`, `p`, `s`, `r`,
`beta`, `gamma`, `phi`, `U`, `Omega`, `K`, `H`, and `d_h`. It will then define the ambient Gaussian
belief and frame construction:

$$
q_i=\mathcal N(\mu_i,\Sigma_i),
\qquad
\mu_i\in\mathbb R^K,
\qquad
\Sigma_i\in\operatorname{SPD}(K),
$$

$$
A_i=\sum_a\phi_i^aG_a\in\mathfrak g,
\qquad
U_i=\exp(A_i),
\qquad
\Omega_{ij}=U_iU_j^{-1}.
$$

For equal gauge blocks, the README will state

$$
G_{\mathrm{block}}=\prod_{h=1}^{H}\operatorname{GL}(d_h),
\qquad
d_h=K/H,
\qquad
U_i=\operatorname{diag}\left(U_i^{(1)},\ldots,U_i^{(H)}\right).
$$

The checked-in chart has `K=20`, `H=2`, and `d_h=10`, so it realizes two `GL^+(10)` gauge blocks.
The prose will distinguish algebra coordinates `phi`, algebra matrices `A`,
and group elements `U`. It will also state that one real exponential lies in the positive
determinant component but does not cover every element of that component.

## Transport and Gaussian comparison energy

For an ambient full Gaussian, transport is the exact pushforward

$$
\widetilde\mu_{ij}=\Omega_{ij}\mu_j,
\qquad
\widetilde\Sigma_{ij}=\Omega_{ij}\Sigma_j\Omega_{ij}^{\top}.
$$

For the implemented diagonal family, the README will immediately show the projected operation

$$
\widetilde\sigma_{ij}
=\operatorname{diag}\left(
\Omega_{ij}\operatorname{Diag}(\sigma_j)\Omega_{ij}^{\top}
\right),
$$

and explain that this is not closure under general `GL(K)` congruence. Exact unrestricted
`GL(K)` covariance applies to the full family; the diagonal cone is preserved for all inputs only
by monomial transformations.

The full-Gaussian forward KL will be displayed as

$$
D_{\mathrm{KL}}(P\Vert Q)
=\frac12\left[
\operatorname{tr}(\Sigma_Q^{-1}\Sigma_P)
+(\mu_P-\mu_Q)^{\top}\Sigma_Q^{-1}(\mu_P-\mu_Q)
-K
+\log\frac{\det\Sigma_Q}{\det\Sigma_P}
\right].
$$

The literal checked-in per-head energy will then specialize to diagonal variances:

$$
E_{ij}^{(h)}
=C_{[0,K_{\max}]}
\left[
\frac12\sum_{k\in I_h}
\left(
\frac{\sigma_{ik}}{\widetilde\sigma_{ij,k}}
+\frac{(\widetilde\mu_{ij,k}-\mu_{ik})^2}{\widetilde\sigma_{ij,k}}
-1
+\log\frac{\widetilde\sigma_{ij,k}}{\sigma_{ik}}
\right)
\right].
$$

The prose will define `I_h`, the variance floor, and the safety map `C`. For the checked-in run,
order-one Renyi is forward KL and `K_max=8K=160`.

## Gauge covariance and flatness

The local gauge statement will use the induced transport law rather than an additive update to
`phi`:

$$
q_i'=(h_i)_*q_i,
\qquad
q_j'=(h_j)_*q_j,
\qquad
\Omega_{ij}'=h_i\Omega_{ij}h_j^{-1},
$$

$$
D_{\mathrm{KL}}\left(
q_i'\Vert(\Omega_{ij}')_*q_j'
\right)
=D_{\mathrm{KL}}\left(
q_i\Vert(\Omega_{ij})_*q_j
\right).
$$

The README will say that this exact result governs the ambient pair score. The current diagonal
projection, linear readout, and untied head mixer are outside that theorem.

Flat Regime-I transport will be summarized by

$$
\Omega_{ij}\Omega_{jk}=\Omega_{ik},
\qquad
\Omega_{ij}\Omega_{jk}\Omega_{ki}=I.
$$

This is an operator identity of the vertex cocycle. The prose will not transfer path independence
to repeated projected-diagonal covariance updates, because diagonal projection is not a group
action.

## Gibbs attention

The existing fixed-row derivation remains and gains the implemented head temperature:

$$
\mathcal F_i(\beta_i)
=\sum_j\beta_{ij}E_{ij}
+\tau_h\sum_j\beta_{ij}\log\frac{\beta_{ij}}{\pi_{ij}},
\qquad
\tau_h=\kappa_\beta\sqrt{d_h},
$$

$$
\beta_{ij}^{*}
=\frac{\pi_{ij}\exp(-E_{ij}/\tau_h)}
{\sum_k\pi_{ik}\exp(-E_{ik}/\tau_h)}.
$$

Uniqueness is stated only on the active support at fixed beliefs, transports, energies, and prior.
Registry-selected scalar energies preserve this Gibbs calculation but do not all acquire the same
ELBO or mixture interpretation. The checked-in beta and gamma temperatures are both `sqrt(10)`.

## Belief- and model-channel objectives

The core target-blind belief objective will be written as

$$
\mathcal F_q
=\sum_i\left[a_iD(q_i\Vert p_i)+R(a_i)\right]
+\lambda_\beta\sum_{h,i,j}\beta_{ij}^{(h)}
\left[
E_{ij}^{q,h}
+\tau_{\beta,h}\log\frac{\beta_{ij}^{(h)}}{\pi_{ij}^{q,h}}
\right],
$$

with the state-dependent self-coupling

$$
R(a)=b_0a-c_0\log a,
\qquad
a_i^*=\frac{c_0}{b_0+D(q_i\Vert p_i)}.
$$

The optional detached two-hop term will be described in prose and identified as zero in the
checked-in profile. For the checked-in `b_0=c_0=1` and initial `q^(0)=p`, the only q iteration
evaluates `D(q^(0)||p)=0` and therefore `a_i^*=1`.

The same-scale model objective will be written as

$$
\mathcal F_s
=\lambda_h\sum_iD_{\mathrm{KL}}(s_i\Vert r)
+\lambda_\gamma\sum_{h,i,j}\gamma_{ij}^{(h)}
\left[
E_{ij}^{s,h}
+\tau_{\gamma,h}\log\frac{\gamma_{ij}^{(h)}}{\pi_{ij}^{s,h}}
\right].
$$

The checked-in coefficients are `lambda_h=0.25`, `lambda_gamma=0.75`, and
`tau_gamma=sqrt(10)`. The global, token-independent `r` is frozen because `learnable_r=False`.
One damped model step produces

$$
q_i^{(0)}=p_i=s_i^{(1)}.
$$

The README will identify this as an implemented same-scale hierarchy, not the full multiscale PIFB
program and not evidence of a slower model timescale.

## Detached hierarchical attention prior

The checked-in belief prior will be shown in two stages. First, the refined model covariance adds
a detached global reliability bias:

$$
\rho_j=-\log\left(2+\operatorname{tr}\Sigma_{s,j}^{(1)}\right),
\qquad
\pi_{hij}^{b}=\operatorname{softmax}_j\left(B_{hij}^{q}+\rho_j\right).
$$

Then the detached model posterior is mixed in probability space:

$$
\overline\pi_{hij}^{q}
=(1-w)\pi_{hij}^{b}
+w\operatorname{stopgrad}(\gamma_{hij}),
\qquad
w=0.5.
$$

The final q attention uses `overline pi` in the Gibbs rule. The README will distinguish the
`lambda_gamma=0.75` model-coupling weight from the independent `w=0.5` prior-mixture weight.

## Frozen-attention MM update

The README will derive the one-hop diagonal-KL target used by the checked-in route. With clamp
masks absorbed into nonnegative effective weights `c_ik` and `w_ijk`, define

$$
P_{ik}
=\frac{c_{ik}}{\sigma_{p,ik}}
+\sum_j\frac{w_{ijk}}{\widetilde\sigma_{ij,k}},
$$

$$
\mu_{ik}^{*}
=\frac{
c_{ik}\mu_{p,ik}/\sigma_{p,ik}
+\sum_jw_{ijk}\widetilde\mu_{ij,k}/\widetilde\sigma_{ij,k}
}{P_{ik}},
\qquad
\sigma_{ik}^{*}
=\frac{c_{ik}+\sum_jw_{ijk}}{P_{ik}}.
$$

The prose will define this as the exact minimizer of a frozen-attention, detached-key,
diagonal-KL majorizer. It is not an exact minimizer of the self-consistent profiled objective.

When covariance is enabled, damping acts in Gaussian natural coordinates:

$$
\Lambda^{+}=(1-\eta)\Lambda+\eta\Lambda^{*},
\qquad
(\Lambda\mu)^{+}
=(1-\eta)\Lambda\mu+\eta\Lambda^{*}\mu^{*}.
$$

The checked-in s step uses this route with `eta=0.75`. The q step freezes covariance and instead
uses

$$
\mu^{(1)}=0.25\mu^{(0)}+0.75\mu^{*},
\qquad
\sigma^{(1)}=\sigma^{(0)},
\qquad
\phi^{(1)}=\phi^{(0)}.
$$

The last equality concerns the q E-step because `e_phi_lr=0`. Frames still train through outer
AdamW at `m_phi_lr=0.010`. The README will say that the q covariance E-step is disabled, not that
all covariance dynamics are absent: the s covariance updates upstream and the active mixer can
transform covariance after the q step.

## Fisher-gradient route

The reusable engine's Gaussian gradient route will be documented separately:

$$
\widetilde\nabla_{\mu}\mathcal F
=\Sigma\nabla_{\mu}\mathcal F,
\qquad
\widetilde\nabla_{\Sigma}\mathcal F
=2\Sigma\operatorname{sym}(\nabla_{\Sigma}\mathcal F)\Sigma.
$$

For diagonal variances this becomes

$$
\widetilde\nabla_{\mu}\mathcal F
=\sigma\odot\nabla_{\mu}\mathcal F,
\qquad
\widetilde\nabla_{\sigma}\mathcal F
=2\sigma^2\odot\nabla_{\sigma}\mathcal F.
$$

The surrounding paragraph will state that these equations describe the registered Gaussian
gradient route, not the current MM route or the plain-AdamW frame update. No convergence theorem is
transferred to the checked-in finite schedule.

## Decode and outer objective

The checked-in decoder and supervised objective will be shown as

$$
z_{iv}=m_i^{\top}W_v+b_v,
\qquad
\mathcal L_{\mathrm{CE}}
=-\frac1M\sum_{b,i:y_{bi}\ne-100}
\log\operatorname{softmax}(z_{bi})_{y_{bi}}.
$$

The optional KL-to-prior boundary will be summarized as

$$
z_{iv}=-\frac{D_{\mathrm{KL}}(q_i^*\Vert p_v^{\mathrm{decode}})}
{\tau_{\mathrm{decode}}}.
$$

The README will note that the decoder registry also contains non-KL modes. For the checked-in
linear boundary, covariance and decode temperature do not enter the logits.

The target shift is `x_i=t_i`, `y_i=t_{i+1}`. At the committed configuration,
`mass_phi=0`, `mstep_self_coupling_weight=0`, `z_loss_weight=0`, and the explicit hyperprior/gamma
outer blocks are gated off under `s_e_step=True`. The backpropagated scalar is therefore exactly
next-token cross-entropy through the unrolled s and q paths. The detached gamma-to-beta fold changes
the forward q update but carries no gradient into s through that fold.

## Objective-boundary statement

The mathematical core will end with a prominent prose statement: the production forward pass does
not optimize one ELBO or one VFE scalar. `F_s` and `F_q` are target-blind structural objectives;
`L_CE` is a separate supervised outer objective. Each inner stage performs one damped,
frozen-attention MM step rather than an exact self-consistent argmin. No shared-functional EM
monotonicity, evidence ascent, convergence, or global free-energy descent follows.

## Final display-block packing

Implementation will consolidate the constituent formulas above into exactly 12 thematic display
blocks:

1. Belief state, algebra-to-group frame map, and equal-block group.
2. Full-Gaussian pushforward and projected diagonal covariance transport.
3. Full-Gaussian KL and the checked-in per-head diagonal energy.
4. Local gauge law, invariant pair score, cocycle composition, and loop identity.
5. Entropy-regularized row objective, head temperature, and Gibbs solution.
6. Belief objective, self-coupling regularizer, and profiled coefficient.
7. Model objective, committed coefficients, and `q^(0)=p=s^(1)` handoff.
8. Precision reliability, detached gamma fold, and the resulting beta prior.
9. Frozen-attention MM precision, mean target, and variance target.
10. Natural-coordinate damping and the checked-in frozen-q-covariance specialization.
11. Full and diagonal Gaussian Fisher-gradient formulas, labeled as the alternate gradient route.
12. Linear and KL-to-prior decode formulas together with next-token cross-entropy.

## Files and change scope

The implementation phase will modify `README.md` and append the existing same-day record in
`docs/2026-07-11-edits.md`. This design specification and its implementation plan are the only
additional permanent files. No source, configuration, test, manuscript, run artifact, or
Research-vault file will change.

## Verification

Verification remains documentation-only because the user explicitly waived pytest. The final gate
will:

1. Resolve every repository-relative link.
2. Check balanced Markdown fences, math delimiters, brackets, and the existing Mermaid graph.
3. Reject forbidden LaTeX spacing commands and unsupported GitHub-math constructs.
4. Run the American-English, banned-prose, stale-claim, private-path, and horizontal-rule scans.
5. Run `git diff --check`.
6. Recheck every checked-in coefficient and branch against committed `train_vfe3.py`.
7. Recheck the MM equations against `vfe3/gradients/kernels.py` and
   `vfe3/inference/e_step.py`.
8. Recheck the gauge and Fisher equations against the current manuscripts and implementation.
9. Obtain independent runtime-mathematics and theory-scope reviews before merge.

The mandatory Git lifecycle remains unchanged: commit and push the task branch, merge and push
`main`, safely fast-forward the user's live branch only if its WIP cannot be altered, remove the
task-owned worktree and local branch, and report the actual final status.
