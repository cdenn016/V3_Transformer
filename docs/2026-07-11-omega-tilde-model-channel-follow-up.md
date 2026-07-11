# Follow-up investigation: separate model-channel transport

Date: 2026-07-11

Scope authority: `docs/2026-07-11-omega-tilde-model-channel-investigation.md`

Status: live-source theory, code, checkpoint, and literature investigation; no production-code change

## Decision

The repository should not implement the memo's E1–E3 sequence as written. A separate
model-channel transport remains a theory-permitted research direction, but the proposed design
mixes two mathematically different objects: a second transport operating inside the transformer's
existing shared coordinate system, and a genuinely independent model bundle with its own gauge
group. The first is implementable with the present identity handoff $q^{(0)}=p=s^{(1)}$, but it is
not the independent-principal-bundle configuration described in PIFB2. The second requires an
explicit cross-fiber bridge before $s^{(1)}$ can seed $q^{(0)}$. The memo provides no such bridge.

The empirical prior also changes after checking the named checkpoint. The model-channel and belief
attention patterns are strongly related, but the two frame-gradient streams are positively aligned,
not conflicting, at the clean-commit near-replay endpoint. A frozen intervention shows that the
checkpoint tensors depend materially on model consensus and modestly on the gamma-to-beta fold under
the same near-replay. These results support a retrained gamma-content baseline and a gradient-routing
control; they do not support beginning with a new full frame table.

The next confirmatory work should therefore retain the shared forward geometry, run a retrained E0
baseline, and add a control that blocks only the model-channel contribution to the shared frame
gradient. A private transport should be considered only after the project chooses between a
same-gauge auxiliary transport and a genuinely independent bundle with a typed bridge.

## Evidence basis and version boundary

The named run directory `vfe3_runs/20260710-180301` no longer exists under that name. Its artifacts
match the renamed `vfe3_runs/142.75_mm-skipsig`: the saved configuration has $K=20$, two
$\mathrm{GL}(10)$ blocks, one layer, one E-step, `e_step_update="mm_exact"`,
`mm_damping=1.0`, `skip_belief_sigma_update=true`, and 15,153,002 parameters. The saved result is
test CE 4.96113 and test PPL 142.75457. `provenance.json` binds the run to dirty commit
`0300a358af02a95616a9e6d90117503d4bf90205`.

The best checkpoint is a legacy raw state dictionary, which current offline-report code correctly
refuses to treat as a self-bound bundle. The resumable `checkpoints/step_15000.pt` does contain the
embedded configuration and optimizer state. Its model tensors are elementwise identical to the raw
`best_model.pt`; its configuration differs from `config.json` only by the JSON list versus Python
tuple representation of `policy_score_terms`. This allowed a clean-commit near-replay, but not an
exact reconstruction of the dirty training tree.

The version split controls the interpretation of the memo:

| Execution graph | Model-channel update | Gamma used by the beta-prior fold | Consequence |
|---|---|---|---|
| Run provenance `0300a358` | Gradient route; `_refine_s` does not receive `e_step_update` or `mm_damping` | Raw model table $s^{(0)}$ | The checkpoint used gradient $s$ refinement and mm-exact $q$ refinement; gamma and beta did not read exactly the same Gaussians. |
| Memo commit `90f2361` | Same historical behavior | Raw $s^{(0)}$ | The memo's claim that the fold reads the same refined $s^{(1)}$ as beta is false for its own committed code lineage. |
| Current `origin/main` at investigation start, `c7064e9` | `_refine_s` receives `e_step_update` and `mm_damping` at `vfe3/model/model.py:783-784` | Refined $s^{(1)}$ is forwarded at `model.py:945-953` and consumed at `model.py:2077-2081` | Current source has the same-Gaussian redundancy mechanism, but it does not replay the training graph of the named checkpoint. |

No single checked-in execution graph supports all of the memo's simultaneous statements about the
checkpoint. Replaying its tensors under current `origin/main` produced CE 6.87 on a validation
sample, rather than the recorded approximately 4.96, so current-main frozen ablations were rejected
as behaviorally unbound. All checkpoint results below use the clean provenance commit and are
explicitly labeled near-replays because the original tree was dirty.

The memo also overstates the decode route. `prior_source="model_channel"` makes refined $s$ the
belief seed and self-prior, so it is prediction-critical. The captured run has
`use_prior_bank=false`, however, and decodes through the separate learned linear
`output_proj_weight` and bias. The $s$ table is not the active decode bank.

## Theory audit

### Independent frames are permitted, not selected by the theory

PIFB2 line 229 introduces the independent model bundle as an alternative that the framework admits.
PIFB2 line 459 then distinguishes shared and independent configurations. This language licenses an
independent frame but does not prefer it, assign it a performance sign, or make it the canonical
transformer realization. PIFB2 line 423 still states without qualification that model transport is
the same group element as belief transport. That sentence needs a tied-versus-independent qualifier
whether or not the transformer changes.

The model-sector covariance theorem itself does not require $K_m=K_q$. It requires a
dimensionally valid action within the $K_m$ model fiber. The matched-dimension restriction matters
for the transformer's direct bridge and for the manuscript's cross-fiber uniqueness extension, not
for ordinary invariance of a model-channel KL under a common invertible pushforward. General
congruence also leaves the diagonal-Gaussian family, so the implemented diagonal projection remains
an approximation under either frame configuration.

### A genuinely independent gauge requires a cross-fiber bridge

Let the state and model gauges act independently by $g_i$ and $h_i$. If a model belief seeds a state
belief through a linear bridge $C_i$, the typed handoff is

$$
q_i^{(0)}=p_i=(C_i)_\#s_i^{(1)},
$$

and covariance requires

$$
C_i' = g_i C_i h_i^{-1}.
$$

Then $(C_i')_\#(h_i)_\#s_i=(g_i)_\#(C_i)_\#s_i$. The current identity handoff $C_i=I$
is covariant only after locking $g_i=h_i$, which reduces the product gauge to its diagonal subgroup.
The transformer papers explicitly use $q^{(0)}=p=s^{(1)}$ at `GL(K)_attention.tex:2174-2177`
while also stating at line 707 that cross-fiber morphisms are not instantiated. PIFB2 lines 459 and
2138 likewise say that the canonical functional contains no cross-bundle morphism.

Consequently, swapping a private frame into `_refine_s` and `_gamma_energy` while leaving
$q=s$ unchanged does not realize PIFB2's independent principal bundles. It realizes a second learned
transport under a shared coordinate identification. That may be a useful ablation, but the report
and manuscripts must name it accurately.

### The proposed conjugation is either redundant or non-equivariant

E1 proposes $\widetilde\Omega=M\Omega M^{-1}$. This is a conjugate, hence equivalent,
representation. With the equivariant bridge $C=M^{-1}$ and transformed model coordinates
$t_i=(M^{-1})_\#s_i$, KL invariance gives

$$
D_{\mathrm{KL}}\left(s_i\middle\|(M\Omega M^{-1})_\#s_j\right)
=
D_{\mathrm{KL}}\left(t_i\middle\|\Omega_\#t_j\right).
$$

The pure full-Gaussian theory therefore predicts exact parity, not new expressive power. If the
implementation retains the identity bridge, fixed diagonal coordinates, or a fixed readout, $M$ can
change the computation, but its effect then comes from relative cross-fiber alignment, projection,
or optimization conditioning. A positive E1 result would not establish that a different invariant
geometry helps. The cleanest first E1 test is an exact-null arm that includes $C=M^{-1}$ and must
match the tied pure path.

The raw matrix $M$ is also not identifiable. At minimum, $M$ and $aM$ produce the same conjugation
for nonzero scalar $a$, and any element in the common centralizer of the observed transports creates
an additional equivalence. Diagnostics should therefore evaluate transported KL values, prediction
effects, and condition numbers rather than interpret individual entries of $M$ across seeds.

### The proposed low-rank algebra offset is flat but not automatically covariant

E2 writes $\widetilde\phi_i=\phi_i+A_i$ with
$A_i=\sum_a c_i^aG_a$. Vertex factorization makes
$\widetilde U_i\widetilde U_j^{-1}$ a flat cocycle for any such $A_i$, but covariance of the
low-rank family does not follow. Under a left gauge $e^\xi$, the required offset becomes the
difference of two Baker-Campbell-Hausdorff series,

$$
A_i' = \operatorname{BCH}(\xi,\phi_i+A_i)
-\operatorname{BCH}(\xi,\phi_i)
=A_i+\frac12[\xi,A_i]+\cdots.
$$

An arbitrary rank-eight span need not contain these commutators. For example,
$A=E_{12}$ and $\xi=E_{21}$ generate
$[\xi,A]=\operatorname{diag}(-1,1)$, which lies outside the one-dimensional span of $E_{12}$.
The memo's assertion that the proposed low-rank offset is covariant is therefore false without an
explicit transformation law and an adjoint-stable parameter family. A group-factorized relative
field such as $\widetilde U_i=U_i\exp(A_i)$ is a better starting point, but it still needs a declared
gauge action and a typed cross-fiber bridge if the two gauges are independent.

### Flatness does not prevent collapse

The cocycle statement is correct:

$$
(\widetilde U_i\widetilde U_j^{-1})
(\widetilde U_j\widetilde U_k^{-1})
=\widetilde U_i\widetilde U_k^{-1}.
$$

The claimed protection against "gauge cheating" is not. Choose any reference Gaussian $s_0$ and
set $s_i=(\widetilde U_i)_\#s_0$. Then

$$
(\widetilde U_i\widetilde U_j^{-1})_\#s_j=s_i
$$

for every ordered pair, so all $N^2$ transported KL terms vanish. PIFB2 line 1654 already identifies
this perfect transported-consensus limit with epistemic death. Flatness constrains the consistency
of edge transports; it does not guarantee nonzero gamma content. An exact SymPy witness with three
agents, rational two-dimensional frames, and a non-diagonal rational covariance verified all nine
pairwise KL values as zero while also verifying every cocycle identity.

### A second frame adds an unanchored gauge orbit and does not create a timescale

The transport is invariant under the global right action
$\widetilde U_i\mapsto\widetilde U_i g$ for every token $i$. A new full table therefore adds another
exact null orbit unless one reference frame is fixed, the quotient is optimized directly, or a
declared gauge slice is imposed. A chart penalty such as `mass_s_phi` is gauge-breaking chart
regularization; it is neither a gauge slice nor a gauge-invariant physical mass and can also alter
relative transports.

A separate parameter permits a separate learning-rate schedule, but no timescale separation follows
from its existence. PIFB2 lines 943, 952, and 956 posit update-rate ratios and make the adiabatic
reading conditional at line 961. The retained transformer route does not establish a slower model
timescale (`GL(K)_attention.tex:2147,2415-2417`; supplementary line 897). An independent frame is
necessary only if the design demands two distinct frame clocks without branchwise gradient scaling.
It is neither necessary for $\eta_s<\eta_q$ nor sufficient to establish an effective slow mode.

### Full-table zero initialization is not a nested control

The learned belief table is initialized as a nonzero random chart
(`vfe3/model/prior_bank.py:263`). A separate absolute table initialized at zero gives
$\widetilde\Omega_{ij}=I$, not the tied baseline $\Omega_{ij}$. Byte-identical initialization
requires a detached copy of the effective group frame,
$\widetilde U_i^{(0)}=U_i^{(0)}$, including positional or reflection factors that contribute to the
active frame. A zero coefficient table is nested only for the relative-offset construction. If a
full table is written as a live $\phi+\delta$, it remains coupled to $\phi$ and is not independent.

## Checkpoint diagnostics

### Methods and limitations

The diagnostics replayed `142.75_mm-skipsig` under clean provenance commit `0300a358`. The flattened
WikiText-103 validation cache contained 245,021 tokens. Window starts were
`round(linspace(0, n_tokens - N - 1, n_windows))`, giving endpoints 0 and 244,892 for $N=128$.
Sixty-four windows were used for D1 and D2; across-window SD values are sample SDs. The environment
available to this investigation exposed CPU PyTorch but not CUDA, so no RTX execution is claimed.
The saved run was trained on an RTX 5090 and records a dirty tree. On 512 held-out windows selected by
the same rule, the clean replay's live mean CE was 4.97582, corresponding to PPL 144.87, near the
saved validation PPL 143.25. This numerical agreement supports using the replay as a diagnostic
witness, but it is not an exact reproduction of the dirty run.

D1 computed gamma from the raw $s^{(0)}$ table, matching the checkpoint-era fold, and beta from the
actual refined $q^{(0)}=s^{(1)}$ state at the one mm-exact belief update. Existing post-update
attention figures were not used because they recompute beta at converged $q^{(1)}$. D2 used a
chain-rule vector-Jacobian decomposition from one captured forward. If $L$ is CE, then

$$
g_s=\left(\frac{\partial s^{(1)}}{\partial\phi}\right)^\top
\frac{\partial L}{\partial s^{(1)}},
\qquad
g_q=\frac{\partial L}{\partial\phi}-g_s.
$$

This separates the model-routed and direct belief-stack gradients exactly for the declared graph
cut. The VJP includes both $s_\mu$ and $s_\Sigma$, and repeated token-position gradients are summed
onto unique active `phi_embed` rows before computing each cosine. Two hooks on the `phi_embed` leaf
cannot separate the routes because autograd calls each with the same accumulated gradient.

### D1: related channels, but not identical near-replay distributions

| Comparison over 64 sequences | Mean | SD across sequences | Range |
|---|---:|---:|---:|
| TV$(\gamma,\beta_{\mathrm{used}})$ | 0.17411 | 0.00628 | 0.15941–0.19120 |
| JS$(\gamma,\beta_{\mathrm{used}})$, nats | 0.02372 | 0.00156 | 0.02030–0.02808 |
| Gamma/beta argmax agreement | 0.93225 | 0.01702 | 0.88281–0.96875 |
| TV between raw-$s$ gamma and refined-$s$ beta under the same static prior | 0.02067 | 0.00172 | 0.01748–0.02726 |

The transported energy geometries were close: changing only raw $s^{(0)}$ to refined $s^{(1)}$
under the same prior moved TV by about 0.021. Precision bias and the hierarchical prior fold account
for most of the approximately 0.174 difference between gamma and the beta actually used by the
belief update. The high argmax agreement supports the memo's claim of related relational content,
but it does not establish that forced decorrelation would improve prediction. Redundancy can be
regularizing, and two distributions can have the same top key while carrying different useful mass.

### D2: the endpoint gradients are cooperative

| Quantity over 64 sequences | Mean | SD | Range |
|---|---:|---:|---:|
| Cosine$(g_s,g_q)$ on active `phi_embed` rows | 0.86448 | 0.03171 | 0.66010–0.89828 |
| $\lVert g_s\rVert/\lVert g_q\rVert$ | 0.65525 | 0.02463 | 0.59052–0.73613 |
| Fraction with negative cosine | 0 | not applicable | 0 of 64 |

This result rejects the clean-commit near-replay endpoint version of the memo's gradient-conflict
mechanism. It does not prove that sharing is always beneficial: gradients can change over training,
and local alignment does not determine held-out generalization. The confirmatory D2 should sample
early, middle, and late checkpoints, report Adam-preconditioned as well as raw alignment, and measure
one-step cross-loss effects. The current evidence provides no reason to pay 10.05 million parameters
merely to separate the two endpoint gradient streams.

### Frozen E0 sensitivity: the checkpoint tensors use gamma in the near-replay

The following paired intervention used 512 evenly spaced validation windows, or 65,536 predicted
tokens, under the provenance execution graph. Percentile intervals use 20,000 shared bootstrap index
rows from NumPy `default_rng(20260711)`, resampling the 512 paired sequence means with replacement;
the reported bounds are the linear 2.5th and 97.5th percentiles.

| Frozen arm | Mean CE | PPL | Paired $\Delta$CE versus live | PPL ratio versus live |
|---|---:|---:|---:|---:|
| Live | 4.97582 | 144.87 | 0 | 1.000 |
| Fold off, gamma consensus retained | 4.99456 | 147.61 | +0.01874, 95% bootstrap CI [0.01736, 0.02015] | 1.0189 |
| Gamma consensus and fold off | 5.19659 | 180.65 | +0.22076, 95% bootstrap CI [0.21312, 0.22861] | 1.2470 |

These are frozen-weight interventions, not retrained ablations. They show that the loaded checkpoint
tensors use gamma consensus strongly and the fold modestly in the clean-commit near-replay; they do
not estimate how well each architecture would learn after retuning. They nevertheless rule out
treating the existing gamma route as functionally inert under that replay. A retrained E0 remains
the missing causal denominator.

## Parameter and control audit

The full independent table contains $50{,}257\times200=10{,}051{,}400$ parameters. Adding it to
the checkpoint gives exactly 25,204,402 parameters, a 66.33% increase; the two frame tables would
then contain 79.76% of all parameters. In float32, the new parameter consumes 40.2 MB, Adam's two
moments consume another 80.4 MB, and its gradient consumes another 40.2 MB. The approximately
120 MB estimate in the memo counts the parameter and Adam moments but not the training gradient or
temporary workspace.

The proposed $K=27,H=2$ width control is invalid because `VFE3Config` requires
`embed_dim % n_heads == 0` at `vfe3/config.py:877-879`. Exact current-config parameter counts are
23,613,854 for $K=26,H=2$ and 26,837,218 for $K=28,H=2$, bracketing the 25,204,402-parameter
independent-frame model by approximately 6.3% below and 6.5% above. Both should be reported if width
is used as a parameter control.

The memo's width fit was reproduced from all 36 `grow_K_GL10` records:

$$
\operatorname{PPL}(K)=1728.0815K^{-1.0489056}+63.96197.
$$

Its numerical prediction at $K=27$ is 118.44, but the sweep changes the number of heads to retain
$d_{\mathrm{head}}=10$, mixes development commits, and trains under a different schedule. It does
not define a realizable $K=27,H=2$ control or a calibrated 20-PPL adoption threshold. Parameter,
fixed-token, fixed-compute, and wall-time comparisons should be reported separately because a wider
belief channel and a second transport spend computation differently.

## External evidence and statistical design

The closest external evidence supports flexible sharing as an experimental question, not a positive
performance prior. Cross-stitch networks learned mixtures of shared and task-specific representations
in particular vision tasks ([Misra et al., CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html)).
Task cooperation and competition also vary by problem, so grouping compatible tasks can outperform
both naive hard sharing and full separation ([Standley et al., ICML 2020](https://proceedings.mlr.press/v119/standley20a.html)).
These models have distinct supervised tasks, whereas beta and gamma are coupled inference channels
serving one language-modeling objective. Their results justify testing soft sharing but do not predict
the sign of an independent $\widetilde\Omega$ effect.

Gradient cosine is a valid local optimization diagnostic. PCGrad defines negative cosine as one
component of harmful interference and reports gains on selected multitask benchmarks
([Yu et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)).
ForkMerge finds that gradient conflict and negative transfer need not correlate and that conflicting
updates can act as regularization
([Jiang et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html)).
D2 can therefore identify a local mechanism but cannot determine generalization by itself.

The proposed non-overlapping mean $\pm1$ SD rule over three seeds is not a calibrated statistical
test. SD measures run dispersion, not uncertainty in a mean, and three runs provide a poor estimate
of either. Benchmark variation includes initialization, data order, and hyperparameter selection
([Bouthillier et al., MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)).
Compute-controlled evaluations can also reverse apparent training-efficiency gains
([Kaddour et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/51f3d6252706100325ddc435ba0ade0e-Abstract-Conference.html)).

The primary endpoint should be held-out token-mean CE, with PPL reported as $\exp(\mathrm{CE})$.
The project should preregister the smallest worthwhile CE reduction $\delta_{\mathrm{CE}}$, estimate
paired seed-difference variance from a pilot, and choose the seed count for 80–90% power. Compatible
arms should share seeds, data order, and stochastic streams. A final E3 adoption claim is an
intersection-union claim: the one-sided upper confidence bound for E3 minus the retrained E0 winner
and the corresponding bound for E3 minus the resource control must both be below
$-\delta_{\mathrm{CE}}$. Model selection through E1/E2 requires fresh confirmatory seeds or an
explicit multiplicity adjustment.

## Revised experiment order

| Gate | Experiment | What it can establish | Required correction |
|---|---|---|---|
| G0 | Bind one execution graph | Reproducible baseline semantics | Choose checkpoint-era gradient-$s$/mm-$q$ behavior or current shared mm-exact behavior. Save a self-bound checkpoint and clean Git provenance. |
| G1 | Retrained E0 | Marginal value of gamma consensus and the fold | The current reachable arms are live, fold-off, and `lambda_gamma=0` plus fold-off. `lambda_gamma=0` with the fold left on is rejected by validation. A full two-factor design needs a separate gamma-availability gate. |
| G2 | Shared-forward gradient control | Whether separating gradients helps without changing geometry or capacity | Keep the tied forward transport and stop only the model-routed gradient into $\phi$. Compare with live at matched initialization and compute. |
| G3 | Typed E1 null | Whether implementation preserves representation equivalence | Implement $M\Omega M^{-1}$ together with $C=M^{-1}$ and require pure-path parity. An identity-bridge variant tests a relative coordinate map, not new invariant geometry. |
| G4 | Covariant partial-private arm | Whether limited relative transport flexibility helps beyond generic capacity | Use a declared group-level relative field, test its transformation law, and compare with a matched $V\times r$ nontransport table. Do not call an arbitrary low-rank algebra span covariant. |
| G5 | Full private arm | Whether a genuinely independent or explicitly same-gauge private transport clears all controls | Copy-initialize the effective group frame, fix the new gauge orbit, disable shared transport reuse, include the cross-fiber bridge if gauges are independent, and compare against retrained E0 plus valid parameter and compute controls. |

E1 and E2 are not a monotone dose series as written: conjugation is an equivalent global
representation, while the low-rank offset is token-specific. Evidence across them may be convergent,
but their ranks do not define one common scalar of model-frame independence.

## Final recommendation

Retain the shared frame for now. The strongest pro-separation argument in the memo was a combination
of redundant relational content and harmful gradient cross-talk. The clean-commit near-replay
supports the first only in a limited sense and rejects the second at the measured endpoint. The
channel is also functionally important to the replayed predictor. Against that evidence, the full
table has a large parameter cost, an invalid proposed width control, an extra gauge orbit, a
non-nested initialization, and a missing cross-fiber map.

The highest-value next result is a clean, retrained E0 comparison followed by the tied-forward
gradient-stop control. If those show that gamma content helps while model-routed frame gradients hurt,
a private transport becomes mechanistically motivated. If the gradient-stop control does not help,
the project should test only low-cost same-gauge flexibility before considering a full table.

The findings are substantial enough to add to the Research vault after explicit approval. The most
appropriate destinations are `[[VFE Transformer Program]]`, `[[GL(K) gauge group]]`, and the current
manuscript revision record, with the checkpoint-version split retained as provenance rather than
presented as a timeless property of the architecture.
