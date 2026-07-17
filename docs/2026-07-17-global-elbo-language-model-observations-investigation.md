# Global ELBO and Language-Model Observations in the Gauge-Theoretic VFE Transformer

**Date:** 2026-07-17
**Scope:** The ELBO question only. This report does not analyze training duration, plateau speed, or the comparative quality of the baseline run.

## Conclusion

The original motivation is sound: language-model training tokens can be observations in a variational generative model. The observation term is then a normalized categorical likelihood, and its negative log likelihood is cross-entropy. Cross-entropy is therefore not foreign to variational free energy. It is the accuracy, or expected negative-log-likelihood, sector of the VFE for discrete observations.

The present implementation does something different. Its model-channel refinement and belief refinement are target-blind computations over the prefix, after which a mean-only categorical decoder predicts the held-out next token. In the investigated baseline, the scored outer loss is exactly cross-entropy. The two inner refinements shape the state through which cross-entropy is differentiated, but they are not posterior-coordinate updates for the held-out observation, and the decoder update is not a fixed-belief M-step on the same scalar.

The full theoretical VFE is much more than the peer term \(D_{\mathrm{KL}}(q_i\Vert\Omega_{ij}q_j)\). It includes belief-prior complexity, model-hyperprior complexity, belief and model consensus channels, the two categorical attention entropies, gauge transport, adaptive precision, and an observation likelihood. Most of those sectors admit a probabilistic role. The problem is not that the complete functional is “only pairwise KL.” The problem is that the complete functional is presently an energy inventory over node-wise belief distributions, not a fixed normalized joint model with a joint variational posterior.

The existing mean-field obstruction is valid but too narrow to settle the question. Mean-field can miss correlations that matter. A correlated state-level ELBO remains possible, but it must introduce the missing correlation-bearing objects: a normalized joint \(Q\), cross-token covariance or edge beliefs, joint entropy or mutual-information corrections, and marginal-consistency constraints. Merely calling the existing marginal functional “structured” does not supply those objects.

Two honest global constructions are available. A structured causal state-space ELBO is the stronger route if the intended claim is a conventional state-level generative model of language. It retains the token likelihood, represents correlations explicitly, and replaces the moving posterior-to-posterior peer KLs by fixed normalized transition factors. The belief-configuration Gibbs lift already proved in `PIFB2.tex` is the stronger route if the goal is to preserve every non-observation sector of the current population VFE as a configuration energy and attach the observation likelihood once. That lift is exact at the level of random belief configurations, but it is not a state-level ELBO and it requires a configuration entropy, a finite partition function, and gauge-volume control.

The observation likelihood cannot be removed while retaining an autoregressive probability model over tokens. It can be reparameterized as a Gaussian-template or other energy-based categorical emission, but its normalized negative log likelihood is still a cross-entropy over vocabulary alternatives. An observation-free VFE can regularize or organize internal beliefs; by itself it cannot assign normalized probabilities to the training tokens or define perplexity.

## 1. What “the token is an observation” means in a language model

For an autoregressive sequence \(x_{1:T}\), a proper language model has the causal factorization

\[
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
\]

With a latent state \(z_t\), the model may define a predictive prior from the prefix and an emission from that latent state,

\[
p_\theta(x_t\mid x_{<t})
=\int p_\theta(x_t\mid z_t,x_{<t})p_\theta(z_t\mid x_{<t})dz_t.
\]

During training, a recognition distribution is allowed to see the observation it is inferring,

\[
Q_\psi(z_t\mid x_{\leq t}),
\]

while the predictive prior is not. A per-token conditional ELBO is then

\[
\mathcal L_t
=\mathbb E_{Q_\psi}\left[\log p_\theta(x_t\mid z_t,x_{<t})\right]
-D_{\mathrm{KL}}\left(
Q_\psi(z_t\mid x_{\leq t})
\middle\Vert
p_\theta(z_t\mid x_{<t})
\right).
\]

This is not target leakage. The posterior sees \(x_t\) only during inference at training time. Generation and next-token evaluation use the predictive prior, which sees only \(x_{<t}\). This prior/posterior distinction is the standard organization used by sequential latent-variable models such as the [VRNN](https://arxiv.org/abs/1506.02216), the [SRNN](https://arxiv.org/abs/1605.07571), and structured nonlinear state-space inference [Krishnan, Shalit, and Sontag](https://arxiv.org/abs/1609.09869).

The current VFE transformer uses another indexing. The state at position \(i\) is initialized from the already-observed \(x_i\), may use \(x_{\leq i}\), and predicts \(x_{i+1}\). The manuscript states this explicitly in [`GL(K)_attention.tex`](../Manuscripts-Theory/GL(K)_attention.tex) at lines 959-967. Its \(q_i\) is therefore a target-blind predictive representation for \(x_{i+1}\), not a posterior that assimilates \(x_{i+1}\). That representation is valid for a deterministic conditional language model. As presently updated, it is not the E-coordinate of the per-token ELBO above because its update omits the expected observation likelihood.

Target blindness by itself does not invalidate an ELBO. Any normalized \(Q(z_t\mid x_{<t})\), even one that excludes \(x_t\), gives a mathematically valid but potentially loose lower bound. It can also serve as a restricted amortized recognition family if its parameters are trained on that same bound. The active mismatch is between objectives: the inner maps optimize target-free consensus surrogates, while the outer step scores point-estimate CE. The mismatch, rather than the absence of the target from the recognition function's inputs alone, defeats the coordinate-ELBO interpretation.

There are two coherent ways to restore the observation semantics. One can reindex the latent so that \(q_t\) is inferred after observing \(x_t\), with a separate prior state used to predict \(x_t\). Alternatively, one can keep the current predictive state at \(i\) and introduce a distinct training posterior \(Q(z_{i+1}\mid x_{\leq i+1})\) that is regularized toward the target-blind prior \(P(z_{i+1}\mid x_{\leq i})\). Both require distinct predictive-prior and observation-conditioned posterior objects. The current single target-blind state does not provide that split.

## 2. The full VFE sector by sector

The canonical pointwise functional in [`PIFB2.tex`](../Manuscripts-Theory/PIFB2.tex) at lines 661-711 contains the belief-prior, model-hyperprior, belief-consensus, model-consensus, attention-entropy, and observation sectors. The transformer manuscript adds adaptive precision \(\alpha_i\) and its regularizer. Writing the combined inventory schematically gives

\[
\begin{aligned}
\mathcal F_{\mathrm{full}}
={}&\sum_i\left[\alpha_iD_{\mathrm{KL}}(q_i\Vert p_i)+R(\alpha_i)\right]
+\lambda_h\sum_iD_{\mathrm{KL}}(s_i\Vert r_i)\\
&+\sum_{ij}\beta_{ij}D_{\mathrm{KL}}(q_i\Vert\Omega_{ij}q_j)
+\tau_q\sum_{ij}\beta_{ij}\log\frac{\beta_{ij}}{\pi_{ij}}\\
&+\sum_{ij}\gamma_{ij}D_{\mathrm{KL}}(s_i\Vert\widetilde\Omega_{ij}s_j)
+\tau_s\sum_{ij}\gamma_{ij}\log\frac{\gamma_{ij}}{\pi^{(s)}_{ij}}\\
&-\sum_i\mathbb E\left[\log p_\theta(o_i\mid k_i,m_i)\right]
+\mathcal F_{\mathrm{optional}}
\end{aligned}.
\]

Here \(\mathcal F_{\mathrm{optional}}\) may include configured frame penalties, connection or curvature terms, and noncanonical experimental regularizers. The transports depend on the gauge frames even when there is no separate frame penalty.

| Sector | Available probabilistic role | Boundary in the present formulation |
|---|---|---|
| \(D_{\mathrm{KL}}(q_i\Vert p_i)\) | Ordinary posterior-prior complexity if \(p_i\) is a fixed generative conditional and \(q_i\) is a posterior factor or marginal. | In a correlated \(Q\), the global entropy is not the sum of node entropies. The node KL sum alone omits total correlation. |
| \(\alpha_iD_i+R(\alpha_i)\) | \(R(\alpha)=b_0\alpha-c_0\log\alpha\) is a Gamma MAP penalty for a positive precision. | A full Bayesian ELBO over uncertain \(\alpha\) also needs a variational distribution and its entropy. A general coefficient on a KL is tempered or generalized unless derived from an explicit model. |
| \(\lambda_hD_{\mathrm{KL}}(s_i\Vert r_i)\) | Model-state posterior versus hyperprior complexity. | The coefficient and the relation between \(s_i\) and the state channel must be generated by a fixed joint, rather than assigned only as a loss weight. |
| \(\beta_{ij}D_{\mathrm{KL}}(q_i\Vert\Omega_{ij}q_j)\) | Exact row-wise source-selection divergence when the transported source templates are frozen. It also has a coherent role as a consensus penalty between local replicas of a shared belief. | With all \(q_j\) live, a variational posterior appears inside what would have to be the generative reference distribution. This is an engineered consensus energy, not yet a fixed-joint transition factor. |
| \(\tau_qD_{\mathrm{KL}}(\beta_i\Vert\pi_i)\) | Categorical source-posterior complexity. | Unit relative weighting follows an ordinary auxiliary-variable ELBO. Arbitrary temperature is a tempered or generalized construction unless the whole model is rescaled with its normalizers. |
| Model-channel \(\gamma\) coupling and entropy | The same source-selection construction can be used for latent model states. | It inherits the moving-source problem and also needs a normalized bridge from the model state \(m_i\) to the belief state \(k_i\). |
| Observation term | Expected log likelihood. For tokens, a categorical emission produces cross-entropy. | The expectation must be under the relevant joint uncertainty. The canonical PIFB2 display contains \(m_i\) in the likelihood but integrates only under \(q_i\), not a joint \(Q_i(k_i,m_i)\). |
| Gauge frame and transport | \(\phi\) can be a deterministic model parameter inside normalized transitions, or a latent variable with a proper prior and posterior. Gauge-covariant quadratic transitions are natural. | A latent-frame ELBO needs frame entropy and a proper reference measure. Noncompact \(\mathrm{GL}^+\) volume requires a quotient, gauge fixing, compact regulator, or coercive prior. |
| Holonomy, curvature, and mass | Curvature or Wilson-type energies can define priors on connection variables. A kinetic term can enter a trajectory or phase-space model. | Frame-derived Regime I, \(\Omega_{ij}=U_iU_j^{-1}\), is pure gauge, with identity loop holonomy and zero curvature. Nonzero Wilson or Yang-Mills sectors require the independent Regime-II connection. These are not automatically likelihood or static ELBO terms, and the manuscript's mass reading requires an additional dynamical postulate. |

There is also a cross-channel specification gap. `PIFB2.tex` says the canonical same-scale functional does not contain a cross-bundle morphism that makes \(s_i\) generate \(p_i\), while [`GL(K)_supplementary.tex`](../Manuscripts-Theory/GL(K)_supplementary.tex) at lines 876-895 proposes

\[
p_i(k_i)=\int p_i(k_i\mid m_i)s_i(m_i)dm_i.
\]

The latter is the beginning of the required bridge, but the two manuscripts do not yet present one common fixed joint that uses it. The retained executable assignment \(q_i^{(0)}=p_i=s_i^{(1)}\) is an update-map handoff, not a normalized conditional distribution \(p(k_i\mid m_i)\).

## 3. Why mean-field was tested, and why it is not the final boundary

The state-level theorem in `PIFB2.tex` at lines 3273-3324 asks whether the current population functional can equal

\[
D_{\mathrm{KL}}\left(\prod_iq_i\middle\Vert p_\theta\right)+c,
\]

for one fixed \(q\)-independent joint. It answers no under stated nondegeneracy conditions when a live nonself peer coupling is present. The theorem is scoped to an open product family, independent factor variation, and fixed attention rows during the variation. It does not rule out a structured posterior, auxiliary variables, a restricted Gaussian family, or the configuration-space construction.

Mean-field was the natural first test because the current theory and code expose one Gaussian \(q_i\) and one Gaussian \(s_i\) per token, and the stated VFE is a sum of functions of those node distributions. That is not a reason to prefer mean-field as physics. It is a statement about the variables the current functional actually contains.

Dropping mean-field creates a second, different representability test. Consider two discrete latent states and a strictly positive interior joint posterior \(Q_0\), so every cell remains positive under a sufficiently small two-sided perturbation,

\[
Q_\epsilon
=Q_0+\epsilon
\begin{pmatrix}
1 & -1\\
-1 & 1
\end{pmatrix}.
\]

For sufficiently small \(\epsilon\), this changes only the correlation; both node marginals remain fixed. Every sector of the current node-marginal VFE remains fixed along this correlation fiber. For any fixed positive joint model \(P\), however,

\[
\frac{d^2}{d\epsilon^2}D_{\mathrm{KL}}(Q_\epsilon\Vert P)
=\frac{1}{Q_{\epsilon,00}}+\frac{1}{Q_{\epsilon,01}}
+\frac{1}{Q_{\epsilon,10}}+\frac{1}{Q_{\epsilon,11}}>0.
\]

The energy expectation under fixed \(P\) is linear in \(Q\); the curvature comes from the joint entropy. Therefore the existing marginal-only functional cannot equal the ELBO of an open structured family that has freely variable correlations. This is not a universal structured-posterior no-go. It says that adding correlations requires adding the terms and state variables that carry correlations.

The same fact appears in the total-correlation identity. If \(q_i\) are the marginals of a joint \(Q\) and the prior factorizes, then

\[
D_{\mathrm{KL}}\left(Q\middle\Vert\prod_ip_i\right)
=\sum_iD_{\mathrm{KL}}(q_i\Vert p_i)
+D_{\mathrm{KL}}\left(Q\middle\Vert\prod_iq_i\right).
\]

The second term is the total correlation. On a tree-structured Markov posterior it can be written using edge mutual informations. Bethe and Kikuchi methods generalize this idea using node and region beliefs, counting numbers, and marginal-consistency constraints. These are the statistical-physics corrections that naive mean-field omits. [Wainwright and Jordan](https://www.cs.columbia.edu/~blei/fogm/2020F/readings/WainwrightJordan2008.pdf) give the general variational geometry, while [Yedidia, Freeman, and Weiss](https://www.merl.com/publications/TR2004-040) derive region free energies and generalized belief propagation.

The current executable state does not contain those variables. [`vfe3/belief.py`](../vfe3/belief.py) stores per-token means and within-token diagonal or full covariance. It does not store cross-token covariance blocks, pair marginals \(q_{ij}\), cavity messages, or marginal-consistency constraints. The \((N,N)\) energy arrays and \(\beta/\gamma\) rows are routing scores; they are not pairwise latent marginals. The “full covariance” family is full within one token, not full over the token population.

## 4. Exact constructions already present in the theory

### 4.1 The augmented hierarchical Gaussian joint

`PIFB2.tex` at lines 3159-3220 defines a proper tree-structured joint with normalized Gaussian parent-child conditionals,

\[
p_\sigma(k,o)
=p(o\mid k^{(0)})
\prod_{(i,\operatorname{pa}(i))}
\mathcal N\left(k_i;\Omega_{i,\operatorname{pa}(i)}k_{\operatorname{pa}(i)},\sigma^2I\right)
p(k_{\mathrm{top}}).
\]

The manuscript subsequently applies a product approximation, but the joint itself does not require mean-field. A structured Gaussian posterior, including the exact Gaussian posterior in the Gaussian-likelihood regime, can be used on the same joint. At zero raw within-scale \(\beta/\gamma\) coupling, the stacked precision is positive definite under the stated top-prior assumptions and the state-level generative interpretation is exact.

This is the most developed starting point for a structured state-level ELBO. What it does not yet provide is the live within-scale attention construction or a language-specific causal state transition.

### 4.2 Frozen auxiliary source rows

For a fixed set of transported source templates, one source-choice row has a proper augmented interpretation. A categorical source variable has prior \(\pi_i\), posterior \(\beta_i\), and conditional component density \(T_{ij}q_j\). At unit temperature, the augmented row KL decomposes into the weighted component KL plus \(D_{\mathrm{KL}}(\beta_i\Vert\pi_i)\).

This identity is exact after the sources are frozen. It does not automatically define a shared joint over all live agent states because every row reuses other variational posteriors as component densities. Compatibility, acyclic factorization, or a normalized global potential is still required.

### 4.3 The belief-configuration Gibbs lift

The theorem at `PIFB2.tex` lines 3413-3466 takes the full belief configuration

\[
X=(q,s,\alpha,\beta,\gamma,\phi,\ldots).
\]

as the random variable. Let \(c\) denote the observed prefix or other declared context, let \(y\) denote the held-out token observations, let \(\rho_0\) be a proper reference probability, and let \(\mathcal F_{\mathrm{vac}}(X;c)\) exclude the token observation term. The context may enter this energy, but the held-out \(y\) may not. When

\[
0<Z_{\mathcal F}(c)
=\int\exp\left[-\mathcal F_{\mathrm{vac}}(X;c)/T_{\mathrm{cfg}}\right]
d\rho_0(X)<\infty,
\]

the prior

\[
dP_{\mathcal F}(X\mid c)
=Z_{\mathcal F}(c)^{-1}
\exp\left[-\mathcal F_{\mathrm{vac}}(X;c)/T_{\mathrm{cfg}}\right]d\rho_0(X).
\]

is normalized. To retain the canonical expected observation term, augment the generative model with \(k_i\mid X,c\sim q_{i,X}\) and \(y_i\mid k_i,c\sim p_\theta(y_i\mid k_i,c)\), and use the tied conditional \(q_{i,X}\) for \(k_i\) in the variational joint. For any correlated \(R(dX\mid c,y)\ll\rho_0\) for which all displayed terms are finite, its negative ELBO is

\[
\begin{aligned}
-\mathcal L_{\mathrm{cfg}}
={}&\mathbb E_R\mathbb E_{q_X}[-\log p_\theta(y\mid k,c)]
+\frac{1}{T_{\mathrm{cfg}}}\mathbb E_R[\mathcal F_{\mathrm{vac}}(X;c)]\\
&+D_{\mathrm{KL}}(R\Vert\rho_0)+\log Z_{\mathcal F}(c)
\end{aligned}.
\]

Equivalently,

\[
\begin{aligned}
-T_{\mathrm{cfg}}\mathcal L_{\mathrm{cfg}}
={}&\mathbb E_R[\mathcal F_{\mathrm{vac}}(X;c)]
+T_{\mathrm{cfg}}\mathbb E_R\mathbb E_{q_X}[-\log p_\theta(y\mid k,c)]\\
&+T_{\mathrm{cfg}}D_{\mathrm{KL}}(R\Vert\rho_0)
+T_{\mathrm{cfg}}\log Z_{\mathcal F}(c)
\end{aligned}.
\]

At \(T_{\mathrm{cfg}}=1\), the non-observation energy and canonical expected observation NLL have their original relative scale. At other temperatures, the observation term is scaled in the temperature-multiplied identity. A different normalized \(p_\theta(y\mid X,c)\) can be attached instead, but that defines a replacement observation sector rather than reproducing the canonical expectation.

This construction preserves every non-observation sector of the current VFE as a prior energy and can reproduce its observation sector through the explicit \(k_i\) draw. It also makes the missing terms visible. A point estimate of \(X\) is a MAP or zero-configuration-temperature approximation; it is not the finite-temperature ELBO because it drops the configuration entropy. If parameters inside \(\mathcal F_{\mathrm{vac}}\) are learned, \(\log Z_{\mathcal F}(c)\) is generally parameter-dependent and its gradient cannot be discarded. A proper gauge reference or quotient is required because raw invariant volume on \(\mathrm{GL}^+\) is noncompact.

This is a genuine answer already present in the theory, but at the meta-level: the random object is a configuration of beliefs and models, not merely the latent token state.

## 5. Recommended state-level construction: a structured causal ELBO

If the scientific claim is that the model performs variational inference over language-generating latent states, the cleanest route is to define one fixed causal joint and let the variational posterior retain correlations.

Let \(z_t\) be the token-state latent, \(m_t\) the model-channel latent, \(a_t\) and \(b_t\) the belief- and model-source choices, and \(\phi\) the gauge connection or frame variables. Restrict source choices to earlier positions or to a declared hierarchy so the graph is acyclic. One possible normalized joint is

\[
\begin{aligned}
p_\theta(x,z,m,a,b,\phi)
={}&p_\theta(\phi)p_\theta(z_0,m_0)\\
&\times\prod_{t=1}^{T}
\pi_t^{(s)}(b_t)
p_\theta(m_t\mid m_{b_t},\phi)
\pi_t^{(q)}(a_t)
p_\theta(z_t\mid z_{a_t},m_t,\phi)
p_\theta(x_t\mid z_t,m_t)
\end{aligned}.
\]

For example, the state transition can be a normalized gauge-covariant Gaussian,

\[
p_\theta(z_t\mid z_j,m_t,\phi)
=\mathcal N\left(z_t;\Omega_{tj}(\phi)f_\theta(z_j,m_t),R_t\right),
\]

with positive-definite \(R_t\). The project does not require a neural \(f_\theta\); it can be an identity, registered linear morphism, or another explicit family consistent with the no-neural-network constraint.

A structured filtering posterior may factor conditionally without factorizing the states marginally,

\[
Q_\psi(z,m,a,b,\phi\mid x)
=Q_\psi(\phi\mid x)
\prod_tQ_\psi(b_t,m_t,a_t,z_t\mid z_{<t},m_{<t},x_{\leq t},\phi).
\]

A smoothing posterior may condition on \(x_{1:T}\). Either choice represents dependencies across tokens. The ELBO is simply

\[
\mathcal L_{\mathrm{structured}}
=\mathbb E_Q\left[\sum_t\log p_\theta(x_t\mid z_t,m_t)\right]
-D_{\mathrm{KL}}\left(
Q(z,m,a,b,\phi\mid x)
\middle\Vert
P_\theta(z,m,a,b,\phi)
\right).
\]

The KL chain rule decomposes this global complexity into expected conditional KLs while retaining correlations. The categorical source variables yield terms of the form \(D_{\mathrm{KL}}(Q(a_t\mid\cdot)\Vert\pi_t^{(q)})\) and the analogous model-channel term, so \(\beta\) and \(\gamma\) can retain a precise source-posterior role.

What changes is the alignment energy. A fixed generative transition produces an expected pair energy under the joint edge marginal,

\[
\mathbb E_{Q(z_t,z_j)}
\left[
\frac{1}{2}
\left\|z_t-\Omega_{tj}z_j\right\|_{R_t^{-1}}^2
\right]
+\frac{1}{2}\log\det(2\pi R_t),
\]

together with the joint entropy. The log-determinant normalization can be discarded as a constant only when \(R_t\) is fixed; it contributes to learning when the transition covariance is trained. This transition contribution does not produce \(D_{\mathrm{KL}}(q_t\Vert\Omega_{tj}q_j)\) exactly, because the latter uses the moving variational marginal \(q_j\) as the covariance-bearing reference distribution. The existing peer KL must therefore be replaced by the derived transition term, retained as an explicitly non-ELBO consensus regularizer, or moved to the configuration-level prior.

For Gaussian latent dynamics, a practical structured family is a joint Gaussian with sparse block precision. Cross-token precision blocks encode the dependencies that the present model omits. Gaussian message passing or sparse linear algebra can optimize the Gaussian part. The categorical token likelihood is nonconjugate. Exact quadrature or an unbiased Monte Carlo estimator can evaluate its ELBO expectation, while a certified lower bound on that expectation gives a looser valid bound. Laplace and ordinary expectation propagation can be useful posterior approximations, but they do not generally preserve an evidence-lower-bound guarantee and must not be labeled ELBO merely because they optimize a local approximation.

A Bethe or Kikuchi version would instead maintain node and edge beliefs \(b_i,b_{ij}\), impose

\[
\int b_{ij}(z_i,z_j)dz_j=b_i(z_i),
\qquad
\int b_{ij}(z_i,z_j)dz_i=b_j(z_j),
\]

and add the corresponding region entropy. It is exact on a tree. On a loopy attention graph its stationary points generally optimize an approximate region free energy, not a guaranteed evidence lower bound. It should be called an ELBO only if an independent bound construction proves that property for the selected regions and counting numbers.

## 6. A second interpretation of the peer KL: distributed consensus around an ELBO

There is another way to preserve the scientific meaning of \(D_{\mathrm{KL}}(q_i\Vert\Omega_{ij}q_j)\) without pretending it is a generative transition. The \(q_i\) may be local, gauge-related replicas of one shared posterior over a common cause. One can then formulate a constrained distributed variational problem,

\[
\max_{q_1,\ldots,q_N}\sum_i\mathcal L_i(q_i)
\quad\text{subject to}\quad
q_i=\Omega_{ij}q_j,
\]

with the observation likelihood partitioned among the local objectives and the global prior and entropy counted exactly once. Bregman-ADMM or related methods may add dual pairings and KL consensus penalties to solve this constrained ELBO.

This interpretation fits the consensus semantics better than forcing every peer belief into a fixed joint density. It does not make the finite-penalty augmented objective itself a model ELBO. At exact consensus the penalty vanishes and the underlying evidence bound is recovered. Away from consensus it is an algorithmic penalty. The current functional contains no dual variables, the attention weights are finite adaptive row weights, and the target likelihood is absent from the local inner problems, so the current update is not an ADMM derivation of token evidence.

A weaker but valid numerical statement is possible: subtracting a nonnegative consensus penalty from an already valid ELBO leaves a lower number and therefore a looser lower bound. That is an ELBO plus regularization, not an identity with the original model evidence.

## 7. What can supplement cross-entropy now

### 7.1 The smallest conventional conditional ELBO

The existing refinements can be treated as a recognition algorithm without claiming that each refinement is an ELBO coordinate. Let \(c_i=x_{\leq i}\) and \(y_i=x_{i+1}\). The refined model channel must first induce a state-channel predictive prior through an explicit normalized bridge,

\[
P_\theta(z_i\mid c_i)
=\int p_\theta(z_i\mid m_i,c_i)s_i^{(1)}(dm_i\mid c_i).
\]

An identity or pushforward bridge is possible only after the model and state fibers are explicitly identified. Let the belief refinement return a normalized Gaussian \(Q_\psi(z_i\mid c_i)\). The model can then score

\[
\mathcal J_{\mathrm{cond}}
=\sum_i\left[
\mathbb E_{Q_\psi(z_i\mid c_i)}
[-\log p_\theta(y_i\mid z_i,c_i)]
+D_{\mathrm{KL}}\left(
Q_\psi(z_i\mid c_i)
\middle\Vert
P_\theta(z_i\mid c_i)
\right)
\right].
\]

Its negative is a valid lower bound on \(\sum_i\log p_\theta(y_i\mid c_i)\) for any normalized \(Q_\psi\), even though \(Q_\psi\) is target-blind. The cost of target blindness is a looser bound and a weaker approximation to the true posterior, not invalidity of the inequality.

This construction is the smallest ordinary ELBO available near the present architecture. It adds a genuine complexity term to the expected token NLL and can be differentiated through the finite refinement maps. The peer and model consensus calculations then define the inference architecture that produces \(Q_\psi\) and \(P_\theta\); they are not all counted again as separate ELBO sectors. It does not make the inner maps coordinate updates, and it does not preserve the complete population VFE as the state-level KL.

To implement this objective faithfully, the decoder must evaluate or bound the expectation under \(Q_\psi\), not only CE at its mean. The prior \(P_\theta\) must also be treated as a normalized conditional rather than merely a copied tuple. If the model channel is itself uncertain, this smaller construction must be extended to integrate over it.

### 7.2 The complete VFE as a generalized regularizer

The immediately available objective is a generalized or regularized predictive loss,

\[
\mathcal J_{\mathrm{reg}}
=\mathcal L_{\mathrm{CE}}
+\lambda_q\mathcal F_q
+\lambda_s\mathcal F_s
+\lambda_\phi\mathcal F_\phi.
\]

This can use the entire canonical VFE inventory and can be trained as one differentiable scalar. It has a coherent loss-based generalized-Bayesian interpretation only when it defines an explicitly normalized update with a finite, nonzero normalizer; [Bissiri, Holmes, and Walker](https://arxiv.org/abs/1306.6430) provide the relevant framework. Without that construction, \(\mathcal J_{\mathrm{reg}}\) is regularized empirical risk. It is not a conventional likelihood ELBO merely because its regularizers are called free energy.

If \(q\) is treated as a genuine Gaussian latent posterior, the observation term should be

\[
\mathbb E_{z\sim q}
\left[-\log\operatorname{Categorical}
\left(x;\operatorname{softmax}(Wz+b)\right)
\right].
\]

The active decoder instead evaluates the categorical NLL at \(z=\mu_q\). Since log-sum-exp is convex,

\[
\mathbb E_q[\mathrm{CE}(Wz+b,x)]
\geq
\mathrm{CE}(W\mathbb E_q[z]+b,x).
\]

Thus the current mean-only CE is an exact NLL for a deterministic hidden state, but not the expected accuracy term of a nondegenerate Gaussian latent posterior. A true latent VFE path should sample or otherwise evaluate the expectation, and it should let the observation update the posterior covariance unless a declared approximation freezes it.

The Gaussian-template decoder in `GL(K)_supplementary.tex` supplies another normalized emission,

\[
p(x=v\mid z)
=\frac{\mathcal N(z;\mu_v,\Sigma_v)}
{\sum_w\mathcal N(z;\mu_w,\Sigma_w)}.
\]

Its NLL is again a vocabulary cross-entropy over Gaussian energy logits. For uncertain \(z\), the correct VFE term is the expectation of that normalized NLL. A softmax of expected log template densities is generally not equal to the expected normalized NLL because expectation does not commute with the log partition.

The current full VFE can therefore supplement CE in two honest ways. It can be an explicit regularizer in \(\mathcal J_{\mathrm{reg}}\), or it can be the configuration-prior energy in the Gibbs ELBO. A state-level ELBO requires the structured-joint redesign in Section 5.

## 8. Why the observation term cannot simply be removed

For a finite vocabulary, every normalized energy-based token model has

\[
-\log p_\theta(x=v\mid h)
=E_\theta(v,h)+\log\sum_w\exp[-E_\theta(w,h)].
\]

This is cross-entropy with logits \(-E_\theta(w,h)\). Changing the energy changes the decoder geometry, but it does not remove the likelihood normalization. If the normalization is omitted, the objective no longer defines token probabilities or perplexity.

An alternative proper scoring rule such as a Brier score can train a categorical predictor, but it is not the log evidence and does not give the ordinary ELBO. A continuous Gaussian reconstruction loss on token embeddings can be an auxiliary observation model, but nearest-neighbor decoding from embedding space is not a normalized language model unless a categorical normalization is restored.

The observation-free VFE contains only prior, consensus, entropy, and geometric sectors. It can prefer coherent beliefs, but it receives no information about which vocabulary item should occur next. In the current architecture it also supplies no target gradient to the output projection. Removing CE without introducing another normalized token likelihood would therefore remove the language-model learning signal rather than replace it with VFE.

Latent-variable language models can be trained by ELBOs, as demonstrated by text VAEs [Bowman et al.](https://arxiv.org/abs/1511.06349), but strong autoregressive decoders may ignore the latent and cause posterior collapse. That is an empirical risk to monitor, not a reason the observation construction is invalid.

## 9. Executable correspondence for the investigated baseline

The active path provides a concrete counterexample to a coordinate-ELBO reading.

The model first calls `forward_beliefs` without targets in [`vfe3/model/model.py`](../vfe3/model/model.py) at lines 1549-1555. The model-channel update refines \(s\) against its hyperprior and \(\gamma\)-weighted consensus terms. The refined \(s_1\) is then assigned as both \(q_0\) and the active belief prior. One frozen-key, damped belief update uses the self and \(\beta\)-consensus sectors. Neither update receives the held-out token.

The target first appears in the categorical decode at lines 1576-1602. Line 1611 sets `loss = ce`. The active configuration has zero frame-mass, M-step self-coupling, CG-energy, and z-loss weights. Because `s_e_step=True`, the separate model-channel outer scalar is gated off at lines 1703-1704. The scored outer loss is therefore exactly CE.

The active `gamma_as_beta_prior` route also folds a detached \(\gamma\) distribution into the \(\beta\) prior. This makes \(s\) affect the value of the later belief objective without receiving the reciprocal derivative that one scalar coordinate update would require. The current frozen-key MM maps are update surrogates rather than exact minimizers of the profiled global scalar, and adding a categorical likelihood would destroy their present Gaussian precision-fusion closed form.

The outer CE gradient through the unrolled filters is valid discriminative training. It trains model tables, frames, the head mixer, and the output projection through their effect on prediction. It should be described as a differentiable conditional language model with VFE-inspired internal refinement, not as variational EM on the held-out token.

## 10. Recommended research decision

If the goal is the smallest exact evidence objective near the baseline, score the conditional latent ELBO in Section 7.1: expected categorical NLL plus \(D_{\mathrm{KL}}(Q_\psi\Vert P_\theta)\), with the finite VFE refinements treated as the recognition computation. This is a genuine ELBO, but it does not make the refinements coordinate updates and it does not include every consensus term as a separate generative sector.

If the goal is the smallest practical supplement using the complete current energy inventory, retain categorical CE and add selected VFE sectors as declared regularizers. Call the result a generalized observation-inclusive free-energy loss, not a state-level ELBO. This tests whether the theory supplies useful inductive bias without requiring a new probabilistic architecture.

If the goal is an exact state-level ELBO claim, build a separate opt-in structured path from the augmented Gaussian joint. Use a causal fixed joint, an explicit \(p(z_t\mid m_t,\cdot)\) bridge, a correlated posterior with sparse block precision or edge beliefs, and an expected normalized categorical emission. Replace the peer posterior KL by a fixed transition energy, or retain it only as a labeled consensus penalty. This is the recommended theoretical route.

If the goal is to preserve every non-observation sector of the current full VFE exactly, use the belief-configuration Gibbs lift. Keep the observation-independent VFE as the configuration prior energy and attach the token likelihood once; use the explicit state draw in Section 4.3 if the canonical expected observation term is to be reproduced. Carry the configuration entropy, partition function, and gauge reference measure. Describe deterministic settling as MAP or a saddle approximation, not the finite-temperature ELBO.

A derivation should not proceed until it passes the following checks:

1. The generative joint is normalized and does not depend on the variational posterior.
2. The observation index is explicit: the predictive prior sees \(x_{<t}\), while the training posterior may see \(x_t\).
3. The structured posterior has an explicit joint entropy or a valid region-entropy approximation, not only node entropies.
4. Every edge belief or cross-covariance satisfies its marginal-consistency constraints.
5. \(s\) affects \(q\) through a normalized conditional, and the observation expectation integrates over every latent variable appearing in the likelihood.
6. \(\beta\) and \(\gamma\) are derived as posterior distributions over explicit source variables, or are labeled as algorithmic routing weights.
7. Gauge variables have a declared status as parameters or latents, with a proper measure in the latter case.
8. The training algorithm either improves the same ELBO in each coordinate or drops the coordinate-EM claim.
9. Evaluation uses the target-blind predictive prior rather than the observation-conditioned posterior.
10. Latent use is measured through posterior-prior KL, prior-predictive versus posterior-predictive performance, covariance calibration, and collapse diagnostics.

The direct answer is therefore yes with a qualification. Training language as observations is exactly how a generative language model should be formulated. The full VFE can supply the latent complexity, consensus, model, and gauge structure around that likelihood. What cannot be done is to replace the normalized observation term with the present observation-free consensus dynamics and still claim a language-model ELBO. The most defensible synthesis is not “VFE instead of CE,” but “CE as the discrete observation term inside a structured, gauge-covariant VFE.”

## Primary references

The standard ELBO and variational-inference framing is reviewed by [Blei, Kucukelbir, and McAuliffe](https://arxiv.org/abs/1601.00670), and the amortized construction is [Kingma and Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). The VFE accuracy-complexity interpretation is given by [Friston](https://doi.org/10.1038/nrn2787), while the coordinate-EM baseline is [Dempster, Laird, and Rubin](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x). Sequential latent-variable precedents are [Chung et al., A Recurrent Latent Variable Model for Sequential Data](https://arxiv.org/abs/1506.02216), [Fraccaro et al., Sequential Neural Models with Stochastic Layers](https://arxiv.org/abs/1605.07571), and [Krishnan, Shalit, and Sontag, Structured Inference Networks for Nonlinear State Space Models](https://arxiv.org/abs/1609.09869). The graphical-model and region-free-energy foundations are [Wainwright and Jordan](https://www.cs.columbia.edu/~blei/fogm/2020F/readings/WainwrightJordan2008.pdf) and [Yedidia, Freeman, and Weiss](https://www.merl.com/publications/TR2004-040). The generalized loss-based interpretation is [Bissiri, Holmes, and Walker](https://arxiv.org/abs/1306.6430). The text-latent precedent and posterior-collapse warning begin with [Bowman et al.](https://arxiv.org/abs/1511.06349).
