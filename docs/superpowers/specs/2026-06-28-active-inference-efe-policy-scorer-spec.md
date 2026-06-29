# V3 Opt-In No-Grad EFE Token-Continuation Policy Scorer — Implementation Spec

Status: DESIGN SPEC. Not implemented. Default-off. No efficacy claimed.

Verified against HEAD `5e88afc`. This document specifies code that does not yet exist; it asserts no result. The scorer ships disabled by default, and with its toggle off the public behavior of `forward()` and `generate()` is byte-identical to HEAD `5e88afc`. Nothing here claims that V3 already implements an active-inference or Expected-Free-Energy (EFE) scorer, nor that any perplexity, calibration, or generation-quality improvement has been demonstrated. Efficacy is the hypothesis the pre-registered experiment of Section 4 tests, with a fixed demotion rule.

## 1. Summary and scope

This spec defines an opt-in, no-grad, flat, one-step EFE policy scorer that reranks candidate token continuations on a frozen V3 checkpoint. The build target for version one (v1) is token continuations: a small fixed menu of candidate next tokens is scored by an EFE functional, and a policy posterior selects among them inside the existing `generate()` loop. The scorer is pure tensor scoring over beliefs and priors V3 already computes. It introduces no learned parameter, no `nn.Linear`, no MLP, and no activation, and it runs entirely under `@torch.no_grad`.

A central honesty constraint sets the scope. At the v1 operating regime (a single look-ahead step over a sigma-free point belief, defined in Section 2.7) the expected-information-gain term of EFE is identically zero, by the mutual-information algebra of Section 2.8. The v1 scorer is therefore a pragmatic, preference-matching reranker: it ranks candidates by how close their predicted outcome distribution sits to an explicit task preference. The genuine epistemic, information-seeking behavior that distinguishes active inference from preference-matching becomes live only when belief state-uncertainty enters the marginalization (the sigma-validated Monte Carlo estimator of Section 2.7) or when a multi-step rollout carries forward genuine state-uncertainty. Both are deferred. The spec keeps the epistemic machinery present, logged, and clearly marked inert at v1, rather than presenting it as an active v1 signal.

The agent-set or community case, in which whole agents or token communities are the policies, is the more principled home for active inference, but it is deferred. It is blocked on two pieces that do not exist in source: an explicit community outcome model, and a non-flat covariant rollout over the model (`s`/`gamma`) channel, whose transport is currently hardcoded flat at `model.py:1047`. Section 6 specifies this case as deferred and keeps it a separate outcome space, with separate metrics and baselines, so that no empirical conclusion crosses between it and the continuation experiment.

The integration map confirmed against live source frames every design choice. `BeliefState` is the Gaussian tuple `(mu, sigma, phi)` with optional `(s, r)` channels (`belief.py:22-30`). `PriorBank.encode` maps token ids to initial beliefs with `q` initialized to `p` (`prior_bank.py:223-228`), and `encode_s` does the same for the model channel (`prior_bank.py:230-247`). The `forward()` method runs the belief pipeline inline (encode at `model.py:660-661`, attention `log_prior`/RoPE at `662-663`, the `vfe_stack` at `725-734`, then `final_norm` at `742-743`) and returns `(B, N, V)` logits on the inference path at `791-794`, never a `BeliefState`; there is no extraction seam today. The `generate()` method is `@torch.no_grad()` (`model.py:1164`), recomputes the full `forward(context)` per token with no key-value or belief cache (`model.py:1199-1201`), and truncates context to the last `cfg.max_seq_len` tokens. The `free_energy.log_likelihood` term is an inert gated stub (`free_energy.py:341`, `401-402`); the default E-step descends through the closed-form `gradients/kernels.py` route and never calls scalar `free_energy()`, so a live observation term is out of scope for a no-grad inference-time scorer. No EFE, policy, or belief-rollout code exists anywhere in source.

The recommended first scorer is therefore default-off, no-grad, flat, and one-step, on a frozen checkpoint, which sidesteps every code blocker. Horizon extension to `H = 2, 4` is a later phase gated on a belief or key-value cache, because the `generate()` recompute makes a horizon-`H` rollout over `Kp` candidates cost on the order of `Kp * H` forward passes per token.

## 2. Active-inference contract (A/B/C/D/E)

This section pins the theory the scorer must instantiate. It maps the discrete active-inference canon (Friston et al. 2017, "Active Inference: A Process Theory"; Da Costa et al. 2020, "Active inference on discrete state-spaces"; Smith, Friston and Whyte 2022 tutorial) onto the concrete V3 objects. Every statement here is a design contract, not a claim that V3 already does any of it.

### 2.0 Setting, outcome space, and horizon

The agent operates on a controlled closed-loop synthetic task in which an emitted token is a genuine action that changes the next observation the environment returns. This closed loop is the reason a fixed-corpus rerank is excluded from v1: in a static corpus the next observation does not depend on what the agent emitted, so no scoring rule over that corpus can be active. A policy $\pi$ is a candidate continuation drawn from a fixed pre-registered candidate generator (element E). For v1 the policy is a single candidate action token $a_\pi \in \mathcal{V}$; the multi-token policy $\pi = (a_1, \dots, a_H)$ is the deferred horizon extension.

The outcome variable $o_\tau \in \mathcal{V}$ is the environment's next observation at look-ahead step $\tau$, a categorical over the vocabulary $\mathcal{V}$ of size $V$. The predicted-outcome distribution under a policy is

$$q(o_\tau \mid \pi) = \sum_{s_\tau} p(o_\tau \mid s_\tau)\, q(s_\tau \mid \pi),$$

a length-$V$ categorical, the element-A likelihood marginalized over the element-B rolled-out belief. This predicted outcome is the agent's own forecast computed from its generative model. The agent never substitutes the environment's true response into this forecast; doing so would be oracle peeking rather than prospective planning (Section 2.2). The horizon $H$ is fixed and finite. The v1 build target is $H = 1$, so the policy sum below collapses to its $\tau = 1$ term. The total expected free energy of a policy is the horizon sum

$$G(\pi) = \sum_{\tau = 1}^{H} \big[\, \mathrm{risk}(\pi, \tau) + \mathrm{ambiguity}(\pi, \tau) \,\big],$$

with the two terms defined in Section 2.6.

### 2.1 A. Outcome likelihood $p(o \mid s)$

The state $s$ is a V3 Gaussian belief, the `BeliefState` $q = \mathcal{N}(\mu_q, \Sigma_q)$ with gauge frame $\phi_q$ (`belief.py:22-30`). On the theoretically pure path `use_prior_bank=True` the outcome likelihood is the existing KL-to-prior decode readout (`prior_bank.py:312-328`),

$$p(o = v \mid s) = \mathrm{softmax}_v\!\left( -\,\frac{\mathrm{KL}\!\big(\mathcal{N}(\mu_q, \Sigma_q)\,\big\|\,\mathcal{N}(\mu_v, \Sigma_v)\big)}{\tau_{\mathrm{eff}}} \right),$$

where $(\mu_v, \Sigma_v)$ is the per-vocabulary prior $\pi_v$ from the prior-bank tables (the model-channel `s` tables when `prior_source == "model_channel"`, `prior_bank.py:289-302`) and $\tau_{\mathrm{eff}} = \tau_{\mathrm{dec}} \cdot \exp(-\,\mathrm{clamp}(\text{decode\_log\_scale}, -3, 3))$ is the effective decode temperature (`prior_bank.py:304-310`). For the default diagonal `decode_mode` the divergence is the diagonal Gaussian KL,

$$\mathrm{KL}(q \,\|\, \pi_v) = \tfrac{1}{2} \sum_{k} \left[ \log\frac{\sigma_{v,k}}{\sigma_{q,k}} + \frac{\sigma_{q,k} + (\mu_{q,k} - \mu_{v,k})^2}{\sigma_{v,k}} - 1 \right],$$

so $p(o \mid s)$ depends on the full belief through the divergence seam (`decode_mode="full"` swaps in the full-covariance KL with no other change). The ablation `use_prior_bank=False` substitutes the linear readout $p(o = v \mid s) = \mathrm{softmax}_v(\mu_q W^\top)_v$ with $W$ a raw $(V, K)$ parameter and $\sigma_q$ discarded. The scorer records which decode path produced any reported number, because the linear readout removes all $\sigma_q$ dependence and would void the sigma-validation gate of Section 2.7.

### 2.2 B. Transition / rollout rule $q(s_\tau \mid \pi)$

The transition rule is how appending a candidate action evolves the belief. It is realized by a public, no-grad belief-rollout helper, the extraction seam that does not exist today. The helper ingests context-plus-candidate token ids and returns the converged belief and, optionally, the decode logits. It runs the same inline pipeline as `forward` (encode, attention log-prior and RoPE, the `vfe_stack`, then `final_norm`) and stops before the loss branch. For a one-step policy the rolled-out belief is the last-position belief after the candidate action $a_\pi$ has been appended and the E-step has reconverged,

$$q(s_1 \mid \pi) = \big[\, \texttt{rollout\_beliefs}([\,\text{context}, a_\pi\,])\,\big]_{\text{last position}}.$$

The scored belief appends the candidate action only. It does not append the environment's deterministic response to that action. This direction reconciles the two transition statements that an earlier draft left in conflict, and it is the opposite of folding the environment response into the score. Folding the realized response into the scored belief (so that, for an information-gathering action, the belief already contains the revealed observation) would let the agent rank candidates by ground-truth future observations it has not yet earned. That is oracle peeking, not prospective active inference: in a real closed loop the response is not available until the action is committed, and substituting it conflates prediction with the answer. The legitimate role of the environment response is to advance the real loop after a token is committed (Section 3.4), not to score candidates. A consequence, made explicit in Section 2.8, is that at $H = 1$ over a point belief an information-gathering action carries no epistemic advantage in the score, which is why the epistemic showcase task is deferred (Section 4 and Section 5).

For v1 the transition is the flat update: the model-channel transport is hardcoded flat (`transport_mode="flat"`, `model.py:1047`, confirmed intentional), the belief-channel transport is Regime-I flat, and no learned connection is consulted. This is why v1 sidesteps the covariant-oracle gap (the `_omega_builder` closure over `belief.sigma` at `e_step.py:426-435` affects only `regime_ii` / `regime_ii_covariant`, which the flat scorer never enters). The non-flat covariant rollout, required before any community-policy claim over the model channel, is deferred (Section 7, guard 5).

### 2.3 C. Preference prior $p(o \mid C)$

The preference is an explicit, peaked distribution over outcomes that encodes the task goal, supplied per task instance by the synthetic environment, not learned and not read from the corpus. Its canonical form is a tempered log-preference,

$$p(o \mid C) = \mathrm{softmax}_o\big(\beta_C\, U_C(o)\big),$$

where $U_C(o)$ is the task utility over observations and $\beta_C > 0$ is the preference precision. Each task instance fixes $U_C$ by naming the goal observation $o^\star$ (or a goal set), so $p(o \mid C)$ peaks on the task target and tends to $\delta_{o^\star}$ as $\beta_C \to \infty$. This peaked preference, $p_\text{task}$, is the v1 experimental preference. Two degenerate choices collapse the construction into ordinary scoring and serve as fixed controls.

The flat preference $p(o \mid C) = 1/V$ (the limit $\beta_C \to 0$) carries no goal. The forward-KL risk under it is the exact identity

$$\mathrm{risk}(\pi) = \mathrm{KL}[\,q(o \mid \pi)\,\|\,\text{uniform}\,] = \log V - H[\,q(o \mid \pi)\,],$$

so the full score, by Section 2.6, reduces to $G(\pi) = \log V - \mathcal{I}(o; s \mid \pi)$, whose minimization maximizes expected information gain. This is the single, consistent definition of the flat-preference arm; the ablation table of Section 4.3 uses exactly this form, not an "ambiguity-only" restatement. The flat preference is the pure-epistemic ablation, and this corrected sign statement supersedes the earlier "be-confident, risk $= -H$" reading, which was inverted. At the v1 sigma-free point belief, $\mathcal{I} \equiv 0$ (Section 2.8), so the flat-preference score collapses to the constant $\log V$ and the policy posterior over the menu is uniform. The flat-preference arm therefore carries no signal at v1 and is meaningful only in the epistemic-live phase; it is reported, not gated, at v1.

The held-out-predictive preference sets $U_C(o) = \log p_{\text{data}}(o)$, giving $p(o \mid C) = p_{\text{data}}(o)$ and

$$\mathrm{risk}(\pi) = \mathrm{KL}[\,q(o \mid \pi)\,\|\,p_{\text{data}}\,] = \mathbb{E}_{q}[\log q(o \mid \pi)] - \mathbb{E}_q[\log p_{\text{data}}(o)],$$

which up to the predictive entropy is the next-observation cross-entropy and tracks ordinary NLL. This is the collapse of EFE into likelihood scoring, demoted to the control arm that the peaked-preference EFE must beat. Because $p_{\text{data}}$ carries no per-episode goal, it cannot steer the controlled task, which is precisely what makes it the control. Both degenerate preferences fall out of the same risk functional as decomposition controls.

### 2.4 D. Initial belief $q(s_0)$

The initial belief is the converged context belief, the state from which every candidate rollout branches. It is the helper's output on the context alone, before any candidate is appended,

$$q(s_0) = \big[\, \texttt{rollout\_beliefs}([\,\text{context}\,])\,\big]_{\text{last position}},$$

the `BeliefState` $(\mu_{s_0}, \Sigma_{s_0}, \phi_{s_0})$. The encode initializes $q = p$ from the prior bank (`prior_bank.py:223-228`), and the E-step refines it; $q(s_0)$ is the post-E-step result. Sharing $q(s_0)$ across candidates, computing it once and rolling each $a_\pi$ forward from it, is the natural site for the deferred belief or key-value cache.

### 2.5 E. Policy / candidate prior $\pi$ and precision $\gamma$

The policy prior $E(\pi)$ is a distribution over the finite candidate set produced by a fixed pre-registered candidate generator (for v1 the top-$K_p$ tokens of the frozen base predictive). The generator and $K_p$ are frozen before any result is read; nothing post-hoc may enlarge, prune, or reweight the candidate set (Section 7, guard 7). The precision $\gamma > 0$ is the inverse temperature over policies. The policy posterior is

$$Q(\pi) = \mathrm{softmax}_\pi\big(\log E(\pi) - \gamma\, G(\pi)\big),$$

which for a uniform candidate prior reduces to $Q(\pi) = \mathrm{softmax}_\pi(-\gamma\, G(\pi))$. Both $E$ and $\gamma$ are fixed scalars or tables in v1, not learned parameters; the no-grad scorer needs no `nn.Parameter` of its own. A learned $\gamma$ would be a separate opt-in, default-off toggle and is out of scope for v1.

### 2.6 The score: risk, ambiguity, and the policy posterior

Per look-ahead step the expected free energy is the risk-plus-ambiguity decomposition,

$$G(\pi, \tau) = \underbrace{\mathrm{KL}\big[\, q(o_\tau \mid \pi)\,\big\|\,p(o_\tau \mid C)\,\big]}_{\mathrm{risk}(\pi, \tau)\ \text{(pragmatic)}} + \underbrace{\mathbb{E}_{q(s_\tau \mid \pi)}\big[\, H[\,p(o_\tau \mid s_\tau)\,]\,\big]}_{\mathrm{ambiguity}(\pi, \tau)\ \text{(expected likelihood entropy)}},$$

with $H[p(o \mid s)] = -\sum_o p(o \mid s) \log p(o \mid s)$ the entropy of the V3 decode readout. Risk is low when predicted outcomes match the task preference; ambiguity is low when the rolled-out states yield confident readouts. Lower $G$ is the preferred policy. The honest definition of the epistemic content is the expected information gain, not the raw magnitude of $\sigma_q$ (Section 2.7) and not the ambiguity term, which is a cost on outcome-likelihood spread rather than an exploration bonus. The information gain is given by the exact mutual-information bridge

$$\mathcal{I}(o_\tau; s_\tau \mid \pi) = H[\,q(o_\tau \mid \pi)\,] - \mathbb{E}_{q(s_\tau \mid \pi)} H[\,p(o_\tau \mid s_\tau)\,] = \text{predictive entropy} - \text{ambiguity}.$$

The equivalent epistemic-pragmatic form is the exact rearrangement

$$G(\pi, \tau) = -\underbrace{\mathbb{E}_{q(o_\tau \mid \pi)}[\log p(o_\tau \mid C)]}_{\text{pragmatic value}} - \underbrace{\mathbb{E}_{q(o_\tau \mid \pi)} \mathrm{KL}\big[\,q(s_\tau \mid o_\tau, \pi)\,\big\|\,q(s_\tau \mid \pi)\,\big]}_{\text{epistemic value}\ =\ \mathcal{I}(o_\tau; s_\tau \mid \pi)},$$

so the full EFE equals the negative pragmatic value minus the expected information gain. The scorer computes and logs risk, ambiguity, predictive entropy, the information-gain bridge $\mathcal{I}$, and the raw continuation log-probability as separate diagnostics, each reported apart from the others, so an apparent gain can never be attributed to EFE when it is a likelihood effect.

### 2.7 Exact-decomposition assumptions and the sigma-validation gate

The risk-plus-ambiguity identity is exact given (i) the generative-model factorization $p(o, s) = p(o \mid s)\,p(s)$ with the preference $p(o \mid C)$ standing in as the target outcome marginal, and (ii) the predictive outcome marginal computed as $q(o_\tau \mid \pi) = \sum_{s_\tau} p(o_\tau \mid s_\tau)\, q(s_\tau \mid \pi)$. The further equality with the pragmatic-plus-epistemic form additionally requires (iii) the conditional state posterior in the information-gain term to be the exact Bayesian posterior $q(s_\tau \mid o_\tau, \pi) = p(s_\tau \mid o_\tau, \pi)$, and (iv) for $H > 1$ a mean-field factorization of beliefs over the horizon.

Under V3 these hold only approximately. Assumption (ii) is approximated because the marginalization over the Gaussian belief has no closed form through the softmax decode; v1 evaluates it at the point belief $s_\tau = \mu_{s_\tau}$ (a sigma-free delta approximation) until the gate of this section passes, after which Monte Carlo over a fixed $S = 16$ samples $\mu^{(s)} \sim \mathcal{N}(\mu_{s_\tau}, \Sigma_{s_\tau})$ is unlocked. Assumption (iii) is approximated because V3 has no native one-shot Bayesian update; v1 deliberately selects the risk-plus-ambiguity form, both terms of which are computable directly from the decode and the rolled-out belief without forming the conditional posterior, so v1 does not depend on (iii). The canonical information-gain diagnostic is therefore the cheap MI bridge $\mathcal{I} = H[q(o \mid \pi)] - \mathbb{E}_q H[p(o \mid s)]$, not the Bayesian-posterior estimator, which would require an inner conditioning E-step and coincides with the bridge only as that E-step converges. Assumption (iv) is vacuous at $H = 1$ and becomes live only in the deferred multi-step phase.

The belief variance $\sigma_q$ may enter the ambiguity term only through the proper functional $\mathbb{E}_{q(s \mid \pi)} H[p(o \mid s)]$, and only after a pre-registered validation has shown that $\sigma_q$ carries epistemic content. The default ambiguity estimator is the sigma-free point form $H[p(o \mid \mu_s)]$. The sigma-dependent Monte Carlo estimator is unlocked only when the gate passes. The gate is a fixed test, pre-registered with its dataset, statistic, and pass threshold before any score is read, run on the checkpoint that would host any sigma-dependent arm: per-token $\sigma_q$ is positively and monotonically associated with realized next-observation NLL/error (Spearman $\rho \ge 0.2$ with a 95 percent bootstrap confidence interval whose lower bound exceeds zero and the measured floor), $\sigma$-binned predictions are calibrated (expected calibration error within $\sigma$ bins below $0.05$), and larger $\sigma_q$ predicts larger realized information gain when the true observation is revealed. Failing the gate does not disable the scorer; it forces the sigma-free point-entropy ambiguity and forbids any claim that V3 sigma is an ambiguity or epistemic-value signal. The gate is a falsifier fixed up front, never adjusted after a result (Section 7, guards 4 and 7).

### 2.8 What is live and what is inert at v1

The v1 operating regime is one look-ahead step ($H = 1$) over a sigma-free point belief. Under it the rolled-out belief $q(s_1 \mid \pi)$ is a delta at $\mu_{s_1}$, so $q(o \mid \pi) = p(o \mid \mu_{s_1})$ and the mutual-information bridge collapses exactly,

$$\mathcal{I}(o; s \mid \pi) = H[\,p(o \mid \mu_{s_1})\,] - \mathbb{E}_{\delta(\mu_{s_1})} H[\,p(o \mid s)\,] = H[\,p(o \mid \mu_{s_1})\,] - H[\,p(o \mid \mu_{s_1})\,] = 0,$$

for every candidate. The expected information gain is identically zero at v1, not merely small. Two consequences follow and are stated plainly rather than hidden. First, the full score collapses to the pragmatic value,

$$G(\pi) = \mathrm{risk}(\pi) + \mathrm{ambiguity}(\pi) = \mathrm{KL}[q(o \mid \pi)\,\|\,p(o \mid C)] + H[q(o \mid \pi)] = -\,\mathbb{E}_{q(o \mid \pi)}[\log p(o \mid C)],$$

the cross-entropy between the predicted outcome and the preference. The v1 scorer is a pragmatic preference-matching reranker. Second, the ambiguity term at the point belief equals the predictive entropy $H[p(o \mid \mu_{s_1})]$, a confidence cost that rewards predictable outcomes; it is not an exploration bonus, and it can penalize an information-gathering action whose immediate predicted outcome is diffuse. Genuine information-seeking requires state-uncertainty in the marginalization, which arrives only through the sigma-validated Monte Carlo estimator (Section 2.7) or through a multi-step rollout that carries forward genuine belief spread.

The following objects are therefore inert at v1 and are reported as such, never presented as live signals: `PolicyScore.epistemic` (the MI bridge $\mathcal{I}$, identically $0$), the epistemic-only arm $G = -\mathcal{I}$ (identically $0$, hence undefined as a ranking), and the flat-preference arm $G = \log V - \mathcal{I}$ (the constant $\log V$, hence a uniform posterior). The risk term, the ambiguity term as a predictive-entropy confidence cost, and the full pragmatic score remain live and varying across candidates. The epistemic falsifiers of Section 4.7 (the sign-flipped epistemic arm and the epistemic-value-versus-realized-uncertainty-reduction correlation) operate on $\mathcal{I}$ and are consequently gated to the epistemic-live phase (sigma-validated Monte Carlo, or $H \ge 2$ with state-uncertainty); applying them at v1 would compare $0$ against $0$ and spuriously fire. This is the design reason the epistemic showcase task is deferred (Sections 4.1 and 5).

## 3. Code and API design

Every signature follows the project convention: tensors first, then `float | torch.Tensor`, then undefined floats/ints/bools, then defined floats/ints/bools, then `Optional`, then `**kwargs`; names, annotations, `=`, and trailing comments vertically aligned; shape comments and type hints throughout; paper notation in variable names. The design adds one extraction seam, three small registries, and one guarded branch in `generate()`. Nothing introduces a learned parameter, and the default `policy_mode='none'` leaves `forward()` and `generate()` byte-identical to HEAD `5e88afc`.

### 3.1 The shared belief-production seam: `forward_beliefs` and `rollout_beliefs`

The contract requires a public no-grad belief-rollout helper returning the converged belief plus optional logits, serving as the single path that `generate()`, the diagnostics replays, and the EFE scorer share. The repository has no such seam: `forward()` (`model.py:644-805`) runs the belief pipeline inline and never exposes the resulting `BeliefState`. The refactor factors the belief-production lines (`660-743`) into a new method `forward_beliefs`, without forking the sequence, and rewrites `forward()` to call it.

```python
def forward_beliefs(
    self,
    token_ids:      torch.Tensor,                    # (B, N) integer token ids

    *,
    return_logits:  bool             = False,        # also decode (B, N, V) logits; else logits is None
    capture:        Optional[dict]   = None,         # out-param: converged pre-transform q* (M-step self-coupling)
    estep_grad_out: Optional[dict]   = None,         # diag out-param: E-step belief-grad norms (forwarded)
) -> 'Tuple[BeliefState, Optional[torch.Tensor]]':
    r"""Run the belief pipeline and return the converged belief q* (post final_norm), optionally
    with the decoded logits.

    Factors the (previously inline) sequence q_i(0) = p_i = encode(token) -> phi <- pos_phi ->
    (optional s-refine) -> precision-bias fold -> vfe_stack (L blocks of E-step belief descent) ->
    final_norm, i.e. the map from token ids to the converged Gaussian tuple q* = (mu*, Sigma*, phi*).
    The returned BeliefState carries mu = final_norm(mu_final, sigma_final), sigma = sigma_final, and
    phi = out.phi UNCHANGED (final_norm transforms only the mean; model.py:742-743), so a caller can
    read q*.phi for the M-step gauge penalty 0.5*mass_phi*(phi**2).mean() (model.py:807-813) exactly as
    forward does today. When ``capture`` is a dict it is forwarded to vfe_stack(capture=...), which
    fills it in-place with the last block's CONVERGED pre-transform q* (model.py:720-734) that the
    M-step self-coupling term reads; the caller passes the same dict it currently builds at
    model.py:723. When ``return_logits`` is set, the per-position outcome distribution p(o | q*_i) is
    produced via ``self.prior_bank.decode(mu*.float(), Sigma*.float())`` inside the SAME fp32 island
    ``with self._amp_off_context(...)`` that forward's inference branch uses (model.py:791-792), so the
    logits are byte-identical to the pre-refactor ``forward(targets=None)`` return.

    This is the single belief-production seam. It is grad-transparent: it carries the SAME internal
    ``run = no_grad() if e_step_gradient=='detach' else nullcontext()`` and ``amp`` contexts ``forward``
    establishes today (model.py:696-706), so a grad-enabled training caller and a no-grad inference
    caller both get the identical forward value. The no-grad property used by the policy layer comes
    from the caller's ``@torch.no_grad`` scope (``generate``, ``rollout_beliefs``), not from this method.
    """
```

`forward()` becomes a thin shell. On the training path it calls `forward_beliefs(token_ids, return_logits=False, capture=cap, estep_grad_out=...)` so no `(B, N, V)` logits tensor is materialized, preserving the fused-chunked memory win (`model.py:764-789`); it then runs its existing decode-plus-cross-entropy assembly reading `belief.mu`/`belief.sigma` exactly as it consumes `mu_final`/`sigma_final` now, adds the `mass_phi` gauge penalty from `belief.phi` (`model.py:807-813`), and adds the `mstep_self_coupling_weight` term reading the filled `cap` (`model.py:814` onward). On the `targets=None` inference path it returns `forward_beliefs(token_ids, return_logits=True)[1]`, so its public signature and three return shapes are preserved. The fused-chunked training branch is unaffected because it consumes `belief.mu`/`belief.sigma` after the call exactly as before.

The public deliverable is a one-line no-grad wrapper naming the rollout intent, called by the scorer and the horizon rollout.

```python
@torch.no_grad()
def rollout_beliefs(
    self,
    token_ids:     torch.Tensor,                     # (B, N) context ids (the action prefix)  -> D

    *,
    return_logits: bool          = True,             # continuation scoring needs the decode    -> A
) -> 'Tuple[BeliefState, Optional[torch.Tensor]]':
    r"""Public no-grad belief rollout: the contract's D (initial belief from current context) and the
    one-step B (transition rule) building block. A single forward of ``token_ids`` through the shared
    belief seam under no_grad, returning (q*, logits). Appending a candidate ACTION token to
    ``token_ids`` and re-calling realizes one transition q*_t -> q*_{t+1}; iterating it H times is the
    fixed-horizon rollout B. The environment's response to a committed action is appended by the loop
    AFTER selection (model.py generate branch), never inside the scored rollout. Returns the SAME
    tensors ``forward`` would, so it adds no new numerical path.
    """
    return self.forward_beliefs(token_ids, return_logits=return_logits)
```

The belief-production part of the diagnostics replays (`diagnostics`/`attention_maps`, around `model.py:1077`, `1109`, `1137`) should migrate to `forward_beliefs` in the same change so the single-path property holds end to end. That migration is recommended but its diagnostics outputs must remain byte-identical; if any cannot be made so trivially it is deferred, since the load-bearing byte-identity requirement is only on `forward()` and `generate()`. The gamma model-channel diagnostics (`_gamma_energy`, `gamma_attention_maps`) stay on their own s-channel route as a distinct functional.

The Phase 0 golden byte-identity test pins the refactor across the configs that exercise the touched branches, not only the inference logits: (a) `forward(targets=None)` inference logits; (b) dense training loss and ce (`targets` given, `decode_mode='diagonal'`); (c) fused-chunked training (`decode_mode='diagonal_chunked'`, `targets` given), confirming no `(B, N, V)` tensor is formed; (d) `mstep_self_coupling_weight > 0`; (e) `mass_phi > 0`; (f) the linear-decode ablation `use_prior_bank=False`. Each output must be byte-identical to HEAD `5e88afc`, and the `return_logits=True` branch must be byte-identical to `forward(targets=None)`, so the "beliefs plus logits" contract is verified rather than asserted.

### 3.2 The `policy_mode` registry and the scorer-callable contract

The scorer seam follows the established add-by-registering pattern (mirroring `alpha_i.py:15-39`: a module dict, a `register_*` decorator, a `get_*` lookup that raises `KeyError` with the available keys). A new module `vfe3/inference/policy.py` holds it.

```python
_POLICIES: Dict[str, Callable] = {}

def register_policy(name: str) -> Callable:
    """Decorator registering a policy scorer under ``name`` (cf. register_alpha)."""
    def _wrap(fn: Callable) -> Callable:
        _POLICIES[name] = fn
        return fn
    return _wrap

def get_policy(name: str) -> Callable:
    """Return the registered policy scorer (KeyError with the available keys if absent)."""
    if name not in _POLICIES:
        raise KeyError(f"no policy mode {name!r}; available: {sorted(_POLICIES)}")
    return _POLICIES[name]
```

The four registered keys are `none`, `logprob_control`, `efe_one_step`, and `efe_rollout`. `none` is never dispatched (the `generate()` branch in Section 3.4 short-circuits before any lookup) and exists only so config validation accepts the default. `logprob_control` is the matched-compute control: it scores candidates by raw continuation log-prob over the same menu and the same rollout cost, with `risk`/`ambiguity` returned as zeros so the diagnostics columns line up. `efe_one_step` is the v1 build (`horizon=1`). `efe_rollout` is the staged horizon extension (`horizon>1`), gated in Section 3.5 on a belief cache.

Every registered scorer obeys one signature and returns one type. The convention places tensors first; the `model` object has no tensor slot, so it sits immediately after the tensor block as a no-default positional. (An alternative considered and rejected for v1 is passing bound callables `rollout_beliefs` and a decode handle instead of the whole model; the model handle is retained because it keeps the call site minimal and the scorer is already model-coupled through the belief seam.)

```python
def policy_scorer(
    context:     torch.Tensor,                       # (B, N) current-context ids                  -> D
    candidates:  torch.Tensor,                       # (B, Kp, L) candidate continuation ids       -> E
    preference:  torch.Tensor,                       # (V,) or (B, V) log p(o | C), broadcastable  -> C

    model:       'VFEModel',                          # belief handles: rollout_beliefs / prior_bank -> A, B

    *,
    gamma:       float                  = 1.0,        # policy precision in softmax(-gamma * G)
    horizon:     int                    = 1,          # fixed finite rollout depth H
    score_terms: Tuple[str, ...]        = ("risk", "ambiguity"),  # which terms enter G(pi)
    log_prior:   Optional[torch.Tensor] = None,       # (B, Kp) log candidate prior E; None -> uniform
    **kwargs,
) -> 'PolicyScore':
    r"""Score a fixed candidate menu by Expected Free Energy and return a policy posterior.

    For each policy pi (a candidate continuation, column of ``candidates``), roll the belief forward
    H = ``horizon`` steps from D = rollout_beliefs(context) by appending the candidate's ACTION tokens
    and re-rolling (the B transition; the environment response is NOT folded into the score, Section
    2.2), decode the horizon outcome distribution q(o | pi) = p(o | q*_pi) (A), and form

        G(pi) = risk(pi) + ambiguity(pi)
              = KL[ q(o | pi) || p(o | C) ]  +  E_{q(s | pi)} H[ p(o | s) ] .

    At the v1 default (horizon=1, sigma-free point belief) the information-gain content I is identically
    0 (Section 2.8): G reduces to the pragmatic cross-entropy -E_{q(o|pi)}[log p(o|C)], `epistemic` is
    returned as exact zeros, and the policy is a pragmatic preference-matching reranker. The policy
    posterior is Q(pi) = softmax_pi( -gamma * G(pi) + log E ), with E the candidate prior (``log_prior``;
    uniform when None). ``score_terms`` selects which terms are summed into G, so the risk-only and
    flat-preference reductions are recoverable without a code change; the epistemic-only reduction
    (G = -I) is inert at v1 and is exposed only for the epistemic-live phase.
    """
```

The return type is a `NamedTuple` (the belief tuple is itself a `NamedTuple`; `belief.py:22`) that keeps the active score and every diagnostic strictly separate, so the raw log-prob is never folded into the metric the policy acts on.

```python
class PolicyScore(NamedTuple):
    score:            torch.Tensor   # (B, Kp) G(pi) = sum of enabled score_terms
    risk:             torch.Tensor   # (B, Kp) KL[q(o|pi) || p(o|C)]            (pragmatic / preference)
    ambiguity:        torch.Tensor   # (B, Kp) E_{q(s|pi)} H[p(o|s)]            (likelihood entropy; = predictive entropy at v1 point belief)
    epistemic:        torch.Tensor   # (B, Kp) MI bridge I; IDENTICALLY 0 at v1 (Section 2.8), logged separately
    log_prob:         torch.Tensor   # (B, Kp) raw continuation log-prob, logged SEPARATELY from score
    policy_posterior: torch.Tensor   # (B, Kp) softmax(-gamma * score + log_prior)
```

The config fields, added to `VFE3Config` and validated in `__post_init__` exactly as the other seam keys are (`_require(self.policy_mode, tuple(sorted(_POLICIES)), "policy_mode")`, cf. `config.py:963`).

```python
policy_mode:        str              = "none"               # registry key: none | logprob_control | efe_one_step | efe_rollout
policy_horizon:     int              = 1                    # fixed finite rollout depth H
policy_top_k:       int              = 8                    # candidate menu size Kp (the fixed generator)
policy_precision:   float            = 1.0                  # policy precision gamma in softmax(-gamma G)
policy_preference:  str              = "task"               # preference registry key -> p(o|C)
policy_score_terms: Tuple[str, ...]  = ("risk", "ambiguity")  # which EFE terms enter G(pi)
```

These six fields are the pre-registration surface: fixing the candidate generator, the horizon, the preference, the precision, and the term set in config before a run structurally forbids the post-hoc-rescue failure. They are read only when `policy_mode != 'none'`, so their defaults are inert on the pure path.

### 3.3 The preference (C) and ambiguity sub-registries

The preference distribution `p(o | C)` is its own small registry (`register_preference`/`get_preference` in `vfe3/inference/policy.py`), so a new preference slots in by registration, never by editing the scorer. Each form returns log-probabilities over the outcome vocabulary and may be global `(V,)` or per-batch `(B, V)`; the controlled task supplies a per-episode `(B, V)` peaked preference (the goal observation differs per episode), while the language-modeling secondary metric uses a global form.

```python
def preference(
    prior_bank: 'PriorBank',                         # vocab handle (V) and any data table

    *,
    eps:        float = 1e-12,
) -> torch.Tensor:                                   # (V,) or (B, V) log p(o | C)
```

Three keys instantiate the design's three arms. `task` is the explicit, peaked goal preference on the controlled closed-loop synthetic task, the genuine pragmatic-EFE arm. `held_out_predictive` sets `p(o | C)` to the data distribution, making `risk` reduce to NLL; it is the control arm EFE must beat. `flat` is the uniform preference, which by Section 2.3 gives `G = log V - I` and is the pure-epistemic ablation, inert at the v1 point belief (constant `log V`) and meaningful only in the epistemic-live phase.

The epistemic/ambiguity term has its own sub-registry. The registered default ambiguity is `likelihood_entropy`, the outcome-likelihood entropy `E_{q(s|pi)} H[p(o|s)]` computed from the decoded predictive categorical at the belief mean (so at v1 it equals the predictive entropy), not from `sigma`. A `sigma_mc` ambiguity variant exists in the same sub-registry but is gated: it raises unless `policy_sigma_ambiguity_validated=True` has been set in config (default `False`), and that flag is documented as forbidden until the pre-registered validation gate of Sections 2.7 and 4.5 passes. This structurally prevents calling `sigma` an ambiguity value, or making the information-gain term live, before it is earned.

### 3.4 The attachment point in `generate()`

`generate()` is already `@torch.no_grad()` (`model.py:1164`) and recomputes the full forward per token (`model.py:1199-1201`). The scorer attaches inside the per-token loop, after the context truncation, as a single guarded branch. When `policy_mode == 'none'` the existing body runs verbatim, so default generation is byte-identical. When `policy_mode != 'none'` the branch builds the fixed candidate menu from the top `policy_top_k` next tokens of the base last-position logits (the pre-registered generator E), scores them through `get_policy(cfg.policy_mode)`, and selects from the policy posterior.

```python
for _ in range(max_new_tokens):
    context = seq[:, -self.cfg.max_seq_len:]                  # (B, <=max_seq_len)
    if self.cfg.policy_mode == "none":
        logits = self.forward(context)[:, -1, :]             # (B, V)  -- existing path, unchanged
        if greedy:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                sorted_probs = sorted_logits.softmax(dim=-1)
                cumprobs = sorted_probs.cumsum(dim=-1)
                remove = cumprobs - sorted_probs >= top_p
                remove_unsorted = remove.scatter(-1, sorted_idx, remove)
                logits = logits.masked_fill(remove_unsorted, float("-inf"))
            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = self._policy_select(context)            # (B, 1)  -- EFE rerank, no_grad
    seq = torch.cat([seq, next_token], dim=-1)
return seq
```

`_policy_select` is a private no-grad method that decodes the base last-position logits once, takes the top `policy_top_k` ids as `candidates`, calls `get_policy(self.cfg.policy_mode)(context, candidates, get_preference(self.cfg.policy_preference)(self.prior_bank), self, gamma=self.cfg.policy_precision, horizon=self.cfg.policy_horizon, score_terms=self.cfg.policy_score_terms)`, and returns `candidates.gather(...)` at `policy_posterior.argmax` (or a `multinomial` draw when sampling is requested). Because the `none` branch is the verbatim pre-existing body and the `else` branch is reached only under a non-default toggle, greedy, temperature, `top_k`, and `top_p` behavior is preserved exactly when `policy_mode='none'`. The `PolicyScore` returned per step is the hook for logging risk, ambiguity, epistemic, and raw log-prob separately. In the closed-loop experiment the environment's deterministic response to the committed `next_token` is appended to `seq` by the driver before the next iteration; that response advances the loop and is never used to score candidates (Section 2.2).

### 3.5 Cost and the horizon precondition

The v1 build is `efe_one_step` with `policy_horizon=1` and flat transport, which is one extra batched `rollout_beliefs` over the `Kp = policy_top_k` candidates per generated token. Because `generate()` recomputes the entire encode, E-step, and decode for the whole context at every step with no cache and truncates context (`model.py:1198-1201`), a horizon-`H` rollout multiplies per-token cost by roughly `Kp * H` (each candidate is re-rolled `H` times from scratch). The one-step scorer therefore pays `Kp` recomputes per token, acceptable for a frozen-checkpoint reranker. The `efe_rollout` key (`policy_horizon>1`) is gated: it must hard-error until a belief or key-value cache exists, because without incremental belief reuse the `Kp * H` recompute makes the compute-matched baselines decisive and the wall-clock honesty check unwinnable. The cache is the binding precondition for the horizon phase only, not for the one-step phase. `efe_rollout` and the agent-set case are spec'd but deferred, kept as separate registry entries and separate experiments.

### 3.6 Pure-path and no-NN compliance

The scorer adds no learned parameters. `rollout_beliefs`/`forward_beliefs` reuse the existing E-step kernels and `prior_bank.decode` and introduce no new numerical path; they expose a belief `forward()` already computes and discards. The preference `p(o | C)` is a fixed tensor produced by the preference registry, never an `nn.Parameter` and never trained. `risk` is a closed-form KL between two categoricals, `ambiguity` is the entropy of the decoded predictive categorical, the policy posterior is a `softmax` over a finite menu, and `gamma`, `policy_horizon`, `policy_top_k`, `policy_precision`, the preference key, and the score-term tuple are plain config scalars and keys. No `nn.Linear`, no MLP, no activation, and no `nn.Parameter` is created anywhere in the policy module; the scorer is tensor scoring over existing beliefs and priors, so it touches none of the documented opt-in NN exceptions and runs entirely under `@torch.no_grad` without invoking `requires_grad`. The default `policy_mode='none'` reads none of the policy fields, never enters the registry, and runs the verbatim pre-existing `generate()` body, while the `forward_beliefs` extraction is a behavior-preserving factoring that returns the identical tensors `forward()` produced before (pinned by the Phase 0 golden test of Section 3.1). The scorer is strictly inference-time: it is never wired into `gradients/kernels.py`, `gradients/oracle.py`, or `free_energy.log_likelihood` (the inert gated stub at `free_energy.py:341`, `401-402`), so no train-time EFE replacement is created or implied.

## 4. Experiment, baselines, and pass/fail

This section pre-registers the experiment a narrow active-inference verdict permits: one finite, falsifiable test of a default-off, no-grad, flat, one-step EFE reranker over candidate token continuations, run on frozen checkpoints. Everything below is fixed here, before any scorer code is written, and may not be revised after a result is seen. The agent-set experiment is deferred (Section 6) and its conclusions may not be shared with this one.

Two checkpoint families carry the evaluation, and they are kept distinct. The synthetic-task checkpoint is trained on the controlled closed-loop dynamics of Section 4.1 so that its predictive model `p(o | context, action)` is accurate, the precondition any model-based planner needs; closed-loop task success, the primary metric, runs on it. The language-modeling checkpoint is the established WikiText operating point (`embed_dim = 20`, 15k steps, linear decode `use_prior_bank=False`, `use_head_mixer=False`, around 144.5 perplexity); the secondary language-modeling and calibration metrics and the language-side sigma diagnostic run on it.

### 4.1 The controlled closed-loop synthetic task

A reranking over a fixed text corpus cannot be active inference, because teacher-forced next-token prediction does not let an emitted token change a later observation. The v1 experiment runs on a controlled, fully observed, closed-loop synthetic environment whose pragmatic payoff is realizable at a one-step horizon, the ring goal-steering ("cursor control") task. This choice is deliberate. An earlier draft proposed a partially observed masked key-value retrieval task in which the only route to success is to probe a masked cell and then answer; that task is unsolvable at `H = 1` (the probe and the answer are two distinct steps) and its payoff is epistemic, so by Section 2.8 it cannot be tested by the v1 scorer at all. The masked-retrieval task is therefore moved to the deferred epistemic-live phase (Section 5, Phase 3), and the v1 primary task is one whose pragmatic structure the v1 scorer can genuinely exercise.

Each episode of the ring task draws from a synthetic vocabulary partitioned into `m = 16` state symbols arranged on a ring `{q_0, ..., q_15}`, three action tokens `{DEC, STAY, INC}`, control tokens `{GOAL, CUR, SEP, EOS}`, and reserved filler, for a vocabulary of `V = 32`. An episode samples an initial state `s_0` and a goal `g != s_0` uniformly on the ring and presents the context `GOAL q_g SEP CUR q_{s_t}`. The agent emits one action token, and the environment applies the deterministic transition `s_{t+1} = (s_t + delta(a)) mod m` with `delta(INC) = +1`, `delta(DEC) = -1`, `delta(STAY) = 0`, then re-renders the context with the new current state. The episode runs for a fixed budget of `T_ep = 10` steps (the maximum ring distance is `m/2 = 8`, so the budget leaves slack only for a near-optimal policy). An episode is scored correct if and only if the state equals the goal at the budget's end; steps-to-goal and the fraction of post-arrival steps held at the goal are secondary.

The world is fully observed: the current state is always shown, so there is no hidden variable to resolve and no epistemic payoff. The task is purely pragmatic, and the peaked preference is what makes it solvable. From any state the cyclic-distance-reducing action strictly reduces the distance to the goal, and the one-step pragmatic score `risk(pi) = KL[q(o | pi) || p_task]` is minimized by exactly that action, because its predicted next observation is the reachable state nearest the goal and therefore closest to the goal-peaked preference. Greedy `H = 1` pragmatic descent reaches the goal in at most `m/2` steps. The data-distribution preference `p_data` and the flat preference `p_flat` carry no per-episode goal, so they cannot steer; this is what makes the explicit peaked preference load-bearing at the v1 horizon. The claim being tested at v1 is pragmatic, that an explicit goal preference steers behavior toward the goal better than the controls and baselines, not epistemic; the spec makes no "epistemic is load-bearing" claim about the v1 task, and a v1 outcome, success or failure, is not evidence about epistemic active inference.

The five contract elements are instantiated against live V3 primitives as follows. D (initial belief) is the converged `BeliefState` returned by `rollout_beliefs` on the current context, factored out of the inline pipeline so generation, diagnostics, and rollouts read one belief trajectory. E (policy/candidate prior) is the top-`Kp` next-token menu under the current logits, with the candidate prior the model's own softmax over that menu; on this task the three action tokens are a subset of the menu. B (transition) is, for each candidate action, re-running `rollout_beliefs` on the context extended by the action token only, flat transport, no gradients, no environment response folded in (Section 2.2). A (outcome likelihood `p(o | s)`) is the decode map: on the operating-point path `use_prior_bank=False` the linear `softmax(mu W^T)`, on the pure path of Phase 4 `use_prior_bank=True` the KL-to-prior decode; the ambiguity term is its entropy and uses no `sigma`. C (preference `p(o | C)`) is the explicit peaked categorical `p_task = softmax(beta_C U_C)` with `U_C` placing utility on the goal symbol `q_g`, `beta_C = 5.0` (goal mass approximately 0.90, off-goal mass approximately 0.10 spread over the other state symbols), and approximately zero mass on non-state tokens.

The outcome `o` at a decision is the next observation token from the task vocabulary. The predictive outcome distribution is `q(o | pi) = E_{q(s | pi)} p(o | s)`, evaluated at the post-action belief, which at v1 is `p(o | mu_s)`. The policy score is `G(pi) = risk(pi) + ambiguity(pi)`, which at the v1 point belief equals the preference cross-entropy `-E_{q(o|pi)}[log p_task(o)]` (Section 2.8), and the policy posterior is `P(pi) = softmax(-gamma G(pi))` over the `Kp` menu. The two control preferences fall out as decomposition controls: `p_data(o | C)` (the empirical next-token distribution) reduces `risk` to NLL and is the control arm EFE must beat, and `p_flat(o | C)` (uniform) gives the constant `log V` at the v1 point belief and is reported as the inert pure-epistemic placeholder until the epistemic-live phase. The default `policy_mode="none"` is the byte-identical reference path.

### 4.2 Pre-registered candidate generator and horizon schedule

The candidate generator is fixed: at each decision the menu is the top-`Kp` next-token set under the current model logits, with `Kp = 8` primary. A sensitivity sweep at `Kp in {4, 16}` is pre-registered as secondary and may not be substituted for the primary should `Kp = 8` fail. The policy precision `gamma` is tuned on a held-out development split of the synthetic task over the fixed grid `gamma in {0.5, 1, 2, 4, 8}` by a single pre-registered rule, argmax dev closed-loop success rate with ties broken toward `gamma = 1`, frozen before any test episode. The temperature-tuned logprob baseline is tuned by the identical rule on the same dev split and the same dev episode-seed list over a matched-cardinality temperature grid, so each comparator carries exactly one scalar tuning degree of freedom and the comparison is fair; fixing `gamma = 1` outright is rejected because it would handicap EFE against a tuned baseline, so the grid stays primary. The untuned canonical precision `gamma = 1` is additionally reported as a mandatory sensitivity point for every gated comparison, so a reader can see whether any win depends on the tuning rather than on the score.

The horizon schedule is staged. The v1 build is one-step, `H = 1`: each token decision is reranked by a single-step lookahead in which the candidate action is applied and the post-action belief and predictive `q(o | pi)` are read once. Deeper lookahead, `H = 2` and `H = 4`, which a probe-then-answer epistemic task needs, is deferred to the cache phase (Phase 3) and gated on first building a belief or key-value cache that makes the recompute cost honest; until then it is not run, and `efe_rollout` with `policy_horizon>1` hard-errors.

### 4.3 Ablation matrix as scheduled arms

The decomposition arms, lesion placebos, and matched-compute baselines are scheduled comparison arms, not merely named falsifiers, and run on identical frozen weights so the comparisons are paired within a checkpoint. The "live at v1" column states whether the arm carries signal at the v1 sigma-free point belief (Section 2.8); arms marked epistemic-live are reported but not gated at v1 and are first claimed in the epistemic-live phase.

| Arm | Definition | Role | Live at v1 |
|---|---|---|---|
| full EFE | `G = risk + ambiguity`, `p_task` preference | the candidate under test | yes (pragmatic) |
| risk-only | `G = risk`, `p_task` | decomposition | yes |
| ambiguity-only | `G = ambiguity` | decomposition; predictive-entropy reranker | yes (confidence, anti-exploratory) |
| epistemic-only | `G = -I` info-gain form | decomposition; sigma-gated | no, `I = 0` at v1 |
| held-out-predictive preference | `p_data`, `risk` reduces to NLL | control; full EFE with `p_task` must beat this | yes |
| flat preference | `p_flat`, `G = log V - I` | pure-epistemic ablation | no, constant `log V` at v1 |
| shuffled-sigma | `sigma` permuted across candidates/positions | placebo; must NOT match true-`sigma` arms | epistemic-live only |
| sign-flipped epistemic | epistemic term negated | placebo; must NOT match full EFE | epistemic-live only |
| random-score | `G` drawn at random | placebo; must be clearly worst | yes |
| temperature-tuned logprob | dev-tuned temperature on current logits | primary matched-compute baseline | yes |
| beam search | matched `Kp` and forward-pass budget | matched-compute baseline | yes |
| best-of-N | `N` matched to `Kp * H` budget | matched-compute baseline | yes |
| predictive-entropy / confidence reranker | rank by next-token entropy or max-prob | cheap near-competitor | yes |
| length-normalized logprob | per-token-normalized sequence logprob | cheap near-competitor | yes |
| top-p (nucleus) | nucleus sampling adapter | sampling baseline | yes |
| locally-typical sampling | typicality-based sampling adapter | sampling baseline | yes |
| greedy / `policy_mode="none"` | unmodified `generate()` | byte-identical reference | yes |

Risk and ambiguity diagnostics are logged separately from raw log-probability for every arm, so a win can be attributed to a component rather than to a relabeled beam score.

### 4.4 Metrics

The primary metric is closed-loop task success rate on the synthetic environment, computed deterministically. The remaining metrics are secondary and reported for every arm. Held-out cross-entropy and perplexity under the policy-adjusted predictive distribution measure whether the policy reweighting helps ordinary language modeling: the one-step policy posterior over the `Kp` menu induces a reweighted next-token distribution whose cross-entropy against the gold token is evaluated on held-out natural text (WikiText, via the existing `test_ce`/`test_ppl` fields on the language-modeling checkpoint), with a synthetic held-out split reported alongside for the task-internal check and the transfer gap noted. Calibration is reported as expected calibration error and the reliability curve (`ece`, `reliability`). Uncertainty signal quality is reported as `sigma`-stratified cross-entropy and the `sigma`-to-cross-entropy Spearman correlation `sigma_ce_spearman`, alongside `sigma_trace_cv`, which at the disjoint 60k-step head-mixer regime sit near 0.176 to 0.19 and 0.11 and have not been measured at the operating point. Generation quality is reported as repetition rate and distinct-n. Length robustness is reported as long-context cross-entropy and perplexity at `N = 192, 256, 384, 512` from the `extrap_ce` evaluation, against the train length of 128. Attention entropy is logged as a behavioral diagnostic. Wall-clock time per token and per episode, with the forward-pass count proxy on the order of `Kp * H`, is the honesty check: any arm that wins only by spending more compute than its matched baselines is reported as such and is not an architectural win.

### 4.5 Seed-floor protocol and the sigma-validation gate

No small-effect claim may be made before the plain-baseline noise floor is measured for the relevant checkpoint family. On the language-modeling operating point (`embed_dim = 20`, 15k steps, linear decode, `use_head_mixer=False`, around 144.5 perplexity) the current ablation data is single-seed (seed 6), and the only multi-seed floor in hand, the `grow_K` scaling sweep, lives in a disjoint 60k-step head-mixer regime with per-cell coefficient of variation between 0.57 and 1.09 percent and a max-minus-min spread up to about 2.1 percent; that floor does not transfer. The protocol therefore requires, before any cross-seed or small-effect claim, training at least three seeds (6, 23, 64 to match the scaling sweep) of the plain baselines at the relevant operating point and computing per-metric mean, standard deviation, and coefficient of variation for the primary success rate and every secondary metric, establishing the per-metric floor band.

The synthetic-task checkpoint family that carries the primary success metric is itself a three-seed set, not a single checkpoint. It is trained with the same seed list (6, 23, 64) and the operating-point architecture (`embed_dim = 20`, linear decode `use_prior_bank=False`, `use_head_mixer=False`) on next-token rendered ring episodes for a sealed fixed budget of 15k optimizer steps, with the final checkpoint taken and no dev-selection knob. Because a model-based planner can only be as good as its forward model, each seed must clear a deterministic predictive-adequacy precondition before it hosts any scorer arm: teacher-forced next-observation (transition) accuracy on held-out episodes at or above 0.98, the deterministic ring transition being easy to fit at this scale. A seed that misses the precondition is excluded and the shortfall reported, never silently dropped. The three admitted seeds supply both the paired within-checkpoint comparisons of Phases 1 and 2 and the cross-seed success-rate floor band, so no separate language-modeling floor is consumed for the primary metric.

For the inference-time phases (Phases 1 through 3) seed variance is removed by construction: EFE and every control and baseline arm run on identical frozen weights, so each comparison is paired within a checkpoint and reported as a paired difference over the same episodes and contexts, with a paired bootstrap or paired permutation test. Cross-seed claims and the deferred train-time phase are the only places that consume the measured floor.

The `sigma`-validation gate is pre-registered and binding. Before any `sigma`-derived quantity may be used as an epistemic or ambiguity signal, before the Monte Carlo ambiguity estimator is unlocked, and before any arm that reads belief covariance (epistemic-only, the `sigma_mc` ambiguity variant, shuffled-sigma as a meaningful contrast) may be claimed rather than merely reported, `sigma` must demonstrably predict realized outcomes on the checkpoint hosting that arm: `sigma_ce_spearman` must reach at least `0.2` with a 95 percent bootstrap confidence interval whose lower bound exceeds zero and the measured floor, `sigma`-stratified cross-entropy must be monotone across strata, and high pre-emission `sigma` must predict higher realized cross-entropy on the emitted token with expected calibration error within `sigma` bins below `0.05`. The gate is currently unmet because it has not been measured at any candidate operating point. If it fails, all `sigma`-derived arms are reported as ablations only and none may be named an epistemic or ambiguity value, and the information-gain term `I` stays at its inert v1 value. The canonical risk and ambiguity terms of Section 4.1 use no `sigma` and are unaffected.

The gate's outcome is recorded as an auditable, versioned artifact, not a transient log. The protocol, the sealed thresholds, and this spec's commit hash live in `docs/research/active-inference/2026-06-28-sigma-gate-prereg.md`, and each measured run writes a machine-readable record to `vfe3_policy_results/sigma_gate/<checkpoint_id>.json` carrying the checkpoint id, the spec commit, the seed list, `sigma_ce_spearman` with its bootstrap confidence interval, the `sigma`-binned expected calibration error, the stratified-cross-entropy monotonicity statistic, the measured floor, and a single `PASS` or `FAIL` stamp. The `VFE3Config` flag `policy_sigma_ambiguity_validated` (default `False`) may be set `True` only together with a `policy_sigma_gate_artifact` reference to a `PASS` record whose spec commit matches; config validation raises if the flag is `True` without a matching passing artifact, so the gate cannot be flipped silently (Section 7, guards 4 and 7).

### 4.6 Pass/fail gates, multiplicity, go/no-go, and demotion

The v1 primary gate is pragmatic and conjunctive. Full EFE with the `p_task` preference must beat, on closed-loop success rate, all of (a) the held-out-predictive `p_data` control, (b) the temperature-tuned logprob baseline, and (c) the matched-compute beam and best-of-N baselines, by a margin exceeding both the measured operating-point seed-floor band and a pre-registered minimum effect size `delta_min = 0.05` absolute success rate, with statistical significance at `alpha = 0.05` after multiple-comparisons correction. The "full EFE beats its best single component" decomposition gate is an epistemic-phase test (it asks whether the ambiguity/epistemic content adds value) and is therefore not part of the v1 pragmatic gate; it is applied in the epistemic-live phase. Significance for the paired inference-time comparisons is a paired bootstrap or permutation test over episodes (and McNemar's test for the paired binary success outcome); cross-seed comparisons use a mixed-effects model with seed as a random effect. Multiplicity over the full arm-by-metric grid is controlled with Benjamini-Hochberg false-discovery-rate control at `q = 0.05`, given the large comparison count, while the small set of primary conjunctive gates uses Holm-Bonferroni at `alpha = 0.05`, so the keep criterion is as strict as each control.

The v1 lesion gates must all hold for the result to count as a genuine pragmatic-EFE effect rather than a relabeled baseline: random-score must be clearly worst, full EFE with `p_task` must beat the `p_data` control, and the closed-loop causality check must pass (the committed action must measurably change the next observation, which the deterministic environment guarantees by construction and the analysis confirms). The epistemic lesion gates (shuffled-sigma must not match true-`sigma` arms; the sign-flipped epistemic arm must not match full EFE; predicted epistemic value must correlate with realized posterior uncertainty reduction) operate on the information-gain term and are deferred to the epistemic-live phase, where `I` is nonzero; applying them at v1 would compare zero against zero and is therefore excluded from the v1 gate set.

The go/no-go gates between phases are explicit. Phase 1 proceeds to Phase 2 only if one-step EFE beats the temperature-tuned logprob baseline on the primary metric beyond the floor. Phase 2 proceeds to Phase 3 only if full EFE with `p_task` beats the `p_data` control and the matched-compute baselines and all v1 lesion gates pass. Phase 3 (epistemic-live: cache, `H >= 2`, sigma-validated Monte Carlo) proceeds to Phase 4 only if a deeper horizon with a live epistemic term helps, or holds, once its compute cost is charged against matched-compute baselines, and only if the epistemic lesion gates and the "full EFE beats its best single component" decomposition gate pass. Phase 4 repeats the surviving arms on the pure KL prior-bank path (`use_prior_bank=True`) as the theoretical-purity confirmation. The train-time auxiliary EFE regularizer is a later phase gated on all prior phases passing, on the `sigma` gate, on a live observation and preference path through the kernels and oracle, and on a covariant rollout where applicable; it is out of scope for v1.

The demotion rule is fixed: if full EFE with `p_task` fails to beat the `p_data` control, the temperature-tuned logprob baseline, and the matched-compute beam and best-of-N baselines on the primary metric beyond the floor and `delta_min`, the feature is demoted to a costly reranker and reported as such, with no rescue.

### 4.7 Fixed falsifiers and pre-registration

The falsifiers are fixed before any run and may not be changed afterward. The v1 (pragmatic) reading is falsified if any of the following holds: matched-compute beam or best-of-N matches or beats full EFE; full EFE with `p_task` fails to beat the `p_data` control; random-score is not clearly worst; or the numerical score reduces to the current free energy plus a diversity heuristic. The epistemic reading, tested only in the epistemic-live phase where `I` is nonzero, is falsified if shuffled-sigma performs as well as true `sigma`, if the sign-flipped epistemic term performs as well as the true epistemic term, if predicted epistemic value does not correlate with realized posterior uncertainty reduction, or if the policy does not causally change future observations in the closed loop. These epistemic falsifiers are explicitly not evaluated at v1, because at the v1 point belief `I` is identically zero and the comparisons would be zero-against-zero and fire spuriously.

The pre-registration is numerically sealed. The pre-registered constants are: ring size `m = 16`, budget `T_ep = 10`, vocabulary `V = 32`, episodes per arm per checkpoint `N_ep = 5000` on a fixed episode-seed list (paired across arms; this gives more than 80 percent power to detect `delta_min = 0.05` at `alpha = 0.05` for a paired binary McNemar comparison at the anticipated success rates); the preferences `p_task` (`beta_C = 5.0`), `p_data`, and `p_flat`; the candidate generator (top-`Kp` by base logit) with `Kp = 8` primary and `{4, 16}` secondary; the precision grid `gamma in {0.5, 1, 2, 4, 8}`; the score weights and term set; the placebos; the Monte Carlo sample count `S = 16`; the synthetic-task checkpoint recipe (seeds 6, 23, 64; `embed_dim = 20`; linear decode; 15k optimizer steps; final checkpoint; predictive-adequacy precondition next-observation accuracy `>= 0.98`); the mandatory `gamma = 1` sensitivity point; and the gate thresholds `delta_min = 0.05`, `alpha = 0.05`, FDR `q = 0.05`, `sigma_ce_spearman >= 0.2` with a bootstrap CI lower bound above zero and the floor, and ECE `<= 0.05`. None of these may be retuned to rescue a failed result. The commit hash of this spec is the pre-registration record, and the analysis is run once against these fixed thresholds.

## 5. Phased plan

Phase 0 (enabling refactor and the sigma measurement). Extract `forward_beliefs` and the public `rollout_beliefs` wrapper (Section 3.1), pinned by the golden byte-identity test that `forward(targets=None)`, the dense and fused-chunked training paths, the `mstep_self_coupling_weight>0` and `mass_phi>0` configs, and the `use_prior_bank=False` ablation return tensors identical to HEAD `5e88afc`. Add the `policy_mode`/preference/ambiguity registries and the `VFE3Config` fields, all default-off, validated like the other seam keys. Migrate the diagnostics belief-production replays to `forward_beliefs` where byte-identity holds. Run, but do not yet gate on, the sigma-validation measurement at the operating point, recording `sigma_ce_spearman`, `sigma`-stratified cross-entropy, and the calibration statistics to the gate artifact of Section 4.5, so the gate of Section 2.7 has a measured value before any `sigma`-derived arm is claimed.

Phase 1 (one-step pragmatic reranker on a frozen checkpoint). Implement `efe_one_step` and `logprob_control`, the `generate()` guarded branch, and `_policy_select`. Train the three-seed synthetic-task checkpoint set on the ring dynamics (Section 4.5), admitting only seeds that clear the predictive-adequacy precondition. Run the controlled closed-loop ring task with `p_task`, `H = 1`, `Kp = 8`, the dev-tuned `gamma`, the sigma-free point-belief ambiguity. Compare against the temperature-tuned logprob baseline and the `p_data` control on success rate with the paired test. Proceed only if the Phase 1 go/no-go gate of Section 4.6 passes.

Phase 2 (decomposition and matched-compute controls). Run the full ablation matrix of Section 4.3, including the risk-only, ambiguity-only, flat-preference, and held-out-predictive arms (the inert arms reported, not gated), random-score, and the beam and best-of-N matched-compute baselines. Apply the v1 lesion gates and the conjunctive pragmatic gate with FDR control. Proceed only if full EFE with `p_task` beats the `p_data` control and the matched-compute baselines and all v1 lesion gates pass.

Phase 3 (epistemic-live: horizon and state-uncertainty, after the cache). Build the belief or key-value cache that makes `Kp * H` recompute honest, then unlock `efe_rollout` for `H = 2, 4` and, if the sigma gate has passed, the `sigma_mc` ambiguity estimator. This is where the deferred partially observed masked key-value retrieval task runs: each episode presents a table of key-to-value pairs with the target value masked and the query `ANSWER target_key`; emitting `PROBE key` makes the environment reveal that key's value as a new observation, and emitting `ANSWER value` commits and terminates. The only reliable route to success is to probe the target then answer, which makes the information-gain term load-bearing, and which is realizable only at `H >= 2` with a live epistemic term, never at v1. Run the epistemic lesion gates (shuffled-sigma, sign-flipped epistemic, epistemic-value-versus-realized-uncertainty-reduction) and the "full EFE beats its best single component" decomposition gate here, add the long-context and sample metrics, and charge the deeper-horizon cost against the matched-compute baselines. Proceed only if a deeper horizon with a live epistemic term helps or holds at matched compute and the epistemic gates pass.

Phase 4 (pure prior-bank path). Repeat the surviving arms on `use_prior_bank=True`, where preferences live in the generative decode space, as the theoretical-purity confirmation.

Phase 5 (train-time EFE, gated and deferred). A train-time auxiliary EFE regularizer that wires a live observation and preference path through `gradients/kernels.py`, `gradients/oracle.py`, and the `free_energy.log_likelihood` term (currently the inert gated stub at `free_energy.py:341`, `401-402`). It is gated on all prior phases passing, on the sigma gate, and on a covariant rollout where applicable, and is out of v1 scope.

## 6. Agent-set / community sub-experiment (spec'd but deferred)

The agent-set or community case treats whole agents, or emergent token communities, as the policies, with their joint outcomes as the observation space. It is the more principled home for active inference because the model (`s`/`gamma`) channel carries the inter-agent structure a community policy would act on. It is deferred and is a separate experiment with a separate outcome space, separate metrics, and separate baselines; no empirical conclusion from the token-continuation experiment of Section 4 may be carried into it, and none of its conclusions may be carried back (Section 7, guard 6).

Two preconditions block it. First, it requires an explicit community outcome model, an observation distribution over community-level outcomes that does not exist in source. Second, it requires a non-flat covariant rollout over the model channel: the model-channel transport is hardcoded flat (`transport_mode="flat"`, `model.py:1047`), and the covariant routes (`regime_ii` / `regime_ii_covariant`) carry the `_omega_builder` closure over `belief.sigma` (`e_step.py:426-435`) whose covariant-oracle behavior must be built and tested before any community-policy claim. Until both exist, no non-flat covariant community-policy claim over the model channel may be made (Section 7, guard 5). The full A/B/C/D/E instantiation for the community case, its outcome model, its candidate generator over agents or communities, its preference, and its metrics, belongs in its own future document and is not specified here beyond this deferral.

The deferral is actionable rather than open-ended. The agent-set sub-experiment is unblocked only when all of the following hold, and at that point its full A/B/C/D/E contract, community outcome model, candidate generator over agents or communities, preference, metrics, baselines, and falsifiers are authored in a separate document `docs/superpowers/specs/<date>-active-inference-agentset-spec.md`: (i) an explicit community outcome observation model `p(o_community | s)` exists and is validated; (ii) a non-flat covariant rollout over the model (`s`/`gamma`) channel exists, replacing the hardcoded `transport_mode="flat"` at `model.py:1047`; (iii) the covariant-oracle `_omega_builder` sigma-leaf behavior (`e_step.py:426-435`) is fixed and tested for `regime_ii` and `regime_ii_covariant`; and (iv) the token-continuation experiment of Section 4 has itself cleared at least its Phase 2 gates, so the agent-set case builds on a validated scorer rather than an untested one. Until every item holds, no non-flat covariant community-policy claim over the model channel may be made.

## 7. Must-not-claims guardrails

The spec asserts none of the following seven claims and structurally prevents each.

Guard 1, that V3 already implements an EFE or active-inference scorer. The scorer is new, default-off code; `policy_mode='none'` is the default and leaves `forward()` and `generate()` byte-identical to HEAD `5e88afc`, pinned by the Phase 0 golden test. The status banner and every docstring state the scorer does not run by default.

Guard 2, that EFE efficacy or any perplexity or generation-quality improvement has been shown. Efficacy is the hypothesis under test with the pre-registered demotion rule of Section 4.6; no result is asserted anywhere in this document, and the raw continuation log-prob is kept in a separate `PolicyScore.log_prob` field, never folded into the score the policy acts on. The spec further states plainly that at v1 the scorer is a pragmatic reranker and the epistemic term is inert, so no v1 result may be read as evidence about epistemic active inference.

Guard 3, that a train-time EFE replacement is justified. The train-time variant is Phase 5, gated on all prior phases plus the sigma gate plus a live observation path; the v1 scorer runs entirely under `@torch.no_grad` and is never wired into `gradients/kernels.py`, `gradients/oracle.py`, or `free_energy.log_likelihood`.

Guard 4, calling V3 `sigma` an ambiguity or epistemic value without validation. The default ambiguity is `likelihood_entropy`, computed from the decode at the belief mean with no `sigma`; the `sigma_mc` variant and the live information-gain term raise or stay inert unless `policy_sigma_ambiguity_validated=True`, which is forbidden until the pre-registered gate of Sections 2.7 and 4.5 passes. The spec states that at the v1 point belief the information-gain term is identically zero, so no epistemic claim is even arithmetically available at v1.

Guard 5, non-flat covariant community-policy claims over the model channel. v1 is flat token continuations only; the non-flat covariant rollout and the agent-set case are deferred to Section 6, and `efe_rollout` with `policy_horizon>1` hard-errors until the cache lands.

Guard 6, sharing conclusions between the continuation and agent-set experiments. They are separate registry keys and separate experiments with separate outcome spaces, metrics, and baselines; Section 6 fences the agent-set case as its own document.

Guard 7, post-hoc rescue by changing preferences, horizon, candidates, or proxies after a failed result. The preferences, horizon schedule, candidate generator and `Kp`, `gamma` grid, score weights, placebos, the task-scale constants, and every numeric threshold are sealed in the config pre-registration surface (Section 3.2) and in Section 4.7, with the spec commit hash as the pre-registration record and the analysis run once against fixed thresholds.

## 8. Resolved cross-facet decisions and remaining open items

The following conflicts across the design facets are resolved as stated. The v1 marginalization of `q(o | pi)` uses the sigma-free point belief `s = mu` until the sigma gate passes, with fixed-`S = 16` Monte Carlo unlocked only thereafter. The transition B is reconciled to a single definition: the scored rolled-out belief appends the candidate action only and never the environment's response, because folding the realized response into the score is oracle peeking rather than prospective planning (Section 2.2); the contract-completeness review's proposed direction, making Section 2.2 match the earlier Section 4.1 by appending `env_response`, is therefore declined, and Section 4.1 is corrected to match Section 2.2 instead. As a consequence the v1 task is re-scoped: the masked key-value probe task is unsolvable and epistemically inert at `H = 1`, so the v1 primary task is the fully observed ring goal-steering task whose pragmatic payoff is realizable at one step, and the probe task moves to the epistemic-live phase (Section 5, Phase 3). At the v1 point belief the information-gain term `I` is identically zero, so `PolicyScore.epistemic`, the epistemic-only arm, and the flat-preference reduction `G = log V - I` are marked inert and reported, not gated; the epistemic falsifiers are gated to the epistemic-live phase. The flat-preference arm is defined once as `G = risk + ambiguity = log V - I` (Section 2.3), which is the constant `log V` at the v1 point belief; the earlier "ambiguity alone" restatement is removed. The flat-preference sign is corrected to `risk = log V - H[q(o | pi)]`, superseding the inverted "be-confident" reading. The `model` handle sits immediately after the tensor block in `policy_scorer` as a no-default positional, with bound-callable passing recorded as a rejected alternative. The diagnostics replays migrate to `forward_beliefs` in Phase 0 where byte-identity holds, while the load-bearing byte-identity requirement remains only on `forward()` and `generate()`. The canonical information-gain diagnostic is the cheap MI bridge `I = H[q(o | pi)] - E_q H[p(o | s)]`, not the Bayesian-posterior estimator. The policy precision `gamma` is dev-tuned over the fixed grid `{0.5, 1, 2, 4, 8}` as primary, to match the dev-tuned logprob baseline's single tuning degree of freedom. Multiplicity uses Benjamini-Hochberg FDR over the full arm-by-metric grid and Holm-Bonferroni over the primary conjunctive gates. The held-out language-modeling check uses WikiText on the language-modeling checkpoint with a synthetic held-out split reported alongside. `p_task` is defined over the immediate next observation for v1 (`H = 1`), with the terminal-episode form deferred. The candidate generator is fixed to top-`Kp` by base logit; nucleus and locally-typical appear only as separate sampling baseline arms, not as candidate generators. `efe_rollout` with `H > 1` hard-errors until the cache lands. The sigma gate flag is `policy_sigma_ambiguity_validated` in `VFE3Config`, default `False`. The numeric pre-registration is sealed in Section 4.7: `delta_min = 0.05`, `alpha = 0.05`, FDR `q = 0.05`, `sigma_ce_spearman >= 0.2` with a bootstrap CI excluding zero and the floor, ECE `<= 0.05`, `beta_C = 5.0`, ring `m = 16`, `T_ep = 10`, `V = 32`, `N_ep = 5000`, `S = 16`.

The four items previously left open are now resolved and folded into the pre-registration. The `gamma` tuning fork is resolved by keeping the dev-tuned grid `{0.5, 1, 2, 4, 8}` as primary, under a single pre-registered selection rule identical to the temperature-tuned baseline's, with `gamma = 1` added as a mandatory sensitivity point (Section 4.2); fixing `gamma = 1` outright is rejected because it would handicap EFE against a tuned baseline. The synthetic-task checkpoint recipe is resolved to a three-seed set (seeds 6, 23, 64) at the operating-point architecture, a sealed 15k-step budget, final-checkpoint evaluation, and a deterministic predictive-adequacy precondition (Section 4.5), with all constants sealed in Section 4.7. The sigma-gate recording is resolved to a versioned pre-registration document plus a machine-readable per-checkpoint JSON record bound to the config flag, so the gate cannot be flipped without a matching passing artifact (Section 4.5). The agent-set case stays deferred by design, not open: its unblock checklist and the document that will carry its full A/B/C/D/E are fixed in Section 6. Nothing in the v1 pre-registration now depends on an unresolved choice.

*Provenance: drafted 2026-06-28 by a multi-agent workflow (3 design facets: code/API, theory/contract, experiment/eval -> synthesize -> contract-completeness + adversarial review -> reconciled finalize; 7 agents). Grounded in the investigation `docs/research/2026-06-28-active-inference-buildout-plan-investigation.md`, the plan `docs/research/2026-06-27-active-inference-policy-investigation.md`, and the binding debate `docs/debates/2026-06-27-active-inference-lm-efficacy/`. Both reviews returned PASS_WITH_FIXES; all meritorious fixes applied. Verified against HEAD 5e88afc.*

## Pre-registration amendments

Three amendments were made during Phase 1 implementation, before the official sealed run, after seed-6
smoke runs showed the pre-registration as written was mechanically broken (the EFE score could not
drive selection). They fix flaws in operationalizing this spec's own intent, not post-hoc rescues; no
conclusion was drawn from the broken runs. Recorded in
`docs/research/active-inference/2026-06-28-prereg-amendments.md`: (A1) the candidate generator for the
ring control task is the three action tokens, not top-Kp (Section 4.2), since top-Kp admits non-action
tokens that freeze the agent; (A2) `p_task`'s non-state mass is a finite floor, not exactly zero
(Section 4.1, "approximately zero"), because exact zero makes the forward KL diverge; (A3) the random
lesion gate is "full EFE beats random by more than `delta_min`", not strict global-argmin (Sections
4.6/4.7), because at v1 all goalless arms cluster near random. All other sealed constants (Section 4.7)
are unchanged.
