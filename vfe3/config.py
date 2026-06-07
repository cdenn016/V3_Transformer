"""Configuration for VFE_3.0. Single dataclass, single validation block.

No CLI parsing (project policy: click-to-run). Edit fields directly, then run.
Every registry-backed seam is selected here by name (divergence, gauge group,
encode/decode mode, alpha form, attention prior, norm, gradient mode), so a
variant swaps without editing call sites.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

# Seams with a live registry (gauge_group, alpha_mode, attention priors, norms; alongside
# transport/retraction/positional/divergence) are validated against that registry in __post_init__
# via tuple(sorted(_REGISTRY)), so a newly registered variant is a valid config value WITHOUT
# editing this validator (the add-by-registering modularity contract). The static tuples below are
# the seams WITHOUT a registry, plus the two intentional second-gates (encode/decode modes, whose
# extra members 'gauge_fixed'/'linear' are reached through use_prior_bank / are NotImplementedError
# stubs, not through these fields).
_VALID_GAUGE_PARAM         = ("phi", "omega_direct")
_VALID_ENCODE_MODES        = ("per_token", "gauge_fixed")
_VALID_DECODE_MODES        = ("diagonal", "diagonal_chunked", "full")
_VALID_GRADIENT_MODES      = ("filtering", "smoothing")
_VALID_PHI_PRECOND_MODES   = ("none", "clip", "killing", "killing_per_block", "pullback")
_VALID_PHI_RETRACT_MODES   = ("euclidean", "bch")
_VALID_POS_PHI_COMPOSE     = ("bch", "euclidean")
_VALID_PRIOR_SOURCES       = ("token", "model_channel")
_VALID_E_STEP_GRADIENTS    = ("unroll", "straight_through", "detach")


@dataclass
class VFE3Config:
    """The single authoritative configuration surface for VFE_3.0.

    Fields are grouped: numerics, divergence seam, model structure, gauge seam,
    belief family, free-energy coupling, attention, E-step, decode/encode, cross-
    block handoff, normalization, and M-step / training. Each ``*_mode`` /
    ``*_family`` / ``*_group`` field is a registry key. Validation runs once in
    ``__post_init__`` in field-declaration order.
    """

    # numerics
    eps:                       float = 1e-6
    kl_max:                    float = 100.0

    # divergence seam
    divergence_family:         str   = "renyi"        # divergence FUNCTIONAL (renyi, ...); alpha_div selects member
    alpha_div:                 float = 1.0            # Renyi order (alpha=1 -> KL)

    # model structure
    vocab_size:                int   = 50257
    embed_dim:                 int   = 64           # K (total belief dimension)
    max_seq_len:               int   = 128          # N (context length)
    n_layers:                  int   = 1            # L (number of blocks)
    n_e_steps:                 int   = 1            # T (E-step inner iterations)
    n_heads:                   int   = 8

    # gauge seam
    gauge_group:               str   = "block_glk"
    gauge_parameterization:    str   = "phi"
    
    # Connection REGIME (registry key): the flat Regime-I phi-cocycle ('flat', default = the pure
    # NO-NN path) vs the non-flat Regime II ('regime_ii'). ORTHOGONAL to gauge_parameterization,
    # which picks how a single flat transport is parameterized; this picks whether the connection is
    # flat at all. Validated against the transport registry below.
    # NEURAL-NETWORK EXCEPTION (sanctioned, default-OFF): 'regime_ii' introduces a LEARNED bilinear
    # edge connection delta_ij^a = mu_i^T W^a mu_j, with W a model-owned nn.Parameter (shape
    # (n_gen, K, K)) trained by backprop on CE -- a documented learned-parameter exception in the
    # spirit of use_head_mixer / alpha_mode='learnable' (see VFEModel.__init__'s connection_W and
    # transport._build_regime_ii). At W=0 (init) or cocycle_relaxation=0 it reduces EXACTLY to the
    # flat cocycle, so init is byte-flat. The pure no-NN path is 'flat' (the default).
    transport_mode:            str   = "flat"
    
    # Homotopy alpha for the Regime-II connection (regime_ii only): delta_ij^a = cocycle_relaxation *
    # (mu_i^T W^a mu_j). 0.0 -> delta=0 -> flat (Regime I); 1.0 -> fully relaxed (Regime II). Ignored
    # by the flat builder.
    cocycle_relaxation:        float = 1.0
   
    # Cross-head GL(K) coupling: a list of directed (head_a, head_b) index pairs that add off-block
    # generators (and a genuinely larger-than-direct-sum subalgebra under the builder's bracket
    # closure) to the gauge basis. Default None = the current block-diagonal GL(d_head)^H gauge.
    # Only a group whose builder accepts the kwarg (block_glk) supports it; validated below.
    cross_couplings:           Optional[List[Tuple[int, int]]] = None

    # head mixer (opt-in, default off): a learned Schur-commutant matrix mixes the equal-size
    # gauge-irrep blocks (under block_glk: the n_heads heads) of the converged belief. Identity
    # init -> step-0 bit-identical to off. Breaks strict gauge equivariance under block_glk's
    # untied per-block gauge (exact at init, deviates as the mixer drifts); the tied_block_glk
    # group restores EXACT equivariance on the full-covariance path. Needs a group with >= 2 equal
    # blocks (block_glk / tied_block_glk), else VFEModel construction raises.
    use_head_mixer:            bool  = False

    # BCH positional encoding (default "learned"): a per-position Lie-algebra element pos_phi_i
    # composed into the token gauge frame via compose_phi BEFORE transport. "learned" owns a model
    # parameter table (max_seq_len, n_gen); "frozen" is the parameter-free i*pos_phi_scale on one
    # axis. The theoretically PURE no-composition path is "none". Validated against the pos_phi
    # registry.
    pos_phi:                   str   = "learned"      # "none" | "learned" | "frozen"
    pos_phi_compose:           str   = "bch"       # composition chart: bch (default) | euclidean
    bch_pe_order:              int   = 4           # BCH Dynkin truncation order (compose_phi order)
    pos_phi_scale:             float = 0.02        # learned-table init scale AND frozen per-position step
    pos_phi_project_slk:       bool  = False       # per-block trace projection (det Omega = 1)

    # gauge-RoPE (default-off): a block-diagonal positional rotation R(theta) folded into the
    # transport (Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T). Means-only by default;
    # rope_full_gauge=True also rotates the covariance sandwich and REQUIRES full covariance.
    pos_rotation:              str   = "none"      # "none" | "rope" (the positional-rotation registry)
    rope_base:                 float = 100.0       # rotary frequency base
    rope_full_gauge:           bool  = False       # rotate covariance too (needs diagonal_covariance=False)

    # belief family
    diagonal_covariance:       bool  = True
    family:                    str   = "gaussian_diagonal"

    # free-energy coupling
    alpha:                     float = 1.0          # constant self-coupling value
    # alpha_mode selects the self-coupling form (registry key). The default-and-pure no-NN forms
    # are 'constant', 'state_dependent', 'state_dependent_per_coord' (closed-form functions of the
    # self-divergence D, no learned parameters) and are unchanged. NEURAL-NETWORK EXCEPTION:
    # 'learnable' introduces a model-owned scalar nn.Parameter log_alpha (alpha = exp(log_alpha))
    # trained by backprop -- a sanctioned, default-OFF learned-parameter exception in the spirit of
    # use_head_mixer / use_prior_bank (see VFEModel.__init__ and alpha_i.alpha_learnable). At init
    # log_alpha=0 -> alpha=1.0, byte-identical to constant alpha=1.0.
    alpha_mode:                str   = "constant"
    b0:                        float = 1.0          # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0:                        float = 1.0          # state-dependent alpha shape (numerator)
    kappa:                     float = 1.0          # temperature tau = kappa * sqrt(K)
    # lambda_beta weights the ENTIRE belief-coupling block of F -- sum_ij [ beta_ij E_ij +
    # tau beta_ij log(beta_ij/pi_ij) ] -- relative to the alpha self-coupling and the likelihood
    # (VFE_2.0 'lambda_align' parity). 1.0 = the canonical/pure F (byte-identical). It scales the
    # POST-softmax block (NOT the energy inside the softmax), so beta = softmax(-E/tau) is unchanged
    # and the analytic kernel stays envelope-consistent with the autograd oracle.
    lambda_beta:               float = 1.0
    # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): a LEARNED lambda_beta. When True the model
    # creates a scalar nn.Parameter log_lambda_beta (lambda_beta = exp(log_lambda_beta)) trained by
    # backprop through the unrolled E-step -- the spirit of alpha_mode='learnable'. Init 0 ->
    # lambda_beta = 1.0, byte-identical to the constant-1.0 pure path at step 0. Default False keeps
    # the path param-free.
    learnable_lambda_beta:     bool  = False
    mass_phi:                  float = 0.0          # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight: float = 0.0         # alpha_hat: overall scale on M-step sum_i alpha_i D(q_i*||p_i) (0 = OFF; alpha_i = the E-step self-coupling form)
    # Hyper-prior weight lambda_h on the model-channel term lambda_h * mean_i KL(s_i||r)
    # (manuscript Participatory_it_from_bit.tex eq:pointwise_free_energy, lines 1241-1249).
    # Default 0.0 = OFF: no s/r tables, loss byte-identical to the single-tier path.
    # This wires the second (model) belief channel s_i + the global hyper-prior r end-to-end; s_i
    # does NOT couple into the belief q / the prediction path (the gamma block below shares the same
    # s tables and is likewise predictively inert). The s->q coupling and the s-channel E-step update
    # remain DEFERRED.
    lambda_h:                  float = 0.0

    # Model-coupling weight gamma_coupling on the model-channel block gamma_coupling * mean_i F_red^s_i,
    # the reduced (envelope) form of sum_ij [ gamma_ij KL(s_i||Omega_ij s_j) + tau_g gamma_ij
    # log(gamma_ij/pi^s_ij) ] (manuscript Participatory_it_from_bit.tex eq:pointwise_free_energy,
    # lines 1241-1249). Default 0.0 = OFF. SECOND INCREMENT: a TRAINING-LOSS regularizer on the
    # model-channel s tables with TIED, DETACHED transport (Omega_tilde = Omega from the converged
    # belief phi.detach()), so the gamma gradient reaches ONLY the s tables and the forward
    # (logits/ce) is byte-identical to the gamma=0 path -- the model channel stays predictively INERT
    # (s does NOT feed q). The detach deliberately severs the phi<-gamma coupling that full tied
    # transport would carry; restoring it is part of the deferred s->q design. gamma_coupling>0 ALONE
    # creates the s tables (the r tables stay hyper-prior-only). NB: the mean over (B, H, N) makes
    # gamma_coupling=1 a per-token-per-head mean weight, NOT the canonical sum-over-ij; the scale is a
    # free coupling.
    gamma_coupling:            float = 0.0
    kappa_gamma:               float = 1.0          # model-channel temperature tau_gamma = kappa_gamma*sqrt(d_head)
    gamma_attention_prior:     str   = "causal"     # pi^s_ij seam for the model channel (its own prior)

    # s->q coupling: REPLACE the belief prior with the model channel, p_i = s_i. This realizes the
    # SAME-SCALE hierarchical-Bayes prior of GL(K)_supplementary.tex:1083-1085 (p_i(k_i) = integral
    # p_i(k_i|m_i) s_i(m_i) dm_i; p_i = s_i is the identity-conditional special case, K_model=K).
    # THEORETICAL TENSION (disclosed, not settled): the main Participatory_it_from_bit.tex:1440 instead
    # makes p_i a CROSS-SCALE shadow -- the meta-agent's belief q^(s+1) transported into agent i's frame
    # -- and states "s_i does not act through p_i at the same scale" (there s_i is regulated only by its
    # own hyper-prior r_i). This toggle deliberately takes the supplementary's same-scale reading; the
    # cross-scale realization would need a meta-agent/scale-(s+1) object that does not exist yet.
    # prior_source selects which table supplies p_i, CONSISTENTLY across encode (q_i(0)=p_i), the E-step
    # self-coupling target alpha*KL(q_i||p_i), AND the decode per-vocab readout -KL(q||p_v). "token"
    # (default): the belief table mu_embed/sigma_log_embed -- byte-identical pure path. "model_channel":
    # the model-channel s tables (coupled by gamma/lambda_h) ARE the belief prior, so the model channel
    # drives predictions; phi stays the belief table (tied, B_state=B_model). Forces the s tables to
    # exist; mu_embed is unused (dead) on this path. NB: with gamma_coupling=lambda_h=0, model_channel is
    # a pure RENAME of mu_embed (s plays mu_embed's role, CE-trained) -- ZERO added capacity; the model
    # channel changes predictions only once gamma>0 / lambda_h>0 shape s beyond CE. Oracle: s copied from
    # the belief prior tables -> byte-identical to "token".
    prior_source:              str   = "token"       # "token" | "model_channel"

    # attention
    include_attention_entropy: bool  = True         # canonical (True) vs surrogate (False)
    attention_prior:           str   = "causal"

    # E-step
    e_mu_lr:                   float = 0.5
    e_sigma_lr:                float = 0.015
    e_phi_lr:                  float = 0.0
    e_sigma_q_trust:           float = 5.0
    sigma_max:                 float = 5.0
   
    gradient_mode:             str   = "filtering"
    
    phi_precond_mode:          str   = "none"
    phi_retract_mode:          str   = "euclidean"  # Lie-algebra step chart: euclidean (sum) or bch
    spd_retract_mode:          str   = "spd_affine" # SPD covariance retraction geometry (registry key)

    # decode / encode
    use_prior_bank:            bool  = True
    decode_tau:                float = 1.0
    decode_mode:               str   = "diagonal"
    
    # decode_chunk_size: vocabulary-chunk width V is iterated over by the fused
    # decode_mode='diagonal_chunked' CE path (the training-path memory win that never
    # materializes the (B,N,V) logit tensor). Ignored by every other decode_mode. Default
    # 8192; validated positive.
    decode_chunk_size:         int   = 8192
    encode_mode:               str   = "per_token"

    # cross-block belief handoff (mu_q -> mu_p)
    prior_handoff_rho:         float = 1.0          # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma:       float = 0.0          # sigma damping (0.0 = frozen at embedding)

    # normalization
    norm_type_block:           str   = "none"
    norm_type_final:           str   = "none"

    # M-step / training
    # E-step backward estimator (manuscript Algorithm 1, GL(K)_attention.tex:2050). Three modes:
    #   'unroll'          (default): fully differentiate through the inner trajectory -- the
    #                     gradient keeps the second-order d delta/d belief_prev terms.
    #   'straight_through': each inner update computes its tangent DETACHED but rebuilds the
    #                     belief grad-connected to the previous belief (mu_next = mu_prev +
    #                     delta.detach(), sigma_next = retract(sigma_prev, delta.detach())), so
    #                     d belief_next/d belief_prev = I flows WITHOUT the second-order term --
    #                     the manuscript's detached-snapshot-plus-live-additive-chain estimator
    #                     (the phi step is already straight-through; this matches it for mu/sigma).
    #                     Forward VALUE is byte-identical to 'unroll'; only the BACKWARD differs.
    #   'detach'          : the whole E-step under torch.no_grad (the legacy detach_e_step=True
    #                     behavior) -- no E-step gradient at all.
    # Reconciled with detach_e_step: the EFFECTIVE mode is 'detach' when detach_e_step=True
    # (back-compat), else e_step_gradient. detach_e_step=True with a non-'unroll' e_step_gradient
    # is contradictory and raises in __post_init__. The default (unroll + detach_e_step=False) is
    # the current fully-unrolled path, byte-identical.
    e_step_gradient:           str   = "unroll"
    detach_e_step:             bool  = False        # False = unroll E-step in the training graph
    # Opt-in (default OFF): make the autograd belief-gradient ORACLE (used for the non-kernel
    # families: gradient_mode='smoothing', family='gaussian_full', alpha_div!=1) return a
    # create_graph (differentiable) gradient under e_step_gradient='unroll', so the unrolled-through-
    # inference signal reaches the prior tables -- matching the closed-form kernel, which already does
    # this. Default OFF preserves the detached oracle (the gradient is truncated for those families,
    # the long-standing behavior), so the default path is unchanged. CAVEAT: this builds a SECOND-ORDER
    # graph through the E-step; it is numerically stable for the DIAGONAL non-kernel families
    # (smoothing / alpha_div!=1) but the full-covariance eigh/cholesky double-backward can produce NaN
    # gradients, so leave OFF for gaussian_full (or expect NaNs there).
    oracle_unroll_grad:        bool  = False
    m_mu_lr:                   float = 0.025
    m_sigma_lr:                float = 0.0025
    m_phi_lr:                  float = 0.015
    weight_decay:              float = 0.05
    batch_size:                int   = 64
    
    # Accumulate gradients over N microbatches before an optimizer step, for a larger
    # effective batch without the memory of one big forward. Each pulled batch is split
    # into N equal chunks along the batch axis, each backed (loss / N) into .grad, then a
    # single clip + optimizer.step() + scheduler.step() fires at the boundary. Default 1 =
    # current single-step behavior (byte-identical: no chunking, no divide).
    grad_accum_steps:          int   = 1
    max_steps:                 int   = 15000
    warmup_steps:              int   = 100
    min_lr:                    float = 1e-4         # absolute cosine-decay floor: each group's LR
    #                          never decays below this. 0.0 recovers the pure half-cosine-to-zero.
    min_lr_frac:               float = 0.0           # fractional cosine-decay floor (default OFF):
    #                          each group's LR never decays below min_lr_frac * its OWN base LR,
    #                          preserving the m_mu:m_sigma:m_phi base ratios into the tail. Combined
    #                          with min_lr as max(min_lr, min_lr_frac*base). 0.0 (with min_lr=0) is
    #                          the pure half-cosine-to-zero path.
    
    seed:                      int   = 0
    log_interval:              int   = 50           # console log every N steps (0 = off)
    eval_interval:             int   = 0            # periodic validation every N steps (0 = off)
    checkpoint_interval:       int   = 0            # save a resumable checkpoint every N steps (0 = off)
    eval_max_batches:          Optional[int] = None # cap the PERIODIC eval pass (None = full split; pure path)
    generate_figures:          bool  = True         # auto-run the single-run publication figures at finalize_run (off the hot path)
    
    # Opt-in mixed precision for CUDA throughput (RTX 5090). None (default) = OFF = the pure fp32
    # path: NO autocast context is entered anywhere in the forward, so the loss/logits are
    # byte-identical to the no-AMP build. 'bf16' / 'fp16' wrap the E-step / belief pipeline in
    # torch.autocast. The cancellation-sensitive decode matmul (_decode_diagonal) AND the
    # cross-entropy stay fp32 even when AMP is on (their inputs are .float()-ed and they run under
    # an autocast(enabled=False) island), as do the existing matrix_exp / SPD-retraction islands
    # (transport.py / retraction.py). TF32 is intentionally NOT used here: its 10-bit mantissa
    # worsens the decode's catastrophic cancellation and breaks the atol-1e-3 decode pin (see
    # docs/perf/2026-05-31-speedup-opportunities.md, "Rejected: global TF32"); bf16/fp16 autocast
    # is the safe alternative precisely because those sensitive ops opt out. bf16 needs no
    # GradScaler and is the recommended default for the 5090; fp16 TRAINING would need a GradScaler
    # in the M-step (train.py) -- a deferred follow-up (this toggle is forward/inference-correctness
    # scope).
    amp_dtype:                 Optional[str] = None

    def __post_init__(self) -> None:
        # numerics
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.kl_max <= 0.0:
            raise ValueError(f"kl_max must be positive, got {self.kl_max}")

        # divergence seam: divergence_family is the FUNCTIONAL (f-divergence) registry key
        # (renyi, squared_hellinger, ...), distinct from `family` (the covariance-structure
        # kernel). alpha_div is the Renyi order, IGNORED by non-alpha functionals (e.g.
        # squared_hellinger). Both are live, modular seams (CLAUDE.md: slot in different
        # f-divergences). Validated against the functional REGISTRY (not a hardcoded literal
        # list) so a newly registered functional is config-selectable without editing here;
        # local import avoids a config <- divergence <- families import cycle.
        from vfe3.divergence import divergence_functionals
        _require(self.divergence_family, divergence_functionals(), "divergence_family")
        if self.alpha_div <= 0.0:
            raise ValueError(f"alpha_div must be positive, got {self.alpha_div}")

        # model structure
        for name in ("vocab_size", "embed_dim", "max_seq_len", "n_heads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.n_e_steps < 1:
            raise ValueError(f"n_e_steps must be >= 1, got {self.n_e_steps}")
        if self.embed_dim % self.n_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by n_heads={self.n_heads}"
            )

        # gauge seam. Validated against the group REGISTRY (not a hardcoded literal) so a newly
        # registered group is a valid config value without editing this validator -- the modularity
        # contract (add-by-registering), matching transport_mode / spd_retract_mode below. Local
        # import avoids a config <- groups import cycle.
        from vfe3.geometry.groups import _GROUPS
        _require(self.gauge_group, tuple(sorted(_GROUPS)), "gauge_group")
        # Sp(2m,R) lives in even dimension K = 2m; reject an odd embed_dim at construction with a
        # clear message rather than letting generate_sp raise mid-build.
        if self.gauge_group == "sp" and self.embed_dim % 2 != 0:
            raise ValueError(
                f"gauge_group='sp' (Sp(2m,R)) requires an EVEN embed_dim (K=2m), got "
                f"embed_dim={self.embed_dim}"
            )
        _require(self.gauge_parameterization, _VALID_GAUGE_PARAM, "gauge_parameterization")
        # transport_mode selects the connection REGIME. Validated against the transport REGISTRY
        # (not a hardcoded literal list) so a newly registered regime is a valid config value
        # without editing this validator. Default 'flat' is the Regime-I phi-cocycle (the pure path
        # always exists); Regime II is design-spec'd and deferred. Local import avoids a
        # config <- transport <- groups import cycle (matching the retraction pattern below).
        from vfe3.geometry.transport import _TRANSPORTS
        _require(self.transport_mode, tuple(sorted(_TRANSPORTS)), "transport_mode")
        # 'omega_direct' (Omega_ij = Omega_i Omega_j^{-1} for general GL(K), det possibly < 0)
        # needs a per-token K x K group element Omega_i. The no-NN belief carries only phi
        # (n_gen Lie-algebra coords), from which the only constructible Omega_i is exp(embed(phi_i))
        # -- making Omega_ij = exp(phi_i) exp(-phi_j), identical to the 'phi' path. There is no
        # source for a non-exponential / det<0 GL(K) element in the belief, so the mode is rejected
        # (live + enforced) rather than silently aliased to 'phi'.
        if self.gauge_parameterization == "omega_direct":
            raise NotImplementedError(
                "gauge_parameterization='omega_direct' needs a per-token GL(K) matrix Omega_i, "
                "but the no-NN belief carries only phi (Lie-algebra coords); exp(phi) reduces it to "
                "the 'phi' path. Use 'phi'."
            )
        # cross_couplings (off-block GL(K) head coupling) is supported only by a group builder that
        # accepts the kwarg (block_glk). Reject otherwise (glk / so_k / tied_block_glk) by inspecting
        # the registered builder's signature, not a hardcoded name list, so support tracks the
        # builders. Each pair must be distinct in-range directed head indices. Local imports avoid a
        # config <- groups <- closure import cycle (matching the divergence / alpha_i / retraction
        # pattern below).
        if self.cross_couplings is not None:
            import inspect
            from vfe3.geometry.groups import get_group
            builder = get_group(self.gauge_group)
            if "cross_couplings" not in inspect.signature(builder).parameters:
                raise ValueError(
                    f"cross_couplings is not supported by gauge_group={self.gauge_group!r} (its "
                    f"builder does not accept the kwarg); use 'block_glk', or leave cross_couplings "
                    f"None"
                )
            if not isinstance(self.cross_couplings, list):
                raise ValueError(
                    f"cross_couplings must be a list of (int, int) head pairs, got "
                    f"{type(self.cross_couplings).__name__}"
                )
            # JSON has no tuple type, so a config.json reloaded by viz.report._load_config hands
            # back list pairs; coerce them to tuples so the round-trip rebuild does not trip the
            # isinstance(pair, tuple) gate below (and downstream builders see a consistent type).
            self.cross_couplings = [tuple(p) if isinstance(p, list) else p
                                    for p in self.cross_couplings]
            for pair in self.cross_couplings:
                if (not isinstance(pair, tuple) or len(pair) != 2
                        or not all(isinstance(x, int) for x in pair)):
                    raise ValueError(
                        f"each cross_couplings entry must be an (int, int) head pair, got {pair!r}"
                    )
                a, b = pair
                if a == b:
                    raise ValueError(f"cross_couplings self-coupling ({a},{a}) not allowed (a != b)")
                if not (0 <= a < self.n_heads and 0 <= b < self.n_heads):
                    raise ValueError(
                        f"cross_couplings head indices ({a},{b}) out of range [0, {self.n_heads})"
                    )

        # belief family. ``family`` selects the covariance-structure divergence kernel
        # (gaussian_diagonal | gaussian_full). ``divergence_family`` is the SEPARATE functional
        # (f-divergence) seam (renyi, ...; validated above), and ``diagonal_covariance`` is a
        # SEPARATE live bool, cross-validated to stay consistent with family. The three are kept
        # distinct and modular per CLAUDE.md (slot in different f-divergences / families); NOT
        # collapsed, and divergence_family is NOT forced equal to family.
        # family is validated against, and its covariance structure read from, the divergence
        # registry (not a hardcoded list / name literal), so a newly registered family is a valid
        # config family and its diagonal-vs-full structure is its declared cov_kind.
        from vfe3.divergence import divergence_families, family_cov_kind
        _require(self.family, divergence_families(), "family")
        family_is_diagonal = family_cov_kind(self.family) == "diagonal"
        if self.diagonal_covariance != family_is_diagonal:
            raise ValueError(
                f"diagonal_covariance={self.diagonal_covariance} contradicts family={self.family!r}; "
                f"set diagonal_covariance={family_is_diagonal} for this family"
            )

        # free-energy coupling
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.mass_phi < 0.0:
            raise ValueError(f"mass_phi must be >= 0, got {self.mass_phi}")
        if self.mstep_self_coupling_weight < 0.0:
            raise ValueError(
                f"mstep_self_coupling_weight must be >= 0, got {self.mstep_self_coupling_weight}"
            )
        if self.lambda_beta < 0.0:
            raise ValueError(f"lambda_beta must be >= 0, got {self.lambda_beta}")
        if self.lambda_h < 0.0:
            raise ValueError(f"lambda_h must be >= 0, got {self.lambda_h}")
        if self.gamma_coupling < 0.0:
            raise ValueError(f"gamma_coupling must be >= 0, got {self.gamma_coupling}")
        if self.kappa_gamma <= 0.0:
            raise ValueError(f"kappa_gamma must be > 0, got {self.kappa_gamma}")
        # attention priors validated against the prior REGISTRY (add-by-registering). Local import
        # avoids a config <- attention_prior cycle; the bound name is reused for attention_prior below.
        from vfe3.attention_prior import _PRIORS
        _require(self.gamma_attention_prior, tuple(sorted(_PRIORS)), "gamma_attention_prior")
        _require(self.prior_source, _VALID_PRIOR_SOURCES, "prior_source")
        for name in ("b0", "c0"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        # alpha_mode validated against the alpha-form REGISTRY (add-by-registering). Local import
        # avoids a config <- alpha_i cycle.
        from vfe3.alpha_i import _ALPHAS
        _require(self.alpha_mode, tuple(sorted(_ALPHAS)), "alpha_mode")
        # A per-coordinate alpha form (state_dependent_per_coord) weights each coordinate's
        # self-divergence by its own alpha^(k), which needs a per-coordinate self-divergence.
        # That decomposition exists only for the diagonal family (full-covariance KL couples
        # coordinates through the trace and log-determinant), so reject the inconsistent pair at
        # construction rather than letting the per-coordinate divergence raise mid-forward.
        from vfe3.alpha_i import alpha_is_per_coord
        if alpha_is_per_coord(self.alpha_mode) and not family_is_diagonal:
            raise ValueError(
                f"alpha_mode={self.alpha_mode!r} needs a per-coordinate self-divergence, which "
                f"exists only for a diagonal-covariance family; got family={self.family!r}. Use "
                f"a diagonal family (e.g. 'gaussian_diagonal') or a per-position alpha_mode."
            )
        # The per-coordinate self-divergence is registered only for the Renyi functional (KL = Renyi
        # at alpha=1); free_energy.self_divergence_per_coord raises for any other divergence_family.
        # Reject the non-Renyi pair at construction too -- mirroring that runtime raise -- so this
        # doubly-opt-in path fails fast at config time rather than only at the first forward (the
        # covariance half is rejected just above).
        if alpha_is_per_coord(self.alpha_mode) and self.divergence_family != "renyi":
            raise ValueError(
                f"alpha_mode={self.alpha_mode!r} needs a per-coordinate self-divergence, which is "
                f"implemented for the 'renyi' functional only (KL = Renyi at alpha=1); got "
                f"divergence_family={self.divergence_family!r}. Use divergence_family='renyi' or a "
                f"per-position alpha_mode."
            )

        # attention
        _require(self.attention_prior, tuple(sorted(_PRIORS)), "attention_prior")

        # E-step
        for name in ("e_mu_lr", "e_sigma_lr", "e_phi_lr"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        _require(self.gradient_mode, _VALID_GRADIENT_MODES, "gradient_mode")
        _require(self.phi_precond_mode, _VALID_PHI_PRECOND_MODES, "phi_precond_mode")
        _require(self.phi_retract_mode, _VALID_PHI_RETRACT_MODES, "phi_retract_mode")
        from vfe3.model.positional_phi import _POS_PHI
        _require(self.pos_phi, tuple(sorted(_POS_PHI)), "pos_phi")
        _require(self.pos_phi_compose, _VALID_POS_PHI_COMPOSE, "pos_phi_compose")
        from vfe3.geometry.rope import _POS_ROTATIONS
        _require(self.pos_rotation, tuple(sorted(_POS_ROTATIONS)), "pos_rotation")
        if self.rope_full_gauge and self.diagonal_covariance:
            raise ValueError(
                "rope_full_gauge=True rotates the covariance sandwich (R Sigma R^T), which the "
                "diagonal-covariance approximation cannot carry; set diagonal_covariance=False."
            )
        # RoPE rotates ADJACENT coordinate pairs (2k, 2k+1); Sp(2m) pairs coordinate i with m+i
        # (J = [[0,I],[-I,0]]), so the rope-wrapped transport R Omega R^T leaves the symplectic group.
        # R is orthogonal (a subset of GL(K)), so the GL(K)-congruence divergence invariance and VFE
        # equivariance survive -- the operator is still valid, just no longer in the SELECTED Sp
        # structure group. Warn (not error): the symplectic property is not consumed downstream.
        if self.pos_rotation == "rope" and self.gauge_group == "sp":
            import warnings
            warnings.warn(
                "pos_rotation='rope' with gauge_group='sp' leaves the symplectic group: RoPE rotates "
                "adjacent pairs (2k, 2k+1) while Sp(2m) pairs i with m+i, so R Omega R^T is no longer "
                "in Sp(2m). The GL(K)-congruence divergence invariance still holds (R is orthogonal), "
                "so the model runs; only the structure-group claim is dropped.",
                UserWarning,
                stacklevel=2,
            )
        # spd_retract_mode selects the SPD covariance retraction geometry. Validated against the
        # retraction REGISTRY (not a hardcoded literal list) so a newly registered retraction is a
        # valid config value without editing this validator. Default 'spd_affine' is the
        # manuscript-canonical affine-invariant exponential map (the pure path always exists).
        from vfe3.geometry.retraction import _RETRACTIONS
        _require(self.spd_retract_mode, tuple(sorted(_RETRACTIONS)), "spd_retract_mode")
        # log_euclidean is a genuinely new variant only for the full-covariance family. On a
        # diagonal family it applies the tangent in the matrix-log chart WITHOUT the affine
        # 1/sigma Fisher whitening, so under this seam's pre-whitened tangent convention it does
        # NOT coincide with the manuscript-canonical 'spd_affine' there -- it is a non-canonical
        # log-chart step. Warn (not error) so the harmless-but-non-canonical pairing surfaces; the
        # user toggles families and retraction modes independently.
        if self.spd_retract_mode == "log_euclidean" and family_is_diagonal:
            import warnings
            warnings.warn(
                "spd_retract_mode='log_euclidean' with a diagonal-covariance family applies the "
                "tangent in the log chart without the affine 1/sigma Fisher whitening, so it does "
                "NOT reduce to the canonical 'spd_affine' here (it is a non-canonical log-chart "
                "step). log_euclidean is a genuinely new variant only for a full-covariance "
                "family; prefer 'spd_affine' on the diagonal family, or use 'gaussian_full'.",
                UserWarning,
                stacklevel=2,
            )
        # 'killing_per_block' builds a per-HEAD Killing metric and requires generators that partition
        # per block (block_glk's independent gl(d_head) per head). The tied gauge's shared generators
        # kron(I_n, gl(d_head)) each act on EVERY block, so the per-block partition does not exist;
        # reject at construction (it otherwise fails cryptically inside the first E-step). The ambient
        # 'killing', 'clip', and 'none' preconditioners are unaffected.
        if self.gauge_group == "tied_block_glk" and self.phi_precond_mode == "killing_per_block":
            raise ValueError(
                "phi_precond_mode='killing_per_block' is incompatible with gauge_group="
                "'tied_block_glk': the shared kron(I_n, gl(d)) generators do not partition per head, "
                "so the per-block Killing metric is undefined. Use 'none', 'clip', or the ambient "
                "'killing'."
            )

        # decode / encode
        if self.decode_tau <= 0.0:
            raise ValueError(f"decode_tau must be positive, got {self.decode_tau}")
        _require(self.decode_mode, _VALID_DECODE_MODES, "decode_mode")
        # decode_mode sets the RANK of the prior-bank KL-decode kernel: 'diagonal'/'diagonal_chunked'
        # consume a diagonal sigma (B,N,K); 'full' consumes a full sigma (B,N,K,K). It must agree with
        # the covariance family, else the rank mismatch is a shape RuntimeError at the first forward.
        # Only the prior-bank decode reads decode_mode; the use_prior_bank=False linear decode ignores
        # it (sigma discarded), so the cross-check is gated on use_prior_bank.
        if self.use_prior_bank and (self.decode_mode == "full") == family_is_diagonal:
            raise ValueError(
                f"decode_mode={self.decode_mode!r} is rank-incompatible with family={self.family!r}: "
                f"'full' decode needs a full-covariance family and 'diagonal'/'diagonal_chunked' decode "
                f"needs a diagonal family. Pair decode_mode='full' with a full family, use a diagonal "
                f"decode_mode with a diagonal family, or set use_prior_bank=False (linear decode)."
            )
        if self.decode_chunk_size < 1:
            raise ValueError(f"decode_chunk_size must be >= 1, got {self.decode_chunk_size}")
        _require(self.encode_mode, _VALID_ENCODE_MODES, "encode_mode")
        # use_prior_bank is the SINGLE decode gate. True (default, pure path): the KL-to-prior
        # readout logits = -KL(q_i || pi_v)/tau_eff over the gauge-orbit prior bank, with the
        # covariance structure selected by decode_mode (diagonal | full). False (VFE_2.0-parity
        # ablation): decode is a plain linear projection mu_q -> logits via a learned (V, K)
        # weight (sigma discarded) -- the one authorized neural exception (a single linear output
        # readout; see CLAUDE.md). Encode and the free-energy self-coupling stay on the PriorBank
        # either way; decode_mode then only describes the (unused) KL structure, so the two knobs
        # cannot silently disagree. The pure KL path always exists under use_prior_bank=True.
        # 'gauge_fixed' encode (gauge orbit from a shared base belief) is a named stub: reject at
        # construction so the failure is at config time, not the first forward pass.
        if self.encode_mode == "gauge_fixed":
            raise NotImplementedError(
                "encode_mode='gauge_fixed' is a named stub (gauge orbit from a shared base belief) "
                "that is not yet implemented; use 'per_token'."
            )

        # handoff (both blends must be convex so the prior stays on the SPD cone:
        # sigma_p_next = (1-rho_s) sigma_p + rho_s sigma_q stays > 0 iff rho_s in [0,1])
        if not (0.0 <= self.prior_handoff_rho <= 1.0):
            raise ValueError(f"prior_handoff_rho must be in [0,1], got {self.prior_handoff_rho}")
        if not (0.0 <= self.prior_handoff_sigma <= 1.0):
            raise ValueError(f"prior_handoff_sigma must be in [0,1], got {self.prior_handoff_sigma}")
        # cocycle_relaxation is the regime_ii homotopy weight delta = cocycle_relaxation * (...) in
        # transport._build_regime_ii (0 -> flat cocycle, 1 -> fully relaxed). It had no guard and feeds
        # the connection directly, so a NaN/inf/out-of-range value propagated silently. The bracketed
        # form also rejects NaN (nan <= 1.0 is False) and +/-inf, unlike a bare `< 0` check.
        if not (0.0 <= self.cocycle_relaxation <= 1.0):
            raise ValueError(f"cocycle_relaxation must be in [0,1], got {self.cocycle_relaxation}")

        # normalization validated against the norm REGISTRY (add-by-registering). Local import
        # avoids a config <- norms cycle.
        from vfe3.geometry.norms import _NORMS
        _valid_norms = tuple(sorted(_NORMS))
        _require(self.norm_type_block, _valid_norms, "norm_type_block")
        _require(self.norm_type_final, _valid_norms, "norm_type_final")

        # M-step / training
        # e_step_gradient selects the E-step backward estimator (unroll | straight_through |
        # detach). Reconciled with the legacy detach_e_step bool WITHOUT breaking it: the
        # EFFECTIVE mode is 'detach' when detach_e_step=True (back-compat), else e_step_gradient
        # (see effective_e_step_gradient). detach_e_step=True with a non-'unroll' e_step_gradient
        # asks for two different things at once, so reject it at construction.
        _require(self.e_step_gradient, _VALID_E_STEP_GRADIENTS, "e_step_gradient")
        if self.detach_e_step and self.e_step_gradient != "unroll":
            raise ValueError(
                f"detach_e_step=True is contradictory with e_step_gradient="
                f"{self.e_step_gradient!r}: detach_e_step already forces the effective mode to "
                f"'detach'. Set detach_e_step=False and use e_step_gradient to select the mode, "
                f"or leave e_step_gradient='unroll'."
            )
        # straight_through detaches the per-iteration E-step tangent, so a learnable parameter whose
        # only loss path IS that tangent receives no gradient and silently freezes. Warn (non-breaking;
        # 'unroll' is the default that trains them) rather than restrict the toggle combination.
        if self.e_step_gradient == "straight_through" and (
            self.alpha_mode == "learnable"
            or self.transport_mode == "regime_ii"
            or self.learnable_lambda_beta
        ):
            import warnings
            warnings.warn(
                "e_step_gradient='straight_through' detaches the per-iteration E-step tangent, so a "
                "learnable parameter that enters the loss only through it (log_alpha under "
                "alpha_mode='learnable', connection_W under transport_mode='regime_ii', log_lambda_beta "
                "under learnable_lambda_beta) receives NO gradient and stays frozen. Use "
                "e_step_gradient='unroll' (the default) to train these.",
                UserWarning,
                stacklevel=2,
            )
        for name in ("m_mu_lr", "m_sigma_lr", "m_phi_lr", "weight_decay", "min_lr", "min_lr_frac"):
            v = getattr(self, name)
            if v < 0.0 or v != v:                            # v != v rejects NaN (which passes < 0.0)
                raise ValueError(f"{name} must be >= 0 (and not NaN), got {v}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.log_interval < 0:
            raise ValueError(f"log_interval must be >= 0, got {self.log_interval}")
        if self.eval_interval < 0:
            raise ValueError(f"eval_interval must be >= 0, got {self.eval_interval}")
        if self.checkpoint_interval < 0:
            raise ValueError(f"checkpoint_interval must be >= 0, got {self.checkpoint_interval}")
        if self.eval_max_batches is not None and self.eval_max_batches < 1:
            raise ValueError(f"eval_max_batches must be >= 1 if set, got {self.eval_max_batches}")
        # amp_dtype: None (default, OFF) = pure fp32 / no autocast; 'bf16' / 'fp16' enable autocast.
        # None is a legal member here, so _require rejects 'fp32' and any other garbage.
        _require(self.amp_dtype, (None, "bf16", "fp16"), "amp_dtype")

    @property
    def tau(self) -> float:
        """Per-head softmax-temperature convenience value tau = kappa * sqrt(d_head).

        NOTE: this is a config-level convenience (used for logging). The ACTIVE attention
        temperature is computed group-aware by ``free_energy.attention_tau(kappa, group.irrep_dims)``,
        which keys off the dimension the energy accumulates over -- the gauge-irrep BLOCK: sqrt(K)
        for a single-block group (glk/so_k/sp, irrep_dims=[K]) and sqrt(d_head) for per-head
        multi-block (block_glk). This property equals the active tau only for an equal-block group
        whose block size is d_head (the default block_glk); on a single-block group it understates it
        by sqrt(n_heads). kappa=1 recovers the Vaswani sqrt(d_k) temperature over the energy dimension.
        """
        return self.kappa * (self.d_head ** 0.5)

    @property
    def tau_gamma(self) -> float:
        """Model-channel softmax temperature tau_gamma = kappa_gamma * sqrt(d_head).

        The gamma model-coupling block's own temperature handle, mirroring `tau` for the belief
        beta block (kappa_gamma=1 -> Vaswani sqrt(d_k) per head). Consumed by the gamma block's
        reduced_free_energy as the -tau_gamma log Z^s envelope temperature.
        """
        return self.kappa_gamma * (self.d_head ** 0.5)

    @property
    def d_head(self) -> int:
        """Per-head belief dimension K // n_heads."""
        return self.embed_dim // self.n_heads

    @property
    def effective_e_step_gradient(self) -> str:
        """The E-step backward estimator actually used, reconciling detach_e_step.

        detach_e_step=True forces 'detach' (the legacy whole-E-step-under-no_grad behavior);
        otherwise the chosen e_step_gradient ('unroll' | 'straight_through' | 'detach') applies.
        The contradictory pair (detach_e_step=True with non-'unroll' e_step_gradient) is rejected
        in __post_init__, so this never silently overrides a meaningful e_step_gradient.
        """
        return "detach" if self.detach_e_step else self.e_step_gradient


def _require(value: Optional[str], valid: tuple, name: str) -> None:
    """Raise ValueError unless ``value`` is one of ``valid``."""
    if value not in valid:
        raise ValueError(f"{name} must be one of {valid}, got {value!r}")
