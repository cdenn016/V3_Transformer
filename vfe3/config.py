"""Configuration for VFE_3.0. Single dataclass, single validation block.

No CLI parsing (project policy: click-to-run). Edit fields directly, then run.
Every registry-backed seam is selected here by name (divergence, gauge group,
encode/decode mode, alpha form, attention prior, norm, gradient mode), so a
variant swaps without editing call sites.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

# Seams with a live registry (gauge_group, lambda_alpha_mode, attention priors, norms; alongside
# transport/retraction/positional/divergence) are validated against that registry in __post_init__
# via tuple(sorted(_REGISTRY)), so a newly registered variant is a valid config value WITHOUT
# editing this validator (the add-by-registering modularity contract). phi_precond_mode,
# phi_retract_mode and pos_phi_compose are ALSO validated against their live registries
# (phi_preconditioner._PRECOND, lie_ops._COMPOSE) for the same reason, as are encode/decode modes
# (_ENCODERS/_DECODERS; their extra members 'gauge_fixed'/'linear' are reached through
# use_prior_bank / are NotImplementedError second-gates after the registry check). The static
# tuples below are the remaining seams WITHOUT a registry.
_VALID_GAUGE_PARAM         = ("phi", "omega_direct")
_VALID_GRADIENT_MODES      = ("filtering", "smoothing")
_VALID_PRIOR_SOURCES       = ("token", "model_channel")
_VALID_E_STEP_GRADIENTS    = ("unroll", "straight_through", "detach")
_VALID_GAUGE_TRANSPORT     = ("on", "off", "frozen")


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
    divergence_family:         str   = "renyi"        # divergence FUNCTIONAL (renyi, ...); renyi_order selects member
    renyi_order:               float = 1.0            # Renyi order (alpha=1 -> KL)

    # model structure
    vocab_size:                int   = 50257
    embed_dim:                 int   = 64           # K (total belief dimension)
    max_seq_len:               int   = 128          # N (context length)
    n_layers:                  int   = 1            # L (number of blocks)
    n_e_steps:                 int   = 1            # T (E-step inner iterations)
    n_heads:                   int   = 8

    # belief-table init scales (PriorBank.__init__): mu_v ~ N(0, mu_init_std^2), every coordinate
    # variance set to the constant sigma_init (stored as log(sigma_init); not random spread), and the
    # gauge frame phi_v ~ N(0, phi_scale^2). mu_init_std and phi_scale may be 0 (deterministic zero
    # table); sigma_init must be > 0 for the log.
    mu_init_std:               float = 0.02         # std of the random mean table mu_embed
    sigma_init:                float = 1.0          # constant initial coordinate variance (sigma_log = log of this)
    phi_scale:                 float = 0.01         # std of the random gauge-frame table phi_embed

    # gauge seam
    gauge_group:               str   = "block_glk"
    gauge_parameterization:    str   = "phi"          # RESERVED axis: 'phi' is the sole live value; 'omega_direct' is rejected at validation (~line 682) until implemented.

    # gauge_transport: ablation meta-toggle over the GL(K) gauge FRAME (the A1 / EXP-2 gauge on/off/
    # frozen experiment; docs/hypotheses/2026-06-21-hypotheses.md). DISTINCT from transport_mode below
    # (flat vs the Regime-II connection): transport_mode decides the connection's flatness, this decides
    # whether the per-token frame exists/learns at all. 'on' (default, PURE PATH): the frame is whatever
    # phi_scale / e_phi_lr / m_phi_lr / pos_phi specify (learned when m_phi_lr>0) -- byte-identical to
    # omitting this toggle. 'off': forces Omega_ij = exp(phi_i) exp(-phi_j) = I EXACTLY by coercing
    # phi_scale=0.0, pos_phi='none', e_phi_lr=0.0, m_phi_lr=0.0 (and REQUIRING pos_rotation='none' +
    # transport_mode='flat', else Omega != I -- raises). 'frozen': a RANDOM but FIXED frame (keeps the
    # set phi_scale>0, coerces e_phi_lr=0.0, m_phi_lr=0.0). __post_init__ applies the coercion and warns.
    # NB the 'off' cell drops the pos_phi_free table (pos_phi='none'), so to match parameter counts
    # against an 'on' baseline set pos_phi='none' in 'on' too, or compare against the equal-param 'frozen'.
    gauge_transport:           str   = "on"          # "on" (pure, learned) | "off" (Omega=I) | "frozen" (random fixed)

    # Connection REGIME (registry key): the flat Regime-I phi-cocycle ('flat', default = the pure
    # NO-NN path) vs the non-flat Regime II ('regime_ii'). ORTHOGONAL to gauge_parameterization,
    # which picks how a single flat transport is parameterized; this picks whether the connection is
    # flat at all. Validated against the transport registry below.
    # NEURAL-NETWORK EXCEPTION (sanctioned, default-OFF): 'regime_ii' introduces a LEARNED bilinear
    # edge connection delta_ij^a = mu_i^T W^a mu_j, with W a model-owned nn.Parameter (shape
    # (n_gen, K, K)) trained by backprop on CE -- a documented learned-parameter exception in the
    # spirit of use_head_mixer / lambda_alpha_mode='learnable' (see VFEModel.__init__'s connection_W and
    # transport._build_regime_ii). At cocycle_relaxation=0 the flat builder's dict is returned
    # byte-identically; at W=0 (the zero-tensor init) the generic path reduces to the flat cocycle
    # to fp32 tolerance (atol 1e-6, pinned -- not bit-exact: the exp(0)=I einsum reorders fp32 ops;
    # audit 2026-06-10 F11). The pure no-NN path is 'flat' (the default). NOTE: regime_ii's belief
    # gradient is served by the autograd ORACLE (the closed-form kernel is the flat-transport
    # gradient and would drop d Omega/d mu), so training connection_W through the unrolled E-step
    # requires oracle_unroll_grad=True (the freeze warning below fires otherwise).
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

    # Close the cross-coupled gauge basis under the Lie bracket (close_under_brackets) so it is a
    # genuine Lie SUBALGEBRA. When a cross_couplings chain spans 3+ heads the raw chain basis is NOT
    # bracket-closed: [E_ab, E_bc] = E_ac lands OUTSIDE the spanned generators, so the pullback
    # structure constants and the BCH composition silently TRUNCATE those out-of-span terms (the
    # transport composes in a non-closed span). Closing the basis adds the missing bracket-generated
    # generators so composition stays inside the algebra. Default None = AUTO: closure is applied iff
    # cross_couplings is set (model.build_group resolves the auto default and reads this field;
    # config only declares it). True / False force it on / off.
    close_basis:               Optional[bool] = None

    # SO(N)/Sp(2m) irrep-tower groups ('so_n' / 'sp_n'): the structure group is SO(group_n)
    # (resp. Sp(group_n), group_n = 2m even) with group_n DECOUPLED from embed_dim. The embedding
    # carries the direct sum of the irreps in irrep_spec -- (label, multiplicity) pairs laid out
    # contiguously in order; heads = irrep blocks (possibly UNEQUAL dims). so_n labels 'l<p>' are
    # the symmetric-traceless rank-p tensor irreps (group_n=3: the spin-p tower, dim 2p+1); sp_n
    # labels 'sym<p>' are Sym^p of the defining rep (dim C(2m+p-1, p)). sum(mult * dim) must equal
    # embed_dim. One shared per-token phi (n_gen = dim of the algebra) drives every block -- a
    # TIED gauge. Both fields are consumed ONLY by these two groups (rejected otherwise).
    group_n:                   Optional[int] = None
    
    # annotation reflects the POST-coercion type: __post_init__ also accepts the JSON/TOML
    # round-trip list-of-lists form ([["l1", 3], ...]) and coerces each pair to a tuple.
    irrep_spec:                Optional[List[Tuple[str, int]]] = None

    # head mixer (opt-in, default off): a learned Schur-commutant matrix mixes the equal-size
    # gauge-irrep blocks (under block_glk: the n_heads heads) of the converged belief. Identity
    # init -> step-0 bit-identical to off. Breaks strict gauge equivariance under block_glk's
    # untied per-block gauge (exact at init, deviates as the mixer drifts); the tied_block_glk
    # group restores EXACT equivariance on the full-covariance path. Needs a group with >= 2
    # blocks (block_glk / tied_block_glk heads, or an so_n/sp_n tower); rejected at config
    # validation otherwise.
    use_head_mixer:            bool  = False

    # CG cross-type coupling (opt-in, default off; so_n/sp_n only): bilinear Clebsch-Gordan
    # between-block update on the means, exactly equivariant for any weights; sigma untouched
    # (means-only phase; see the 2026-06-09 design spec). NEURAL-NETWORK EXCEPTION (sanctioned,
    # default-off): learned scalar path weights, zero-init (step 0 byte-identical).
    use_cg_coupling:           bool  = False

    # BCH positional encoding (default "learned"): a per-position Lie-algebra element pos_phi_i
    # composed into the token gauge frame via compose_phi BEFORE transport. "learned" owns a model
    # parameter table (max_seq_len, n_gen); "frozen" is the parameter-free i*pos_phi_scale on one
    # axis. The theoretically PURE no-composition path is "none". Validated against the pos_phi
    # registry.
    pos_phi:                   str   = "learned"      # "none" | "learned" | "frozen"
    pos_phi_compose:           str   = "bch"       # composition chart: bch (default) | euclidean
    
    bch_pe_order:              int   = 4           # BCH Dynkin truncation order (compose_phi order) 4 is just as good as 6
   
    pos_phi_scale:             float = 0.02        # learned-table init scale AND frozen per-position step
    pos_phi_project_slk:       bool  = False       # per-block trace projection (det Omega = 1)

    # gauge-RoPE (default-off): a block-diagonal positional rotation R(theta) folded into the
    # transport (Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T). Means-only by default;
    # rope_full_gauge=True also rotates the covariance sandwich and REQUIRES full covariance.
    pos_rotation:              str   = "none"      # "none" | "rope" (the positional-rotation registry)
    rope_base:                 float = 100.0       # rotary frequency base
    rope_full_gauge:           bool  = False       # rotate covariance too (needs family="gaussian_full")
    # rope_on_value=True (default) is the coherent single-gauge path: the gauge-RoPE rotation feeds
    # BOTH the attention score and the value aggregation. =False factors the transport into an
    # attention gauge and a value gauge (GL(K)_attention.tex:1909): the score keeps the rotation but
    # the value aggregation mu_hat_i = sum_j beta_ij Omega_ij mu_j uses the UN-rotated base -- RoPE's
    # "position-dependent attention, position-independent values" asymmetry. Decoupled breaks beta's
    # coupling-sum stationarity, so the belief gradient routes to the autograd oracle (no closed-form
    # kernel). Inert unless pos_rotation='rope'.
    rope_on_value:             bool  = True        # False -> value aggregation uses the un-rotated base (RoPE on Q/K only)

    # belief family. ``family`` is the SINGLE covariance-structure toggle (a registry key;
    # gaussian_diagonal | gaussian_full | ...). The diagonal-vs-full flag is its derived,
    # read-only ``diagonal_covariance`` property (see below) -- one source of truth, no second
    # field to keep in sync.
    family:                    str   = "gaussian_diagonal"

    # free-energy coupling
    lambda_alpha:              float = 1.0          # constant self-coupling value
    
    # lambda_alpha_mode selects the self-coupling form (registry key). The default-and-pure no-NN forms
    # are 'constant', 'state_dependent', 'state_dependent_per_coord' (closed-form functions of the
    # self-divergence D, no learned parameters) and are unchanged. NEURAL-NETWORK EXCEPTION:
    # 'learnable' introduces a model-owned scalar nn.Parameter log_alpha (alpha = exp(log_alpha))
    # trained by backprop -- a sanctioned, default-OFF learned-parameter exception in the spirit of
    # use_head_mixer / use_prior_bank (see VFEModel.__init__ and alpha_i.alpha_learnable). At init
    # log_alpha=0 -> alpha=1.0, byte-identical to constant alpha=1.0.
    lambda_alpha_mode:         str   = "constant"
    
    b0:                        'float | List[float]' = 1.0   # state-dependent alpha shape: alpha* = c0/(b0 + D); list -> (K,) per-coord
    c0:                        'float | List[float]' = 1.0   # state-dependent alpha shape (numerator); list -> (K,) per-coord
    kappa_beta:                'float | List[float]' = 1.0   # sharpness; list (len n_heads) -> per-head tau
   
    # lambda_beta weights the ENTIRE belief-coupling block of F -- sum_ij [ beta_ij E_ij +
    # tau beta_ij log(beta_ij/pi_ij) ] -- relative to the alpha self-coupling and the likelihood.
    # 1.0 = the canonical/pure F (byte-identical). It scales the
    # POST-softmax block (NOT the energy inside the softmax), so beta = softmax(-E/tau) is unchanged
    # and the analytic kernel stays envelope-consistent with the autograd oracle.
    lambda_beta:               float = 1.0
    
    # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): a LEARNED lambda_beta. When True the model
    # creates a scalar nn.Parameter log_lambda_beta (lambda_beta = exp(log_lambda_beta)) trained by
    # backprop through the unrolled E-step -- the spirit of lambda_alpha_mode='learnable'. Init 0 ->
    # lambda_beta = 1.0, byte-identical to the constant-1.0 pure path at step 0. Default False keeps
    # the path param-free.
    learnable_lambda_beta:      bool  = False
    
    mass_phi:                   float = 0.0          # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight: float = 0.0         # alpha_hat: overall scale on M-step sum_i alpha_i D(q_i*||p_i) (0 = OFF; alpha_i = the E-step self-coupling form)
    
    # Hyper-prior weight lambda_h on the model-channel term lambda_h * mean_i KL(s_i||r)
    # (manuscript Participatory_it_from_bit.tex eq:pointwise_free_energy, lines 1241-1249).
    # Default 0.0 = OFF: no s/r tables, loss byte-identical to the single-tier path.
    # This wires the second (model) belief channel s_i + the global hyper-prior r end-to-end; s_i
    # does NOT couple into the belief q / the prediction path (the gamma block below shares the same
    # s tables and is likewise predictively inert). The s->q coupling and the s-channel E-step update
    # remain DEFERRED.
    lambda_h:                  float = 0.0

    # Un-freeze the global hyper-prior centroid r (r_mu/r_sigma_log). Default False = FROZEN (current
    # behavior, byte-identical): a fixed centroid the model beliefs s_i are regularized toward, the
    # stand-in for the manuscript's "higher, slower meta-level". True trains r by gradient (grouped in
    # build_optimizer like the s tables, mean@m_p_mu_lr / log-scale@m_p_sigma_lr) as an empirical-Bayes
    # population centroid -- meaningful ONLY when s carries an independent data force
    # (prior_source='model_channel' routes the CE gradient into s). With s unanchored the only force on
    # s/r is lambda_h*KL(s||r), whose joint optimum is s_i=r=const (KL->0, the regularizer vanishes);
    # __post_init__ warns for that regime. The token-dependent top-down r_i=Omega_tilde[s^(s+1)]
    # (PIFB eq:cross_scale_shadow) needs a GENUINELY EMERGED scale-(s+1) meta-agent and is OUT OF SCOPE for
    # this single-scale transformer (PIFB lines 554/636 treat single-scale r_i as a PRIMITIVE boundary;
    # line 2334 assigns the full transport to MAgent_Model) -- NOT a deferred gap. learnable_r is the
    # same-scale empirical-Bayes stand-in (frozen-vs-learned axis; still token-uniform).
    learnable_r:               bool  = False

    # How the un-frozen centroid r is updated (only consulted when learnable_r=True; ignored when r
    # is frozen). 'gradient' (default): r trains by the AdamW M-step like the s/embedding tables
    # (byte-identical to the pre-toggle learnable_r behavior). 'barycenter': r is set each M-step to
    # the closed-form forward-KL barycenter (population centroid) of the s tables --
    # r_mu = mean_v s_mu_v, r_sigma = mean_v[s_sigma_v + (s_mu_v - r_mu)^2] -- the exact minimizer of
    # the UNIFORM-over-vocab sum_v KL(s_v||r) (the same closed-form-stationary-point treatment
    # alpha*/beta*/gamma* already receive), so r is NOT grouped in the optimizer and never receives a
    # gradient. CAVEAT (audit 2026-06-13): this is a uniform-over-VOCABULARY empirical-Bayes centroid,
    # one row per vocab type. The SCORED hyper-prior term reduces with mean() over (B,N) token
    # OCCURRENCES, i.e. the frequency-weighted sum_v f_v KL(s_v||r); the uniform barycenter equals that
    # argmin only when token frequencies are uniform (for a Zipfian vocab the two minimizers differ).
    # So even in the scored s_e_step=False regime the barycenter is the exact M-step of the uniform-vocab
    # objective, NOT of the frequency-weighted scored loss. Under s_e_step=True (r coupled to the CE
    # through the unrolled _refine_s) it is further only a consistent population target, not the argmin
    # -- use r_update_mode='gradient' there. See the 2026-06-13 r/lambda_h spec.
    r_update_mode:             str   = "gradient"   # "gradient" | "barycenter"

    # lambda_h_mode selects the hyper-prior coupling form (the model-fiber analogue of lambda_alpha_mode;
    # registry vfe3/lambda_h_i.py). The default-and-pure forms are 'constant' (lambda_h = cfg.lambda_h,
    # today's bare scalar) and 'state_dependent' (the closed-form envelope lambda_h*_i =
    # c0_h/(b0_h + KL(s_i||r)), which REQUIRES R_h(lambda_h)=b0_h*lambda_h - c0_h*log lambda_h added to
    # F and the s E-step for the envelope cancellation -- threaded automatically through both paths).
    # 'state_dependent_per_coord' is the per-coordinate sibling (lambda_h^(k)* = c0_h^(k)/(b0_h^(k)+KL_k),
    # mirroring lambda_alpha_mode's per-coord form): each model coordinate is shrunk toward r by its own
    # envelope weight, fed the UNSUMMED per-coordinate KL_k(s||r). It needs a coordinate-decomposable
    # divergence on the (always-diagonal) s/r tables (renyi/KL/...; __post_init__ rejects full-cov /
    # squared_hellinger) and accepts (K,) b0_h/c0_h. NB it is NOT a width-robustness lever -- summing the
    # per-coordinate envelopes over K is linear in K, unlike per-token 'state_dependent' (log in K).
    # NEURAL-NETWORK EXCEPTION: 'learnable' introduces a model-owned scalar nn.Parameter log_lambda_h
    # (lambda_h = exp(log_lambda_h)), a sanctioned default-OFF sibling of lambda_alpha_mode='learnable' /
    # learnable_lambda_beta. At init log_lambda_h=log(cfg.lambda_h) -> lambda_h=cfg.lambda_h, so a
    # learnable model is byte-identical to constant lambda_h at step 0. Non-'constant' modes require
    # lambda_h>0 (the channel-on gate and, for 'learnable', the log-init value); __post_init__ warns.
    lambda_h_mode:             str   = "constant"   # "constant" | "state_dependent" | "state_dependent_per_coord" | "learnable"
    b0_h:                      'float | List[float]' = 1.0   # state-dependent lambda_h shape: lambda_h* = c0_h/(b0_h + KL(s||r)); list -> (K,) per-coord
    c0_h:                      'float | List[float]' = 1.0   # state-dependent lambda_h shape (numerator); max precision c0_h/b0_h; list -> (K,) per-coord

    # Model-coupling weight lambda_gamma on the model-channel block lambda_gamma * mean_i F_red^s_i,
    # the reduced (envelope) form of sum_ij [ gamma_ij KL(s_i||Omega_ij s_j) + tau_g gamma_ij
    # log(gamma_ij/pi^s_ij) ] (manuscript Participatory_it_from_bit.tex eq:pointwise_free_energy,
    # lines 1241-1249). Default 0.0 = OFF. SECOND INCREMENT: a TRAINING-LOSS regularizer on the
    # model-channel s tables with TIED, DETACHED transport (Omega_tilde = Omega from the converged
    # belief phi.detach()), so the gamma gradient reaches ONLY the s tables and the forward
    # (logits/ce) is byte-identical to the gamma=0 path -- the model channel stays predictively INERT
    # (s does NOT feed q). The detach deliberately severs the phi<-gamma coupling that full tied
    # transport would carry; restoring it is part of the deferred s->q design. lambda_gamma>0 ALONE
    # creates the s tables (the r tables stay hyper-prior-only). NB: the mean over (B, H, N) makes
    # lambda_gamma=1 a per-token-per-head mean weight, NOT the canonical sum-over-ij; the scale is a
    # free coupling.
    lambda_gamma:              float = 0.0
    kappa_gamma:               'float | List[float]' = 1.0   # model-channel sharpness; list -> per-head tau_gamma
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
    # exist; mu_embed is unused (dead) on this path. NB: with lambda_gamma=lambda_h=0, model_channel is
    # a pure RENAME of mu_embed (s plays mu_embed's role, CE-trained) -- ZERO added capacity; the model
    # channel changes predictions only once gamma>0 / lambda_h>0 shape s beyond CE. Oracle: s copied from
    # the belief prior tables -> byte-identical to "token".
    prior_source:              str   = "token"       # "token" | "model_channel"

    # attention
    include_attention_entropy: bool  = True         # canonical (True) vs surrogate (False)
    
    beta_attention_prior:      str   = "causal"

    alibi_slope:               float = 1.0          # base slope for alibi/causal_alibi priors (Press et al. schedule)
    attention_window:          int   = 128          # band half-width for the windowed/causal_windowed priors

    # Precision-weighted attention (diagnostic, default OFF): fold a detached per-key reliability
    # bias -log(precision_attention_b0 + tr Sigma_j) into the attention log_prior, so attention
    # down-weights high-variance (unreliable) keys BEFORE the softmax. A uniform-over-keys Sigma is
    # softmax-absorbed (no effect); only key-to-key variance in Sigma changes attention. Detached ->
    # the closed-form belief kernel stays exact (the bias enters like any fixed attention prior). OFF
    # -> log_prior stays the sequence-only (N,N)/(H,N,N) bias (byte-identical).
    precision_weighted_attention: bool  = False
    precision_attention_b0:       float = 1.0       # b0 in the per-key reliability -log(b0 + tr Sigma_j); > 0
    # precision_weighted_attention only: compute the per-key reliability PER HEAD -- trace over each
    # gauge block's coords (-log(b0 + tr_h Sigma_j)) -- so each head down-weights keys by its OWN
    # block uncertainty, better aligned with per-head attention. Default False = global (trace over
    # all K coords). Inert (warns) when precision_weighted_attention is off.
    precision_attention_per_head: bool  = False

    t5_num_buckets:            int   = 32           # t5_relative_bias: relative-position bucket count
    t5_max_distance:           int   = 128          # t5_relative_bias: log-bucketing horizon (beyond -> last bucket)
    t5_learnable_bias:         bool  = False        # t5_relative_bias: learn the per-bucket bias table b_{i-j}
                                                    #   (sanctioned no-NN exception, default OFF; needs a t5 channel)

    # E-step
    e_q_mu_lr:                 float = 0.5
    e_q_sigma_lr:              float = 0.015
    e_phi_lr:                  float = 0.0

    # Live model channel s (default OFF -> the manuscript's frozen slow channel). When True, s is
    # refined by its own E-step each forward and fed in as the belief's prior (dynamic fiber tie,
    # manuscript line 1399). Requires prior_source='model_channel' so the s-tables are the model's
    # vocab table for encode AND decode. e_s_*_lr are the refine learning rates; small -> slow
    # channel, and e_s_lr=0 collapses to the static model_channel tie. Inert when s_e_step=False.
    s_e_step:                  bool  = False
    e_s_mu_lr:                 float = 0.1
    e_s_sigma_lr:              float = 0.1

    
    
    
    # E-step MEAN trust region (default OFF = current unbounded update). When set to
    # a float, every per-iteration mean step delta_mu = e_q_mu_lr*nat_mu is clamped in sigma-whitened
    # units to at most this many standard deviations (mu_trust_mode='box' per-coord, 'ball' = 2-norm
    # Mahalanobis ball). None reproduces the bare mu = mu - e_q_mu_lr*nat_mu bit-for-bit. A strong
    # setting is e_mu_q_trust=5.0, mu_trust_mode='box'.
    e_mu_q_trust:              Optional[float] = None
    e_sigma_q_trust:           float = 5.0
    
    mu_trust_mode:             str   = "box"          # "box" | "ball" (consulted only when e_mu_q_trust is not None)
    # E-step MEAN preconditioner (B3/EXP-14 mu-arm ablation). 'fisher' (default, pure) descends the
    # Fisher natural gradient nat_mu = Sigma*grad_mu (diagonal Gaussian); 'raw' descends the raw
    # Euclidean grad_mu. The sigma retraction is unchanged either way, so this isolates the MEAN
    # preconditioner (the sigma sector already whitens by 1/sigma in the affine retraction).
    e_step_mu_precond:         str   = "fisher"       # "fisher" | "raw"

    sigma_max:                 float = 10.0
   
    gradient_mode:             str   = "filtering"
    
    phi_precond_mode:          str   = "none"
    phi_retract_mode:          str   = "bch"  # Lie-algebra step chart: euclidean (sum) or bch
    spd_retract_mode:          str   = "spd_affine" # SPD covariance retraction geometry (registry key)

    # decode / encode
    use_prior_bank:            bool  = False
    decode_bias:               bool  = False  # use_prior_bank=False only: learned per-vocab log-unigram bias on logits=mu_q@W^T+b (zero-init, weight-decay-free). Inert (warns) under use_prior_bank=True.
    # use_prior_bank=False only (diagnostic, default OFF): feed the precision-weighted mean -- the
    # diagonal natural parameter eta = Sigma^-1 mu = mu/(sigma+eps) -- to the linear head instead of
    # the bare mean mu, so the belief covariance Sigma_q enters the DISCRIMINATIVE readout. Tests
    # whether Sigma carries predictive signal at the decode WITHOUT the generative/capacity confound
    # of the prior-bank KL decode. OFF -> logits = mu_q @ W^T (byte-identical). Inert (warns) under
    # use_prior_bank=True (the KL decode already consumes sigma_q).
    decode_precision_scaled:   bool  = False
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
    # families: gradient_mode='smoothing', family='gaussian_full', renyi_order!=1) return a
    # create_graph (differentiable) gradient under e_step_gradient='unroll', so the unrolled-through-
    # inference signal reaches the prior tables -- matching the closed-form kernel, which already does
    # this. Default OFF preserves the detached oracle (the gradient is truncated for those families,
    # the long-standing behavior), so the default path is unchanged. NUMERICS: this builds a
    # SECOND-ORDER graph through the E-step. The SINGLE training backward is FINITE for ALL the
    # non-kernel families, full covariance included: the SPD retractions route their eigendecomposition
    # through the gap-regularized _eigh_damped (retraction.py), whose Lorentzian-damped adjoint is
    # finite on the degenerate isotropic Sigma = I default gaussian_full init (stock eigh backward is
    # 100% NaN there), and the full-cov KL/entropy use safe_cholesky. So oracle_unroll_grad=True is the
    # correct setting to keep the through-inference signal live under gaussian_full (pinned across
    # renyi_order in {0.5,1,1.5} by tests/test_fullcov_alpha_roadmap_2026_06_13.py). The residual caveat
    # is HIGHER order only: a genuine SECOND backward (double-grad / Hessian-vector product) can still
    # NaN at large Renyi order (renyi_order>=2, the blend leaving the convex regime); do not rely on this
    # toggle for >1st-order autograd on gaussian_full.
    oracle_unroll_grad:        bool  = False
    
    m_p_mu_lr:                 float = 0.025
    m_p_sigma_lr:              float = 0.0025
    m_phi_lr:                  float = 0.015
    
    
    # Geometrically-correct gauge M-step (opt-in, default OFF -> plain AdamW on phi_embed/pos_phi_free).
    # When True the gauge-frame prior tables (phi_embed, pos_phi_free) are updated by NATURAL-GRADIENT
    # descent + heavy-ball momentum under the phi_precond_mode metric (set phi_precond_mode=
    # "pullback_per_block" for the exact exp-map metric), instead of being placed in AdamW. This is the
    # only way the gauge geometry reaches the M-step: a position-dependent metric cannot ride inside
    # AdamW -- Adam's per-coordinate normalization re-flattens any metric. Here the gauge group is
    # stepped manually WITHOUT Adam normalization (gauge_optim clears p.grad=None after the natural-
    # gradient step), so the Killing metric is NOT a no-op: being conformal (a scalar * I in the
    # Frobenius-orthonormal E_ij basis), killing/killing_per_block rescale the phi step by that
    # constant conformal factor -- a direction-preserving effective-LR change, an exact no-op ONLY
    # under Adam's scale-invariance. The non-conformal pullback metric additionally changes the step
    # direction (it reshapes the gradient along the metric's eigendirections -- the genuinely position-
    # dependent geometry). Per-token metric solves run on the active rows
    # only but are still real compute, so this is an opt-in extreme path; the pure AdamW path is the
    # default. Everything except the gauge frame (mu/sigma/decode/...) stays on AdamW.
    m_phi_natural_grad:        bool  = False
    m_gauge_momentum:          float = 0.9   # heavy-ball momentum for the natural-gradient gauge step
    
    weight_decay:              float = 0.05
   
    # SEPARATE AdamW weight decay for the gauge-frame coordinate tables (phi_embed, learned
    # pos_phi_free), default 0.065. Decoupled decay on phi sets an LR-invariant frame-norm ceiling
    # (|phi*| ~ E[normalized-grad]/wd) that pulls the transport exp(phi.G) toward the identity; this
    # knob decouples that from the belief-table weight_decay so it can be swept (set 0 for full
    # gauge-frame protection). Inert under m_phi_natural_grad=True (phi is natural-gradient stepped,
    # AdamW decay 0 on the gauge groups regardless).
    phi_weight_decay:          float = 0.065

    # SEPARATE AdamW weight decay for the Regime-II edge connection connection_W (audit 2026-06-10
    # F9): the analogue of phi_weight_decay's frame-norm ceiling for the learned connection, whose
    # growth drives the ||mu||^2 ||W||-scaled edge factor. Default None = inherit the global
    # weight_decay (the long-standing behavior, unchanged); set explicitly for an LR-invariant
    # connection-norm ceiling that pulls the transport toward the flat cocycle.
    connection_weight_decay:   Optional[float] = None
    batch_size:                int   = 64
    
    # Accumulate gradients over N microbatches before an optimizer step, for a larger
    # effective batch without the memory of one big forward. Each pulled batch is split
    # into N equal chunks along the batch axis, each backed (loss / N) into .grad, then a
    # single clip + optimizer.step() + scheduler.step() fires at the boundary. Default 1 =
    # current single-step behavior (byte-identical: no chunking, no divide).
    grad_accum_steps:          int   = 1
    max_steps:                 int   = 15000
    warmup_steps:              int   = 100
    
    min_lr:                    float = 0         # absolute cosine-decay floor: each group's LR
    #                          never decays below this. 0.0 recovers the pure half-cosine-to-zero.
    
    min_lr_frac:               float = 0.0           # fractional cosine-decay floor (default OFF):
    #                          each group's LR never decays below min_lr_frac * its OWN base LR,
    #                          preserving the m_mu:m_sigma:m_phi base ratios into the tail. Combined
    #                          with min_lr as max(min_lr, min_lr_frac*base). 0.0 (with min_lr=0) is
    #                          the pure half-cosine-to-zero path.
    
    seed:                      int   = 0
    log_interval:              int   = 100           # console log every N steps (0 = off)
    eval_interval:             int   = 2000            # periodic validation every N steps (0 = off)
    checkpoint_interval:       int   = 0            # save a resumable checkpoint every N steps (0 = off)
    
    # Opt-in training RESUME (default None = OFF = the pure from-scratch path): a path to a
    # checkpoints/step_<N>.pt written by checkpoint_interval. When set, train() restores the model
    # weights, the AdamW optimizer state (momentum), the RNG, and rebuilds the per-group cosine
    # LambdaLR at the saved step, then continues from step N to max_steps. An explicit train(resume_from=...)
    # argument takes precedence over this field. None leaves train() byte-identical to the from-scratch loop.
    resume_from:               Optional[str] = None

    # Opt-in EMA / Polyak weight averaging (default OFF = the pure path: no shadow, the trained model
    # IS the last SGD iterate). When on, train() keeps an exponential moving average of the trainable
    # tables, s <- ema_decay*s + (1-ema_decay)*theta after each optimizer step, swaps it in for
    # evaluation/best-save, and copies it into the model at the end. EMA reads params only (no RNG, no
    # grad/optimizer touch), so the SGD trajectory is byte-identical to the OFF path; only the reported
    # eval and the final weights change. ema_decay must be in (0, 1) when use_ema is on.
    use_ema:                   bool  = False
    ema_decay:                 float = 0.999

    eval_max_batches:          Optional[int] = None # cap the PERIODIC eval pass (None = full split; pure path)
    generate_figures:          bool          = True         # auto-run the single-run publication figures at finalize_run (off the hot path)
    
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

        # gauge_transport ablation meta-toggle (A1 / EXP-2). Coerce the gauge-frame fields UP FRONT so
        # every downstream validation and the optimizer/table wiring see the resolved state. 'on' is a
        # no-op (the pure learned-frame path). 'off' forces Omega_ij = I; 'frozen' freezes a random frame.
        _require(self.gauge_transport, _VALID_GAUGE_TRANSPORT, "gauge_transport")
        if self.gauge_transport == "off":
            # Omega = exp(phi_i) exp(-phi_j) = I needs a ZERO frame AND no other transport modifier, so
            # the two non-flat transport layers must also be off (else Omega != I and the cell is mislabeled).
            if self.pos_rotation != "none":
                raise ValueError(
                    f"gauge_transport='off' requires Omega=I, but pos_rotation={self.pos_rotation!r} folds "
                    f"a positional rotation into the transport. Set pos_rotation='none'."
                )
            if self.transport_mode != "flat":
                raise ValueError(
                    f"gauge_transport='off' requires Omega=I, but transport_mode={self.transport_mode!r} "
                    f"adds a non-flat connection edge factor. Set transport_mode='flat'."
                )
            self.phi_scale = 0.0
            self.pos_phi   = "none"
            self.e_phi_lr  = 0.0
            self.m_phi_lr  = 0.0
            import warnings
            warnings.warn(
                "gauge_transport='off': forcing the gauge frame to the identity (phi_scale=0.0, "
                "pos_phi='none', e_phi_lr=0.0, m_phi_lr=0.0), so Omega_ij=I for all i,j.",
                UserWarning, stacklevel=2,
            )
        elif self.gauge_transport == "frozen":
            if self.phi_scale <= 0.0:
                raise ValueError(
                    f"gauge_transport='frozen' needs a nonzero random frame (phi_scale>0), got "
                    f"phi_scale={self.phi_scale}; use gauge_transport='off' for the identity frame."
                )
            self.e_phi_lr = 0.0
            self.m_phi_lr = 0.0
            import warnings
            warnings.warn(
                f"gauge_transport='frozen': freezing the random gauge frame at init "
                f"(phi_scale={self.phi_scale}, e_phi_lr=0.0, m_phi_lr=0.0); Omega is fixed, not learned.",
                UserWarning, stacklevel=2,
            )

        # divergence seam: divergence_family is the FUNCTIONAL (f-divergence) registry key
        # (renyi, squared_hellinger, ...), distinct from `family` (the covariance-structure
        # kernel). renyi_order is the Renyi order, IGNORED by non-alpha functionals (e.g.
        # squared_hellinger). Both are live, modular seams (CLAUDE.md: slot in different
        # f-divergences). Validated against the functional REGISTRY (not a hardcoded literal
        # list) so a newly registered functional is config-selectable without editing here;
        # local import avoids a config <- divergence <- families import cycle.
        from vfe3.divergence import divergence_functionals
        _require(self.divergence_family, divergence_functionals(), "divergence_family")
        if self.renyi_order <= 0.0:
            raise ValueError(f"renyi_order must be positive, got {self.renyi_order}")

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
        # so_n / sp_n irrep towers: validate group_n + irrep_spec at construction (labels resolve,
        # block dims sum to embed_dim) so a bad spec fails here with the computed dims, not inside
        # the generator build. Local import avoids a config <- irreps cycle.
        if self.gauge_group in ("so_n", "sp_n"):
            _algebra = "so" if self.gauge_group == "so_n" else "sp"
            if self.group_n is None or self.irrep_spec is None:
                raise ValueError(
                    f"gauge_group={self.gauge_group!r} requires both group_n and irrep_spec "
                    f"([(label, mult), ...]); got group_n={self.group_n}, "
                    f"irrep_spec={self.irrep_spec}"
                )
            if not isinstance(self.group_n, int) or self.group_n < 2:
                raise ValueError(f"group_n must be an int >= 2, got {self.group_n!r}")
            if self.gauge_group == "sp_n" and self.group_n % 2 != 0:
                raise ValueError(
                    f"gauge_group='sp_n' (Sp(2m,R)) requires an EVEN group_n (= 2m), got "
                    f"group_n={self.group_n}"
                )
            if not isinstance(self.irrep_spec, list) or not self.irrep_spec:
                raise ValueError(
                    f"irrep_spec must be a non-empty list of (label, mult) pairs, got "
                    f"{self.irrep_spec!r}"
                )
            # JSON has no tuple type; coerce reloaded list pairs to tuples (the cross_couplings
            # round-trip pattern above).
            self.irrep_spec = [tuple(p) if isinstance(p, list) else p for p in self.irrep_spec]
            from vfe3.geometry.irreps import irrep_dim
            _block_dims: List[int] = []
            for _entry in self.irrep_spec:
                if (not isinstance(_entry, tuple) or len(_entry) != 2
                        or not isinstance(_entry[0], str)
                        or not isinstance(_entry[1], int) or _entry[1] < 1):
                    raise ValueError(
                        f"each irrep_spec entry must be a (label: str, mult: int >= 1) pair, "
                        f"got {_entry!r}"
                    )
                _d = irrep_dim(self.group_n, algebra=_algebra, label=_entry[0])
                _block_dims.extend([_d] * _entry[1])
            if sum(_block_dims) != self.embed_dim:
                raise ValueError(
                    f"irrep_spec blocks {_block_dims} sum to {sum(_block_dims)} != "
                    f"embed_dim={self.embed_dim} (group_n={self.group_n})"
                )
            # One shared phi drives EVERY block, so the generators do not partition per block:
            # the per-block phi preconditioners are undefined (the tied_block_glk footprint).
            if self.phi_precond_mode in ("killing_per_block", "pullback_per_block"):
                raise ValueError(
                    f"phi_precond_mode={self.phi_precond_mode!r} is incompatible with "
                    f"gauge_group={self.gauge_group!r}: the shared so/sp generators act in every "
                    f"irrep block, so a per-block metric is undefined. Use 'none', 'clip', or "
                    f"the ambient 'killing'."
                )
            # ALiBi-family priors are built with H = n_heads (model._attention_log_prior) while
            # the energy head axis is the number of irrep blocks; require they agree so the
            # (H, N, N) prior aligns with the (..., H, N, N) energy.
            for _pname in ("beta_attention_prior", "gamma_attention_prior"):
                if (getattr(self, _pname) in ("alibi", "causal_alibi")
                        and self.n_heads != len(_block_dims)):
                    raise ValueError(
                        f"{_pname}={getattr(self, _pname)!r} builds an (n_heads, N, N) prior but "
                        f"gauge_group={self.gauge_group!r} has {len(_block_dims)} irrep blocks; "
                        f"set n_heads={len(_block_dims)} or use a headless prior "
                        f"(uniform/causal/...)."
                    )
        elif self.group_n is not None or self.irrep_spec is not None:
            raise ValueError(
                f"group_n/irrep_spec are consumed only by gauge_group 'so_n'/'sp_n'; got "
                f"gauge_group={self.gauge_group!r}"
            )
        if self.use_cg_coupling and self.gauge_group not in ("so_n", "sp_n"):
            raise ValueError(
                f"use_cg_coupling requires an irrep-labeled tower group ('so_n'/'sp_n'); got "
                f"gauge_group={self.gauge_group!r}"
            )
        # The head mixer needs >= 2 gauge blocks to mix (audit 2026-06-09 overnight PP1). Two
        # single-block cases, handled differently:
        #   (1) A head-block group (block_glk/tied_block_glk) with n_heads < 2 is a single-HEAD
        #       artifact of a head/K sweep -- one block, nothing to mix. AUTO-DISABLE the mixer and
        #       warn (rather than raise) so use_head_mixer=True can stay set across a sweep without a
        #       manual toggle-off at n_heads=1.
        #   (2) A genuinely single-BLOCK group (glk/so_k/sp) cannot mix regardless of head count --
        #       a static misconfiguration that still RAISES (HeadMixer would reject it at VFEModel
        #       construction anyway). so_n/sp_n mix per isotypic component (block counts validated above).
        if (self.use_head_mixer
                and self.gauge_group in ("block_glk", "tied_block_glk")
                and self.n_heads < 2):
            import warnings
            warnings.warn(
                f"use_head_mixer=True with gauge_group={self.gauge_group!r} and n_heads={self.n_heads} "
                f"yields a single gauge block (nothing to mix); auto-disabling use_head_mixer. Set "
                f"n_heads >= 2 to use the head mixer.",
                UserWarning,
                stacklevel=2,
            )
            self.use_head_mixer = False
        if self.use_head_mixer and self.gauge_group in ("glk", "so_k", "sp"):
            raise ValueError(
                f"use_head_mixer=True needs >= 2 gauge blocks to mix, but "
                f"gauge_group={self.gauge_group!r} is a single-block group. Use "
                f"block_glk/tied_block_glk with n_heads >= 2, or an so_n/sp_n irrep tower."
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
            # Cross-coupling's off-block generators destroy the per-head direct sum, so the group
            # builder reports a SINGLE irrep block [K]: n_heads no longer sets the runtime
            # attention-head count and the energy is one full-K block. Fail FAST here on the
            # combinations that would otherwise crash late at VFEModel construction / first forward,
            # and warn on the semantic shift (cross-coupling audit 2026-06-12). The coherent
            # single-block cross-coupled gauge itself still runs.
            import warnings
            for _pname in ("beta_attention_prior", "gamma_attention_prior"):
                if getattr(self, _pname) in ("alibi", "causal_alibi"):
                    raise ValueError(
                        f"{_pname}={getattr(self, _pname)!r} is incompatible with cross_couplings: an "
                        f"alibi prior builds an (n_heads, N, N) per-head bias, but cross-coupled "
                        f"block_glk collapses irrep_dims to a single block [K] (one runtime attention "
                        f"head). Use a headless prior (uniform/causal/causal_windowed/...) with "
                        f"cross_couplings."
                    )
            if self.use_head_mixer:
                raise ValueError(
                    "use_head_mixer=True is incompatible with cross_couplings: cross-coupled block_glk "
                    "collapses irrep_dims to a single block [K], and the head mixer needs >= 2 blocks "
                    "to mix. Disable use_head_mixer or remove cross_couplings."
                )
            for _name in ("kappa_beta", "kappa_gamma"):
                if isinstance(getattr(self, _name), (list, tuple)):
                    raise ValueError(
                        f"{_name} per-head list is incompatible with cross_couplings: the cross-coupled "
                        f"group has a single irrep block [K], not n_heads blocks. Use a scalar {_name}."
                    )
            if self.family == "gaussian_diagonal":
                warnings.warn(
                    "cross_couplings with family='gaussian_diagonal' is an APPROXIMATION: the off-block "
                    "GL(K) congruence g*Sigma*g^T is not diagonal, so the diagonal readout is not "
                    "gauge-invariant. family='gaussian_full' (diagonal_covariance=False) is the exact "
                    "congruence path.",
                    UserWarning, stacklevel=2,
                )
            if self.n_heads > 1:
                warnings.warn(
                    f"cross_couplings collapses block_glk to a single irrep block [K]: n_heads="
                    f"{self.n_heads} no longer sets the attention-head count (runtime is 1 head) and "
                    f"the softmax temperature shifts from kappa*sqrt(d_head) to kappa*sqrt(K). This is "
                    f"the coherent single-block cross-coupled gauge; n_heads only partitions the "
                    f"generators.",
                    UserWarning, stacklevel=2,
                )

        # belief family. ``family`` is the SINGLE covariance-structure toggle: a registry key
        # whose declared ``cov_kind`` ('diagonal'|'full') is the one source of truth for the
        # diagonal-vs-full path, exposed read-only as the derived ``diagonal_covariance`` property.
        # ``divergence_family`` is the SEPARATE functional (f-divergence) seam (renyi, ...;
        # validated above) and is NOT forced equal to family. family is validated against the
        # divergence registry (not a hardcoded list / name literal), so a newly registered family
        # is a valid config value and its diagonal-vs-full structure is its declared cov_kind;
        # ``family_is_diagonal`` below drives the downstream rank / decode / alpha consistency guards.
        from vfe3.divergence import divergence_families, family_cov_kind
        _require(self.family, divergence_families(), "family")
        family_is_diagonal = family_cov_kind(self.family) == "diagonal"

        # free-energy coupling
        for _name in ("kappa_beta", "kappa_gamma"):
            _v = getattr(self, _name)
            if isinstance(_v, (list, tuple)):
                if self.gauge_group in ("so_n", "sp_n"):
                    # heads = irrep blocks (possibly unequal dims): one kappa entry per block
                    # (irrep_spec already validated/coerced in the gauge seam above).
                    _n_blocks = sum(m for _, m in self.irrep_spec)
                    if len(_v) != _n_blocks:
                        raise ValueError(
                            f"{_name} list must have one entry per irrep block; gauge_group="
                            f"{self.gauge_group!r} has {_n_blocks} blocks, got {len(_v)}")
                elif self.gauge_group not in ("block_glk", "tied_block_glk"):
                    raise ValueError(
                        f"{_name} list (per-head) requires an equal-block group "
                        f"(block_glk/tied_block_glk); got gauge_group={self.gauge_group!r}")
                elif len(_v) != self.n_heads:
                    raise ValueError(
                        f"{_name} list must have length n_heads={self.n_heads}, got {len(_v)}")
                if any(x <= 0.0 for x in _v):
                    raise ValueError(f"{_name} entries must be > 0, got {_v}")
            else:
                if _v <= 0.0:
                    raise ValueError(f"{_name} must be positive, got {_v}")
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
        if self.lambda_gamma < 0.0:
            raise ValueError(f"lambda_gamma must be >= 0, got {self.lambda_gamma}")
        # attention priors validated against the prior REGISTRY (add-by-registering). Local import
        # avoids a config <- attention_prior cycle; the bound name is reused for beta_attention_prior below.
        from vfe3.attention_prior import _PRIORS
        _require(self.gamma_attention_prior, tuple(sorted(_PRIORS)), "gamma_attention_prior")
        _require(self.prior_source, _VALID_PRIOR_SOURCES, "prior_source")
        # prior-shape knobs (threaded into model._attention_log_prior; audit P9): the windowed
        # band half-width and the T5 bucketing parameters must be positive ints.
        for name in ("attention_window", "t5_max_distance"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be an int >= 1, got {getattr(self, name)!r}")
        # t5_num_buckets needs >= 2: the bucket function uses max_exact = num_buckets // 2, and
        # num_buckets=1 -> max_exact=0 -> division by zero in the log-bucketing (attention_prior).
        if not isinstance(self.t5_num_buckets, int) or self.t5_num_buckets < 2:
            raise ValueError(f"t5_num_buckets must be an int >= 2, got {self.t5_num_buckets!r}")
        # The log-bucketing divides by log(max_distance / max_exact) with max_exact = num_buckets // 2
        # (the non-bidirectional/causal half-range; bidirectional uses num_buckets // 4, a fortiori
        # safe under this guard). max_distance == max_exact -> log(1) = 0 -> division by zero ->
        # garbage .long() bucket index (IndexError into the bias table); max_distance < max_exact ->
        # a NEGATIVE bucket index that silently reads the wrong end of the table. Require strict
        # max_distance > num_buckets // 2 so every bucket index stays in [0, num_buckets) (defaults
        # 128 > 16 are safe). Mirrors the num_buckets >= 2 guard above.
        if self.t5_max_distance <= self.t5_num_buckets // 2:
            raise ValueError(
                f"t5_max_distance must be > t5_num_buckets // 2 (={self.t5_num_buckets // 2}) so the "
                f"T5 log-bucketing denominator log(t5_max_distance / (t5_num_buckets // 2)) is "
                f"positive; got t5_max_distance={self.t5_max_distance}, t5_num_buckets="
                f"{self.t5_num_buckets}."
            )
        for name in ("b0", "c0"):
            _v = getattr(self, name)
            if isinstance(_v, (list, tuple)):
                # A (K,) per-coordinate b0/c0 shapes the PER-COORDINATE alpha*_k = c0_k/(b0_k + D_k);
                # every other alpha form carries a per-position scalar D, against which a (K,) list
                # either crashes at the first forward or (when K == N) silently mis-broadcasts.
                # Reject the pair at construction (audit 2026-06-09 P2). Local import matches the
                # alpha_i pattern below.
                from vfe3.alpha_i import alpha_is_per_coord
                if not alpha_is_per_coord(self.lambda_alpha_mode):
                    raise ValueError(
                        f"{name} list (per-coordinate) requires a per-coordinate alpha form "
                        f"(lambda_alpha_mode='state_dependent_per_coord'); got "
                        f"lambda_alpha_mode={self.lambda_alpha_mode!r}"
                    )
                if len(_v) != self.embed_dim:
                    raise ValueError(
                        f"{name} list must have length embed_dim={self.embed_dim}, got {len(_v)}")
                if any(x <= 0.0 for x in _v):
                    raise ValueError(f"{name} entries must be > 0, got {_v}")
            else:
                if _v <= 0.0:
                    raise ValueError(f"{name} must be positive, got {_v}")
        if self.sigma_init <= 0.0:
            raise ValueError(f"sigma_init must be positive (log is taken), got {self.sigma_init}")
        for name in ("mu_init_std", "phi_scale"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        # lambda_alpha_mode validated against the alpha-form REGISTRY (add-by-registering). Local import
        # avoids a config <- alpha_i cycle.
        from vfe3.alpha_i import _ALPHAS
        _require(self.lambda_alpha_mode, tuple(sorted(_ALPHAS)), "lambda_alpha_mode")
        # lambda_h_mode validated against the hyper-prior-coupling registry (the model-fiber mirror of
        # lambda_alpha_mode). r_update_mode selects the centroid M-step (gradient vs closed-form barycenter).
        from vfe3.lambda_h_i import _LAMBDA_H_MODES
        _require(self.lambda_h_mode, _LAMBDA_H_MODES, "lambda_h_mode")
        _require(self.r_update_mode, ("gradient", "barycenter"), "r_update_mode")
        # A per-coordinate alpha form (state_dependent_per_coord) weights each coordinate's
        # self-divergence by its own alpha^(k), which needs a per-coordinate self-divergence.
        # That decomposition exists only for the diagonal family (full-covariance KL couples
        # coordinates through the trace and log-determinant), so reject the inconsistent pair at
        # construction rather than letting the per-coordinate divergence raise mid-forward.
        from vfe3.alpha_i import alpha_is_per_coord
        if alpha_is_per_coord(self.lambda_alpha_mode) and not family_is_diagonal:
            raise ValueError(
                f"lambda_alpha_mode={self.lambda_alpha_mode!r} needs a per-coordinate self-divergence, which "
                f"exists only for a diagonal-covariance family; got family={self.family!r}. Use "
                f"a diagonal family (e.g. 'gaussian_diagonal') or a per-position lambda_alpha_mode."
            )
        # The per-coordinate self-divergence exists only for a divergence that DECOMPOSES
        # coordinate-wise on a diagonal Gaussian -- the per-coordinate functional registry: Renyi/KL
        # and the two divergences AFFINE in it (Bhattacharyya = 0.5 D_{1/2}, Jeffreys = KL + KL_rev).
        # squared_hellinger is excluded (H^2 = 1 - exp(-D_{1/2}/2) is a nonlinear transform of the
        # SUMMED divergence). free_energy.self_divergence_per_coord raises at runtime for an
        # unregistered functional; reject the pair at construction too -- mirroring that raise -- so
        # this doubly-opt-in path fails fast at config time (the covariance half is rejected above).
        from vfe3.divergence import has_per_coord_functional, divergence_functionals_per_coord
        if alpha_is_per_coord(self.lambda_alpha_mode) and not has_per_coord_functional(self.divergence_family):
            raise ValueError(
                f"lambda_alpha_mode={self.lambda_alpha_mode!r} needs a per-coordinate self-divergence, which is "
                f"implemented only for divergences that decompose coordinate-wise on a diagonal "
                f"Gaussian ({divergence_functionals_per_coord()}); got "
                f"divergence_family={self.divergence_family!r} (e.g. 'squared_hellinger' does not "
                f"decompose). Use a decomposable divergence or a per-position lambda_alpha_mode."
            )
        # Calibration footgun (B4): the state-dependent envelope alpha* = c0/(b0 + D) is correct for
        # ANY divergence, but its RANGE collapses when D is BOUNDED. squared_hellinger has D = H^2 in
        # [0, 1), so the alpha* ratio max/min = 1 + 1/b0; at the default b0 = 1 that is only ~2x (vs
        # ~1 + kl_max/b0 for the unbounded KL), so the adaptive coupling is nearly inert. Warn (the
        # b0/c0 are free, positivity-validated fields) -- set b0 ~ Dmax/10 (here ~0.1) to restore a
        # wide alpha* range. Only squared_hellinger is bounded-small; bhattacharyya/jeffreys are
        # bounded by kl_max/2 / 2*kl_max, wide enough that b0 = O(1) is not degenerate.
        if (self.lambda_alpha_mode in ("state_dependent", "state_dependent_per_coord")
                and self.divergence_family == "squared_hellinger"):
            _b0_vals = self.b0 if isinstance(self.b0, (list, tuple)) else [self.b0]
            if min(_b0_vals) >= 1.0:
                import warnings
                warnings.warn(
                    f"lambda_alpha_mode={self.lambda_alpha_mode!r} with divergence_family='squared_hellinger' and "
                    f"b0={self.b0}: squared Hellinger is bounded (H^2 in [0,1)), so the state-dependent "
                    f"alpha* = c0/(b0 + D) spans only a ~{1.0 + 1.0/min(_b0_vals):.1f}x range and the "
                    f"adaptive self-coupling is nearly constant. Set b0 ~ 0.1 (Dmax/10) for a wide "
                    f"alpha* range, or use an unbounded divergence (renyi/KL).",
                    UserWarning,
                    stacklevel=2,
                )

        # lambda_h_mode='state_dependent_per_coord': the model-fiber mirror of the alpha per-coord
        # guards above, keyed on lambda_h_mode (lambda_h delegates to the shared alpha registry, so
        # lambda_h_is_per_coord defers to alpha_is_per_coord). The per-coordinate hyper-prior envelope
        # lambda_h^(k)* = c0_h^(k)/(b0_h^(k) + KL_k(s||r)) needs (a) (K,) b0_h/c0_h shaping each
        # coordinate's weight, (b) a per-coordinate KL(s||r), which exists only for a coordinate-
        # decomposable divergence on the (always-diagonal) s/r tables. Reject inconsistent pairs at
        # construction rather than letting the per-coordinate divergence raise mid-forward.
        from vfe3.lambda_h_i import lambda_h_is_per_coord
        for name in ("b0_h", "c0_h"):
            _v = getattr(self, name)
            if isinstance(_v, (list, tuple)):
                if not lambda_h_is_per_coord(self.lambda_h_mode):
                    raise ValueError(
                        f"{name} list (per-coordinate) requires a per-coordinate lambda_h form "
                        f"(lambda_h_mode='state_dependent_per_coord'); got "
                        f"lambda_h_mode={self.lambda_h_mode!r}"
                    )
                if len(_v) != self.embed_dim:
                    raise ValueError(
                        f"{name} list must have length embed_dim={self.embed_dim}, got {len(_v)}")
                if any(x <= 0.0 for x in _v):
                    raise ValueError(f"{name} entries must be > 0, got {_v}")
        if lambda_h_is_per_coord(self.lambda_h_mode) and not family_is_diagonal:
            raise ValueError(
                f"lambda_h_mode={self.lambda_h_mode!r} needs a per-coordinate hyper-prior divergence, "
                f"which exists only for a diagonal-covariance family; got family={self.family!r}. Use "
                f"a diagonal family (e.g. 'gaussian_diagonal') or a per-position lambda_h_mode."
            )
        if lambda_h_is_per_coord(self.lambda_h_mode) and not has_per_coord_functional(self.divergence_family):
            raise ValueError(
                f"lambda_h_mode={self.lambda_h_mode!r} needs a per-coordinate hyper-prior divergence, "
                f"implemented only for divergences that decompose coordinate-wise on a diagonal Gaussian "
                f"({divergence_functionals_per_coord()}); got divergence_family={self.divergence_family!r} "
                f"(e.g. 'squared_hellinger' does not decompose). Use a decomposable divergence or a "
                f"per-position lambda_h_mode."
            )

        # attention
        _require(self.beta_attention_prior, tuple(sorted(_PRIORS)), "beta_attention_prior")

        # E-step
        for name in ("e_q_mu_lr", "e_q_sigma_lr", "e_phi_lr", "e_s_mu_lr", "e_s_sigma_lr"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.s_e_step:
            # Intentional narrowing (not a membership check): of the valid prior_source values, only
            # 'model_channel' routes encode AND decode through the s-tables, which the live s-refine
            # anchors the belief to. 'token' would decode against the separate belief table.
            if self.prior_source != "model_channel":
                raise ValueError(
                    "s_e_step=True requires prior_source='model_channel' so the s-tables are the "
                    f"model's vocab table for encode and decode; got prior_source={self.prior_source!r}."
                )
            # The live s-refine (_refine_s) and the s/r tables are DIAGONAL by construction (the s
            # table is (V,K) and the centroid r is (K,)). Under a full-covariance family the belief
            # sigma is (B,N,K,K) and the diagonal refined-s would overwrite it with a (B,N,K) tensor,
            # crashing deep in the full kernel with an opaque shape error. Reject at construction so
            # the (unsupported) full-cov s-channel fails fast with a clear message; the diagonal
            # family is the supported pure path for the s E-step.
            if not family_is_diagonal:
                raise ValueError(
                    "s_e_step=True refines the model channel as a DIAGONAL Gaussian (the s/r tables "
                    f"are diagonal by construction), incompatible with family={self.family!r}. Use a "
                    "diagonal-covariance family (e.g. 'gaussian_diagonal') for the live s E-step."
                )
            # The s-refine (_refine_s) hardcodes family='gaussian_diagonal': the model channel is
            # uniformly DiagonalGaussian by design. A non-Gaussian (but diagonal) belief family
            # therefore still runs a GAUSSIAN s E-step while the belief runs its own family -- a
            # well-posed mixed prior/posterior (no NaN), but the model channel is NOT refined in the
            # belief's family. Warn (do not raise) so the double-opt-in (s_e_step + non-Gaussian
            # family) is not silent; the pure path is family='gaussian_diagonal'.
            elif self.family != "gaussian_diagonal":
                import warnings
                warnings.warn(
                    f"s_e_step=True refines the model channel as a Gaussian (_refine_s hardcodes "
                    f"family='gaussian_diagonal'), but family={self.family!r}: the s-channel E-step "
                    f"runs Gaussian while the belief is {self.family!r}. This is a well-posed "
                    f"mixed-family prior/posterior (no NaN), but the model channel is not refined in "
                    f"the belief's family. Use family='gaussian_diagonal' to match, or accept the "
                    f"mixed-family s-refine.",
                    UserWarning, stacklevel=2,
                )
            if self.lambda_h == 0.0 and self.lambda_gamma == 0.0:
                import warnings
                warnings.warn(
                    "s_e_step=True with lambda_h=0 and lambda_gamma=0: the s-refine has no force, "
                    "so s1==s0 and the channel reduces to the static prior_source='model_channel' tie.",
                    UserWarning, stacklevel=2,
                )
        # learnable_r diagnostics. The centroid r is created only under lambda_h>0 or s_e_step
        # (prior_bank.py), and the forward hyper-prior term that governs it runs only when lambda_h>0
        # and not s_e_step. (s_e_step is excluded here: it structurally requires model_channel, so r is
        # both created and CE-anchored -- never inert, never collapsing.)
        if self.learnable_r and not self.s_e_step:
            import warnings
            if self.lambda_h == 0.0:
                # Inert: with lambda_h=0 no r table is created (lambda_gamma>0 builds only the s
                # tables), so the toggle silently does nothing -- warn rather than no-op in silence.
                warnings.warn(
                    "learnable_r=True has no effect: the hyper-prior centroid r is created only when "
                    "lambda_h>0 or s_e_step=True. Set lambda_h>0 (with prior_source='model_channel') "
                    "to train r.",
                    UserWarning, stacklevel=2,
                )
            elif self.prior_source != "model_channel":
                # Collapse guard: un-freezing r while the forward hyper-prior term is live but s is NOT
                # data-anchored leaves lambda_h*KL(s||r) the only force on s/r. Its joint optimum is
                # s_i=r=const (KL->0), so r collapses onto s and the term vanishes -- exactly what
                # freezing r guards against. Only the CE gradient (routed into s by
                # prior_source='model_channel') anchors s; gamma is a self-referential consensus among
                # the same free s and does not prevent the collapse, so this warns regardless of gamma.
                warnings.warn(
                    f"learnable_r=True with an unanchored model channel (prior_source={self.prior_source!r}, "
                    "lambda_h>0): the only force on s and r is lambda_h*KL(s||r), whose joint optimum is "
                    "s_i=r=const (KL->0, the hyper-prior term vanishes). Anchor s to data with "
                    "prior_source='model_channel', or keep r frozen (learnable_r=False).",
                    UserWarning, stacklevel=2,
                )
        # lambda_h_mode inert guard: a non-'constant' mode (state_dependent envelope / learnable
        # log_lambda_h) only takes effect when the hyper-prior channel is live (lambda_h>0 creates
        # r and gates the forward term; learnable additionally needs lambda_h>0 for its log-init).
        if self.lambda_h_mode != "constant" and self.lambda_h == 0.0:
            import warnings
            warnings.warn(
                f"lambda_h_mode={self.lambda_h_mode!r} has no effect with lambda_h=0: the hyper-prior "
                "channel (and its centroid r) is created only when lambda_h>0 or s_e_step=True, and the "
                "weight defaults to lambda_h. Set lambda_h>0 to activate the state-dependent/learnable "
                "hyper-prior precision.",
                UserWarning, stacklevel=2,
            )
        # r_update_mode='barycenter' is a no-op unless r is un-frozen (learnable_r=True); a frozen r
        # is never updated by either mechanism.
        if self.r_update_mode == "barycenter" and not self.learnable_r:
            import warnings
            warnings.warn(
                "r_update_mode='barycenter' has no effect with learnable_r=False: a frozen centroid r "
                "is never updated. Set learnable_r=True to enable the closed-form barycenter M-step.",
                UserWarning, stacklevel=2,
            )
        # r_update_mode='barycenter' is the EXACT M-step only when KL(s||r) is r's sole objective, i.e.
        # the scored s_e_step=False regime. Under s_e_step=True the scored hyper-prior term is gated off
        # (model.py, `... and not s_e_step`) and r enters the loss only through the unrolled _refine_s,
        # so r is coupled to the cross-entropy and the closed-form moment-matched barycenter is NOT the
        # argmin of what the model minimizes -- AdamW-through-unroll (r_update_mode='gradient') is the
        # consistent update there (2026-06-13 r/lambda_h spec). Warn (non-breaking) rather than restrict.
        if self.r_update_mode == "barycenter" and self.s_e_step:
            import warnings
            warnings.warn(
                "r_update_mode='barycenter' with s_e_step=True applies an INEXACT M-step: the closed-form "
                "barycenter is the exact M-step only in the scored s_e_step=False regime. Under s_e_step "
                "the hyper-prior centroid r is coupled to the cross-entropy through the unrolled _refine_s, "
                "so the moment-matched centroid is a consistent population target, not the variational "
                "optimum. Use r_update_mode='gradient' for the coupled s_e_step regime.",
                UserWarning, stacklevel=2,
            )
        # barycenter_r_ is the closed-form ALPHA=1 forward-KL m-projection (moment match) of the s
        # tables and reads NO cfg, so it is the EXACT M-step only for the canonical KL objective:
        # renyi_order=1, divergence_family='renyi', lambda_h_mode='constant'. The scored hyper-prior
        # gradient (_hyper_prior_kl) descends D_alpha(s||r) at cfg.renyi_order / cfg.divergence_family
        # with the lambda_h_mode envelope, so under any non-canonical setting the closed-form barycenter
        # and a gradient M-step descend DIFFERENT divergences/weightings and no longer share a fixed
        # point. Warn (non-breaking) rather than restrict (2026-06-14 multi-expert investigation of the
        # model.py TODO(B)); gated on learnable_r so it fires only when the barycenter actually runs.
        if (self.r_update_mode == "barycenter" and self.learnable_r
                and (self.renyi_order != 1.0 or self.divergence_family != "renyi"
                     or self.lambda_h_mode != "constant")):
            import warnings
            warnings.warn(
                "r_update_mode='barycenter' applies the alpha=1 forward-KL moment-match centroid, the "
                "exact M-step only for renyi_order=1.0, divergence_family='renyi', and "
                f"lambda_h_mode='constant'. The active config (renyi_order={self.renyi_order}, "
                f"divergence_family={self.divergence_family!r}, lambda_h_mode={self.lambda_h_mode!r}) "
                "makes the scored hyper-prior gradient descend a different divergence/weighting, so the "
                "closed-form barycenter and a gradient M-step no longer share a fixed point. Use "
                "r_update_mode='gradient' for a consistent M-step under non-KL / state_dependent / "
                "learnable settings.",
                UserWarning, stacklevel=2,
            )
        _require(self.gradient_mode, _VALID_GRADIENT_MODES, "gradient_mode")
        from vfe3.geometry.phi_preconditioner import _PRECOND
        _require(self.phi_precond_mode, tuple(sorted(_PRECOND)), "phi_precond_mode")
        # The natural-gradient gauge M-step (m_phi_natural_grad=True) carries a POSITION-DEPENDENT
        # metric only under the pullback family ('pullback' / 'pullback_per_block'). With any other
        # phi_precond_mode it degenerates to plain heavy-ball momentum SGD on phi with NO geometric
        # metric: 'killing'/'killing_per_block' are conformal (a scalar * I) -> only an effective-LR
        # rescale, and 'none'/'clip' apply no metric at all. Warn (non-breaking) so the geometric step
        # is actually selected; set phi_precond_mode='pullback_per_block' for the documented step.
        if self.m_phi_natural_grad and self.phi_precond_mode not in ("pullback", "pullback_per_block"):
            import warnings
            warnings.warn(
                f"m_phi_natural_grad=True with phi_precond_mode={self.phi_precond_mode!r} runs plain "
                "heavy-ball momentum SGD on phi with NO position-dependent metric: "
                "'killing'/'killing_per_block' are conformal (only an effective-LR rescale) and "
                "'none'/'clip' apply no metric. Set phi_precond_mode='pullback_per_block' for the "
                "documented geometric (pullback natural-gradient) gauge M-step.",
                UserWarning,
                stacklevel=2,
            )
        
        from vfe3.geometry.lie_ops import _COMPOSE      # phi retraction & pos-phi share the compose registry
        _require(self.phi_retract_mode, tuple(sorted(_COMPOSE)), "phi_retract_mode")
        from vfe3.model.positional_phi import _POS_PHI
        _require(self.pos_phi, tuple(sorted(_POS_PHI)), "pos_phi")
        _require(self.pos_phi_compose, tuple(sorted(_COMPOSE)), "pos_phi_compose")
        from vfe3.geometry.rope import _POS_ROTATIONS
        _require(self.pos_rotation, tuple(sorted(_POS_ROTATIONS)), "pos_rotation")
        if self.rope_full_gauge and self.diagonal_covariance:
            raise ValueError(
                "rope_full_gauge=True rotates the covariance sandwich (R Sigma R^T), which the "
                "diagonal-covariance approximation cannot carry; set family='gaussian_full'."
            )
        # RoPE rotates ADJACENT coordinate pairs (2k, 2k+1); Sp(2m) pairs coordinate i with m+i
        # (J = [[0,I],[-I,0]]), so the rope-wrapped transport R Omega R^T leaves the symplectic group.
        # R is orthogonal, so each per-pair divergence D(q_i || R_i Omega_ij R_j^T q_j) stays a
        # well-defined GL(K)-congruence value. But the position-FIXED R(theta_i) do not commute
        # with a global gauge element g, so gauge-RoPE breaks the global gauge EQUIVARIANCE of
        # F/beta for EVERY group -- including rope_full_gauge=True (executable probe, audit
        # 2026-06-09 G1: beta residual 0.44-0.61 under rope vs 7e-6 without). RoPE is therefore a
        # deliberate residual gauge-FIXING layer, not an equivariant operator; the pure
        # (equivariant) positional path remains pos_phi (a gauge element composed into phi).
        # Warn (not error): the symplectic property is not consumed downstream.
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
        # Same structural fact for the irrep-tower groups: RoPE's adjacent-pair rotation R is
        # orthogonal but generally NOT in the irrep image rho(SO(N)) / rho(Sp(2m)) inside GL(K),
        # so R Omega R^T leaves the SELECTED structure group (the GL(K)-congruence divergence
        # invariance survives; only the structure-group claim is dropped).
        if self.pos_rotation == "rope" and self.gauge_group in ("so_n", "sp_n"):
            import warnings
            warnings.warn(
                f"pos_rotation='rope' with gauge_group={self.gauge_group!r} leaves the structure "
                f"group: the coordinate-pair rotation R(theta) is generally not in the irrep image "
                f"of SO(N)/Sp(2m), so R Omega R^T is no longer in the selected group. The "
                f"GL(K)-congruence divergence invariance still holds (R is orthogonal), so the "
                f"model runs; only the structure-group claim is dropped.",
                UserWarning,
                stacklevel=2,
            )
        # Means-only RoPE (the default pairing) transports mu under R Omega R^T but Sigma under
        # the UN-rotated Omega -- an affine-incoherent operator pair: the Mahalanobis form is not
        # preserved (executable probe, audit 2026-06-09 G2: invariant drift 0.18 -> 0.74, while
        # rope_full_gauge=True preserves it). It stays available as a deliberate cheap
        # approximation; warn so the non-coherent pairing is explicit.
        if self.pos_rotation == "rope" and not self.rope_full_gauge:
            import warnings
            warnings.warn(
                "pos_rotation='rope' with rope_full_gauge=False rotates the transported MEANS but "
                "not the covariance sandwich -- an affine-incoherent transport pair (Mahalanobis "
                "invariants are not preserved). The coherent gauge-RoPE is rope_full_gauge=True "
                "with a full-covariance family.",
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
        # decode_mode / encode_mode validated against the LIVE decode/encode REGISTRIES (not the
        # static _VALID_*_MODES literals) so a newly registered kernel is config-selectable without
        # editing this validator (the add-by-registering contract, matching gauge_group / transport_mode
        # / spd_retract_mode above). Local import avoids a config <- prior_bank cycle. decode_mode must
        # NOT accept 'linear': that kernel is reached only through the use_prior_bank=False second-gate
        # (a learned linear readout), never via decode_mode, so it is excluded from the valid set.
        from vfe3.model.prior_bank import _DECODERS, _ENCODERS
        _require(self.decode_mode, tuple(sorted(set(_DECODERS) - {"linear"})), "decode_mode")
        # decode_mode sets the RANK of the prior-bank KL-decode kernel: 'diagonal'/'diagonal_chunked'
        # consume a diagonal sigma (B,N,K); 'full'/'full_chunked' consume a full sigma (B,N,K,K). It
        # must agree with the covariance family, else the rank mismatch is a shape RuntimeError at the
        # first forward. The use_prior_bank=False linear decode discards sigma and reads decode_mode
        # for exactly one thing: '*_chunked' selects the fused chunked-CE training path over
        # logits = mu @ W^T (vram audit 2026-06-10) -- rank is irrelevant there, so the cross-check
        # stays gated on use_prior_bank.
        decode_is_full = self.decode_mode in ("full", "full_chunked")
        if self.use_prior_bank and decode_is_full == family_is_diagonal:
            raise ValueError(
                f"decode_mode={self.decode_mode!r} is rank-incompatible with family={self.family!r}: "
                f"'full'/'full_chunked' decode needs a full-covariance family and "
                f"'diagonal'/'diagonal_chunked' decode needs a diagonal family. Pair a full decode_mode "
                f"with a full family, use a diagonal decode_mode with a diagonal family, or set "
                f"use_prior_bank=False (linear decode)."
            )
        if self.decode_chunk_size < 1:
            raise ValueError(f"decode_chunk_size must be >= 1, got {self.decode_chunk_size}")
        # decode_bias is a learned per-vocab log-unigram bias on the use_prior_bank=False LINEAR
        # decode (logits = mu_q @ W^T + b). On the prior-bank KL path the per-vocab priors
        # (mu_p, sigma_p) already carry the unigram role, so the bias is never created there; warn
        # so a True+prior_bank pair is not silently a no-op.
        if self.decode_bias and self.use_prior_bank:
            import warnings
            warnings.warn(
                "decode_bias=True is inert when use_prior_bank=True: the KL-to-prior decode's "
                "per-vocab priors already play the log-unigram role. Set use_prior_bank=False "
                "(linear decode) for the learned bias to take effect.",
                UserWarning,
            )
        # decode_precision_scaled feeds the precision-weighted mean (eta=mu/sigma) to the LINEAR head;
        # on the KL-to-prior path sigma_q already enters the readout, so the toggle is inert there --
        # warn (mirrors decode_bias).
        if self.decode_precision_scaled and self.use_prior_bank:
            import warnings
            warnings.warn(
                "decode_precision_scaled=True is inert when use_prior_bank=True: the KL-to-prior "
                "decode already consumes sigma_q. Set use_prior_bank=False (linear decode) for the "
                "precision-weighted mean to take effect.",
                UserWarning,
            )
        # precision_weighted_attention's per-key reliability -log(b0 + tr Sigma_j) needs a positive b0.
        if self.precision_weighted_attention and self.precision_attention_b0 <= 0.0:
            raise ValueError(
                f"precision_attention_b0 must be positive (the b0 in the per-key reliability "
                f"-log(b0 + tr Sigma_j)), got {self.precision_attention_b0}")
        # precision_attention_per_head only shapes the bias WHEN precision_weighted_attention is on.
        if self.precision_attention_per_head and not self.precision_weighted_attention:
            import warnings
            warnings.warn(
                "precision_attention_per_head=True is inert when precision_weighted_attention=False: "
                "there is no reliability bias to shape per head. Enable precision_weighted_attention.",
                UserWarning,
            )
        # use_prior_bank decode is a FIXED alpha=1 KL readout on the hardcoded Gaussian family
        # (prior_bank.reference_decode / the fused kernels call divergence.kl); it does NOT read
        # renyi_order / divergence_family. An opt-in non-KL/non-alpha=1 seam therefore minimizes the
        # E-step under one divergence and reads logits out under another -- warn so the mismatch is a
        # deliberate choice (the pure path is use_prior_bank=True with renyi / renyi_order=1).
        if self.use_prior_bank and (self.renyi_order != 1.0 or self.divergence_family != "renyi"):
            import warnings
            warnings.warn(
                f"use_prior_bank=True decodes at a FIXED alpha=1 KL, but the E-step minimizes under "
                f"divergence_family={self.divergence_family!r}/renyi_order={self.renyi_order}: inference "
                f"and the KL-to-prior readout use different divergences.",
                UserWarning,
                stacklevel=2,
            )
        # The use_prior_bank decode kernels hardcode the GAUSSIAN family (prior_bank.reference_decode /
        # the fused kernels call get_family('gaussian_diagonal')/('gaussian_full'); they read neither
        # `family` nor `divergence_family`). A non-Gaussian belief family runs a genuine non-Gaussian
        # E-step but is then projected to logits through the WRONG (Gaussian) metric -- the converged
        # belief is correct, only its readout uses the wrong divergence (argmax can flip). No
        # Gaussian-only decode kernel for the other families exists, so warn (the pure readout paths
        # are a Gaussian family, or use_prior_bank=False's linear decode which is family-agnostic).
        if self.use_prior_bank and self.family not in ("gaussian_diagonal", "gaussian_full"):
            import warnings
            warnings.warn(
                f"use_prior_bank=True decodes through a hardcoded GAUSSIAN KL readout, but "
                f"family={self.family!r} is non-Gaussian: the E-step minimizes in the {self.family!r} "
                f"geometry while the logits are read out under the Gaussian metric (the converged "
                f"belief is correct; only its projection to logits uses the wrong divergence). Use a "
                f"Gaussian family for the KL-to-prior decode, or use_prior_bank=False (the linear "
                f"decode is family-agnostic).",
                UserWarning,
                stacklevel=2,
            )
        # Full-covariance compute discarded at the decode boundary (B4): use_prior_bank=False decodes
        # by the linear readout logits = mu_q @ W^T, which DISCARDS sigma. With a full-covariance family
        # the E-step still evolves a (B, N, K, K) covariance (it shapes the mean trajectory, so the
        # result is correct, not wasted) but that covariance never reaches the output. Warn so the
        # combination is a deliberate choice, not a silent surprise; the full covariance reaches the
        # logits only under the use_prior_bank=True KL decode (decode_mode='full'/'full_chunked').
        if not family_is_diagonal and not self.use_prior_bank:
            import warnings
            warnings.warn(
                f"family={self.family!r} (full covariance) with use_prior_bank=False: the linear "
                f"decode logits = mu_q @ W^T discards the converged (B,N,K,K) covariance at the output "
                f"boundary (it still shapes the E-step mean trajectory, so the result is correct). For "
                f"the covariance to reach the logits use use_prior_bank=True with "
                f"decode_mode='full'/'full_chunked' (the KL-to-prior decode).",
                UserWarning,
                stacklevel=2,
            )
        # encode_mode validated against the live encoder registry (per_token + the 'gauge_fixed'
        # stub, which the existing NotImplementedError guard below then rejects).
        _require(self.encode_mode, tuple(sorted(_ENCODERS)), "encode_mode")
        # use_prior_bank is the SINGLE decode gate. True (default, pure path): the KL-to-prior
        # readout logits = -KL(q_i || pi_v)/tau_eff over the gauge-orbit prior bank, with the
        # covariance structure selected by decode_mode (diagonal | full). False (the linear-decode
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
        # regime_ii x gaussian_full (audit 2026-06-10 F10): the per-edge factor exp(delta . G) is
        # non-orthogonal for the non-compact groups, and the FULL-covariance sandwich
        # Omega Sigma Omega^T can go indefinite at fp32 -- the full-family KL then masks the NaN
        # Cholesky to kl_max SILENTLY (the edge contributes a saturated constant, and under
        # oracle_unroll_grad=True the double-backward can NaN connection_W.grad). Warn (non-
        # breaking, mirroring the estimator warnings below); prefer the diagonal family or a
        # compact so tower when running regime_ii at fp32 full covariance.
        if self.transport_mode in ("regime_ii", "regime_ii_covariant") and self.family == "gaussian_full":
            import warnings
            warnings.warn(
                "transport_mode='regime_ii' with family='gaussian_full': the non-orthogonal edge "
                "factor can drive the transported full covariance indefinite at fp32; the full-"
                "family KL masks the failed Cholesky to kl_max SILENTLY, and the unrolled oracle's "
                "double-backward can produce NaN connection_W gradients. Prefer gaussian_diagonal "
                "or a compact so-tower group with regime_ii.",
                UserWarning,
                stacklevel=2,
            )

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
        # Both 'straight_through' and 'detach' sever the per-iteration E-step tangent, so a learnable
        # parameter whose only loss path IS that tangent receives no gradient and silently freezes.
        # (detach_e_step=True also detaches but is forced to e_step_gradient='unroll' here and is warned
        # at the model level, so keying off the e_step_gradient literal covers exactly the un-warned
        # routes.) Warn (non-breaking; 'unroll' is the default that trains them) rather than restrict
        # the toggle combination.
        # (pos_phi='learned' ALSO freezes under these estimators, but it is the DEFAULT and is warned
        # at the MODEL level -- VFEModel.__init__, keyed on the EFFECTIVE estimator -- so it is left out
        # of this config-level predicate to keep the default config silent here.)
        if self.e_step_gradient in ("straight_through", "detach") and (
            self.lambda_alpha_mode == "learnable"
            or self.transport_mode in ("regime_ii", "regime_ii_covariant")
            or self.learnable_lambda_beta
            or (self.lambda_h_mode == "learnable" and self.s_e_step)
            or (self.learnable_r and self.r_update_mode == "gradient" and self.s_e_step)
        ):
            import warnings
            warnings.warn(
                f"e_step_gradient={self.e_step_gradient!r} severs the per-iteration E-step tangent, so a "
                "learnable parameter that enters the loss only through it (log_alpha under "
                "lambda_alpha_mode='learnable', connection_W under transport_mode='regime_ii', log_lambda_beta "
                "under learnable_lambda_beta, log_lambda_h under lambda_h_mode='learnable'+s_e_step, "
                "r_mu/r_sigma_log under learnable_r+r_update_mode='gradient'+s_e_step) "
                "receives NO gradient and stays frozen. Use e_step_gradient='unroll' (the default) to "
                "train these.",
                UserWarning,
                stacklevel=2,
            )
        # A SECOND un-warned freeze route: under e_step_gradient='unroll' the closed-form belief-gradient
        # KERNEL keeps the unrolled E-step signal live to the prior tables, but the autograd ORACLE
        # fallback (served for every NON-kernel family) returns a DETACHED tangent unless
        # oracle_unroll_grad=True, truncating that signal. The kernel covers exactly
        # gradient_mode=='filtering' AND family=='gaussian_diagonal' AND divergence_family=='renyi' AND
        # renyi_order==1.0 AND include_attention_entropy (verified against gradients.kernels.use_kernel);
        # any other combination routes to the oracle. When it does and an E-step-only learnable param is
        # active (lambda_alpha_mode='learnable' / transport_mode='regime_ii' / learnable_lambda_beta /
        # pos_phi='learned'), that param receives NO gradient through the detached oracle. Warn
        # (non-breaking); oracle_unroll_grad=True restores the differentiable oracle gradient.
        # transport_mode='regime_ii' ALWAYS routes to the oracle (audit 2026-06-10 F1: the kernel
        # is the flat-transport gradient and drops d Omega/d mu), regardless of the kernel-family
        # predicate below -- so connection_W training requires oracle_unroll_grad=True on every
        # regime_ii config.
        # The decoupled value gauge (pos_rotation='rope' + rope_on_value=False) also forces the oracle
        # route -- uses_kernel_route gates on `and not decoupled_value_gauge` (kernels.py) -- so the
        # freeze warning must include it or it silently misses a frozen E-step param there (r2 id2).
        # AUTO-ENABLE the differentiable oracle for the non-flat regimes (2026-06-18): the learned
        # connection (connection_W / connection_M) enters the loss ONLY through the unrolled E-step,
        # so it needs oracle_unroll_grad=True to receive a gradient. Enable it here rather than only
        # warning the user to set it. Inert under e_step_gradient != 'unroll' / detach_e_step (where
        # the E-step tangent is severed regardless -- those paths keep their own freeze warnings).
        if self.transport_mode in ("regime_ii", "regime_ii_covariant") and not self.oracle_unroll_grad:
            self.oracle_unroll_grad = True
        _routes_to_oracle = (
            self.transport_mode in ("regime_ii", "regime_ii_covariant")
            or (self.pos_rotation == "rope" and not self.rope_on_value)
            or not (
                self.gradient_mode == "filtering"
                and self.family == "gaussian_diagonal"
                and self.divergence_family == "renyi"
                and self.renyi_order == 1.0
                and self.include_attention_entropy
            )
        )
        if (
            self.e_step_gradient == "unroll"
            and not self.oracle_unroll_grad
            and _routes_to_oracle
            and (
                self.lambda_alpha_mode == "learnable"
                or self.transport_mode in ("regime_ii", "regime_ii_covariant")
                or self.learnable_lambda_beta
                or (self.lambda_h_mode == "learnable" and self.s_e_step)
                or (self.learnable_r and self.r_update_mode == "gradient" and self.s_e_step)
                or self.pos_phi == "learned"
            )
        ):
            import warnings
            warnings.warn(
                "e_step_gradient='unroll' but this family/gradient_mode routes the belief gradient to "
                "the autograd ORACLE (NOT the closed-form kernel: that covers only "
                "gradient_mode='filtering' + family='gaussian_diagonal' + divergence_family='renyi' + "
                "renyi_order=1.0 + include_attention_entropy=True), which returns a DETACHED tangent while "
                "oracle_unroll_grad=False. A learnable parameter that enters the loss only through the "
                "E-step tangent (log_alpha under lambda_alpha_mode='learnable', connection_W under "
                "transport_mode='regime_ii', log_lambda_beta under learnable_lambda_beta, log_lambda_h "
                "under lambda_h_mode='learnable'+s_e_step, r_mu/r_sigma_log under "
                "learnable_r+r_update_mode='gradient'+s_e_step, pos_phi_free under pos_phi='learned') "
                "therefore receives NO gradient and stays frozen. Set "
                "oracle_unroll_grad=True to make the oracle return a differentiable (unrolled) gradient.",
                UserWarning,
                stacklevel=2,
            )
        if self.e_mu_q_trust is not None and self.e_mu_q_trust <= 0.0:
            raise ValueError(f"e_mu_q_trust must be > 0 or None, got {self.e_mu_q_trust}")
        if self.mu_trust_mode not in ("box", "ball"):
            raise ValueError(f"mu_trust_mode must be 'box' or 'ball', got {self.mu_trust_mode!r}")
        if self.e_step_mu_precond not in ("fisher", "raw"):
            raise ValueError(f"e_step_mu_precond must be 'fisher' or 'raw', got {self.e_step_mu_precond!r}")
        for name in ("m_p_mu_lr", "m_p_sigma_lr", "m_phi_lr", "weight_decay", "phi_weight_decay", "min_lr", "min_lr_frac"):
            v = getattr(self, name)
            if v < 0.0 or v != v:                            # v != v rejects NaN (which passes < 0.0)
                raise ValueError(f"{name} must be >= 0 (and not NaN), got {v}")
        if self.connection_weight_decay is not None and (
                self.connection_weight_decay < 0.0
                or self.connection_weight_decay != self.connection_weight_decay):
            raise ValueError(
                f"connection_weight_decay must be >= 0 (and not NaN) or None, "
                f"got {self.connection_weight_decay}"
            )
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
        if self.resume_from is not None and not isinstance(self.resume_from, str):
            raise ValueError(
                f"resume_from must be None or a path string to a checkpoints/step_<N>.pt, got "
                f"{type(self.resume_from).__name__}"
            )
        if self.eval_max_batches is not None and self.eval_max_batches < 1:
            raise ValueError(f"eval_max_batches must be >= 1 if set, got {self.eval_max_batches}")
        # EMA decay validated only when the averaging is on (inert otherwise -> the OFF path never reads it).
        if self.use_ema and not (0.0 < self.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1) when use_ema=True, got {self.ema_decay}")
        # amp_dtype: None (default, OFF) = pure fp32 / no autocast; 'bf16' / 'fp16' enable autocast.
        # None is a legal member here, so _require rejects 'fp32' and any other garbage. fp16 is
        # accepted for FORWARD/inference; fp16 TRAINING still needs a GradScaler in the M-step
        # (train.py) -- a documented buildout -- so it is not enforced-rejected here (that would also
        # block the legitimate fp16 inference path that tests/test_amp.py pins).
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
        When kappa is a per-head list, returns the mean tau (for logging only).
        """
        k = float(sum(self.kappa_beta) / len(self.kappa_beta)) if isinstance(self.kappa_beta, (list, tuple)) else self.kappa_beta
        return k * (self.d_head ** 0.5)

    @property
    def tau_gamma(self) -> float:
        """Model-channel softmax temperature tau_gamma = kappa_gamma * sqrt(d_head).

        The gamma model-coupling block's own temperature handle, mirroring `tau` for the belief
        beta block (kappa_gamma=1 -> Vaswani sqrt(d_k) per head). Consumed by the gamma block's
        reduced_free_energy as the -tau_gamma log Z^s envelope temperature.
        When kappa_gamma is a per-head list, returns the mean tau_gamma (for logging only).
        """
        k = float(sum(self.kappa_gamma) / len(self.kappa_gamma)) if isinstance(self.kappa_gamma, (list, tuple)) else self.kappa_gamma
        return k * (self.d_head ** 0.5)

    @property
    def diagonal_covariance(self) -> bool:
        """Whether the belief covariance is diagonal -- DERIVED from ``family``.

        ``family`` is the single covariance-structure toggle; this returns its declared
        ``cov_kind == 'diagonal'`` (gaussian_diagonal -> True, gaussian_full -> False). Read-only:
        switch the covariance structure by setting ``family``, never this. Consumers that want a
        fast boolean (e.g. PriorBank) read it once at construction.
        """
        from vfe3.divergence import family_cov_kind
        return family_cov_kind(self.family) == "diagonal"

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


def _require(value: Optional[str], valid: Tuple[object, ...], name: str) -> None:
    """Raise ValueError unless ``value`` is one of ``valid``."""
    if value not in valid:
        raise ValueError(f"{name} must be one of {valid}, got {value!r}")
