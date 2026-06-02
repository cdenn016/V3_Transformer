"""Configuration for VFE_3.0. Single dataclass, single validation block.

No CLI parsing (project policy: click-to-run). Edit fields directly, then run.
Every registry-backed seam is selected here by name (divergence, gauge group,
encode/decode mode, alpha form, attention prior, norm, gradient mode), so a
variant swaps without editing call sites.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

_VALID_GAUGE_GROUPS        = ("glk", "block_glk", "tied_block_glk", "so_k")
_VALID_GAUGE_PARAM         = ("phi", "omega_direct")
_VALID_ENCODE_MODES        = ("per_token", "gauge_fixed")
_VALID_DECODE_MODES        = ("diagonal", "diagonal_chunked", "full")
_VALID_GRADIENT_MODES      = ("filtering", "smoothing")
_VALID_ALPHA_MODES         = ("constant", "state_dependent", "state_dependent_per_coord", "learnable")
_VALID_PHI_PRECOND_MODES   = ("none", "clip", "killing", "killing_per_block", "pullback")
_VALID_PHI_RETRACT_MODES   = ("euclidean", "bch")
_VALID_ATTENTION_PRIORS    = ("uniform", "causal", "alibi")
_VALID_NORMS               = ("none", "mahalanobis")
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
    mass_phi:                  float = 0.0          # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight: float = 0.0         # alpha_hat * sum_i KL(q_i*||p_i) M-step term (0 = OFF)
    # Hyper-prior weight lambda_h on the model-channel term lambda_h * mean_i KL(s_i||r)
    # (manuscript Participatory_it_from_bit.tex eq:pointwise_free_energy, lines 1241-1249).
    # Default 0.0 = OFF: no s/r tables, loss byte-identical to the single-tier path.
    # FIRST INCREMENT: this wires the second (model) belief channel s_i + the global hyper-prior
    # r end-to-end at the smallest scope; s_i does NOT yet couple into the belief q / the
    # prediction path, and the gamma model-coupling block + the s-channel E-step update are
    # DEFERRED to increment 2.
    lambda_h:                  float = 0.0

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
    seed:                      int   = 0
    log_interval:              int   = 50           # console log every N steps (0 = off)
    eval_interval:             int   = 0            # periodic validation every N steps (0 = off)
    checkpoint_interval:       int   = 0            # save a resumable checkpoint every N steps (0 = off)
    eval_max_batches:          Optional[int] = None # cap the PERIODIC eval pass (None = full split; pure path)
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

        # gauge seam
        _require(self.gauge_group, _VALID_GAUGE_GROUPS, "gauge_group")
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
        # (live + enforced) rather than silently aliased to 'phi'. compute_transport_operators_direct
        # remains for an external-Omega regime this belief design does not provide.
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
        if self.lambda_h < 0.0:
            raise ValueError(f"lambda_h must be >= 0, got {self.lambda_h}")
        for name in ("b0", "c0"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        _require(self.alpha_mode, _VALID_ALPHA_MODES, "alpha_mode")
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

        # attention
        _require(self.attention_prior, _VALID_ATTENTION_PRIORS, "attention_prior")

        # E-step
        for name in ("e_mu_lr", "e_sigma_lr", "e_phi_lr"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        _require(self.gradient_mode, _VALID_GRADIENT_MODES, "gradient_mode")
        _require(self.phi_precond_mode, _VALID_PHI_PRECOND_MODES, "phi_precond_mode")
        _require(self.phi_retract_mode, _VALID_PHI_RETRACT_MODES, "phi_retract_mode")
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

        # normalization
        _require(self.norm_type_block, _VALID_NORMS, "norm_type_block")
        _require(self.norm_type_final, _VALID_NORMS, "norm_type_final")

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
        for name in ("m_mu_lr", "m_sigma_lr", "m_phi_lr", "weight_decay"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
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
        """Attention softmax temperature tau = kappa * sqrt(d_head).

        Per-head dimension d_head = embed_dim // n_heads, so kappa=1 recovers standard
        scaled dot-product attention (Vaswani sqrt(d_k)) PER HEAD. Audit finding 6c: the
        manuscript's free-energy functional (eq:pointwise) writes tau = kappa*sqrt(K) over
        the full belief, but its standard-attention recovery is derived per-head with
        sqrt(d_k); the code follows the recovery convention so that kappa=1 is the
        Vaswani temperature.
        """
        return self.kappa * (self.d_head ** 0.5)

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


def _require(value: str, valid: tuple, name: str) -> None:
    """Raise ValueError unless ``value`` is one of ``valid``."""
    if value not in valid:
        raise ValueError(f"{name} must be one of {valid}, got {value!r}")
