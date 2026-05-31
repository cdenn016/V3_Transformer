"""Configuration for VFE_3.0. Single dataclass, single validation block.

No CLI parsing (project policy: click-to-run). Edit fields directly, then run.
Every registry-backed seam is selected here by name (divergence, gauge group,
encode/decode mode, alpha form, attention prior, norm, gradient mode), so a
variant swaps without editing call sites.
"""

from dataclasses import dataclass
from typing import Optional

_VALID_DIVERGENCE_FAMILIES = ("gaussian_diagonal", "gaussian_full")
_VALID_DIVERGENCE_FUNCTIONALS = ("renyi",)
_VALID_GAUGE_GROUPS        = ("glk", "block_glk", "so_k")
_VALID_GAUGE_PARAM         = ("phi", "omega_direct")
_VALID_ENCODE_MODES        = ("per_token", "gauge_fixed")
_VALID_DECODE_MODES        = ("diagonal", "full")
_VALID_GRADIENT_MODES      = ("filtering", "smoothing")
_VALID_ALPHA_MODES         = ("constant", "state_dependent", "state_dependent_per_coord")
_VALID_PHI_PRECOND_MODES   = ("none", "clip", "killing", "killing_per_block", "pullback")
_VALID_PHI_RETRACT_MODES   = ("euclidean", "bch")
_VALID_ATTENTION_PRIORS    = ("uniform", "causal", "alibi")
_VALID_NORMS               = ("none", "mahalanobis")


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

    # belief family
    diagonal_covariance:       bool  = True
    family:                    str   = "gaussian_diagonal"

    # free-energy coupling
    alpha:                     float = 1.0          # constant self-coupling value
    alpha_mode:                str   = "constant"
    b0:                        float = 1.0          # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0:                        float = 1.0          # state-dependent alpha shape (numerator)
    kappa:                     float = 1.0          # temperature tau = kappa * sqrt(K)
    mass_phi:                  float = 0.0          # (mass_phi/2) ||phi||^2 penalty

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

    # decode / encode
    use_prior_bank:            bool  = True
    decode_tau:                float = 1.0
    decode_mode:               str   = "diagonal"
    encode_mode:               str   = "per_token"

    # cross-block belief handoff (mu_q -> mu_p)
    prior_handoff_rho:         float = 1.0          # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma:       float = 0.0          # sigma damping (0.0 = frozen at embedding)

    # normalization
    norm_type_block:           str   = "none"
    norm_type_final:           str   = "none"

    # M-step / training
    detach_e_step:             bool  = False        # False = unroll E-step in the training graph
    m_mu_lr:                   float = 0.025
    m_sigma_lr:                float = 0.0025
    m_phi_lr:                  float = 0.015
    weight_decay:              float = 0.05
    batch_size:                int   = 64
    max_steps:                 int   = 15000
    warmup_steps:              int   = 100
    seed:                      int   = 0

    def __post_init__(self) -> None:
        # numerics
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.kl_max <= 0.0:
            raise ValueError(f"kl_max must be positive, got {self.kl_max}")

        # divergence seam: divergence_family is the FUNCTIONAL (f-divergence) registry key
        # (renyi, ...), distinct from `family` (the covariance-structure kernel). alpha_div is
        # the Renyi order. Both are live, modular seams (CLAUDE.md: slot in different f-divergences).
        _require(self.divergence_family, _VALID_DIVERGENCE_FUNCTIONALS, "divergence_family")
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

        # belief family. ``family`` selects the covariance-structure divergence kernel
        # (gaussian_diagonal | gaussian_full); ``diagonal_covariance`` is a SEPARATE live bool,
        # cross-validated to stay consistent with it (kept distinct per the modularity design,
        # not collapsed). It threads into the PriorBank encode to choose diagonal vs full SPD.
        _require(self.family, _VALID_DIVERGENCE_FAMILIES, "family")
        if self.diagonal_covariance != (self.family == "gaussian_diagonal"):
            raise ValueError(
                f"diagonal_covariance={self.diagonal_covariance} contradicts family={self.family!r}; "
                f"set diagonal_covariance={self.family == 'gaussian_diagonal'} for this family"
            )

        # free-energy coupling
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.mass_phi < 0.0:
            raise ValueError(f"mass_phi must be >= 0, got {self.mass_phi}")
        for name in ("b0", "c0"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        _require(self.alpha_mode, _VALID_ALPHA_MODES, "alpha_mode")

        # attention
        _require(self.attention_prior, _VALID_ATTENTION_PRIORS, "attention_prior")

        # E-step
        for name in ("e_mu_lr", "e_sigma_lr", "e_phi_lr"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        _require(self.gradient_mode, _VALID_GRADIENT_MODES, "gradient_mode")
        _require(self.phi_precond_mode, _VALID_PHI_PRECOND_MODES, "phi_precond_mode")
        _require(self.phi_retract_mode, _VALID_PHI_RETRACT_MODES, "phi_retract_mode")

        # decode / encode
        if self.decode_tau <= 0.0:
            raise ValueError(f"decode_tau must be positive, got {self.decode_tau}")
        _require(self.decode_mode, _VALID_DECODE_MODES, "decode_mode")
        _require(self.encode_mode, _VALID_ENCODE_MODES, "encode_mode")
        # The PriorBank IS the only encode/decode boundary; there is no specified alternative.
        # use_prior_bank=False is a live knob that rejects the unsupported value rather than
        # silently doing nothing (it was a dead config seam).
        if not self.use_prior_bank:
            raise NotImplementedError(
                "use_prior_bank=False has no alternative encode/decode path; the PriorBank is "
                "the only belief-encode/decode boundary in VFE_3.0"
            )
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
        for name in ("m_mu_lr", "m_sigma_lr", "m_phi_lr", "weight_decay"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    @property
    def tau(self) -> float:
        """Attention softmax temperature tau = kappa * sqrt(K)."""
        return self.kappa * (self.embed_dim ** 0.5)

    @property
    def d_head(self) -> int:
        """Per-head belief dimension K // n_heads."""
        return self.embed_dim // self.n_heads


def _require(value: str, valid: tuple, name: str) -> None:
    """Raise ValueError unless ``value`` is one of ``valid``."""
    if value not in valid:
        raise ValueError(f"{name} must be one of {valid}, got {value!r}")
