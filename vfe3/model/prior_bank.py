r"""PriorBank for VFE_3.0: learnable Gaussian vocab priors + the KL decode boundary.

Holds the per-vocabulary prior pi_v = N(mu_v, Sigma_v) with gauge frame phi_v as
PARAMETER TABLES (nn.Parameter -- priors, not neural maps; the no-NN rule bans
nn.Linear/MLP/activations, not learnable parameters). encode(token_ids) looks them
up into the initial belief (q = p); decode(mu_q, sigma_q) scores the posterior
against every prior as logits = -KL(q || pi_v)/tau_eff (the divergence seam),
replacing a linear output projection.

Modularity:
    encode_mode registry -- ``per_token`` (table lookup, default); ``gauge_fixed`` a
        named stub (gauge orbit from a shared base belief).
    decode_mode registry -- ``diagonal`` (fused closed form, default); ``diagonal_chunked``
        (fused decode+CE, inference delegates to ``diagonal``); ``full`` (exact full-covariance
        Cholesky decode); ``full_chunked`` (full-cov KL via the diagonal-prior closed form);
        ``family`` / ``family_chunked`` (family/divergence-consistent decode: logits =
        -D_configured(q || pi_v)/tau_eff through the CONFIGURED family and divergence functional,
        both covariance ranks); ``expected_likelihood_chunked`` (log N(mu_q; mu_v, Sigma_q + Sigma_v)
        Gaussian-convolution scoring, diagonal only); plus the registered-but-config-excluded
        ``linear`` ablation kernel (reached via use_prior_bank=False).

Decode seam (PB-14): the family-consistent ``family``/``family_chunked`` kernels AND the
authoritative ``reference_decode`` score logits = -D_configured(q || pi_v)/tau_eff through the
CONFIGURED family (``self.family``) and divergence functional (``self.divergence_family`` at
``self.renyi_order``), so the readout matches the E-step geometry. The fast ``diagonal``/``full``
kernels remain the OPTIMIZED gaussian_* + renyi(alpha=1) implementations (they hardcode gaussian
alpha=1 KL and ignore divergence_family/renyi_order); config pairs those single-rank kernels only
with a canonical gaussian/renyi/alpha=1 seam, and REQUIRES a ``family_consistent`` decoder for any
non-Gaussian family or noncanonical divergence under ``use_prior_bank=True``. The registry seam is
honored at the COVARIANCE-STRUCTURE granularity (``DecodeRegistration.covariance_kinds``): a new
covariance structure or a new family-consistent readout is added by writing-and-registering a decode
kernel, never by editing a call site. The full kernels score a full q against the intentionally
DIAGONAL vocabulary-prior table (promoted with diag_embed only when the family is full).
``reference_decode`` is the slow per-V seam-call cross-check the fused canonical kernels are pinned
to EXACTLY (and under ``log_softmax``) on the canonical path.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Protocol, Tuple

import torch
import torch.utils.checkpoint as _checkpoint
from torch import nn

from vfe3.belief import BeliefState
from vfe3.divergence import family_cov_kind, get_family, get_functional, kl
from vfe3.families.base import _logdet_chol
from vfe3.families.covariance_tables import (
    covariance_from_packed,
    packed_from_covariance,
    packed_strict_lower_size,
)
from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.numerics import bounded_variance_from_log, safe_cholesky


# ---------------------------------------------------------------------------
# Registries: mode name -> callable. Variants swap by config; add a variant by
# writing-and-registering it, never by editing call sites.
#   encode: fn(pb, token_ids) -> BeliefState
#   decode: fn(pb, mu_q, sigma_q, tau_eff) -> logits (B, N, V)
# ---------------------------------------------------------------------------
_ENCODERS: 'Dict[str, EncodeCallable]' = {}


@dataclass(frozen=True)
class DecodeRegistration:
    """A decode callable and all routing capabilities attached to that callable.

    ``covariance_kinds`` is the resolved set of family covariance structures the decoder scores
    ("diagonal" and/or "full"); config validates ``family_cov_kind(cfg.family) in covariance_kinds``
    rather than treating ``supports_full`` as an exclusive rank bit, so a dual-rank decoder (e.g.
    ``family``) accepts BOTH a diagonal and a full family. ``supports_full`` is retained (public,
    read by legacy callers) and stays coherent with the set: it is ``"full" in covariance_kinds``.
    ``family_consistent`` flags a decoder that scores logits = -D_configured(q||p_v)/tau_eff through
    the CONFIGURED family AND divergence functional (as opposed to the fast kernels' hardcoded
    gaussian alpha=1 KL); config requires a family_consistent decoder for any non-Gaussian family or
    noncanonical divergence under ``use_prior_bank=True``.

    Direct construction ``DecodeRegistration(callable, supports_full, supports_chunked, fused_ce)``
    stays source-compatible: the two new fields default, and ``__post_init__`` derives the legacy
    singleton ``covariance_kinds`` from ``supports_full`` when it is not supplied.
    """

    callable:          'DecodeCallable'
    supports_full:     bool
    supports_chunked:  bool
    fused_ce:          'Optional[FusedCECallable]'
    family_consistent: bool                        = False
    covariance_kinds:  'Optional[FrozenSet[str]]'  = None

    def __post_init__(self) -> None:
        # Resolve the covariance-kind set. Omitted -> the legacy singleton derived from
        # supports_full (a frozen dataclass, so the resolved value is written via object.__setattr__).
        if self.covariance_kinds is None:
            object.__setattr__(
                self, "covariance_kinds",
                frozenset({"full"} if self.supports_full else {"diagonal"}),
            )
        else:
            object.__setattr__(self, "covariance_kinds", frozenset(self.covariance_kinds))


_DECODERS: Dict[str, DecodeRegistration] = {}

# Once-per-process guard for the decode_unigram_prior=True-with-unset-table warning
# (the decode then degenerates to the current uniform-prior behavior).
_WARNED_UNIGRAM_UNSET: bool = False


def register_encode(
    name: str,

    *,
    override: bool = False,
) -> 'Callable[[EncodeCallable], EncodeCallable]':
    """Decorator registering an encode kernel under ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: 'EncodeCallable') -> 'EncodeCallable':
        if name in _ENCODERS and not override:
            raise KeyError(f"encode mode {name!r} already registered; pass override=True to replace")
        _ENCODERS[name] = fn
        return fn
    return _wrap


def get_encode(name: str) -> 'EncodeCallable':
    """Return the registered encode kernel for ``name`` (KeyError if absent)."""
    if name not in _ENCODERS:
        raise KeyError(
            f"no encode mode registered under {name!r}; available: {sorted(_ENCODERS)}"
        )
    return _ENCODERS[name]


def register_decode(
    name: str,

    *,
    supports_full:     Optional[bool]              = None,
    supports_chunked:  bool                        = False,
    override:          bool                        = False,
    family_consistent: bool                        = False,
    fused_ce:          'Optional[FusedCECallable]'      = None,
    covariance_kinds:  'Optional[FrozenSet[str]]'       = None,
) -> 'Callable[[DecodeCallable], DecodeCallable]':
    """Decorator registering a decode kernel under ``name``.

    ``covariance_kinds`` is the resolved set of family covariance structures the decoder scores.
    OMITTED -> derive the legacy singleton from ``supports_full`` (``{"full"}`` when True, else
    ``{"diagonal"}``); SUPPLIED -> derive ``supports_full`` from membership (``"full" in kinds``) and
    reject an explicitly contradictory legacy ``supports_full``. Every existing
    ``register_decode(..., supports_full=True|False)`` call therefore keeps its old behavior.
    ``family_consistent`` marks a decoder that reads logits out through the CONFIGURED family and
    divergence functional. ``supports_chunked`` advertises a fused chunked-CE training path, whose
    callable is ``fused_ce``. The callable and all capabilities are replaced atomically, so an
    override cannot retain stale routing metadata from the prior registration.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    if covariance_kinds is None:
        resolved_full  = bool(supports_full) if supports_full is not None else False
        resolved_kinds = frozenset({"full"} if resolved_full else {"diagonal"})
    else:
        resolved_kinds = frozenset(covariance_kinds)
        if not resolved_kinds or not resolved_kinds <= {"diagonal", "full"}:
            raise ValueError(
                f"decode mode {name!r} covariance_kinds must be a nonempty subset of "
                f"{{'diagonal', 'full'}}, got {sorted(resolved_kinds)}"
            )
        resolved_full = "full" in resolved_kinds
        if supports_full is not None and bool(supports_full) != resolved_full:
            raise ValueError(
                f"decode mode {name!r} has contradictory metadata: supports_full={supports_full} "
                f"but covariance_kinds={sorted(resolved_kinds)} implies supports_full={resolved_full}"
            )

    def _wrap(fn: 'DecodeCallable') -> 'DecodeCallable':
        if name in _DECODERS and not override:
            raise KeyError(f"decode mode {name!r} already registered; pass override=True to replace")
        if supports_chunked != (fused_ce is not None):
            raise ValueError(
                f"decode mode {name!r} must declare supports_chunked=True exactly when fused_ce "
                f"is provided"
            )
        _DECODERS[name] = DecodeRegistration(
            callable=fn,
            supports_full=resolved_full,
            supports_chunked=supports_chunked,
            fused_ce=fused_ce,
            family_consistent=family_consistent,
            covariance_kinds=resolved_kinds,
        )
        return fn
    return _wrap


def get_decode_registration(name: str) -> DecodeRegistration:
    """Return the complete registration record for ``name`` (KeyError if absent)."""
    if name not in _DECODERS:
        raise KeyError(
            f"no decode mode registered under {name!r}; available: {sorted(_DECODERS)}"
        )
    return _DECODERS[name]


def get_decode(name: str) -> 'DecodeCallable':
    """Return the registered decode kernel for ``name`` (KeyError if absent)."""
    return get_decode_registration(name).callable


class PriorBank(nn.Module):
    r"""Learnable Gaussian vocab priors; encode (lookup) and decode (-KL/tau_eff).

    The tables ``mu_embed`` (V, K), ``sigma_log_embed`` (V, K), ``phi_embed`` (V, n_gen)
    parameterize the priors pi_v = N(mu_v, exp(sigma_log_v)) with gauge frame phi_v.
    They are PRIORS (nn.Parameter), not a neural map: there is no nn.Linear/MLP/activation
    anywhere in this module. The learnable scalar ``decode_log_scale`` tunes the decode
    temperature.
    """

    output_proj_weight: Optional[nn.Parameter]   # (V, K) linear-decode weight; None unless use_prior_bank=False
    output_proj_bias:   Optional[nn.Parameter]   # (V,) linear-decode log-unigram bias; None unless use_prior_bank=False and decode_bias

    def __init__(
        self,
        vocab_size:   int,
        K:            int,
        n_gen:        int,

        *,
        mu_init_std:         float = 0.02,
        sigma_init:          float = 1.0,
        phi_scale:           float = 0.01,
        decode_tau:          float = 1.0,
        eps:                 float = 1e-6,
        diagonal_covariance: bool  = True,
        family:              str   = "gaussian_diagonal",
        divergence_family:   str   = "renyi",
        renyi_order:         float = 1.0,
        use_prior_bank:      bool  = True,
        decode_bias:         bool  = False,
        encode_mode:         str   = "per_token",
        decode_mode:         str   = "diagonal",
        decode_chunk_size:   int   = 8192,
        lambda_h:            float = 0.0,
        lambda_gamma:        float = 0.0,
        prior_source:        str   = "token",
        s_frame_mode:        str   = "tied",
        s_e_step:            bool  = False,
        learnable_r:         bool  = False,

        unigram_kappa:        float = 1.0,
        decode_unigram_prior: bool  = False,
        untie_decode_bank:    bool  = False,

        gauge_parameterization: str                 = "phi",
        irrep_dims:             Optional[List[int]] = None,
        omega_reflection:       str                 = "off",
        phi_reflection:         str                 = "off",
        omega_compact_storage:  bool                = False,
        gauge_group_is_tied:    bool                = False,
        gauge_group_name:       Optional[str]       = None,
    ) -> None:
        super().__init__()
        if gauge_parameterization == "omega_direct" and encode_mode == "per_token_additive":
            raise ValueError(
                "gauge_parameterization='omega_direct' is incompatible with "
                "encode_mode='per_token_additive': the additive encoder returns no stored omega "
                "frame. Use encode_mode='per_token', or gauge_parameterization='phi' for the "
                "additive control."
            )
        if type(omega_compact_storage) is not bool:
            raise ValueError(
                "omega_compact_storage must be a bool, got "
                f"{type(omega_compact_storage).__name__}: {omega_compact_storage!r}"
            )
        if omega_compact_storage:
            compact_groups = {"block_glk", "tied_block_glk"}
            if gauge_group_name not in compact_groups:
                raise ValueError(
                    "omega_compact_storage requires explicit gauge_group_name='block_glk' or "
                    f"'tied_block_glk'; got {gauge_group_name!r}")
            expected_tied = gauge_group_name == "tied_block_glk"
            if gauge_group_is_tied != expected_tied:
                raise ValueError(
                    "gauge_group_is_tied is inconsistent with gauge_group_name: "
                    f"group={gauge_group_name!r}, tied={gauge_group_is_tied!r}")
            if irrep_dims is None:
                raise ValueError("omega_compact_storage requires explicit irrep_dims")
            if len(irrep_dims) <= 1:
                raise ValueError(
                    "omega_compact_storage requires more than one irrep block; "
                    f"got irrep_dims={irrep_dims!r}")
            if any(type(d) is not int or d <= 0 for d in irrep_dims):
                raise ValueError(
                    "omega_compact_storage requires every irrep dimension to be a positive int; "
                    f"got irrep_dims={irrep_dims!r}")
            if len(set(irrep_dims)) != 1:
                raise ValueError(
                    "omega_compact_storage requires equal irrep dimensions; "
                    f"got irrep_dims={irrep_dims!r}")
            if sum(irrep_dims) != K:
                raise ValueError(
                    f"omega_compact_storage requires sum(irrep_dims)==K; "
                    f"got sum={sum(irrep_dims)}, K={K}")
        self.vocab_size = vocab_size
        self.K = K
        self.n_gen = n_gen
        self.decode_tau = decode_tau
        self.eps = eps
        self.diagonal_covariance = diagonal_covariance
        # family drives the model-channel (s/r) covariance rank: 'full' -> packed strict-lower
        # Cholesky tables (SPD covariance), else the diagonal log-variance tables. The vocabulary
        # prior and decode variance tables stay diagonal in EVERY family (PB-11).
        self.family = family
        # divergence_family / renyi_order drive the family-consistent decode kernels
        # (decode_mode='family'/'family_chunked'): logits = -D_configured(q||p_v)/tau_eff scored
        # through get_functional(divergence_family) at alpha=renyi_order. The fast gaussian kernels
        # (diagonal/full) ignore them (they hardcode gaussian alpha=1 KL); config only pairs those
        # with a canonical gaussian/renyi/alpha=1 seam. Defaults reproduce the old fixed-KL readout.
        self.divergence_family = divergence_family
        self.renyi_order = renyi_order
        self._s_cov_kind = family_cov_kind(family)
        self.use_prior_bank = use_prior_bank
        self.encode_mode = encode_mode
        self.decode_mode = decode_mode
        self.decode_chunk_size = decode_chunk_size
        self.prior_source = prior_source
        self.s_frame_mode = s_frame_mode
        self.s_e_step = s_e_step
        self.unigram_kappa = unigram_kappa
        self.decode_unigram_prior = decode_unigram_prior
        self.gauge_parameterization = gauge_parameterization
        self.gauge_group_name = gauge_group_name
        self.irrep_dims = irrep_dims
        # untie applies to the KL-to-bank decode only (the linear ablation is already untied by
        # construction), so the flag is resolved against use_prior_bank once, here.
        self.untie_decode_bank = untie_decode_bank and use_prior_bank

        sigma_log_init = float(torch.log(torch.tensor(sigma_init)))
        self.mu_embed         = nn.Parameter(mu_init_std * torch.randn(vocab_size, K))
        self.sigma_log_embed  = nn.Parameter(torch.full((vocab_size, K), sigma_log_init))
        self.phi_embed        = nn.Parameter(phi_scale * torch.randn(vocab_size, n_gen))
        if s_frame_mode == "phi_tilde":
            self.s_phi_embed = nn.Parameter(self.phi_embed.detach().clone())
        self.decode_log_scale = nn.Parameter(torch.zeros(1))

        # Arm-2 control (encode_mode='per_token_additive'): a NON-structural use of the SAME learned
        # (V, n_gen) phi table. A FROZEN random readout R (K, n_gen) maps each token's n_gen-dim code
        # to an additive K-dim mean shift, and encode returns phi=0 so Omega = exp(phi.G) = I (no gl(g)
        # transport). Isolates raw phi-table CAPACITY (V*n_gen learned params, matched to the gauge
        # cell) from the gl(g) generator STRUCTURE. R is a buffer (not a Parameter, so learned-param
        # count is unchanged), seeded for reproducibility, scaled 1/sqrt(n_gen) so the per-dim shift
        # std matches phi_scale at init. Deliberately breaks gauge equivariance -- that IS the control.
        if encode_mode == "per_token_additive":
            _r_gen = torch.Generator().manual_seed(0)
            self.register_buffer(
                "additive_R",
                torch.randn(K, n_gen, generator=_r_gen) / (float(n_gen) ** 0.5),
            )

        # use_prior_bank=False (linear-decode ablation): decode is a plain linear projection
        # logits = mu_q @ W^T through a learned (V, K) weight, the single authorized neural
        # exception (a lone linear output readout; see CLAUDE.md). Realized as a raw nn.Parameter
        # matmul -- NOT an nn.Linear/MLP -- so no neural-layer class enters the module. Created
        # only on the ablation path so the pure path (use_prior_bank=True) carries no extra weight.
        # Xavier-uniform init (PyTorch's nn.Linear default), no bias (a constant shift in
        # V that softmax/cross-entropy absorbs). Encode stays the prior-bank lookup either way.
        if use_prior_bank:
            self.output_proj_weight = None
            self.output_proj_bias   = None
        else:
            self.output_proj_weight = nn.Parameter(torch.empty(vocab_size, K))
            nn.init.xavier_uniform_(self.output_proj_weight)
            # Optional per-vocab bias (decode_bias): a *per-class* bias is NOT a softmax-invariant
            # constant shift -- it is a learned log-unigram prior, and a 50k Zipfian vocab is the
            # opposite of balanced, so the bias-free map can represent token base rates only by
            # spending rank-K mean capacity. Zero-init -> logits bit-identical to decode_bias=False
            # at construction (drawn AFTER the weight, so the weight's RNG is unchanged); the CE
            # gradient drives it toward log p(token). Routed to a weight-decay-free optimizer group
            # in build_optimizer (decaying a unigram prior toward zero biases it to flat).
            self.output_proj_bias = (
                nn.Parameter(torch.zeros(vocab_size)) if decode_bias else None
            )

        # MODEL CHANNEL (manuscript eq:pointwise_free_energy), default-OFF. The model-channel belief
        # tables s_mu_embed/s_sigma_log_embed (V, K) -- a per-token DIAGONAL Gaussian s_i looked up
        # like the belief tables -- back BOTH the hyper-prior term lambda_h*KL(s||r) and the gamma
        # model-coupling block lambda_gamma*F_red^s, so they are created whenever EITHER channel is
        # active (lambda_h>0 OR lambda_gamma>0). The global hyper-prior r_mu/r_sigma_log (K,) -- a
        # single diagonal Gaussian the s_i are regularized toward (the manuscript centroid) -- is
        # consumed ONLY by the hyper-prior term, so it stays gated on lambda_h>0. These are PRIORS
        # (nn.Parameter), not a neural map. They are created LAST and only on the active-channel path:
        # the default (both 0) path draws zero new RNG, so the belief tables above are byte-unchanged
        # and the pure path is param-free. s drawn BEFORE r preserves the existing lambda_h>0 RNG order
        # (byte-identical to the hyper-prior-only build). s init mirrors the belief tables (small mu,
        # sigma matching sigma_init); r init: mu=0, sigma matching sigma_init -- so s != r at init
        # (KL(s||r) > 0, the channel has a gradient). When prior_source='model_channel', these same
        # tables supply the belief prior p, including their packed full covariance; s_e_step may
        # additionally refine that prior before the belief-channel E-step.
        if lambda_h > 0.0 or lambda_gamma > 0.0 or prior_source == "model_channel" or s_e_step:
            self.s_mu_embed        = nn.Parameter(mu_init_std * torch.randn(vocab_size, K))
            self.s_sigma_log_embed = nn.Parameter(torch.full((vocab_size, K), sigma_log_init))
            if self._s_cov_kind == "full":
                # gaussian_full model channel (PB-11): the packed strict-lower Cholesky (V, K*(K-1)//2)
                # completing s_sigma_log_embed's diagonal into a full SPD covariance L L^T. ZERO-init
                # (torch.zeros, no RNG) so the initial model-channel covariances are exactly diagonal
                # AND the RNG order of every subsequent table is byte-unchanged from the pre-PB-11 build.
                # Diagonal/Laplace channels create no packed key -> pure diagonal state_dict is identical.
                self.s_sigma_lower_embed = nn.Parameter(
                    torch.zeros(vocab_size, packed_strict_lower_size(K)))
        if lambda_h > 0.0 or s_e_step:
            # Hyper-prior centroid r (r_mu, r_sigma_log): the centroid the model beliefs s_i are
            # regularized toward via lambda_h*KL(s_i||r). DEFAULT FROZEN (learnable_r=False,
            # requires_grad=False): the fixed centroid the manuscript determines "from a higher, slower
            # meta-level" (GL(K)_supplementary.tex:1081); with no meta-level built, a FIXED r is the
            # manuscript-consistent stand-in, and freezing prevents the KL(s||r)->0 collapse that freely
            # training r alongside an unanchored s would cause. learnable_r=True un-freezes r as an
            # empirical-Bayes population centroid (grouped in build_optimizer like the s tables);
            # meaningful only when s carries an independent data force (prior_source='model_channel'),
            # which VFE3Config.__post_init__ warns about.
            self.r_mu              = nn.Parameter(torch.zeros(K), requires_grad=learnable_r)
            self.r_sigma_log       = nn.Parameter(torch.full((K,), sigma_log_init), requires_grad=learnable_r)
            if self._s_cov_kind == "full":
                # The packed strict-lower Cholesky of the centroid r (gaussian_full, PB-11): zero-init
                # (r starts diagonal), grouped/frozen exactly like r_sigma_log via learnable_r.
                self.r_sigma_lower = nn.Parameter(
                    torch.zeros(packed_strict_lower_size(K)), requires_grad=learnable_r)
            # DESIGN NOTE (audit 2026-06-15): the token-dependent top-down hyper-prior
            # r_i = Omega_tilde[s_I^{(s+1)}] (PIFB eq:cross_scale_shadow / eq:topdown_priors) is the
            # model-fiber transport of a GENUINELY EMERGED scale-(s+1) meta-agent, and is OUT OF SCOPE for
            # this single-scale transformer -- NOT a deferred gap. The manuscript treats single-scale r_i as
            # a PRIMITIVE boundary condition (PIFB lines 554, 636) and assigns the full transport + Ouroboros
            # tower to MAgent_Model/gauge_agent/ (PIFB line 2334). The frozen global r above IS the sanctioned
            # s_max boundary -- the named "held at its initial value rather than recomputed" special case of
            # the self-referential closure (PIFB line 2332). learnable_r is the same-scale empirical-Bayes
            # stand-in (a different axis: frozen-vs-learned, still token-uniform).

        # Unigram log-prior decode table (decode_unigram_prior=True): a non-trainable (V,) buffer
        # log pi_v holding the smoothed corpus unigram log-frequencies, added to EVERY decode
        # path's logits as kappa * log pi_v (the Bayes class prior; a DATA statistic set by
        # set_unigram_log_prior, not a learned parameter). Created only on the toggled path
        # (matching additive_R / the s tables) so the default state_dict is byte-identical.
        # Init zeros = the current implicit uniform prior; decode warns once per process while
        # the table is still unset.
        if decode_unigram_prior:
            self.register_buffer("unigram_log_prior", torch.zeros(vocab_size))
            self._unigram_set = False                                   # flips on set_unigram_log_prior
        # Untied decode bank (untie_decode_bank=True, use_prior_bank=True only): decode reads its
        # OWN (V, K) tables decode_mu_embed / decode_sigma_log_embed, cloned from the tables decode
        # would otherwise read (_prior_mu_table -- the s tables under prior_source='model_channel',
        # else the encode tables) so step 0 is byte-identical, then trained separately. Encode and
        # the alpha-KL self-coupling target keep the original tables. Cloning draws no RNG, so the
        # default path's table init is byte-unchanged.
        if self.untie_decode_bank:
            self.decode_mu_embed        = nn.Parameter(self._prior_mu_table().clone().detach())
            self.decode_sigma_log_embed = nn.Parameter(self._prior_sigma_log_table().clone().detach())

        # omega_direct: a per-token GL(K) group element table (identity init -> step-0 == trivial gauge).
        # Created ONLY on the omega_direct path so the default state_dict is byte-identical. Block-
        # diagonal by construction for block_glk (identity is diagonal; the group retraction keeps it so).
        #
        # omega_compact_storage (opt-in, default OFF): for an EQUAL-block group (untied block_glk /
        # tied tied_block_glk; irrep_dims = [d]*H, H>1) the full (V,K,K) table wastes ~H x (off-blocks
        # frozen zero). Store the H distinct blocks (V,H,d,d) untied, or the ONE shared block (V,d,d)
        # tied -- both matching phi_embed's V*n_gen param count exactly (V*H*d^2 / V*d^2 = V*n_gen).
        # encode carries these blocks in CompactBlockElement; inverse and transport stay blockwise,
        # while explicit compatibility callers may request a dense element. Compaction changes the
        # table SHAPE (would break a Phase-1 (V,K,K) checkpoint), so
        # the opt-in flag is the state-dict safety: default OFF keeps the shipped (V,K,K) path
        # byte-identical. Single-block groups (glk/so_k/sp) and the irrep towers (so_n/sp_n) keep
        # (V,K,K) this phase (nothing to compact / element-vs-coordinate tension deferred).
        self._omega_compact = False
        self._omega_tied    = bool(gauge_group_is_tied)
        self.reflection_scope = "full_element"
        if (gauge_group_name == "block_glk"
                and irrep_dims is not None
                and len(irrep_dims) > 1
                and (omega_reflection != "off" or phi_reflection != "off")):
            # Multi-block block_glk is a product GL(d)^H with 2^H orientation sectors. The existing
            # reflection proposal is diag(-1,1,...) at K scale, so it probes block 0 only; keep the
            # proposal unchanged and label its intentionally limited scope instead of implying all
            # sectors. One-head and cross-coupled block_glk report irrep_dims=[K] and therefore use
            # the complete represented element rather than this product-group label.
            self.reflection_scope = "block_0_probe"
            warnings.warn(
                "block-GL reflection is a block-0 probe: the existing proposal flips only the first "
                "GL(d) block and does not explore all 2^H product-group orientation sectors.",
                UserWarning,
                stacklevel=2,
            )
        if gauge_parameterization == "omega_direct":
            dims = irrep_dims
            compact = omega_compact_storage
            self._omega_compact = compact
            if compact:
                H, d = len(dims), dims[0]
                eye_d = torch.eye(d)
                if gauge_group_is_tied:                       # (V,d,d): one block shared across H heads
                    self.omega_embed = nn.Parameter(eye_d.expand(vocab_size, d, d).clone())
                else:                                         # (V,H,d,d): H independent blocks
                    self.omega_embed = nn.Parameter(eye_d.expand(vocab_size, H, d, d).clone())
                    if omega_reflection == "init_seed":
                        # reflection_element(K) = diag(-1,1,...,1) is block 0 = reflection_element(d),
                        # blocks 1..H-1 = I_d, so seeding block 0 of every OTHER token assembles to the
                        # identical det<0 element the full (V,K,K) path seeds. (tied rejects init_seed at
                        # config, so the (V,d,d) branch needs no seed.)
                        from vfe3.geometry.generators import reflection_element
                        Rd = reflection_element(d)
                        with torch.no_grad():
                            self.omega_embed[1::2, 0] = Rd
            else:
                eye_K = torch.eye(K)
                self.omega_embed = nn.Parameter(eye_K.expand(vocab_size, K, K).clone())
                if omega_reflection == "init_seed":
                    from vfe3.geometry.generators import reflection_element
                    R = reflection_element(K)
                    with torch.no_grad():                    # seed every OTHER token into the det<0 sheet
                        self.omega_embed[1::2] = R

        # phi_reflection: a per-token discrete reflection sign R_i (det<0 iff sign==-1), prepended to
        # exp(phi_i) as g_i = R_i exp(phi_i) (see docs/superpowers/specs/2026-07-08-phi-reflection-
        # design.md). A register_buffer, NOT nn.Parameter: discrete state flipped by the Metropolis
        # move, not gradient. Created ONLY on the phi path when phi_reflection != 'off' so the default
        # state_dict is byte-identical. Default all +1 (identity, det>0); 'init_seed' seeds every OTHER
        # token to -1, mirroring omega_embed's [1::2] init_seed above.
        if gauge_parameterization == "phi" and phi_reflection != "off":
            self.register_buffer("reflection_sign", torch.ones(vocab_size))
            if phi_reflection == "init_seed":
                with torch.no_grad():
                    self.reflection_sign[1::2] = -1.0

    def encode(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> BeliefState:
        r"""Look up the per-token Gaussian prior as the initial belief (q = p)."""
        belief = get_encode(self.encode_mode)(self, token_ids)
        if hasattr(self, "reflection_sign"):
            belief = belief._replace(reflection=self.reflection_sign[token_ids])
        return belief

    def _omega_lookup(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> 'torch.Tensor | CompactBlockElement':
        r"""Look up the per-token gauge frame U_i without changing its storage representation.

        Non-compact (default): a plain (V, K, K) table lookup. Compact
        (``_omega_compact``): return ``CompactBlockElement`` around the live looked-up
        (B, N, H, d, d) / tied (B, N, d, d) blocks. Inverse and transport contractions consume
        those blocks directly. Dense K x K reconstruction is available only through the container's
        explicit compatibility method ``to_dense()``.
        """
        g = self.omega_embed[token_ids]                                      # (B,N,K,K) or (B,N,H,d,d)/(B,N,d,d)
        if not self._omega_compact:
            return g
        return CompactBlockElement(g, self.K, tied=self._omega_tied)

    def encode_s(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Look up the per-token model-channel belief s_i = N(s_mu, s_sigma).

        Returns (s_mu, s_sigma) with s_mu (B, N, K); the covariance rank FOLLOWS the family
        (``family_cov_kind``): a diagonal/Laplace family yields the positive variances
        exp(s_sigma_log).clamp(min=eps) as (B, N, K), while ``gaussian_full`` assembles the packed
        strict-lower Cholesky into the full SPD covariance L L^T as (B, N, K, K). Available on the
        active-model-channel path (lambda_h>0, lambda_gamma>0, prior_source='model_channel', or
        s_e_step, where the s tables are created); consumed as ``get_family(cfg.family)(s_mu,
        s_sigma)`` by the hyper-prior term lambda_h*KL(s_i||r). The s->q coupling is a separate
        path: ``prior_source='model_channel'`` routes the belief prior to these same s tables,
        including the packed full covariance when the configured family is ``gaussian_full``.
        """
        s_mu = self.s_mu_embed[token_ids]                                       # (B, N, K)
        if self._s_cov_kind == "full":
            s_sigma = covariance_from_packed(
                self.s_sigma_log_embed[token_ids], self.s_sigma_lower_embed[token_ids], eps=self.eps,
            )                                                                     # (B, N, K, K)
        else:
            s_sigma = bounded_variance_from_log(
                self.s_sigma_log_embed[token_ids], eps=self.eps,
            )                                                                     # (B, N, K)
        return s_mu, s_sigma

    def r_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The global hyper-prior centroid r = N(r_mu, r_sigma) with covariance rank FOLLOWING the
        family (``family_cov_kind``).

        Returns (r_mu, r_sigma) with r_mu (K,); a diagonal/Laplace family yields the positive
        variances exp(r_sigma_log).clamp(min=eps) as (K,), while ``gaussian_full`` assembles the
        packed strict-lower Cholesky into the full SPD covariance L L^T as (K, K). Available on the
        centroid path (lambda_h>0 or s_e_step, where the r tables are created); consumed as
        ``get_family(cfg.family)(r_mu, r_sigma)`` by the hyper-prior term (replacing the direct
        log-variance reads so a full family carries its off-diagonal centroid covariance).
        """
        r_mu = self.r_mu                                                        # (K,)
        if self._s_cov_kind == "full":
            r_sigma = covariance_from_packed(
                self.r_sigma_log, self.r_sigma_lower, eps=self.eps,
            )                                                                     # (K, K)
        else:
            r_sigma = bounded_variance_from_log(self.r_sigma_log, eps=self.eps)  # (K,)
        return r_mu, r_sigma

    def s_phi(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> torch.Tensor:                   # (B, N, n_gen) model-channel frame coordinates
        r"""Look up the independently stored model-channel frame coordinates."""
        return self.s_phi_embed[token_ids]

    @torch.no_grad()
    def barycenter_r_(self) -> None:
        r"""Closed-form forward-KL barycenter M-step for the hyper-prior centroid r (IN PLACE).

        Sets r to the moment-matched centroid (m-projection) of the model-channel s tables:
        ``r_mu = mean_v s_mu_v`` and ``r_sigma = mean_v[s_sigma_v + (s_mu_v - r_mu)^2]`` (within-table
        variance plus the spread of the means) -- the unique minimizer of ``sum_v KL(s_v || r)`` for
        diagonal Gaussians (Amari-Nagaoka m-projection = moment matching; the diagonal unit-weight
        specialization of the manuscript meta-agent barycenter). Computed over the FULL vocab s tables
        (the population centroid, batch-independent), under no_grad: in r_update_mode='barycenter' r is
        NOT an optimizer leaf (requires_grad=False), so it carries no gradient and is set here once per
        M-step (driven from train_step).

        POPULATION (audit 2026-06-13): this is the exact argmin of the UNIFORM-over-vocab objective
        ``sum_v KL(s_v||r)`` -- one equal-weight row per vocab type. The scored hyper-prior term
        (``_hyper_prior_term``) reduces with mean() over (B,N) token OCCURRENCES, i.e. the
        frequency-weighted ``sum_v f_v KL(s_v||r)``; the uniform centroid equals that argmin only for a
        uniform token distribution, so for a Zipfian vocab the two differ. Treat this as the
        empirical-Bayes prior-over-TYPES centroid, NOT the argmin of the frequency-weighted scored loss.
        It is also the UNCLAMPED moment-match, whereas the scored KL runs through kl_max (so the two
        targets diverge for far-drifted rows). Under s_e_step=True r additionally couples to the CE
        through _refine_s, so it is only a consistent population target there -- prefer
        r_update_mode='gradient' for the scored s_e_step=False exactness and the s_e_step coupled regime.

        DIVERGENCE (audit 2026-06-14): this closed form is the ALPHA=1 (KL) m-projection and reads NO
        cfg, so it is the exact M-step only for the canonical KL objective (renyi_order=1,
        divergence_family='renyi', lambda_h_mode='constant'). The scored gradient path descends
        D_alpha(s||r) at cfg.renyi_order / cfg.divergence_family with the lambda_h_mode envelope, so
        under any non-canonical setting the 'barycenter' and 'gradient' r-updates do NOT share a fixed
        point (VFE3Config.__post_init__ warns). It also drops the model-fiber transport Omega_tilde and
        the per-type weights of the manuscript meta-agent barycenter, so it is a same-scale,
        UNTRANSPORTED, uniform-weight centroid -- not the cross-scale shadow r_i=Omega_tilde[s^(s+1)].

        FAMILY (PB-11): for ``gaussian_full`` the moment match runs over FULL covariances --
        ``r_Sigma = mean_v[Sigma_s_v + (s_mu_v - r_mu)(s_mu_v - r_mu)^T]`` (within-covariance plus
        the outer product of the mean spread), the full-Gaussian m-projection -- and is written back
        through the packed Cholesky (r_sigma_log + r_sigma_lower). The diagonal branch is unchanged.
        """
        if self._s_cov_kind == "full":
            s_mu = self.s_mu_embed                                               # (V, K)
            s_sigma = covariance_from_packed(
                self.s_sigma_log_embed, self.s_sigma_lower_embed, eps=self.eps,
            )                                                                     # (V, K, K)
            r_mu = s_mu.mean(dim=0)                                              # (K,)
            centered = s_mu - r_mu                                               # (V, K)
            outer = centered.unsqueeze(-1) * centered.unsqueeze(-2)              # (V, K, K)
            r_sigma = (s_sigma + outer).mean(dim=0)                             # (K, K) within + between
            r_log_diag, r_packed = packed_from_covariance(r_sigma, eps=self.eps)
            self.r_mu.copy_(r_mu)
            self.r_sigma_log.copy_(r_log_diag)
            self.r_sigma_lower.copy_(r_packed)
            return
        s_mu = self.s_mu_embed                                                   # (V, K)
        s_sigma = bounded_variance_from_log(self.s_sigma_log_embed, eps=self.eps)  # (V, K)
        r_mu = s_mu.mean(dim=0)                                                  # (K,)
        r_var = (s_sigma + (s_mu - r_mu) ** 2).mean(dim=0)                       # (K,) within + between
        self.r_mu.copy_(r_mu)
        self.r_sigma_log.copy_(torch.log(r_var.clamp(min=self.eps)))

    def _prior_mu_table(self) -> torch.Tensor:
        r"""The (V, K) mean prior table feeding p_i: the model-channel s tables when
        prior_source=='model_channel' (s->q REPLACE: p_i = s_i), else the belief table mu_embed
        (default). Routed through ONE accessor so encode (q_i(0)=p_i), the E-step self-coupling
        target alpha*KL(q_i||p_i), and the decode per-vocab readout -KL(q||p_v) all consume the SAME
        prior, keeping p_i = s_i consistent. On the default 'token' path this returns self.mu_embed
        (the identical tensor), so the pre-toggle path is byte-identical.
        """
        return self.s_mu_embed if self.prior_source == "model_channel" else self.mu_embed

    def _prior_sigma_log_table(self) -> torch.Tensor:
        r"""The (V, K) log-variance prior table feeding p_i; the model-channel sibling of
        _prior_mu_table (see there). 'token' -> self.sigma_log_embed (byte-identical)."""
        return self.s_sigma_log_embed if self.prior_source == "model_channel" else self.sigma_log_embed

    def _decode_mu_table(self) -> torch.Tensor:
        r"""The (V, K) mean table the DECODE boundary scores against: the untied decode table
        decode_mu_embed when untie_decode_bank created it, else the shared prior table
        (_prior_mu_table). Encode and the E-step self-coupling target always read the prior
        table, so the untie toggle splits ONLY the decode readout. On the default (tied) path
        this returns the identical tensor _prior_mu_table does -- byte-identical.
        """
        return self.decode_mu_embed if self.untie_decode_bank else self._prior_mu_table()

    def _decode_sigma_log_table(self) -> torch.Tensor:
        r"""The (V, K) log-variance decode table; the sigma sibling of _decode_mu_table (see there)."""
        return self.decode_sigma_log_embed if self.untie_decode_bank else self._prior_sigma_log_table()

    @torch.no_grad()
    def set_unigram_log_prior(
        self,
        counts: torch.Tensor,            # (V,) corpus unigram COUNTS (integer or float, >= 0)
    ) -> None:
        r"""Fill the unigram decode table with add-one-smoothed log-frequencies (IN PLACE).

            log pi_v = log((counts_v + 1) / (sum_v counts_v + V)),
        the Laplace (add-one) smoothed unigram log-prior: every token gets one pseudo-count, so
        zero-count tokens carry a finite log pi_v = -log(total + V) instead of -inf, and
        sum_v pi_v = 1 exactly. Requires construction with decode_unigram_prior=True (the buffer
        exists only on the toggled path).
        """
        if not self.decode_unigram_prior:
            raise RuntimeError(
                "set_unigram_log_prior requires decode_unigram_prior=True at construction "
                "(the unigram_log_prior buffer exists only on the toggled path)."
            )
        if counts.shape != (self.vocab_size,):
            raise ValueError(
                f"counts must have shape ({self.vocab_size},), got {tuple(counts.shape)}"
            )
        counts_f = counts.to(dtype=self.unigram_log_prior.dtype,
                             device=self.unigram_log_prior.device)          # (V,)
        self.unigram_log_prior.copy_(
            torch.log((counts_f + 1.0) / (counts_f.sum() + float(self.vocab_size)))
        )
        self._unigram_set = True

    def _unigram_bias(self) -> torch.Tensor:
        r"""The (V,) additive decode bias kappa * log pi_v (decode_unigram_prior=True only).

        Warns ONCE PER PROCESS while the table is still all-zero (never set): the decode then
        degenerates to the pre-toggle uniform prior (kappa * 0 = 0, a value no-op). A table
        restored nonzero through load_state_dict counts as set.
        """
        global _WARNED_UNIGRAM_UNSET
        if not self._unigram_set:
            if bool((self.unigram_log_prior != 0.0).any()):
                self._unigram_set = True                                 # restored via state_dict
            elif not _WARNED_UNIGRAM_UNSET:
                _WARNED_UNIGRAM_UNSET = True
                warnings.warn(
                    "decode_unigram_prior=True but the unigram_log_prior table is unset "
                    "(all-zero): the decode degenerates to the uniform prior. Call "
                    "PriorBank.set_unigram_log_prior(counts) with the (V,) corpus counts.",
                    UserWarning, stacklevel=3,
                )
        return self.unigram_kappa * self.unigram_log_prior

    def _tau_eff(
        self,
        tau: Optional[float] = None,     # override decode_tau; None -> self.decode_tau
    ) -> torch.Tensor:
        r"""Effective decode temperature tau_eff = tau * exp(-clamp(decode_log_scale, -3, 3))."""
        base_tau = self.decode_tau if tau is None else tau
        return base_tau * torch.exp(-self.decode_log_scale.clamp(-3.0, 3.0))

    def decode(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances

        *,
        tau:     Optional[float] = None,  # override decode_tau; None -> self.decode_tau
    ) -> torch.Tensor:                   # (B, N, V) logits
        r"""Decode logits via the selected kernel; ``use_prior_bank`` is the single gate.

        True (the opt-in pure path): the KL-to-prior readout -KL(q_i || pi_v)/tau_eff with the
        covariance structure given by ``decode_mode`` (diagonal | full). False (ablation): the
        ``linear`` kernel logits = mu_q @ W^T (sigma_q and tau_eff ignored). Routing here -- not
        through a second config value -- keeps ``decode_mode`` and ``use_prior_bank`` from ever
        silently disagreeing (the linear path simply does not consult ``decode_mode``).

        Under ``decode_unigram_prior=True`` the unigram log-prior bias kappa * log pi_v is added
        HERE, after the registered kernel, so every decode mode (linear included) gets it from
        one seam; toggle off adds nothing (byte-identical)."""
        mode = self.decode_mode if self.use_prior_bank else "linear"
        logits = get_decode(mode)(self, mu_q, sigma_q, self._tau_eff(tau))
        if self.decode_unigram_prior:
            logits = logits + self._unigram_bias()                       # (B, N, V) + (V,)
        return logits

    def reference_decode(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances

        *,
        tau:     Optional[float] = None,  # override decode_tau; None -> self.decode_tau
    ) -> torch.Tensor:                   # (B, N, V) logits = -D_configured(q || pi_v)/tau_eff
        r"""Authoritative reference decode: -D_configured(q_i || pi_v)/tau_eff via the seam.

        Dispatches through the CONFIGURED family (``self.family``) and divergence functional
        (``self.divergence_family`` at ``self.renyi_order``), broadcasting the seam over the
        vocabulary V in one shot (general but slow, O(B*N*V*K)). This is the same computation the
        registered ``family`` kernel performs, so it stays the oracle for the fast canonical kernels:
        for a canonical gaussian + renyi + alpha=1 config it equals the fused ``diagonal``/``full``
        kernels exactly (and under log-softmax); for a non-Gaussian family or a noncanonical
        divergence it reads the belief out under the SAME geometry the E-step minimized.

        The seam is invoked with ``kl_max=inf``: a DECODE must preserve the full divergence ranking
        over the vocabulary, so the saturation policy (default ``kl_max=100``, which flattens every
        distant prior to a single -100 logit and destroys the argmax) is disabled here. The full q is
        scored against the intentionally DIAGONAL vocabulary-prior table (promoted with diag_embed
        only for a full family). (``nan_to_num`` inside ``safe_kl_clamp`` still maps NaN/+inf from
        degenerate pairs to +inf -> -inf logits.)
        """
        tau_eff = self._tau_eff(tau)
        logits = _decode_family(self, mu_q, sigma_q, tau_eff)           # configured family/divergence
        if self.decode_unigram_prior:
            logits = logits + self._unigram_bias()                       # same seam as decode()
        return logits

    def _validate_fused_ce_targets(
        self,
        targets: torch.Tensor,           # (B, N) next-token ids

        *,
        ignore_index: int = -100,
    ) -> None:
        """Reject nonignored targets outside the vocabulary before fused CE reduction."""
        counted = targets != ignore_index
        invalid = counted & ((targets < 0) | (targets >= self.vocab_size))
        if bool(invalid.any()):
            invalid_target = int(targets[invalid][0].item())
            raise IndexError(f"Target {invalid_target} is out of bounds.")

    def decode_ce_diagonal_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        z_loss_weight: float           = 0.0,   # z-loss coefficient on mean(logsumexp^2); 0.0 = OFF
        tau:           Optional[float] = None,   # override decode_tau; None -> self.decode_tau
        chunk_size:    Optional[int]   = None,   # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int             = -100,
    ) -> torch.Tensor:                   # () scalar mean cross-entropy
        r"""Fused chunked-vocab cross-entropy: the ``diagonal`` decode CE WITHOUT a (B, N, V) tensor.

        Iterates the vocabulary in chunks ``[v0, v1)``, computing each chunk's logits with the SAME
        closed form (and the SAME global centering offset ``c = mean_v(mu_v)``) as ``_decode_diagonal``,
        reducing each chunk to its per-position ``logsumexp`` and gathering the target-token logit, so
        the full ``(B, N, V)`` logit tensor is never materialized. Per position the cross-entropy is
        ``logsumexp_v(logit_v) - logit_target`` (= -log-softmax at the target); the loss is the mean
        over non-ignored positions, exactly matching ``F.cross_entropy(decode(...), targets, ignore_index)``.

        The offset ``c`` is a per-coordinate ``(1, K)`` mean over ALL V, computed in one ``O(V*K)``
        pass with no big tensor, so it is IDENTICAL to the full path (the closed form is
        offset-invariant: ``(mu_q - c) - (mu_v - c) == mu_q - mu_v``). The V-axis reduction (the
        chunk ``logsumexp`` and the target gather) happens INSIDE a gradient-checkpointed function
        that returns only the two ``(B, N)`` per-chunk summaries, so the ``(B, N, Vc)`` chunk logit
        is born and dies inside the checkpoint -- it is recomputed in backward and never crosses the
        boundary (without this the downstream ``logsumexp``/``exp``/``gather`` would save it and the
        peak would stay ``(B, N, V)``). Recompute is deterministic (no RNG here), so value and
        gradient match the full path exactly.

        ``decode_unigram_prior=True`` adds the chunk slice of kappa * log pi_v to each chunk's
        logits BEFORE its logsumexp/gather, so the streamed CE equals the dense CE over the
        shifted logits. ``z_loss_weight > 0`` adds z_loss_weight * mean_i(logsumexp_v logit)^2
        (the streamed total logsumexp, already computed for the CE) -- the guard keeps 0.0
        byte-identical to the pre-kwarg path.
        """
        self._validate_fused_ce_targets(targets, ignore_index=ignore_index)
        tau_eff = self._tau_eff(tau)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size

        sigma_v_all = bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps,
        )                                                                             # (V, K)
        mu_v_all = self._decode_mu_table()                                  # (V, K) decode table (untied if set)
        c = mu_v_all.mean(dim=0, keepdim=True)                              # (1, K) global v-independent shift
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        mc_q = mu_q - c                                                     # (B, N, K) centered query means
        lhs = torch.cat([sigma_q + mc_q ** 2, -2.0 * mc_q], dim=-1)         # (B, N, 2K)
        # Per-position, v-INDEPENDENT term of -KL/tau_eff: it cancels in the CE difference
        # (logsumexp - target_logit) but is carried so each chunk's logits equal _decode_diagonal's.
        per_pos = self.K + torch.log(sigma_q.clamp(min=self.eps)).sum(-1, keepdim=True)  # (B, N, 1)

        def _chunk_summaries(lhs_:    torch.Tensor, per_pos_:        torch.Tensor,
                             mu_v_c:  torch.Tensor, inv_v_c:         torch.Tensor,
                             lsum_c:  torch.Tensor, in_chunk_f:      torch.Tensor,
                             local_idx: torch.Tensor,
                             u_c:     Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""Reduce one vocab chunk to (lse_chunk, target_contrib), both (B, N), on the inside.

            logit_{i,v} = -0.5(a_v - per_pos)/tau_eff over the chunk (see _decode_diagonal). The
            full (B, N, Vc) chunk logit lives only here so checkpointing frees it after forward.
            ``in_chunk_f`` is a 0/1 (B, N) mask selecting positions whose target falls in this chunk.
            """
            rhs = torch.cat([inv_v_c, mu_v_c * inv_v_c], dim=-1)            # (Vc, 2K), mu_v_c already centered
            a_v = lhs_ @ rhs.transpose(-1, -2)                             # (B, N, Vc)
            a_v = a_v + (mu_v_c ** 2 * inv_v_c).sum(-1) + lsum_c            # + sum_k(mc_v^2/sigma_v + log sigma_v)
            logit_chunk = -0.5 * (a_v - per_pos_) / tau_eff                # (B, N, Vc)
            if u_c is not None:
                logit_chunk = logit_chunk + u_c                            # unigram log-prior chunk slice
            lse_chunk = torch.logsumexp(logit_chunk, dim=-1)               # (B, N)
            gathered = logit_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
            return lse_chunk, gathered * in_chunk_f                        # zero where target not in chunk

        valid = targets != ignore_index                                    # (B, N) bool
        lse_chunks = []
        target_logit = torch.zeros(mu_q.shape[:-1], device=mu_q.device, dtype=mu_q.dtype)  # (B, N)

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            mc_v_c = (mu_v_all[v0:v1] - c)                                  # (Vc, K) centered prior means
            inv_v_c = 1.0 / sigma_v_all[v0:v1]                             # (Vc, K)
            lsum_c = torch.log(sigma_v_all[v0:v1]).sum(-1)                 # (Vc,)
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            # Target gather indices: positions whose target lands in [v0, v1). Ignored positions have
            # target < 0 < v0, so they never match -> target_logit stays 0 for them and `valid` excludes
            # them from the mean. local_idx is clamped to a safe range for the out-of-window rows.
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            if torch.is_grad_enabled() and lhs.requires_grad:
                lse_chunk, contrib = _checkpoint.checkpoint(
                    _chunk_summaries, lhs, per_pos, mc_v_c, inv_v_c, lsum_c, in_chunk_f, local_idx,
                    u_c, use_reentrant=False,
                )
            else:
                lse_chunk, contrib = _chunk_summaries(
                    lhs, per_pos, mc_v_c, inv_v_c, lsum_c, in_chunk_f, local_idx, u_c
                )
            lse_chunks.append(lse_chunk)
            target_logit = target_logit + contrib                          # exactly one chunk contributes per valid pos

        # Combine the per-chunk logsumexps into the full-V logsumexp. The stacked summaries are
        # (n_chunks, B, N) = B*N*ceil(V/chunk), negligible vs (B, N, V).
        logsumexp_v = torch.logsumexp(torch.stack(lse_chunks, dim=0), dim=0)  # (B, N)
        ce_per_pos = logsumexp_v - target_logit                           # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        # (Matches the full path, whose F.cross_entropy mean over zero counted tokens would be NaN.)
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        if z_loss_weight > 0.0:
            # z-loss: z_loss_weight * mean_i (log Z_i)^2 over the counted positions, log Z_i the
            # streamed full-V logsumexp above -- calibrates log Z ~ 0 so the decode approximates a
            # normalized observation model. The 0.0 guard keeps the default path byte-identical.
            ce = ce + z_loss_weight * (logsumexp_v ** 2 * valid).sum() / valid.sum().clamp_min(1)
        return ce

    def _full_cov_query_invariants(
        self,
        sigma_q: torch.Tensor,           # (B, N, K, K) posterior covariances
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Per-position, v-INDEPENDENT pieces of the full-cov KL against a DIAGONAL prior.

        Scoring a full q = N(mu_q, Sigma_q) against a diagonal prior pi_v = N(mu_v, diag(sigma_v))
        is the gaussian_full KL with a DIAGONAL second covariance, which collapses every per-pair
        (K, K) Cholesky into matmuls over V PLUS one per-position log|Sigma_q|. This returns the two
        pieces that depend on q only (not on the vocabulary):
            diag_sq = diag(Sigma_q)                    (B, N, K)  -- the raw query variances
            logdet_q = log|Sigma_q|                    (B, N)
        with the same round-zero-first factorization as the gaussian_full closed form
        (families/gaussian.py renyi_closed_form), so the diagonal-prior closed form is value-equal
        to ``_decode_full`` (the per-pair Cholesky seam) without ever forming a (B, N, V, K, K)
        workspace. ``safe_cholesky`` (jittered, never raises) yields a finite log-det where its
        ``ok`` mask is True; a position where every jitter round fails (non-PD Sigma_q) gets
        logdet_q = -inf, so per_pos = K + logdet_q drives every vocab logit to -inf there --
        matching the dense ``_decode_full`` path (gaussian_full's ok-gating -> NaN -> kl_max=inf
        -> -inf logit). The SPD retraction keeps Sigma_q PD in training, so ok is all-True and
        the -inf branch never engages on the pure path.
        """
        diag_sq = torch.diagonal(sigma_q, dim1=-2, dim2=-1)                # (B, N, K) = diag(Sigma_q)
        L, ok = safe_cholesky(sigma_q, eps=self.eps, rounds=5)
        logdet_q = _logdet_chol(L)                                         # (B, N)
        logdet_q = torch.where(ok, logdet_q, logdet_q.new_full((), float("-inf")))
        return diag_sq, logdet_q

    def decode_ce_full_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K, K) posterior covariances
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        z_loss_weight: float           = 0.0,   # z-loss coefficient on mean(logsumexp^2); 0.0 = OFF
        tau:           Optional[float] = None,   # override decode_tau; None -> self.decode_tau
        chunk_size:    Optional[int]   = None,   # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int             = -100,
    ) -> torch.Tensor:                   # () scalar mean cross-entropy
        r"""Fused chunked-vocab cross-entropy for the FULL-covariance KL decode WITHOUT the dense
        (B, N, V) logits OR the (B, N, V, K, K) per-pair Cholesky workspace ``_decode_full`` builds.

        The prior table is DIAGONAL (sigma_log_embed), so KL(q_full || pi_v_diag) needs no per-pair
        (K, K) Cholesky: the v-dependent trace and Mahalanobis terms are matmuls over V and the only
        (K, K) work is ONE log|Sigma_q| per position (``_full_cov_query_invariants``). This is the
        full-cov twin of ``decode_ce_diagonal_chunked`` -- same streaming logsumexp + target gather
        inside a gradient checkpoint, same global centering offset c = mean_v(mu_v) for fp32
        stability -- with the diagonal query variance replaced by diag(Sigma_q), the prior kept at
        its existing variance floor, and the per-position v-independent term K + sum_k log sigma_q
        replaced by K + log|Sigma_q|. Value-equal to F.cross_entropy(_decode_full(...)) to the
        decode's atol-1e-3 (tests/test_fullcov_alpha_roadmap_2026_06_13.py). The unigram-prior
        chunk-slice add and the z_loss_weight term follow ``decode_ce_diagonal_chunked`` exactly
        (see there); both default OFF / byte-identical.
        """
        self._validate_fused_ce_targets(targets, ignore_index=ignore_index)
        tau_eff = self._tau_eff(tau)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size

        sigma_v_all = bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps,
        )                                                                             # (V, K)
        mu_v_all = self._decode_mu_table()                                  # (V, K) decode table (untied if set)
        c = mu_v_all.mean(dim=0, keepdim=True)                              # (1, K) global v-independent shift
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        diag_sq, logdet_q = self._full_cov_query_invariants(sigma_q)       # (B,N,K), (B,N)
        mc_q = mu_q - c                                                     # (B, N, K) centered query means
        lhs = torch.cat([diag_sq + mc_q ** 2, -2.0 * mc_q], dim=-1)         # (B, N, 2K)
        # v-INDEPENDENT term of -KL/tau_eff (cancels in the CE difference, carried so each chunk's
        # logits equal _decode_full's): K + log|Sigma_q| (the full-cov analogue of K + sum_k log sigma_q).
        per_pos = self.K + logdet_q.unsqueeze(-1)                          # (B, N, 1)

        def _chunk_summaries(lhs_:    torch.Tensor, per_pos_:        torch.Tensor,
                             mu_v_c:  torch.Tensor, inv_v_c:         torch.Tensor,
                             lsum_c:  torch.Tensor, in_chunk_f:      torch.Tensor,
                             local_idx: torch.Tensor,
                             u_c:     Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""Reduce one vocab chunk to (lse_chunk, target_contrib), both (B, N), on the inside.

            a_v = sum_k[(diag(Sigma_q) + (mc_q-mc_v)^2)/sigma_v] + sum_k log sigma_v
                = trace_term + mahalanobis + log|diag(sigma_v)|, the gaussian_full KL with a
            diagonal prior; logit = -0.5(a_v - per_pos)/tau_eff. The (B, N, Vc) chunk logit lives
            only here so checkpointing frees it after forward.
            """
            rhs = torch.cat([inv_v_c, mu_v_c * inv_v_c], dim=-1)            # (Vc, 2K), mu_v_c centered
            a_v = lhs_ @ rhs.transpose(-1, -2)                             # (B, N, Vc)
            a_v = a_v + (mu_v_c ** 2 * inv_v_c).sum(-1) + lsum_c            # + sum_k(mc_v^2/sigma_v + log sigma_v)
            logit_chunk = -0.5 * (a_v - per_pos_) / tau_eff                # (B, N, Vc)
            if u_c is not None:
                logit_chunk = logit_chunk + u_c                            # unigram log-prior chunk slice
            lse_chunk = torch.logsumexp(logit_chunk, dim=-1)               # (B, N)
            gathered = logit_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
            return lse_chunk, gathered * in_chunk_f                        # zero where target not in chunk

        valid = targets != ignore_index                                    # (B, N) bool
        lse_chunks = []
        target_logit = torch.zeros(mu_q.shape[:-1], device=mu_q.device, dtype=mu_q.dtype)  # (B, N)

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            mc_v_c = (mu_v_all[v0:v1] - c)                                  # (Vc, K) centered prior means
            inv_v_c = 1.0 / sigma_v_all[v0:v1]                             # (Vc, K) = 1/sigma_v
            lsum_c = torch.log(sigma_v_all[v0:v1]).sum(-1)                 # (Vc,) = sum_k log sigma_v
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            if torch.is_grad_enabled() and lhs.requires_grad:
                lse_chunk, contrib = _checkpoint.checkpoint(
                    _chunk_summaries, lhs, per_pos, mc_v_c, inv_v_c, lsum_c, in_chunk_f, local_idx,
                    u_c, use_reentrant=False,
                )
            else:
                lse_chunk, contrib = _chunk_summaries(
                    lhs, per_pos, mc_v_c, inv_v_c, lsum_c, in_chunk_f, local_idx, u_c
                )
            lse_chunks.append(lse_chunk)
            target_logit = target_logit + contrib                          # exactly one chunk contributes per valid pos

        logsumexp_v = torch.logsumexp(torch.stack(lse_chunks, dim=0), dim=0)  # (B, N)
        ce_per_pos = logsumexp_v - target_logit                            # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        if z_loss_weight > 0.0:
            # z-loss on the streamed log Z (see decode_ce_diagonal_chunked); 0.0 guard = byte-identical.
            ce = ce + z_loss_weight * (logsumexp_v ** 2 * valid).sum() / valid.sum().clamp_min(1)
        return ce

    def decode_ce_linear_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        z_loss_weight: float                 = 0.0,   # z-loss coefficient on mean(logsumexp^2); 0.0 = OFF
        chunk_size:    Optional[int]         = None,  # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int                   = -100,
    ) -> torch.Tensor:                   # () scalar mean cross-entropy
        r"""Fused chunked-vocab cross-entropy for the LINEAR decode (``use_prior_bank=False``).

        The ``_decode_linear`` CE -- ``logits = x @ W^T (+ b)`` -> ``F.cross_entropy`` -- WITHOUT
        the (B, N, V) logit tensor (plus cross_entropy's same-size log-softmax copy, both retained
        for backward on the dense path; the dominant decode VRAM at large B, vram audit 2026-06-10).
        Same streaming contract as ``decode_ce_diagonal_chunked``: each vocab chunk's logits are
        born and die inside a gradient-checkpointed reduction that returns only the (B, N) chunk
        logsumexp and target-logit summaries; recompute is deterministic, so value and gradient (to
        mu_q, W, and b) match the dense path exactly. The unigram-prior chunk-slice add and the
        z_loss_weight term follow ``decode_ce_diagonal_chunked`` (see there); both default OFF.
        """
        self._validate_fused_ce_targets(targets, ignore_index=ignore_index)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size
        W = self.output_proj_weight                                        # (V, K)
        bias = self.output_proj_bias                                       # (V,) or None
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        def _chunk_summaries(mu_:     torch.Tensor, w_c:       torch.Tensor,
                             in_chunk_f: torch.Tensor, local_idx: torch.Tensor,
                             b_c:     Optional[torch.Tensor],
                             u_c:     Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""Reduce one vocab chunk to (lse_chunk, target_contrib), both (B, N), on the inside."""
            logit_chunk = mu_ @ w_c.transpose(-1, -2)                      # (B, N, Vc)
            if b_c is not None:
                logit_chunk = logit_chunk + b_c                            # learned log-unigram prior
            if u_c is not None:
                logit_chunk = logit_chunk + u_c                            # fixed unigram log-prior slice
            lse_chunk = torch.logsumexp(logit_chunk, dim=-1)               # (B, N)
            gathered = logit_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
            return lse_chunk, gathered * in_chunk_f                        # zero where target not in chunk

        valid = targets != ignore_index                                    # (B, N) bool
        lse_chunks = []
        target_logit = torch.zeros(mu_q.shape[:-1], device=mu_q.device, dtype=mu_q.dtype)  # (B, N)

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            w_c = W[v0:v1]                                                 # (Vc, K)
            b_c = bias[v0:v1] if bias is not None else None                # (Vc,) or None
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            if torch.is_grad_enabled() and (mu_q.requires_grad or W.requires_grad):
                lse_chunk, contrib = _checkpoint.checkpoint(
                    _chunk_summaries, mu_q, w_c, in_chunk_f, local_idx, b_c, u_c,
                    use_reentrant=False,
                )
            else:
                lse_chunk, contrib = _chunk_summaries(mu_q, w_c, in_chunk_f, local_idx, b_c, u_c)
            lse_chunks.append(lse_chunk)
            target_logit = target_logit + contrib                          # exactly one chunk contributes per valid pos

        logsumexp_v = torch.logsumexp(torch.stack(lse_chunks, dim=0), dim=0)  # (B, N)
        ce_per_pos = logsumexp_v - target_logit                           # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        if z_loss_weight > 0.0:
            # z-loss on the streamed log Z (see decode_ce_diagonal_chunked); 0.0 guard = byte-identical.
            ce = ce + z_loss_weight * (logsumexp_v ** 2 * valid).sum() / valid.sum().clamp_min(1)
        return ce

    def decode_ce_expected_likelihood_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        z_loss_weight: float           = 0.0,   # z-loss coefficient on mean(logsumexp^2); 0.0 = OFF
        tau:           Optional[float] = None,   # override decode_tau; None -> self.decode_tau
        chunk_size:    Optional[int]   = None,   # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int             = -100,
    ) -> torch.Tensor:                   # () scalar mean cross-entropy
        r"""Fused chunked-vocab cross-entropy for the EXPECTED-LIKELIHOOD decode (diagonal only).

        The fused-CE twin of ``decode_mode='expected_likelihood_chunked'`` (see
        ``_decode_expected_likelihood_chunked`` for the scoring math): the same streaming contract
        as ``decode_ce_diagonal_chunked`` -- each chunk's (B, N, Vc) logits are born and die inside
        a gradient-checkpointed reduction returning only the (B, N) chunk logsumexp and target
        summaries, so the (B, N, V) tensor is never materialized. The couplings sigma_q + sigma_v
        block the diagonal kernel's single-matmul trick, so each chunk broadcasts a (B, N, Vc, K)
        workspace instead (bounded by the chunk width; freed by the checkpoint). The unigram-prior
        chunk-slice add and the z_loss_weight term follow ``decode_ce_diagonal_chunked`` (see
        there); both default OFF.
        """
        self._validate_fused_ce_targets(targets, ignore_index=ignore_index)
        tau_eff = self._tau_eff(tau)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size

        sigma_v_all = bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps,
        )                                                                             # (V, K)
        mu_v_all = self._decode_mu_table()                                  # (V, K) decode table (untied if set)
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        def _chunk_summaries(mu_q_:   torch.Tensor, sigma_q_:   torch.Tensor,
                             mu_v_c:  torch.Tensor, sigma_v_c:  torch.Tensor,
                             in_chunk_f: torch.Tensor, local_idx: torch.Tensor,
                             u_c:     Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""Reduce one vocab chunk to (lse_chunk, target_contrib), both (B, N), on the inside."""
            d = mu_q_.unsqueeze(-2) - mu_v_c                               # (B, N, Vc, K)
            s = sigma_q_.unsqueeze(-2) + sigma_v_c                         # (B, N, Vc, K) convolved variances
            logit_chunk = -0.5 * (d ** 2 / s + torch.log(s)).sum(-1) / tau_eff   # (B, N, Vc)
            if u_c is not None:
                logit_chunk = logit_chunk + u_c                            # unigram log-prior chunk slice
            lse_chunk = torch.logsumexp(logit_chunk, dim=-1)               # (B, N)
            gathered = logit_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
            return lse_chunk, gathered * in_chunk_f                        # zero where target not in chunk

        valid = targets != ignore_index                                    # (B, N) bool
        lse_chunks = []
        target_logit = torch.zeros(mu_q.shape[:-1], device=mu_q.device, dtype=mu_q.dtype)  # (B, N)

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            mu_v_c = mu_v_all[v0:v1]                                       # (Vc, K)
            sigma_v_c = sigma_v_all[v0:v1]                                 # (Vc, K)
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            if torch.is_grad_enabled() and (mu_q.requires_grad or mu_v_all.requires_grad):
                lse_chunk, contrib = _checkpoint.checkpoint(
                    _chunk_summaries, mu_q, sigma_q, mu_v_c, sigma_v_c, in_chunk_f, local_idx,
                    u_c, use_reentrant=False,
                )
            else:
                lse_chunk, contrib = _chunk_summaries(
                    mu_q, sigma_q, mu_v_c, sigma_v_c, in_chunk_f, local_idx, u_c
                )
            lse_chunks.append(lse_chunk)
            target_logit = target_logit + contrib                          # exactly one chunk contributes per valid pos

        logsumexp_v = torch.logsumexp(torch.stack(lse_chunks, dim=0), dim=0)  # (B, N)
        ce_per_pos = logsumexp_v - target_logit                           # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        if z_loss_weight > 0.0:
            # z-loss on the streamed log Z (see decode_ce_diagonal_chunked); 0.0 guard = byte-identical.
            ce = ce + z_loss_weight * (logsumexp_v ** 2 * valid).sum() / valid.sum().clamp_min(1)
        return ce

    def decode_ce_family_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) or (B, N, K, K) posterior (co)variances
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        z_loss_weight: float           = 0.0,   # z-loss coefficient on mean(logsumexp^2); 0.0 = OFF
        tau:           Optional[float] = None,   # override decode_tau; None -> self.decode_tau
        chunk_size:    Optional[int]   = None,   # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int             = -100,
    ) -> torch.Tensor:                   # () scalar mean cross-entropy
        r"""Fused chunked-vocab cross-entropy for the FAMILY-consistent decode (``decode_mode=
        'family_chunked'``) WITHOUT the dense (B, N, V) logits.

        The family-consistent twin of ``decode_ce_diagonal_chunked``: each vocab chunk streams
        through the SAME registered functional ``get_functional(self.divergence_family)`` at
        ``alpha=self.renyi_order`` (logits = -D_configured(q || pi_v)/tau_eff, ``kl_max=inf``) and the
        same fused log-sum-exp/gather reduction inside a gradient checkpoint, so the (B, N, V) tensor
        is never materialized. The vocabulary prior table is DIAGONAL; a FULL family promotes each
        chunk with ``diag_embed`` and materializes only a (B, N, Vc, K, K) functional workspace inside
        the checkpoint (never a full SPD vocabulary table). Value/gradient-equal to the dense
        ``family`` decode -> cross-entropy. The unigram-prior chunk-slice add and the z_loss_weight
        term follow ``decode_ce_diagonal_chunked`` (see there); both default OFF.
        """
        self._validate_fused_ce_targets(targets, ignore_index=ignore_index)
        tau_eff = self._tau_eff(tau)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size

        family_cls = get_family(self.family)
        is_full = family_cls.cov_kind == "full"
        functional = get_functional(self.divergence_family)

        sigma_v_all = bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps,
        )                                                                             # (V, K) diagonal prior
        mu_v_all = self._decode_mu_table()                                  # (V, K) decode table (untied if set)
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        q_mu = mu_q.unsqueeze(-2)                                            # (B, N, 1, K)
        q_sigma = sigma_q.unsqueeze(-3 if is_full else -2)                   # (B, N, 1, K[, K])

        def _chunk_summaries(q_mu_:   torch.Tensor, q_sigma_:   torch.Tensor,
                             mu_v_c:  torch.Tensor, sigma_v_c:  torch.Tensor,
                             in_chunk_f: torch.Tensor, local_idx: torch.Tensor,
                             u_c:     Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""Reduce one vocab chunk to (lse_chunk, target_contrib), both (B, N), on the inside.

            The functional workspace ((B, N, Vc) diagonal / (B, N, Vc, K, K) full) is born and dies
            here so checkpointing frees it after forward; recompute is deterministic (the functional
            has no RNG), so value and gradient match the dense family decode exactly.
            """
            q = family_cls(q_mu_, q_sigma_)
            p = family_cls(mu_v_c, sigma_v_c)
            energy = functional(q, p, alpha=self.renyi_order,
                                kl_max=float("inf"), eps=self.eps)         # (B, N, Vc)
            logit_chunk = -energy / tau_eff                                # (B, N, Vc)
            if u_c is not None:
                logit_chunk = logit_chunk + u_c                            # unigram log-prior chunk slice
            lse_chunk = torch.logsumexp(logit_chunk, dim=-1)               # (B, N)
            gathered = logit_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
            return lse_chunk, gathered * in_chunk_f                        # zero where target not in chunk

        valid = targets != ignore_index                                    # (B, N) bool
        lse_chunks = []
        target_logit = torch.zeros(mu_q.shape[:-1], device=mu_q.device, dtype=mu_q.dtype)  # (B, N)

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            mu_v_c = mu_v_all[v0:v1]                                       # (Vc, K)
            sigma_v_c = (torch.diag_embed(sigma_v_all[v0:v1]) if is_full
                         else sigma_v_all[v0:v1])                          # (Vc, K[, K]) diag-embedded if full
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            if torch.is_grad_enabled() and (mu_q.requires_grad or mu_v_all.requires_grad):
                lse_chunk, contrib = _checkpoint.checkpoint(
                    _chunk_summaries, q_mu, q_sigma, mu_v_c, sigma_v_c, in_chunk_f, local_idx,
                    u_c, use_reentrant=False,
                )
            else:
                lse_chunk, contrib = _chunk_summaries(
                    q_mu, q_sigma, mu_v_c, sigma_v_c, in_chunk_f, local_idx, u_c
                )
            lse_chunks.append(lse_chunk)
            target_logit = target_logit + contrib                          # exactly one chunk contributes per valid pos

        logsumexp_v = torch.logsumexp(torch.stack(lse_chunks, dim=0), dim=0)  # (B, N)
        ce_per_pos = logsumexp_v - target_logit                            # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        if z_loss_weight > 0.0:
            # z-loss on the streamed log Z (see decode_ce_diagonal_chunked); 0.0 guard = byte-identical.
            ce = ce + z_loss_weight * (logsumexp_v ** 2 * valid).sum() / valid.sum().clamp_min(1)
        return ce


EncodeCallable = Callable[[PriorBank, torch.Tensor], BeliefState]
DecodeCallable = Callable[
    [PriorBank, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


class GeometricFusedCECallable(Protocol):
    """Fused CE contract for covariance-aware geometric decoders."""

    def __call__(
        self,
        pb:            PriorBank,
        mu_q:          torch.Tensor,
        sigma_q:       torch.Tensor,
        targets:       torch.Tensor,

        *,
        z_loss_weight: float           = 0.0,
        tau:           Optional[float] = None,
        chunk_size:    Optional[int]   = None,
        ignore_index:  int             = -100,
    ) -> torch.Tensor:
        ...


class LinearFusedCECallable(Protocol):
    """Fused CE contract for the mean-only linear decoder."""

    def __call__(
        self,
        pb:            PriorBank,
        mu_q:          torch.Tensor,
        targets:       torch.Tensor,

        *,
        z_loss_weight: float         = 0.0,
        chunk_size:    Optional[int] = None,
        ignore_index:  int           = -100,
    ) -> torch.Tensor:
        ...


FusedCECallable = GeometricFusedCECallable | LinearFusedCECallable


def _encode_prior_sigma(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> torch.Tensor:
    """Look up the configured belief prior covariance without discarding model-channel rank."""
    log_diag = pb._prior_sigma_log_table()[token_ids]                    # (B, N, K)
    if pb.diagonal_covariance:
        return bounded_variance_from_log(log_diag, eps=pb.eps)
    if pb.prior_source == "model_channel":
        return covariance_from_packed(
            log_diag,
            pb.s_sigma_lower_embed[token_ids],
            eps=pb.eps,
        )                                                                # (B, N, K, K)
    return torch.diag_embed(bounded_variance_from_log(log_diag, eps=pb.eps))


@register_encode("per_token")
def _encode_per_token(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""Per-token table lookup: token_ids -> (mu_v, sigma_v, phi_v) as the belief q = p.

    Diagonal family: sigma is the (B, N, K) variance vector. A full token-table prior promotes its
    diagonal variances to (B, N, K, K). A full ``model_channel`` prior reconstructs the complete
    packed s covariance, so the s-to-p route preserves learned correlations. The mean and gauge
    tables are shared across families.
    """
    mu = pb._prior_mu_table()[token_ids]                                     # (B, N, K) prior (s if model_channel)
    sigma = _encode_prior_sigma(pb, token_ids)                               # (B,N,K) or (B,N,K,K)
    phi = pb.phi_embed[token_ids]                                            # (B, N, n_gen)
    omega = pb._omega_lookup(token_ids) if getattr(pb, "gauge_parameterization", "phi") == "omega_direct" else None
    return BeliefState(mu=mu, sigma=sigma, phi=phi, omega=omega)


@register_encode("per_token_additive")
def _encode_per_token_additive(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""Arm-2 control: the SAME learned (V, n_gen) phi table used NON-structurally.

    Each token's phi code is mapped by the FROZEN readout ``pb.additive_R`` (K, n_gen) to an additive
    mean shift ``mu += phi @ R^T``, and the returned phi is ZERO so the transport
    ``Omega = exp(phi.G) exp(-phi.G) = I`` (no gl(g) congruence). The learned parameter count is the
    gauge cell's (``V*n_gen`` in ``phi_embed``; ``R`` is a frozen buffer), so this isolates raw phi-table
    CAPACITY from the gl(g) generator STRUCTURE -- the capacity-vs-structure control for the blocks_K48
    REMAND (docs/2026-07-05-blocks-k48-followup-experiment-spec.md, Arm 2a). Deliberately NOT gauge
    equivariant; use with ``transport_mode='flat'`` and ``pos_phi='none'`` so no other channel transports.
    """
    mu = pb._prior_mu_table()[token_ids]                                     # (B, N, K) prior (s if model_channel)
    sigma = _encode_prior_sigma(pb, token_ids)                               # (B,N,K) or (B,N,K,K)
    phi_code = pb.phi_embed[token_ids]                                       # (B, N, n_gen) learned table
    mu = mu + phi_code @ pb.additive_R.t()                                   # (B, N, K) structure-free shift
    phi = torch.zeros_like(phi_code)                                         # Omega = I: no gl(g) transport
    return BeliefState(mu=mu, sigma=sigma, phi=phi)


@register_encode("gauge_fixed")
def _encode_gauge_fixed(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""NAMED STUB: gauge-fixed encode (gauge orbit from a shared base belief).

    Deferred: would realize every prior as a gauge transform of one shared base
    belief, so the vocabulary varies only along the gauge orbit. Not yet implemented.
    """
    raise NotImplementedError(
        "encode_mode='gauge_fixed' is a named stub (gauge orbit from a shared base); "
        "use 'per_token'."
    )


@register_decode("diagonal")
def _decode_diagonal(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Exact diagonal -KL/tau_eff in closed form via a single fused matmul.

        KL = 0.5[ sum_k(sigma_q/sigma_v + (mu_q-mu_v)^2/sigma_v) - K + sum_k log(sigma_v/sigma_q) ]
    The v-dependent part A_v expands the Mahalanobis/trace terms into one matmul:
        lhs = [sigma_q + mc_q^2, -2 mc_q]            (B, N, 2K)
        rhs = [1/sigma_v,        mc_v/sigma_v]       (V, 2K)
        A_v = lhs @ rhs^T + sum_k(mc_v^2/sigma_v + log sigma_v)
            == sum_k(sigma_q/sigma_v + (mc_q-mc_v)^2/sigma_v) + sum_k log sigma_v
            == 2 KL + K + sum_k log sigma_q.
    The per-position (-K - sum_k log sigma_q) is v-INDEPENDENT (drops under softmax) but
    is KEPT so logits == -KL/tau_eff EXACTLY.

    NUMERICS: the Mahalanobis term ``(mu_q - mu_v)^2`` is reconstructed by the matmul as
    ``mc_q^2 - 2 mc_q mc_v + mc_v^2``, a subtraction of large near-equal quantities that
    catastrophically cancels in float32 once the means carry a large common offset (the
    error grows like eps * mu^2 / sigma_v and breaks the atol-1e-3 seam pin at modest
    |mu| / tight sigma_v). We remove the common offset BEFORE the matmul by subtracting
    the v-independent shift ``c = mean_v(mu_v)`` (per dim) from both means; since
    ``(mu_q - c) - (mu_v - c) == mu_q - mu_v`` the closed form is unchanged exactly while
    the cancelled magnitude collapses to the residual spread of the means.
    """
    sigma_v = bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps)  # (V, K)
    mu_v = pb._decode_mu_table()                                        # (V, K) decode table (untied if set)
    inv_v = 1.0 / sigma_v                                               # (V, K) = 1/sigma_v

    c = mu_v.mean(dim=0, keepdim=True)                                  # (1, K) v-independent shift
    mc_v = mu_v - c                                                     # (V, K) centered prior means
    mc_q = mu_q - c                                                     # (B, N, K) centered query means

    lhs = torch.cat([sigma_q + mc_q ** 2, -2.0 * mc_q], dim=-1)          # (B, N, 2K)
    rhs = torch.cat([inv_v, mc_v * inv_v], dim=-1)                       # (V, 2K)
    a_v = lhs @ rhs.transpose(-1, -2)                                    # (B, N, V): sum_k[(sigma_q+mc_q^2-2 mc_q mc_v)/sigma_v]
    a_v = a_v + (mc_v ** 2 * inv_v).sum(-1) + torch.log(sigma_v).sum(-1)  # + sum_k(mc_v^2/sigma_v + log sigma_v)
    # a_v == sum_k(sigma_q/sigma_v + (mc_q-mc_v)^2/sigma_v) + sum_k log sigma_v
    #     == sum_k(sigma_q/sigma_v + (mu_q-mu_v)^2/sigma_v) + sum_k log sigma_v = 2 KL + K + sum_k log sigma_q
    per_pos = pb.K + torch.log(sigma_q.clamp(min=pb.eps)).sum(-1, keepdim=True)   # (B, N, 1) = K + sum_k log sigma_q
    kl_v = (0.5 * (a_v - per_pos)).clamp(min=0.0)                        # (B, N, V); KL>=0 floor matches
    return -kl_v / tau_eff                                               # reference_decode's safe_kl_clamp (r2 id17)


@register_decode(
    "diagonal_chunked",
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_diagonal_chunked,
)
def _decode_diagonal_chunked(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Inference (targets=None) decode for ``decode_mode='diagonal_chunked'``: full diagonal logits.

    The chunked mode's training memory win is the FUSED decode+CE in ``decode_ce_diagonal_chunked``
    (it never forms ``(B, N, V)``). When ``decode`` is called for logits (sampling / generation /
    inference), correctness is what matters, so this delegates to the exact ``diagonal`` kernel --
    the returned logits are byte-identical to ``decode_mode='diagonal'``.
    """
    return _decode_diagonal(pb, mu_q, sigma_q, tau_eff)


@register_decode("full", supports_full=True)
def _decode_full(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K, K) posterior covariances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Exact full-covariance decode logits_{i,v} = -KL(q_i || pi_v)/tau_eff via Cholesky.

    Scores the full-covariance posterior q_i = N(mu_q, Sigma_q) against every vocab prior
    pi_v through the ``gaussian_full`` divergence seam (Cholesky KL). The prior table is
    diagonal (sigma_log_embed), embedded as a diagonal full covariance diag(exp(sigma_log_v))
    so a full q is scored against it. As in ``reference_decode`` the seam is invoked with
    ``kl_max=inf`` so the full KL ranking over the vocabulary is preserved (decode must not
    saturate distant priors to a single logit). General but O(B*N*V*K^3) (per-pair Cholesky):
    the theoretically pure full-covariance path, not the fast diagonal kernel.
    """
    mu_v = pb._decode_mu_table()                                         # (V, K) decode table (untied if set)
    sigma_v = torch.diag_embed(
        bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps)
    )                                                                                       # (V, K, K) diagonal-as-full
    mu_q_b = mu_q.unsqueeze(-2)                                          # (B, N, 1, K)
    sigma_q_b = sigma_q.unsqueeze(-3)                                    # (B, N, 1, K, K)
    full = get_family("gaussian_full")
    kl_v = kl(
        full(mu_q_b, sigma_q_b),
        full(mu_v, sigma_v),
        kl_max=float("inf"),
        eps=pb.eps,
    )                                                                       # (B, N, V)
    return -kl_v / tau_eff


@register_decode(
    "full_chunked",
    supports_full=True,
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_full_chunked,
)
def _decode_full_chunked(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K, K) posterior covariances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Inference (targets=None) decode for ``decode_mode='full_chunked'``: full-cov KL logits via
    the DIAGONAL-prior closed form -- NO per-pair (K, K) Cholesky.

    The training memory win is the fused decode+CE in ``decode_ce_full_chunked`` (never forms
    (B, N, V)); for logits (sampling / generation) this materializes (B, N, V) -- inherent to
    producing every vocab logit -- but still avoids the (B, N, V, K, K) Cholesky/solve workspace
    that ``_decode_full`` builds, by exploiting the diagonal prior (see ``_full_cov_query_invariants``).
    Value-equal to ``decode_mode='full'`` to atol-1e-3 (tests/test_fullcov_alpha_roadmap_2026_06_13.py).
    """
    sigma_v = bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps)  # (V, K) diagonal decode variances
    mu_v = pb._decode_mu_table()                                         # (V, K) decode table (untied if set)
    inv_v = 1.0 / sigma_v                                                # (V, K) = 1/sigma_v

    diag_sq, logdet_q = pb._full_cov_query_invariants(sigma_q)           # (B,N,K), (B,N)
    c = mu_v.mean(dim=0, keepdim=True)                                   # (1, K) v-independent shift
    mc_v = mu_v - c                                                      # (V, K) centered prior means
    mc_q = mu_q - c                                                      # (B, N, K) centered query means

    lhs = torch.cat([diag_sq + mc_q ** 2, -2.0 * mc_q], dim=-1)          # (B, N, 2K)
    rhs = torch.cat([inv_v, mc_v * inv_v], dim=-1)                       # (V, 2K)
    a_v = lhs @ rhs.transpose(-1, -2)                                    # (B, N, V): trace + mahalanobis core
    a_v = a_v + (mc_v ** 2 * inv_v).sum(-1) + torch.log(sigma_v).sum(-1)  # + sum_k(mc_v^2/sigma_v + log sigma_v)
    per_pos = pb.K + logdet_q.unsqueeze(-1)                              # (B, N, 1) = K + log|Sigma_q|
    kl_v = (0.5 * (a_v - per_pos)).clamp(min=0.0)                        # (B, N, V); KL>=0 floor matches
    return -kl_v / tau_eff                                               # _decode_diagonal (audit 2026-07-05 m5)


@register_decode(
    "family",
    covariance_kinds=frozenset({"diagonal", "full"}),
    family_consistent=True,
)
def _decode_family(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) or (B, N, K, K) posterior (co)variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -D_configured(q || pi_v)/tau_eff
    r"""Family/divergence-consistent decode (PB-14): logits = -D_configured(q_i || pi_v)/tau_eff.

    Scores the posterior q_i against every vocabulary prior pi_v through the CONFIGURED belief family
    ``pb.family`` and divergence functional ``pb.divergence_family`` at ``alpha=pb.renyi_order``, so
    the readout matches the E-step geometry rather than a hardcoded gaussian alpha=1 KL. As in the
    other decode kernels the seam is invoked with ``kl_max=inf`` (a DECODE must preserve the full
    divergence ranking over the vocabulary). The vocabulary prior table is intentionally DIAGONAL in
    every family (PB-11); a full family promotes it with ``diag_embed`` so a full q is scored against
    a diagonal-as-full prior. Broadcasting the functional over V materializes a (B, N, V) energy
    (a full family a (B, N, V, K, K) workspace): general but O(B*N*V*...); the training memory win is
    the fused CE twin ``decode_ce_family_chunked``. For a canonical gaussian + renyi + alpha=1 config
    this equals the fast ``diagonal``/``full`` kernels (and ``reference_decode`` is pinned to it)."""
    family_cls = get_family(pb.family)
    q_sigma = sigma_q.unsqueeze(-3 if family_cls.cov_kind == "full" else -2)
    q = family_cls(mu_q.unsqueeze(-2), q_sigma)
    p_sigma_diag = bounded_variance_from_log(
        pb._decode_sigma_log_table(), eps=pb.eps
    )                                                                       # (V, K) diagonal prior variances
    p_sigma = (
        torch.diag_embed(p_sigma_diag)
        if family_cls.cov_kind == "full"
        else p_sigma_diag
    )                                                                       # (V, K) or (V, K, K)
    p = family_cls(pb._decode_mu_table(), p_sigma)
    functional = get_functional(pb.divergence_family)
    energy = functional(q, p, alpha=pb.renyi_order,
                        kl_max=float("inf"), eps=pb.eps)                    # (B, N, V)
    return -energy / tau_eff


@register_decode(
    "family_chunked",
    covariance_kinds=frozenset({"diagonal", "full"}),
    family_consistent=True,
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_family_chunked,
)
def _decode_family_chunked(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) or (B, N, K, K) posterior (co)variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -D_configured(q || pi_v)/tau_eff
    r"""Inference (targets=None) decode for ``decode_mode='family_chunked'``: full family logits.

    The chunked mode's training memory win is the FUSED decode+CE in ``decode_ce_family_chunked``
    (it never forms (B, N, V)). When ``decode`` is called for logits (sampling / generation /
    inference), correctness is what matters, so this delegates to the exact ``family`` kernel --
    the returned logits are byte-identical to ``decode_mode='family'``.
    """
    return _decode_family(pb, mu_q, sigma_q, tau_eff)


@register_decode(
    "expected_likelihood_chunked",
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_expected_likelihood_chunked,
)
def _decode_expected_likelihood_chunked(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) expected-likelihood logits
    r"""Expected-likelihood decode: logits from the exact Gaussian-convolution marginal.

        E_{x~q_i}[N(x; mu_v, Sigma_v)] = N(mu_q_i; mu_v, Sigma_q_i + Sigma_v)
    (the Gaussian convolution identity: the expectation of one Gaussian density under another
    integrates to a Gaussian in the mean difference with the SUMMED covariances). Taking the log,
    dropping the v-independent constant -(K/2) log(2 pi), and tempering by tau_eff (diagonal
    family):

        logit_{i,v} = -1/(2 tau_eff) * sum_k [ (mu_q - mu_v)^2 / (sigma_q + sigma_v)
                                               + log(sigma_q + sigma_v) ].

    Unlike the -KL readout this scores q as an OBSERVATION model marginal (Bayes-exact up to the
    dropped constant), so a diffuse prior pi_v is penalized through log(sigma_q + sigma_v) rather
    than rewarded through the KL's 1/sigma_v flattening. DIAGONAL family only (registered without
    is_full, so the config rank cross-check pairs it with diagonal families by construction).
    Chunked over the vocabulary: the couplings sigma_q + sigma_v block the diagonal kernel's
    single-matmul trick, so each chunk broadcasts a (B, N, Vc, K) workspace; the (B, N, V) output
    is inherent to producing every logit (the training memory win is the fused CE twin
    ``decode_ce_expected_likelihood_chunked``).
    """
    sigma_v_all = bounded_variance_from_log(
        pb._decode_sigma_log_table(), eps=pb.eps,
    )                                                                         # (V, K)
    mu_v_all = pb._decode_mu_table()                                     # (V, K) decode table (untied if set)
    chunk = pb.decode_chunk_size
    V = pb.vocab_size

    logit_chunks = []
    for v0 in range(0, V, chunk):
        v1 = min(v0 + chunk, V)
        d = mu_q.unsqueeze(-2) - mu_v_all[v0:v1]                         # (B, N, Vc, K)
        s = sigma_q.unsqueeze(-2) + sigma_v_all[v0:v1]                   # (B, N, Vc, K) convolved variances
        logit_chunks.append(-0.5 * (d ** 2 / s + torch.log(s)).sum(-1) / tau_eff)  # (B, N, Vc)
    return torch.cat(logit_chunks, dim=-1)                               # (B, N, V)


@register_decode(
    "linear",
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_linear_chunked,
)
def _decode_linear(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances (DISCARDED)
    tau_eff: torch.Tensor,               # () effective temperature (DISCARDED)
) -> torch.Tensor:                       # (B, N, V) logits = mu_q @ W^T (+ b)
    r"""Linear-projection decode (use_prior_bank=False): logits = mu_q @ W^T (+ b).

    The one authorized neural exception: a single learned (V, K) output weight applied to the
    converged mean, with NO KL geometry at the decode boundary (the decode temperature is discarded;
    only encode + the E-step remain gauge-aware). Realized as a raw nn.Parameter matmul, not an
    nn.Linear module. With ``decode_bias`` a learned per-vocab log-unigram bias ``b`` (V,) is added
    (see __init__). The pure KL-readout path is always available under use_prior_bank=True; this is
    the opt-in ablation the user uses to compare with/without the prior-bank decode.
    """
    x = mu_q                                                           # bare converged mean
    logits = x @ pb.output_proj_weight.transpose(-1, -2)               # (B, N, V)
    if pb.output_proj_bias is not None:
        logits = logits + pb.output_proj_bias                           # learned log-unigram prior
    return logits
