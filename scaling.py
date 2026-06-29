r"""Click-to-run PARAMETER-scaling experiment runner for the VFE_3.0 transformer.

Scaling experiments are with respect to NUMBER OF PARAMETERS. This runner loops a size grid x a
seed list, training each (size, seed) cell into its own self-contained ``RunArtifacts`` directory and
calling ``finalize_run`` so EVERY point carries the canonical held-out TEST cross-entropy and the
enriched ``scaling_point`` block (n_params, n_gen, active-params-per-token, FLOP proxies, wall-clock).
There is no CLI arg parsing (project policy): edit the ``CONFIG`` dict and the active ``ROUTES`` at the
bottom, then run ``python scaling.py``. Aggregate + fit + plot afterwards with ``scaling_analysis.py``.

WHY A PARAMETER AXIS IS SUBTLE HERE (read before picking a grid). The pure-path parameters are the
prior tables only: ``mu_embed (V,K)``, ``sigma_log_embed (V,K)``, ``phi_embed (V,n_gen)``, and a scalar
(prior_bank.py). So ``N = 2*V*K + V*n_gen + 1`` with ``V=50257``, and ``phi_embed = V*n_gen`` usually
DOMINATES. ``n_gen`` is set by the gauge group: for ``block_glk`` it is ``K^2/n_heads`` (so FEWER/larger
blocks = MORE params -- the opposite sign of a standard transformer); ``glk`` is ``K^2``; ``so_k`` is
``K(K-1)/2``; the ``so_n``/``sp_n`` towers decouple ``n_gen`` from ``K`` entirely. Three consequences:
the gauge group / n_heads is a FIRST-CLASS parameter lever; growing ``embed_dim`` moves ``N`` on two
fronts (linear ``2VK`` + quadratic ``V*n_gen``); and ``n_layers`` / ``n_e_steps`` / full-covariance add
ZERO parameters (they are inference-compute axes at flat ``N``, plotted separately, NEVER on ``L(N)``).

The baseline operating point is the self-contained ``config`` dict IN THIS FILE (it is NOT imported
from ``train_vfe3.py`` -- edit ``config`` below to set what every scaling cell trains around). ``BASELINE``
is bound to it right after the dict is built. Each cell overrides only the scale knob(s). Equal-token
budget: ``max_steps`` / ``batch_size`` /
``max_seq_len`` are held fixed across the parameter routes, so ``tokens_seen`` is constant and the fitted
exponent is the equal-data exponent. A missing tokenized cache raises ``FileNotFoundError`` (no
synthetic substitution); build the corpus cache first (see ``vfe3/data``).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Anaconda + PyTorch each ship a
#   libiomp5md.dll; the duplicate OpenMP init aborts the process (seen with n_e_steps>1). This MUST
#   run before `import torch`. The clean fix is one OpenMP in the env (e.g. `conda install nomkl`);
#   override by exporting KMP_DUPLICATE_LIB_OK yourself. See docs/edits/2026-06-05.

import copy
import gc
import json
import logging
import time
from dataclasses import asdict
from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from vfe3.config import VFE3Config
from vfe3.data.datasets import make_dataloader, tokens_per_char as _tokens_per_char
from vfe3.model.model import VFEModel, build_group
from vfe3.run_artifacts import RunArtifacts, finalize_run
from vfe3.train import coverage_lines, train

logger = logging.getLogger("scaling")


# =============================================================================
# CLICK-TO-RUN KNOBS  -- edit, then run.
# =============================================================================
CONFIG: Dict[str, Any] = {
    # Which routes to run (keys of ROUTES), in order. See the ROUTE MENU above the ROUTES registry.
    "routes":     ["grow_K_GL10"], # "grow_K_GL10","blocks_K48",                       
                   #"blocksize", "grow_K", "group", "model_channel", "infer_T", "infer_L"

    # Seeds per cell. Graduated budget is sensible (more seeds at the cheap small end); the simplest
    # honest default is one shared list applied to every cell -- trim/extend per your compute budget.
    "seeds":      [6, 64, 23],

    "device":     "auto",                                   # 'auto' -> CUDA (RTX 5090) else CPU

    # Dataset for every run (NOT a VFE3Config field; the loader seam). Held-out CE is comparable across
    # sizes only within one tokenizer/corpus.
    "dataset":    "wikitext-103",                           # "wikitext-103" | "wiki-ja" | "wiki-en" | ...

    # Cap the TRAIN stream for fast scaling passes (validation/test always read in full). None = full.
    "max_tokens": None,

    # Override every run's max_steps (None = use the local `config` max_steps below). HOLD THIS FIXED across the
    # parameter routes for an equal-token budget (so tokens_seen is constant and the exponent is clean).
    "max_steps":  None,

    # Skip cells whose run dir already holds a summary.json built from the SAME config (idempotent
    # reruns / crash recovery), exactly like ablation.py's resume.
    "resume":     True,

    "output_dir": "vfe3_scaling_results",
}

config = dict(
    

    #################################
    #            Training
    #################################
    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*)
    
    embed_dim                 = 120,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 12,
    
    max_seq_len               = 128,                 # N, context length
    
    batch_size                = 64,
    max_steps                 = 60000,
    
    n_layers                  = 1,                   # L, number of blocks
    n_e_steps                 = 1 ,                   # T, E-step inner iterations
    
    seed                      = 6,
    warmup_steps              = 100,
    
    #################################
    # f-divergence and e/m family
    #################################
    
    divergence_family         = "renyi",   # "renyi", "squared_hellinger","bhattacharyya", "jeffreys",
    renyi_order               = 1.0,       # Renyi order (1.0 -> KL)

    family                    = "gaussian_diagonal", # "gaussian_diagonal" | "gaussian_full" | "laplace_diagonal" (single covariance toggle; diagonal_covariance is derived)
    
    #################################
    #        Initialization
    #################################
    mu_init_std               = 0.065,     # std of the random mean table mu_embed
    sigma_init                = 3,         # constant initial coordinate variance (sigma_log = log of this)
    phi_scale                 = 0.06,      # std
    
    
    #################################
    #        Encode/Decode          #
    #################################
    decode_bias               = True,     # only if use_prior_bank = False
    use_head_mixer            = True,      # opt-in Schur-commutant head mixer (needs >=2 equal blocks (block_glk/tied_block_glk) OR a labeled irrep tower (so_n/sp_n: per-isotypic-component mixing; mults-one towers get scalar gains));
                                           # breaks strict equivariance under block_glk (exact at init); EXACT under tied_block_glk (full-cov)
    
    use_prior_bank            = False,               # True: KL-to-prior decode (pure path). False: linear projection
                                                     # mu->logits ablation (encode stays on the prior bank)
    decode_tau                = 0.008,
    decode_precision_scaled   = False,               # use_prior_bank=False only: feed the precision-weighted mean
                                                     # eta=mu/sigma (natural param) to the linear head so Sigma enters
                                                     # the discriminative readout (diagnostic; OFF = bare-mu linear)
    decode_mode               = 'diagonal_chunked',
    oracle_unroll_grad        = False,
    
    #################################
    #          Gauge Group
    #################################
    gauge_transport           = "on",         # gauge-frame ABLATION (A1/EXP-2): "on" (pure, learned frame)
                                              #   | "off" (Omega=I exactly: forces phi_scale=0, pos_phi='none',
                                              #     e_phi_lr=m_phi_lr=0; needs transport_mode='flat' + pos_rotation='none')
                                              #   | "frozen" (random fixed frame: e_phi_lr=m_phi_lr=0, phi_scale kept).
                                              #   NOT transport_mode (flat vs regime_ii). docs/hypotheses/2026-06-21-hypotheses.md
    gauge_parameterization    = "phi",        # "phi" | "omega_direct" (omega_direct: live-rejected, no belief source)
    
    m_phi_natural_grad        = False,        # natural gradient on phi m-step
    
    m_gauge_update_rule       = "heavy_ball",       #'adam' or 'heavy_ball'
    
    phi_precond_mode          = "pullback_per_block",  # "none" | "clip" | "killing" | "killing_per_block" | "pullback" | "pullback_per_block"
    phi_retract_mode          = "bch",                # "euclidean" | "bch"
    spd_retract_mode          = "spd_affine",         # SPD covariance retraction (registry: "spd_affine" | "log_euclidean")

    
    gauge_group               = "block_glk",    # "glk" | "block_glk" | "tied_block_glk" | "so_k" | "sp" | "so_n" | "sp_n"
                                                     # tied_block_glk: one shared GL(d) frame across heads (kron(I_n, gl(d)))

    # so_n / sp_n irrep towers (heads = irreps). Structure group SO(group_n) / Sp(group_n) with
    # group_n DECOUPLED from embed_dim; irrep_spec = [(label, mult), ...] blocks laid out in order,
    # block dims summing to embed_dim. Labels: so_n 'l<p>' = symmetric-traceless rank-p irrep
    # (group_n=3: spin-p, dim 2p+1); sp_n 'sym<p>' = Sym^p of the defining rep (dim C(2m+p-1, p)).
    # One shared per-token phi drives EVERY block (TIED gauge; n_gen = dim of the algebra), and
    # unequal block dims get per-head tau_h = kappa_h*sqrt(d_h). Both REQUIRED for so_n/sp_n,
    # must stay None for every other group. CONSTRAINTS for these groups: phi_precond_mode must
    # be "none"/"clip"/"killing" (the per-block modes are rejected -- tied generators do not
    # partition per block); use_head_mixer mixes per isotypic component (equal-mult towers mix copies; mults-one towers get scalar gains); alibi-family priors need
    # n_heads == number of blocks.
    # embed_dim=20 examples:
    #   so_n: group_n=3, irrep_spec=[("l2", 4)]                            # 4 equal spin-2 heads (mixer OK)
    #   so_n: group_n=3, irrep_spec=[("l0",1),("l1",1),("l3",1),("l4",1)]  # spins 0,1,3,4 = 1+3+7+9 (unequal: mixer = per-head scalar gains)
    #   sp_n: group_n=4, irrep_spec=[("sym2", 2)]                          # 2 equal Sym^2(R^4) heads, dim 10 each
    
    group_n                   = None,                # so_n/sp_n only: N of SO(N) / 2m of Sp(2m)
    irrep_spec                = None,                # so_n/sp_n only: [(label, mult), ...]; dims sum == embed_dim

    use_cg_coupling           = False,               # so_n/sp_n only: CG cross-type coupling (bilinear, exactly
                                                     # equivariant, means-only sigma; zero-init path weights)

    ####################################
    # Non-Flat Connection - Regime II
    ####################################
    transport_mode            = "flat",     # "flat" (Regime-I phi-cocycle) | "regime_ii" (learned bilinear edge
                                            # connection delta=mu^T W mu; gauge-invariant only at W=0; NN exception, default-off)
                                            # | "regime_ii_covariant" (Route B: gauge-COVARIANT non-flat connection
                                            # delta=M . invariant-features(q_i, Omega^0 q_j); covariant for any M; NN exception, default-off)
    cocycle_relaxation        =   1.0,        # regime_ii / regime_ii_covariant homotopy: 0.0 -> flat, 1.0 -> fully relaxed (ignored by flat)
    cross_couplings           = None,       # off-block GL(K) head pairs e.g. [(0, 1)]; block_glk only (None = block-diagonal gauge)
                                               #if enabled and head-mixer = True or causal_alibi it will fail
    close_basis               = False,
    ####################################
    #       Positional Encoding
    #    BCH gauge-frame PE (pos_phi)
    #     gauge-RoPE (pos_rotation)
    ####################################

    pos_phi                   = "learned",           # "none" (pure path) | "learned" | "frozen"
    pos_rotation              = "none",              # "none" | "rope" (block-diagonal positional rotation folded into transport)
    pos_phi_compose           = "bch",               # composition chart: "bch" | "euclidean"
               
    pos_phi_scale             = 0.02,                # learned-table init scale AND frozen per-position step
    
    rope_base                 = 100.0,               # rotary frequency base
    rope_full_gauge           = False,               # rotate the covariance sandwich too (REQUIRES family="gaussian_full")
    rope_on_value             = False,
    
    ######################################
    #                Self Energy:  
    #        Sum_i alpha_i * KL(q_i||p_i)
    ######################################
    lambda_alpha_mode          = "state_dependent",  # "constant" | "learnable" | "state_dependent" | "state_dependent_per_coord"
    lambda_h_mode              = "constant",  # "constant" | "state_dependent" (lambda_h*=c0_h/(b0_h+KL); +R_h) | "learnable" (NN exc.)
    
    b0                         = 1.0,                 # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0                         = 1.0,                 # state-dependent alpha shape (numerator)
       
    lambda_alpha               = 1,          # constant self-coupling value
    lambda_h                   = 0.25,       # hyper-prior weight lambda_h * mean_i KL(s_i||r) (0 = OFF; >0 creates s/r tables)
    #lambda h ~0.25/6 = 0.04 for K=160 d=20
    
    b0_h                       = 1.0,        # state-dependent lambda_h shape: lambda_h* = c0_h/(b0_h + KL(s||r))
    c0_h                       = 1.0,        # state-dependent lambda_h shape (numerator); max precision c0_h/b0_h

    # Further Regularizers
    mass_phi                   = 0.0,       # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight = 0.0,      # alpha_hat * sum_i KL(q_i*||p_i) M-step term (0 = OFF)
    
    
    ##################################################
    #              Attention Energy: 
    # lambda_beta*Sum_i beta_ij * KL(q_i||Omega_ij q_j) 
    ##################################################
    
    lambda_beta                = 1.0,        # belief-coupling block weight (1.0 = pure F)    
    lambda_gamma               = 0.75,       # model-channel coupling (0 = OFF; >0 creates s tables, predictively inert by default)
         

    ########################################
    #     Attention Belief/Model Settings
    #            & Temperatures
    ########################################
    
    kappa_beta                = 1, #[1, 0.5],        # tau = kappa * sqrt(d_head); kappa=1 -> Vaswani temperature
    kappa_gamma               = 1, #[1, 0.5],        # model-channel temperature tau_gamma = kappa_gamma*sqrt(d_head)
        
    beta_attention_prior      = "causal_alibi",        # "uniform" | "causal" | "alibi" | "causal_alibi" | "windowed" | "causal_windowed" | "t5_relative_bias"
    gamma_attention_prior     = "causal_alibi",        # model-channel prior pi^s_ij (same 7 keys): "uniform" | "causal" | "alibi" | "causal_alibi" | "windowed" | "causal_windowed" | "t5_relative_bias"

    t5_learnable_bias         = False,           # learn the per-bucket T5 bias table b_{i-j} (sanctioned NN exception, default OFF; needs a t5_relative_bias channel)

    precision_weighted_attention = True,        # down-weight high-variance keys: fold detached -log(b0 + tr Sigma_j)
                                                 # into the attention prior (diagnostic; OFF = position-only prior)
    precision_attention_b0       = 2.0,          # b0 in the per-key reliability -log(b0 + tr Sigma_j); > 0
    precision_attention_per_head = False,        # per-key reliability PER HEAD (trace over each block's coords) vs
                                                 # global (all K); needs precision_weighted_attention=True
    #################################
    #         Belief E-step
    #         Learning Rates
    #################################
    
    e_q_mu_lr                 = 0.9,
    e_q_sigma_lr              = 0.001,
    e_phi_lr                  = 0.00,     
    
    
    ####################################
    #       Model E-step LR's
    #      If s_e_step = True
    # and prior_source = 'model_channel'
    ####################################
    
    r_update_mode             = "gradient",          # "gradient" (AdamW M-step; correct under s_e_step) | "barycenter" (closed-form forward-KL centroid of s; exact M-step in the scored s_e_step=False regime)
    prior_source              = "model_channel",    # belief prior p_i: "token" or "model_channel"
    learnable_r               = True,               # un-freeze hyper-prior centroid r (empirical-Bayes)
    s_e_step                  = True,
    
    e_s_mu_lr                 = 0.85,
    e_s_sigma_lr              = 0.1,
    
    #################################
    #    Embedding/Priors M-step 
    #        Learning Rates
    #################################
        
    m_p_mu_lr                 = 0.015,   
    m_p_sigma_lr              = 0.0045,     
    m_phi_lr                  = 0.015,   
    
    weight_decay              = 0.02,
    phi_weight_decay          = 0.05,
    
    min_lr                    = 0,       # absolute cosine-decay LR floor (0.0 = pure cosine)
    min_lr_frac               = 0.01,    # proportional LR floor, max(min_lr, frac*base); OFF
    
    #################################
    #     Layer Normalization 
    #        and Hand-Off
    #################################
    
    norm_type_block           = "none",              # "none" | "mahalanobis"
    norm_type_final           = "none",              # "none" | "mahalanobis"
    
    prior_handoff_rho         = 0,                 # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma       = 0,                 # sigma damping in [0,1] (0.0 = frozen at embedding)
    
    #################################
    #        Numerical Safety
    #################################
    
    e_mu_q_trust              = None,
    e_sigma_q_trust           = 10.0,
    sigma_max                 = 100.0,
    
    #################################
    #         Misc/Logging
    #################################     
    amp_dtype                 = None,      # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32
        
    log_interval              = 100,       # console log every N steps (0 = off)
    eval_interval             = 1500,      # periodic validation every N steps (0 = off)
    checkpoint_interval       = 25000,     # save a resumable checkpoint every N steps (0 = off)

    use_ema                   = False,     # EMA/Polyak averaging of the trained tables (default OFF = pure
                                           # path: model is the last SGD iterate). ON: eval/best-save/final
                                           # model use the running average s <- ema_decay*s + (1-ema_decay)*theta
    ema_decay                 = 0.95,     # EMA decay in (0,1); only read when use_ema=True
)

# kl_max is the numerical safety-net clamp on EVERY divergence (KL(q||p), KL(s||r), pairwise energy),
# next to eps -- NOT an operating ceiling. Diagonal KL is a sum over K coords (~0.8 nats/coord trained),
# so the K-INDEPENDENT 100.0 default binds for ~100% of tokens at large K (K* ~ 126), silently zeroing
# the hyper-prior self-gradient and gradient-freezing learnable r (the kernel self_mask, gradients/
# kernels.py:129). Scale it with K so it binds only on genuine NaN/inf/Cholesky blowups; F is provably
# kl_max-independent below the ceiling (safe_kl_clamp is the identity there). See docs/2026-06-21-edits.md.
config["kl_max"] = 8 * config["embed_dim"]

# The scaling operating point IS the local ``config`` dict above (self-contained; NOT train_vfe3.py).
# Edit ``config`` to change what every cell trains around; each ROUTE cell overrides only its scale knob(s).
BASELINE: Dict[str, Any] = config


# =============================================================================
# ROUTE BUILDERS  -- each returns a list of cells; a cell is
#   {"label", "route", "scale_knob", "overrides": {VFE3Config field: value, ...}}.
# Every cell's overrides must independently satisfy VFE3Config.__post_init__ (n_heads | embed_dim,
# use_head_mixer needs >= 2 equal blocks, alibi priors need n_heads == n_blocks, ...); a cell that
# violates a cross-field constraint is caught at construction and recorded as a config-error point
# (never crashes the grid), but the defaults below are pre-satisfied so they construct cleanly.
# =============================================================================

def route_grow_k(embed_dims: List[int], n_heads: int = 4) -> List[Dict[str, Any]]:
    r"""Grow N by widening embed_dim at a FIXED block_glk head count (route A). Mixed linear+quadratic
    route: 2VK grows linearly, phi_embed = V*K^2/n_heads quadratically. n_heads stays equal to the
    block count so the baseline causal_alibi prior and the head mixer remain valid."""
    return [{"label": f"K{k}", "route": "grow_K", "scale_knob": "embed_dim",
             "overrides": {"embed_dim": k, "n_heads": n_heads, "gauge_group": "block_glk"}}
            for k in embed_dims]


_VFE3_DEFAULTS = {f.name: f.default for f in _dc_fields(VFE3Config)}


def _baseline_value(key: str) -> float:
    r"""Operating-point value of a VFE3Config float field: the local ``config`` dict (BASELINE) if it
    sets it, else the dataclass default. Lets the muP route scale LRs / init relative to the anchor
    width without hard-coding the operating point."""
    return float(BASELINE.get(key, _VFE3_DEFAULTS.get(key)))


def route_grow_k_mup(embed_dims: List[int], n_heads: int = 4, anchor_k: int = 20) -> List[Dict[str, Any]]:
    r"""muP width-stability route for the inverse-K exponent (F1/EXP-6). For each K it emits a matched
    PAIR: a 'fixed' arm at the baseline mean LR/init (the width-fixed control) and a 'mup' arm that
    scales the E/M-step mean LRs ~ anchor_k/K and the mean-init std ~ sqrt(anchor_k/K) (Tensor-Programs
    muP for width). BOTH arms recompute kl_max = 8*K per cell -- the baseline freezes kl_max at
    8*train_K (a width confound that over-relaxes every small-K cell and zeros the hyper-prior self-
    gradient near K ~ 126; see docs/experiments/2026-06-21-experiment-readiness.md). Anchored at
    K=anchor_k, where the muP factors are 1 and the two arms coincide. Scoped to this route so the
    other routes (incl. an active blocksize run) are untouched. The fitted exponent should be compared
    against embed_dim, not n_params (which is K^2-dominated by phi_embed)."""
    base_eqmu = _baseline_value("e_q_mu_lr")
    base_mpmu = _baseline_value("m_p_mu_lr")
    base_init = _baseline_value("mu_init_std")
    cells: List[Dict[str, Any]] = []
    for k in embed_dims:
        common = {"embed_dim": k, "n_heads": n_heads, "gauge_group": "block_glk", "kl_max": 8 * k}
        cells.append({"label": f"K{k}_fixed", "route": "grow_K_mup", "scale_knob": "embed_dim",
                      "overrides": dict(common)})
        w = anchor_k / k                                     # muP width ratio (1 at the anchor)
        cells.append({"label": f"K{k}_mup", "route": "grow_K_mup", "scale_knob": "embed_dim",
                      "overrides": {**common, "e_q_mu_lr": base_eqmu * w,
                                    "m_p_mu_lr": base_mpmu * w, "mu_init_std": base_init * (w ** 0.5)}})
    return cells


def route_blocksize(embed_dim: int, n_heads_list: List[int]) -> List[Dict[str, Any]]:
    r"""Grow N by SHRINKING the head count at fixed K (route B): block_glk n_gen = K^2/n_heads, so
    fewer/larger blocks -> more params. n_heads == n_blocks keeps causal_alibi + the mixer valid."""
    return [{"label": f"K{embed_dim}_h{h}", "route": "blocksize", "scale_knob": "n_heads",
             "overrides": {"embed_dim": embed_dim, "n_heads": h, "gauge_group": "block_glk"}}
            for h in n_heads_list]


def route_grow_k_fixed_block(embed_dims: List[int], block: int) -> List[Dict[str, Any]]:
    r"""Grow N by widening K at a FIXED gauge block size GL(``block``): n_heads = K/block scales WITH K
    (every head is one GL(block) frame). Because block_glk has n_gen = K^2/n_heads = block*K, phi_embed
    = V*block*K grows LINEARLY in K here -- contrast ``route_grow_k`` (fixed n_heads, so the block grows
    and n_gen ~ K^2). kl_max = 8*K per cell (avoids the frozen-kl_max width confound). The single-block
    K=block cell drops the head mixer (the Schur-commutant mixer needs >=2 equal blocks). K must be a
    positive multiple of ``block``; non-multiples are skipped with a warning."""
    cells: List[Dict[str, Any]] = []
    for k in embed_dims:
        if k <= 0 or k % block != 0:
            logger.warning("  [skip] grow_K_GL%d: K=%d is not a positive multiple of block=%d", block, k, block)
            continue
        h = k // block
        ov: Dict[str, Any] = {"embed_dim": k, "n_heads": h, "gauge_group": "block_glk", "kl_max": 8 * k}
        if h < 2:
            ov["use_head_mixer"] = False                     # 1 block: nothing for the head mixer to mix
        cells.append({"label": f"K{k}_GL{block}", "route": f"grow_K_GL{block}",
                      "scale_knob": "embed_dim", "overrides": ov})
    return cells


def route_vary_block_fixed_k(embed_dim: int, blocks: List[int]) -> List[Dict[str, Any]]:
    r"""Fixed K, vary the gauge block size GL(``b``): n_heads = K/b, so block_glk n_gen = K^2/n_heads =
    b*K. FEWER/LARGER blocks (bigger b) = MORE params (n_gen up) -- the opposite sign of a standard
    transformer, and the parameter axis here. This is the ``route_blocksize`` idea written in GL(b) terms
    (b=K/n_heads). kl_max = 8*K (constant; K is fixed across the route). The single-block b=K cell (one
    GL(K) frame) drops the head mixer. Each b must divide K; non-divisors are skipped with a warning."""
    cells: List[Dict[str, Any]] = []
    for b in blocks:
        if b <= 0 or embed_dim % b != 0:
            logger.warning("  [skip] blocks_K%d: block=%d does not divide K=%d", embed_dim, b, embed_dim)
            continue
        h = embed_dim // b
        ov: Dict[str, Any] = {"embed_dim": embed_dim, "n_heads": h, "gauge_group": "block_glk",
                              "kl_max": 8 * embed_dim}
        if h < 2:
            ov["use_head_mixer"] = False
        cells.append({"label": f"K{embed_dim}_GL{b}", "route": f"blocks_K{embed_dim}",
                      "scale_knob": "n_heads", "overrides": ov})
    return cells


def route_group(embed_dim: int) -> List[Dict[str, Any]]:
    r"""Grow/shrink N by changing the gauge GROUP at fixed K (route C): tied_block_glk (tiny n_gen) ->
    block_glk -> so_k span a wide n_gen range. A headless 'causal' beta prior is used for ALL arms so
    the attention prior is identical across groups (alibi would otherwise need n_heads == n_blocks,
    which differs by group); single-block arms also drop the head mixer (nothing to mix). glk (K^2,
    the largest) is left commented -- at K=64 it is ~212M params, near the single-GPU ceiling once 3x
    AdamW moments count; uncomment if VRAM allows."""
    headless = {"beta_attention_prior": "causal"}
    return [
        {"label": f"K{embed_dim}_tied_h8", "route": "group", "scale_knob": "gauge_group",
         "overrides": {"embed_dim": embed_dim, "n_heads": 8, "gauge_group": "tied_block_glk", **headless}},
        {"label": f"K{embed_dim}_block_h8", "route": "group", "scale_knob": "gauge_group",
         "overrides": {"embed_dim": embed_dim, "n_heads": 8, "gauge_group": "block_glk", **headless}},
        {"label": f"K{embed_dim}_so_k", "route": "group", "scale_knob": "gauge_group",
         "overrides": {"embed_dim": embed_dim, "n_heads": 1, "gauge_group": "so_k",
                       "use_head_mixer": False, **headless}},
        # {"label": f"K{embed_dim}_glk", "route": "group", "scale_knob": "gauge_group",
        #  "overrides": {"embed_dim": embed_dim, "n_heads": 1, "gauge_group": "glk",
        #                "use_head_mixer": False, **headless}},
    ]


def route_model_channel() -> List[Dict[str, Any]]:
    r"""Grow N by ~2VK by turning ON the model-channel s tables (route D), vs a pure single-tier token
    prior. A coarse 2-point route: the 'token' arm strips the s/r tables (prior_source='token',
    s_e_step/lambda_h/lambda_gamma off), the 'model_channel' arm keeps the baseline channel. Only the
    s-table mass counts as real added capacity when gamma/lambda_h shape s beyond CE."""
    return [
        {"label": "token_prior", "route": "model_channel", "scale_knob": "model_channel",
         "overrides": {"prior_source": "token", "s_e_step": False, "learnable_r": False,
                       "lambda_h": 0.0, "lambda_gamma": 0.0}},
        {"label": "model_channel", "route": "model_channel", "scale_knob": "model_channel",
         "overrides": {}},                                   # baseline already runs the channel
    ]


def route_inference_t(n_e_steps_list: List[int]) -> List[Dict[str, Any]]:
    r"""FLAT-N inference-compute axis: more E-step inner iterations T at constant params. route tagged
    'inference' so the analyzer plots it on the inference-capacity figure, NEVER the L(N) curve."""
    return [{"label": f"T{t}", "route": "inference", "scale_knob": "n_e_steps",
             "overrides": {"n_e_steps": t}} for t in n_e_steps_list]


def route_inference_l(n_layers_list: List[int]) -> List[Dict[str, Any]]:
    r"""FLAT-N inference-compute axis: stacked blocks L at constant params (depth re-primes the one
    shared PriorBank, adding zero parameters). route 'inference' (flat-N), like route_inference_t."""
    return [{"label": f"L{n}", "route": "inference", "scale_knob": "n_layers",
             "overrides": {"n_layers": n}} for n in n_layers_list]


# ============================ ROUTE MENU (how to set up a run) ===============================
# A ROUTE is one way of moving N (number of params). Pick which ones run in CONFIG["routes"] above;
# edit each route's grid HERE (the call args). The predicted n_params is printed per cell before any
# training so you can size a grid to the GPU first. Geometric (~2x) spacing in N gives the cleanest fit.
#
#   PARAMETER routes (plotted on the L(N) = test_CE-vs-N power-law curve):
#     grow_K_GL10   GROW K at a FIXED block GL(10): K=10,20,...; n_heads=K/10; n_gen=10*K (LINEAR in K).
#     blocks_K48    FIXED K=48, VARY the block GL(b): b=48,24,12,8,6; n_heads=K/b; n_gen=b*K
#                   (bigger block = FEWER heads = MORE params -- e.g. GL(48) is the largest model here).
#     grow_K        grow K at FIXED n_heads=4 (so the block grows with K; n_gen ~ K^2/4, quadratic).
#     grow_K_mup    grow_K + a muP LR/init-rescaled twin per K (F1/EXP-6 width-stability), kl_max=8*K.
#     blocksize     FIXED K=64, vary n_heads in {8,4,2} (= block GL(8),GL(16),GL(32)); same idea as
#                   blocks_K48 but parameterized by n_heads instead of GL(b).
#     group         FIXED K=64, swap the gauge GROUP: tied_block_glk -> block_glk -> so_k (spans n_gen).
#     model_channel token-prior (no s-tables) vs the full model channel (+2VK params).
#   INFERENCE routes (FLAT N -- plotted on a SEPARATE inference-capacity figure, NEVER on L(N)):
#     infer_T       n_e_steps in {1,2,4,8} at constant params.   infer_L  n_layers in {1,2,4,6}.
#
# To add your own: call a route builder with a new grid and give it a key; add that key to CONFIG["routes"].
ROUTES: Dict[str, List[Dict[str, Any]]] = {
    "grow_K_GL10":   route_grow_k_fixed_block([90, 100, 110, 120], block=10),
    "blocks_K48":    route_vary_block_fixed_k(48, [48, 24, 12, 8, 6]),
    "grow_K":        route_grow_k([20, 40, 60, 80, 100, 120], n_heads=4),
    "grow_K_mup":    route_grow_k_mup([20, 40, 80, 120], n_heads=4, anchor_k=20),  # F1/EXP-6 (fixed vs muP)
    "blocksize":     route_blocksize(64, [8, 4, 2]),
    "group":         route_group(64),
    "model_channel": route_model_channel(),
    "infer_T":       route_inference_t([1, 2, 4, 8]),
    "infer_L":       route_inference_l([1, 2, 4, 6]),
}





# =============================================================================
# PARAMETER PREDICTION  -- size a grid to the GPU before committing to long runs.
# =============================================================================

def predict_n_params(cfg: VFE3Config) -> Tuple[int, int]:
    r"""Predicted total ``n_params`` and ``n_gen`` for ``cfg``, by building only the (cheap) gauge group
    and summing the prior-table sizes per ``PriorBank`` (prior_bank.py). Exact on the pure path; the
    small head-mixer / CG / connection_W / learnable-scalar tables (when toggled on) are omitted, so a
    tiny predicted-vs-actual gap there is expected and only printed, never enforced."""
    n_gen = int(build_group(cfg).generators.shape[0])
    V, K = int(cfg.vocab_size), int(cfg.embed_dim)
    n = 2 * V * K + V * n_gen + 1                            # mu_embed, sigma_log_embed, phi_embed, decode_log_scale
    if not cfg.use_prior_bank:
        n += V * K                                          # output_proj_weight
        if cfg.decode_bias:
            n += V                                          # output_proj_bias
    if cfg.lambda_h > 0.0 or cfg.lambda_gamma > 0.0 or cfg.prior_source == "model_channel" or cfg.s_e_step:
        n += 2 * V * K                                      # s_mu_embed, s_sigma_log_embed
    if cfg.lambda_h > 0.0 or cfg.s_e_step:
        n += 2 * K                                          # r_mu, r_sigma_log
    if cfg.pos_phi == "learned":
        n += int(cfg.max_seq_len) * n_gen                   # pos_phi_free
    return n, n_gen


# =============================================================================
# LOADERS  -- memoised on the fields that actually change the stream (mirrors ablation.get_loader).
# =============================================================================
_LOADER_CACHE: Dict[Tuple[Any, ...], Any] = {}


def get_loader(
    dataset:    str,
    seq_len:    int,
    batch_size: int,
    split:      str,

    *,
    max_tokens: Optional[int] = None,
) -> Any:
    r"""Split-aware DataLoader for ``dataset``/``split`` (a missing cache raises ``FileNotFoundError``).

    Memoised on ``(dataset, seq_len, batch_size, split, cap)`` so the corpus loads once across the grid.
    Only the train stream shuffles / drops the partial last batch; validation/test read the whole split
    in a stable order so the held-out metric is a full-corpus measurement. ``max_tokens`` caps the train
    split only. No synthetic substitution for a missing real corpus."""
    cap = max_tokens if split == "train" else None
    key = (dataset, seq_len, batch_size, split, cap)
    if key not in _LOADER_CACHE:
        _LOADER_CACHE[key] = make_dataloader(dataset, split, seq_len, batch_size,
                                             shuffle=(split == "train"), drop_last=(split == "train"),
                                             max_tokens=cap)
    return _LOADER_CACHE[key]


# =============================================================================
# SINGLE-CELL EXECUTOR  -- one independent (size, seed) run (replicates _run_once's body).
# =============================================================================

def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cell_cfg_dict(overrides: Dict[str, Any], seed: int, max_steps: Optional[int]) -> Dict[str, Any]:
    r"""The exact kwargs a cell's VFE3Config is built from: baseline + overrides + run knobs. Single
    source of truth, shared by ``run_cell`` and the resume staleness check."""
    d = copy.deepcopy(dict(BASELINE))
    d.update(overrides)
    d["seed"] = int(seed)
    d["checkpoint_interval"] = 0                             # no per-cell step_N.pt blowup
    d["generate_figures"] = False                           # single-run replay figures are off the scaling path
    if max_steps is not None:
        d["max_steps"] = int(max_steps)
    return d


def _cell_is_current(run_dir: Path, cfg: VFE3Config, dataset: str) -> bool:
    r"""True iff the run dir already holds a summary.json AND its config.json equals the config we would
    build now (guards resume against baseline drift / a changed dataset)."""
    if not (run_dir / "summary.json").exists() or not (run_dir / "config.json").exists():
        return False
    try:
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        built = json.loads(json.dumps(asdict(cfg), default=str))
    except Exception:
        return False
    return saved.get("dataset") == dataset and saved.get("config") == built


def run_cell(
    cell:       Dict[str, Any],
    run_dir:    Path,
    seed:       int,

    *,
    dataset:    str,
    device:     torch.device,
    max_tokens: Optional[int] = None,
    max_steps:  Optional[int] = None,
) -> Dict[str, Any]:
    r"""Build a fresh model from baseline+overrides, train it, score the held-out TEST split via
    ``finalize_run``, and return a harvest dict. A cross-field config violation is caught and returned
    as ``error_kind='config'`` (not raised), keeping it distinct from a training crash."""
    label = cell["label"]
    cfg_dict = _cell_cfg_dict(cell["overrides"], seed, max_steps)
    try:
        cfg = VFE3Config(**cfg_dict)
    except (ValueError, NotImplementedError, TypeError) as exc:
        logger.warning("  [config rejected] %s: %s", label, exc)
        return {"label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
                "error_kind": "config", "error": str(exc), "seed": int(seed),
                "test_ce": None, "n_params": None}

    if CONFIG["resume"] and _cell_is_current(run_dir, cfg, dataset):
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        sp = summary.get("scaling_point", {})
        print(f"    [CACHED] {label} s{seed}  test_ce={sp.get('test_ce')}  N={sp.get('n_params')}")
        return {"label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
                "error_kind": None, "seed": int(seed), "cached": True,
                "test_ce": sp.get("test_ce"), "n_params": sp.get("n_params")}

    pred_n, n_gen = predict_n_params(cfg)
    _seed_everything(cfg.seed)
    model = VFEModel(cfg).to(device)
    actual_n = int(sum(p.numel() for p in model.parameters()))
    gap = "" if actual_n == pred_n else f"  (predicted {pred_n:,}; +{actual_n - pred_n:,} small modules)"
    print(f"    {label} s{seed} | K={cfg.embed_dim} h={cfg.n_heads} {cfg.gauge_group} "
          f"n_gen={n_gen} | N={actual_n:,}{gap} | steps={cfg.max_steps}")

    train_loader = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "train", max_tokens=max_tokens)
    val_loader   = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "validation")
    test_loader  = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "test")

    # Order-INDEPENDENT data stream: model build consumed config-dependent RNG, so reseed AFTER it and
    # re-seed each loader's generator so every cell sees the same batch sequence regardless of grid
    # position (per-seed variance is then init/optimization variance, not a data-order artifact).
    _seed_everything(cfg.seed)
    for loader in (train_loader, val_loader, test_loader):
        if getattr(loader, "generator", None) is not None:
            loader.generator.manual_seed(cfg.seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    artifacts = RunArtifacts(run_dir, cfg, model, dataset=dataset, device=device,
                             timestamp=datetime.now().isoformat(timespec="seconds"))
    # Cell provenance the per-run config.json does not carry: which ROUTE / scale knob produced this N.
    artifacts.save_json("scaling_cell.json", {
        "label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
        "overrides": json.loads(json.dumps(cell["overrides"], default=str)),
        "predicted_n_params": pred_n, "n_gen": n_gen, "seed": int(seed),
    })

    val_tpc = _tokens_per_char(dataset, "validation") or 1.0
    test_tpc = _tokens_per_char(dataset, "test") or 1.0
    t0 = time.perf_counter()
    losses = train(model, train_loader, cfg, n_steps=cfg.max_steps,
                   log_interval=cfg.log_interval, eval_interval=cfg.eval_interval,
                   val_loader=val_loader, tokens_per_char=val_tpc, device=device,
                   logger=logger, artifacts=artifacts, generate_samples=False)
    wall = time.perf_counter() - t0
    results = finalize_run(model, artifacts, cfg, test_loader=test_loader, losses=losses,
                           tokens_per_char=test_tpc, device=device, wall_time=wall, logger=logger)
    return {"label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
            "error_kind": None, "seed": int(cfg.seed), "cached": False,
            "test_ce": results.get("test_ce"), "test_ppl": results.get("test_ppl"),
            "n_params": actual_n, "n_gen": n_gen, "wall_time_s": wall}


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# MAIN  (click-to-run; edit CONFIG / ROUTES above)
# =============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if CONFIG["device"] == "auto" else torch.device(CONFIG["device"]))
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    route_names = CONFIG["routes"]
    for name in route_names:
        if name not in ROUTES:
            raise ValueError(f"unknown route {name!r}; choose from {sorted(ROUTES)}")
    seeds = list(CONFIG["seeds"])
    n_cells = sum(len(ROUTES[n]) for n in route_names)
    print(f"\nVFE_3.0 parameter-scaling suite\n  device:  {device}\n  dataset: {CONFIG['dataset']}"
          f"\n  output:  {output_dir}\n  seeds:   {seeds}\n  routes:  {', '.join(route_names)}"
          f"\n  total:   {n_cells} cells x {len(seeds)} seeds = {n_cells * len(seeds)} runs")

    for name in route_names:
        cells = ROUTES[name]
        print(f"\n{'=' * 70}\nROUTE: {name}  ({len(cells)} cells x {len(seeds)} seeds)\n{'=' * 70}")
        for cell in cells:
            for seed in seeds:
                run_dir = output_dir / name / cell["label"] / f"s{seed}"
                try:
                    res = run_cell(cell, run_dir, int(seed), dataset=CONFIG["dataset"], device=device,
                                   max_tokens=CONFIG["max_tokens"], max_steps=CONFIG["max_steps"])
                except Exception as exc:                     # a training crash must not kill the suite
                    logger.exception("route %s / %s s%d crashed", name, cell["label"], seed)
                    res = {"label": cell["label"], "route": name, "error_kind": "train",
                           "error": str(exc), "seed": int(seed), "test_ce": None, "n_params": None}
                finally:
                    _cleanup()
                if res.get("error_kind") is None and not res.get("cached"):
                    print(f"      -> test_ce={res.get('test_ce')}  ppl={res.get('test_ppl')}  "
                          f"({res.get('wall_time_s', 0.0):.0f}s)")

    print(f"\nALL ROUTES COMPLETE. Aggregate + fit + plot with:  python scaling_analysis.py"
          f"  (reads {output_dir}/)")


if __name__ == "__main__":
    main()
