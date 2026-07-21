r"""Click-to-run PARAMETER-scaling experiment runner for the VFE_3.0 transformer.

Scaling experiments are with respect to NUMBER OF PARAMETERS. This runner loops a size grid x a
seed list, training each (size, seed) cell into its own self-contained ``RunArtifacts`` directory and
calling ``finalize_run`` so EVERY point carries the canonical held-out TEST cross-entropy and the
enriched ``scaling_point`` block (n_params, n_gen, active-params-per-token, FLOP proxies, wall-clock).
There is no CLI arg parsing (project policy): edit the ``CONFIG`` dict and the active ``ROUTES`` at the
bottom, then run ``python scaling.py``. Aggregate + fit + plot afterwards with ``scaling_analysis.py``.

WHY A PARAMETER AXIS IS SUBTLE HERE (read before picking a grid). The token-prior pure path has
``mu_embed (V,K)``, ``sigma_log_embed (V,K)``, ``phi_embed (V,n_gen)``, and a scalar
(prior_bank.py), so ``N = 2*V*K + V*n_gen + 1`` with ``V=50257``. A built-in model-channel route
uses its s tables instead of allocating a redundant base pair; an untied decoder adds a second
``2*V*K`` table pair. ``phi_embed = V*n_gen`` usually DOMINATES. ``n_gen`` is set by the gauge group:
for ``block_glk`` it is ``K^2/n_heads`` (so FEWER/larger
blocks = MORE params -- the opposite sign of a standard transformer); ``glk`` is ``K^2``; ``so_k`` is
``K(K-1)/2``; the ``so_n``/``sp_n`` towers decouple ``n_gen`` from ``K`` entirely. Three consequences:
the gauge group / n_heads is a FIRST-CLASS parameter lever; growing ``embed_dim`` moves ``N`` on two
fronts (linear ``2VK`` + quadratic ``V*n_gen``); and ``n_layers`` / ``n_e_steps`` add ZERO parameters.
Full covariance adds packed-lower coordinates only when an s channel or r centroid is active.

The baseline operating point is the self-contained ``config`` dict IN THIS FILE (it is NOT imported
from ``train_vfe3.py`` -- edit ``config`` below to set what every scaling cell trains around). ``BASELINE``
is bound to it right after the dict is built. Each cell overrides only the scale knob(s). Equal-token
budget: ``max_steps`` / ``batch_size`` /
``max_seq_len`` are held fixed across the parameter routes, so ``tokens_seen`` is constant and the fitted
exponent is the equal-data exponent. A missing tokenized cache raises ``FileNotFoundError`` (no
synthetic substitution); build the corpus cache first (see ``vfe3/data``).
"""

import os
if os.environ.get("VFE3_ALLOW_DUPLICATE_OPENMP") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import gc
import hashlib
import json
import logging
import math
import stat
import time
import warnings
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from vfe3.config import VFE3Config, migrate_serialized_config
from vfe3.data.datasets import (
    _tokenizer_tag,
    cache_source_identity,
    make_dataloader,
    tokens_per_char as _tokens_per_char,
)
from vfe3.model.head_mixer import HeadMixer
from vfe3.model.model import VFEModel, build_group
from vfe3.model.prior_bank import get_decode_registration, get_encode_registration
from vfe3.path_utils import portable_path_component_key
from vfe3.run_artifacts import RunArtifacts, _git_code_identity, _write_json_atomic, finalize_run
from vfe3.runtime import seed_everything
from vfe3.train import coverage_lines, train

logger = logging.getLogger("scaling")

_SCALING_CELL_SCHEMA_VERSION = 2
_SCALING_REUSE_DIGEST_FIELD = "reuse_contract_sha256"
_SCALING_OWNER_SCHEMA_VERSION = 1
_SCALING_OWNER_FILENAME = "scaling_cell_owner.json"
_SCALING_SUMMARY_DIGEST_FIELD = "scaling_reuse_contract_sha256"


# =============================================================================
# CLICK-TO-RUN KNOBS  -- edit, then run.
# =============================================================================
CONFIG: Dict[str, Any] = {
    # Which routes to run (keys of ROUTES), in order. See the ROUTE MENU above the ROUTES registry.
    "routes":     ["grow_K"], # "grow_K_GL10","blocks_K48", "grow_K_mup" (seems to be identical)                      
                   #"blocksize", "grow_K", "group", "model_channel", "infer_T", "infer_L"

    # Seeds per cell. Graduated budget is sensible (more seeds at the cheap small end); the simplest
    # honest default is one shared list applied to every cell -- trim/extend per your compute budget.
    "seeds":      [6, 64, 23],

    "device":     "auto",                                   # 'auto' -> CUDA (RTX 5090) else CPU

    # Dataset for every run (NOT a VFE3Config field; the loader seam). Held-out CE is comparable across
    # sizes only within one tokenizer/corpus.
    "dataset":    "wikitext-103",              # "wikitext-103" | "wiki-ja" | "wiki-en" | "wiki-ar"

    # Cap the TRAIN stream for fast scaling passes (validation/test always read in full). None = full.
    "max_tokens": None,

    # Override every run's max_steps (None = use the local `config` max_steps below). HOLD THIS FIXED across the
    # parameter routes for an equal-token budget (so tokens_seen is constant and the exponent is clean).
    "max_steps":  None,

    # Protect verified completed cells with the same effective config and corpus. Prior-code records
    # are skipped with explicit provenance labeling; partial cells are restarted.
    "resume":     True,

    "output_dir": "vfe3_scaling_results",
}

config = dict(
    

    #################################
    #            Training
    #################################
    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*)
    
    embed_dim                 = 20,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 2,
    
    max_seq_len               = 128,                 # N, context length
    
    batch_size                = 64,
    max_steps                 = 15000,
    
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
    sigma_init                = 4,         # constant initial coordinate variance (sigma_log = log of this)
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
    decode_mode               = 'diagonal_chunked',  #"full_chunked", "diagonal_chunked", "expected_likelihood_chunked", "full", "family", "family_chunked" (family/family_chunked: divergence-consistent KL-to-prior decode, use_prior_bank=True)
    encode_mode               = "per_token",   #"per_token_additive"


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
    s_frame_mode              = "tied",       # "tied" | "phi_tilde" (independent model-channel gauge frame)


    omega_retract_mode        = "lie_exp",  # omega_direct group-manifold retraction: 'lie_exp' | 'cayley'
    omega_reflection           = "off",      # omega_direct det<0 seeding: 'off' (det>0 only) | 'init_seed' | 'metropolis'
    omega_metropolis_temperature  = 1.0 ,   # T in the metropolis det-sign accept min(1, exp(-dF/T)); >0
    omega_metropolis_every        = 100,       # cadence in optimizer steps for the metropolis det-sign sweep; >=1

    # (counts train-loop iterations, 1:1 with optimizer steps INCLUDING under grad_accum_steps>1,
    # which chunks intra-step -- see vfe3/train.py::train_step docstring; diverges only when a step's
    # update is dropped by the NaN/Inf skip_step guard, see spec Sec.4)
    omega_compact_storage        = False,     # opt-in compact (V,H,d,d)/(V,d,d) block storage (equal-block groups)
    omega_reorth_every            = 0 ,        # SO-group re-orthogonalization cadence in M-steps (0 = off)
    phi_reflection               = "off",      # phi-path det<0 via R*exp(phi): 'off' (default) | 'init_seed' | 'metropolis'; reuses omega_metropolis_temperature/every




    m_phi_update_mode         = "adamw",      # "adamw" | "pullback_group"
    m_phi_group_trust_radius  = 0.1,          # embedded Frobenius bound on the group factor
    
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
    cg_covariance_mode        = "passthrough",       # CG covariance pushforward: "passthrough" (means-only, pure) |
                                                     # "delta_full" (delta-method sigma_out=sym(J Sigma J^T); needs family="gaussian_full")
    cg_energy_weight          = 0.0,                 # CG moment-energy regularizer (0.0 = OFF; >0 adds once
                                                     # cg_energy_weight*mean_layers(mean_tokens D(q_post||q_pre)); needs use_cg_coupling=True)

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
    lambda_alpha_mode          = "state_dependent_per_coord",  # "constant" | "state_dependent" | "state_dependent_per_coord"
    lambda_h_mode              = "constant",  # "constant" | "state_dependent" (lambda_h*=c0_h/(b0_h+KL); +R_h)
    
    b0                         = 1.0,                 # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0                         = 1.0,                 # state-dependent alpha shape (numerator)
       
    lambda_alpha               = 1,          # constant self-coupling value
    lambda_h                   = 0.25,       # hyper-prior weight lambda_h * mean_i KL(s_i||r) (0 = OFF; >0 creates s/r tables)

    
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

    learnable_kappa_beta      = False,       # learn per-head kappa_beta = exp(log_kappa_beta), init from kappa_beta above
                                             # (t5-exception family; freezes under detach/straight_through E-step)
    learnable_kappa_gamma     = False,       # learn per-head kappa_gamma (trains under any estimator on the scored
                                             # lambda_gamma>0 path; under s_e_step needs an 'unroll' E-step)

    beta_attention_prior      = "causal_alibi_noself",        # "uniform" | "causal" | "alibi" | "causal_alibi" | "windowed" | "causal_windowed" | "t5_relative_bias"
    gamma_attention_prior     = "causal_alibi_noself",        # model-channel prior pi^s_ij (same 7 keys): "uniform" | "causal" | "alibi" | "causal_alibi" | "windowed" | "causal_windowed" | "t5_relative_bias"

    t5_learnable_bias         = False,           # learn the per-bucket T5 bias table b_{i-j} (sanctioned NN exception, default OFF; needs a t5_relative_bias channel)

    precision_weighted_attention = False,        # down-weight high-variance keys: fold detached -log(b0 + tr Sigma_j)
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
    learnable_r               = False,               # un-freeze hyper-prior centroid r (empirical-Bayes)
    s_e_step                  = True,
    
    e_s_mu_lr                 = 0.85,
    e_s_sigma_lr              = 0.1,
    
    #################################
    #    Embedding/Priors M-step 
    #        Learning Rates
    #################################
        
    m_p_mu_lr                 = 0.015,     #0.015
    m_p_sigma_lr              = 0.01,
    m_phi_lr                  = 0.01,

    m_s_phi_lr                = 0.007,         # M-step LR for independent model-channel frame (phi_tilde)
    
    weight_decay              = 0.02,   #0.03
    phi_weight_decay          = 0.03, #0.03
    
    min_lr                    = 0,       # absolute cosine-decay LR floor (0.0 = pure cosine)
    min_lr_frac               = 0.01,    # proportional LR floor, max(min_lr, frac*base); OFF
    
    #################################
    #     Layer Normalization 
    #        and Hand-Off
    #################################
    
    layernorm_affine          = False,
    norm_type_block           = "none",             # "none" | "mahalanobis"
    norm_type_final           = "none",              # "none" | "mahalanobis"
    
    prior_handoff_rho         = 1,                 # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma       = 0.1,                 # sigma damping in [0,1] (0.0 = frozen at embedding)
    
    #################################
    #        Numerical Safety
    #################################
    
    e_mu_q_trust              = None,
    e_sigma_q_trust           = 10.0,
    sigma_max                 = 10.0,
    
    #################################
    #         Misc/Logging
    #################################     
    amp_dtype                 = None,      # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32
        
    log_interval              = 100,       # console log every N steps (0 = off)
    eval_interval             = 1500,      # periodic validation every N steps (0 = off)
    checkpoint_interval       = 15000,     # save a resumable checkpoint every N steps (0 = off)

    generate_figures          = False,     # OFF: skip the heavy-compute figure set at finalize_run (UMAP
                                           # belief-category triptych, model/belief UMAP, belief bank, E-step
                                           # replay, holonomy sampling) + the per-eval attention/gamma heatmaps.
                                           # True re-enables; make_figures.py re-runs them for a trained run.
                                           # The cheap dashboards (loss/val-ppl/holonomy/free-energy) still write.

    use_ema                   = False,     # EMA/Polyak averaging of the trained tables (default OFF = pure
                                           # path: model is the last SGD iterate). ON: eval/best-save/final
                                           # model use the running average s <- ema_decay*s + (1-ema_decay)*theta
    ema_decay                 = 0.95,     # EMA decay in (0,1); only read when use_ema=True

    ############################################################
    #   Tier-1/Tier-2 improvement toggles (2026-07-05)
    #   docs/2026-07-05-improvement-ideas.md -- ALL default OFF
    #   (byte-identical to the pre-toggle build when left as-is)
    ############################################################

    # --- E-step update rule ---
    e_step_update             = "mm_exact",  # "gradient" (pure current path) | "mm_exact" (closed-form MM
                                             # coordinate minimizer at frozen beta: precision fusion in ONE
                                             # iteration, same cost; kernel route only)
    mm_damping                = 0.75,         # mm_exact damping eta in (0,1]; 1.0 = exact minimizer

    # --- randomized-depth E-step (recurrent-depth recipe) ---
    randomize_e_steps         = False,       # training forwards sample T ~ Uniform{e_steps_min..e_steps_max}
    e_steps_min               = 1,
    e_steps_max               = 3,
    e_steps_backprop_last     = 0,           # truncated backprop: no_grad all but the last k iterations (0 = OFF)
    e_step_halt_tol           = None,        # eval halting: break when mean KL(q^t||q^{t-1}) < tol (None = OFF)

    # --- decode / objective ---
    decode_unigram_prior      = False,       # add kappa*log pi_v (corpus unigram, data statistic) to decode logits
    unigram_kappa             = 1.0,         # tempering on log pi_v (1.0 = exact Bayes class prior)

    # decode_mode "expected_likelihood_chunked" is also new: sigma-aware Gaussian-convolution readout
    # log N(mu_q; mu_v, Sigma_q + Sigma_v) - select it above under use_prior_bank=True.
    untie_decode_bank         = False,       # use_prior_bank=True only: decode reads its OWN cloned (V,K) tables
    z_loss_weight             = 0,           # z-loss on the decode partition: w * mean(logsumexp^2) (0 = OFF)
    sigma_weight_decay        = 0.01,           # AdamW decay for log-variance tables (None = inherit weight_decay;
                                             # 0.0 exempts sigma from the unintended log-sigma->0 pull)

    # --- attention / coupling ---
    gamma_as_beta_prior       = True,        # fold DETACHED gamma posterior into beta's prior (h->s->p->q);
                                             # needs lambda_gamma > 0
    gamma_prior_weight        = 0.5,         # mixture weight w in [0,1]: pi = (1-w) softmax(B) + w gamma
    lambda_twohop             = 0.0,         # two-hop coupling F2 = lam2 sum_ik (beta@beta)_ik KL_ik (0 = OFF;
                                             # exact composed transport, effective depth 2 at L=1)
    query_adaptive_tau        = False,       # per-query tau_i = tau_h (1 + c tr_h Sigma_i / d_h), detached
    query_tau_c               = 1.0,         # strength c >= 0 (read only when query_adaptive_tau=True)
    # New attention priors (select above): "causal_noself" / "causal_alibi_noself" mask the E_ii ~ 0
    # self-edge attention sink (diagonal -inf except (0,0)).

    # --- training mechanics ---
    grad_clip                 = 1.0,         # gradient clip: global L2 norm unless grad_clip_per_role; None/0.0 disables
    grad_clip_per_role        = True,        # clip grads per role (mu/sigma/phi) instead of one global norm
                                             # (global is phi-dominated and silently rescales other roles)
    skip_belief_sigma_update  = True,        # skip the belief-channel sigma E-step update (dead-compute ablation
                                             # for linear-decode configs; user asserts sigma has no consumer)

    # --- compute reclamation (exactness-preserving perf; default OFF) ---
    exp_fp64_mode             = "dim",       # "dim" (long-standing: fp64 when block dim >= 20) | "norm" (fp64 only
                                             # when clamped ||M||_F >= exp_fp64_norm_threshold; d_head=25 blocks
                                             # currently run fp64 PERMANENTLY under "dim")
    exp_fp64_norm_threshold   = 15.0,        # "norm" mode threshold
    share_refine_s_transport  = True,        # build the flat transport ONCE per forward, share s-refine + belief
                                             # E-step (+ all layers); valid on flat/e_phi_lr=0/no-rope configs
    compile_pair_kernel       = False,       # torch.compile the closed-form pair kernel (eager fallback + warn)
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
    token-prior route: 2VK grows linearly, phi_embed = V*K^2/n_heads quadratically. n_heads stays equal to the
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


def route_vary_block_fixed_k(
    embed_dim:   int,
    blocks:      List[int],

    *,
    gauge_group:     str                     = "block_glk",
    tag:             Optional[str]           = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    r"""Fixed K, vary the gauge block size GL(``b``): n_heads = K/b, so block_glk n_gen = K^2/n_heads =
    b*K. FEWER/LARGER blocks (bigger b) = MORE params (n_gen up) -- the opposite sign of a standard
    transformer, and the parameter axis here. This is the ``route_blocksize`` idea written in GL(b) terms
    (b=K/n_heads). kl_max = 8*K (constant; K is fixed across the route). The single-block b=K cell (one
    GL(K) frame) drops the head mixer. Each b must divide K; non-divisors are skipped with a warning.

    ``gauge_group`` selects the structure group per cell (default ``block_glk``, the untied per-head
    GL(d_head); pass ``tied_block_glk`` for the tied gauge, n_gen = d_head^2 instead of b*K, under which
    the Schur-commutant head mixer stays exactly equivariant). ``tag`` overrides the internal cell route
    tag (default ``f'blocks_K{embed_dim}'``); give a distinct tag to a variant route (different budget or
    group) so ``scaling_analysis`` keeps its points separate from the base ``blocks_K48`` run."""
    cells: List[Dict[str, Any]] = []
    for b in blocks:
        if b <= 0 or embed_dim % b != 0:
            logger.warning("  [skip] blocks_K%d: block=%d does not divide K=%d", embed_dim, b, embed_dim)
            continue
        h = embed_dim // b
        ov: Dict[str, Any] = {"embed_dim": embed_dim, "n_heads": h, "gauge_group": gauge_group,
                              "kl_max": 8 * embed_dim}
        if gauge_group == "tied_block_glk":
            ov["phi_precond_mode"] = "killing"
        if h < 2:
            ov["use_head_mixer"] = False
        if extra_overrides:
            ov.update(extra_overrides)
        cells.append({"label": f"K{embed_dim}_GL{b}", "route": tag or f"blocks_K{embed_dim}",
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
         "overrides": {"embed_dim": embed_dim, "n_heads": 8, "gauge_group": "tied_block_glk",
                       "phi_precond_mode": "killing", **headless}},
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
    r"""Compare a routed model-channel prior with a single-tier token prior (route D). The built-in
    model-channel path replaces the base prior pair with the s pair, so this route need not grow N by
    2VK unless another independent s use or decoder bank is enabled. The 'token' arm strips the s/r
    tables; the 'model_channel' arm keeps the baseline channel."""
    return [
        {"label": "model_channel", "route": "model_channel", "scale_knob": "model_channel",
         "overrides": {}},   
        
        {"label": "token_prior", "route": "model_channel", "scale_knob": "model_channel",
         "overrides": {"prior_source": "token", "s_e_step": False, "learnable_r": False,
                       "lambda_h": 0.0, "lambda_gamma": 0.0, "gamma_as_beta_prior": False}},
                                        # baseline already runs the channel
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
#     model_channel token-prior vs the routed model-channel capacity (counted from active tables).
#   INFERENCE routes (FLAT N -- plotted on a SEPARATE inference-capacity figure, NEVER on L(N)):
#     infer_T       n_e_steps in {1,2,4,8} at constant params.   infer_L  n_layers in {1,2,4,6}.
#
# To add your own: call a route builder with a new grid and give it a key; add that key to CONFIG["routes"].
ROUTES: Dict[str, List[Dict[str, Any]]] = {
    "grow_K_GL10":   route_grow_k_fixed_block([20, 30, 40, 50, 60, 80, 100], block=10),
    
    "blocks_K48":    route_vary_block_fixed_k(48, [3, 6, 8, 12, 24]),
    # blocks_K48 follow-up (S1 window GL(3)..GL(24)) at the current BASELINE batch_size=64 => 491.52M
    # tokens/run, the MATCHED budget that removes the 2x Chinchilla D-slice confound vs grow_K_GL10.
    # Distinct keys/tags so scaling_analysis keeps these points separate from the 245.76M blocks_K48 run.
    "blocks_K48_2x":      route_vary_block_fixed_k(48, [3, 6, 8, 12, 24], tag="blocks_K48_2x"),
    
    # Arm 3: tied gauge (n_gen = d_head^2 instead of 48*b) at matched budget -- does per-block UNTIED
    # richness drive the S1 curve, or does the tied variant match it at far fewer params? Under the tied
    # gauge the head mixer stays exactly equivariant, so this is also the equivariance-clean arm.
    "blocks_K48_tied_2x": route_vary_block_fixed_k(48, [3, 6, 8, 12, 24],
                                                   gauge_group="tied_block_glk", tag="blocks_K48_tied_2x"),
    # Arm 2a: the non-gauge capacity control. Keep block_glk so n_gen (= 48*b) matches each gauge cell,
    # but encode NON-structurally (encode_mode='per_token_additive' + pos_phi='none'): the learned
    # (V, n_gen) phi table drives an additive mean shift through a frozen readout, and Omega = I. Tests
    # whether raw phi-table capacity, minus the gl(g) structure, reproduces the S1 curve.
    "blocks_K48_ctrl_2x": route_vary_block_fixed_k(
        48, [3, 6, 8, 12, 24], tag="blocks_K48_ctrl_2x",
        extra_overrides={"encode_mode": "per_token_additive", "pos_phi": "none"}),
    
    "grow_K":        route_grow_k([20, 40, 60, 80, 100, 120], n_heads=4),
    "grow_K_mup":    route_grow_k_mup([20, 40, 60, 80, 100], n_heads=4, anchor_k=20),  # F1/EXP-6 (fixed vs muP)
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
    and summing the prior-table sizes per ``PriorBank`` (prior_bank.py). Exact on the pure path and for
    the active head mixer; CG / connection_W / learnable-scalar tables remain outside this sizing model."""
    group = build_group(cfg)
    n_gen = int(group.generators.shape[0])
    V, K = int(cfg.vocab_size), int(cfg.embed_dim)
    encoder = get_encode_registration(cfg.encode_mode)
    decoder = get_decode_registration(cfg.decode_mode if cfg.use_prior_bank else "linear")
    model_channel_route = cfg.prior_source == "model_channel"
    base_mean = not (
        model_channel_route
        and encoder.can_omit_base_mean
        and decoder.can_omit_base_mean
    )
    base_variance = not (
        model_channel_route
        and encoder.can_omit_base_variance
        and decoder.can_omit_base_variance
    )
    n = V * n_gen + 1                                        # phi_embed, decode_log_scale
    n += int(base_mean) * V * K                               # routed token-prior mean table
    n += int(base_variance) * V * K                           # routed token-prior variance table
    if not cfg.use_prior_bank:
        n += V * K                                          # output_proj_weight
        if cfg.decode_bias:
            n += V                                          # output_proj_bias
    elif cfg.untie_decode_bank:
        n += 2 * V * K                                      # independent routed decode tables
    lower = K * (K - 1) // 2 if not cfg.diagonal_covariance else 0
    if cfg.lambda_h > 0.0 or cfg.lambda_gamma > 0.0 or cfg.prior_source == "model_channel" or cfg.s_e_step:
        n += 2 * V * K                                      # s_mu_embed, s_sigma_log_embed
        n += V * lower                                      # full s packed strict-lower table
    if cfg.lambda_h > 0.0 or cfg.s_e_step:
        n += 2 * K                                          # r_mu, r_sigma_log
        n += lower                                          # full r packed strict-lower centroid
    if cfg.pos_phi == "learned":
        n += int(cfg.max_seq_len) * n_gen                   # pos_phi_free
    if cfg.use_head_mixer:
        n += HeadMixer.parameter_count(group.irrep_dims, group.irrep_labels)
    return n, n_gen


# =============================================================================
# LOADERS  -- memoised on the fields that actually change the stream (mirrors ablation.get_loader).
# =============================================================================
_LOADER_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _require_scaling_seed(value: object, field: str = "seed") -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be an exact nonnegative integer, got {value!r}")
    return value


def get_loader(
    dataset:    str,
    seq_len:    int,
    batch_size: int,
    split:      str,

    *,
    data_seed:  Optional[int] = None,
    max_tokens: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> Any:
    r"""Split-aware DataLoader for ``dataset``/``split`` (a missing cache raises ``FileNotFoundError``).

    Memoized on ``(dataset, seq_len, batch_size, split, cap, vocab_size, data_seed)`` so the corpus
    loads once across cells with the same data and sampling contract.
    Only the train stream shuffles / drops the partial last batch; validation/test read the whole split
    in a stable order so the held-out metric is a full-corpus measurement. ``max_tokens`` caps the train
    split only. No synthetic substitution for a missing real corpus."""
    cap = max_tokens if split == "train" else None
    seeded = split == "train" and data_seed is not None
    validated_data_seed = _require_scaling_seed(data_seed, "data_seed") if seeded else None
    key = (dataset, seq_len, batch_size, split, cap, vocab_size,
           validated_data_seed)
    if key not in _LOADER_CACHE:
        generator = torch.Generator().manual_seed(validated_data_seed) if seeded else None
        _LOADER_CACHE[key] = make_dataloader(dataset, split, seq_len, batch_size,
                                             shuffle=(split == "train"), drop_last=(split == "train"),
                                             max_tokens=cap, vocab_size=vocab_size,
                                             generator=generator)
    return _LOADER_CACHE[key]


# =============================================================================
# SINGLE-CELL EXECUTOR  -- one independent (size, seed) run (replicates _run_once's body).
# =============================================================================

def _cell_cfg_dict(overrides: Dict[str, Any], seed: int, max_steps: Optional[int]) -> Dict[str, Any]:
    r"""The exact kwargs a cell's VFE3Config is built from: baseline + overrides + run knobs. Single
    source of truth, shared by ``run_cell`` and the resume staleness check."""
    d = copy.deepcopy(dict(BASELINE))
    d.update(overrides)
    d["seed"] = _require_scaling_seed(seed)
    d["checkpoint_interval"] = 0                             # no per-cell step_N.pt blowup
    d["generate_figures"] = False                           # finalize-time publication probes/figures are off this path
    if max_steps is not None:
        d["max_steps"] = int(max_steps)
    return d


def _current_code_identity() -> Dict[str, object]:
    """Current repository code identity used to validate a persisted scaling cell."""
    return _git_code_identity(Path(__file__).resolve().parent)


def _validated_scaling_code_identity(value: object) -> Dict[str, object]:
    """Return one detached, usable Git identity for a scaling invocation."""
    if not isinstance(value, Mapping):
        raise TypeError("scaling code identity must be a mapping")
    detached = json.loads(json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False, allow_nan=False))
    git_sha = detached.get("git_sha")
    git_dirty = detached.get("git_dirty")
    fingerprint = detached.get("git_dirty_fingerprint")
    if not isinstance(git_sha, str) or not git_sha or type(git_dirty) is not bool:
        raise ValueError("scaling code identity is unavailable")
    if ((git_dirty and (not isinstance(fingerprint, str) or not fingerprint))
            or (not git_dirty and fingerprint is not None)):
        raise ValueError("scaling code identity has an inconsistent dirty-tree fingerprint")
    return detached


_SCALING_SOURCE_SPLITS = ("train", "validation", "test")


def _validated_data_source_identities(value: object) -> Dict[str, Dict[str, object]]:
    """Return a detached, JSON-safe identity for exactly the splits scaling consumes."""
    if not isinstance(value, Mapping) or set(value) != set(_SCALING_SOURCE_SPLITS):
        raise ValueError(
            "scaling data source identities must contain exactly train, validation, and test")
    normalized: Dict[str, Dict[str, object]] = {}
    for split in _SCALING_SOURCE_SPLITS:
        source = value[split]
        if not isinstance(source, Mapping):
            raise TypeError(f"{split} source identity must be a mapping")
        detached = json.loads(json.dumps(
            dict(source), sort_keys=True, ensure_ascii=False, allow_nan=False))
        if (not isinstance(detached, dict)
                or not isinstance(detached.get("format"), str)
                or not detached["format"]
                or type(detached.get("size_bytes")) is not int
                or detached["size_bytes"] < 0
                or not isinstance(detached.get("sha256"), str)
                or not detached["sha256"]):
            raise ValueError(f"{split} source identity is incomplete")
        normalized[split] = detached
    return normalized


def _data_source_identities(dataset: str) -> Dict[str, Dict[str, object]]:
    """Current cached corpus identities for every split consumed by a scaling cell."""
    return _validated_data_source_identities({
        split: cache_source_identity(dataset, split)
        for split in _SCALING_SOURCE_SPLITS
    })


def _loader_data_source_identities(
    train_loader: object,
    val_loader:   object,
    test_loader:  object,

    *,
    dataset:    str,
    max_tokens: Optional[int],
) -> Dict[str, Dict[str, object]]:
    """Return source identities from the immutable loader datasets actually used by one cell."""
    sources: Dict[str, object] = {}
    for split, loader, expected_cap in (
        ("train", train_loader, max_tokens),
        ("validation", val_loader, None),
        ("test", test_loader, None),
    ):
        data_identity = getattr(getattr(loader, "dataset", None), "data_identity", None)
        if (not isinstance(data_identity, Mapping)
                or data_identity.get("schema_version") != 2
                or data_identity.get("dataset") != dataset
                or data_identity.get("split") != split
                or data_identity.get("max_tokens") != expected_cap):
            raise RuntimeError(f"{split} loader data identity is unavailable or mismatched")
        sources[split] = data_identity.get("source")
    return _validated_data_source_identities(sources)


def _canonical_json_sha256(value: object) -> str:
    """SHA-256 of one finite, JSON-safe value under a stable canonical encoding."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scaling_reuse_contract_sha256(cellmeta: Mapping[str, object]) -> str:
    """Digest a scaling-cell reuse contract without its self-referential digest field."""
    contract = dict(cellmeta)
    contract.pop(_SCALING_REUSE_DIGEST_FIELD, None)
    return _canonical_json_sha256(contract)


def _cached_scaling_metrics(summary: Mapping[str, object]) -> Dict[str, object]:
    """Return the internally consistent metric payload served by the cached-cell path."""
    scaling_point = summary.get("scaling_point")
    if not isinstance(scaling_point, Mapping):
        return {}
    required = ("n_params", "test_ce", "test_ppl", "test_bits_per_token", "test_bpc")
    if any(key not in summary or key not in scaling_point for key in required):
        return {}
    if any(summary[key] != scaling_point[key] for key in required):
        return {}
    return {key: scaling_point[key] for key in required}


def _cell_resume_status(
    run_dir: Path,
    cfg:     VFE3Config,
    dataset: str,

    max_tokens: Optional[int] = None,

    *,
    source_identities: Optional[Mapping[str, object]] = None,
    code_identity:     Optional[Mapping[str, object]] = None,
) -> Optional[str]:
    r"""Classify one verified completed cell for restart handling.

    ``current`` means the effective config, corpus, and code identity all match this invocation.
    ``recorded_prior_code`` means the internally bound result has the same effective config and corpus
    but was produced under another code identity; restart protects and skips that completed record,
    while downstream analysis retains the saved provenance and can reject mixed-code aggregation.
    ``None`` means the cell is missing, partial, corrupt, or belongs to another effective experiment.
    Historical configs are compared only after the repository's strict serialized-config migration,
    so retired inert fields do not force a completed cell to be overwritten."""
    if (not (run_dir / "summary.json").exists()
            or not (run_dir / "config.json").exists()
            or (run_dir / "scaling_failure.json").exists()):
        return None
    try:
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        built = json.loads(json.dumps(asdict(cfg), default=str))
    except Exception:
        return None
    if (not isinstance(saved, Mapping) or not isinstance(summary, Mapping)
            or saved.get("dataset") != dataset
            or not isinstance(saved.get("config"), Mapping)):
        return None
    raw_saved_config = dict(saved["config"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            migrated = migrate_serialized_config(
                raw_saved_config,
                source=str(run_dir / "config.json"),
                strict_unknown=True,
            )
        saved_effective = json.loads(json.dumps(asdict(migrated.config), default=str))
    except (TypeError, ValueError):
        return None
    if saved_effective != built:
        return None
    try:
        cellmeta = json.loads((run_dir / "scaling_cell.json").read_text(encoding="utf-8"))
        provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        if not isinstance(cellmeta, Mapping) or not isinstance(provenance, Mapping):
            return None
    except Exception:
        return None
    if cellmeta.get("schema_version") != _SCALING_CELL_SCHEMA_VERSION:
        return None
    if cellmeta.get("dataset") != dataset:
        return None
    try:
        saved_config_sha256 = _canonical_json_sha256(raw_saved_config)
        contract_sha256 = _scaling_reuse_contract_sha256(cellmeta)
    except (TypeError, ValueError):
        return None
    saved_contract_sha256 = cellmeta.get(_SCALING_REUSE_DIGEST_FIELD)
    if (not isinstance(saved_contract_sha256, str)
            or len(saved_contract_sha256) != 64
            or saved_contract_sha256 != contract_sha256
            or summary.get(_SCALING_SUMMARY_DIGEST_FIELD) != saved_contract_sha256
            or cellmeta.get("config_sha256") != saved_config_sha256):
        return None
    if _scaling_result_status(_cached_scaling_metrics(summary)) != "complete":
        return None
    if cellmeta.get("max_tokens", None) != (int(max_tokens) if max_tokens is not None else None):
        return None
    saved_sources = cellmeta.get("data_sources")
    if not isinstance(saved_sources, Mapping):
        return None
    try:
        recorded = _validated_scaling_code_identity(cellmeta.get("code_identity"))
        current_sources = (
            _data_source_identities(dataset)
            if source_identities is None
            else _validated_data_source_identities(source_identities)
        )
        current = (
            _validated_scaling_code_identity(_current_code_identity())
            if code_identity is None
            else _validated_scaling_code_identity(code_identity)
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if dict(saved_sources) != current_sources:
        return None
    if any(provenance.get(key) != value for key, value in recorded.items()):
        return None
    return "current" if recorded == current else "recorded_prior_code"


def _cell_is_current(
    run_dir: Path,
    cfg:     VFE3Config,
    dataset: str,

    max_tokens: Optional[int] = None,

    *,
    source_identities: Optional[Mapping[str, object]] = None,
    code_identity:     Optional[Mapping[str, object]] = None,
) -> bool:
    """Return whether a completed cell matches the current effective experiment and code."""
    return _cell_resume_status(
        run_dir,
        cfg,
        dataset,
        max_tokens=max_tokens,
        source_identities=source_identities,
        code_identity=code_identity,
    ) == "current"


def _path_is_reparse_point(path: Path) -> bool:
    """Return whether ``path`` is a symlink, junction, or other filesystem reparse point."""
    junction_probe = getattr(os.path, "isjunction", lambda _path: False)
    if path.is_symlink() or junction_probe(path):
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _trusted_scaling_output_dir(path: Path) -> Path:
    """Create and return a real output directory reached without any reparse-point ancestor."""
    absolute = Path(os.path.abspath(path))
    existing_chain = [candidate for candidate in reversed(absolute.parents) if candidate.exists()]
    if absolute.exists():
        existing_chain.append(absolute)
    for candidate in existing_chain:
        if _path_is_reparse_point(candidate):
            raise ValueError(f"scaling output path crosses a symlink or junction: {candidate}")
    absolute.mkdir(parents=True, exist_ok=True)
    if _path_is_reparse_point(absolute) or not absolute.is_dir():
        raise ValueError("scaling output path must be a real directory")
    return absolute.resolve(strict=True)


def _trusted_scaling_run_dir(
    output_dir: Path,
    route:      str,
    label:      str,
    seed:       int,
) -> Path:
    """Return one real direct-child cell tree contained by ``output_dir``."""
    seed = _require_scaling_seed(seed)
    root = _trusted_scaling_output_dir(output_dir)
    components = (route, label, f"s{seed}")
    current = root
    for component in components:
        portable_path_component_key(component)
        candidate = current / component
        if candidate.exists() and (
                _path_is_reparse_point(candidate) or not candidate.is_dir()):
            raise ValueError(f"unsafe scaling cell path: {candidate}")
        candidate.mkdir(exist_ok=True)
        if _path_is_reparse_point(candidate) or candidate.resolve(strict=True).parent != current:
            raise ValueError(f"scaling cell escaped its owned output tree: {candidate}")
        current = candidate.resolve(strict=True)
    try:
        current.relative_to(root)
    except ValueError as exc:
        raise ValueError("scaling cell escaped its owned output tree") from exc
    _authorize_scaling_cell(current, route, label, seed)
    return current


def _archive_scaling_design(design_path: Path) -> Optional[Path]:
    """Move a prior invocation manifest into append-only history before creating another."""
    if not design_path.exists():
        if design_path.is_symlink() or _path_is_reparse_point(design_path):
            raise ValueError("scaling design path may not be a symlink or reparse point")
        return None
    if (_path_is_reparse_point(design_path) or not design_path.is_file()
            or design_path.resolve(strict=True).parent != design_path.parent.resolve(strict=True)):
        raise ValueError("scaling design path must be a direct regular file")
    archive_root = design_path.parent / ".invocations"
    if archive_root.exists():
        if _path_is_reparse_point(archive_root) or not archive_root.is_dir():
            raise ValueError("scaling invocation archive must be a real directory")
    else:
        archive_root.mkdir()
    for index in range(1, 1_000_000):
        archived = archive_root / f"scaling_design__invocation_{index:04d}.json"
        if archived.exists() or archived.is_symlink() or _path_is_reparse_point(archived):
            continue
        design_path.rename(archived)
        return archived
    raise RuntimeError("scaling invocation archive exhausted its numeric namespace")


def _scaling_owner_payload(route: str, label: str, seed: int) -> Dict[str, object]:
    """Return the exact identity that authorizes scaling-cell attempt archival."""
    portable_path_component_key(route, field="scaling owner route")
    portable_path_component_key(label, field="scaling owner label")
    return {
        "schema_version": _SCALING_OWNER_SCHEMA_VERSION,
        "route":          route,
        "label":          label,
        "seed":           _require_scaling_seed(seed),
    }


def _scaling_owner_is_exact(
    value:    Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    """Whether a decoded owner has exactly the versioned identity schema and Python types."""
    return bool(
        set(value) == set(expected)
        and type(value.get("schema_version")) is int
        and type(value.get("route")) is str
        and type(value.get("label")) is str
        and type(value.get("seed")) is int
        and value == expected
    )


def _read_regular_scaling_json(path: Path, *, role: str) -> Dict[str, object]:
    """Read a direct regular, non-reparse JSON object without following a redirect."""
    if path.is_symlink() or _path_is_reparse_point(path) or not path.is_file():
        raise ValueError(f"{role} must be a regular non-reparse file")
    try:
        if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
            raise ValueError(f"{role} resolves outside its scaling cell")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must contain a JSON object")
    return dict(value)


def _authorize_scaling_cell(run_dir: Path, route: str, label: str, seed: int) -> Path:
    r"""Validate exact ownership or promote one exact regular legacy cell marker once."""
    expected = _scaling_owner_payload(route, label, seed)
    owner_path = run_dir / _SCALING_OWNER_FILENAME
    children = list(run_dir.iterdir())
    if not children:
        _write_json_atomic(owner_path, expected)
        return owner_path

    owner_present = (
        owner_path.exists()
        or owner_path.is_symlink()
        or _path_is_reparse_point(owner_path)
    )
    if owner_present:
        owner = _read_regular_scaling_json(
            owner_path,
            role="scaling cell ownership sentinel",
        )
        if not _scaling_owner_is_exact(owner, expected):
            raise ValueError(
                "scaling cell ownership sentinel does not match route, label, and seed"
            )
        return owner_path

    legacy_path = run_dir / "scaling_cell.json"
    try:
        legacy = _read_regular_scaling_json(
            legacy_path,
            role="legacy scaling cell ownership marker",
        )
    except ValueError as exc:
        raise ValueError(
            "nonempty scaling cell has no valid ownership sentinel or promotable legacy marker"
        ) from exc
    if (
        legacy.get("route") != route
        or legacy.get("label") != label
        or type(legacy.get("seed")) is not int
        or legacy.get("seed") != seed
    ):
        raise ValueError(
            "legacy scaling cell ownership marker does not match route, label, and seed"
        )
    _write_json_atomic(owner_path, expected)
    return owner_path


def _archive_scaling_attempt(
    run_dir: Path,

    *,
    route: str,
    label: str,
    seed:  int,
) -> Path:
    r"""Move one owned cell attempt intact into its append-only archive.

    A fresh ``RunArtifacts`` instance must never target a populated directory because its writers
    replace canonical filenames. Archiving the entire directory first preserves every runner-owned
    artifact and every user-added file without interpreting or deleting either class.
    """
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    if _path_is_reparse_point(run_dir):
        raise ValueError("scaling cell directory may not be a symlink or junction")
    if not run_dir.is_dir():
        raise ValueError("scaling cell path exists but is not a directory")
    _authorize_scaling_cell(run_dir, route, label, seed)
    archive_root = run_dir.parent / ".attempts"
    if archive_root.exists():
        if _path_is_reparse_point(archive_root) or not archive_root.is_dir():
            raise ValueError("scaling attempt archive must be a real directory")
    else:
        archive_root.mkdir()
    for index in range(1, 1_000_000):
        archived = archive_root / f"s{seed}__attempt_{index:04d}"
        if archived.exists() or archived.is_symlink() or _path_is_reparse_point(archived):
            continue
        run_dir.rename(archived)
        return archived
    raise RuntimeError("scaling attempt archive exhausted its numeric namespace")


def _scaling_cell_has_payload(run_dir: Path) -> bool:
    """Return whether a cell contains anything beyond its ownership sentinel."""
    return run_dir.is_dir() and any(
        path.name != _SCALING_OWNER_FILENAME for path in run_dir.iterdir()
    )


def run_cell(
    cell:       Dict[str, Any],
    run_dir:    Path,
    seed:       int,

    *,
    dataset:            str,
    device:             torch.device,
    max_tokens:         Optional[int]                  = None,
    max_steps:          Optional[int]                  = None,
    source_identities:  Optional[Mapping[str, object]] = None,
    code_identity:      Optional[Mapping[str, object]] = None,
) -> Dict[str, Any]:
    r"""Build a fresh model from baseline+overrides, train it, score the held-out TEST split via
    ``finalize_run``, and return a harvest dict. A cross-field config violation is caught and returned
    as ``error_kind='config'`` (not raised), keeping it distinct from a training crash."""
    seed = _require_scaling_seed(seed)
    invocation_sources = (
        None if source_identities is None
        else _validated_data_source_identities(source_identities)
    )
    invocation_code_identity = (
        None if code_identity is None
        else _validated_scaling_code_identity(code_identity)
    )
    label = cell["label"]
    cfg_dict = _cell_cfg_dict(cell["overrides"], seed, max_steps)
    try:
        cfg = VFE3Config(**cfg_dict)
    except (ValueError, NotImplementedError, TypeError) as exc:
        preserve_existing = _scaling_cell_has_payload(run_dir)
        logger.warning("  [config rejected] %s: %s", label, exc)
        return {"label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
                "error_kind": "config", "error": str(exc), "seed": seed,
                "preserve_existing": preserve_existing,
                "test_ce": None, "test_ppl": None, "test_bits_per_token": None,
                "test_bpc": None, "n_params": None}

    resume_status: Optional[str] = None
    if CONFIG["resume"]:
        if _cell_is_current(
            run_dir, cfg, dataset, max_tokens=max_tokens,
            source_identities=invocation_sources,
            code_identity=invocation_code_identity,
        ):
            resume_status = "current"
        else:
            resume_status = _cell_resume_status(
                run_dir, cfg, dataset, max_tokens=max_tokens,
                source_identities=invocation_sources,
                code_identity=invocation_code_identity,
            )
    if resume_status is not None:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        metrics = _cached_scaling_metrics(summary)
        marker = "CACHED" if resume_status == "current" else "RECORDED"
        suffix = (
            ""
            if resume_status == "current"
            else "  (complete prior-code result preserved; provenance retained)"
        )
        print(f"    [{marker}] {label} s{seed}  test_ce={metrics['test_ce']}  "
              f"N={metrics['n_params']}{suffix}")
        return {"label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
                "error_kind": None, "seed": seed, "cached": True,
                "cache_status": resume_status,
                **metrics}

    if _scaling_cell_has_payload(run_dir):
        archived = _archive_scaling_attempt(
            run_dir,
            route=cell["route"],
            label=label,
            seed=seed,
        )
        print(f"    [ARCHIVED] {label} s{seed} previous attempt preserved at {archived}")
        run_dir.mkdir()
        _authorize_scaling_cell(run_dir, cell["route"], label, seed)

    pred_n, n_gen = predict_n_params(cfg)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    model = VFEModel(cfg).to(device)
    actual_n = int(sum(p.numel() for p in model.parameters()))
    gap = "" if actual_n == pred_n else f"  (predicted {pred_n:,}; +{actual_n - pred_n:,} small modules)"
    print(f"    {label} s{seed} | K={cfg.embed_dim} h={cfg.n_heads} {cfg.gauge_group} "
          f"n_gen={n_gen} | N={actual_n:,}{gap} | steps={cfg.max_steps}")

    train_loader = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "train",
                              data_seed=cfg.seed, max_tokens=max_tokens, vocab_size=cfg.vocab_size)
    val_loader   = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "validation",
                              vocab_size=cfg.vocab_size)
    test_loader  = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "test",
                              vocab_size=cfg.vocab_size)

    if invocation_sources is not None:
        loaded_sources = _loader_data_source_identities(
            train_loader,
            val_loader,
            test_loader,
            dataset=dataset,
            max_tokens=max_tokens,
        )
        if loaded_sources != invocation_sources:
            raise RuntimeError(
                "scaling corpus identity drifted between the invocation snapshot and loader build")
    else:
        # Direct API callers retain the historical behavior: take one fresh source snapshot for this
        # standalone cell. ``main`` always supplies its invocation-owned shared snapshot instead.
        loaded_sources = _data_source_identities(dataset)
    cell_code_identity = (
        _validated_scaling_code_identity(_current_code_identity())
        if invocation_code_identity is None else invocation_code_identity
    )

    # Order-INDEPENDENT data stream: model build consumed config-dependent RNG, so reseed AFTER it and
    # re-seed each loader's generator so every cell sees the same batch sequence regardless of grid
    # position (per-seed variance is then init/optimization variance, not a data-order artifact).
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    for loader in (train_loader, val_loader, test_loader):
        if getattr(loader, "generator", None) is not None:
            loader.generator.manual_seed(cfg.seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    artifacts = RunArtifacts(run_dir, cfg, model, dataset=dataset, device=device,
                             timestamp=datetime.now().isoformat(timespec="seconds"))
    # Cell provenance the per-run config.json does not carry: which ROUTE / scale knob produced this N.
    cellmeta: Dict[str, object] = {
        "schema_version": _SCALING_CELL_SCHEMA_VERSION,
        "label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
        "overrides": json.loads(json.dumps(cell["overrides"], default=str)),
        "predicted_n_params": pred_n, "n_gen": n_gen, "seed": seed,
        "max_tokens": (int(max_tokens) if max_tokens is not None else None),
        "dataset": dataset,
        "config_sha256": _canonical_json_sha256(
            json.loads(json.dumps(asdict(cfg), default=str))),
        "code_identity": cell_code_identity,
        "data_sources": loaded_sources,
    }
    reuse_contract_sha256 = _scaling_reuse_contract_sha256(cellmeta)
    cellmeta[_SCALING_REUSE_DIGEST_FIELD] = reuse_contract_sha256
    artifacts.save_json("scaling_cell.json", cellmeta)

    val_tpc = _tokens_per_char(dataset, "validation")
    test_tpc = _tokens_per_char(dataset, "test")
    t0 = time.perf_counter()
    losses = train(model, train_loader, cfg, n_steps=cfg.max_steps,
                   grad_clip=cfg.grad_clip,
                   log_interval=cfg.log_interval, eval_interval=cfg.eval_interval,
                   val_loader=val_loader, tokens_per_char=val_tpc, device=device,
                   logger=logger, artifacts=artifacts, generate_samples=False)
    wall = time.perf_counter() - t0
    results = finalize_run(
        model,
        artifacts,
        cfg,
        tokens_per_char=test_tpc,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        losses=losses,
        data_seed=cfg.seed,
        max_tokens=max_tokens,
        tokenizer_tag=_tokenizer_tag(dataset),
        device=device,
        wall_time=wall,
        logger=logger,
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise RuntimeError("scaling summary must be a JSON object")
    bound_summary = dict(summary)
    bound_summary[_SCALING_SUMMARY_DIGEST_FIELD] = reuse_contract_sha256
    _write_json_atomic(summary_path, bound_summary)
    return {"label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
            "error_kind": None, "seed": int(cfg.seed), "cached": False,
            "test_ce": results.get("test_ce"), "test_ppl": results.get("test_ppl"),
            "test_bits_per_token": results.get("test_bits_per_token"),
            "test_bpc": results.get("test_bpc"),
            "n_params": actual_n, "n_gen": n_gen, "wall_time_s": wall}


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# MAIN  (click-to-run; edit CONFIG / ROUTES above)
# =============================================================================

def validate_routes() -> None:
    """Construct every declared route arm before any training or output publication begins."""
    errors: List[Tuple[str, str, str]] = []
    for route_name, cells in ROUTES.items():
        for cell in cells:
            try:
                VFE3Config(**_cell_cfg_dict(cell["overrides"], 0, None))
            except (TypeError, ValueError) as exc:
                errors.append((route_name, str(cell["label"]), str(exc)))
    if errors:
        detail = "\n".join(
            f"  route {route!r}, arm {label!r}: {error}"
            for route, label, error in errors
        )
        raise ValueError(f"scaling route construction failed:\n{detail}")


def _validated_scaling_seeds(raw_seeds: object) -> List[int]:
    if not isinstance(raw_seeds, (list, tuple)) or not raw_seeds:
        raise ValueError("CONFIG['seeds'] must be a non-empty list or tuple of unique integers")
    if any(type(seed) is not int for seed in raw_seeds):
        raise ValueError("CONFIG['seeds'] must contain exact integers")
    seeds = list(raw_seeds)
    if any(seed < 0 for seed in seeds):
        raise ValueError("CONFIG['seeds'] must contain nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"CONFIG['seeds'] must be unique, got {seeds!r}")
    return seeds


def _scaling_path_component_key(value: str, *, field: str) -> str:
    """Validate one portable directory name and return its collision key."""
    return portable_path_component_key(value, field=field)


def _validated_scaling_routes(raw_routes: object) -> List[str]:
    if not isinstance(raw_routes, (list, tuple)) or not raw_routes:
        raise ValueError("CONFIG['routes'] must be a non-empty list or tuple of unique route names")
    if any(not isinstance(name, str) or not name for name in raw_routes):
        raise ValueError("CONFIG['routes'] must contain nonempty strings")
    route_names = list(raw_routes)
    if len(set(route_names)) != len(route_names):
        raise ValueError(f"CONFIG['routes'] must be unique, got {route_names!r}")
    route_keys = [
        _scaling_path_component_key(name, field="scaling route name")
        for name in route_names
    ]
    if len(set(route_keys)) != len(route_keys):
        raise ValueError(
            f"CONFIG['routes'] must not contain filesystem aliases, got {route_names!r}"
        )
    for name in route_names:
        if name not in ROUTES:
            raise ValueError(f"unknown route {name!r}; choose from {sorted(ROUTES)}")
        cells = ROUTES[name]
        if not cells:
            raise ValueError(f"route {name!r} must contain at least one scaling cell")
        labels = [cell.get("label") if isinstance(cell, Mapping) else None for cell in cells]
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError(f"route {name!r} cell labels must be nonempty strings")
        if len(set(labels)) != len(labels):
            raise ValueError(f"route {name!r} cell labels must be unique, got {labels!r}")
        label_keys = [
            _scaling_path_component_key(label, field=f"route {name!r} cell label")
            for label in labels
        ]
        if len(set(label_keys)) != len(label_keys):
            raise ValueError(
                f"route {name!r} cell labels must not contain filesystem aliases, "
                f"got {labels!r}"
            )
    return route_names


def _scaling_design(route_names: List[str], seeds: List[int]) -> Dict[str, Any]:
    cells = []
    for route_name in route_names:
        for cell in ROUTES[route_name]:
            for seed in seeds:
                cells.append({
                    "route": route_name,
                    "label": cell["label"],
                    "seed": seed,
                    "scale_knob": cell["scale_knob"],
                    "run_dir": f"{route_name}/{cell['label']}/s{seed}",
                    "status": "pending",
                })
    return {
        "schema_version": 1,
        "routes": list(route_names),
        "seeds": list(seeds),
        "status": "pending",
        "cells": cells,
    }


def _scaling_result_status(result: Mapping[str, Any]) -> str:
    if result.get("error_kind") is not None:
        return "failed"

    def _finite_positive(value: object) -> bool:
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(float(value)) and float(value) > 0.0)

    if type(result.get("n_params")) is not int or result["n_params"] <= 0:
        return "nonfinite"
    required_metrics = ("test_ce", "test_ppl", "test_bits_per_token", "test_bpc")
    if any(key not in result for key in required_metrics):
        return "nonfinite"
    test_ce = result["test_ce"]
    test_ppl = result["test_ppl"]
    test_bits = result["test_bits_per_token"]
    test_bpc = result["test_bpc"]
    if not all(_finite_positive(value) for value in (test_ce, test_ppl, test_bits)):
        return "nonfinite"
    if test_bpc is not None and not _finite_positive(test_bpc):
        return "nonfinite"
    expected_ppl = math.exp(min(float(test_ce), 20.0))
    expected_bits = float(test_ce) / math.log(2.0)
    if (not math.isclose(float(test_ppl), expected_ppl, rel_tol=1e-9, abs_tol=1e-12)
            or not math.isclose(float(test_bits), expected_bits, rel_tol=1e-9, abs_tol=1e-12)):
        return "nonfinite"
    return "complete"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    route_names = _validated_scaling_routes(CONFIG["routes"])
    seeds = _validated_scaling_seeds(CONFIG["seeds"])
    validate_routes()
    try:
        invocation_code_identity = _validated_scaling_code_identity(
            _current_code_identity())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("scaling code identity snapshot failed before execution: %s", exc)
        return 1
    try:
        invocation_sources = _validated_data_source_identities(
            _data_source_identities(CONFIG["dataset"]))
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("scaling source identity snapshot failed before execution: %s", exc)
        return 1
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if CONFIG["device"] == "auto" else torch.device(CONFIG["device"]))
    try:
        output_dir = _trusted_scaling_output_dir(Path(CONFIG["output_dir"]))
    except (OSError, ValueError) as exc:
        logger.error("unsafe scaling output directory: %s", exc)
        return 1

    design = _scaling_design(route_names, seeds)
    design_path = output_dir / "scaling_design.json"
    try:
        archived_design = _archive_scaling_design(design_path)
        if archived_design is not None:
            print(f"  [ARCHIVED] previous scaling design preserved at {archived_design}")
        _write_json_atomic(design_path, design)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("scaling design publication refused to overwrite prior data: %s", exc)
        return 1
    design_cells = {
        (cell["route"], cell["label"], int(cell["seed"])): cell
        for cell in design["cells"]
    }
    incomplete = False
    n_cells = sum(len(ROUTES[n]) for n in route_names)
    print(f"\nVFE_3.0 parameter-scaling suite\n  device:  {device}\n  dataset: {CONFIG['dataset']}"
          f"\n  output:  {output_dir}\n  seeds:   {seeds}\n  routes:  {', '.join(route_names)}"
          f"\n  total:   {n_cells} cells x {len(seeds)} seeds = {n_cells * len(seeds)} runs")

    for name in route_names:
        cells = ROUTES[name]
        print(f"\n{'=' * 70}\nROUTE: {name}  ({len(cells)} cells x {len(seeds)} seeds)\n{'=' * 70}")
        for cell in cells:
            for seed in seeds:
                try:
                    run_dir = _trusted_scaling_run_dir(
                        output_dir, name, cell["label"], int(seed)
                    )
                    res = run_cell(cell, run_dir, int(seed), dataset=CONFIG["dataset"], device=device,
                                   max_tokens=CONFIG["max_tokens"], max_steps=CONFIG["max_steps"],
                                   source_identities=invocation_sources,
                                   code_identity=invocation_code_identity)
                except Exception as exc:                     # a training crash must not kill the suite
                    logger.exception("route %s / %s s%d crashed", name, cell["label"], seed)
                    res = {"label": cell["label"], "route": name, "error_kind": "train",
                           "error": str(exc), "seed": int(seed), "test_ce": None,
                           "test_ppl": None, "test_bits_per_token": None,
                           "test_bpc": None, "n_params": None}
                finally:
                    _cleanup()
                status = _scaling_result_status(res)
                published = dict(res)
                published["status"] = status
                if status == "nonfinite":
                    if published.get("error_kind") is None:
                        published["error_kind"] = "nonfinite"
                    if published.get("error") is None:
                        published["error"] = (
                            "scaling result is incomplete, non-finite, non-positive, or inconsistent")
                persist_result = not res.get("cached") and not res.get("preserve_existing")
                if persist_result:
                    try:
                        run_dir = _trusted_scaling_run_dir(
                            output_dir, name, cell["label"], int(seed)
                        )
                        _write_json_atomic(run_dir / "scaling_result.json", published)
                        if status != "complete":
                            _write_json_atomic(run_dir / "scaling_failure.json", published)
                            incomplete = True
                    except (OSError, ValueError) as exc:
                        published["status"] = "failed"
                        published["error_kind"] = "path"
                        published["error"] = str(exc)
                        status = "failed"
                        incomplete = True
                        logger.error(
                            "refusing per-cell publication outside the owned scaling tree: %s", exc
                        )
                elif status != "complete":
                    incomplete = True
                manifest_cell = design_cells[(name, cell["label"], int(seed))]
                manifest_cell.update({
                    "status": status,
                    "error_kind": published.get("error_kind"),
                    "error": published.get("error"),
                    "cache_status": published.get("cache_status"),
                })
                design["status"] = "incomplete" if incomplete else "running"
                _trusted_scaling_output_dir(output_dir)
                _write_json_atomic(design_path, design)
                if status == "complete" and not res.get("cached"):
                    print(f"      -> test_ce={res.get('test_ce')}  ppl={res.get('test_ppl')}  "
                          f"({res.get('wall_time_s', 0.0):.0f}s)")

    invocation_errors: List[str] = []
    try:
        terminal_sources = _validated_data_source_identities(
            _data_source_identities(CONFIG["dataset"]))
        if terminal_sources != invocation_sources:
            invocation_errors.append(
                "data source identities drifted during the scaling invocation")
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        invocation_errors.append(f"terminal data source identity snapshot failed: {exc}")
    try:
        terminal_code_identity = _validated_scaling_code_identity(_current_code_identity())
        if terminal_code_identity != invocation_code_identity:
            invocation_errors.append(
                "code identity drifted during the scaling invocation")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        invocation_errors.append(f"terminal code identity snapshot failed: {exc}")
    if invocation_errors:
        incomplete = True
        design["error"] = "; ".join(invocation_errors)

    design["status"] = "incomplete" if incomplete else "complete"
    _trusted_scaling_output_dir(output_dir)
    _write_json_atomic(design_path, design)
    if incomplete:
        print(f"\nROUTES INCOMPLETE. Inspect {design_path} and per-cell scaling_failure.json files.")
        return 1
    print(f"\nALL ROUTES COMPLETE. Aggregate + fit + plot with:  python scaling_analysis.py"
          f"  (reads {output_dir}/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
