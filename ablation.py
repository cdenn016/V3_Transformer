r"""Click-to-run hyperparameter ablation/sweep runner for the VFE_3.0 transformer.

Sweeps one (or several) ``VFE3Config`` field(s) around the operating point defined in
``train_vfe3.py``. There is no CLI arg parsing (project policy): edit the ``CONFIG`` dict
at the bottom, pick a sweep, then run ``python ablation.py``.

Two sweep shapes are supported, both declared in the ``SWEEPS`` registry:

  * single-field  -- vary ONE field across an explicit ``values`` list or an arithmetic
    ``range = [start, stop, step]`` (one-at-a-time ablation around the baseline);
  * multi-arm     -- a ``configs`` list of named arms, each a dict of field overrides,
    for categorical comparisons whose arms differ in more than one field (e.g. a
    full-covariance arm that flips ``family`` and the per-coordinate alpha form together).

The baseline is the self-contained ``BASELINE_CONFIG`` dict below -- a full ``VFE3Config`` toggle
set kept deliberately separate from ``train_vfe3.py`` so an ablation can pin its own (fast)
operating point. It tracks train_vfe3.py's K=20 / block_glk point; keep the two in sync by hand
when the training operating point moves. ``kl_max`` is derived as ``8 * embed_dim`` below exactly
as in train_vfe3.py, and ``DATA_SEED`` mirrors ``train_vfe3.DATA_SEED``, so an identical config
trains on the identical batch order and reproduces train_vfe3.py run-for-run. Each run gets a self-contained
``RunArtifacts`` directory (``config.json``, ``metrics.csv``, ``best_model.pt``, figures)
nested under its sweep, plus an ``ablation_result.json`` headline used for resume and the
sweep-level leaderboard.

Model selection here is VALIDATION-ONLY (``best_val_ppl``): the held-out test split is NOT
scored per cell (that would leak the test set into selection and cost a full extra eval per
run). To get the test number for the winning configuration, copy its fields into
``train_vfe3.py`` and run that -- ``train_vfe3.py`` calls ``finalize_run`` for the test eval.

Three guards make this safe for VFE_3.0's strict config surface:

  1. every swept field name is checked against the real ``VFE3Config`` dataclass fields at
     startup, so a typo aborts loudly instead of being silently dropped (which would make
     every run identical and read as "this field does not matter");
  2. the data loader is rebuilt whenever a swept field changes ``dataset`` / ``max_seq_len``
     / ``batch_size`` (a memoised factory keyed on those), so a ``batch_size`` sweep does
     not silently reuse the wrong loader;
  3. a config-construction failure (a cross-field violation caught by
     ``VFE3Config.__post_init__``) is tagged ``error_kind = "config"`` and kept DISTINCT
     from a training crash (``"train"``), so a mis-specified cell is not silently bucketed
     as ``ppl = inf``.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Anaconda + PyTorch each ship a
#   libiomp5md.dll; the duplicate OpenMP init aborts the process (seen with n_e_steps>1). This MUST
#   run before `import torch`. The clean fix is one OpenMP in the env (e.g. `conda install nomkl`);
#   override by exporting KMP_DUPLICATE_LIB_OK yourself. See docs/edits/2026-06-05.

import copy
import csv
import gc
import hashlib
import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from vfe3.config import VFE3Config
from vfe3.data.datasets import make_dataloader, tokens_per_char
from vfe3.metrics import (
    attention_entropy,
    gauge_equivariance_residual,
    guard_saturation,
    head_mixer_gauge_residual,
    rank_one_residual,
)
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts
from vfe3.runtime import seed_everything
from vfe3.train import coverage_lines, evaluate, train
from vfe3.viz.extract import across_layer_belief_trace, attention_entropy_cov_gap, converged_state

logger = logging.getLogger("ablation")


# DATA_SEED (EXP-1 variance floor), mirroring train_vfe3.DATA_SEED: when set to an int, the TRAIN
# loader's shuffle order is fixed to this seed via an explicit generator, INDEPENDENT of the
# per-cell model seed. None keeps the legacy behavior (shuffle drawn from the global RNG, which
# run_single's post-build reseed pins to cfg.seed). Keep this EQUAL to train_vfe3.DATA_SEED so the
# two entry points train on the same batch order and reproduce each other under an identical config.
DATA_SEED: Optional[int] = 3


# =============================================================================
# BASELINE CONFIG  -- self-contained operating point: EVERY VFE3Config toggle.
# =============================================================================


BASELINE_CONFIG: Dict[str, Any] = dict(
    
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
    decode_mode               = 'diagonal_chunked',  #"full_chunked", "diagonal_chunked", "expected_likelihood_chunked", "diagonal_untied", "full"
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
    lambda_alpha_mode          = "state_dependent",  # "constant" | "state_dependent" | "state_dependent_per_coord"
    lambda_h_mode              = "constant",  # "constant" | "state_dependent" (lambda_h*=c0_h/(b0_h+KL); +R_h)
    
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

    learnable_kappa_beta      = False,       # learn per-head kappa_beta = exp(log_kappa_beta), init from kappa_beta above
                                             # (t5-exception family; freezes under detach/straight_through E-step)
    learnable_kappa_gamma     = False,       # learn per-head kappa_gamma (trains under any estimator on the scored
                                             # lambda_gamma>0 path; under s_e_step needs an 'unroll' E-step)

    beta_attention_prior      = "causal_alibi_noself",        # "uniform" | "causal" | "alibi" | "causal_alibi" | "windowed" | "causal_windowed" | "t5_relative_bias"
    gamma_attention_prior     = "causal_alibi_noself",        # model-channel prior pi^s_ij (same 7 keys): "uniform" | "causal" | "alibi" | "causal_alibi" | "windowed" | "causal_windowed" | "t5_relative_bias"

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
    learnable_r               = False,               # un-freeze hyper-prior centroid r (empirical-Bayes)
    s_e_step                  = True,
    
    e_s_mu_lr                 = 0.85,
    e_s_sigma_lr              = 0.1,
    
    #################################
    #    Embedding/Priors M-step 
    #        Learning Rates
    #################################
        
    m_p_mu_lr                 = 0.0125,   
    m_p_sigma_lr              = 0.01,     
    m_phi_lr                  = 0.010,   
    
    weight_decay              = 0.02,
    phi_weight_decay          = 0.05,
    
    min_lr                    = 0,       # absolute cosine-decay LR floor (0.0 = pure cosine)
    min_lr_frac               = 0.01,    # proportional LR floor, max(min_lr, frac*base); OFF
    
    #################################
    #     Layer Normalization 
    #        and Hand-Off
    #################################
    
    layernorm_affine          = False,
    norm_type_block           = "none",             # "none" | "mahalanobis"
    norm_type_final           = "none",              # "none" | "mahalanobis"
    
    prior_handoff_rho         = 0,                 # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma       = 0,                 # sigma damping in [0,1] (0.0 = frozen at embedding)
    
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
    sigma_weight_decay        = None,        # AdamW decay for log-variance tables (None = inherit weight_decay;
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
    grad_clip_per_role        = True,        # clip grads per role (mu/sigma/phi) instead of one global norm
                                             # (global is phi-dominated and silently rescales other roles)
    skip_belief_sigma_update  = True,        # skip the belief-channel sigma E-step update (dead-compute ablation
                                             # for linear-decode configs; user asserts sigma has no consumer)

    # --- compute reclamation (exactness-preserving perf; default OFF) ---
    transport_mean_per_head   = False,       # per-head transport_mean einsum (~n_heads x fewer FLOPs, allclose 1e-6)
    exp_fp64_mode             = "dim",       # "dim" (long-standing: fp64 when block dim >= 20) | "norm" (fp64 only
                                             # when clamped ||M||_F >= exp_fp64_norm_threshold; d_head=25 blocks
                                             # currently run fp64 PERMANENTLY under "dim")
    exp_fp64_norm_threshold   = 15.0,        # "norm" mode threshold
    share_refine_s_transport  = True,        # build the flat transport ONCE per forward, share s-refine + belief
                                             # E-step (+ all layers); valid on flat/e_phi_lr=0/no-rope configs
    compile_pair_kernel       = False,       # torch.compile the closed-form pair kernel (eager fallback + warn)
)

# kl_max is the numerical safety-net clamp on EVERY divergence, scaled with K exactly as in
# train_vfe3.py (``config["kl_max"] = 8 * config["embed_dim"]``) so the two entry points share one
# objective under an identical config -- the K-independent 100.0 dataclass default would silently
# differ from train_vfe3's 160 at K=20. A sweep that changes embed_dim must override kl_max itself
# (the K=28 multi-arm below already sets kl_max=224).
BASELINE_CONFIG["kl_max"] = 8 * BASELINE_CONFIG["embed_dim"]



# =============================================================================
# SWEEP REGISTRY  -- each entry sweeps real VFE3Config field(s); edit freely.
# =============================================================================
# Schema per sweep:
#   description : str                       one-line human summary (printed + plotted)
#   single-field form:
#     param         : str                   the VFE3Config field to vary
#     values        : [v1, v2, ...]   OR    
#     range : [start, stop, step]
#     
#   baseline_value: Any                   the train_vfe3 value (for reference only)
#   multi-arm form:
#     configs       : [{label: str, <field>: <value>, ...}, ...]
#   optional, both forms:
#     requires      : {field: value, ...}   prerequisite overrides merged into EVERY run of
#                                            this sweep BEFORE the swept field, used to keep a
#                                            cross-field constraint satisfied so the cell is a
#                                            clean single-variable comparison rather than a
#                                            config error.

# One sweep per sweepable VFE3Config toggle. `requires` pre-satisfies a cross-field constraint
# so the cell is a clean single-variable comparison rather than a config error; multi-arm
# `configs` is used where arms must differ in several fields at once. The few fields that are NOT
# meaningfully ablatable are listed under NON_SWEPT_FIELDS below.
SWEEPS: Dict[str, Dict[str, Any]] = {

    # === model structure / capacity ========================================
    # NB the embed_dim / n_heads value lists assume the baseline embed_dim=20, n_heads=2
    # (n_heads must divide embed_dim); adjust them if you change those in BASELINE_CONFIG.
    "embed_dim": {
        "description": "total belief dimension K (kept divisible by n_heads=2)",
        "param": "embed_dim", "values": [20, 40, 64],
    },
    
    
    "n_heads": {  # n_heads=1 has no >=2 equal blocks -> disable the head mixer for a clean sweep
        "description": "number of gauge-irrep blocks / heads (divisors of embed_dim=20)",
        "param": "n_heads", "values": [1, 2, 4, 5], "requires": {"use_head_mixer": False},
    },
    
    
    "n_layers": {
        "description": "number of stacked blocks L",
        "param": "n_layers", "values": [1, 2, 3, 5],
    },
    "n_e_steps": {
        "description": "E-step inner iterations T per block",
        "param": "n_e_steps", "values": [1, 2, 3, 5],
    },

    
    
    
    
    # === gauge seam ========================================================
    # use_head_mixer (True at baseline) needs >= 2 equal blocks (block_glk / tied_block_glk) or a labeled irrep tower (so_n/sp_n);
    # the single-block glk / so_k / sp arms turn it off so the model constructs.
    # so_n / sp_n irrep-tower arms (heads = irreps; group_n decoupled from K): irrep_spec dims
    # must sum to the baseline embed_dim=20, and the TIED gauge rejects the per-block phi
    # preconditioners, so these arms override phi_precond_mode (baseline pullback_per_block).
    # Equal-block arms keep the head mixer (kron(A, I_d) IS the Schur commutant of mult copies
    # of one irrep, exactly equivariant under the tied gauge).
    "gauge_group": {
        "description": "gauge group",
        "configs": [
            {"label": "block_glk",      "gauge_group": "block_glk"},
            {"label": "tied_block_glk", "gauge_group": "tied_block_glk"},
            {"label": "glk",            "gauge_group": "glk",  "use_head_mixer": False},
            {"label": "so_k",           "gauge_group": "so_k", "use_head_mixer": False},
            {"label": "sp",             "gauge_group": "sp",   "use_head_mixer": False},
            {"label": "so3_spin2x4",    "gauge_group": "so_n", "group_n": 3,
             "irrep_spec": [("l2", 4)],                       "phi_precond_mode": "killing"},
            #{"label": "so3_tower",      "gauge_group": "so_n", "group_n": 3,
            # "irrep_spec": [("l0", 1), ("l1", 1), ("l3", 1), ("l4", 1)],
            #                                                  "phi_precond_mode": "killing"},
            {"label": "sp4_sym2x2",     "gauge_group": "sp_n", "group_n": 4,
             "irrep_spec": [("sym2", 2)],                     "phi_precond_mode": "killing"},
        ],
    },


    # Phase 3 gave omega_direct s-channel frame-fidelity (the gamma / model-coupling channel now
    # transports the s tables by the stored U, not exp(phi)), so every cell INHERITS BASELINE_CONFIG's
    # gamma-on settings (lambda_gamma / s_e_step / gamma_as_beta_prior) -- an apples-to-apples gamma-on
    # comparison across gauge charts, phi vs omega_direct per gauge_group.
    # One phi baseline plus one omega_direct cell per gauge_group in vfe3.config._OMEGA_GROUPS, reusing
    # the group_n/irrep_spec/phi_precond_mode payloads from the gauge_group arm above (~383-398):
    #   - glk/so_k/sp are single-block groups -> n_heads=1 and use_head_mixer=False so the runtime
    #     block count matches the per-head attention priors.
    #   - block_glk/tied_block_glk keep the baseline's use_head_mixer=True (n_heads=2 is >= 2 equal
    #     blocks); tied_block_glk additionally overrides phi_precond_mode='killing' because the
    #     baseline's 'killing_per_block' generators do not partition per head under the tied gauge
    #     (same override the cache_supported tied_block_glk arm at ~528-533 uses).
    #   - so_n/sp_n towers keep phi_precond_mode='killing' (ambient; the shared generators do not
    #     partition per irrep block either). The so3_spin2x4 tower additionally overrides n_heads=4:
    #     BASELINE_CONFIG's causal_alibi priors build an (n_heads, N, N) bias that must equal the
    #     4-block irrep count (same override the cg_coupling arm at ~556-557 uses for this tower) --
    #     the plain gauge_group arm's so3_spin2x4 cell (~391-392) lacks this override and currently
    #     fails to construct against BASELINE_CONFIG's alibi priors, a pre-existing issue out of scope
    #     here. sp4_sym2x2's 2 blocks already equal n_heads=2, so no override is needed there.
    "gauge_parameterization": {
        "description": "gauge frame chart: phi (exp coords) vs omega_direct (stored GL(K) element), per gauge_group",
        "configs": [
            {"label": "phi",                         "gauge_parameterization": "phi"},

            {"label": "omega_direct_glk",             "gauge_parameterization": "omega_direct",
             "gauge_group": "glk",                    "n_heads": 1, "use_head_mixer": False},

            {"label": "omega_direct_block_glk",       "gauge_parameterization": "omega_direct",
             "gauge_group": "block_glk",               "n_heads": 2},

            {"label": "omega_direct_tied_block_glk",  "gauge_parameterization": "omega_direct",
             "gauge_group": "tied_block_glk",          "n_heads": 2, "phi_precond_mode": "killing"},

            {"label": "omega_direct_so_k",            "gauge_parameterization": "omega_direct",
             "gauge_group": "so_k",                    "n_heads": 1, "use_head_mixer": False},

            {"label": "omega_direct_sp",               "gauge_parameterization": "omega_direct",
             "gauge_group": "sp",                       "n_heads": 1, "use_head_mixer": False},

            {"label": "omega_direct_so3_spin2x4",     "gauge_parameterization": "omega_direct",
             "gauge_group": "so_n", "group_n": 3, "irrep_spec": [("l2", 4)],
             "phi_precond_mode": "killing", "n_heads": 4},

            {"label": "omega_direct_sp4_sym2x2",      "gauge_parameterization": "omega_direct",
             "gauge_group": "sp_n", "group_n": 4, "irrep_spec": [("sym2", 2)],
             "phi_precond_mode": "killing"},
        ],
        "requires": {"transport_mode": "flat", "pos_phi": "none"},
    },


    "transport_mode": {  # regime_ii / regime_ii_covariant are learned connections (sanctioned NN exceptions)
        "description": "connection regime: flat phi-cocycle vs learned non-flat (regime_ii bilinear, regime_ii_covariant Route B)",
        "configs": [
            {"label": "flat",                "transport_mode": "flat"},
            {"label": "regime_ii",           "transport_mode": "regime_ii"},
            {"label": "regime_ii_covariant", "transport_mode": "regime_ii_covariant"},
        ],
    },
    
    
    "cocycle_relaxation": {
        "description": "regime_ii homotopy (0 -> flat, 1 -> fully relaxed)",
        "param": "cocycle_relaxation", "values": [0.0, 0.5, 1.0],
        "requires": {"transport_mode": "regime_ii"},
    },
    
    
    "cross_couplings": {  # the off-block coupling merges the heads into one super-block, so the
                          # >=2-block head mixer cannot apply -> turn it off for a clean comparison
        "description": "cross-head GL(K) coupling (block-diagonal vs one coupled pair)",
        "configs": [
            {"label": "none",     "cross_couplings": None},
            {"label": "pair_0_1", "cross_couplings": [(0, 1)]},
        ],
        "requires": {"use_head_mixer": False},
    },

    # === experiment arms (2026-06-21 readiness build; run by name via CONFIG["sweep"]=...) ====
    # Each declares ``collect_diagnostics: True`` so run_single does one converged-state replay per
    # cell and tacks the gauge/entropy/equivariance scalars onto sweep_results.csv (see
    # _cell_diagnostics). They are registered but kept OUT of SWEEP_ORDER so the default run is
    # unchanged; select one with CONFIG["sweep"] = "gauge_transport" (etc.).

    "multiseed_floor": {  # I1 / EXP-1: the discipline gate -- across-seed variance floor on the baseline.
        # One arm = the production operating point (no overrides), replicated across 5 seeds. The across-
        # seed mean/SD of primary_val_ppl is the noise band every single-seed ablation "win" must clear.
        # The runner reseeds the data order per cell to a FIXED stream, so this SD is init+optimization
        # only -- a LOWER bound on deployment variance. Read the band off sweep_results.csv: the 5 rows
        # share the base label 'baseline' (run dirs baseline__s6 ... baseline__s17), differing in the
        # seed column; mean/SD over them is the floor (or use multiseed_analysis.py for the figure).
        "description": "across-seed variance floor on the baseline operating point [I1/EXP-1]",
        "collect_diagnostics": True,
        "seeds": [6, 64, 23, 3, 17],
        "configs": [
            {"label": "baseline"},
        ],
    },

    "gauge_transport": {  # A1 / EXP-2: the program's central causal claim -- gauge ON vs OFF vs frozen-random.
        # 'off' coerces the frame to identity (phi_scale=0, e/m_phi_lr=0, pos_phi='none' -> Omega=I);
        # 'frozen' keeps a nonzero random frame with the LRs zeroed; 'on' is the learned baseline.
        # use_head_mixer is forced OFF in every arm so the contrast is flat-GL(K)-transport-vs-none
        # (matched param count). The depth arm L in {1,2} is folded in (rank collapse cannot show at L=1).
        "description": "GL(K) gauge transport on/off(Omega=I)/frozen-random, at L in {1,2} [A1/EXP-2]",
        "collect_diagnostics": True,
        "seeds": [6, 64, 23],                                # I1: across-seed error bar on the headline result
        "configs": [
            {"label": "on_L1",     "gauge_transport": "on",     "use_head_mixer": False, "n_layers": 1},
            {"label": "off_L1",    "gauge_transport": "off",    "use_head_mixer": False, "n_layers": 1},
            {"label": "frozen_L1", "gauge_transport": "frozen", "use_head_mixer": False, "n_layers": 1},
            {"label": "on_L2",     "gauge_transport": "on",     "use_head_mixer": False, "n_layers": 2},
            {"label": "off_L2",    "gauge_transport": "off",    "use_head_mixer": False, "n_layers": 2},
            {"label": "frozen_L2", "gauge_transport": "frozen", "use_head_mixer": False, "n_layers": 2},
        ],
    },

    "attention_entropy": {  # C1 / EXP-4: canonical F (entropy term ON) vs entropy-suppressed surrogate.
        # The production kernel never computes the entropy term, so BOTH arms are forced onto the
        # oracle route (oracle_unroll_grad=True) -- this isolates the -tau^{-1} Cov_beta term from the
        # kernel-vs-oracle route. __post_init__ does NOT auto-enable the oracle for entropy=False, so
        # the surrogate arm must set it explicitly.
        # 2x2 entropy x kappa: the companion low-kappa arm (kappa_beta=0.25) guards against a sharp-
        # attention null -- Cov_beta scales with attention diffuseness, so a gap that vanishes at the
        # baseline kappa=1 may still bite at low kappa. cov_gap (the -tau^{-1} Cov_beta magnitude) is
        # collected per cell. Both arms force oracle_unroll_grad=True (the kernel never computes the
        # entropy term; __post_init__ does not auto-enable the oracle for entropy=False).
        "description": "attention-entropy canonical vs surrogate, kappa in {1.0, 0.25}, on the oracle [C1/EXP-4]",
        "collect_diagnostics": True,
        "requires": {"e_step_update": "gradient"},
        "seeds": [6, 64, 23],                                # I1: across-seed error bar (esp. the low-kappa gap)
        "configs": [
            {"label": "canon_k1.0",  "include_attention_entropy": True,  "oracle_unroll_grad": True, "kappa_beta": 1.0},
            {"label": "surr_k1.0",   "include_attention_entropy": False, "oracle_unroll_grad": True, "kappa_beta": 1.0},
            {"label": "canon_k0.25", "include_attention_entropy": True,  "oracle_unroll_grad": True, "kappa_beta": 0.25},
            {"label": "surr_k0.25",  "include_attention_entropy": False, "oracle_unroll_grad": True, "kappa_beta": 0.25},
        ],
    },

    "gauge_equivariance": {  # A2 / EXP-9: exact-equivariant tied gauge vs strictly-broken untied gauge.
        # Under block_glk the per-head gauge is UNTIED so the head mixer breaks equivariance as A
        # drifts from I; tied_block_glk restores it exactly. Full covariance + PriorBank so the
        # builder-break residual (the new head_mixer_gauge_residual, surfaced as builder_resid) is the
        # full-cov certificate -- ~eps for the tied arm, climbing for the untied one.
        "description": "tied (exact) vs untied (head-mixer drift) gauge equivariance, full cov [A2/EXP-9]",
        "collect_diagnostics": True,
        "requires": {"e_step_update": "gradient"},
        # s_e_step=False: the live model-channel E-step is diagonal-only (s/r tables are diagonal by
        # construction) and rejects family='gaussian_full', which EXP-9 requires (the covariance break
        # only shows under the full-cov mixer; the diagonal closed form is equivariant under diagonal
        # gauges). Both arms share it, so the tied-vs-untied contrast stays controlled.
        # decode_mode='full_chunked' pairs the full-covariance family with the KL-to-prior decode so
        # the converged covariance reaches the logits (the baseline 'diagonal_chunked' is rank-incompatible).
        # phi_precond_mode='killing' (ambient) on BOTH arms: the tied gauge's shared kron(I_n, gl(d))
        # generators do not partition per head, so the baseline 'killing_per_block' is undefined there;
        # the ambient Killing metric works for both groups, leaving gauge_group (tied vs untied) the
        # only difference between the two arms.
        # precision_weighted_attention=False on BOTH arms: the reliability bias -log(b0 + tr Sigma_j)
        # uses tr Sigma, which is NOT invariant under the GL(K) congruence Sigma->g Sigma g^T, so with
        # it on even the exact-equivariant tied arm carries a gauge-non-covariant element in its forward
        # operating point. Off makes the tied arm's whole forward gauge-exact (modulo the head mixer being
        # studied), so the gauge-equivariance read is clean. (It does not enter the builder_resid /
        # gauge_resid certificates either way -- those are computed off mu/sigma/omega, not log_prior.)
        "configs": [
            {"label": "untied_block_glk", "gauge_group": "block_glk", "use_head_mixer": True,
             "family": "gaussian_full", "use_prior_bank": True, "decode_mode": "full_chunked",
             "phi_precond_mode": "killing", "s_e_step": False, "precision_weighted_attention": False},
            {"label": "tied_block_glk",   "gauge_group": "tied_block_glk", "use_head_mixer": True,
             "family": "gaussian_full", "use_prior_bank": True, "decode_mode": "full_chunked",
             "phi_precond_mode": "killing", "s_e_step": False, "precision_weighted_attention": False},
            # PARAM-MATCHED CONTROL: the untied arm carries +55.6% params (14.10M vs 9.06M tied) because
            # the per-head gauge gives the head mixer a larger Schur commutant, so the raw tied-vs-untied
            # PPL gap is only an UPPER BOUND on the equivariance tax. This arm widens the exact-equivariant
            # tied model to embed_dim=28 (measured 15.50M params, +10% OVER untied -- a conservative match:
            # tied is handed MORE capacity), with kl_max=8*28=224 to keep the per-K convention. CAVEAT: the
            # match is bought with a larger belief width K=28 (vs 20), so this trades the param confound for
            # a width difference on this arm only; if tied_wide STILL loses to untied, capacity is not the
            # explanation and the equivariance tax is real.
            {"label": "tied_block_glk_wide", "gauge_group": "tied_block_glk", "use_head_mixer": True,
             "family": "gaussian_full", "use_prior_bank": True, "decode_mode": "full_chunked",
             "phi_precond_mode": "killing", "s_e_step": False, "precision_weighted_attention": False,
             "embed_dim": 28, "kl_max": 224},
        ],
    },

    "cg_coupling": {  # A3 / EXP-10: the only exactly-equivariant cross-irrep channel, off vs on.
        # SO(3) l2 x4 isotypic tower (sum of dims = 4*5 = 20 = baseline K); the l2 (x) l2 -> l2 CG
        # path is admissible so the coupling is non-trivial. means-only, zero-init (byte-identical at
        # step 0). e_step_gradient='unroll' so the learned path weights actually train (a 'detach'
        # E-step would freeze them and collapse on==off). Head mixer off to isolate the CG channel.
        "description": "Clebsch-Gordan cross-irrep coupling off vs on (SO(3) l2 x4 tower) [A3/EXP-10]",
        "collect_diagnostics": True,
        # n_heads=4: the tower has 4 irrep blocks (l2 x4), and the baseline ALiBi-family prior builds
        # an (n_heads, N, N) bias that must align with the block/head axis.
        "configs": [
            {"label": "cg_off", "gauge_group": "so_n", "group_n": 3, "irrep_spec": [("l2", 4)],
             "n_heads": 4, "phi_precond_mode": "killing", "use_head_mixer": False,
             "use_cg_coupling": False, "e_step_gradient": "unroll"},
            {"label": "cg_on",  "gauge_group": "so_n", "group_n": 3, "irrep_spec": [("l2", 4)],
             "n_heads": 4, "phi_precond_mode": "killing", "use_head_mixer": False,
             "use_cg_coupling": True, "e_step_gradient": "unroll"},
        ],
    },

    "n_e_steps_em": {  # C2 / EXP-5: structural non-Neal-Hinton EM -- PPL vs n_e_steps, unroll vs
        # straight_through. The straight_through arm removes the deepening-graph confound of unrolling
        # the E-step (its trajectory builds no graph), isolating the inference-iteration effect on PPL.
        # e_phi_lr=0 keeps the gauge preconditioner off the E-step across the sweep. (The F-vs-CE
        # decorrelation half additionally needs a persisted final E-step F/token -- a separate infra
        # gap noted in the readiness doc.)
        "description": "n_e_steps {1,2,3,5,8} x e_step_gradient {unroll, straight_through}, e_phi_lr=0 [C2/EXP-5]",
        "configs": [
            {"label": f"T{t}_{g}", "n_e_steps": t, "e_step_gradient": g, "e_phi_lr": 0.0}
            for t in (1, 2, 3, 5, 8) for g in ("unroll", "straight_through")
        ],
    },

    "gauge_mstep_optim": {  # D1 / EXP-8: gauge M-step optimizer geometry.
        # AdamW-on-phi vs the pullback natural-grad M-step (the exact exp-map metric, which reshapes
        # the step DIRECTION) vs killing (conformal: a direction-preserving effective-LR rescale,
        # cos(nat,grad)=1 -- the control that ISOLATES "reshaped direction" from "rescaled LR").
        # e_phi_lr=0 keeps the preconditioner off the E-step so this measures the M-step only.
        "description": "gauge M-step: AdamW vs pullback natural-grad vs killing conformal [D1/EXP-8]",
        "requires": {"e_phi_lr": 0.0},
        "configs": [
            {"label": "adamw",    "m_phi_natural_grad": False},
            {"label": "pullback", "m_phi_natural_grad": True, "phi_precond_mode": "pullback_per_block"},
            {"label": "killing",  "m_phi_natural_grad": True, "phi_precond_mode": "killing_per_block"},
        ],
    },

    "m_phi_lr_natgrad": {  # D1 / EXP-8: the natural-grad LR-mis-scaling sub-experiment.
        # GaugeNaturalGradAdamW steps phi manually (bypassing Adam's per-coord normalization), so the
        # AdamW-tuned m_phi_lr mis-scales; a log-spaced sweep should place the natural-grad optimum
        # >=2x from the AdamW value. Gated to the pullback natural-grad path.
        "description": "log-spaced m_phi_lr on the pullback natural-grad M-step [D1/EXP-8]",
        "param": "m_phi_lr", "values": [0.0005, 0.0015, 0.005, 0.015, 0.05, 0.15],
        "requires": {"m_phi_natural_grad": True, "phi_precond_mode": "pullback_per_block", "e_phi_lr": 0.0},
    },

    "mass_phi": {  # D1 / EXP-8: the regime knob (NOT phi_weight_decay, which is hard-zeroed under
        # natural-grad). The pullback advantage is predicted to shrink as mass_phi rises (the frame-
        # norm penalty pulls phi toward 0, where ad_phi -> 0 and the pullback metric -> I).
        "description": "mass_phi frame-norm penalty -- natural-grad regime knob [D1/EXP-8]",
        "param": "mass_phi", "values": [0.0, 0.001, 0.01, 0.1],
        "requires": {"m_phi_natural_grad": True, "phi_precond_mode": "pullback_per_block", "e_phi_lr": 0.0},
    },

    "fisher_mu_precond": {  # B3 / EXP-14: Fisher natural-gradient vs raw-Euclidean E-step MEAN preconditioner.
        # nat_mu = Sigma*grad_mu (Fisher, the default/pure mean step) vs the raw Euclidean grad_mu; the
        # SPD sigma retraction is UNCHANGED either way, so this isolates the MEAN arm. e_phi_lr=0 keeps
        # the gauge preconditioner off the E-step. The sigma-arm is out of scope (the affine retraction
        # already whitens by 1/sigma, so a 'raw' sigma step needs a different retraction, not this knob).
        "description": "E-step mean preconditioner: Fisher nat-grad vs raw Euclidean x n_e_steps [B3/EXP-14]",
        "requires": {"e_phi_lr": 0.0},
        "seeds": [6, 64, 23],                                # I1: reseed to test raw_T5 divergence is generic
        "configs": [
            {"label": f"{p}_T{t}", "e_step_mu_precond": p, "n_e_steps": t}
            for p in ("fisher", "raw") for t in (1, 3, 5)
        ],
    },

    # === lower-priority runnable diagnostics (E1/E2/E3/A4; 2026-06-22 audit tail) ===========
    "amp_dtype": {  # E2 / EXP-23: bf16 autocast transport-matmul exposure. decode/CE/SPD/transport are
        # fp32-islanded, so the only genuine bf16 exposure is the upstream transport matmuls; the
        # predicted null (PPL/entropy within noise) certifies bf16 as a safe throughput default.
        "description": "autocast dtype: fp32 vs bf16 vs fp16 (transport-matmul exposure) [E2/EXP-23]",
        "configs": [
            {"label": "fp32", "amp_dtype": None},
            {"label": "bf16", "amp_dtype": "bf16"},
            {"label": "fp16", "amp_dtype": "fp16"},
        ],
    },

    "spd_retract_mode": {  # E1 / EXP-20: SPD chart -- spd_affine (whitens by 1/sigma) vs log_euclidean.
        # sigma_max pinned to the live 10.0 (not the dead BASELINE 1000 that never binds) so the
        # congruence-break ceiling can bind; guard_sigma_ceil_frac (already logged per eval) reads it.
        "description": "SPD retraction chart: spd_affine vs log_euclidean (sigma_max=10) [E1/EXP-20]",
        "configs": [
            {"label": "spd_affine",    "spd_retract_mode": "spd_affine",    "sigma_max": 10.0},
            {"label": "log_euclidean", "spd_retract_mode": "log_euclidean", "sigma_max": 10.0},
        ],
    },

    

    "e_mu_q_trust": {  # E3 / EXP-24: mean trust region (BOX mode) as a stability guard. The endpoint
        # is the NaN/loss-spike rate (nonfinite_frac, already logged per eval), NOT PPL -- it is
        # near-inert at the production embed_dim<=64 / e_q_mu_lr operating point; it binds only at
        # large embed_dim or raised LR. Numeric radii (single-field 'param'/'values') so _plot_one_sweep
        # draws the x-sorted LINE plot like m_phi_lr; the unbounded 'off' (e_mu_q_trust=None) baseline
        # cannot sit on a numeric axis, so run it separately in train_vfe3 if you need the endpoint.
        "description": "E-step mean trust-region radius (box mode); endpoint = nonfinite_frac [E3/EXP-24]",
        "param": "e_mu_q_trust", "values": [1.0, 2.0, 5.0],
        "requires": {"mu_trust_mode": "box"},
    },

    "e_mu_q_trust_ball": {  # E3 / EXP-24, BALL mode: the SAME mean trust region under the 2-norm ball
        # clip (mu_trust_mode='ball') instead of the per-coordinate box. Same numeric radii -> LINE plot;
        # 'requires' pins mu_trust_mode='ball' on every cell (consulted only when e_mu_q_trust is set).
        "description": "E-step mean trust-region radius (ball mode); endpoint = nonfinite_frac [E3/EXP-24]",
        "param": "e_mu_q_trust", "values": [1.0, 2.0, 5.0],
        "requires": {"mu_trust_mode": "ball"},
    },

    "regime_ii": {  # A4 / EXP-15: trained Regime-II connection trainability (flat vs learned connection_W).
        # transport_mode='regime_ii' auto-enables oracle_unroll_grad and creates the learned bilinear
        # connection_W; connection_w_norm + holonomy_deviation are logged per eval -> the holonomy-vs-
        # ||connection|| trainability scatter (_plot_holonomy_trainability). Opt-in, equivariance-
        # breaking at nonzero W (default OFF; user-accepted -- see CLAUDE.md exception (3)).
        "description": "Regime-II connection: flat vs learned; holonomy vs ||connection|| trainability [A4/EXP-15]",
        "requires": {"e_step_update": "gradient"},
        "configs": [
            {"label": "flat",      "transport_mode": "flat"},
            {"label": "regime_ii", "transport_mode": "regime_ii"},
        ],
    },

    "rho_handoff": {  # F2 / EXP-7: prior-anchoring as the FFN-brake substitute against rank collapse.
        # The Dong rank-one residual r(X) of the per-token mean cloud is read off PER LAYER from one
        # deep model per arm (across_layer_belief_trace -> rank_resid_by_layer in each cell's
        # ablation_result.json), so the depth-overlay figure compares decay RATES, not absolute level
        # (the no-anchor control plateaus rather than collapsing to rank one). lambda_alpha_mode is
        # pinned 'constant' on every arm so lambda_alpha is the literal anchor strength (the baseline
        # state-dependent alpha is off); n_layers=4 gives a 4-point depth curve. The anchored arms
        # carry both the previous-layer handoff (rho=1) and the embedding anchor (rho=0); the
        # no-anchor arm zeroes the self-coupling (lambda_alpha=1e-3). The *_ephi pair re-runs with
        # e_phi_lr>0 so the gauge frame is genuinely per-layer-independent (default e_phi_lr=0 freezes
        # Omega across depth).
        "description": "prior-anchoring (lambda_alpha x rho x e_phi_lr) vs rank collapse, r(X) by depth [F2/EXP-7]",
        "collect_diagnostics": True,
        "configs": [
            {"label": "anchor_rho1",      "lambda_alpha": 1.0,  "prior_handoff_rho": 1.0,
             "lambda_alpha_mode": "constant", "n_layers": 4, "e_phi_lr": 0.0},
            {"label": "anchor_rho0",      "lambda_alpha": 1.0,  "prior_handoff_rho": 0.0,
             "lambda_alpha_mode": "constant", "n_layers": 4, "e_phi_lr": 0.0},
            {"label": "noanchor",         "lambda_alpha": 1e-3, "prior_handoff_rho": 0.0,
             "lambda_alpha_mode": "constant", "n_layers": 4, "e_phi_lr": 0.0},
            {"label": "anchor_rho1_ephi", "lambda_alpha": 1.0,  "prior_handoff_rho": 1.0,
             "lambda_alpha_mode": "constant", "n_layers": 4, "e_phi_lr": 0.02},
            {"label": "noanchor_ephi",    "lambda_alpha": 1e-3, "prior_handoff_rho": 0.0,
             "lambda_alpha_mode": "constant", "n_layers": 4, "e_phi_lr": 0.02},
        ],
    },

    "pos_extrapolation": {  # H1 / EXP-13: offset priors extrapolate, absolute (learned/RoPE) do not.
        # Train at max_seq_len, then eval the SAME model at growing N (collect_extrapolation -> the
        # CE-vs-N curve persisted per cell). Offset attention priors (causal_alibi, t5_relative_bias)
        # are functions of |i-j| and rebuild at runtime N; the absolute schemes (pos_phi='learned'
        # table, pos_rotation='rope') do not. t5_max_distance is raised to 512 (>= max eval N) so the
        # T5 arm measures offset extrapolation, not bucket saturation. Each arm isolates ONE positional
        # mechanism (the others off), sharing the causal mask. requires pins ALL arms onto ONE
        # belief-gradient route (oracle_unroll_grad=True, the EXP-4 discipline): the rope arm's
        # decoupled value gauge (rope_on_value=False) routes to the autograd oracle, which would
        # otherwise return a DETACHED tangent at oracle_unroll_grad=False -- a different training route
        # than the kernel-route alibi/t5/learned arms, confounding the CE-vs-N contrast. max_seq_len is
        # pinned to the trained 128 so the eval grows to 4x=512 == t5_max_distance (no T5 bucket
        # saturation; the figure measures offset extrapolation, not the bucket horizon).
        "description": "positional extrapolation: offset (alibi/t5) vs absolute (learned/rope), eval @ growing N [H1/EXP-13]",
        "collect_extrapolation": True,
        "requires": {
            "oracle_unroll_grad": True,
            "max_seq_len":        128,
            "e_step_update":      "gradient",
        },
        "configs": [
            {"label": "alibi",   "beta_attention_prior": "causal_alibi",
             "pos_phi": "none",    "pos_rotation": "none"},
            {"label": "t5",      "beta_attention_prior": "t5_relative_bias", "t5_max_distance": 512,
             "pos_phi": "none",    "pos_rotation": "none"},
            {"label": "learned", "beta_attention_prior": "causal",
             "pos_phi": "learned", "pos_rotation": "none"},
            {"label": "rope",    "beta_attention_prior": "causal",
             "pos_phi": "none",    "pos_rotation": "rope"},
        ],
    },

    # === positional encoding ===============================================
    
    "pos_phi": {
        "description": "BCH positional encoding mode",
        "param": "pos_phi", "values": ["none", "learned", "frozen"],
    },
    

    
    "pos_phi_project_slk": {
        "description": "per-block trace projection (det Omega = 1) on pos_phi",
        "param": "pos_phi_project_slk", "values": [False, True],
        "requires": {"pos_phi": "learned"},
    },
    
    "pos_rotation": {
        "description": "gauge-RoPE positional rotation (means-only) on/off",
        "configs": [
            {"label": "none", "pos_rotation": "none"},
            {"label": "rope", "pos_rotation": "rope"},
        ],
    },
    
    
    "pos_phi_scale": {
        "description": "learned pos_phi table init scale",
        "param": "pos_phi_scale", "range": [0.01, 0.1, 0.01],
        "requires": {"pos_phi": "learned"},
    },    
    
    "rope_base": {
        "description": "RoPE rotary frequency base",
        "param": "rope_base", "values": [10.0, 100.0, 1000.0],
        "requires": {"pos_rotation": "rope"},
    },
    "rope_full_gauge": {  # rotating the covariance sandwich needs full covariance
        "description": "RoPE means-only vs full-gauge (rotates covariance; needs full cov)",
        "configs": [
            {"label": "means_only", "pos_rotation": "rope"},
            {"label": "full_gauge", "pos_rotation": "rope", "rope_full_gauge": True,
                                    "family": "gaussian_full",
                                    "lambda_alpha_mode": "state_dependent"},
        ],
    },

    "use_ema": {
        "description": "EMA/Polyak weight averaging off vs on (eval/best-save/final use the average)",
        "param": "use_ema", "values": [False, True],
    },
    "ema_decay": {
        "description": "EMA decay rate (slower average as decay -> 1); requires use_ema=True",
        "param": "ema_decay", "values": [0.99, 0.999, 0.9995], "requires": {"use_ema": True},
    },

    # === belief family =====================================================
    # The full arm flips family (which derives diagonal_covariance) and moves off the per-coordinate
    # alpha form (diagonal-only), both of which a naive single-field sweep would have rejected.
    "covariance": {
        "description": "belief covariance structure (diagonal vs full Gaussian)",
        "configs": [
            {"label": "diagonal", "family": "gaussian_diagonal"},
            {"label": "full",     "family": "gaussian_full",
                                  "lambda_alpha_mode": "state_dependent"},
        ],
    },

    "e_sigma_q_trust": {
        "description": "E-step SPD retraction trust radius",
        "param": "e_sigma_q_trust", "values": [10.0, 15],
    },

    # === free-energy coupling ==============================================
    
    
    "lambda_h_mode": {
        "description": "self-coupling lambda_h form",
        "param": "lambda_h_mode",
        "values": ["constant", "state_dependent", "state_dependent_per_coord"],
    },
    
    "b0_h": {
        "description": "state-dependent alpha shape b0_h (lambda_h* = c0_h/(b0_h + D))",
        "param": "b0_h", "values": [0.1, 1, 5.0], "requires": {"lambda_h_mode": "state_dependent"},
    },
    "c0_h": {
        "description": "state-dependent alpha shape c0_h (numerator)",
        "param": "c0_h", "values": [0.1, 1.0, 5.0], "requires": {"lambda_h_mode": "state_dependent"},
    },
    
   
    
    
    
    


    "mstep_self_coupling_weight": {
        "description": "M-step self-coupling term alpha_hat * sum_i KL(q_i*||p_i)",
        "param": "mstep_self_coupling_weight", "values": [0.00, 1e-4, 0.001, 0.005, 0.01],
    },
    
    
    "precision_attention_b0": {
        "description": "precision_attention_b0",
        "param": "precision_attention_b0", "values": [1, 1.75, 2, 2.25],
    },
    "precision_attention_per_head": {  # per-key reliability per head (block trace) vs global (all K)
        "description": "precision-weighted attention reliability: global trace vs per-head block trace",
        "param": "precision_attention_per_head", "values": [False, True],
        "requires": {"precision_weighted_attention": True},
    },
    



    
    
    
    
    
    
   

    

    
    # === belief-table init scales (PriorBank) ===============================
    "mu_init_std": {
        "description": "init std of the prior mean table mu_embed ~ N(0, std^2)",
        "param": "mu_init_std", "values": [0.010, 0.04, 0.06, 0.075, 0.1],
    },
    
    "sigma_init": {
        "description": "constant initial coordinate variance of the prior table (>0)",
        "param": "sigma_init", "values": [0.5, 1, 2, 3, 4],
    },
    
    "phi_scale": {
        "description": "init std of the gauge-frame table phi_embed ~ N(0, std^2)",
        "param": "phi_scale", "values": [0.01, 0.03, 0.05, 0.07, 0.09, 0.125],
    },
    

    
    
    "decode_tau": {
        "description": "KL-to-prior decode temperature",
        "param": "decode_tau", "values": [0.007, 0.008, 0.009], "requires": {"use_prior_bank": True},
    },
    
    
    
    
    
    
    
    "lambda_beta": {
        "description": "belief-coupling block weight (1.0 = pure F)",
        "param": "lambda_beta", "values": [0, 0.25, 0.5, 0.75, 1, 2],
    },
    
    
    
    "lambda_gamma": {
        "description": "model-channel coupling weight (>0 creates s tables)",
        "param": "lambda_gamma", "values": [0, 0.25, 0.5, 0.75, 0.85],
    },
    
    
    "lambda_h": {
        "description": "hyper-prior weight lambda_h * mean_i KL(s_i||r) (>0 creates s/r tables)",
        "param": "lambda_h", "values": [0.0, 0.2, 0.25, 0.4, 0.75, 1],
    },
    
    
    
    
    
    "kappa_gamma": {
        "description": "model-channel temperature tau_gamma = kappa_gamma * sqrt(d_head)",
        "param": "kappa_gamma", "values": [0.9, 1, 1.1], 
    },
    
    "kappa_beta": {
       "description": "attention temperature tau = kappa * sqrt(d_head)",
       "param": "kappa_beta", "values": [0.9, 1, 1.1],
    },

    "kappa_beta_per_head": {  # per-head tau_h = kappa_beta[h]*sqrt(d_head); list len MUST == n_heads
        # Lists assume the baseline n_heads=2 on  equal-block group (block_glk/tied_block_glk);
        # single-block groups (glk/so_k/sp) reject a list. Mean held at 1.0 so this isolates the
        # per-head ASYMMETRY from the global-temperature axis the scalar 'kappa_beta' sweep covers;
        # [1.0, 1.0] is the uniform reference and is byte-identical to the scalar kappa_beta=1 baseline.
        "description": "per-head belief attention temperature (asymmetry at fixed mean 1.0, n_heads=2)",
        "configs": [
            {"label": "uniform_1.0",   "kappa_beta": [1.0, 1.0]},
            {"label": "split_0.8_1.2", "kappa_beta": [0.8, 1.2]},
            {"label": "split_1.2_0.8", "kappa_beta": [1.2, 0.8]},
            {"label": "split_0.6_1.4", "kappa_beta": [0.6, 1.4]},
            {"label": "split_1.4_0.6", "kappa_beta": [1.4, 0.6]},
            # geo-mean-tau confound controls (B11-a/EXP-11): tied arms whose scalar kappa equals the
            # GEOMETRIC mean of a dispersed pair, isolating per-head ASYMMETRY from the geo-mean tau
            # shift the arithmetic-mean-1.0 dispersed arms carry. sqrt(0.8*1.2)=0.97980,
            # sqrt(0.6*1.4)=0.91652. They sit at dispersion 0 on the dispersion-vs-PPL figure.
            {"label": "geomean_0.8_1.2", "kappa_beta": [0.97980, 0.97980]},
            {"label": "geomean_0.6_1.4", "kappa_beta": [0.91652, 0.91652]},
        ],
    },
   
  
    
    
    "lambda_alpha": {
        "description": "constant self-coupling value (lambda_alpha_mode=constant)",
        "param": "lambda_alpha", "values": [0.0, 0.25, 0.5, 0.75, 1, 2.5, 5], "requires": {"lambda_alpha_mode": "constant"},
    },
    

    
    

    
    
    
    "renyi_order": {  # B2 / EXP-12: Renyi alpha-attention sweep + the alpha>1 non-PD saturation diagnostic.
        # oracle_unroll_grad MUST be on for a fair divergence-order comparison: renyi_order != 1 routes
        # the autograd oracle, whose default (detached) gradient truncates the through-inference signal
        # to the priors/gauge-frame tables, while renyi_order == 1 uses the always-live analytic kernel.
        # Without this the sweep measures gradient-truncation, not divergence order (it makes alpha != 1
        # spuriously ~2.5x faster AND worse). No-op at renyi_order == 1 (the kernel ignores the toggle).
        # alpha<1 is mass-covering, alpha>1 mode-seeking; for alpha>1 the non-PD blend saturates to
        # kl_max with zero gradient (S27), predicting a non-monotone H(beta)-vs-alpha tail.
        # collect_diagnostics captures attn_entropy (H(beta)) + energy_klmax_frac (the saturation
        # fraction) per cell -> the renyi_saturation figure.
        "description": "Renyi divergence order alpha (both sides of 1) + non-PD saturation diagnostic [B2/EXP-12]",
        "param": "renyi_order", "values": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        "requires": {"oracle_unroll_grad": True},
        "collect_diagnostics": True,
    },
    
    
    
    
    
    
    
    
    "e_s_mu_lr": {
       "description": "E-step natural-gradient step size for mu_s",
       "param": "e_s_mu_lr", "values": [0.6, 0.7, 0.8, 0.9],
    },
    
    
    
    
    "e_q_mu_lr": {
       "description": "E-step natural-gradient step size for mu_q",
       "param": "e_q_mu_lr", "values": [0.5, 0.7, 0.9, 1],
    },
   
    "e_q_sigma_lr": {
       "description": "E-step retraction step size for sigma_q",
       "param": "e_q_sigma_lr", "values": [0, 0.0005, 0.0015],
    },
   
    "e_phi_lr": {
       "description": "E-step gauge-frame step size for phi",
       "param": "e_phi_lr", "values": [0.0, 0.005, 0.01],
    },
    
        
    
   
   
   
   
    "m_p_mu_lr": {
        "description": "M-step LR for the prior-bank means",
        "param": "m_p_mu_lr", "values": [0.005, 0.015, 0.016, 0.017, 0.025],
    },
    
    "m_p_sigma_lr": {
        "description": "M-step LR for the prior-bank variances",
        "param": "m_p_sigma_lr", "values": [0.002, 0.003, 0.0035, 0.004, 0.01],
    },
    
    "m_phi_lr": {
        "description": "M-step LR for the gauge-frame parameters (phi)",
        "param": "m_phi_lr", "values": [0.0075, 0.009, 0.01, 0.0115],
    },
    
    
    
    
    "sigma_max": {  # E1 / EXP-20: does the SPD variance ceiling ever bind? (read guard_sigma_ceil_frac)
        "description": "SPD variance ceiling sigma_max (binding vs slack) [E1/EXP-20]",
        "param": "sigma_max", "values": [20, 30, 40],
    },
    
    
    "weight_decay": {
        "description": "AdamW weight decay",
        "param": "weight_decay", "values": [0.005, 0.015, 0.035, 0.05, 0.075],
    }, 
   
    
    
    "phi_weight_decay":{
        "description": "weight decay on phi",
        "param": "phi_weight_decay", "values": [0.005, 0.015, 0.035, 0.05, 0.075, 0.1],
    },

    "mm_damping": {  # MM-exact E-step damping eta in (0,1]; 'requires' pins e_step_update='mm_exact' so the
        # damped coordinate-minimizer step is actually taken every cell (1.0 = full exact minimizer).
        "description": "MM-exact E-step damping eta in (0,1] (requires mm_exact E-step)",
        "param": "mm_damping", "values": [0.25, 0.5, 0.75, 1.0],
        "requires": {"e_step_update": "mm_exact"},
    },

    "query_tau_c": {  # query-adaptive temperature strength c >= 0 (0 = inert); 'requires' forces
        # query_adaptive_tau=True so tau_i = tau_h (1 + c * tr_h Sigma_i / d_h) is live on every cell.
        "description": "query-adaptive temperature strength c (requires query_adaptive_tau=True)",
        "param": "query_tau_c", "values": [0.5, 1.0, 2.0, 4.0],
        "requires": {"query_adaptive_tau": True},
    },

    "lambda_twohop": {  # two-hop coupling F2 = lam2 sum_ik (beta@beta)_ik KL_ik (0 = OFF = pure canonical F;
        # effective depth 2 at L=1). Numeric values -> _plot_one_sweep draws the x-sorted LINE plot.
        "description": "two-hop coupling weight lambda_twohop (0 = OFF)",
        "param": "lambda_twohop", "values": [0.0, 0.001, 0.005, 0.01],
    },

    "sigma_weight_decay": {  # separate AdamW weight decay for the log-variance tables (None = inherit
        # weight_decay). Numeric radii -> LINE plot; the None (inherit) baseline runs via train_vfe3.
        "description": "AdamW weight decay on the log-variance (sigma) tables",
        "param": "sigma_weight_decay", "values": [0.0, 0.02, 0.05, 0.1],
    },

    "warmup_steps": {  # LR warmup length before the cosine decay (0 = no warmup, straight into cosine).
        "description": "LR warmup steps before cosine decay",
        "param": "warmup_steps", "values": [0, 10, 100, 500, 1000, 5000],
    },

    
}


# Fields deliberately NOT swept, with the reason:
#   vocab_size             fixed by the dataset
#   encode_mode            only 'per_token' is live ('gauge_fixed' is a rejected stub)
#   divergence_family      only 'renyi' is registered (renyi_order is its live knob)
#   seed                   set per run from CONFIG['seed'] (the runner reseeds each cell)
#   max_steps              run length, set via CONFIG['max_steps']
#   log/eval/checkpoint_interval, eval_max_batches   bookkeeping, not model behavior
# (decode_mode='full' is a valid value left OUT of the decode_mode sweep: on the prior-bank path it
#  drives the full-covariance SPD retraction's eigh to non-convergence -- a deferred robust-eigh
#  issue, separate from the now-fixed full-cov KL Cholesky.)
# (gauge_parameterization is swept via the "gauge_parameterization" arm in SWEEPS, not here.)
NON_SWEPT_FIELDS = (
    "vocab_size", "encode_mode", "divergence_family", "seed",
    "max_steps", "log_interval", "eval_interval", "checkpoint_interval", "eval_max_batches",
)


# Which sweeps run (and in what order) when CONFIG["sweep"] is None. This is a CURATED subset of
# the full SWEEPS registry above (every key in SWEEPS is also runnable on its own via
# CONFIG["sweep"]="<name>"); add or remove names to shape a session. Cheap-to-expensive is a good
# ordering for a single GPU. Set CONFIG["list_only"]=True (with sweep=None) to print every sweep.
SWEEP_ORDER: List[str] = [
   
    "mm_damping",
   "query_tau_c",
   "lambda_twohop",
   "sigma_weight_decay",
   "warmup_steps", 
   
  #"gauge_transport",
 # "attention_entropy",
 # "gauge_equivariance",
  #"cg_coupling",
 # "fisher_mu_precond",
  
 # "n_e_steps_em",
 # "gauge_mstep_optim",
  
  #"m_phi_lr_natgrad",   havent run 
 
 # "pos_extrapolation",
 # "rho_handoff",
   
  #"kappa_beta_per_head",
  
  # "precision_attention_b0",
  # "decode_tau",
  # "lambda_alpha",
  
 # "sigma_max",
  "weight_decay",
  "phi_weight_decay",
    
    "mu_init_std",
    "phi_scale",
    "sigma_init", 
   
    "lambda_beta",
    "lambda_gamma",
   "lambda_h",
   # "renyi_order",
   # "lambda_alpha",
   "e_mu_q_trust",
   "e_mu_q_trust_ball",

   "e_s_mu_lr",
   "e_q_mu_lr",
    
   #"e_q_sigma_lr",

    #"pos_phi_scale",  
   
   "m_phi_lr",
   "m_p_mu_lr",
   "m_p_sigma_lr",
   
  # "kappa_beta",
  # "kappa_gamma",   
    
   # "mass_phi",
   # "mstep_self_coupling_weight",

   # --- 2026-07-11 restored/added sweeps ---
   


]


# =============================================================================
# CLICK-TO-RUN KNOBS  -- edit, then run.
# =============================================================================
CONFIG: Dict[str, Any] = {
    # The default run is ONE contiguous flow: train each sweep, then (per sweep) write its CSV,
    # print its analysis table, and save its PPL figure, then the next sweep; after all sweeps,
    # the cross-sweep comparison plot + best-per-sweep summary. A later re-run with a different
    # value list TACKS its new cells onto the existing per-sweep figure (union of cell markers).
    # Set list_only=True to instead just print the sweep registry and exit (no training).
    "list_only":   False,

    # One sweep name, or None -> every sweep in SWEEP_ORDER.
    "sweep":       None,

    # 'auto' picks CUDA when present (the RTX 5090), else CPU.
    "device":      "auto",

    # Dataset for every run in the session (NOT a VFE3Config field; the loader seam).
    #   "wikitext-103" | "wikitext-2" | "wiki-en" | "wiki-ja"
    "dataset":     "wikitext-103",

    # Cap the TRAIN stream for fast sweeps (validation is always read in full). None = full.
    "max_tokens":  None,

    # Override every run's max_steps (None = use the train_vfe3 baseline value).
    "max_steps":   None,

    "seed":        6,

    # Skip cells that already wrote ablation_result.json (idempotent reruns / crash recovery).
    "resume":      True,

    "output_dir":  "vfe3_ablation_results",
}


# =============================================================================
# FIELD VALIDATION  -- guard #1: a typo'd field name aborts loudly.
# =============================================================================
_VFE3_FIELDS = {f.name for f in dataclass_fields(VFE3Config)}


def _swept_field_names(sweep: Dict[str, Any]) -> List[str]:
    r"""Every VFE3Config field a sweep touches: its ``param``/``configs`` keys and ``requires``."""
    names: List[str] = list(sweep.get("requires", {}).keys())
    if "configs" in sweep:
        for arm in sweep["configs"]:
            names.extend(k for k in arm if k != "label")
    elif "param" in sweep:
        names.append(sweep["param"])
    return names


def validate_sweeps(sweep_names: List[str]) -> None:
    r"""Abort with the offending names unless every swept field is a real VFE3Config field.

    VFE3Config(**cfg) would silently ignore an unknown kwarg only if it were dropped first;
    here a bad name would instead raise a TypeError mid-run (or, worse under a dict-merge
    that pre-filtered, vanish and make every cell identical). Catching it once at startup
    turns a subtle "this parameter has no effect" result into an immediate, named error.
    """
    offenders: List[Tuple[str, str]] = []
    for name in sweep_names:
        sweep = SWEEPS[name]
        if "configs" not in sweep and "param" not in sweep:
            raise ValueError(f"sweep {name!r} declares neither 'param'/'values' nor 'configs'")
        for field in _swept_field_names(sweep):
            if field not in _VFE3_FIELDS:
                offenders.append((name, field))
    if offenders:
        lines = "\n".join(f"  sweep {s!r}: {f!r} is not a VFE3Config field" for s, f in offenders)
        raise ValueError(
            "ablation SWEEPS reference field(s) that do not exist on VFE3Config "
            f"(typo? renamed?):\n{lines}"
        )


# =============================================================================
# RUN-CONFIG EXPANSION
# =============================================================================

def _expand_range(spec: List[Union[int, float]]) -> List[Union[int, float]]:
    r"""Expand a ``[start, stop, step]`` range into an explicit inclusive list."""
    if len(spec) != 3:
        raise ValueError(f"'range' must be [start, stop, step], got {spec!r}")
    start, stop, step = spec
    if step == 0:
        raise ValueError("'range' step must be non-zero")
    if start != stop and (stop - start) * step < 0:          # sign-mismatched -> empty sweep otherwise
        raise ValueError(
            f"'range' step sign disagrees with [start, stop] direction: {spec!r}"
        )
    all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in spec)
    values: List[Union[int, float]] = []
    tol = abs(step) * 1e-9
    n = int(round((stop - start) / step))
    for i in range(n + 2):
        v = start + i * step
        if (step > 0 and v > stop + tol) or (step < 0 and v < stop - tol):
            break
        values.append(v if all_int else round(v, 10))
    if not values:
        raise ValueError(f"'range' expanded to no values: {spec!r}")
    return values


def _sweep_values(sweep: Dict[str, Any]) -> List[Any]:
    if "values" in sweep:
        return list(sweep["values"])
    if "range" in sweep:
        return _expand_range(sweep["range"])
    raise KeyError(f"single-field sweep must define 'values' or 'range': {sweep!r}")


def sweep_n_runs(sweep: Dict[str, Any]) -> int:
    n_cells = len(sweep["configs"]) if "configs" in sweep else len(_sweep_values(sweep))
    return n_cells * len(sweep.get("seeds") or [None])         # x seeds when the sweep is multi-seeded


def make_run_overrides(sweep_name: str) -> List[Tuple[str, Dict[str, Any]]]:
    r"""(label, overrides) pairs for a sweep; ``requires`` is folded into every override dict.

    The returned ``overrides`` is the FULL set of field changes for that cell (prerequisites
    first, then the swept field/arm), so the caller merges one dict onto the baseline.
    """
    sweep = SWEEPS[sweep_name]
    requires = sweep.get("requires", {})
    runs: List[Tuple[str, Dict[str, Any]]] = []
    if "configs" in sweep:
        for arm in sweep["configs"]:
            arm = dict(arm)
            label = arm.pop("label")
            runs.append((label, {**requires, **arm}))
    else:
        param = sweep["param"]
        for value in _sweep_values(sweep):
            runs.append((f"{param}={value}", {**requires, param: value}))
    return runs


# =============================================================================
# LOADERS  -- guard #2: memoised on the fields that actually change the stream.
# =============================================================================
_LOADER_CACHE: Dict[Tuple[Any, ...], Any] = {}


def get_loader(
    dataset:     str,
    seq_len:     int,
    batch_size:  int,
    split:       str,

    *,
    max_tokens:  Optional[int] = None,
    vocab_size:  Optional[int] = None,
) -> Any:
    r"""DataLoader for ``dataset``/``split``. A missing cache raises ``FileNotFoundError``.

    Memoised on ``(dataset, seq_len, batch_size, split, cap, vocab_size)`` so runs that do not change
    those reuse one cached loader (the corpus cache loads once), while a sweep over
    ``batch_size`` / ``max_seq_len`` / ``vocab_size`` correctly builds a distinct, matching loader.
    ``max_tokens`` caps only the train split (validation is always full). The loader never
    substitutes synthetic data for a missing real corpus -- that would mislabel synthetic numbers
    as a corpus measurement.
    """
    cap = max_tokens if split == "train" else None
    key = (dataset, seq_len, batch_size, split, cap, vocab_size)
    if key in _LOADER_CACHE:
        return _LOADER_CACHE[key]
    # Split-aware loader semantics, mirroring train_vfe3._select_loader: only the train stream is
    # shuffled and tail-dropped; validation/test read the WHOLE split in a stable order so the
    # held-out PPL is a full-corpus measurement (make_dataloader defaults to the train regime, so
    # the eval flags must be passed explicitly here). The TRAIN shuffle order is fixed to DATA_SEED
    # when set (an explicit generator, as in train_vfe3._select_loader); None -> no generator ->
    # legacy global-RNG shuffle, pinned to cfg.seed by run_single's post-build reseed. The cached
    # generator's state advances across cells; run_single re-pins it to DATA_SEED per cell.
    gen = torch.Generator().manual_seed(int(DATA_SEED)) if (split == "train" and DATA_SEED is not None) else None
    loader = make_dataloader(dataset, split, seq_len, batch_size,
                             shuffle=(split == "train"), drop_last=(split == "train"),
                             max_tokens=cap, vocab_size=vocab_size, generator=gen)
    _LOADER_CACHE[key] = loader
    return loader


# =============================================================================
# SINGLE-RUN EXECUTOR
# =============================================================================

def _cell_cfg_dict(
    overrides:  Dict[str, Any],

    *,
    seed:       int,
    max_steps:  Optional[int] = None,
) -> Dict[str, Any]:
    r"""The exact kwargs dict a cell's VFE3Config is built from (baseline + overrides + run knobs).

    Single source of truth for cell construction, shared by ``run_single`` and the resume
    staleness check so the cached-config comparison is faithful.
    """
    d = copy.deepcopy(BASELINE_CONFIG)
    d.update(overrides)
    d["checkpoint_interval"] = 0                             # no per-cell step_N.pt blowup
    d["seed"] = int(seed)
    if max_steps is not None:
        d["max_steps"] = int(max_steps)
    return d


@torch.no_grad()
def _cell_diagnostics(
    model:   VFEModel,
    cfg:     VFE3Config,
    loader:  Any,
    device:  torch.device,
) -> Dict[str, Any]:
    r"""Per-cell converged-state diagnostics for the gauge / entropy / equivariance experiments.

    OPT-IN: only runs when a sweep declares ``collect_diagnostics: True``, so the default tuning
    sweeps pay nothing. One faithful converged-state replay of validation sequence 0
    (``viz.extract.converged_state``) feeds the scalars the experiments read out of
    ``sweep_results.csv``:

      * ``attn_entropy``        mean attention-row entropy H(beta)                       [C1/EXP-4]
      * ``omega_identity_dev``  max |Omega_ij - I| over off-diagonal pairs (confirms     [A1/EXP-2]
                                gauge_transport='off' gives Omega = I to float eps)
      * ``builder_resid``       median builder-break gauge residual of the head mixer    [A2/EXP-9]
                                (eps under the tied gauge, grows under the untied one)
      * ``gauge_resid_in/out``  median in-/out-of-group congruence residual of the       [A1/EXP-2,
                                converged attention operator                              A3/EXP-10]

    Each probe is isolated: a failure (e.g. OOM on the O(N^2 K^2) Omega at large K) is logged and
    drops that scalar, never crashing the cell. The dominant cost is building Omega inside
    ``converged_state``; the experiment sweeps run at the baseline K=20, where it is cheap.
    """
    out: Dict[str, Any] = {}
    try:
        batch = next(iter(loader))
        token_ids = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        cstate = converged_state(model, token_ids)
    except Exception as exc:                                  # converged-state replay failed wholesale
        logger.warning("  [diagnostics: converged_state failed] %s", exc)
        return out

    def _probe(name: str, fn) -> None:
        try:
            out[name] = float(fn())
        except Exception as exc:
            logger.warning("  [diagnostics: %s skipped] %s", name, exc)

    _probe("attn_entropy", lambda: attention_entropy(cstate["beta"]))

    def _omega_identity_dev() -> torch.Tensor:
        omega = cstate["omega"]                               # (N, N, K, K)
        n, k = omega.shape[0], omega.shape[-1]
        off = ~torch.eye(n, dtype=torch.bool, device=omega.device)
        eye = torch.eye(k, device=omega.device, dtype=omega.dtype)
        return (omega[off] - eye).abs().max()
    _probe("omega_identity_dev", _omega_identity_dev)

    # Pass the diagonal/full flag explicitly (cfg.diagonal_covariance) rather than letting the
    # metrics infer it from shape -- a diagonal sigma is (N, K), which is ambiguous with a full
    # (N, K, K) whenever the sequence length N happens to equal K.
    diag = cfg.diagonal_covariance
    if model.head_mixer is not None:
        def _builder_resid() -> torch.Tensor:
            r = head_mixer_gauge_residual(cstate["mu"], cstate["sigma"], model.head_mixer, model.group,
                                          diagonal=diag)
            return torch.cat([r["mu_residual"], r["sigma_residual"]]).median()
        _probe("builder_resid", _builder_resid)

    try:
        res = gauge_equivariance_residual(
            cstate["mu"], cstate["sigma"], cstate["omega"], model.group,
            kappa=cfg.kappa_beta, renyi_order=cfg.renyi_order, kl_max=cfg.kl_max,
            eps=cfg.eps, divergence_family=cfg.divergence_family, diagonal=diag)
        out["gauge_resid_in"]  = float(res["energy_in_group"].median())
        out["gauge_resid_out"] = float(res["energy_out_group"].median())
    except Exception as exc:
        logger.warning("  [diagnostics: gauge_resid skipped] %s", exc)

    # Dong rank-one residual r(X) of the converged per-token mean cloud (F2/EXP-7). The final-layer
    # scalar is a CSV column (a cheap rank-collapse readout for every diagnostics sweep); the full
    # per-layer curve is one extra deep replay and rides in the cell JSON only (not CSV) for the
    # rank-residual depth-overlay figure (_plot_rank_collapse).
    _probe("rank_resid", lambda: rank_one_residual(cstate["mu"]))
    try:
        curve = across_layer_belief_trace(model, token_ids)["rank_one_residual"]
        out["rank_resid_by_layer"] = [float(x) for x in curve]
    except Exception as exc:
        logger.warning("  [diagnostics: rank_resid_by_layer skipped] %s", exc)

    # -tau^{-1} Cov_beta(E, dE) attention-entropy gradient gap (C1/EXP-4): the magnitude of the
    # belief gradient the canonical entropy term adds over the surrogate, on the converged belief.
    # Gated to L=1: the extractor builds the single-block operating point, so it would report a
    # misleading layer-1 value on the L>1 cells of other sweeps (gauge_transport L2, rho_handoff L4).
    if cfg.n_layers == 1:
        _probe("cov_gap", lambda: attention_entropy_cov_gap(model, token_ids)["cov_gap"])

    # kl_max saturation fraction of the converged pairwise energy (B2/EXP-12): for Renyi alpha>1 the
    # non-PD blend pins E_ij at kl_max with zero gradient, so this fraction climbs in the alpha>1 tail
    # and explains a non-monotone H(beta)-vs-alpha curve.
    _probe("energy_klmax_frac", lambda: guard_saturation(
        cstate["sigma"], cstate["energy"], cstate["self_div"], diagonal=diag,
        eps=cfg.eps, sigma_max=cfg.sigma_max, kl_max=cfg.kl_max)["energy_klmax_frac"])
    return out


@torch.no_grad()
def _eval_at_growing_n(model: Any, cfg: VFE3Config, dataset: str, device: torch.device) -> List[Dict[str, Any]]:
    r"""H1/EXP-13: held-out CE at sequence lengths from the trained ``max_seq_len`` outward.

    Re-windows the validation split at each N (anchor ``max_seq_len``, then 1.5/2/3/4x) via
    ``get_loader`` and scores the SAME trained model. Offset attention priors (alibi / t5, functions
    of |i-j|) and the causal mask rebuild at runtime N and extrapolate; the absolute schemes
    (``pos_phi='learned'`` table -- now clamped past the table; RoPE) degrade. Each N is isolated: a
    too-short split or any failure drops that point, never the cell. Returns ``[{n, ce, ppl}, ...]``."""
    base = int(cfg.max_seq_len)
    n_list = sorted({base} | {int(round(base * m)) for m in (1.5, 2.0, 3.0, 4.0)})
    out: List[Dict[str, Any]] = []
    for n in n_list:
        try:
            loader = get_loader(dataset, n, cfg.batch_size, "validation",
                                vocab_size=cfg.vocab_size)
            m = evaluate(model, loader, device=device)
            out.append({"n": n, "ce": float(m["ce"]), "ppl": float(m["ppl"])})
        except Exception as exc:                              # short split / OOM at large N -> drop point
            logger.warning("  [extrapolation eval N=%d skipped] %s", n, exc)
    return out


def run_single(
    label:       str,
    overrides:   Dict[str, Any],
    run_dir:     Path,

    *,
    dataset:             str,
    device:              torch.device,
    seed:                int,
    collect_diagnostics:   bool        = False,
    collect_extrapolation: bool        = False,
    max_tokens:          Optional[int] = None,
    max_steps:           Optional[int] = None,
) -> Dict[str, Any]:
    r"""Build a fresh model from baseline+overrides, train it, and score validation.

    Returns a headline dict with ``primary_val_ppl`` (= min of any periodic best and the
    final validation PPL) and bookkeeping. A cross-field config violation is caught and
    returned as ``error_kind = "config"`` (not raised), keeping it distinct from a training
    crash; the headline is ``inf`` either way so it sorts to the bottom of the leaderboard.
    """
    cfg_dict = _cell_cfg_dict(overrides, seed=seed, max_steps=max_steps)
    try:
        cfg = VFE3Config(**cfg_dict)
    except (ValueError, NotImplementedError, TypeError) as exc:
        logger.warning("  [config rejected] %s: %s", label, exc)
        return {"label": label, "error_kind": "config", "error": str(exc),
                "primary_val_ppl": float("inf"), "seed": int(seed),
                "overrides": _jsonable(overrides)}

    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    model = VFEModel(cfg).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))

    train_loader = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "train",
                              max_tokens=max_tokens, vocab_size=cfg.vocab_size)
    val_loader   = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "validation",
                              vocab_size=cfg.vocab_size)

    # Bits-per-CHARACTER correction for the val BPC, mirroring train_vfe3 (None -- synthetic / no
    # tiktoken / cache absent -- keeps 1.0 = honest bits-per-token). Memoized, so cells share it.
    val_tpc = tokens_per_char(dataset, "validation") or 1.0

    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(run_dir, cfg, model, dataset=dataset, device=device)

    # Reproducible, ORDER-INDEPENDENT data stream. Model construction above consumes a
    # config-dependent amount of RNG, and a cached loader's shuffle (its own generator, or the
    # global RNG for make_dataloader) otherwise advances across runs -- so without this reseed
    # the same config would see different batches depending on its position in the sweep, and
    # the comparison would be confounded by data order. Reseeding here, after the model is built,
    # pins every cell to the same batch sequence regardless of order.
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    for loader, is_train in ((train_loader, True), (val_loader, False)):  # synthetic loaders carry their own generator
        if getattr(loader, "generator", None) is not None:
            # The train loader's DATA_SEED generator (get_loader) is re-pinned to DATA_SEED, not
            # cfg.seed: its state advanced across cells (memoised loader), and train_vfe3 hands
            # each run a FRESH generator at DATA_SEED -- reseeding here reproduces that exactly.
            loader.generator.manual_seed(int(DATA_SEED) if (is_train and DATA_SEED is not None) else cfg.seed)

    print(f"    K={cfg.embed_dim} heads={len(model.group.irrep_dims)} group={cfg.gauge_group} "
          f"family={cfg.family} | steps={cfg.max_steps} batch={cfg.batch_size} | {n_params:,} params")
    for _cov in coverage_lines(train_loader, cfg.max_steps, dataset):
        print(f"   {_cov}")

    losses = train(
        model, train_loader, cfg,
        n_steps=cfg.max_steps,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=val_loader,
        tokens_per_char=val_tpc,
        device=device,
        logger=logger,
        artifacts=artifacts,
        generate_samples=False,                              # pure silent path: no sample text
    )

    # Unconditional final validation pass: guarantees a number even when max_steps is below
    # eval_interval (a periodic eval never fired). best_val_ppl is the lowest the periodic
    # eval saw (inf if none); the headline takes the better of the two.
    m = evaluate(model, val_loader, tokens_per_char=val_tpc, device=device)
    best = artifacts.best_val_ppl
    primary = min(best, m["ppl"]) if best != float("inf") else m["ppl"]

    result = {
        "label":            label,
        "error_kind":       None,
        "primary_val_ppl":  float(primary),
        "final_val_ppl":    float(m["ppl"]),
        "final_val_ce":     float(m["ce"]),
        "final_val_bpc":    float(m["bpc"]),
        "best_val_ppl":     (float(best) if best != float("inf") else None),
        "final_train_loss": (float(losses[-1]) if losses else None),
        "n_params":         n_params,
        "seed":             int(cfg.seed),
        "overrides":        _jsonable(overrides),
        "max_tokens":       (int(max_tokens) if max_tokens is not None else None),
    }
    if collect_diagnostics:                                  # opt-in converged-state probes (S2)
        result.update(_cell_diagnostics(model, cfg, val_loader, device))
    if collect_extrapolation:                                # opt-in growing-N eval (H1/EXP-13)
        result["extrap_ce"] = _eval_at_growing_n(model, cfg, dataset, device)
    return result


def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    r"""Coerce override values (e.g. tuples in cross_couplings) to JSON-friendly forms."""
    return json.loads(json.dumps(d, default=str))


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# SWEEP DRIVER
# =============================================================================
_CSV_COLUMNS = [
    "sweep", "label", "error_kind", "primary_val_ppl", "final_val_ppl",
    "final_val_ce", "final_val_bpc", "best_val_ppl", "final_train_loss",
    "n_params",
    # opt-in per-cell converged-state diagnostics (S2; empty unless the sweep sets collect_diagnostics)
    "attn_entropy", "omega_identity_dev", "builder_resid", "gauge_resid_in", "gauge_resid_out",
    "rank_resid", "cov_gap", "energy_klmax_frac",
    "wall_time_s", "seed", "error",
]

_DIAGNOSTIC_RESULT_KEYS = {
    "attn_entropy", "omega_identity_dev", "builder_resid", "gauge_resid_in", "gauge_resid_out",
    "rank_resid", "rank_resid_by_layer", "cov_gap", "energy_klmax_frac",
}


def _cell_is_current(
    run_dir:    Path,
    overrides:  Dict[str, Any],

    *,
    seed:                  int,
    dataset:               str,
    collect_diagnostics:   bool = False,
    collect_extrapolation: bool = False,
    max_steps:             Optional[int] = None,
    max_tokens:            Optional[int] = None,
) -> bool:
    r"""True iff a completed cell's persisted config.json matches the config we would build now.

    Guards resume against baseline drift: ``ablation_result.json`` is keyed only by the
    ``param=value`` label, which does NOT encode the imported ``train_vfe3`` baseline. Editing
    an unrelated baseline field (e.g. ``embed_dim``) would otherwise let a stale result be
    served as current. A cell is skipped only when its saved VFE3Config equals the freshly
    built one (config-error cells have no config.json, so they are always re-run -- cheap).

    The session ``dataset`` is NOT a VFE3Config field (it is the loader seam), so it is compared
    separately against the persisted top-level ``config.json["dataset"]`` -- otherwise a rerun on a
    DIFFERENT dataset would serve the wrong-dataset cell as current (the VFE3Config would match).

    ``max_tokens`` (the loader train-split token cap) is likewise a loader seam, not a VFE3Config
    field, so it never lands in config.json: a smoke cell capped at 10k tokens and a later full run
    would otherwise compare byte-identical. It is persisted in the ``ablation_result.json`` marker
    and compared here (a marker missing the key reads as None -> a capped re-run fails closed).

    A current marker must also record successful completion with a finite terminal validation PPL.
    Requested diagnostic and extrapolation collections are cache requirements: the marker must record
    the corresponding request flag and contain its output, so enabling either collection cannot reuse
    a headline-only cell.
    """
    cj = run_dir / "config.json"
    if not cj.exists():
        return False
    try:
        built = json.loads(json.dumps(asdict(VFE3Config(
            **_cell_cfg_dict(overrides, seed=seed, max_steps=max_steps))), default=str))
        saved_obj = json.loads(cj.read_text(encoding="utf-8"))
    except Exception:                                        # unbuildable now / unreadable -> re-run
        return False
    if saved_obj.get("dataset") != dataset:                  # session dataset changed -> re-run
        return False
    if saved_obj.get("config") != built:
        return False
    try:
        marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8"))
    except Exception:                                        # no/unreadable marker -> re-run
        return False
    if not isinstance(marker, dict):                         # parseable but incomplete marker -> re-run
        return False
    cur = int(max_tokens) if max_tokens is not None else None
    if marker.get("max_tokens", None) != cur:
        return False
    if marker.get("status") != "success" or marker.get("error_kind") is not None:
        return False
    try:
        terminal_ppl = float(marker["final_val_ppl"])
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(terminal_ppl):
        return False
    saved_diagnostics   = marker.get("collect_diagnostics")
    saved_extrapolation = marker.get("collect_extrapolation")
    if type(saved_diagnostics) is not bool or saved_diagnostics != collect_diagnostics:
        return False
    if type(saved_extrapolation) is not bool or saved_extrapolation != collect_extrapolation:
        return False
    if collect_diagnostics:
        if not any(key in marker for key in _DIAGNOSTIC_RESULT_KEYS):
            return False
    if collect_extrapolation:
        if not isinstance(marker.get("extrap_ce"), list):
            return False
    return True


def _sanitize(label: str) -> str:
    r"""A filesystem-safe single path component (no separators, parent tokens, or drive colon).

    The char-replace is lossy ('a=b', 'a b', 'a/b' all map to 'a_b'), so a stable short hash of
    the RAW label is appended: distinct labels get distinct run dirs, while the map stays
    deterministic in the label so the resume [CACHED] path finds the same dir on re-run.
    """
    out = label
    for bad, repl in (("=", "_"), (" ", "_"), ("/", "_"), ("\\", "_"), ("..", "_"), (":", "_")):
        out = out.replace(bad, repl)
    out = out.lstrip("._") or "_"
    h = hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
    return f"{out}__{h}"


def _write_sweep_csv(sweep_dir: Path, results: List[Dict[str, Any]]) -> None:
    r"""Rewrite ``sweep_results.csv`` as the complete frame (fixed columns; missing keys blank)."""
    with open(sweep_dir / "sweep_results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in _CSV_COLUMNS})


def _collect_sweep_results(sweep_dir: Path) -> List[Dict[str, Any]]:
    r"""The union of every persisted cell under ``sweep_dir`` (each ``*/ablation_result.json``).

    This is what makes a re-run "tack on": every cell label maps to its own subdirectory, so a
    sweep re-run with a DIFFERENT value list (e.g. ``kappa_beta=0.5,2.2,3.7`` after ``1,2,3,4``) writes
    new cell dirs alongside the old ones, and this union picks up all of them. Re-running the SAME
    label overwrites that one marker while the others persist, so the union is additive and never
    subtracts (to drop a point, delete its cell directory). ``sorted`` keeps CSV row order
    deterministic. Unreadable, non-object, failed, errored, and nonfinite-terminal markers are
    skipped rather than entering the analysis frame.
    """
    results: List[Dict[str, Any]] = []
    for marker in sorted(sweep_dir.glob("*/ablation_result.json")):
        try:
            result = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:                                       # unreadable marker -> skip
            continue
        if not isinstance(result, Mapping):
            continue
        if result.get("status") != "success" or result.get("error_kind") is not None:
            continue
        try:
            terminal_ppl = float(result["final_val_ppl"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(terminal_ppl):
            continue
        results.append(dict(result))
    return results


def run_sweep(
    sweep_name:  str,
    output_dir:  Path,

    *,
    dataset:     str,
    device:      torch.device,
    seed:        int,
    resume:      bool,
    max_tokens:  Optional[int] = None,
    max_steps:   Optional[int] = None,
) -> List[Dict[str, Any]]:
    r"""Run every cell of one sweep; per-cell failures are isolated so the sweep completes."""
    sweep = SWEEPS[sweep_name]
    sweep_dir = output_dir / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    runs = make_run_overrides(sweep_name)
    collect_diagnostics   = bool(sweep.get("collect_diagnostics", False))
    collect_extrapolation = bool(sweep.get("collect_extrapolation", False))
    # Multi-seed (I1/EXP-1): a sweep may declare ``seeds`` to replicate every cell across seeds for an
    # across-seed error bar. Each (cell, seed) gets its own ``{label}__s{seed}`` run dir and result row
    # (the seed also lives in the existing ``seed`` column), so the across-seed aggregate is a plain
    # group-by on the base label. A sweep WITHOUT ``seeds`` keeps the single-seed label/run-dir exactly,
    # so every existing sweep is byte-identical.
    cell_seeds = [int(s) for s in sweep.get("seeds", [])] or [int(seed)]
    multiseed = bool(sweep.get("seeds"))
    cells = [((f"{label}__s{s}" if multiseed else label), overrides, s)
             for (label, overrides) in runs for s in cell_seeds]

    print(f"\n{'=' * 70}\nSWEEP: {sweep_name} ({len(cells)} runs"
          f"{f' = {len(runs)} cells x {len(cell_seeds)} seeds' if multiseed else ''})"
          f"\n  {sweep['description']}"
          f"\n  Output: {sweep_dir}{'  [resume ON]' if resume else ''}\n{'=' * 70}")

    results: List[Dict[str, Any]] = []
    for i, (label, overrides, cell_seed) in enumerate(cells):
        run_dir = sweep_dir / _sanitize(label)
        run_dir.mkdir(parents=True, exist_ok=True)
        marker = run_dir / "ablation_result.json"

        if resume and marker.exists():
            if _cell_is_current(run_dir, overrides, seed=cell_seed, max_steps=max_steps,
                                max_tokens=max_tokens, dataset=dataset,
                                collect_diagnostics=collect_diagnostics,
                                collect_extrapolation=collect_extrapolation):
                print(f"\n--- {i + 1}/{len(cells)}: {label}  [CACHED] ---")
                results.append(json.loads(marker.read_text(encoding="utf-8")))
                continue
            print(f"\n--- {i + 1}/{len(cells)}: {label}  [config changed -> re-running] ---")
        else:
            print(f"\n--- {i + 1}/{len(cells)}: {label} ---")
        t0 = time.perf_counter()
        try:
            result = run_single(label, overrides, run_dir, dataset=dataset, device=device,
                                 seed=cell_seed, collect_diagnostics=collect_diagnostics,
                                 collect_extrapolation=collect_extrapolation,
                                 max_tokens=max_tokens, max_steps=max_steps)
        except Exception as exc:                             # a training crash must not kill the sweep
            logger.exception("sweep %s / %s crashed", sweep_name, label)
            result = {"label": label, "error_kind": "train", "error": str(exc),
                      "primary_val_ppl": float("inf"), "seed": int(cell_seed),
                      "overrides": _jsonable(overrides)}
        finally:
            _cleanup()

        result.setdefault("error_kind", None)
        result["collect_diagnostics"]   = collect_diagnostics
        result["collect_extrapolation"] = collect_extrapolation
        try:
            terminal_ppl = float(result["final_val_ppl"])
        except (KeyError, TypeError, ValueError):
            terminal_ppl = float("inf")
        result["final_val_ppl"] = terminal_ppl
        successful = result["error_kind"] is None and math.isfinite(terminal_ppl)
        result["status"] = "success" if successful else "failed"
        result["sweep"] = sweep_name
        result["wall_time_s"] = time.perf_counter() - t0
        marker.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        results.append(result)

        ppl = result["primary_val_ppl"]
        tag = f" [{result['error_kind'].upper()}]" if result.get("error_kind") else ""
        print(f"\n\n  -> val PPL {ppl:.3f}{tag}  ({result['wall_time_s']:.0f}s)\n")
        if i == 0 and len(cells) > 1:
            est = result["wall_time_s"] * len(cells)
            print(f"  ** ~{est / 60:.0f} min estimated for the full {len(cells)}-run sweep")

        # Keep the CSV whole AND accumulated after each cell: write the union of every persisted
        # marker (this cell, the rest of this run, and any prior run's cells) so the tacked-on
        # frame is live even mid-sweep.
        _write_sweep_csv(sweep_dir, _collect_sweep_results(sweep_dir))

    (sweep_dir / "sweep_meta.json").write_text(json.dumps({
        "sweep_name":  sweep_name,
        "description": sweep["description"],
        "n_runs":      len(cells),
        "dataset":     dataset,
        "seed":        (cell_seeds if multiseed else seed),
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")

    # Final whole-frame write over the accumulated union (also covers the all-cached case, where
    # the per-cell write above never fires). The best line and the return value are the union too.
    union = _collect_sweep_results(sweep_dir)
    _write_sweep_csv(sweep_dir, union)

    finished = [r for r in union if _as_float(r.get("primary_val_ppl")) < float("inf")]
    if finished:
        best = min(finished, key=lambda r: _as_float(r.get("primary_val_ppl")))
        print(f"\nSWEEP COMPLETE: {sweep_name}  ->  best {best['label']} "
              f"(val PPL {_as_float(best['primary_val_ppl']):.3f})")
    else:
        print(f"\nSWEEP COMPLETE: {sweep_name}  ->  no successful run")
    return union


# =============================================================================
# ANALYSIS  (reads sweep_results.csv; no model re-run)
# =============================================================================

def _read_sweep_csv(sweep_dir: Path) -> List[Dict[str, Any]]:
    path = sweep_dir / "sweep_results.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("inf")


def _base_label(label: str) -> str:
    r"""Strip the ``__s<seed>`` suffix a multi-seed sweep appends, so the seeded cells of one config
    collapse to one base label (``a2_on__s0`` / ``a2_on__s1`` -> ``a2_on``). A non-seeded label is
    returned unchanged."""
    base, _, tail = str(label).rpartition("__s")
    return base if (base and tail.lstrip("-").isdigit()) else str(label)


def _seed_aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    r"""Group finished cells by base label into across-seed statistics: n, and the mean / SD / CV of the
    primary val PPL. A single-seed base reports n=1, SD=0. Sorted by mean PPL; bases with no finite PPL
    are dropped. The honest endpoint for reading a sub-3-percent ablation effect against seed noise."""
    groups: Dict[str, List[float]] = {}
    for r in rows:
        ppl = _as_float(r.get("primary_val_ppl"))
        if ppl < float("inf"):
            groups.setdefault(_base_label(r.get("label", "")), []).append(ppl)
    agg: List[Dict[str, Any]] = []
    for base, vals in groups.items():
        n = len(vals)
        mean = sum(vals) / n
        sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
        agg.append({"label": base, "n": n, "mean": mean, "sd": sd,
                    "cv": (sd / mean) if mean > 0 else float("nan")})
    return sorted(agg, key=lambda d: d["mean"])


def _aggregate_cells(
    cells:      List[Dict[str, Any]],
    key_fields: 'tuple | list',
    avg_fields: 'tuple | list',
) -> List[Dict[str, Any]]:
    r"""Collapse per-seed cells that share the same ``key_fields`` into one cell whose ``avg_fields`` are
    the across-seed means (non-finite values dropped; empty -> nan); non-averaged fields inherit the first
    member. Used so a multi-seed sweep's ``__s<seed>`` cells aggregate into one point/bar before the
    headline ablation figures plot them, instead of being skipped or read off one arbitrary seed
    (audit 2026-07-06 M4)."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for c in cells:
        groups.setdefault(tuple(c.get(k) for k in key_fields), []).append(c)
    out: List[Dict[str, Any]] = []
    for members in groups.values():
        agg = dict(members[0])
        for f in avg_fields:
            vals = [float(m[f]) for m in members
                    if isinstance(m.get(f), (int, float))
                    and float(m[f]) == float(m[f]) and abs(float(m[f])) != float("inf")]
            agg[f] = (sum(vals) / len(vals)) if vals else float("nan")
        out.append(agg)
    return out


def analyze_sweep(sweep_dir: Path) -> None:
    rows = _read_sweep_csv(sweep_dir)
    if not rows:
        print(f"No results in {sweep_dir}")
        return
    for r in rows:
        r["_ppl"] = _as_float(r.get("primary_val_ppl"))
    rows.sort(key=lambda r: r["_ppl"])

    print(f"\n{'=' * 70}\nANALYSIS: {sweep_dir.name}\n{'=' * 70}")
    print(f"{'label':<34}{'val PPL':>12}{'params':>12}{'note':>10}")
    print("-" * 68)
    for r in rows:
        ppl = "inf" if r["_ppl"] == float("inf") else f"{r['_ppl']:.3f}"
        params = f"{int(_as_float(r.get('n_params'))):,}" if r.get("n_params") not in ("", None) else "-"
        note = r.get("error_kind") or ""
        print(f"{r['label']:<34}{ppl:>12}{params:>12}{note:>10}")

    finished = [r for r in rows if r["_ppl"] < float("inf")]
    if len(finished) > 1:
        best = finished[0]["_ppl"]
        print(f"\nrelative to best ({finished[0]['label']}):")
        for r in finished:
            print(f"  {r['label']:<34}{(r['_ppl'] - best) / best * 100:+.1f}%")

    # Across-seed aggregation (only when a base label carries multiple seeds): n / mean / SD / CV of the
    # val PPL per config, so a sub-3-percent effect is read against the seed noise rather than as a
    # single-seed point win. Native here rather than left advisory.
    agg = _seed_aggregate(rows)
    if any(a["n"] > 1 for a in agg):
        print(f"\nacross-seed aggregation ({sum(a['n'] for a in agg)} runs -> {len(agg)} cells):")
        print(f"{'label':<34}{'n':>4}{'mean PPL':>12}{'SD':>10}{'CV%':>8}")
        print("-" * 68)
        for a in agg:
            print(f"{a['label']:<34}{a['n']:>4}{a['mean']:>12.3f}{a['sd']:>10.3f}{a['cv'] * 100:>8.1f}")


def summarize_sweeps(output_dir: Path) -> None:
    r"""Cross-sweep comparison table: the best (lowest val PPL) cell of every persisted sweep.

    Printed once after all sweeps in a run (the per-sweep tables come from ``analyze_sweep`` as
    each sweep finishes). Scans EVERY sweep dir under ``output_dir`` so earlier-session sweeps
    are included, not just this run's.
    """
    print(f"\n{'=' * 70}\nBEST PER SWEEP  ({output_dir})\n{'=' * 70}")
    sweep_dirs = [d for d in sorted(output_dir.iterdir())
                  if d.is_dir() and (d / "sweep_results.csv").exists()]
    if not sweep_dirs:
        print("No completed sweeps found.")
        return
    print(f"{'sweep':<24}{'best config':<30}{'val PPL':>10}")
    print("-" * 64)
    for d in sweep_dirs:
        rows = [r for r in _read_sweep_csv(d) if _as_float(r.get("primary_val_ppl")) < float("inf")]
        if not rows:
            continue
        best = min(rows, key=lambda r: _as_float(r.get("primary_val_ppl")))
        print(f"{d.name:<24}{best['label']:<30}{_as_float(best['primary_val_ppl']):>10.3f}")


# =============================================================================
# PLOTS
# =============================================================================

def _plt_or_none() -> Any:
    r"""Import matplotlib and apply the publication style once; return the module or ``None``.

    Plotting is best-effort -- a headless box or a missing matplotlib must never abort a sweep --
    so a failure prints once and returns ``None``, and the caller silently skips the figure.
    ``set_publication_style`` is idempotent, so repeated calls across per-sweep figures are fine.
    """
    try:
        import matplotlib.pyplot as plt
        from vfe3.viz.figures import set_publication_style
        set_publication_style()
        return plt
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"plotting unavailable ({exc}); skipping figure")
        return None


def _plot_one_sweep(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Write ``figures/<sweep>.png`` from the sweep's ACCUMULATED CSV (so a tacked-on re-run shows
    every point). Numeric ``param=value`` labels -> line plot, x-sorted by value; categorical arms
    -> bar plot, sorted by PPL. No-op when the sweep has no finished cell or matplotlib is absent.
    """
    rows = [r for r in _read_sweep_csv(sweep_dir) if _as_float(r.get("primary_val_ppl")) < float("inf")]
    if not rows:
        return
    plt = _plt_or_none()
    if plt is None:
        return

    labels = [r["label"] for r in rows]
    ppls = [_as_float(r["primary_val_ppl"]) for r in rows]

    # Numeric param=value labels -> line plot; categorical arms -> sorted bar plot.
    numeric: Optional[List[float]] = []
    for lab in labels:
        try:
            numeric.append(float(str(lab).split("=")[-1]))
        except ValueError:
            numeric = None
            break

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if numeric is not None:
        order = sorted(range(len(numeric)), key=lambda k: numeric[k])
        ax.plot([numeric[k] for k in order], [ppls[k] for k in order], "o-", lw=2, ms=7)
        ax.set_xlabel(sweep_dir.name)
        ax.set_ylabel("validation PPL")            # numeric line plot: PPL on the y-axis
    else:
        order = sorted(range(len(ppls)), key=lambda k: ppls[k])
        vals = [ppls[k] for k in order]
        ax.barh(range(len(order)), vals,
                color=["#2ca02c" if j == 0 else "#1f77b4" for j in range(len(order))])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([labels[k] for k in order])
        ax.invert_yaxis()
        ax.set_xlabel("validation PPL")            # horizontal bars: PPL on the x-axis (m30)
        # Zoom the value axis to a padded [min, max] window (not [0, max]) so near-equal PPLs are
        # visually distinguishable instead of all reading as full-length bars, and annotate each bar
        # with its value so the truncated bars are not misread.
        lo, hi = min(vals), max(vals)
        if hi > lo:
            span = hi - lo
            ax.set_xlim(lo - 0.25 * span, hi + 0.30 * span)
        for j, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", xy=(v, j), xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8, clip_on=False)
    ax.set_title(sweep_dir.name)
    fig.tight_layout()
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  figure -> {out}")


def _plot_seed_aggregate(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Write ``figures/<sweep>_seed_aggregate.png`` -- across-seed mean val PPL +/- 1 SD per base label
    (the seed-grouped error bars the per-row table flattens). A no-op unless some base label has >= 2
    seeds, so it is safe to call after every sweep; a single-seed sweep draws nothing."""
    agg = sorted(_seed_aggregate(_read_sweep_csv(sweep_dir)), key=lambda d: d["mean"])
    if not any(a["n"] > 1 for a in agg):
        return
    plt = _plt_or_none()
    if plt is None:
        return
    y = list(range(len(agg)))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(agg) + 1.5))
    ax.errorbar([a["mean"] for a in agg], y, xerr=[a["sd"] for a in agg], fmt="o",
                color="#0072B2", ecolor="#888888", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a['label']} (n={a['n']})" for a in agg])
    ax.invert_yaxis()
    ax.set_xlabel("validation PPL (across-seed mean $\\pm$ 1 SD)")
    ax.set_title(f"{sweep_dir.name}: seed-grouped aggregation")
    fig.tight_layout()
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_seed_aggregate.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  figure -> {out}")


def _plot_rank_collapse(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Write ``figures/<sweep>_rank_collapse.png`` -- the F2/EXP-7 Dong r(X)-vs-depth overlay, one
    line per arm -- from the per-cell ``rank_resid_by_layer`` curves persisted in each
    ``ablation_result.json`` (off ``collect_diagnostics``; see ``_cell_diagnostics``). A no-op (writes
    nothing) unless at least one finished cell carries a >=2-layer rank curve, so it is safe to call
    after every sweep -- only the deep collect_diagnostics sweeps (rho_handoff) produce the curve."""
    arms: Dict[str, Any] = {}
    for r in _collect_sweep_results(sweep_dir):
        curve = r.get("rank_resid_by_layer")
        if (isinstance(curve, list) and len(curve) >= 2
                and _as_float(r.get("primary_val_ppl")) < float("inf")):
            arms[str(r.get("label", "?"))] = curve
    if not arms:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_rank_residual_by_depth
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"rank-collapse figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_rank_collapse.png"
    fig = plot_rank_residual_by_depth(arms, path=str(out))
    plt.close(fig)
    print(f"  figure -> {out}")


def _plot_attention_entropy(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Write the C1/EXP-4 figures -- the canonical-vs-surrogate PPL gap (grouped by kappa) and the
    -tau^{-1} Cov_beta gradient-gap magnitude vs kappa -- from the per-cell ``include_attention_entropy``
    / ``kappa_beta`` overrides and the ``cov_gap`` diagnostic. A no-op unless >= 2 finished cells carry
    an ``include_attention_entropy`` override, so it is safe to call after every sweep (only the
    attention_entropy sweep produces these)."""
    cells_ppl, cells_gap = [], []
    for r in _collect_sweep_results(sweep_dir):
        ov = r.get("overrides", {}) or {}
        if "include_attention_entropy" not in ov or _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        kap = _as_float(ov.get("kappa_beta", BASELINE_CONFIG.get("kappa_beta")))
        if kap >= float("inf"):                                  # non-scalar kappa (per-head list) -> skip
            continue
        ent = bool(ov["include_attention_entropy"])
        cells_ppl.append({"include_attention_entropy": ent, "kappa": kap,
                          "ppl": _as_float(r.get("primary_val_ppl"))})
        if _as_float(r.get("cov_gap")) < float("inf"):
            cells_gap.append({"include_attention_entropy": ent, "kappa": kap,
                              "cov_gap": _as_float(r.get("cov_gap"))})
    cells_ppl = _aggregate_cells(cells_ppl, ("include_attention_entropy", "kappa"), ("ppl",))   # M4
    cells_gap = _aggregate_cells(cells_gap, ("include_attention_entropy", "kappa"), ("cov_gap",))
    if len(cells_ppl) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_cov_gap_vs_kappa, plot_entropy_ppl_gap
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"attention-entropy figures unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_ppl_gap.png"
    plt.close(plot_entropy_ppl_gap(cells_ppl, path=str(out)))
    print(f"  figure -> {out}")
    if len(cells_gap) >= 2:
        out = fig_dir / f"{sweep_dir.name}_cov_gap.png"
        plt.close(plot_cov_gap_vs_kappa(cells_gap, path=str(out)))
        print(f"  figure -> {out}")


def _plot_wallclock_convergence(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Write ``figures/<sweep>_wallclock_convergence.png`` -- the D1/EXP-8 per-wall-clock convergence
    overlay (val PPL vs cumulative wall time, one line per arm, with steps/wall-to-target annotated) --
    from each cell's ``metrics.csv`` eval rows (val_ppl + wall_clock_s). A no-op unless >= 2 cells carry
    >= 2 eval points, so it is safe to call after every sweep (the gauge M-step sweeps populate it)."""
    arms: List[Dict[str, Any]] = []
    for cell in sorted(sweep_dir.glob("*/metrics.csv")):
        steps, ppls, walls = [], [], []
        try:
            with open(cell, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    vp, wc = _as_float(r.get("val_ppl")), _as_float(r.get("wall_clock_s"))
                    if vp < float("inf") and wc < float("inf"):     # eval rows carry both
                        steps.append(_as_float(r.get("step")))
                        ppls.append(vp)
                        walls.append(wc)
        except Exception:                                           # unreadable metrics.csv -> skip cell
            continue
        if len(ppls) >= 2:
            arms.append({"label": cell.parent.name, "step": steps, "val_ppl": ppls, "wall_clock_s": walls})
    if len(arms) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_wallclock_convergence
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"wallclock-convergence figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_wallclock_convergence.png"
    plt.close(plot_wallclock_convergence(arms, path=str(out)))
    print(f"  figure -> {out}")


def _plot_gauge_transport(sweep_dir: Path, fig_dir: Path) -> None:
    r"""A1/EXP-2 gauge ON/frozen/OFF(Omega=I) grouped-bar (val PPL by depth, omega_identity_dev
    annotated) from the gauge_transport cells. No-op unless >= 2 cells label as <mode>_<depth> with
    mode in {on,off,frozen}."""
    cells: List[Dict[str, Any]] = []
    for r in _collect_sweep_results(sweep_dir):
        parts = _base_label(str(r.get("label", ""))).split("_")   # strip __s<seed> before parsing (M4)
        if len(parts) != 2 or parts[0] not in ("on", "off", "frozen"):
            continue
        if _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        cells.append({"mode": parts[0], "depth": parts[1],
                      "ppl": _as_float(r.get("primary_val_ppl")),
                      "omega_dev": _as_float(r.get("omega_identity_dev"))})
    cells = _aggregate_cells(cells, ("mode", "depth"), ("ppl", "omega_dev"))   # collapse seeds (M4)
    if len(cells) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_gauge_transport_bars
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"gauge-transport figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_gauge_bars.png"
    plt.close(plot_gauge_transport_bars(cells, path=str(out)))
    print(f"  figure -> {out}")


def _plot_cg_coupling(sweep_dir: Path, fig_dir: Path) -> None:
    r"""A3/EXP-10 combined PPL + median equivariance-residual bar from the cg_coupling cells (gated on
    the use_cg_coupling override). No-op unless >= 2 such cells finished."""
    cells: List[Dict[str, Any]] = []
    for r in _collect_sweep_results(sweep_dir):
        ov = r.get("overrides", {}) or {}
        if "use_cg_coupling" not in ov or _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        cells.append({"label": str(r.get("label", "")), "ppl": _as_float(r.get("primary_val_ppl")),
                      "resid": _as_float(r.get("gauge_resid_in"))})
    if len(cells) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_ppl_equivariance_bars
    except Exception as exc:
        print(f"cg-coupling figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_ppl_equiv.png"
    plt.close(plot_ppl_equivariance_bars(cells, path=str(out)))
    print(f"  figure -> {out}")


def _plot_kappa_dispersion(sweep_dir: Path, fig_dir: Path) -> None:
    r"""H2/EXP-11 PPL vs per-head temperature dispersion std(kappa_beta), read from each cell's
    kappa_beta LIST override (the scalar-kappa sweeps carry no list -> skipped). No-op unless >= 2
    list-valued cells finished."""
    cells: List[Dict[str, Any]] = []
    for r in _collect_sweep_results(sweep_dir):
        kb = (r.get("overrides", {}) or {}).get("kappa_beta")
        if not isinstance(kb, (list, tuple)) or len(kb) < 2:
            continue
        if _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        try:
            vals = [float(x) for x in kb]
        except (TypeError, ValueError):
            continue
        m = sum(vals) / len(vals)
        disp = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5      # population std of the kappa list
        cells.append({"label": str(r.get("label", "")), "dispersion": disp,
                      "ppl": _as_float(r.get("primary_val_ppl"))})
    if len(cells) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_kappa_dispersion
    except Exception as exc:
        print(f"kappa-dispersion figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_kappa_dispersion.png"
    plt.close(plot_kappa_dispersion(cells, path=str(out)))
    print(f"  figure -> {out}")


def _plot_gauge_residual_drift(sweep_dir: Path, fig_dir: Path) -> None:
    r"""A2/EXP-9 builder-break gauge residual vs step (tied vs untied) from each cell's metrics.csv
    ``val_builder_resid`` eval series. No-op unless >= 2 cells carry >= 2 eval points."""
    arms: List[Dict[str, Any]] = []
    for cell in sorted(sweep_dir.glob("*/metrics.csv")):
        steps, res = [], []
        try:
            with open(cell, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    rv = _as_float(row.get("val_builder_resid"))
                    if rv < float("inf"):                              # eval rows carry the residual
                        steps.append(_as_float(row.get("step")))
                        res.append(rv)
        except Exception:                                             # unreadable metrics.csv -> skip
            continue
        if len(res) >= 2:
            arms.append({"label": cell.parent.name, "step": steps, "resid": res})
    if len(arms) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_gauge_residual_drift
    except Exception as exc:
        print(f"gauge-residual-drift figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_residual_drift.png"
    plt.close(plot_gauge_residual_drift(arms, path=str(out)))
    print(f"  figure -> {out}")


def _plot_holonomy_trainability(sweep_dir: Path, fig_dir: Path) -> None:
    r"""A4/EXP-15 holonomy-vs-||connection|| scatter from each cell's metrics.csv (connection_w_norm +
    holonomy_deviation per eval). connection_w_norm is logged only on a regime_ii run, so this no-ops
    unless a cell carries >= 2 such eval rows (the flat arm is correctly excluded)."""
    arms: List[Dict[str, Any]] = []
    for cell in sorted(sweep_dir.glob("*/metrics.csv")):
        steps, cn, hol = [], [], []
        try:
            with open(cell, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    c, h = _as_float(row.get("connection_w_norm")), _as_float(row.get("holonomy_deviation"))
                    if c < float("inf") and h < float("inf"):       # regime_ii eval rows carry both
                        steps.append(_as_float(row.get("step")))
                        cn.append(c)
                        hol.append(h)
        except Exception:                                           # unreadable metrics.csv -> skip
            continue
        if len(cn) >= 2:
            arms.append({"label": cell.parent.name, "step": steps, "connection_norm": cn, "holonomy": hol})
    if not arms:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_holonomy_trainability
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"holonomy-trainability figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_holonomy_trainability.png"
    plt.close(plot_holonomy_trainability(arms, path=str(out)))
    print(f"  figure -> {out}")


def _plot_mu_precond(sweep_dir: Path, fig_dir: Path) -> None:
    r"""B3/EXP-14 PPL-vs-n_e_steps figure, Fisher vs raw mean preconditioner, from the
    fisher_mu_precond cells (label '<precond>_T<n_e_steps>'). No-op unless >= 2 such cells finished."""
    cells: List[Dict[str, Any]] = []
    for r in _collect_sweep_results(sweep_dir):
        lab = _base_label(str(r.get("label", "")))   # strip __s<seed> before parsing (M4)
        if "_T" not in lab or _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        pre, _, t = lab.partition("_T")
        if pre not in ("fisher", "raw"):
            continue
        try:
            n_e = int(t)
        except ValueError:
            continue
        cells.append({"precond": pre, "n_e_steps": n_e, "ppl": _as_float(r.get("primary_val_ppl"))})
    cells = _aggregate_cells(cells, ("precond", "n_e_steps"), ("ppl",))   # collapse seeds (M4)
    if len(cells) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_mu_precond
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"mu-precond figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_mu_precond.png"
    plt.close(plot_mu_precond(cells, path=str(out)))
    print(f"  figure -> {out}")


def _plot_renyi_saturation(sweep_dir: Path, fig_dir: Path) -> None:
    r"""B2/EXP-12 H(beta)-vs-alpha + kl_max-saturation-vs-alpha figure from the renyi_order cells
    (label 'renyi_order=<alpha>', with attn_entropy + energy_klmax_frac diagnostics). No-op unless
    >= 2 such cells finished."""
    cells: List[Dict[str, Any]] = []
    for r in _collect_sweep_results(sweep_dir):
        lab = str(r.get("label", ""))
        if not lab.startswith("renyi_order=") or _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        try:
            alpha = float(lab.split("=")[-1])
        except ValueError:
            continue
        cells.append({"alpha": alpha, "attn_entropy": _as_float(r.get("attn_entropy")),
                      "energy_klmax_frac": _as_float(r.get("energy_klmax_frac"))})
    if len(cells) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_renyi_saturation
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"renyi-saturation figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_renyi_saturation.png"
    plt.close(plot_renyi_saturation(cells, path=str(out)))
    print(f"  figure -> {out}")


def _plot_pos_extrapolation(sweep_dir: Path, fig_dir: Path) -> None:
    r"""H1/EXP-13 CE-vs-N extrapolation overlay (one line per positional scheme) from each cell's
    ``extrap_ce`` curve (persisted by run_single under collect_extrapolation). No-op unless >= 2 cells
    carry a >= 2-point curve. The train length (the smallest evaluated N = max_seq_len) is marked."""
    arms: Dict[str, Any] = {}
    all_n: List[float] = []
    for r in _collect_sweep_results(sweep_dir):
        curve = r.get("extrap_ce")
        if isinstance(curve, list) and len(curve) >= 2:
            arms[str(r.get("label", "?"))] = curve
            all_n += [float(p["n"]) for p in curve if isinstance(p, dict) and "n" in p]
    if len(arms) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_pos_extrapolation
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"pos-extrapolation figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_extrapolation.png"
    plt.close(plot_pos_extrapolation(arms, train_n=(min(all_n) if all_n else None), path=str(out)))
    print(f"  figure -> {out}")


def _plot_sensitivity(output_dir: Path, fig_dir: Path) -> None:
    r"""Cross-sweep comparison: a PPL-range (worst - best) bar per sweep, sorted by sensitivity.

    Made once after all sweeps. Scans EVERY persisted sweep under ``output_dir`` (not just this
    run's), matching the per-sweep figures' accumulated view.
    """
    sweep_dirs = [d for d in sorted(output_dir.iterdir())
                  if d.is_dir() and (d / "sweep_results.csv").exists()]
    sensitivity: List[Tuple[str, float, str]] = []           # (sweep, ppl range, best label)
    for d in sweep_dirs:
        rows = [r for r in _read_sweep_csv(d) if _as_float(r.get("primary_val_ppl")) < float("inf")]
        if not rows:
            continue
        ppls = [_as_float(r["primary_val_ppl"]) for r in rows]
        best = min(rows, key=lambda r: _as_float(r["primary_val_ppl"]))
        sensitivity.append((d.name, max(ppls) - min(ppls), best["label"]))
    if not sensitivity:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    sensitivity.sort(key=lambda t: t[1], reverse=True)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(sensitivity))))
    ax.barh(range(len(sensitivity)), [s[1] for s in sensitivity], color="#d62728", alpha=0.8)
    ax.set_yticks(range(len(sensitivity)))
    ax.set_yticklabels([f"{s[0]}\n(best: {s[2]})" for s in sensitivity])
    ax.invert_yaxis()
    ax.set_xlabel("validation PPL range (worst - best)")
    ax.set_title("hyperparameter sensitivity")
    fig.tight_layout()
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / "sensitivity_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"comparison figure -> {out}")


# =============================================================================
# MAIN  (click-to-run; edit CONFIG above)
# =============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = Path(CONFIG["output_dir"])

    if CONFIG["list_only"]:
        # Every registered sweep (CONFIG["sweep"]=None) or just the named one; an asterisk marks
        # those in the curated SWEEP_ORDER that a None-sweep run would execute.
        names = sorted(SWEEPS) if CONFIG["sweep"] is None else [CONFIG["sweep"]]
        active = set(SWEEP_ORDER)
        print(f"\nRegistered sweeps ({len(names)} shown; * = in SWEEP_ORDER):\n")
        print(f"  {'name':<28}{'runs':>5}  description")
        print("-" * 90)
        for name in names:
            s = SWEEPS[name]
            mark = "*" if name in active else " "
            print(f"{mark} {name:<28}{sweep_n_runs(s):>5}  {s['description']}")
        print(f"\n{len(SWEEPS)} sweeps registered; SWEEP_ORDER runs {len(SWEEP_ORDER)} "
              f"({sum(sweep_n_runs(SWEEPS[n]) for n in SWEEP_ORDER)} runs).")
        return

    # ---- contiguous run: per sweep { train -> analyze table -> PPL figure }, then comparison ----
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])

    sweep_names = SWEEP_ORDER if CONFIG["sweep"] is None else [CONFIG["sweep"]]
    for name in sweep_names:
        if name not in SWEEPS:
            raise ValueError(f"unknown sweep {name!r}; choose from {sorted(SWEEPS)}")
    validate_sweeps(sweep_names)                             # guard #1: loud field check

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    print(f"\nVFE_3.0 ablation suite\n  device:  {device}\n  dataset: {CONFIG['dataset']}"
          f"\n  output:  {output_dir}\n  seed:    {CONFIG['seed']}"
          f"\n  sweeps:  {', '.join(sweep_names)}")

    for name in sweep_names:
        run_sweep(name, output_dir, dataset=CONFIG["dataset"], device=device,
                  seed=CONFIG["seed"], resume=CONFIG["resume"],
                  max_tokens=CONFIG["max_tokens"], max_steps=CONFIG["max_steps"])
        sweep_dir = output_dir / name
        analyze_sweep(sweep_dir)                             # this sweep's table (accumulated)
        _plot_one_sweep(sweep_dir, fig_dir)                 # this sweep's PPL figure (tacked on)
        _plot_seed_aggregate(sweep_dir, fig_dir)           # multi-seed mean+/-SD forest (no-op if single-seed)
        _plot_rank_collapse(sweep_dir, fig_dir)            # F2/EXP-7 r(X)-by-depth (no-op if absent)
        _plot_attention_entropy(sweep_dir, fig_dir)        # C1/EXP-4 PPL-gap + Cov-gap (no-op if absent)
        _plot_wallclock_convergence(sweep_dir, fig_dir)    # D1/EXP-8 wall-clock convergence (no-op if absent)
        _plot_gauge_transport(sweep_dir, fig_dir)          # A1/EXP-2 gauge on/off/frozen bars (no-op if absent)
        _plot_cg_coupling(sweep_dir, fig_dir)              # A3/EXP-10 PPL+equivariance bars (no-op if absent)
        _plot_kappa_dispersion(sweep_dir, fig_dir)         # H2/EXP-11 kappa dispersion (no-op if absent)
        _plot_gauge_residual_drift(sweep_dir, fig_dir)     # A2/EXP-9 residual drift (no-op if absent)
        _plot_pos_extrapolation(sweep_dir, fig_dir)        # H1/EXP-13 CE-vs-N extrapolation (no-op if absent)
        _plot_renyi_saturation(sweep_dir, fig_dir)         # B2/EXP-12 entropy+saturation vs alpha (no-op if absent)
        _plot_mu_precond(sweep_dir, fig_dir)               # B3/EXP-14 Fisher-vs-raw mean precond (no-op if absent)
        _plot_holonomy_trainability(sweep_dir, fig_dir)    # A4/EXP-15 holonomy vs ||connection|| (no-op if absent)

    # ---- after all sweeps: the cross-sweep comparison ----
    _plot_sensitivity(output_dir, fig_dir)
    summarize_sweeps(output_dir)


if __name__ == "__main__":
    main()
