r"""Click-to-run hyperparameter ablation/sweep runner for the VFE_3.0 transformer.

Sweeps one (or several) ``VFE3Config`` field(s) around the operating point defined in
``train_vfe3.py``. There is no CLI arg parsing (project policy): edit the ``CONFIG`` dict
at the bottom, pick a sweep, then run ``python ablation.py``.

Two sweep shapes are supported, both declared in the ``SWEEPS`` registry:

  * single-field  -- vary ONE field across an explicit ``values`` list or an arithmetic
    ``range = [start, stop, step]`` (one-at-a-time ablation around the baseline);
  * multi-arm     -- a ``configs`` list of named arms, each a dict of field overrides,
    for categorical comparisons whose arms differ in more than one field (e.g. a
    full-covariance arm that flips ``family`` AND ``diagonal_covariance`` together).

The baseline is IMPORTED from ``train_vfe3.py`` (``from train_vfe3 import config``), so a
sweep ablates around exactly what a normal ``train_vfe3.py`` run would train -- there is no
second copy of the operating point to drift out of sync. Each run gets a self-contained
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

import copy
import csv
import gc
import json
import logging
import time
from dataclasses import asdict, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from vfe3.config import VFE3Config
from vfe3.data.datasets import make_dataloader
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts
from vfe3.train import evaluate, train

# Only the zero-dependency synthetic stream is borrowed from train_vfe3 (the corpus-cache
# fallback); the baseline operating point is self-contained below.
from train_vfe3 import synthetic_period3_loader

logger = logging.getLogger("ablation")


# =============================================================================
# BASELINE CONFIG  -- self-contained operating point: EVERY VFE3Config toggle.
# =============================================================================
# A sweep ablates one (or a few) of these around this point. This is an INDEPENDENT copy
# (train_vfe3.py keeps its own click-to-run config); edit here for ablations. Grouped exactly as
# vfe3/config.py; registry fields list valid keys inline. The SWEEPS below pre-satisfy any
# cross-field constraint per sweep via `requires` / `configs`.
BASELINE_CONFIG: Dict[str, Any] = dict(
    # numerics
    eps                        = 1e-6,
    kl_max                     = 100.0,

    # divergence seam (f-divergence FUNCTIONAL, distinct from `family`)
    divergence_family          = "renyi",                    # "renyi"
    alpha_div                  = 1.0,                         # Renyi order (1.0 -> KL)

    # model structure
    vocab_size                 = 50257,                       # gpt2/tiktoken vocab (dataset-fixed)
    embed_dim                  = 20,                          # K (must be divisible by n_heads)
    max_seq_len                = 128,                         # N, context length
    n_layers                   = 1,                           # L, number of blocks
    n_e_steps                  = 1,                           # T, E-step inner iterations
    n_heads                    = 2,

    # gauge seam
    gauge_group                = "block_glk",                 # "glk"|"block_glk"|"tied_block_glk"|"so_k"|"sp"
    gauge_parameterization     = "phi",                       # "phi" (omega_direct: live-rejected)
    transport_mode             = "flat",                      # "flat" (pure no-NN) | "regime_ii" (learned, NN exception)
    cocycle_relaxation         = 1.0,                         # regime_ii homotopy (ignored by flat)
    cross_couplings            = None,                        # off-block GL(K) head pairs e.g. [(0, 1)]; block_glk only
    use_head_mixer             = True,                        # Schur-commutant head mixer (needs >=2 equal blocks)

    # positional encoding
    pos_phi                    = "learned",                   # "none" (pure path) | "learned" | "frozen"
    pos_phi_compose            = "bch",                       # "bch" | "euclidean"
    bch_pe_order               = 4,
    pos_phi_scale              = 0.02,
    pos_phi_project_slk        = False,
    pos_rotation               = "none",                      # "none" | "rope"
    rope_base                  = 100.0,
    rope_full_gauge            = False,                       # rotate covariance (REQUIRES diagonal_covariance=False)

    # belief family (diagonal_covariance MUST equal family == "gaussian_diagonal")
    diagonal_covariance        = True,
    family                     = "gaussian_diagonal",         # "gaussian_diagonal" | "gaussian_full"

    # free-energy coupling
    alpha                      = 1.0,                         # constant self-coupling value
    alpha_mode                 = "state_dependent_per_coord", # "constant"|"state_dependent"|"state_dependent_per_coord"|"learnable"
    b0                         = 1.0,
    c0                         = 1.0,
    kappa                      = 1.0,                         # tau = kappa * sqrt(d_head)
    mass_phi                   = 0.0,
    mstep_self_coupling_weight = 0.0,
    lambda_h                   = 0.0,                         # hyper-prior weight (>0 creates s/r tables)
    gamma_coupling             = 0.0,                         # model-channel coupling (>0 creates s tables)
    kappa_gamma                = 1.0,
    gamma_attention_prior      = "causal",                    # "uniform" | "causal" | "alibi"
    prior_source               = "token",                     # "token" | "model_channel"

    # attention
    include_attention_entropy  = True,                        # canonical F vs entropy-suppressed surrogate
    attention_prior            = "causal",                    # "uniform" | "causal" | "alibi"

    # E-step
    e_mu_lr                    = 0.7,
    e_sigma_lr                 = 0.025,
    e_phi_lr                   = 0.0,
    e_sigma_q_trust            = 5.0,
    sigma_max                  = 5.0,
    gradient_mode              = "filtering",                 # "filtering" | "smoothing"
    phi_precond_mode           = "killing",                   # "none"|"clip"|"killing"|"killing_per_block"|"pullback"
    phi_retract_mode           = "bch",                       # "euclidean" | "bch"
    spd_retract_mode           = "spd_affine",                # "spd_affine" | "log_euclidean"

    # decode / encode
    use_prior_bank             = False,                       # True: KL-to-prior (pure) | False: linear projection
    decode_tau                 = 1.0,
    decode_mode                = "diagonal",                  # "diagonal" | "diagonal_chunked" | "full"
    decode_chunk_size          = 8192,
    encode_mode                = "per_token",                 # "per_token" (gauge_fixed: live-rejected)

    # cross-block belief handoff
    prior_handoff_rho          = 0.0,                         # 1.0 = full flow; 0.0 = frozen
    prior_handoff_sigma        = 0.0,

    # normalization
    norm_type_block            = "none",                      # "none" | "mahalanobis"
    norm_type_final            = "none",                      # "none" | "mahalanobis"

    # M-step / training
    e_step_gradient            = "unroll",                    # "unroll" | "straight_through" | "detach"
    detach_e_step              = False,
    grad_accum_steps           = 1,
    m_mu_lr                    = 0.01,
    m_sigma_lr                 = 0.0021,
    m_phi_lr                   = 0.009,
    weight_decay               = 0.05,
    batch_size                 = 64,
    max_steps                  = 15000,
    warmup_steps               = 100,
    seed                       = 6,                           # overridden per run by CONFIG["seed"]
    log_interval               = 100,
    eval_interval              = 1000,
    checkpoint_interval        = 15000,                       # forced to 0 per cell (no checkpoint blowup)
    eval_max_batches           = None,
    amp_dtype                  = None,                        # None (pure fp32) | "bf16" | "fp16"
)


# =============================================================================
# SWEEP REGISTRY  -- each entry sweeps real VFE3Config field(s); edit freely.
# =============================================================================
# Schema per sweep:
#   description : str                       one-line human summary (printed + plotted)
#   single-field form:
#     param         : str                   the VFE3Config field to vary
#     values        : [v1, v2, ...]   OR    range : [start, stop, step]
#     baseline_value: Any                   the train_vfe3 value (for reference only)
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
        "param": "n_layers", "values": [1, 2, 3],
    },
    "n_e_steps": {
        "description": "E-step inner iterations T per block",
        "param": "n_e_steps", "values": [1, 2, 4],
    },
    "max_seq_len": {  # loader-affecting: the runner rebuilds the loader per cell
        "description": "context length N",
        "param": "max_seq_len", "values": [64, 128, 256],
    },

    # === divergence / numerics =============================================
    "alpha_div": {
        "description": "Renyi divergence order (1.0 -> KL; != 1 routes the non-kernel oracle)",
        "param": "alpha_div", "values": [0.5, 1.0, 1.5, 2.0],
    },
    "kl_max": {
        "description": "per-pair KL clamp",
        "param": "kl_max", "values": [50.0, 100.0, 200.0],
    },
    "eps": {
        "description": "numerical floor (variance / log stability)",
        "param": "eps", "values": [1e-7, 1e-6, 1e-5],
    },

    # === gauge seam ========================================================
    # use_head_mixer (True at baseline) needs >= 2 equal blocks (block_glk / tied_block_glk);
    # the single-block glk / so_k / sp arms turn it off so the model constructs.
    "gauge_group": {
        "description": "gauge group",
        "configs": [
            {"label": "block_glk",      "gauge_group": "block_glk"},
            {"label": "tied_block_glk", "gauge_group": "tied_block_glk"},
            {"label": "glk",            "gauge_group": "glk",  "use_head_mixer": False},
            {"label": "so_k",           "gauge_group": "so_k", "use_head_mixer": False},
            {"label": "sp",             "gauge_group": "sp",   "use_head_mixer": False},
        ],
    },
    "transport_mode": {  # regime_ii is the learned bilinear connection (sanctioned NN exception)
        "description": "connection regime: flat phi-cocycle vs learned non-flat (regime_ii)",
        "configs": [
            {"label": "flat",      "transport_mode": "flat"},
            {"label": "regime_ii", "transport_mode": "regime_ii"},
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

    # === positional encoding ===============================================
    "pos_phi": {
        "description": "BCH positional encoding mode",
        "param": "pos_phi", "values": ["none", "learned", "frozen"],
    },
    "pos_phi_compose": {
        "description": "BCH composition chart",
        "param": "pos_phi_compose", "values": ["bch", "euclidean"],
        "requires": {"pos_phi": "learned"},
    },
    "bch_pe_order": {
        "description": "BCH Dynkin truncation order",
        "param": "bch_pe_order", "values": [2, 4, 6],
        "requires": {"pos_phi": "learned", "pos_phi_compose": "bch"},
    },
    "pos_phi_scale": {
        "description": "learned pos_phi table init scale",
        "param": "pos_phi_scale", "values": [0.005, 0.02, 0.1],
        "requires": {"pos_phi": "learned"},
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
                                    "diagonal_covariance": False, "family": "gaussian_full",
                                    "alpha_mode": "state_dependent"},
        ],
    },

    # === belief family =====================================================
    # The full arm flips family + diagonal_covariance together and moves off the per-coordinate
    # alpha form (diagonal-only), all of which a naive single-field sweep would have rejected.
    "covariance": {
        "description": "belief covariance structure (diagonal vs full Gaussian)",
        "configs": [
            {"label": "diagonal", "family": "gaussian_diagonal", "diagonal_covariance": True},
            {"label": "full",     "family": "gaussian_full",     "diagonal_covariance": False,
                                  "alpha_mode": "state_dependent"},
        ],
    },

    # === free-energy coupling ==============================================
    "alpha": {
        "description": "constant self-coupling value (alpha_mode=constant)",
        "param": "alpha", "values": [0.5, 1.0, 2.0], "requires": {"alpha_mode": "constant"},
    },
    "alpha_mode": {  # 'learnable' is the NN-exception scalar log_alpha (now optimizer-grouped)
        "description": "self-coupling alpha form",
        "param": "alpha_mode",
        "values": ["constant", "state_dependent", "state_dependent_per_coord", "learnable"],
    },
    "b0": {
        "description": "state-dependent alpha shape b0 (alpha* = c0/(b0 + D))",
        "param": "b0", "values": [0.5, 1.0, 2.0], "requires": {"alpha_mode": "state_dependent"},
    },
    "c0": {
        "description": "state-dependent alpha shape c0 (numerator)",
        "param": "c0", "values": [0.5, 1.0, 2.0], "requires": {"alpha_mode": "state_dependent"},
    },
    "kappa": {
        "description": "attention temperature tau = kappa * sqrt(d_head)",
        "param": "kappa", "values": [0.5, 0.7, 1.0, 1.4, 2.0],
    },
    "mass_phi": {
        "description": "gauge prior weight (mass_phi / 2) ||phi||^2",
        "param": "mass_phi", "values": [0.0, 1e-4, 1e-3, 1e-2],
    },
    "mstep_self_coupling_weight": {
        "description": "M-step self-coupling term alpha_hat * sum_i KL(q_i*||p_i)",
        "param": "mstep_self_coupling_weight", "values": [0.0, 0.1, 1.0],
    },
    "lambda_h": {
        "description": "hyper-prior weight lambda_h * mean_i KL(s_i||r) (>0 creates s/r tables)",
        "param": "lambda_h", "values": [0.0, 0.1, 1.0],
    },
    "gamma_coupling": {
        "description": "model-channel coupling weight (>0 creates s tables)",
        "param": "gamma_coupling", "values": [0.0, 0.1, 1.0],
    },
    "kappa_gamma": {
        "description": "model-channel temperature tau_gamma = kappa_gamma * sqrt(d_head)",
        "param": "kappa_gamma", "values": [0.5, 1.0, 2.0], "requires": {"gamma_coupling": 1.0},
    },
    "gamma_attention_prior": {
        "description": "model-channel attention prior pi^s_ij",
        "param": "gamma_attention_prior", "values": ["uniform", "causal", "alibi"],
        "requires": {"gamma_coupling": 1.0},
    },
    "prior_source": {  # model_channel makes the s tables the belief prior p_i
        "description": "belief-prior source table (token vs model channel)",
        "configs": [
            {"label": "token",         "prior_source": "token"},
            {"label": "model_channel", "prior_source": "model_channel"},
        ],
    },

    # === attention =========================================================
    "entropy_term": {
        "description": "canonical free energy (entropy term) vs entropy-suppressed surrogate",
        "configs": [
            {"label": "canonical", "include_attention_entropy": True},
            {"label": "surrogate", "include_attention_entropy": False},
        ],
    },
    "attention_prior": {
        "description": "attention prior pi_ij",
        "param": "attention_prior", "values": ["uniform", "causal", "alibi"],
    },

    # === E-step ============================================================
    "e_mu_lr": {
        "description": "E-step natural-gradient step size for mu_q",
        "param": "e_mu_lr", "values": [0.3, 0.5, 0.7, 0.9, 1.1],
    },
    "e_sigma_lr": {
        "description": "E-step retraction step size for sigma_q",
        "param": "e_sigma_lr", "values": [0.0, 0.01, 0.025, 0.05],
    },
    "e_phi_lr": {
        "description": "E-step gauge-frame step size for phi",
        "param": "e_phi_lr", "values": [0.0, 0.005, 0.01],
    },
    "e_sigma_q_trust": {
        "description": "E-step SPD retraction trust radius",
        "param": "e_sigma_q_trust", "values": [1.0, 5.0, 10.0],
    },
    "sigma_max": {
        "description": "upper bound on belief variance",
        "param": "sigma_max", "values": [2.0, 5.0, 10.0],
    },
    "gradient_mode": {
        "description": "E-step coupling gradient (filtering = query-side only; smoothing = full)",
        "param": "gradient_mode", "values": ["filtering", "smoothing"],
    },
    "phi_precond_mode": {
        "description": "gauge-step preconditioner on the phi update",
        "param": "phi_precond_mode", "values": ["none", "clip", "killing", "killing_per_block"],
    },
    "phi_retract_mode": {
        "description": "phi Lie-algebra step chart",
        "param": "phi_retract_mode", "values": ["euclidean", "bch"],
    },
    "spd_retract_mode": {  # log_euclidean warns (not errors) on a diagonal family
        "description": "SPD covariance retraction geometry",
        "param": "spd_retract_mode", "values": ["spd_affine", "log_euclidean"],
    },

    # === decode / encode ===================================================
    "decode_head": {
        "description": "KL-to-prior decode (pure path) vs learned linear projection (VFE_2.0 parity)",
        "configs": [
            {"label": "prior_bank",    "use_prior_bank": True},
            {"label": "linear_decode", "use_prior_bank": False},
        ],
    },
    "decode_tau": {
        "description": "KL-to-prior decode temperature",
        "param": "decode_tau", "values": [0.5, 1.0, 2.0], "requires": {"use_prior_bank": True},
    },
    # Only the two diagonal KL-decode variants are swept. The full-cov KL readout (decode_mode=
    # 'full') is no longer the blocker -- its Cholesky was hardened with safe_cholesky -- but on the
    # prior-bank path it drives the full-covariance SPD retraction (retraction.py retract_spd_full)
    # into an eigh that fails to converge on the ill-conditioned spectrum; a gap-regularized robust
    # eigh there is explicitly deferred in the codebase, so the 'full' arm stays excluded (see
    # NON_SWEPT_FIELDS). Full-covariance TRAINING is still exercised by the `covariance` sweep.
    "decode_mode": {
        "description": "KL decode covariance structure (diagonal vs chunked, on the pure bank)",
        "configs": [
            {"label": "diagonal",         "use_prior_bank": True, "decode_mode": "diagonal"},
            {"label": "diagonal_chunked", "use_prior_bank": True, "decode_mode": "diagonal_chunked"},
        ],
    },
    "decode_chunk_size": {
        "description": "vocab-chunk width for the chunked decode",
        "param": "decode_chunk_size", "values": [4096, 8192, 16384],
        "requires": {"use_prior_bank": True, "decode_mode": "diagonal_chunked"},
    },

    # === cross-block belief handoff ========================================
    "prior_handoff_rho": {
        "description": "mu_q -> mu_p handoff (0 = frozen priors, 1 = full flow)",
        "param": "prior_handoff_rho", "values": [0.0, 0.5, 1.0],
    },
    "prior_handoff_sigma": {
        "description": "sigma_q -> sigma_p handoff damping",
        "param": "prior_handoff_sigma", "values": [0.0, 0.5, 1.0],
    },

    # === normalization =====================================================
    "norm_type_block": {
        "description": "inner (per-block) normalization",
        "param": "norm_type_block", "values": ["none", "mahalanobis"],
    },
    "norm_type_final": {
        "description": "final (outer) normalization",
        "param": "norm_type_final", "values": ["none", "mahalanobis"],
    },

    # === M-step / training =================================================
    "e_step_gradient": {  # 'detach' is consistent only with detach_e_step=False (the baseline)
        "description": "E-step backward estimator",
        "param": "e_step_gradient", "values": ["unroll", "straight_through", "detach"],
    },
    "detach_e_step": {
        "description": "detach the whole E-step (no E-step gradient)",
        "param": "detach_e_step", "values": [False, True],
    },
    "m_mu_lr": {
        "description": "M-step LR for the prior-bank means",
        "param": "m_mu_lr", "values": [0.005, 0.01, 0.025],
    },
    "m_sigma_lr": {
        "description": "M-step LR for the prior-bank variances",
        "param": "m_sigma_lr", "values": [0.001, 0.0021, 0.005],
    },
    "m_phi_lr": {
        "description": "M-step LR for the gauge-frame parameters (phi)",
        "param": "m_phi_lr", "values": [0.0, 0.003, 0.006, 0.009, 0.015],
    },
    "weight_decay": {
        "description": "AdamW weight decay",
        "param": "weight_decay", "values": [0.0, 0.05, 0.1],
    },
    "batch_size": {  # loader-affecting: the runner rebuilds the loader per cell
        "description": "training batch size",
        "param": "batch_size", "values": [32, 64, 128],
    },
    "grad_accum_steps": {
        "description": "gradient-accumulation microbatches per optimizer step",
        "param": "grad_accum_steps", "values": [1, 2, 4],
    },
    "warmup_steps": {
        "description": "LR warmup steps",
        "param": "warmup_steps", "values": [50, 100, 500],
    },
    "amp_dtype": {  # fp16 training would need a GradScaler (deferred); bf16 is the safe arm
        "description": "mixed precision (pure fp32 vs bf16 autocast)",
        "configs": [
            {"label": "fp32", "amp_dtype": None},
            {"label": "bf16", "amp_dtype": "bf16"},
        ],
    },
}


# Fields deliberately NOT swept, with the reason:
#   vocab_size             fixed by the dataset
#   gauge_parameterization only 'phi' is live ('omega_direct' is config-rejected)
#   encode_mode            only 'per_token' is live ('gauge_fixed' is a rejected stub)
#   divergence_family      only 'renyi' is registered (alpha_div is its live knob)
#   seed                   set per run from CONFIG['seed'] (the runner reseeds each cell)
#   max_steps              run length, set via CONFIG['max_steps']
#   log/eval/checkpoint_interval, eval_max_batches   bookkeeping, not model behavior
# (decode_mode='full' is a valid value left OUT of the decode_mode sweep: on the prior-bank path it
#  drives the full-covariance SPD retraction's eigh to non-convergence -- a deferred robust-eigh
#  issue, separate from the now-fixed full-cov KL Cholesky.)
NON_SWEPT_FIELDS = (
    "vocab_size", "gauge_parameterization", "encode_mode", "divergence_family", "seed",
    "max_steps", "log_interval", "eval_interval", "checkpoint_interval", "eval_max_batches",
)


# Which sweeps run (and in what order) when CONFIG["sweep"] is None. This is a CURATED subset of
# the full SWEEPS registry above (every key in SWEEPS is also runnable on its own via
# CONFIG["sweep"]="<name>"); add or remove names to shape a session. Cheap-to-expensive is a good
# ordering for a single GPU. `mode="list"` (with sweep=None) prints every registered sweep.
SWEEP_ORDER: List[str] = [
    "kappa",
    "alpha_mode",
    "attention_prior",
    "entropy_term",
    "decode_head",
    "gauge_group",
    "phi_precond_mode",
    "covariance",
]


# =============================================================================
# CLICK-TO-RUN KNOBS  -- edit, then run.
# =============================================================================
CONFIG: Dict[str, Any] = {
    # Action: 'train' (run sweeps), 'analyze' (print tables), 'plot' (figures), 'list'.
    "mode":        "train",

    # One sweep name, or None -> every sweep in SWEEP_ORDER.
    "sweep":       None,

    # 'auto' picks CUDA when present (the RTX 5090), else CPU.
    "device":      "auto",

    # Dataset for every run in the session (NOT a VFE3Config field; the loader seam).
    #   "wikitext-103" | "wikitext-2" | "wiki-en" | "wiki-ja" | "synthetic-period3"
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
    all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in spec)
    values: List[Union[int, float]] = []
    tol = abs(step) * 1e-9
    n = int(round((stop - start) / step))
    for i in range(n + 2):
        v = start + i * step
        if (step > 0 and v > stop + tol) or (step < 0 and v < stop - tol):
            break
        values.append(v if all_int else round(v, 10))
    return values


def _sweep_values(sweep: Dict[str, Any]) -> List[Any]:
    if "values" in sweep:
        return list(sweep["values"])
    if "range" in sweep:
        return _expand_range(sweep["range"])
    raise KeyError(f"single-field sweep must define 'values' or 'range': {sweep!r}")


def sweep_n_runs(sweep: Dict[str, Any]) -> int:
    return len(sweep["configs"]) if "configs" in sweep else len(_sweep_values(sweep))


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
    seed:        int           = 0,
) -> Any:
    r"""DataLoader for ``dataset``/``split``, falling back to the synthetic stream if absent.

    Memoised on ``(dataset, seq_len, batch_size, split, cap)`` so runs that do not change
    those reuse one cached loader (the corpus cache loads once), while a sweep over
    ``batch_size`` / ``max_seq_len`` correctly builds a distinct, matching loader. ``max_tokens``
    caps only the train split (validation is always full).
    """
    cap = max_tokens if split == "train" else None
    key = (dataset, seq_len, batch_size, split, cap)
    if key in _LOADER_CACHE:
        return _LOADER_CACHE[key]
    if dataset == "synthetic-period3":
        loader = synthetic_period3_loader(seq_len=seq_len, batch_size=batch_size, seed=seed)
    else:
        try:
            loader = make_dataloader(dataset, split, seq_len, batch_size, max_tokens=cap)
        except FileNotFoundError:
            logger.warning("cache for %r/%r absent; falling back to synthetic-period3", dataset, split)
            loader = synthetic_period3_loader(seq_len=seq_len, batch_size=batch_size, seed=seed)
    _LOADER_CACHE[key] = loader
    return loader


# =============================================================================
# SINGLE-RUN EXECUTOR
# =============================================================================

def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def run_single(
    label:       str,
    overrides:   Dict[str, Any],
    run_dir:     Path,

    *,
    dataset:     str,
    device:      torch.device,
    seed:        int,
    max_tokens:  Optional[int] = None,
    max_steps:   Optional[int] = None,
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

    _seed_everything(cfg.seed)
    model = VFEModel(cfg).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))

    train_loader = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "train",
                              max_tokens=max_tokens, seed=cfg.seed)
    val_loader   = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "validation",
                              seed=cfg.seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(run_dir, cfg, model, dataset=dataset, device=device)

    # Reproducible, ORDER-INDEPENDENT data stream. Model construction above consumes a
    # config-dependent amount of RNG, and a cached loader's shuffle (its own generator, or the
    # global RNG for make_dataloader) otherwise advances across runs -- so without this reseed
    # the same config would see different batches depending on its position in the sweep, and
    # the comparison would be confounded by data order. Reseeding here, after the model is built,
    # pins every cell to the same batch sequence regardless of order.
    _seed_everything(cfg.seed)
    for loader in (train_loader, val_loader):                # synthetic loaders carry their own generator
        if getattr(loader, "generator", None) is not None:
            loader.generator.manual_seed(cfg.seed)

    print(f"    K={cfg.embed_dim} heads={cfg.n_heads} group={cfg.gauge_group} "
          f"family={cfg.family} | steps={cfg.max_steps} batch={cfg.batch_size} | {n_params:,} params")

    losses = train(
        model, train_loader, cfg,
        n_steps=cfg.max_steps,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=val_loader,
        device=device,
        logger=logger,
        artifacts=artifacts,
        generate_samples=False,                              # pure silent path: no sample text
    )

    # Unconditional final validation pass: guarantees a number even when max_steps is below
    # eval_interval (a periodic eval never fired). best_val_ppl is the lowest the periodic
    # eval saw (inf if none); the headline takes the better of the two.
    m = evaluate(model, val_loader, device=device)
    best = artifacts.best_val_ppl
    primary = min(best, m["ppl"]) if best != float("inf") else m["ppl"]

    return {
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
    }


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
    "n_params", "wall_time_s", "seed", "error",
]


def _cell_is_current(
    run_dir:    Path,
    overrides:  Dict[str, Any],

    *,
    seed:       int,
    max_steps:  Optional[int] = None,
) -> bool:
    r"""True iff a completed cell's persisted config.json matches the config we would build now.

    Guards resume against baseline drift: ``ablation_result.json`` is keyed only by the
    ``param=value`` label, which does NOT encode the imported ``train_vfe3`` baseline. Editing
    an unrelated baseline field (e.g. ``embed_dim``) would otherwise let a stale result be
    served as current. A cell is skipped only when its saved VFE3Config equals the freshly
    built one (config-error cells have no config.json, so they are always re-run -- cheap).
    """
    cj = run_dir / "config.json"
    if not cj.exists():
        return False
    try:
        built = json.loads(json.dumps(asdict(VFE3Config(
            **_cell_cfg_dict(overrides, seed=seed, max_steps=max_steps))), default=str))
        saved = json.loads(cj.read_text(encoding="utf-8")).get("config")
    except Exception:                                        # unbuildable now / unreadable -> re-run
        return False
    return saved == built


def _sanitize(label: str) -> str:
    r"""A filesystem-safe single path component (no separators, parent tokens, or drive colon)."""
    out = label
    for bad, repl in (("=", "_"), (" ", "_"), ("/", "_"), ("\\", "_"), ("..", "_"), (":", "_")):
        out = out.replace(bad, repl)
    return out.lstrip("._") or "_"


def _write_sweep_csv(sweep_dir: Path, results: List[Dict[str, Any]]) -> None:
    r"""Rewrite ``sweep_results.csv`` as the complete frame (fixed columns; missing keys blank)."""
    with open(sweep_dir / "sweep_results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in _CSV_COLUMNS})


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

    print(f"\n{'=' * 70}\nSWEEP: {sweep_name} ({len(runs)} runs)\n  {sweep['description']}"
          f"\n  Output: {sweep_dir}{'  [resume ON]' if resume else ''}\n{'=' * 70}")

    results: List[Dict[str, Any]] = []
    for i, (label, overrides) in enumerate(runs):
        run_dir = sweep_dir / _sanitize(label)
        run_dir.mkdir(parents=True, exist_ok=True)
        marker = run_dir / "ablation_result.json"

        if resume and marker.exists():
            if _cell_is_current(run_dir, overrides, seed=seed, max_steps=max_steps):
                print(f"\n--- {i + 1}/{len(runs)}: {label}  [CACHED] ---")
                results.append(json.loads(marker.read_text(encoding="utf-8")))
                continue
            print(f"\n--- {i + 1}/{len(runs)}: {label}  [config changed -> re-running] ---")
        else:
            print(f"\n--- {i + 1}/{len(runs)}: {label} ---")
        t0 = time.perf_counter()
        try:
            result = run_single(label, overrides, run_dir, dataset=dataset, device=device,
                                 seed=seed, max_tokens=max_tokens, max_steps=max_steps)
        except Exception as exc:                             # a training crash must not kill the sweep
            logger.exception("sweep %s / %s crashed", sweep_name, label)
            result = {"label": label, "error_kind": "train", "error": str(exc),
                      "primary_val_ppl": float("inf"), "seed": int(seed),
                      "overrides": _jsonable(overrides)}
        finally:
            _cleanup()

        result["sweep"] = sweep_name
        result["wall_time_s"] = time.perf_counter() - t0
        marker.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        results.append(result)

        ppl = result["primary_val_ppl"]
        tag = f" [{result['error_kind'].upper()}]" if result.get("error_kind") else ""
        print(f"  -> val PPL {ppl:.3f}{tag}  ({result['wall_time_s']:.0f}s)")
        if i == 0 and len(runs) > 1:
            est = result["wall_time_s"] * len(runs)
            print(f"  ** ~{est / 60:.0f} min estimated for the full {len(runs)}-run sweep")

        _write_sweep_csv(sweep_dir, results)               # keep the CSV whole after each cell

    (sweep_dir / "sweep_meta.json").write_text(json.dumps({
        "sweep_name":  sweep_name,
        "description": sweep["description"],
        "n_runs":      len(runs),
        "dataset":     dataset,
        "seed":        seed,
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")

    finished = [r for r in results if r.get("primary_val_ppl", float("inf")) < float("inf")]
    if finished:
        best = min(finished, key=lambda r: r["primary_val_ppl"])
        print(f"\nSWEEP COMPLETE: {sweep_name}  ->  best {best['label']} "
              f"(val PPL {best['primary_val_ppl']:.3f})")
    else:
        print(f"\nSWEEP COMPLETE: {sweep_name}  ->  no successful run")
    return results


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


def analyze_all(output_dir: Path) -> None:
    print(f"\n{'=' * 70}\nABLATION SUMMARY  ({output_dir})\n{'=' * 70}")
    sweep_dirs = [d for d in sorted(output_dir.iterdir())
                  if d.is_dir() and (d / "sweep_results.csv").exists()]
    if not sweep_dirs:
        print("No completed sweeps found.")
        return
    for d in sweep_dirs:
        analyze_sweep(d)

    print(f"\n{'=' * 70}\nBEST PER SWEEP\n{'=' * 70}")
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

def generate_plots(output_dir: Path) -> None:
    r"""Per-sweep PPL line/bar figures plus a cross-sweep sensitivity (PPL-range) summary."""
    try:
        import matplotlib.pyplot as plt
        from vfe3.viz.figures import set_publication_style
        set_publication_style()
    except Exception as exc:                                 # plotting is best-effort, never fatal
        print(f"plotting unavailable ({exc}); skipping figures")
        return

    sweep_dirs = [d for d in sorted(output_dir.iterdir())
                  if d.is_dir() and (d / "sweep_results.csv").exists()]
    if not sweep_dirs:
        print("No sweeps to plot.")
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    sensitivity: List[Tuple[str, float, str]] = []           # (sweep, ppl range, best label)
    for d in sweep_dirs:
        rows = [r for r in _read_sweep_csv(d) if _as_float(r.get("primary_val_ppl")) < float("inf")]
        if not rows:
            continue
        labels = [r["label"] for r in rows]
        ppls = [_as_float(r["primary_val_ppl"]) for r in rows]

        # Numeric param=value labels -> line plot; categorical arms -> sorted bar plot.
        numeric = []
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
            ax.set_xlabel(d.name)
        else:
            order = sorted(range(len(ppls)), key=lambda k: ppls[k])
            ax.barh(range(len(order)), [ppls[k] for k in order],
                    color=["#2ca02c" if j == 0 else "#1f77b4" for j in range(len(order))])
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([labels[k] for k in order])
            ax.invert_yaxis()
        ax.set_ylabel("validation PPL")
        ax.set_title(d.name)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{d.name}.png")
        plt.close(fig)

        best = min(rows, key=lambda r: _as_float(r["primary_val_ppl"]))
        sensitivity.append((d.name, max(ppls) - min(ppls), best["label"]))

    if sensitivity:
        sensitivity.sort(key=lambda t: t[1], reverse=True)
        fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(sensitivity))))
        ax.barh(range(len(sensitivity)), [s[1] for s in sensitivity], color="#d62728", alpha=0.8)
        ax.set_yticks(range(len(sensitivity)))
        ax.set_yticklabels([f"{s[0]}\n(best: {s[2]})" for s in sensitivity])
        ax.invert_yaxis()
        ax.set_xlabel("validation PPL range (worst - best)")
        ax.set_title("hyperparameter sensitivity")
        fig.tight_layout()
        fig.savefig(fig_dir / "sensitivity_summary.png")
        plt.close(fig)
    print(f"figures -> {fig_dir}")


# =============================================================================
# MAIN  (click-to-run; edit CONFIG above)
# =============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mode = CONFIG["mode"]
    output_dir = Path(CONFIG["output_dir"])

    if mode == "list":
        # Every registered sweep (CONFIG["sweep"]=None) or just the named one; an asterisk marks
        # those in the curated SWEEP_ORDER that a None-sweep `train` run would execute.
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

    if mode == "analyze":
        analyze_all(output_dir)
        return
    if mode == "plot":
        generate_plots(output_dir)
        return
    if mode != "train":
        raise ValueError(f"CONFIG['mode']={mode!r} not in {{'train','analyze','plot','list'}}")

    # ---- train mode --------------------------------------------------------
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
    print(f"\nVFE_3.0 ablation suite\n  device:  {device}\n  dataset: {CONFIG['dataset']}"
          f"\n  output:  {output_dir}\n  seed:    {CONFIG['seed']}"
          f"\n  sweeps:  {', '.join(sweep_names)}")

    for name in sweep_names:
        run_sweep(name, output_dir, dataset=CONFIG["dataset"], device=device,
                  seed=CONFIG["seed"], resume=CONFIG["resume"],
                  max_tokens=CONFIG["max_tokens"], max_steps=CONFIG["max_steps"])
        generate_plots(output_dir)                           # refresh figures after each sweep

    analyze_all(output_dir)


if __name__ == "__main__":
    main()
