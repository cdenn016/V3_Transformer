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
if os.environ.get("VFE3_ALLOW_DUPLICATE_OPENMP") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import ast
import copy
import csv
import gc
import hashlib
import json
import logging
import math
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from vfe3.config import VFE3Config
from vfe3.data.datasets import (
    _sha256_file,
    _tokenizer_tag,
    cache_source_identity,
    make_dataloader,
    tokens_per_char,
)
from vfe3.metrics import (
    attention_entropy,
    gauge_equivariance_residual,
    guard_saturation,
    head_mixer_gauge_residual,
    rank_one_residual,
)
from vfe3.model.model import VFEModel
from vfe3.path_utils import filesystem_slug, portable_path_component_key
from vfe3.process_utils import run_process_tree
from vfe3.run_artifacts import (
    RunArtifacts,
    _atomic_replace,
    _pure_path_report,
    _unique_sibling_temp,
    _verified_process_code_identity,
    _write_json_atomic,
    finalize_validation_run,
    semantic_config_fingerprint,
)
from vfe3.runtime import seed_everything
from vfe3.train import coverage_lines, evaluate, train
from vfe3.viz.extract import (
    across_layer_belief_trace,
    attention_entropy_cov_gap,
    converged_state,
    per_unit_eval_nats,
)
logger = logging.getLogger("ablation")
_ABLATION_FIGURE_TIMEOUT_SECONDS = 2 * 60 * 60
_RESERVED_SWEEP_NAMES = {"figures", "__sensitivity__"}
_RESERVED_SWEEP_KEYS = {
    portable_path_component_key(name, field="reserved sweep name")
    for name in _RESERVED_SWEEP_NAMES
}
_DECLARATIVE_ABLATION_NAMES = {
    "BASELINE_CONFIG",
    "CONFIG",
    "DATA_SEED",
    "SWEEPS",
    "SWEEP_ORDER",
}


def _assignment_root_name(target: ast.expr) -> Optional[str]:
    """Return the root name assigned by one top-level assignment target."""
    while isinstance(target, (ast.Attribute, ast.Subscript)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _is_declarative_ablation_statement(statement: ast.stmt) -> bool:
    """Whether a top-level statement changes only registry/config data bound elsewhere."""
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        return any(
            _assignment_root_name(target) in _DECLARATIVE_ABLATION_NAMES
            for target in targets
        )
    return False


def _ablation_runner_source_sha256() -> str:
    r"""Hash executable runner logic while excluding separately contracted declarations."""
    source_path = Path(__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    tree.body = [
        statement
        for statement in tree.body
        if not _is_declarative_ablation_statement(statement)
    ]
    payload = ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_PROCESS_ABLATION_RUNNER_SHA256 = _ablation_runner_source_sha256()


def _verified_ablation_runner_source_sha256() -> str:
    """Return the import-time runner identity or fail after executable runner logic changes."""
    current = _ablation_runner_source_sha256()
    if current != _PROCESS_ABLATION_RUNNER_SHA256:
        raise RuntimeError(
            "ablation runner logic changed after this Python process imported it; restart the "
            "Spyder kernel before starting another sweep"
        )
    return _PROCESS_ABLATION_RUNNER_SHA256


def _git_code_identity() -> Dict[str, object]:
    r"""Return the exact imported-package and runner-logic identity for ablation contracts.

    The legacy name remains as a monkeypatch seam. Sweep values, order, baseline, output policy, and
    data seed are excluded here because their effective values are already bound explicitly by the
    aggregation or per-cell contract. This lets an edit that only adds a sweep value tack onto the
    prior compatible cohort without treating the declaration edit itself as model-code drift.
    """
    return {
        "package_code_sha256":  _verified_process_code_identity(),
        "ablation_runner_sha256": _verified_ablation_runner_source_sha256(),
    }


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
    
    batch_size                = 32,
    max_steps                 = 30000,
    
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
    pos_phi_compose           = "bch",               # composition: "bch" | "euclidean" | exact "group_product"
               
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
    phi_mstep_max_matrix_norm = None,    # opt-in post-M-step projection bound; None leaves the chart unbounded

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

    "estep_depth_damping": {
        "description": "fixed versus randomized MM-exact inference depth and damping",
        "collect_diagnostics": True,
        "seeds": [6, 64, 23],
        "requires": {
            "e_step_update": "mm_exact",
            "e_steps_backprop_last": 0,
            "e_step_halt_tol": None,
        },
        "configs": [
            {"label": "fixed_T1_eta1.00", "n_e_steps": 1, "mm_damping": 1.0,
             "randomize_e_steps": False},
            {"label": "fixed_T3_eta1.00", "n_e_steps": 3, "mm_damping": 1.0,
             "randomize_e_steps": False},
            {"label": "fixed_T5_eta1.00", "n_e_steps": 5, "mm_damping": 1.0,
             "randomize_e_steps": False},
            {"label": "fixed_T5_eta0.75", "n_e_steps": 5, "mm_damping": 0.75,
             "randomize_e_steps": False},
            {"label": "random_T1-5_evalT5_eta1.00", "n_e_steps": 5, "mm_damping": 1.0,
             "randomize_e_steps": True, "e_steps_min": 1, "e_steps_max": 5},
            {"label": "random_T1-5_evalT5_eta0.75", "n_e_steps": 5, "mm_damping": 0.75,
             "randomize_e_steps": True, "e_steps_min": 1, "e_steps_max": 5},
        ],
    },

    "phi_chart_control": {
        "description": "matched phi chart regularization, step scale, pullback descent, and projection",
        "collect_diagnostics": True,
        "seeds": [6, 64, 23],
        "requires": {
            "gauge_parameterization": "phi",
            "e_phi_lr": 0.0,
            "mass_phi": 0.0,
            "m_phi_update_mode": "adamw",
            "phi_precond_mode": "pullback_per_block",
            "phi_mstep_max_matrix_norm": None,
        },
        "configs": [
            {"label": "adamw_unbounded", "m_phi_lr": 0.01},
            {"label": "adamw_mass0.01", "m_phi_lr": 0.01, "mass_phi": 0.01},
            {"label": "adamw_lr0.003", "m_phi_lr": 0.003},
            {"label": "pullback_group_lr0.0015", "m_phi_lr": 0.0015,
             "m_phi_update_mode": "pullback_group", "transport_chart_max_norm": 6.0},
            {"label": "adamw_projected_norm5", "m_phi_lr": 0.01,
             "phi_mstep_max_matrix_norm": 5.0},
        ],
    },

    "pos_phi_composition": {
        "description": "truncated BCH versus exact group product versus no positional gauge factor",
        "collect_diagnostics": True,
        "seeds": [6, 64, 23],
        "requires": {
            "gauge_parameterization": "phi",
            "transport_mode": "flat",
            "s_frame_mode": "tied",
            "pos_rotation": "none",
        },
        "configs": [
            {"label": "bch", "pos_phi": "learned", "pos_phi_compose": "bch"},
            {"label": "group_product", "pos_phi": "learned", "pos_phi_compose": "group_product"},
            {"label": "none", "pos_phi": "none", "pos_phi_compose": "bch"},
        ],
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
            {"label": "tied_block_glk", "gauge_group": "tied_block_glk",
                                          "phi_precond_mode": "killing"},
            {"label": "glk",            "gauge_group": "glk",  "use_head_mixer": False},
            {"label": "so_k",           "gauge_group": "so_k", "use_head_mixer": False},
            {"label": "sp",             "gauge_group": "sp",   "use_head_mixer": False},
            {"label": "so3_spin2x4",    "gauge_group": "so_n", "group_n": 3, "n_heads": 4,
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
            {"label": "regime_ii",           "transport_mode": "regime_ii",
                                                "e_step_update": "gradient"},
            {"label": "regime_ii_covariant", "transport_mode": "regime_ii_covariant",
                                                "e_step_update": "gradient"},
        ],
    },
    
    
    "cocycle_relaxation": {
        "description": "regime_ii homotopy (0 -> flat, 1 -> fully relaxed)",
        "param": "cocycle_relaxation", "values": [0.0, 0.5, 1.0],
        "requires": {"transport_mode": "regime_ii", "e_step_update": "gradient"},
    },
    
    
    "cross_couplings": {  # the off-block coupling merges the heads into one super-block, so the
                          # >=2-block head mixer cannot apply -> turn it off for a clean comparison
        "description": "cross-head GL(K) coupling (block-diagonal vs one coupled pair)",
        "configs": [
            {"label": "none",     "cross_couplings": None},
            {"label": "pair_0_1", "cross_couplings": [(0, 1)],
                                  "beta_attention_prior": "causal_noself",
                                  "gamma_attention_prior": "causal_noself"},
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
        "required_diagnostics": ["cov_gap"],
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
        "required_diagnostics": ["builder_resid"],
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
        # lambda_alpha_mode pinned 'constant' on every arm: the baseline's 'state_dependent_per_coord'
        # needs a per-coordinate self-divergence, which exists only for a diagonal-covariance family
        # (vfe3/config.py's alpha_is_per_coord/family_is_diagonal guard), but every arm here is
        # family='gaussian_full'. 'constant' is the pure default (VFE3Config.lambda_alpha_mode) and is
        # valid for any family, so it is the minimal, theory-neutral override that keeps the
        # tied-vs-untied gauge-equivariance contrast the only difference between arms.
        "configs": [
            {"label": "untied_block_glk", "gauge_group": "block_glk", "use_head_mixer": True,
             "family": "gaussian_full", "use_prior_bank": True, "decode_mode": "full_chunked",
             "phi_precond_mode": "killing", "s_e_step": False, "precision_weighted_attention": False,
             "lambda_alpha_mode": "constant"},
            {"label": "tied_block_glk",   "gauge_group": "tied_block_glk", "use_head_mixer": True,
             "family": "gaussian_full", "use_prior_bank": True, "decode_mode": "full_chunked",
             "phi_precond_mode": "killing", "s_e_step": False, "precision_weighted_attention": False,
             "lambda_alpha_mode": "constant"},
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
             "embed_dim": 28, "kl_max": 224, "lambda_alpha_mode": "constant"},
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
        # AdamW-on-phi versus stateless pullback group descent. e_phi_lr=0 keeps the
        # preconditioner off the E-step so this measures the M-step only.
        "description": "gauge M-step: AdamW vs pullback group descent [D1/EXP-8]",
        "requires": {"e_phi_lr": 0.0},
        "configs": [
            {"label": "adamw", "m_phi_update_mode": "adamw"},
            {"label": "pullback_group", "m_phi_update_mode": "pullback_group",
             "phi_precond_mode": "pullback_per_block", "gauge_group": "block_glk",
             "embed_dim": 10, "n_heads": 2, "e_phi_lr": 0.0,
             "transport_chart_max_norm": 6.0},
        ],
    },

    "m_phi_lr_pullback_group": {  # D1 / EXP-8: the pullback-group LR sub-experiment.
        # Group descent bypasses Adam's per-coordinate normalization, so the AdamW-tuned
        # m_phi_lr need not transfer. Gated to the certified pullback-group path.
        "description": "log-spaced m_phi_lr on the pullback group M-step [D1/EXP-8]",
        "param": "m_phi_lr", "values": [0.0005, 0.0015, 0.005, 0.015, 0.05, 0.15],
        "requires": {"m_phi_update_mode": "pullback_group",
                     "phi_precond_mode": "pullback_per_block", "e_phi_lr": 0.0,
                     "transport_chart_max_norm": 6.0},
    },

    "mass_phi": {  # D1 / EXP-8: the regime knob (NOT phi_weight_decay, which is hard-zeroed under
        # pullback group descent). The pullback advantage is predicted to shrink as mass_phi rises (the frame-
        # norm penalty pulls phi toward 0, where ad_phi -> 0 and the pullback metric -> I).
        "description": "mass_phi frame-norm penalty: pullback-group regime knob [D1/EXP-8]",
        "param": "mass_phi", "values": [0.0, 0.001, 0.01, 0.1],
        "requires": {"m_phi_update_mode": "pullback_group",
                     "phi_precond_mode": "pullback_per_block", "e_phi_lr": 0.0,
                     "transport_chart_max_norm": 6.0},
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
        "required_diagnostics": ["rank_resid_by_layer"],
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
        "min_extrapolation_points": 2,
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
            {"label": "rope", "pos_rotation": "rope", "e_step_update": "gradient"},
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
        "requires": {"pos_rotation": "rope", "e_step_update": "gradient"},
    },
    "rope_full_gauge": {  # rotating the covariance sandwich needs full covariance
        "description": "RoPE means-only vs full-gauge (rotates covariance; needs full cov)",
        "configs": [
            {"label": "means_only", "pos_rotation": "rope", "e_step_update": "gradient"},
            {"label": "full_gauge", "pos_rotation": "rope", "rope_full_gauge": True,
                                    "e_step_update": "gradient",
                                    "family": "gaussian_full",
                                    "lambda_alpha_mode": "state_dependent"},
        ],
    },

    

    # === belief family =====================================================
    # The full arm flips family (which derives diagonal_covariance) and moves off the per-coordinate
    # alpha form (diagonal-only), both of which a naive single-field sweep would have rejected.
    "covariance": {
        "description": "belief covariance structure (diagonal vs full Gaussian)",
        "configs": [
            {"label": "diagonal", "family": "gaussian_diagonal"},
            {"label": "full",     "family": "gaussian_full", "e_step_update": "gradient",
                                  "lambda_alpha_mode": "state_dependent"},
        ],
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
        "configs": [
            {"label": "renyi_order=0.5", "renyi_order": 0.5, "e_step_update": "gradient"},
            {"label": "renyi_order=0.8", "renyi_order": 0.8, "e_step_update": "gradient"},
            {"label": "renyi_order=1.0", "renyi_order": 1.0},
            {"label": "renyi_order=1.2", "renyi_order": 1.2, "e_step_update": "gradient"},
            {"label": "renyi_order=1.5", "renyi_order": 1.5, "e_step_update": "gradient"},
            {"label": "renyi_order=2.0", "renyi_order": 2.0, "e_step_update": "gradient"},
        ],
        "requires": {"oracle_unroll_grad": True},
        "collect_diagnostics": True,
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


    "e_sigma_q_trust": {
        "description": "E-step SPD retraction trust radius",
        "param": "e_sigma_q_trust", "values": [10.0, 15],
    },
    
    
    "sigma_max": {  # E1 / EXP-20: does the SPD variance ceiling ever bind? (read guard_sigma_ceil_frac)
        "description": "SPD variance ceiling sigma_max (binding vs slack) [E1/EXP-20]",
        "param": "sigma_max", "values": [15],
    },
    
    
    
    
    
    

    # === free-energy coupling ==============================================
    
    
    
    
   
    
    
    
    


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
        "param": "mu_init_std", "values": [0.01, 0.04, 0.065, 0.075, 0.1],
    },
    
    "sigma_init": {
        "description": "constant initial coordinate variance of the prior table (>0)",
        "param": "sigma_init", "values": [4, 4.25],
    },
    
    "phi_scale": {
        "description": "init std of the gauge-frame table phi_embed ~ N(0, std^2)",
        "param": "phi_scale", "values": [0.01, 0.03, 0.05, 0.06, 0.07, 0.1, 0.15, 0.2],
    },
    

    
    
    
    
    
    
    
    "lambda_alpha": {
        "description": "constant self-coupling value (lambda_alpha_mode=constant)",
        "param": "lambda_alpha", "values": [0.0, 0.25, 0.5, 0.75, 1, 2.5, 5], "requires": {"lambda_alpha_mode": "constant"},
    },
    
    
    "lambda_beta": {
        "description": "belief-coupling block weight (1.0 = pure F)",
        "param": "lambda_beta", "values": [0, 0.5, 0.75, 1, 2],
    },
    
    
    
    "lambda_gamma": {
        "description": "model-channel coupling weight (>0 creates s tables)",
        "param": "lambda_gamma", "values": [0.5, 0.7, 0.75, 0.8],
    },
    
    
    "lambda_h": {
        "description": "hyper-prior weight lambda_h * mean_i KL(s_i||r) (>0 creates s/r tables)",
        "param": "lambda_h", "values": [0.2, 0.225, 0.25, 0.275],
    },
    
    
    
    
    
    
    
    
    
    "kappa_gamma": {
        "description": "model-channel temperature tau_gamma = kappa_gamma * sqrt(d_head)",
        "param": "kappa_gamma", "values": [0.9, 1, 1.1], 
    },
    
    "kappa_beta": {
       "description": "attention temperature tau = kappa * sqrt(d_head)",
       "param": "kappa_beta", "values": [0.9, 1, 1.1],
    },
    
    "decode_tau": {
        "description": "KL-to-prior decode temperature",
        "param": "decode_tau", "values": [0.007, 0.008, 0.009], "requires": {"use_prior_bank": True},
    },
    
    

    
    
    
    
    "e_s_mu_lr": {
       "description": "E-step natural-gradient step size for mu_s",
       "param": "e_s_mu_lr", "values": [0.8, 0.85, 0.9],
    },
    
    "e_s_sigma_lr": {
       "description": "E-step retraction step size for sigma_s",
       "param": "e_s_sigma_lr", "values": [0, 0.005, 0.01, 0.05, 0.1],
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
        "param": "m_p_mu_lr", "values": [0.014, 0.015, 0.016, 0.018],
    },
    
    "m_p_sigma_lr": {
        "description": "M-step LR for the prior-bank variances",
        "param": "m_p_sigma_lr", "values": [0.008, 0.009, 0.01, 0.011, 0.012],
    },
    
    "m_phi_lr": {
        "description": "M-step LR for the gauge-frame parameters (phi)",
        "param": "m_phi_lr", "values": [0.009, 0.0095, 0.01, 0.0105],
    },
    

    "s_frame_mode": {
        "description": "model-channel gauge frame: tied belief frame vs independent phi_tilde",
        "param": "s_frame_mode", "values": ["tied", "phi_tilde"],
        "requires": {
            "gauge_parameterization": "phi",
            "phi_reflection": "off",
            "pos_rotation": "none",
            "s_e_step": True,
            "prior_source": "model_channel",
            "share_refine_s_transport": False,
        },
    },

    "m_s_phi_lr": {
        "description": "M-step LR for the independent model-channel gauge frame (phi_tilde)",
        "param": "m_s_phi_lr", "values": [0.005, 0.0065, 0.007, 0.0075, 0.0085, 0.009, 0.01],
        "requires": {
            "gauge_parameterization": "phi",
            "phi_reflection": "off",
            "pos_rotation": "none",
            "s_e_step": True,
            "prior_source": "model_channel",
            "share_refine_s_transport": False,
            "s_frame_mode": "phi_tilde",
        },
    },
    
    
    
    
   
    
    
    
    
    
    
    
    
    "weight_decay": {
        "description": "AdamW weight decay",
        "param": "weight_decay", "values": [0.015, 0.025],
    }, 
   
    
    
    "phi_weight_decay":{
        "description": "weight decay on phi",
        "param": "phi_weight_decay", "values": [0.02, 0.03, 0.035, 0.04, 0.05],
    },

    "sigma_weight_decay": {  # separate AdamW weight decay for the log-variance tables (None = inherit
        # weight_decay). Numeric radii -> LINE plot; the None (inherit) baseline runs via train_vfe3.
        "description": "AdamW weight decay on the log-variance (sigma) tables",
        "param": "sigma_weight_decay", "values": [0, 0.01, 0.02, 0.035, 0.05],
    },







    "mm_damping": {  # MM-exact E-step damping eta in (0,1]; 'requires' pins e_step_update='mm_exact' so the
        # damped coordinate-minimizer step is actually taken every cell (1.0 = full exact minimizer).
        "description": "MM-exact E-step damping eta in (0,1] (requires mm_exact E-step)",
        "param": "mm_damping", "values": [0.8, 0.85],
        "requires": {"e_step_update": "mm_exact"},
    },

    
    
    "query_tau_c": {  # query-adaptive temperature strength c >= 0 (0 = inert); 'requires' forces
        # query_adaptive_tau=True so tau_i = tau_h (1 + c * tr_h Sigma_i / d_h) is live on every cell.
        "description": "query-adaptive temperature strength c (requires query_adaptive_tau=True)",
        "param": "query_tau_c", "values": [0, 0.5, 0.75, 1.0, 2.0, 4.0],
        "requires": {"query_adaptive_tau": True},
    },

    
    
    "lambda_twohop": {  # two-hop coupling F2 = lam2 sum_ik (beta@beta)_ik KL_ik (0 = OFF = pure canonical F;
        # effective depth 2 at L=1). Numeric values -> _plot_one_sweep draws the x-sorted LINE plot.
        "description": "two-hop coupling weight lambda_twohop (0 = OFF)",
        "param": "lambda_twohop", "values": [0.0, 0.001, 0.005, 0.01],
    },

      "gamma_prior_weight": {
          "description": "gamma_prior_weight",
          "param": "gamma_prior_weight", "values": [0.4, 0.45, 0.55, 0.6],
      },

    "warmup_steps": {  # LR warmup length before the cosine decay (0 = no warmup, straight into cosine).
        "description": "LR warmup steps before cosine decay",
        "param": "warmup_steps", "values": [50, 150, 200, 1500],
    },


}


# ---- PB-07 report sweeps: opt-in, NOT in SWEEP_ORDER, values DERIVED from the live entries above ----
# component_ablation_forest is a single-seed multi-arm sweep that publishes per-cell paired-token nats
# (paired_token_bootstrap=True) so ablation_forest_kwargs can plot a within-run bootstrap band of the
# ablation delta in bits/token against the "baseline" arm. e_q_mu_sigma_lr_grid is the Cartesian
# product of the two one-dimensional E-step learning-rate sweeps plus their baseline operating point,
# so lr_grid_heatmap_kwargs can expose ridge interactions the 1-D slices cannot. Both are derived from
# the live one-dimensional entries (not copied), so they track a future edit to those value lists.
_GRID_MU_LRS = sorted(set([
    *SWEEPS["e_q_mu_lr"]["values"],
    BASELINE_CONFIG["e_q_mu_lr"],
]))
_GRID_SIGMA_LRS = sorted(set([
    *SWEEPS["e_q_sigma_lr"]["values"],
    BASELINE_CONFIG["e_q_sigma_lr"],
]))

SWEEPS["component_ablation_forest"] = {
    "description": "paired-token component ablation forest",
    "configs": [
        {"label": "baseline"},
        {"label": "head_mixer_off", "use_head_mixer": False},
        {"label": "precision_attention_off", "precision_weighted_attention": False},
    ],
    "paired_token_bootstrap": True,
    "forest_baseline_label": "baseline",
}
SWEEPS["e_q_mu_sigma_lr_grid"] = {
    "description": "joint q-mean and q-covariance E-step learning-rate grid",
    "configs": [
        {"label": f"mu={mu:g},sigma={sigma:g}",
         "e_q_mu_lr": mu, "e_q_sigma_lr": sigma}
        for sigma in _GRID_SIGMA_LRS
        for mu in _GRID_MU_LRS
    ],
    "grid_x": "e_q_mu_lr",
    "grid_y": "e_q_sigma_lr",
    "grid_x_values": _GRID_MU_LRS,
    "grid_y_values": _GRID_SIGMA_LRS,
    "grid_baseline": (BASELINE_CONFIG["e_q_mu_lr"], BASELINE_CONFIG["e_q_sigma_lr"]),
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
  #"component_ablation_forest",
  #"gauge_group",
  #"transport_mode",
 # "gauge_equivariance",
  "pos_extrapolation",
  "estep_depth_damping",
  #"e_q_mu_sigma_lr_grid",


  #"m_s_phi_lr",
  #"gamma_prior_weight",
  # "m_phi_lr",
  #"weight_decay",
  #"sigma_init",
 # "s_frame_mode",


  # "m_p_mu_lr",
  # "m_p_sigma_lr",

   #"sigma_weight_decay",

   #"phi_weight_decay",


    #"mu_init_std",
    #"phi_scale",


  # "mm_damping",
  # "query_tau_c",
   #"lambda_h",
   #"lambda_gamma",




   # "lambda_beta",

  #"gauge_transport",
 # "attention_entropy",
 # "gauge_equivariance",
  #"cg_coupling",
 # "fisher_mu_precond",

 # "n_e_steps_em",
 # "gauge_mstep_optim",

  #"m_phi_lr_pullback_group",   not run

 # "pos_extrapolation",
 # "rho_handoff",

  #"kappa_beta_per_head",

  # "precision_attention_b0",
  # "decode_tau",




   # "sigma_max",



   # "renyi_order",

   #"e_mu_q_trust",
   #"e_mu_q_trust_ball",

   # "lambda_alpha",
    #"e_s_sigma_lr",
     #"e_s_mu_lr",
   #"e_q_sigma_lr",
    #"e_q_mu_lr",

    #"pos_phi_scale",


   #"lambda_twohop",
  # "kappa_beta",
  # "kappa_gamma",

   # "mass_phi",
   # "mstep_self_coupling_weight",



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


def _validated_sweep_name(value: object) -> str:
    """Return one safe, non-reserved output-directory component for a sweep."""
    if not isinstance(value, str):
        raise ValueError(f"sweep name is unsafe or reserved: {value!r}")
    key = portable_path_component_key(value, field="sweep name")
    if key in _RESERVED_SWEEP_KEYS:
        raise ValueError(f"sweep name is unsafe or reserved: {value!r}")
    return value


def validate_sweeps(sweep_names: List[str]) -> None:
    r"""Abort unless every named arm references real fields and constructs against the baseline.

    VFE3Config(**cfg) would silently ignore an unknown kwarg only if it were dropped first;
    here a bad name would instead raise a TypeError mid-run (or, worse under a dict-merge
    that pre-filtered, vanish and make every cell identical). Catching it once at startup
    turns a subtle "this parameter has no effect" result into an immediate, named error.
    """
    portable_names: Dict[str, str] = {}
    offenders: List[Tuple[str, str]] = []
    for name in sweep_names:
        validated = _validated_sweep_name(name)
        key = portable_path_component_key(validated, field="sweep name")
        prior = portable_names.get(key)
        if prior is not None and prior != validated:
            raise ValueError(f"sweep names {prior!r} and {validated!r} alias on a portable filesystem")
        portable_names[key] = validated
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
    construction_errors: List[Tuple[str, str, str]] = []
    for name in sweep_names:
        runs = make_run_overrides(name)
        labels = [label for label, _overrides in runs]
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError(f"sweep {name!r} has a non-string or empty label")
        duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
        if duplicate_labels:
            raise ValueError(f"sweep {name!r} has duplicate expanded label(s): {duplicate_labels}")
        for label, overrides in runs:
            cfg = dict(BASELINE_CONFIG)
            cfg.update(overrides)
            try:
                VFE3Config(**cfg)
            except (TypeError, ValueError) as exc:
                construction_errors.append((name, label, str(exc)))
    if construction_errors:
        lines = "\n".join(
            f"  sweep {sweep!r}, arm {label!r}: {error}"
            for sweep, label, error in construction_errors
        )
        raise ValueError(f"ablation arm construction failed:\n{lines}")


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

    Memoized on ``(dataset, seq_len, batch_size, split, cap, vocab_size)`` so runs that do not change
    those reuse the already-loaded immutable dataset and its attached byte identity, while a sweep over
    ``batch_size`` / ``max_seq_len`` / ``vocab_size`` correctly builds a distinct, matching loader.
    ``max_tokens`` caps only the train split (validation is always full). The loader never
    substitutes synthetic data for a missing real corpus -- that would mislabel synthetic numbers
    as a corpus measurement.
    """
    data_seed_override = _validated_data_seed_override()
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
    gen = (
        torch.Generator().manual_seed(data_seed_override)
        if split == "train" and data_seed_override is not None else None
    )
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
    d["seed"] = _require_exact_seed(seed, "seed")
    if max_steps is not None:
        d["max_steps"] = int(max_steps)
    return d


def _gauge_reporting_fields(cfg: VFE3Config) -> Dict[str, object]:
    """Return the executable gauge-purity classification carried by every ablation row."""
    report = _pure_path_report(cfg, [])
    return {
        "head_mixer_compatibility":    cfg.head_mixer_compatibility,
        "head_mixer_gauge_compatible": bool(cfg.head_mixer_gauge_compatible),
        "on_gauge_pure_path":          bool(report["on_gauge_pure_path"]),
    }


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
        omega = cstate["omega"]
        if not isinstance(omega, torch.Tensor):
            to_dense = getattr(omega, "to_dense_omega", None)
            if not callable(to_dense):
                raise TypeError(
                    "omega identity diagnostics require a tensor or an explicit dense converter"
                )
            omega = to_dense()
        if not isinstance(omega, torch.Tensor) or omega.ndim < 4:
            raise TypeError("omega identity diagnostics require (..., N, N, K, K) transport")
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
    dataset:                str,
    device:                 torch.device,
    seed:                   int,
    collect_diagnostics:    bool          = False,
    collect_extrapolation:  bool          = False,
    paired_token_bootstrap: bool          = False,
    max_tokens:             Optional[int] = None,
    max_steps:              Optional[int] = None,
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
                "primary_val_ppl": float("inf"), "seed": seed,
                "overrides": _jsonable(overrides)}

    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    model = VFEModel(cfg).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))

    train_loader = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "train",
                              max_tokens=max_tokens, vocab_size=cfg.vocab_size)
    val_loader   = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "validation",
                              vocab_size=cfg.vocab_size)
    loaded_data_sources = _loader_ablation_source_identities(
        train_loader,
        val_loader,
        dataset=dataset,
        max_tokens=max_tokens,
    )

    # Bits-per-CHARACTER correction for val BPC, mirroring train_vfe3. When normalization is
    # unavailable, BPC stays null and the separately named bits-per-token metric remains defined.
    val_tpc = tokens_per_char(dataset, "validation")

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
            loader.generator.manual_seed(
                _effective_data_seed(cfg.seed) if is_train else cfg.seed)

    print(f"    K={cfg.embed_dim} heads={len(model.group.irrep_dims)} group={cfg.gauge_group} "
          f"family={cfg.family} | steps={cfg.max_steps} batch={cfg.batch_size} | {n_params:,} params")
    for _cov in coverage_lines(train_loader, cfg.max_steps, dataset):
        print(f"   {_cov}")

    # Terminal artifact set (PB-02): a VALIDATION-ONLY finalizer runs once as a train() callback --
    # immediately after the final optimizer step -- so even a default cell (log/eval interval above
    # max_steps, checkpoint_interval=0) publishes a complete resumable artifact set (terminal metrics
    # row, best_model.pt, validation_results.json, summary.json, provenance, figures, and a resumable
    # terminal checkpoint) WITHOUT ever scoring a test split. The finalizer's returned merge mapping
    # carries the headline (primary/final validation, best, terminal_checkpoint); run_single overlays it
    # on the label/error/seed/overrides/token-cap metadata below. Periodic log/eval/checkpoint cadence is
    # no longer relied on for the artifact set -- generation/checkpointing is this one post-final step.
    terminal_result: Dict[str, Any] = {}
    train_start = time.perf_counter()

    def _terminal_callback(state, callback_losses):
        terminal_result.update(finalize_validation_run(
            model, artifacts, cfg, val_loader,
            tokens_per_char=val_tpc,
            train_loader=train_loader,
            losses=callback_losses,
            data_seed=_effective_data_seed(cfg.seed),
            max_tokens=max_tokens,
            tokenizer_tag=_tokenizer_tag(dataset),
            device=device,
            wall_time=time.perf_counter() - train_start,
            logger=logger,
            terminal_state=state,
        ))

    train(
        model, train_loader, cfg,
        n_steps=cfg.max_steps,
        grad_clip=cfg.grad_clip,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=val_loader,
        tokens_per_char=val_tpc,
        device=device,
        logger=logger,
        artifacts=artifacts,
        generate_samples=False,                              # pure silent path: no sample text
        terminal_callback=_terminal_callback,
    )

    result: Dict[str, Any] = {
        "label":                label,
        "error_kind":           None,
        "n_params":             n_params,
        "seed":                 int(cfg.seed),
        "overrides":            _jsonable(overrides),
        "max_tokens":           (int(max_tokens) if max_tokens is not None else None),
        "_loaded_data_sources": loaded_data_sources,
    }
    result.update(_gauge_reporting_fields(cfg))
    result.update(terminal_result)                           # primary/final/best/terminal_checkpoint headline

    # PB-07 opt-in: publish the per-token validation nats (the paired within-run bootstrap the
    # component-ablation forest consumes) as an atomic sibling artifact, and record its exact
    # byte/tensor identity in the marker. Written BEFORE run_sweep publishes the reuse contract and
    # completion marker, so the post-contract validator binds bytes that already exist on disk. The
    # stable, unshuffled validation loader plus the shared seed make the vector position-aligned
    # across arms. This per-unit extraction is the ONLY added validation replay; aggregate final
    # validation stays owned by the terminal callback above.
    token_identity: Optional[Dict[str, Any]] = None
    if paired_token_bootstrap:
        per_token = per_unit_eval_nats(model, val_loader, device=device)["per_token_nats"]
        final_path = run_dir / "val_token_nats.pt"
        with _unique_sibling_temp(final_path) as tmp_path:
            torch.save(per_token.detach().cpu(), tmp_path)
            _atomic_replace(final_path, tmp_path)
        token_identity = {
            "path":       final_path.name,
            "sha256":     _sha256_file(final_path),
            "size_bytes": final_path.stat().st_size,
            "numel":      int(per_token.numel()),
            "dtype":      str(per_token.dtype),
        }
    result["paired_token_bootstrap"] = bool(paired_token_bootstrap)
    result["val_token_nats_path"] = (
        token_identity["path"] if token_identity is not None else None
    )
    result["val_token_nats_sha256"] = (
        token_identity["sha256"] if token_identity is not None else None
    )
    result["val_token_nats_size_bytes"] = (
        token_identity["size_bytes"] if token_identity is not None else None
    )
    result["val_token_nats_numel"] = (
        token_identity["numel"] if token_identity is not None else None
    )
    result["val_token_nats_dtype"] = (
        token_identity["dtype"] if token_identity is not None else None
    )

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
    "final_val_ce", "final_val_bits_per_token", "final_val_bpc", "best_val_ppl", "final_train_loss",
    "n_params", "head_mixer_compatibility", "head_mixer_gauge_compatible",
    "on_gauge_pure_path",
    # opt-in per-cell converged-state diagnostics (S2; empty unless the sweep sets collect_diagnostics)
    "attn_entropy", "omega_identity_dev", "builder_resid", "gauge_resid_in", "gauge_resid_out",
    "rank_resid", "cov_gap", "energy_klmax_frac",
    # opt-in paired-token artifact identity (PB-07; empty unless the sweep sets paired_token_bootstrap)
    "paired_token_bootstrap", "val_token_nats_path", "val_token_nats_sha256",
    "val_token_nats_size_bytes", "val_token_nats_numel", "val_token_nats_dtype",
    "wall_time_s", "seed", "error",
]

_DIAGNOSTIC_RESULT_KEYS = {
    "attn_entropy", "omega_identity_dev", "builder_resid", "gauge_resid_in", "gauge_resid_out",
    "rank_resid", "rank_resid_by_layer", "cov_gap", "energy_klmax_frac",
}
_BASE_REQUIRED_DIAGNOSTIC_KEYS = {
    "attn_entropy",
    "energy_klmax_frac",
    "gauge_resid_in",
    "gauge_resid_out",
    "omega_identity_dev",
    "rank_resid",
}


_CELL_CONTRACT_SCHEMA_VERSION = 3
_SWEEP_AGGREGATION_CONTRACT_SCHEMA_VERSION = 4
_ABLATION_CELL_OWNER_SCHEMA_VERSION = 1
_ABLATION_CELL_OWNER_FILENAME = "ablation_cell_owner.json"
_SWEEP_DIAGNOSTIC_FLAG_KEYS = {
    "collect_diagnostics",
    "collect_extrapolation",
    "paired_token_bootstrap",
}


def _require_exact_seed(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be an exact non-negative integer, got {value!r}")
    return value


def _validated_sweep_seeds(sweep: Mapping[str, object], fallback_seed: object) -> List[int]:
    """Resolve one sweep's seed axis without bool/fraction/string coercion or duplicate cells."""
    fallback = _require_exact_seed(fallback_seed, "seed")
    if "seeds" not in sweep:
        return [fallback]
    raw = sweep["seeds"]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("sweep['seeds'] must be a non-empty list or tuple of unique integers")
    seeds = [_require_exact_seed(seed, "sweep['seeds'] entry") for seed in raw]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"sweep['seeds'] must be unique, got {seeds!r}")
    return seeds


def _validated_data_seed_override() -> Optional[int]:
    """Validate the optional global train-loader seed without coercion."""
    if DATA_SEED is None:
        return None
    return _require_exact_seed(DATA_SEED, "DATA_SEED")


def _effective_data_seed(fallback_seed: object) -> int:
    """Return the validated override or one exact per-cell fallback seed."""
    override = _validated_data_seed_override()
    return override if override is not None else _require_exact_seed(fallback_seed, "seed")


_ABLATION_SOURCE_SPLITS = ("train", "validation")


def _validated_ablation_source_identities(
    value: object,
) -> Dict[str, Dict[str, object]]:
    """Return a detached, JSON-safe identity for exactly the splits an ablation cell consumes."""
    if not isinstance(value, Mapping) or set(value) != set(_ABLATION_SOURCE_SPLITS):
        raise ValueError(
            "ablation data source identities must contain exactly train and validation")
    normalized: Dict[str, Dict[str, object]] = {}
    for split in _ABLATION_SOURCE_SPLITS:
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


def _ablation_source_identities(
    dataset:   str,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    """Hash each corpus split once for a shared pre-sweep filesystem snapshot."""
    return _validated_ablation_source_identities({
        split: cache_source_identity(dataset, split, cache_dir=cache_dir)
        for split in _ABLATION_SOURCE_SPLITS
    })


def _loader_ablation_source_identities(
    train_loader: object,
    val_loader:   object,

    *,
    dataset:    str,
    max_tokens: Optional[int],
) -> Optional[Dict[str, Dict[str, object]]]:
    r"""Snapshot the identities attached to the immutable datasets actually used by one cell.

    ``make_dataloader`` binds every loaded tensor to the source identity that remained stable across
    its load. A custom loader without that contract remains usable by ``run_single`` directly, but
    the sweep receives ``None`` and therefore refuses to publish a reusable success contract.
    """
    sources: Dict[str, object] = {}
    try:
        for split, loader, expected_cap in (
            ("train", train_loader, max_tokens),
            ("validation", val_loader, None),
        ):
            data_identity = getattr(getattr(loader, "dataset", None), "data_identity", None)
            if (not isinstance(data_identity, Mapping)
                    or data_identity.get("schema_version") != 2
                    or data_identity.get("dataset") != dataset
                    or data_identity.get("split") != split
                    or data_identity.get("max_tokens") != expected_cap):
                raise ValueError(f"{split} loader data identity is unavailable or mismatched")
            sources[split] = data_identity.get("source")
        return _validated_ablation_source_identities(sources)
    except (TypeError, ValueError):
        logger.warning("  [loaded corpus identity unavailable -> reuse contract forbidden]")
        return None


def _validated_ablation_code_identity(value: object) -> Dict[str, object]:
    """Return one detached, usable source identity for an ablation invocation."""
    if not isinstance(value, Mapping):
        raise TypeError("ablation code identity must be a mapping")
    detached = json.loads(json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False, allow_nan=False))
    if set(detached) == {"package_code_sha256", "ablation_runner_sha256"}:
        for field in ("package_code_sha256", "ablation_runner_sha256"):
            digest = detached[field]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)):
                raise ValueError(f"ablation code identity {field} is not a SHA-256 digest")
        return detached

    # Legacy shape remains accepted for persisted-fixture and direct-API compatibility. A current
    # production contract uses the two source digests above, so a legacy cohort never equals it.
    git_sha = detached.get("git_sha")
    git_dirty = detached.get("git_dirty")
    fingerprint = detached.get("git_dirty_fingerprint")
    if not isinstance(git_sha, str) or not git_sha or type(git_dirty) is not bool:
        raise ValueError("ablation code identity is unavailable")
    if ((git_dirty and (not isinstance(fingerprint, str) or not fingerprint))
            or (not git_dirty and fingerprint is not None)):
        raise ValueError("ablation code identity has an inconsistent dirty-tree fingerprint")
    return detached


def _validated_diagnostic_flags(value: object) -> Dict[str, bool]:
    """Validate the three invocation-wide artifact requests bound into every cell contract."""
    if not isinstance(value, Mapping) or set(value) != _SWEEP_DIAGNOSTIC_FLAG_KEYS:
        raise ValueError(
            "ablation diagnostic flags must contain exactly collect_diagnostics, "
            "collect_extrapolation, and paired_token_bootstrap"
        )
    if any(type(value[key]) is not bool for key in _SWEEP_DIAGNOSTIC_FLAG_KEYS):
        raise TypeError("ablation diagnostic flags must be exact booleans")
    return {key: bool(value[key]) for key in sorted(_SWEEP_DIAGNOSTIC_FLAG_KEYS)}


def _validated_required_diagnostic_keys(value: object) -> List[str]:
    """Return a unique, sorted list of diagnostic marker fields required for completion."""
    if not isinstance(value, (list, tuple)):
        raise TypeError("required diagnostic keys must be a list or tuple")
    keys = list(value)
    if (any(not isinstance(key, str) or key not in _DIAGNOSTIC_RESULT_KEYS for key in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError("required diagnostic keys must be unique known result fields")
    return sorted(keys)


def _validated_min_extrapolation_points(value: object) -> int:
    """Return the exact non-negative number of finite distinct-N extrapolation points required."""
    if type(value) is not int or value < 0:
        raise ValueError("min_extrapolation_points must be an exact non-negative integer")
    return value


def _validated_seed_design(value: object) -> List[int]:
    """Return the sorted, unique model-seed panel used for cross-sweep comparisons."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("ablation seed design must be a non-empty list or tuple")
    seeds = [_require_exact_seed(seed, "seed_design") for seed in value]
    if len(seeds) != len(set(seeds)):
        raise ValueError("ablation seed design must not contain duplicates")
    return sorted(seeds)


def _baseline_semantic_config_fingerprint(max_steps: Optional[int]) -> str:
    """Fingerprint the effective unswept operating point independently of model seed."""
    cfg = VFE3Config(**_cell_cfg_dict({}, seed=0, max_steps=max_steps))
    return semantic_config_fingerprint(asdict(cfg))


def _sweep_required_outputs(sweep: Mapping[str, object]) -> Tuple[List[str], int]:
    """Resolve the output-completeness profile declared by one sweep."""
    collect_diagnostics = bool(sweep.get("collect_diagnostics", False))
    extra = sweep.get("required_diagnostics", ())
    required = _validated_required_diagnostic_keys(extra)
    if collect_diagnostics:
        required = sorted(set(required) | _BASE_REQUIRED_DIAGNOSTIC_KEYS)
    elif required:
        raise ValueError("required_diagnostics needs collect_diagnostics=True")

    collect_extrapolation = bool(sweep.get("collect_extrapolation", False))
    default_points = 2 if collect_extrapolation else 0
    minimum = _validated_min_extrapolation_points(
        sweep.get("min_extrapolation_points", default_points)
    )
    if collect_extrapolation and minimum < 2:
        raise ValueError("collect_extrapolation requires at least two extrapolation points")
    if not collect_extrapolation and minimum:
        raise ValueError("min_extrapolation_points needs collect_extrapolation=True")
    return required, minimum


def _requested_outputs_are_complete(
    result: Mapping[str, object],

    *,
    required_diagnostic_keys: Sequence[str],
    min_extrapolation_points: int,
) -> bool:
    """Whether one result contains every finite output promised by its sweep contract."""
    try:
        required = _validated_required_diagnostic_keys(required_diagnostic_keys)
        minimum = _validated_min_extrapolation_points(min_extrapolation_points)
    except (TypeError, ValueError):
        return False
    for key in required:
        value = result.get(key)
        if key == "rank_resid_by_layer":
            if not isinstance(value, list) or len(value) < 2:
                return False
            try:
                if any(not math.isfinite(float(item)) for item in value):
                    return False
            except (TypeError, ValueError):
                return False
            continue
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    if minimum == 0:
        return True
    curve = result.get("extrap_ce")
    if not isinstance(curve, list) or len(curve) < minimum:
        return False
    seen_n = set()
    for point in curve:
        if not isinstance(point, Mapping):
            return False
        n = point.get("n")
        if type(n) is not int or n <= 0 or n in seen_n:
            return False
        seen_n.add(n)
        try:
            ce = float(point["ce"])
            ppl = float(point["ppl"])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(ce) or not math.isfinite(ppl) or ppl <= 0.0:
            return False
    return True


def _validated_sweep_aggregation_contract(value: object) -> Dict[str, object]:
    """Return one normalized invocation contract for cross-cell scientific aggregation."""
    if not isinstance(value, Mapping):
        raise TypeError("sweep aggregation contract must be a mapping")
    required = {
        "schema_version",
        "baseline_semantic_config_fingerprint",
        "seed_design",
        "dataset",
        "device",
        "tokenizer_tag",
        "data_seed_override",
        "max_tokens",
        "max_steps",
        "source_identities",
        "code_identity",
        "diagnostic_flags",
        "required_diagnostic_keys",
        "min_extrapolation_points",
    }
    if set(value) != required:
        raise ValueError("sweep aggregation contract has missing or unknown fields")
    if value.get("schema_version") != _SWEEP_AGGREGATION_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported sweep aggregation contract schema")
    baseline_fingerprint = value.get("baseline_semantic_config_fingerprint")
    if (not isinstance(baseline_fingerprint, str) or len(baseline_fingerprint) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in baseline_fingerprint)):
        raise ValueError("ablation baseline semantic config fingerprint must be a SHA-256 digest")
    seed_design = _validated_seed_design(value.get("seed_design"))
    dataset = value.get("dataset")
    device = value.get("device")
    tokenizer_tag = value.get("tokenizer_tag")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("sweep aggregation dataset must be a non-empty string")
    if not isinstance(device, str) or not device:
        raise ValueError("sweep aggregation device must be a non-empty string")
    if not isinstance(tokenizer_tag, str) or not tokenizer_tag:
        raise ValueError("sweep aggregation tokenizer_tag must be a non-empty string")

    data_seed_override = value.get("data_seed_override")
    if data_seed_override is not None:
        data_seed_override = _require_exact_seed(data_seed_override, "data_seed_override")
    max_tokens = value.get("max_tokens")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 0):
        raise ValueError("sweep aggregation max_tokens must be an exact non-negative integer or null")
    max_steps = value.get("max_steps")
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("sweep aggregation max_steps must be an exact positive integer or null")

    diagnostic_flags = _validated_diagnostic_flags(value.get("diagnostic_flags"))
    required_diagnostic_keys = _validated_required_diagnostic_keys(
        value.get("required_diagnostic_keys")
    )
    min_extrapolation_points = _validated_min_extrapolation_points(
        value.get("min_extrapolation_points")
    )
    if diagnostic_flags["collect_diagnostics"] != bool(required_diagnostic_keys):
        raise ValueError("diagnostic collection and required keys are inconsistent")
    if diagnostic_flags["collect_extrapolation"]:
        if min_extrapolation_points < 2:
            raise ValueError("extrapolation collection requires at least two points")
    elif min_extrapolation_points != 0:
        raise ValueError("unrequested extrapolation cannot require output points")

    return {
        "schema_version":     _SWEEP_AGGREGATION_CONTRACT_SCHEMA_VERSION,
        "baseline_semantic_config_fingerprint": baseline_fingerprint.lower(),
        "seed_design":        seed_design,
        "dataset":            dataset,
        "device":             device,
        "tokenizer_tag":      tokenizer_tag,
        "data_seed_override": data_seed_override,
        "max_tokens":         max_tokens,
        "max_steps":          max_steps,
        "source_identities":  _validated_ablation_source_identities(
            value.get("source_identities")
        ),
        "code_identity":      _validated_ablation_code_identity(value.get("code_identity")),
        "diagnostic_flags":   diagnostic_flags,
        "required_diagnostic_keys": required_diagnostic_keys,
        "min_extrapolation_points": min_extrapolation_points,
    }


def _sweep_aggregation_contract(
    dataset:          str,
    diagnostic_flags: Mapping[str, bool],

    *,
    data_seed_override: Optional[int],
    max_tokens:        Optional[int],
    max_steps:         Optional[int],
    seed_design:       Sequence[int],
    source_identities: Mapping[str, object],
    code_identity:     Mapping[str, object],
    device:            str = "cpu",
    required_diagnostic_keys: Optional[Sequence[str]] = None,
    min_extrapolation_points: Optional[int]           = None,
) -> Dict[str, object]:
    """Build the common invocation identity under which unlike labels, values, and seeds compare."""
    flags = _validated_diagnostic_flags(diagnostic_flags)
    required_keys = _validated_required_diagnostic_keys(
        (_BASE_REQUIRED_DIAGNOSTIC_KEYS if flags["collect_diagnostics"] else ())
        if required_diagnostic_keys is None else required_diagnostic_keys
    )
    minimum = _validated_min_extrapolation_points(
        (2 if flags["collect_extrapolation"] else 0)
        if min_extrapolation_points is None else min_extrapolation_points
    )
    return _validated_sweep_aggregation_contract({
        "schema_version":     _SWEEP_AGGREGATION_CONTRACT_SCHEMA_VERSION,
        "baseline_semantic_config_fingerprint": _baseline_semantic_config_fingerprint(max_steps),
        "seed_design":        list(seed_design),
        "dataset":            dataset,
        "device":             device,
        "tokenizer_tag":      _tokenizer_tag(dataset),
        "data_seed_override": data_seed_override,
        "max_tokens":         max_tokens,
        "max_steps":          max_steps,
        "source_identities":  source_identities,
        "code_identity":      code_identity,
        "diagnostic_flags":   flags,
        "required_diagnostic_keys": required_keys,
        "min_extrapolation_points": minimum,
    })


def _cell_contract(
    cfg:              VFE3Config,
    dataset:          str,
    diagnostic_flags: Mapping[str, bool],

    *,
    data_seed:         int,
    device:            str                          = "cpu",
    max_tokens:        Optional[int]                  = None,
    cache_dir:         Optional[Path]                 = None,
    source_identities: Optional[Mapping[str, object]] = None,
    code_identity:     Optional[Mapping[str, object]] = None,
    required_diagnostic_keys: Optional[Sequence[str]] = None,
    min_extrapolation_points: Optional[int]           = None,
) -> Dict[str, object]:
    r"""Build the versioned contract that authorizes reuse of one ablation cell.

    The contract binds a cached cell to the full identity a bare ``param=value`` label omits, so
    resume fails closed after ANY of them drifts (audit 2026-07-12 PB-01): the semantic-config
    fingerprint (the imported baseline, not just the swept field), the session dataset / data seed /
    token cap / tokenizer tag (loader seams that never land in ``config.json``), the per-split corpus
    identity (byte size + streamed SHA-256, so a rebuilt or re-tokenized cache is caught even at the
    same label), the repository code identity (HEAD plus a dirty-tree fingerprint), and the requested
    diagnostic/extrapolation collections. ``cache_source_identity`` raises when a split's cache is
    absent, which the caller converts into "forbid reuse" rather than a stale hit.
    """
    data_seed = _require_exact_seed(data_seed, "data_seed")
    if not isinstance(device, str) or not device:
        raise ValueError("ablation cell device must be a non-empty string")
    sources = (
        _ablation_source_identities(dataset, cache_dir=cache_dir)
        if source_identities is None
        else _validated_ablation_source_identities(source_identities)
    )
    code = (
        _validated_ablation_code_identity(_git_code_identity())
        if code_identity is None
        else _validated_ablation_code_identity(code_identity)
    )
    flags = _validated_diagnostic_flags(diagnostic_flags)
    required_keys = _validated_required_diagnostic_keys(
        (_BASE_REQUIRED_DIAGNOSTIC_KEYS if flags["collect_diagnostics"] else ())
        if required_diagnostic_keys is None else required_diagnostic_keys
    )
    minimum = _validated_min_extrapolation_points(
        (2 if flags["collect_extrapolation"] else 0)
        if min_extrapolation_points is None else min_extrapolation_points
    )
    if flags["collect_diagnostics"] != bool(required_keys):
        raise ValueError("diagnostic collection and required keys are inconsistent")
    if flags["collect_extrapolation"] != (minimum >= 2):
        raise ValueError("extrapolation collection and minimum output points are inconsistent")
    return {
        "schema_version":              _CELL_CONTRACT_SCHEMA_VERSION,
        "semantic_config_fingerprint": semantic_config_fingerprint(asdict(cfg)),
        "dataset":                     dataset,
        "device":                      device,
        "data_seed":                   data_seed,
        "max_tokens":                  int(max_tokens) if max_tokens is not None else None,
        "tokenizer_tag":               _tokenizer_tag(dataset),
        "train_source":                sources["train"],
        "validation_source":           sources["validation"],
        "code_identity":               code,
        "diagnostic_flags":            flags,
        "required_diagnostic_keys":    required_keys,
        "min_extrapolation_points":    minimum,
    }


def _expected_cell_contract_from_aggregation(
    result:               Mapping[str, object],
    aggregation_contract: Mapping[str, object],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Reconstruct one adjacent row's exact cell contract under the current common cohort."""
    aggregation = _validated_sweep_aggregation_contract(aggregation_contract)
    overrides = result.get("overrides")
    if not isinstance(overrides, Mapping):
        raise TypeError("ablation result overrides must be a mapping")
    seed = _require_exact_seed(result.get("seed"), "ablation result seed")
    if seed not in aggregation["seed_design"]:
        raise ValueError("ablation result seed is outside the sweep aggregation seed design")
    cfg = VFE3Config(**_cell_cfg_dict(
        dict(overrides),
        seed=seed,
        max_steps=aggregation["max_steps"],
    ))
    data_seed_override = aggregation["data_seed_override"]
    contract = _cell_contract(
        cfg,
        str(aggregation["dataset"]),
        aggregation["diagnostic_flags"],
        data_seed=(seed if data_seed_override is None else int(data_seed_override)),
        device=str(aggregation["device"]),
        max_tokens=aggregation["max_tokens"],
        source_identities=aggregation["source_identities"],
        code_identity=aggregation["code_identity"],
        required_diagnostic_keys=aggregation["required_diagnostic_keys"],
        min_extrapolation_points=int(aggregation["min_extrapolation_points"]),
    )
    contract["tokenizer_tag"] = aggregation["tokenizer_tag"]
    return contract, _gauge_reporting_fields(cfg)


def _gauge_purity_summary(results: List[Dict[str, Any]]) -> Dict[str, object]:
    """Summarize row-level gauge classifications without collapsing mixed sweeps into a pure claim."""
    classifications = {
        str(result["label"]): str(result["head_mixer_compatibility"])
        for result in results
    }
    return {
        "classifications_by_label": classifications,
        "contains_independent_head_nonintertwiner": any(
            value == "independent_head_nonintertwiner"
            for value in classifications.values()
        ),
        "all_rows_on_gauge_pure_path": bool(results) and all(
            result.get("on_gauge_pure_path") is True for result in results
        ),
    }


def _expected_cell_contract_or_none(
    overrides: Mapping[str, object],
    dataset: str,
    diagnostic_flags: Mapping[str, bool],
    *,
    seed:              int,
    device:            str                              = "cpu",
    max_steps:         Optional[int]                  = None,
    max_tokens:        Optional[int]                  = None,
    source_identities: Optional[Mapping[str, object]] = None,
    code_identity:     Optional[Mapping[str, object]] = None,
    required_diagnostic_keys: Optional[Sequence[str]] = None,
    min_extrapolation_points: Optional[int]           = None,
) -> Optional[Dict[str, object]]:
    r"""Build the reuse contract inside the per-cell failure boundary, else forbid reuse.

    Wraps both ``VFE3Config`` construction and the corpus source hashing so a rejected config or a
    missing/corrupt cache raises here rather than in the sweep loop: the exception is logged and
    ``None`` is returned, which only forbids cache reuse (``run_single`` then executes inside its own
    isolation and records the actual config/data error without aborting later cells). The data-order
    seed recorded in the contract is the effective TRAIN-loader seed -- ``DATA_SEED`` when set (shared
    across per-model seeds), else the cell seed -- matching ``run_single``'s post-build reseed.
    """
    try:
        cfg = VFE3Config(**_cell_cfg_dict(dict(overrides), seed=seed, max_steps=max_steps))
        data_seed = _require_exact_seed(
            DATA_SEED if DATA_SEED is not None else seed,
            "DATA_SEED" if DATA_SEED is not None else "seed",
        )
        return _cell_contract(
            cfg,
            dataset,
            diagnostic_flags,
            data_seed=data_seed,
            device=device,
            max_tokens=max_tokens,
            source_identities=source_identities,
            code_identity=code_identity,
            required_diagnostic_keys=required_diagnostic_keys,
            min_extrapolation_points=min_extrapolation_points,
        )
    except Exception as exc:                                  # unbuildable config / unhashable corpus
        logger.warning("  [contract unavailable -> reuse forbidden] %s", exc)
        return None


def _terminal_checkpoint_identity(
    run_dir: Path,
    marker:  Mapping[str, object],
) -> Optional[Dict[str, object]]:
    """Return the owned terminal checkpoint's immutable identity, or ``None`` if unsafe."""
    raw_path = marker.get("terminal_checkpoint")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    root = run_dir.resolve()
    raw = Path(raw_path)
    candidates = [raw] if raw.is_absolute() else [raw, run_dir / raw]
    for candidate in candidates:
        try:
            if candidate.is_symlink() or _path_is_junction(candidate):
                continue
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(root)
            if not relative.parts or not resolved.is_file():
                continue
            if any(part in {"", ".", ".."} for part in relative.parts):
                continue
            return {
                "terminal_checkpoint_relpath":    relative.as_posix(),
                "terminal_checkpoint_sha256":     _sha256_file(resolved),
                "terminal_checkpoint_size_bytes": resolved.stat().st_size,
            }
        except (OSError, ValueError):
            continue
    return None


def _terminal_checkpoint_is_current(
    run_dir: Path,
    marker:  Mapping[str, object],
) -> bool:
    """Revalidate a published terminal checkpoint's location and exact bytes on every reuse."""
    current = _terminal_checkpoint_identity(run_dir, marker)
    if current is None:
        return False
    return all(marker.get(key) == value for key, value in current.items())


def _cell_is_current(
    run_dir:           Path,
    expected_contract: Mapping[str, object],
) -> bool:
    r"""Return true only for a successful cell with an exactly matching contract.

    Fails closed on anything short of a completed success bound to ``expected_contract``: a legacy
    directory carrying only the old ``ablation_result.json`` success marker and no
    ``cell_contract.json`` re-runs, as does a marker that is unreadable, non-mapping, failed, or
    lacks a finite terminal validation PPL. The published contract must parse, be a mapping, carry
    the current schema version, and equal the freshly rebuilt ``expected_contract`` field for field
    (ordinary nested mapping equality) -- any code, corpus, tokenizer, dataset, seed, token-cap,
    semantic-config, or diagnostic-flag drift leaves the two unequal and forbids reuse.
    """
    try:
        marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8"))
    except Exception:                                        # no/unreadable success marker -> re-run
        return False
    if not isinstance(marker, Mapping):                      # parseable but not an object -> re-run
        return False
    if marker.get("status") != "success" or marker.get("error_kind") is not None:
        return False
    expected_fingerprint = semantic_config_fingerprint(dict(expected_contract))
    if marker.get("cell_contract_fingerprint") != expected_fingerprint:
        return False
    try:
        terminal_ppl = float(marker["final_val_ppl"])
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(terminal_ppl):
        return False
    if not _requested_outputs_are_complete(
        marker,
        required_diagnostic_keys=expected_contract.get("required_diagnostic_keys", ()),
        min_extrapolation_points=expected_contract.get("min_extrapolation_points", 0),
    ):
        return False
    if not _terminal_checkpoint_is_current(run_dir, marker):
        return False
    contract_path = run_dir / "cell_contract.json"
    if not contract_path.exists():                           # legacy dir / never-published contract
        return False
    try:
        loaded = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:                                        # truncated / corrupt contract -> re-run
        return False
    if not isinstance(loaded, Mapping):
        return False
    if loaded.get("schema_version") != _CELL_CONTRACT_SCHEMA_VERSION:
        return False
    return (dict(loaded) == dict(expected_contract)
            and semantic_config_fingerprint(dict(loaded)) == expected_fingerprint)


def _paired_token_artifact_is_current(run_dir: Path, *, required: bool) -> bool:
    r"""Return true only when the requested paired-token artifact is present and its identity verifies.

    The cell contract's sorted ``diagnostic_flags`` binds the REQUEST (was paired_token_bootstrap
    asked for); this post-contract validator binds the requested ARTIFACT's exact bytes and tensor
    schema so a stale, tampered, or same-shape-but-different-bytes ``val_token_nats.pt`` forces a
    re-run rather than a cache hit. When the sweep did not request the artifact (``required`` false)
    there is nothing to bind, so it returns true immediately. Otherwise the completion marker must
    record a real ``paired_token_bootstrap`` true, ``val_token_nats_path == "val_token_nats.pt"``,
    and the exact SHA-256 / byte size / tensor length / dtype fields, the file must exist, its
    recomputed streamed hash and byte size must match, and a ``weights_only`` safe load must yield a
    finite, nonempty, one-dimensional tensor whose numel and string dtype equal the marker. Any
    missing, malformed, or nonexact field fails closed.
    """
    if not required:
        return True
    try:
        marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8"))
    except Exception:                                        # no/unreadable marker -> re-run
        return False
    if not isinstance(marker, Mapping):
        return False
    if marker.get("paired_token_bootstrap") is not True:
        return False
    if marker.get("val_token_nats_path") != "val_token_nats.pt":
        return False
    expected_sha   = marker.get("val_token_nats_sha256")
    expected_size  = marker.get("val_token_nats_size_bytes")
    expected_numel = marker.get("val_token_nats_numel")
    expected_dtype = marker.get("val_token_nats_dtype")
    if not (isinstance(expected_sha, str) and expected_sha):
        return False
    path = run_dir / "val_token_nats.pt"
    if not path.is_file():
        return False
    if path.stat().st_size != expected_size:
        return False
    if _sha256_file(path) != expected_sha:
        return False
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.ndim != 1 or tensor.numel() == 0:
        return False
    if int(tensor.numel()) != expected_numel or str(tensor.dtype) != expected_dtype:
        return False
    return bool(torch.isfinite(tensor).all())


def _sanitize(label: str) -> str:
    r"""A filesystem-safe single path component (no separators, parent tokens, or drive colon).

    The char-replace is lossy ('a=b', 'a b', 'a/b' all map to 'a_b'), so a stable short hash of
    the RAW label is appended: distinct labels get distinct run dirs, while the map stays
    deterministic in the label so the resume [CACHED] path finds the same dir on re-run.
    """
    return filesystem_slug(label)


def _write_sweep_csv(sweep_dir: Path, results: List[Dict[str, Any]]) -> None:
    r"""Atomically replace ``sweep_results.csv`` with one complete fixed-column frame."""
    final_path = sweep_dir / "sweep_results.csv"
    with _unique_sibling_temp(final_path) as temporary_path:
        with open(temporary_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for result in results:
                writer.writerow({key: result.get(key, "") for key in _CSV_COLUMNS})
        _atomic_replace(final_path, temporary_path)


def _collect_sweep_results(
    sweep_dir: Path,

    *,
    aggregation_contract: Optional[Mapping[str, object]] = None,
) -> List[Dict[str, Any]]:
    r"""Collect only successful cells whose full contracts belong to one invocation cohort.

    Labels and swept values may differ. Every row is nevertheless reconstructed from the cohort's
    common code, corpus, tokenizer, token budget, diagnostic requests, and baseline plus that row's
    own overrides and seed. A base label is retained only with exactly one admitted row for every
    seed in the cohort's declared panel. Missing, malformed, fingerprint-mismatched, adjacent, or
    seed-incomplete contracts are excluded before CSV publication, plotting, or winner selection.
    """
    if aggregation_contract is None:
        try:
            meta = json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))
            aggregation_contract = meta["aggregation_contract"]
        except Exception:
            return []
    try:
        aggregation = _validated_sweep_aggregation_contract(aggregation_contract)
    except (TypeError, ValueError):
        return []

    results: List[Dict[str, Any]] = []
    for marker in sorted(sweep_dir.glob("*/ablation_result.json")):
        run_dir = marker.parent
        if not _cell_dir_is_owned(sweep_dir, run_dir):
            continue
        try:
            result = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:                                       # unreadable marker -> skip
            continue
        if not isinstance(result, Mapping):
            continue
        label = result.get("label")
        if (not isinstance(label, str) or not label
                or result.get("sweep") != sweep_dir.name
                or run_dir.name != _sanitize(label)):
            continue
        seed = result.get("seed")
        if (type(seed) is not int
                or not _ablation_cell_owner_is_exact(
                    run_dir, sweep_dir.name, label, seed)):
            continue
        if result.get("status") != "success" or result.get("error_kind") is not None:
            continue
        try:
            terminal_ppl = float(result["final_val_ppl"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(terminal_ppl):
            continue
        try:
            expected_contract, gauge_fields = _expected_cell_contract_from_aggregation(
                result,
                aggregation,
            )
        except Exception:
            continue
        if not _cell_is_current(run_dir, expected_contract):
            continue
        if not _paired_token_artifact_is_current(
            run_dir,
            required=bool(aggregation["diagnostic_flags"]["paired_token_bootstrap"]),
        ):
            continue
        if any(result.get(key) != value for key, value in gauge_fields.items()):
            continue
        results.append(dict(result))

    required_seeds = set(aggregation["seed_design"])
    panels: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        panels.setdefault(_base_label(str(result["label"])), []).append(result)
    complete_bases = {
        base
        for base, members in panels.items()
        if (
            len(members) == len(required_seeds)
            and {member["seed"] for member in members} == required_seeds
        )
    }
    return [
        result
        for result in results
        if _base_label(str(result["label"])) in complete_bases
    ]


def _path_is_junction(path: Path) -> bool:
    """Return whether ``path`` is any Windows reparse point, including pre-3.12 junctions."""
    probe = getattr(os.path, "isjunction", None)
    if probe is not None and probe(path):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(reparse_flag and attributes & reparse_flag)


def _cell_dir_is_owned(sweep_dir: Path, run_dir: Path) -> bool:
    """Return true only for a real direct child directory that cannot redirect cleanup elsewhere."""
    try:
        return bool(
            run_dir.parent == sweep_dir
            and run_dir.is_dir()
            and not run_dir.is_symlink()
            and not _path_is_junction(run_dir)
            and run_dir.resolve().parent == sweep_dir.resolve()
        )
    except OSError:
        return False


def _prepare_owned_output_child(root: Path, name: str, *, role: str) -> Path:
    """Create or validate one real direct-child directory under an explicit output root."""
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ValueError(f"{role} name must be one non-empty path component")
    component = Path(name)
    if component.name != name or component.is_absolute() or component.drive:
        raise ValueError(f"{role} name must be one safe path component: {name!r}")

    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError(f"{role} output root is not a directory: {root}")
    child = root / name
    if child.is_symlink() or _path_is_junction(child):
        raise ValueError(f"{role} directory may not be a symlink, junction, or reparse point")
    if child.exists() and not child.is_dir():
        raise ValueError(f"{role} path exists but is not a directory: {child}")
    child.mkdir(parents=False, exist_ok=True)
    if not _cell_dir_is_owned(root, child):
        raise ValueError(f"{role} directory resolves outside its output root")
    return child


def _ablation_cell_owner_payload(
    sweep_name: str,
    label:      str,
    seed:       int,
) -> Dict[str, object]:
    """Return the exact versioned identity that authorizes destructive cell cleanup."""
    if type(sweep_name) is not str or not sweep_name:
        raise ValueError("ablation cell ownership sweep must be a nonempty string")
    if type(label) is not str or not label:
        raise ValueError("ablation cell ownership label must be a nonempty string")
    return {
        "schema_version": _ABLATION_CELL_OWNER_SCHEMA_VERSION,
        "sweep":          sweep_name,
        "label":          label,
        "seed":           _require_exact_seed(seed, "ablation cell ownership seed"),
    }


def _read_regular_json_mapping(path: Path, *, role: str) -> Dict[str, object]:
    """Read one direct regular, non-reparse JSON object or fail without following a redirect."""
    if path.is_symlink() or _path_is_junction(path) or not path.is_file():
        raise ValueError(f"{role} must be a regular non-reparse file")
    try:
        if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
            raise ValueError(f"{role} resolves outside its cell directory")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must contain a JSON object")
    return dict(value)


def _ablation_cell_owner_is_exact(
    run_dir:    Path,
    sweep_name: str,
    label:      str,
    seed:       int,
) -> bool:
    """Return true only when the regular ownership sentinel has the exact requested identity."""
    try:
        expected = _ablation_cell_owner_payload(sweep_name, label, seed)
        owner = _read_regular_json_mapping(
            run_dir / _ABLATION_CELL_OWNER_FILENAME,
            role="ablation cell ownership sentinel",
        )
    except ValueError:
        return False
    return bool(
        set(owner) == set(expected)
        and type(owner.get("schema_version")) is int
        and type(owner.get("sweep")) is str
        and type(owner.get("label")) is str
        and type(owner.get("seed")) is int
        and owner == expected
    )


def _authorize_ablation_cell_cleanup(
    run_dir:    Path,
    sweep_name: str,
    label:      str,
    seed:       int,
) -> Path:
    r"""Validate existing ownership or promote one exact legacy marker before any overwrite.

    An empty cell receives a fresh sentinel. A nonempty cell must already carry that exact sentinel,
    except that a pre-sentinel generation may be promoted once from its regular result marker when
    label, seed, and any present sweep field match this invocation exactly. Malformed or mismatched
    state never authorizes cleanup.
    """
    expected = _ablation_cell_owner_payload(sweep_name, label, seed)
    owner_path = run_dir / _ABLATION_CELL_OWNER_FILENAME
    children = list(run_dir.iterdir())
    if not children:
        _write_json_atomic(owner_path, expected)
        return owner_path

    owner_present = (
        owner_path.exists()
        or owner_path.is_symlink()
        or _path_is_junction(owner_path)
    )
    if owner_present:
        if not _ablation_cell_owner_is_exact(run_dir, sweep_name, label, seed):
            raise ValueError(
                "ablation cell ownership sentinel does not match the requested sweep, label, and seed"
            )
        return owner_path

    marker_path = run_dir / "ablation_result.json"
    try:
        legacy = _read_regular_json_mapping(
            marker_path,
            role="legacy ablation cell ownership marker",
        )
    except ValueError as exc:
        raise ValueError(
            "nonempty ablation cell has no valid ownership sentinel or promotable legacy marker"
        ) from exc
    if (
        legacy.get("label") != label
        or type(legacy.get("seed")) is not int
        or legacy.get("seed") != seed
        or ("sweep" in legacy and legacy.get("sweep") != sweep_name)
    ):
        raise ValueError(
            "legacy ablation cell ownership marker does not match the requested sweep, label, and seed"
        )
    _write_json_atomic(owner_path, expected)
    return owner_path


def _start_owned_cell_generation(
    sweep_dir: Path,
    run_dir:   Path,

    *,
    sweep_name: str,
    label:      str,
    seed:       int,
) -> None:
    r"""Invalidate the old marker, then empty only this owned cell directory before recomputation.

    A matching resume never calls this helper. Before any existing file is overwritten, a versioned
    sentinel must bind the directory to this exact sweep, label, and seed; one regular legacy result
    marker may establish that ownership once. Writing the ``running`` marker then makes a partial
    cleanup or process crash non-reportable. Every other direct child except the sentinel is removed,
    preventing optional artifacts, figures, and checkpoints from a prior owned generation from
    surviving beside the new one. Symlinks and junctions are removed as links and never traversed.
    """
    if run_dir.parent != sweep_dir:
        raise ValueError("ablation cell directory must be a direct child of its sweep directory")
    if run_dir.exists() and (run_dir.is_symlink() or _path_is_junction(run_dir)):
        raise ValueError("ablation cell directory may not be a symlink or junction")
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError("ablation cell path exists but is not a directory")
    run_dir.mkdir(parents=False, exist_ok=True)
    if not _cell_dir_is_owned(sweep_dir, run_dir):
        raise ValueError("ablation cell directory resolves outside its sweep directory")

    owner_path = _authorize_ablation_cell_cleanup(
        run_dir,
        sweep_name,
        label,
        seed,
    )
    marker = run_dir / "ablation_result.json"
    _write_json_atomic(marker, {
        "status": "running",
        "sweep":  sweep_name,
        "label":  label,
        "seed":   seed,
    })
    for child in list(run_dir.iterdir()):
        if child in {marker, owner_path}:
            continue
        if child.is_symlink():
            child.unlink()
        elif _path_is_junction(child):
            child.rmdir()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _invalidate_code_drifted_cells(
    sweep_dir: Path,
    cells:     List[Tuple[str, Dict[str, Any], int]],
    error:     str,
) -> None:
    """Atomically downgrade every successful current-invocation marker after terminal code drift."""
    for label, _overrides, _seed in cells:
        run_dir = sweep_dir / _sanitize(label)
        marker_path = run_dir / "ablation_result.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(marker, Mapping) or marker.get("status") != "success":
            continue
        failed = dict(marker)
        failed.update({
            "status":                    "failed",
            "error_kind":                "code_identity_drift",
            "error":                     error,
            "cell_contract_fingerprint": None,
        })
        _write_json_atomic(marker_path, failed)
        try:
            (run_dir / "cell_contract.json").unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("  [%s] failed to remove drifted cell contract: %s", label, exc)


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
    sweep_name = _validated_sweep_name(sweep_name)
    sweep = SWEEPS[sweep_name]
    cell_seeds = _validated_sweep_seeds(sweep, seed)
    data_seed_override = _validated_data_seed_override()
    sweep_dir = _prepare_owned_output_child(output_dir, sweep_name, role="ablation sweep")
    runs = make_run_overrides(sweep_name)
    run_labels = [label for label, _overrides in runs]
    if any(not isinstance(label, str) or not label for label in run_labels):
        raise ValueError(f"sweep {sweep_name!r} has a non-string or empty label")
    duplicate_labels = sorted({label for label in run_labels if run_labels.count(label) > 1})
    if duplicate_labels:
        raise ValueError(
            f"sweep {sweep_name!r} has duplicate expanded label(s): {duplicate_labels}"
        )
    collect_diagnostics    = bool(sweep.get("collect_diagnostics", False))
    collect_extrapolation  = bool(sweep.get("collect_extrapolation", False))
    paired_token_bootstrap = bool(sweep.get("paired_token_bootstrap", False))
    required_diagnostic_keys, min_extrapolation_points = _sweep_required_outputs(sweep)
    # Multi-seed (I1/EXP-1): a sweep may declare ``seeds`` to replicate every cell across seeds for an
    # across-seed error bar. Each (cell, seed) gets its own ``{label}__s{seed}`` run dir and result row
    # (the seed also lives in the existing ``seed`` column), so the across-seed aggregate is a plain
    # group-by on the base label. A sweep WITHOUT ``seeds`` keeps the single-seed label/run-dir exactly,
    # so every existing sweep is byte-identical.
    multiseed = "seeds" in sweep
    cells = [((f"{label}__s{s}" if multiseed else label), overrides, s)
             for (label, overrides) in runs for s in cell_seeds]

    report_metadata = {
        "paired_token_bootstrap": paired_token_bootstrap,
        "required_diagnostic_keys": required_diagnostic_keys,
        "min_extrapolation_points": min_extrapolation_points,
        "forest_baseline_label":  sweep.get("forest_baseline_label"),
        "grid_x":                 sweep.get("grid_x"),
        "grid_y":                 sweep.get("grid_y"),
        "grid_x_values":          sweep.get("grid_x_values"),
        "grid_y_values":          sweep.get("grid_y_values"),
        "grid_baseline":          (list(sweep["grid_baseline"])
                                   if sweep.get("grid_baseline") is not None else None),
    }
    running_meta = {
        "sweep_name":              sweep_name,
        "description":             sweep["description"],
        "n_runs":                  len(cells),
        "n_successful_requested":  0,
        "failed_requested_labels": [label for label, _overrides, _seed in cells],
        "dataset":                 dataset,
        "device":                  str(device),
        "seed":                    (cell_seeds if multiseed else seed),
        "timestamp":               time.strftime("%Y-%m-%d %H:%M:%S"),
        "status":                  "running",
        "error":                   None,
        "aggregation_contract":    None,
        "gauge_purity":            _gauge_purity_summary([]),
        **report_metadata,
    }
    # Invalidate any prior complete view before even the invocation identity probes run. Cell
    # artifacts remain available for exact-contract resume, but a crash cannot leave the old
    # metadata/CSV pair externally visible as the outcome of this new invocation.
    _write_json_atomic(sweep_dir / "sweep_meta.json", running_meta)
    _write_sweep_csv(sweep_dir, [])

    try:
        invocation_code_identity = _validated_ablation_code_identity(_git_code_identity())
    except Exception as exc:
        error = f"pre-sweep code identity unavailable: {exc}"
        logger.error("  [%s -> sweep aborted]", error)
        running_meta.update({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status":    "incomplete",
            "error":     error,
        })
        _write_json_atomic(sweep_dir / "sweep_meta.json", running_meta)
        return []
    contract_code_identity: Mapping[str, object] = invocation_code_identity

    print(f"\n{'=' * 70}\nSWEEP: {sweep_name} ({len(cells)} runs"
          f"{f' = {len(runs)} cells x {len(cell_seeds)} seeds' if multiseed else ''})"
          f"\n  {sweep['description']}"
          f"\n  Output: {sweep_dir}{'  [resume ON]' if resume else ''}\n{'=' * 70}")

    diagnostic_flags = {
        "collect_diagnostics":    collect_diagnostics,
        "collect_extrapolation":  collect_extrapolation,
        "paired_token_bootstrap": paired_token_bootstrap,
    }
    try:
        sweep_source_identities: Optional[Dict[str, Dict[str, object]]] = (
            _ablation_source_identities(dataset))
    except Exception as exc:                                  # missing/corrupt/unreadable source
        logger.warning("  [pre-sweep corpus identity unavailable -> reuse forbidden] %s", exc)
        sweep_source_identities = None
    # An explicit empty mapping tells the contract builder that the invocation-owned snapshot failed.
    # It must not fall back to per-cell filesystem hashing; every cell then fails closed at publication.
    contract_source_identities: Mapping[str, object] = (
        sweep_source_identities if sweep_source_identities is not None else {})
    try:
        aggregation_contract: Optional[Dict[str, object]] = (
            _sweep_aggregation_contract(
                dataset,
                diagnostic_flags,
                data_seed_override=data_seed_override,
                max_tokens=max_tokens,
                max_steps=max_steps,
                seed_design=cell_seeds,
                source_identities=contract_source_identities,
                code_identity=contract_code_identity,
                device=str(device),
                required_diagnostic_keys=required_diagnostic_keys,
                min_extrapolation_points=min_extrapolation_points,
            )
            if sweep_source_identities is not None else None
        )
    except Exception as exc:
        logger.warning("  [aggregation contract unavailable -> sweep incomplete] %s", exc)
        aggregation_contract = None
    running_meta["aggregation_contract"] = aggregation_contract
    _write_json_atomic(sweep_dir / "sweep_meta.json", running_meta)
    results: List[Dict[str, Any]] = []
    for i, (label, overrides, cell_seed) in enumerate(cells):
        run_dir = sweep_dir / _sanitize(label)
        marker = run_dir / "ablation_result.json"
        contract_path = run_dir / "cell_contract.json"

        # Reuse is authorized only by a versioned contract that binds this cell to its code identity,
        # per-split corpus hashes, and semantic config (audit 2026-07-12 PB-01). The expected contract
        # is built inside the per-cell failure boundary for every recomputation, not only resume.
        # An unbuildable config or unhashable corpus returns None, which forbids both reuse and later
        # success publication because the generation active at training start is unknown.
        expected_contract = _expected_cell_contract_or_none(
            overrides, dataset, diagnostic_flags, seed=cell_seed,
            device=str(device),
            max_steps=max_steps, max_tokens=max_tokens,
            source_identities=contract_source_identities,
            code_identity=contract_code_identity,
            required_diagnostic_keys=required_diagnostic_keys,
            min_extrapolation_points=min_extrapolation_points)
        expected_gauge_fields: Dict[str, object] = {}
        try:
            if aggregation_contract is None:
                raise ValueError("invocation aggregation contract is unavailable")
            cohort_cell_contract, expected_gauge_fields = _expected_cell_contract_from_aggregation(
                {"overrides": overrides, "seed": cell_seed},
                aggregation_contract,
            )
            if expected_contract is None or cohort_cell_contract != expected_contract:
                raise ValueError("cell contract does not match the invocation aggregation contract")
        except Exception as exc:
            logger.warning("  [contract unavailable -> reuse forbidden] %s", exc)
            expected_contract = None
        if resume and marker.exists():
            # The contract binds the request (diagnostic_flags carries paired_token_bootstrap); a
            # separate post-contract validator binds the requested artifact's exact bytes/schema, so a
            # missing or drifted val_token_nats.pt forbids reuse even when the contract still matches.
            if (_cell_dir_is_owned(sweep_dir, run_dir)
                    and _ablation_cell_owner_is_exact(
                        run_dir, sweep_name, label, cell_seed)
                    and expected_contract is not None
                    and _cell_is_current(run_dir, expected_contract)
                    and _paired_token_artifact_is_current(run_dir, required=paired_token_bootstrap)):
                print(f"\n--- {i + 1}/{len(cells)}: {label}  [CACHED] ---")
                cached_result = json.loads(marker.read_text(encoding="utf-8"))
                cached_result.update(expected_gauge_fields)
                cached_result["sweep"] = sweep_name
                cached_result["label"] = label
                cached_result["seed"] = cell_seed
                cached_result["overrides"] = _jsonable(overrides)
                _write_json_atomic(marker, cached_result)
                results.append(cached_result)
                continue
            print(f"\n--- {i + 1}/{len(cells)}: {label}  [contract changed -> re-running] ---")
        else:
            print(f"\n--- {i + 1}/{len(cells)}: {label} ---")
        t0 = time.perf_counter()
        generation_started = False
        try:
            _start_owned_cell_generation(
                sweep_dir,
                run_dir,
                sweep_name=sweep_name,
                label=label,
                seed=cell_seed,
            )
        except Exception as exc:
            logger.exception("sweep %s / %s setup failed", sweep_name, label)
            result = {"label": label, "error_kind": "setup", "error": str(exc),
                      "primary_val_ppl": float("inf"), "seed": int(cell_seed),
                      "overrides": _jsonable(overrides)}
        else:
            generation_started = True
            try:
                result = run_single(label, overrides, run_dir, dataset=dataset, device=device,
                                    seed=cell_seed, collect_diagnostics=collect_diagnostics,
                                    collect_extrapolation=collect_extrapolation,
                                    paired_token_bootstrap=paired_token_bootstrap,
                                    max_tokens=max_tokens, max_steps=max_steps)
            except Exception as exc:                         # one training crash must not kill the sweep
                logger.exception("sweep %s / %s training crashed", sweep_name, label)
                result = {"label": label, "error_kind": "train", "error": str(exc),
                          "primary_val_ppl": float("inf"), "seed": int(cell_seed),
                          "overrides": _jsonable(overrides)}
        finally:
            _cleanup()

        loaded_sources_raw = result.pop("_loaded_data_sources", None)
        try:
            loaded_source_identities = _validated_ablation_source_identities(loaded_sources_raw)
        except (TypeError, ValueError):
            loaded_source_identities = None
        result.setdefault("error_kind", None)
        result["label"] = label
        result["seed"] = int(cell_seed)
        result["overrides"] = _jsonable(overrides)
        result.update(expected_gauge_fields or {
            "head_mixer_compatibility":    "unavailable",
            "head_mixer_gauge_compatible": False,
            "on_gauge_pure_path":          False,
        })
        result["collect_diagnostics"]   = collect_diagnostics
        result["collect_extrapolation"] = collect_extrapolation
        # The request flag always lands in the marker; the artifact identity fields default to null on
        # any path (crash, or a sweep that did not request the artifact) so the CSV/adapters stay whole.
        result["paired_token_bootstrap"] = paired_token_bootstrap
        result.setdefault("val_token_nats_path", None)
        result.setdefault("val_token_nats_sha256", None)
        result.setdefault("val_token_nats_size_bytes", None)
        result.setdefault("val_token_nats_numel", None)
        result.setdefault("val_token_nats_dtype", None)
        try:
            terminal_ppl = float(result["final_val_ppl"])
        except (KeyError, TypeError, ValueError):
            terminal_ppl = float("inf")
        result["final_val_ppl"] = terminal_ppl
        successful = result["error_kind"] is None and math.isfinite(terminal_ppl)

        # Requested collections are terminal artifacts. Every declared diagnostic must be finite,
        # and extrapolation must contain the contracted number of distinct finite sequence lengths.
        if successful and not _requested_outputs_are_complete(
                result,
                required_diagnostic_keys=required_diagnostic_keys,
                min_extrapolation_points=min_extrapolation_points):
            logger.warning("  [%s] requested outputs incomplete -> cell marked failed", label)
            successful = False

        # A production terminal checkpoint must be a real owned file under this cell. Its relative
        # path, size, and SHA-256 are published into the marker and revalidated on every resume.
        if successful:
            checkpoint_identity = _terminal_checkpoint_identity(run_dir, result)
            if checkpoint_identity is None:
                logger.warning("  [%s] terminal checkpoint is missing or unowned -> cell marked failed", label)
                successful = False
            else:
                result.update(checkpoint_identity)

        # A successful recomputation is cached only when its contract can be rebuilt now. Never
        # publish the pre-run resume contract: code or corpus identity can drift while training. If
        # a pre-run contract was available, require exact agreement with the post-run rebuild;
        # otherwise convert the cell to a failed result rather than mix two generations.
        contract = None
        if successful:
            if loaded_source_identities is None:
                logger.warning(
                    "  [%s] loaded corpus identity unavailable -> cell marked failed", label)
                successful = False
            else:
                post_run_contract = _expected_cell_contract_or_none(
                    overrides, dataset, diagnostic_flags, seed=cell_seed,
                    device=str(device),
                    max_steps=max_steps, max_tokens=max_tokens,
                    source_identities=loaded_source_identities,
                    code_identity=contract_code_identity,
                    required_diagnostic_keys=required_diagnostic_keys,
                    min_extrapolation_points=min_extrapolation_points)
                if expected_contract is None or post_run_contract is None:
                    successful = False
                elif dict(post_run_contract) != dict(expected_contract):
                    logger.warning(
                        "  [%s] contract drifted during recomputation -> cell marked failed", label)
                    successful = False
                else:
                    contract = post_run_contract
        if successful and not _terminal_checkpoint_is_current(run_dir, result):
            logger.warning("  [%s] terminal checkpoint changed before publication -> cell marked failed", label)
            successful = False
            contract = None
        result["status"] = "success" if successful else "failed"
        result["cell_contract_fingerprint"] = (
            semantic_config_fingerprint(contract)
            if successful and contract is not None else None
        )
        result["sweep"] = sweep_name
        result["wall_time_s"] = time.perf_counter() - t0

        # Publish the contract atomically (same-dir tmp + os.replace) BEFORE the completion marker, so
        # a reader that sees the success marker always sees a fully-written contract; a failed cell
        # publishes no contract. Neither file is written after a training exception.
        if generation_started and successful and contract is not None:
            _write_json_atomic(contract_path, contract)
        if generation_started:
            _write_json_atomic(marker, result)
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
        _write_sweep_csv(sweep_dir, _collect_sweep_results(
            sweep_dir,
            aggregation_contract=(aggregation_contract or {}),
        ))

    try:
        terminal_code_identity = _validated_ablation_code_identity(_git_code_identity())
        terminal_code_identity_error = None
    except Exception as exc:
        terminal_code_identity = None
        terminal_code_identity_error = f"terminal code identity snapshot failed: {exc}"
    if terminal_code_identity_error is not None:
        code_identity_error = terminal_code_identity_error
    else:
        code_identity_error = (
            None if terminal_code_identity == invocation_code_identity
            else "code identity drifted during the ablation invocation"
        )
    if code_identity_error is not None:
        _invalidate_code_drifted_cells(sweep_dir, cells, code_identity_error)

    # The final frame is the one compatible cohort only. A requested label counts as successful
    # exactly when its current marker, contract, optional paired artifact, and gauge disclosure all
    # passed that collector; historical survivors cannot make an incomplete invocation complete.
    union = _collect_sweep_results(
        sweep_dir,
        aggregation_contract=(aggregation_contract or {}),
    )
    collected_labels = {str(result["label"]) for result in union}
    failed_requested_labels = [
        label for label, _overrides, _cell_seed in cells
        if label not in collected_labels
    ]
    if code_identity_error is not None:
        sweep_error = code_identity_error
    elif aggregation_contract is None:
        sweep_error = "sweep aggregation contract was unavailable"
    elif failed_requested_labels:
        sweep_error = (
            f"{len(failed_requested_labels)} of {len(cells)} requested cells failed or lacked "
            "a current contract: " + ", ".join(failed_requested_labels)
        )
    else:
        sweep_error = None
    sweep_status = "complete" if sweep_error is None else "incomplete"
    gauge_purity = _gauge_purity_summary(union)

    # Publish the complete accumulated frame before the terminal metadata. A reader that observes
    # status="complete" therefore cannot see a truncated or previous-generation CSV.
    _write_sweep_csv(sweep_dir, union)
    _write_json_atomic(sweep_dir / "sweep_meta.json", {
        "sweep_name":  sweep_name,
        "description": sweep["description"],
        "n_runs":      len(cells),
        "n_successful_requested": len(cells) - len(failed_requested_labels),
        "failed_requested_labels": failed_requested_labels,
        "dataset":     dataset,
        "device":      str(device),
        "seed":        (cell_seeds if multiseed else seed),
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "status":      sweep_status,
        "error":       sweep_error,
        "aggregation_contract": aggregation_contract,
        "gauge_purity": gauge_purity,
        # PB-07 report metadata (null for ordinary sweeps): lets the ablation-forest / joint-LR-grid
        # adapters operate after a process restart, when only the persisted sweep view survives.
        **report_metadata,
    })

    if sweep_error is not None:
        print(f"\nSWEEP INCOMPLETE: {sweep_name}  ->  {sweep_error}")
        return []
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
    print(f"{'label':<34}{'val PPL':>12}{'params':>12}  gauge classification")
    print("-" * 104)
    for r in rows:
        ppl = "inf" if r["_ppl"] == float("inf") else f"{r['_ppl']:.3f}"
        params = f"{int(_as_float(r.get('n_params'))):,}" if r.get("n_params") not in ("", None) else "-"
        gauge = r.get("head_mixer_compatibility") or "unavailable"
        print(f"{r['label']:<34}{ppl:>12}{params:>12}  {gauge}")

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


def _sweep_is_complete(sweep_dir: Path) -> bool:
    """Return true only when one persisted sweep explicitly completed its invocation."""
    try:
        meta = json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(meta, Mapping) and meta.get("status") == "complete"


def _cross_sweep_cohort_identity(
    aggregation_contract: Mapping[str, object],
) -> Dict[str, object]:
    r"""Project a sweep contract to fields that make validation-PPL comparisons commensurate."""
    contract = _validated_sweep_aggregation_contract(aggregation_contract)
    return {
        key: contract[key]
        for key in (
            "schema_version",
            "baseline_semantic_config_fingerprint",
            "seed_design",
            "dataset",
            "device",
            "tokenizer_tag",
            "data_seed_override",
            "max_tokens",
            "max_steps",
            "source_identities",
            "code_identity",
        )
    }


def _sweep_matches_cohort(
    sweep_dir:       Path,
    cohort_identity: Mapping[str, object],
) -> bool:
    """Whether one complete persisted sweep belongs to an exact comparison cohort."""
    try:
        meta = json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))
        return (
            isinstance(meta, Mapping)
            and meta.get("status") == "complete"
            and _cross_sweep_cohort_identity(meta["aggregation_contract"])
            == dict(cohort_identity)
        )
    except Exception:
        return False


def summarize_sweeps(
    output_dir: Path,

    *,
    cohort_identity: Optional[Mapping[str, object]] = None,
) -> None:
    r"""Cross-sweep comparison table: the best (lowest val PPL) cell of every persisted sweep.

    Printed once after all sweeps in a run (the per-sweep tables come from ``analyze_sweep`` as
    each sweep finishes). Scans EVERY sweep dir under ``output_dir`` so earlier-session sweeps
    are included, not just this run's.
    """
    print(f"\n{'=' * 70}\nBEST PER SWEEP  ({output_dir})\n{'=' * 70}")
    sweep_dirs = [
        directory
        for directory in sorted(output_dir.iterdir())
        if (
            directory.is_dir()
            and (directory / "sweep_results.csv").exists()
            and cohort_identity is not None
            and _sweep_matches_cohort(directory, cohort_identity)
        )
    ]
    if not sweep_dirs:
        print("No completed sweeps found in the current comparison cohort.")
        return
    print(f"{'sweep':<24}{'best config':<30}{'val PPL':>10}  gauge classification")
    print("-" * 104)
    for d in sweep_dirs:
        rows = [r for r in _read_sweep_csv(d) if _as_float(r.get("primary_val_ppl")) < float("inf")]
        if not rows:
            continue
        best = min(rows, key=lambda r: _as_float(r.get("primary_val_ppl")))
        gauge = best.get("head_mixer_compatibility") or "unavailable"
        print(f"{d.name:<24}{best['label']:<30}{_as_float(best['primary_val_ppl']):>10.3f}  {gauge}")


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


def _gauge_disclosure_text(gauge_purity: object) -> str:
    """Return the explicit gauge/head-mixer disclosure printed on every ablation figure."""
    if not isinstance(gauge_purity, Mapping):
        return "Gauge classification: unavailable"
    classifications = gauge_purity.get("classifications_by_label")
    values = (
        sorted({str(value) for value in classifications.values()})
        if isinstance(classifications, Mapping) else []
    )
    value_text = ", ".join(values) if values else "unavailable"
    if gauge_purity.get("contains_independent_head_nonintertwiner") is True:
        return (
            "Gauge classification: independent_head_nonintertwiner present; "
            "this figure is not gauge-pure"
        )
    if gauge_purity.get("all_rows_on_gauge_pure_path") is True:
        return f"Gauge classification: all rows gauge-pure; head mixer(s): {value_text}"
    return f"Gauge classification: not all rows are gauge-pure; head mixer(s): {value_text}"


def _sweep_gauge_purity(sweep_dir: Path) -> object:
    """Read one sweep's persisted gauge-purity summary without trusting plot input rows."""
    try:
        metadata = json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    return metadata.get("gauge_purity") if isinstance(metadata, Mapping) else None


def _annotate_gauge_disclosure(fig: Any, gauge_purity: object) -> None:
    """Add a visible, figure-level gauge disclosure independent of individual axes."""
    fig.text(
        0.5,
        0.01,
        _gauge_disclosure_text(gauge_purity),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#b22222",
    )


def _save_ablation_figure(fig: Any, out: Path, sweep_dir: Path, plt: Any) -> None:
    """Annotate, save, and close one legacy ablation figure under the common disclosure policy."""
    _annotate_gauge_disclosure(fig, _sweep_gauge_purity(sweep_dir))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


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
    _save_ablation_figure(fig, out, sweep_dir, plt)
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
    _save_ablation_figure(fig, out, sweep_dir, plt)
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
    fig = plot_rank_residual_by_depth(arms)
    _save_ablation_figure(fig, out, sweep_dir, plt)
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
    _save_ablation_figure(plot_entropy_ppl_gap(cells_ppl), out, sweep_dir, plt)
    print(f"  figure -> {out}")
    if len(cells_gap) >= 2:
        out = fig_dir / f"{sweep_dir.name}_cov_gap.png"
        _save_ablation_figure(plot_cov_gap_vs_kappa(cells_gap), out, sweep_dir, plt)
        print(f"  figure -> {out}")


def _compatible_cell_dirs(sweep_dir: Path) -> List[Path]:
    """Return only cell directories admitted by the persisted aggregation contract."""
    return [
        sweep_dir / _sanitize(str(result["label"]))
        for result in _collect_sweep_results(sweep_dir)
    ]


def _plot_wallclock_convergence(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Write ``figures/<sweep>_wallclock_convergence.png`` -- the D1/EXP-8 per-wall-clock convergence
    overlay (val PPL vs cumulative wall time, one line per arm, with steps/wall-to-target annotated) --
    from each cell's ``metrics.csv`` eval rows (val_ppl + wall_clock_s). A no-op unless >= 2 cells carry
    >= 2 eval points, so it is safe to call after every sweep (the gauge M-step sweeps populate it)."""
    arms: List[Dict[str, Any]] = []
    for run_dir in _compatible_cell_dirs(sweep_dir):
        cell = run_dir / "metrics.csv"
        if not cell.is_file():
            continue
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
    _save_ablation_figure(plot_wallclock_convergence(arms), out, sweep_dir, plt)
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
    _save_ablation_figure(plot_gauge_transport_bars(cells), out, sweep_dir, plt)
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
    _save_ablation_figure(plot_ppl_equivariance_bars(cells), out, sweep_dir, plt)
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
    _save_ablation_figure(plot_kappa_dispersion(cells), out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_gauge_residual_drift(sweep_dir: Path, fig_dir: Path) -> None:
    r"""A2/EXP-9 builder-break gauge residual vs step (tied vs untied) from each cell's metrics.csv
    ``val_builder_resid`` eval series. No-op unless >= 2 cells carry >= 2 eval points."""
    arms: List[Dict[str, Any]] = []
    for run_dir in _compatible_cell_dirs(sweep_dir):
        cell = run_dir / "metrics.csv"
        if not cell.is_file():
            continue
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
    _save_ablation_figure(plot_gauge_residual_drift(arms), out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_holonomy_trainability(sweep_dir: Path, fig_dir: Path) -> None:
    r"""A4/EXP-15 holonomy-vs-||connection|| scatter from each cell's metrics.csv (connection_w_norm +
    holonomy_deviation per eval). connection_w_norm is logged only on a regime_ii run, so this no-ops
    unless a cell carries >= 2 such eval rows (the flat arm is correctly excluded)."""
    arms: List[Dict[str, Any]] = []
    for run_dir in _compatible_cell_dirs(sweep_dir):
        cell = run_dir / "metrics.csv"
        if not cell.is_file():
            continue
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
    _save_ablation_figure(plot_holonomy_trainability(arms), out, sweep_dir, plt)
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
    _save_ablation_figure(plot_mu_precond(cells), out, sweep_dir, plt)
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
    _save_ablation_figure(plot_renyi_saturation(cells), out, sweep_dir, plt)
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
    _save_ablation_figure(
        plot_pos_extrapolation(arms, train_n=(min(all_n) if all_n else None)),
        out,
        sweep_dir,
        plt,
    )
    print(f"  figure -> {out}")


def _plot_sensitivity(
    output_dir: Path,
    fig_dir:    Path,

    *,
    cohort_identity: Optional[Mapping[str, object]] = None,
) -> None:
    r"""Cross-sweep comparison: a PPL-range (worst - best) bar per sweep, sorted by sensitivity.

    Made once after all sweeps. Scans EVERY persisted sweep under ``output_dir`` (not just this
    run's), matching the per-sweep figures' accumulated view.
    """
    sweep_dirs = [
        directory
        for directory in sorted(output_dir.iterdir())
        if (
            directory.is_dir()
            and (directory / "sweep_results.csv").exists()
            and cohort_identity is not None
            and _sweep_matches_cohort(directory, cohort_identity)
        )
    ]
    sensitivity: List[Tuple[str, float, str, str]] = []      # sweep, PPL range, best label, gauge class
    gauge_summaries: List[Mapping[str, object]] = []
    for d in sweep_dirs:
        rows = [r for r in _read_sweep_csv(d) if _as_float(r.get("primary_val_ppl")) < float("inf")]
        if not rows:
            continue
        gauge_summary = _sweep_gauge_purity(d)
        if isinstance(gauge_summary, Mapping):
            gauge_summaries.append(gauge_summary)
        ppls = [_as_float(r["primary_val_ppl"]) for r in rows]
        best = min(rows, key=lambda r: _as_float(r["primary_val_ppl"]))
        sensitivity.append((
            d.name,
            max(ppls) - min(ppls),
            best["label"],
            best.get("head_mixer_compatibility") or "unavailable",
        ))
    if not sensitivity:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    sensitivity.sort(key=lambda t: t[1], reverse=True)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(sensitivity))))
    ax.barh(range(len(sensitivity)), [s[1] for s in sensitivity], color="#d62728", alpha=0.8)
    ax.set_yticks(range(len(sensitivity)))
    ax.set_yticklabels([f"{s[0]}\n(best: {s[2]}; gauge: {s[3]})" for s in sensitivity])
    ax.invert_yaxis()
    ax.set_xlabel("validation PPL range (worst - best)")
    ax.set_title("hyperparameter sensitivity")
    fig.tight_layout()
    classifications = {
        f"{index}:{label}": value
        for index, summary in enumerate(gauge_summaries)
        for label, value in (
            summary.get("classifications_by_label", {}).items()
            if isinstance(summary.get("classifications_by_label"), Mapping) else ()
        )
    }
    _annotate_gauge_disclosure(fig, {
        "classifications_by_label": classifications,
        "contains_independent_head_nonintertwiner": any(
            summary.get("contains_independent_head_nonintertwiner") is True
            for summary in gauge_summaries
        ),
        "all_rows_on_gauge_pure_path": bool(gauge_summaries) and all(
            summary.get("all_rows_on_gauge_pure_path") is True
            for summary in gauge_summaries
        ),
    })
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / "sensitivity_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"comparison figure -> {out}")


def emit_registered_figures(
    specs:      Any,
    context:    Mapping[str, object],
    output_dir: Path,
) -> List[Path]:
    r"""Compatibility wrapper whose Matplotlib-backed implementation is imported only on demand."""
    from vfe3.viz.specs import emit_registered_figures as _emit_registered_figures

    return _emit_registered_figures(specs, context, output_dir)


def _render_sweep_figures(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Render one complete sweep's registered and legacy figures inside a plotting worker."""
    from vfe3.viz.specs import FigureSpec
    from vfe3.viz.sweep_adapters import ablation_forest_kwargs, lr_grid_heatmap_kwargs

    meta_path = sweep_dir / "sweep_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, Mapping) or meta.get("status") != "complete":
        raise ValueError(f"sweep {sweep_dir.name!r} has no complete metadata")
    rows = _collect_sweep_results(sweep_dir)
    for row in rows:
        row["_cell_dir"] = str(sweep_dir / _sanitize(str(row["label"])))
    report_context = {
        "sweep_dir":      sweep_dir,
        "rows":           rows,
        "gauge_purity":   meta.get("gauge_purity"),
        "baseline_label": meta.get("forest_baseline_label"),
        "grid_x":         meta.get("grid_x"),
        "grid_y":         meta.get("grid_y"),
        "grid_x_values":  meta.get("grid_x_values"),
        "grid_y_values":  meta.get("grid_y_values"),
        "baseline":       tuple(meta["grid_baseline"]) if meta.get("grid_baseline") else None,
    }
    specs = []
    if meta.get("paired_token_bootstrap") is True:
        specs.append(FigureSpec(
            "ablation_forest",
            f"{sweep_dir.name}_ablation_forest.png",
            lambda ctx: ablation_forest_kwargs(
                ctx["sweep_dir"],
                ctx["baseline_label"],
                admitted_rows=ctx["rows"],
            ),
            postprocess=lambda fig, ctx: _annotate_gauge_disclosure(
                fig, ctx["gauge_purity"]
            ),
        ))
    if meta.get("grid_x") and meta.get("grid_y"):
        specs.append(FigureSpec(
            "lr_grid_heatmap",
            f"{sweep_dir.name}_lr_grid_heatmap.png",
            lambda ctx: lr_grid_heatmap_kwargs(
                ctx["rows"],
                ctx["grid_x"],
                ctx["grid_y"],
                ctx["grid_x_values"],
                ctx["grid_y_values"],
                ctx["baseline"],
            ),
            postprocess=lambda fig, ctx: _annotate_gauge_disclosure(
                fig, ctx["gauge_purity"]
            ),
        ))
    emit_registered_figures(specs, report_context, fig_dir)
    _plot_one_sweep(sweep_dir, fig_dir)
    _plot_seed_aggregate(sweep_dir, fig_dir)
    _plot_rank_collapse(sweep_dir, fig_dir)
    _plot_attention_entropy(sweep_dir, fig_dir)
    _plot_wallclock_convergence(sweep_dir, fig_dir)
    _plot_gauge_transport(sweep_dir, fig_dir)
    _plot_cg_coupling(sweep_dir, fig_dir)
    _plot_kappa_dispersion(sweep_dir, fig_dir)
    _plot_gauge_residual_drift(sweep_dir, fig_dir)
    _plot_pos_extrapolation(sweep_dir, fig_dir)
    _plot_renyi_saturation(sweep_dir, fig_dir)
    _plot_mu_precond(sweep_dir, fig_dir)
    _plot_holonomy_trainability(sweep_dir, fig_dir)


def _run_ablation_figures_isolated(
    output_dir: Path,

    *,
    scope:           str,
    invalidate:      bool                          = False,
    cohort_identity: Optional[Mapping[str, object]] = None,
) -> bool:
    r"""Render one ablation figure scope in a disposable process with child-only OMP policy."""
    output_dir = output_dir.resolve()
    with _unique_sibling_temp(output_dir / "ablation_figure_request.json") as request_path:
        request_path.write_text(json.dumps({
            "mode":    "ablation",
            "run_dir": str(output_dir),
            "scope":   scope,
            "invalidate": invalidate,
            "cohort_identity": (
                dict(cohort_identity) if cohort_identity is not None else None
            ),
        }), encoding="utf-8")
        environment = os.environ.copy()
        environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        environment["VFE3_FIGURE_REQUEST"] = str(request_path)
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            completed = run_process_tree(
                [sys.executable, "-m", "vfe3.viz.figure_worker"],
                cwd=str(Path(__file__).resolve().parent),
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_ABLATION_FIGURE_TIMEOUT_SECONDS,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "isolated ablation figure process exceeded %d seconds; numeric sweep results are saved",
                _ABLATION_FIGURE_TIMEOUT_SECONDS,
            )
            return False
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "isolated ablation figure process could not start (%s); numeric sweep results are saved",
                exc,
            )
            return False
    if completed.stdout.strip():
        logger.info("isolated ablation figure process:\n%s", completed.stdout.rstrip())
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no stderr"
        logger.warning(
            "isolated ablation figure process exited with code %d (%s); numeric sweep results are saved",
            completed.returncode,
            detail[-4000:],
        )
        return False
    if completed.stderr.strip():
        logger.info(
            "isolated ablation figure process diagnostics:\n%s",
            completed.stderr.rstrip(),
        )
    return True


# =============================================================================
# MAIN  (click-to-run; edit CONFIG above)
# =============================================================================

def _write_ablation_run_status(
    output_dir: Path,

    *,
    status:                str,
    requested_sweeps:      List[str],
    incomplete_sweeps:     List[str],
    failed_figure_scopes:  List[str],
) -> None:
    """Atomically publish the terminal completeness of one click-to-run invocation."""
    _write_json_atomic(output_dir / "ablation_run_status.json", {
        "schema_version":       1,
        "status":               status,
        "requested_sweeps":     list(requested_sweeps),
        "incomplete_sweeps":    list(dict.fromkeys(incomplete_sweeps)),
        "failed_figure_scopes": list(dict.fromkeys(failed_figure_scopes)),
        "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def main() -> int:
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
        return 0

    # ---- contiguous run: per sweep { train -> analyze table -> PPL figure }, then comparison ----
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])

    sweep_names = SWEEP_ORDER if CONFIG["sweep"] is None else [CONFIG["sweep"]]
    for name in sweep_names:
        if name not in SWEEPS:
            raise ValueError(f"unknown sweep {name!r}; choose from {sorted(SWEEPS)}")
    validate_sweeps(list(SWEEPS))                            # all declared arms must construct upfront

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nVFE_3.0 ablation suite\n  device:  {device}\n  dataset: {CONFIG['dataset']}"
          f"\n  output:  {output_dir}\n  seed:    {CONFIG['seed']}"
          f"\n  sweeps:  {', '.join(sweep_names)}")

    current_cohort: Optional[Dict[str, object]] = None
    incomplete_sweeps: List[str] = []
    failed_figure_scopes: List[str] = []
    _write_ablation_run_status(
        output_dir,
        status="running",
        requested_sweeps=list(sweep_names),
        incomplete_sweeps=incomplete_sweeps,
        failed_figure_scopes=failed_figure_scopes,
    )
    if not _run_ablation_figures_isolated(
            output_dir,
            scope="__sensitivity__",
            invalidate=True):
        failed_figure_scopes.append("__sensitivity__:invalidate")
        incomplete_sweeps.extend(sweep_names)
        logger.error(
            "could not invalidate prior sensitivity figures; refusing to publish a mixed generation"
        )
        _write_ablation_run_status(
            output_dir,
            status="incomplete",
            requested_sweeps=list(sweep_names),
            incomplete_sweeps=incomplete_sweeps,
            failed_figure_scopes=failed_figure_scopes,
        )
        return 1
    for name in sweep_names:
        if not _run_ablation_figures_isolated(output_dir, scope=name, invalidate=True):
            failed_figure_scopes.append(f"{name}:invalidate")
            incomplete_sweeps.append(name)
            logger.error(
                "could not invalidate prior %r figures; refusing to run beside stale output", name
            )
            continue
        run_sweep(name, output_dir, dataset=CONFIG["dataset"], device=device,
                  seed=CONFIG["seed"], resume=CONFIG["resume"],
                  max_tokens=CONFIG["max_tokens"], max_steps=CONFIG["max_steps"])
        sweep_dir = output_dir / name

        try:
            meta = json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("sweep %s has no readable completion metadata: %s", name, exc)
            incomplete_sweeps.append(name)
            continue
        if not isinstance(meta, Mapping) or meta.get("status") != "complete":
            logger.error(
                "sweep %s is incomplete; later requested sweeps will still run", name)
            incomplete_sweeps.append(name)
            continue
        try:
            sweep_cohort = _cross_sweep_cohort_identity(meta["aggregation_contract"])
        except Exception as exc:
            logger.error("sweep %s has invalid comparison metadata: %s", name, exc)
            incomplete_sweeps.append(name)
            continue
        if current_cohort is None:
            current_cohort = sweep_cohort
        elif sweep_cohort != current_cohort:
            logger.error("sweep %s belongs to a different comparison cohort", name)
            incomplete_sweeps.append(name)
            continue
        analyze_sweep(sweep_dir)                            # this sweep's numeric table (accumulated)
        if not _run_ablation_figures_isolated(output_dir, scope=name):
            failed_figure_scopes.append(f"{name}:render")

    # ---- after all sweeps: the cross-sweep comparison ----
    if current_cohort is not None:
        if not _run_ablation_figures_isolated(
                output_dir,
                scope="__sensitivity__",
                cohort_identity=current_cohort):
            failed_figure_scopes.append("__sensitivity__:render")
        summarize_sweeps(output_dir, cohort_identity=current_cohort)
    if incomplete_sweeps:
        logger.error(
            "incomplete sweeps withheld from analysis: %s",
            ", ".join(incomplete_sweeps),
        )
    status = "complete" if not incomplete_sweeps and not failed_figure_scopes else "incomplete"
    _write_ablation_run_status(
        output_dir,
        status=status,
        requested_sweeps=list(sweep_names),
        incomplete_sweeps=incomplete_sweeps,
        failed_figure_scopes=failed_figure_scopes,
    )
    if failed_figure_scopes:
        logger.error(
            "requested figure scopes failed: %s",
            ", ".join(failed_figure_scopes),
        )
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
