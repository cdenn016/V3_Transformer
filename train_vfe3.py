r"""Click-to-run training entry for the VFE_3.0 transformer.

Edit the ``config`` dict below, pick a ``DATASET``,
then run ``python train_vfe3.py``. There is no CLI arg parsing.

The ``config`` dict exposes the commonly-tuned ``VFE3Config`` toggles, grouped exactly as
in ``vfe3/config.py``; each registry-backed ``*_mode`` / ``*_family`` / ``*_group`` field
lists its valid keys inline. Any ``VFE3Config`` field omitted here simply takes its dataclass
default -- add it to this dict to tune it. The default ``DATASET = "wikitext-103"`` trains on the
cached gpt2/tiktoken corpus (vocab 50257) under ``~/.cache/tokenized_cache``; the
``config`` defaults (``vocab_size=50257``) are kept consistent with it so click-to-run
works out of the box. ``MAX_TOKENS`` caps the training stream for fast smoke runs.

A missing tokenized cache raises ``FileNotFoundError`` rather than substituting toy data:
held-out numbers are never silently computed on a synthetic stream and mislabeled as the
real corpus. Build the corpus cache first (see ``vfe3/data``).

A full ``max_steps`` run on the 116.8M-token wikitext-103 train split is a real (not
smoke) job: run it on the CUDA interpreter (the RTX 5090), or drop ``MAX_TOKENS`` /
``max_steps`` for a quick slice on CPU.
"""

import os
if os.environ.get("VFE3_ALLOW_DUPLICATE_OPENMP") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Third-party toolchain noise, filtered BEFORE torch is imported (triton emits its
# "Failed to find cuobjdump/nvdisasm" UserWarnings while torch is loading, and a warning
# already emitted cannot be un-emitted). Disassembly tools only -- absent on the Windows pip
# wheel, unused by this project, and irrelevant to numerics or throughput. See vfe3/quiet.py.
from vfe3.quiet import silence_toolchain_warnings
silence_toolchain_warnings()

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import (
    _tokenizer_tag,
    _validate_path_component,
    cached_token_count,
    make_dataloader,
)
from vfe3.runtime import seed_everything
from vfe3.train import _fmt_tau, evaluate, train





# Cached tokenized corpus (gpt2/tiktoken -> vocab_size 50257). Caches live in
# ~/.cache/tokenized_cache; a missing cache raises (no synthetic substitution).
#   "wikitext-103" | "wikitext-2" | "wiki-en" | "wiki-ja" | "wiki-ar" 100277
DATASET = "wikitext-103"

# Cap the *training* stream for fast smoke runs (the validation split is always read
# in full -- it is small). None = the full corpus (116.8M tokens for wikitext-103).
MAX_TOKENS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multi-seed training: launch NUM_RUNS independent runs back-to-back on click-to-run, run i using
# SEEDS[i]. Each run is fully independent (its own model, RNG, and artifacts dir -- the seed is in the
# run-folder label so they never collide). NUM_RUNS=1 with SEEDS=[] keeps the single-run path on the
# config `seed` above, unchanged. Example: NUM_RUNS=3, SEEDS=[3, 64, 23] trains all three seeds.
NUM_RUNS = 1
SEEDS    = 6,#[6,23,54,66,122]        # 54 is best seed so far; must list at least NUM_RUNS seeds when NUM_RUNS > 1

# DATA_SEED (EXP-1 variance floor): when set to an int, the TRAIN loader's shuffle order is fixed to
# this seed via an explicit generator, INDEPENDENT of the per-run model seed -- so a multi-seed run
# shares ONE batch order across seeds and the measured across-seed SD is init+optimization variance,
# NOT data-order noise. None (default) keeps the legacy behavior (shuffle drawn from the global RNG,
# which the post-build reseed pins to cfg.seed -- byte-identical to before, reproducible vs ablation.py).
DATA_SEED = 3


config = dict(
    

    #################################
    #            Training
    #################################
    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*  100277)
    
    embed_dim                 = 20,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 2,
    
    max_seq_len               = 128,                 # N, context length
    eval_stride               = None,
    
    batch_size                = 64,
    max_steps                 = 15000,
    
    n_layers                  = 1,                   # L, number of blocks
    n_e_steps                 = 1 ,                   # T, E-step inner iterations
    s_e_step_n_iter           = None,              #None = same as n_e_steps
    
    seed                      = 6,
    warmup_steps              = 100,
    
    #################################
    # f-divergence and e/m family
    #################################
    
    divergence_family         = "renyi",   # "renyi", "squared_hellinger","bhattacharyya", "jeffreys",
    renyi_order               = 1.0,       # Renyi order (1.0 -> KL)

    family                    = "gaussian_full", # "gaussian_diagonal" | "gaussian_full" | "laplace_diagonal" | "gaussian_frame_diagonal"
                                                     # | "gaussian_diagonal_exact" (single covariance toggle; diagonal_covariance is derived)
                                                     # "gaussian_frame_diagonal" (default OFF): covariance diagonal in the agent's OWN fiber frame,
                                                     # Sigma_i = U_i diag(sigma_i) U_i^T, which IS closed under GL(K) -- so the pair energy is the EXACT
                                                     # full-covariance divergence at diagonal cost instead of the diagonal-of-sandwich truncation.
                                                     # CONSEQUENCE: for the Regime-I coboundary the relative frame cancels, so Omega leaves the pair
                                                     # energy and the gauge no longer shapes the belief coupling. A DIFFERENT model, not a refinement.
                                                     # REQUIRES e_step_update="gradient": mm_exact is a hand kernel registered only for
                                                     # "gaussian_diagonal", so this family routes to the autograd oracle (slower per step).
                                                     # NOTE phi_embed receives NO gradient under this family (nothing on the belief path reads it),
                                                     # so it stays at init and decays under phi_weight_decay -- compare against gauge_transport="off".
                                                     #
                                                     # "gaussian_diagonal_exact" (default OFF): the SAME diagonal Gaussian as "gaussian_diagonal"
                                                     # (same self-divergence, decode, retraction); only the pair energy differs. Instead of pushing
                                                     # the key FORWARD and truncating Omega diag(s_j) Omega^T to its diagonal, it pulls the QUERY BACK
                                                     # by Omega^{-1} (a divergence is invariant under a common invertible pushforward), which leaves
                                                     # the second argument exactly diagonal and needs the first only through its diagonal and its
                                                     # determinant. The energy is then the EXACT congruence KL at the same cost, and unlike
                                                     # "gaussian_frame_diagonal" the gauge STAYS in the coupling (phi keeps its gradient).
                                                     # REQUIRES divergence_family="renyi" + renyi_order=1.0: the pullback identity is KL-specific
                                                     #   (a Renyi blend of the dense pushforward with a diagonal covariance does not reduce). Raises.
                                                     # REQUIRES e_step_update="gradient": no hand kernel is registered for this family, so it routes
                                                     #   to the autograd oracle (slower per step); mm_exact is rejected at config time.
                                                     # REQUIRES oracle_unroll_grad=True under e_step_gradient="unroll": the oracle otherwise returns a
                                                     #   DETACHED tangent and phi_embed receives no gradient -- silently a frozen-gauge run.
                                                     # TRANSPORT: flat vertex-factored cocycle (the fast route inverts by the pair transpose
                                                     #   Omega_ij^{-1} = Omega_ji) or a dense pairwise Omega (the O(N^2 K^3) reference route).
                                                     #   regime_ii / direct-link / RoPE-wrapped transports raise rather than silently truncating.
    
    #################################
    #        Initialization
    #################################
    mu_init_std               = 0.001, #0.065,     # std of the random mean table mu_embed
    sigma_init                = 2.5,         # constant initial coordinate variance (sigma_log = log of this)
    phi_scale                 = 0.06,      # std
    pos_phi_scale             = 0.02,                # learned-table init scale AND frozen per-position step
    
    
    e_step_mu_precond         = "fisher",       # "fisher" | "raw"
    #################################
    #        Encode/Decode          #
    #################################
    decode_bias               = True,     # only if use_prior_bank = False
    use_head_mixer            = False,      # opt-in Schur-commutant head mixer (needs >=2 equal blocks (block_glk/tied_block_glk) OR a labeled irrep tower (so_n/sp_n: per-isotypic-component mixing; mults-one towers get scalar gains));
                                           # breaks strict equivariance under block_glk (exact at init); EXACT under tied_block_glk (full-cov)
    
    use_prior_bank            = True,               # True: KL-to-prior decode (pure path). False: linear projection
                                                     # mu->logits ablation (encode stays on the prior bank)
    decode_tau                = 0.01,
    decode_mode               = 'full_chunked',  #"family_chunked (set chunks to 512 or default/K^2)", "full_chunked", "diagonal_chunked", "expected_likelihood_chunked", "full", "family", "family_chunked" (family/family_chunked: divergence-consistent KL-to-prior decode, use_prior_bank=True)
    decode_chunk_size         = 8192,
    
    encode_mode               = "per_token",   #"per_token_additive"
    
    
    oracle_unroll_grad        = True,
    
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
    transport_chart_max_norm  = None, 
    phi_mstep_max_matrix_norm = 12,
    
    m_phi_group_trust_radius  = 0.1,          # embedded Frobenius bound on the group factor
    
    phi_precond_mode          = "pullback_per_block",  # "none" | "clip" | "killing" | "killing_per_block" | "pullback" | "pullback_per_block"
                                                       # needs e_phi_lr>0
    
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
     transport_mode            = "flat",     # "flat" 
                                                       # "regime_ii"    
                                                       #  "regime_ii_covariant" 
                                                       # "regime_ii_link"   
                                                       # "regime_ii_link_charted"
                                           
    cocycle_relaxation        =  1.0,        # regime_ii / regime_ii_covariant homotopy: 0.0 -> flat, 1.0 -> fully relaxed (ignored by flat)
    cross_couplings           =  None, #[(0, 1)],       # off-block GL(K) head pairs e.g. [(0, 1)]; block_glk only (None = block-diagonal gauge)
                                               #if enabled and head-mixer = True or causal_alibi it will fail
    close_basis               =  False,
    ####################################
    #       Positional Encoding
    #    BCH gauge-frame PE (pos_phi)
    #     gauge-RoPE (pos_rotation)
    ####################################

    pos_phi                   = "learned",           # "none" (pure path) | "learned" | "frozen"
    pos_rotation              = "none",              # "none" | "rope" (block-diagonal positional rotation folded into transport)
    pos_phi_compose           = "group_product",     # composition chart: "bch" | "euclidean" |"group_product"
               
    
    rope_base                 = 100.0,               # rotary frequency base
    rope_insertion            = "right",             # "right" (PURE: frame V_i = U_i R_i^T, so Omega_ij = U_i R(theta_j - theta_i) U_j^-1 --
                                                     #   the relative rotation sits BETWEEN query and key content, per GL(K)_attention.tex and
                                                     #   Su et al. Eq 16. Exactly (i-j)-dependent for any phi, and exactly gauge covariant.)
                                                     # "left" (LEGACY, what shipped pre-2026-07-26: Omega_ij = R_i U_i U_j^-1 R_j^T, rotations
                                                     #   OUTSIDE the content operator, so the transport is conjugated by the query's ABSOLUTE
                                                     #   angle. Measured on the K=20 rope run: 69% of the pair-energy spread is absolute-position
                                                     #   contamination and gauge covariance fails at 5.6. Kept only to reproduce old runs.)
    rope_full_gauge           = False,               # rotate the covariance sandwich too (REQUIRES family="gaussian_full")
    rope_on_value             = False,
    
    ######################################
    #                Self Energy:  
    #        Sum_i alpha_i * KL(q_i||p_i)
    ######################################
    lambda_alpha_mode          = "constant",  # "constant" | "state_dependent" | "state_dependent_per_coord"
    lambda_h_mode              = "constant",  # "constant" | "state_dependent" (lambda_h*=c0_h/(b0_h+KL); +R_h)
    
    b0                         = 1.0,                 # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0                         = 1.0,                 # state-dependent alpha shape (numerator)
       
    lambda_alpha               = 1,          # constant self-coupling value
    lambda_h                   = 0.0,       # hyper-prior weight lambda_h * mean_i KL(s_i||r) (0 = OFF; >0 creates s/r tables)
    
    
    b0_h                       = 1.0,        # state-dependent lambda_h shape: lambda_h* = c0_h/(b0_h + KL(s||r))
    c0_h                       = 1.0,        # state-dependent lambda_h shape (numerator); max precision c0_h/b0_h

    # Further Regularizers
    mass_phi                   = 0.0,       # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight = 0.0,      # alpha_hat * sum_i KL(q_i*||p_i) M-step term (0 = OFF) 0.25
    
    
    ##################################################
    #              Attention Energy: 
    # lambda_beta*Sum_i beta_ij * KL(q_i||Omega_ij q_j) 
    ##################################################
    
    lambda_beta                = 1.0,        # belief-coupling block weight (1.0 = pure F)    
    lambda_gamma               = 0,       # model-channel coupling (0 = OFF; >0 creates s tables, predictively inert by default)
         

    ########################################
    #     Attention Belief/Model Settings
    #            & Temperatures
    ########################################
    
    kappa_beta                   = 1, #[1, 0.5],        # tau = kappa * sqrt(d_head); kappa=1 -> Vaswani temperature
    kappa_gamma                  = 1, #[1, 0.5],        # model-channel temperature tau_gamma = kappa_gamma*sqrt(d_head)

    learnable_kappa_beta         = False,       # learn per-head kappa_beta = exp(log_kappa_beta), init from kappa_beta above
                                             # (t5-exception family; freezes under detach/straight_through E-step)
    learnable_kappa_gamma        = False,       # learn per-head kappa_gamma (trains under any estimator on the scored
                                             # lambda_gamma>0 path; under s_e_step needs an 'unroll' E-step)

    beta_attention_prior         = "causal_alibi_noself",        # "uniform" | "causal" | "alibi" | "causal_alibi" | "windowed" | "causal_windowed" | "t5_relative_bias"
    gamma_attention_prior        = "causal_alibi_noself",        # model-channel prior pi^s_ij (same 7 keys): "uniform" | "causal" | "alibi" | "causal_alibi" | "windowed" | "causal_windowed" | "t5_relative_bias"

    alibi_slope                  = 1,

    t5_learnable_bias            = False,           # learn the per-bucket T5 bias table b_{i-j} (sanctioned NN exception, default OFF; needs a t5_relative_bias channel)

    precision_weighted_attention = False,        # down-weight high-variance keys: fold detached -log(b0 + tr Sigma_j)
                                                 # into the attention prior (diagnostic; OFF = position-only prior)
    precision_attention_b0       = 2.0,          # b0 in the per-key reliability -log(b0 + tr Sigma_j); > 0
    precision_attention_per_head = False,        # per-key reliability PER HEAD (trace over each block's coords) vs
                                                 # global (all K); needs precision_weighted_attention=True
    #################################
    #         Belief E-step
    #         Learning Rates
    #################################
    
    e_q_mu_lr                 = 0.3,
    e_q_sigma_lr              = 0.025,
    e_phi_lr                  = 0.00,     
    
    
    ####################################
    #       Model E-step LR's
    #      If s_e_step = True
    # and prior_source = 'model_channel'
    ####################################
    
    r_update_mode             = "gradient",          # "gradient" (AdamW M-step; correct under s_e_step) | "barycenter" (closed-form forward-KL centroid of s; exact M-step in the scored s_e_step=False regime)
    prior_source              = "token",    # belief prior p_i: "token" or "model_channel"
    learnable_r               = False,               # un-freeze hyper-prior centroid r (empirical-Bayes)
    s_e_step                  = False,
    
    e_s_mu_lr                 = 0.85,
    e_s_sigma_lr              = 0.1,
    
    #################################
    #    Embedding/Priors M-step 
    #        Learning Rates
    #################################
        
    m_p_mu_lr                 = 0.005,     
    m_p_sigma_lr              = 0.002,     
    m_phi_lr                  = 0.0025,    #0.0025 pure path
    
    m_s_phi_lr                = 0.007,         
    
    weight_decay              = 0.02,   
    phi_weight_decay          = 0.00,   
    sigma_weight_decay        = 0.0,           # AdamW decay for log-variance tables (None = inherit weight_decay;
                                             # 0.0 exempts sigma from the unintended log-sigma->0 pull)
    mu_weight_decay           = 0,            # AdamW decay for the MEAN-role tables: mu_embed, s_mu_embed,
                                             # decode_mu_embed, output_proj_weight (None = inherit weight_decay).
                                             # Decoupled decay reaches EVERY row of a live embedding table on
                                             # every step, so rare rows are crushed faster than their gather
                                             # gradient restores them (measured: zero-count rows at norm 0.000,
                                             # count 1-15 BELOW init). 0.0 exempts the mean sector; set None to
                                             # restore the pre-2026-07-26 inherit-the-global behavior.

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
    
    e_mu_q_trust              = 1,
    e_sigma_q_trust           = 1,
    sigma_max                 = 100,
    
    #################################
    #         Misc/Logging
    #################################     
    amp_dtype                 = None,      # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32
        
    log_interval              = 100,       # console log every N steps (0 = off)
    eval_interval             = 1500,      # periodic validation every N steps (0 = off)
    checkpoint_interval       = 15000,     # save a resumable checkpoint every N steps (0 = off)

    



    # --- E-step update rule ---
    e_step_update             = "gradient",  # "gradient" (pure current path) | "mm_exact" (closed-form MM
                                             # coordinate minimizer at frozen beta: precision fusion in ONE
                                             # iteration, same cost; kernel route only)
    mm_damping                = 1,         # mm_exact damping eta in (0,1]; 1.0 = exact minimizer

    # --- randomized-depth E-step (recurrent-depth recipe) ---
    randomize_e_steps         = False,       # training forwards sample T ~ Uniform{e_steps_min..e_steps_max}
    e_steps_min               = 1,
    e_steps_max               = 3,
    e_steps_backprop_last     = 0,           # truncated backprop: no_grad all but the last k iterations (0 = OFF)
    e_step_halt_tol           = None,        # eval halting: break when mean KL(q^t||q^{t-1}) < tol (None = OFF)

    # --- decode / objective ---
    decode_unigram_prior      = True,       # add kappa*log pi_v (corpus unigram, data statistic) to decode logits
    unigram_kappa             = 1,         # tempering on log pi_v (1.0 = exact Bayes class prior)
    
    # decode_mode "expected_likelihood_chunked" is also new: sigma-aware Gaussian-convolution readout
    # log N(mu_q; mu_v, Sigma_q + Sigma_v) - select it above under use_prior_bank=True.
    untie_decode_bank         = True,       # use_prior_bank=True only: decode reads its OWN cloned (V,K) tables
    z_loss_weight             = 0,           # z-loss on the decode partition: w * mean(logsumexp^2) (0 = OFF)
   

    # --- attention / coupling ---
    gamma_as_beta_prior       = False,        # fold DETACHED gamma posterior into beta's prior (h->s->p->q);
                                             # needs lambda_gamma > 0
    gamma_prior_weight        = 0.5,         # mixture weight w in [0,1]: pi = (1-w) softmax(B) + w gamma
    lambda_twohop             = 0.0,         # two-hop coupling F2 = lam2 sum_ik (beta@beta)_ik KL_ik (0 = OFF;
                                             # exact composed transport, effective depth 2 at L=1)
    query_adaptive_tau        = False,       # per-query tau_i = tau_h (1 + c tr_h Sigma_i / d_h), detached
    query_tau_c               = 1.0,         # strength c >= 0 (read only when query_adaptive_tau=True)
    

    #   "off"      -- the PURE path and the default: no data term, byte-identical to before.
    #   "shared"   -- reuse the decode table (output_proj_weight). Whitepaper-faithful, no new
    #                 parameters, but V3 decodes position t against x_{t+1} while the emission pulls
    #                 toward x_t, so one linear map carries both roles with no nonlinear head between.
    #   "separate" -- own (V, K) table. Removes that competition at the cost of decoupling the factor
    #                 from the decoder that actually scores the prediction.
    # Both live modes are fixed-basis linear maps and therefore NOT gauge-equivariant, the same
    # footprint the linear decode already carries; "off" keeps the gauge-pure path.
    emission_mode    = "off",       # "off" | "shared" | "separate"
    emission_weight  = 0.0,            # 0.0 reproduces the "off" path byte-identically


    # --- training mechanics ---
    grad_clip                 = 1.0,         # gradient clip: global L2 norm unless grad_clip_per_role; None/0.0 disables
    grad_clip_per_role        = True,        # clip grads per role (mu/sigma/phi) instead of one global norm
                                             # (global is phi-dominated and silently rescales other roles)
    skip_belief_sigma_update  = False,        # skip the belief-channel sigma E-step update (dead-compute ablation
                                             # for linear-decode configs; user asserts sigma has no consumer)

    # --- compute reclamation (exactness-preserving perf; default OFF) ---
    exp_fp64_mode             = "dim",       # "dim" (long-standing: fp64 when block dim >= 20) | "norm" (fp64 only
                                             # when clamped ||M||_F >= exp_fp64_norm_threshold; d_head=25 blocks
                                             # currently run fp64 PERMANENTLY under "dim")
    exp_fp64_norm_threshold   = 15.0,        # "norm" mode threshold
    share_refine_s_transport  = True,        # build the flat transport ONCE per forward, share s-refine + belief
                                             # E-step (+ all layers); valid on flat/e_phi_lr=0/no-rope configs
    compile_pair_kernel       = False,       # torch.compile the closed-form pair kernel (eager fallback + warn)

    full_cov_kl_precision     = "fp32_escalate",  # "fp64" | "fp32_escalate"  (family="gaussian_full" only)
                                             # "fp64" = the historical unconditional float64 island.
                                             # "fp32_escalate" runs the (B,N,N,K,K) pair KL in float32 and
                                             # recomputes in float64 ONLY when a Cholesky actually fails --
                                             # data-keyed, not a blanket downcast. Verified single-pass through
                                             # cond(Sigma)~1e6, escalates at ~1e9; post-softmax attention moves
                                             # <=2e-4 on adversarial synthetic spectra, 4e-6 on real trained ones.
                                             
    full_cov_congruence_precision = "fp32_escalate",      # "fp64" | "fp32_escalate"                                        

    safe_cholesky_jitter_mode     = "relative",      # "absolute" | "relative"
    mu_trust_cholesky_rounds = 3,

    congruence_cond_escalation           = False,         # True -> slow ...escalate on the conditioning proxy
    emit_expensive_diagnostics           = True,    
    generate_figures                     = True,      # OFF: strict opt-out for all finalization plots, plot-only
                                                      # probes/model replays, and per-eval attention/gamma heatmaps.
                                                      # True re-enables; make_figures.py later rebuilds the replayable
                                                      # model-report, saved-probe, and persisted-history set.
    

)

# kl_max is the numerical safety-net clamp on EVERY divergence (KL(q||p), KL(s||r), pairwise energy),
# next to eps -- NOT an operating ceiling. Diagonal KL is a sum over K coords (~0.8 nats/coord trained),
# so the K-INDEPENDENT 100.0 default binds for ~100% of tokens at large K (K* ~ 126), silently zeroing
# the hyper-prior self-gradient and gradient-freezing learnable r (the kernel self_mask, gradients/
# kernels.py:129). Scale it with K so it binds only on genuine NaN/inf/Cholesky blowups; F is provably
# kl_max-independent below the ceiling (safe_kl_clamp is the identity there). See docs/2026-06-21-edits.md.
config["kl_max"] = 8 * config["embed_dim"]


# Where each run's artifacts go: vfe3_runs/<timestamp>_<label>/ while training (config.json,
# metrics.csv, checkpoints/, best_model.pt, test_results.json, summary.json, *.png), renamed to
# vfe3_runs/<test_ppl>_<label>/ (timestamp dropped) at finalize. None disables persistence.
RUN_ROOT = "vfe3_runs"


def _banner(model, cfg: VFE3Config, dataset: str, device: str, n_steps: int,
            train_loader=None, full_corpus_tokens: 'int | None' = None) -> str:
    from vfe3.train import coverage_lines, parameter_report
    rep = parameter_report(model, device=device)
    bar = "=" * 64
    cov = (coverage_lines(train_loader, n_steps, dataset, full_corpus_tokens=full_corpus_tokens)
           if train_loader is not None else [])
    live_note = (f" ({rep['live']:,} live, {rep['dead']:,} dead)"
                 if rep["probed"] and rep["dead"] else "")
    dead_line = ([" dead under config (no grad): "
                  + ", ".join(n.replace("prior_bank.", "") for n in rep["dead_names"])]
                 if rep["probed"] and rep["dead_names"] else [])
    return "\n".join([
        bar,
        f" Gauge VFE Transformer | {rep['total']:,} params{live_note} | {device}",
        bar,
        f" K={cfg.embed_dim}  N={cfg.max_seq_len}  L={cfg.n_layers}  "
        f"heads={len(model.group.irrep_dims)}  "  # runtime attention heads = irrep blocks (cross_couplings -> 1)
        f"group={cfg.gauge_group}  family={cfg.family}",
        f" steps={n_steps}  batch={cfg.batch_size}  dataset={dataset}",
        *cov,
        *dead_line,
        f" M-LRs: mu={cfg.m_p_mu_lr}  sigma={cfg.m_p_sigma_lr}  "
        f"phi={cfg.m_phi_lr}  s_phi={cfg.m_s_phi_lr}",
        f" VFE: lambda_alpha={cfg.lambda_alpha}  kappa_beta={cfg.kappa_beta}  "
        f"tau={_fmt_tau(cfg, model)}  mass_phi={cfg.mass_phi}",
        f" seed={cfg.seed}",
        bar,
    ])


def _validated_data_seed() -> Optional[int]:
    if DATA_SEED is not None and (type(DATA_SEED) is not int or DATA_SEED < 0):
        raise ValueError(
            f"DATA_SEED must be None or an exact non-negative integer, got {DATA_SEED!r}")
    return DATA_SEED


def _select_loader(
    dataset: str,
    cfg:     VFE3Config,

    *,
    split:   str = "train",
) -> DataLoader:
    r"""Loader for ``dataset``/``split``. A missing cache raises ``FileNotFoundError``.

    ``MAX_TOKENS`` caps only the train split (smoke runs); the small validation/test splits are
    always read in full. The loader never substitutes synthetic data for a missing real corpus --
    that would silently compute held-out numbers on a toy stream and mislabel them as the corpus.

    Split-aware loader semantics: only TRAIN shuffles and drops the partial last batch; VALIDATION
    and TEST read the whole split in deterministic order (shuffle=False, drop_last=False) so the
    held-out metric is a stable corpus measurement, not a randomly-varying ~97% subset.

    ``cfg.eval_stride`` applies to VALIDATION and TEST only. None (the default) leaves the disjoint
    ``stride == max_seq_len`` windows untouched; a smaller value selects the sliding-window
    evaluation, whose overlapping prefixes are masked so every transition is still scored exactly
    once (see :class:`~vfe3.data.datasets.TokenWindows`). Train is never strided -- it shuffles and
    drops its tail, so the exactly-once contract does not apply to it.
    """
    is_train = (split == "train")
    cap = MAX_TOKENS if is_train else None
    # Fix the TRAIN shuffle order across seeds (EXP-1) only when DATA_SEED is set; None -> no generator
    # -> legacy global-RNG shuffle (byte-identical). Val/test do not shuffle, so a generator is moot there.
    data_seed = _validated_data_seed()
    gen = torch.Generator().manual_seed(data_seed) if (is_train and data_seed is not None) else None
    return make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size,
                           shuffle=is_train, drop_last=is_train, max_tokens=cap,
                           stride=(None if is_train else cfg.eval_stride),
                           vocab_size=cfg.vocab_size, generator=gen)


def _run_label(cfg: VFE3Config, dataset: str) -> str:
    r"""Descriptive run label ``<dataset>_K<embed_dim>_<group>[_linear][_mix][_cross]_s<seed>`` (no
    timestamp, no PPL).

    The stable part of the run-folder name: ``_run_dir`` prefixes it with a timestamp while the run is
    in progress, and ``_rename_run_by_ppl`` swaps that prefix for the test perplexity at finalize. The
    ``_s<seed>`` suffix keeps a multi-seed launch's run folders distinct (and identifiable by seed).
    """
    _validate_path_component(dataset, "dataset")
    tags = (("_linear" if not cfg.use_prior_bank else "")
            + ("_mix" if cfg.use_head_mixer else "")
            + ("_cross" if cfg.cross_couplings else ""))
    return f"{dataset}_K{cfg.embed_dim}_{cfg.gauge_group}{tags}_s{cfg.seed}"


def _reserve_directory(root: Path, base: str) -> Path:
    """Atomically reserve ``root/base`` and retry deterministic numeric suffixes on collision."""
    root.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        candidate = root / (base if suffix == 1 else f"{base}_{suffix}")
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        return candidate


def _run_dir(
    cfg:     VFE3Config,
    dataset: str,

    *,
    run_root: 'str | None' = None,
) -> 'str | None':
    r"""Reserve ``<run_root>/<timestamp>_<label>/`` (None when persistence is disabled).

    The timestamp keeps concurrent runs from colliding while training; ``_rename_run_by_ppl`` drops it in
    favor of the held-out test perplexity once ``finalize_run`` has scored the test split.
    """
    selected_root = RUN_ROOT if run_root is None else run_root
    if selected_root is None:
        return None
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{_run_label(cfg, dataset)}"
    return str(_reserve_directory(Path(selected_root), base))


def _rename_run_by_ppl(
    run_dir:  str,                       # in-progress timestamped run directory
    label:    str,                       # descriptive part to keep (see _run_label)
    test_ppl: 'float | None',            # held-out test perplexity (None / non-finite -> no rename)

    logger:   logging.Logger,
) -> str:
    r"""Rename ``run_dir`` to ``vfe3_runs/<test_ppl:.2f>_<label>/`` so runs sort by test perplexity.

    The folder is created with a timestamp prefix (the PPL is unknown until ``finalize_run`` scores the
    test split); this swaps that prefix for the formatted test PPL and drops the timestamp. Returns the
    new path -- or the original unchanged when the PPL is missing/non-finite (the timestamped name is
    then the only stable handle) or when the OS refuses the move (an open handle / locked directory --
    the numeric results are already on disk, so a failed rename is logged, never fatal). A name clash
    gets a ``_2``, ``_3``, ... suffix so an existing run is never clobbered.
    """
    import math
    from pathlib import Path

    src = Path(run_dir)
    if test_ppl is None or not math.isfinite(test_ppl) or not src.exists():
        return run_dir
    dst = src.parent / f"{test_ppl:.2f}_{label}"
    i = 2
    while dst.exists():
        dst = src.parent / f"{test_ppl:.2f}_{label}_{i}"
        i += 1
    try:
        src.rename(dst)
    except OSError as exc:                                # open handle / locked dir -> keep run, log it
        logger.warning("could not rename run dir to %s (%s); kept %s", dst.name, exc, src.name)
        return run_dir
    logger.info("Renamed run dir -> %s", dst.name)
    return str(dst)


def _run_once(
    seed:   int,
    logger: logging.Logger,

    *,
    run_root: 'str | None' = None,
) -> None:
    r"""One full, independent training run at ``seed`` (build -> train -> val -> test/finalize).

    Builds a fresh ``VFE3Config`` from ``config`` with ``seed`` overridden, seeds the RNG, and runs the
    complete pipeline into its own seed-labeled artifacts dir. Called once per resolved seed by
    :func:`main`, so a multi-seed launch yields one independent, comparable run folder per seed.
    """
    import time

    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import RunArtifacts, finalize_run

    cfg = VFE3Config(**{**config, "seed": seed})         # per-run seed override (config `seed` is the default)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    model = VFEModel(cfg).to(DEVICE)
    train_loader = _select_loader(DATASET, cfg, split="train")
    val_loader = _select_loader(DATASET, cfg, split="validation")

    # Bits-per-CHARACTER correction so PPL/BPC compare across tokenizers and languages (gpt2 vs
    # cl100k; en/ja/ar -- a cl100k token spans ~3 Japanese codepoints). tokens_per_char =
    # n_tokens/n_codepoints from the held-out stream; None (synthetic / no tiktoken / cache absent)
    # -> BPC unavailable, while evaluate publishes the always-defined bits_per_token separately.
    from vfe3.data.datasets import tokens_per_char as _tokens_per_char
    val_tpc = _tokens_per_char(DATASET, "validation")

    # Run-artifacts directory (config.json, metrics.csv, checkpoints/, best_model.pt, figures).
    # None disables persistence (RUN_ROOT = None); the synthetic fallback also runs unsaved-free.
    run_dir = _run_dir(cfg, DATASET, run_root=run_root)
    artifacts = None
    if run_dir is not None:
        from datetime import datetime
        artifacts = RunArtifacts(run_dir, cfg, model, dataset=DATASET, device=DEVICE,
                                 timestamp=datetime.now().isoformat(timespec="seconds"))
        logger.info("Saving run artifacts to %s", run_dir)

    # Full uncapped corpus size for the "stream is X% of full" banner line -- only computed when
    # MAX_TOKENS actually caps the train stream (the default None loads the whole corpus, so no cap line).
    full_corpus_tokens = None
    if MAX_TOKENS is not None:
        try:
            full_corpus_tokens = cached_token_count(DATASET, "train")
        except FileNotFoundError:
            full_corpus_tokens = None
    logger.info(_banner(model, cfg, DATASET, DEVICE, cfg.max_steps,
                        train_loader=train_loader, full_corpus_tokens=full_corpus_tokens))
    # Reseed AFTER model construction so the train data-shuffle order does NOT depend on the
    # config-dependent amount of global RNG VFEModel(cfg) consumes at init. make_dataloader builds
    # the train loader with no explicit generator, so its RandomSampler draws each epoch permutation
    # from the GLOBAL RNG at the first iter(loader) inside train(); leaving the RNG model-advanced
    # here would make this entry point train on a DIFFERENT batch order than ablation.py for an
    # identical config+seed (model init itself is already identical). Mirrors ablation.run_single's
    # post-build reseed so the two entry points reproduce each other.
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    t0 = time.perf_counter()
    losses = train(
        model, train_loader, cfg,
        n_steps=cfg.max_steps,
        grad_clip=cfg.grad_clip,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=val_loader,
        tokens_per_char=val_tpc,
        device=torch.device(DEVICE),
        logger=logger,
        artifacts=artifacts,
        sample_dataset=DATASET,          # audit 2026-07-27: key the sample tokenizer, do not guess
    )
    wall = time.perf_counter() - t0

    m = evaluate(model, val_loader, tokens_per_char=val_tpc, device=torch.device(DEVICE))
    logger.info("=" * 64)
    logger.info(                                          # val-only summary; CE is the loss (no separate train loss here)
        "Final (val) | CE: %.4f | PPL: %.1f | BPT: %.4f | BPC: %s",
        m["ce"], m["ppl"], m["bits_per_token"],
        (f"{float(m['bpc']):.4f}" if m["bpc"] is not None else "unavailable"),
    )

    # End-of-run held-out TEST evaluation on the reloaded best-val checkpoint, plus summary +
    # figures, on the dataset's test split (a missing cache raises -- no synthetic substitution).
    if artifacts is not None:
        test_loader = _select_loader(DATASET, cfg, split="test")
        test_tpc = _tokens_per_char(DATASET, "test")
        results = finalize_run(
            model,
            artifacts,
            cfg,
            tokens_per_char=test_tpc,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            losses=losses,
            data_seed=DATA_SEED,
            max_tokens=MAX_TOKENS,
            tokenizer_tag=_tokenizer_tag(DATASET),
            device=torch.device(DEVICE),
            wall_time=wall,
            logger=logger,
        )
        run_dir = _rename_run_by_ppl(run_dir, _run_label(cfg, DATASET), results.get("test_ppl"), logger)
        logger.info("Artifacts written to %s", run_dir)


def _resolve_seeds(
    run_config: Dict[str, object],
    seeds:      Sequence[int],
    num_runs:   int,
) -> List[int]:
    """Resolve click-to-run seeds without silently overriding a one-run config seed."""
    if isinstance(num_runs, bool) or not isinstance(num_runs, int) or num_runs <= 0:
        raise ValueError(f"NUM_RUNS must be a positive integer, got {num_runs!r}")
    config_seed = run_config.get("seed")
    if type(config_seed) is not int or config_seed < 0:
        raise ValueError(f"config seed must be an exact non-negative integer, got {config_seed!r}")
    if seeds:
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("SEEDS must contain exact non-negative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"SEEDS must be unique, got {list(seeds)!r}")
        if len(seeds) < num_runs:
            raise ValueError(
                f"SEEDS lists {len(seeds)} seed(s) but NUM_RUNS={num_runs}; "
                "provide at least NUM_RUNS seeds"
            )
        if num_runs == 1 and seeds[0] != config_seed:
            raise ValueError(
                f"SEEDS[0]={seeds[0]} conflicts with config seed={config_seed}; "
                "make the one-run seed values agree"
            )
        return list(seeds[:num_runs])
    if num_runs != 1:
        raise ValueError(f"NUM_RUNS={num_runs} > 1 but SEEDS is empty; list one seed per run in SEEDS")
    return [config_seed]


def main() -> None:
    r"""Click-to-run entry: train ``NUM_RUNS`` independent seeds back-to-back (default: one run on the
    config ``seed``). Run i uses ``SEEDS[i]``; each run is fully independent with its own seed-labeled
    artifacts directory. A multi-seed launch owns one atomically reserved group containing its request
    manifest and one comparable run folder per seed.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("train_vfe3")
    _validated_data_seed()                                  # before reserving any run/manifest directory
    seeds = _resolve_seeds(config, seeds=SEEDS, num_runs=NUM_RUNS)
    run_root = None
    request_manifest = None
    request_path = None
    if RUN_ROOT is not None and len(seeds) > 1:
        from datetime import datetime
        from vfe3.run_artifacts import _write_json_atomic
        group = _reserve_directory(
            Path(RUN_ROOT),
            f"multiseed_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        run_root = str(group)
        request_path = group / "multiseed_request.json"
        request_manifest = {
            "schema_version": 1,
            "status": "pending",
            "seeds": seeds,
            "cells": [{"seed": seed, "status": "pending"} for seed in seeds],
        }
        _write_json_atomic(request_path, request_manifest)
        logger.info("Multi-seed run group: %s", group)
    if request_manifest is not None and request_path is not None:
        request_manifest["status"] = "running"
        _write_json_atomic(request_path, request_manifest)
    for i, s in enumerate(seeds):
        if len(seeds) > 1:
            logger.info("\n%s\n# Run %d/%d  (seed=%d)\n%s", "#" * 64, i + 1, len(seeds), int(s), "#" * 64)
        try:
            _run_once(int(s), logger, run_root=run_root)
        except Exception as exc:
            if request_manifest is not None and request_path is not None:
                request_manifest["status"] = "failed"
                request_manifest["cells"][i].update({
                    "status": "failed",
                    "error": str(exc),
                })
                _write_json_atomic(request_path, request_manifest)
            raise
        if request_manifest is not None and request_path is not None:
            request_manifest["cells"][i]["status"] = "complete"
            request_manifest["status"] = "complete" if i == len(seeds) - 1 else "running"
            _write_json_atomic(request_path, request_manifest)


if __name__ == "__main__":
    main()
