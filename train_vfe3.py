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
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Anaconda + PyTorch each ship a
#   libiomp5md.dll; the duplicate OpenMP init aborts the process (seen with n_e_steps>1). This MUST
#   run before `import torch`. The clean fix is one OpenMP in the env (e.g. `conda install nomkl`);
#   override by exporting KMP_DUPLICATE_LIB_OK yourself. See docs/edits/2026-06-05.

import logging

import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import make_dataloader
from vfe3.train import _fmt_tau, evaluate, train





# Cached tokenized corpus (gpt2/tiktoken -> vocab_size 50257). Caches live in
# ~/.cache/tokenized_cache; a missing cache raises (no synthetic substitution).
#   "wikitext-103" | "wikitext-2" | "wiki-en" | "wiki-ja" | "wiki-ar"
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
SEEDS    = [6]        # e.g. [3, 64, 23]; must list at least NUM_RUNS seeds when NUM_RUNS > 1


config = dict(
    

    #################################
    #            Training
    #################################
    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*)
    
    embed_dim                 = 80,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 8,
    
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
    
    phi_precond_mode          = "killing_per_block",  # "none" | "clip" | "killing" | "killing_per_block" | "pullback" | "pullback_per_block"
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
    m_p_sigma_lr              = 0.0035,     
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


# Where each run's artifacts go: vfe3_runs/<timestamp>_<label>/ while training (config.json,
# metrics.csv, checkpoints/, best_model.pt, test_results.json, summary.json, *.png), renamed to
# vfe3_runs/<test_ppl>_<label>/ (timestamp dropped) at finalize. None disables persistence.
RUN_ROOT = "vfe3_runs"


def _banner(model, cfg: VFE3Config, dataset: str, device: str, n_steps: int,
            train_loader=None, full_corpus_tokens: 'int | None' = None) -> str:
    from vfe3.train import coverage_lines
    n_params = sum(p.numel() for p in model.parameters())
    bar = "=" * 64
    cov = (coverage_lines(train_loader, n_steps, dataset, full_corpus_tokens=full_corpus_tokens)
           if train_loader is not None else [])
    return "\n".join([
        bar,
        f" Gauge VFE Transformer | {n_params} params | {device}",
        bar,
        f" K={cfg.embed_dim}  N={cfg.max_seq_len}  L={cfg.n_layers}  "
        f"heads={len(model.group.irrep_dims)}  "  # runtime attention heads = irrep blocks (cross_couplings -> 1)
        f"group={cfg.gauge_group}  family={cfg.family}",
        f" steps={n_steps}  batch={cfg.batch_size}  dataset={dataset}",
        *cov,
        f" M-LRs: mu={cfg.m_p_mu_lr}  sigma={cfg.m_p_sigma_lr}  phi={cfg.m_phi_lr}",
        f" VFE: lambda_alpha={cfg.lambda_alpha}  kappa_beta={cfg.kappa_beta}  "
        f"tau={_fmt_tau(cfg, model)}  mass_phi={cfg.mass_phi}",
        f" seed={cfg.seed}",
        bar,
    ])


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
    """
    is_train = (split == "train")
    cap = MAX_TOKENS if is_train else None
    return make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size,
                           shuffle=is_train, drop_last=is_train, max_tokens=cap)


def _run_label(cfg: VFE3Config, dataset: str) -> str:
    r"""Descriptive run label ``<dataset>_K<embed_dim>_<group>[_linear][_mix][_cross]_s<seed>`` (no
    timestamp, no PPL).

    The stable part of the run-folder name: ``_run_dir`` prefixes it with a timestamp while the run is
    in progress, and ``_rename_run_by_ppl`` swaps that prefix for the test perplexity at finalize. The
    ``_s<seed>`` suffix keeps a multi-seed launch's run folders distinct (and identifiable by seed).
    """
    tags = (("_linear" if not cfg.use_prior_bank else "")
            + ("_mix" if cfg.use_head_mixer else "")
            + ("_cross" if cfg.cross_couplings else ""))
    return f"{dataset}_K{cfg.embed_dim}_{cfg.gauge_group}{tags}_s{cfg.seed}"


def _run_dir(cfg: VFE3Config, dataset: str) -> 'str | None':
    r"""In-progress run dir ``vfe3_runs/<timestamp>_<label>/`` (None if RUN_ROOT is None).

    The timestamp keeps concurrent runs from colliding while training; ``_rename_run_by_ppl`` drops it in
    favour of the held-out test perplexity once ``finalize_run`` has scored the test split.
    """
    if RUN_ROOT is None:
        return None
    from datetime import datetime
    from pathlib import Path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(Path(RUN_ROOT) / f"{stamp}_{_run_label(cfg, dataset)}")


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


def _run_once(seed: int, logger: logging.Logger) -> None:
    r"""One full, independent training run at ``seed`` (build -> train -> val -> test/finalize).

    Builds a fresh ``VFE3Config`` from ``config`` with ``seed`` overridden, seeds the RNG, and runs the
    complete pipeline into its own seed-labelled artifacts dir. Called once per resolved seed by
    :func:`main`, so a multi-seed launch yields one independent, comparable run folder per seed.
    """
    import time

    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import RunArtifacts, finalize_run

    cfg = VFE3Config(**{**config, "seed": seed})         # per-run seed override (config `seed` is the default)
    torch.manual_seed(cfg.seed)
    model = VFEModel(cfg).to(DEVICE)
    train_loader = _select_loader(DATASET, cfg, split="train")
    val_loader = _select_loader(DATASET, cfg, split="validation")

    # Bits-per-CHARACTER correction so PPL/BPC compare across tokenizers and languages (gpt2 vs
    # cl100k; en/ja/ar -- a cl100k token spans ~3 Japanese codepoints). tokens_per_char =
    # n_tokens/n_codepoints from the held-out stream; None (synthetic / no tiktoken / cache absent)
    # -> 1.0 = honest bits-per-token. One cheap decode pass over the small val/test stream.
    from vfe3.data.datasets import tokens_per_char as _tokens_per_char
    val_tpc = _tokens_per_char(DATASET, "validation") or 1.0

    # Run-artifacts directory (config.json, metrics.csv, checkpoints/, best_model.pt, figures).
    # None disables persistence (RUN_ROOT = None); the synthetic fallback also runs unsaved-free.
    run_dir = _run_dir(cfg, DATASET)
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
        from vfe3.data.datasets import load_cached_tokens
        try:
            full_corpus_tokens = int(load_cached_tokens(DATASET, "train").numel())
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
    torch.manual_seed(cfg.seed)
    t0 = time.perf_counter()
    losses = train(
        model, train_loader, cfg,
        n_steps=cfg.max_steps,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=val_loader,
        tokens_per_char=val_tpc,
        device=torch.device(DEVICE),
        logger=logger,
        artifacts=artifacts,
    )
    wall = time.perf_counter() - t0

    m = evaluate(model, val_loader, tokens_per_char=val_tpc, device=torch.device(DEVICE))
    logger.info("=" * 64)
    logger.info(                                          # val-only summary; CE is the loss (no separate train loss here)
        "Final (val) | CE: %.4f | PPL: %.1f | BPC: %.4f",
        m["ce"], m["ppl"], m["bpc"],
    )

    # End-of-run held-out TEST evaluation on the reloaded best-val checkpoint, plus summary +
    # figures, on the dataset's test split (a missing cache raises -- no synthetic substitution).
    if artifacts is not None:
        test_loader = _select_loader(DATASET, cfg, split="test")
        test_tpc = _tokens_per_char(DATASET, "test") or 1.0
        results = finalize_run(model, artifacts, cfg, test_loader=test_loader, losses=losses,
                               tokens_per_char=test_tpc, device=torch.device(DEVICE), wall_time=wall, logger=logger)
        run_dir = _rename_run_by_ppl(run_dir, _run_label(cfg, DATASET), results.get("test_ppl"), logger)
        logger.info("Artifacts written to %s", run_dir)


def main() -> None:
    r"""Click-to-run entry: train ``NUM_RUNS`` independent seeds back-to-back (default: one run on the
    config ``seed``). Run i uses ``SEEDS[i]``; each run is fully independent with its own seed-labelled
    artifacts directory, so a multi-seed launch produces one comparable run folder per seed.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("train_vfe3")
    if SEEDS:
        if len(SEEDS) < NUM_RUNS:
            raise ValueError(
                f"SEEDS lists {len(SEEDS)} seed(s) but NUM_RUNS={NUM_RUNS}; provide at least NUM_RUNS seeds")
        seeds = list(SEEDS[:NUM_RUNS])
    elif NUM_RUNS != 1:
        raise ValueError(f"NUM_RUNS={NUM_RUNS} > 1 but SEEDS is empty; list one seed per run in SEEDS")
    else:
        seeds = [config["seed"]]                          # single run on the config seed (unchanged path)
    for i, s in enumerate(seeds):
        if len(seeds) > 1:
            logger.info("\n%s\n# Run %d/%d  (seed=%d)\n%s", "#" * 64, i + 1, len(seeds), int(s), "#" * 64)
        _run_once(int(s), logger)


if __name__ == "__main__":
    main()
