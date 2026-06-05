r"""Click-to-run training entry for the VFE_3.0 transformer.

Mirrors VFE_2.0 ``train_vfe.py``: edit the ``config`` dict below, pick a ``DATASET``,
then run ``python train_vfe3.py``. There is no CLI arg parsing.

The ``config`` dict exposes EVERY ``VFE3Config`` toggle, grouped exactly as in
``vfe3/config.py``; each registry-backed ``*_mode`` / ``*_family`` / ``*_group`` field
lists its valid keys inline. The default ``DATASET = "wikitext-103"`` trains on the
cached gpt2/tiktoken corpus (vocab 50257) under ``~/.cache/tokenized_cache``; the
``config`` defaults (``vocab_size=50257``) are kept consistent with it so click-to-run
works out of the box. ``MAX_TOKENS`` caps the training stream for fast smoke runs.

If a real corpus' cache is absent the loader falls back to a deterministic period-3
token stream (the cutover anchor used by ``tests/test_train.py``), so this file never
crashes for lack of data. Selecting ``DATASET = "synthetic-period3"`` forces that stream.

A full ``max_steps`` run on the 116.8M-token wikitext-103 train split is a real (not
smoke) job: run it on the CUDA interpreter (the RTX 5090), or drop ``MAX_TOKENS`` /
``max_steps`` for a quick slice on CPU.
"""

import logging

import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows, make_dataloader
from vfe3.free_energy import attention_tau
from vfe3.train import evaluate, train


# --- click-to-run knobs -------------------------------------------------------
SEED = 6

# Cached tokenized corpus (gpt2/tiktoken -> vocab_size 50257) or the zero-dependency
# synthetic anchor. Caches live in ~/.cache/tokenized_cache.
#   "wikitext-103" | "wikitext-2" | "wiki-en" | "wiki-ja" | wiki-ar | "synthetic-period3"
DATASET = "wikitext-103"

# Cap the *training* stream for fast smoke runs (the validation split is always read
# in full -- it is small). None = the full corpus (116.8M tokens for wikitext-103).
MAX_TOKENS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- full config: every VFE3Config toggle, grouped as in vfe3/config.py --------
# Edit any field, then run. Values are the wikitext-103 (real-corpus) defaults.
# Registry fields list their valid keys inline; cross-field constraints are noted.
config = dict(
    

    # model structure
    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*)
    
    embed_dim                 = 20,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 2,
    
    max_seq_len               = 128,                 # N, context length
    
    batch_size                = 64,
    max_steps                 = 15000,
    
    n_layers                  = 1,                   # L, number of blocks
    n_e_steps                 = 1,                   # T, E-step inner iterations
    
    

    # gauge seam
    gauge_group               = "block_glk",    # "glk" | "block_glk" | "tied_block_glk" | "so_k"
                                                     # tied_block_glk: one shared GL(d) frame across heads (kron(I_n, gl(d)))
    gauge_parameterization    = "phi",               # "phi" | "omega_direct" (omega_direct: live-rejected, no belief source)
    use_head_mixer            = True,               # opt-in Schur-commutant head mixer (needs >=2 equal blocks: block_glk/tied_block_glk);
                                                     # breaks strict equivariance under block_glk (exact at init); EXACT under tied_block_glk (full-cov)

    # connection regime (orthogonal to gauge_parameterization)
    transport_mode            = "flat",              # "flat" (Regime-I phi-cocycle, pure no-NN) | "regime_ii"
                                                     # (learned bilinear edge connection; sanctioned NN exception, default-off)
    cocycle_relaxation        = 1.0,                 # regime_ii homotopy: 0.0 -> flat, 1.0 -> fully relaxed (ignored by flat)
    cross_couplings           = None,                # off-block GL(K) head pairs e.g. [(0, 1)]; block_glk only (None = block-diagonal gauge)

    # positional encoding -- BCH gauge-frame PE (pos_phi) + gauge-RoPE (pos_rotation)
    pos_phi                   = "learned",           # "none" (pure path) | "learned" | "frozen"
    pos_phi_compose           = "bch",               # composition chart: "bch" | "euclidean"
    
       
    pos_phi_scale             = 0.02,                # learned-table init scale AND frozen per-position step
    
    pos_phi_project_slk       = False,               # per-block trace projection (det Omega = 1)
    
    pos_rotation              = "none",              # "none" | "rope" (block-diagonal positional rotation folded into transport)
    rope_base                 = 100.0,               # rotary frequency base
    rope_full_gauge           = False,               # rotate the covariance sandwich too (REQUIRES diagonal_covariance=False)

    # belief family -- diagonal_covariance MUST equal (family == "gaussian_diagonal")
    diagonal_covariance       = True,
    family                    = "gaussian_diagonal", # "gaussian_diagonal" | "gaussian_full"

    # free-energy coupling
    alpha                     = 1.0,                 # constant self-coupling value
    alpha_mode                = "state_dependent_per_coord",          # "constant" | "state_dependent" | "state_dependent_per_coord"
    b0                        = 1.0,                 # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0                        = 1.0,                 # state-dependent alpha shape (numerator)
    
    kappa                     = 1.0,                 # tau = kappa * sqrt(d_head); kappa=1 -> Vaswani temperature

    lambda_beta               = 1.0,                 # belief-coupling block weight (1.0 = pure F; VFE_2.0 lambda_align)
    learnable_lambda_beta     = False,               # learn lambda_beta (NN exception; exp(log_lambda_beta), trained on CE)

    mass_phi                  = 0.0,                 # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight = 0.0,                # alpha_hat * sum_i KL(q_i*||p_i) M-step term (0 = OFF)
    
    
    lambda_h                  = 0.0,                 # hyper-prior weight lambda_h * mean_i KL(s_i||r) (0 = OFF; >0 creates s/r tables)
    gamma_coupling            = 0.0,                 # model-channel coupling (0 = OFF; >0 creates s tables, predictively inert by default)
    kappa_gamma               = 1.0,                 # model-channel temperature tau_gamma = kappa_gamma*sqrt(d_head)
    gamma_attention_prior     = "causal",            # model-channel prior pi^s_ij: "uniform" | "causal" | "alibi"
    prior_source              = "token",             # which table supplies the belief prior p_i: "token" | "model_channel"

    # attention
    include_attention_entropy = True,                # canonical F (True) vs entropy-suppressed surrogate (False)
    attention_prior           = "causal",            # "uniform" | "causal" | "alibi"

    # E-step
    e_mu_lr                   = 0.7,
    e_sigma_lr                = 0.025,
    e_phi_lr                  = 0.0,
    
    e_sigma_q_trust           = 5.0,
    sigma_max                 = 5.0,
    
    gradient_mode             = "filtering",          # "filtering" | "smoothing"
    
    phi_precond_mode          = "killing",  # "none" | "clip" | "killing" | "killing_per_block" | "pullback"
    phi_retract_mode          = "bch",                # "euclidean" | "bch"
    spd_retract_mode          = "spd_affine",         # SPD covariance retraction (registry: "spd_affine" | "log_euclidean")

    # decode / encode
    use_prior_bank            = False,                # True: KL-to-prior decode (pure path). False: linear projection
                                                     # mu->logits ablation (VFE_2.0 parity; encode stays on the prior bank)
    decode_tau                = 1.0,
    decode_mode               = "diagonal",          # "diagonal" | "diagonal_chunked" | "full"
    decode_chunk_size         = 8192,                # vocab-chunk width for decode_mode="diagonal_chunked" (ignored otherwise)
    encode_mode               = "per_token",         # "per_token" | "gauge_fixed" (gauge_fixed: live-rejected stub)

    # cross-block belief handoff (mu_q -> mu_p)
    prior_handoff_rho         = 0.0,                 # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma       = 0.0,                 # sigma damping in [0,1] (0.0 = frozen at embedding)

    # normalization
    norm_type_block           = "none",              # "none" | "mahalanobis"
    norm_type_final           = "none",              # "none" | "mahalanobis"

    # M-step / training
    e_step_gradient           = "unroll",            # E-step backward: "unroll" | "straight_through" | "detach" (reconciled w/ detach_e_step)
    detach_e_step             = False,               # False = unroll the E-step in the training graph (True forces effective "detach")
    grad_accum_steps          = 1,                   # microbatches accumulated before an optimizer step (1 = single-step)

    m_mu_lr                   = 0.015,
    m_sigma_lr                = 0.0021,
    m_phi_lr                  = 0.015,
    
    weight_decay              = 0.065,
    
    # numerics
    eps                       = 1e-6,
    kl_max                    = 100.0,

    # divergence seam -- the f-divergence FUNCTIONAL (distinct from `family` below)
    divergence_family         = "renyi",             # "renyi"
    alpha_div                 = 1.0,                  # Renyi order (1.0 -> KL)
    
    warmup_steps              = 100,
    seed                      = SEED,
    
    log_interval              = 200,                  # console log every N steps (0 = off)
    eval_interval             = 1500,                   # periodic validation every N steps (0 = off)
    checkpoint_interval       = 15000,                  # save a resumable checkpoint every N steps (0 = off)
    
    
    eval_max_batches          = None,                 # cap the PERIODIC eval pass (None = full split)
    amp_dtype                 = None,                 # None (pure fp32) | "bf16" | "fp16" (opt-in autocast; CUDA throughput)
)


# Where each run's artifacts go: vfe3_runs/<timestamp>_<label>/ (config.json, metrics.csv,
# checkpoints/, best_model.pt, test_results.json, summary.json, *.png). None disables persistence.
RUN_ROOT = "vfe3_runs"


def synthetic_period3_loader(period=3, n=600, seq_len=8, batch_size=8, seed=0) -> DataLoader:
    r"""Deterministic period-3 token stream (mirrors tests/test_train.py).

    ``n`` is grown to ``seq_len * batch_size * 4`` when needed so the stream yields at
    least a few ``batch_size`` windows under ``drop_last`` -- otherwise large click-run
    dims (``seq_len=128, batch_size=64``) would produce zero batches.
    """
    g = torch.Generator().manual_seed(seed)
    n = max(n, seq_len * batch_size * 4)
    base = torch.arange(period).repeat(n // period + 2)
    ds = TokenWindows(base[:n].to(torch.long), seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)


def _banner(model, cfg: VFE3Config, dataset: str, device: str, n_steps: int) -> str:
    n_params = sum(p.numel() for p in model.parameters())
    bar = "=" * 64
    return "\n".join([
        bar,
        f" Gauge VFE Transformer | {n_params} params | {device}",
        bar,
        f" K={cfg.embed_dim}  N={cfg.max_seq_len}  L={cfg.n_layers}  heads={cfg.n_heads}  "
        f"group={cfg.gauge_group}  family={cfg.family}",
        f" steps={n_steps}  batch={cfg.batch_size}  dataset={dataset}",
        f" M-LRs: mu={cfg.m_mu_lr}  sigma={cfg.m_sigma_lr}  phi={cfg.m_phi_lr}",
        f" VFE: alpha={cfg.alpha}  kappa={cfg.kappa}  "
        f"tau={attention_tau(cfg.kappa, model.group.irrep_dims):.4f}  mass_phi={cfg.mass_phi}",
        f" seed={cfg.seed}",
        bar,
    ])


def _select_loader(
    dataset: str,
    cfg:     VFE3Config,
    logger:  logging.Logger,

    *,
    split:   str = "train",
) -> DataLoader:
    r"""Loader for ``dataset``/``split``; falls back to the synthetic stream if absent.

    ``MAX_TOKENS`` caps only the train split (smoke runs); the small validation split is
    always read in full. The synthetic anchor ignores ``split`` (its train == val).

    Split-aware loader semantics: only TRAIN shuffles and drops the partial last batch; VALIDATION
    and TEST read the whole split in deterministic order (shuffle=False, drop_last=False) so the
    held-out metric is a stable corpus measurement, not a randomly-varying ~97% subset.
    """
    is_train = (split == "train")
    if dataset == "synthetic-period3":
        return synthetic_period3_loader(seq_len=cfg.max_seq_len, batch_size=cfg.batch_size, seed=cfg.seed)
    cap = MAX_TOKENS if is_train else None
    try:
        return make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size,
                               shuffle=is_train, drop_last=is_train, max_tokens=cap)
    except FileNotFoundError:
        logger.warning("cache for %r/%r absent; falling back to synthetic-period3", dataset, split)
        return synthetic_period3_loader(seq_len=cfg.max_seq_len, batch_size=cfg.batch_size, seed=cfg.seed)


def _run_dir(cfg: VFE3Config, dataset: str) -> 'str | None':
    r"""``vfe3_runs/<timestamp>_<dataset>_K<embed_dim>_<group>[_linear][_mix]/`` (None if RUN_ROOT is None)."""
    if RUN_ROOT is None:
        return None
    from datetime import datetime
    from pathlib import Path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tags = "" + ("_linear" if not cfg.use_prior_bank else "") + ("_mix" if cfg.use_head_mixer else "")
    return str(Path(RUN_ROOT) / f"{stamp}_{dataset}_K{cfg.embed_dim}_{cfg.gauge_group}{tags}")


def main() -> None:
    import time

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("train_vfe3")

    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import RunArtifacts, finalize_run

    cfg = VFE3Config(**config)
    torch.manual_seed(cfg.seed)
    model = VFEModel(cfg).to(DEVICE)
    train_loader = _select_loader(DATASET, cfg, logger, split="train")
    val_loader = _select_loader(DATASET, cfg, logger, split="validation")

    # Run-artifacts directory (config.json, metrics.csv, checkpoints/, best_model.pt, figures).
    # None disables persistence (RUN_ROOT = None); the synthetic fallback also runs unsaved-free.
    run_dir = _run_dir(cfg, DATASET)
    artifacts = None
    if run_dir is not None:
        from datetime import datetime
        artifacts = RunArtifacts(run_dir, cfg, model, dataset=DATASET, device=DEVICE,
                                 timestamp=datetime.now().isoformat(timespec="seconds"))
        logger.info("Saving run artifacts to %s", run_dir)

    logger.info(_banner(model, cfg, DATASET, DEVICE, cfg.max_steps))
    t0 = time.perf_counter()
    losses = train(
        model, train_loader, cfg,
        n_steps=cfg.max_steps,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=val_loader,
        device=torch.device(DEVICE),
        logger=logger,
        artifacts=artifacts,
    )
    wall = time.perf_counter() - t0

    m = evaluate(model, val_loader, device=torch.device(DEVICE))
    logger.info("=" * 64)
    logger.info(                                          # val-only summary; CE is the loss (no separate train loss here)
        "Final (val) | CE: %.4f | PPL: %.1f | BPC: %.4f",
        m["ce"], m["ppl"], m["bpc"],
    )

    # End-of-run held-out TEST evaluation on the reloaded best-val checkpoint, plus summary +
    # figures. Uses the dataset's test split when its cache exists (wikitext-* do); falls back to
    # the validation loader for the synthetic anchor (no separate test stream).
    if artifacts is not None:
        test_loader = (val_loader if DATASET == "synthetic-period3"
                       else _select_loader(DATASET, cfg, logger, split="test"))
        finalize_run(model, artifacts, cfg, test_loader=test_loader, losses=losses,
                     device=torch.device(DEVICE), wall_time=wall, logger=logger)
        logger.info("Artifacts written to %s", run_dir)


if __name__ == "__main__":
    main()
