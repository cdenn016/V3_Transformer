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

The baseline operating point is IMPORTED from ``train_vfe3.config`` (this script never edits
``train_vfe3.py``), so a scaling run scales around exactly what a normal ``train_vfe3.py`` run trains.
Each cell overrides only the scale knob(s). Equal-token budget: ``max_steps`` / ``batch_size`` /
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from train_vfe3 import config as BASELINE          # the train_vfe3 operating point (import, not edit)
from vfe3.config import VFE3Config
from vfe3.data.datasets import make_dataloader, tokens_per_char as _tokens_per_char
from vfe3.model.model import VFEModel, build_group
from vfe3.run_artifacts import RunArtifacts, finalize_run
from vfe3.train import coverage_lines, train

logger = logging.getLogger("scaling")


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


def route_blocksize(embed_dim: int, n_heads_list: List[int]) -> List[Dict[str, Any]]:
    r"""Grow N by SHRINKING the head count at fixed K (route B): block_glk n_gen = K^2/n_heads, so
    fewer/larger blocks -> more params. n_heads == n_blocks keeps causal_alibi + the mixer valid."""
    return [{"label": f"K{embed_dim}_h{h}", "route": "blocksize", "scale_knob": "n_heads",
             "overrides": {"embed_dim": embed_dim, "n_heads": h, "gauge_group": "block_glk"}}
            for h in n_heads_list]


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


# The full route registry. Each value is a list of cells; CONFIG["routes"] selects which run. Edit the
# grids freely -- the predicted n_params is printed per cell before training so a grid can be sized to
# the GPU first. Geometric (~2x) spacing in N gives even leverage to the log-log fit.
ROUTES: Dict[str, List[Dict[str, Any]]] = {
    "grow_K":        route_grow_k([20, 40, 80, 160], n_heads=4),
    "blocksize":     route_blocksize(64, [8, 4, 2]),
    "group":         route_group(64),
    "model_channel": route_model_channel(),
    "infer_T":       route_inference_t([1, 2, 4, 8]),
    "infer_L":       route_inference_l([1, 2, 4, 6]),
}


# =============================================================================
# CLICK-TO-RUN KNOBS  -- edit, then run.
# =============================================================================
CONFIG: Dict[str, Any] = {
    # Which routes to run (keys of ROUTES), in order. A curated subset for a single-GPU session.
    "routes":     ["grow_K", "blocksize", "group", "model_channel", "infer_T", "infer_L"],

    # Seeds per cell. Graduated budget is sensible (more seeds at the cheap small end); the simplest
    # honest default is one shared list applied to every cell -- trim/extend per your compute budget.
    "seeds":      [6, 64, 23],

    "device":     "auto",                                   # 'auto' -> CUDA (RTX 5090) else CPU

    # Dataset for every run (NOT a VFE3Config field; the loader seam). Held-out CE is comparable across
    # sizes only within one tokenizer/corpus.
    "dataset":    "wikitext-103",                           # "wikitext-103" | "wikitext-2" | "wiki-en" | ...

    # Cap the TRAIN stream for fast scaling passes (validation/test always read in full). None = full.
    "max_tokens": None,

    # Override every run's max_steps (None = use the train_vfe3 baseline). HOLD THIS FIXED across the
    # parameter routes for an equal-token budget (so tokens_seen is constant and the exponent is clean).
    "max_steps":  None,

    # Skip cells whose run dir already holds a summary.json built from the SAME config (idempotent
    # reruns / crash recovery), exactly like ablation.py's resume.
    "resume":     True,

    "output_dir": "vfe3_scaling_results",
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
