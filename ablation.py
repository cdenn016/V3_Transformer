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

# The baseline operating point IS train_vfe3.py's config (single source of truth) plus its
# zero-dependency synthetic stream (the fallback when a corpus cache is absent).
from train_vfe3 import config as BASELINE_CONFIG
from train_vfe3 import synthetic_period3_loader

logger = logging.getLogger("ablation")


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

SWEEPS: Dict[str, Dict[str, Any]] = {

    # --- attention temperature ----------------------------------------------
    "kappa": {
        "description":    "attention temperature tau = kappa * sqrt(d_head)",
        "param":          "kappa",
        "values":         [0.5, 0.7, 1.0, 1.4, 2.0],
        "baseline_value": 1.0,
    },

    # --- self-coupling form (closed-form, no learned params) ----------------
    "alpha_mode": {
        "description":    "self-coupling alpha form (constant vs state-dependent)",
        "param":          "alpha_mode",
        "values":         ["constant", "state_dependent", "state_dependent_per_coord"],
        "baseline_value": "state_dependent_per_coord",
    },

    # --- E-step belief-mean step size ---------------------------------------
    "e_mu_lr": {
        "description":    "E-step natural-gradient step size for mu_q",
        "param":          "e_mu_lr",
        "values":         [0.3, 0.5, 0.7, 0.9, 1.1],
        "baseline_value": 0.7,
    },

    # --- M-step gauge-frame LR ----------------------------------------------
    "m_phi_lr": {
        "description":    "M-step learning rate for the gauge-frame parameters (phi)",
        "param":          "m_phi_lr",
        "values":         [0.0, 0.003, 0.006, 0.009, 0.015],
        "baseline_value": 0.009,
    },

    # --- gauge-frame L2 prior ------------------------------------------------
    "mass_phi": {
        "description":    "gauge prior weight (mass_phi / 2) ||phi||^2",
        "param":          "mass_phi",
        "values":         [0.0, 1e-4, 1e-3, 1e-2],
        "baseline_value": 0.0,
    },

    # --- phi preconditioner (block_glk supports killing_per_block) -----------
    "phi_precond_mode": {
        "description":    "gauge-step preconditioner on the phi update",
        "param":          "phi_precond_mode",
        "values":         ["none", "clip", "killing", "killing_per_block"],
        "baseline_value": "killing",
    },

    # --- attention prior pi_ij ----------------------------------------------
    "attention_prior": {
        "description":    "attention prior pi_ij (uniform vs causal vs alibi)",
        "param":          "attention_prior",
        "values":         ["uniform", "causal", "alibi"],
        "baseline_value": "causal",
    },

    # --- positional encoding (the pos_phi seam) -----------------------------
    "pos_phi": {
        "description":    "BCH positional encoding mode (none vs learned vs frozen)",
        "param":          "pos_phi",
        "values":         ["none", "learned", "frozen"],
        "baseline_value": "learned",
    },

    # --- canonical F vs entropy-suppressed surrogate (multi-arm) ------------
    "entropy_term": {
        "description": "canonical free energy (entropy term) vs entropy-suppressed surrogate",
        "configs": [
            {"label": "canonical", "include_attention_entropy": True},
            {"label": "surrogate", "include_attention_entropy": False},
        ],
    },

    # --- decode head: pure KL-to-prior vs linear projection (multi-arm) -----
    "decode_head": {
        "description": "KL-to-prior decode (pure path) vs learned linear projection (VFE_2.0 parity)",
        "configs": [
            {"label": "prior_bank",    "use_prior_bank": True},
            {"label": "linear_decode", "use_prior_bank": False},
        ],
    },

    # --- gauge group (multi-arm; head mixer needs >= 2 equal blocks) --------
    # use_head_mixer (True at baseline) requires >= 2 equal gauge-irrep blocks, which only
    # block_glk / tied_block_glk provide; the single-block glk / so_k arms turn it off so the
    # model constructs. Each arm therefore differs in exactly the group + the mixer it forces.
    "gauge_group": {
        "description": "gauge group (block_glk / tied_block_glk / glk / so_k)",
        "configs": [
            {"label": "block_glk",      "gauge_group": "block_glk"},
            {"label": "tied_block_glk", "gauge_group": "tied_block_glk"},
            {"label": "glk",            "gauge_group": "glk",  "use_head_mixer": False},
            {"label": "so_k",           "gauge_group": "so_k", "use_head_mixer": False},
        ],
    },

    # --- diagonal vs full covariance (multi-arm with cross-field requires) ---
    # The full-covariance arm flips family AND diagonal_covariance together, and also moves
    # off the per-coordinate alpha form (which is only defined for a diagonal family) -- a
    # textbook case where a naive single-field sweep would be rejected by __post_init__.
    "covariance": {
        "description": "belief covariance structure (diagonal vs full Gaussian)",
        "configs": [
            {"label": "diagonal", "family": "gaussian_diagonal", "diagonal_covariance": True},
            {"label": "full",     "family": "gaussian_full",     "diagonal_covariance": False,
                                   "alpha_mode": "state_dependent"},
        ],
    },
}


# Which sweeps run (and in what order) when CONFIG["sweep"] is None. Comment lines out to
# narrow a session; cheap-to-expensive is a good ordering for a single GPU.
SWEEP_ORDER: List[str] = [
    "kappa",
    "alpha_mode",
    "attention_prior",
    "entropy_term",
    "decode_head",
    # "e_mu_lr",
    # "m_phi_lr",
    # "mass_phi",
    # "phi_precond_mode",
    # "pos_phi",
    # "gauge_group",
    # "covariance",
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
        names = SWEEP_ORDER if CONFIG["sweep"] is None else [CONFIG["sweep"]]
        print(f"\nAvailable sweeps ({len(names)} selected of {len(SWEEPS)}):\n")
        print(f"{'name':<22}{'runs':>6}  description")
        print("-" * 78)
        for name in names:
            s = SWEEPS[name]
            print(f"{name:<22}{sweep_n_runs(s):>6}  {s['description']}")
        print(f"\ntotal runs: {sum(sweep_n_runs(SWEEPS[n]) for n in names)}")
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
