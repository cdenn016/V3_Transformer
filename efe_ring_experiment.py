r"""Phase 1 pre-registered experiment: the one-step EFE pragmatic-reranker gate on the controlled
closed-loop ring goal-steering task (spec docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md,
Section 4).

CLICK-TO-RUN: edit CONFIG below, then run. No CLI args. Trains the three-seed synthetic-task
checkpoint set, applies the predictive-adequacy precondition (>= 0.98), dev-tunes gamma over the
sealed grid, runs the v1-live arm matrix on a paired test episode set, and applies the conjunctive
go/no-go gate with paired McNemar tests (Holm-Bonferroni over the primary gates) plus bootstrap CIs.

Compute: the iterative E-step is slow on CPU; RUN THIS ON THE GPU (it auto-uses CUDA when available).
The sealed budgets below reproduce the pre-registration. To smoke-test the pipeline quickly first,
set CONFIG['steps'] and CONFIG['n_episodes']/['n_dev'] small (this is logged in the output, so any
deviation from the sealed run is visible -- no silent caps).
"""
import hashlib
import json
import logging
import math
import os
import pickle
import time
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from vfe3.inference import ring_task as rt
from vfe3.run_artifacts import _atomic_replace, semantic_config_fingerprint

logger = logging.getLogger(__name__)

# ---- pre-registration surface (spec Section 4.7); reduce only for a logged smoke test ----
CONFIG = dict(
    seeds=(6, 23, 64),               # sealed seed list
    steps=15000,                     # sealed training budget per seed
    batch_size=256,
    lr=3e-3,
    log_every=100,                   # print training step/loss/rate/ETA every N steps (0 = silent)
    n_dev=1000,                      # dev episodes for gamma tuning
    n_episodes=5000,                 # sealed test episodes per arm (paired)
    budget=10,                       # T_ep
    candidate_mode="actions",        # "actions" (the 3 control actions) | "top_k" (top-Kp tokens)
    top_k=8,                         # Kp menu size (only used when candidate_mode="top_k")
    beta_C=5.0,                      # preference precision
    gamma_grid=(0.5, 1.0, 2.0, 4.0, 8.0),
    adequacy_threshold=0.98,
    delta_min=0.05,                  # minimum effect size (absolute success rate)
    alpha=0.05,
    out_dir="vfe3_policy_results/ring_v1",
    resume=True,                     # reuse current per-seed bundles under out_dir/seeds; False = full recompute
)


# ---- Phase 2 baseline pre-registration constants (kept separate from the user-edited CONFIG) ----
TEMP_GRID     = (0.5, 1.0, 2.0, 4.0, 8.0)  # temp grid for the temp-tuned logprob baseline (matches gamma_grid cardinality)
NUCLEUS_TOP_P = 0.9                          # nucleus (top-p) sampling baseline mass
TYPICAL_P     = 0.9                          # locally-typical sampling baseline mass
FDR_Q         = 0.05                         # Benjamini-Hochberg FDR level over the arm grid (spec 4.6)


def sample_episodes(n, seed, device):
    # Draw a uniform NONZERO ring offset and add it to s0, so goals are uniform over the M-1 states
    # g != s0 (spec Section 4.1). The earlier "resample collisions to (s0+1)%M" remapping doubled the
    # mass on the clockwise neighbor (delta=1) and distorted the directional action prior (audit F2).
    g = torch.Generator().manual_seed(seed)
    s0 = torch.randint(0, rt.M, (n,), generator=g)
    offset = torch.randint(1, rt.M, (n,), generator=g)   # uniform offset in 1..M-1
    goals = (s0 + offset) % rt.M                          # uniform over g != s0
    return goals.to(device), s0.to(device)


def mcnemar(correct_a, correct_b):
    a = correct_a.bool().cpu().numpy()
    b = correct_b.bool().cpu().numpy()
    n_b = int((a & ~b).sum())                            # a right, b wrong
    n_c = int((~a & b).sum())                            # a wrong, b right
    if n_b + n_c == 0:
        return n_b, n_c, 0.0, 1.0
    chi2 = (abs(n_b - n_c) - 1) ** 2 / (n_b + n_c)       # continuity-corrected
    p = math.erfc(math.sqrt(chi2 / 2.0))                 # chi2, 1 dof, survival
    return n_b, n_c, chi2, p


def bootstrap_diff_ci(correct_a, correct_b, n_boot=2000, alpha=0.05, seed=0):
    a = correct_a.float().cpu().numpy()
    b = correct_b.float().cpu().numpy()
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def bh_fdr(pvals, q=0.05):
    # Benjamini-Hochberg step-up: control FDR at q over the comparisons. Returns {name: (p, significant)}.
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    crit = 0
    for k, (_, p) in enumerate(items, start=1):
        if p <= k / m * q:
            crit = k                                         # largest rank meeting the BH threshold
    return {name: (p, (rank + 1) <= crit) for rank, (name, p) in enumerate(items)}


def tune_gamma(model, goals, s0, cfg):
    best_g, best_sr = None, -1.0
    for gamma in cfg["gamma_grid"]:
        out = rt.run_episodes(model, goals, s0, "efe_one_step", preference_key="task",
                              gamma=gamma, candidate_mode=cfg["candidate_mode"], top_k=cfg["top_k"],
                              beta_C=cfg["beta_C"], budget=cfg["budget"])
        sr = float(out["correct"].float().mean())
        print(f"     tune gamma={gamma}: dev_success={sr:.3f}", flush=True)
        # argmax dev success; ties broken toward gamma = 1.0 (spec Section 4.2)
        if sr > best_sr or (sr == best_sr and gamma == 1.0):
            best_sr, best_g = sr, gamma
    return best_g, best_sr


def tune_temperature(model, goals, s0, cfg, seed):
    # The temperature-tuned logprob baseline: one tuning DOF, the identical selection rule as gamma
    # (argmax dev success, ties toward T=1.0), over a matched-cardinality grid (spec Section 4.2).
    best_t, best_sr = None, -1.0
    for i, temp in enumerate(TEMP_GRID):
        out = rt.run_episodes(model, goals, s0, "temp_sample", temperature=temp,
                              candidate_mode=cfg["candidate_mode"], top_k=cfg["top_k"], budget=cfg["budget"],
                              generator=torch.Generator().manual_seed(seed + 300 + i))
        sr = float(out["correct"].float().mean())
        print(f"     tune temp={temp}: dev_success={sr:.3f}", flush=True)
        if sr > best_sr or (sr == best_sr and temp == 1.0):
            best_sr, best_t = sr, temp
    return best_t, best_sr


def arm_results(out):
    return dict(
        success=float(out["correct"].float().mean()),
        mean_steps_to_goal=float(out["steps_to_goal"].float().mean()),
        frac_at_goal=float(out["frac_at_goal"].float().mean()),
        mean_risk=float(out.get("mean_risk", 0.0)),            # scorer-arm component diagnostics (spec 4.4)
        mean_ambiguity=float(out.get("mean_ambiguity", 0.0)),
    )


def run_checkpoint(model, cfg, device, seed):
    print(f"   [seed {seed}] tuning gamma over {cfg['gamma_grid']} on {cfg['n_dev']} dev episodes...", flush=True)
    dev_goals, dev_s0 = sample_episodes(cfg["n_dev"], seed + 100, device)
    gamma, dev_sr = tune_gamma(model, dev_goals, dev_s0, cfg)
    temp, temp_dev_sr = tune_temperature(model, dev_goals, dev_s0, cfg, seed)
    print(f"   [seed {seed}] gamma*={gamma} (dev={dev_sr:.3f}); temp*={temp} (dev={temp_dev_sr:.3f}); "
          f"running the arm matrix on {cfg['n_episodes']} paired test episodes...", flush=True)
    goals, s0 = sample_episodes(cfg["n_episodes"], seed + 200, device)   # paired test set
    base_kw = dict(candidate_mode=cfg["candidate_mode"], top_k=cfg["top_k"], budget=cfg["budget"])

    def efe(pref, terms, g):
        return rt.run_episodes(model, goals, s0, "efe_one_step", preference_key=pref, score_terms=terms,
                               gamma=g, beta_C=cfg["beta_C"], **base_kw)

    def _run(name, fn):
        out = fn()
        print(f"     arm {name:18s} success={float(out['correct'].float().mean()):.3f}", flush=True)
        return out

    outs = {
        "full_efe_tuned":     _run("full_efe_tuned",     lambda: efe("task", ("risk", "ambiguity"), gamma)),
        "full_efe_g1":        _run("full_efe_g1",        lambda: efe("task", ("risk", "ambiguity"), 1.0)),  # sensitivity
        "risk_only":          _run("risk_only",          lambda: efe("task", ("risk",), gamma)),
        "ambiguity_only":     _run("ambiguity_only",     lambda: efe("task", ("ambiguity",), gamma)),
        "flat_pref":          _run("flat_pref",          lambda: efe("flat", ("risk", "ambiguity"), gamma)),  # inert v1
        "p_data_control":     _run("p_data_control",     lambda: efe("held_out_predictive", ("risk", "ambiguity"), gamma)),
        "temp_tuned_logprob": _run("temp_tuned_logprob", lambda: rt.run_episodes(model, goals, s0, "temp_sample",
                                            temperature=temp, generator=torch.Generator().manual_seed(seed + 11), **base_kw)),
        "logprob_baseline":   _run("logprob_baseline",   lambda: rt.run_episodes(model, goals, s0, "logprob_control",
                                            gamma=1.0, **base_kw)),
        "nucleus":            _run("nucleus",            lambda: rt.run_episodes(model, goals, s0, "nucleus",
                                            top_p=NUCLEUS_TOP_P, generator=torch.Generator().manual_seed(seed + 12), **base_kw)),
        "typical":            _run("typical",            lambda: rt.run_episodes(model, goals, s0, "typical",
                                            top_p=TYPICAL_P, generator=torch.Generator().manual_seed(seed + 13), **base_kw)),
        "greedy_ref":         _run("greedy_ref",         lambda: rt.run_episodes(model, goals, s0, "greedy_ref", **base_kw)),
        "random":             _run("random",             lambda: rt.run_episodes(model, goals, s0, "random",
                                            generator=torch.Generator().manual_seed(seed + 7), **base_kw)),
    }
    metrics = {k: arm_results(v) for k, v in outs.items()}

    efe_correct = outs["full_efe_tuned"]["correct"]
    gates = {}
    # conjunctive PRIMARY gate, Holm-Bonferroni (spec Section 4.6): full EFE must beat the p_data control
    # and the temperature-tuned logprob baseline by > delta_min with corrected significance. The
    # matched-compute beam / best-of-N primaries are deferred to the horizon phase (Phase 3), where they
    # are not degenerate (see docs/research/active-inference/2026-06-28-phase2-scope-note.md).
    primaries = {}
    for name in ("p_data_control", "temp_tuned_logprob"):
        nb, nc, chi2, p = mcnemar(efe_correct, outs[name]["correct"])
        diff, lo, hi = bootstrap_diff_ci(efe_correct, outs[name]["correct"], alpha=cfg["alpha"])
        primaries[name] = dict(mcnemar_b=nb, mcnemar_c=nc, chi2=chi2, p=p, diff=diff, ci=[lo, hi])
    ordered = sorted(primaries.items(), key=lambda kv: kv[1]["p"])
    m = len(ordered)
    holm_pass = {}
    holm_rejected = True                                      # Holm step-down: once one p fails, all higher-p fail
    for rank, (name, st) in enumerate(ordered):
        thresh = cfg["alpha"] / (m - rank)
        holm_rejected = holm_rejected and (st["p"] < thresh)
        holm_pass[name] = bool(holm_rejected and st["diff"] > cfg["delta_min"])
    gates["primary"] = {k: {**primaries[k], "holm_pass": holm_pass[k]} for k in primaries}
    # Benjamini-Hochberg FDR over the broader arm grid (full EFE vs every other arm on the primary
    # success metric), the multiplicity control for the exploratory matrix (spec Section 4.6, q=0.05).
    grid_p = {name: mcnemar(efe_correct, outs[name]["correct"])[3] for name in outs if name != "full_efe_tuned"}
    gates["fdr_grid"] = {name: {"p": p, "significant": sig} for name, (p, sig) in bh_fdr(grid_p, q=FDR_Q).items()}
    # v1 lesion gates (spec Sections 4.6/4.7). Random must NOT match EFE: "full EFE beats random by more
    # than delta_min" (a strict global-argmin test is wrong here because every goalless arm legitimately
    # collapses to ~random and any one can edge random by a single episode without weakening the lesion).
    # Closed-loop causality: the committed action must measurably change the next observation.
    gates["random_clearly_beaten"] = bool(
        metrics["full_efe_tuned"]["success"] - metrics["random"]["success"] > cfg["delta_min"])
    gates["closed_loop_causal"] = rt.closed_loop_causality_holds()
    gates["go"] = bool(all(holm_pass.values()) and gates["random_clearly_beaten"]
                       and gates["closed_loop_causal"])
    return dict(gamma=gamma, temp=temp, dev_success=dev_sr, metrics=metrics, gates=gates)


# ---- per-seed durable bundles + resume state machine (audit PB-04) -------------------------------
# Each seed is trained and evaluated independently and published atomically to
# <out_dir>/seeds/seed_<seed>.pt so a crash never discards a finished seed. A 'trained' bundle carries
# the model + adequacy (evaluation still pending); a 'complete' bundle also carries the validated
# 12-arm result. Resume reuses a bundle ONLY when it is current -- same semantic experiment config,
# same executable code, intact schema -- and fails closed (recomputes) on any absent/malformed/stale/
# incompatible bundle. In-step optimizer resume is out of scope.

_SEED_BUNDLE_SCHEMA_VERSION = 1


def _semantic_experiment_config(cfg: Mapping[str, object]) -> Dict[str, object]:
    """Return every training/evaluation field that can change one seed's result.

    resume / out_dir / log_every are EXCLUDED: they set the reuse policy and I/O cadence, never the
    trained weights or the measured arm matrix. The Phase-2 baseline constants TEMP_GRID /
    NUCLEUS_TOP_P / TYPICAL_P / FDR_Q (module-level, not in CONFIG) DO shape the scored arms, so they
    join the CONFIG fields in the semantic identity.
    """
    included = ("steps", "batch_size", "lr", "n_dev", "n_episodes", "budget",
                "candidate_mode", "top_k", "beta_C", "gamma_grid",
                "adequacy_threshold", "delta_min", "alpha")
    semantic: Dict[str, object] = {key: cfg[key] for key in included}
    semantic["TEMP_GRID"]     = TEMP_GRID
    semantic["NUCLEUS_TOP_P"] = NUCLEUS_TOP_P
    semantic["TYPICAL_P"]     = TYPICAL_P
    semantic["FDR_Q"]         = FDR_Q
    return semantic


def _efe_ring_code_identity(root: Optional[Path] = None) -> str:
    """Hash the executable ring entry point and package Python sources, never result files.

    Digests the relative path plus bytes of ``efe_ring_experiment.py`` and every sorted
    ``vfe3/**/*.py`` (``__pycache__`` excluded). It NEVER inspects git status, docs, ``out_dir``, seed
    bundles, or aggregate results, so publishing a seed bundle or ``ring_v1_results.json`` under the
    source tree leaves the identity unchanged while an edit to the executable code moves it.
    """
    base = Path(__file__).resolve().parent if root is None else Path(root).resolve()
    sources = []
    entry = base / "efe_ring_experiment.py"
    if entry.is_file():
        sources.append(entry)
    package = base / "vfe3"
    if package.is_dir():
        sources.extend(sorted(p for p in package.glob("**/*.py") if "__pycache__" not in p.parts))
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validated_complete_result(result: object, adequacy: float) -> Optional[Dict[str, object]]:
    """Return a safe aggregate-ready copy of one complete result, else None.

    Explicit so no restored mapping is indexed before validation: the top-level and result adequacy
    must agree as finite reals (never bools), ``admitted`` must be a real bool, and when admitted the
    tuning scalars, the exact 12-arm metrics key set (each arm a finite ``success``), and a real-bool
    ``gates["go"]`` must all be present.
    """
    if not isinstance(result, Mapping):
        return None
    copied = dict(result)
    result_adequacy = copied.get("adequacy")
    if (not isinstance(result_adequacy, Real) or isinstance(result_adequacy, bool)
            or not math.isfinite(float(result_adequacy))
            or float(result_adequacy) != float(adequacy)):
        return None
    admitted_value = copied.get("admitted")
    if type(admitted_value) is not bool:
        return None
    if admitted_value:
        for name in ("gamma", "temp"):
            value = copied.get(name)
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not math.isfinite(float(value))):
                return None
        metrics = copied.get("metrics")
        gates = copied.get("gates")
        expected_arms = {
            "full_efe_tuned", "full_efe_g1", "risk_only", "ambiguity_only",
            "flat_pref", "p_data_control", "temp_tuned_logprob", "logprob_baseline",
            "nucleus", "typical", "greedy_ref", "random",
        }
        if not isinstance(metrics, Mapping) or set(metrics) != expected_arms:
            return None
        if not isinstance(gates, Mapping) or type(gates.get("go")) is not bool:
            return None
        for arm in metrics.values():
            if not isinstance(arm, Mapping):
                return None
            success = arm.get("success")
            if (not isinstance(success, Real) or isinstance(success, bool)
                    or not math.isfinite(float(success))):
                return None
    return copied


def _save_seed_bundle(
    path:              Path,
    model:             rt.VFEModel,
    experiment_config: Mapping[str, object],
    result:            Optional[Mapping[str, object]],

    *,
    seed:     int,
    adequacy: float,
    status:   str,
) -> Path:
    r"""Atomically publish a `trained` or `complete` seed bundle.

    Written through ``seed_path.with_suffix(".pt.tmp")`` + the shared atomic-replace helper (same-dir
    tmp + ``os.replace``), so the trained and complete states are each independently publishable and a
    crash never leaves a truncated ``.pt`` at the final name.
    """
    semantic     = dict(experiment_config)
    model_config = asdict(model.cfg)
    bundle = {
        "schema_version":              _SEED_BUNDLE_SCHEMA_VERSION,
        "status":                      status,
        "seed":                        int(seed),
        "semantic_config":             semantic,
        "semantic_config_fingerprint": semantic_config_fingerprint(semantic),
        "code_identity_sha256":        _efe_ring_code_identity(),
        "model_config":                model_config,
        "model_config_fingerprint":    semantic_config_fingerprint(model_config),
        "model_state":                 model.state_dict(),
        "adequacy":                    float(adequacy),
        "result":                      (dict(result) if result is not None else None),
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(bundle, tmp)
    _atomic_replace(path, tmp)
    return path


def _load_seed_bundle_if_current(
    path:              Path,
    experiment_config: Mapping[str, object],
    device:            torch.device,

    *,
    seed: int,
) -> Optional[Tuple[rt.VFEModel, Dict[str, object]]]:
    r"""Rebuild the model and return a current validated seed bundle, else return None.

    Fails closed on an absent/malformed/stale/incompatible bundle: the schema, semantic experiment
    fingerprint, executable code identity, adequacy, and (for a complete bundle) the full result must
    all validate, and the model must rebuild + load strictly. A complete bundle's stored ``result`` is
    replaced with the ``_validated_complete_result`` copy before returning. Expected file/safe-load/
    schema/state-dict failures are caught, logged, and turned into None (the state machine then
    recomputes); programming errors surface.
    """
    if not path.is_file():
        return None
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(bundle, Mapping):
            raise RuntimeError("bundle is not a mapping")
        if bundle.get("schema_version") != _SEED_BUNDLE_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported schema_version {bundle.get('schema_version')!r}")
        status = bundle.get("status")
        if status not in ("trained", "complete"):
            raise RuntimeError(f"status {status!r} is neither 'trained' nor 'complete'")
        if bundle.get("seed") != int(seed):
            raise RuntimeError(f"bundle seed {bundle.get('seed')!r} != requested {int(seed)}")
        if bundle.get("semantic_config_fingerprint") != semantic_config_fingerprint(dict(experiment_config)):
            raise RuntimeError("semantic experiment config drift")
        if bundle.get("code_identity_sha256") != _efe_ring_code_identity():
            raise RuntimeError("executable code identity drift")
        adequacy = bundle.get("adequacy")
        if (not isinstance(adequacy, Real) or isinstance(adequacy, bool)
                or not math.isfinite(float(adequacy))):
            raise RuntimeError("adequacy is not a finite real number")
        adequacy = float(adequacy)
        result = bundle.get("result")
        if status == "trained":
            if result is not None:
                raise RuntimeError("a trained bundle must not carry a result")
            validated_result = None
        else:
            validated_result = _validated_complete_result(result, adequacy)
            if validated_result is None:
                raise RuntimeError("complete-result schema validation failed")
        model_config = bundle.get("model_config")
        if not isinstance(model_config, Mapping):
            raise RuntimeError("model_config is not a mapping")
        if bundle.get("model_config_fingerprint") != semantic_config_fingerprint(dict(model_config)):
            raise RuntimeError("model_config fingerprint mismatch")
        model_state = bundle.get("model_state")
        if not isinstance(model_state, Mapping):
            raise RuntimeError("model_state is not a mapping")
        model = rt.VFEModel(rt.VFE3Config(**dict(model_config)))
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        logger.warning("seed %s bundle at %s is unusable (%s); recomputing",
                       seed, getattr(path, "name", path), exc)
        return None
    return model, {"status": status, "adequacy": adequacy, "result": validated_result}


def main():
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  | sealed run: {cfg['seeds']} seeds x {cfg['steps']} steps, "
          f"{cfg['n_episodes']} test episodes/arm")
    if device.type == "cpu":
        print("WARNING: running on CPU; the iterative E-step makes this very slow. Run on the GPU.")

    results = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
               "device": str(device), "checkpoints": {}}

    out_dir  = Path(cfg["out_dir"])
    seed_dir = out_dir / "seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_dir.mkdir(parents=True, exist_ok=True)
    semantic_cfg = _semantic_experiment_config(cfg)
    admitted = []

    for seed in cfg["seeds"]:
        seed_path = seed_dir / f"seed_{int(seed)}.pt"
        bundle = (_load_seed_bundle_if_current(seed_path, semantic_cfg, device, seed=seed)
                  if cfg["resume"] else None)
        if bundle is not None and bundle[1]["status"] == "complete":
            entry = dict(bundle[1]["result"])
            print(f"\n[seed {seed}] resumed COMPLETE from {seed_path.name} "
                  f"(adequacy={entry['adequacy']:.4f}, "
                  f"{'ADMIT' if entry['admitted'] else 'EXCLUDE'})", flush=True)
        else:
            if bundle is None:
                t0 = time.time()
                print(f"\n[seed {seed}] training {cfg['steps']} steps (batch {cfg['batch_size']}) "
                      f"on {device}...", flush=True)
                model, adequacy = rt.train_ring_checkpoint(
                    seed=seed,
                    steps=cfg["steps"],
                    batch_size=cfg["batch_size"],
                    lr=cfg["lr"],
                    log_every=cfg["log_every"],
                    device=str(device),
                )
                _save_seed_bundle(seed_path, model, semantic_cfg, None,
                                  seed=seed, adequacy=adequacy, status="trained")
                tail = f"  [{time.time() - t0:.0f}s]"
            else:
                model, saved = bundle
                adequacy = float(saved["adequacy"])
                tail = "  (resumed trained; evaluating)"
            is_admitted = adequacy >= cfg["adequacy_threshold"]
            print(f"[seed {seed}] adequacy={adequacy:.4f} "
                  f"({'ADMIT' if is_admitted else 'EXCLUDE'}){tail}", flush=True)
            entry = {"adequacy": adequacy, "admitted": is_admitted}
            if is_admitted:
                entry.update(run_checkpoint(model, cfg, str(device), seed))
                g = entry["gates"]
                print(f"   gamma*={entry['gamma']} temp*={entry['temp']}  "
                      f"full_efe={entry['metrics']['full_efe_tuned']['success']:.3f}  "
                      f"temp_lp={entry['metrics']['temp_tuned_logprob']['success']:.3f}  "
                      f"p_data={entry['metrics']['p_data_control']['success']:.3f}  "
                      f"random={entry['metrics']['random']['success']:.3f}  "
                      f"causal={g['closed_loop_causal']}  GO={g['go']}")
            _save_seed_bundle(seed_path, model, semantic_cfg, entry,
                              seed=seed, adequacy=adequacy, status="complete")
        if bool(entry["admitted"]):
            admitted.append(seed)
        results["checkpoints"][str(seed)] = entry

    # cross-seed aggregate + overall go/no-go (all admitted seeds must individually pass)
    if admitted:
        def mean_sr(arm):
            return float(np.mean([results["checkpoints"][str(s)]["metrics"][arm]["success"] for s in admitted]))
        arms = list(results["checkpoints"][str(admitted[0])]["metrics"])
        results["aggregate"] = {"admitted_seeds": admitted,
                                "mean_success": {a: mean_sr(a) for a in arms},
                                "all_seeds_go": all(results["checkpoints"][str(s)]["gates"]["go"] for s in admitted)}
        print("\n=== cross-seed mean success ===")
        for a in sorted(arms, key=lambda x: -results["aggregate"]["mean_success"][x]):
            print(f"  {a:18s} {results['aggregate']['mean_success'][a]:.3f}")
        verdict = ("GO: full EFE beats the controls beyond delta_min with the lesion gates"
                   if results["aggregate"]["all_seeds_go"]
                   else "NO-GO / DEMOTE: the pre-registered demotion rule applies (spec Section 4.6)")
        results["verdict"] = verdict
        print("\nVERDICT:", verdict)
    else:
        results["verdict"] = "NO checkpoint cleared the predictive-adequacy precondition"
        print("\nVERDICT:", results["verdict"])

    # Publish the cross-seed aggregate only after every requested seed has a complete entry above, and
    # atomically (same-dir tmp + os.replace) so a crash never leaves a truncated ring_v1_results.json.
    out_path = os.path.join(cfg["out_dir"], "ring_v1_results.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
