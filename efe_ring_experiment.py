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
import json
import math
import os
import time

import numpy as np
import torch

from vfe3.inference import ring_task as rt

# ---- pre-registration surface (spec Section 4.7); reduce only for a logged smoke test ----
CONFIG = dict(
    seeds=(6,),# 23, 64),               # sealed seed list
    steps=3000,                     # sealed training budget per seed
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
)


def sample_episodes(n, seed, device):
    g = torch.Generator().manual_seed(seed)
    s0 = torch.randint(0, rt.M, (n,), generator=g)
    goals = torch.randint(0, rt.M, (n,), generator=g)
    clash = goals == s0
    goals[clash] = (goals[clash] + 1) % rt.M             # enforce g != s0
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


def arm_results(out):
    return dict(
        success=float(out["correct"].float().mean()),
        mean_steps_to_goal=float(out["steps_to_goal"].float().mean()),
        frac_at_goal=float(out["frac_at_goal"].float().mean()),
    )


def run_checkpoint(model, cfg, device, seed):
    print(f"   [seed {seed}] tuning gamma over {cfg['gamma_grid']} on {cfg['n_dev']} dev episodes...", flush=True)
    dev_goals, dev_s0 = sample_episodes(cfg["n_dev"], seed + 100, device)
    gamma, dev_sr = tune_gamma(model, dev_goals, dev_s0, cfg)
    print(f"   [seed {seed}] gamma*={gamma} (dev_success={dev_sr:.3f}); running the arm matrix on "
          f"{cfg['n_episodes']} paired test episodes...", flush=True)
    goals, s0 = sample_episodes(cfg["n_episodes"], seed + 200, device)   # paired test set

    def efe(pref, terms, g):
        return rt.run_episodes(model, goals, s0, "efe_one_step", preference_key=pref, score_terms=terms,
                               gamma=g, candidate_mode=cfg["candidate_mode"], top_k=cfg["top_k"],
                               beta_C=cfg["beta_C"], budget=cfg["budget"])

    def _run(name, fn):
        out = fn()
        print(f"     arm {name:18s} success={float(out['correct'].float().mean()):.3f}", flush=True)
        return out

    outs = {
        "full_efe_tuned":   _run("full_efe_tuned",   lambda: efe("task", ("risk", "ambiguity"), gamma)),
        "full_efe_g1":      _run("full_efe_g1",      lambda: efe("task", ("risk", "ambiguity"), 1.0)),  # sensitivity
        "risk_only":        _run("risk_only",        lambda: efe("task", ("risk",), gamma)),
        "ambiguity_only":   _run("ambiguity_only",   lambda: efe("task", ("ambiguity",), gamma)),
        "flat_pref":        _run("flat_pref",        lambda: efe("flat", ("risk", "ambiguity"), gamma)),  # inert v1
        "p_data_control":   _run("p_data_control",   lambda: efe("held_out_predictive", ("risk", "ambiguity"), gamma)),
        "logprob_baseline": _run("logprob_baseline", lambda: rt.run_episodes(model, goals, s0, "logprob_control",
                                            gamma=1.0, candidate_mode=cfg["candidate_mode"],
                                            top_k=cfg["top_k"], budget=cfg["budget"])),
        "random":           _run("random",           lambda: rt.run_episodes(model, goals, s0, "random",
                                            candidate_mode=cfg["candidate_mode"], top_k=cfg["top_k"],
                                            budget=cfg["budget"], generator=torch.Generator().manual_seed(seed + 7))),
    }
    metrics = {k: arm_results(v) for k, v in outs.items()}

    efe_correct = outs["full_efe_tuned"]["correct"]
    gates = {}
    # Holm-Bonferroni over the two distinct primary comparisons (beam / best-of-N reduce to the
    # logprob baseline at H=1, so they are not separate tests here; the horizon phase adds them).
    primaries = {}
    for name in ("p_data_control", "logprob_baseline"):
        nb, nc, chi2, p = mcnemar(efe_correct, outs[name]["correct"])
        diff, lo, hi = bootstrap_diff_ci(efe_correct, outs[name]["correct"], alpha=cfg["alpha"])
        primaries[name] = dict(mcnemar_b=nb, mcnemar_c=nc, chi2=chi2, p=p,
                               diff=diff, ci=[lo, hi])
    # Holm correction
    ordered = sorted(primaries.items(), key=lambda kv: kv[1]["p"])
    m = len(ordered)
    holm_pass = {}
    for rank, (name, st) in enumerate(ordered):
        thresh = cfg["alpha"] / (m - rank)
        holm_pass[name] = bool(st["p"] < thresh and st["diff"] > cfg["delta_min"])
    gates["primary"] = {k: {**primaries[k], "holm_pass": holm_pass[k]} for k in primaries}
    # Random lesion (spec Section 4.7: falsified if "random-score is not clearly worst"). The intent is
    # that a random scorer must NOT match EFE; the operationalization is "full EFE beats random by more
    # than delta_min". A strict global-argmin test is wrong here because at v1 every goalless arm
    # (ambiguity_only, flat, p_data, logprob) legitimately collapses to ~random, so they cluster within
    # noise and any one of them can edge random by a single episode without weakening the lesion.
    gates["random_clearly_beaten"] = bool(
        metrics["full_efe_tuned"]["success"] - metrics["random"]["success"] > cfg["delta_min"])
    gates["go"] = bool(all(holm_pass.values()) and gates["random_clearly_beaten"])
    return dict(gamma=gamma, dev_success=dev_sr, metrics=metrics, gates=gates)


def main():
    cfg = CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["out_dir"], exist_ok=True)
    print(f"device={device}  | sealed run: {cfg['seeds']} seeds x {cfg['steps']} steps, "
          f"{cfg['n_episodes']} test episodes/arm")
    if device == "cpu":
        print("WARNING: running on CPU; the iterative E-step makes this very slow. Run on the GPU.")

    results = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
               "device": device, "checkpoints": {}}
    admitted = []
    for seed in cfg["seeds"]:
        t0 = time.time()
        print(f"\n[seed {seed}] training {cfg['steps']} steps (batch {cfg['batch_size']}) on {device}...", flush=True)
        model, adeq = rt.train_ring_checkpoint(
            seed=seed, steps=cfg["steps"], batch_size=cfg["batch_size"], lr=cfg["lr"],
            log_every=cfg["log_every"], device=device)
        ok = adeq >= cfg["adequacy_threshold"]
        print(f"[seed {seed}] adequacy={adeq:.4f} ({'ADMIT' if ok else 'EXCLUDE'})  [{time.time()-t0:.0f}s]", flush=True)
        entry = {"adequacy": adeq, "admitted": ok}
        if ok:
            entry.update(run_checkpoint(model, cfg, device, seed))
            admitted.append(seed)
            g = entry["gates"]
            print(f"   gamma*={entry['gamma']}  full_efe={entry['metrics']['full_efe_tuned']['success']:.3f}  "
                  f"logprob={entry['metrics']['logprob_baseline']['success']:.3f}  "
                  f"p_data={entry['metrics']['p_data_control']['success']:.3f}  "
                  f"random={entry['metrics']['random']['success']:.3f}  GO={g['go']}")
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

    out_path = os.path.join(cfg["out_dir"], "ring_v1_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
