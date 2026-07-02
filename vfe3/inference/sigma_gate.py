r"""The sigma-validation gate measurement for the active-inference EFE policy scorer (spec
``docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md`` Sections 2.7, 4.5;
pre-registration ``docs/research/active-inference/2026-06-28-sigma-gate-prereg.md``).

The gate is the binding precondition for ANY sigma-derived epistemic/ambiguity arm (the ``sigma_mc``
ambiguity estimator, the epistemic-only arm, shuffled-sigma as a meaningful contrast). Theory: at a
sigma-free POINT belief the MI-bridge information gain ``I = H[q(o|pi)] - E_q H[p(o|s)]`` is identically
zero at EVERY horizon, so a live epistemic term requires belief covariance ``sigma`` that demonstrably
predicts realized outcomes. This module measures whether it does, on a given checkpoint, and writes a
versioned PASS/FAIL artifact the ``policy_sigma_ambiguity_validated`` config flag is bound to
(``config.py`` Guard 4).

Sealed gate (spec Section 4.5, thresholds in 4.7), all must hold for PASS:
  1. ``sigma_ce_spearman >= 0.2`` AND its 95% bootstrap CI lower bound > 0 AND > the measured floor.
  2. sigma-stratified cross-entropy is monotone (non-decreasing) across sigma strata.
  3. sigma-binned expected calibration error < 0.05.
The "floor" is a permutation null: the high quantile of the Spearman rho under sigma<->CE shuffling, the
noise level the real correlation must clear. These are operationalizations of the spec's prose; the
choices (strict-monotone flag plus a reported rank statistic; permutation floor; sigma-quantile bins for
the ECE) are recorded in the pre-registration note.

Pure measurement functions take aligned per-token 1-D tensors and are device/grad agnostic; the
``measure_sigma_gate`` orchestrator pulls those tensors from ``belief_ce_bank`` and writes the artifact.
"""
import hashlib
import json
import os
import re
from typing import Dict, Optional, Tuple

import torch

from vfe3.metrics import cv, spearman_rho


def spearman_bootstrap_ci(
    sigma: torch.Tensor,             # (M,) per-token belief-covariance trace tr(Sigma_q)
    ce:    torch.Tensor,             # (M,) per-token realized cross-entropy (nats)

    *,
    n_boot: int   = 2000,
    alpha:  float = 0.05,
    seed:   int   = 0,
) -> Tuple[float, float, float]:     # (rho, ci_lo, ci_hi)
    r"""Spearman rho(tr Sigma_q, CE) with a paired-token bootstrap (1-alpha) percentile CI. Resamples
    token indices with replacement and recomputes the rank correlation each draw."""
    sigma = sigma.flatten().to(torch.float64).cpu()
    ce = ce.flatten().to(torch.float64).cpu()
    n = sigma.numel()
    rho = spearman_rho(sigma, ce)
    g = torch.Generator().manual_seed(seed)
    boots = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        boots[b] = spearman_rho(sigma[idx], ce[idx])
    lo = float(torch.quantile(boots, alpha / 2))
    hi = float(torch.quantile(boots, 1 - alpha / 2))
    return rho, lo, hi


def permutation_floor(
    sigma: torch.Tensor,             # (M,) tr(Sigma_q)
    ce:    torch.Tensor,             # (M,) per-token CE

    *,
    n_perm: int   = 1000,
    q:      float = 0.95,
    seed:   int   = 0,
) -> float:                          # the noise floor the real rho's CI lower bound must exceed
    r"""The measured floor: the ``q`` quantile of the Spearman rho under the null that sigma carries no
    information about CE (sigma permuted against CE). A real correlation must clear this noise band."""
    sigma = sigma.flatten().to(torch.float64).cpu()
    ce = ce.flatten().to(torch.float64).cpu()
    n = sigma.numel()
    g = torch.Generator().manual_seed(seed + 1)
    null = torch.empty(n_perm, dtype=torch.float64)
    for p in range(n_perm):
        perm = torch.randperm(n, generator=g)
        null[p] = spearman_rho(sigma[perm], ce)
    return float(torch.quantile(null, q))


def sigma_stratified_ce(
    sigma: torch.Tensor,             # (M,) tr(Sigma_q)
    ce:    torch.Tensor,             # (M,) per-token CE

    *,
    n_strata: int = 10,
) -> Dict[str, object]:
    r"""Bin tokens into ``n_strata`` equal-count sigma strata (ordered by sigma) and report the per-
    stratum mean sigma and mean CE. ``monotone`` is True iff the stratum-mean CE is non-decreasing
    across strata (the sealed gate flag); ``mono_spearman`` (rank correlation of stratum index vs
    stratum-mean CE) is the robust diagnostic reported alongside."""
    sigma = sigma.flatten().to(torch.float64).cpu()
    ce = ce.flatten().to(torch.float64).cpu()
    order = sigma.argsort()
    parts = torch.tensor_split(order, n_strata)
    s_means = torch.stack([sigma[b].mean() for b in parts])
    c_means = torch.stack([ce[b].mean() for b in parts])
    diffs = c_means[1:] - c_means[:-1]
    monotone = bool((diffs >= 0).all())
    mono_spearman = spearman_rho(torch.arange(n_strata, dtype=torch.float64), c_means)
    return dict(sigma_means=s_means.tolist(), ce_means=c_means.tolist(),
                monotone=monotone, mono_spearman=mono_spearman)


def sigma_binned_ece(
    sigma:   torch.Tensor,           # (M,) tr(Sigma_q)
    conf:    torch.Tensor,           # (M,) per-token predicted confidence (max softmax prob)
    correct: torch.Tensor,           # (M,) per-token correctness (1.0 if argmax == gold else 0.0)

    *,
    n_bins: int = 10,
) -> float:                          # sum_b (n_b/M) |mean(conf_b) - mean(correct_b)|
    r"""Expected calibration error within sigma bins (spec Section 4.5): bin tokens into ``n_bins``
    equal-count sigma-quantile bins and average the |confidence - accuracy| gap, weighted by bin size.
    Below 0.05 means the model stays calibrated within each uncertainty stratum."""
    sigma = sigma.flatten().to(torch.float64).cpu()
    conf = conf.flatten().to(torch.float64).cpu()
    correct = correct.flatten().to(torch.float64).cpu()
    n = sigma.numel()
    order = sigma.argsort()
    ece = 0.0
    for b in torch.tensor_split(order, n_bins):
        ece += (b.numel() / n) * abs(float(conf[b].mean()) - float(correct[b].mean()))
    return ece


def evaluate_sigma_gate(
    sigma:   torch.Tensor,           # (M,) tr(Sigma_q)
    ce:      torch.Tensor,           # (M,) per-token CE (nats)
    conf:    torch.Tensor,           # (M,) per-token max softmax prob
    correct: torch.Tensor,           # (M,) per-token correctness (0/1)

    *,
    spearman_min: float = 0.2,       # sealed (spec 4.7)
    ece_max:      float = 0.05,      # sealed (spec 4.7)
    n_strata:     int   = 10,
    n_bins:       int   = 10,
    n_boot:       int   = 2000,
    n_perm:       int   = 1000,
    alpha:        float = 0.05,
    seed:         int   = 0,
) -> Dict[str, object]:
    r"""Run the full sigma-validation gate on aligned per-token tensors and return the record dict with
    every statistic plus a single ``status`` of "PASS"/"FAIL". PASS iff spearman >= ``spearman_min`` and
    its bootstrap CI lower bound exceeds both zero and the permutation floor, the stratified CE is
    monotone, and the sigma-binned ECE < ``ece_max`` (spec Section 4.5)."""
    rho, ci_lo, ci_hi = spearman_bootstrap_ci(sigma, ce, n_boot=n_boot, alpha=alpha, seed=seed)
    floor = permutation_floor(sigma, ce, n_perm=n_perm, seed=seed)
    strat = sigma_stratified_ce(sigma, ce, n_strata=n_strata)
    ece = sigma_binned_ece(sigma, conf, correct, n_bins=n_bins)
    passed = bool(rho >= spearman_min and ci_lo > 0.0 and ci_lo > floor
                  and strat["monotone"] and ece < ece_max)
    return dict(
        n_tokens=int(sigma.numel()),
        sigma_ce_spearman=rho,
        spearman_ci=[ci_lo, ci_hi],
        permutation_floor=floor,
        sigma_trace_cv=cv(sigma) if sigma.numel() >= 2 else 0.0,
        stratified_ce=strat,
        sigma_binned_ece=ece,
        thresholds=dict(spearman_min=spearman_min, ece_max=ece_max, alpha=alpha,
                        n_strata=n_strata, n_bins=n_bins, n_boot=n_boot, n_perm=n_perm),
        status="PASS" if passed else "FAIL",
    )


def write_sigma_gate_artifact(
    record:        Dict[str, object],

    *,
    checkpoint_id: str,
    spec_commit:   str,
    seeds:         Tuple[int, ...],
    out_dir:       str = "vfe3_policy_results/sigma_gate",
) -> str:                            # the written artifact path
    r"""Write the versioned, machine-readable gate artifact (spec Section 4.5) carrying the checkpoint
    id, the spec commit hash, the seed list, and the full record with its PASS/FAIL stamp. The config
    flag ``policy_sigma_ambiguity_validated`` may be set True only with a ``policy_sigma_gate_artifact``
    reference to a PASS record whose spec commit matches (config.py Guard 4)."""
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(checkpoint_id=checkpoint_id, spec_commit=spec_commit, seeds=list(seeds), **record)
    # Slugify the FILENAME only (a checkpoint_id carrying os.sep / '..' / a drive colon must not
    # escape out_dir); the payload above keeps the RAW checkpoint_id for provenance. The slug is
    # lossy ('ckpt a' / 'ckpt:a' / 'ckpt/a' all map to 'ckpt_a'), so a stable short hash of the
    # RAW id disambiguates: distinct checkpoint_ids never overwrite each other's PASS/FAIL record
    # (mirrors the ablation.py _sanitize C15 fix; audit 2026-07-01 round-3).
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", checkpoint_id).strip("._") or "artifact"
    h    = hashlib.sha1(checkpoint_id.encode("utf-8")).hexdigest()[:8]
    path = os.path.join(out_dir, f"{slug}__{h}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def verify_gate_artifact(
    path: str,

    *,
    expected_spec_commit: Optional[str] = None,
    require_pass:         bool          = True,
) -> Dict[str, object]:
    r"""Load a sigma-gate artifact and raise ``ValueError`` unless it is a usable PASS record (spec
    Section 4.5, Guards 4/7): the file must exist and parse, carry ``status=='PASS'`` (when
    ``require_pass``), and -- when ``expected_spec_commit`` is given -- its ``spec_commit`` must match.
    This is the CONTENT check that stops a FAIL, unreadable, or stale-spec artifact from silently
    validating ``policy_sigma_ambiguity_validated``. Returns the loaded record. Pass
    ``expected_spec_commit=None`` to skip the commit match when the caller does not know the live spec
    commit (config validation); the Phase-3 consumer that unlocks the sigma arm passes the live commit."""
    if not os.path.isfile(path):
        raise ValueError(f"sigma-gate artifact {path!r} does not exist")
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"sigma-gate artifact {path!r} is unreadable JSON: {exc}")
    if require_pass and rec.get("status") != "PASS":
        raise ValueError(
            f"sigma-gate artifact {path!r} has status={rec.get('status')!r}, not 'PASS'; the gate must "
            f"PASS before policy_sigma_ambiguity_validated can be set (spec Section 4.5).")
    if expected_spec_commit is not None and rec.get("spec_commit") != expected_spec_commit:
        raise ValueError(
            f"sigma-gate artifact {path!r} spec_commit={rec.get('spec_commit')!r} does not match the "
            f"current spec commit {expected_spec_commit!r}; re-measure the gate (spec Section 4.7).")
    return rec


@torch.no_grad()
def measure_sigma_gate(
    model:         'object',         # VFEModel hosting the arm
    loader:        'object',         # held-out (tokens, targets) DataLoader

    *,
    checkpoint_id: str,
    spec_commit:   str,
    seeds:         Tuple[int, ...],
    out_dir:       str            = "vfe3_policy_results/sigma_gate",
    max_batches:   Optional[int]  = 20,
    device:        Optional[torch.device] = None,
    write:         bool           = True,
    **gate_kwargs,
) -> Dict[str, object]:
    r"""End-to-end gate run: pull aligned per-token (tr_sigma, ce, conf, correct) from
    ``belief_ce_bank`` on the held-out loader, evaluate the gate, and (by default) write the artifact.
    Returns the full record. ``write=False`` is for tests."""
    from vfe3.viz.extract import belief_ce_bank
    bank = belief_ce_bank(model, loader, device=device, max_batches=max_batches)
    record = evaluate_sigma_gate(bank["tr_sigma"], bank["ce"], bank["conf"], bank["correct"],
                                 **gate_kwargs)
    if write:
        record["artifact_path"] = write_sigma_gate_artifact(
            record, checkpoint_id=checkpoint_id, spec_commit=spec_commit, seeds=seeds, out_dir=out_dir)
    return record
