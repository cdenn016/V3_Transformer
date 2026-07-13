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
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

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
    model_behavior_sha256:      Optional[str]                  = None,
    code_identity_sha256:       Optional[str]                  = None,
    measurement_context:        Optional[Mapping[str, object]] = None,
    measurement_context_sha256: Optional[str]                  = None,
) -> str:                            # the written artifact path
    r"""Write the versioned, machine-readable gate artifact (spec Section 4.5) carrying the checkpoint
    id, the (now content-based) spec identity, the seed list, and the full record with its PASS/FAIL
    stamp. When supplied, the PB-06 provenance fields (``model_behavior_sha256``,
    ``code_identity_sha256``, ``measurement_context`` + its fingerprint) are folded into the payload so a
    consumer can bind the artifact to the exact model/source/data measured; the writer RECOMPUTES the
    context fingerprint from the stored mapping and raises on a mismatch. The writer also REFUSES to
    publish or overwrite a PASS under a resolved-FAIL preregistration (a reproduction is diagnostic-only;
    a real PASS is a separate reviewed manifest-only update)."""
    if record.get("status") == "PASS":
        try:
            prereg = load_sigma_gate_preregistry().get(spec_commit)
        except ValueError:
            prereg = None
        if prereg is not None and prereg.get("status") == "FAIL":
            raise ValueError(
                f"refusing to write a PASS sigma-gate artifact under a resolved-FAIL preregistration for "
                f"identity {spec_commit!r}; a reproduction is diagnostic-only (spec Section 4.5).")
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(checkpoint_id=checkpoint_id, spec_commit=spec_commit, seeds=list(seeds), **record)
    if measurement_context is not None:
        from vfe3.run_artifacts import semantic_config_fingerprint
        recomputed = semantic_config_fingerprint(measurement_context)
        if measurement_context_sha256 is not None and recomputed != measurement_context_sha256:
            raise ValueError(
                "sigma-gate measurement_context_sha256 does not match the stored measurement_context")
        payload["measurement_context"] = dict(measurement_context)
        payload["measurement_context_sha256"] = recomputed
    if model_behavior_sha256 is not None:
        payload["model_behavior_sha256"] = model_behavior_sha256
    if code_identity_sha256 is not None:
        payload["code_identity_sha256"] = code_identity_sha256
    # Slugify the FILENAME only (a checkpoint_id carrying os.sep / '..' / a drive colon must not
    # escape out_dir); the payload above keeps the RAW checkpoint_id for provenance. The slug is
    # lossy ('ckpt a' / 'ckpt:a' / 'ckpt/a' all map to 'ckpt_a'), so a stable short hash of the
    # RAW id disambiguates: distinct checkpoint_ids never overwrite each other's PASS/FAIL record
    # (mirrors the ablation.py _sanitize C15 fix; audit 2026-07-01 round-3).
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", checkpoint_id).strip("._") or "artifact"
    h    = hashlib.sha1(checkpoint_id.encode("utf-8")).hexdigest()[:8]
    path = os.path.join(out_dir, f"{slug}__{h}.json")
    tmp  = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=out_dir,
            prefix=os.path.basename(path) + ".", suffix=".tmp", delete=False,
        ) as f:
            tmp = f.name
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
            except OSError:
                pass
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
    model_behavior_sha256:      Optional[str]                  = None,
    code_identity_sha256:       Optional[str]                  = None,
    measurement_context:        Optional[Mapping[str, object]] = None,
    measurement_context_sha256: Optional[str]                  = None,
    **gate_kwargs,
) -> Dict[str, object]:
    r"""End-to-end gate run: pull aligned per-token (tr_sigma, ce, conf, correct) from
    ``belief_ce_bank`` on the held-out loader, evaluate the gate, and (by default) write the artifact.
    Returns the full record. ``write=False`` is for tests. The PB-06 provenance fields
    (``model_behavior_sha256``, ``code_identity_sha256``, ``measurement_context`` + its fingerprint) are
    passed straight through to :func:`write_sigma_gate_artifact`."""
    from vfe3.viz.extract import belief_ce_bank
    bank = belief_ce_bank(model, loader, device=device, max_batches=max_batches)
    record = evaluate_sigma_gate(bank["tr_sigma"], bank["ce"], bank["conf"], bank["correct"],
                                 **gate_kwargs)
    if write:
        record["artifact_path"] = write_sigma_gate_artifact(
            record, checkpoint_id=checkpoint_id, spec_commit=spec_commit, seeds=seeds, out_dir=out_dir,
            model_behavior_sha256=model_behavior_sha256, code_identity_sha256=code_identity_sha256,
            measurement_context=measurement_context,
            measurement_context_sha256=measurement_context_sha256)
    return record


# ======================================================================================
# PB-06: content-based governing identity, consumer code identity, the sealed measurement context,
# the preregistration manifest, and the strict sigma-consumer gate.
# ======================================================================================

# The governing preregistration records: the spec plus the sigma-gate pre-registration note. The
# specification IDENTITY is a hash of THESE FILES' normalized content -- never a git commit SHA -- so
# the commit that restored them is not circular with the identity that authorizes the gate.
GOVERNING_SPEC_PATHS: Tuple[str, ...] = (
    "docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md",
    "docs/research/active-inference/2026-06-28-sigma-gate-prereg.md",
)

# Sealed gate-statistic thresholds. These MUST equal the ``thresholds`` mapping ``evaluate_sigma_gate``
# stamps into a record, so a measured artifact's thresholds and its sealed context agree byte-for-byte.
SEALED_GATE_THRESHOLDS: Dict[str, object] = {
    "spearman_min": 0.2,
    "ece_max":      0.05,
    "alpha":        0.05,
    "n_strata":     10,
    "n_bins":       10,
    "n_boot":       2000,
    "n_perm":       1000,
}

# Sealed loader / sampler / statistic context (spec Sections 4.5/4.7). The consumer derives the SAME
# mapping from the live config and current corpus; any drift fails the gate closed.
SEALED_MEASUREMENT_CONTEXT: Dict[str, object] = {
    "dataset":           "wikitext-103",
    "split":             "test",
    "requested_seq_len": 128,
    "batch_size":        16,
    "max_batches":       20,
    "shuffle":           False,
    "drop_last":         True,
    "seeds":             [6, 23, 64],
    "sigma_samples":     16,
    "mc_seed":           0,
    "sampling_rule":     "antithetic_shared_v1",
    "statistic_seed":    0,
}

_PREREGISTRY_PATH = Path(__file__).resolve().parent / "sigma_gate_preregistry.json"


def _governing_root(root: 'Optional[str | Path]') -> Path:
    r"""The repository root the governing/source paths are resolved against (``vfe3/inference`` -> up 2)."""
    return Path(__file__).resolve().parents[2] if root is None else Path(root)


def _normalize_text_bytes(raw: bytes) -> bytes:
    r"""UTF-8 decode then canonicalize CRLF and lone CR to LF, so a checkout's line-ending convention
    (this host runs ``core.autocrlf=true``) never changes an identity. Raises on undecodable bytes."""
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def sigma_gate_spec_identity(
    root: 'Optional[str | Path]' = None,
) -> str:
    r"""Content-based governing identity of the sigma gate (PB-06). Sorts the governing relative paths
    lexicographically, then folds each UTF-8 path and its CRLF/CR-normalized UTF-8 content into a
    SHA-256, with an 8-byte big-endian length delimiter immediately before each path and content
    payload. It NEVER includes a git commit SHA, so the restored-doc commit is not circular; and it is
    LF/CRLF/CR-invariant. Any missing or undecodable governing file yields ``"unknown"`` (fail-closed).
    The producer and every consumer read the identity through this one helper."""
    base = _governing_root(root)
    digest = hashlib.sha256()
    for rel in sorted(GOVERNING_SPEC_PATHS):
        try:
            content = (base / rel).read_bytes()
            normalized = _normalize_text_bytes(content)
        except (OSError, UnicodeDecodeError):
            return "unknown"
        rel_bytes = rel.encode("utf-8")
        digest.update(len(rel_bytes).to_bytes(8, "big"))
        digest.update(rel_bytes)
        digest.update(len(normalized).to_bytes(8, "big"))
        digest.update(normalized)
    return digest.hexdigest()


def sigma_consumer_code_identity(
    root: 'Optional[str | Path]' = None,
) -> str:
    r"""Content identity of the executable sigma-arm sources (PB-06): every sorted ``vfe3/**/*.py`` file
    plus ``sigma_gate_measure.py``, each folded as a length-delimited normalized relative path and
    CRLF-normalized content, excluding ``__pycache__``. Generated artifacts, docs, tests, and the JSON
    preregistry are naturally excluded (only ``*.py`` under ``vfe3`` and the one measurement script are
    declared), so writing/replacing the gate JSON or updating the preregistry leaves this identity
    unchanged while editing a copied policy/model source changes it. RAISES if any declared source is
    unreadable -- an integrity error, unlike the fail-closed ``"unknown"`` of the spec identity."""
    base = _governing_root(root)
    declared = [p for p in (base / "vfe3").rglob("*.py") if "__pycache__" not in p.parts]
    declared.append(base / "sigma_gate_measure.py")
    entries = sorted((p.relative_to(base).as_posix(), p) for p in declared)
    digest = hashlib.sha256()
    for rel, path in entries:
        try:
            normalized = _normalize_text_bytes(path.read_bytes())
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"sigma consumer code identity cannot read declared source {rel!r}: {exc}")
        rel_bytes = rel.encode("utf-8")
        digest.update(len(rel_bytes).to_bytes(8, "big"))
        digest.update(rel_bytes)
        digest.update(len(normalized).to_bytes(8, "big"))
        digest.update(normalized)
    return digest.hexdigest()


def sigma_measurement_context(
    cfg: 'object',

    *,
    cache_dir: 'Optional[Path]' = None,
) -> Dict[str, object]:
    r"""Return the sealed loader/statistic/sampler context plus the current corpus identity (PB-06).

    Starts from :data:`SEALED_MEASUREMENT_CONTEXT`, adds the effective ``min(128, cfg.max_seq_len)``,
    the tokenizer tag, the sealed ``thresholds``, and ``cache_source_identity(dataset, split,
    cache_dir)`` (the artifact-integrity plan's byte-exact corpus binding). A missing or changed corpus
    raises (fail-closed) so the consumer cannot silently measure against different data."""
    from vfe3.data.datasets import _tokenizer_tag, cache_source_identity
    dataset = SEALED_MEASUREMENT_CONTEXT["dataset"]
    split = SEALED_MEASUREMENT_CONTEXT["split"]
    max_seq_len = cfg["max_seq_len"] if isinstance(cfg, Mapping) else cfg.max_seq_len
    context = dict(SEALED_MEASUREMENT_CONTEXT)
    context["effective_seq_len"] = min(int(context["requested_seq_len"]), int(max_seq_len))
    context["tokenizer_tag"] = _tokenizer_tag(dataset)
    context["thresholds"] = dict(SEALED_GATE_THRESHOLDS)
    context["cache_source_identity"] = cache_source_identity(dataset, split, cache_dir=cache_dir)
    return context


def canonical_json_sha256(
    path: 'str | Path',
) -> str:
    r"""SHA-256 of an artifact's CANONICAL JSON (parse then
    ``json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``), so LF/CRLF,
    indentation, and key order cannot change identity. Unreadable/undecodable JSON raises ``ValueError``."""
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"sigma-gate artifact {str(path)!r} is unreadable JSON: {exc}")
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_sigma_gate_preregistry(
    path: 'Optional[str | Path]' = None,
) -> Dict[str, object]:
    r"""Load the tracked, non-code authorization manifest (fail-closed) as a mapping keyed by the exact
    content-based governing identity. Production resolves this through the module at call time, so a test
    may substitute a temporary manifest without an unpatchable alias."""
    target = _PREREGISTRY_PATH if path is None else Path(path)
    try:
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"sigma-gate preregistry {str(target)!r} is unreadable: {exc}")
    if not isinstance(data, dict):
        raise ValueError("sigma-gate preregistry must be a JSON object keyed by governing identity")
    return data


def _computed_gate_pass(
    record:     Mapping[str, object],
    thresholds: Mapping[str, object],
) -> bool:
    r"""Recompute the sealed PASS predicate from a record's stored statistics + thresholds."""
    return bool(
        float(record["sigma_ce_spearman"]) >= float(thresholds["spearman_min"])
        and float(record["spearman_ci"][0]) > 0.0
        and float(record["spearman_ci"][0]) > float(record["permutation_floor"])
        and record["stratified_ce"]["monotone"] is True
        and float(record["sigma_binned_ece"]) < float(thresholds["ece_max"])
    )


def verify_sigma_prereg_gate(
    path: str,

    *,
    actual_spec_identity: str,
) -> Dict[str, object]:
    r"""Prereg-aware artifact verifier for VFE3Config construction (PB-06): reject unregistered /
    resolved-FAIL identities, artifact byte-hash mismatch, stale spec, and a status/statistic
    contradiction. It does NOT verify the not-yet-constructed live model or current corpus -- those live
    checks are the consumer boundary (:func:`verify_sigma_consumer_gate`)."""
    prereg = load_sigma_gate_preregistry().get(actual_spec_identity)
    if prereg is None or prereg.get("status") != "PASS":
        raise ValueError("sigma-gate governing identity is not registered as PASS")
    if canonical_json_sha256(Path(path)) != prereg.get("artifact_sha256"):
        raise ValueError("sigma-gate artifact bytes do not match the preregistration registry")
    record = verify_gate_artifact(path, expected_spec_commit=actual_spec_identity, require_pass=True)
    thresholds = record.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("sigma-gate artifact carries no thresholds mapping")
    if record.get("status") != ("PASS" if _computed_gate_pass(record, thresholds) else "FAIL"):
        raise ValueError("sigma-gate status contradicts its stored statistics")
    return record


def verify_sigma_consumer_gate(
    path: str,

    *,
    actual_model_behavior_sha256:      str,
    actual_spec_identity:              str,
    actual_code_identity_sha256:       str,
    actual_measurement_context_sha256: str,
) -> Dict[str, object]:
    r"""Strict consumer-boundary gate over DERIVED live identities (PB-06). Rejects an unregistered or
    resolved-FAIL governing identity before reading PASS; a canonical byte-hash that does not match the
    manifest; a stale spec / non-PASS record; a model-behavior, code, or sealed-context fingerprint that
    does not match the LIVE model/source/data; a duplicated seed/threshold provenance contradicting the
    sealed context; and a stored status that contradicts the recomputed statistics. Returns the record."""
    from vfe3.run_artifacts import semantic_config_fingerprint
    prereg = load_sigma_gate_preregistry().get(actual_spec_identity)
    if prereg is None or prereg.get("status") != "PASS":
        raise ValueError("sigma-gate governing identity is not registered as PASS")
    if canonical_json_sha256(Path(path)) != prereg.get("artifact_sha256"):
        raise ValueError("sigma-gate artifact bytes do not match the preregistration registry")
    record = verify_gate_artifact(
        path,
        expected_spec_commit=actual_spec_identity,
        require_pass=True,
    )
    if record.get("model_behavior_sha256") != actual_model_behavior_sha256:
        raise ValueError(
            "sigma-gate model-behavior fingerprint does not match the live model"
        )
    if record.get("code_identity_sha256") != actual_code_identity_sha256:
        raise ValueError("sigma-gate code identity does not match the live source")
    context = record.get("measurement_context")
    if (not isinstance(context, Mapping)
            or semantic_config_fingerprint(context) != record.get("measurement_context_sha256")
            or record.get("measurement_context_sha256") != actual_measurement_context_sha256):
        raise ValueError("sigma-gate measurement context does not match sealed live data/statistics")
    if record.get("seeds") != context.get("seeds"):
        raise ValueError("sigma-gate duplicated seed provenance contradicts its sealed context")
    if record.get("thresholds") != context.get("thresholds"):
        raise ValueError("sigma-gate measured thresholds contradict its sealed context")
    thresholds = context["thresholds"]
    computed_pass = _computed_gate_pass(record, thresholds)
    if record.get("status") != ("PASS" if computed_pass else "FAIL"):
        raise ValueError("sigma-gate status contradicts its stored statistics")
    return record
