r"""Click-to-run: measure the sigma-validation gate at the operating-point checkpoint and write the
PASS/FAIL artifact (spec docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md
Section 4.5; pre-registration docs/research/active-inference/2026-06-28-sigma-gate-prereg.md).

The gate is the BINDING precondition for any sigma-derived epistemic/ambiguity arm in Phase 3 (sigma_mc,
epistemic-only, shuffled-sigma). It tests whether the belief covariance tr(Sigma_q) demonstrably
predicts realized cross-entropy on a held-out split (Spearman + bootstrap CI vs a permutation floor,
sigma-stratified-CE monotonicity, sigma-binned ECE). PASS unlocks those arms; FAIL keeps them
reported-only and the information-gain term inert.

CLICK-TO-RUN: point CONFIG['checkpoint'] at an operating-point checkpoint, set the model arch to MATCH
that checkpoint, then run. Compute is a forward-pass eval over a few held-out batches; run on the GPU.
The artifact is written to vfe3_policy_results/sigma_gate/<checkpoint_id>.json with the spec commit hash,
so the policy_sigma_ambiguity_validated config flag can be bound to a matching PASS record (Guard 4).
"""
import dataclasses
import hashlib
import os
import subprocess

import torch

from vfe3.config import VFE3Config
from vfe3.data.datasets import make_dataloader
from vfe3.inference.sigma_gate import measure_sigma_gate
from vfe3.model.model import VFEModel

SPEC = "docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md"

CONFIG = dict(
    checkpoint="",                       # REQUIRED: path to the operating-point checkpoint (.pt)
    checkpoint_id="wikitext103_ed20_15k",  # artifact filename stem (the hosting checkpoint's id)
    seeds=(6, 23, 64),                   # sealed seed list (recorded in the artifact)
    dataset="wikitext-103",
    split="test",
    seq_len=128,                         # match the train length
    batch_size=16,
    max_batches=20,                      # held-out batches to join (sigma <-> CE)
    out_dir="vfe3_policy_results/sigma_gate",
    # The model arch is read from the checkpoint's stored config (save_checkpoint saves asdict(cfg)),
    # so any saved checkpoint loads as-is with no manual arch entry to get wrong.
)


def spec_commit():
    r"""Provenance stamp for the governing spec (audit F2, 2026-06-28). Returns the spec's last commit
    hash when the spec is tracked and clean; otherwise a content-bound stamp, so a gate artifact can
    never look commit-bound while the governing spec is actually untracked or dirty (the old code fell
    back to `git rev-parse HEAD`, binding the artifact to the code revision rather than the spec text).
    Formats:
      <commit>               tracked and clean
      <commit>+dirty:<sha12> tracked with uncommitted edits
      untracked:<sha12>      not in git
      unknown                git or the spec file is unavailable
    where <sha12> is the first 12 hex digits of sha256(spec bytes)."""
    def _git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return ""
    try:
        with open(SPEC, "rb") as f:
            sha12 = hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return "unknown"
    commit = _git("log", "-1", "--format=%H", "--", SPEC)
    if not commit:
        return f"untracked:{sha12}"
    dirty = bool(_git("status", "--porcelain", "--", SPEC))
    return f"{commit}+dirty:{sha12}" if dirty else commit


def load_model_from_checkpoint(path, device):
    r"""Rebuild the exact model from a checkpoint's stored config (save_checkpoint stores asdict(cfg)),
    so the arch always matches whatever checkpoint is given -- no manual arch entry to get wrong.
    Tolerates a missing .pt extension. weights_only=True is safe (the bundle is tensors + a dict config)."""
    p = path if os.path.isfile(path) else path + ".pt"
    ckpt = torch.load(p, map_location=device, weights_only=True)
    saved = ckpt.get("config")
    if saved is None:
        raise ValueError(f"checkpoint {p!r} has no stored config; cannot rebuild the model arch")
    fields = {f.name for f in dataclasses.fields(VFE3Config)}
    cfg = VFE3Config(**{k: v for k, v in saved.items() if k in fields})
    model = VFEModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def main():
    cfg = CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not cfg["checkpoint"]:
        raise ValueError("set CONFIG['checkpoint'] to an operating-point checkpoint path before running")
    print(f"device={device}  checkpoint={cfg['checkpoint']}  spec_commit={spec_commit()[:12]}")
    if device == "cpu":
        print("WARNING: running on CPU; the iterative E-step belief replay is slow. Run on the GPU.")

    model, mcfg = load_model_from_checkpoint(cfg["checkpoint"], device)
    print(f"  loaded arch: embed_dim={mcfg.embed_dim} n_layers={mcfg.n_layers} "
          f"use_prior_bank={mcfg.use_prior_bank} use_head_mixer={mcfg.use_head_mixer} "
          f"max_seq_len={mcfg.max_seq_len} vocab={mcfg.vocab_size}")
    seq_len = min(cfg["seq_len"], mcfg.max_seq_len)
    loader = make_dataloader(cfg["dataset"], cfg["split"], seq_len, cfg["batch_size"],
                             shuffle=False, drop_last=True)

    record = measure_sigma_gate(
        model, loader, checkpoint_id=cfg["checkpoint_id"], spec_commit=spec_commit(),
        seeds=cfg["seeds"], out_dir=cfg["out_dir"], max_batches=cfg["max_batches"], device=device)

    print(f"\n=== sigma-validation gate: {record['status']} ===")
    print(f"  n_tokens            {record['n_tokens']}")
    print(f"  sigma_ce_spearman   {record['sigma_ce_spearman']:.4f}  "
          f"CI={[round(x, 4) for x in record['spearman_ci']]}  (>= 0.20, CI_lo > 0 and floor)")
    print(f"  permutation_floor   {record['permutation_floor']:.4f}")
    print(f"  sigma_trace_cv      {record['sigma_trace_cv']:.4f}")
    print(f"  stratified monotone {record['stratified_ce']['monotone']}  "
          f"(rank trend {record['stratified_ce']['mono_spearman']:.3f})")
    print(f"  sigma_binned_ece    {record['sigma_binned_ece']:.4f}  (< 0.05)")
    print(f"  artifact            {record.get('artifact_path')}")
    if record["status"] == "PASS":
        print("\nPASS: sigma is a validated epistemic signal on this checkpoint. The Phase 3 epistemic "
              "arms (sigma_mc, epistemic-only) may be CLAIMED; set policy_sigma_ambiguity_validated=True "
              "with policy_sigma_gate_artifact pointing at this record.")
    else:
        print("\nFAIL: sigma is NOT a validated epistemic signal on this checkpoint. The information-gain "
              "term stays inert and all sigma-derived arms remain reported-only (spec Section 4.5).")


if __name__ == "__main__":
    main()
