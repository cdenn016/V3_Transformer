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
import os

import torch

from vfe3.config import VFE3Config
from vfe3.data.datasets import make_dataloader
from vfe3.inference.sigma_gate import (
    SEALED_MEASUREMENT_CONTEXT,
    measure_sigma_gate,
    sigma_consumer_code_identity,
    sigma_gate_spec_identity,
    sigma_measurement_context,
)
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import (
    model_behavior_fingerprint,
    semantic_config_fingerprint,
    sigma_behavior_config,
)

SPEC = "docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md"


def _verify_sealed_config(cfg):
    r"""Reject any edited CONFIG value that differs from the sealed measurement context (PB-06), so the
    producer cannot silently measure another dataset/split/loader than the pre-registered one."""
    sealed = {
        "dataset":     SEALED_MEASUREMENT_CONTEXT["dataset"],
        "split":       SEALED_MEASUREMENT_CONTEXT["split"],
        "seq_len":     SEALED_MEASUREMENT_CONTEXT["requested_seq_len"],
        "batch_size":  SEALED_MEASUREMENT_CONTEXT["batch_size"],
        "max_batches": SEALED_MEASUREMENT_CONTEXT["max_batches"],
        "seeds":       tuple(SEALED_MEASUREMENT_CONTEXT["seeds"]),
    }
    for key, want in sealed.items():
        got = tuple(cfg[key]) if key == "seeds" else cfg[key]
        if got != want:
            raise ValueError(
                f"sigma_gate_measure CONFIG[{key!r}]={got!r} differs from the sealed measurement context "
                f"{want!r}; the gate must measure the pre-registered context (spec Sections 4.5/4.7).")

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
    _verify_sealed_config(cfg)
    spec = sigma_gate_spec_identity()
    if spec == "unknown":
        raise ValueError(
            "the governing specification identity is 'unknown' (missing/undecodable governing docs); "
            "restore them before measuring the gate (spec Sections 4.5/4.7).")
    print(f"device={device}  checkpoint={cfg['checkpoint']}  spec_identity={spec[:12]}")
    if device == "cpu":
        print("WARNING: running on CPU; the iterative E-step belief replay is slow. Run on the GPU.")

    model, mcfg = load_model_from_checkpoint(cfg["checkpoint"], device)
    print(f"  loaded arch: embed_dim={mcfg.embed_dim} n_layers={mcfg.n_layers} "
          f"use_prior_bank={mcfg.use_prior_bank} use_head_mixer={mcfg.use_head_mixer} "
          f"max_seq_len={mcfg.max_seq_len} vocab={mcfg.vocab_size}")
    seq_len = min(cfg["seq_len"], mcfg.max_seq_len)
    loader = make_dataloader(cfg["dataset"], cfg["split"], seq_len, cfg["batch_size"],
                             shuffle=False, drop_last=True, vocab_size=mcfg.vocab_size)

    # PB-06 provenance the consumer gate binds to: the exact non-policy behavior fingerprint, the
    # declared-source code identity, and the sealed loader/statistic context + its fingerprint.
    behavior = model_behavior_fingerprint(sigma_behavior_config(mcfg), model.state_dict())
    code_identity = sigma_consumer_code_identity()
    meas_context = sigma_measurement_context(mcfg)
    context_fp = semantic_config_fingerprint(meas_context)

    record = measure_sigma_gate(
        model, loader, checkpoint_id=cfg["checkpoint_id"], spec_commit=spec,
        seeds=cfg["seeds"], out_dir=cfg["out_dir"], max_batches=cfg["max_batches"], device=device,
        model_behavior_sha256=behavior, code_identity_sha256=code_identity,
        measurement_context=meas_context, measurement_context_sha256=context_fp)

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
