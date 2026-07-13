r"""Pre-registration-invariant tests for the EFE ring experiment script (efe_ring_experiment.py;
spec Section 4.1). Pins sampler CORRECTNESS -- uniform goal != s0 -- which the audit 2026-06-28 found
unpinned (a green suite survived a biased sampler, finding F2/F6).

Config TOGGLES (steps, seeds, batch_size, ...) are intentionally NOT pinned here: they are the user's
live pre-registration surface, edited between runs, and any deviation is already logged in the run
output. Only behavioral contracts of the harness functions are pinned.

The per-seed durable-bundle / resume tests (audit PB-04) pin the state machine: a trained bundle
skips training, a complete bundle skips training AND evaluation, a stale (code/config-drifted) or
malformed bundle fails closed and recomputes, and the aggregate is published atomically. Every model
here is a CPU-tiny embed_dim=4 checkpoint and the expensive training / arm-matrix are monkeypatched
with counters, so the whole file stays well under the CPU test budget.
"""
import json
import os

import pytest
import torch

import efe_ring_experiment as exp
from vfe3.inference import ring_task as rt


def test_sample_episodes_excludes_start_and_is_uniform():
    n = 200_000
    goals, s0 = exp.sample_episodes(n, seed=0, device=torch.device("cpu"))
    assert goals.shape == (n,) and s0.shape == (n,)
    assert bool((goals != s0).all())                          # g != s0 always (spec Section 4.1)
    assert int(goals.min()) >= 0 and int(goals.max()) < rt.M  # on the ring
    # the ring offset (goal - s0) mod M must be uniform over the M-1 nonzero values, none ~2x another.
    offset = (goals - s0) % rt.M
    counts = torch.bincount(offset, minlength=rt.M).float()
    assert float(counts[0]) == 0.0                            # never the zero offset
    expected = n / (rt.M - 1)
    # pre-fix the clockwise neighbor (offset 1) carried ~2x the mass; require all within 10% of uniform.
    assert float((counts[1:] - expected).abs().max()) < 0.1 * expected


def test_sample_episodes_respects_device():
    goals, s0 = exp.sample_episodes(8, seed=1, device=torch.device("cpu"))
    assert goals.device.type == "cpu" and s0.device.type == "cpu"
    assert goals.shape == (8,) and s0.shape == (8,)


def test_bh_fdr_step_up_control():
    # Phase 2 multiplicity control (spec 4.6): Benjamini-Hochberg over the arm grid.
    all_sig = exp.bh_fdr({"a": 0.001, "b": 0.002, "c": 0.003}, q=0.05)
    assert all(sig for _, sig in all_sig.values())            # tiny p-values -> all rejected
    none_sig = exp.bh_fdr({"a": 0.9, "b": 0.8, "c": 0.95}, q=0.05)
    assert not any(sig for _, sig in none_sig.values())       # large p-values -> none rejected
    # step-up: p=[0.01, 0.04, 0.5], m=3, q=0.05 -> only the largest passing rank k=1 ('a') rejected
    mixed = exp.bh_fdr({"a": 0.01, "b": 0.04, "c": 0.5}, q=0.05)
    assert mixed["a"][1] is True and mixed["b"][1] is False and mixed["c"][1] is False
    # step-up property: a later rank passing lifts the earlier ones too
    lifted = exp.bh_fdr({"a": 0.02, "b": 0.02, "c": 0.02}, q=0.05)
    assert all(sig for _, sig in lifted.values())             # k=3 passes (0.02 <= 0.05) -> all rejected


# ---- per-seed durable bundle + resume state machine (audit PB-04) -------------------------------

EXPECTED_ARMS = (
    "full_efe_tuned", "full_efe_g1", "risk_only", "ambiguity_only",
    "flat_pref", "p_data_control", "temp_tuned_logprob", "logprob_baseline",
    "nucleus", "typical", "greedy_ref", "random",
)


def _tiny_config():
    # CPU-tiny (embed_dim=4, one head-block, one layer, one E-step): builds in well under a second.
    return rt.VFE3Config(
        vocab_size=rt.V, embed_dim=4, n_heads=2, max_seq_len=rt.SEQ_LEN,
        n_layers=1, n_e_steps=1, use_prior_bank=False, use_head_mixer=False,
    )


def _tiny_model():
    return rt.VFEModel(_tiny_config())


def _test_config(tmp_path, **overrides):
    cfg = dict(exp.CONFIG)
    cfg.update(seeds=(0,), steps=1, out_dir=str(tmp_path), resume=True)
    cfg.update(overrides)
    return cfg


def _run_checkpoint_output():
    # The (monkeypatched) arm-matrix output: the fields run_checkpoint contributes to a seed entry.
    return {
        "gamma":       1.0,
        "temp":        1.0,
        "dev_success": 0.5,
        "metrics":     {arm: {"success": 0.5, "mean_steps_to_goal": 1.0, "frac_at_goal": 0.5,
                              "mean_risk": 0.0, "mean_ambiguity": 0.0} for arm in EXPECTED_ARMS},
        "gates":       {"go": True, "closed_loop_causal": True},   # mirrors run_checkpoint's gate keys
    }


def _complete_result(adequacy=0.99):
    # A full 'complete' seed entry = {adequacy, admitted} + the arm-matrix output.
    entry = {"adequacy": adequacy, "admitted": True}
    entry.update(_run_checkpoint_output())
    return entry


def _fake_train(counter, adequacy=0.99):
    def _train(*, seed, steps, batch_size, lr, log_every, device, **kw):
        counter[0] += 1
        return _tiny_model().to(device), adequacy
    return _train


def _fake_run_checkpoint(counter):
    def _run(model, cfg, device, seed):
        counter[0] += 1
        return _run_checkpoint_output()
    return _run


def test_seed_bundle_round_trip_restores_exact_weights(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model()
    semantic_cfg = exp._semantic_experiment_config(_test_config(tmp_path))
    path = tmp_path / "seed_0.pt"
    exp._save_seed_bundle(path, model, semantic_cfg, None,
                          seed=0, adequacy=0.99, status="trained")
    loaded = exp._load_seed_bundle_if_current(path, semantic_cfg, torch.device("cpu"), seed=0)
    assert loaded is not None
    restored, saved = loaded
    assert saved["status"] == "trained" and saved["result"] is None
    assert not restored.training                              # rebuilt in eval mode
    orig, new = model.state_dict(), restored.state_dict()
    assert set(orig) == set(new)
    for key in orig:
        assert torch.equal(orig[key], new[key])              # byte-exact weight restore


def test_trained_seed_bundle_skips_training_but_runs_evaluation(tmp_path, monkeypatch):
    cfg = _test_config(tmp_path)
    semantic_cfg = exp._semantic_experiment_config(cfg)
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    exp._save_seed_bundle(seed_dir / "seed_0.pt", _tiny_model(), semantic_cfg, None,
                          seed=0, adequacy=0.99, status="trained")
    train_counter, eval_counter = [0], [0]
    monkeypatch.setattr(exp, "CONFIG", cfg)
    monkeypatch.setattr(rt, "train_ring_checkpoint", _fake_train(train_counter))
    monkeypatch.setattr(exp, "run_checkpoint", _fake_run_checkpoint(eval_counter))
    exp.main()
    assert train_counter[0] == 0                              # training skipped (resumed 'trained')
    assert eval_counter[0] == 1                               # but evaluation ran
    bundle = torch.load(seed_dir / "seed_0.pt", weights_only=True)
    assert bundle["status"] == "complete"                    # promoted to 'complete'


def test_complete_seed_bundle_skips_training_and_evaluation(tmp_path, monkeypatch):
    cfg = _test_config(tmp_path)
    semantic_cfg = exp._semantic_experiment_config(cfg)
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    exp._save_seed_bundle(seed_dir / "seed_0.pt", _tiny_model(), semantic_cfg, _complete_result(),
                          seed=0, adequacy=0.99, status="complete")
    train_counter, eval_counter = [0], [0]
    monkeypatch.setattr(exp, "CONFIG", cfg)
    monkeypatch.setattr(rt, "train_ring_checkpoint", _fake_train(train_counter))
    monkeypatch.setattr(exp, "run_checkpoint", _fake_run_checkpoint(eval_counter))
    exp.main()
    assert train_counter[0] == 0 and eval_counter[0] == 0    # nothing recomputed
    results = json.loads((tmp_path / "ring_v1_results.json").read_text())
    assert results["checkpoints"]["0"]["admitted"] is True   # entry restored from the bundle


def test_seed_bundle_rejects_code_or_experiment_drift(tmp_path, monkeypatch):
    cfg = _test_config(tmp_path)
    semantic_cfg = exp._semantic_experiment_config(cfg)
    path = tmp_path / "seed_0.pt"
    torch.manual_seed(0)
    exp._save_seed_bundle(path, _tiny_model(), semantic_cfg, None,
                          seed=0, adequacy=0.99, status="trained")
    dev = torch.device("cpu")
    assert exp._load_seed_bundle_if_current(path, semantic_cfg, dev, seed=0) is not None
    # experiment drift: a changed semantic field invalidates the bundle
    drifted = dict(semantic_cfg, steps=int(semantic_cfg["steps"]) + 1)
    assert exp._load_seed_bundle_if_current(path, drifted, dev, seed=0) is None
    # code drift: a changed executable code identity invalidates it
    monkeypatch.setattr(exp, "_efe_ring_code_identity", lambda root=None: "drifted-code-identity")
    assert exp._load_seed_bundle_if_current(path, semantic_cfg, dev, seed=0) is None


def _corrupt_trained_nan_adequacy(b):
    b["adequacy"] = float("nan")


def _corrupt_trained_bool_adequacy(b):
    b["adequacy"] = True


def _corrupt_missing_admitted(b):
    del b["result"]["admitted"]


def _corrupt_nonbool_admitted(b):
    b["result"]["admitted"] = 1


def _corrupt_missing_arm(b):
    del b["result"]["metrics"]["random"]


def _corrupt_extra_arm(b):
    b["result"]["metrics"]["bogus_arm"] = {"success": 0.5}


def _corrupt_missing_gates(b):
    del b["result"]["gates"]


def _corrupt_nonbool_go(b):
    b["result"]["gates"]["go"] = 1


def _corrupt_nonfinite_success(b):
    b["result"]["metrics"]["random"]["success"] = float("inf")


def _corrupt_adequacy_mismatch(b):
    b["result"]["adequacy"] = 0.5                            # top-level stays 0.99


MALFORMED_CASES = [
    ("trained",  None,               _corrupt_trained_nan_adequacy),
    ("trained",  None,               _corrupt_trained_bool_adequacy),
    ("complete", _complete_result,   _corrupt_missing_admitted),
    ("complete", _complete_result,   _corrupt_nonbool_admitted),
    ("complete", _complete_result,   _corrupt_missing_arm),
    ("complete", _complete_result,   _corrupt_extra_arm),
    ("complete", _complete_result,   _corrupt_missing_gates),
    ("complete", _complete_result,   _corrupt_nonbool_go),
    ("complete", _complete_result,   _corrupt_nonfinite_success),
    ("complete", _complete_result,   _corrupt_adequacy_mismatch),
]


@pytest.mark.parametrize("status,result_factory,mutate", MALFORMED_CASES,
                         ids=[m.__name__ for _, _, m in MALFORMED_CASES])
def test_malformed_seed_bundle_retrains(tmp_path, monkeypatch, status, result_factory, mutate):
    cfg = _test_config(tmp_path)
    semantic_cfg = exp._semantic_experiment_config(cfg)
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / "seed_0.pt"
    torch.manual_seed(0)
    result = result_factory() if result_factory is not None else None
    exp._save_seed_bundle(path, _tiny_model(), semantic_cfg, result,
                          seed=0, adequacy=0.99, status=status)
    bundle = torch.load(path, weights_only=True)
    mutate(bundle)
    torch.save(bundle, path)
    # the loader fails closed on the malformed bundle
    assert exp._load_seed_bundle_if_current(path, semantic_cfg, torch.device("cpu"), seed=0) is None
    # and the state machine recomputes the seed exactly once (train + evaluate)
    train_counter, eval_counter = [0], [0]
    monkeypatch.setattr(exp, "CONFIG", cfg)
    monkeypatch.setattr(rt, "train_ring_checkpoint", _fake_train(train_counter))
    monkeypatch.setattr(exp, "run_checkpoint", _fake_run_checkpoint(eval_counter))
    exp.main()
    assert train_counter[0] == 1 and eval_counter[0] == 1


def test_aggregate_result_is_atomically_replaced(tmp_path, monkeypatch):
    cfg = _test_config(tmp_path, seeds=(0, 1))
    calls = []
    real_replace = os.replace

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    train_counter, eval_counter = [0], [0]
    monkeypatch.setattr(exp, "CONFIG", cfg)
    monkeypatch.setattr(rt, "train_ring_checkpoint", _fake_train(train_counter))
    monkeypatch.setattr(exp, "run_checkpoint", _fake_run_checkpoint(eval_counter))
    monkeypatch.setattr(exp.os, "replace", spy)
    exp.main()
    out_path = tmp_path / "ring_v1_results.json"
    assert out_path.is_file()
    results = json.loads(out_path.read_text())
    assert set(results["checkpoints"]) == {"0", "1"}         # every requested seed has an entry
    aggregate_calls = [(s, d) for s, d in calls if d.endswith("ring_v1_results.json")]
    assert aggregate_calls, "aggregate JSON must be published through os.replace"
    assert all(s.endswith(".tmp") for s, d in aggregate_calls)   # same-dir tmp + atomic rename


def test_efe_ring_code_identity_ignores_seed_and_aggregate_publication_but_changes_with_source(tmp_path):
    root = tmp_path / "tree"
    (root / "vfe3" / "sub").mkdir(parents=True)
    (root / "efe_ring_experiment.py").write_bytes(b"# entry point\nX = 1\n")
    (root / "vfe3" / "__init__.py").write_bytes(b"")
    (root / "vfe3" / "mod.py").write_bytes(b"A = 2\n")
    (root / "vfe3" / "sub" / "inner.py").write_bytes(b"B = 3\n")
    id1 = exp._efe_ring_code_identity(root=root)
    # publish seed bundles + aggregate results + a __pycache__ artifact: identity must NOT move
    seeds = root / "vfe3_policy_results" / "ring_v1" / "seeds"
    seeds.mkdir(parents=True)
    (seeds / "seed_0.pt").write_bytes(b"\x00binary-bundle")
    (root / "vfe3_policy_results" / "ring_v1" / "ring_v1_results.json").write_bytes(b"{}")
    (root / "vfe3" / "__pycache__").mkdir()
    (root / "vfe3" / "__pycache__" / "cached.py").write_bytes(b"cached = 9\n")
    assert exp._efe_ring_code_identity(root=root) == id1
    # editing an executable source file DOES move the identity
    (root / "efe_ring_experiment.py").write_bytes(b"# entry point\nX = 2\n")
    assert exp._efe_ring_code_identity(root=root) != id1
