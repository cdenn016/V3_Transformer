r"""Checkpoint RESUME (load side, PL8): a killed run continues from a saved checkpoint and
yields the SAME weights as an uninterrupted run.

The SAVE side (``run_artifacts.save_checkpoint``) predates this and was write-only; these
tests pin the load half: ``run_artifacts.load_checkpoint`` restores model + optimizer + RNG
+ step, and ``train(resume_from=...)`` rebuilds the per-group cosine ``LambdaLR`` at the
saved step so the continuation is numerically equivalent to a straight run.

Determinism is forced by a CONSTANT token stream: every ``TokenWindows`` window is identical,
so every batch is identical regardless of the loader's shuffle/iterator position. Both the
straight run and the resumed run therefore see the same data at every step, and the only
thing that can make their final weights differ is a missing restore leg -- model state,
optimizer momentum (exp_avg/exp_avg_sq), or the scheduler's ``last_epoch``.
"""

import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.ema import EMA
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import (
    RunArtifacts,
    finalize_run,
    finalize_validation_run,
    load_checkpoint,
    semantic_config_fingerprint,
)
from vfe3.train import TrainingTerminalState, build_optimizer, train


def _const_loader(seq_len: int = 8, bs: int = 4) -> DataLoader:
    # CONSTANT stream: every window is identical, so every batch is identical regardless of
    # loader position -> the straight run and the resumed run see the same data at every step.
    base = torch.full((seq_len * 6,), 1, dtype=torch.long)
    ds = TokenWindows(base, seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True)


def _cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return VFE3Config(**base)


def _params(model: torch.nn.Module):
    return [p.detach().clone() for p in model.parameters()]


def test_config_resume_from_default_none_and_validated():
    assert VFE3Config().resume_from is None                     # off by default (pure path)
    assert VFE3Config(resume_from="ckpt.pt").resume_from == "ckpt.pt"
    with pytest.raises(ValueError):
        VFE3Config(resume_from=123)                             # not a str/path


def test_load_checkpoint_restores_model_and_returns_step(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    with torch.no_grad():                                       # make the saved state differ from a fresh init
        model.prior_bank.mu_embed.add_(0.5)
    saved = _params(model)
    art.save_checkpoint(4, model, opt, cfg)

    fresh = VFEModel(cfg)                                       # a different random init
    assert not torch.equal(fresh.prior_bank.mu_embed, model.prior_bank.mu_embed)
    step = load_checkpoint(tmp_path / "r" / "checkpoints" / "step_4.pt", fresh)
    assert step == 4
    for restored, original in zip(_params(fresh), saved):
        assert torch.equal(restored, original)                 # exact model-state restore


def test_load_checkpoint_restores_optimizer_state(tmp_path):
    # Drive a real train() run so the checkpoint is written from the actual internal optimizer's
    # populated AdamW state (exp_avg/exp_avg_sq/step), then reload it into a fresh optimizer.
    cfg = _cfg(checkpoint_interval=3)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    train(model, _const_loader(), cfg, n_steps=3, artifacts=art)
    ckpt = tmp_path / "r" / "checkpoints" / "step_3.pt"
    assert ckpt.exists()

    fresh = VFEModel(cfg)
    fresh_opt = build_optimizer(fresh, cfg)
    assert len(fresh_opt.state) == 0                            # fresh optimizer has no momentum yet
    load_checkpoint(ckpt, fresh, fresh_opt)
    assert len(fresh_opt.state) > 0                             # AdamW momentum buffers restored from the run
    # the restored 'step' counter matches the number of completed optimizer steps
    any_state = next(iter(fresh_opt.state.values()))
    assert int(any_state["step"]) == 3


def test_load_checkpoint_restamps_current_optimizer_group_metadata(tmp_path):
    saved_cfg = _cfg(m_p_mu_lr=0.03, m_p_sigma_lr=0.02, m_phi_lr=0.01,
                     weight_decay=0.2)
    saved_model = VFEModel(saved_cfg)
    saved_opt = build_optimizer(saved_model, saved_cfg)
    art = RunArtifacts(tmp_path / "saved", saved_cfg, saved_model)
    ckpt = art.save_checkpoint(1, saved_model, saved_opt, saved_cfg)

    current_cfg = _cfg(m_p_mu_lr=0.003, m_p_sigma_lr=0.002, m_phi_lr=0.001,
                       weight_decay=0.0)
    current_model = VFEModel(current_cfg)
    current_opt = build_optimizer(current_model, current_cfg)
    current_metadata = [{k: v for k, v in group.items() if k != "params"}
                        for group in current_opt.param_groups]
    current_params = [list(group["params"]) for group in current_opt.param_groups]

    load_checkpoint(ckpt, current_model, current_opt)

    for group, metadata, params in zip(current_opt.param_groups, current_metadata, current_params):
        assert {k: v for k, v in group.items() if k != "params"} == metadata
        assert len(group["params"]) == len(params)
        assert all(loaded is current for loaded, current in zip(group["params"], params))


def test_missing_checkpoint_preserves_file_not_found(tmp_path):
    model = VFEModel(_cfg())
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model)


def test_resume_matches_uninterrupted_run(tmp_path):
    r"""The end-to-end equivalence: straight 4-step run == (2 steps -> checkpoint -> resume to 4).

    Pins all three restore legs at once -- if model state, optimizer momentum, OR the LR
    schedule's last_epoch is not restored, the continuation diverges and this fails."""
    cfg = _cfg(checkpoint_interval=2)

    torch.manual_seed(0)                                        # Run A: straight through
    model_a = VFEModel(cfg)
    init_a = _params(model_a)
    train(model_a, _const_loader(), cfg, n_steps=4)
    final_a = _params(model_a)
    assert any(not torch.equal(i, f) for i, f in zip(init_a, final_a))   # actually trained (non-vacuous)

    torch.manual_seed(0)                                        # Run B: train 2, checkpoint, resume to 4
    model_b = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model_b)
    train(model_b, _const_loader(), cfg, n_steps=2, artifacts=art)
    ckpt = tmp_path / "run" / "checkpoints" / "step_2.pt"
    assert ckpt.exists()

    model_c = VFEModel(cfg)                                     # fresh model resumes from the checkpoint
    losses_c = train(model_c, _const_loader(), cfg, n_steps=4, resume_from=ckpt)
    final_c = _params(model_c)

    assert len(losses_c) == 2                                   # only the remaining steps 2,3 ran
    for a, c in zip(final_a, final_c):
        torch.testing.assert_close(a, c, atol=1e-6, rtol=1e-5)  # bit-equivalent continuation


def test_resume_matches_uninterrupted_run_geometric_mstep(tmp_path):
    r"""Resume equivalence for the GEOMETRIC M-step optimizer (m_phi_natural_grad=True).

    This branch develops the gauge-geometric M-step, whose GaugeNaturalGradAdamW keeps a
    heavy-ball ``gauge_mom`` buffer in ``self.state[p]``. Resume restores the optimizer via the
    inherited ``state_dict``/``load_state_dict``; this pins that ``gauge_mom`` actually round-trips
    (a dropped buffer would silently restore wrong gauge momentum and diverge here)."""
    cfg = _cfg(checkpoint_interval=2, m_phi_natural_grad=True, m_phi_lr=0.05,
               phi_precond_mode="pullback_per_block")            # the documented geometric gauge M-step

    torch.manual_seed(0)
    model_a = VFEModel(cfg)
    phi0 = model_a.prior_bank.phi_embed.detach().clone()
    train(model_a, _const_loader(), cfg, n_steps=4)
    final_a = _params(model_a)
    assert not torch.equal(phi0, model_a.prior_bank.phi_embed)   # the gauge frame actually moved (non-vacuous)

    torch.manual_seed(0)
    model_b = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model_b)
    train(model_b, _const_loader(), cfg, n_steps=2, artifacts=art)
    ckpt = tmp_path / "run" / "checkpoints" / "step_2.pt"
    opt_state = torch.load(ckpt, weights_only=False)["optimizer_state"]
    assert any("gauge_mom" in s for s in opt_state["state"].values())   # the buffer was actually saved

    model_c = VFEModel(cfg)
    train(model_c, _const_loader(), cfg, n_steps=4, resume_from=ckpt)
    for a, c in zip(final_a, _params(model_c)):
        torch.testing.assert_close(a, c, atol=1e-6, rtol=1e-5)   # gauge-momentum round-trips correctly


def _shuffled_loader(seq_len: int = 8, bs: int = 4, n: int = 480,
                     data_seed: int = 123, loader_seed: int = 0) -> DataLoader:
    # NONCONSTANT random stream + shuffle=True (RandomSampler): distinct windows, so the batch
    # sequence actually depends on the sampler's in-flight epoch permutation.
    dg = torch.Generator().manual_seed(data_seed)
    g = torch.Generator().manual_seed(loader_seed)
    base = torch.randint(0, 6, (n,), generator=dg)
    ds = TokenWindows(base.to(torch.long), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, generator=g)


def test_shuffled_resume_matches_uninterrupted_run(tmp_path):
    r"""A shuffled six-step run is identical to three steps plus an exact resume."""
    cfg = _cfg(checkpoint_interval=3, max_steps=6)

    torch.manual_seed(0)
    model_a = VFEModel(cfg)
    losses_a = train(model_a, _shuffled_loader(n=105), cfg, n_steps=6)
    final_a = _params(model_a)

    torch.manual_seed(0)
    model_b = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model_b)
    loader_b = _shuffled_loader(n=105)
    epoch_start_generator_state = loader_b.generator.get_state().clone()
    losses_b = train(model_b, loader_b, cfg, n_steps=3, artifacts=art)
    ckpt = tmp_path / "run" / "checkpoints" / "step_3.pt"
    assert ckpt.exists()
    saved_data_state = torch.load(ckpt, weights_only=True)["data_state"]
    assert set(saved_data_state) == {"epoch_start_generator_state", "batches_consumed", "epoch"}
    assert torch.equal(saved_data_state["epoch_start_generator_state"], epoch_start_generator_state)
    assert saved_data_state["batches_consumed"] == 3
    assert saved_data_state["epoch"] == 0

    model_c = VFEModel(cfg)
    resume_art = RunArtifacts(tmp_path / "resumed", cfg, model_c)
    losses_c = train(model_c, _shuffled_loader(n=105), cfg, n_steps=6,
                     artifacts=resume_art, resume_from=ckpt)

    assert len(losses_b) == len(losses_c) == 3
    assert losses_b + losses_c == losses_a
    for uninterrupted, resumed in zip(final_a, _params(model_c)):
        assert torch.equal(uninterrupted, resumed)
    resumed_data_state = torch.load(
        tmp_path / "resumed" / "checkpoints" / "step_6.pt", weights_only=True)["data_state"]
    assert resumed_data_state["epoch"] == 1
    assert resumed_data_state["batches_consumed"] == 3
    assert not torch.equal(resumed_data_state["epoch_start_generator_state"],
                           epoch_start_generator_state)


def test_exact_shuffled_resume_requires_loader_generator(tmp_path):
    cfg = _cfg(checkpoint_interval=1, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    loader = _shuffled_loader()
    loader.generator = None
    loader.sampler.generator = None
    art = RunArtifacts(tmp_path / "run", cfg, model)
    train(model, loader, cfg, n_steps=1, artifacts=art)

    resume_loader = _shuffled_loader()
    resume_loader.generator = None
    resume_loader.sampler.generator = None
    with pytest.raises(RuntimeError, match="exact shuffled resume.*generator"):
        train(VFEModel(cfg), resume_loader, cfg, n_steps=2,
              resume_from=tmp_path / "run" / "checkpoints" / "step_1.pt")


@pytest.mark.parametrize(("field", "bad_value"), [
    ("batches_consumed", 1.5),
    ("batches_consumed", True),
    ("batches_consumed", -1),
    ("epoch", 1.5),
    ("epoch", True),
    ("epoch", -1),
])
def test_save_checkpoint_rejects_malformed_data_cursor(tmp_path, field, bad_value):
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    data_state = {
        "epoch_start_generator_state": torch.Generator().manual_seed(0).get_state(),
        "batches_consumed":            0,
        "epoch":                       0,
    }
    data_state[field] = bad_value

    with pytest.raises(ValueError, match=rf"{field}.*non-negative integer"):
        art.save_checkpoint(1, model, build_optimizer(model, cfg), cfg, data_state=data_state)


@pytest.mark.parametrize(("field", "bad_value"), [
    ("batches_consumed", 1.5),
    ("batches_consumed", True),
    ("batches_consumed", -1),
    ("epoch", 1.5),
    ("epoch", True),
    ("epoch", -1),
])
def test_load_checkpoint_rejects_malformed_data_cursor(tmp_path, field, bad_value):
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    valid_data_state = {
        "epoch_start_generator_state": torch.Generator().manual_seed(0).get_state(),
        "batches_consumed":            0,
        "epoch":                       0,
    }
    checkpoint = art.save_checkpoint(
        1, model, build_optimizer(model, cfg), cfg, data_state=valid_data_state)
    bundle = torch.load(checkpoint, weights_only=True)
    bundle["data_state"][field] = bad_value
    malformed_checkpoint = tmp_path / f"malformed-{field}.pt"
    torch.save(bundle, malformed_checkpoint)

    with pytest.raises(ValueError, match=rf"{field}.*non-negative integer"):
        load_checkpoint(malformed_checkpoint, VFEModel(cfg), data_state={})


def test_resume_restores_best_val_state(tmp_path):
    r"""C2 (audit 2026-07-01): best_val_ppl/best_step are bundled by save_checkpoint and restored
    into the RunArtifacts passed to load_checkpoint, so a resumed continuation with no post-resume
    improvement still reports the run-wide best."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "a", cfg, model)
    assert art.maybe_save_best(3, model, 8.5) is True           # model-selection state to carry over
    art.save_checkpoint(4, model, opt, cfg)

    fresh = VFEModel(cfg)
    new_art = RunArtifacts(tmp_path / "b", cfg, fresh)
    assert new_art.best_val_ppl == float("inf") and new_art.best_step is None
    load_checkpoint(tmp_path / "a" / "checkpoints" / "step_4.pt", fresh, artifacts=new_art)
    assert new_art.best_val_ppl == 8.5
    assert new_art.best_step == 3


def test_resume_with_ema_from_non_ema_ckpt_shadow_tracks_loaded_weights(tmp_path):
    r"""C3 (audit 2026-07-01): resuming a use_ema=True run from a use_ema=False checkpoint (no
    bundled ema_state) must reseed the shadow from the LOADED weights, not the pre-load fresh
    init. With zero remaining steps the final ``copy_to`` writes the shadow into the model, so
    the resumed model must sit exactly at the checkpoint weights."""
    cfg_a = _cfg(checkpoint_interval=2, use_ema=False)
    torch.manual_seed(0)
    model_a = VFEModel(cfg_a)
    art = RunArtifacts(tmp_path / "run", cfg_a, model_a)
    train(model_a, _const_loader(), cfg_a, n_steps=2, artifacts=art)
    saved = _params(model_a)
    ckpt = tmp_path / "run" / "checkpoints" / "step_2.pt"
    assert torch.load(ckpt, weights_only=False)["ema_state"] is None   # genuinely a non-EMA bundle

    cfg_b = _cfg(checkpoint_interval=2, use_ema=True, ema_decay=0.9)
    torch.manual_seed(1)                                        # a DIFFERENT fresh init than the saved weights
    model_b = VFEModel(cfg_b)
    train(model_b, _const_loader(), cfg_b, n_steps=2, resume_from=ckpt)   # zero remaining steps
    for s, b in zip(saved, _params(model_b)):
        assert torch.equal(s, b)                                # shadow == loaded weights, not fresh init


def test_resume_from_cfg_field_is_picked_up(tmp_path):
    r"""cfg.resume_from (click-to-run) is honored when no explicit resume_from arg is passed."""
    cfg = _cfg(checkpoint_interval=2)
    torch.manual_seed(0)
    model_b = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model_b)
    train(model_b, _const_loader(), cfg, n_steps=2, artifacts=art)
    ckpt = tmp_path / "run" / "checkpoints" / "step_2.pt"

    cfg_resume = _cfg(checkpoint_interval=2, resume_from=str(ckpt))
    model_c = VFEModel(cfg_resume)
    losses_c = train(model_c, _const_loader(), cfg_resume, n_steps=4)
    assert len(losses_c) == 2                                   # resumed from step 2 via the cfg field


def test_resume_config_drift_reports_grad_clip_change(tmp_path):
    r"""PB-15: changing only grad_clip across a resume is caught by the existing config-drift
    warning in load_checkpoint (the same mechanism test_resume_warns_on_config_drift pins for
    e_q_mu_lr), proving grad_clip participates in that comparison like any other semantic field."""
    cfg = _cfg(grad_clip=1.0)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    path = art.save_checkpoint(2, model, opt, cfg)

    drifted = VFE3Config(**{**cfg.__dict__, "grad_clip": 0.25})
    with pytest.warns(UserWarning, match=r"config drift.*grad_clip"):
        load_checkpoint(path, model, opt, cfg=drifted)

    # identical config (grad_clip unchanged) -> silent
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        load_checkpoint(path, model, opt, cfg=cfg)


# --------------------------------------------------------------------------- PB-03: portable best weights
#
# A cross-run-directory resume used to restore only the best_val_ppl/best_step SCALARS, so the
# selected weights (best_model.pt) never followed the checkpoint into the new run_dir; finalize then
# saw finite best metadata whose file did not exist. save_checkpoint now embeds a VALIDATED best-model
# bundle, and load_checkpoint publishes it (or a legacy sibling) into the new run, failing closed on
# any missing/tampered/semantically-incompatible bundle. Selection compatibility is judged on the
# SELECTION PROJECTION of the config (architecture/objective fields), so a resume-path or output-cadence
# change cannot invalidate otherwise identical weights, while the full internal fingerprint still
# detects excluded-field tampering. Tiny CPU models throughout (embed_dim=4).


def _eval_loader(seq_len: int = 8, bs: int = 4) -> DataLoader:
    base = torch.arange(3).repeat(20)
    ds = TokenWindows(base[: seq_len * 6].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True)


def _no_git(monkeypatch) -> None:
    monkeypatch.setattr(
        "vfe3.run_artifacts._git_code_identity",
        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})


def _make_terminal_state(model, cfg, *, ema=None) -> TrainingTerminalState:
    r"""A TrainingTerminalState from the model's current (raw) weights + a fresh optimizer + RNG."""
    opt = build_optimizer(model, cfg)
    raw = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    rng = {"cpu": torch.get_rng_state().clone(),
           "cuda": ([s.clone() for s in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available() else None)}
    return TrainingTerminalState(
        step=int(cfg.max_steps), optimizer=opt, scaler=None, ema=ema,
        metropolis_generator=torch.Generator().manual_seed(0),
        data_state=None, raw_model_state=raw, rng_state=rng)


def _build_embedded_best_checkpoint(run_dir, cfg, *, best_ppl=5.0, best_step=2, final_step=4):
    r"""Write a checkpoint whose embedded best bundle carries DISTINCT selected weights.

    Returns (checkpoint_path, best_state, final_state). The best weights (saved to best_model.pt at
    best_ppl) differ from the checkpoint's own model_state, so a later equality check proves the
    SELECTED weights -- not the latest weights -- were carried across the resume."""
    torch.manual_seed(0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(run_dir, cfg, model)
    with torch.no_grad():
        model.prior_bank.mu_embed.add_(0.5)                     # BEST weights
    best_state = {n: p.detach().clone() for n, p in model.named_parameters()}
    art.maybe_save_best(best_step, model, best_ppl)
    with torch.no_grad():
        model.prior_bank.mu_embed.add_(0.5)                     # FINAL weights (distinct from best)
    final_state = {n: p.detach().clone() for n, p in model.named_parameters()}
    ckpt = art.save_checkpoint(final_step, model, opt, cfg)
    return ckpt, best_state, final_state


def _strip_to_legacy(ckpt_path) -> None:
    r"""Remove the best_model_bundle field so the checkpoint reads as a pre-PB-03 (legacy) bundle."""
    bundle = torch.load(ckpt_path, weights_only=False)
    del bundle["best_model_bundle"]
    torch.save(bundle, ckpt_path)


def test_cross_run_resume_restores_embedded_best_bundle(tmp_path):
    cfg = _cfg()
    ckpt, best_state, final_state = _build_embedded_best_checkpoint(tmp_path / "A", cfg)
    assert any(not torch.equal(best_state[n], final_state[n]) for n in best_state)   # distinct

    fresh = VFEModel(cfg)                                        # a different init, a NEW run_dir
    new_art = RunArtifacts(tmp_path / "B", cfg, fresh)
    assert new_art.best_val_ppl == float("inf") and new_art.best_step is None
    load_checkpoint(ckpt, fresh, artifacts=new_art)

    assert new_art.best_val_ppl == 5.0 and new_art.best_step == 2
    assert new_art.best_path.is_file()                          # published into the NEW run_dir
    published = torch.load(new_art.best_path, weights_only=True)["model_state"]
    for n, v in best_state.items():
        assert torch.equal(published[n], v)                     # the SELECTED (best) weights moved
    live = dict(fresh.named_parameters())
    for n, v in final_state.items():
        assert torch.equal(live[n].detach(), v)                 # the model itself got the checkpoint weights


def test_legacy_cross_run_resume_imports_sibling_best_bundle(tmp_path):
    cfg = _cfg()
    ckpt, best_state, _ = _build_embedded_best_checkpoint(tmp_path / "A", cfg)
    _strip_to_legacy(ckpt)
    assert "best_model_bundle" not in torch.load(ckpt, weights_only=False)   # genuinely legacy
    assert (tmp_path / "A" / "best_model.pt").is_file()         # sibling <old_run>/best_model.pt present

    fresh = VFEModel(cfg)
    new_art = RunArtifacts(tmp_path / "B", cfg, fresh)
    load_checkpoint(ckpt, fresh, artifacts=new_art)

    assert new_art.best_val_ppl == 5.0 and new_art.best_step == 2
    assert new_art.best_path.is_file()
    published = torch.load(new_art.best_path, weights_only=True)["model_state"]
    for n, v in best_state.items():
        assert torch.equal(published[n], v)                     # sibling best imported into the new run


def test_resume_without_best_weights_drops_unreachable_best_metadata(tmp_path):
    cfg = _cfg()
    ckpt, _, _ = _build_embedded_best_checkpoint(tmp_path / "A", cfg)
    _strip_to_legacy(ckpt)
    (tmp_path / "A" / "best_model.pt").unlink()                 # neither embedded nor sibling weights

    fresh = VFEModel(cfg)
    new_art = RunArtifacts(tmp_path / "B", cfg, fresh)
    with pytest.warns(UserWarning):
        load_checkpoint(ckpt, fresh, artifacts=new_art)

    assert new_art.best_val_ppl == float("inf")                 # unreachable best scalar dropped
    assert new_art.best_step is None
    assert not new_art.best_path.exists()


def test_checkpoint_rejects_finite_best_without_weights(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    art.best_val_ppl = 5.0                                      # finite selection scalar...
    art.best_step = 2
    assert not art.best_path.exists()                          # ...with no readable best_model.pt

    with pytest.raises(RuntimeError):
        art.save_checkpoint(4, model, opt, cfg)                # integrity error -> no checkpoint


def test_finalize_rejects_best_metadata_without_best_weights(tmp_path, monkeypatch):
    _no_git(monkeypatch)
    cfg = _cfg(generate_figures=False)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    art.best_val_ppl = 5.0                                      # finite metadata, unreachable weights
    art.best_step = 2
    assert not art.best_path.exists()

    with pytest.raises(RuntimeError, match="no reachable weights"):
        finalize_run(model, art, cfg, test_loader=None)


def test_validation_finalizer_scores_terminal_ema_before_best_selection(tmp_path, monkeypatch):
    _no_git(monkeypatch)
    torch.manual_seed(0)
    cfg = _cfg(use_ema=True, ema_decay=0.5, generate_figures=False)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)

    # EMA shadow captured at init, then the raw weights are perturbed so the DEPLOYED EMA differs.
    ema = EMA(model)
    deployed_ema = {name: t.detach().clone() for name, t in ema.shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.75)

    # A DISTINCT prior best whose PPL (4.0) is LOWER than the final EMA result (9.0).
    torch.manual_seed(123)
    prior = VFEModel(cfg)
    art.maybe_save_best(1, prior, 4.0)
    prior_state = torch.load(art.best_path, weights_only=True)["model_state"]
    prior_bytes = art.best_path.read_bytes()

    calls = []

    def fake_evaluate(m, loader, *, tokens_per_char=1.0, device=None):
        calls.append({name: t.detach().clone() for name, t in m.state_dict().items()})
        return {"ce": 2.0, "ppl": 9.0, "bpc": 2.5}

    monkeypatch.setattr("vfe3.train.evaluate", fake_evaluate)

    state = _make_terminal_state(model, cfg, ema=ema)
    mapping = finalize_validation_run(
        model, art, cfg, _eval_loader(), losses=[1.0, 0.9],
        terminal_state=state, device=torch.device("cpu"))

    assert len(calls) == 1                                      # scored the terminal EMA exactly once
    scored = calls[0]
    for name, ema_w in deployed_ema.items():
        assert torch.equal(scored[name], ema_w)                # ...on the deployed EMA weights
    assert any(not torch.equal(scored[name], prior_state[name]) for name in deployed_ema)  # not prior best
    assert art.best_path.read_bytes() == prior_bytes           # prior best file untouched (9.0 !< 4.0)
    assert mapping["primary_val_ppl"] == 4.0                   # the better prior best remains primary
    assert mapping["best_val_ppl"] == 4.0
    assert mapping["final_val_ppl"] == 9.0                     # final fields are the terminal score
    assert mapping["final_val_ce"] == 2.0


def test_selection_projection_migrates_missing_defaults_and_rejects_unknown_fields():
    from vfe3.run_artifacts import _selection_semantic_config

    cfg = _cfg()
    live_projection = _selection_semantic_config(cfg)
    serialized = asdict(cfg)

    # The stored FULL fingerprint is a stable function of the raw mapping (the tamper-check basis).
    assert semantic_config_fingerprint(serialized) == semantic_config_fingerprint(
        dict(reversed(list(serialized.items()))))

    # A genuinely older mapping missing a defaulted behavior field acquires the CURRENT default.
    older = dict(serialized)
    del older["decode_tau"]
    assert "decode_tau" not in older
    assert _selection_semantic_config(older) == live_projection

    # An unknown newer field fails closed rather than being silently ignored.
    newer = dict(serialized)
    newer["a_field_from_the_future"] = 123
    assert semantic_config_fingerprint(newer) == semantic_config_fingerprint(
        dict(reversed(list(newer.items()))))
    with pytest.raises(ValueError):
        _selection_semantic_config(newer)


def test_selection_projection_grad_clip_migrates_to_default_and_differentiates():
    r"""PB-15 cross-plan regression: a raw legacy mapping predating grad_clip (simulated by
    stripping the field from asdict(VFE3Config())) migrates through _selection_semantic_config
    to the CURRENT default (grad_clip=1.0), matching the live default projection exactly -- an
    explicit non-default grad_clip=0.25 must project differently. The raw legacy mapping's own
    full fingerprint is verified BEFORE the projection (the tamper-check basis, mirroring
    test_selection_projection_migrates_missing_defaults_and_rejects_unknown_fields above), and an
    unknown field still fails closed rather than being silently ignored by config_from_serialized."""
    from vfe3.run_artifacts import _selection_semantic_config

    legacy = asdict(VFE3Config())
    assert legacy["grad_clip"] == 1.0
    del legacy["grad_clip"]
    assert "grad_clip" not in legacy

    # The raw legacy mapping's own full fingerprint is a stable function of key order.
    assert semantic_config_fingerprint(legacy) == semantic_config_fingerprint(
        dict(reversed(list(legacy.items()))))

    live_default_projection = _selection_semantic_config(VFE3Config())
    assert live_default_projection["grad_clip"] == 1.0
    assert _selection_semantic_config(legacy) == live_default_projection

    explicit = dict(legacy)
    explicit["grad_clip"] = 0.25
    explicit_projection = _selection_semantic_config(explicit)
    assert explicit_projection["grad_clip"] == 0.25
    assert explicit_projection != live_default_projection

    unknown = dict(legacy)
    unknown["a_field_from_the_future"] = 123
    with pytest.raises(ValueError):
        _selection_semantic_config(unknown)


def test_resume_from_only_difference_survives_finalization(tmp_path, monkeypatch):
    _no_git(monkeypatch)
    cfg_a = _cfg(generate_figures=False)                        # resume_from=None
    ckpt, best_state, _ = _build_embedded_best_checkpoint(tmp_path / "A", cfg_a)

    # Resume into B under a config that differs ONLY in resume_from (and trust flags).
    cfg_b = _cfg(generate_figures=False, resume_from=str(ckpt))
    model_b = VFEModel(cfg_b)
    new_art = RunArtifacts(tmp_path / "B", cfg_b, model_b)
    load_checkpoint(ckpt, model_b, artifacts=new_art, cfg=cfg_b)
    assert new_art.best_val_ppl == 5.0                         # imported despite the resume_from diff
    assert new_art.best_path.is_file()
    res = finalize_run(model_b, new_art, cfg_b, test_loader=_const_loader())
    assert res["reloaded_best"] is True                        # finalize reloaded the imported best

    # Control: a behavior-field difference (decode_tau) must REJECT the same bundle.
    cfg_c = _cfg(generate_figures=False, decode_tau=2.0)
    model_c = VFEModel(cfg_c)
    art_c = RunArtifacts(tmp_path / "C", cfg_c, model_c)
    with pytest.raises(RuntimeError):
        load_checkpoint(ckpt, model_c, artifacts=art_c, cfg=cfg_c)


def test_stale_contract_rerun_replaces_old_unselected_best(tmp_path, monkeypatch):
    _no_git(monkeypatch)
    torch.manual_seed(0)
    cfg = _cfg(generate_figures=False)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)

    # A stale best_model.pt left by a previous cell: distinct weights, on disk, but the recomputed
    # cell begins with INFINITE in-memory best metadata (it must not silently select these weights).
    stale = VFEModel(cfg)
    with torch.no_grad():
        for p in stale.parameters():
            p.add_(3.0)
    art.maybe_save_best(1, stale, 2.0)
    stale_bytes = art.best_path.read_bytes()
    stale_state = torch.load(art.best_path, weights_only=True)["model_state"]
    art.best_val_ppl = float("inf")                            # recomputed-cell in-memory reset
    art.best_step = None

    state = _make_terminal_state(model, cfg)
    mapping = finalize_validation_run(
        model, art, cfg, _eval_loader(), losses=[1.0, 0.9],
        terminal_state=state, device=torch.device("cpu"))

    assert art.best_path.is_file()
    assert art.best_path.read_bytes() != stale_bytes           # terminal validation replaced the file
    new_state = torch.load(art.best_path, weights_only=True)["model_state"]
    assert any(not torch.equal(new_state[k], stale_state[k]) for k in stale_state)
    assert math.isfinite(mapping["primary_val_ppl"])
    assert Path(mapping["terminal_checkpoint"]).exists()       # success contract published
    assert (tmp_path / "r" / "summary.json").exists()


@pytest.mark.parametrize("corruption", [
    "stale_fingerprint", "missing_key", "extra_key",
    "wrong_shape", "wrong_dtype", "non_tensor",
])
def test_corrupt_embedded_best_bundle_rejected_on_resume(tmp_path, corruption):
    cfg = _cfg()
    ckpt, _, _ = _build_embedded_best_checkpoint(tmp_path / "A", cfg)
    bundle = torch.load(ckpt, weights_only=False)
    embedded = bundle["best_model_bundle"]
    model_state = embedded["model_state"]
    a_key = next(iter(model_state))
    if corruption == "stale_fingerprint":
        embedded["config_fingerprint"] = "0" * 64
    elif corruption == "missing_key":
        del model_state[a_key]
    elif corruption == "extra_key":
        model_state["phantom_param"] = torch.zeros(2)
    elif corruption == "wrong_shape":
        model_state[a_key] = torch.zeros(model_state[a_key].shape + (1,))
    elif corruption == "wrong_dtype":
        model_state[a_key] = model_state[a_key].to(torch.float64)
    elif corruption == "non_tensor":
        model_state[a_key] = [1.0, 2.0, 3.0]
    torch.save(bundle, ckpt)

    fresh = VFEModel(cfg)
    new_art = RunArtifacts(tmp_path / "B", cfg, fresh)
    with pytest.raises(RuntimeError):
        load_checkpoint(ckpt, fresh, artifacts=new_art)


def test_full_model_channel_packed_tables_round_trip(tmp_path):
    r"""PB-11: the full-covariance model-channel packed Cholesky tables (s_sigma_lower_embed and
    r_sigma_lower) survive a checkpoint save/load exactly, so a resumed full-Gaussian model channel
    restores its off-diagonal covariance and not just the diagonal log-variance."""
    cfg = _cfg(family="gaussian_full", decode_mode="full", lambda_h=0.5)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    with torch.no_grad():                                       # make the packed tables genuinely nonzero
        model.prior_bank.s_sigma_lower_embed.normal_(0.0, 0.4)
        model.prior_bank.r_sigma_lower.normal_(0.0, 0.4)
    saved_s = model.prior_bank.s_sigma_lower_embed.detach().clone()
    saved_r = model.prior_bank.r_sigma_lower.detach().clone()
    art.save_checkpoint(4, model, opt, cfg)

    fresh = VFEModel(cfg)                                       # a fresh model whose packed tables are still zero
    assert not torch.equal(fresh.prior_bank.s_sigma_lower_embed, saved_s)
    load_checkpoint(tmp_path / "r" / "checkpoints" / "step_4.pt", fresh)
    assert torch.equal(fresh.prior_bank.s_sigma_lower_embed, saved_s)
    assert torch.equal(fresh.prior_bank.r_sigma_lower, saved_r)
