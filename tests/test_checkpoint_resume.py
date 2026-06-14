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

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, load_checkpoint
from vfe3.train import build_optimizer, train


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
