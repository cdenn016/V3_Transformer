r"""EMA / Polyak weight averaging (``vfe3.ema.EMA``, default-OFF toggle ``use_ema``).

A passive shadow of the trainable params, updated ``s <- decay*s + (1-decay)*theta`` after each
optimizer step, swapped in for evaluation/best-save and copied into the model at the end of
training. It must NOT perturb the SGD trajectory (draws no RNG, touches no grad/optimizer state):
the per-step CE history ``train(...)`` returns is therefore byte-identical with the toggle on or off.
"""

import copy

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.ema import EMA
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, load_checkpoint
from vfe3.train import build_optimizer, train


# --------------------------------------------------------------------------- helpers

class _Toy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(3))
        self.b = torch.nn.Parameter(torch.ones(2))
        self.frozen = torch.nn.Parameter(torch.full((2,), 5.0), requires_grad=False)


def _const_loader(seq_len: int = 8, bs: int = 4) -> DataLoader:
    base = torch.full((seq_len * 6,), 1, dtype=torch.long)
    return DataLoader(TokenWindows(base, seq_len), batch_size=bs, shuffle=False, drop_last=True)


def _cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return VFE3Config(**base)


def _params(model: torch.nn.Module):
    return [p.detach().clone() for p in model.parameters()]


# --------------------------------------------------------------------------- EMA unit

def test_ema_tracks_only_requires_grad_params():
    ema = EMA(_Toy(), decay=0.9)
    assert set(ema.shadow) == {"a", "b"}                 # 'frozen' (requires_grad=False) is not tracked


def test_ema_shadow_init_equals_params():
    toy = _Toy()
    ema = EMA(toy, decay=0.9)
    assert torch.equal(ema.shadow["a"], toy.a.detach())
    assert torch.equal(ema.shadow["b"], toy.b.detach())
    assert ema.shadow["a"] is not toy.a                  # a detached clone, not an alias


def test_ema_update_is_convex_blend():
    toy = _Toy()
    ema = EMA(toy, decay=0.9)                             # shadow['a'] == 0
    with torch.no_grad():
        toy.a.fill_(1.0)                                 # param moves to 1
    ema.update(toy)
    # s <- 0.9*0 + 0.1*1 = 0.1
    torch.testing.assert_close(ema.shadow["a"], torch.full((3,), 0.1))


def test_ema_store_copy_restore_roundtrip():
    toy = _Toy()
    ema = EMA(toy, decay=0.9)                             # shadow == init (a=0, b=1)
    with torch.no_grad():                                # diverge live params from the shadow
        toy.a.fill_(7.0)
        toy.b.fill_(8.0)
    live = _params(toy)
    ema.store(toy)
    ema.copy_to(toy)                                     # model now holds the shadow
    torch.testing.assert_close(toy.a.detach(), ema.shadow["a"])
    torch.testing.assert_close(toy.b.detach(), ema.shadow["b"])
    ema.restore(toy)                                     # model back to the live SGD params
    for restored, original in zip(_params(toy), live):
        assert torch.equal(restored, original)


def test_ema_resets_shadow_after_load_when_no_ema_state(tmp_path):
    r"""C3 (audit 2026-07-01): loading a checkpoint that carries NO ema_state (a use_ema=False or
    legacy bundle) must RESEED the shadow from the just-loaded weights -- the shadow built at
    ``EMA.__init__`` clones the PRE-load fresh init, and blending real weights into random-init
    noise would corrupt the running average."""
    cfg = _cfg()
    torch.manual_seed(0)
    model_a = VFEModel(cfg)
    opt = build_optimizer(model_a, cfg)
    with torch.no_grad():                                # make the saved weights distinguishable
        model_a.prior_bank.mu_embed.add_(0.5)
    art = RunArtifacts(tmp_path / "r", cfg, model_a)
    art.save_checkpoint(2, model_a, opt, cfg)            # ema=None -> bundle stores ema_state=None

    torch.manual_seed(1)                                 # a DIFFERENT fresh init
    model_b = VFEModel(cfg)
    ema_b = EMA(model_b, decay=0.9)                      # shadow clones B's PRE-load init
    pre = {name: t.clone() for name, t in ema_b.shadow.items()}
    load_checkpoint(tmp_path / "r" / "checkpoints" / "step_2.pt", model_b, ema=ema_b)

    params = dict(model_b.named_parameters())
    assert set(ema_b.shadow) == {n for n, p in params.items() if p.requires_grad}   # same filter as __init__
    for name, shadow in ema_b.shadow.items():
        assert torch.equal(shadow, params[name].detach())        # reseeded from the LOADED weights
    assert any(not torch.equal(pre[n], ema_b.shadow[n]) for n in pre)   # ...not the pre-load init


def test_ema_state_dict_roundtrip():
    toy = _Toy()
    ema = EMA(toy, decay=0.9)
    with torch.no_grad():
        toy.a.fill_(3.0)
    ema.update(toy)
    sd = copy.deepcopy(ema.state_dict())

    ema2 = EMA(_Toy(), decay=0.5)                         # different decay + fresh shadow
    ema2.load_state_dict(sd)
    assert ema2.decay == 0.9
    for k in ema.shadow:
        assert torch.equal(ema2.shadow[k], ema.shadow[k])


# --------------------------------------------------------------------------- config

def test_config_ema_defaults_off():
    cfg = VFE3Config()
    assert cfg.use_ema is False
    assert cfg.ema_decay == 0.999


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
def test_config_ema_decay_validated_when_on(bad):
    with pytest.raises(ValueError):
        VFE3Config(use_ema=True, ema_decay=bad)


def test_config_ema_decay_unvalidated_when_off():
    VFE3Config(use_ema=False, ema_decay=1.0)             # inert when off -> not validated


# --------------------------------------------------------------------------- train integration

def test_ema_does_not_perturb_training_trajectory():
    # EMA is a passive observer: same seed -> identical per-step CE history with the toggle on/off.
    torch.manual_seed(0)
    m_off = VFEModel(_cfg(use_ema=False))
    losses_off = train(m_off, _const_loader(), _cfg(use_ema=False), n_steps=4)

    torch.manual_seed(0)
    m_on = VFEModel(_cfg(use_ema=True, ema_decay=0.9))
    losses_on = train(m_on, _const_loader(), _cfg(use_ema=True, ema_decay=0.9), n_steps=4)

    assert losses_on == losses_off                        # trajectory untouched


def test_ema_changes_final_model():
    # With EMA on, the returned model holds the averaged weights -> differs from the SGD-final model.
    torch.manual_seed(0)
    m_off = VFEModel(_cfg(use_ema=False))
    train(m_off, _const_loader(), _cfg(use_ema=False), n_steps=4)

    torch.manual_seed(0)
    m_on = VFEModel(_cfg(use_ema=True, ema_decay=0.9))
    train(m_on, _const_loader(), _cfg(use_ema=True, ema_decay=0.9), n_steps=4)

    assert any(not torch.equal(a, b) for a, b in zip(_params(m_off), _params(m_on)))


def test_ema_eval_swap_runs_and_restores():
    # Exercise the eval-block store->copy_to->restore path; training must still be unperturbed.
    torch.manual_seed(0)
    m_off = VFEModel(_cfg(use_ema=False))
    losses_off = train(m_off, _const_loader(), _cfg(use_ema=False), n_steps=4,
                       eval_interval=2, val_loader=_const_loader(), generate_samples=False)

    torch.manual_seed(0)
    m_on = VFEModel(_cfg(use_ema=True, ema_decay=0.9))
    losses_on = train(m_on, _const_loader(), _cfg(use_ema=True, ema_decay=0.9), n_steps=4,
                      eval_interval=2, val_loader=_const_loader(), generate_samples=False)

    assert losses_on == losses_off                        # eval swap restores live weights before next step


def test_ema_resume_matches_uninterrupted_run(tmp_path):
    r"""EMA shadow round-trips through a checkpoint: straight run == (checkpoint -> resume) for the
    AVERAGED final model. A dropped ema_state would re-seed the shadow from the resumed SGD weights
    and the averaged final model would diverge here."""
    cfg = _cfg(checkpoint_interval=2, use_ema=True, ema_decay=0.9)

    torch.manual_seed(0)                                  # Run A: straight through
    model_a = VFEModel(cfg)
    train(model_a, _const_loader(), cfg, n_steps=4)
    final_a = _params(model_a)                            # averaged (EMA copied in at end)

    torch.manual_seed(0)                                  # Run B: 2 steps -> checkpoint -> resume to 4
    model_b = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model_b)
    train(model_b, _const_loader(), cfg, n_steps=2, artifacts=art)
    ckpt = tmp_path / "run" / "checkpoints" / "step_2.pt"
    assert "ema_state" in torch.load(ckpt, weights_only=False)   # shadow was actually persisted

    model_c = VFEModel(cfg)
    train(model_c, _const_loader(), cfg, n_steps=4, resume_from=ckpt)
    for a, c in zip(final_a, _params(model_c)):
        torch.testing.assert_close(a, c, atol=1e-6, rtol=1e-5)
