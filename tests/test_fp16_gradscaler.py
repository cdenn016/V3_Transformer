r"""fp16 training GradScaler (close the silent-mistrain footgun). Spec:
docs/superpowers/specs/2026-06-08-fp16-gradscaler-design.md.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.ema import EMA
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, train, train_step


def _tiny_cfg(**overrides) -> VFE3Config:
    base = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
    base.update(overrides)
    return VFE3Config(**base)


def _sched(opt):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda _step: 1.0)


def _batch(cfg, b=2, n=4):
    tok = torch.randint(0, cfg.vocab_size, (b, n))
    tgt = torch.randint(0, cfg.vocab_size, (b, n))
    return tok, tgt


def test_gradscaler_disabled_is_byte_identical_to_unscaled_step():
    # amp_dtype=None -> scaler enabled=False -> the wired train_step must produce the SAME
    # parameter update as the original unscaled loss.backward()/optimizer.step() path.
    cfg = _tiny_cfg()  # amp_dtype defaults to None
    torch.manual_seed(0); mA = VFEModel(cfg)
    torch.manual_seed(0); mB = VFEModel(cfg)
    tok, tgt = _batch(cfg)

    optA = build_optimizer(mA, cfg); schA = _sched(optA)
    train_step(mA, optA, schA, tok, tgt, grad_clip=1.0)

    optB = build_optimizer(mB, cfg)
    optB.zero_grad(set_to_none=True)
    _, lossB, _ = mB(tok, tgt)
    lossB.backward()
    torch.nn.utils.clip_grad_norm_(mB.parameters(), 1.0)
    optB.step()

    for (na, pa), (nb, pb) in zip(mA.named_parameters(), mB.named_parameters()):
        assert torch.equal(pa, pb), f"param {na} diverged from the unscaled reference"


def test_fp16_forward_keeps_logits_and_sigma_finite():
    # The SPD/sigma islands must stay fp32 under amp_dtype='fp16' (no Cholesky/eigh NaN).
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(amp_dtype="fp16"))
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    lg = m(tok)
    assert torch.isfinite(lg).all()


def _run_one_fp16_step(cfg):
    torch.manual_seed(0)
    m = VFEModel(cfg)
    opt = build_optimizer(m, cfg); sch = _sched(opt)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True)
    before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    tok, tgt = _batch(cfg)
    train_step(m, opt, sch, tok, tgt, grad_clip=1.0, scaler=scaler)
    moved = any(not torch.equal(before[n], p) for n, p in m.named_parameters()
                if p.requires_grad and n in before)
    return m, scaler, moved


def test_fp16_scaler_scales_and_updates_params():
    # Under amp_dtype='fp16' with an ENABLED scaler, a train_step must move parameters
    # (i.e. fp16 grads are scaled, not underflowed to zero).
    cfg = _tiny_cfg(amp_dtype="fp16")
    m, scaler, moved = _run_one_fp16_step(cfg)
    assert torch.isfinite(torch.tensor(scaler.get_scale()))
    assert moved, "no parameter moved under the enabled fp16 scaler (gradients underflowed?)"


def test_fp16_with_gauge_natural_grad_steps_the_frame():
    # Option A: fp16 + m_phi_natural_grad (custom GaugeManifoldAdamW) must compose with the
    # scaler -- the gauge-frame table must move under a scaled step (not silently no-op).
    cfg = _tiny_cfg(amp_dtype="fp16", m_phi_natural_grad=True, pos_phi="learned")
    torch.manual_seed(0)
    m = VFEModel(cfg)
    opt = build_optimizer(m, cfg); sch = _sched(opt)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True)
    phi0 = m.prior_bank.phi_embed.detach().clone()
    tok, tgt = _batch(cfg)
    train_step(m, opt, sch, tok, tgt, grad_clip=1.0, scaler=scaler)
    assert not torch.equal(phi0, m.prior_bank.phi_embed), "gauge frame did not move under fp16 scaler"


def test_ema_does_not_advance_on_gradscaler_overflow(monkeypatch, device):
    torch.manual_seed(0)
    cfg = _tiny_cfg(amp_dtype="fp16", use_ema=True, ema_decay=0.9, max_steps=1)
    model = VFEModel(cfg).to(device)
    model.prior_bank.mu_embed.register_hook(
        lambda grad: torch.full_like(grad, float("inf")))
    tok, tgt = _batch(cfg)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    updates = []
    scalers = []
    update = EMA.update
    grad_scaler = torch.amp.GradScaler

    def _capture_scaler(*args, **kwargs):
        scaler = grad_scaler(*args, **kwargs)
        scalers.append((scaler, float(scaler.get_scale())))
        return scaler

    def _record_update(self, live_model):
        updates.append(True)
        update(self, live_model)

    monkeypatch.setattr(torch.amp, "GradScaler", _capture_scaler)
    monkeypatch.setattr(EMA, "update", _record_update)
    train(model, [(tok, tgt)], cfg, n_steps=1, device=device, generate_samples=False)

    assert updates == []
    assert len(scalers) == 1
    scaler, initial_scale = scalers[0]
    assert scaler.is_enabled()
    assert float(scaler.get_scale()) < initial_scale
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name])
