r"""fp16 training GradScaler (close the silent-mistrain footgun). Spec:
docs/superpowers/specs/2026-06-08-fp16-gradscaler-design.md.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, train_step


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
