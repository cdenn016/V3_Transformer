r"""Regression tests for the 2026-06-17 round-2 (deep) audit fixes.

See docs/audits/audit-2026-06-17-deep.md. One test per confirmed behavioral fix.
"""

import inspect

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, train_step


def _cfg(**over) -> VFE3Config:
    base = dict(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1, n_e_steps=1)
    base.update(over)
    return VFE3Config(**base)


def _sched(opt):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda _s: 1.0)


# ---- id16: non-finite loss must skip the optimizer step (no AdamW poison) on the default path ----
def test_nonfinite_loss_skips_step_no_param_poison():
    torch.manual_seed(0)
    base = VFEModel(_cfg())
    opt = build_optimizer(base, base.cfg)
    sch = _sched(opt)
    tok = torch.randint(0, 20, (2, 5))
    tgt = torch.randint(0, 20, (2, 5))

    class _NaNLoss(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m; self.cfg = m.cfg
        def forward(self, t, y=None, **kw):
            logits, loss, ce = self.m(t, y, **kw)
            return logits, loss * float("nan"), ce
        def parameters(self, *a, **k): return self.m.parameters(*a, **k)
        def named_parameters(self, *a, **k): return self.m.named_parameters(*a, **k)

    before = [p.detach().clone() for p in base.parameters()]
    train_step(_NaNLoss(base), opt, sch, tok, tgt, grad_clip=1.0)   # NaN loss -> step skipped
    for b, p in zip(before, base.parameters()):
        assert torch.isfinite(p).all(), "AdamW poisoned by a NaN-gradient step"
        assert torch.equal(b, p), "params moved on a non-finite-loss step"


def test_finite_loss_still_steps():
    # control: a finite loss DOES update params (the skip only fires on non-finite).
    torch.manual_seed(0)
    m = VFEModel(_cfg())
    opt = build_optimizer(m, m.cfg)
    sch = _sched(opt)
    tok = torch.randint(0, 20, (2, 5))
    tgt = torch.randint(0, 20, (2, 5))
    before = [p.detach().clone() for p in m.parameters()]
    train_step(m, opt, sch, tok, tgt, grad_clip=1.0)
    assert any(not torch.equal(b, p) for b, p in zip(before, m.parameters()))


# ---- id0 / id15: free_energy_value accept-and-ignores mass_phi; phi_alignment_loss threads rope ----
def test_free_energy_value_accepts_mass_phi():
    from vfe3.inference.e_step import free_energy_value
    assert "mass_phi" in inspect.signature(free_energy_value).parameters


def test_phi_alignment_loss_threads_rope():
    from vfe3.inference.e_step import phi_alignment_loss
    p = inspect.signature(phi_alignment_loss).parameters
    assert {"rope", "rope_on_cov", "rope_on_value"} <= set(p)


# ---- id1: condition_number diagonal branch surfaces +inf for a non-positive spectrum ----
def test_condition_number_diagonal_non_positive_is_inf():
    from vfe3.numerics import condition_number
    assert torch.isinf(condition_number(torch.tensor([0.0, 1.0, 3.0])))
    assert torch.isinf(condition_number(torch.tensor([-1.0, 1.0, 3.0])))
    assert torch.isclose(condition_number(torch.tensor([1.0, 4.0])), torch.tensor(4.0))  # PD unchanged


# ---- id19: EMA.update must not propagate a non-finite live param into the shadow ----
def test_ema_update_skips_nonfinite_param():
    from vfe3.ema import EMA
    m = torch.nn.Linear(3, 2)
    ema = EMA(m, decay=0.9)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        m.weight.fill_(float("nan"))                  # poison a live param
    ema.update(m)
    for k, v in ema.shadow.items():
        assert torch.isfinite(v).all(), f"shadow {k} poisoned by a NaN live param"
    assert torch.equal(ema.shadow["weight"], before["weight"])  # the NaN param's shadow held


# ---- id22: precision bias is folded in diagnostics() too (multi-block AND single-block) ----
def test_precision_bias_folded_in_diagnostics_multiblock():
    cfg_on = _cfg(precision_weighted_attention=True)
    cfg_off = _cfg(precision_weighted_attention=False)
    torch.manual_seed(0); m_on = VFEModel(cfg_on)
    torch.manual_seed(0); m_off = VFEModel(cfg_off)
    with torch.no_grad():
        for mm in (m_on, m_off):
            mm.prior_bank.sigma_log_embed.copy_(torch.randn_like(mm.prior_bank.sigma_log_embed))
    tok = torch.randint(0, 20, (1, 5))
    d_on = m_on.diagnostics(tok)
    d_off = m_off.diagnostics(tok)
    # the reliability bias changes the attention distribution -> the F-decomposition differs.
    assert d_on["attn_entropy"] != d_off["attn_entropy"]


def test_precision_bias_diagnostics_single_block_runs():
    # single-block (glk) group has a HEADLESS energy; the diagnostics fold must shape the bias
    # without a head axis (would mis-broadcast otherwise). Just assert it runs + is finite.
    cfg = _cfg(n_heads=1, gauge_group="glk", precision_weighted_attention=True)
    torch.manual_seed(0); m = VFEModel(cfg)
    assert len(m.group.irrep_dims) == 1
    with torch.no_grad():
        m.prior_bank.sigma_log_embed.copy_(torch.randn_like(m.prior_bank.sigma_log_embed))
    tok = torch.randint(0, 20, (1, 5))
    d = m.diagnostics(tok)
    assert torch.isfinite(torch.tensor(d["attn_entropy"]))
    m.attention_maps(tok)                              # must also run under single-block precision
