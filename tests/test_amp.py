r"""Opt-in mixed-precision (amp_dtype) tests.

The pure fp32 path (amp_dtype=None, the default) must enter NO autocast context and stay
byte-identical to the no-AMP build; the bf16/fp16 path must run and stay finite; and the
cancellation-sensitive decode + cross-entropy must stay fp32 even when AMP is on (the fp32
island holds). The full suite is the broader default-off byte-identity oracle (every other
test runs amp_dtype=None); these tests pin the AMP wiring specifically.
"""

from contextlib import nullcontext
from unittest import mock

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import _val_diagnostics


def _tiny_model(**overrides) -> VFEModel:
    base = dict(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.0, use_prior_bank=True)
    base.update(overrides)
    return VFEModel(VFE3Config(**base))


def test_default_off_enters_no_autocast_context():
    # amp_dtype=None (default) must NOT instantiate any torch.autocast object in forward:
    # the default path is byte-identical to the no-AMP build because no autocast context is
    # ever entered. Patch torch.autocast to a tripwire that fails if called.
    model = _tiny_model()
    assert model.cfg.amp_dtype is None
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))

    def _tripwire(*a, **k):
        raise AssertionError("torch.autocast was entered on the amp_dtype=None default path")

    with mock.patch("torch.autocast", _tripwire):
        _, loss, _ = model(tok, tgt)
    assert torch.isfinite(loss)


def test_default_off_forward_is_deterministic_and_finite():
    # The default (amp_dtype=None) forward is unchanged: same seed -> identical logits/loss.
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    torch.manual_seed(0); m1 = _tiny_model()
    torch.manual_seed(0); m2 = _tiny_model()
    l1, loss1, _ = m1(tok, tgt)
    l2, loss2, _ = m2(tok, tgt)
    assert torch.equal(l1, l2) and torch.equal(loss1, loss2)


def test_bf16_forward_runs_and_is_finite():
    # amp_dtype='bf16' wraps the E-step in autocast (device-resolved, so the CPU box exercises
    # it) and returns a finite loss. CPU bf16 autocast is REAL casting (not a no-op), so the loss
    # is only loosely close to fp32 -- assert finite + shape + a relaxed allclose. The point is the
    # path is wired and the fp32 islands hold (no crash, finite).
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    torch.manual_seed(0); m_fp32 = _tiny_model()
    torch.manual_seed(0); m_bf16 = _tiny_model(amp_dtype="bf16")
    logits_fp32, loss_fp32, _ = m_fp32(tok, tgt)
    logits_bf16, loss_bf16, _ = m_bf16(tok, tgt)
    assert logits_bf16.shape == logits_fp32.shape
    assert torch.isfinite(loss_bf16)
    # bf16 mantissa is ~3 decimal digits; the loss should be in the same ballpark as fp32.
    assert torch.allclose(loss_bf16, loss_fp32, atol=0.5, rtol=0.1)


def test_fp16_forward_runs_and_is_finite():
    # amp_dtype='fp16' is accepted and runs a forward without error (forward-correctness scope;
    # fp16 TRAINING would need a GradScaler in the M-step -- a documented follow-up).
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    m = _tiny_model(amp_dtype="fp16")
    _, loss, _ = m(tok, tgt)
    assert torch.isfinite(loss)


def test_decode_and_ce_stay_fp32_under_bf16():
    # The cancellation-sensitive decode + CE must run in fp32 even under amp_dtype='bf16': the
    # island holds. Intercept PriorBank.decode to record its INPUT dtype (the .float() guard) and
    # its OUTPUT logits dtype, and assert the CE-input logits are fp32. This pins the protection
    # the islands exist for (the atol-1e-3 decode pin) regardless of the autocast E-step.
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    model = _tiny_model(amp_dtype="bf16")

    seen = {}
    real_decode = model.prior_bank.decode

    def _spy(mu_q, sigma_q, **kw):
        seen["mu_in"] = mu_q.dtype
        seen["sigma_in"] = sigma_q.dtype
        out = real_decode(mu_q, sigma_q, **kw)
        seen["logits_out"] = out.dtype
        return out

    with mock.patch.object(model.prior_bank, "decode", _spy):
        _, loss, _ = model(tok, tgt)

    # Decode inputs are .float()-ed (the load-bearing fp32 guard), so the cancellation-sensitive
    # matmul runs in fp32, and the logits feeding CE are fp32.
    assert seen["mu_in"] == torch.float32
    assert seen["sigma_in"] == torch.float32
    assert seen["logits_out"] == torch.float32
    assert torch.isfinite(loss)


def test_bf16_backward_reaches_prior_tables():
    # Under bf16 the loss must still backprop through the unrolled E-step to the prior tables
    # (the AMP wrap does not sever the training graph). bf16 needs no GradScaler.
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    model = _tiny_model(amp_dtype="bf16", e_phi_lr=0.02)
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0


def test_bf16_head_mixer_validation_diagnostics_run_in_fp32():
    r"""A bf16 head-mixer forward must leave the off-graph diagnostic replay in fp32."""
    model = _tiny_model(
        amp_dtype="bf16",
        compact_phi_block_transport=True,
        gauge_group="block_glk",
        transport_mean_per_head=True,
        use_head_mixer=True,
    )
    tok = torch.randint(0, 20, (2, 5))
    tgt = torch.randint(0, 20, (2, 5))

    model(tok, tgt)  # match the successful validation forward that precedes the diagnostic replay
    snapshot = model.build_diagnostic_snapshot(tok)
    diagnostics = model.diagnostics(tok, snapshot=snapshot)
    val_diagnostics = _val_diagnostics(model, [(tok, tgt)], torch.device("cpu"))

    assert snapshot.stack_output.mu.dtype == torch.float32
    assert snapshot.stack_output.sigma.dtype == torch.float32
    assert snapshot.beta_maps.dtype == torch.float32
    assert torch.isfinite(torch.tensor(diagnostics["total"]))
    assert torch.isfinite(torch.tensor(val_diagnostics["val_free_energy_total"]))
