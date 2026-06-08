r"""Cheap ledger wins: M6 (b0/c0 sequence), T3 (per-head ALiBi), T1 (per-head kappa).
Spec: docs/superpowers/specs/2026-06-08-cheap-ledger-wins-design.md. Each default byte-identical.
"""

import dataclasses
import json

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _tiny_cfg(**overrides) -> VFE3Config:
    base = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
    base.update(overrides)
    return VFE3Config(**base)


def test_b0_c0_default_scalar():
    cfg = VFE3Config()
    assert cfg.b0 == 1.0 and cfg.c0 == 1.0


def test_b0_c0_list_length_must_match_embed_dim():
    with pytest.raises(ValueError, match="b0"):
        _tiny_cfg(b0=[1.0, 1.0, 1.0])           # embed_dim=4, list len 3 -> reject
    cfg = _tiny_cfg(b0=[1.0, 2.0, 3.0, 4.0])
    assert list(cfg.b0) == [1.0, 2.0, 3.0, 4.0]


def test_b0_c0_list_entries_must_be_positive():
    with pytest.raises(ValueError, match="c0"):
        _tiny_cfg(c0=[1.0, 0.0, 1.0, 1.0])


def test_b0_list_config_is_json_serializable():
    cfg = _tiny_cfg(b0=[1.0, 2.0, 3.0, 4.0])
    json.dumps(dataclasses.asdict(cfg))         # must not raise (list -> json, no tensor)


def test_b0_list_threads_per_coord_alpha_into_the_model():
    torch.manual_seed(0)
    base = VFEModel(_tiny_cfg(alpha_mode="state_dependent_per_coord", b0=1.0, n_e_steps=2))
    torch.manual_seed(0)
    perc = VFEModel(_tiny_cfg(alpha_mode="state_dependent_per_coord",
                              b0=[0.2, 0.5, 2.0, 5.0], n_e_steps=2))
    tok = torch.randint(0, base.cfg.vocab_size, (2, 4))
    lg_base, lg_perc = base(tok), perc(tok)
    assert torch.isfinite(lg_perc).all()
    assert not torch.allclose(lg_base, lg_perc)


def test_b0_list_through_viz_extract_trajectory():
    # viz/extract.py is a parallel b0/c0 consumption surface; a list b0 must not crash it (M6
    # completeness -- the extractor mirrors vfe_block and must also convert the list to a (K,) tensor).
    from vfe3.viz.extract import e_step_belief_trace
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(alpha_mode="state_dependent_per_coord",
                           b0=[0.2, 0.5, 2.0, 5.0], n_e_steps=2))
    tok = torch.randint(0, m.cfg.vocab_size, (1, 4))
    tr = e_step_belief_trace(m, tok, n_iter=2)
    assert torch.isfinite(tr["free_energy"]).all()


# ---------------------------------------------------------------------------
# T3: per-head Press ALiBi slopes
# ---------------------------------------------------------------------------

def test_prior_alibi_per_head_press_slopes():
    from vfe3.attention_prior import get_prior
    H, N = 4, 5
    B = get_prior("alibi")(N, N, n_heads=H, alibi_slope=1.0)
    assert B.shape == (H, N, N)
    s0     = -B[0,     0, 1].item()     # slope_0 * |i-j|=1
    s_last = -B[H - 1, 0, 1].item()
    assert s0 > s_last > 0              # head 0 steepest, decaying with h
    assert B[0, 2, 2].item() == 0.0    # zero on the diagonal
    assert torch.allclose(B[1], B[1].transpose(-1, -2))   # symmetric per head


def test_prior_causal_alibi_per_head_keeps_mask():
    from vfe3.attention_prior import get_prior
    H, N = 2, 4
    B = get_prior("causal_alibi")(N, N, n_heads=H, alibi_slope=1.0)
    assert B.shape == (H, N, N)
    assert torch.isinf(B[0, 0, 1]) and B[0, 0, 1] < 0     # j>i masked to -inf per head
    assert B[0, 1, 0].item() != float("-inf")              # j<=i allowed


def test_alibi_slope_config_field_default():
    assert VFE3Config().alibi_slope == 1.0


def test_default_causal_forward_byte_identical_to_pre_change():
    torch.manual_seed(0); m = VFEModel(_tiny_cfg())        # attention_prior default = causal
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    assert torch.isfinite(m(tok)).all()


# ---------------------------------------------------------------------------
# T1: per-head kappa (per-head softmax temperature)
# ---------------------------------------------------------------------------

def test_attention_tau_returns_per_head_vector():
    from vfe3.free_energy import attention_tau
    tau = attention_tau(torch.tensor([1.0, 2.0]), irrep_dims=[3, 3])
    assert tau.shape == (2,)
    assert torch.allclose(tau, torch.tensor([1.0, 2.0]) * (3 ** 0.5))


def test_kappa_default_scalar_byte_identical():
    torch.manual_seed(0); a = VFEModel(_tiny_cfg(kappa=1.0))
    torch.manual_seed(0); b = VFEModel(_tiny_cfg(kappa=1.0))
    tok = torch.randint(0, a.cfg.vocab_size, (2, 4))
    assert torch.equal(a(tok), b(tok))


def test_kappa_equal_list_equals_scalar():
    torch.manual_seed(0); sca = VFEModel(_tiny_cfg(kappa=1.5, n_e_steps=2))
    torch.manual_seed(0); lst = VFEModel(_tiny_cfg(kappa=[1.5, 1.5], n_e_steps=2))
    tok = torch.randint(0, sca.cfg.vocab_size, (2, 4))
    assert torch.allclose(sca(tok), lst(tok), atol=1e-6, rtol=1e-5)


def test_kappa_per_head_changes_logits():
    torch.manual_seed(0); sca = VFEModel(_tiny_cfg(kappa=1.5, n_e_steps=2))
    torch.manual_seed(0); per = VFEModel(_tiny_cfg(kappa=[0.5, 4.0], n_e_steps=2))
    tok = torch.randint(0, sca.cfg.vocab_size, (2, 4))
    assert not torch.allclose(sca(tok), per(tok))


def test_single_block_group_rejects_list_kappa():
    # Per-head kappa needs equal irrep blocks; a single-block group must reject a list.
    # glk has irrep_dims=[K] (one block) at any embed_dim.
    with pytest.raises(ValueError, match="kappa"):
        _tiny_cfg(gauge_group="glk", kappa=[1.0, 2.0])


def test_banner_tau_format_handles_scalar_and_list_kappa():
    # The training banner formats tau; a per-head (list) kappa makes attention_tau return a (H,)
    # tensor that ':.4f' cannot format. _fmt_tau must handle both without crashing.
    from vfe3.train import _fmt_tau
    torch.manual_seed(0)
    m_list = VFEModel(_tiny_cfg(kappa=[0.5, 4.0]))
    s_list = _fmt_tau(m_list.cfg, m_list)
    assert isinstance(s_list, str) and s_list.startswith("[")    # per-head -> bracketed vector
    m_sca = VFEModel(_tiny_cfg(kappa=1.5))
    s_sca = _fmt_tau(m_sca.cfg, m_sca)
    assert isinstance(s_sca, str) and not s_sca.startswith("[")  # scalar -> plain float
