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
