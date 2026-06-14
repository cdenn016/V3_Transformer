r"""Audit-fix pins for vfe3/model/model.py (V4 pos_phi guard, _amp_context mapping,
V6 diagnostics reg/entropy threading, V2 close_basis forwarding).

Each test targets the exact contract changed in model.py and is independent of the
concurrently-edited config.py / metrics.py / groups.py beyond their already-live contracts.
"""

import contextlib
import math
import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel, build_group


def _tiny_cfg(**overrides) -> VFE3Config:
    r"""Minimal model config (embed_dim=4, n_heads=2, vocab=8, seq=4, 1 layer)."""
    base = dict(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
    )
    base.update(overrides)
    return VFE3Config(**base)


# ---------------------------------------------------------------------------
# (1) V4 -- pos_phi='learned' freeze warning fires for the STRING estimators
#     ('straight_through' / 'detach'), not only the legacy detach_e_step bool.
# ---------------------------------------------------------------------------

def test_pos_phi_freeze_warning_fires_for_straight_through():
    cfg = _tiny_cfg(pos_phi="learned", e_step_gradient="straight_through")
    assert cfg.effective_e_step_gradient == "straight_through"
    with pytest.warns(UserWarning, match="pos_phi"):
        VFEModel(cfg)


def test_pos_phi_freeze_warning_fires_for_detach_estimator():
    cfg = _tiny_cfg(pos_phi="learned", e_step_gradient="detach")
    assert cfg.effective_e_step_gradient == "detach"
    with pytest.warns(UserWarning, match="pos_phi"):
        VFEModel(cfg)


def test_pos_phi_freeze_warning_silent_on_unroll_default():
    # The pure default (unroll, detach_e_step=False) must NOT warn: pos_phi_free is trainable.
    cfg = _tiny_cfg(pos_phi="learned")  # e_step_gradient defaults to 'unroll'
    assert cfg.effective_e_step_gradient == "unroll"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning would raise
        VFEModel(cfg)


# ---------------------------------------------------------------------------
# (2) _amp_context: explicit mapping (no silent fp16 fallthrough). Raises on a
#     manually-forced bad value; returns the right autocast dtype for bf16; the
#     None default returns a nullcontext.
# ---------------------------------------------------------------------------

def test_amp_context_none_is_nullcontext():
    model = VFEModel(_tiny_cfg())  # amp_dtype defaults to None
    ctx = model._amp_context(torch.device("cpu"))
    assert isinstance(ctx, contextlib.nullcontext)


def test_amp_context_bf16_dtype():
    # config rejects fp16 at construction; bf16 is the only constructible non-None value.
    model = VFEModel(_tiny_cfg(amp_dtype="bf16"))
    ctx = model._amp_context(torch.device("cpu"))
    assert isinstance(ctx, torch.autocast)
    assert ctx.fast_dtype == torch.bfloat16


def test_amp_context_raises_on_forced_bad_value():
    # Force a value config would reject (mutate post-construction) to hit the defensive raise.
    model = VFEModel(_tiny_cfg())
    model.cfg.amp_dtype = "garbage"
    with pytest.raises(ValueError, match="unsupported amp_dtype"):
        model._amp_context(torch.device("cpu"))


# ---------------------------------------------------------------------------
# (3) V6 -- diagnostics threads alpha_reg (state-dependent only) and
#     include_attention_entropy into metrics.free_energy_terms. Use a spy on the
#     module attribute (diagnostics does ``from vfe3 import metrics`` then
#     ``metrics.free_energy_terms(...)``, resolved at call time).
# ---------------------------------------------------------------------------

def _spy_free_energy_terms(monkeypatch):
    import vfe3.metrics as M
    seen = {}
    orig = M.free_energy_terms

    def spy(*a, **k):
        seen["alpha_reg"] = k.get("alpha_reg")
        seen["iae"] = k.get("include_attention_entropy")
        return orig(*a, **k)

    monkeypatch.setattr(M, "free_energy_terms", spy)
    return seen


def test_diagnostics_threads_reg_when_state_dependent(monkeypatch):
    cfg = _tiny_cfg(lambda_alpha_mode="state_dependent_per_coord")
    model = VFEModel(cfg)
    token_ids = torch.zeros((1, cfg.max_seq_len), dtype=torch.long)
    seen = _spy_free_energy_terms(monkeypatch)
    d = model.diagnostics(token_ids)
    assert seen["alpha_reg"] is not None
    assert seen["iae"] == cfg.include_attention_entropy
    assert "total" in d


def test_diagnostics_reg_is_none_when_constant(monkeypatch):
    cfg = _tiny_cfg(lambda_alpha_mode="constant")
    model = VFEModel(cfg)
    token_ids = torch.zeros((1, cfg.max_seq_len), dtype=torch.long)
    seen = _spy_free_energy_terms(monkeypatch)
    model.diagnostics(token_ids)
    assert seen["alpha_reg"] is None
    assert seen["iae"] == cfg.include_attention_entropy


def test_diagnostics_exposes_wilson_holonomy_zero_on_flat_default():
    # The Wilson-action density 1 - Re Tr(H)/K is exposed alongside the Frobenius holonomy and is
    # ~0 on the default flat phi-cocycle (every triangle closes), the complement of the certificate.
    cfg = _tiny_cfg()
    model = VFEModel(cfg)
    token_ids = torch.zeros((1, cfg.max_seq_len), dtype=torch.long)
    d = model.diagnostics(token_ids)
    assert "holonomy_wilson" in d
    assert math.isfinite(d["holonomy_wilson"])
    assert abs(d["holonomy_wilson"]) < 1e-3


# ---------------------------------------------------------------------------
# (4) V2 -- close_basis wiring. Default path is byte-identical; the new close=True
#     branch (cross_couplings chain) constructs without error.
# ---------------------------------------------------------------------------

def test_default_group_unchanged_by_close_basis_field():
    # Default (close_basis=None, cross_couplings=None) -> close resolves to False -> same group.
    g_default = build_group(_tiny_cfg())
    g_explicit_none = build_group(_tiny_cfg(close_basis=None))
    assert g_default.name == g_explicit_none.name
    assert g_default.irrep_dims == g_explicit_none.irrep_dims
    assert torch.equal(g_default.generators, g_explicit_none.generators)
    # And constructing a full model on the default succeeds.
    VFEModel(_tiny_cfg())


def test_close_basis_true_with_cross_couplings_constructs():
    # Exercises the new close=True branch through to the block_glk builder.
    cfg = _tiny_cfg(gauge_group="block_glk", cross_couplings=[(0, 1)], close_basis=True)
    g = build_group(cfg)
    assert g.name == "block_glk"
    assert g.generators.shape[-1] == cfg.embed_dim  # (n_gen, K, K)
    # Full model build also succeeds with the closed cross-coupled basis.
    VFEModel(cfg)
