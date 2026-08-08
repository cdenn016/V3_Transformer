r"""``family_chunked`` decode: bound the FULL-family functional workspace, and gate the checkpoint
on what not-checkpointing actually retains (audit 2026-08-07).

Two defects, both reachable only when ``decode_mode='family_chunked'`` meets a full family, and both
measured on an RTX 5090 before the fix.

**(1) One ``decode_chunk_size`` knob, two workspace units.** Five decode kernels read
``decode_chunk_size`` (prior_bank.py:1185, 1382, 1496, 1580, 1673) and it is sized for the ones whose
per-slice workspace is ``(B, N, Vc)``. A full family's is ``(B, N, Vc, K, K)`` -- ``K*K`` times larger
-- and nothing scaled the value on the way in: ``config.py:2515`` validates only ``>= 1``, and
``model.py:1609`` passes no ``chunk_size`` override, so the raw integer reached the loop. At the live
``B=64, N=128, K=20, V=50257, decode_chunk_size=8192`` that is a measured
``RuntimeError: CUDA out of memory. Tried to allocate 100.00 GiB`` on the FIRST slice, against a
6887 MiB peak for the whole ``full_chunked`` decode at the identical shape. The
``train_vfe3.py:161`` comment ("set chunks to 512 or default/K^2") was an unenforced suggestion, so
the failure mode was a hand-tuning obligation the config never stated.

**(2) The checkpoint gate measured a quantity the knob cannot move.** ``_decode_ce_should_checkpoint``
was handed ONE slice's bytes, but declining to checkpoint retains EVERY slice's workspace into
backward at once. That total is ``B*N*V*inner*itemsize`` -- invariant under the slice width. So
narrowing the slice to bound the transient (defect 1's fix) would have walked the per-slice estimate
under the 2 GiB threshold and silently switched checkpointing OFF, retaining the whole vocabulary
instead of one slice: strictly worse than the OOM it was fixing. The estimate was also 2x low on its
own terms, counting one ``(B, N, Vc, K, K)`` tensor where ``_full_gaussian_kl_terms`` allocates two
simultaneously (the forward and back substitutions at families/gaussian.py:117-118).

Pins: (a) the ceiling binds for a full family and is inert for a diagonal one; (b) narrowing is
value- and gradient-identical, because slicing a logsumexp is a tiling choice; (c) the checkpoint
decision is invariant under ``decode_chunk_size``; (d) the workspace estimate counts both solve
buffers; (e) the slice width never reaches zero, however small the budget.
"""

import pytest
import torch

import vfe3.model.prior_bank as prior_bank
from vfe3.model.prior_bank import (
    DECODE_CE_FAMILY_WORKSETS,
    _decode_ce_chunk_activation_bytes,
    _decode_ce_family_effective_chunk,
    _decode_ce_should_checkpoint,
)

from tests.test_amp import _tiny_model


def _bank(**kw):
    model = _tiny_model(gauge_group="block_glk", n_heads=2, family="gaussian_full",
                        decode_mode="family_chunked", **kw)
    return model, model.prior_bank


def _inputs(model):
    B, N, K = 2, 5, model.cfg.embed_dim
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(B, N, K, generator=g)
    sigma = torch.eye(K).expand(B, N, K, K).clone()
    return mu, sigma, torch.randint(0, 20, (B, N), generator=g)


# -- (a) the ceiling binds for a full family, and is inert for a diagonal one -------------------

def test_effective_chunk_narrows_a_full_family_to_the_workspace_ceiling():
    r"""At the live shape the raw 8192 is cut to whatever keeps two (B,N,Vc,K,K) buffers under 1 GiB."""
    ref = torch.empty(64, 128, 20)                       # live B, N; fp32
    inner = 20 * 20
    got = _decode_ce_family_effective_chunk(ref, 8192, inner)

    per_entry = 64 * 128 * inner * DECODE_CE_FAMILY_WORKSETS * 4
    assert got == prior_bank.DECODE_CE_FAMILY_WORKSPACE_BYTES // per_entry
    assert got < 8192, "the ceiling did not bind at the shape that measured a 100 GiB allocation"
    # The bound it advertises actually holds.
    assert got * per_entry <= prior_bank.DECODE_CE_FAMILY_WORKSPACE_BYTES


def test_effective_chunk_is_inert_for_a_diagonal_family():
    r"""inner == 1 is every (B, N, Vc) kernel: the knob keeps its exact meaning, byte-identical."""
    ref = torch.empty(64, 128, 20)
    assert _decode_ce_family_effective_chunk(ref, 8192, 1) == 8192
    assert _decode_ce_family_effective_chunk(ref, 3, 1) == 3


def test_effective_chunk_never_returns_zero():
    r"""A budget smaller than one vocabulary entry must still make progress, not hang on range(0,V,0)."""
    ref = torch.empty(4096, 4096, 8)
    assert _decode_ce_family_effective_chunk(ref, 8192, 64 * 64) >= 1


def test_effective_chunk_never_widens_the_request():
    r"""The ceiling is a cap, not a target: a user asking for a narrow slice keeps it."""
    ref = torch.empty(2, 5, 20)
    assert _decode_ce_family_effective_chunk(ref, 3, 20 * 20) == 3


# -- (b) narrowing is value- and gradient-identical ---------------------------------------------

@pytest.fixture
def spy(monkeypatch):
    r"""Record the vocab width of every functional call the decode makes."""
    widths = []
    real = prior_bank.get_functional

    def _wrapped(name):
        fn = real(name)

        def _spy(q, p, **kw):
            out = fn(q, p, **kw)
            widths.append(out.shape[-1])
            return out
        return _spy

    monkeypatch.setattr(prior_bank, "get_functional", _wrapped)
    return widths


def _one_entry_budget(model):
    r"""Exactly one vocabulary entry's workspace at the ``_inputs`` shape -> maximal retiling.

    Derived from the model's real K rather than hard-coded: ``_tiny_model`` defaults to
    ``embed_dim=4``, and a budget written against a guessed K silently fails to bind, which makes
    every value-equality assertion below compare a tiling against itself.
    """
    B, N, K = 2, 5, model.cfg.embed_dim
    return B * N * K * K * DECODE_CE_FAMILY_WORKSETS * 4      # fp32


def test_narrowed_slices_match_the_unnarrowed_ce(monkeypatch, spy):
    r"""Forcing the ceiling to bind must not move the loss: slicing a logsumexp is a tiling choice."""
    model, pb = _bank(decode_chunk_size=20)
    mu, sigma, targets = _inputs(model)
    V = pb.vocab_size

    wide = pb.decode_ce_family_chunked(mu, sigma, targets)
    wide_widths = list(spy)
    spy.clear()

    # Squeeze the budget so the same call is forced into many narrow slices.
    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES", _one_entry_budget(model))
    narrow = pb.decode_ce_family_chunked(mu, sigma, targets)

    # The retiling MUST actually have happened, or the equality below proves nothing.
    assert max(narrow_w := spy) == 1 and len(narrow_w) == V, f"ceiling did not bind: {narrow_w}"
    assert len(wide_widths) < len(narrow_w), "the two runs used the same tiling"
    assert sum(wide_widths) == sum(narrow_w) == V, "a tiling dropped vocabulary entries"

    assert torch.allclose(wide, narrow, rtol=0, atol=1e-6), f"{wide.item()} != {narrow.item()}"


def test_narrowed_slices_match_the_unnarrowed_gradient(monkeypatch, spy):
    r"""The gradient to the prior tables must survive the retiling too, not just the value."""
    model, pb = _bank(decode_chunk_size=20)
    mu, sigma, targets = _inputs(model)

    def _grad():
        pb.zero_grad(set_to_none=True)
        pb.decode_ce_family_chunked(mu, sigma, targets).backward()
        return pb.mu_embed.grad.clone()

    wide = _grad()
    n_wide = len(spy)
    spy.clear()

    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES", _one_entry_budget(model))
    narrow = _grad()

    assert len(spy) > n_wide, "the ceiling did not bind -- the comparison would be vacuous"
    assert torch.allclose(wide, narrow, rtol=0, atol=1e-6)
    assert torch.isfinite(narrow).all()
    assert narrow.abs().sum() > 0, "a zero gradient would satisfy the equality vacuously"


def test_declining_the_prior_hoist_still_promotes_each_slice(monkeypatch, spy):
    r"""The (V, K, K) promotion is hoisted only when it is small; the fallback must still promote.

    A full family needs a ``(Vc, K, K)`` prior slice. When the whole-table hoist is declined (the
    table would be the new memory problem -- 8.9 GB at K=210) the loop must ``diag_embed`` each
    slice itself. Slicing the UN-promoted ``(V, K)`` table instead hands the functional a diagonal
    dispersion for a full family, which is a shape error at best and a silently wrong divergence at
    worst -- so this pins the fallback against the hoisted path, not merely against "it ran".
    """
    model, pb = _bank(decode_chunk_size=20)
    mu, sigma, targets = _inputs(model)

    hoisted = pb.decode_ce_family_chunked(mu, sigma, targets)
    spy.clear()

    # One byte under the promoted table's size: the hoist is declined, the loop must promote.
    V, K = pb.vocab_size, model.cfg.embed_dim
    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES", V * K * K * 4 - 1)
    fallback = pb.decode_ce_family_chunked(mu, sigma, targets)

    assert spy, "the decode made no functional call"
    assert torch.allclose(hoisted, fallback, rtol=0, atol=1e-6)
    assert torch.isfinite(fallback)


# -- (c) the checkpoint decision does not depend on the slice width -----------------------------

def test_checkpoint_gate_is_invariant_under_chunk_size():
    r"""Not-checkpointing retains every slice, so the decision is keyed on B*N*V, not B*N*Vc.

    This is the regression that made defect 1's fix dangerous: under the old per-slice estimate,
    narrowing the slice walked the number under the threshold and turned checkpointing off.
    """
    ref = torch.empty(64, 128, 20)
    V, inner = 50257, 20 * 20 * DECODE_CE_FAMILY_WORKSETS

    total = _decode_ce_chunk_activation_bytes(ref, V, inner=inner)
    decisions = {
        _decode_ce_should_checkpoint("auto", True, total)          # gate sees the SAME total
        for _ in (1, 20, 512, 8192)                                 # regardless of slice width
    }
    assert decisions == {True}, "the whole-vocabulary retention must still trip the auto threshold"

    # And the old per-slice estimate really would have flipped it -- this is not a hypothetical.
    per_slice = _decode_ce_chunk_activation_bytes(ref, 40, inner=inner)
    assert not _decode_ce_should_checkpoint("auto", True, per_slice)


# -- (d) the estimate counts BOTH triangular-solve buffers --------------------------------------

def test_workspace_estimate_counts_both_solve_buffers():
    r"""_full_gaussian_kl_terms holds Y and Z simultaneously; counting one halved the figure."""
    assert DECODE_CE_FAMILY_WORKSETS == 2
    ref = torch.empty(8, 32, 20)
    one = _decode_ce_chunk_activation_bytes(ref, 512, inner=400)
    both = _decode_ce_chunk_activation_bytes(ref, 512, inner=400 * DECODE_CE_FAMILY_WORKSETS)
    assert both == 2 * one


# -- (f) decode_av_precision is reported inert on the family route -------------------------------

def test_decode_av_precision_is_reported_inert_on_the_family_route():
    r"""The family route never calls _decode_av, so tuning its precision must not fail silently.

    ``decode_av_precision`` is accepted, validated (config.py:2379) and published process-wide
    (model.py:258), but its only reader is ``_decode_av`` (prior_bank.py:180), which the
    family-consistent kernels do not call -- they hand the pair to the registered functional, whose
    precision is ``full_cov_kl_precision``. Before audit 2026-08-07 the dead-knob oracle had no rule
    for it, so the setting was silently ignored.
    """
    from vfe3.config import VFE3Config

    base = dict(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                use_prior_bank=True, family="gaussian_full", gauge_group="block_glk")

    def _inert_mentions(mode):
        with pytest.warns(Warning) as record:
            VFE3Config(**base, decode_mode=mode, decode_av_precision="fp64")
        return any("decode_av_precision" in str(r.message) for r in record)

    assert _inert_mentions("family_chunked"), "the family route must report the knob inert"
    # full_chunked genuinely reads it, so it must NOT be reported inert there.
    assert not _inert_mentions("full_chunked"), "full_chunked reads _decode_av -- not inert"


# -- (e) the route still runs end to end at a binding ceiling ------------------------------------

@pytest.mark.parametrize("multiple", [1, 8])
def test_forward_backward_is_finite_at_a_binding_ceiling(monkeypatch, multiple):
    model, pb = _bank(decode_chunk_size=8192)
    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES",
                        multiple * _one_entry_budget(model))
    mu, sigma, targets = _inputs(model)

    ce = pb.decode_ce_family_chunked(mu, sigma, targets)
    ce.backward()

    assert torch.isfinite(ce)
    assert torch.isfinite(pb.mu_embed.grad).all()
