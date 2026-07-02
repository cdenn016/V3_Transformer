r"""Round-3 geometry wiring pins (audit 2026-07-01 round-3).

(a) register_transport fails closed on duplicate keys; override=True refreshes the
    needs-set metadata (no stale membership survives a replacement).
(b) build_belief_transport routes belief state by REGISTRY membership
    (_TRANSPORT_NEEDS_MU / _TRANSPORT_NEEDS_SIGMA), never by literal mode names.
(c) stable_matrix_exp_pair's clamp_monitor diagnostic warns iff opted in.
(d) VFE3Config.transport_clamp_monitor exists, defaults False, and round-trips.
"""

import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    _TRANSPORT_BATCH_INDEPENDENT,
    _TRANSPORT_NEEDS_MU,
    _TRANSPORT_NEEDS_SIGMA,
    _TRANSPORTS,
    register_transport,
    stable_matrix_exp_pair,
)
from vfe3.inference.e_step import build_belief_transport


def test_register_transport_duplicate_fails_closed():
    assert "flat" in _TRANSPORTS
    with pytest.raises(KeyError, match="already registered"):
        @register_transport("flat")
        def _dup(phi, group, **kwargs):          # pragma: no cover - never registered
            raise AssertionError("unreachable")
    assert _TRANSPORTS["flat"].__name__ == "_build_flat"   # registry untouched by the failed attempt


def test_register_transport_override_refreshes_needs_sets():
    orig = _TRANSPORTS["flat"]
    assert "flat" not in _TRANSPORT_NEEDS_MU
    assert "flat" not in _TRANSPORT_NEEDS_SIGMA
    try:
        @register_transport("flat", needs_mu=True, override=True)
        def _dummy_mu(phi, group, **kwargs):
            return orig(phi, group, **kwargs)
        assert _TRANSPORTS["flat"] is _dummy_mu
        assert "flat" in _TRANSPORT_NEEDS_MU               # new flags applied

        # A second override WITHOUT needs_mu must not inherit stale membership.
        @register_transport("flat", override=True)
        def _dummy_plain(phi, group, **kwargs):
            return orig(phi, group, **kwargs)
        assert "flat" not in _TRANSPORT_NEEDS_MU           # stale membership discarded
    finally:
        register_transport("flat", override=True)(orig)    # restore builder + (flag-free) metadata
    assert _TRANSPORTS["flat"] is orig
    assert "flat" not in _TRANSPORT_NEEDS_MU
    assert "flat" not in _TRANSPORT_NEEDS_SIGMA
    assert "flat" not in _TRANSPORT_BATCH_INDEPENDENT


def test_build_belief_transport_forwards_mu_to_needs_mu_builder():
    grp = get_group("so_k")(K=4)
    g = torch.Generator().manual_seed(0)
    B, N, K = 1, 3, 4
    phi = torch.zeros(B, N, grp.generators.shape[0])
    mu = torch.randn(B, N, K, generator=g)
    sigma = torch.rand(B, N, K, generator=g) + 0.1
    rec: dict = {}
    name = "_round3_probe_needs_mu"
    try:
        @register_transport(name, needs_mu=True)
        def _probe(phi_, group_, **kwargs):
            rec.update(kwargs)
            omega = torch.eye(K).expand(B, N, N, K, K).contiguous()   # flat-shaped dict
            return {"Omega": omega}
        out = build_belief_transport(phi, grp, transport_mode=name, mu=mu, sigma=sigma)
        assert rec["mu"] is mu                             # needs_mu -> live means forwarded
        assert rec["sigma"] is None                        # not needs_sigma -> gated to None
        assert out.shape == (B, N, N, K, K)
    finally:
        _TRANSPORTS.pop(name, None)
        _TRANSPORT_NEEDS_MU.discard(name)
        _TRANSPORT_NEEDS_SIGMA.discard(name)
        _TRANSPORT_BATCH_INDEPENDENT.discard(name)


def test_build_belief_transport_gates_mu_to_none_for_flat():
    grp = get_group("so_k")(K=4)                           # single block -> dense (non-fused) path
    orig = _TRANSPORTS["flat"]
    g = torch.Generator().manual_seed(1)
    phi = 0.1 * torch.randn(1, 3, grp.generators.shape[0], generator=g)
    mu = torch.randn(1, 3, 4, generator=g)
    rec: dict = {}
    try:
        @register_transport("flat", override=True)
        def _recording_flat(phi_, group_, **kwargs):
            rec.update(kwargs)
            return orig(phi_, group_, **kwargs)
        out = build_belief_transport(phi, grp, transport_mode="flat", mu=mu)
        assert rec["mu"] is None                           # flat not in _TRANSPORT_NEEDS_MU
        assert rec["sigma"] is None
        assert out.shape == (1, 3, 3, 4, 4)
    finally:
        register_transport("flat", override=True)(orig)
    assert _TRANSPORTS["flat"] is orig


def test_stable_matrix_exp_pair_clamp_monitor_warns_when_clamp_fires():
    m = 20.0 * torch.eye(4).expand(2, 4, 4)                # ||M||_F = 40 > max_norm = 15
    with pytest.warns(RuntimeWarning, match="Frobenius clamp"):
        stable_matrix_exp_pair(m, clamp_monitor=True)


def test_stable_matrix_exp_pair_clamp_monitor_off_is_silent():
    m = 20.0 * torch.eye(4).expand(2, 4, 4)                # clamp fires, but the monitor is off
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # any warning -> test failure
        stable_matrix_exp_pair(m)


def test_config_transport_clamp_monitor_defaults_false_and_roundtrips():
    assert VFE3Config().transport_clamp_monitor is False
    assert VFE3Config(transport_clamp_monitor=True).transport_clamp_monitor is True
