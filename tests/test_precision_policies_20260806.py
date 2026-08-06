r"""Numerics policies added by the WAVE3 Tier C pass (audit 2026-08-06 C2, C3, C6).

Each is a new toggle whose DEFAULT reproduces the historical behavior exactly, so every run
already on disk stays bit-reproducible; the opt-in is what changes numerics.

C2/F1  ``full_cov_kl_precision="fp32_escalate_cond"`` -- the existing ``"fp32_escalate"`` keys its
       escalation on the Cholesky ``ok`` mask, which is effectively always True (the jitter ladder
       repairs every float32 PD loss at t=0), so that policy is unconditional float32. The new
       spelling keys on a conditioning proxy instead.
C3/F18 ``safe_cholesky_jitter_mode="relative"`` -- the ridge ``eps*10^t`` is absolute, so it doubles
       an eigenvalue sitting on the ``eps`` floor and is a 1e-8 no-op at ``sigma_max``.
C6/F29 ``mu_trust_cholesky_rounds`` -- at 0 one marginally non-PD ``sigma_q`` silently drops to a
       NON-equivariant diagonal whitening; a positive value keeps it on the equivariant path.
"""

import pytest
import torch

import vfe3.families.gaussian as gaussian_mod
import vfe3.numerics as numerics_mod
from vfe3.config import VFE3Config
from vfe3.families.gaussian import FullGaussian, _full_gaussian_kl_terms
from vfe3.numerics import (
    apply_mu_trust_region,
    mu_trust_fallback_elements,
    reset_mu_trust_fallback_elements,
    safe_cholesky,
    set_mu_trust_cholesky_rounds,
    set_safe_cholesky_jitter_mode,
)

K = 20


@pytest.fixture(autouse=True)
def _restore():
    kl = gaussian_mod.full_cov_kl_precision()
    jitter = numerics_mod.safe_cholesky_jitter_mode()
    rounds = numerics_mod.mu_trust_cholesky_rounds()
    reset_mu_trust_fallback_elements()
    yield
    gaussian_mod.set_full_cov_kl_precision(kl)
    set_safe_cholesky_jitter_mode(jitter)
    set_mu_trust_cholesky_rounds(rounds)
    reset_mu_trust_fallback_elements()


def _spd(cond_exp, seed=0):
    r"""(K, K) SPD with a prescribed condition number, in float32 STORAGE."""
    g = torch.Generator().manual_seed(seed)
    w = torch.logspace(0, -cond_exp, K, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(K, K, generator=g, dtype=torch.float64))
    s = q @ torch.diag(w) @ q.T
    return (0.5 * (s + s.T)).float()


def _kl(mu, sigma, policy):
    gaussian_mod.set_full_cov_kl_precision(policy)
    if policy == "fp64":
        mu, sigma = mu.double(), sigma.double()
    q = FullGaussian(mu, sigma.unsqueeze(0))
    t = FullGaussian(mu + 0.1, sigma.unsqueeze(0))
    return float(q.renyi_closed_form(t, alpha=1.0, kl_max=1e9, eps=1e-6)[0])


# -- C2: the conditioning-keyed escalation ----------------------------------------------------


def test_ok_mask_trigger_is_effectively_dead():
    r"""The premise of C2: safe_cholesky's ladder repairs everything, so `ok` never fires."""
    failures = 0
    for cond_exp in (8, 10, 12):
        for seed in range(15):
            sigma = _spd(cond_exp, seed)
            _, ok, _ = _full_gaussian_kl_terms(
                torch.zeros(1, K), sigma.unsqueeze(0), torch.zeros(1, K), sigma.unsqueeze(0),
                K, 1e-6)
            failures += int(not bool(ok.all()))
    assert failures == 0, "if this ever fires, the C2 rationale needs re-measuring"


@pytest.mark.parametrize("cond_exp,should_fire", [(3, False), (5, False), (6, True), (8, True)])
def test_conditioning_trigger_fires_where_it_should(cond_exp, should_fire):
    sigma = _spd(cond_exp)
    _, _, inv_cond = _full_gaussian_kl_terms(
        torch.zeros(1, K), sigma.unsqueeze(0), torch.zeros(1, K), sigma.unsqueeze(0), K, 1e-6)
    fires = bool((inv_cond < gaussian_mod._FULL_COV_KL_COND_FLOOR).any())
    assert fires is should_fire


def test_escalation_removes_the_float32_arithmetic_error():
    r"""Reference is float64 ARITHMETIC on the SAME float32-rounded inputs, which is what the
    escalation can actually recover -- float32 STORAGE loss at high conditioning is irrecoverable
    by any compute precision, and this policy does not claim otherwise."""
    g = torch.Generator().manual_seed(1)
    mu = torch.randn(1, K, generator=g)
    for cond_exp in (6, 7, 8):
        sigma = _spd(cond_exp, seed=cond_exp)
        reference = _kl(mu, sigma, "fp64")
        plain = _kl(mu, sigma, "fp32_escalate")
        keyed = _kl(mu, sigma, "fp32_escalate_cond")
        rel = lambda v: abs(v - reference) / max(abs(reference), 1e-12)   # noqa: E731
        assert rel(keyed) < 1e-6, f"cond 1e{cond_exp}: escalation did not recover the arithmetic"
        assert rel(keyed) < rel(plain) / 100.0


@pytest.mark.parametrize("cond_exp", [3, 4, 5])
def test_below_the_floor_the_two_float32_policies_agree_bitwise(cond_exp):
    g = torch.Generator().manual_seed(2)
    mu = torch.randn(1, K, generator=g)
    sigma = _spd(cond_exp, seed=cond_exp)
    assert _kl(mu, sigma, "fp32_escalate") == _kl(mu, sigma, "fp32_escalate_cond")


def test_fp64_policy_is_unaffected_by_the_new_spelling():
    cfg = VFE3Config(vocab_size=32, embed_dim=K, n_heads=2, max_seq_len=8)
    assert cfg.full_cov_kl_precision == "fp64"
    with pytest.raises(ValueError, match="full_cov_kl_precision"):
        VFE3Config(vocab_size=32, embed_dim=K, n_heads=2, max_seq_len=8,
                   full_cov_kl_precision="nonsense")


# -- C3: relative jitter -----------------------------------------------------------------------


def test_absolute_is_the_default_and_is_byte_identical():
    g = torch.Generator().manual_seed(3)
    a = torch.randn(4, 6, 6, generator=g)
    m = a @ a.transpose(-1, -2) + 6 * torch.eye(6)
    m[1] = -3.0 * torch.eye(6)                      # forces the ladder
    set_safe_cholesky_jitter_mode("absolute")
    factor_a, ok_a = safe_cholesky(m, eps=1e-6, rounds=5)
    factor_b, ok_b = safe_cholesky(m, eps=1e-6, rounds=5, jitter_mode="absolute")
    assert torch.equal(factor_a, factor_b) and torch.equal(ok_a, ok_b)


def test_relative_and_absolute_agree_when_the_ladder_never_fires():
    g = torch.Generator().manual_seed(4)
    a = torch.randn(4, 6, 6, generator=g)
    m = a @ a.transpose(-1, -2) + 6 * torch.eye(6)   # healthy, ladder inert
    assert torch.equal(
        safe_cholesky(m, eps=1e-6, rounds=5, jitter_mode="absolute")[0],
        safe_cholesky(m, eps=1e-6, rounds=5, jitter_mode="relative")[0])


def test_relative_ridge_scales_with_the_matrix():
    r"""The defect: one absolute ridge doubles an eigenvalue at the eps floor and is a no-op at
    sigma_max. Relative gives both the same treatment."""
    floor = torch.full((1, 4, 4), 0.0)
    floor[0] = 1e-6 * torch.eye(4)
    big = 100.0 * torch.eye(4).unsqueeze(0)
    set_safe_cholesky_jitter_mode("relative")
    for matrix in (floor, big):
        factor, ok = safe_cholesky(matrix, eps=1e-6, rounds=3)
        assert bool(ok.all()) and torch.isfinite(factor).all()


def test_jitter_mode_validates():
    with pytest.raises(ValueError, match="jitter_mode"):
        safe_cholesky(torch.eye(3).unsqueeze(0), rounds=1, jitter_mode="sideways")
    with pytest.raises(ValueError, match="safe_cholesky_jitter_mode"):
        VFE3Config(vocab_size=32, embed_dim=K, n_heads=2, max_seq_len=8,
                   safe_cholesky_jitter_mode="sideways")


# -- C6: mu-trust-region rounds ----------------------------------------------------------------


def test_rounds_rescue_the_element_onto_the_equivariant_path():
    dim = 5
    sigma = torch.eye(dim).expand(4, dim, dim).clone()
    sigma[2] = torch.eye(dim) * 1e-9 - 1e-8 * torch.ones(dim, dim)      # marginally non-PD
    delta = 0.1 * torch.randn(4, dim, generator=torch.Generator().manual_seed(5))

    counts = {}
    for rounds in (0, 5):
        set_mu_trust_cholesky_rounds(rounds)
        reset_mu_trust_fallback_elements()
        out = apply_mu_trust_region(delta, sigma, trust=1.0, mode="ball",
                                    is_diagonal=False, eps=1e-6, family="gaussian_full")
        assert torch.isfinite(out).all()
        counts[rounds] = mu_trust_fallback_elements()

    assert counts[0] == 1, "the default must still take the non-equivariant fallback"
    assert counts[5] == 0, "rounds>0 must keep the element on the equivariant path"


def test_rounds_validate():
    with pytest.raises(ValueError, match="mu_trust_cholesky_rounds"):
        VFE3Config(vocab_size=32, embed_dim=K, n_heads=2, max_seq_len=8,
                   mu_trust_cholesky_rounds=-1)
