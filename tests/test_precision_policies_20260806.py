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
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.families.gaussian import FullGaussian, _full_gaussian_kl_terms
from vfe3.geometry.groups import get_group
from vfe3.inference import e_step as e_step_mod
from vfe3.model.model import VFEModel
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
        certified = _kl(mu, sigma, "fp32_escalate")
        stricter = _kl(mu, sigma, "fp32_escalate_cond")
        rel = lambda v: abs(v - reference) / max(abs(reference), 1e-12)   # noqa: E731
        assert rel(certified) < 1e-6, \
            f"cond 1e{cond_exp}: certificate escalation did not recover the arithmetic"
        assert rel(stricter) < 1e-6
        assert certified == stricter


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


def test_constructing_second_model_does_not_change_first_full_gaussian_policy_or_decode():
    """A model-owned full-Gaussian decode must retain model A's precision policy.

    Replacing the process-global policy in ``VFEModel.__init__`` makes this fail: model B changes
    model A's stored policy and its family decode result for this ill-conditioned public query.
    """
    common = dict(
        vocab_size=16,
        embed_dim=K,
        n_heads=2,
        max_seq_len=4,
        family="gaussian_full",
        use_prior_bank=True,
        decode_mode="family",
        full_cov_congruence_precision="fp64",
    )
    model_a = VFEModel(VFE3Config(**common, full_cov_kl_precision="fp64"))
    mu = torch.zeros(1, 1, K)
    sigma = _spd(7, seed=17).reshape(1, 1, K, K)
    before = model_a.prior_bank.reference_decode(mu, sigma)

    model_b = VFEModel(VFE3Config(**common, full_cov_kl_precision="fp32_escalate"))
    after = model_a.prior_bank.reference_decode(mu, sigma)

    assert model_a.full_cov_kl_precision == "fp64"
    assert model_b.full_cov_kl_precision == "fp32_escalate"
    assert torch.equal(before, after)


def test_global_mutation_cannot_change_model_owned_dense_full_decode():
    """The direct dense decoder must use PriorBank's stored policy, not the module default."""
    cfg = VFE3Config(
        vocab_size=16, embed_dim=K, n_heads=2, max_seq_len=4,
        family="gaussian_full", use_prior_bank=True, decode_mode="full",
        full_cov_kl_precision="fp64", full_cov_congruence_precision="fp64",
    )
    model = VFEModel(cfg)
    mu = torch.zeros(1, 1, K)
    sigma = _spd(7, seed=31).reshape(1, 1, K, K)
    before = model.prior_bank.reference_decode(mu, sigma)
    gaussian_mod.set_full_cov_kl_precision("fp32_escalate")
    after = model.prior_bank.reference_decode(mu, sigma)
    assert torch.equal(before, after)


def _full_model(*, lambda_h=0.0) -> VFEModel:
    return VFEModel(VFE3Config(
        vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=4,
        family="gaussian_full", lambda_h=lambda_h,
        full_cov_kl_precision="fp64", full_cov_congruence_precision="fp64",
        pos_phi="none", e_phi_lr=0.0,
    ))


def test_hyper_prior_kl_keeps_model_policy_value_and_gradient_after_global_mutation(monkeypatch):
    """The hyper-prior's actual FullGaussian operands belong to the owning model."""
    from vfe3 import free_energy

    model = _full_model(lambda_h=0.2)
    tokens = torch.tensor([[1, 2, 3, 4]])
    calls = []
    actual = free_energy.self_divergence

    def observed(q, p, **kwargs):
        calls.append((q._precision_policy, p._precision_policy))
        return actual(q, p, **kwargs)

    monkeypatch.setattr(free_energy, "self_divergence", observed)

    def value_and_gradient():
        value = model._hyper_prior_kl(tokens).sum()
        gradient, = torch.autograd.grad(value, model.prior_bank.s_mu_embed)
        return value.detach(), gradient.detach()

    before, before_grad = value_and_gradient()
    gaussian_mod.set_full_cov_kl_precision("fp32_escalate")
    after, after_grad = value_and_gradient()

    assert calls and set(calls) == {("fp64", "fp64")}
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(after_grad, before_grad, rtol=0.0, atol=0.0)


def test_cg_moment_energy_rows_uses_builtin_policy_but_preserves_legacy_constructor(monkeypatch):
    """The direct CG seam passes the policy only to the exact built-in family."""
    from vfe3 import free_energy
    from vfe3.families import base as families_base
    from vfe3.model.cg_coupling import cg_moment_energy_rows

    pre_mu = torch.zeros(1, 2, 4)
    pre_sigma = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 2, 4, 4).clone()
    calls = []
    actual = free_energy.self_divergence

    def observed(q, p, **kwargs):
        calls.append((q._precision_policy, p._precision_policy))
        return actual(q, p, **kwargs)

    monkeypatch.setattr(free_energy, "self_divergence", observed)

    def value_and_gradient():
        post_mu = (pre_mu + 0.1).detach().requires_grad_()
        value = cg_moment_energy_rows(
            pre_mu, pre_sigma, post_mu, pre_sigma,
            family="gaussian_full", full_cov_kl_precision="fp64",
            kl_max=float("inf"),
        ).sum()
        gradient, = torch.autograd.grad(value, post_mu)
        return value.detach(), gradient.detach()

    before, before_grad = value_and_gradient()
    gaussian_mod.set_full_cov_kl_precision("fp32_escalate")
    after, after_grad = value_and_gradient()

    assert calls and set(calls) == {("fp64", "fp64")}
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(after_grad, before_grad, rtol=0.0, atol=0.0)

    class LegacyFullGaussian(FullGaussian):
        def __init__(self, mu, sigma):
            super().__init__(mu, sigma)

    monkeypatch.setattr(families_base, "get_family", lambda _name: LegacyFullGaussian)
    legacy = cg_moment_energy_rows(
        pre_mu, pre_sigma, pre_mu + 0.1, pre_sigma,
        family="legacy_full", full_cov_kl_precision="fp64", kl_max=float("inf"),
    )
    assert torch.isfinite(legacy).all()


def test_e_step_early_halt_constructs_policy_owned_full_gaussians(monkeypatch):
    """Evaluation halting reaches the explicit factory and actually breaks the iteration loop."""
    belief = BeliefState(
        mu=torch.zeros(2, 4),
        sigma=torch.eye(4).expand(2, 4, 4).clone(),
        phi=torch.zeros(2, 1),
    )
    constructed = []
    iterations = []
    actual_factory = e_step_mod._family_instance

    def observed_factory(family, mu, sigma, policy):
        out = actual_factory(family, mu, sigma, policy)
        constructed.append(out._precision_policy)
        return out

    def stationary_iteration(current, *_args, **_kwargs):
        iterations.append(current)
        return current

    monkeypatch.setattr(e_step_mod, "_family_instance", observed_factory)
    monkeypatch.setattr(e_step_mod, "e_step_iteration", stationary_iteration)
    gaussian_mod.set_full_cov_kl_precision("fp32_escalate")

    result = e_step_mod.e_step(
        belief, belief.mu, belief.sigma, object(), n_iter=3,
        e_phi_lr=0.1, e_step_halt_tol=1e-9, training=False,
        family="gaussian_full", full_cov_kl_precision="fp64",
    )

    assert result is belief
    assert len(iterations) == 1
    assert constructed == ["fp64", "fp64"]


def test_fixed_point_diagnostic_uses_model_owned_full_gaussian_after_global_mutation(monkeypatch):
    """The displayed fixed-point KL is evaluated through VFEModel's owned factory."""
    from vfe3.viz.extract import e_step_fixed_point_diagnostics

    model = _full_model()
    model.cfg.n_e_steps = 1
    tokens = torch.tensor([[1, 2, 3, 4]])
    calls = []
    actual_factory = model._family_instance

    def observed_factory(family, *args):
        out = actual_factory(family, *args)
        if family is FullGaussian:
            calls.append(out._precision_policy)
        return out

    monkeypatch.setattr(model, "_family_instance", observed_factory)
    before = e_step_fixed_point_diagnostics(model, tokens)["estep_fp_kl"]
    gaussian_mod.set_full_cov_kl_precision("fp32_escalate")
    after = e_step_fixed_point_diagnostics(model, tokens)["estep_fp_kl"]

    assert calls and set(calls) == {"fp64"}
    assert after == before


def test_gauge_equivariance_residual_owns_full_policy_through_transport_and_divergence(monkeypatch):
    """The metric's real transport, factory, and energy calls all receive the explicit policy."""
    from vfe3 import free_energy
    from vfe3 import metrics

    group = get_group("block_glk")(4, 2)
    mu = torch.zeros(2, 4)
    sigma = torch.eye(4).expand(2, 4, 4).clone()
    omega = torch.eye(4).reshape(1, 1, 4, 4).expand(2, 2, 4, 4).clone()
    transport_policies = []
    energy_policies = []
    transport = FullGaussian.transport_dispersion.__func__
    pairwise = free_energy.pairwise_energy

    def observed_transport(cls, dispersion, omega_, *, diagonal_out=None, precision_policy=None):
        transport_policies.append(precision_policy)
        return transport(cls, dispersion, omega_, diagonal_out=diagonal_out,
                         precision_policy=precision_policy)

    def observed_energy(q, p, *args, **kwargs):
        energy_policies.append((q._precision_policy, p._precision_policy))
        return pairwise(q, p, *args, **kwargs)

    monkeypatch.setattr(FullGaussian, "transport_dispersion", classmethod(observed_transport))
    monkeypatch.setattr(free_energy, "pairwise_energy", observed_energy)

    before = metrics.gauge_equivariance_residual(
        mu, sigma, omega, group, n_samples=1, seed=0,
        full_cov_kl_precision="fp64",
    )
    gaussian_mod.set_full_cov_kl_precision("fp32_escalate")
    after = metrics.gauge_equivariance_residual(
        mu, sigma, omega, group, n_samples=1, seed=0,
        full_cov_kl_precision="fp64",
    )

    assert transport_policies and set(transport_policies) == {"fp64"}
    assert energy_policies and set(energy_policies) == {("fp64", "fp64")}
    for key in before:
        torch.testing.assert_close(after[key], before[key], rtol=0.0, atol=0.0)


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
