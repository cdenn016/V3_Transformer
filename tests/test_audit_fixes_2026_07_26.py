r"""Pins for the 2026-07-26 deep-audit remediation, batch 3 (docs/audit-results.md).

C-04 _direct_link_diagonal_covariance gets the autocast island, float64 escalation and
     clamp(min=0.0) its compact/factored siblings already carry.
C-05 the dense pairwise Omega and the dense diagonal congruence evaluate inside an fp32
     island, so Omega_ii = I (and therefore E_ii = 0) survives bf16 autocast.
D-02 mm_exact_update calls uses_kernel_route internally and fails closed instead of scoring
     one objective and fusing another.
E-01 gamma_as_beta_prior rejects a gamma prior that may attend future keys beta forbids.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.geometry.transport import (
    DirectLinkTransport,
    _direct_link_diagonal_covariance,
    compute_transport_operators,
    transport_covariance,
)
from vfe3.gradients.kernels import mm_exact_update
from vfe3.model.model import build_group


# --- C-04: the direct-link diagonal congruence ------------------------------------------

def _ill_conditioned_direct_link(seed: int, n: int = 3, k: int = 4):
    r"""A charted direct link at ``||phi||_F`` ~ 8 with a strongly anisotropic sigma.

    This is the regime where the mixed-sign regrouping ``r^T C_j r`` cancels: the audit measured
    29/300 draws returning a negative transported variance under bf16 autocast (worst -2.85e+09)
    before the island/escalation/floor were added.
    """
    torch.manual_seed(seed)
    g = torch.randn(n, k, k)
    g = g / g.norm(dim=(-2, -1), keepdim=True) * 8.0
    direct = DirectLinkTransport(
        torch.matrix_exp(1.5 * torch.randn(n, n, k, k)),
        torch.matrix_exp(g),
        torch.matrix_exp(-g),
    )
    sigma = torch.rand(n, k).pow(6) * 1e3 + 1e-4
    return direct, sigma


def test_direct_link_diagonal_covariance_never_returns_a_negative_variance():
    # A negative variance reaches the divergence kernel's clamp(min=eps)=1e-6, inverting that
    # key's precision weight by ~6 orders and saturating E_ij at kl_max. The floor makes it
    # unreachable; over the seeds below the unfixed route went negative under autocast.
    for seed in range(40):
        direct, sigma = _ill_conditioned_direct_link(seed)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            out = _direct_link_diagonal_covariance(direct, sigma)
        assert bool((out >= 0.0).all()), f"negative transported variance at seed {seed}"


def test_direct_link_diagonal_covariance_holds_the_fp32_island_under_autocast():
    # The two siblings (_compact_factored_diagonal_covariance, _factored_diagonal_covariance)
    # carry autocast(enabled=False); this route did not, so under AMP the whole congruence was
    # built in bf16 -- the very cancellation the working dtype exists to control.
    direct, sigma = _ill_conditioned_direct_link(0)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = _direct_link_diagonal_covariance(direct, sigma)
    assert out.dtype is torch.float32
    reference = _direct_link_diagonal_covariance(direct, sigma)
    assert torch.equal(out, reference)          # the island makes AMP byte-identical to fp32


def test_direct_link_bare_edge_branch_also_holds_the_island():
    # The bare link has no vertex factors and its squared form is manifestly nonnegative, so it
    # needs no escalation -- but it is the same autocast-eligible einsum and needs the island.
    torch.manual_seed(5)
    direct = DirectLinkTransport(torch.matrix_exp(0.5 * torch.randn(3, 3, 4, 4)))
    sigma = torch.rand(3, 4) + 0.1
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = _direct_link_diagonal_covariance(direct, sigma)
    assert out.dtype is torch.float32
    assert torch.equal(out, _direct_link_diagonal_covariance(direct, sigma))


# --- C-05: the dense operator and the dense diagonal congruence --------------------------

def _dense_omega(k: int = 4, n: int = 3):
    cfg = VFE3Config(vocab_size=7, embed_dim=k, max_seq_len=n + 1, n_layers=1,
                     gauge_group="glk", n_heads=1)
    group = build_group(cfg)
    torch.manual_seed(2)
    phi = 0.3 * torch.randn(1, n, group.generators.shape[0])
    return phi, group


def test_dense_omega_self_link_is_the_identity_under_bf16_autocast():
    # Omega_ii = I is what makes the structural self energy E_ii = 0 and keeps the self pair out
    # of pair_mask. The vertex exps were already pinned to fp32, but the pair einsum was
    # autocast-eligible and rebuilt Omega in bf16: measured ||Omega_ii - I||_inf = 4.76e-03.
    phi, group = _dense_omega()
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        omega_amp = compute_transport_operators(phi, group)["Omega"]
    omega_fp32 = compute_transport_operators(phi, group)["Omega"]
    assert omega_amp.dtype is torch.float32
    assert torch.equal(omega_amp, omega_fp32)
    eye = torch.eye(omega_amp.shape[-1])
    self_link = torch.diagonal(omega_amp, dim1=1, dim2=2).permute(0, 3, 1, 2)
    assert float((self_link - eye).abs().max()) < 1e-5


def test_dense_diagonal_congruence_holds_the_fp32_island():
    # transport_covariance's dense diagonal branch is the last congruence route that ran at the
    # autocast dtype: measured 4.63e-03 relative error against fp64 under bf16.
    phi, group = _dense_omega()
    omega = compute_transport_operators(phi, group)["Omega"][0]
    torch.manual_seed(4)
    sigma = torch.rand(omega.shape[0], omega.shape[-1]) + 0.1
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = transport_covariance(omega, sigma, diagonal_out=True)
    assert out.dtype is torch.float32
    reference = torch.einsum("ijkl,ijkl,jl->ijk",
                             omega.double(), omega.double(), sigma.double())
    rel = float(((out.double() - reference).abs() / reference.abs().clamp(min=1e-12)).max())
    assert rel < 1e-5, rel


# --- D-02: mm_exact_update fails closed off its route ------------------------------------

def _mm_inputs(n: int = 4, k: int = 3):
    torch.manual_seed(6)
    mu = torch.randn(n, k)
    sigma = torch.rand(n, k) + 0.5
    mu_p = torch.randn(n, k)
    sigma_p = torch.rand(n, k) + 0.5
    omega = torch.eye(k).expand(n, n, k, k).contiguous()
    return mu, sigma, mu_p, sigma_p, omega


@pytest.mark.parametrize(
    "family,divergence_family",
    [
        ("gaussian_diagonal_exact", "renyi"),
        ("gaussian_diagonal", "jeffreys"),
        ("gaussian_diagonal", "bhattacharyya"),
        ("gaussian_diagonal", "squared_hellinger"),
    ],
)
def test_mm_exact_update_rejects_configs_off_the_kernel_route(family, divergence_family):
    # Before the guard all four RAN: the grid came from pairwise_energy rather than
    # fam.coupling_energy, and a non-Renyi divergence drove beta and pair_mask before being fused
    # with hardcoded diagonal-Gaussian-KL precision expressions.
    mu, sigma, mu_p, sigma_p, omega = _mm_inputs()
    with pytest.raises(ValueError, match="closed-form minimizer"):
        mm_exact_update(mu, sigma, mu_p, sigma_p, omega,
                        family=family, divergence_family=divergence_family)


def test_mm_exact_update_still_serves_its_covered_route():
    mu, sigma, mu_p, sigma_p, omega = _mm_inputs()
    mu_star, sigma_star = mm_exact_update(mu, sigma, mu_p, sigma_p, omega,
                                          family="gaussian_diagonal",
                                          divergence_family="renyi")
    assert mu_star.shape == mu.shape and sigma_star.shape == sigma.shape
    assert bool(torch.isfinite(mu_star).all()) and bool((sigma_star > 0).all())


# --- E-01: the gamma-as-beta-prior causal-support gate -----------------------------------

def _gamma_prior_cfg(beta: str, gamma: str) -> VFE3Config:
    return VFE3Config(
        vocab_size=11, embed_dim=4, max_seq_len=6, n_layers=1, n_heads=1,
        gauge_group="glk", n_e_steps=1, prior_source="model_channel",
        beta_attention_prior=beta, gamma_attention_prior=gamma,
        lambda_gamma=0.5, gamma_as_beta_prior=True, attention_window=2,
    )


@pytest.mark.parametrize("gamma", ["uniform", "alibi", "windowed"])
def test_gamma_as_beta_prior_rejects_a_future_reaching_gamma_prior(gamma):
    # gamma is normalized over its OWN key row before beta's mask is applied and the mixture is
    # renormalized, so gamma's normalizer Z_i does not cancel: a future key in Z_i changes the
    # belief log-prior at strictly past entries (measured up to 5.9e-05 at beta='causal').
    with pytest.raises(ValueError, match="leaks future tokens"):
        _gamma_prior_cfg("causal", gamma)


@pytest.mark.parametrize(
    "beta,gamma",
    [
        ("causal", "causal"),
        ("causal_alibi_noself", "causal_alibi_noself"),
        # A gamma allowing a PAST key beta forbids (here the self key, which causal_alibi_noself
        # masks and causal does not) shifts the mixture weight but leaks no future information.
        ("causal_alibi_noself", "causal"),
        # Neither channel is causal: there is no causal contract to violate.
        ("uniform", "uniform"),
    ],
)
def test_gamma_as_beta_prior_accepts_causally_consistent_pairings(beta, gamma):
    cfg = _gamma_prior_cfg(beta, gamma)
    assert cfg.gamma_as_beta_prior is True


def test_gamma_prior_support_gate_reads_the_registry_not_a_name_list():
    # The gate builds both priors through vfe3.attention_prior at max_seq_len, so a newly
    # registered prior is covered without editing config.py -- and a window-style support is
    # evaluated at the length it is actually used at, where 'windowed' does reach the future.
    from vfe3.attention_prior import _PRIORS, register_prior

    @register_prior("_audit_20260726_probe_future")
    def _probe(n_query, n_key, *, device=None, dtype=torch.float32, **kwargs):
        return torch.zeros(n_query, n_key, device=device, dtype=dtype)   # allows every key

    try:
        with pytest.raises(ValueError, match="leaks future tokens"):
            _gamma_prior_cfg("causal", "_audit_20260726_probe_future")
    finally:
        _PRIORS.pop("_audit_20260726_probe_future", None)


# --- E-03: the family scope gate fires at config validation, not mid-forward ------------

def _scope_cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=7, embed_dim=4, max_seq_len=4, n_layers=1,
                gauge_group="glk", n_heads=1)
    base.update(kw)
    return VFE3Config(**base)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(family="gaussian_diagonal_exact", renyi_order=0.5), "KL only"),
        (dict(family="gaussian_diagonal_exact", divergence_family="squared_hellinger",
              decode_mode="family"), "KL only"),
        (dict(family="gaussian_diagonal_exact", transport_mode="regime_ii_link"),
         "pair_inverse"),
        (dict(family="gaussian_frame_diagonal", transport_mode="regime_ii"), "coboundary"),
        (dict(family="gaussian_frame_diagonal", transport_mode="regime_ii_link"), "coboundary"),
    ],
)
def test_family_scope_violations_are_rejected_at_config_construction(kwargs, match):
    # Every one of these was ACCEPTED by the config and raised at the FIRST FORWARD, where
    # ablation.py catches config errors only around VFE3Config(**cfg_dict) and the mid-forward raise
    # falls to the outer handler, misfiling the cell as error_kind="train".
    with pytest.raises(ValueError, match=match):
        _scope_cfg(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(family="gaussian_diagonal_exact"),                                # flat: coboundary
        dict(family="gaussian_diagonal_exact", transport_mode="regime_ii"),    # dense: invertible
        dict(family="gaussian_frame_diagonal"),                                # flat: coboundary
        dict(family="gaussian_diagonal", transport_mode="regime_ii_link"),     # no requirement
    ],
)
def test_family_scope_gate_leaves_supported_pairings_alone(kwargs):
    assert _scope_cfg(**kwargs) is not None


def test_scope_gate_reads_the_transport_registry_not_a_mode_name_list():
    # The requirement is satisfied by capability LEVEL, so a newly registered transport that
    # declares 'coboundary' serves the frame family without editing config.py.
    from vfe3.geometry.transport import (
        PAIR_TRANSPORT_KINDS,
        _TRANSPORTS,
        get_transport_registration,
        register_transport,
    )

    assert PAIR_TRANSPORT_KINDS == ("opaque", "pair_inverse", "coboundary")
    flat = get_transport_registration("flat")
    assert flat.pair_transport_kind == "coboundary"
    assert flat.satisfies("coboundary") and flat.satisfies("pair_inverse") and flat.satisfies("any")
    link = get_transport_registration("regime_ii_link")
    assert not link.satisfies("pair_inverse") and not link.satisfies("coboundary")
    assert link.satisfies("any")

    @register_transport("_audit_20260726_probe_coboundary",
                        covariance_class="probe", pair_transport_kind="coboundary")
    def _probe(phi, group, **kwargs):
        return get_transport_registration("flat").callable(phi, group, **kwargs)

    try:
        cfg = _scope_cfg(family="gaussian_frame_diagonal",
                         transport_mode="_audit_20260726_probe_coboundary")
        assert cfg.transport_mode == "_audit_20260726_probe_coboundary"
    finally:
        _TRANSPORTS.pop("_audit_20260726_probe_coboundary", None)


def test_register_transport_rejects_an_unknown_capability_level():
    from vfe3.geometry.transport import register_transport

    with pytest.raises(ValueError, match="pair_transport_kind"):
        register_transport("_audit_20260726_bad", covariance_class="probe",
                           pair_transport_kind="invertible")


# --- E-04: the registry capability replaces the two hardcoded family tuples --------------

@pytest.mark.parametrize(
    "family,expected",
    [
        ("gaussian_diagonal", True),
        ("gaussian_full", True),
        ("gaussian_diagonal_exact", True),      # overrides coupling_energy only
        ("gaussian_frame_diagonal", True),      # overrides the two transport seams only
        ("laplace_diagonal", False),
    ],
)
def test_gaussian_pointwise_algebra_is_declared_on_the_family(family, expected):
    from vfe3.families.base import get_family

    assert get_family(family).gaussian_pointwise_algebra is expected


@pytest.mark.parametrize("family", ["gaussian_diagonal_exact", "gaussian_frame_diagonal"])
def test_barycenter_r_update_accepts_a_gaussian_pointwise_family(family):
    # Both were rejected by the old literal though barycenter_r_ branches only on cov_kind and
    # implements exactly the diagonal moment match these families inherit.
    assert _scope_cfg(family=family, r_update_mode="barycenter") is not None


def test_barycenter_r_update_still_rejects_a_non_gaussian_family():
    with pytest.raises(ValueError, match="gaussian_pointwise_algebra"):
        _scope_cfg(family="laplace_diagonal", r_update_mode="barycenter",
                   decode_mode="family", divergence_family="renyi")


@pytest.mark.parametrize("family", ["gaussian_diagonal_exact", "gaussian_frame_diagonal"])
def test_fast_decode_kernels_accept_a_gaussian_pointwise_family(family):
    # The fast decode kernels assume the Gaussian POINTWISE readout, which both satisfy.
    assert _scope_cfg(family=family, use_prior_bank=True, decode_mode="diagonal") is not None


# --- E-05: the invariance verifier covers the whole family registry ---------------------

def test_check_admissible_dispatches_through_the_family_registry():
    from vfe3.families.base import divergence_families
    from vfe3.geometry.groups import check_admissible, get_group

    grp = get_group("glk")(K=4)
    covered, gaps = [], []
    for family in divergence_families():
        try:
            check_admissible(grp, family, n_samples=3)
            covered.append(family)
        except NotImplementedError:
            gaps.append(family)
    # Before E-05 the verifier raised for everything but three hardcoded names, one of which
    # ('gaussian') is not even registered -- so neither 2026-07 family could be reached.
    assert "gaussian_diagonal_exact" in covered
    assert "gaussian_frame_diagonal" in covered
    assert gaps == ["laplace_diagonal"]      # no full-covariance readout: the extension point


def test_dead_gaussian_declaration_is_gone():
    from vfe3.geometry.groups import declared_invariant_families, get_group

    for name in ("glk", "block_glk", "tied_block_glk", "so_k", "so_n", "sp", "sp_n"):
        assert declared_invariant_families(name) == ("gaussian_full",)
        assert get_group(name).invariant_families == ("gaussian_full",)
