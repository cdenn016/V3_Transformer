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


# --- E-07: rope_on_value's three contradicting defaults -----------------------------------

def test_rope_on_value_default_agrees_across_every_seam():
    r"""One value, four places. The config field was flipped to False by a live-config sweep
    (`6dbfb58 "config: commit the live working-tree configuration as-is"`) while RopeTransport,
    vfe_block and the e_step defaults all kept True, so a direct caller forwarding `rope=` without
    `rope_on_value=` silently got the COUPLED gauge -- the opposite of what the config declared.
    """
    import inspect

    from vfe3.geometry.transport import RopeTransport
    from vfe3.model.block import vfe_block

    config_default = VFE3Config().rope_on_value
    assert config_default is True
    assert RopeTransport.__dataclass_fields__["on_value"].default is config_default
    assert inspect.signature(vfe_block).parameters["rope_on_value"].default is config_default


def test_rope_is_not_inert_at_the_default_value_gauge():
    r"""At rope_on_value=False the rotation barely reaches the logits (measured 7.9e-06 against
    2.1e-02 coupled), which is why test_rope_changes_logits_vs_no_rope was red: RoPE was very nearly
    a no-op on the shipped default. The coupled single-gauge path is the one that actually rotates.
    """
    from vfe3.model.model import VFEModel

    def _cfg(**kw):
        base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                    n_e_steps=2, gauge_group="block_glk", e_step_update="gradient")
        base.update(kw)
        return VFE3Config(**base)

    torch.manual_seed(0)
    tokens = torch.randint(0, 6, (2, 8))
    plain = VFEModel(_cfg(pos_rotation="none"))
    roped = VFEModel(_cfg(pos_rotation="rope"))          # rope_on_value now defaults True
    roped.load_state_dict(plain.state_dict())
    with torch.no_grad():
        assert float((plain(tokens) - roped(tokens)).abs().max()) > 1e-4


# --- D-03: the purity ledger records the E-step update axis --------------------------------

def _pure_cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=7, embed_dim=4, n_heads=1, max_seq_len=5, n_layers=1, n_e_steps=1,
                gauge_group="glk", use_prior_bank=True, lambda_alpha_mode="constant")
    base.update(kw)
    return VFE3Config(**base)


def test_mm_exact_run_can_no_longer_publish_on_pure_path():
    # mm_exact minimizes the STRICT-PAIR-MASKED surrogate, which drops the structural E_ii = 0
    # self-pairs, so it is not the canonical objective. The ledger used to certify such a run as
    # pure because no flag covered the update rule at all.
    from vfe3.run_artifacts import _pure_path_report

    report = _pure_path_report(_pure_cfg(e_step_update="mm_exact"), [])
    assert report["pure_flags"]["gradient_e_step_update"] is False
    assert report["on_pure_path"] is False
    assert [k for k, v in report["pure_flags"].items() if not v] == ["gradient_e_step_update"]


def test_gradient_run_is_still_certified_pure():
    from vfe3.run_artifacts import _pure_path_report

    report = _pure_path_report(_pure_cfg(e_step_update="gradient"), [])
    assert report["pure_flags"]["gradient_e_step_update"] is True
    assert report["on_pure_path"] is True


@pytest.mark.parametrize("update,damping", [("gradient", 1.0), ("mm_exact", 0.5)])
def test_config_toggles_record_the_update_rule_and_its_damping(update, damping):
    from vfe3.run_artifacts import _pure_path_report

    toggles = _pure_path_report(
        _pure_cfg(e_step_update=update, mm_damping=damping), [])["config_toggles"]
    assert toggles["e_step_update"] == update
    assert toggles["mm_damping"] == pytest.approx(damping)


# --- E-06: a length-1 tau is a scalar, not a one-head vector -------------------------------

@pytest.mark.parametrize("gauge_group", ["glk", "so_k"])
def test_learnable_kappa_beta_runs_on_a_single_irrep_block_group(gauge_group):
    r"""``log_kappa_beta`` is built with shape ``(len(irrep_dims),) = (1,)`` on a single-block group,
    and ``_broadcast_tau`` used to reshape any 1-d tau to ``(H, 1, 1)`` -- prepending a phantom head
    to a HEADLESS ``(..., N, N)`` energy. It propagated into beta and grad_sigma and broke the
    attention-map einsum outright, while the equivalent explicit per-head kappa list is rejected at
    config time.
    """
    from vfe3.model.model import VFEModel

    cfg = VFE3Config(vocab_size=7, embed_dim=4, n_heads=1, max_seq_len=5, n_layers=1,
                     n_e_steps=1, gauge_group=gauge_group, learnable_kappa_beta=True)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 7, (1, 5))

    with torch.no_grad():
        logits = model(tokens)
        maps = model.attention_maps(tokens)

    assert logits.shape == (1, 5, 7)
    assert maps.shape[-2:] == (5, 5)


def test_broadcast_tau_collapses_length_one_but_keeps_real_heads():
    from vfe3.free_energy import _broadcast_tau

    headless_energy = torch.zeros(5, 5)
    assert _broadcast_tau(torch.tensor([2.0]), headless_energy).dim() == 0
    multihead_energy = torch.zeros(3, 5, 5)
    assert _broadcast_tau(torch.tensor([1.0, 2.0, 3.0]), multihead_energy).shape == (3, 1, 1)


# --- E-17: the declared n_e_steps domain matches what the code does ------------------------

def test_n_e_steps_domain_admits_the_zero_depth_probes_use():
    # Two probes drive the field to 0 by direct assignment (the depth-sensitivity sweep and the
    # zero-e-steps counterfactual), so a `>= 1` rule was false of live instances.
    assert _pure_cfg(n_e_steps=0).n_e_steps == 0


def test_zero_n_e_steps_warns_that_inference_is_disabled():
    with pytest.warns(UserWarning, match="runs NO belief E-step"):
        _pure_cfg(n_e_steps=0)


def test_negative_n_e_steps_is_still_rejected():
    with pytest.raises(ValueError, match="n_e_steps must be >= 0"):
        _pure_cfg(n_e_steps=-1)


# --- A-02: means-only RoPE is a hybrid, and the run record says so ------------------------

def _rope_pair(seed: int = 0, n: int = 4, k: int = 4, heads: int = 2):
    r"""A flat block cocycle plus a block-diagonal orthogonal RoPE rotation."""
    from vfe3.geometry.transport import FactoredTransport, RopeTransport

    d = k // heads
    torch.manual_seed(seed)
    algebra = 0.3 * torch.randn(n, k, k)
    block_mask = torch.zeros(k, k)
    for h in range(heads):
        block_mask[h * d:(h + 1) * d, h * d:(h + 1) * d] = 1.0
    algebra = algebra * block_mask
    base = FactoredTransport(torch.matrix_exp(algebra), torch.matrix_exp(-algebra),
                             irrep_dims=[d] * heads, same_frame_flat_cocycle=True)
    angles = torch.randn(n, heads)
    rope = torch.zeros(n, k, k)
    for i in range(n):
        for h in range(heads):
            cos, sin = torch.cos(angles[i, h]), torch.sin(angles[i, h])
            rope[i, h * d:h * d + 2, h * d:h * d + 2] = torch.tensor([[cos, -sin], [sin, cos]])
    return (base,
            RopeTransport(base=base, rope=rope, on_cov=False, same_frame_flat_cocycle=True),
            RopeTransport(base=base, rope=rope, on_cov=True, same_frame_flat_cocycle=True))


def test_means_only_rope_rotates_the_mean_but_not_the_covariance():
    r"""The transported pair is the pushforward of the key under NO single operator.

    This is the half of A-02 the challenge tier re-scoped and left open: "the mean transports under
    R_i Omega_ij R_j^T while the covariance transports under bare Omega_ij -- no single frame works.
    Needs its own check; neither side established it." Established here.
    """
    from vfe3.geometry.transport import transport_covariance, transport_mean

    base, hybrid, full = _rope_pair()
    mu = torch.randn(4, 4)
    sigma = torch.rand(4, 4) + 0.5

    mu_hybrid = transport_mean(hybrid, mu)
    mu_rotated = transport_mean(full, mu)
    mu_plain = transport_mean(base, mu)
    sigma_hybrid = transport_covariance(hybrid, sigma, diagonal_out=True)
    sigma_plain = transport_covariance(base, sigma, diagonal_out=True)

    assert torch.equal(mu_hybrid, mu_rotated)               # the MEAN is rotated ...
    assert float((mu_hybrid - mu_plain).abs().max()) > 1e-3
    assert torch.equal(sigma_hybrid, sigma_plain)           # ... and the COVARIANCE is not.


def test_means_only_rope_preserves_the_structural_self_pair():
    r"""What the asymmetry must NOT break: Omega_ii = I and R_i R_i^T = I, so the self-link is exact
    on both slots and the structural E_ii = 0 (and the pair_mask keyed off it) is untouched.
    """
    from vfe3.geometry.transport import transport_covariance, transport_mean

    _, hybrid, _ = _rope_pair()
    mu = torch.randn(4, 4)
    sigma = torch.rand(4, 4) + 0.5
    index = torch.arange(4)

    assert torch.equal(transport_mean(hybrid, mu)[index, index], mu)
    assert torch.equal(
        transport_covariance(hybrid, sigma, diagonal_out=True)[index, index], sigma)


def test_means_only_covariance_is_not_the_truncated_rotated_pushforward():
    # It is a DIFFERENT object, not a diagonal projection of the right one -- so it cannot be
    # explained away as the same controlled truncation the diagonal congruence already documents.
    from vfe3.geometry.transport import transport_covariance

    base, hybrid, _ = _rope_pair()
    sigma = torch.rand(4, 4) + 0.5
    shipped = transport_covariance(hybrid, sigma, diagonal_out=True)
    omega = torch.einsum("ikl,jlm->ijkm", base.exp_phi, base.exp_neg_phi)
    rotated = torch.einsum("ikl,ijlm,jnm->ijkn", hybrid.rope, omega, hybrid.rope)
    truncated = torch.einsum("ijkl,ijkl,jl->ijk", rotated, rotated, sigma)
    assert float((truncated - shipped).abs().max()) > 1e-2


@pytest.mark.parametrize(
    "pos_rotation,full_gauge,expected",
    [
        ("none", False, "not_applicable"),
        ("rope", False, "means_only_unrotated_covariance"),
        ("rope", True, "exact_rope_congruence"),
    ],
)
def test_run_report_records_which_rope_regime_the_energy_used(pos_rotation, full_gauge, expected):
    from vfe3.run_artifacts import _pure_path_report

    kwargs = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1, n_e_steps=1,
                  gauge_group="block_glk", pos_rotation=pos_rotation, rope_full_gauge=full_gauge)
    if full_gauge:                        # R Sigma R^T is dense: full gauge needs the full family
        kwargs.update(family="gaussian_full", e_step_update="gradient", decode_mode="full")
    report = _pure_path_report(VFE3Config(**kwargs), [])
    assert report["config_toggles"]["rope_pair_energy_exactness"] == expected
    if pos_rotation != "none":            # a RoPE run is already off the gauge-pure path
        assert report["gauge_flags"]["no_positional_rotation"] is False
