import pytest

from vfe3.config import VFE3Config


def test_config_defaults():
    cfg = VFE3Config()
    assert cfg.eps == 1e-6
    assert cfg.kl_max == 100.0
    assert cfg.divergence_family == "renyi"
    assert cfg.renyi_order == 1.0


def test_config_rejects_unknown_family():
    with pytest.raises(ValueError):
        VFE3Config(divergence_family="not_a_family")


def test_divergence_family_is_functional_seam_distinct_from_family():
    """divergence_family selects the divergence FUNCTIONAL (renyi, ...); family selects the
    covariance structure. They are distinct seams: a covariance-family value is not a valid
    functional, and the default functional is 'renyi'."""
    cfg = VFE3Config()
    assert cfg.divergence_family == "renyi" and cfg.family == "gaussian_diagonal"
    with pytest.raises(ValueError):
        VFE3Config(divergence_family="gaussian_diagonal")   # covariance family != functional
    with pytest.raises(ValueError):
        VFE3Config(divergence_family="not_a_functional")


def test_config_rejects_nonpositive_alpha():
    with pytest.raises(ValueError):
        VFE3Config(renyi_order=0.0)


def test_config_rejects_nonpositive_eps():
    with pytest.raises(ValueError):
        VFE3Config(eps=0.0)


def test_config_rejects_nonpositive_kl_max():
    with pytest.raises(ValueError):
        VFE3Config(kl_max=0.0)


def test_gauge_transport_default_on_is_noop():
    """gauge_transport='on' (default) is the pure learned-frame path: it must NOT coerce any of the
    gauge-frame fields, so a config built with explicit frame settings is byte-identical to omitting
    the toggle."""
    cfg = VFE3Config(phi_scale=0.06, pos_phi="learned", e_phi_lr=0.3, m_phi_lr=0.02)
    assert cfg.gauge_transport == "on"
    assert cfg.phi_scale == 0.06 and cfg.pos_phi == "learned"
    assert cfg.e_phi_lr == 0.3 and cfg.m_phi_lr == 0.02


def test_gauge_transport_off_forces_identity_frame():
    """gauge_transport='off' coerces the frame to the identity (Omega_ij=I): phi_scale=0, pos_phi='none',
    e_phi_lr=0, m_phi_lr=0, with a warning recording the coercion."""
    with pytest.warns(UserWarning, match="gauge_transport='off'"):
        cfg = VFE3Config(gauge_transport="off", phi_scale=0.06, pos_phi="learned",
                         e_phi_lr=0.3, m_phi_lr=0.02)
    assert cfg.phi_scale == 0.0 and cfg.pos_phi == "none"
    assert cfg.e_phi_lr == 0.0 and cfg.m_phi_lr == 0.0


def test_gauge_transport_off_rejects_rope():
    """A RoPE rotation folded into the transport makes Omega != I, so it is incompatible with the
    Omega=I contract of gauge_transport='off'."""
    with pytest.raises(ValueError, match="pos_rotation"):
        VFE3Config(gauge_transport="off", pos_rotation="rope")


def test_gauge_transport_off_rejects_regime_ii():
    """A non-flat Regime-II connection edge factor makes Omega != I, so it is incompatible with
    gauge_transport='off'."""
    with pytest.raises(ValueError, match="transport_mode"):
        VFE3Config(gauge_transport="off", transport_mode="regime_ii")


@pytest.mark.parametrize("phi_reflection", ["init_seed", "metropolis"])
def test_gauge_transport_off_rejects_phi_reflection(phi_reflection):
    with pytest.raises(ValueError, match="reflection"):
        VFE3Config(gauge_transport="off", gauge_group="glk",
                   phi_reflection=phi_reflection)


def test_gauge_transport_frozen_freezes_lrs_keeps_random_frame():
    """gauge_transport='frozen' keeps the random frame (phi_scale unchanged, pos_phi unchanged) but
    freezes its learning (e_phi_lr=0, m_phi_lr=0)."""
    with pytest.warns(UserWarning, match="gauge_transport='frozen'"):
        cfg = VFE3Config(gauge_transport="frozen", phi_scale=0.06, pos_phi="learned",
                         e_phi_lr=0.3, m_phi_lr=0.02)
    assert cfg.phi_scale == 0.06 and cfg.pos_phi == "learned"
    assert cfg.e_phi_lr == 0.0 and cfg.m_phi_lr == 0.0


def test_gauge_transport_frozen_rejects_zero_phi_scale():
    """'frozen' is a RANDOM fixed frame; a zero phi_scale would make it the identity, which is the
    'off' mode -- reject the ambiguous pair."""
    with pytest.raises(ValueError):
        VFE3Config(gauge_transport="frozen", phi_scale=0.0)


def test_gauge_transport_rejects_unknown():
    with pytest.raises(ValueError):
        VFE3Config(gauge_transport="bogus")


def test_cross_couplings_accepts_list_pairs_from_json_roundtrip():
    """A config.json reloaded by viz.report._load_config gives LIST pairs (JSON has no tuples);
    VFE3Config(**reloaded) must accept them so a cold-start generate_figures does not crash on the
    isinstance(pair, tuple) validator. Coercion normalizes them back to tuples."""
    cfg = VFE3Config(gauge_group="block_glk", embed_dim=4, n_heads=2,
                     cross_couplings=[[0, 1]])          # lists, exactly as JSON round-trips tuples
    assert cfg.cross_couplings == [(0, 1)]              # actual tuples (a stored list would fail this)


def test_load_config_drops_legacy_diagonal_covariance_key(tmp_path):
    """Old config.json files (written when diagonal_covariance was a dataclass field) carry the key
    under 'config'; report._load_config must drop it (it is now a derived property of family) so a
    figure regeneration from a pre-change run does not crash on the unexpected kwarg."""
    import json
    from dataclasses import asdict
    from vfe3.viz.report import _load_config

    cfg_dict = asdict(VFE3Config())
    cfg_dict["diagonal_covariance"] = True                   # legacy field, no longer a constructor arg
    (tmp_path / "config.json").write_text(
        json.dumps({"config": cfg_dict, "dataset": "wikitext-103"}))
    cfg, dataset = _load_config(tmp_path)
    assert cfg.diagonal_covariance is True and dataset == "wikitext-103"


def test_config_rejects_nan_min_lr():
    import math
    with pytest.raises(ValueError):
        VFE3Config(min_lr=math.nan)


def test_config_rejects_nan_min_lr_frac():
    import math
    with pytest.raises(ValueError):
        VFE3Config(min_lr_frac=math.nan)


@pytest.mark.parametrize("overrides", [
    {"transport_mode": "regime_ii"},
])
def test_straight_through_with_each_learnable_trigger_warns(overrides):
    """straight_through severs the per-iteration E-step tangent, so a learnable param that enters the
    loss ONLY through it (connection_W) gets no gradient. The regime_ii trigger must warn
    (non-breaking: 'unroll' is the default that trains it)."""
    with pytest.warns(UserWarning, match="frozen"):
        VFE3Config(e_step_gradient="straight_through", **overrides)


def test_detach_with_learnable_trigger_warns():
    """The 'detach' sibling severs the same tangent and must warn too (the un-warned route:
    detach_e_step=True is forced to 'unroll' here and is warned at the model level instead)."""
    with pytest.warns(UserWarning, match="frozen"):
        VFE3Config(e_step_gradient="detach", transport_mode="regime_ii")


def test_straight_through_without_learnable_does_not_warn():
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        VFE3Config(e_step_gradient="straight_through")          # no learnable param active
    assert not any("frozen" in str(w.message) for w in caught)


# --- Phase 7 full-config fields --------------------------------------------
def test_config_model_defaults():
    cfg = VFE3Config()
    assert cfg.embed_dim == 64 and cfg.n_heads == 8 and cfg.n_layers == 1
    assert cfg.gauge_group == "block_glk" and cfg.decode_mode == "diagonal"
    assert cfg.use_prior_bank is False              # default is the linear-decode ablation


def test_tau_is_kappa_sqrt_d_head():
    # Audit finding 6c: tau = kappa * sqrt(d_head) (per-head, Vaswani sqrt(d_k)),
    # NOT sqrt(embed_dim). embed_dim=16, n_heads=4 -> d_head=4 -> sqrt(d_head)=2.
    cfg = VFE3Config(embed_dim=16, n_heads=4, kappa_beta=1.5)
    assert abs(cfg.tau - 1.5 * 2.0) < 1e-9
    assert cfg.d_head == 4


def test_config_rejects_embed_dim_not_divisible_by_heads():
    with pytest.raises(ValueError):
        VFE3Config(embed_dim=10, n_heads=3)


def test_config_rejects_unknown_gauge_group_and_decode_mode():
    with pytest.raises(ValueError):
        VFE3Config(gauge_group="not_a_group")
    with pytest.raises(ValueError):
        VFE3Config(decode_mode="not_a_mode")


def test_config_sp_gauge_group_requires_even_embed_dim():
    # Sp(2m,R) lives in even dimension K=2m. Even embed_dim is accepted; odd raises a clear
    # ValueError. Existing groups are unaffected by the even-dim guard.
    cfg = VFE3Config(embed_dim=4, n_heads=2, gauge_group="sp")
    assert cfg.gauge_group == "sp"
    # an odd embed_dim is rejected (n_heads=1 so the divisibility guard cannot mask it):
    with pytest.raises(ValueError):
        VFE3Config(embed_dim=5, n_heads=1, gauge_group="sp")
    # a non-sp group with an odd embed_dim is fine (guard is sp-specific):
    assert VFE3Config(embed_dim=5, n_heads=1, gauge_group="glk").embed_dim == 5


def test_config_accepts_diagonal_chunked_decode_and_validates_chunk_size():
    """diagonal_chunked is the fused chunked-vocab decode+CE mode; decode_chunk_size must be > 0."""
    assert VFE3Config().decode_chunk_size == 8192            # default
    cfg = VFE3Config(decode_mode="diagonal_chunked", decode_chunk_size=4096)
    assert cfg.decode_mode == "diagonal_chunked" and cfg.decode_chunk_size == 4096
    with pytest.raises(ValueError):
        VFE3Config(decode_chunk_size=0)
    with pytest.raises(ValueError):
        VFE3Config(decode_chunk_size=-1)


def test_config_rejects_negative_learning_rate_and_bad_rho():
    with pytest.raises(ValueError):
        VFE3Config(e_q_mu_lr=-0.1)
    with pytest.raises(ValueError):
        VFE3Config(prior_handoff_rho=1.5)


# --- Audit 2026-05-31: dead / trapping toggles are live + rejected, not silent ----
def test_config_accepts_omega_direct_on_gl_groups():
    """omega_direct is now a live element-valued chart on the GL groups (glk, block_glk)."""
    for grp, over in (("glk", {}), ("block_glk", {"n_heads": 2})):
        cfg = VFE3Config(gauge_parameterization="omega_direct", gauge_group=grp,
                         embed_dim=4, n_heads=over.get("n_heads", 1), transport_mode="flat",
                         pos_phi="none", e_phi_lr=0.0)
        assert cfg.gauge_parameterization == "omega_direct"
    assert VFE3Config(gauge_parameterization="phi").gauge_parameterization == "phi"


def test_config_rejects_omega_direct_off_scope():
    r"""Phase 2 widened the gauge_group whitelist to all seven omega-eligible groups (so_k is now
    accepted -- see test_omega_direct_accepts_all_eligible_groups), so only the still-live scope
    rejections (E-step frame refinement, non-flat transport) remain here."""
    with pytest.raises(ValueError):                       # E-step frame refinement not supported yet
        VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4,
                   n_heads=1, pos_phi="none", e_phi_lr=0.1)
    with pytest.raises(ValueError):                       # only the flat regime in Phase 1
        VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4, n_heads=1,
                   transport_mode="regime_ii", pos_phi="none")


@pytest.mark.parametrize("pos_phi", ["learned", "frozen"])
def test_omega_direct_rejects_active_positional_phi(pos_phi):
    with pytest.raises(ValueError, match="pos_phi"):
        VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4,
                   n_heads=1, e_phi_lr=0.0, pos_phi=pos_phi)


def test_omega_direct_rejects_additive_encoder():
    with pytest.raises(ValueError, match="per_token_additive"):
        VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4,
                   n_heads=1, e_phi_lr=0.0, pos_phi="none",
                   encode_mode="per_token_additive")


def test_config_omega_retract_and_reflection_validated():
    with pytest.raises(ValueError):
        VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4, n_heads=1,
                   omega_retract_mode="bogus", pos_phi="none")
    assert VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4, n_heads=1,
                      omega_retract_mode="cayley", pos_phi="none").omega_retract_mode == "cayley"


@pytest.mark.parametrize("value", [1, 0, "False", None])
def test_omega_compact_storage_requires_strict_bool(value):
    with pytest.raises(ValueError, match="omega_compact_storage"):
        VFE3Config(omega_compact_storage=value)

    assert VFE3Config(omega_compact_storage=False).omega_compact_storage is False
    assert VFE3Config(omega_compact_storage=True).omega_compact_storage is True


@pytest.mark.parametrize("value", [-1, 1.5, True, "2", None])
def test_omega_reorth_every_requires_nonnegative_integer(value):
    with pytest.raises(ValueError, match="omega_reorth_every"):
        VFE3Config(omega_reorth_every=value)

    assert VFE3Config(omega_reorth_every=0).omega_reorth_every == 0
    assert VFE3Config(omega_reorth_every=3).omega_reorth_every == 3


@pytest.mark.parametrize("value", [0, -1, 1.5, True, False, "2", None])
def test_omega_metropolis_every_requires_positive_integer(value):
    with pytest.raises(ValueError, match="omega_metropolis_every"):
        VFE3Config(omega_metropolis_every=value)

    assert VFE3Config(omega_metropolis_every=1).omega_metropolis_every == 1
    assert VFE3Config(omega_metropolis_every=3).omega_metropolis_every == 3


def test_config_accepts_use_prior_bank_false():
    """use_prior_bank=False is the live linear-projection decode ablation:
    encode/self-coupling stay on the PriorBank, only decode becomes a plain mu->logits
    projection. It must construct cleanly (no NotImplementedError); it is the current default."""
    assert VFE3Config().use_prior_bank is False
    cfg = VFE3Config(use_prior_bank=True)
    assert cfg.use_prior_bank is True


def test_config_rejects_gauge_fixed_encode():
    """encode_mode='gauge_fixed' is an unimplemented stub; rejected at construction, not at forward."""
    with pytest.raises(NotImplementedError):
        VFE3Config(encode_mode="gauge_fixed")


def test_config_has_state_dependent_alpha_shape_params():
    """b0/c0 (state-dependent-alpha shape parameters) are configurable and validated positive."""
    cfg = VFE3Config()
    assert cfg.b0 == 1.0 and cfg.c0 == 1.0
    with pytest.raises(ValueError):
        VFE3Config(b0=0.0)
    with pytest.raises(ValueError):
        VFE3Config(c0=-1.0)


def test_diagonal_covariance_derived_from_family():
    """diagonal_covariance is a DERIVED read-only property of family (single source of truth),
    not a settable field: it tracks the family's cov_kind and passing it as a kwarg raises."""
    assert VFE3Config(family="gaussian_diagonal").diagonal_covariance is True
    assert VFE3Config(family="gaussian_full", decode_mode="full").diagonal_covariance is False
    with pytest.raises(TypeError):
        VFE3Config(diagonal_covariance=False)          # no longer a constructor field


def test_per_coord_alpha_requires_diagonal_family():
    """state_dependent_per_coord needs a per-coordinate self-divergence, which exists only for
    the diagonal family (full-cov KL does not decompose coordinate-wise). The inconsistent pair
    is rejected at construction; the diagonal pairing (the default family) is accepted."""
    with pytest.raises(ValueError):
        VFE3Config(lambda_alpha_mode="state_dependent_per_coord",
                   family="gaussian_full", decode_mode="full")
    VFE3Config(lambda_alpha_mode="state_dependent_per_coord")          # family defaults to diagonal -> ok


def test_per_coord_alpha_requires_renyi_functional():
    """state_dependent_per_coord weights each coordinate by its own alpha^(k), needing a per-coordinate
    self-divergence. That decomposition exists only for a divergence that DECOMPOSES coordinate-wise
    (the per-coordinate functional registry: Renyi/KL, Bhattacharyya, Jeffreys); squared_hellinger is
    excluded because H^2 = 1 - exp(-D_{1/2}/2) is a nonlinear transform of the summed divergence.
    A non-decomposable divergence_family otherwise constructs fine and crashes only at the first
    forward (free_energy.self_divergence_per_coord raises). Reject the pair at construction, mirroring
    the covariance guard above. The DEFAULT diagonal family is used so the covariance guard does not
    mask this functional check; the Renyi default is accepted (no over-rejection)."""
    with pytest.raises(ValueError):
        VFE3Config(lambda_alpha_mode="state_dependent_per_coord", divergence_family="squared_hellinger")
    VFE3Config(lambda_alpha_mode="state_dependent_per_coord")          # divergence_family defaults to renyi -> ok


def test_decode_mode_full_rejects_diagonal_family():
    # 'full' KL-decode consumes a (B,N,K,K) sigma; with a diagonal family (sigma (B,N,K)) it is a
    # shape crash at the first prior-bank forward -- reject at construction (the shipped-arm direction).
    # use_prior_bank=True: the rank check is a PRIOR-BANK decode constraint (linear decode ignores
    # decode_mode; that skip is covered by test_decode_mode_family_cross_check_skipped_under_linear_decode).
    with pytest.raises(ValueError):
        VFE3Config(family="gaussian_diagonal", decode_mode="full",
                   use_prior_bank=True)


def test_decode_mode_diagonal_rejects_full_family():
    # The reverse rank mismatch: a full family (sigma (B,N,K,K)) fed to a diagonal decode kernel.
    # use_prior_bank=True: the rank check is a prior-bank decode constraint (see the linear-decode
    # skip test below).
    with pytest.raises(ValueError):
        VFE3Config(family="gaussian_full", decode_mode="diagonal",
                   use_prior_bank=True)


def test_decode_mode_family_cross_check_skipped_under_linear_decode():
    # use_prior_bank=False decodes via the linear projection and ignores decode_mode, so the rank
    # cross-check does not apply: a 'full' decode_mode with a diagonal family is accepted.
    cfg = VFE3Config(family="gaussian_diagonal",
                     decode_mode="full", use_prior_bank=False)
    assert cfg.use_prior_bank is False


def test_rope_with_sp_gauge_warns_structure_group():
    # RoPE rotates adjacent coordinate pairs (2k,2k+1); Sp(2m) pairs i with m+i, so the rope-wrapped
    # transport leaves the symplectic group. The GL(K)-congruence divergence invariance survives, so
    # this is a WARNING (not an error): the operator is still a valid GL(K) element, just not in Sp.
    with pytest.warns(UserWarning, match="symplectic"):
        VFE3Config(gauge_group="sp", pos_rotation="rope")


def test_tied_block_glk_rejects_killing_per_block():
    """killing_per_block builds a per-HEAD Killing metric and needs generators that partition per
    block (block_glk's independent gl(d) per head). tied_block_glk's shared kron(I_n, gl(d))
    generators each act on EVERY block, so that preconditioner does not apply -- reject at config
    time (else it fails cryptically at the first forward). A compatible preconditioner is accepted."""
    with pytest.raises(ValueError):
        VFE3Config(embed_dim=4, n_heads=2, gauge_group="tied_block_glk",
                   phi_precond_mode="killing_per_block")
    for ok in ("none", "clip", "killing"):
        VFE3Config(embed_dim=4, n_heads=2, gauge_group="tied_block_glk", phi_precond_mode=ok)


def test_config_phi_retract_mode_validated():
    """phi_retract_mode selects the Lie-algebra composition chart (euclidean | bch)."""
    assert VFE3Config().phi_retract_mode == "bch"          # operating-point default (2026-06-10)
    assert VFE3Config(phi_retract_mode="euclidean").phi_retract_mode == "euclidean"
    assert VFE3Config(phi_retract_mode="bch").phi_retract_mode == "bch"
    with pytest.raises(ValueError):
        VFE3Config(phi_retract_mode="not_a_mode")


def test_config_spd_retract_mode_validated():
    """spd_retract_mode selects the SPD covariance retraction geometry (registry key)."""
    assert VFE3Config().spd_retract_mode == "spd_affine"
    with pytest.raises(ValueError):
        VFE3Config(spd_retract_mode="not_a_mode")


def test_config_transport_mode_validated():
    """transport_mode selects the connection-regime (registry key); default 'flat'."""
    assert VFE3Config().transport_mode == "flat"
    assert VFE3Config(transport_mode="flat").transport_mode == "flat"
    with pytest.raises(ValueError):
        VFE3Config(transport_mode="not_a_mode")


def test_config_eval_max_batches_default_none_and_validated():
    """eval_max_batches caps the PERIODIC validation pass (diagnostic only). Default None is
    the pure path -- the full validation split is read, as before. A positive int caps it;
    a non-positive int is rejected (the final post-training eval stays uncapped at the call site)."""
    assert VFE3Config().eval_max_batches is None
    assert VFE3Config(eval_max_batches=50).eval_max_batches == 50
    with pytest.raises(ValueError):
        VFE3Config(eval_max_batches=0)


def test_diagonal_covariance_is_read_only_property():
    """diagonal_covariance is a derived read-only property (no setter): family is the single
    source of truth, so assigning it raises AttributeError."""
    cfg = VFE3Config()
    assert cfg.diagonal_covariance is True
    with pytest.raises(AttributeError):
        cfg.diagonal_covariance = False


def test_config_lambda_h_default_zero_and_validated():
    """lambda_h is the hyper-prior weight KL(s_i||r) (manuscript eq:pointwise_free_energy);
    default 0.0 = OFF (pure single-tier path). A negative weight is rejected; 0.0 and a
    positive weight are accepted."""
    assert VFE3Config().lambda_h == 0.0
    assert VFE3Config(lambda_h=0.5).lambda_h == 0.5
    with pytest.raises(ValueError):
        VFE3Config(lambda_h=-1.0)


def test_config_cross_couplings_default_none_and_validated():
    """cross_couplings (off-block GL(K) head coupling) defaults None (current behavior). A valid
    list of distinct in-range directed head pairs is accepted under block_glk; out-of-range or
    self-coupling (a == b) pairs raise; a group whose builder does not accept the kwarg
    (glk / so_k / tied_block_glk) raises when cross_couplings is set."""
    assert VFE3Config().cross_couplings is None
    # valid: distinct in-range pair under block_glk (embed_dim 8 / n_heads 2 -> heads {0, 1})
    cfg = VFE3Config(embed_dim=8, n_heads=2, gauge_group="block_glk", cross_couplings=[(0, 1)])
    assert cfg.cross_couplings == [(0, 1)]
    # self-coupling a == b is rejected
    with pytest.raises(ValueError):
        VFE3Config(embed_dim=8, n_heads=2, gauge_group="block_glk", cross_couplings=[(0, 0)])
    # out-of-range head index is rejected (head 2 not in [0, 2))
    with pytest.raises(ValueError):
        VFE3Config(embed_dim=8, n_heads=2, gauge_group="block_glk", cross_couplings=[(0, 2)])
    # an unsupported group (builder does not accept the kwarg) is rejected
    with pytest.raises(ValueError):
        VFE3Config(embed_dim=8, n_heads=2, gauge_group="so_k", cross_couplings=[(0, 1)])
    with pytest.raises(ValueError):
        VFE3Config(embed_dim=8, n_heads=2, gauge_group="tied_block_glk", cross_couplings=[(0, 1)])


def test_config_amp_dtype_default_none_and_validated():
    """amp_dtype is the opt-in mixed-precision toggle: None (default) = OFF (pure fp32, no
    autocast), 'bf16' / 'fp16' enable autocast. 'fp32' and any other string are rejected (None
    is the only OFF value; there is no 'fp32' member -- fp32 is amp_dtype=None)."""
    assert VFE3Config().amp_dtype is None
    assert VFE3Config(amp_dtype="bf16").amp_dtype == "bf16"
    assert VFE3Config(amp_dtype="fp16").amp_dtype == "fp16"
    with pytest.raises(ValueError):
        VFE3Config(amp_dtype="fp32")
    with pytest.raises(ValueError):
        VFE3Config(amp_dtype="bfloat16")


def test_pos_phi_default_is_learned_and_validates():
    cfg = VFE3Config()
    assert cfg.pos_phi == "learned"        # default-ON; the pure no-composition path is "none"
    assert cfg.pos_phi_compose == "bch"
    assert cfg.bch_pe_order == 4


def test_pos_phi_rejects_unknown_mode():
    with pytest.raises(ValueError):
        VFE3Config(pos_phi="banana")


def test_pos_phi_compose_rejects_unknown():
    with pytest.raises(ValueError):
        VFE3Config(pos_phi_compose="quaternion")


def test_config_accepts_newly_registered_family_without_editing_config():
    """A new family registered with cov_kind='diagonal' is a valid config family and its derived
    diagonal_covariance property reads the registered cov_kind, without editing config.py (no
    hardcoded family-name list)."""
    from vfe3.families.base import register_family, _FAMILIES
    from vfe3.families.gaussian import DiagonalGaussian

    name = "laplace_diagonal_test"

    @register_family(name)
    class _LaplaceDiagonal(DiagonalGaussian):                            # cov_kind = "diagonal"
        pass

    try:
        cfg = VFE3Config(family=name)                                    # must NOT raise
        assert cfg.family == name
        assert cfg.diagonal_covariance is True                          # derived from registered cov_kind
    finally:
        _FAMILIES.pop(name, None)


def test_rope_defaults_off_and_full_gauge_requires_full_cov():
    cfg = VFE3Config()
    assert cfg.pos_rotation == "none" and cfg.rope_full_gauge is False
    with pytest.raises(ValueError):
        VFE3Config(pos_rotation="rope", rope_full_gauge=True)            # family defaults to diagonal
    # full-gauge with full covariance is allowed
    VFE3Config(pos_rotation="rope", rope_full_gauge=True,
               family="gaussian_full", decode_mode="full")


def test_gauge_group_validation_reads_registry_not_static_list():
    r"""Audit F4: a newly registered gauge group must be a valid config value WITHOUT editing
    config.py (the modularity contract -- add-by-registering). Mirrors transport_mode /
    spd_retract_mode, which already validate against their registries. RED against the static
    _VALID_GAUGE_GROUPS tuple."""
    from vfe3.geometry.groups import _GROUPS, get_group, register_group
    name = "audit_probe_glk_alias"
    if name not in _GROUPS:
        register_group(name)(lambda K, *a, **k: get_group("glk")(K))
    cfg = VFE3Config(embed_dim=4, n_heads=1, gauge_group=name)
    assert cfg.gauge_group == name                       # accepted via the registry, not a literal list


def test_decode_mode_linear_stays_a_rejected_second_gate():
    r"""Audit F4 (deliberate NON-change): decode_mode='linear' is reached via use_prior_bank=False,
    NOT through decode_mode; it must stay rejected as a decode_mode value (intentional second-gate,
    not a registry oversight). Guards against accidentally opening it when registry-validating the
    OTHER seams."""
    with pytest.raises(ValueError):
        VFE3Config(decode_mode="linear")


# --- audit 2026-07-01 fixes (F2-config, F12, F7, C5, F9) ---


def test_sigma_max_validated():
    r"""Audit 2026-07-01 F2: sigma_max caps the covariance in the SPD retractions
    (clamp max=sigma_max), so a nonfinite or sub-eps cap must be rejected at construction --
    it would push the clamped covariance below eps / negative / NaN."""
    with pytest.raises(ValueError):
        VFE3Config(sigma_max=-1.0)
    with pytest.raises(ValueError):
        VFE3Config(sigma_max=float("nan"))
    with pytest.raises(ValueError):
        VFE3Config(sigma_max=1e-9)                        # below the default eps=1e-6
    assert VFE3Config(sigma_max=10.0).sigma_max == 10.0   # the default cap still constructs


def test_bool_fields_reject_truthy_non_bools():
    r"""Audit 2026-07-01 F12: trust_resume_checkpoint / generate_figures / use_ema are consumed
    by truthiness (load_checkpoint reads bool(cfg.trust_resume_checkpoint), so the string
    "False" would coerce to True and enable the unsafe weights_only=False fallback). Non-bool
    values are rejected at construction; real-bool defaults still construct."""
    with pytest.raises(ValueError):
        VFE3Config(trust_resume_checkpoint="False")
    with pytest.raises(ValueError):
        VFE3Config(generate_figures=1)
    with pytest.raises(ValueError):
        VFE3Config(use_ema="0")
    cfg = VFE3Config()
    assert cfg.trust_resume_checkpoint is False and cfg.generate_figures is True


def test_max_steps_and_warmup_steps_validated():
    r"""Audit 2026-07-01 F12: max_steps feeds range(start_step, n_steps) (a float TypeErrors
    late in range(); 0 is an empty training loop) and warmup_steps feeds the scheduler
    (negative is meaningless; 0 = no warmup stays legal via the scheduler's max(1, ...))."""
    with pytest.raises(ValueError):
        VFE3Config(max_steps=0)
    with pytest.raises(ValueError):
        VFE3Config(max_steps=4.0)
    with pytest.raises(ValueError):
        VFE3Config(warmup_steps=-1)
    assert VFE3Config(warmup_steps=0).warmup_steps == 0   # no-warmup is a valid config
    assert VFE3Config().max_steps == 15000                # default still constructs


def test_bch_pe_order_zero_rejected_when_bch_compose_active():
    r"""Audit 2026-07-01 F12: compose_bch adds the Dynkin corrections only at order >= 1, so
    pos_phi_compose='bch' with bch_pe_order=0 degrades to plain additive composition still
    labeled BCH -- rejected. Inert under pos_phi='none' (never reaches compose_bch)."""
    with pytest.raises(ValueError):
        VFE3Config(pos_phi="learned", pos_phi_compose="bch", bch_pe_order=0)
    assert VFE3Config().bch_pe_order == 4                 # default (order 4) constructs
    VFE3Config(pos_phi="none", bch_pe_order=0)            # inert combination -> accepted


def test_nonflat_transport_with_active_model_channel_warns():
    r"""Audit 2026-07-01 F7 (safe variant): the s-channel (_gamma_energy / _refine_s) transports
    the s tables under the FLAT phi-cocycle only, so a non-flat belief transport plus an active
    model channel (lambda_gamma>0 or s_e_step=True) runs different connections per channel --
    the model-channel comparison is not gauge-covariant. Non-breaking UserWarning."""
    with pytest.warns(UserWarning, match="non-flat"):
        VFE3Config(transport_mode="regime_ii", lambda_gamma=1.0)
    with pytest.warns(UserWarning, match="non-flat"):
        VFE3Config(transport_mode="regime_ii_covariant", s_e_step=True,
                   prior_source="model_channel", lambda_gamma=1.0)


def test_flat_or_inactive_model_channel_no_nonflat_warning():
    r"""Audit 2026-07-01 F7 negative control: flat transport with an active model channel, and
    non-flat transport with an INACTIVE model channel (lambda_gamma=0, s_e_step=False), must
    NOT emit the flat s-cocycle warning."""
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        VFE3Config(transport_mode="flat", lambda_gamma=1.0, s_e_step=True,
                   prior_source="model_channel")
        VFE3Config(transport_mode="regime_ii", lambda_gamma=0.0)
    assert not [r for r in rec if "FLAT phi-cocycle" in str(r.message)]


def test_regime_ii_covariant_diagonal_family_warns_controlled_approximation():
    r"""Audit 2026-07-01 C5 (safe variant): the diagonal covariance cone is not closed under a
    general GL(K) congruence Omega Sigma Omega^T, so regime_ii_covariant with
    family='gaussian_diagonal' is a controlled approximation of the exact covariant transport --
    non-breaking UserWarning."""
    with pytest.warns(UserWarning, match="CONTROLLED APPROXIMATION"):
        VFE3Config(transport_mode="regime_ii_covariant", family="gaussian_diagonal")


def test_regime_ii_covariant_full_family_no_approximation_warning():
    r"""Audit 2026-07-01 C5 negative control: family='gaussian_full' IS closed under the GL(K)
    congruence, so no approximation warning fires (the complementary gaussian_full fp32
    numerics warning may still fire; only the approximation label is asserted absent)."""
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        VFE3Config(transport_mode="regime_ii_covariant", family="gaussian_full",
                   decode_mode="full")
    assert not [r for r in rec if "CONTROLLED APPROXIMATION" in str(r.message)]


def test_force_large_figures_default_off():
    r"""Audit 2026-07-01 F9 (safe variant): force_large_figures is the opt-in override for the
    finalize_run figure-pass memory guard; default False keeps the guard so small-run behavior
    is unchanged."""
    assert VFE3Config().force_large_figures is False
    assert VFE3Config(force_large_figures=True).force_large_figures is True


@pytest.mark.parametrize("gamma_toggle", [
    dict(lambda_gamma=0.1),                                        # gamma model-coupling loss term
    dict(s_e_step=True, prior_source="model_channel"),            # _refine_s s-channel E-step
    dict(gamma_as_beta_prior=True, lambda_gamma=0.1),             # hierarchical gamma-in-belief-prior fold
])
def test_omega_direct_accepts_active_gamma_channel(gamma_toggle):
    r"""Phase-3 gave the gamma / model-coupling (s) channel omega frame-fidelity: _gamma_energy
    (lambda_gamma>0 / gamma_as_beta_prior), _fold_gamma_prior, and _refine_s (s_e_step) now transport
    the s tables by the stored frame U_i U_j^{-1} under omega_direct -- exactly like the belief channel
    -- so the config gate that used to reject the combination is gone. Each active-gamma toggle now
    CONSTRUCTS. (s_e_step alone -- no lambda_gamma -- also constructs; it only carries the orthogonal
    prior_source='model_channel' requirement, which has nothing to do with the omega frame.)"""
    cfg = VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk",
                     embed_dim=4, n_heads=1, pos_phi="none", **gamma_toggle)
    assert cfg.gauge_parameterization == "omega_direct"


def test_omega_direct_pure_channel_off_constructs():
    r"""The default (gamma-off) omega_direct config is accepted: lambda_gamma=0, s_e_step=False,
    gamma_as_beta_prior=False leave no s-channel transport to mis-frame."""
    cfg = VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4,
                     n_heads=1, pos_phi="none")
    assert cfg.gauge_parameterization == "omega_direct"


def test_omega_direct_accepts_all_eligible_groups():
    common = dict(gauge_parameterization="omega_direct", transport_mode="flat", e_phi_lr=0.0,
                  lambda_gamma=0.0, s_e_step=False, pos_phi="none")
    for grp, over in (("glk", {}), ("block_glk", {"n_heads": 2}), ("tied_block_glk", {"n_heads": 2}),
                      ("sp", {}), ("so_k", {}),
                      ("so_n", {"group_n": 3, "irrep_spec": [("l0", 1), ("l1", 1)]}),
                      ("sp_n", {"embed_dim": 5, "group_n": 4, "irrep_spec": [("sym0", 1), ("sym1", 1)]})):
        kw = dict(embed_dim=over.pop("embed_dim", 4), n_heads=over.pop("n_heads", 1),
                  gauge_group=grp, use_head_mixer=False, **over, **common)
        assert VFE3Config(**kw).gauge_parameterization == "omega_direct"


def test_omega_direct_reflection_cross_check_fail_closed():
    import pytest
    base = dict(gauge_parameterization="omega_direct", transport_mode="flat", e_phi_lr=0.0,
                lambda_gamma=0.0, s_e_step=False, omega_reflection="init_seed", use_head_mixer=False,
                pos_phi="none")
    # reject init_seed where it is vacuous / group-incorrect
    with pytest.raises(ValueError):    # sp: no det<0 component
        VFE3Config(embed_dim=4, n_heads=1, gauge_group="sp", **base)
    with pytest.raises(ValueError):    # so_n: needs rho(O(N)) image, deferred
        VFE3Config(embed_dim=4, n_heads=1, gauge_group="so_n", group_n=3,
                   irrep_spec=[("l0", 1), ("l1", 1)], **base)
    with pytest.raises(ValueError):    # tied_block_glk: ambient seed breaks the tie, deferred
        VFE3Config(embed_dim=4, n_heads=2, gauge_group="tied_block_glk", **base)
    # accept init_seed where it is group-correct
    assert VFE3Config(embed_dim=4, n_heads=1, gauge_group="so_k", **base).omega_reflection == "init_seed"
    assert VFE3Config(embed_dim=4, n_heads=1, gauge_group="glk", **base).omega_reflection == "init_seed"


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_constant_lambda_alpha_requires_finite_nonnegative_value(value):
    with pytest.raises(ValueError, match="lambda_alpha"):
        VFE3Config(lambda_alpha_mode="constant", lambda_alpha=value)
    assert VFE3Config(lambda_alpha_mode="constant", lambda_alpha=0.0).lambda_alpha == 0.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_rope_base_requires_finite_positive_value(value):
    with pytest.raises(ValueError, match="rope_base"):
        VFE3Config(rope_base=value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_alibi_slope_requires_finite_nonnegative_value(value):
    with pytest.raises(ValueError, match="alibi_slope"):
        VFE3Config(alibi_slope=value)
    assert VFE3Config(alibi_slope=0.0).alibi_slope == 0.0


def test_rope_warning_names_only_registered_positional_modes():
    from vfe3.model.positional_phi import _POS_PHI

    with pytest.warns(UserWarning) as caught:
        VFE3Config(pos_rotation="rope")
    messages = " ".join(str(item.message) for item in caught)

    assert "sinusoidal" not in messages
    assert "frozen" in messages
    assert {"frozen", "learned"}.issubset(_POS_PHI)


def test_omega_direct_capability_comes_from_group_registration():
    from vfe3.geometry.groups import GaugeGroup, _GROUPS, get_group, register_group

    capable_name = "audit_omega_capable_alias"
    incapable_name = "audit_omega_incapable_alias"

    @register_group(capable_name, omega_direct_capable=True)
    def _build_capable(K, *args, **kwargs):
        base = get_group("glk")(K, *args, **kwargs)
        return GaugeGroup(
            name=capable_name,
            generators=base.generators,
            irrep_dims=base.irrep_dims,
            skew_symmetric=base.skew_symmetric,
        )

    @register_group(incapable_name)
    def _build_incapable(K, *args, **kwargs):
        base = get_group("glk")(K, *args, **kwargs)
        return GaugeGroup(
            name=incapable_name,
            generators=base.generators,
            irrep_dims=base.irrep_dims,
            skew_symmetric=base.skew_symmetric,
            omega_direct_capable=True,
        )

    try:
        cfg = VFE3Config(
            embed_dim=4,
            n_heads=1,
            gauge_group=capable_name,
            gauge_parameterization="omega_direct",
            pos_phi="none",
            e_phi_lr=0.0,
        )
        assert cfg.gauge_group == capable_name
        assert get_group("glk")(4).omega_direct_capable is True
        assert _build_capable(4).omega_direct_capable is True
        assert _build_incapable(4).omega_direct_capable is False
        assert get_group(capable_name)(4).omega_direct_capable is True
        assert get_group(incapable_name)(4).omega_direct_capable is False
        with pytest.raises(ValueError, match="omega_direct"):
            VFE3Config(
                embed_dim=4,
                n_heads=1,
                gauge_group=incapable_name,
                gauge_parameterization="omega_direct",
                pos_phi="none",
                e_phi_lr=0.0,
            )
    finally:
        _GROUPS.pop(capable_name, None)
        _GROUPS.pop(incapable_name, None)
