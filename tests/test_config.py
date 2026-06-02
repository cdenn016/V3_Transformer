import pytest

from vfe3.config import VFE3Config


def test_config_defaults():
    cfg = VFE3Config()
    assert cfg.eps == 1e-6
    assert cfg.kl_max == 100.0
    assert cfg.divergence_family == "renyi"
    assert cfg.alpha_div == 1.0


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
        VFE3Config(alpha_div=0.0)


def test_config_rejects_nonpositive_eps():
    with pytest.raises(ValueError):
        VFE3Config(eps=0.0)


def test_config_rejects_nonpositive_kl_max():
    with pytest.raises(ValueError):
        VFE3Config(kl_max=0.0)


# --- Phase 7 full-config fields --------------------------------------------
def test_config_model_defaults():
    cfg = VFE3Config()
    assert cfg.embed_dim == 64 and cfg.n_heads == 8 and cfg.n_layers == 1
    assert cfg.gauge_group == "block_glk" and cfg.decode_mode == "diagonal"
    assert cfg.use_prior_bank is True


def test_tau_is_kappa_sqrt_d_head():
    # Audit finding 6c: tau = kappa * sqrt(d_head) (per-head, Vaswani sqrt(d_k)),
    # NOT sqrt(embed_dim). embed_dim=16, n_heads=4 -> d_head=4 -> sqrt(d_head)=2.
    cfg = VFE3Config(embed_dim=16, n_heads=4, kappa=1.5)
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
        VFE3Config(e_mu_lr=-0.1)
    with pytest.raises(ValueError):
        VFE3Config(prior_handoff_rho=1.5)


# --- Audit 2026-05-31: dead / trapping toggles are live + rejected, not silent ----
def test_config_rejects_omega_direct_gauge_parameterization():
    """omega_direct needs a per-token GL(K) matrix the no-NN belief (phi only) cannot supply,
    so it is rejected at construction rather than silently aliased to the 'phi' path."""
    with pytest.raises(NotImplementedError):
        VFE3Config(gauge_parameterization="omega_direct")
    assert VFE3Config(gauge_parameterization="phi").gauge_parameterization == "phi"


def test_config_accepts_use_prior_bank_false():
    """use_prior_bank=False is the live linear-projection decode ablation (VFE_2.0 parity):
    encode/self-coupling stay on the PriorBank, only decode becomes a plain mu->logits
    projection. It must construct cleanly (no NotImplementedError); the default stays True."""
    assert VFE3Config().use_prior_bank is True
    cfg = VFE3Config(use_prior_bank=False)
    assert cfg.use_prior_bank is False


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


def test_diagonal_covariance_must_agree_with_family():
    """diagonal_covariance is a live bool cross-validated against family (kept distinct, not collapsed)."""
    with pytest.raises(ValueError):
        VFE3Config(diagonal_covariance=False)          # family defaults to gaussian_diagonal
    # the consistent full-covariance pair is accepted (divergence_family stays the functional seam)
    VFE3Config(family="gaussian_full", diagonal_covariance=False, decode_mode="full")


def test_per_coord_alpha_requires_diagonal_family():
    """state_dependent_per_coord needs a per-coordinate self-divergence, which exists only for
    the diagonal family (full-cov KL does not decompose coordinate-wise). The inconsistent pair
    is rejected at construction; the diagonal pairing (the default family) is accepted."""
    with pytest.raises(ValueError):
        VFE3Config(alpha_mode="state_dependent_per_coord",
                   family="gaussian_full", diagonal_covariance=False, decode_mode="full")
    VFE3Config(alpha_mode="state_dependent_per_coord")          # family defaults to diagonal -> ok


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
    assert VFE3Config().phi_retract_mode == "euclidean"
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


def test_config_diagonal_covariance_cross_check_uses_cov_kind():
    """The diagonal_covariance consistency check is driven by the family's declared cov_kind,
    not the literal family == 'gaussian_diagonal'."""
    VFE3Config(family="gaussian_full", diagonal_covariance=False)        # full + non-diagonal: ok
    with pytest.raises(ValueError):
        VFE3Config(family="gaussian_full", diagonal_covariance=True)     # full + diagonal: mismatch


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


def test_config_accepts_newly_registered_family_without_editing_config():
    """A new family registered with cov_kind='diagonal' is a valid config family and passes the
    diagonal_covariance cross-check without editing config.py (no hardcoded family-name list)."""
    from vfe3.families.base import register_family, _FAMILIES
    from vfe3.families.gaussian import DiagonalGaussian

    name = "laplace_diagonal_test"

    @register_family(name)
    class _LaplaceDiagonal(DiagonalGaussian):                            # cov_kind = "diagonal"
        pass

    try:
        cfg = VFE3Config(family=name, diagonal_covariance=True)          # must NOT raise
        assert cfg.family == name
        with pytest.raises(ValueError):
            VFE3Config(family=name, diagonal_covariance=False)          # cov_kind diagonal != False
    finally:
        _FAMILIES.pop(name, None)
