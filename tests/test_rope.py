import torch

from vfe3.geometry.rope import build_rope_rotation, get_pos_rotation


def test_rope_rotation_is_orthogonal_and_block_diagonal():
    irrep_dims = [4, 4]                                     # two head-blocks of size 4
    R = build_rope_rotation(torch.arange(6), irrep_dims, base=100.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    assert R.shape == (6, 8, 8)
    eye = torch.eye(8).expand(6, 8, 8)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)   # orthogonal
    # off-block entries are exactly zero (block-diagonal on irrep_dims)
    assert torch.count_nonzero(R[:, 0:4, 4:8]) == 0
    assert torch.count_nonzero(R[:, 4:8, 0:4]) == 0


def test_rope_position_zero_is_identity():
    R = build_rope_rotation(torch.arange(3), [4], base=100.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    assert torch.allclose(R[0], torch.eye(4), atol=1e-6)   # position 0 -> angle 0 -> I


def test_pos_rotation_none_registered():
    assert get_pos_rotation("none")(torch.arange(3), [4], base=100.0,
                                    device=torch.device("cpu"), dtype=torch.float32) is None


from vfe3.geometry.transport import RopeTransport, transport_mean, transport_covariance


def test_rope_mean_at_identity_omega_is_relative():
    # Omega = I -> Omega^RoPE = R(theta_i) R(theta_j)^T = R(theta_i - theta_j): relative position.
    N, K = 5, 4
    R = build_rope_rotation(torch.arange(N), [K], base=100.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    omega_I = torch.eye(K).expand(N, N, K, K).contiguous()
    mu = torch.randn(N, K)
    rt = RopeTransport(base=omega_I, rope=R, on_cov=False)
    mu_t = transport_mean(rt, mu)                            # (N, N, K)
    mu_const = torch.ones(N, K)
    rt_c = RopeTransport(base=omega_I, rope=R, on_cov=False)
    t = transport_mean(rt_c, mu_const)                      # (N, N, K)
    # rows of equal (i-j) give equal transported vectors (relative-position property)
    assert torch.allclose(t[2, 1], t[3, 2], atol=1e-5)      # both are (i-j)=1
    assert torch.allclose(t[3, 1], t[4, 2], atol=1e-5)      # both are (i-j)=2


def test_rope_mean_only_leaves_covariance_unrotated():
    N, K = 4, 4
    R = build_rope_rotation(torch.arange(N), [K], base=10.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    omega_I = torch.eye(K).expand(N, N, K, K).contiguous()
    sigma = torch.rand(N, K) + 0.5
    rt = RopeTransport(base=omega_I, rope=R, on_cov=False)
    plain = transport_covariance(omega_I, sigma)            # un-rotated diagonal sandwich
    roped = transport_covariance(rt, sigma)                 # mu-only -> ignores rope
    assert torch.allclose(plain, roped, atol=1e-6)


from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import build_belief_transport


def test_build_belief_transport_wraps_in_ropetransport_when_rope_set():
    g = get_group("block_glk")(8, 2)
    phi = torch.randn(1, 6, g.generators.shape[0])
    R = build_rope_rotation(torch.arange(6), g.irrep_dims, base=100.0,
                            device=phi.device, dtype=phi.dtype)
    out = build_belief_transport(phi, g, transport_mode="flat", rope=R, rope_on_cov=False)
    assert isinstance(out, RopeTransport)
    # rope=None reproduces the plain build (no wrapper).
    plain = build_belief_transport(phi, g, transport_mode="flat")
    assert not isinstance(plain, RopeTransport)


from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _rope_cfg(**kw):
    base = dict(vocab_size=6, embed_dim=8, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0, gauge_group="block_glk",
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return VFE3Config(**base)


def test_rope_changes_logits_vs_no_rope():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    base = VFEModel(_rope_cfg(pos_rotation="none"))
    roped = VFEModel(_rope_cfg(pos_rotation="rope"))
    roped.load_state_dict(base.state_dict())
    assert not torch.allclose(base(x), roped(x), atol=1e-5)   # RoPE perturbs attention -> logits


def test_attention_maps_reflect_rope():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (1, 8))
    base = VFEModel(_rope_cfg(pos_rotation="none"))
    roped = VFEModel(_rope_cfg(pos_rotation="rope"))
    roped.load_state_dict(base.state_dict())
    a = base.attention_maps(x)
    b = roped.attention_maps(x)
    assert a.shape == b.shape
    assert not torch.allclose(a, b, atol=1e-5)             # RoPE changes the per-head attention


def test_diagnostics_runs_under_rope():
    # diagnostics() replays the E-step and must thread the ACTIVE rope into vfe_stack (matching the
    # forward), so the rope!=None convergence branch is exercised. Default configs use pos_rotation
    # 'none', so without this the threaded path is never run. Assert the metrics come back finite
    # (abs(v) < inf is False for both +/-inf and NaN).
    torch.manual_seed(0)
    x = torch.randint(0, 6, (1, 8))
    roped = VFEModel(_rope_cfg(pos_rotation="rope"))
    diag = roped.diagnostics(x)
    assert diag and all(abs(v) < float("inf") for v in diag.values())


from vfe3.gradients.kernels import belief_gradients
from vfe3.gradients.oracle import belief_gradients_autograd


def _full_cov_cfg(**kw):
    """_rope_cfg defaults plus the full-covariance pair (family/decode_mode); diagonal_covariance derived."""
    base = dict(vocab_size=6, embed_dim=8, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0, gauge_group="block_glk",
                warmup_steps=1, max_steps=4,
                family="gaussian_full", decode_mode="full")
    base.update(kw)
    return VFE3Config(**base)


def test_rope_means_only_kernel_matches_oracle():
    # The analytic belief-gradient kernel must still agree with autograd-of-F when the transport is
    # rope-rotated (means-only). Both consume the RopeTransport opaquely; agreement isolates RoPE.
    torch.manual_seed(0)
    g = get_group("block_glk")(8, 2)
    N, K, n_gen = 5, 8, g.generators.shape[0]
    phi = torch.randn(1, N, n_gen) * 0.1
    R = build_rope_rotation(torch.arange(N), g.irrep_dims, base=100.0,
                            device=phi.device, dtype=phi.dtype)
    omega = build_belief_transport(phi, g, transport_mode="flat", rope=R, rope_on_cov=False)
    mu   = torch.randn(1, N, K); sigma   = torch.rand(1, N, K) + 0.5
    mu_p = torch.randn(1, N, K); sigma_p = torch.rand(1, N, K) + 0.5
    kw = dict(tau=1.0, renyi_order=1.0, kl_max=100.0, eps=1e-6, b0=1.0, c0=1.0, value=1.0,
              include_attention_entropy=True, gradient_mode="filtering", family="gaussian_diagonal",
              divergence_family="renyi", lambda_alpha_mode="constant", irrep_dims=g.irrep_dims, log_prior=None)
    gk = belief_gradients(mu, sigma, mu_p, sigma_p, omega, **kw)          # hand kernel
    go = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega, **kw)  # autograd-of-F oracle
    assert torch.allclose(gk[0], go[0], atol=1e-5)
    assert torch.allclose(gk[1], go[1], atol=1e-5)


def test_rope_full_gauge_covariance_equals_manual_sandwich():
    # Full-gauge covariance: transport_covariance(RopeTransport(on_cov=True)) must equal the manual
    # sandwich with the rotated operator Omega'_ij = R_i Omega_ij R_j^T. Pure property; no model.
    torch.manual_seed(0)
    N, K = 4, 4
    R = build_rope_rotation(torch.arange(N), [K], base=10.0,
                            device=torch.device("cpu"), dtype=torch.float64)
    omega = torch.randn(N, N, K, K, dtype=torch.float64)
    A = torch.randn(N, K, K, dtype=torch.float64)
    sigma = A @ A.transpose(-1, -2) + K * torch.eye(K, dtype=torch.float64)   # SPD full cov
    got = transport_covariance(RopeTransport(base=omega, rope=R, on_cov=True), sigma)
    Op = torch.einsum("ikl,ijlm,jnm->ijkn", R, omega, R)                      # R_i Omega_ij R_j^T
    manual = torch.einsum("ijkl,jlm,ijnm->ijkn", Op, sigma, Op)
    assert torch.allclose(got, manual, atol=1e-9)


def test_full_gauge_model_runs_forward_backward():
    # Reachability: a full-covariance rope_full_gauge model trains (finite gradients) end to end.
    cfg = _full_cov_cfg(pos_rotation="rope", rope_full_gauge=True)
    torch.manual_seed(0)
    m = VFEModel(cfg)
    x = torch.randint(0, 6, (1, 6)); y = torch.randint(0, 6, (1, 6))
    _, loss, _ = m(x, y); loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(m.prior_bank.mu_embed.grad).all()


def test_2x2_positional_ablation_runs():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    for pr in ("none", "rope"):
        for pp in ("none", "learned"):
            m = VFEModel(_rope_cfg(pos_rotation=pr, pos_phi=pp))
            out = m(x)
            assert out.shape[0] == 2 and torch.isfinite(out).all()
