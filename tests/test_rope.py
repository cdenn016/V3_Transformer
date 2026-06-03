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
