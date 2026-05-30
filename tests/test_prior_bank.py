import torch
import torch.nn.functional as F
from vfe3.belief import BeliefState
from vfe3.divergence import kl as _kl
from vfe3.model.prior_bank import PriorBank


def _reference_decode(pb, mu_q, sigma_q, tau):
    # General reference: -KL(q_i || pi_v)/tau by broadcasting the divergence seam over V.
    mu_v = pb.mu_embed; sigma_v = torch.exp(pb.sigma_log_embed)
    mu_q_b = mu_q.unsqueeze(-2)                               # (B,N,1,K)
    sigma_q_b = sigma_q.unsqueeze(-2)
    klv = _kl(mu_q_b, sigma_q_b, mu_v, sigma_v)               # (B,N,V) via broadcast
    return -klv / tau


def test_encode_shapes_and_positive_sigma():
    V, K, n_gen = 20, 4, 16
    pb = PriorBank(V, K, n_gen)
    tokens = torch.randint(0, V, (2, 5))
    b = pb.encode(tokens)
    assert isinstance(b, BeliefState)
    assert b.mu.shape == (2, 5, K) and b.sigma.shape == (2, 5, K) and b.phi.shape == (2, 5, n_gen)
    assert (b.sigma > 0).all()


def test_encode_is_a_lookup():
    V, K, n_gen = 6, 3, 9
    pb = PriorBank(V, K, n_gen)
    b = pb.encode(torch.tensor([[0, 0]]))
    assert torch.allclose(b.mu[0, 0], b.mu[0, 1])             # same token -> same prior


def test_decode_matches_divergence_seam_exactly():
    V, K, n_gen = 12, 4, 16
    pb = PriorBank(V, K, n_gen)
    mu_q = torch.randn(2, 3, K); sigma_q = torch.rand(2, 3, K) + 0.5
    logits = pb.decode(mu_q, sigma_q)
    ref = _reference_decode(pb, mu_q, sigma_q, pb.decode_tau)  # decode_log_scale=0 -> tau_eff=decode_tau
    assert torch.allclose(logits, ref, atol=1e-3)             # EXACT -KL/tau (per-position term kept)
    # shift-invariant pin (robust to a dropped-constant variant):
    assert torch.allclose(F.log_softmax(logits, dim=-1), F.log_softmax(ref, dim=-1), atol=1e-4)


def test_decode_tau_scaling():
    V, K = 10, 3
    pb = PriorBank(V, K, 9)
    mu_q = torch.randn(1, 2, K); sigma_q = torch.rand(1, 2, K) + 0.5
    l1 = pb.decode(mu_q, sigma_q, tau=1.0)
    l2 = pb.decode(mu_q, sigma_q, tau=2.0)
    assert torch.allclose(l1, 2.0 * l2, atol=1e-3)            # logits ~ 1/tau
