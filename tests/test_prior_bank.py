import torch
import torch.nn.functional as F
from vfe3.belief import BeliefState
from vfe3.divergence import kl as _kl
from vfe3.families.gaussian import DiagonalGaussian
from vfe3.model.prior_bank import PriorBank


def _reference_decode(pb, mu_q, sigma_q, tau):
    # General reference: -KL(q_i || pi_v)/tau by broadcasting the divergence seam over V.
    # kl_max=inf: a DECODE preserves the full KL ranking, so the seam's default
    # kl_max=100 saturation (which flattens distant priors to a single -100 logit) is
    # disabled -- matching the unclamped fused kernel across the whole input domain.
    mu_v = pb.mu_embed; sigma_v = torch.exp(pb.sigma_log_embed)
    mu_q_b = mu_q.unsqueeze(-2)                               # (B,N,1,K)
    sigma_q_b = sigma_q.unsqueeze(-2)
    klv = _kl(DiagonalGaussian(mu_q_b, sigma_q_b), DiagonalGaussian(mu_v, sigma_v),
              kl_max=float("inf"))                           # (B,N,V) via broadcast
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
    rng = torch.Generator().manual_seed(0)
    V, K, n_gen = 12, 4, 16
    pb = PriorBank(V, K, n_gen)
    mu_q = torch.randn(2, 3, K, generator=rng); sigma_q = torch.rand(2, 3, K, generator=rng) + 0.5
    logits = pb.decode(mu_q, sigma_q)
    ref = _reference_decode(pb, mu_q, sigma_q, pb.decode_tau)  # decode_log_scale=0 -> tau_eff=decode_tau
    assert torch.allclose(logits, ref, atol=1e-3)             # EXACT -KL/tau (per-position term kept)
    # shift-invariant pin (robust to a dropped-constant variant):
    assert torch.allclose(F.log_softmax(logits, dim=-1), F.log_softmax(ref, dim=-1), atol=1e-4)


def test_decode_tau_scaling():
    rng = torch.Generator().manual_seed(1)
    V, K = 10, 3
    pb = PriorBank(V, K, 9)
    mu_q = torch.randn(1, 2, K, generator=rng); sigma_q = torch.rand(1, 2, K, generator=rng) + 0.5
    l1 = pb.decode(mu_q, sigma_q, tau=1.0)
    l2 = pb.decode(mu_q, sigma_q, tau=2.0)
    assert torch.allclose(l1, 2.0 * l2, atol=1e-3)            # logits ~ 1/tau


def test_decode_matches_seam_in_large_kl_regime():
    # Regression for the clamp-asymmetry defect: tight priors + separated means drive
    # KL >> 100, where the seam's default kl_max=100 would saturate to a flat -100 and
    # destroy the ranking. Both decode paths use kl_max=inf, so they must still agree
    # EXACTLY and predict the SAME token (argmax preserved, no flattening).
    torch.manual_seed(1)
    V, K, n_gen = 6, 4, 16
    pb = PriorBank(V, K, n_gen)
    with torch.no_grad():
        pb.sigma_log_embed.fill_(-4.0)                       # sigma_v = exp(-4) ~ 0.018 (tight)
        pb.mu_embed.normal_(0.0, 1.0)
    mu_q = 5.0 * torch.ones(1, 1, K); sigma_q = torch.ones(1, 1, K)
    logits = pb.decode(mu_q, sigma_q)
    ref = _reference_decode(pb, mu_q, sigma_q, pb.decode_tau)
    implied_kl = (-logits * pb.decode_tau)
    assert implied_kl.max().item() > 100.0                   # genuinely past the old clamp
    assert torch.allclose(logits, ref, atol=1e-3)            # EXACT pin holds in the clamped regime
    assert torch.allclose(                                   # shift-invariant pin holds
        F.log_softmax(logits, dim=-1), F.log_softmax(ref, dim=-1), atol=1e-4
    )
    assert logits.argmax(-1).item() == ref.argmax(-1).item()  # same predicted token (no flattening)


def test_decode_exact_at_large_mean_offset():
    # Regression for the catastrophic-cancellation defect: a large common offset on the
    # means makes the expanded-square matmul subtract large near-equal quantities. The
    # mean-centered fused kernel must still match the seam to atol 1e-3 (the un-centered
    # version exceeded it near offset ~100 and grew ~mu^2 thereafter).
    torch.manual_seed(0)
    V, K, n_gen = 8, 4, 16
    pb = PriorBank(V, K, n_gen)
    with torch.no_grad():
        pb.mu_embed.normal_(0.0, 0.1).add_(1000.0)           # means clustered far from zero
    mu_q = (1000.0 + 0.1) * torch.ones(1, 1, K); sigma_q = torch.ones(1, 1, K)
    logits = pb.decode(mu_q, sigma_q)
    ref = _reference_decode(pb, mu_q, sigma_q, pb.decode_tau)
    assert torch.allclose(logits, ref, atol=1e-3)
    assert torch.allclose(
        F.log_softmax(logits, dim=-1), F.log_softmax(ref, dim=-1), atol=1e-4
    )
