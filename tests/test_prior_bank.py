import pytest
import torch
import torch.nn.functional as F
from vfe3.belief import BeliefState
from vfe3.model.prior_bank import PriorBank


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
    # decode_log_scale=0 -> tau_eff=decode_tau
    ref = pb.reference_decode(mu_q, sigma_q, tau=pb.decode_tau)
    assert torch.allclose(logits, ref, atol=1e-3)             # EXACT -KL/tau (per-position term kept)
    # shift-invariant pin (robust to a dropped-constant variant):
    assert torch.allclose(F.log_softmax(logits, dim=-1), F.log_softmax(ref, dim=-1), atol=1e-4)


@pytest.mark.parametrize(
    ("prior_source", "untie_decode_bank"),
    [
        ("token", False),
        ("token", True),
        ("model_channel", False),
        ("model_channel", True),
    ],
    ids=["tied-token", "untied-token", "tied-model-channel", "untied-model-channel"],
)
def test_decode_matches_reference_across_table_routes(
    prior_source:      str,
    untie_decode_bank: bool,
) -> None:
    V, K, n_gen = 5, 3, 9
    pb = PriorBank(
        V,
        K,
        n_gen,
        decode_tau=1.7,
        prior_source=prior_source,
        untie_decode_bank=untie_decode_bank,
    )
    with torch.no_grad():
        pb._decode_mu_table().copy_(torch.linspace(-0.75, 0.75, V * K).reshape(V, K))
        pb._decode_sigma_log_table().copy_(
            torch.tensor([-20.0, -0.25, 81.0, -20.0, 0.5]).unsqueeze(-1).expand(V, K)
        )

    mu_q = torch.tensor([[[0.2, -0.3, 0.4]]])
    sigma_q = torch.tensor([[[0.8, 1.1, 0.6]]])
    with pytest.warns(RuntimeWarning, match="max_log=80"):
        logits = pb.decode(mu_q, sigma_q, tau=1.7)
    with pytest.warns(RuntimeWarning, match="max_log=80"):
        ref = pb.reference_decode(mu_q, sigma_q, tau=1.7)
    assert torch.allclose(logits, ref, atol=1e-3, rtol=0.0)


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
    ref = pb.reference_decode(mu_q, sigma_q, tau=pb.decode_tau)
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
    ref = pb.reference_decode(mu_q, sigma_q, tau=pb.decode_tau)
    assert torch.allclose(logits, ref, atol=1e-3)
    assert torch.allclose(
        F.log_softmax(logits, dim=-1), F.log_softmax(ref, dim=-1), atol=1e-4
    )
