r"""Categorical emission factor in the belief's Markov blanket (cfg.emission_mode, default OFF).

VFE 4.0 ``eq:state-model-markov-blanket-potentials`` puts ``log L(x_s | z_s, m_s)`` inside the
potential for the continuous latent, so an observation-conditioned belief update includes the
emission. V3's E-step has no data term at all. This pins the Bohning majorizer, the two readout
modes, the fail-closed validation, and -- the property the whole direction rests on -- that the
factor reads the CURRENT token and never the held-out target.

Everything is tiny and CPU-bound: K = 4, V = 16, N = 4.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.emission import bohning_curvature_diagonal, bohning_emission_terms
from vfe3.model.model import VFEModel

BASE = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
            e_step_update="mm_exact", use_prior_bank=False)


def _tokens(seed=0, batch=2, length=4, vocab=16):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (batch, length))


def _forward(**overrides):
    torch.manual_seed(0)
    cfg = VFE3Config(**BASE, **overrides)
    model = VFEModel(cfg)
    tokens, targets = _tokens(1), _tokens(2)
    _, loss, ce = model(tokens, targets=targets)
    return model, loss, ce


# --------------------------------------------------------------------------- the majorizer

def test_curvature_diagonal_matches_the_dense_bohning_matrix():
    r"""``d = diag(W^T H W)`` exactly, with ``H = (1/2)(I_V - 11^T/V)``."""
    torch.manual_seed(0)
    vocab, dim = 16, 4
    weight = torch.randn(vocab, dim)
    H = 0.5 * (torch.eye(vocab) - torch.ones(vocab, vocab) / vocab)
    dense = weight.T @ H @ weight                                    # (K, K)
    assert torch.allclose(bohning_curvature_diagonal(weight), dense.diagonal(), atol=1e-6)


def test_full_curvature_is_a_true_majorizer():
    r"""The FULL ``W^T H W`` bound holds for every displacement -- the property Bohning proves and
    the reason this route is admissible at all. The diagonal restriction the model consumes does
    NOT inherit it; that gap is measured in the next test rather than assumed away."""
    torch.manual_seed(0)
    vocab, dim = 16, 4
    weight = torch.randn(vocab, dim)
    H = 0.5 * (torch.eye(vocab) - torch.ones(vocab, vocab) / vocab)
    curvature = weight.T @ H @ weight
    z0 = torch.randn(dim)
    logits0 = weight @ z0
    p0 = torch.softmax(logits0, dim=-1)
    target = 5

    def exact(z):
        return -torch.log_softmax(weight @ z, dim=-1)[target]

    def bound(z):
        delta = z - z0
        return (exact(z0) - delta @ (weight.T @ (torch.eye(vocab)[target] - p0))
                + 0.5 * delta @ curvature @ delta)

    for step in range(25):                                           # many directions and magnitudes
        torch.manual_seed(100 + step)
        z = z0 + torch.randn(dim) * (0.1 + 0.5 * step)
        assert bound(z) >= exact(z) - 1e-5, f"majorization violated at step {step}"


def test_diagonal_restriction_is_not_claimed_to_majorize():
    r"""Guards the docstring's honesty: there exists a direction where the diagonal-only quadratic
    falls BELOW the full one, so the strict bound genuinely does not survive the restriction."""
    torch.manual_seed(0)
    vocab, dim = 16, 4
    weight = torch.randn(vocab, dim)
    H = 0.5 * (torch.eye(vocab) - torch.ones(vocab, vocab) / vocab)
    curvature = weight.T @ H @ weight
    diagonal = torch.diag(bohning_curvature_diagonal(weight))
    gaps = []
    for step in range(50):
        torch.manual_seed(200 + step)
        v = torch.randn(dim)
        gaps.append((v @ diagonal @ v - v @ curvature @ v).item())
    assert min(gaps) < 0.0, "expected the diagonal form to undershoot in some direction"


def test_linear_term_is_the_pullback_of_the_residual():
    r"""``g = W^T(e_{x_t} - softmax(W mu_p + b))``, and the streaming tiles reproduce the dense form."""
    torch.manual_seed(0)
    vocab, dim = 16, 4
    weight = torch.randn(vocab, dim)
    bias = torch.randn(vocab)
    mu_p = torch.randn(2, 4, dim)
    tokens = _tokens(3, vocab=vocab)

    _, g = bohning_emission_terms(mu_p, weight, tokens, bias=bias)
    probs = torch.softmax(mu_p @ weight.T + bias, dim=-1)             # dense reference
    onehot = torch.nn.functional.one_hot(tokens, vocab).to(probs.dtype)
    assert torch.allclose(g, (onehot - probs) @ weight, atol=1e-5)


def test_streaming_is_independent_of_the_vocab_tile():
    torch.manual_seed(0)
    weight = torch.randn(16, 4)
    mu_p = torch.randn(2, 4, 4)
    tokens = _tokens(3)
    _, g_big = bohning_emission_terms(mu_p, weight, tokens, vocab_chunk=1024)
    _, g_small = bohning_emission_terms(mu_p, weight, tokens, vocab_chunk=3)
    assert torch.allclose(g_big, g_small, atol=1e-6)


# --------------------------------------------------------------------------- the pure path

def test_default_is_off_and_builds_no_emission():
    cfg = VFE3Config(**BASE)
    assert cfg.emission_mode == "off" and cfg.emission_weight == 0.0
    model = VFEModel(cfg)
    assert model.prior_bank.emission_proj_weight is None
    assert model._emission_terms(_tokens(1), torch.zeros(2, 4, 4)) is None


def test_zero_weight_is_byte_identical_to_off():
    _, loss_off, _ = _forward()
    with pytest.warns(UserWarning, match="INERT"):
        _, loss_zero, _ = _forward(emission_mode="shared", emission_weight=0.0)
    assert torch.equal(loss_off, loss_zero)


@pytest.mark.parametrize("mode", ["shared", "separate"])
def test_live_emission_moves_the_belief(mode):
    _, loss_off, _ = _forward()
    _, loss_on, _ = _forward(emission_mode=mode, emission_weight=0.5)
    assert not torch.equal(loss_off, loss_on)


@pytest.mark.parametrize("mode,table", [("shared", "output_proj_weight"),
                                        ("separate", "emission_proj_weight")])
def test_the_readout_table_receives_an_e_step_gradient(mode, table):
    r"""The 'shared' mode exists to tie the belief's inference objective to the decoder that scores
    the prediction; that tie is only real if the E-step reaches the table."""
    model, loss, _ = _forward(emission_mode=mode, emission_weight=0.5)
    loss.backward()
    grad = getattr(model.prior_bank, table).grad
    assert grad is not None and float(grad.norm()) > 0.0


def test_separate_mode_allocates_its_own_table_and_shared_does_not():
    _, model_shared = None, VFEModel(VFE3Config(**BASE, emission_mode="shared", emission_weight=0.5))
    assert model_shared.prior_bank.emission_proj_weight is None
    model_separate = VFEModel(VFE3Config(**BASE, emission_mode="separate", emission_weight=0.5))
    table = model_separate.prior_bank.emission_proj_weight
    assert table is not None and tuple(table.shape) == (16, 4)


def test_separate_table_is_in_an_optimizer_group():
    from vfe3.train import build_optimizer
    cfg = VFE3Config(**BASE, emission_mode="separate", emission_weight=0.5)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    table = model.prior_bank.emission_proj_weight
    assert any(any(p is table for p in g["params"]) for g in opt.param_groups), \
        "emission_proj_weight must be stepped, not silently frozen"


# --------------------------------------------------------------------------- non-leakage

def test_emission_reads_the_current_token_and_never_the_target():
    r"""THE load-bearing property. The factor is admissible only because it conditions on x_t, which
    the prefix already contains. If the belief moved with the held-out target, evaluation would run
    a target-conditioned recognition law and every reported perplexity would be invalid."""
    torch.manual_seed(0)
    cfg = VFE3Config(**BASE, emission_mode="shared", emission_weight=0.5)
    model = VFEModel(cfg)
    tokens = _tokens(1)
    belief_a, _ = model.forward_beliefs(tokens)
    belief_b, _ = model.forward_beliefs(tokens)
    assert torch.equal(belief_a.mu, belief_b.mu)                     # determinism control

    # the belief is produced from token_ids alone; targets never enter its production
    _, loss_1, ce_1 = model(tokens, targets=_tokens(2))
    _, loss_2, ce_2 = model(tokens, targets=_tokens(9))
    belief_c, _ = model.forward_beliefs(tokens)
    assert torch.equal(belief_a.mu, belief_c.mu), "belief moved with the target -- that is a leak"
    assert not torch.equal(ce_1, ce_2), "different targets must still change the CE"


def test_emission_changes_with_the_current_token():
    """The converse: it must actually depend on x_t, or it is not an observation factor."""
    torch.manual_seed(0)
    cfg = VFE3Config(**BASE, emission_mode="shared", emission_weight=0.5)
    model = VFEModel(cfg)
    belief_a, _ = model.forward_beliefs(_tokens(1))
    belief_b, _ = model.forward_beliefs(_tokens(4))
    assert not torch.equal(belief_a.mu, belief_b.mu)


# --------------------------------------------------------------------------- fail closed

def test_emission_requires_the_closed_form_route():
    with pytest.raises(ValueError, match="requires e_step_update='mm_exact'"):
        VFE3Config(**{**BASE, "e_step_update": "gradient"},
                   emission_mode="shared", emission_weight=0.5)


def test_shared_mode_requires_the_linear_decode():
    with pytest.raises(ValueError, match="output_proj_weight"):
        VFE3Config(**{**BASE, "use_prior_bank": True},
                   emission_mode="shared", emission_weight=0.5)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="emission_mode must be"):
        VFE3Config(**BASE, emission_mode="bohning")


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError, match="emission_weight must be"):
        VFE3Config(**BASE, emission_mode="shared", emission_weight=-1.0)
