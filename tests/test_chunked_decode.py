r"""Equivalence gate for the fused chunked-vocab decode+CE (decode_mode='diagonal_chunked').

The chunked CE must equal the full-decode CE (decode -> F.cross_entropy) to atol-1e-3 (the
cancellation-sensitive decode's golden tolerance, per docs/perf/2026-05-31-speedup-opportunities.md)
for several chunk sizes, honor ignore_index identically, and give prior-table grads allclose to the
full path. The chunking is a memory reassociation of the SAME CE, never a formula change.
"""

import torch
import torch.nn.functional as F

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _full_ce(model: VFEModel, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Reference CE on the full-decode 'diagonal' path: decode -> F.cross_entropy."""
    logits = model.prior_bank.decode(*_converged(model, tokens))      # (B, N, V)
    flat_logits = logits.reshape(-1, model.cfg.vocab_size)
    flat_targets = targets.reshape(-1)
    return F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)


def _converged(model: VFEModel, tokens: torch.Tensor):
    """Run the E-step stack and return (mu_final, sigma_final) the decode consumes."""
    from vfe3.model.stack import vfe_stack
    B, N = tokens.shape
    beliefs = model.prior_bank.encode(tokens)
    log_prior = model._attention_log_prior(N, tokens.device)
    out = vfe_stack(beliefs, beliefs.mu, beliefs.sigma, model.group, model.cfg,
                    log_prior=log_prior, block_norm=model.block_norm)
    return out.mu.float(), out.sigma.float()


def _model(vocab_size=64, **kw):
    cfg = VFE3Config(vocab_size=vocab_size, embed_dim=4, n_heads=2, max_seq_len=6,
                     n_layers=1, n_e_steps=1, e_mu_lr=0.05, e_phi_lr=0.0, **kw)
    return VFEModel(cfg)


def test_chunked_ce_matches_full_ce_multiple_chunk_sizes():
    torch.manual_seed(0)
    V = 64
    tokens = torch.randint(0, V, (3, 6))
    targets = torch.randint(0, V, (3, 6))
    full = _model(vocab_size=V)
    full_loss = _full_ce(full, tokens, targets)
    # chunk sizes: divides V evenly, does NOT divide V, and >= V (single chunk).
    for chunk in (16, 8192, V, 7, 100):
        ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=chunk)
        ch.load_state_dict(full.state_dict())
        mu, sigma = _converged(ch, tokens)
        ce = ch.prior_bank.decode_ce_diagonal_chunked(mu, sigma, targets)
        assert torch.allclose(ce, full_loss, atol=1e-3), (
            f"chunk={chunk}: chunked CE {ce.item()} != full CE {full_loss.item()}"
        )


def test_chunked_ce_honors_ignore_index():
    torch.manual_seed(1)
    V = 50
    tokens = torch.randint(0, V, (2, 6))
    targets = torch.randint(0, V, (2, 6))
    targets[0, 0] = -100
    targets[1, 3] = -100
    targets[1, 5] = -100
    full = _model(vocab_size=V)
    full_loss = _full_ce(full, tokens, targets)
    ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=13)
    ch.load_state_dict(full.state_dict())
    mu, sigma = _converged(ch, tokens)
    ce = ch.prior_bank.decode_ce_diagonal_chunked(mu, sigma, targets)
    assert torch.allclose(ce, full_loss, atol=1e-3)


def test_chunked_ce_all_ignore_is_finite_zero():
    # Every target == -100: full path emits a finite grad-connected zero; chunked must too.
    torch.manual_seed(2)
    V = 32
    tokens = torch.randint(0, V, (2, 4))
    targets = torch.full((2, 4), -100)
    ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=10)
    mu, sigma = _converged(ch, tokens)
    mu = mu.detach().requires_grad_(True)
    ce = ch.prior_bank.decode_ce_diagonal_chunked(mu, sigma, targets)
    assert torch.isfinite(ce) and ce.item() == 0.0
    ce.backward()                                          # grad-connected (no autograd error)


def test_chunked_ce_grad_matches_full():
    torch.manual_seed(3)
    V = 48
    tokens = torch.randint(0, V, (2, 5))
    targets = torch.randint(0, V, (2, 5))

    full = _model(vocab_size=V)
    full_loss = _full_ce(full, tokens, targets)
    full_loss.backward()
    g_mu_full = full.prior_bank.mu_embed.grad.clone()
    g_sig_full = full.prior_bank.sigma_log_embed.grad.clone()

    ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=11)
    ch.load_state_dict(full.state_dict())
    # Drive the chunked CE through the full forward so the gradient reaches the prior tables.
    _, loss, _ = ch(tokens, targets)
    loss.backward()
    g_mu_ch = ch.prior_bank.mu_embed.grad
    g_sig_ch = ch.prior_bank.sigma_log_embed.grad

    assert torch.allclose(g_mu_ch, g_mu_full, atol=1e-3)
    assert torch.allclose(g_sig_ch, g_sig_full, atol=1e-3)


def test_chunked_forward_loss_matches_diagonal_forward_loss():
    # End-to-end: model.forward under diagonal_chunked returns the same loss/ce as diagonal.
    torch.manual_seed(4)
    V = 40
    tokens = torch.randint(0, V, (3, 6))
    targets = torch.randint(0, V, (3, 6))
    base = _model(vocab_size=V)
    _, loss_full, ce_full = base(tokens, targets)

    ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=9)
    ch.load_state_dict(base.state_dict())
    out = ch(tokens, targets)
    logits_ch, loss_ch, ce_ch = out
    assert logits_ch is None                               # fused path forms no (B,N,V) logits
    assert torch.allclose(loss_ch, loss_full, atol=1e-3)
    assert torch.allclose(ce_ch, ce_full, atol=1e-3)


def test_chunked_inference_targets_none_returns_full_logits():
    # targets=None (inference/generation) falls back to materialized logits equal to diagonal.
    torch.manual_seed(5)
    V = 40
    tokens = torch.randint(0, V, (2, 5))
    base = _model(vocab_size=V)
    logits_full = base(tokens)

    ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=9)
    ch.load_state_dict(base.state_dict())
    logits_ch = ch(tokens)
    assert logits_ch.shape == (2, 5, V)
    assert torch.allclose(logits_ch, logits_full, atol=1e-3)
