r"""C9 forward-fidelity tests for the viz belief extractors (audit 2026-07-01).

``_encode_one`` and ``belief_bank`` must replay model.forward's belief pipeline EXACTLY --
including the ``s_e_step`` model-channel anchor and the ``precision_weighted_attention``
log-prior fold (``model._fold_precision_bias``) -- so every figure describes the model that
actually trained. Pins:

  * ``converged_state`` (via the shared ``_encode_one``) equals the forward belief under
    ``precision_weighted_attention=True`` (pre-fix: the fold was missing -> a different beta);
  * ``belief_bank`` equals the forward belief under ``s_e_step=True`` +
    ``precision_weighted_attention=True`` (pre-fix: neither anchor nor fold was applied);
  * on the DEFAULT config (both toggles off) the fold + anchor are exact no-ops: the fixed
    extractors are byte-identical to the pre-fix replay, so the pure path is untouched.

Each fidelity test carries a discrimination guard (the pre-fix replay DIFFERS from forward),
so the equality assertions cannot pass vacuously. The forward reference is
``forward_beliefs(capture=...)['out']`` -- the raw pre-final_norm ``vfe_stack`` output, which
is exactly the tensor the extractors return.

Device-agnostic (CPU default; set VFE3_TEST_DEVICE=cuda for the GPU).
"""
import os

import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.model.stack import vfe_stack
from vfe3.viz import extract

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _prefix_encode_one(model, token_ids: torch.Tensor):
    r"""The PRE-C9 ``_encode_one`` replay: encode -> pos_phi, NO s-anchor, NO precision fold."""
    enc = model.prior_bank.encode(token_ids[:1])
    belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=model._apply_pos_phi(enc.phi[0]))
    n = belief.mu.shape[0]
    log_prior = model._attention_log_prior(n, token_ids.device)
    rope = model._rope_rotation(n, token_ids.device)
    return belief, log_prior, rope


def _stack(model, belief, log_prior, rope):
    r"""The extractors' shared ``vfe_stack`` replay (mirrors belief_bank / converged_state)."""
    cfg = model.cfg
    return vfe_stack(
        belief, belief.mu, belief.sigma, model.group, cfg,
        log_prior=log_prior, block_norm=model.block_norm,
        head_mixer=model.head_mixer, cg_coupling=model.cg_coupling,
        lambda_beta=cfg.lambda_beta,
        connection_W=getattr(model, "connection_W", None),
        connection_M=getattr(model, "connection_M", None),
        connection_L=getattr(model, "connection_L", None),
        rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
    )


def _forward_reference(model, tokens: torch.Tensor):
    r"""forward's raw pre-final_norm converged belief (``capture['out']``, the extractor target)."""
    cap: dict = {}
    with torch.no_grad():
        model.forward_beliefs(tokens, capture=cap)
    return cap["out"]


def test_extractor_belief_matches_forward_under_precision_weighted_attention():
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=8,
                     precision_weighted_attention=True)
    model = VFEModel(cfg).to(DEVICE)
    with torch.no_grad():                            # vary tr Sigma_j across the vocab so the
        model.prior_bank.sigma_log_embed.add_(       # reliability bias is NOT a constant key shift
            0.5 * torch.randn_like(model.prior_bank.sigma_log_embed))
    tokens = torch.randint(0, 16, (1, 8), device=DEVICE)
    ref = _forward_reference(model, tokens)          # (1, N, K) raw stack output
    state = extract.converged_state(model, tokens)   # consumes the shared _encode_one
    assert torch.allclose(state["mu"],    ref.mu[0],    atol=1e-5)
    assert torch.allclose(state["sigma"], ref.sigma[0], atol=1e-5)
    assert torch.allclose(state["phi"],   ref.phi[0],   atol=1e-5)
    # Discrimination guard: the PRE-fix replay (raw, unfolded prior) converges to a DIFFERENT
    # belief, so the equality above is not vacuous.
    with torch.no_grad():
        belief0, raw_prior, rope0 = _prefix_encode_one(model, tokens)
        out0 = _stack(model, belief0, raw_prior, rope0)
    assert (out0.mu - ref.mu[0]).abs().max().item() > 1e-4


def test_belief_bank_matches_forward_under_s_e_step():
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=8,
                     s_e_step=True, prior_source="model_channel", lambda_h=0.5,
                     precision_weighted_attention=True)
    model = VFEModel(cfg).to(DEVICE)
    with torch.no_grad():                            # model_channel: encode reads the s tables
        model.prior_bank.s_sigma_log_embed.add_(
            0.5 * torch.randn_like(model.prior_bank.s_sigma_log_embed))
    tokens = torch.randint(0, 16, (2, 8), device=DEVICE)
    b, n = tokens.shape
    ref = _forward_reference(model, tokens)          # (B, N, K) raw stack output
    bank = extract.belief_bank(model, [tokens])
    assert torch.allclose(bank["mu"],    ref.mu.reshape(b * n, -1),                       atol=1e-5)
    assert torch.allclose(bank["sigma"], ref.sigma.reshape(b * n, *ref.sigma.shape[2:]), atol=1e-5)
    assert torch.allclose(bank["phi"],   ref.phi.reshape(b * n, -1),                     atol=1e-5)
    # Discrimination guard: the PRE-fix bank body (no s-anchor, no fold) diverges from forward.
    with torch.no_grad():
        beliefs0 = model.prior_bank.encode(tokens)
        beliefs0 = beliefs0._replace(phi=model._apply_pos_phi(beliefs0.phi))
        raw_prior = model._attention_log_prior(n, tokens.device)
        rope0 = model._rope_rotation(n, tokens.device)
        out0 = _stack(model, beliefs0, raw_prior, rope0)
    assert (out0.mu - ref.mu).abs().max().item() > 1e-4


def test_extractor_fold_and_anchor_are_noops_on_default_config():
    # Regression guard: with precision_weighted_attention=False and s_e_step=False (the pure
    # default path) the C9 edits are exact no-ops -- the fixed extractors reproduce the pre-fix
    # replay byte-for-byte.
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=8)
    model = VFEModel(cfg).to(DEVICE)
    model.eval()
    tokens = torch.randint(0, 16, (2, 8), device=DEVICE)
    with torch.no_grad():
        belief, log_prior, rope = extract._encode_one(model, tokens)
        belief0, log_prior0, _ = _prefix_encode_one(model, tokens)
    assert torch.equal(belief.mu,    belief0.mu)
    assert torch.equal(belief.sigma, belief0.sigma)
    assert torch.equal(belief.phi,   belief0.phi)
    assert (log_prior is None and log_prior0 is None) or torch.equal(log_prior, log_prior0)
    bank = extract.belief_bank(model, [tokens])
    with torch.no_grad():                            # the PRE-fix bank body (no anchor, no fold)
        beliefs0 = model.prior_bank.encode(tokens)
        beliefs0 = beliefs0._replace(phi=model._apply_pos_phi(beliefs0.phi))
        raw_prior = model._attention_log_prior(tokens.shape[1], tokens.device)
        rope0 = model._rope_rotation(tokens.shape[1], tokens.device)
        out0 = _stack(model, beliefs0, raw_prior, rope0)
    b, n = tokens.shape
    assert torch.equal(bank["mu"],    out0.mu.reshape(b * n, -1))
    assert torch.equal(bank["sigma"], out0.sigma.reshape(b * n, *out0.sigma.shape[2:]))
    assert torch.equal(bank["phi"],   out0.phi.reshape(b * n, -1))
