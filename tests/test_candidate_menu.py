r"""Tests for the bounded H-step candidate-policy menu (vfe3/inference/candidate_menu.py, audit
PB-05). The beam is bounded (at most ``width`` live sequences at every depth), not the Cartesian
top-``width`` product, so the second test checks it against an EXHAUSTIVE two-step search on a
tiny vocabulary and the third traces the batch size ``model.rollout_beliefs`` sees at every depth.
"""

import torch

from vfe3.config import VFE3Config
from vfe3.inference.candidate_menu import build_topk_policy_menu
from vfe3.model.model import VFEModel


def tiny_model(seed: int = 0, **overrides) -> VFEModel:
    """A tiny fixed-seed VFEModel (K < 6, single-digit dims elsewhere) -- the CPU-test fixture
    pattern established in tests/test_generate.py's ``_tiny_model``."""
    base = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0, seed=seed)
    base.update(overrides)
    cfg = VFE3Config(**base)
    torch.manual_seed(seed)
    return VFEModel(cfg)


def test_topk_policy_menu_has_horizon_shape_and_normalized_prior():
    model = tiny_model(embed_dim=4, n_heads=1, vocab_size=11)
    context = torch.tensor([[1, 2]], dtype=torch.long)
    base_logits = model.rollout_beliefs(
        context, return_logits=True, decode_last=True
    )[1][:, 0, :]
    candidates, log_prior = build_topk_policy_menu(
        context, base_logits, model, horizon=3, width=4
    )
    assert candidates.shape == (1, 4, 3)
    assert log_prior.shape == (1, 4)
    torch.testing.assert_close(log_prior.exp().sum(-1), torch.ones(1))


def test_topk_policy_menu_matches_exhaustive_search_on_tiny_vocabulary():
    model = tiny_model(embed_dim=4, n_heads=1, vocab_size=4)
    context = torch.tensor([[1, 2]], dtype=torch.long)
    _, decoded = model.rollout_beliefs(
        context, return_logits=True, decode_last=True
    )
    base_logits = decoded[:, 0, :]
    candidates, log_prior = build_topk_policy_menu(
        context, base_logits, model, horizon=2, width=4
    )

    first = torch.arange(4).reshape(1, 4, 1)
    expanded_context = context.unsqueeze(1).expand(1, 4, 2)
    extended = torch.cat([expanded_context, first], dim=-1).reshape(4, 3)
    _, next_decoded = model.rollout_beliefs(
        extended, return_logits=True, decode_last=True
    )
    first_logp = torch.log_softmax(base_logits[0], dim=-1)
    next_logp = torch.log_softmax(next_decoded[:, 0, :], dim=-1)
    joint = first_logp[:, None] + next_logp
    expected_score, flat = joint.reshape(-1).topk(4)
    expected = torch.stack((flat // 4, flat % 4), dim=-1)

    torch.testing.assert_close(candidates[0], expected)
    torch.testing.assert_close(
        log_prior[0], torch.log_softmax(expected_score, dim=-1)
    )


def test_topk_policy_menu_batches_beams_and_never_exceeds_width(monkeypatch):
    model = tiny_model(embed_dim=4, n_heads=1, vocab_size=7)
    context = torch.tensor([[1, 2], [2, 3]], dtype=torch.long)
    _, decoded = model.rollout_beliefs(
        context, return_logits=True, decode_last=True
    )
    base_logits = decoded[:, 0, :]
    observed_batch_sizes = []
    original = model.rollout_beliefs

    def traced_rollout(token_ids, *args, **kwargs):
        observed_batch_sizes.append(token_ids.shape[0])
        return original(token_ids, *args, **kwargs)

    monkeypatch.setattr(model, "rollout_beliefs", traced_rollout)
    candidates, _ = build_topk_policy_menu(
        context, base_logits, model, horizon=4, width=3
    )

    assert candidates.shape == (2, 3, 4)
    assert observed_batch_sizes == [6, 6, 6]
