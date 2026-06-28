r"""Phase 1 tests for the ring goal-steering environment (vfe3/inference/ring_task.py; spec Section 4.1).
Deterministic env/data/preference tests (no training) plus a harness smoke test on an untrained model.
"""
import torch

from vfe3.config import VFE3Config
from vfe3.inference import ring_task as rt
from vfe3.model.model import VFEModel


def test_transition_wraps_on_ring():
    s = torch.tensor([0, 5, 15])
    inc = torch.tensor([rt.INC, rt.INC, rt.INC])
    dec = torch.tensor([rt.DEC, rt.DEC, rt.DEC])
    stay = torch.tensor([rt.STAY, rt.STAY, rt.STAY])
    assert rt.transition(s, inc).tolist() == [1, 6, 0]      # 15 -> 0 wrap
    assert rt.transition(s, dec).tolist() == [15, 4, 14]    # 0 -> 15 wrap
    assert rt.transition(s, stay).tolist() == [0, 5, 15]
    # a non-action token is a wasted STAY (delta 0)
    assert rt.transition(torch.tensor([7]), torch.tensor([rt.GOAL])).tolist() == [7]


def test_render_context_layout():
    ctx = rt.render_context(torch.tensor([3, 9]), torch.tensor([1, 14]))
    assert ctx.shape == (2, 5)
    assert ctx[0].tolist() == [rt.GOAL, 3, rt.SEP, rt.CUR, 1]
    assert ctx[1].tolist() == [rt.GOAL, 9, rt.SEP, rt.CUR, 14]


def test_sample_batch_shapes_and_transition_target():
    gen = torch.Generator().manual_seed(0)
    tokens, targets = rt.sample_batch(64, generator=gen)
    assert tokens.shape == (64, rt.SEQ_LEN) and targets.shape == (64, rt.SEQ_LEN)
    assert (targets[:, -1] == -100).all()                  # no target after the final state
    assert torch.equal(targets[:, :-1], tokens[:, 1:])     # next-token shift
    # the rendered final state is the true transition of (state, action)
    s, a, s_next = tokens[:, 4], tokens[:, 5], tokens[:, 6]
    assert torch.equal(s_next, rt.transition(s, a))
    assert ((a == rt.DEC) | (a == rt.STAY) | (a == rt.INC)).all()


def test_ring_preference_is_distance_graded():
    goals = torch.tensor([4])
    logp = rt.ring_preference(goals, beta_C=5.0)           # (1, V)
    p = logp.exp()[0]
    assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-5)
    assert int(p.argmax()) == 4                            # peaks on the goal
    assert float(p[rt.M:].sum()) < 1e-6                    # ~0 mass on non-state tokens
    # strictly decreasing with ring distance: p(goal) > p(dist 1) > p(dist 2)
    assert float(p[4]) > float(p[5]) > float(p[6])
    assert float(p[5]) == float(p[3])                      # symmetric in ring distance (dist 1 both sides)


def test_ring_preference_keeps_efe_score_finite():
    # Regression: a -inf preference floor on non-state tokens made forward KL(q || p_task) diverge to
    # +inf wherever the model's q had off-state mass, collapsing score to inf and the posterior to nan.
    from vfe3.inference.policy import get_policy
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=rt.V, embed_dim=12, n_heads=2, max_seq_len=rt.SEQ_LEN,
                     n_layers=1, n_e_steps=1, use_prior_bank=False)
    model = VFEModel(cfg)
    model.eval()
    goals = torch.tensor([3, 9])
    ctx = rt.render_context(goals, torch.tensor([0, 5]))
    base = model.forward(ctx)[:, -1, :]
    topk = base.topk(8, dim=-1).indices
    pref = rt.ring_preference(goals, beta_C=5.0)
    with torch.no_grad():
        out = get_policy("efe_one_step")(ctx, topk.unsqueeze(-1), pref, model, gamma=1.0, base_logits=base)
    assert torch.isfinite(out.risk).all()
    assert torch.isfinite(out.score).all()
    assert torch.isfinite(out.policy_posterior).all()
    assert torch.allclose(out.policy_posterior.sum(-1), torch.ones(2), atol=1e-5)


def test_run_episodes_smoke_shapes_on_untrained_model():
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=rt.V, embed_dim=12, n_heads=2, max_seq_len=rt.SEQ_LEN,
                     n_layers=1, n_e_steps=1, use_prior_bank=False)
    model = VFEModel(cfg)
    model.eval()
    goals = torch.tensor([2, 7, 11, 0])
    s0 = torch.tensor([5, 7, 3, 8])                        # episode 1 starts at goal already
    out = rt.run_episodes(model, goals, s0, "efe_one_step", gamma=1.0, top_k=8, budget=5)
    assert set(out) == {"correct", "steps_to_goal", "frac_at_goal"}
    assert out["correct"].shape == (4,) and out["correct"].dtype == torch.bool
    assert out["steps_to_goal"].shape == (4,) and out["frac_at_goal"].shape == (4,)
    assert (out["steps_to_goal"] <= 5).all()


def test_predictive_adequacy_runs_and_is_a_fraction():
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=rt.V, embed_dim=12, n_heads=2, max_seq_len=rt.SEQ_LEN,
                     n_layers=1, n_e_steps=1, use_prior_bank=False)
    model = VFEModel(cfg)
    model.eval()
    acc = rt.predictive_adequacy(model, n=256, generator=torch.Generator().manual_seed(1))
    assert 0.0 <= acc <= 1.0
