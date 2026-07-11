r"""Phase 1 tests for the ring goal-steering environment (vfe3/inference/ring_task.py; spec Section 4.1).
Deterministic env/data/preference tests (no training) plus a harness smoke test on an untrained model.
"""
import pytest
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
    logp = rt.ring_preference(goals, beta_C=5.0, support_floor=-45.0)  # (1, V)
    p = logp.exp()[0]
    assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-5)
    assert int(p.argmax()) == 4                            # peaks on the goal
    assert float(p[rt.M:].sum()) < 1e-6                    # ~0 mass on non-state tokens
    # strictly decreasing with ring distance: p(goal) > p(dist 1) > p(dist 2)
    assert float(p[4]) > float(p[5]) > float(p[6])
    assert float(p[5]) == float(p[3])                      # symmetric in ring distance (dist 1 both sides)


def test_ring_preference_support_floor_is_explicit_and_normalized():
    goals = torch.tensor([4])
    hard = rt.ring_preference(goals, beta_C=5.0)
    assert torch.isfinite(hard[:, :rt.M]).all()
    assert torch.isneginf(hard[:, rt.M:]).all()
    finite = rt.ring_preference(goals, beta_C=5.0, support_floor=-45.0)
    assert torch.isfinite(finite).all()
    assert torch.allclose(finite.exp().sum(dim=-1), torch.ones(1), atol=1e-6)


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
    pref = rt.ring_preference(goals, beta_C=5.0, support_floor=-45.0)
    with torch.no_grad():
        out = get_policy("efe_one_step")(ctx, topk.unsqueeze(-1), pref, model, gamma=1.0, base_logits=base)
    assert torch.isfinite(out.risk).all()
    assert torch.isfinite(out.score).all()
    assert torch.isfinite(out.policy_posterior).all()
    assert torch.allclose(out.policy_posterior.sum(-1), torch.ones(2), atol=1e-5)


def test_decode_menu_rejects_invalid_sampling_inputs():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    for temperature in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            rt._decode_menu(logits, "temp_sample", temperature=temperature)
    for top_p in (0.0, -0.1, 1.1, float("nan"), float("inf")):
        for mode in ("nucleus", "typical"):
            with pytest.raises(ValueError, match="top_p"):
                rt._decode_menu(logits, mode, top_p=top_p)
    for bad_logits in (
        torch.tensor([[0.0, float("nan")]]),
        torch.tensor([[0.0, float("inf")]]),
        torch.full((1, 2), float("-inf")),
    ):
        with pytest.raises(ValueError, match="finite"):
            rt._decode_menu(bad_logits, "temp_sample")


def test_run_episodes_smoke_shapes_on_untrained_model():
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=rt.V, embed_dim=12, n_heads=2, max_seq_len=rt.SEQ_LEN,
                     n_layers=1, n_e_steps=1, use_prior_bank=False)
    model = VFEModel(cfg)
    model.eval()
    goals = torch.tensor([2, 7, 11, 0])
    s0 = torch.tensor([5, 7, 3, 8])                        # episode 1 starts at goal already
    out = rt.run_episodes(model, goals, s0, "efe_one_step", gamma=1.0, top_k=8, budget=5)
    assert set(out) == {"correct", "steps_to_goal", "frac_at_goal", "mean_risk", "mean_ambiguity"}
    assert out["correct"].shape == (4,) and out["correct"].dtype == torch.bool
    assert out["steps_to_goal"].shape == (4,) and out["frac_at_goal"].shape == (4,)
    assert (out["steps_to_goal"] <= 5).all()
    assert torch.isfinite(out["mean_risk"]) and torch.isfinite(out["mean_ambiguity"])  # scorer diagnostics


def test_run_episodes_rejects_invalid_context_before_policy_side_effects(monkeypatch):
    cfg = VFE3Config(vocab_size=rt.V, embed_dim=12, n_heads=2, max_seq_len=5,
                     n_layers=1, n_e_steps=1, use_prior_bank=False)
    model = VFEModel(cfg)
    goals = torch.tensor([2])
    states = torch.tensor([5])

    def fail_forward(*args, **kwargs):
        pytest.fail("invalid ring context reached base forward")

    monkeypatch.setattr(model, "forward", fail_forward)
    with pytest.raises(
        ValueError,
        match=r"context length N=5 plus candidate length L=1 exceeds max_seq_len=5",
    ):
        rt.run_episodes(model, goals, states, "efe_one_step", budget=1)

    monkeypatch.setattr(
        rt,
        "render_context",
        lambda goals, states: torch.empty((goals.shape[0], 0), dtype=torch.long),
    )
    with pytest.raises(ValueError, match=r"context must be nonempty, got N=0"):
        rt.run_episodes(model, goals, states, "efe_one_step", budget=1)


def test_run_episodes_phase2_baseline_arms_run_and_stay_valid():
    # Phase 2 (spec 4.3): greedy reference + goal-free sampling baselines run end-to-end and carry no
    # scorer diagnostics (they do not score risk/ambiguity).
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=rt.V, embed_dim=12, n_heads=2, max_seq_len=rt.SEQ_LEN,
                     n_layers=1, n_e_steps=1, use_prior_bank=False)
    model = VFEModel(cfg)
    model.eval()
    goals = torch.tensor([2, 7, 11, 0])
    s0 = torch.tensor([5, 7, 3, 8])
    for mode, kw in (("greedy_ref", {}), ("temp_sample", {"temperature": 2.0}),
                     ("nucleus", {"top_p": 0.9}), ("typical", {"top_p": 0.9})):
        out = rt.run_episodes(model, goals, s0, mode, budget=5,
                              generator=torch.Generator().manual_seed(2), **kw)
        assert out["correct"].shape == (4,) and out["correct"].dtype == torch.bool
        assert (out["steps_to_goal"] <= 5).all()
        assert float(out["mean_risk"]) == 0.0 and float(out["mean_ambiguity"]) == 0.0


def test_closed_loop_causality_holds_by_construction():
    # v1 lesion gate (spec 4.6): the committed action measurably changes the next observation.
    assert rt.closed_loop_causality_holds() is True


def test_predictive_adequacy_runs_and_is_a_fraction():
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=rt.V, embed_dim=12, n_heads=2, max_seq_len=rt.SEQ_LEN,
                     n_layers=1, n_e_steps=1, use_prior_bank=False)
    model = VFEModel(cfg)
    model.eval()
    acc = rt.predictive_adequacy(model, n=256, generator=torch.Generator().manual_seed(1))
    assert 0.0 <= acc <= 1.0
