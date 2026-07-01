r"""Phase 1 correctness tests for the one-step EFE policy scorer
(vfe3/inference/policy.py; spec docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md).

Two layers: (1) pure-algebra unit tests of the EFE terms on hand-computable categoricals, with no
model; (2) end-to-end scorer tests on a small VFEModel, headlined by the honesty invariant that the
information-gain term I is IDENTICALLY zero at the v1 operating regime (horizon=1, sigma-free point
belief; spec Section 2.8), so the score collapses to the pragmatic cross-entropy.
"""
import math

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.inference.policy import (
    PolicyScore,
    _efe_terms,
    _rollout_predictive,
    get_ambiguity,
    get_policy,
    get_preference,
)
from vfe3.model.model import VFEModel


# ---------- (1) pure-algebra unit tests (no model) ----------

def test_efe_terms_match_hand_computation():
    q = torch.tensor([[[0.5, 0.3, 0.2]]])                    # (1,1,3)
    pC = torch.tensor([0.2, 0.3, 0.5])                       # (3,)
    risk, pred_ent = _efe_terms(q.log(), pC.log())
    # risk = KL(q||pC); pred_ent = H[q]; risk + pred_ent = cross-entropy -sum q log pC
    assert abs(float(risk) - 0.27490) < 1e-4
    assert abs(float(pred_ent) - 1.02965) < 1e-4
    ce = -(q * pC.log()).sum()
    assert abs(float(risk + pred_ent) - float(ce)) < 1e-5   # the pragmatic-collapse identity


def test_likelihood_entropy_ambiguity_equals_predictive_entropy():
    q_log = torch.log_softmax(torch.randn(2, 4, 7), dim=-1)
    _, pred_ent = _efe_terms(q_log, torch.log_softmax(torch.randn(7), dim=-1))
    amb = get_ambiguity("likelihood_entropy")(q_log)
    assert torch.allclose(amb, pred_ent, atol=1e-6)          # so the MI bridge I = pred_ent - amb == 0


# ---------- (2) end-to-end scorer tests on a small model ----------

def _model(**kw):
    d = dict(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=12, n_layers=2,
             n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.02, use_prior_bank=False)
    d.update(kw)
    torch.manual_seed(0)
    return VFEModel(VFE3Config(**d))


def _menu(model, context, Kp=4):
    with torch.no_grad():
        base = model.forward(context)[:, -1, :]
    topk = base.topk(Kp, dim=-1).indices
    return topk.unsqueeze(-1), base                          # (B, Kp, 1), (B, V)


def test_scorer_shapes_and_posterior_normalized():
    m = _model()
    ctx = torch.tensor([[1, 2, 3, 4, 5]])
    cand, base = _menu(m, ctx, Kp=4)
    pref = get_preference("task")(m.prior_bank, goal=7, beta_C=5.0)
    with torch.no_grad():
        out = get_policy("efe_one_step")(ctx, cand, pref, m, gamma=1.0, base_logits=base)
    assert isinstance(out, PolicyScore)
    for t in (out.score, out.risk, out.ambiguity, out.epistemic, out.log_prob, out.policy_posterior):
        assert t.shape == (1, 4)
    assert torch.allclose(out.policy_posterior.sum(-1), torch.ones(1), atol=1e-6)


def test_epistemic_identically_zero_at_v1():
    # THE honesty invariant (spec Section 2.8): at horizon=1 over the point belief the MI bridge is 0.
    m = _model()
    ctx = torch.tensor([[3, 1, 4, 1, 5, 9]])
    cand, base = _menu(m, ctx, Kp=5)
    pref = get_preference("task")(m.prior_bank, goal=2, beta_C=5.0)
    with torch.no_grad():
        out = get_policy("efe_one_step")(ctx, cand, pref, m, gamma=2.0, base_logits=base)
    assert torch.equal(out.epistemic, torch.zeros_like(out.epistemic))   # exactly zero, not merely small


def test_score_is_risk_plus_ambiguity_and_collapses_to_cross_entropy():
    m = _model()
    ctx = torch.tensor([[2, 7, 1, 8, 2]])
    cand, base = _menu(m, ctx, Kp=4)
    pref = get_preference("task")(m.prior_bank, goal=5, beta_C=5.0)
    with torch.no_grad():
        out = get_policy("efe_one_step")(ctx, cand, pref, m, base_logits=base)
        q_log, _ = _rollout_predictive(ctx, cand, m, base_logits=base)
    assert torch.allclose(out.score, out.risk + out.ambiguity, atol=1e-6)
    ce = -(q_log.exp() * pref.view(1, 1, -1)).sum(-1)        # -E_q[log p(o|C)]
    assert torch.allclose(out.score, ce, atol=1e-5)          # G = risk + ambiguity = cross-entropy


def test_flat_preference_gives_constant_log_V():
    # spec Section 2.3 / 2.8: flat preference -> G = log V - I = log V (constant) at the v1 point belief.
    m = _model()
    ctx = torch.tensor([[1, 2, 3, 4, 5]])
    cand, base = _menu(m, ctx, Kp=6)
    pref = get_preference("flat")(m.prior_bank)
    with torch.no_grad():
        out = get_policy("efe_one_step")(ctx, cand, pref, m, base_logits=base)
    assert torch.allclose(out.score, torch.full_like(out.score, math.log(16)), atol=1e-4)
    # constant score -> uniform posterior over the menu
    assert torch.allclose(out.policy_posterior, torch.full_like(out.policy_posterior, 1.0 / 6), atol=1e-4)


def test_score_terms_select_components():
    m = _model()
    ctx = torch.tensor([[1, 2, 3, 4, 5]])
    cand, base = _menu(m, ctx, Kp=4)
    pref = get_preference("task")(m.prior_bank, goal=7, beta_C=5.0)
    with torch.no_grad():
        full = get_policy("efe_one_step")(ctx, cand, pref, m, base_logits=base)
        risk_only = get_policy("efe_one_step")(ctx, cand, pref, m, score_terms=("risk",), base_logits=base)
        amb_only = get_policy("efe_one_step")(ctx, cand, pref, m, score_terms=("ambiguity",), base_logits=base)
    assert torch.allclose(risk_only.score, full.risk, atol=1e-6)
    assert torch.allclose(amb_only.score, full.ambiguity, atol=1e-6)


def test_policy_posterior_matches_softmax_formula():
    m = _model()
    ctx = torch.tensor([[1, 2, 3, 4, 5]])
    cand, base = _menu(m, ctx, Kp=4)
    pref = get_preference("task")(m.prior_bank, goal=7, beta_C=5.0)
    gamma = 3.0
    log_prior = torch.log_softmax(torch.randn(1, 4), dim=-1)
    with torch.no_grad():
        out = get_policy("efe_one_step")(ctx, cand, pref, m, gamma=gamma, log_prior=log_prior, base_logits=base)
    manual = torch.softmax(-gamma * out.score + log_prior, dim=-1)
    assert torch.allclose(out.policy_posterior, manual, atol=1e-6)


def test_logprob_control_scores_by_logprob_with_zero_efe_terms():
    m = _model()
    ctx = torch.tensor([[1, 2, 3, 4, 5]])
    cand, base = _menu(m, ctx, Kp=4)
    pref = get_preference("flat")(m.prior_bank)
    with torch.no_grad():
        out = get_policy("logprob_control")(ctx, cand, pref, m, gamma=2.0, base_logits=base)
    z = torch.zeros(1, 4)
    assert torch.equal(out.risk, z) and torch.equal(out.ambiguity, z) and torch.equal(out.epistemic, z)
    assert torch.allclose(out.score, -out.log_prob, atol=1e-6)
    assert torch.allclose(out.policy_posterior, torch.softmax(2.0 * out.log_prob, dim=-1), atol=1e-6)


def test_gates_and_horizon_guards_raise():
    m = _model()
    ctx = torch.tensor([[1, 2, 3, 4, 5]])
    cand, base = _menu(m, ctx, Kp=4)
    pref = get_preference("flat")(m.prior_bank)
    with pytest.raises(ValueError):                          # efe_one_step is H=1 only
        get_policy("efe_one_step")(ctx, cand, pref, m, horizon=2, base_logits=base)
    # efe_rollout (H>1) is unlocked by the Phase-3a cache but REQUIRES it: this _model() (n_layers=2,
    # n_e_steps=2, e_phi_lr>0) is not cache-supported, so a correctly-shaped H=2 call still raises.
    cand2 = torch.randint(0, m.cfg.vocab_size, (1, 4, 2))
    with pytest.raises(NotImplementedError):
        get_policy("efe_rollout")(ctx, cand2, pref, m, horizon=2)
    with pytest.raises(RuntimeError):                        # sigma_mc gated on the sigma-validation gate
        get_ambiguity("sigma_mc")(torch.log_softmax(torch.randn(1, 4, 16), dim=-1))


def test_sigma_mc_raise_names_flag_has_no_consumer():
    # audit F5 (2026-07-01): the gate message must state that policy_sigma_ambiguity_validated ALONE
    # does not unlock sigma_mc -- the flag is a precondition record with no executable consumer until
    # a Phase-3 consumer that reads the validated artifact is added.
    with pytest.raises(RuntimeError) as e:
        get_ambiguity("sigma_mc")(torch.log_softmax(torch.randn(1, 4, 16), dim=-1))
    msg = str(e.value)
    assert "policy_sigma_ambiguity_validated" in msg
    assert "does NOT unlock" in msg and "NO executable consumer" in msg


def test_generate_policy_branch_dispatches_and_stays_in_vocab():
    # policy_mode != 'none' routes generate() through _policy_select; flat preference at v1 gives a
    # uniform EFE score, so greedy selection follows the candidate prior E (base menu) -- a valid run.
    m = _model(policy_mode="efe_one_step", policy_preference="flat", policy_top_k=5, policy_precision=1.0)
    prompt = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        seq = m.generate(prompt, 4, greedy=True)
    assert seq.shape == (1, 7)
    assert int(seq.max()) < 16 and int(seq.min()) >= 0
    assert torch.equal(seq[:, :3], prompt)                  # prompt preserved


def test_preference_builders():
    m = _model()
    V = 16
    flat = get_preference("flat")(m.prior_bank)
    assert torch.allclose(flat, torch.full((V,), -math.log(V)), atol=1e-6)
    # task: peaks on the goal, ~0 mass outside the support set
    support = torch.arange(0, 8)
    task = get_preference("task")(m.prior_bank, goal=3, beta_C=5.0, support=support)
    p = task.exp()
    assert int(p.argmax()) == 3
    assert float(p[8:].sum()) < 1e-5                         # non-support tokens carry ~0 mass
    # batched goal -> (B, V)
    task_b = get_preference("task")(m.prior_bank, goal=torch.tensor([1, 4]), beta_C=5.0)
    assert task_b.shape == (2, V)
    assert int(task_b[0].exp().argmax()) == 1 and int(task_b[1].exp().argmax()) == 4
    # held_out_predictive returns log p_data
    p_data = torch.softmax(torch.randn(V), dim=-1)
    hop = get_preference("held_out_predictive")(m.prior_bank, p_data=p_data)
    assert torch.allclose(hop, p_data.log(), atol=1e-5)


def test_preference_builders_are_device_aware():
    # audit F5 (2026-06-28): the generic builders must honor the requested device, else generate() on
    # CUDA builds a CPU preference and hits a CPU/CUDA mismatch. Default (device=None) stays on CPU for
    # direct callers. CUDA-conditional, per the audit's fix note.
    m = _model()
    assert get_preference("flat")(m.prior_bank).device.type == "cpu"          # default -> CPU
    assert get_preference("flat")(m.prior_bank, device=torch.device("cpu")).device.type == "cpu"
    assert get_preference("task")(m.prior_bank, goal=3, device=torch.device("cpu")).device.type == "cpu"
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        assert get_preference("flat")(m.prior_bank, device=dev).device.type == "cuda"
        assert get_preference("task")(m.prior_bank, goal=torch.tensor([1, 4]),
                                      device=dev).device.type == "cuda"
