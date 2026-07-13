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
from vfe3.contracts import AmbiguityEstimate
from vfe3.inference.policy import (
    PolicyScore,
    _amb_sigma_mc,
    _antithetic_shared_state_samples,
    _efe_terms,
    _policy_posterior,
    _rollout_predictive,
    _validate_policy_context,
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


def test_efe_terms_handle_zero_probability_without_nan():
    q_log = torch.tensor([[[0.0, float("-inf")]]])
    preference = torch.full((2,), -math.log(2.0))
    risk, pred_ent = _efe_terms(q_log, preference)
    assert torch.isfinite(risk).all() and torch.isfinite(pred_ent).all()
    assert torch.allclose(risk, torch.full_like(risk, math.log(2.0)))
    assert torch.equal(pred_ent, torch.zeros_like(pred_ent))


def test_likelihood_entropy_ambiguity_equals_predictive_entropy():
    q_log = torch.log_softmax(torch.randn(2, 4, 7), dim=-1)
    _, pred_ent = _efe_terms(q_log, torch.log_softmax(torch.randn(7), dim=-1))
    est = get_ambiguity("likelihood_entropy")(q_log)         # AmbiguityEstimate (PB-06)
    assert torch.equal(est.predictive_log_prob, q_log)       # point predictive returned unchanged
    assert torch.allclose(est.expected_conditional_entropy, pred_ent, atol=1e-6)  # MI bridge I == 0


# ---------- (2) end-to-end scorer tests on a small model ----------

def _model(**kw):
    d = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=12, n_layers=2,
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


def test_policy_posterior_rejects_all_infinite_candidate_row():
    score = torch.full((1, 4), float("inf"))
    with pytest.raises(ValueError, match="no finite candidate"):
        _policy_posterior(score, 1.0, None)


def test_policy_posterior_rejects_nan_and_positive_infinity_with_row_indices():
    score = torch.tensor([
        [0.0, float("nan"), float("inf")],
        [0.0, float("inf"), 1.0],
        [0.0, float("-inf"), float("inf")],
    ])
    with pytest.raises(ValueError, match=r"NaN or \+inf.*rows \[0, 2\]"):
        _policy_posterior(score, 1.0, None)


def test_policy_posterior_retains_partial_negative_infinity_support():
    score = torch.tensor([[0.0, float("inf"), 1.0]])
    posterior = _policy_posterior(score, 1.0, None)
    expected = torch.softmax(torch.tensor([[0.0, float("-inf"), -1.0]]), dim=-1)
    assert torch.equal(posterior[:, 1], torch.zeros(1))
    assert torch.allclose(posterior, expected)


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


def test_logprob_control_does_not_double_count_base_prior():
    m = _model()
    ctx = torch.tensor([[1, 2, 3, 4, 5]])
    cand = torch.tensor([[[0], [1], [2], [3]]])
    base = torch.linspace(-1.0, 1.0, m.cfg.vocab_size).unsqueeze(0)
    pref = get_preference("flat")(m.prior_bank)
    menu_log_prior = torch.log_softmax(torch.gather(base, 1, cand.squeeze(-1)), dim=-1)
    with torch.no_grad():
        out = get_policy("logprob_control")(
            ctx, cand, pref, m, gamma=1.0, log_prior=menu_log_prior, base_logits=base)
    expected = torch.softmax(out.log_prob, dim=-1)
    double_counted = torch.softmax(2.0 * out.log_prob, dim=-1)
    assert torch.allclose(out.policy_posterior, expected, atol=1e-6)
    assert not torch.allclose(out.policy_posterior, double_counted, atol=1e-6)


def test_gates_and_horizon_guards_raise(monkeypatch):
    from vfe3.inference import policy as policy_module
    m = _model()
    ctx = torch.tensor([[1, 2, 3, 4, 5]])
    cand, base = _menu(m, ctx, Kp=4)
    pref = get_preference("flat")(m.prior_bank)

    def fail_rollout(*args, **kwargs):
        pytest.fail("invalid one-step policy reached rollout")

    monkeypatch.setattr(policy_module, "_rollout_predictive", fail_rollout)
    with pytest.raises(ValueError):                          # efe_one_step is H=1 only
        get_policy("efe_one_step")(ctx, cand, pref, m, horizon=2, base_logits=base)
    for mode in ("efe_one_step", "logprob_control"):
        cand2 = torch.randint(0, m.cfg.vocab_size, (1, 4, 2))
        with pytest.raises(ValueError, match=r"candidate length L=2 must equal horizon=1"):
            get_policy(mode)(ctx, cand2, pref, m, horizon=1, base_logits=base)
        empty_candidates = torch.empty((1, 4, 0), dtype=torch.long)
        with pytest.raises(ValueError, match=r"candidate length must be > 0, got L=0"):
            get_policy(mode)(ctx, empty_candidates, pref, m, horizon=1, base_logits=base)
        empty_context = torch.empty((1, 0), dtype=torch.long)
        with pytest.raises(ValueError, match=r"context must be nonempty, got N=0"):
            get_policy(mode)(empty_context, cand, pref, m, horizon=1, base_logits=base)
    # efe_rollout (H>1) is unlocked by the Phase-3a cache but REQUIRES it: this _model() (n_layers=2,
    # n_e_steps=2, e_phi_lr>0) is not cache-supported, so a correctly-shaped H=2 call still raises.
    cand2 = torch.randint(0, m.cfg.vocab_size, (1, 4, 2))
    with pytest.raises(NotImplementedError):
        get_policy("efe_rollout")(ctx, cand2, pref, m, horizon=2)
    with pytest.raises(RuntimeError):                        # sigma_mc fails closed absent gate identities
        get_ambiguity("sigma_mc")(
            torch.log_softmax(torch.randn(1, 4, 16), dim=-1),
            mu=torch.randn(1, 4, 8), sigma=torch.rand(1, 4, 8) + 0.5, model=m, num_samples=16)


def test_policy_context_validator_accepts_exact_boundary():
    context = torch.ones((1, 7), dtype=torch.long)
    assert _validate_policy_context(context, 1, 8) is None


def test_sigma_mc_direct_dispatch_without_identities_fails_closed():
    # PB-06 Task 4: the estimator body is implemented now, but a direct registry dispatch with NO
    # derived consumer-gate identity still fails closed (the four identities are supplied only after
    # VFEModel.generate verifies verify_sigma_consumer_gate). The message must name the validated flag
    # and state it does NOT unlock the arm by itself, and that the identities are REQUIRED.
    m = _model()
    with pytest.raises(RuntimeError) as e:
        get_ambiguity("sigma_mc")(
            torch.log_softmax(torch.randn(1, 4, 16), dim=-1),
            mu=torch.randn(1, 4, 8), sigma=torch.rand(1, 4, 8) + 0.5, model=m, num_samples=16)
    msg = str(e.value)
    assert "policy_sigma_ambiguity_validated" in msg
    assert "does NOT unlock" in msg and "REQUIRED" in msg


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


def test_policy_rejects_context_plus_candidate_over_limit(monkeypatch):
    m = _model(policy_mode="efe_one_step", policy_preference="flat", max_seq_len=8)
    context = torch.randint(0, m.cfg.vocab_size, (1, 8))

    def fail_forward_beliefs(*args, **kwargs):
        pytest.fail("overlength policy context reached base decode")

    monkeypatch.setattr(m, "forward_beliefs", fail_forward_beliefs)
    with pytest.raises(
        ValueError,
        match=r"context length N=8 plus candidate length L=1 exceeds max_seq_len=8",
    ):
        m.generate(context, max_new_tokens=1, greedy=True)


def test_policy_menu_rejects_nonfinite_base_logits(monkeypatch):
    from vfe3.inference import policy as policy_module
    m = _model(
        policy_mode="efe_one_step",
        policy_preference="flat",
        policy_top_k=4,
    )
    V = m.cfg.vocab_size
    context = torch.randint(0, V, (2, 3))
    injected = {"logits": torch.zeros(2, 1, V)}

    def fake_forward_beliefs(
        token_ids,
        *,
        return_logits=False,
        decode_last=False,
        **kwargs,
    ):
        assert return_logits and decode_last
        return None, injected["logits"]

    monkeypatch.setattr(m, "forward_beliefs", fake_forward_beliefs)

    def fake_get_policy(name):
        def fake_policy(context, candidates, preference, model, **kwargs):
            B, Kp = candidates.shape[:2]
            zeros = torch.zeros(B, Kp)
            posterior = torch.zeros(B, Kp)
            posterior[:, 0] = 1.0
            return PolicyScore(zeros, zeros, zeros, zeros, zeros, posterior)
        return fake_policy

    monkeypatch.setattr(policy_module, "get_policy", fake_get_policy)

    injected["logits"] = torch.zeros(2, 1, V)
    injected["logits"][0] = float("-inf")
    with pytest.raises(ValueError, match=r"policy base logits have no finite value in rows \[0\]"):
        m._policy_select(context, greedy=True)

    injected["logits"] = torch.zeros(2, 1, V)
    injected["logits"][0, 0, 0] = float("nan")
    injected["logits"][1, 0, 1] = float("inf")
    with pytest.raises(
        ValueError,
        match=r"policy base logits contain NaN or \+inf values in rows \[0, 1\]",
    ):
        m._policy_select(context, greedy=True)

    injected["logits"] = torch.zeros(2, 1, V)
    injected["logits"][0] = float("-inf")
    injected["logits"][0, 0, 0] = 0.0
    with pytest.raises(
        ValueError,
        match=r"policy menu logits contain non-finite retained values in rows \[0\]",
    ):
        m._policy_select(context, greedy=True)

    injected["logits"] = torch.full((2, 1, V), float("-inf"))
    injected["logits"][:, :, :4] = torch.arange(4, dtype=torch.float32)
    selected = m._policy_select(context, greedy=True)
    assert selected.shape == (2, 1)                        # filtered-out -inf remains valid


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
    assert torch.isneginf(task[8:]).all()                    # default support remains exact/hard
    finite_task = get_preference("task")(
        m.prior_bank, goal=3, beta_C=5.0, support=support, support_floor=-30.0)
    assert torch.isfinite(finite_task).all()
    assert torch.allclose(finite_task.exp().sum(), torch.tensor(1.0), atol=1e-6)
    assert float(finite_task.exp()[8:].sum()) < 1e-5         # explicit finite-floor preference
    for bad_floor in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="support_floor"):
            get_preference("task")(
                m.prior_bank, goal=3, beta_C=5.0, support=support, support_floor=bad_floor)
    # batched goal -> (B, V)
    task_b = get_preference("task")(m.prior_bank, goal=torch.tensor([1, 4]), beta_C=5.0)
    assert task_b.shape == (2, V)
    assert int(task_b[0].exp().argmax()) == 1 and int(task_b[1].exp().argmax()) == 4
    # held_out_predictive returns log p_data
    p_data = torch.softmax(torch.randn(V), dim=-1)
    hop = get_preference("held_out_predictive")(m.prior_bank, p_data=p_data)
    assert torch.allclose(hop, p_data.log(), atol=1e-5)


def test_rollout_predictive_state_full_path_carries_terminal_state():
    # PB-06: the state-carrying rollout returns the terminal belief mean/covariance alongside q_log and
    # the raw continuation log-prob, and the two-tensor wrapper stays byte-identical to (q_log, log_prob).
    from vfe3.contracts import PolicyRollout
    from vfe3.inference.policy import _rollout_predictive_state
    from vfe3.inference.belief_cache import cache_supported
    m = _model()                                            # n_layers=2, n_e_steps=2 -> NOT cache-supported
    assert not cache_supported(m.cfg)
    ctx = torch.tensor([[1, 2, 3, 4, 5]])
    cand, base = _menu(m, ctx, Kp=4)
    B, Kp, K = 1, 4, m.cfg.embed_dim
    with torch.no_grad():
        state = _rollout_predictive_state(ctx, cand, m, base_logits=base)
        q_log, log_prob = _rollout_predictive(ctx, cand, m, base_logits=base)
    assert isinstance(state, PolicyRollout)
    assert torch.equal(state.q_log, q_log) and torch.equal(state.log_prob, log_prob)
    assert state.mu.shape == (B, Kp, K)                     # (B, Kp, K) terminal mean
    assert state.sigma.shape == (B, Kp, K)                  # (B, Kp, K) diagonal terminal covariance
    assert torch.isfinite(state.mu).all() and (state.sigma > 0).all()


def test_rollout_predictive_state_full_covariance_shape():
    m = _model(n_heads=1, family="gaussian_full", use_prior_bank=True, decode_mode="full")
    from vfe3.inference.policy import _rollout_predictive_state
    ctx = torch.tensor([[1, 2, 3, 4, 5]])
    cand, base = _menu(m, ctx, Kp=3)
    B, Kp, K = 1, 3, m.cfg.embed_dim
    with torch.no_grad():
        state = _rollout_predictive_state(ctx, cand, m, base_logits=base)
    assert state.sigma.shape == (B, Kp, K, K)               # (B, Kp, K, K) full terminal covariance
    assert state.mu.shape == (B, Kp, K)


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


# ---------- (3) the gated sigma_mc ambiguity estimator (PB-06 Task 4) ----------
#
# These exercise the estimator BODY under an ARTIFICIAL / SYNTHETIC PASS: verify_sigma_consumer_gate is
# monkeypatched open so the estimator runs. This is control-flow / numerical plumbing only; it does NOT
# validate the empirical sigma arm (the shipped preregistry keeps the production identity FAIL).

# The four derived consumer-gate identities are stub values here; the real ones flow from
# VFEModel.generate after it verifies the live gate.
_SIGMA_IDS = dict(model_behavior_sha256="mb", spec_identity="sp",
                  code_identity_sha256="cd", measurement_context_sha256="ctx")


def _open_gate(monkeypatch):
    """Hold verify_sigma_consumer_gate open (synthetic PASS) so the estimator body runs."""
    from vfe3.inference import sigma_gate
    monkeypatch.setattr(sigma_gate, "verify_sigma_consumer_gate", lambda *a, **k: {"status": "PASS"})


def _sigma_mc_model(**kw):
    d = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
             n_e_steps=1, e_phi_lr=0.0, use_prior_bank=True, decode_mode="diagonal")
    d.update(kw)
    torch.manual_seed(0)
    return VFEModel(VFE3Config(**d))


def _explicit_diag_sigma_mc(m, mu, sigma):
    """Recompute the antithetic_shared_v1 marginal + expected conditional entropy explicitly (diagonal)."""
    S, half = 16, 8
    B, Kp, K = mu.shape
    gen = torch.Generator(device=mu.device).manual_seed(0)
    z = torch.randn(B, 1, half, K, generator=gen, device=mu.device, dtype=mu.dtype)
    mu_e = mu.unsqueeze(2)
    delta = sigma.clamp_min(m.cfg.eps).sqrt().unsqueeze(2) * z
    samples = torch.cat([mu_e + delta, mu_e - delta], dim=2)     # (B, Kp, S, K), positive-then-negative
    slp = torch.log_softmax(m.prior_bank.decode_point(samples), dim=-1)
    marg = torch.log_softmax(torch.logsumexp(slp, dim=2) - math.log(S), dim=-1)
    prob = slp.exp()
    ent = -torch.where(prob > 0, prob * slp, torch.zeros_like(prob)).sum(-1).mean(2)
    return marg, ent


def test_sigma_mc_artificial_pass_matches_explicit_antithetic_draws(monkeypatch):
    # Artificial-PASS estimator test: embed_dim=4, a fixed terminal Gaussian, explicit antithetic draws
    # compared for BOTH the predictive marginal AND the expected conditional entropy.
    _open_gate(monkeypatch)
    m = _sigma_mc_model()
    B, Kp, K, V = 2, 3, m.cfg.embed_dim, m.cfg.vocab_size
    torch.manual_seed(1)
    mu = torch.randn(B, Kp, K)
    sigma = torch.rand(B, Kp, K) + 0.5
    q_log = torch.log_softmax(m.prior_bank.decode_point(mu), dim=-1)     # unused by sigma_mc
    est = _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
    assert isinstance(est, AmbiguityEstimate)
    assert est.predictive_log_prob.shape == (B, Kp, V)
    assert est.expected_conditional_entropy.shape == (B, Kp)
    marg, ent = _explicit_diag_sigma_mc(m, mu, sigma)
    assert torch.allclose(est.predictive_log_prob, marg, atol=1e-6)
    assert torch.allclose(est.expected_conditional_entropy, ent, atol=1e-6)


def test_sigma_mc_zero_covariance_approaches_likelihood_entropy(monkeypatch):
    _open_gate(monkeypatch)
    m = _sigma_mc_model()
    B, Kp, K = 1, 2, m.cfg.embed_dim
    torch.manual_seed(2)
    mu = torch.randn(B, Kp, K)
    sigma = torch.zeros(B, Kp, K)                                # zero covariance -> eps-floor point decode
    q_log = torch.log_softmax(m.prior_bank.decode_point(mu), dim=-1)
    est = _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
    like = get_ambiguity("likelihood_entropy")(q_log)
    assert torch.allclose(est.expected_conditional_entropy,
                          like.expected_conditional_entropy, atol=1e-2)
    assert torch.allclose(est.predictive_log_prob, q_log, atol=1e-2)


def test_sigma_mc_nonzero_covariance_changes_the_estimator(monkeypatch):
    _open_gate(monkeypatch)
    m = _sigma_mc_model()
    B, Kp, K = 1, 2, m.cfg.embed_dim
    torch.manual_seed(3)
    mu = torch.randn(B, Kp, K)
    sigma = torch.full((B, Kp, K), 2.0)                          # substantial covariance
    q_log = torch.log_softmax(m.prior_bank.decode_point(mu), dim=-1)
    est = _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
    like = get_ambiguity("likelihood_entropy")(q_log)
    assert not torch.allclose(est.expected_conditional_entropy,
                              like.expected_conditional_entropy, atol=1e-3)
    assert not torch.allclose(est.predictive_log_prob, q_log, atol=1e-3)


def test_sigma_mc_candidate_permutation_permutes_both_tensors(monkeypatch):
    _open_gate(monkeypatch)
    m = _sigma_mc_model()
    B, Kp, K = 1, 4, m.cfg.embed_dim
    torch.manual_seed(4)
    mu = torch.randn(B, Kp, K)
    sigma = torch.rand(B, Kp, K) + 0.5
    q_log = torch.log_softmax(m.prior_bank.decode_point(mu), dim=-1)
    est = _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
    perm = torch.tensor([2, 0, 3, 1])
    est_p = _amb_sigma_mc(q_log[:, perm], mu=mu[:, perm], sigma=sigma[:, perm],
                          model=m, num_samples=16, **_SIGMA_IDS)
    # shared-across-candidate noise -> permuting candidates permutes both result tensors exactly
    assert torch.allclose(est_p.predictive_log_prob, est.predictive_log_prob[:, perm], atol=1e-6)
    assert torch.allclose(est_p.expected_conditional_entropy,
                          est.expected_conditional_entropy[:, perm], atol=1e-6)


def test_sigma_mc_repeated_calls_deterministic_and_global_rng_unchanged(monkeypatch):
    _open_gate(monkeypatch)
    m = _sigma_mc_model()
    B, Kp, K = 1, 2, m.cfg.embed_dim
    torch.manual_seed(5)
    mu = torch.randn(B, Kp, K)
    sigma = torch.rand(B, Kp, K) + 0.5
    q_log = torch.log_softmax(m.prior_bank.decode_point(mu), dim=-1)
    torch.manual_seed(999)
    before = torch.get_rng_state()
    e1 = _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
    e2 = _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
    after = torch.get_rng_state()
    assert torch.equal(before, after)                            # local generator: global RNG untouched
    assert torch.equal(e1.predictive_log_prob, e2.predictive_log_prob)
    assert torch.equal(e1.expected_conditional_entropy, e2.expected_conditional_entropy)


def test_sigma_mc_mi_bridge_is_nonnegative(monkeypatch):
    _open_gate(monkeypatch)
    m = _sigma_mc_model()
    B, Kp, K = 2, 3, m.cfg.embed_dim
    torch.manual_seed(7)
    mu = torch.randn(B, Kp, K)
    sigma = torch.rand(B, Kp, K) + 1.0
    q_log = torch.log_softmax(m.prior_bank.decode_point(mu), dim=-1)
    est = _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
    p = est.predictive_log_prob.exp()
    pred_ent = -torch.where(p > 0, p * est.predictive_log_prob, torch.zeros_like(p)).sum(-1)
    mi = pred_ent - est.expected_conditional_entropy               # Jensen: H[E_s p] >= E_s H[p]
    assert bool((mi >= -1e-6).all())


def test_sigma_mc_risk_matches_kl_of_the_sampled_marginal(monkeypatch):
    _open_gate(monkeypatch)
    m = _sigma_mc_model()
    B, Kp, K = 1, 2, m.cfg.embed_dim
    torch.manual_seed(8)
    mu = torch.randn(B, Kp, K)
    sigma = torch.rand(B, Kp, K) + 0.5
    q_log = torch.log_softmax(m.prior_bank.decode_point(mu), dim=-1)
    est = _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
    pref = get_preference("flat")(m.prior_bank)
    risk, _ = _efe_terms(est.predictive_log_prob, pref)           # risk uses the MC marginal, not q_log
    p = est.predictive_log_prob.exp()
    kl = (torch.where(p > 0, p * est.predictive_log_prob, torch.zeros_like(p))
          - p * pref.view(1, 1, -1)).sum(-1)
    assert torch.allclose(risk, kl, atol=1e-5)


def test_sigma_mc_marginal_normalized_and_inf_tails_stay_nan_free(monkeypatch):
    # Extreme logits with exact -inf tails: the marginal must still normalize to 1 and the entropy must
    # stay NaN-free (the where(prob>0, ...) guard blocks 0 * -inf).
    _open_gate(monkeypatch)
    m = _sigma_mc_model()
    B, Kp, K, V = 1, 2, m.cfg.embed_dim, m.cfg.vocab_size
    torch.manual_seed(9)
    mu = torch.randn(B, Kp, K)
    sigma = torch.rand(B, Kp, K) + 0.5

    def extreme_decode(mu_s):
        logits = torch.full((*mu_s.shape[:-1], V), float("-inf"))
        logits[..., 0] = 60.0 * mu_s.sum(-1)                     # two finite classes; the rest exact -inf
        logits[..., 1] = -60.0 * mu_s.sum(-1)
        return logits

    monkeypatch.setattr(m.prior_bank, "decode_point", extreme_decode)
    q_log = torch.log_softmax(torch.randn(B, Kp, V), dim=-1)
    est = _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
    assert torch.allclose(est.predictive_log_prob.exp().sum(-1), torch.ones(B, Kp), atol=1e-5)
    assert not torch.isnan(est.expected_conditional_entropy).any()
    assert torch.isfinite(est.expected_conditional_entropy).all()


def test_sigma_mc_full_family_matches_explicit_cholesky_draws(monkeypatch):
    # Full family: the sampler reparameterizes through safe_cholesky (never raw cholesky); pin the
    # marginal + entropy against an explicit safe-Cholesky recomputation, and receive the FULL decode.
    from vfe3.numerics import safe_cholesky
    _open_gate(monkeypatch)
    m = _sigma_mc_model(n_heads=1, family="gaussian_full", decode_mode="full")
    B, Kp, K = 1, 2, m.cfg.embed_dim
    torch.manual_seed(10)
    mu = torch.randn(B, Kp, K)
    A = torch.randn(B, Kp, K, K)
    sigma = A @ A.transpose(-1, -2) + torch.eye(K)               # SPD full covariance
    q_log = torch.log_softmax(m.prior_bank.decode_point(mu), dim=-1)
    est = _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
    S, half = 16, 8
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(B, 1, half, K, generator=gen)
    L, _ok = safe_cholesky(sigma, eps=m.cfg.eps, rounds=5)
    delta = (L.unsqueeze(2) @ z.unsqueeze(-1)).squeeze(-1)
    mu_e = mu.unsqueeze(2)
    samples = torch.cat([mu_e + delta, mu_e - delta], dim=2)
    slp = torch.log_softmax(m.prior_bank.decode_point(samples), dim=-1)
    marg = torch.log_softmax(torch.logsumexp(slp, dim=2) - math.log(S), dim=-1)
    prob = slp.exp()
    ent = -torch.where(prob > 0, prob * slp, torch.zeros_like(prob)).sum(-1).mean(2)
    assert torch.allclose(est.predictive_log_prob, marg, atol=1e-5)
    assert torch.allclose(est.expected_conditional_entropy, ent, atol=1e-5)


def test_sigma_mc_full_family_zero_and_near_singular_stay_finite(monkeypatch):
    # The full zero / near-singular covariance case must remain finite and converge to likelihood_entropy
    # through the repository's safe-Cholesky jitter policy.
    _open_gate(monkeypatch)
    m = _sigma_mc_model(n_heads=1, family="gaussian_full", decode_mode="full")
    B, Kp, K = 1, 2, m.cfg.embed_dim
    torch.manual_seed(11)
    mu = torch.randn(B, Kp, K)
    q_log = torch.log_softmax(m.prior_bank.decode_point(mu), dim=-1)
    like = get_ambiguity("likelihood_entropy")(q_log)
    eye = torch.eye(K).expand(B, Kp, K, K)
    for sigma in (torch.zeros(B, Kp, K, K), 1e-10 * eye.clone()):
        est = _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
        assert not torch.isnan(est.predictive_log_prob).any()
        assert torch.isfinite(est.expected_conditional_entropy).all()
        assert torch.allclose(est.expected_conditional_entropy,
                              like.expected_conditional_entropy, atol=1e-2)


def test_sigma_mc_rejects_sample_counts_other_than_16(monkeypatch):
    _open_gate(monkeypatch)
    m = _sigma_mc_model()
    B, Kp, K = 1, 2, m.cfg.embed_dim
    mu = torch.randn(B, Kp, K)
    sigma = torch.rand(B, Kp, K) + 0.5
    q_log = torch.log_softmax(m.prior_bank.decode_point(mu), dim=-1)
    for bad in (8, 15, 32):
        with pytest.raises(ValueError, match="num_samples=16"):
            _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=bad, **_SIGMA_IDS)
    # the sealed sampler itself rejects a non-16 S even when called directly
    with pytest.raises(ValueError, match="num_samples=16"):
        _antithetic_shared_state_samples(mu, sigma, m.cfg.eps, num_samples=8)


def test_sigma_mc_reverifies_consumer_gate_defense_in_depth(monkeypatch):
    # The estimator re-verifies the consumer gate (defense in depth), rereading the artifact per sigma
    # dispatch: an open gate lets it run, a gate that now rejects makes it fail closed.
    from vfe3.inference import sigma_gate
    m = _sigma_mc_model()
    B, Kp, K = 1, 2, m.cfg.embed_dim
    mu = torch.randn(B, Kp, K)
    sigma = torch.rand(B, Kp, K) + 0.5
    q_log = torch.log_softmax(m.prior_bank.decode_point(mu), dim=-1)
    calls = {"n": 0}

    def _spy(path, **k):
        calls["n"] += 1
        return {"status": "PASS"}

    monkeypatch.setattr(sigma_gate, "verify_sigma_consumer_gate", _spy)
    _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
    assert calls["n"] == 1                                        # re-read once per sigma dispatch

    def _reject(path, **k):
        raise ValueError("sigma-gate governing identity is not registered as PASS")

    monkeypatch.setattr(sigma_gate, "verify_sigma_consumer_gate", _reject)
    with pytest.raises(ValueError, match="not registered as PASS"):
        _amb_sigma_mc(q_log, mu=mu, sigma=sigma, model=m, num_samples=16, **_SIGMA_IDS)
