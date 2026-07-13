r"""Golden tests for the EFE belief-prefix cache (vfe3/inference/belief_cache.py, Phase 3a).

The load-bearing test is GOLDEN EQUIVALENCE: on the supported config the cache-accelerated
``rollout_predictive_cached`` (which computes only the appended positions' E-step rows against a
shared context) must reproduce the full per-candidate recompute ``policy._rollout_predictive`` to
within float tolerance, for both decode paths (linear and KL-to-prior), single and multi-token
appends (H = 1 and H > 1), and single- and multi-head groups. Equality is to tolerance, not bytes:
the partial recompute reduces over a different key-axis length and float addition is non-associative
(the cacheability audit wf_a12bc02f-988 flagged this explicitly). The second test pins the
``cache_supported`` guard so an unsupported config falls back to the full recompute rather than
silently taking a wrong fast path.
"""
from types import SimpleNamespace

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.inference import belief_cache as belief_cache_module
from vfe3.inference.belief_cache import cache_supported, rollout_predictive_cached
from vfe3.inference.policy import _rollout_predictive
from vfe3.model.model import VFEModel


def _supported_ns(**over):
    """A minimal stand-in carrying exactly the fields ``cache_supported`` reads, in the supported
    regime; tests the predicate without tripping VFE3Config's cross-field validation (e.g. s_e_step
    requires prior_source='model_channel')."""
    base = dict(
        n_layers=1, n_e_steps=1, gradient_mode="filtering", family="gaussian_diagonal",
        divergence_family="renyi", renyi_order=1.0, include_attention_entropy=True,
        transport_mode="flat", e_phi_lr=0.0, beta_attention_prior="causal", s_e_step=False,
        precision_weighted_attention=False, pos_rotation="none", use_head_mixer=False,
        use_cg_coupling=False, e_step_mu_precond="fisher", e_mu_q_trust=None,
        phi_reflection="off",
        # M3 (audit 2026-07-06): result-changing toggles the cache does not replicate, at their
        # cache-eligible defaults so _supported_ns() stays supported.
        lambda_twohop=0.0, e_step_update="gradient", skip_belief_sigma_update=False,
        query_adaptive_tau=False, gamma_as_beta_prior=False, learnable_kappa_beta=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _model(**kw):
    # Defaults are the cache-supported regime: n_layers=1, n_e_steps=1, e_phi_lr=0, flat transport,
    # causal belief prior, filtering, gaussian_diagonal + KL. Only dims/seed are pinned here.
    d = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=16)
    d.update(kw)
    torch.manual_seed(0)
    return VFEModel(VFE3Config(**d))


@pytest.mark.parametrize("use_prior_bank", [False, True])   # linear (ring) and KL-to-prior decode
@pytest.mark.parametrize("L", [1, 3])                        # H = 1 (one-step) and H > 1 (rollout)
@pytest.mark.parametrize("n_heads", [1, 2])                  # single-block and equal-block (factored) groups
@pytest.mark.parametrize("attention_prior", ["causal", "causal_alibi", "causal_windowed"])
def test_cached_matches_full_rollout(use_prior_bank, L, n_heads, attention_prior):
    prior_kwargs = ({"attention_window": 2} if attention_prior == "causal_windowed" else {})
    m = _model(use_prior_bank=use_prior_bank, n_heads=n_heads,
               beta_attention_prior=attention_prior, **prior_kwargs)
    assert cache_supported(m.cfg)                            # the fast path must actually engage here

    B, N, Kp, V = 2, 5, 4, m.cfg.vocab_size
    torch.manual_seed(1)
    context = torch.randint(0, V, (B, N))
    candidates = torch.randint(0, V, (B, Kp, L))

    with torch.no_grad():
        base_logits = m.forward(context)[:, -1, :]
        q_full, lp_full = _rollout_predictive(context, candidates, m, base_logits=base_logits)
        q_cache, lp_cache = rollout_predictive_cached(context, candidates, m, base_logits=base_logits)

    # The raw continuation log-prob is computed identically on both paths -> exact.
    assert torch.allclose(lp_cache, lp_full, atol=1e-6)
    # The predicted-outcome distribution is reproduced to float tolerance (partial vs full recompute).
    assert torch.allclose(q_cache, q_full, atol=1e-5, rtol=1e-4), \
        f"max |dq|={float((q_cache - q_full).abs().max()):.2e}"


def test_cached_matches_full_with_norm_keyed_matrix_exp(monkeypatch):
    threshold = 1e-8
    m = _model(n_heads=1, exp_fp64_mode="norm", exp_fp64_norm_threshold=threshold)
    assert cache_supported(m.cfg)

    B, N, Kp, L, V = 1, 4, 3, 2, m.cfg.vocab_size
    context = torch.tensor([[0, 1, 2, 3]])
    candidates = torch.tensor([[[4, 5], [6, 7], [8, 9]]])

    with torch.no_grad():
        base_logits = m.forward(context)[:, -1, :]
        ctx_exp = context.unsqueeze(1).expand(B, Kp, N)
        ext = torch.cat([ctx_exp, candidates], dim=2).reshape(B * Kp, N + L)
        _belief, logits = m.rollout_beliefs(ext, return_logits=True)
        q_full = torch.log_softmax(logits[:, -1, :], dim=-1).reshape(B, Kp, V)

    calls = []
    original_compute = belief_cache_module.compute_transport_operators

    def tracked_compute(phi, group, **kwargs):
        calls.append((
            kwargs.get("exp_fp64_mode", "dim"),
            kwargs.get("exp_fp64_norm_threshold", 5.0),
        ))
        return original_compute(phi, group, **kwargs)

    monkeypatch.setattr(belief_cache_module, "compute_transport_operators", tracked_compute)
    with torch.no_grad():
        q_cache, lp_cache = rollout_predictive_cached(
            context, candidates, m, base_logits=base_logits)

    lp_full = torch.gather(torch.log_softmax(base_logits, dim=-1), 1, candidates[:, :, 0])
    assert calls == [("norm", threshold), ("norm", threshold)]
    assert torch.allclose(lp_cache, lp_full, atol=1e-6)
    assert torch.allclose(q_cache, q_full, atol=1e-5, rtol=1e-4), \
        f"max |dq|={float((q_cache - q_full).abs().max()):.2e}"


def test_cache_supported_guard():
    assert cache_supported(_supported_ns())                 # the supported regime
    assert cache_supported(VFE3Config())                    # and the REAL default config is that regime
    # Each toggle below independently leaves the verified fast path and must force the fallback.
    for kw in (
        dict(n_layers=2),
        dict(n_e_steps=2),
        dict(e_phi_lr=0.02),
        dict(transport_mode="regime_ii"),
        dict(gradient_mode="smoothing"),
        dict(include_attention_entropy=False),
        dict(beta_attention_prior="uniform"),
        dict(s_e_step=True),
        dict(precision_weighted_attention=True),
        dict(pos_rotation="rope"),
        dict(use_head_mixer=True),
        dict(use_cg_coupling=True),
        dict(e_step_mu_precond="raw"),
        dict(e_mu_q_trust=2.0),
        dict(family="laplace_diagonal"),
        dict(divergence_family="alpha"),
        dict(renyi_order=0.5),
    ):
        assert not cache_supported(_supported_ns(**kw)), kw


def test_cache_supported_rejects_phi_reflection():
    assert cache_supported(_supported_ns(phi_reflection="off"))
    assert not cache_supported(_supported_ns(phi_reflection="init_seed"))
    assert not cache_supported(_supported_ns(phi_reflection="metropolis"))


def test_cache_supported_gates_result_changing_toggles():
    # M3 (audit 2026-07-06): six result-changing toggles the cached kernel does NOT replicate must
    # disable the fast path, else rollout_predictive_cached silently diverges from the full recompute
    # it is golden-pinned to equal (two-hop coupling, the exact-minimizer E-step, the sigma-freeze,
    # query-adaptive tau, the gamma prior fold, and the learned per-block kappa).
    assert cache_supported(_supported_ns())                       # baseline: supported
    for kw in (
        dict(lambda_twohop=0.5),
        dict(e_step_update="mm_exact"),
        dict(skip_belief_sigma_update=True),
        dict(query_adaptive_tau=True),
        dict(gamma_as_beta_prior=True),
        dict(learnable_kappa_beta=True),
    ):
        assert not cache_supported(_supported_ns(**kw)), kw


# ---------- efe_rollout (H>1), unlocked by the cache ----------

def test_efe_rollout_unlocked_on_supported_config():
    from vfe3.inference.policy import PolicyScore, _efe_terms, get_policy, get_preference
    m = _model()                                            # cache-supported (defaults)
    B, N, Kp, H, V = 2, 5, 3, 3, m.cfg.vocab_size
    torch.manual_seed(2)
    ctx = torch.randint(0, V, (B, N))
    cand = torch.randint(0, V, (B, Kp, H))                  # Kp H-action policy sequences
    pref = get_preference("flat")(m.prior_bank)
    with torch.no_grad():
        out = get_policy("efe_rollout")(ctx, cand, pref, m, gamma=1.0, horizon=H)
    assert isinstance(out, PolicyScore)
    for t in (out.score, out.risk, out.ambiguity, out.policy_posterior):
        assert t.shape == (B, Kp)
    assert torch.allclose(out.policy_posterior.sum(-1), torch.ones(B), atol=1e-6)
    # audit F3 (2026-07-01): efe_rollout is a TERMINAL-OUTCOME scorer, NOT a per-step horizon sum.
    # Independently roll each H-action candidate to convergence and score q(o|pi_H) read from the
    # LAST appended position only; the scorer's risk must match that terminal-predictive KL (to the
    # cache-vs-full float tolerance pinned by test_cached_matches_full_rollout above).
    ctx_exp = ctx.unsqueeze(1).expand(B, Kp, N)              # (B, Kp, N)
    ext = torch.cat([ctx_exp, cand], dim=2).reshape(B * Kp, N + H)
    with torch.no_grad():
        _belief, logits = m.rollout_beliefs(ext, return_logits=True)  # (B*Kp, N+H, V)
    q_log_terminal = torch.log_softmax(logits[:, -1, :], dim=-1).reshape(B, Kp, -1)
    risk_terminal, _ = _efe_terms(q_log_terminal, pref)
    assert torch.allclose(out.risk, risk_terminal, atol=1e-5, rtol=1e-4), \
        f"max |drisk|={float((out.risk - risk_terminal).abs().max()):.2e}"


def test_efe_rollout_guards():
    from vfe3.inference.policy import get_policy, get_preference
    m = _model()
    V = m.cfg.vocab_size
    ctx = torch.randint(0, V, (1, 4))
    pref = get_preference("flat")(m.prior_bank)
    cand3 = torch.randint(0, V, (1, 2, 3))
    with torch.no_grad():
        with pytest.raises(ValueError):                    # horizon must be > 1
            get_policy("efe_rollout")(ctx, cand3, pref, m, horizon=1)
        with pytest.raises(ValueError):                    # horizon must equal candidate length
            get_policy("efe_rollout")(ctx, cand3, pref, m, horizon=2)
        m2 = _model(n_e_steps=2)                            # NOT cache-supported -> must raise, not fall back
        cand2 = torch.randint(0, V, (1, 2, 2))
        with pytest.raises(NotImplementedError):
            get_policy("efe_rollout")(ctx, cand2, pref, m2, horizon=2)


def test_efe_rollout_reflection_fails_closed():
    from vfe3.inference.policy import get_policy, get_preference

    m = _model(phi_reflection="init_seed")
    H = 2
    context = torch.tensor([[0, 1, 2, 3]])
    candidates = torch.tensor([[[4, 5], [6, 7]]])
    preference = get_preference("flat")(m.prior_bank)

    with torch.no_grad():
        with pytest.raises(NotImplementedError, match="requires the belief-prefix cache"):
            get_policy("efe_rollout")(
                context, candidates, preference, m, horizon=H)


def test_efe_rollout_rejects_context_plus_horizon_over_limit():
    from vfe3.inference.policy import get_policy, get_preference
    m = _model(max_seq_len=8)
    V, H = m.cfg.vocab_size, 2
    exact_ctx = torch.randint(0, V, (1, 6))
    ctx = torch.randint(0, V, (1, 7))
    cand = torch.randint(0, V, (1, 3, H))
    pref = get_preference("flat")(m.prior_bank)
    with torch.no_grad():
        exact = get_policy("efe_rollout")(exact_ctx, cand, pref, m, horizon=H)
        assert exact.score.shape == (1, 3)                 # N + H == max_seq_len remains valid
        with pytest.raises(
            ValueError,
            match=r"context length N=7 plus candidate length L=2 exceeds max_seq_len=8",
        ):
            get_policy("efe_rollout")(ctx, cand, pref, m, horizon=H)


@pytest.mark.parametrize("use_prior_bank", [False, True])
@pytest.mark.parametrize("L", [1, 3])
def test_rollout_predictive_state_cached_carries_terminal_state(use_prior_bank, L):
    # PB-06: the cached state-carrying rollout returns the terminal belief mean/covariance (post
    # block_norm/final_norm) at the last appended position, matches the full state path, and keeps the
    # two-tensor wrapper byte-identical to (q_log, log_prob).
    from vfe3.contracts import PolicyRollout
    from vfe3.inference.belief_cache import rollout_predictive_state_cached
    from vfe3.inference.policy import _rollout_predictive_state
    m = _model(use_prior_bank=use_prior_bank, n_heads=1)
    assert cache_supported(m.cfg)
    B, N, Kp, V, K = 2, 5, 4, m.cfg.vocab_size, m.cfg.embed_dim
    torch.manual_seed(3)
    context = torch.randint(0, V, (B, N))
    candidates = torch.randint(0, V, (B, Kp, L))
    with torch.no_grad():
        base_logits = m.forward(context)[:, -1, :]
        state = rollout_predictive_state_cached(context, candidates, m, base_logits=base_logits)
        q_cache, lp_cache = rollout_predictive_cached(context, candidates, m, base_logits=base_logits)
        full = _rollout_predictive_state(context, candidates, m, base_logits=base_logits)
    assert isinstance(state, PolicyRollout)
    assert torch.equal(state.q_log, q_cache) and torch.equal(state.log_prob, lp_cache)
    assert state.mu.shape == (B, Kp, K) and state.sigma.shape == (B, Kp, K)
    # _rollout_predictive_state routes through the cache on a supported config, so it IS this state.
    assert torch.equal(full.mu, state.mu) and torch.equal(full.sigma, state.sigma)
    assert (state.sigma > 0).all()


def test_noncache_policy_rollout_decodes_only_terminal_position(monkeypatch):
    m = _model(n_e_steps=2)                                # unsupported cache -> full rollout path
    B, N, Kp, L, V = 2, 5, 3, 2, m.cfg.vocab_size
    context = torch.randint(0, V, (B, N))
    candidates = torch.randint(0, V, (B, Kp, L))
    base_logits = m.forward(context)[:, -1, :]
    original_rollout_beliefs = m.rollout_beliefs
    calls = []

    def tracked_rollout_beliefs(
        token_ids,
        *,
        return_logits=True,
        decode_last=False,
    ):
        belief, logits = original_rollout_beliefs(
            token_ids, return_logits=return_logits, decode_last=decode_last)
        calls.append((return_logits, decode_last, tuple(logits.shape)))
        return belief, logits

    monkeypatch.setattr(m, "rollout_beliefs", tracked_rollout_beliefs)
    with torch.no_grad():
        q_log, log_prob = _rollout_predictive(
            context, candidates, m, base_logits=base_logits)

    assert q_log.shape == (B, Kp, V)
    assert log_prob.shape == (B, Kp)
    assert calls == [(True, True, (B * Kp, 1, V))]
