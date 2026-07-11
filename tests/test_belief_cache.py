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
    d = dict(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=16)
    d.update(kw)
    torch.manual_seed(0)
    return VFEModel(VFE3Config(**d))


@pytest.mark.parametrize("use_prior_bank", [False, True])   # linear (ring) and KL-to-prior decode
@pytest.mark.parametrize("L", [1, 3])                        # H = 1 (one-step) and H > 1 (rollout)
@pytest.mark.parametrize("n_heads", [1, 2])                  # single-block and equal-block (factored) groups
def test_cached_matches_full_rollout(use_prior_bank, L, n_heads):
    m = _model(use_prior_bank=use_prior_bank, n_heads=n_heads)
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


def test_efe_rollout_rejects_context_plus_horizon_over_limit():
    from vfe3.inference.policy import get_policy, get_preference
    m = _model(max_seq_len=8)
    V, H = m.cfg.vocab_size, 2
    ctx = torch.randint(0, V, (1, 7))
    cand = torch.randint(0, V, (1, 3, H))
    pref = get_preference("flat")(m.prior_bank)
    with torch.no_grad():
        with pytest.raises(
            ValueError,
            match=r"context length N=7 plus candidate length L=2 exceeds max_seq_len=8",
        ):
            get_policy("efe_rollout")(ctx, cand, pref, m, horizon=H)
