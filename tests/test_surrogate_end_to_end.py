r"""Surrogate (include_attention_entropy=False) exercised END-TO-END through VFEModel (PL19).

The canonical free energy carries the attention-entropy term tau*sum_ij beta_ij log(beta_ij/pi_ij);
the "entropy-suppressed surrogate" (the convention standard transformers train under) drops it. The
existing coverage tests the -tau^-1 Cov gradient gap with LOCAL closures only; these tests add the
end-to-end legs the model never covered:

  1. With the flag False the hand kernel is skipped and the autograd ORACLE runs -- assert the model
     forwards and backpropagates to the prior tables through that branch.
  2. The flag is LIVE in the E-step descent (not just diagnostics): matched-weight canonical vs
     surrogate models produce different logits, because the E-step minimizes a different objective.
  3. The free-energy GATE is exact: at the model's converged beliefs, `total` includes the entropy
     term iff the flag is set, and the canonical total exceeds the surrogate total by exactly
     lambda_beta * attention_entropy (through metrics.free_energy_terms, the function diagnostics uses).
"""

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=2, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0)
    base.update(kw)
    return VFE3Config(**base)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 12, (2, 8), generator=g)
    targets = torch.randint(0, 12, (2, 8), generator=g)
    return tokens, targets


def test_surrogate_forward_and_backward_through_oracle_branch():
    # include_attention_entropy=False -> the closed-form hand kernel is skipped and the autograd
    # oracle runs; with oracle_unroll_grad=True it returns a DIFFERENTIABLE (unrolled) tangent, so
    # the surrogate must still forward + backprop to the prior tables through VFEModel. (With the
    # default oracle_unroll_grad=False the oracle tangent is detached -- config warns -- and the loss
    # is not differentiable; that is the documented non-training-path, not what the surrogate trains under.)
    cfg = _cfg(include_attention_entropy=False, oracle_unroll_grad=True)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens, targets = _batch()
    logits = model(tokens)
    _, loss, ce = model(tokens, targets)                    # fused training CE may omit logits
    assert torch.isfinite(logits).all() and torch.isfinite(loss) and torch.isfinite(ce)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "surrogate backward reached no parameter"
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0   # the M-step actually has signal


def test_surrogate_changes_the_e_step_descent_vs_canonical():
    # Matched weights (same seed) -> identical tables; the only difference is the objective the
    # E-step descends, so the converged-belief logits must differ when the entropy term is dropped.
    torch.manual_seed(0)
    m_true = VFEModel(_cfg(include_attention_entropy=True))
    torch.manual_seed(0)
    m_false = VFEModel(_cfg(include_attention_entropy=False))
    # sanity: identical initial weights
    for a, b in zip(m_true.parameters(), m_false.parameters()):
        assert torch.equal(a, b)
    tokens, _ = _batch()
    with torch.no_grad():
        lt = m_true(tokens)[0]
        lf = m_false(tokens)[0]
    assert not torch.allclose(lt, lf), "the surrogate descended the same beliefs as canonical"


def test_free_energy_gate_differs_by_exactly_the_entropy_term():
    # At the model's converged beliefs, diagnostics' `total` includes the entropy term iff the flag
    # is set. The canonical total exceeds the surrogate total (entropy excluded) by exactly
    # lambda_beta * attention_entropy, evaluated at the SAME beliefs.
    cfg = _cfg(include_attention_entropy=True)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens, _ = _batch()
    d = model.diagnostics(tokens)
    lb = cfg.lambda_beta                                     # 1.0 on the pure path
    total_with_entropy = d["total"]
    total_without_entropy = d["self_coupling"] + lb * d["belief_coupling"]
    assert abs((total_with_entropy - total_without_entropy) - lb * d["attention_entropy"]) < 1e-4

    # and the surrogate model's diagnostics `total` omits the entropy term entirely
    cfg_s = _cfg(include_attention_entropy=False)
    torch.manual_seed(0)
    model_s = VFEModel(cfg_s)
    ds = model_s.diagnostics(tokens)
    assert abs(ds["total"] - (ds["self_coupling"] + lb * ds["belief_coupling"])) < 1e-4
