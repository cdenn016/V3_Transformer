r"""Audit V6 fix: ``metrics.free_energy_terms`` must agree with ``free_energy.free_energy``.

The pre-fix decomposition dropped the alpha-regularizer R(alpha) from the self-coupling term
and unconditionally added the attention-entropy contribution to ``total``, so the logged/CSV
total disagreed with the scalar the E-step actually minimizes under a state-dependent alpha or
the entropy-suppressed surrogate. These tests pin the corrected, default-preserving behavior.
"""

import torch

from vfe3.free_energy import attention_weights, free_energy
from vfe3.metrics import free_energy_terms


def _inputs():
    r"""Small synthetic (self_div, energy, alpha, alpha_reg, log_prior) in float32."""
    torch.manual_seed(0)
    N = 4
    self_div = torch.rand(N, dtype=torch.float32) + 0.1          # (N,) D(q_i||p_i) > 0
    energy = torch.rand(N, N, dtype=torch.float32)               # (N, N) E_ij
    alpha = torch.rand(N, dtype=torch.float32) + 0.5             # (N,) self-coupling > 0
    alpha_reg = torch.rand(N, dtype=torch.float32) + 0.2         # (N,) R(alpha_i) nonzero
    log_prior = torch.randn(N, N, dtype=torch.float32)           # (N, N) attention log-prior
    return self_div, energy, alpha, alpha_reg, log_prior


def test_defaults_byte_identical_to_pre_fix():
    r"""(1) alpha_reg=None, include_attention_entropy=True reproduces the inline free-energy-terms
    recomputation exactly. The entropy log-prior uses the exact ``log_softmax`` (audit m8; formerly
    ``log(softmax(.).clamp)``, which floored a finite deep-tail prior at ~-27.6 nats)."""
    self_div, energy, alpha, _, log_prior = _inputs()
    tau, lambda_beta, eps = 1.3, 0.9, 1e-12
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)

    out = free_energy_terms(
        self_div, energy, beta, alpha,
        tau=tau, lambda_beta=lambda_beta, log_prior=log_prior,
    )

    # Recompute the EXACT expressions inline (identical ops -> identical float32->float).
    exp_self = float((alpha * self_div).sum())
    exp_belief = float((beta * energy).sum())
    from vfe3.free_energy import _broadcast_tau
    log_pi = torch.log_softmax(log_prior, dim=-1)   # m8: exact log-prior (was torch.log(softmax(.).clamp))
    _tau_e = _broadcast_tau(tau, energy)            # mirror metrics' exact op order (tau inside the sum)
    exp_entropy = float((_tau_e * (beta * (torch.log(beta.clamp(min=eps)) - log_pi))).sum())
    exp_total = exp_self + float(lambda_beta) * (exp_belief + exp_entropy)

    assert out["self_coupling"] == exp_self
    assert out["belief_coupling"] == exp_belief
    assert out["attention_entropy"] == exp_entropy
    assert out["total"] == exp_total
    assert set(out) == {"self_coupling", "belief_coupling", "attention_entropy", "total"}


def test_alpha_reg_adds_into_self_and_total():
    r"""(2) A nonzero alpha_reg increases self_coupling and total by alpha_reg.sum() (and strictly)."""
    self_div, energy, alpha, alpha_reg, log_prior = _inputs()
    tau, lambda_beta = 1.0, 1.0
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)

    base = free_energy_terms(self_div, energy, beta, alpha,
                             tau=tau, lambda_beta=lambda_beta, log_prior=log_prior)
    reg = free_energy_terms(self_div, energy, beta, alpha,
                            tau=tau, lambda_beta=lambda_beta, log_prior=log_prior,
                            alpha_reg=alpha_reg)

    delta = float(alpha_reg.sum())
    # belief_coupling and attention_entropy are untouched by alpha_reg.
    assert reg["belief_coupling"] == base["belief_coupling"]
    assert reg["attention_entropy"] == base["attention_entropy"]
    # self_coupling and total both rise by exactly alpha_reg.sum() (within float32 rounding).
    assert reg["self_coupling"] > base["self_coupling"]
    assert reg["total"] > base["total"]
    torch.testing.assert_close(
        torch.tensor(reg["self_coupling"] - base["self_coupling"]),
        torch.tensor(delta), rtol=1e-5, atol=1e-5,
    )
    torch.testing.assert_close(
        torch.tensor(reg["total"] - base["total"]),
        torch.tensor(delta), rtol=1e-5, atol=1e-5,
    )


def test_total_matches_free_energy_scalar():
    r"""(3) The reported ``total`` equals ``free_energy(...)`` for matching inputs, both entropy gates."""
    self_div, energy, alpha, alpha_reg, log_prior = _inputs()
    tau, lambda_beta = 1.1, 0.7
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)

    for include in (True, False):
        out = free_energy_terms(
            self_div, energy, beta, alpha,
            tau=tau, lambda_beta=lambda_beta,
            include_attention_entropy=include,
            log_prior=log_prior, alpha_reg=alpha_reg,
        )
        f = free_energy(
            self_div, energy, alpha,
            tau=tau, lambda_beta=lambda_beta,
            include_attention_entropy=include,
            log_prior=log_prior, alpha_reg=alpha_reg,
            log_likelihood=None,
        )
        torch.testing.assert_close(
            torch.tensor(out["total"]), torch.tensor(float(f)),
            rtol=1e-5, atol=1e-5,
        )
        # The diagnostic entropy value is reported regardless of the gate.
        assert "attention_entropy" in out
