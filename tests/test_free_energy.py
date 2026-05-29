import math

import torch

from vfe3.free_energy import (
    attention_weights,
    effective_temperature,
    log_partition,
    reduced_free_energy,
)

# A concrete non-uniform setup reused across tests.
_E   = torch.tensor([1.0, 2.0, 0.5])               # distinct per-key energies
_PI  = torch.tensor([0.5, 0.3, 0.2])               # normalized non-uniform prior
_B   = torch.log(_PI)                              # log-prior bias
_TAU = 2.0


def test_temperature_is_kappa_sqrt_k():
    assert math.isclose(effective_temperature(1.5, 16), 1.5 * 4.0, rel_tol=1e-6)


def test_beta_is_softmax_logprior_minus_energy_over_tau():
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    logits = _B - _E / _TAU
    expect = torch.softmax(logits, dim=-1)
    assert torch.allclose(beta, expect, atol=1e-6)
    assert torch.allclose(beta.sum(-1), torch.tensor(1.0), atol=1e-6)


def test_envelope_identity_canonical_block_equals_neg_tau_logZ():
    # Sum_j beta* E + tau Sum_j beta* log(beta*/pi) == -tau log Z, with non-uniform pi.
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    pi = torch.softmax(_B, dim=-1)
    canon_block = (beta * _E).sum(-1) + _TAU * (beta * (torch.log(beta) - torch.log(pi))).sum(-1)
    fred = reduced_free_energy(_E, log_prior=_B, tau=_TAU)        # -tau log Z
    assert torch.allclose(canon_block, fred, atol=1e-5)
    # hand-computed literal backstop (catches a tau*log N offset):
    assert torch.allclose(fred, torch.tensor(1.1264), atol=1e-3)


def test_stationarity_residual_constant_across_keys():
    # At beta*, E_j + tau log(beta*_j/pi_j) is the SAME for every key j (= -tau log Z).
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    pi = torch.softmax(_B, dim=-1)
    residual = _E + _TAU * (torch.log(beta) - torch.log(pi))
    assert (residual.max() - residual.min()).abs() < 1e-5
    assert torch.allclose(residual.mean(), reduced_free_energy(_E, log_prior=_B, tau=_TAU), atol=1e-5)
