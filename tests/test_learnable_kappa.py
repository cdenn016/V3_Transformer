r"""Learnable per-irrep-block softmax temperatures (learnable_kappa_beta / learnable_kappa_gamma).

Design: docs/2026-07-05-learnable-per-head-kappa-design.md. Both toggles are default-OFF members
of the t5-exception family (sanctioned learned-scalar exceptions to the no-NN constraint): the
model stores log_kappa_* as an nn.Parameter of shape (len(group.irrep_dims),) and consumes
kappa = exp(log_kappa) through attention_tau, so tau = kappa * sqrt(d_block) stays strictly
positive and step 0 is byte-identical to the config-scalar path (log 1.0 = 0, exp 0 = 1 exactly).
A per-block scalar temperature multiplies the gauge-invariant per-block energy and touches no
gauge transport, so equivariance is preserved.

All models here are tiny (K = 4, single-digit dims) and CPU-bound per the project testing rules.
"""

import csv
import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _cfg(**kw):
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0)
    base.update(kw)
    return VFE3Config(**base)


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 12, (2, 8), generator=g)
    y = torch.randint(0, 12, (2, 8), generator=g)
    return x, y


# --- defaults / pure path ----------------------------------------------------

def test_learnable_kappa_defaults_false():
    cfg = VFE3Config()
    assert cfg.learnable_kappa_beta is False
    assert cfg.learnable_kappa_gamma is False


def test_absent_when_off():
    # Pure path: neither toggle creates a parameter (no attribute), state_dict stays param-free.
    m = VFEModel(_cfg())
    assert not hasattr(m, "log_kappa_beta")
    assert not hasattr(m, "log_kappa_gamma")
    assert not any(k.startswith("log_kappa") for k in m.state_dict())


# --- step-0 byte-identity ----------------------------------------------------

def test_step0_byte_identity_beta():
    # exp(log 1.0) = 1.0 exactly: the learnable model's forward is byte-identical to the
    # config-scalar path at construction (parameter creation draws zero RNG).
    x, y = _batch()
    torch.manual_seed(0)
    m_on = VFEModel(_cfg(learnable_kappa_beta=True))
    torch.manual_seed(0)
    m_off = VFEModel(_cfg())
    logits_on, loss_on, ce_on = m_on(x, y)
    logits_off, loss_off, ce_off = m_off(x, y)
    assert torch.equal(logits_on, logits_off)
    assert torch.equal(loss_on, loss_off)
    assert torch.equal(ce_on, ce_off)


def test_step0_byte_identity_gamma():
    # Same identity for the model channel: the scored gamma block (lambda_gamma > 0) at
    # kappa = exp(0) = 1.0 equals the config-scalar gamma block.
    x, y = _batch()
    torch.manual_seed(0)
    m_on = VFEModel(_cfg(learnable_kappa_gamma=True, lambda_gamma=0.1))
    torch.manual_seed(0)
    m_off = VFEModel(_cfg(lambda_gamma=0.1))
    _, loss_on, _ = m_on(x, y)
    _, loss_off, _ = m_off(x, y)
    assert torch.equal(loss_on, loss_off)


def test_perturbed_kappa_changes_loss():
    # The parameter is live: shifting log_kappa_beta changes the attention temperature and
    # therefore the loss (guards against a silently-disconnected parameter).
    x, y = _batch()
    torch.manual_seed(0)
    m = VFEModel(_cfg(learnable_kappa_beta=True))
    _, loss1, _ = m(x, y)
    with torch.no_grad():
        m.log_kappa_beta.add_(1.0)
    _, loss2, _ = m(x, y)
    assert not torch.equal(loss1, loss2)


# --- gradient flow / freeze --------------------------------------------------

def test_gradient_flows_under_unroll():
    # Canonical kernel route, e_step_gradient='unroll' (default): tau is never detached, so both
    # log-kappas receive a real, nonzero gradient from one backward pass.
    x, y = _batch()
    torch.manual_seed(0)
    m = VFEModel(_cfg(learnable_kappa_beta=True, learnable_kappa_gamma=True, lambda_gamma=0.1))
    _, loss, _ = m(x, y)
    loss.backward()
    assert m.log_kappa_beta.grad is not None
    assert torch.isfinite(m.log_kappa_beta.grad).all()
    assert m.log_kappa_beta.grad.abs().sum() > 0
    assert m.log_kappa_gamma.grad is not None
    assert torch.isfinite(m.log_kappa_gamma.grad).all()
    assert m.log_kappa_gamma.grad.abs().sum() > 0


@pytest.mark.parametrize("estimator", ["detach", "straight_through"])
def test_kappa_beta_frozen_under_severing_estimators(estimator):
    # kappa_beta enters the loss ONLY through the E-step softmax temperature; both severing
    # estimators cut that path, so the parameter receives no gradient (the family's
    # detach_e_step footgun) and construction warns.
    x, y = _batch()
    torch.manual_seed(0)
    with pytest.warns(UserWarning, match="log_kappa_beta"):
        m = VFEModel(_cfg(learnable_kappa_beta=True, e_step_gradient=estimator))
    _, loss, _ = m(x, y)
    loss.backward()
    assert m.log_kappa_beta.grad is None


def test_kappa_gamma_trains_through_scored_gamma_term_under_detach():
    # The scored gamma block (lambda_gamma > 0, s_e_step=False) is assembled at the LOSS level,
    # outside the E-step no_grad wrapper, so log_kappa_gamma trains even under 'detach' -- and
    # construction must NOT emit a freeze warning for it.
    x, y = _batch()
    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = VFEModel(_cfg(learnable_kappa_gamma=True, lambda_gamma=0.1, e_step_gradient="detach"))
    assert not any("log_kappa_gamma" in str(w.message) for w in rec)
    _, loss, _ = m(x, y)
    loss.backward()
    assert m.log_kappa_gamma.grad is not None
    assert m.log_kappa_gamma.grad.abs().sum() > 0


def test_kappa_gamma_frozen_under_s_e_step_detach():
    # Under s_e_step=True the gamma temperature is consumed only inside _refine_s's E-step
    # (the scored block is skipped), so 'detach' severs its only path: grad is None and
    # construction warns.
    x, y = _batch()
    torch.manual_seed(0)
    with pytest.warns(UserWarning, match="log_kappa_gamma"):
        m = VFEModel(_cfg(learnable_kappa_gamma=True, lambda_gamma=0.1, s_e_step=True,
                          prior_source="model_channel", e_step_gradient="detach"))
    _, loss, _ = m(x, y)
    loss.backward()
    assert m.log_kappa_gamma.grad is None


def test_kappa_gamma_inert_warning_when_no_gamma_path():
    # lambda_gamma == 0 and s_e_step=False: kappa_gamma reaches no loss term, so the toggle is
    # inert -- warn (mirrors the t5_learnable_bias inert warning) rather than silently no-op.
    torch.manual_seed(0)
    with pytest.warns(UserWarning, match="inert"):
        VFEModel(_cfg(learnable_kappa_gamma=True))


# --- shape / group coverage --------------------------------------------------

def test_shape_block_glk_per_head():
    m = VFEModel(_cfg(learnable_kappa_beta=True))          # block_glk, n_heads=2
    assert isinstance(m.log_kappa_beta, torch.nn.Parameter)
    assert m.log_kappa_beta.shape == (2,)
    assert torch.equal(m.log_kappa_beta.detach(), torch.zeros(2))   # log 1.0 = 0 exactly
    assert any(p is m.log_kappa_beta for p in m.parameters())


def test_shape_so_n_irrep_tower():
    # so_n tower [l0, l1] -> 2 irrep blocks of UNEQUAL dims [1, 3]: the parameter is sized by
    # len(irrep_dims), not n_heads.
    m = VFEModel(_cfg(learnable_kappa_beta=True, gauge_group="so_n", group_n=3,
                      irrep_spec=[("l0", 1), ("l1", 1)]))
    assert m.log_kappa_beta.shape == (2,)


def test_shape_single_block_vacuity_warning():
    # A single-irrep-block group makes per-head learning vacuous (one scalar temperature):
    # config warns but stays valid, and the model learns one scalar.
    with pytest.warns(UserWarning, match="single irrep block"):
        cfg = _cfg(learnable_kappa_beta=True, n_heads=1)
    m = VFEModel(cfg)
    assert m.log_kappa_beta.shape == (1,)


def test_list_init_used_elementwise():
    # A per-head kappa_beta list seeds the parameter elementwise: log_kappa = log(kappa_config).
    m = VFEModel(_cfg(learnable_kappa_beta=True, kappa_beta=[1.0, 2.0]))
    assert torch.equal(m.log_kappa_beta.detach(), torch.log(torch.tensor([1.0, 2.0])))


def test_training_logs_learned_kappa_stats(tmp_path):
    from vfe3.run_artifacts import RunArtifacts
    from vfe3.train import train

    torch.manual_seed(0)
    cfg = _cfg(learnable_kappa_beta=True, learnable_kappa_gamma=True, lambda_gamma=0.1,
               max_steps=2, log_interval=1)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic", device="cpu")
    train(model, [_batch(1), _batch(2)], cfg, n_steps=2, log_interval=1, eval_interval=0,
          artifacts=art, generate_samples=False)
    with open(tmp_path / "run" / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    for key in ("kappa_beta_mean", "kappa_beta_var", "kappa_gamma_mean", "kappa_gamma_var"):
        assert key in rows[0]
        assert rows[0][key] not in ("", "nan", None)


def test_training_logs_per_block_kappa_and_tau(tmp_path):
    # Per-block companion to the aggregate mean/var above: one kappa_<ch>_b<i> and one
    # tau_<ch>_b<i> = kappa_b * sqrt(d_b) column per irrep block, so finalize_run can draw a
    # line per block in each of the kappa/tau panels.
    from vfe3.run_artifacts import RunArtifacts
    from vfe3.train import train

    torch.manual_seed(0)
    cfg = _cfg(learnable_kappa_beta=True, learnable_kappa_gamma=True, lambda_gamma=0.1,
               max_steps=2, log_interval=1)              # block_glk, n_heads=2 -> 2 irrep blocks
    model = VFEModel(cfg)
    assert model.log_kappa_beta.shape == (2,)            # multi-block precondition for this test
    dims = model.group.irrep_dims
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic", device="cpu")
    train(model, [_batch(1), _batch(2)], cfg, n_steps=2, log_interval=1, eval_interval=0,
          artifacts=art, generate_samples=False)
    with open(tmp_path / "run" / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    for ch in ("beta", "gamma"):
        for bi in range(2):
            for key in (f"kappa_{ch}_b{bi}", f"tau_{ch}_b{bi}"):
                assert key in rows[0], f"{key} missing from metrics.csv"
                assert rows[0][key] not in ("", "nan", None)
    for bi in range(2):                                  # tau_b == kappa_b * sqrt(d_b)
        kb = float(rows[0][f"kappa_beta_b{bi}"])
        tb = float(rows[0][f"tau_beta_b{bi}"])
        assert tb == pytest.approx(kb * float(dims[bi]) ** 0.5, rel=1e-5)


# --- optimizer wiring ----------------------------------------------------------

def test_optimizer_grouping():
    # Both parameters land in exactly one group: role='mu', weight_decay=0.0 (a temperature
    # decayed toward the fixed calibration biases the softmax), no gauge flag; the
    # exact-coverage guard passes.
    from vfe3.train import build_optimizer
    torch.manual_seed(0)
    m = VFEModel(_cfg(learnable_kappa_beta=True, learnable_kappa_gamma=True, lambda_gamma=0.1))
    opt = build_optimizer(m, m.cfg)                        # coverage guard raises if ungrouped
    for p in (m.log_kappa_beta, m.log_kappa_gamma):
        gs = [g for g in opt.param_groups if any(q is p for q in g["params"])]
        assert len(gs) == 1
        g = gs[0]
        assert g["lr"] == m.cfg.m_p_mu_lr
        assert g["weight_decay"] == 0.0
        assert g["role"] == "mu"
        assert not g.get("gauge", False)
