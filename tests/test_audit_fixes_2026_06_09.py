r"""Regression pins for the audit-2026-06-09 punch-list fixes: the ALiBi x single-block guard
(P1), the filtering-kernel pair-term saturation mask (P7), windowed/T5 prior-parameter
threading (P9), the gamma-channel entropy toggle (P6), and the resume config-drift warning /
scaler persistence (P11). The b0/c0 x lambda_alpha_mode cross-check (P2) is pinned in
test_cheap_ledger_wins; the oracle truncation warning (P8) and the RoPE means-only warning (P5)
fire inside existing suites.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, load_checkpoint
from vfe3.train import build_optimizer


def _cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0)
    base.update(kw)
    return VFE3Config(**base)


# ------------------------------------------------- P1: alibi-family x head-axis guard

def test_alibi_with_single_block_group_rejected_at_construction():
    # so_k has irrep_dims=[K] (energy head axis 1); an (n_heads=2, N, N) prior would broadcast
    # into the batch axis -- silent corruption at B=1, crash at B>1 (audit P1).
    for pname in ("beta_attention_prior", "gamma_attention_prior"):
        with pytest.raises(ValueError, match="energy head axis"):
            VFEModel(_cfg(gauge_group="so_k", **{pname: "causal_alibi"}))


def test_alibi_single_block_runs_with_n_heads_1_at_any_batch():
    # n_heads=1 is the legal single-block pairing: the (1, N, N) prior is squeezed to the (N, N)
    # single-block convention, so a B=3 forward (the pre-fix crash case) runs with sane shapes.
    cfg = _cfg(gauge_group="so_k", n_heads=1, beta_attention_prior="causal_alibi")
    model = VFEModel(cfg)
    prior = model._attention_log_prior(5, torch.device("cpu"))
    assert prior.shape == (5, 5)                                # squeezed, no phantom head axis
    tok = torch.randint(0, 12, (3, 5)); tgt = torch.randint(0, 12, (3, 5))
    logits, loss, _ = model(tok, tgt)
    assert logits.shape == (3, 5, 12)
    assert torch.isfinite(loss)


def test_alibi_block_glk_with_matching_heads_still_constructs():
    model = VFEModel(_cfg(gauge_group="block_glk", beta_attention_prior="causal_alibi"))
    assert model._attention_log_prior(5, torch.device("cpu")).shape == (2, 5, 5)


# ------------------------------------------------- P7: pair-term saturation mask

def test_kernel_matches_oracle_on_fully_saturated_rows():
    # With kl_max tiny every pairwise energy saturates the clamp; autograd of F then carries a
    # ZERO pair gradient (d clamp/dE = 0). Pre-fix the kernel's unmasked pair term deviated by
    # orders of magnitude here (audit P7/N1); post-fix kernel == oracle.
    from vfe3.gradients.kernels import belief_gradients
    from vfe3.gradients.oracle import belief_gradients_autograd
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.transport import compute_transport_operators

    torch.manual_seed(0)
    grp = get_group("glk")(K=2)
    N, K = 4, 2
    mu = torch.randn(N, K); sigma = torch.rand(N, K) + 0.5
    mu_p = mu + 3.0                                              # large self/pair divergences
    sigma_p = torch.rand(N, K) + 0.5
    phi = 0.1 * torch.randn(1, N, grp.generators.shape[0])
    omega = compute_transport_operators(phi, grp)["Omega"][0]    # (N, N, K, K)
    kw = dict(tau=1.0, kl_max=1e-4, eps=1e-6, irrep_dims=grp.irrep_dims)
    g_mu_k, g_sig_k = belief_gradients(mu, sigma, mu_p, sigma_p, omega, **kw)
    g_mu_o, g_sig_o = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega, **kw)
    assert torch.allclose(g_mu_k, g_mu_o, atol=1e-6)
    assert torch.allclose(g_sig_k, g_sig_o, atol=1e-6)


# ------------------------------------------------- P9: prior parameter threading

def test_attention_window_threads_into_the_windowed_prior():
    model = VFEModel(_cfg(beta_attention_prior="causal_windowed", attention_window=1))
    prior = model._attention_log_prior(5, torch.device("cpu"))  # (N, N)
    d = torch.arange(5).unsqueeze(-1) - torch.arange(5).unsqueeze(0)   # i - j
    inside = (d >= 0) & (d <= 1)
    assert torch.isinf(prior[~inside]).all() and (prior[~inside] < 0).all()
    assert (prior[inside] == 0).all()                            # window=1 actually honored


def test_prior_shape_knobs_validated_positive():
    for name in ("attention_window", "t5_num_buckets", "t5_max_distance"):
        with pytest.raises(ValueError, match=name):
            _cfg(**{name: 0})


# ------------------------------------------------- P6: gamma honors the entropy toggle

def test_gamma_block_honors_include_attention_entropy():
    tok = torch.randint(0, 12, (2, 5)); tgt = torch.randint(0, 12, (2, 5))
    losses = {}
    for ent in (True, False):
        torch.manual_seed(0)
        m = VFEModel(_cfg(lambda_gamma=0.5, include_attention_entropy=ent))
        torch.manual_seed(1)
        _, loss, _ = m(tok, tgt)
        assert torch.isfinite(loss)
        losses[ent] = float(loss)
    # canonical envelope (-tau log Z) vs surrogate (sum gamma E) differ by the entropy term
    assert losses[True] != losses[False]


# ------------------------------------------------- P11: resume drift warning

def test_resume_warns_on_config_drift(tmp_path):
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                     warmup_steps=1, max_steps=4)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    path = art.save_checkpoint(2, model, opt, cfg)
    drifted = VFE3Config(**{**cfg.__dict__, "e_q_mu_lr": 0.05})
    with pytest.warns(UserWarning, match=r"config drift.*e_q_mu_lr"):
        load_checkpoint(path, model, opt, cfg=drifted)
    # identical config -> silent
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        load_checkpoint(path, model, opt, cfg=cfg)


def test_checkpoint_bundle_carries_scaler_state_slot(tmp_path):
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                     warmup_steps=1, max_steps=4)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    disabled = torch.amp.GradScaler(device="cpu", enabled=False)
    path = art.save_checkpoint(1, model, opt, cfg, scaler=disabled)
    ckpt = torch.load(path, weights_only=False)
    assert "scaler_state" in ckpt and ckpt["scaler_state"] is None   # disabled scaler -> None
    # a load with a scaler given and a None state is a documented no-op (pre-scaler bundles)
    load_checkpoint(path, model, opt, scaler=disabled)
