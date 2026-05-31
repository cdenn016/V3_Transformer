import torch
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.model.block import vfe_block
from vfe3.model.stack import vfe_stack


def _belief(N=4, K=4, n_gen=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return BeliefState(
        mu=torch.randn(N, K, generator=g),
        sigma=torch.rand(N, K, generator=g) + 0.5,
        phi=0.1 * torch.randn(N, n_gen, generator=g),
    )


def test_block_runs_e_step_and_preserves_shapes():
    cfg = VFE3Config(embed_dim=4, n_heads=2, n_e_steps=2, e_mu_lr=0.05, e_phi_lr=0.0)
    grp = get_group("block_glk")(4, 2)
    b = _belief(K=4, n_gen=grp.generators.shape[0])
    out = vfe_block(b, b.mu, b.sigma, grp, cfg)
    assert out.mu.shape == b.mu.shape and (out.sigma > 0).all()


def test_stack_handoff_updates_prior_across_blocks():
    cfg = VFE3Config(embed_dim=4, n_heads=2, n_layers=3, n_e_steps=1,
                     e_mu_lr=0.05, e_phi_lr=0.0, prior_handoff_rho=1.0)
    grp = get_group("block_glk")(4, 2)
    b = _belief(K=4, n_gen=grp.generators.shape[0])
    out = vfe_stack(b, b.mu, b.sigma, grp, cfg)
    assert out.mu.shape == b.mu.shape and (out.sigma > 0).all()
    # with rho=1 and a nonzero E-step, the stack moves the belief off the input
    assert not torch.allclose(out.mu, b.mu, atol=1e-4)


from vfe3.model.model import VFEModel


def test_model_forward_shapes_and_loss():
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=2,
                     n_e_steps=1, e_mu_lr=0.05, e_phi_lr=0.0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (3, 5))
    logits = model(tokens)
    assert logits.shape == (3, 5, 20)
    targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = model(tokens, targets)
    assert loss.shape == () and torch.isfinite(loss)


def test_loss_backward_reaches_prior_tables():
    # THE crown jewel: the unrolled E-step keeps the training graph connected, so the
    # M-step gradient reaches the encode/phi prior parameters (not just decode).
    cfg = VFE3Config(vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                     n_e_steps=2, e_mu_lr=0.05, e_phi_lr=0.02, gradient_mode="filtering")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 15, (2, 4)); targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
    assert model.prior_bank.phi_embed.grad is not None
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0          # gradient actually flows
    assert model.prior_bank.phi_embed.grad.abs().sum() > 0


def test_phi_e_step_actually_moves_phi():
    # Isolate the phi E-step: the crown jewel's phi_embed.grad>0 check passes even with
    # e_phi_lr=0 (phi flows via the in-graph transport omega in the kernel), so it does
    # not by itself exercise the phi update. Here we assert the OUTPUT belief.phi changes
    # between e_phi_lr=0 (phi step skipped) and e_phi_lr>0 (phi natgrad applied), which
    # exercises the phi E-step itself, not only the transport coupling.
    grp = get_group("block_glk")(4, 2)
    n_gen = grp.generators.shape[0]

    def run(e_phi_lr):
        b = _belief(K=4, n_gen=n_gen)
        cfg = VFE3Config(embed_dim=4, n_heads=2, n_layers=1, n_e_steps=2,
                         e_mu_lr=0.05, e_phi_lr=e_phi_lr)
        return vfe_stack(b, b.mu, b.sigma, grp, cfg).phi

    phi_off = run(0.0)
    phi_on = run(0.05)
    assert not torch.allclose(phi_off, phi_on, atol=1e-6)  # the phi step moves phi


def test_detach_e_step_with_phi_runs_and_reaches_decode_priors():
    # The documented fixed-point/truncated toggle (detach_e_step=True) must complete a
    # forward+backward even with the phi E-step enabled (e_phi_lr>0): the phi natgrad runs
    # under its own enable_grad island, so the blanket no_grad does not strangle it.
    # Under detach the E-step is severed, so priors are reached only via decode (mu/sigma);
    # phi_embed is frozen (no gradient path) -- pin both as the as-built semantics.
    cfg = VFE3Config(vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                     n_e_steps=2, e_mu_lr=0.05, e_phi_lr=0.02, detach_e_step=True)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 15, (2, 4)); targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)                                    # no crash, finite loss
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None              # decode reaches mu prior
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0
    assert model.prior_bank.phi_embed.grad is None                 # phi frozen under detach


def test_prior_handoff_sigma_must_be_convex():
    import pytest
    # rho_s outside [0,1] is a non-convex sigma blend that can drive the prior variance
    # negative off the SPD cone; the config must reject it (matching prior_handoff_rho).
    for bad in (2.0, -1.0):
        with pytest.raises(ValueError):
            VFE3Config(prior_handoff_sigma=bad)
    VFE3Config(prior_handoff_sigma=0.5)  # in-range is accepted


def test_all_ignore_batch_yields_finite_zero_loss():
    # A microbatch where every target == -100 must not poison the loss with NaN.
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                     n_e_steps=1, e_mu_lr=0.05)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 4)); targets = torch.full((2, 4), -100)
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.prior_bank.mu_embed.grad).all()


def test_model_has_no_nn_layers():
    import torch.nn as nn
    cfg = VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3)
    model = VFEModel(cfg)
    for m in model.modules():
        assert not isinstance(m, (nn.Linear, nn.MultiheadAttention, nn.RNNBase, nn.Conv1d))


def test_build_group_dispatches_on_arity_for_every_group():
    from vfe3.model.model import build_group
    # build_group must construct every registered group by positional arity alone,
    # so a new group slots in via register_group without editing the dispatcher.
    for g, n_gen_expected in (("glk", None), ("block_glk", None), ("so_k", None)):
        cfg = VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3, gauge_group=g)
        grp = build_group(cfg)
        assert grp.generators.shape[-1] == 4  # K


def test_group_generators_follow_dtype_move():
    # GaugeGroup is not an nn.Module; its generators must still follow model.to(dtype)
    # so the E-step transport (belief.phi, which moves) is matmul'd against same-dtype
    # generators. _apply re-maps them.
    cfg = VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3)
    model = VFEModel(cfg)
    model.to(torch.float64)
    assert model.prior_bank.mu_embed.dtype == torch.float64
    assert model.group.generators.dtype == torch.float64


def test_norm_type_final_is_wired():
    # norm_type_final must actually be applied before decode (not a dead seam): a
    # configured final norm changes the logits relative to 'none'.
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                n_e_steps=1, e_mu_lr=0.05)
    tok = torch.randint(0, 12, (2, 4))
    torch.manual_seed(0); m_norm = VFEModel(VFE3Config(**base, norm_type_final="mahalanobis"))
    torch.manual_seed(0); m_none = VFEModel(VFE3Config(**base, norm_type_final="none"))
    assert not torch.allclose(m_norm(tok), m_none(tok))


# --- Audit 2026-05-31 --------------------------------------------------------
def test_seed_pins_prior_bank_initialization():
    """The documented cfg.seed pins the PriorBank table init so a run is reproducible."""
    cfg = VFE3Config(vocab_size=40, embed_dim=8, n_heads=2, max_seq_len=4, seed=123)
    m1 = VFEModel(cfg)
    m2 = VFEModel(cfg)
    assert torch.equal(m1.prior_bank.mu_embed, m2.prior_bank.mu_embed)
    assert torch.equal(m1.prior_bank.phi_embed, m2.prior_bank.phi_embed)
    m3 = VFEModel(VFE3Config(vocab_size=40, embed_dim=8, n_heads=2, max_seq_len=4, seed=999))
    assert not torch.equal(m1.prior_bank.mu_embed, m3.prior_bank.mu_embed)


def test_killing_per_block_precond_runs_through_forward():
    # phi_precond_mode='killing_per_block' is a validated config seam; with e_phi_lr>0 it
    # must not crash the forward. The per-block Killing metric needs the group's irrep_dims,
    # which the E-step must thread into precondition_phi_gradient.
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                     n_e_steps=2, e_mu_lr=0.05, e_phi_lr=0.05,
                     phi_precond_mode="killing_per_block")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 4)); targets = torch.randint(0, 12, (2, 4))
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)


def test_b0_c0_threaded_into_state_dependent_alpha():
    """cfg.b0/c0 reach the state-dependent self-coupling alpha; changing b0 changes the belief."""
    grp = get_group("block_glk")(4, 2)
    n_gen = grp.generators.shape[0]

    def run(b0):
        b = _belief(K=4, n_gen=n_gen)
        cfg = VFE3Config(embed_dim=4, n_heads=2, n_layers=1, n_e_steps=2, e_mu_lr=0.1,
                         e_phi_lr=0.0, alpha_mode="state_dependent", b0=b0)
        return vfe_block(b, b.mu, b.sigma, grp, cfg).mu

    assert not torch.allclose(run(1.0), run(8.0), atol=1e-5)


def test_phi_retract_mode_bch_reachable_and_differs():
    """phi_retract_mode='bch' is reachable through the E-step and differs from 'euclidean'."""
    grp = get_group("block_glk")(4, 2)
    n_gen = grp.generators.shape[0]

    def run(mode):
        b = _belief(K=4, n_gen=n_gen)
        cfg = VFE3Config(embed_dim=4, n_heads=2, n_layers=1, n_e_steps=3, e_mu_lr=0.05,
                         e_phi_lr=0.2, phi_retract_mode=mode)
        return vfe_stack(b, b.mu, b.sigma, grp, cfg).phi

    assert not torch.allclose(run("euclidean"), run("bch"), atol=1e-7)
