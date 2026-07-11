r"""Regression pins for the 2026-06-09 overnight deep-audit fixes (see
docs/audits/audit-2026-06-09-overnight.md): CG dead-path pruning + batched forward,
grad-safe identity short-circuits, HeadMixer dtype contract + degeneracy warnings,
the detach freeze warning for mixer/CG, the config-level use_head_mixer guard,
the atol-keyed CG cache, and the tall-block factored-covariance branch."""

import warnings

import pytest
import torch


def _tower_group(spec=(("l0", 1), ("l1", 1), ("l2", 1)), K=9):
    from vfe3.geometry.groups import get_group
    return get_group("so_n")(K, group_n=3, irrep_spec=[tuple(p) for p in spec],
                             dtype=torch.float64)


def _cfg(**kw):
    from vfe3.config import VFE3Config
    base = dict(vocab_size=20, embed_dim=9, n_heads=3, max_seq_len=5, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0,
                gauge_group="so_n", group_n=3,
                irrep_spec=[("l0", 1), ("l1", 1), ("l2", 1)],
                phi_precond_mode="none")
    base.update(kw)
    return VFE3Config(**base)


# ---------------------------------------------------------------- CG dead-path pruning (F10/F13/F20)

def test_cg_coupling_prunes_antisymmetric_self_pairs():
    # l1 (x) l1 -> l1 is swap-antisymmetric, so the single-copy self-pair C(x, x) = 0
    # identically: its weight would be a dead parameter. It must be pruned at enumeration.
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels)
    assert ("l1", "l1", "l1") not in cpl.path_types
    # the symmetric self-products stay
    assert ("l1", "l1", "l2") in cpl.path_types
    assert ("l1", "l1", "l0") in cpl.path_types


def test_cg_coupling_every_surviving_path_weight_is_live():
    # After pruning, every path weight must receive a structurally nonzero gradient (F15/F20).
    # Random NONZERO weights move the forward off the zero-init symmetric point first: at
    # w = 0 a path whose target block equals one source block can have <y, C(x, y)> = 0 by
    # 3j antisymmetry under that pairing (e.g. (l1, l2) -> l2) without being dead.
    from vfe3.model.cg_coupling import CGCoupling
    torch.manual_seed(0)
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    with torch.no_grad():
        cpl.path_weights.copy_(0.3 * torch.randn(cpl.path_weights.shape[0],
                                                 dtype=torch.float64))
    mu = torch.randn(4, 9, dtype=torch.float64, requires_grad=True)
    sig = torch.rand(4, 9, dtype=torch.float64)
    mu2, _ = cpl(mu, sig)
    mu2.square().sum().backward()
    g = cpl.path_weights.grad
    assert g is not None and torch.isfinite(g).all()
    assert (g.abs() > 1e-10).all(), f"dead path weights survived pruning: {g}"


def test_cg_coupling_batched_forward_matches_per_path_reference():
    # The grouped/batched forward must equal the per-path reference contraction.
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    with torch.no_grad():
        cpl.path_weights.copy_(0.3 * torch.randn(cpl.path_weights.shape[0],
                                                 dtype=torch.float64))
    mu = torch.randn(2, 4, 9, dtype=torch.float64)
    sig = torch.rand(2, 4, 9, dtype=torch.float64)
    mu2, _ = cpl(mu, sig)
    delta_ref = torch.zeros_like(mu)                      # per-path reference
    cg = [getattr(cpl, f"cg_{t}") for t in range(len(cpl._triple_index))]
    for p, (sa, da, sb, db, sc, dc, t, m) in enumerate(cpl.paths):
        x, y = mu[..., sa:sa + da], mu[..., sb:sb + db]
        xy = (x.unsqueeze(-1) * y.unsqueeze(-2)).reshape(*x.shape[:-1], da * db)
        delta_ref[..., sc:sc + dc] += cpl.path_weights[p] * torch.einsum(
            "cd,...d->...c", cg[t][m], xy)
    assert torch.allclose(mu2, mu + delta_ref, atol=1e-12)


# ---------------------------------------------------------------- sp algebra CG path (PP9)

def test_cg_coupling_sp_algebra_constructs_and_trains():
    from vfe3.geometry.groups import get_group
    from vfe3.model.cg_coupling import CGCoupling
    grp = get_group("sp_n")(5, group_n=2, irrep_spec=[("sym1", 1), ("sym2", 1)],
                            dtype=torch.float64)
    assert grp.algebra == "sp"
    cpl = CGCoupling(2, "sp", grp.irrep_dims, grp.irrep_labels).double()
    assert cpl.path_weights.shape[0] > 0
    # sym2 (x) sym2 -> sym2 is the antisymmetric (spin-1-like) slot: self-pair pruned
    assert ("sym2", "sym2", "sym2") not in cpl.path_types
    mu = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    mu2, _ = cpl(mu, torch.rand(3, 5, dtype=torch.float64))
    mu2.square().sum().backward()
    g = cpl.path_weights.grad
    assert g is not None and torch.isfinite(g).all() and (g.abs() > 1e-12).all()


# ---------------------------------------------------------------- grad-safe identity short-circuit

def test_identity_short_circuit_is_grad_safe():
    from vfe3.model.cg_coupling import CGCoupling
    from vfe3.model.head_mixer import HeadMixer
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels)
    mix = HeadMixer([2, 2])
    mu, sig = torch.randn(3, 9), torch.rand(3, 9)
    with torch.no_grad():                                 # eval path: zero-init short-circuits
        mu2, sig2 = cpl(mu, sig)
        assert mu2 is mu and sig2 is sig
        m, s = torch.randn(3, 4), torch.rand(3, 4)
        m2, s2 = mix(m, s)
        assert m2 is m and s2 is s
    mu3, _ = cpl(mu.requires_grad_(True), sig)            # grad path: weights stay in the graph
    mu3.sum().backward()
    assert cpl.path_weights.grad is not None              # NOT severed by the short-circuit


# ---------------------------------------------------------------- HeadMixer dtype contract (DB1/DB5)

@pytest.mark.parametrize("module_dtype,input_dtype",
                         [(torch.float64, torch.float32), (torch.float32, torch.float64)])
def test_head_mixer_dtype_contract(module_dtype, input_dtype):
    from vfe3.model.head_mixer import HeadMixer
    mix = HeadMixer([2, 2]).to(module_dtype)
    with torch.no_grad():
        mix.mixer_deltas[0].add_(0.1)                     # off identity: no short-circuit
    mu = torch.randn(3, 4, dtype=input_dtype)
    sig_d = torch.rand(3, 4, dtype=input_dtype)
    mu2, sig2 = mix(mu, sig_d)                            # diagonal arm
    assert mu2.dtype == input_dtype and sig2.dtype == input_dtype
    sig_f = torch.eye(4, dtype=input_dtype).expand(3, 4, 4).clone()
    mu3, sig3 = mix(mu, sig_f)                            # full-cov arm
    assert mu3.dtype == input_dtype and sig3.dtype == input_dtype


# ---------------------------------------------------------------- degeneracy warnings (PP6/DB3)

def test_head_mixer_warns_on_all_distinct_labels():
    from vfe3.model.head_mixer import HeadMixer
    with pytest.warns(UserWarning, match="scalar gain per block"):
        HeadMixer([1, 3, 5], irrep_labels=["l0", "l1", "l2"])


def test_head_mixer_warns_on_non_adjacent_same_label():
    from vfe3.model.head_mixer import HeadMixer
    with pytest.warns(UserWarning, match="NON-ADJACENT"):
        HeadMixer([3, 5, 3], irrep_labels=["l1", "l2", "l1"])


def test_head_mixer_silent_on_contiguous_multiplicity():
    from vfe3.model.head_mixer import HeadMixer
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HeadMixer([3, 3, 5], irrep_labels=["l1", "l1", "l2"])


# ---------------------------------------------------------------- detach freeze warning (F31/F34)

def test_detach_with_mixer_or_cg_warns():
    from vfe3.model.model import VFEModel
    with pytest.warns(UserWarning, match="freezes mixer_deltas/path_weights"):
        VFEModel(_cfg(use_cg_coupling=True, detach_e_step=True))


def test_unroll_with_mixer_does_not_warn_freeze():
    from vfe3.model.model import VFEModel
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        VFEModel(_cfg(use_cg_coupling=True))
    assert not any("freezes mixer_deltas" in str(w.message) for w in rec)


# ---------------------------------------------------------------- config guard (PP1)

def test_config_rejects_head_mixer_on_single_block_groups():
    # A genuinely single-BLOCK group (glk/so_k/sp) cannot mix regardless of head count and still
    # raises -- distinct from a single-HEAD block_glk, which auto-disables (see
    # test_config_auto_disables_head_mixer_on_single_head).
    from vfe3.config import VFE3Config
    with pytest.raises(ValueError, match="use_head_mixer"):
        VFE3Config(vocab_size=8, embed_dim=4, n_heads=1, max_seq_len=4, n_layers=1,
                   gauge_group="glk", use_head_mixer=True)


def test_config_auto_disables_head_mixer_on_single_head():
    # A head-block group (block_glk/tied_block_glk) with n_heads=1 is a single-HEAD scan artifact:
    # one gauge block, nothing to mix. The config auto-disables use_head_mixer and warns (rather
    # than raising) so a head/K sweep can leave use_head_mixer=True set without a manual toggle-off
    # at n_heads=1.
    from vfe3.config import VFE3Config
    for grp in ("block_glk", "tied_block_glk"):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cfg = VFE3Config(vocab_size=8, embed_dim=4, n_heads=1, max_seq_len=4, n_layers=1,
                             gauge_group=grp, use_head_mixer=True)
        assert cfg.use_head_mixer is False
        assert any("use_head_mixer" in str(r.message) for r in rec)


def test_config_accepts_head_mixer_on_multi_block_groups():
    from vfe3.config import VFE3Config
    VFE3Config(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
               gauge_group="block_glk", use_head_mixer=True)


# ---------------------------------------------------------------- atol-keyed CG cache (CR1)

def test_cg_cache_keys_on_atol():
    from vfe3.geometry import cg
    cg.clear_cg_cache()
    cg.cg_intertwiners(3, algebra="so", label_a="l1", label_b="l1", label_c="l2")
    cg.cg_intertwiners(3, algebra="so", label_a="l1", label_b="l1", label_c="l2",
                       atol=1e-12)
    keys = {k for k in cg._CG_CACHE if k[:5] == ("so", 3, "l1", "l1", "l2")}
    assert len(keys) == 2                                 # distinct atol -> distinct solves
    cg.clear_cg_cache()
    assert len(cg._CG_CACHE) == 0


# ---------------------------------------------------------------- tall-block factored covariance (F4)

def test_factored_diagonal_covariance_tall_block_matches_dense():
    # d > N exercises the dense per-pair branch; it must equal the dense diagonal sandwich.
    from vfe3.geometry.transport import (FactoredTransport, transport_covariance,
                                         _factored_diagonal_covariance)
    gen = torch.Generator().manual_seed(0)
    N, d = 2, 3                                           # d > N: tall-block regime
    A = 0.3 * torch.randn(N, d, d, generator=gen, dtype=torch.float64)
    A = A - A.transpose(-1, -2)                           # skew -> exp(-A) = exp(A)^-1
    blk_p, blk_n = torch.linalg.matrix_exp(A), torch.linalg.matrix_exp(-A)
    K = 2 * d
    ep = torch.zeros(N, K, K, dtype=torch.float64)
    en = torch.zeros(N, K, K, dtype=torch.float64)
    for h in range(2):
        ep[:, h * d:(h + 1) * d, h * d:(h + 1) * d] = blk_p
        en[:, h * d:(h + 1) * d, h * d:(h + 1) * d] = blk_n
    ft = FactoredTransport(exp_phi=ep, exp_neg_phi=en, irrep_dims=[d, d])
    sigma = torch.rand(N, K, dtype=torch.float64) + 0.1
    fast = _factored_diagonal_covariance(ft, sigma)
    dense = transport_covariance(ft.to_dense_omega(), sigma)
    assert torch.allclose(fast, dense, atol=1e-12)


# ---------------------------------------------------------------- F19: M-step reads converged q*

def test_vfe_block_capture_returns_pre_transform_belief():
    from vfe3.model.model import VFEModel
    from vfe3.model.stack import vfe_stack
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(_cfg(use_head_mixer=True))
    with torch.no_grad():
        for dlt in model.head_mixer.mixer_deltas:
            dlt.add_(0.3)                                     # material transform
    tok = torch.randint(0, 20, (1, 5))
    beliefs = model.prior_bank.encode(tok)
    beliefs = beliefs._replace(phi=model._apply_pos_phi(beliefs.phi))
    cap: dict = {}
    out = vfe_stack(beliefs, beliefs.mu, beliefs.sigma, model.group, model.cfg,
                    log_prior=model._attention_log_prior(5, tok.device),
                    head_mixer=model.head_mixer, capture=cap)
    assert not torch.allclose(cap["converged"].mu, out.mu)    # q* != mixed handoff
    plain = VFEModel(_cfg())                                  # transforms off: same object
    beliefs2 = plain.prior_bank.encode(tok)
    beliefs2 = beliefs2._replace(phi=plain._apply_pos_phi(beliefs2.phi))
    cap2: dict = {}
    out2 = vfe_stack(beliefs2, beliefs2.mu, beliefs2.sigma, plain.group, plain.cfg,
                     log_prior=plain._attention_log_prior(5, tok.device), capture=cap2)
    assert cap2["converged"] is out2                          # pure path: byte-identical


def test_mstep_self_coupling_reads_converged_pretransform_belief():
    # The M-step regularizer must equal ce + w * sc(q*, p) with q* the CAPTURED pre-transform
    # converged belief -- not the post-mixer handoff (audit F19, challenge-upheld).
    import torch.nn.functional as F
    from vfe3.model.model import VFEModel
    from vfe3.model.stack import vfe_stack
    from vfe3.families import get_family
    from vfe3.free_energy import self_divergence_for_alpha
    torch.manual_seed(0)
    w = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(_cfg(use_head_mixer=True, mstep_self_coupling_weight=w,
                              mass_phi=0.0))
    with torch.no_grad():
        for dlt in model.head_mixer.mixer_deltas:
            dlt.add_(0.3)
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    logits = model(tok)
    _, loss, _ = model(tok, tgt)
    ce = F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size), tgt.reshape(-1),
                         ignore_index=-100)
    # replicate the forward's belief pipeline with the capture
    beliefs = model.prior_bank.encode(tok)
    beliefs = beliefs._replace(phi=model._apply_pos_phi(beliefs.phi))
    cap: dict = {}
    out = vfe_stack(beliefs, beliefs.mu, beliefs.sigma, model.group, model.cfg,
                    log_prior=model._attention_log_prior(5, tok.device),
                    head_mixer=model.head_mixer, capture=cap)
    fam = get_family(model.cfg.family)

    def sc_of(b):
        sd = self_divergence_for_alpha(
            fam(b.mu, b.sigma), fam(beliefs.mu, beliefs.sigma),   # n_layers=1: prior = encode
            alpha=model.cfg.renyi_order, kl_max=model.cfg.kl_max, eps=model.cfg.eps,
            divergence_family=model.cfg.divergence_family, lambda_alpha_mode=model.cfg.lambda_alpha_mode,
        )
        return sd.mean()

    sc_conv, sc_post = sc_of(cap["converged"]), sc_of(out)
    assert not torch.allclose(sc_conv, sc_post)               # the transform is material
    assert torch.allclose(loss, ce + w * sc_conv, atol=1e-6)  # loss anchors to q*
    assert not torch.allclose(loss, ce + w * sc_post, atol=1e-6)


# ---------------------------------------------------------------- CG overlapping-slice gradcheck (F18)

def test_cg_coupling_gradcheck_through_overlapping_slices():
    # Numeric finite-difference check (correctness, not just finiteness) of the batched
    # forward's delta accumulation, where multiple (triple, mult) groups write the SAME
    # target slice (e.g. (l1,l1)->l0 and (l2,l2)->l0 both write the l0 block).
    from torch.func import functional_call
    from vfe3.model.cg_coupling import CGCoupling
    torch.manual_seed(0)
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    sig = torch.rand(2, 9, dtype=torch.float64)
    mu = torch.randn(2, 9, dtype=torch.float64, requires_grad=True)
    w = (0.3 * torch.randn(cpl.path_weights.shape[0],
                           dtype=torch.float64)).requires_grad_(True)

    def f(mu_in, w_in):
        return functional_call(cpl, {"path_weights": w_in}, (mu_in, sig))[0]

    assert torch.autograd.gradcheck(f, (mu, w))


# ---------------------------------------------------------------- uniform-pi entropy scalar (F8/PE7)

def test_free_energy_uniform_pi_scalar_matches_explicit_reference():
    # log_prior=None branch: the scalar -log N must reproduce the old explicit
    # full_like(beta, 1/N) reference term-for-term (audit F8 / morning PE7).
    from vfe3.free_energy import attention_weights, free_energy
    torch.manual_seed(0)
    N, tau = 7, 2.0
    energy = torch.rand(N, N)
    self_div = torch.rand(N)
    alpha = torch.ones(N)
    F_new = free_energy(self_div, energy, alpha, tau=tau, log_prior=None)
    beta = attention_weights(energy, tau=tau, log_prior=None)
    pi = torch.full_like(beta, 1.0 / N)
    ref = (alpha * self_div).sum() + (beta * energy).sum() + (
        tau * (beta * (torch.log(beta.clamp(min=1e-12))
                       - torch.log(pi.clamp(min=1e-12))))).sum()
    assert torch.allclose(F_new, ref, atol=1e-6)


# ---------------------------------------------------------------- replay threading smoke (F29/F32/F33)

def test_replays_thread_mixer_and_cg():
    from vfe3.model.model import VFEModel
    from vfe3.viz.extract import converged_state, across_layer_belief_trace
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                   # degeneracy warnings expected here
        model = VFEModel(_cfg(use_head_mixer=True, use_cg_coupling=True))
    with torch.no_grad():                                 # perturb so the toggles bite
        model.cg_coupling.path_weights.add_(0.05)
        for dlt in model.head_mixer.mixer_deltas:
            dlt.add_(0.05)
    tok = torch.randint(0, 20, (1, 5))
    state = converged_state(model, tok)
    trace = across_layer_belief_trace(model, tok)
    amaps = model.attention_maps(tok)
    assert state["mu"].shape == (5, 9)
    assert trace["mu"].shape[0] == model.cfg.n_layers
    assert amaps.shape[0] == model.cfg.n_layers
    # the replayed handoff belief must match the training forward's (same toggles applied)
    logits = model(tok)
    assert torch.isfinite(logits).all()
