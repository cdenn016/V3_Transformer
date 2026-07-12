import csv
import gc
import json
import logging
import math
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, finalize_validation_run, load_checkpoint
from vfe3.train import (
    TrainingTerminalState,
    build_optimizer,
    evaluate,
    lr_lambda,
    train,
    train_step,
    _floor_lr_lambdas,
)


def test_optimizer_groups_priors_by_m_lr():
    # use_prior_bank=True: the prior-bank decode has no output_proj_weight group, so the four prior
    # tables map to exactly four LR groups (the linear-decode default adds an output_proj group).
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     m_p_mu_lr=0.01, m_p_sigma_lr=0.002, m_phi_lr=0.005, use_prior_bank=True)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    # m_p_sigma_lr=0.002, then TWO groups at m_phi_lr=0.005 (phi_embed + the default pos_phi='learned'
    # pos_phi_free table, grouped at m_phi_lr in train.py), then m_p_mu_lr=0.01.
    assert lrs == [0.002, 0.005, 0.005, 0.01]
    # every PriorBank parameter is covered by exactly one group
    n_params = sum(len(g["params"]) for g in opt.param_groups)
    assert n_params == len(list(model.parameters()))


def test_phi_clamp_monitor_threshold_matches_transport_clamp():
    # M2 (audit 2026-07-06): the M-step drift monitor must trip at the SAME Frobenius norm the
    # transport clamp actually fires at -- else a phi whose embedded norm lands in (clamp, monitor]
    # silently receives the surrogate exp(max_norm*M/||M||) transport with no warning. Pin the two
    # defaults equal so they cannot drift apart again.
    import inspect
    from vfe3.train import _warn_phi_transport_clamp
    from vfe3.geometry.transport import stable_matrix_exp_pair
    monitor_thr = inspect.signature(_warn_phi_transport_clamp).parameters["max_norm"].default
    clamp_thr   = inspect.signature(stable_matrix_exp_pair).parameters["max_norm"].default
    assert monitor_thr == clamp_thr, f"monitor trips at {monitor_thr}, clamp fires at {clamp_thr}"


def test_phi_clamp_monitor_reuses_cached_gram(monkeypatch):
    import vfe3.train as train_module

    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4)
    model = VFEModel(cfg)
    calls = []
    real_einsum = torch.einsum

    def _counted_einsum(equation, *operands, **kwargs):
        if equation == "aij,bij->ab":
            calls.append(equation)
        return real_einsum(equation, *operands, **kwargs)

    train_module._PHI_CLAMP_WARNED = False
    train_module._PHI_CLAMP_GRAM_CACHE.clear()
    monkeypatch.setattr(torch, "einsum", _counted_einsum)

    train_module._warn_phi_transport_clamp(model)
    train_module._warn_phi_transport_clamp(model)

    assert calls == ["aij,bij->ab"]


def test_phi_clamp_monitor_invalidates_cached_gram_on_generator_version(monkeypatch):
    import vfe3.train as train_module

    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4)
    model = VFEModel(cfg)
    calls = []
    real_einsum = torch.einsum

    def _counted_einsum(equation, *operands, **kwargs):
        if equation == "aij,bij->ab":
            calls.append(equation)
        return real_einsum(equation, *operands, **kwargs)

    train_module._PHI_CLAMP_WARNED = False
    train_module._PHI_CLAMP_GRAM_CACHE.clear()
    monkeypatch.setattr(torch, "einsum", _counted_einsum)

    train_module._warn_phi_transport_clamp(model)
    with torch.no_grad():
        model.group.generators.add_(0.0)              # value-stable mutation still increments _version
    train_module._warn_phi_transport_clamp(model)

    assert calls == ["aij,bij->ab", "aij,bij->ab"]


def test_phi_clamp_gram_cache_uses_full_metadata_key_without_value_reads():
    from vfe3.train import _phi_clamp_gram_key

    class _MetadataOnlyGenerator:
        shape = (3, 4, 4)
        device = torch.device("cuda")
        dtype = torch.float32
        _version = 7

        def item(self):
            raise AssertionError("cache key read CUDA tensor data")

        def cpu(self):
            raise AssertionError("cache key copied CUDA tensor data to the host")

        def tolist(self):
            raise AssertionError("cache key copied CUDA tensor data to the host")

        def __bool__(self):
            raise AssertionError("cache key synchronized through tensor truthiness")

    generators = _MetadataOnlyGenerator()
    key = _phi_clamp_gram_key(generators)

    assert key == (
        id(generators),
        (3, 4, 4),
        torch.device("cuda"),
        torch.float32,
        7,
    )


def test_phi_clamp_gram_cache_stores_detached_tensor():
    import vfe3.train as train_module

    generators = torch.randn(2, 2, 2, requires_grad=True)
    train_module._PHI_CLAMP_GRAM_CACHE.clear()

    gram = train_module._cached_phi_clamp_gram(generators)

    assert gram.requires_grad is False
    assert gram.grad_fn is None
    train_module._PHI_CLAMP_GRAM_CACHE.clear()


def test_phi_clamp_gram_cache_weakref_cleanup():
    import vfe3.train as train_module

    generators = torch.zeros(2, 2, 2)
    model = SimpleNamespace(
        group=SimpleNamespace(generators=generators),
        prior_bank=SimpleNamespace(phi_embed=None),
        pos_phi_free=None,
    )
    train_module._PHI_CLAMP_WARNED = False
    train_module._PHI_CLAMP_GRAM_CACHE.clear()

    train_module._warn_phi_transport_clamp(model)
    generator_ref = weakref.ref(generators)
    assert train_module._PHI_CLAMP_GRAM_CACHE

    del model
    del generators
    gc.collect()

    assert generator_ref() is None
    assert train_module._PHI_CLAMP_GRAM_CACHE == {}


def test_parameter_report_leaves_global_rng_untouched():
    # m31: parameter_report's probe forward draws the GLOBAL RNG under randomize_e_steps=True (the
    # E-step count randint), so it must snapshot/restore around the probe -- its docstring promises the
    # global stream is untouched (the deprecated run_training entry's batch order depends on it).
    from vfe3.train import parameter_report
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1, n_e_steps=2,
                     randomize_e_steps=True, e_steps_min=1, e_steps_max=4)
    model = VFEModel(cfg)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    parameter_report(model)
    assert torch.equal(before, torch.get_rng_state()), "parameter_report advanced the global RNG"


def test_optimizer_groups_regime_ii_connection():
    # connection_W (transport_mode='regime_ii') is a trainable model-level nn.Parameter;
    # build_optimizer must group it so it actually trains and so its exact-coverage guard does not
    # raise. A >=2-head block group lets regime_ii build.
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     transport_mode="regime_ii")
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)                           # must not raise (coverage guard)
    grouped = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(model.connection_W) in grouped


def test_lr_lambda_warmup_then_cosine():
    cfg = VFE3Config(warmup_steps=10, max_steps=100)
    assert abs(lr_lambda(0, cfg) - 0.0) < 1e-6
    assert abs(lr_lambda(10, cfg) - 1.0) < 1e-6            # peak at end of warmup
    assert lr_lambda(55, cfg) < 1.0 and lr_lambda(55, cfg) > 0.0
    assert abs(lr_lambda(100, cfg) - 0.0) < 1e-3           # ~0 at max_steps


def test_scheduler_floors_lr_at_min_lr():
    # The cosine multiplier decays to 0 at max_steps; the floored per-group scheduler built in
    # train() must keep EACH group's absolute LR >= cfg.min_lr there (and beyond). Build the real
    # optimizer+scheduler, fast-forward to max_steps, and check every group.
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     warmup_steps=2, max_steps=10, min_lr=1e-5)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _floor_lr_lambdas(base_lrs, cfg))
    for _ in range(cfg.max_steps + 5):                     # step past max_steps into the clamped tail
        sched.step()
    for lr in sched.get_last_lr():
        assert lr >= cfg.min_lr - 1e-12                    # floored, never decays to zero
        assert math.isclose(lr, cfg.min_lr, rel_tol=1e-9)  # at the tail every group sits exactly on the floor


def test_scheduler_min_lr_zero_is_pure_cosine():
    # min_lr=0.0 is the theoretically pure path: the floor max(0/base, cosine)=cosine, so the LR
    # decays to exactly zero at max_steps, identical to the unfloored half-cosine.
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     warmup_steps=2, max_steps=10, min_lr=0.0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _floor_lr_lambdas(base_lrs, cfg))
    for _ in range(cfg.max_steps):
        sched.step()
    for lr in sched.get_last_lr():
        assert abs(lr) < 1e-9                              # pure cosine reaches zero


def test_fractional_floor_scales_each_group_to_min_lr_frac_times_base():
    # Option B: min_lr_frac floors EACH group's absolute LR at min_lr_frac * its own base LR,
    # so the m_mu:m_sigma:m_phi base ratios are preserved into the cosine tail (unlike the shared
    # absolute min_lr, which floors every group at the same value regardless of base).
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     warmup_steps=2, max_steps=10, min_lr=0.0, min_lr_frac=0.01,
                     m_p_mu_lr=0.02, m_p_sigma_lr=0.004, m_phi_lr=0.01)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _floor_lr_lambdas(base_lrs, cfg))
    for _ in range(cfg.max_steps + 5):                     # into the clamped tail
        sched.step()
    for base, lr in zip(base_lrs, sched.get_last_lr()):
        assert math.isclose(lr, cfg.min_lr_frac * base, rel_tol=1e-9)


def test_floor_is_max_of_absolute_min_lr_and_fractional():
    # Both knobs live together: each group floors at max(min_lr, min_lr_frac * base). With
    # min_lr=1e-3, min_lr_frac=0.01: mu(base 0.2)->frac 2e-3 wins; sigma(base 0.05)->abs 1e-3 wins.
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     warmup_steps=2, max_steps=10, min_lr=1e-3, min_lr_frac=0.01,
                     m_p_mu_lr=0.2, m_p_sigma_lr=0.05, m_phi_lr=0.2)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _floor_lr_lambdas(base_lrs, cfg))
    for _ in range(cfg.max_steps + 5):
        sched.step()
    for base, lr in zip(base_lrs, sched.get_last_lr()):
        assert math.isclose(lr, max(cfg.min_lr, cfg.min_lr_frac * base), rel_tol=1e-9)


def test_floor_lambdas_handle_zero_base_lr_without_dividing():
    # A deliberately frozen channel (m_phi_lr=0) gives a group with base LR 0. The floor builder
    # must NOT compute min_lr/0 (ZeroDivisionError), and the frozen group must stay at 0 -- an
    # absolute min_lr does not resurrect a channel the user chose to freeze.
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     warmup_steps=2, max_steps=10, min_lr=1e-5, m_phi_lr=0.0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    assert 0.0 in base_lrs                                 # the frozen phi group is present
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _floor_lr_lambdas(base_lrs, cfg))  # must not raise
    for _ in range(cfg.max_steps + 5):
        sched.step()
    for base, lr in zip(base_lrs, sched.get_last_lr()):
        if base == 0.0:
            assert lr == 0.0                               # frozen stays frozen
        else:
            assert lr >= cfg.min_lr - 1e-12


# The active alphabet of the period-3 stream is {0,1,2}; a structure-BLIND predictor
# (one that learns only the unigram frequencies of the active tokens) is pinned at the
# marginal entropy ln(3) ~ 1.0986. Beating that floor by a margin is the discriminating
# evidence that the model learned the period-3 NEXT-TOKEN structure, not just the marginal.
_MARGINAL_ENTROPY_P3 = math.log(3)                              # unigram floor of the 3 active tokens
_CUTOVER_MARGIN      = 0.05                                     # nats below the marginal a learner must reach


def _periodic_loader(V=6, period=3, n=600, seq_len=8, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(period).repeat(n // period + 2)         # 0,1,2,0,1,2,...
    ds = TokenWindows(base[: n].to(torch.long), seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      generator=g)


def test_run_training_applies_cfg_seed_and_deterministic_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import vfe3.train as train_module
    from vfe3.runtime import seed_everything

    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, batch_size=2, deterministic=False, seed=37)
    loader = _periodic_loader(V=cfg.vocab_size, n=96, seq_len=cfg.max_seq_len,
                              batch_size=cfg.batch_size, seed=cfg.seed)
    monkeypatch.setattr(train_module, "make_dataloader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(train_module, "train", lambda *args, **kwargs: [])

    was_algorithms = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    was_cudnn_deterministic = torch.backends.cudnn.deterministic
    was_cudnn_benchmark = torch.backends.cudnn.benchmark
    had_cublas = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    was_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        seed_everything(999, deterministic=True)
        model, _ = train_module.run_training(cfg, dataset="synthetic", n_steps=0)
        assert model.cfg is cfg
        assert torch.initial_seed() == cfg.seed
        assert torch.are_deterministic_algorithms_enabled() is False
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.use_deterministic_algorithms(was_algorithms, warn_only=was_warn_only)
        torch.backends.cudnn.deterministic = was_cudnn_deterministic
        torch.backends.cudnn.benchmark = was_cudnn_benchmark
        if had_cublas:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = was_cublas
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def _random3_loader(V=6, n=600, seq_len=8, batch_size=8, data_seed=101, loader_seed=0):
    # UNLEARNABLE negative control: each token drawn iid uniform over the SAME 3 active
    # tokens {0,1,2}. The next token is independent of the current one, so the irreducible
    # CE floor is exactly the marginal entropy ln(3) -- there is no structure to learn. A
    # model that genuinely learns next-token structure on the periodic stream must NOT clear
    # the same sub-ln(3) anchor here, where clearing it could only be in-sample noise-fitting.
    dg = torch.Generator().manual_seed(data_seed)
    g = torch.Generator().manual_seed(loader_seed)
    base = torch.randint(0, 3, (n,), generator=dg)             # iid over {0,1,2}, no period
    ds = TokenWindows(base.to(torch.long), seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      generator=g)


def _structured_cfg() -> VFE3Config:
    # The period-3 shift (0->1->2->0) is a DIRECTED map: predicting the next token from the
    # current one. Causal attention only AVERAGES past beliefs, so with the gauge frame
    # frozen (e_phi_lr=m_phi_lr=0) the priors collapse to the symmetric "predict the marginal
    # over the active tokens" optimum and CE pins at exactly ln(3) ~ 1.099 (verified across
    # every learning-rate / E-step / depth / alpha sweep -- see the changelog). The gauge
    # transport Omega_ij(phi) is the one degree of freedom that applies a DIRECTED (non-
    # averaging) rotation to coupled beliefs; turning it on (e_phi_lr, m_phi_lr > 0) breaks
    # the symmetry and is the ONLY mechanism that drives CE below ln(3). The cutover anchor
    # below therefore gates exactly that sub-marginal (phi) improvement: a future config edit
    # that detunes phi drops CE back to the ln(3) pin and fails the anchor loudly.
    return VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                      n_e_steps=3, e_q_mu_lr=0.3, e_phi_lr=0.3,
                      m_p_mu_lr=0.05, m_p_sigma_lr=0.01, m_phi_lr=0.05, warmup_steps=5, max_steps=200)


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


# Promoted from xfail to a hard gate (audit t3, 2026-07-07 re-validation): the per-head-(per-irrep-block)
# beta model learns the period-3 next-token map to CE ~= 0.002, clearing the ln(3) - 0.05 floor by +1.05
# nats -- byte-identical on CPU and on an RTX 5090 across the 3 fixed seeds, so no LR re-tuning is needed
# (the old ~0.047 thin margin was the retired SINGLE-beta model). The negative control
# test_random_stream_does_not_clear_cutover_anchor guards the other side. Threshold deliberately NOT
# massaged (audit honesty rule).
def test_training_decreases_loss_on_structured_stream():
    # CUTOVER (spec section 10): the assembled VFEModel must LEARN the period-3 next-token
    # structure, certified by BEATING the active-alphabet unigram floor ln(3) -- not merely
    # by ending below 0.6*init (which is coupled to the ln(vocab) init magnitude, sits BELOW
    # ln(3) only by ~0.024 nats, and is razor-thin against an unlearnable random(3) stream;
    # see test_random_stream_does_not_clear_cutover_anchor). We assert the MEDIAN end-loss
    # over 3 seeds clears the absolute marginal-entropy anchor, so a single unlucky phi init
    # cannot flip the gate. (No held-out split is needed: the random-control floor sits ABOVE
    # this anchor, so in-sample noise-fitting alone cannot clear it.)
    ends = []
    for seed in range(3):
        torch.manual_seed(seed)
        cfg = _structured_cfg()
        model = VFEModel(cfg)
        losses = train(model, _periodic_loader(V=6, period=3, seed=seed), cfg, n_steps=200)
        ends.append(losses[-1])
        if seed == 0:                                          # documented secondary readout
            assert losses[-1] < 0.6 * losses[0]               # init-relative drop (start ~ln(6))
    assert _median(ends) < _MARGINAL_ENTROPY_P3 - _CUTOVER_MARGIN   # beats the unigram floor -> LEARNS the period


def test_random_stream_does_not_clear_cutover_anchor():
    # NEGATIVE CONTROL for the cutover: the same model/config on an UNLEARNABLE iid random(3)
    # stream must NOT clear the marginal-entropy anchor. The irreducible floor is ln(3) (no
    # next-token structure), so the only way to drop below ln(3) - margin in-sample would be
    # noise-fitting finite-sample fluctuations -- which this asserts does NOT happen, proving
    # the structured test's anchor certifies STRUCTURE learning rather than ended-just-under
    # -the-marginal. (data_seed 101 verified to end ~1.12, robustly above the anchor.)
    torch.manual_seed(0)
    cfg = _structured_cfg()
    model = VFEModel(cfg)
    losses = train(model, _random3_loader(data_seed=101), cfg, n_steps=200)
    assert losses[-1] == losses[-1]                            # finite (no NaN)
    assert losses[-1] >= _MARGINAL_ENTROPY_P3 - _CUTOVER_MARGIN   # cannot beat a floor that has no structure


def test_training_smoke_on_real_wikitext2_if_present():
    # SOFTER SMOKE (NOT a structure-learning proof): on a real wikitext-2 slice this asserts
    # only that training is FINITE (no NaN) and the CE DECREASES by a margin. The ~2-nat drop
    # here (10.825 -> ~8.78 from a uniform-init ln(V)=10.825) is achievable by learning the
    # unigram token distribution alone: the same config on the SAME tokens randomly PERMUTED
    # (sequential structure destroyed, only the marginal histogram surviving) still drops well
    # past the 0.05 bar. So this guards numerics and end-to-end wiring on real vocabulary; it
    # does NOT certify next-token structure learning. The structured period-3 cutover (above),
    # which beats the active-alphabet marginal entropy, is the learnability gate.
    import pytest
    from vfe3.data.datasets import load_cached_tokens
    try:
        toks = load_cached_tokens("wikitext-2", "validation")
    except FileNotFoundError:
        pytest.skip("wikitext-2 cache absent")
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=50257, embed_dim=8, n_heads=2, max_seq_len=16, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.3, e_phi_lr=0.0,
                     m_p_mu_lr=0.05, m_p_sigma_lr=0.01, m_phi_lr=0.0, warmup_steps=3, max_steps=30)
    model = VFEModel(cfg)
    ds = TokenWindows(toks[:4000], 16)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    losses = train(model, loader, cfg, n_steps=30)
    assert all(map(lambda x: x == x, losses))                   # finite (no NaN)
    assert losses[-1] < losses[0] - 0.05                        # CE decreases (marginal learning suffices)


def test_evaluate_returns_finite_ppl_bpc_consistent_with_ce():
    torch.manual_seed(0)
    cfg = _structured_cfg()
    model = VFEModel(cfg)
    loader = _periodic_loader(seed=0)
    m = evaluate(model, loader)
    assert set(m.keys()) == {"ce", "ppl", "bpc"}
    assert all(math.isfinite(v) for v in m.values())
    assert m["ppl"] == pytest.approx(math.exp(min(m["ce"], 20.0)))
    assert m["bpc"] == pytest.approx(m["ce"] / math.log(2.0))


def test_silent_and_logging_paths_are_bitwise_identical(caplog):
    torch.manual_seed(0)
    cfg_a = _structured_cfg()
    model_a = VFEModel(cfg_a)
    loader_a = _periodic_loader(seed=0)
    losses_silent = train(model_a, loader_a, cfg_a, n_steps=20)

    torch.manual_seed(0)
    cfg_b = _structured_cfg()
    model_b = VFEModel(cfg_b)
    loader_b = _periodic_loader(seed=0)
    with caplog.at_level("INFO"):
        losses_logged = train(model_b, loader_b, cfg_b, n_steps=20, log_interval=1, eval_interval=0)

    assert losses_silent == losses_logged           # exact: logging must not perturb the hot path
    assert any("Step 1/20" in r.message for r in caplog.records)


class _CountingLoader:
    """Yields a fixed list of (tokens, targets) batches, counting how many are consumed.

    Used to prove the PERIODIC eval is capped: ``evaluate`` iterates the loader and breaks at
    ``max_batches``, so a capped periodic pass draws fewer batches than the loader holds.
    """

    def __init__(self, batches):
        self.batches = batches
        self.count = 0

    def __iter__(self):
        for b in self.batches:
            self.count += 1
            yield b


def test_train_caps_periodic_eval_at_eval_max_batches():
    # train() must thread cfg.eval_max_batches into the PERIODIC validation pass so a large
    # val split is not fully re-scanned every eval_interval steps. With a 5-batch val loader,
    # eval_interval=1 over 2 steps (2 eval calls) and eval_max_batches=2, the loader must be
    # drawn 2*2=4 times, not 2*5=10.
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0, warmup_steps=1, max_steps=2,
                     eval_max_batches=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    train_loader = _periodic_loader(V=6, period=3, seq_len=8, batch_size=4, seed=0)
    val_batches = [(torch.randint(0, 3, (4, 8)), torch.randint(0, 3, (4, 8))) for _ in range(5)]
    val_loader = _CountingLoader(val_batches)
    train(model, train_loader, cfg, n_steps=2, eval_interval=1, val_loader=val_loader)
    assert val_loader.count == 4          # 2 eval calls x cap 2, not x 5


def test_sample_decode_emits_sample_line_each_eval(caplog):
    # When a decoder is supplied, train() prints a "Sample:" line below the BPC value every eval;
    # when it is None (the default) nothing is generated. A decode/gen error must not raise.
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0, warmup_steps=1, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    decode = lambda ids: " ".join(str(int(t)) for t in ids)        # trivial token->text decoder
    with caplog.at_level("INFO"):
        train(model, _periodic_loader(seed=0), cfg, n_steps=2, eval_interval=1,
              val_loader=_periodic_loader(seed=1), sample_decode=decode,
              sample_new_tokens=3, sample_prompt_len=4)
    assert sum("Sample:" in r.message for r in caplog.records) == 2  # one per eval step


def test_tiny_vocab_auto_default_stays_silent(caplog):
    # The vocab-gated auto-default decoder is None for a tiny synthetic/test vocab (6 is not a
    # real tokenizer size), so with no explicit decoder no Sample line is emitted -- the pure
    # silent path is preserved with no extra toggle.
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0, warmup_steps=1, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    with caplog.at_level("INFO"):
        train(model, _periodic_loader(seed=0), cfg, n_steps=2, eval_interval=1,
              val_loader=_periodic_loader(seed=1))
    assert not any("Sample:" in r.message for r in caplog.records)


def test_generate_samples_false_is_silent_at_real_vocab(caplog):
    # The pure-path opt-out: generate_samples=False suppresses sampling even at a real gpt2 vocab
    # where the auto-default would otherwise fire (CLAUDE.md: a reachable silent path under a toggle).
    pytest.importorskip("tiktoken")
    cfg = VFE3Config(vocab_size=50257, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0, warmup_steps=1, max_steps=1)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    with caplog.at_level("INFO"):
        train(model, _periodic_loader(seed=0), cfg, n_steps=1, eval_interval=1,
              val_loader=_periodic_loader(seed=1), generate_samples=False)
    assert not any("Sample:" in r.message for r in caplog.records)


def test_auto_default_sample_decoder_emits_at_gpt2_vocab(caplog):
    # At a real gpt2 vocab the auto-default decoder activates with NO explicit decoder and NO
    # entry-file wiring: a Sample line prints below BPC each eval (skip if tiktoken is absent).
    pytest.importorskip("tiktoken")
    cfg = VFE3Config(vocab_size=50257, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0, warmup_steps=1, max_steps=1)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    with caplog.at_level("INFO"):
        train(model, _periodic_loader(seed=0), cfg, n_steps=1, eval_interval=1,
              val_loader=_periodic_loader(seed=1), sample_new_tokens=3, sample_prompt_len=4)
    assert any("Sample:" in r.message for r in caplog.records)


def test_train_vfe3_clickrun_importable_and_runs_one_step():
    from dataclasses import replace

    from train_vfe3 import config as cr_config

    cfg_full = VFE3Config(**cr_config)         # importability: the LIVE click-run config must construct
    # t2: run the train STEP on a dim-shrunk copy so this CPU test's cost cannot balloon with the user's
    # live WIP toggles (seq_len=128, batch_size=64, more layers). The gauge/family/decode structural
    # toggles are preserved -- only the size dims shrink -- so the click-run code path is still exercised.
    cfg = replace(cfg_full, max_seq_len=16, batch_size=2, n_layers=1)
    loader = _periodic_loader(n=cfg.max_seq_len * cfg.batch_size * 4,
                              seq_len=cfg.max_seq_len, batch_size=cfg.batch_size, seed=cfg.seed)
    batch = next(iter(loader))
    assert len(batch) == 2
    torch.manual_seed(cfg.seed)
    model = VFEModel(cfg)
    losses = train(model, loader, cfg, n_steps=1)
    assert len(losses) == 1 and math.isfinite(losses[0])


def test_build_optimizer_groups_pos_phi_free():
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     pos_phi="learned", m_phi_lr=0.009)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)                      # must NOT raise the coverage AssertionError
    grouped = {p for g in opt.param_groups for p in g["params"]}
    assert model.pos_phi_free in grouped


def test_train_step_skips_on_nonfinite_grad_with_finite_loss():
    r"""F1 (audit 2026-07-01): a FINITE scalar loss can still carry a NaN parameter gradient
    through the unrolled E-step; the finite-GRADIENT gate must skip the optimizer step so AdamW's
    exp_avg/exp_avg_sq moment buffers are never poisoned. The scheduler still steps (resume
    rebuilds LambdaLR assuming exactly one scheduler.step per loop iteration)."""
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                     warmup_steps=1, max_steps=4)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg))
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 6, (2, 8), generator=g)
    targets = torch.randint(0, 6, (2, 8), generator=g)

    train_step(model, opt, sched, tokens, targets, grad_clip=1.0)   # clean step populates AdamW state
    assert opt.state                                                # moment buffers exist (non-vacuous)
    # A hook that poisons ONE parameter's gradient while the forward (and its scalar loss) stays finite.
    model.prior_bank.mu_embed.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    metrics = {}
    loss = train_step(model, opt, sched, tokens, targets, grad_clip=1.0, metrics_out=metrics)

    assert math.isfinite(loss)
    assert metrics["loss_finite"] == 1.0                            # the scalar loss WAS finite
    assert metrics["grad_finite"] == 0.0                            # ...but the gradient was not
    assert metrics["step_skipped"] == 1.0                           # so the optimizer step was skipped
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for state in opt.state.values():                                # AdamW moments stay clean
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                assert torch.isfinite(state[key]).all()
    assert sched.last_epoch == 2                                    # scheduler stepped UNCONDITIONALLY


def test_attention_map_replay_failure_does_not_kill_training(tmp_path, monkeypatch, caplog):
    r"""F11 (audit 2026-07-01): the attention/gamma map replays are argument expressions evaluated
    in the CALLER, outside the save helpers' internal try/except -- a replay error must be caught
    by the caller-side guard (warn + continue), never abort training."""
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0, warmup_steps=1, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)

    def _boom(*a, **k):
        raise RuntimeError("replay boom")

    monkeypatch.setattr(model, "attention_maps", _boom)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    with caplog.at_level(logging.WARNING):
        losses = train(model, _periodic_loader(seed=0), cfg, n_steps=2, eval_interval=1,
                       val_loader=_periodic_loader(seed=1), artifacts=art)
    assert len(losses) == 2                                     # train() completed despite the failure
    assert any("attention-map replay failed" in r.getMessage() for r in caplog.records)


def test_val_diagnostics_failure_resets_to_nan(tmp_path, monkeypatch):
    r"""F11 (audit 2026-07-01): a _val_diagnostics failure must RESET the held-out probes to NaN
    (blank CSV cells) instead of carrying the previous eval's values forward as if fresh."""
    import vfe3.train as vt

    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0, warmup_steps=1, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    calls = {"n": 0}

    def _flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"val_free_energy_total": 1.23}              # eval 1: a fresh probe value
        raise RuntimeError("diagnostics boom")                  # eval 2: replay failure

    monkeypatch.setattr(vt, "_val_diagnostics", _flaky)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    train(model, _periodic_loader(seed=0), cfg, n_steps=2, eval_interval=1,
          val_loader=_periodic_loader(seed=1), artifacts=art)
    with open(tmp_path / "run" / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert calls["n"] == 2                                      # both evals hit the probe
    assert rows[0]["val_free_energy_total"] == "1.23"           # eval 1 wrote the fresh value
    assert rows[1]["val_free_energy_total"] == ""               # eval 2 failed -> NaN -> blank, NOT stale 1.23


def test_select_loader_is_split_aware(monkeypatch):
    r"""Audit F1: _select_loader must request shuffle=False, drop_last=False for validation/test
    (stable corpus metric) and shuffle=True, drop_last=True only for train. RED against the old
    _select_loader, which called make_dataloader with neither override (so val/test inherited the
    shuffle=True, drop_last=True training defaults)."""
    import train_vfe3

    captured = {}

    def fake_make_dataloader(dataset, split, seq_len, batch_size, *,
                             shuffle=True, drop_last=True, max_tokens=None, **kw):
        captured[split] = {"shuffle": shuffle, "drop_last": drop_last}
        return ("loader", split)

    monkeypatch.setattr(train_vfe3, "make_dataloader", fake_make_dataloader)
    cfg = VFE3Config()
    for split in ("train", "validation", "test"):
        train_vfe3._select_loader("wikitext-103", cfg, split=split)

    assert captured["train"] == {"shuffle": True, "drop_last": True}
    assert captured["validation"] == {"shuffle": False, "drop_last": False}
    assert captured["test"] == {"shuffle": False, "drop_last": False}


def test_select_loader_raises_on_missing_cache(monkeypatch):
    r"""H1: a missing real-corpus cache must RAISE, never silently substitute the synthetic
    period-3 stream. The old fallback computed held-out numbers on a 3-token toy stream and
    persisted them mislabeled as the real corpus (config.json / test_results.json / run-folder name)."""
    import train_vfe3

    def raise_fnf(*a, **kw):
        raise FileNotFoundError("no cache")

    monkeypatch.setattr(train_vfe3, "make_dataloader", raise_fnf)
    cfg = VFE3Config()
    for split in ("train", "validation", "test"):
        with pytest.raises(FileNotFoundError):
            train_vfe3._select_loader("wikitext-103", cfg, split=split)


# =============================================================================
# Terminal callback + validation-only finalizer (PB-02, audit 2026-07-12)
# =============================================================================
# A default ablation cell (log/eval interval above max_steps, checkpoint_interval=0) trains fine but
# used to save no metrics.csv / best_model.pt / resumable bundle. train() now invokes a terminal
# callback once, immediately after the final optimizer step, and finalize_validation_run() turns that
# snapshot into a complete VALIDATION-ONLY artifact set (never a test split) plus a resumable checkpoint
# whose weights + optimizer moments are BOTH the raw iterate (never EMA weights paired with raw moments).
# Tiny CPU models throughout: embed_dim=4, n_heads=1, n_layers=1, n_e_steps=1, two batches.


def _terminal_cfg(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=1, max_seq_len=8, n_layers=1,
                n_e_steps=1, gauge_group="glk", use_head_mixer=False,
                e_q_mu_lr=0.5, e_q_sigma_lr=0.05, e_phi_lr=0.0,
                m_p_mu_lr=0.1, m_p_sigma_lr=0.05, m_phi_lr=0.0,
                warmup_steps=1, max_steps=2, kl_max=32)
    base.update(kw)
    return VFE3Config(**base)


def _terminal_loader(seed=7, n=64, seq_len=8, bs=4):
    # Two batches (8 windows / bs 4, drop_last); shuffled with an explicit generator so the terminal
    # checkpoint captures a resumable data cursor.
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(3).repeat(n // 3 + 2)
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, generator=g)


def _terminal_val_loader(n=48, seq_len=8, bs=4):
    base = torch.arange(3).repeat(n // 3 + 2)
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False)


def _make_terminal_state(model, cfg, *, ema=None):
    r"""A real TrainingTerminalState built from the model's current (raw) weights + optimizer + RNG."""
    opt = build_optimizer(model, cfg)
    raw = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    rng = {"cpu": torch.get_rng_state().clone(),
           "cuda": ([s.clone() for s in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available() else None)}
    return TrainingTerminalState(
        step=int(cfg.max_steps), optimizer=opt, scaler=None, ema=ema,
        metropolis_generator=torch.Generator().manual_seed(0),
        data_state=None, raw_model_state=raw, rng_state=rng)


def test_train_terminal_callback_receives_resumable_state():
    torch.manual_seed(0)
    cfg = _terminal_cfg(use_ema=True, ema_decay=0.5)
    model = VFEModel(cfg)
    loader = _terminal_loader()
    captured = {"calls": 0}

    def cb(state, callback_losses):
        captured["calls"] += 1
        captured["state"] = state
        captured["losses_len"] = len(callback_losses)

    train(model, loader, cfg, n_steps=cfg.max_steps, terminal_callback=cb)

    assert captured["calls"] == 1                                   # invoked EXACTLY once
    st = captured["state"]
    assert st.step == cfg.max_steps                                 # the completed-step count
    assert isinstance(st.optimizer, torch.optim.Optimizer)
    assert st.ema is not None                                       # use_ema=True -> the live EMA
    assert isinstance(st.metropolis_generator, torch.Generator)
    assert set(st.raw_model_state) == set(model.state_dict())       # full state_dict, strictly reloadable
    assert "cpu" in st.rng_state and st.rng_state["cpu"] is not None
    assert captured["losses_len"] == cfg.max_steps
    # The callback fires BEFORE the trailing ema.copy_to: the captured raw_model_state is the raw
    # last-iterate, while train() returns the EMA weights, so at least one trainable table differs.
    final = model.state_dict()
    assert any(not torch.equal(st.raw_model_state[n], final[n])
               for n, p in model.named_parameters() if p.requires_grad)


def test_validation_finalizer_records_validation_without_test_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False,
                                         "git_dirty_fingerprint": None})
    torch.manual_seed(0)
    cfg = _terminal_cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    state = _make_terminal_state(model, cfg)
    mapping = finalize_validation_run(
        model, art, cfg, _terminal_val_loader(), losses=[1.0, 0.9],
        terminal_state=state, device=torch.device("cpu"))

    summary = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert summary["selection_split"] == "validation"
    for forbidden in ("test_ce", "test_ppl", "test_bpc"):
        assert forbidden not in summary
    for required in ("primary_val_ppl", "final_val_ce", "final_val_ppl", "final_val_bpc",
                     "best_val_ppl", "best_step", "n_steps", "n_params", "final_train_loss",
                     "wall_time_s", "terminal_checkpoint", "figures_written"):
        assert required in summary
    assert isinstance(summary["figures_written"], list)

    vr = json.loads((tmp_path / "r" / "validation_results.json").read_text())
    assert vr["selection_split"] == "validation"
    assert "test_ppl" not in vr

    assert set(mapping) == {"primary_val_ppl", "final_val_ppl", "final_val_ce", "final_val_bpc",
                            "best_val_ppl", "best_step", "final_train_loss", "n_params",
                            "terminal_checkpoint"}
    # After the terminal maybe_save_best the primary equals the selected finite best (no earlier best).
    assert mapping["primary_val_ppl"] == mapping["best_val_ppl"] == mapping["final_val_ppl"]
    assert Path(mapping["terminal_checkpoint"]).exists()
    assert (tmp_path / "r" / "checkpoints" / f"step_{cfg.max_steps}.pt").exists()


def test_validation_finalizer_appends_to_existing_metrics_schema(tmp_path, monkeypatch):
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False,
                                         "git_dirty_fingerprint": None})
    torch.manual_seed(0)
    cfg = _terminal_cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    # An established (richer) training schema locks the CSV fieldnames on its first row; the terminal
    # row's five keys are a SUBSET, so the append writes blanks for the extra columns rather than crashing.
    art.log_metrics({"step": 1, "train_loss": 1.0, "train_ce": 1.0, "val_ce": float("nan"),
                     "val_ppl": float("nan"), "val_bpc": float("nan"), "lr_mu": 0.01,
                     "attn_entropy": 0.5})
    state = _make_terminal_state(model, cfg)
    finalize_validation_run(model, art, cfg, _terminal_val_loader(), losses=[1.0, 0.9],
                            terminal_state=state, device=torch.device("cpu"))

    with open(tmp_path / "r" / "metrics.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert "attn_entropy" in rows[0]                                # original schema preserved
    assert rows[-1]["step"] == str(cfg.max_steps)                   # terminal row appended
    assert rows[-1]["val_ppl"] != ""                               # terminal validation recorded
    assert rows[-1]["attn_entropy"] == ""                          # non-terminal column left blank


def test_terminal_callback_restores_cpu_and_cuda_rng(tmp_path, monkeypatch):
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False,
                                         "git_dirty_fingerprint": None})
    torch.manual_seed(0)
    cfg = _terminal_cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    val = _terminal_val_loader()
    captured = {}

    def cb(state, callback_losses):
        # Global RNG captured immediately after the last training step (== state.rng_state).
        captured["cpu"] = state.rng_state["cpu"].clone()
        captured["cuda"] = ([s.clone() for s in state.rng_state["cuda"]]
                            if state.rng_state["cuda"] is not None else [])
        # The finalizer draws RNG (eval + figures) then restores the captured state in its finally.
        finalize_validation_run(model, art, cfg, val, losses=callback_losses,
                                terminal_state=state, device=torch.device("cpu"))

    train(model, _terminal_loader(), cfg, n_steps=cfg.max_steps, artifacts=art,
          val_loader=val, terminal_callback=cb)

    assert torch.equal(torch.get_rng_state(), captured["cpu"])      # CPU stream rewound
    cuda_now = ([s for s in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [])               # empty list on a CPU-only host
    assert [s.tolist() for s in cuda_now] == [s.tolist() for s in captured["cuda"]]


def test_terminal_checkpoint_resumes_optimizer_rng_and_next_step(tmp_path, monkeypatch):
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False,
                                         "git_dirty_fingerprint": None})
    torch.manual_seed(0)
    cfg = _terminal_cfg(max_steps=2)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    val = _terminal_val_loader()
    loader = _terminal_loader()
    result = {}

    def cb(state, callback_losses):
        result.update(finalize_validation_run(
            model, art, cfg, val, losses=callback_losses, train_loader=loader,
            terminal_state=state, device=torch.device("cpu")))

    train(model, loader, cfg, n_steps=cfg.max_steps, artifacts=art,
          val_loader=val, terminal_callback=cb)

    ckpt = result["terminal_checkpoint"]
    assert Path(ckpt).exists()
    bundle = torch.load(ckpt, weights_only=True)                   # safe-loads (tensors + config only)
    assert bundle["step"] == cfg.max_steps
    for slot in ("model_state", "optimizer_state", "rng_state", "scaler_state", "ema_state",
                 "data_state", "best_val_ppl", "best_step"):
        assert slot in bundle

    # Resume: a fresh model + optimizer continues from the saved step for EXACTLY one more step.
    torch.manual_seed(0)
    cfg_resume = _terminal_cfg(max_steps=cfg.max_steps + 1)
    fresh = VFEModel(cfg_resume)
    losses_resume = train(fresh, _terminal_loader(), cfg_resume, n_steps=cfg.max_steps + 1,
                          resume_from=ckpt, device=torch.device("cpu"))
    assert len(losses_resume) == 1                                 # start_step == max_steps -> one step
    assert math.isfinite(losses_resume[0])


def test_terminal_checkpoint_ema_raw_weights_resume_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False,
                                         "git_dirty_fingerprint": None})
    # UNINTERRUPTED control (raw trajectory, no EMA): 3 steps.
    torch.manual_seed(0)
    cfg_ctrl = _terminal_cfg(max_steps=3, use_ema=False)
    ctrl = VFEModel(cfg_ctrl)
    train(ctrl, _terminal_loader(), cfg_ctrl, n_steps=3, device=torch.device("cpu"))
    ctrl_raw = {n: p.detach().clone() for n, p in ctrl.named_parameters() if p.requires_grad}

    # INTERRUPTED first leg: 2 steps WITH EMA. EMA.update draws no RNG and (no periodic eval) never
    # touches the live weights, so the raw 2-step trajectory matches the control's first two steps.
    torch.manual_seed(0)
    cfg_ema = _terminal_cfg(max_steps=2, use_ema=True, ema_decay=0.5)
    interrupted = VFEModel(cfg_ema)
    art = RunArtifacts(tmp_path / "r", cfg_ema, interrupted)
    val = _terminal_val_loader()
    loader = _terminal_loader()
    grab = {}

    def cb(state, callback_losses):
        grab["raw2"] = {k: v.clone() for k, v in state.raw_model_state.items()}
        grab.update(finalize_validation_run(
            interrupted, art, cfg_ema, val, losses=callback_losses, train_loader=loader,
            terminal_state=state, device=torch.device("cpu")))

    train(interrupted, loader, cfg_ema, n_steps=2, artifacts=art,
          val_loader=val, terminal_callback=cb)
    ckpt = grab["terminal_checkpoint"]

    # The returned model + best_model.pt use the EMA weights (differ from the raw iterate).
    ema_final = {n: p.detach().clone() for n, p in interrupted.named_parameters() if p.requires_grad}
    assert any(not torch.equal(ema_final[n], grab["raw2"][n]) for n in ema_final)
    best_bundle = torch.load(tmp_path / "r" / "best_model.pt", weights_only=True)
    assert any(not torch.equal(best_bundle["model_state"][n], grab["raw2"][n]) for n in ema_final)

    # Resume the RAW step from the checkpoint (raw weights + raw optimizer moments) and match control.
    torch.manual_seed(0)
    cfg_res = _terminal_cfg(max_steps=3, use_ema=False)
    resumed = VFEModel(cfg_res)
    train(resumed, _terminal_loader(), cfg_res, n_steps=3, resume_from=ckpt, device=torch.device("cpu"))
    for n, p in resumed.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p.detach(), ctrl_raw[n], atol=1e-5, rtol=1e-4), n
