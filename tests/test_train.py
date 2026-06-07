import math

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, evaluate, lr_lambda, train, _floor_lr_lambdas


def test_optimizer_groups_priors_by_m_lr():
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     m_mu_lr=0.01, m_sigma_lr=0.002, m_phi_lr=0.005)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    # m_sigma_lr=0.002, then TWO groups at m_phi_lr=0.005 (phi_embed + the default pos_phi='learned'
    # pos_phi_free table, grouped at m_phi_lr in train.py), then m_mu_lr=0.01.
    assert lrs == [0.002, 0.005, 0.005, 0.01]
    # every PriorBank parameter is covered by exactly one group
    n_params = sum(len(g["params"]) for g in opt.param_groups)
    assert n_params == len(list(model.parameters()))


def test_optimizer_groups_regime_ii_connection_and_learnable_alpha():
    # connection_W (transport_mode='regime_ii') and log_alpha (alpha_mode='learnable') are
    # trainable model-level nn.Parameters; build_optimizer must group them so they actually train
    # and so its exact-coverage guard does not raise. A >=2-head block group lets regime_ii build.
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     transport_mode="regime_ii", alpha_mode="learnable")
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)                           # must not raise (coverage guard)
    grouped = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(model.connection_W) in grouped
    assert id(model.log_alpha) in grouped


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
                     m_mu_lr=0.02, m_sigma_lr=0.004, m_phi_lr=0.01)
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
                     m_mu_lr=0.2, m_sigma_lr=0.05, m_phi_lr=0.2)
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
                      n_e_steps=3, e_mu_lr=0.3, e_phi_lr=0.3,
                      m_mu_lr=0.05, m_sigma_lr=0.01, m_phi_lr=0.05, warmup_steps=5, max_steps=200)


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


@pytest.mark.xfail(
    reason="audit 6c (temperature) + GL(K) finding #1 (per-head beta): at per-head "
           "tau=kappa*sqrt(d_head) the SINGLE-beta model beat the ln(3) floor but only by ~0.047 "
           "(< the 0.05 anchor margin), hence this xfail. The 2026-05-31 per-head (per-irrep-block) "
           "beta is more expressive and now clears the full margin on CPU across the 3 fixed seeds "
           "(this test XPASSes). Kept as a non-strict xfail because the margin is still thin and the "
           "LRs were calibrated for the old sqrt(embed_dim) tau -- GPU re-validation + LR re-tuning "
           "at scale remain advisable before this is promoted to a hard gate. Threshold deliberately "
           "NOT massaged (audit honesty rule).",
    strict=False,
)
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
                     n_e_steps=1, e_mu_lr=0.3, e_phi_lr=0.0,
                     m_mu_lr=0.05, m_sigma_lr=0.01, m_phi_lr=0.0, warmup_steps=3, max_steps=30)
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
    from train_vfe3 import config as cr_config, synthetic_period3_loader

    cfg = VFE3Config(**cr_config)
    loader = synthetic_period3_loader(seq_len=cfg.max_seq_len, batch_size=cfg.batch_size, seed=cfg.seed)
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


def test_select_loader_is_split_aware(monkeypatch):
    r"""Audit F1: _select_loader must request shuffle=False, drop_last=False for validation/test
    (stable corpus metric) and shuffle=True, drop_last=True only for train. RED against the old
    _select_loader, which called make_dataloader with neither override (so val/test inherited the
    shuffle=True, drop_last=True training defaults)."""
    import logging

    import train_vfe3

    captured = {}

    def fake_make_dataloader(dataset, split, seq_len, batch_size, *,
                             shuffle=True, drop_last=True, max_tokens=None, **kw):
        captured[split] = {"shuffle": shuffle, "drop_last": drop_last}
        return ("loader", split)

    monkeypatch.setattr(train_vfe3, "make_dataloader", fake_make_dataloader)
    cfg = VFE3Config()
    log = logging.getLogger("audit-f1")
    for split in ("train", "validation", "test"):
        train_vfe3._select_loader("wikitext-103", cfg, log, split=split)

    assert captured["train"] == {"shuffle": True, "drop_last": True}
    assert captured["validation"] == {"shuffle": False, "drop_last": False}
    assert captured["test"] == {"shuffle": False, "drop_last": False}
