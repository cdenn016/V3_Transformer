r"""The end-of-run mechanism diagnostics must answer their question or say they could not.

Three probes were promoted out of scratchpad scripts on 2026-07-25, after the shipped depth probe
spent weeks reporting the model channel under the belief loop's name. The property each must have
is the one that failure lacked: when a probe cannot measure what it is named for, it reports that
explicitly instead of returning numbers that describe something else.

Every model here is TINY (K = 4, single-digit everything) and CPU-bound per CLAUDE.md.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import (
    collect_beta_channel_decomposition,
    collect_context_sensitivity,
    collect_estep_character,
)


def _model(**overrides) -> VFEModel:
    cfg = VFE3Config(
        embed_dim=4, n_heads=2, vocab_size=11, max_seq_len=6, batch_size=2,
        n_e_steps=2, max_steps=1, lambda_alpha_mode="constant",
        **{"n_layers": 1, **overrides},
    )
    torch.manual_seed(0)
    return VFEModel(cfg)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.randint(0, 11, (2, 6)), torch.randint(0, 11, (2, 6))


# --------------------------------------------------------------------- E-step character


def test_precision_split_replica_is_exact_against_the_live_kernel() -> None:
    r"""The probe recomputes the fusion; if that replica drifts from the kernel it is worthless.

    ``recompute_max_abs_err`` is the probe's own falsification hook: it reproduces ``mu_star`` from
    the (a, prior_prec, pair_prec, pair_mean) it reports, so a nonzero residual means the reported
    shares no longer describe the fusion the model ran.
    """
    model = _model(e_step_update="mm_exact")
    tokens, _ = _batch()

    record = collect_estep_character(model, tokens, depths=[1, 2])

    assert record["precision_split_available"] is True
    assert record["recompute_max_abs_err"] == pytest.approx(0.0, abs=1e-5)


def test_precision_shares_are_a_partition() -> None:
    model = _model(e_step_update="mm_exact")
    tokens, _ = _batch()

    record = collect_estep_character(model, tokens, depths=[1, 2])

    for point in record["points"]:
        assert 0.0 <= point["pair_precision_share"] <= 1.0
        assert point["pair_precision_share"] + point["prior_precision_share"] == pytest.approx(1.0)


def test_step1_is_its_own_reference_direction() -> None:
    r"""Depth 1 compares against itself, so its cosine is exactly 1 and its share exactly 1."""
    model = _model(e_step_update="mm_exact")
    tokens, _ = _batch()

    record = collect_estep_character(model, tokens, depths=[1, 3])
    first = next(p for p in record["points"] if p["belief_depth"] == 1)

    assert first["cos_dir_vs_step1"] == pytest.approx(1.0, abs=1e-5)
    assert first["step1_share_of_displacement"] == pytest.approx(1.0, abs=1e-5)


def test_character_probe_reports_unavailable_off_the_kernel_route() -> None:
    r"""Without the mm_exact fusion there is no precision split, so none is invented."""
    model = _model(e_step_update="gradient")
    tokens, _ = _batch()

    record = collect_estep_character(model, tokens, depths=[1, 2])

    assert record["precision_split_available"] is False
    assert all(point["pair_precision_share"] is None for point in record["points"])


def test_character_probe_restores_both_depths_and_training_mode() -> None:
    model = _model(e_step_update="mm_exact", s_e_step_n_iter=3)
    model.train()
    tokens, _ = _batch()

    collect_estep_character(model, tokens, depths=[1, 2])

    assert model.cfg.n_e_steps == 2
    assert model.cfg.s_e_step_n_iter == 3
    assert model.training is True


def test_character_probe_rejects_a_zero_depth() -> None:
    r"""Depth 0 runs no E-step, so there is no displacement or fusion to characterize."""
    model = _model(e_step_update="mm_exact")
    tokens, _ = _batch()

    with pytest.raises(ValueError, match="depths"):
        collect_estep_character(model, tokens, depths=[0, 1])


# --------------------------------------------------------------------- beta channels


def test_beta_full_arm_reproduces_the_unpatched_forward() -> None:
    r"""The harness check: if the patched 'full' arm drifts, the other arms measure nothing."""
    import torch.nn.functional as F

    model = _model(e_step_update="mm_exact")
    tokens, targets = _batch()
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        reference = float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)))

    record = collect_beta_channel_decomposition(model, tokens, targets)

    assert record["available"] is True
    assert record["ce"]["full"] == pytest.approx(reference, abs=1e-5)
    assert record["delta_ce"]["content_channel"] == pytest.approx(
        record["ce"]["no_energy"] - record["ce"]["full"], abs=1e-9)


def test_energy_ablation_removes_the_temperature_dependence() -> None:
    r"""Ablating the energy must make CE INDEPENDENT of the softmax temperature, exactly.

    A magnitude assertion is worthless here: an untrained K=4 model has near-uniform attention, so
    all three arms agree to ~1e-6 nats and `!=` would be testing float noise. This invariant is
    exact instead. `beta = softmax(log_prior - E/tau)` depends on `tau` only through `E`, so the
    `no_energy` arm must give the SAME cross-entropy at two very different `kappa_beta` values while
    the `full` arm must not. If the patch ever stops reaching the live softmax, `no_energy` starts
    tracking `tau` and this fails.
    """
    tokens, targets = _batch()
    records = {}
    for kappa in (1.0, 0.01):
        model = _model(e_step_update="mm_exact", kappa_beta=kappa,
                       beta_attention_prior="causal_alibi_noself")
        records[kappa] = collect_beta_channel_decomposition(model, tokens, targets)

    assert records[1.0]["ce"]["no_energy"] == pytest.approx(
        records[0.01]["ce"]["no_energy"], abs=1e-9)
    assert records[1.0]["ce"]["full"] != records[0.01]["ce"]["full"]


def test_content_channel_grows_when_the_energy_is_made_dominant() -> None:
    r"""Cooling the softmax hands the routing to the energy, so ablating it must cost strictly more."""
    tokens, targets = _batch()
    deltas = {}
    for kappa in (1.0, 0.01):
        model = _model(e_step_update="mm_exact", kappa_beta=kappa,
                       beta_attention_prior="causal_alibi_noself")
        record = collect_beta_channel_decomposition(model, tokens, targets)
        deltas[kappa] = abs(record["delta_ce"]["content_channel"])

    assert deltas[0.01] > deltas[1.0]


def test_beta_probe_works_on_both_belief_routes() -> None:
    r"""`belief_gradients` and `mm_exact_update` both read the patched binding, so both decompose.

    An earlier draft asserted the gradient route was undecomposable. It is not: `kernels.py` calls
    `attention_weights` on both paths, so the probe is route-agnostic and the docstring claiming
    otherwise was the thing that was wrong.
    """
    tokens, targets = _batch()

    for update in ("mm_exact", "gradient"):
        model = _model(e_step_update=update, beta_attention_prior="causal_alibi_noself")
        record = collect_beta_channel_decomposition(model, tokens, targets)
        assert record["available"] is True, update
        assert record["delta_ce"]["positional_channel"] is not None, update


def test_flat_prior_model_reports_no_positional_channel() -> None:
    r"""With no attention log-prior there is nothing to flatten, so `no_prior` equals `full`."""
    model = _model(e_step_update="mm_exact")
    tokens, targets = _batch()

    record = collect_beta_channel_decomposition(model, tokens, targets)

    assert record["delta_ce"]["positional_channel"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------- context sensitivity


def test_raw_prior_control_is_exactly_zero_for_a_token_lookup() -> None:
    r"""With both loops off the belief IS the per-token prior, so a new prefix cannot move it.

    This control is what makes the other stages readable: a nonzero reading here would mean the
    probe is measuring its own noise rather than context injection.
    """
    model = _model(prior_source="token", s_e_step=False, e_step_update="mm_exact")
    tokens, _ = _batch()

    record = collect_context_sensitivity(model, tokens, n_replicates=2)

    assert record["stages"]["raw_prior"] == pytest.approx(0.0, abs=1e-6)
    assert record["stages"]["belief_estep"] > 0.0


def test_frozen_model_channel_emits_no_model_channel_stage() -> None:
    model = _model(prior_source="token", s_e_step=False, e_step_update="mm_exact")
    tokens, _ = _batch()

    record = collect_context_sensitivity(model, tokens, n_replicates=1)

    assert record["model_channel_live"] is False
    assert "model_channel" not in record["stages"]


def test_context_probe_restores_depths_and_training_mode() -> None:
    model = _model(prior_source="token", s_e_step=False, e_step_update="mm_exact",
                   s_e_step_n_iter=3)
    model.train()
    tokens, _ = _batch()

    collect_context_sensitivity(model, tokens, n_replicates=1)

    assert model.cfg.n_e_steps == 2
    assert model.cfg.s_e_step_n_iter == 3
    assert model.training is True


def test_context_probe_rejects_a_nonpositive_replicate_count() -> None:
    model = _model(prior_source="token", s_e_step=False, e_step_update="mm_exact")
    tokens, _ = _batch()

    with pytest.raises(ValueError, match="n_replicates"):
        collect_context_sensitivity(model, tokens, n_replicates=0)


# --------------------------------------------------------------------- the toggle


def test_expensive_diagnostics_default_off_and_must_be_a_real_bool() -> None:
    r"""The cheap tier always runs; only the context probe is gated, and "False" must not enable it."""
    assert VFE3Config(embed_dim=4, n_heads=2, vocab_size=11, max_seq_len=6, batch_size=2,
                      n_layers=1, max_steps=1).emit_expensive_diagnostics is False

    with pytest.raises(ValueError, match="emit_expensive_diagnostics"):
        VFE3Config(embed_dim=4, n_heads=2, vocab_size=11, max_seq_len=6, batch_size=2,
                   n_layers=1, max_steps=1, emit_expensive_diagnostics="False")


# ------------------------------------------------------- audit 2026-07-26 regressions
#
# Every test above runs `s_e_step=False`, which is why none of them caught B-01: the model
# channel's own E-step threads `cfg.e_step_update` and runs BEFORE the belief stack, so an
# untagged spy read the s-channel's fusion as the belief's. These pin the channel/layer
# attribution directly.


def _two_channel_model(**overrides) -> VFEModel:
    r"""A model whose model channel is LIVE, so s-channel calls exist to be misattributed."""
    return _model(prior_source="model_channel", s_e_step=True, lambda_gamma=0.5,
                  e_step_update="mm_exact", **overrides)


def test_character_probe_counts_belief_calls_only_when_the_model_channel_is_live() -> None:
    r"""The realized count must be the BELIEF depth, not belief + s (audit 2026-07-26 B-01).

    With `s_e_step_n_iter=3` and a swept belief depth of 2, an untagged probe sees 5 fusion calls
    and reads `recorded[0]` from the s-channel. The fixed probe reports 2.
    """
    model = _two_channel_model(s_e_step_n_iter=3)
    tokens, _ = _batch()

    record = collect_estep_character(model, tokens, depths=[1, 2])

    assert record["measured_channel"] == "belief"
    assert record["measured_layer"] == 0
    assert [p["n_belief_calls"] for p in record["points"]] == [1, 2]


def test_character_probe_ignores_layers_past_the_first() -> None:
    r"""At `n_layers=2` the window must not span the stack (audit 2026-07-26 B-01)."""
    model = _two_channel_model(n_layers=2, s_e_step_n_iter=1)
    tokens, _ = _batch()

    record = collect_estep_character(model, tokens, depths=[2])

    assert record["points"][0]["n_belief_calls"] == 2       # layer 0 only, not 4 across two layers


def test_beta_probe_intercepts_the_phi_substep_binding() -> None:
    r"""`inference/e_step.py` binds `attention_weights` itself (audit 2026-07-26 B-02).

    Under `e_phi_lr > 0` with `lambda_twohop != 0` the phi substep called that unpatched binding,
    so all three arms ran the TRUE beta while the record still claimed `available: True`. Patching
    the definition site makes the interception total; the observable consequence is that the
    ablated arms now actually differ from `full` on a config that exercises the phi substep.
    """
    from vfe3 import free_energy as free_energy_module
    from vfe3.inference import e_step as e_step_module

    # The seam under test: every binder must alias the same object before patching.
    assert e_step_module.attention_weights is free_energy_module.attention_weights

    model = _model(e_step_update="gradient", e_phi_lr=0.05, lambda_twohop=0.3,
                   beta_attention_prior="causal_alibi_noself")
    tokens, targets = _batch()

    record = collect_beta_channel_decomposition(model, tokens, targets)

    assert record["available"] is True
    assert record["ce"]["no_energy"] != record["ce"]["full"]


def test_character_probe_restores_the_patched_helpers_it_installs() -> None:
    r"""The probe now patches `_refine_s` and `vfe_block` too; all must be restored."""
    from vfe3.model import stack as stack_module

    model = _two_channel_model()
    before_refine = VFEModel._refine_s
    before_block = stack_module.vfe_block
    tokens, _ = _batch()

    collect_estep_character(model, tokens, depths=[1])

    assert VFEModel._refine_s is before_refine
    assert stack_module.vfe_block is before_block
