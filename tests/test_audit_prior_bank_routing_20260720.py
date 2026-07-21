"""Task 10 regressions for executable prior-table routing and persistence."""

from collections import OrderedDict
from dataclasses import asdict
from functools import wraps

import pytest
import torch

import scaling
from vfe3.config import VFE3Config
from vfe3.ema import EMA
from vfe3.model.model import VFEModel
from vfe3.model import prior_bank as prior_bank_module
from vfe3.run_artifacts import (
    RunArtifacts,
    _cost_model_fields,
    _validate_best_model_mapping,
    _validate_checkpoint_model_state,
    _validate_ema_state,
    load_checkpoint,
    semantic_config_fingerprint,
)
from vfe3.train import build_optimizer, evaluate
from vfe3.viz.extract import _model_device


def _cfg(prior_source: str, route: str, **overrides: object) -> VFE3Config:
    values = {
        "vocab_size": 11,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 4,
        "n_layers": 1,
        "n_e_steps": 1,
        "pos_phi": "none",
        "prior_source": prior_source,
        "use_prior_bank": route != "linear",
        "untie_decode_bank": route == "untied",
        "decode_mode": "diagonal",
        "decode_bias": False,
    }
    values.update(overrides)
    return VFE3Config(**values)


def _forward(model: VFEModel) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    tokens = torch.tensor([[1, 2, 3, 4]], device=next(model.parameters()).device)
    targets = torch.tensor([[2, 3, 4, 5]], device=tokens.device)
    return model(tokens, targets)


@pytest.mark.parametrize("route", ["linear", "tied", "untied"])
def test_model_channel_routes_register_no_dormant_base_tables(route: str) -> None:
    cfg = _cfg("model_channel", route)
    model = VFEModel(cfg)
    pb = model.prior_bank

    assert "mu_embed" in pb._parameters and pb.mu_embed is None
    assert "sigma_log_embed" in pb._parameters and pb.sigma_log_embed is None
    assert "prior_bank.mu_embed" not in dict(model.named_parameters())
    assert "prior_bank.sigma_log_embed" not in dict(model.named_parameters())
    assert "prior_bank.mu_embed" not in model.state_dict()
    assert "prior_bank.sigma_log_embed" not in model.state_dict()

    optimizer = build_optimizer(model, cfg)
    grouped = {parameter for group in optimizer.param_groups for parameter in group["params"]}
    assert grouped == {parameter for parameter in model.parameters() if parameter.requires_grad}
    assert all(parameter is not None for parameter in grouped)

    logits, loss, ce = _forward(model)
    if logits is not None:
        assert torch.isfinite(logits).all()
    assert torch.isfinite(loss) and torch.isfinite(ce)


@pytest.mark.parametrize("route", ["linear", "tied", "untied"])
def test_token_routes_retain_both_tables_and_seeded_initialization(route: str) -> None:
    cfg = _cfg("token", route)
    torch.manual_seed(1729)
    left = VFEModel(cfg)
    torch.manual_seed(1729)
    right = VFEModel(cfg)

    for name in ("mu_embed", "sigma_log_embed"):
        left_parameter = getattr(left.prior_bank, name)
        right_parameter = getattr(right.prior_bank, name)
        assert isinstance(left_parameter, torch.nn.Parameter)
        assert torch.equal(left_parameter, right_parameter)
        assert f"prior_bank.{name}" in left.state_dict()


def test_omitted_base_mean_preserves_downstream_seeded_initialization_order() -> None:
    cfg = _cfg("model_channel", "linear")
    torch.manual_seed(314159)
    torch.randn(cfg.vocab_size, cfg.embed_dim)  # historical dormant base-mean draw
    n_gen = VFEModel(cfg).group.generators.shape[0]

    torch.manual_seed(314159)
    torch.randn(cfg.vocab_size, cfg.embed_dim)
    expected_phi = cfg.phi_scale * torch.randn(cfg.vocab_size, n_gen)
    torch.nn.init.xavier_uniform_(torch.empty(cfg.vocab_size, cfg.embed_dim))
    expected_s_mu = cfg.mu_init_std * torch.randn(cfg.vocab_size, cfg.embed_dim)

    torch.manual_seed(314159)
    model = VFEModel(cfg)
    assert torch.equal(model.prior_bank.phi_embed, expected_phi)
    assert torch.equal(model.prior_bank.s_mu_embed, expected_s_mu)


def test_unknown_registry_routes_conservatively_retain_base_tables() -> None:
    encode_name = "_task10_unknown_encode"
    decode_name = "_task10_unknown_decode"
    selected_encode = prior_bank_module.get_encode("per_token")
    selected_decode = prior_bank_module.get_decode("diagonal")

    def _encode(pb, token_ids):
        return selected_encode(pb, token_ids)

    def _decode(pb, mu_q, sigma_q, tau_eff):
        return selected_decode(pb, mu_q, sigma_q, tau_eff)

    prior_bank_module.register_encode(encode_name)(_encode)
    prior_bank_module.register_decode(decode_name)(_decode)
    try:
        unknown_encode = prior_bank_module.PriorBank(
            7, 4, 8, prior_source="model_channel", encode_mode=encode_name,
        )
        unknown_decode = prior_bank_module.PriorBank(
            7, 4, 8, prior_source="model_channel", decode_mode=decode_name,
        )
        assert unknown_encode.mu_embed is not None and unknown_encode.sigma_log_embed is not None
        assert unknown_decode.mu_embed is not None and unknown_decode.sigma_log_embed is not None
    finally:
        prior_bank_module._ENCODERS.pop(encode_name, None)
        prior_bank_module._ENCODER_REGISTRATIONS.pop(encode_name, None)
        prior_bank_module._DECODERS.pop(decode_name, None)


@pytest.mark.parametrize("kind", ["alias", "wrapped"])
def test_encode_callable_identity_cannot_inherit_base_table_omission(kind: str) -> None:
    name = f"_task10_conservative_encode_{kind}"
    selected = prior_bank_module.get_encode("per_token")
    if kind == "alias":
        candidate = selected
    else:
        @wraps(selected)
        def candidate(pb, token_ids):
            assert pb.mu_embed is not None and pb.sigma_log_embed is not None
            return selected(pb, token_ids)

    prior_bank_module.register_encode(name)(candidate)
    try:
        pb = prior_bank_module.PriorBank(
            7, 4, 8, prior_source="model_channel", encode_mode=name,
        )
        assert pb.mu_embed is not None and pb.sigma_log_embed is not None
        belief = pb.encode(torch.tensor([[0, 1, 2]]))
        assert torch.isfinite(belief.mu).all() and torch.isfinite(belief.sigma).all()
    finally:
        prior_bank_module._ENCODERS.pop(name, None)
        prior_bank_module._ENCODER_REGISTRATIONS.pop(name, None)


@pytest.mark.parametrize("kind", ["alias", "wrapped"])
def test_decode_callable_identity_cannot_inherit_base_table_omission(kind: str) -> None:
    name = f"_task10_conservative_decode_{kind}"
    selected = prior_bank_module.get_decode("diagonal")
    if kind == "alias":
        candidate = selected
    else:
        @wraps(selected)
        def candidate(pb, mu_q, sigma_q, tau_eff):
            assert pb.mu_embed is not None and pb.sigma_log_embed is not None
            return selected(pb, mu_q, sigma_q, tau_eff)

    prior_bank_module.register_decode(name)(candidate)
    try:
        pb = prior_bank_module.PriorBank(
            7, 4, 8, prior_source="model_channel", decode_mode=name,
        )
        assert pb.mu_embed is not None and pb.sigma_log_embed is not None
        belief = pb.encode(torch.tensor([[0, 1, 2]]))
        logits = pb.decode(belief.mu, belief.sigma)
        assert torch.isfinite(logits).all()
    finally:
        prior_bank_module._DECODERS.pop(name, None)


@pytest.mark.parametrize("route", ["linear", "tied", "untied"])
def test_realized_and_predicted_counts_have_route_parity(route: str) -> None:
    token_cfg = _cfg("token", route)
    model_cfg = _cfg("model_channel", route)
    token = VFEModel(token_cfg)
    model_channel = VFEModel(model_cfg)

    token_count = sum(parameter.numel() for parameter in token.parameters())
    model_count = sum(parameter.numel() for parameter in model_channel.parameters())
    assert token_count == model_count
    assert scaling.predict_n_params(token_cfg)[0] == token_count
    assert scaling.predict_n_params(model_cfg)[0] == model_count


def test_model_channel_device_dtype_anchors_and_smokes() -> None:
    cfg = _cfg("model_channel", "linear")
    converted = VFEModel(cfg).to(dtype=torch.float64)
    assert _model_device(converted) == next(converted.parameters()).device
    assert converted._attention_log_prior(4, _model_device(converted)).dtype == torch.float64
    assert converted._rope_rotation(4, _model_device(converted)) is None

    model = VFEModel(cfg)
    batch = [(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))]
    metrics = evaluate(model, batch)
    assert all(torch.isfinite(torch.tensor(value)) for key, value in metrics.items()
               if key != "bpc" and value is not None)


def test_learning_rate_reporting_is_resolved_by_group_role() -> None:
    from vfe3.train import _learning_rates_by_role

    cfg = _cfg(
        "model_channel",
        "linear",
        m_p_mu_lr=0.031,
        m_p_sigma_lr=0.017,
        m_phi_lr=0.009,
    )
    optimizer = build_optimizer(VFEModel(cfg), cfg)
    lrs = [float(group["lr"]) for group in optimizer.param_groups]
    expected = {"mu": 0.031, "sigma": 0.017, "phi": 0.009}
    assert _learning_rates_by_role(optimizer.param_groups, lrs) == expected
    assert _learning_rates_by_role(
        list(reversed(optimizer.param_groups)), list(reversed(lrs))) == expected


def _legacy_model_state(model: VFEModel, *, fill: float = 0.0) -> OrderedDict[str, torch.Tensor]:
    state = OrderedDict(model.state_dict())
    pb = model.prior_bank
    state["prior_bank.mu_embed"] = torch.full(
        (pb.vocab_size, pb.K), fill, dtype=pb.phi_embed.dtype, device=pb.phi_embed.device)
    state["prior_bank.sigma_log_embed"] = torch.full(
        (pb.vocab_size, pb.K), fill, dtype=pb.phi_embed.dtype, device=pb.phi_embed.device)
    return state


def test_legacy_dormant_model_tables_normalize_nonmutating_for_direct_and_artifact_loads() -> None:
    cfg = _cfg("model_channel", "linear")
    model = VFEModel(cfg)
    expected_output = _forward(model)
    legacy = _legacy_model_state(model, fill=123.0)
    legacy_keys = set(legacy)

    model.load_state_dict(legacy)
    actual_output = _forward(model)
    assert set(legacy) == legacy_keys
    assert actual_output[0] is expected_output[0] is None
    assert torch.equal(actual_output[1], expected_output[1])
    assert torch.equal(actual_output[2], expected_output[2])

    validated = _validate_checkpoint_model_state(legacy, model.state_dict(), "legacy")
    assert set(validated) == set(model.state_dict())
    assert set(legacy) == legacy_keys

    config = asdict(cfg)
    bundle = {
        "model_state": legacy,
        "config": config,
        "config_fingerprint": semantic_config_fingerprint(config),
    }
    validated_bundle = _validate_best_model_mapping(bundle, cfg, model.state_dict(), "legacy best")
    assert set(validated_bundle["model_state"]) == set(model.state_dict())


@pytest.mark.parametrize("bad", ["unexpected.weight", "prior_bank.mu_embed.extra"])
def test_legacy_normalizer_does_not_hide_arbitrary_unexpected_keys(bad: str) -> None:
    model = VFEModel(_cfg("model_channel", "linear"))
    state = _legacy_model_state(model)
    state[bad] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="keys|Unexpected key"):
        model.load_state_dict(state)
    with pytest.raises(RuntimeError, match="keys do not match"):
        _validate_checkpoint_model_state(state, model.state_dict(), "bad legacy")


@pytest.mark.parametrize("field", ["mu_embed", "sigma_log_embed"])
@pytest.mark.parametrize("failure", ["shape", "dtype", "layout", "finite"])
def test_legacy_dormant_table_validation_is_exact(field: str, failure: str) -> None:
    model = VFEModel(_cfg("model_channel", "linear"))
    state = _legacy_model_state(model)
    key = f"prior_bank.{field}"
    if failure == "shape":
        state[key] = state[key][:-1]
    elif failure == "dtype":
        state[key] = state[key].double()
    elif failure == "layout":
        state[key] = state[key].to_sparse()
    else:
        state[key][0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="legacy dormant prior table"):
        model.load_state_dict(state)


def test_token_prior_loading_remains_strict() -> None:
    model = VFEModel(_cfg("token", "linear"))
    state = OrderedDict(model.state_dict())
    del state["prior_bank.mu_embed"]
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state)


def test_exact_legacy_ema_shadow_entries_are_normalized_nonmutating() -> None:
    model = VFEModel(_cfg("model_channel", "linear"))
    ema = EMA(model, decay=0.9)
    shadow = OrderedDict(ema.shadow)
    pb = model.prior_bank
    shadow["prior_bank.mu_embed"] = torch.zeros(pb.vocab_size, pb.K)
    shadow["prior_bank.sigma_log_embed"] = torch.zeros(pb.vocab_size, pb.K)
    saved = {"decay": 0.9, "shadow": shadow}

    validated = _validate_ema_state(saved, ema, require_state=True)
    assert validated is not saved
    assert set(validated["shadow"]) == set(ema.shadow)
    assert "prior_bank.mu_embed" in saved["shadow"]


def test_legacy_optimizer_resume_fails_before_mutating_live_state(tmp_path) -> None:
    cfg = _cfg("model_channel", "linear")
    source = VFEModel(cfg)
    optimizer = build_optimizer(source, cfg)
    artifacts = RunArtifacts(tmp_path / "source", cfg, source)
    checkpoint = artifacts.save_checkpoint(0, source, optimizer, cfg)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_state"] = _legacy_model_state(source)
    groups = payload["optimizer_state"]["param_groups"]
    parameter_ids = [parameter_id for group in groups for parameter_id in group["params"]]
    legacy_mu_id = max(parameter_ids) + 1
    legacy_sigma_id = legacy_mu_id + 1
    legacy_mu_group = dict(next(group for group in groups if group.get("role") == "mu"))
    legacy_mu_group["params"] = [legacy_mu_id]
    groups.insert(0, legacy_mu_group)
    sigma_group = next(group for group in groups if group.get("role") == "sigma")
    sigma_group["params"] = [legacy_sigma_id, *sigma_group["params"]]
    torch.save(payload, checkpoint)

    target = VFEModel(cfg)
    target_optimizer = build_optimizer(target, cfg)
    with torch.no_grad():
        target.prior_bank.phi_embed.fill_(7.0)
    before = target.prior_bank.phi_embed.detach().clone()
    with pytest.raises(RuntimeError, match="weights-only restart with a fresh optimizer"):
        load_checkpoint(
            checkpoint,
            target,
            target_optimizer,
            restore_rng=False,
            cfg=cfg,
        )
    assert torch.equal(target.prior_bank.phi_embed, before)


@pytest.mark.parametrize(
    ("prior_source", "route", "expected_decode"),
    [
        ("token", "linear", 11 * 4),
        ("model_channel", "linear", 11 * 4),
        ("token", "tied", 2 * 11 * 4),
        ("model_channel", "tied", 2 * 11 * 4),
        ("token", "untied", 2 * 11 * 4),
        ("model_channel", "untied", 2 * 11 * 4),
    ],
)
def test_cost_model_counts_only_realized_encode_and_decode_consumers(
    prior_source: str,
    route: str,
    expected_decode: int,
) -> None:
    cfg = _cfg(prior_source, route)
    model = VFEModel(cfg)
    out = _cost_model_fields(model, cfg, n_params=1, tokens_seen=3)
    token_row = 2 * cfg.embed_dim + model.group.generators.shape[0]
    assert out["active_params_per_token"] == token_row + expected_decode


@pytest.mark.parametrize("prior_source", ["token", "model_channel"])
@pytest.mark.parametrize("route", ["linear", "tied", "untied"])
@pytest.mark.parametrize("lambda_h", [0.0, 0.25])
def test_full_gaussian_realized_prediction_and_active_cost_are_exact(
    prior_source: str,
    route:        str,
    lambda_h:     float,
) -> None:
    cfg = _cfg(
        prior_source,
        route,
        family="gaussian_full",
        decode_mode="full",
        lambda_h=lambda_h,
        lambda_gamma=0.0,
        s_e_step=False,
    )
    model = VFEModel(cfg)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert scaling.predict_n_params(cfg)[0] == actual

    V, K = cfg.vocab_size, cfg.embed_dim
    lower = K * (K - 1) // 2
    n_gen = model.group.generators.shape[0]
    s_active = prior_source == "model_channel" or lambda_h > 0.0
    encode_row = 2 * K + n_gen + (lower if prior_source == "model_channel" else 0)
    decode_readout = V * K if route == "linear" else 2 * V * K
    independent_s_row = (
        2 * K + lower
        if s_active and prior_source != "model_channel"
        else 0
    )
    hyperprior_centroid = 2 * K + lower if lambda_h > 0.0 else 0
    expected_active = encode_row + decode_readout + independent_s_row + hyperprior_centroid
    cost = _cost_model_fields(model, cfg, n_params=actual, tokens_seen=3)
    assert cost["active_params_per_token"] == expected_active


@pytest.mark.parametrize(
    ("family", "decode_mode"),
    [("gaussian_diagonal", "diagonal"), ("gaussian_full", "full")],
)
def test_s_e_step_active_cost_counts_executed_r_centroid(
    family:      str,
    decode_mode: str,
) -> None:
    cfg = _cfg(
        "model_channel",
        "linear",
        family=family,
        decode_mode=decode_mode,
        lambda_h=0.0,
        lambda_gamma=0.25,
        s_e_step=True,
    )
    model = VFEModel(cfg)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert scaling.predict_n_params(cfg)[0] == actual

    V, K = cfg.vocab_size, cfg.embed_dim
    lower = 0 if family == "gaussian_diagonal" else K * (K - 1) // 2
    n_gen = model.group.generators.shape[0]
    encode_row = 2 * K + lower + n_gen
    r_centroid = 2 * K + lower
    cost = _cost_model_fields(model, cfg, n_params=actual, tokens_seen=3)
    assert cost["active_params_per_token"] == encode_row + V * K + r_centroid
