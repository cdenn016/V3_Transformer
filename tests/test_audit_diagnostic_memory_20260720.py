"""Regression coverage for bounded held-out diagnostic snapshot memory."""

from types import SimpleNamespace

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
import vfe3.train as train_module
from vfe3.train import _val_diagnostics, evaluate, train


_ACTIVE_BATCH_SIZE = 256
_ACTIVE_SEQUENCE_LENGTH = 128
_ACTIVE_VOCAB_SIZE = 50_257
_FLOAT32_BYTES = 4
_BOUNDED_LOGITS_COPY_PAIR_BYTES = (
    2 * 1 * _ACTIVE_SEQUENCE_LENGTH * _ACTIVE_VOCAB_SIZE * _FLOAT32_BYTES
)
_HISTORICAL_LOGITS_COPY_PAIR_BYTES = (
    2 * _ACTIVE_BATCH_SIZE * _ACTIVE_SEQUENCE_LENGTH * _ACTIVE_VOCAB_SIZE * _FLOAT32_BYTES
)
_CUDA_PEAK_MULTIPLE = 4


class _DiagnosticSnapshotSpy(torch.nn.Module):
    """Minimal diagnostic model that forbids any forward replay after snapshot capture."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(diagonal_covariance=True)
        self.head_mixer = None
        self.forward_calls = 0
        self.build_shapes: list[tuple[int, ...]] = []
        self.consumer_calls: list[tuple[str, tuple[int, ...], object]] = []

    def forward_beliefs(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_calls += 1
        if self.forward_calls > 1:
            raise AssertionError("held-out diagnostic consumer replayed forward_beliefs")
        logits = torch.zeros(
            (*token_ids.shape, 8),
            dtype=torch.float32,
            device=token_ids.device,
        )
        logits[..., 0] = 4.0
        return logits

    def build_diagnostic_snapshot(self, token_ids: torch.Tensor) -> object:
        self.build_shapes.append(tuple(token_ids.shape))
        return SimpleNamespace(logits=self.forward_beliefs(token_ids))

    def diagnostics(
        self,
        token_ids: torch.Tensor,

        *,
        snapshot: object,
    ) -> dict[str, float]:
        self.consumer_calls.append(("diagnostics", tuple(token_ids.shape), snapshot))
        return {
            "self_coupling":          0.0,
            "self_divergence":        0.0,
            "belief_coupling":        0.0,
            "attention_entropy":      0.0,
            "total":                  0.0,
            "attn_entropy":           0.0,
            "effective_rank":         1.0,
            "belief_cond_median":     1.0,
            "attn_entropy_min":       0.0,
            "holonomy_wilson":        0.0,
            "cocycle_residual":       0.0,
            "gauge_invariant_spread": 0.0,
            "fisher_trace_mean":      0.0,
            "belief_cond_p95":        1.0,
            "phi_norm_mean":          0.0,
            "phi_norm_std":           0.0,
            "guard_sigma_floor_frac": 0.0,
            "guard_sigma_ceil_frac":  0.0,
            "guard_energy_klmax_frac": 0.0,
            "nonfinite_frac":         0.0,
        }

    def attention_maps(
        self,
        token_ids: torch.Tensor,

        *,
        snapshot: object,
    ) -> torch.Tensor:
        self.consumer_calls.append(("attention_maps", tuple(token_ids.shape), snapshot))
        length = int(token_ids.shape[1])
        return torch.full(
            (1, 1, length, length),
            1.0 / length,
            dtype=torch.float32,
            device=token_ids.device,
        )


class _PopulationEvaluationSpy(torch.nn.Module):
    """Record the ordinary validation population passed through predictive scoring."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.shapes: list[tuple[int, ...]] = []
        self.scored_targets = 0

    def forward(
        self,
        token_ids: torch.Tensor,
        targets:   torch.Tensor,
    ) -> tuple[None, None, torch.Tensor]:
        self.shapes.append(tuple(token_ids.shape))
        self.scored_targets += int((targets != -100).sum())
        return None, None, torch.tensor(2.0, device=token_ids.device)


def test_val_diagnostics_slices_first_sequence_before_snapshot_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe3 import metrics
    from vfe3.viz import extract

    token_ids = torch.arange(
        _ACTIVE_BATCH_SIZE * _ACTIVE_SEQUENCE_LENGTH,
        dtype=torch.long,
    ).reshape(_ACTIVE_BATCH_SIZE, _ACTIVE_SEQUENCE_LENGTH) % 8
    targets = torch.arange(_ACTIVE_BATCH_SIZE, dtype=torch.long).remainder(8)
    targets = targets[:, None].expand(-1, _ACTIVE_SEQUENCE_LENGTH).clone()
    model = _DiagnosticSnapshotSpy()
    snapshots: list[object] = []
    covariance_flags: list[bool | None] = []

    real_build = model.build_diagnostic_snapshot

    def build_snapshot(tokens: torch.Tensor) -> object:
        snapshot = real_build(tokens)
        snapshots.append(snapshot)
        return snapshot

    def trace(
        _model: object,
        tokens: torch.Tensor,

        *,
        snapshot: object,
    ) -> dict[str, torch.Tensor]:
        model.consumer_calls.append(("e_step_belief_trace", tuple(tokens.shape), snapshot))
        state = torch.zeros((2, 1, _ACTIVE_SEQUENCE_LENGTH, 2))
        return {
            "free_energy": torch.tensor([1.0, 0.0]),
            "mu":          state,
            "sigma":       state + 1.0,
            "phi":         state,
        }

    def residuals(
        _mu: torch.Tensor,
        _sigma: torch.Tensor,
        _phi: torch.Tensor,

        *,
        diagonal: bool | None = None,
        eps:      float = 1e-12,
    ) -> dict[str, torch.Tensor]:
        del eps
        covariance_flags.append(diagonal)
        zero = torch.zeros(1)
        return {"r_mu": zero, "r_sigma": zero, "r_phi": zero}

    def fixed_point(
        _model: object,
        tokens: torch.Tensor,

        *,
        snapshot: object,
    ) -> dict[str, float]:
        model.consumer_calls.append(("e_step_fixed_point", tuple(tokens.shape), snapshot))
        return {}

    monkeypatch.setattr(model, "build_diagnostic_snapshot", build_snapshot)
    monkeypatch.setattr(extract, "e_step_belief_trace", trace)
    monkeypatch.setattr(extract, "e_step_fixed_point_diagnostics", fixed_point)
    monkeypatch.setattr(metrics, "estep_residuals", residuals)

    result = _val_diagnostics(model, [(token_ids, targets)], torch.device("cpu"))

    assert model.build_shapes == [(1, _ACTIVE_SEQUENCE_LENGTH)]
    assert len(snapshots) == 1
    assert model.forward_calls == 1
    assert covariance_flags == [True]
    assert {name for name, _, _ in model.consumer_calls} == {
        "diagnostics",
        "attention_maps",
        "e_step_belief_trace",
        "e_step_fixed_point",
    }
    assert all(shape == (1, _ACTIVE_SEQUENCE_LENGTH) for _, shape, _ in model.consumer_calls)
    assert all(snapshot is snapshots[0] for _, _, snapshot in model.consumer_calls)
    expected_first_row_ce = torch.log(torch.exp(torch.tensor(4.0)) + 7.0) - 4.0
    assert result.metrics["pos_loss_first_q"] == pytest.approx(float(expected_first_row_ce), abs=1e-6)
    assert result.metrics["pos_loss_last_q"] == pytest.approx(float(expected_first_row_ce), abs=1e-6)
    assert result.metrics["pos_loss_ratio"] == pytest.approx(1.0)

    population_model = _PopulationEvaluationSpy()
    scores = evaluate(
        population_model,
        [(token_ids, targets)],
        device=torch.device("cpu"),
    )
    assert population_model.shapes == [(_ACTIVE_BATCH_SIZE, _ACTIVE_SEQUENCE_LENGTH)]
    assert population_model.scored_targets == _ACTIVE_BATCH_SIZE * _ACTIVE_SEQUENCE_LENGTH
    assert scores["ce"] == pytest.approx(2.0)


def test_periodic_eval_reuses_one_held_out_snapshot_for_metrics_and_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        batch_size=2,
        generate_figures=True,
        use_ema=False,
    )
    model = VFEModel(cfg)
    train_tokens = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    train_targets = torch.roll(train_tokens, shifts=-1, dims=1)
    val_tokens = torch.tensor([[7, 6, 5, 4], [6, 5, 4, 3]])
    val_targets = torch.roll(val_tokens, shifts=-1, dims=1)
    builds: list[tuple[torch.Tensor, object]] = []
    consumers: list[tuple[str, torch.Tensor, object]] = []
    real_build = model.build_diagnostic_snapshot
    real_diagnostics = model.diagnostics
    real_attention_maps = model.attention_maps
    real_gamma_attention_maps = model.gamma_attention_maps

    def build_snapshot(tokens: torch.Tensor) -> object:
        snapshot = real_build(tokens)
        builds.append((tokens.detach().clone(), snapshot))
        return snapshot

    def diagnostics(tokens: torch.Tensor, *, snapshot: object | None = None) -> dict[str, float]:
        if snapshot is not None:
            consumers.append(("diagnostics", tokens.detach().clone(), snapshot))
        return real_diagnostics(tokens, snapshot=snapshot)

    def attention_maps(tokens: torch.Tensor, *, snapshot: object | None = None) -> torch.Tensor:
        if snapshot is not None:
            consumers.append(("attention_maps", tokens.detach().clone(), snapshot))
        return real_attention_maps(tokens, snapshot=snapshot)

    def gamma_attention_maps(tokens: torch.Tensor, *, snapshot: object | None = None) -> object:
        if snapshot is not None:
            consumers.append(("gamma_attention_maps", tokens.detach().clone(), snapshot))
        return real_gamma_attention_maps(tokens, snapshot=snapshot)

    def fake_train_step(*args: object, **kwargs: object) -> float:
        del args
        metrics = kwargs["metrics_out"]
        status = kwargs["status_out"]
        assert isinstance(metrics, dict)
        assert isinstance(status, dict)
        metrics["train_ce"] = 1.0
        status["did_step"] = False
        return 1.0

    class _Artifacts:
        def __init__(self) -> None:
            self.saved: list[tuple[str, int, object]] = []

        def maybe_save_best(self, step: int, _model: object, _ppl: float) -> None:
            del step

        def save_attention_maps(self, step: int, maps: object, *, logger: object) -> None:
            del logger
            self.saved.append(("beta", step, maps))

        def save_gamma_attention_maps(self, step: int, maps: object, *, logger: object) -> None:
            del logger
            self.saved.append(("gamma", step, maps))

        def log_metrics(self, _row: dict[str, float]) -> None:
            return None

    monkeypatch.setattr(model, "build_diagnostic_snapshot", build_snapshot)
    monkeypatch.setattr(model, "diagnostics", diagnostics)
    monkeypatch.setattr(model, "attention_maps", attention_maps)
    monkeypatch.setattr(model, "gamma_attention_maps", gamma_attention_maps)
    monkeypatch.setattr(train_module, "train_step", fake_train_step)
    artifacts = _Artifacts()

    train(
        model,
        [(train_tokens, train_targets)],
        cfg,
        n_steps=1,
        eval_interval=1,
        val_loader=[(val_tokens, val_targets)],
        artifacts=artifacts,
        generate_samples=False,
    )

    assert len(builds) == 1
    assert torch.equal(builds[0][0], val_tokens[:1])
    held_out_snapshot = builds[0][1]
    assert {name for name, _, _ in consumers} == {
        "diagnostics",
        "attention_maps",
        "gamma_attention_maps",
    }
    assert all(torch.equal(tokens, val_tokens[:1]) for _, tokens, _ in consumers)
    assert all(snapshot is held_out_snapshot for _, _, snapshot in consumers)
    assert [channel for channel, _, _ in artifacts.saved] == ["beta", "gamma"]


def test_h1_logits_copy_pair_allocation_arithmetic_is_bounded() -> None:
    assert _BOUNDED_LOGITS_COPY_PAIR_BYTES == 51_463_168
    assert _HISTORICAL_LOGITS_COPY_PAIR_BYTES == 13_174_571_008
    assert _HISTORICAL_LOGITS_COPY_PAIR_BYTES == (
        _ACTIVE_BATCH_SIZE * _BOUNDED_LOGITS_COPY_PAIR_BYTES
    )


@pytest.mark.cuda
def test_cuda_diagnostic_snapshot_peak_is_bounded_to_one_sequence() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    device = torch.device("cuda")
    cfg = VFE3Config(
        vocab_size=_ACTIVE_VOCAB_SIZE,
        embed_dim=2,
        n_heads=1,
        max_seq_len=_ACTIVE_SEQUENCE_LENGTH,
        n_layers=1,
        n_e_steps=1,
    )
    model = VFEModel(cfg).to(device).eval()
    token_ids = torch.zeros(
        (1, _ACTIVE_SEQUENCE_LENGTH),
        dtype=torch.long,
        device=device,
    )
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    snapshot = model.build_diagnostic_snapshot(token_ids)
    torch.cuda.synchronize(device)
    peak_growth = torch.cuda.max_memory_allocated(device) - baseline

    assert snapshot.logits.shape == (1, _ACTIVE_SEQUENCE_LENGTH, _ACTIVE_VOCAB_SIZE)
    assert peak_growth < _CUDA_PEAK_MULTIPLE * _BOUNDED_LOGITS_COPY_PAIR_BYTES
