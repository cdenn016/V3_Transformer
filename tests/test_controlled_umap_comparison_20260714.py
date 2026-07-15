"""Controlled belief-geometry comparison regressions (2026-07-14)."""

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.viz import extract, report


def _tiny_model(*, model_channel: bool = False) -> VFEModel:
    overrides = {}
    if model_channel:
        overrides = {
            "s_e_step": True,
            "prior_source": "model_channel",
            "lambda_h": 0.25,
            "lambda_gamma": 0.75,
        }
    cfg = VFE3Config(
        vocab_size=20,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_q_mu_lr=0.1,
        e_phi_lr=0.0,
        **overrides,
    )
    torch.manual_seed(0)
    return VFEModel(cfg)


def _token_batches() -> list[torch.Tensor]:
    return [
        torch.arange(0, 8).reshape(2, 4),
        torch.arange(8, 16).reshape(2, 4),
    ]


def test_belief_bank_max_tokens_slices_every_aligned_field():
    bank = extract.belief_bank(_tiny_model(), _token_batches(), max_tokens=11)

    aligned = ("mu", "sigma", "phi", "token_ids", "seq_idx", "pos_idx")
    assert {bank[key].shape[0] for key in aligned} == {11}
    assert bank["token_ids"].tolist() == list(range(11))
    assert bank["seq_idx"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    assert bank["pos_idx"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]


def test_belief_bank_max_sequences_is_exact_and_position_aligned():
    bank = extract.belief_bank(_tiny_model(), _token_batches(), max_sequences=3)

    assert bank["mu"].shape[0] == 12
    assert bank["seq_idx"].tolist() == [0] * 4 + [1] * 4 + [2] * 4
    assert bank["pos_idx"].tolist() == [0, 1, 2, 3] * 3


def test_model_channel_bank_max_tokens_slices_every_aligned_field():
    bank = extract.model_channel_bank(
        _tiny_model(model_channel=True),
        _token_batches(),
        max_tokens=11,
    )

    assert bank is not None
    aligned = ("mu", "sigma", "token_ids", "seq_idx", "pos_idx")
    assert {bank[key].shape[0] for key in aligned} == {11}
    assert bank["token_ids"].tolist() == list(range(11))
    assert bank["pos_idx"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]


@pytest.mark.parametrize("bank_name", ["belief_bank", "model_channel_bank"])
def test_banks_reject_ambiguous_population_caps(bank_name):
    model = _tiny_model(model_channel=bank_name == "model_channel_bank")
    bank_fn = getattr(extract, bank_name)

    with pytest.raises(ValueError, match="max_tokens.*max_sequences"):
        bank_fn(model, _token_batches(), max_tokens=8, max_sequences=2)


@pytest.mark.parametrize("cap_name", ["max_tokens", "max_sequences"])
@pytest.mark.parametrize("cap_value", [0, -1])
def test_belief_bank_rejects_nonpositive_population_caps(cap_name, cap_value):
    with pytest.raises(ValueError, match=cap_name):
        extract.belief_bank(_tiny_model(), _token_batches(), **{cap_name: cap_value})


@pytest.mark.parametrize(
    ("seq_len", "batch_size", "expected_batches"),
    [(128, 64, 2), (256, 32, 2), (512, 16, 2)],
)
def test_report_default_requests_same_controlled_token_population(
    seq_len,
    batch_size,
    expected_batches,
):
    cfg = SimpleNamespace(max_seq_len=seq_len, batch_size=batch_size)

    max_tokens, max_sequences, n_batches = report._resolve_bank_budget(
        cfg,
        max_tokens=None,
        max_sequences=None,
    )

    assert max_tokens == 16_384
    assert max_sequences is None
    assert n_batches == expected_batches


def test_cluster_coordinates_use_pca_not_umap_above_ten_dimensions():
    from vfe3.viz import embedding_comparison

    features = np.random.default_rng(0).normal(size=(80, 20))
    coordinates, description = embedding_comparison.cluster_coordinates(features)
    repeated, repeated_description = embedding_comparison.cluster_coordinates(features)

    assert coordinates.shape == (80, 10)
    assert description == "PCA 10-D"
    assert repeated_description == description
    assert np.array_equal(coordinates, repeated)


def test_cluster_coordinates_preserve_native_features_at_ten_dimensions():
    from vfe3.viz import embedding_comparison

    features = np.random.default_rng(1).normal(size=(40, 10))
    coordinates, description = embedding_comparison.cluster_coordinates(features)

    assert description == "native 10-D"
    assert np.array_equal(coordinates, features)
    assert coordinates is not features


def test_token_fingerprint_is_stable_and_order_sensitive():
    from vfe3.viz import embedding_comparison

    token_ids = np.array([2, 7, 11, 2], dtype=np.int64)

    assert embedding_comparison.token_fingerprint(token_ids) == embedding_comparison.token_fingerprint(
        torch.tensor(token_ids)
    )
    assert embedding_comparison.token_fingerprint(token_ids) != embedding_comparison.token_fingerprint(
        token_ids[::-1]
    )


def test_relative_position_quartiles_are_sequence_length_normalized():
    from vfe3.viz import embedding_comparison

    positions = np.arange(8)

    assert embedding_comparison.relative_position_quartiles(positions, seq_len=8).tolist() == [
        0, 0, 1, 1, 2, 2, 3, 3,
    ]


def _controlled_record_fixture():
    from vfe3.viz import embedding_comparison

    rng = np.random.default_rng(4)
    features = np.vstack([
        rng.normal(-1.5, 0.25, size=(50, 12)),
        rng.normal(+1.5, 0.25, size=(50, 12)),
    ])
    base = features[:, :2]
    coordinates = {
        seed: base + rng.normal(0.0, 0.002 * (seed + 1), size=base.shape)
        for seed in embedding_comparison.CONTROLLED_SEEDS
    }
    cluster_labels = np.repeat([0, 1], 50)
    token_ids = np.tile([3, 8], 50)
    bpe_labels = token_ids % 2
    function_content_labels = cluster_labels.copy()
    seq_idx = np.repeat(np.arange(10), 10)
    pos_idx = np.tile(np.arange(10), 10)
    contract = embedding_comparison.controlled_contract(
        kind="Belief",
        channel="mu",
        feature_dim=features.shape[1],
        feature_chart="Euclidean means",
        clustering_space="PCA 10-D",
    )
    return {
        "features": features,
        "coords_by_seed": coordinates,
        "cluster_labels": cluster_labels,
        "token_ids": token_ids,
        "bpe_labels": bpe_labels,
        "function_content_labels": function_content_labels,
        "seq_idx": seq_idx,
        "pos_idx": pos_idx,
        "seq_len": 10,
        "contract": contract,
    }


def test_controlled_record_reports_projection_and_confound_metrics(tmp_path):
    from vfe3.viz import embedding_comparison

    record = embedding_comparison.controlled_embedding_record(**_controlled_record_fixture())
    sidecar = embedding_comparison.write_json_atomic(record, tmp_path / "belief_umap_mu.json")
    loaded = json.loads(sidecar.read_text(encoding="utf-8"))

    assert loaded["projection"]["trustworthiness"]["count"] == 5
    assert np.isfinite(loaded["projection"]["neighbor_overlap"]["mean"])
    ami = loaded["clusters"]["adjusted_mutual_information"]
    assert set(ami) == {"bpe", "function_content", "position_quartile", "sequence_identity"}
    assert loaded["sample"]["token_count"] == 100
    assert len(loaded["sample"]["token_sha256"]) == 64
    assert not list(tmp_path.glob(".belief_umap_mu.*.tmp"))


def test_controlled_record_uses_null_reason_for_unavailable_taxonomy():
    from vfe3.viz import embedding_comparison

    fixture = _controlled_record_fixture()
    fixture["bpe_labels"] = None
    record = embedding_comparison.controlled_embedding_record(**fixture)

    silhouette = record["native_space"]["silhouette"]["bpe"]
    ami = record["clusters"]["adjusted_mutual_information"]["bpe"]
    assert silhouette["value"] is None and silhouette["reason"] == "labels unavailable"
    assert ami["value"] is None and ami["reason"] == "labels unavailable"


def test_controlled_plot_uses_fixed_display_and_pca_clustering(tmp_path):
    from vfe3.viz import embedding_comparison, figures

    rng = np.random.default_rng(9)
    features = np.vstack([
        rng.normal(-1.0, 0.2, size=(80, 12)),
        rng.normal(+1.0, 0.2, size=(80, 12)),
    ])
    bank = {
        "mu": torch.tensor(features, dtype=torch.float32),
        "sigma": torch.ones((160, 12), dtype=torch.float32),
        "phi": torch.zeros((160, 2), dtype=torch.float32),
        "token_ids": torch.arange(160) % 8,
        "seq_idx": torch.arange(160) // 8,
        "pos_idx": torch.arange(160) % 8,
    }

    class RecordingWorker:
        def __init__(self):
            self.calls = []

        def embed(self, values, *, n_neighbors, min_dist, n_components, seed):
            self.calls.append((n_neighbors, min_dist, n_components, seed))
            array = np.asarray(values, dtype=float)
            return array[:, :n_components] + seed * 1e-5

    worker = RecordingWorker()
    image_path = tmp_path / "belief_umap_mu.png"
    sidecar_path = tmp_path / "belief_umap_mu.json"
    figure = figures.plot_belief_umap(
        bank,
        "mu",
        controlled=True,
        decode=lambda ids: f" token{int(ids[0])}",
        umap_worker=worker,
        path=str(image_path),
        sidecar_path=str(sidecar_path),
    )
    record = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert figure.axes[0].get_title().startswith("Belief mu - controlled clusters")
    assert worker.calls == [
        (embedding_comparison.CONTROLLED_N_NEIGHBORS,
         embedding_comparison.CONTROLLED_MIN_DIST, 2, seed)
        for seed in embedding_comparison.CONTROLLED_SEEDS
    ]
    assert record["display"]["seeds"] == list(embedding_comparison.CONTROLLED_SEEDS)
    assert record["clustering"]["space"] == "PCA 10-D"
    assert record["channel"]["feature_chart"] == "Euclidean means"
    assert image_path.is_file() and sidecar_path.is_file()
    figures.plt.close(figure)


def test_exploratory_plot_retains_adaptive_single_embedding(tmp_path):
    from vfe3.viz import figures

    rng = np.random.default_rng(2)
    bank = {
        "mu": torch.tensor(rng.normal(size=(40, 4)), dtype=torch.float32),
        "sigma": torch.ones((40, 4), dtype=torch.float32),
        "phi": torch.zeros((40, 2), dtype=torch.float32),
        "token_ids": torch.arange(40) % 4,
        "seq_idx": torch.arange(40) // 4,
        "pos_idx": torch.arange(40) % 4,
    }

    class RecordingWorker:
        def __init__(self):
            self.calls = []

        def embed(self, values, *, n_neighbors, min_dist, n_components, seed):
            self.calls.append((n_neighbors, min_dist, n_components, seed))
            return np.asarray(values, dtype=float)[:, :n_components]

    worker = RecordingWorker()
    figure = figures.plot_belief_umap(
        bank,
        "mu",
        umap_worker=worker,
        path=str(tmp_path / "exploratory.png"),
    )

    assert figure.axes[0].get_title().startswith("Belief mu - exploratory clusters")
    assert len(worker.calls) == 1
    assert worker.calls[0][0] == figures._UMAP_N_NEIGHBORS
    figures.plt.close(figure)
