r"""Regression tests for the two-panel data-integrity remediation.

These tests cover S2-D1 through S2-D7, first-panel P3, and the duplicated
P8/S2-D6 bounded-character-count finding.
"""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import scaling
import vfe3.data.datasets as datasets_mod
import vfe3.run_artifacts as artifacts_mod
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows, cache_path, cached_token_count, load_cached_tokens
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, load_checkpoint
from vfe3.train import build_optimizer, evaluate


_DATASET = "wiki-en"


def _write_bin_cache(
    root:     Path,
    values:   Iterable[int],

    *,
    split:    str = "train",
    dtype:    str = "int32",
    n_tokens: int | None = None,
) -> Path:
    path = cache_path(_DATASET, split, suffix="bin", cache_dir=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(list(values), dtype=np.dtype(dtype))
    array.tofile(path)
    metadata = {
        "n_tokens": int(array.size if n_tokens is None else n_tokens),
        "dtype":    dtype,
    }
    Path(str(path) + ".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


@pytest.mark.parametrize("n_tokens", [0, -1])
def test_binary_cache_rejects_nonpositive_token_count_before_mapping(
    tmp_path: Path,
    n_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_bin_cache(tmp_path, [], n_tokens=n_tokens)
    mapped = False

    def forbidden_memmap(*args: object, **kwargs: object) -> None:
        nonlocal mapped
        mapped = True
        raise AssertionError("invalid binary metadata reached np.memmap")

    monkeypatch.setattr(datasets_mod.np, "memmap", forbidden_memmap)
    with pytest.raises(ValueError, match="n_tokens.*positive"):
        load_cached_tokens(_DATASET, "train", cache_dir=tmp_path)
    assert mapped is False


@pytest.mark.parametrize(
    ("values", "n_tokens", "dtype", "relation"),
    [
        (range(3), 4, "int32", "truncated"),
        (range(5), 4, "int32", "extended"),
        (range(4), 4, "int64", "dtype-inconsistent"),
    ],
)
def test_binary_cache_requires_exact_metadata_byte_length(
    tmp_path: Path,
    values:   Iterable[int],
    n_tokens: int,
    dtype:    str,
    relation: str,
) -> None:
    path = _write_bin_cache(tmp_path, values, n_tokens=n_tokens, dtype="int32")
    if relation == "dtype-inconsistent":
        Path(str(path) + ".meta.json").write_text(
            json.dumps({"n_tokens": n_tokens, "dtype": dtype}), encoding="utf-8")
    expected = n_tokens * np.dtype(dtype).itemsize
    actual = path.stat().st_size
    with pytest.raises(ValueError, match=rf"file bytes.*{actual}.*expected.*{expected}"):
        load_cached_tokens(_DATASET, "train", cache_dir=tmp_path)
    with pytest.raises(ValueError, match=rf"file bytes.*{actual}.*expected.*{expected}"):
        cached_token_count(_DATASET, "train", cache_dir=tmp_path)


def test_binary_cache_requires_sidecar_before_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = cache_path(_DATASET, "train", suffix="bin", cache_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.arange(4, dtype=np.int32).tofile(path)
    monkeypatch.setattr(
        datasets_mod.np,
        "memmap",
        lambda *args, **kwargs: pytest.fail("missing sidecar reached np.memmap"),
    )
    with pytest.raises(FileNotFoundError, match="meta.json"):
        load_cached_tokens(_DATASET, "train", cache_dir=tmp_path)


def test_padded_final_evaluation_window_emits_every_transition_once() -> None:
    tokens = torch.arange(10, dtype=torch.long)
    windows = TokenWindows(tokens, seq_len=4, stride=4, pad_final=True)
    observed_targets: List[int] = []
    for inputs, targets in windows:
        assert inputs.shape == targets.shape == (4,)
        observed_targets.extend(int(value) for value in targets if int(value) != -100)
    assert observed_targets == tokens[1:].tolist()
    assert windows[-1][1].tolist() == [9, -100, -100, -100]


class _ConstantCEModel(torch.nn.Module):
    def __init__(self, ce: float) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.prior_bank = type("PriorBankStub", (), {"mu_embed": self.anchor})()
        self.ce = ce

    def forward(
        self,
        tokens:  torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[None, None, torch.Tensor]:
        del tokens, targets
        return None, None, self.anchor * 0.0 + self.ce


def test_evaluate_marks_bpc_unavailable_and_names_bits_per_token() -> None:
    model = _ConstantCEModel(math.log(2.0))
    batch = (torch.zeros((1, 3), dtype=torch.long), torch.ones((1, 3), dtype=torch.long))
    metrics = evaluate(model, [batch], tokens_per_char=None)
    assert metrics == {
        "ce":             pytest.approx(math.log(2.0)),
        "ppl":            pytest.approx(2.0),
        "bits_per_token": pytest.approx(1.0),
        "bpc":            None,
    }


def test_scaling_preserves_unavailable_character_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(
        seed=3, embed_dim=4, n_heads=2, gauge_group="block_glk", max_steps=1,
        max_seq_len=4, batch_size=1, vocab_size=8, grad_clip=1.0,
        log_interval=1, eval_interval=1, deterministic=False,
    )
    captured: Dict[str, object] = {}

    monkeypatch.setitem(scaling.CONFIG, "resume", False)
    monkeypatch.setattr(scaling, "_cell_cfg_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(scaling, "VFE3Config", lambda **kwargs: cfg)
    monkeypatch.setattr(scaling, "predict_n_params", lambda value: (2, 1))
    monkeypatch.setattr(scaling, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(scaling, "VFEModel", lambda value: torch.nn.Linear(1, 1))
    monkeypatch.setattr(
        scaling,
        "get_loader",
        lambda *args, **kwargs: SimpleNamespace(generator=None),
    )
    monkeypatch.setattr(
        scaling,
        "RunArtifacts",
        lambda *args, **kwargs: SimpleNamespace(save_json=lambda *a, **k: None),
    )
    monkeypatch.setattr(scaling, "_data_source_identities", lambda *args, **kwargs: {})
    monkeypatch.setattr(scaling, "_tokenizer_tag", lambda dataset: "fixture-tokenizer")
    monkeypatch.setattr(scaling, "_tokens_per_char", lambda *args, **kwargs: None)

    def fake_train(*args: object, **kwargs: object) -> List[float]:
        captured["validation_tpc"] = kwargs["tokens_per_char"]
        return []

    def fake_finalize(*args: object, **kwargs: object) -> Dict[str, object]:
        captured["test_tpc"] = kwargs["tokens_per_char"]
        batch = (
            torch.zeros((1, 3), dtype=torch.long),
            torch.ones((1, 3), dtype=torch.long),
        )
        metrics = evaluate(_ConstantCEModel(math.log(2.0)), [batch], tokens_per_char=None)
        captured["published"] = metrics
        return {
            "test_ce":             metrics["ce"],
            "test_ppl":            metrics["ppl"],
            "test_bits_per_token": metrics["bits_per_token"],
            "test_bpc":            metrics["bpc"],
        }

    monkeypatch.setattr(scaling, "train", fake_train)
    monkeypatch.setattr(scaling, "finalize_run", fake_finalize)
    result = scaling.run_cell(
        {"label": "fixture", "route": "fixture", "scale_knob": 1, "overrides": {}},
        tmp_path / "run",
        3,
        dataset="fixture",
        device=torch.device("cpu"),
    )

    assert result["error_kind"] is None
    assert captured["validation_tpc"] is None
    assert captured["test_tpc"] is None
    assert captured["published"] == {
        "ce":             pytest.approx(math.log(2.0)),
        "ppl":            pytest.approx(2.0),
        "bits_per_token": pytest.approx(1.0),
        "bpc":            None,
    }


def test_tokens_per_char_is_incremental_unicode_exact_and_content_keyed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = cache_path("wikitext-103", "test", suffix="pt", cache_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64), path)
    datasets_mod._TOKENS_PER_CHAR_CACHE.clear()
    datasets_mod._CACHE_SOURCE_IDENTITY_MEMO.clear()
    seen_chunk_sizes: List[int] = []
    byte_pieces = {
        0: b"\xf0",
        1: b"\x9f",
        2: b"\x98",
        3: b"\x80",
        4: b"a",
    }

    def decoder(ids: Iterable[int]) -> bytes:
        materialized = [int(value) for value in ids]
        seen_chunk_sizes.append(len(materialized))
        return b"".join(byte_pieces[value] for value in materialized)

    monkeypatch.setattr(
        datasets_mod,
        "get_tiktoken_byte_decoder",
        lambda dataset: decoder,
        raising=False,
    )
    first = datasets_mod.tokens_per_char(
        "wikitext-103", "test", cache_dir=tmp_path, chunk_tokens=1)
    assert first == pytest.approx(5.0 / 2.0)                 # one emoji plus one ASCII codepoint
    assert max(seen_chunk_sizes) == 1

    # Replace the same cache path and shape in place. The memo key must include the new source
    # identity, so this is recomputed rather than serving the prior 5/2 normalization.
    torch.save(torch.full((5,), 4, dtype=torch.int64), path)
    second = datasets_mod.tokens_per_char(
        "wikitext-103", "test", cache_dir=tmp_path, chunk_tokens=1)
    assert second == pytest.approx(1.0)
    assert second != first


def test_training_unigram_count_uses_bounded_native_width_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = torch.tensor([0, 1, 1, 3, 3, 3, 5, 5, 5, 5], dtype=torch.int32)
    real_bincount = torch.bincount
    chunk_sizes: List[int] = []

    def tracked_bincount(chunk: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        chunk_sizes.append(int(chunk.numel()))
        assert chunk.device.type == "cpu"
        return real_bincount(chunk, *args, **kwargs)

    monkeypatch.setattr(artifacts_mod.torch, "bincount", tracked_bincount)
    counts = artifacts_mod._bincount_token_chunks(tokens, vocab_size=7, chunk_tokens=3)
    assert counts.tolist() == [1, 2, 0, 3, 0, 4, 0]
    assert max(chunk_sizes) <= 3


class _ArtifactSink:
    def __init__(self) -> None:
        self.saved: List[Dict[str, object]] = []

    def save_json(self, name: str, value: Dict[str, object]) -> None:
        assert name == "provenance.json"
        self.saved.append(value)


def test_provenance_reuses_one_immutable_split_digest_across_finalizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = torch.arange(20, dtype=torch.int32)
    loader = DataLoader(TokenWindows(tokens, seq_len=4), batch_size=2)
    model = torch.nn.Linear(1, 1)
    cfg = VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
    calls = 0
    real_hash = artifacts_mod._sha256_tensor_content

    def counted_hash(value: torch.Tensor, *, chunk_tokens: int = 128 * 1024) -> str:
        nonlocal calls
        calls += 1
        return real_hash(value, chunk_tokens=chunk_tokens)

    monkeypatch.setattr(artifacts_mod, "_sha256_tensor_content", counted_hash)
    monkeypatch.setattr(
        artifacts_mod,
        "_git_code_identity",
        lambda: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None},
    )
    sink_a = _ArtifactSink()
    sink_b = _ArtifactSink()
    logger = artifacts_mod.logging.getLogger(__name__)
    artifacts_mod._write_provenance(sink_a, cfg, model, logger, train_loader=loader)
    artifacts_mod._write_provenance(sink_b, cfg, model, logger, train_loader=loader)
    assert calls == 1
    assert sink_a.saved[0]["train_data_sha256"] == sink_b.saved[0]["train_data_sha256"]


def _checkpoint_cfg() -> VFE3Config:
    return VFE3Config(
        vocab_size=16,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
    )


def _data_identity() -> Dict[str, object]:
    return {
        "schema_version":       1,
        "dataset":              _DATASET,
        "split":                "train",
        "tokenizer_tag":        "tiktoken_cl100k",
        "tokenizer_encoding":   "cl100k_base",
        "tokenizer_vocab_size": 100277,
        "model_vocab_size":     16,
        "max_tokens":           4,
        "source": {
            "format":        "bin",
            "tokenizer_tag": "tiktoken_cl100k",
            "size_bytes":    16,
            "sha256":        "a" * 64,
            "meta":          {"n_tokens": 4, "dtype": "int32"},
            "meta_sha256":   "b" * 64,
        },
    }


_DELETE = object()
_INVALID_IDENTITY_SCHEMA_CASES = [
    pytest.param(("source", "format"), _DELETE, id="missing-source-format"),
    pytest.param(("source", "tokenizer_tag"), _DELETE, id="missing-source-tokenizer"),
    pytest.param(("source", "size_bytes"), _DELETE, id="missing-source-size"),
    pytest.param(("source", "sha256"), _DELETE, id="missing-content-digest"),
    pytest.param(("source", "meta"), _DELETE, id="missing-binary-metadata"),
    pytest.param(("source", "meta_sha256"), _DELETE, id="missing-metadata-digest"),
    pytest.param(("source", "meta", "n_tokens"), _DELETE, id="missing-token-count"),
    pytest.param(("source", "meta", "dtype"), _DELETE, id="missing-binary-dtype"),
    pytest.param(("source", "format"), "", id="empty-source-format"),
    pytest.param(("source", "sha256"), "", id="empty-content-digest"),
    pytest.param(("source", "size_bytes"), 15, id="inexact-binary-byte-identity"),
]


@pytest.mark.parametrize(("path", "replacement"), _INVALID_IDENTITY_SCHEMA_CASES)
def test_data_identity_schema_rejects_malformed_source_before_any_restore(
    tmp_path:   Path,
    path:       Tuple[str, ...],
    replacement: object,
) -> None:
    cfg = _checkpoint_cfg()
    saved_model = VFEModel(cfg)
    saved_optimizer = build_optimizer(saved_model, cfg)
    artifacts = RunArtifacts(tmp_path / "saved", cfg, saved_model)
    valid_identity = _data_identity()
    valid_checkpoint = artifacts.save_checkpoint(
        1,
        saved_model,
        saved_optimizer,
        cfg,
        data_state={
            "epoch_start_generator_state": torch.Generator().manual_seed(7).get_state(),
            "batches_consumed":            1,
            "epoch":                       0,
            "data_identity":               valid_identity,
        },
    )

    malformed = copy.deepcopy(valid_identity)
    target = malformed
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    if replacement is _DELETE:
        del target[path[-1]]  # type: ignore[arg-type]
    else:
        target[path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(ValueError, match="data_identity"):
        artifacts.save_checkpoint(
            2,
            saved_model,
            saved_optimizer,
            cfg,
            data_state={
                "epoch_start_generator_state": torch.Generator().manual_seed(7).get_state(),
                "batches_consumed":            1,
                "epoch":                       0,
                "data_identity":               malformed,
            },
        )

    bundle = torch.load(valid_checkpoint, weights_only=True)
    bundle["data_state"]["data_identity"] = malformed
    malformed_checkpoint = tmp_path / f"malformed-{'-'.join(path)}.pt"
    torch.save(bundle, malformed_checkpoint)

    live_model = VFEModel(cfg)
    live_optimizer = build_optimizer(live_model, cfg)
    model_before = {
        name: value.detach().clone() for name, value in live_model.state_dict().items()
    }
    cursor = {"sentinel": 9}
    torch.manual_seed(991)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="data_identity"):
        load_checkpoint(
            malformed_checkpoint,
            live_model,
            live_optimizer,
            data_state=cursor,
            expected_data_identity=valid_identity,
        )

    assert cursor == {"sentinel": 9}
    assert live_optimizer.state == {}
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, value in live_model.state_dict().items():
        assert torch.equal(value, model_before[name])


_IDENTITY_MUTATIONS = [
    pytest.param(("dataset",), "wiki-ja", id="dataset"),
    pytest.param(("tokenizer_tag",), "other-tokenizer", id="tokenizer-tag"),
    pytest.param(("tokenizer_encoding",), "other-encoding", id="tokenizer-encoding"),
    pytest.param(("tokenizer_vocab_size",), 100278, id="tokenizer-vocabulary"),
    pytest.param(("model_vocab_size",), 17, id="model-vocabulary"),
    pytest.param(("max_tokens",), None, id="max-tokens"),
    pytest.param(("source", "format"), "pt", id="cache-format"),
    pytest.param(("source", "size_bytes"), 20, id="cache-size"),
    pytest.param(("source", "meta"), {"n_tokens": 5, "dtype": "int32"}, id="cache-metadata"),
    pytest.param(("source", "meta_sha256"), "c" * 64, id="metadata-digest"),
    pytest.param(("source", "sha256"), "d" * 64, id="content-digest"),
]


@pytest.mark.parametrize(("path", "replacement"), _IDENTITY_MUTATIONS)
def test_resume_rejects_every_data_identity_mismatch_before_rng_or_cursor_restore(
    tmp_path:   Path,
    path:       Tuple[str, ...],
    replacement: object,
) -> None:
    cfg = _checkpoint_cfg()
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "saved", cfg, model)
    identity = _data_identity()
    checkpoint = artifacts.save_checkpoint(
        1,
        model,
        build_optimizer(model, cfg),
        cfg,
        data_state={
            "epoch_start_generator_state": torch.Generator().manual_seed(7).get_state(),
            "batches_consumed":            1,
            "epoch":                       0,
            "data_identity":               identity,
        },
    )

    live = copy.deepcopy(identity)
    target = live
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = replacement  # type: ignore[index]
    cursor = {"sentinel": 9}
    fresh_model = VFEModel(cfg)
    torch.manual_seed(991)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="data identity mismatch"):
        load_checkpoint(
            checkpoint,
            fresh_model,
            data_state=cursor,
            expected_data_identity=live,
        )
    assert cursor == {"sentinel": 9}
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_resume_rejects_cursor_without_identity_contract(tmp_path: Path) -> None:
    cfg = _checkpoint_cfg()
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "saved", cfg, model)
    checkpoint = artifacts.save_checkpoint(
        1,
        model,
        build_optimizer(model, cfg),
        cfg,
        data_state={
            "epoch_start_generator_state": torch.Generator().manual_seed(7).get_state(),
            "batches_consumed":            1,
            "epoch":                       0,
            "data_identity":               _data_identity(),
        },
    )
    bundle = torch.load(checkpoint, weights_only=True)
    del bundle["data_state"]["data_identity"]
    torch.save(bundle, checkpoint)

    with pytest.raises(RuntimeError, match="missing.*data identity"):
        load_checkpoint(
            checkpoint,
            VFEModel(cfg),
            data_state={},
            expected_data_identity=_data_identity(),
        )


def test_exact_resume_rejects_missing_data_state_before_any_mutation(tmp_path: Path) -> None:
    cfg = _checkpoint_cfg()
    saved_model = VFEModel(cfg)
    with torch.no_grad():
        for parameter in saved_model.parameters():
            parameter.fill_(0.25)
    saved_optimizer = build_optimizer(saved_model, cfg)
    saved_parameter = next(saved_model.parameters())
    saved_optimizer.state[saved_parameter] = {
        "step":       torch.tensor(1.0),
        "exp_avg":    torch.ones_like(saved_parameter),
        "exp_avg_sq": torch.ones_like(saved_parameter),
    }
    checkpoint = RunArtifacts(tmp_path / "saved", cfg, saved_model).save_checkpoint(
        1,
        saved_model,
        saved_optimizer,
        cfg,
        data_state=None,
    )

    live_model = VFEModel(cfg)
    live_optimizer = build_optimizer(live_model, cfg)
    model_before = {
        name: value.detach().clone() for name, value in live_model.state_dict().items()
    }
    cursor = {"sentinel": 9}
    torch.manual_seed(991)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="missing.*data_state.*exact resume"):
        load_checkpoint(
            checkpoint,
            live_model,
            live_optimizer,
            data_state=cursor,
            expected_data_identity=_data_identity(),
        )

    assert cursor == {"sentinel": 9}
    assert live_optimizer.state == {}
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, value in live_model.state_dict().items():
        assert torch.equal(value, model_before[name])
