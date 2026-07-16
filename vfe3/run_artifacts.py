r"""Run artifacts for VFE_3.0 training: the persistence + reporting layer.

A training run produces a self-contained directory::

    run_dir/
      config.json        full VFE3Config + run metadata (n_params, dataset, device, timestamp)
      metrics.csv        one row per periodic eval (step, train_loss, lr, val_ce/ppl/bpc, diagnostics)
      checkpoints/
        step_<N>.pt      resumable {step, model_state, optimizer_state, config}
      best_model.pt      {model_state, config, config_fingerprint} at the lowest validation PPL
      test_results.json  end-of-run TEST-split eval on the reloaded best checkpoint
      summary.json       headline numbers (best_val_ppl, test_ppl, wall_time, ...)
      loss_curve.png     training cross-entropy trajectory
      val_ppl.png        validation perplexity trajectory (log-y, best marked)
      holonomy.png / gauge_trace_spread.png   gauge-geometry diagnostics
      free_energy_decomposition.png   per-token F budget snapshot + early/mid/late evolution
      free_energy_codescent.png       F-vs-validation-CE co-descent (twin axis)

``RunArtifacts`` is OPT-IN: ``train`` only touches it when an instance is passed, so the silent
path (``artifacts=None``) writes nothing and is unchanged. ``finalize_run`` reloads the best-val
checkpoint, scores the held-out test split, and writes the summary + figures. Figure generation
is best-effort (a plotting/dependency error is logged, never fatal) so the numeric results
survive a viz problem.
"""

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, fields
from functools import lru_cache
from numbers import Real
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch

from vfe3.config import VFE3Config, config_from_serialized
from vfe3.contracts import DataState, DataStateBuffer
from vfe3.ema import EMA
from vfe3.runtime import deterministic_state

if TYPE_CHECKING:                                        # forward ref only: train imports RunArtifacts
    from vfe3.train import TrainingTerminalState         # at top level, so a runtime import here cycles


def _require_nonnegative_int(value: object, field: str) -> int:
    """Return an exact nonnegative integer cursor; reject coercible lookalikes."""
    if type(value) is not int or value < 0:
        raise ValueError(f"data_state {field} must be a non-negative integer")
    return value


_DATA_IDENTITY_SCHEMA_VERSION = 2
_DATA_IDENTITY_FIELDS = {
    "schema_version",
    "dataset",
    "split",
    "tokenizer_tag",
    "tokenizer_encoding",
    "tokenizer_vocab_size",
    "model_vocab_size",
    "max_tokens",
    "source",
    "iterator",
}
_DATA_SOURCE_IDENTITY_FIELDS = {
    "format",
    "tokenizer_tag",
    "size_bytes",
    "sha256",
    "meta",
    "meta_sha256",
}
_BINARY_TOKEN_DTYPE_BYTES = {
    "uint8":  1,
    "int8":   1,
    "int16":  2,
    "int32":  4,
    "int64":  8,
}
_TENSOR_TOKEN_DTYPE_BYTES = {
    f"torch.{name}": itemsize for name, itemsize in _BINARY_TOKEN_DTYPE_BYTES.items()
}
_DATA_ITERATOR_IDENTITY_FIELDS = {
    "dataset_type",
    "seq_len",
    "stride",
    "pad_final",
    "n_windows",
    "batch_size",
    "drop_last",
    "sampler",
    "sampler_replacement",
    "sampler_num_samples",
}


def _require_identity_sha256(value: object, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)):
        raise ValueError(f"data_state data_identity {field} must be a 64-digit SHA-256 hex digest")
    return value


@lru_cache(maxsize=1)
def _package_code_identity() -> str:
    r"""Return one process-cached content identity for every executable ``vfe3/**/*.py`` source.

    A run captures this digest once when its artifact owner is constructed. Generated files, tests,
    documentation, Git metadata, and output directories are excluded, so artifact publication cannot
    change the identity. New Python processes naturally recompute it after a source edit.
    """
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    sources = sorted(
        path for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    if not sources:
        raise RuntimeError("training code identity found no vfe3 Python sources")
    for path in sources:
        relative = path.relative_to(root.parent).as_posix().encode("utf-8")
        try:
            content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        except OSError as exc:
            raise RuntimeError(
                f"training code identity cannot read {path.relative_to(root.parent)!s}") from exc
        for payload in (relative, content):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _require_identity_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"data_state data_identity {field} must be a positive integer")
    return value


def _normalized_data_identity(
    value: object,
) -> Dict[str, object]:
    r"""Return an owned JSON-normalized data contract or fail closed.

    JSON normalization prevents a caller from mutating nested checkpoint identity state after the
    save boundary and gives equality one canonical set of scalar/container semantics.
    """
    if not isinstance(value, Mapping):
        raise ValueError("data_state data_identity must be a mapping")
    try:
        normalized = json.loads(json.dumps(dict(value), sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise ValueError("data_state data_identity must contain only JSON-compatible values") from exc
    if not isinstance(normalized, dict):
        raise ValueError("data_state data_identity must normalize to a mapping")
    missing = sorted(_DATA_IDENTITY_FIELDS - normalized.keys())
    if missing:
        raise ValueError(f"data_state data_identity is missing required field(s) {missing}")
    schema_version = normalized.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != _DATA_IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            f"data_state data_identity schema_version must be {_DATA_IDENTITY_SCHEMA_VERSION}")
    for field in ("dataset", "split"):
        if not isinstance(normalized.get(field), str) or not normalized[field]:
            raise ValueError(f"data_state data_identity {field} must be a nonempty string")
    tokenizer_tag = normalized.get("tokenizer_tag")
    tokenizer_encoding = normalized.get("tokenizer_encoding")
    tokenizer_vocab_size = normalized.get("tokenizer_vocab_size")
    if tokenizer_tag is not None and (not isinstance(tokenizer_tag, str) or not tokenizer_tag):
        raise ValueError(
            "data_state data_identity tokenizer_tag must be a nonempty string or null")
    if (tokenizer_encoding is not None
            and (not isinstance(tokenizer_encoding, str) or not tokenizer_encoding)):
        raise ValueError(
            "data_state data_identity tokenizer_encoding must be a nonempty string or null")
    if tokenizer_vocab_size is not None:
        _require_identity_positive_int(tokenizer_vocab_size, "tokenizer_vocab_size")
    _require_identity_positive_int(normalized.get("model_vocab_size"), "model_vocab_size")
    max_tokens = normalized.get("max_tokens")
    if max_tokens is not None:
        _require_identity_positive_int(max_tokens, "max_tokens")

    iterator = normalized.get("iterator")
    if not isinstance(iterator, dict):
        raise ValueError("data_state data_identity iterator must be a mapping")
    missing_iterator = sorted(_DATA_ITERATOR_IDENTITY_FIELDS - iterator.keys())
    if missing_iterator:
        raise ValueError(
            f"data_state data_identity iterator is missing required field(s) {missing_iterator}")
    if not isinstance(iterator.get("dataset_type"), str) or not iterator["dataset_type"]:
        raise ValueError(
            "data_state data_identity iterator dataset_type must be a nonempty string")
    for field in ("seq_len", "stride", "n_windows", "batch_size", "sampler_num_samples"):
        _require_identity_positive_int(iterator.get(field), f"iterator {field}")
    for field in ("pad_final", "drop_last"):
        if not isinstance(iterator.get(field), bool):
            raise ValueError(f"data_state data_identity iterator {field} must be a boolean")
    sampler = iterator.get("sampler")
    replacement = iterator.get("sampler_replacement")
    if sampler not in {"random", "sequential"}:
        raise ValueError(
            "data_state data_identity iterator sampler must be 'random' or 'sequential'")
    if sampler == "random":
        if not isinstance(replacement, bool):
            raise ValueError(
                "data_state data_identity iterator random sampler replacement must be a boolean")
    elif replacement is not None:
        raise ValueError(
            "data_state data_identity iterator sequential sampler replacement must be null")

    source = normalized.get("source")
    if not isinstance(source, dict):
        raise ValueError("data_state data_identity source must be a mapping")
    missing_source = sorted(_DATA_SOURCE_IDENTITY_FIELDS - source.keys())
    if missing_source:
        raise ValueError(
            f"data_state data_identity source is missing required field(s) {missing_source}")
    source_format = source.get("format")
    if source_format not in {"pt", "bin", "tensor"}:
        raise ValueError(
            "data_state data_identity source format must be one of 'pt', 'bin', or 'tensor'")
    source_tokenizer = source.get("tokenizer_tag")
    if source_format in {"pt", "bin"}:
        if not isinstance(source_tokenizer, str) or not source_tokenizer:
            raise ValueError(
                "data_state data_identity source tokenizer_tag must be a nonempty string")
        if (not isinstance(tokenizer_tag, str) or not tokenizer_tag
                or not isinstance(tokenizer_encoding, str) or not tokenizer_encoding):
            raise ValueError(
                "data_state data_identity file sources require tokenizer tag and encoding")
        _require_identity_positive_int(tokenizer_vocab_size, "tokenizer_vocab_size")
    elif source_tokenizer is not None and (
            not isinstance(source_tokenizer, str) or not source_tokenizer):
        raise ValueError(
            "data_state data_identity source tokenizer_tag must be a nonempty string or null")
    if source_tokenizer != tokenizer_tag:
        raise ValueError(
            "data_state data_identity tokenizer_tag must agree with source tokenizer_tag")
    size_bytes = _require_identity_positive_int(source.get("size_bytes"), "source size_bytes")
    _require_identity_sha256(source.get("sha256"), "source sha256")

    meta = source.get("meta")
    meta_sha256 = source.get("meta_sha256")
    if source_format == "pt":
        if meta is not None or meta_sha256 is not None:
            raise ValueError(
                "data_state data_identity pt source metadata fields must be null")
    else:
        if not isinstance(meta, dict):
            raise ValueError(
                f"data_state data_identity {source_format} source meta must be a mapping")
        missing_meta = sorted({"n_tokens", "dtype"} - meta.keys())
        if missing_meta:
            raise ValueError(
                f"data_state data_identity source meta is missing required field(s) {missing_meta}")
        n_tokens = _require_identity_positive_int(
            meta.get("n_tokens"), "source meta n_tokens")
        dtype = meta.get("dtype")
        dtype_bytes = (
            _BINARY_TOKEN_DTYPE_BYTES.get(dtype)
            if source_format == "bin"
            else _TENSOR_TOKEN_DTYPE_BYTES.get(dtype)
        )
        if dtype_bytes is None:
            raise ValueError(
                f"data_state data_identity source meta dtype {dtype!r} is unsupported")
        expected_bytes = n_tokens * dtype_bytes
        if size_bytes != expected_bytes:
            raise ValueError(
                "data_state data_identity source size_bytes must equal "
                f"n_tokens * dtype.itemsize ({expected_bytes})")
        if source_format == "bin":
            _require_identity_sha256(meta_sha256, "source meta_sha256")
        elif meta_sha256 is not None:
            raise ValueError(
                "data_state data_identity tensor source meta_sha256 must be null")
    return normalized


def semantic_config_fingerprint(
    config: Mapping[str, Any],
) -> str:
    """Return the stable SHA-256 fingerprint of a normalized semantic config mapping."""
    normalized = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _phi_chart_norm_route(
    model: torch.nn.Module,
    cfg:   object,
) -> Optional[str]:
    """Persist the exact norm route only when the projected phi M-step is enabled."""
    if getattr(cfg, "phi_mstep_max_matrix_norm", None) is None:
        return None
    route = getattr(model.group, "phi_norm_route", None)
    return route() if route is not None else "dense_fallback"


def sigma_behavior_config(
    cfg: "VFE3Config | Mapping[str, object]",
) -> Dict[str, object]:
    r"""Return the non-policy config projection that controls belief transition and decode (PB-06).

    Every ``policy_*`` field chooses the candidate MENU, horizons, preferences, score weights, ambiguity
    dispatch, and gate authorization AROUND an already defined candidate; none of them alters the
    underlying ``rollout_predictive_state`` belief transition or the ``PriorBank.decode`` distribution
    whose sigma utility the gate measured. So the sigma-gate model-behavior fingerprint is bound to every
    non-policy field (``decode_tau``, family/divergence, the E-step, transport, prior-bank settings, ...)
    and INVARIANT to the policy fields: a checkpoint measured under ``policy_mode='none'`` and consumed
    under ``policy_mode='efe_rollout'`` with different preference/score/top-k/horizon/gate fields carry
    the SAME projection. Accepts a live :class:`VFE3Config` (via ``asdict``) or a serialized mapping."""
    if isinstance(cfg, VFE3Config):
        base: Dict[str, object] = asdict(cfg)
    elif isinstance(cfg, Mapping):
        base = dict(cfg)
    else:
        raise TypeError(
            f"sigma_behavior_config expects a VFE3Config or mapping, got {type(cfg).__name__}")
    return {key: value for key, value in base.items() if not str(key).startswith("policy_")}


def model_behavior_fingerprint(
    semantic_config: Mapping[str, object],
    state_dict:      Mapping[str, torch.Tensor],
) -> str:
    r"""Hash canonical semantic config plus sorted tensor metadata and bytes (PB-06).

    Binds a sigma-gate artifact to the EXACT model whose sigma utility was measured: the digest is
    prefixed with ``semantic_config_fingerprint(semantic_config)`` (typically
    :func:`sigma_behavior_config`), then folds each state-dict entry in SORTED key order -- the
    length-delimited key, the tensor ``dtype`` and ``shape``, and the raw contiguous ``uint8`` byte view
    of ``tensor.detach().cpu().contiguous().reshape(-1)``. Key order cannot change the digest (sorted),
    but any changed weight value, dtype, or shape does, as does any non-policy behavior field (via the
    prefix). Non-tensor state-dict values are rejected."""
    digest = hashlib.sha256()
    digest.update(semantic_config_fingerprint(semantic_config).encode("utf-8"))
    digest.update(b"\0state\0")
    for key in sorted(state_dict):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"model_behavior_fingerprint expects tensor state-dict values; entry {key!r} is "
                f"{type(value).__name__}")
        key_bytes = str(key).encode("utf-8")
        digest.update(len(key_bytes).to_bytes(8, "big"))
        digest.update(key_bytes)
        meta = f"{value.dtype}|{tuple(value.shape)}".encode("utf-8")
        digest.update(len(meta).to_bytes(8, "big"))
        digest.update(meta)
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _atomic_replace(
    final: Path,                         # destination (the artifact name readers load)
    tmp:   Path,                         # same-directory temp file, already fully written

    *,
    delay:   float = 0.2,
    retries: int   = 5,
) -> None:
    r"""Atomically publish ``tmp`` over ``final`` via ``os.replace`` (same-volume rename).

    Same-directory temp + ``os.replace`` makes the publish an atomic rename on one volume, so a
    crash or power loss mid-write can never leave a truncated JSON or corrupt ``.pt`` at the final
    name (audit 2026-07-01 C11). Retries with backoff on ``PermissionError`` -- Windows can hold a
    transient open-handle lock on the destination (this host has hit it on ``best_model.pt``) --
    and re-raises any other error (and the last ``PermissionError``) so a real failure is never
    swallowed. On the raising paths the orphaned ``tmp`` is best-effort deleted first (audit
    2026-07-01 round-3); between retries it must survive (it is the source of the next replace)."""
    def _cleanup_tmp() -> None:
        try:                                             # cleanup failure must not mask the original error
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    for i in range(retries):
        try:
            os.replace(tmp, final)
            return
        except PermissionError:
            if i == retries - 1:
                _cleanup_tmp()
                raise
            time.sleep(delay)
        except Exception:
            _cleanup_tmp()
            raise


@contextmanager
def _unique_sibling_temp(final: Path) -> Iterator[Path]:
    r"""Yield a uniquely reserved same-directory temporary path for one artifact writer.

    ``mkstemp`` performs the reservation atomically, so concurrent writers to the same final
    artifact never share a temporary filename.  The caller must publish with
    :func:`_atomic_replace`; this context also removes an unconsumed temporary after any failure.
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        dir=str(final.parent),
        prefix=f".{final.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp = Path(raw_path)
    try:
        yield tmp
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _write_json_atomic(final: Path, obj: object) -> Path:
    """Serialize JSON through a unique same-directory temporary and atomically publish it."""
    final = Path(final)
    with _unique_sibling_temp(final) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        _atomic_replace(final, tmp)
    return final


def _selection_semantic_config(
    config: 'VFE3Config | Mapping[str, object]',
) -> Dict[str, object]:
    r"""Project a config down to the fields that determine the SELECTED weights.

    Model selection depends on architecture, family/transport/decode, optimizer/schedule, and every
    objective weight -- but NOT on resume bookkeeping/policy (``resume_from``,
    ``trust_resume_checkpoint``) or output cadence (``log_interval``, ``checkpoint_interval``,
    ``generate_figures``). Comparing this projection lets
    a cross-run resume carry otherwise-identical selected weights even when the resumed run changed
    its resume path or figure/log cadence, while every architecture/objective difference still
    invalidates the bundle.

    A live :class:`VFE3Config` starts from ``asdict`` directly. A SERIALIZED mapping is first checked
    key-by-key against the current fields -- an unknown newer field FAILS CLOSED rather than being
    silently ignored (as :func:`config_from_serialized` would) -- and is then migrated through
    ``config_from_serialized`` so a genuinely older mapping acquires the current defaults for any
    field it predates. The raw mapping's full ``config_fingerprint`` is verified elsewhere (before
    this normalization), so default migration never hides excluded-field tampering."""
    if isinstance(config, VFE3Config):
        normalized = asdict(config)
    elif isinstance(config, Mapping):
        known = {field.name for field in fields(VFE3Config)}
        unknown = sorted(str(key) for key in config if key not in known)
        if unknown:
            raise ValueError(
                f"best-model selection config carries field(s) unknown to this code version "
                f"{unknown}; refusing to migrate (an artifact from a newer code version)")
        normalized = asdict(config_from_serialized(
            config, source="best-model selection compatibility"))
    else:
        raise TypeError(
            f"_selection_semantic_config expects a VFE3Config or mapping, got {type(config).__name__}")
    for key in (
        "resume_from",
        "trust_resume_checkpoint",
        "log_interval",
        "checkpoint_interval",
        "generate_figures",
    ):
        normalized.pop(key, None)
    return normalized


def _validate_best_model_mapping(
    bundle:               object,
    cfg:                  Optional[VFE3Config],
    expected_model_state: Mapping[str, torch.Tensor],
    context:              str,
) -> Dict[str, object]:
    """Validate an already-loaded selected-weights mapping without mutating live state."""
    legacy_fields = {"model_state", "config", "config_fingerprint"}
    current_fields = legacy_fields | {"code_identity_sha256", "selection_data_identity"}
    if not isinstance(bundle, Mapping) or set(bundle) not in (legacy_fields, current_fields):
        raise RuntimeError(f"{context} is not a semantic best-model mapping")
    saved_config = bundle["config"]
    if not isinstance(saved_config, Mapping):
        raise RuntimeError(f"{context} has a non-mapping config")
    if bundle["config_fingerprint"] != semantic_config_fingerprint(saved_config):
        raise RuntimeError(f"{context} has a config fingerprint mismatch")
    if (cfg is not None
            and semantic_config_fingerprint(_selection_semantic_config(saved_config))
            != semantic_config_fingerprint(_selection_semantic_config(cfg))):
        raise RuntimeError(f"{context} does not match the active selection config")
    saved_state = bundle["model_state"]
    if not isinstance(saved_state, Mapping):
        raise RuntimeError(f"{context} has a non-mapping model_state")
    if set(saved_state) != set(expected_model_state):
        raise RuntimeError(f"{context} model_state keys do not match the live model")
    for key, expected in expected_model_state.items():
        actual = saved_state[key]
        if not isinstance(actual, torch.Tensor):
            raise RuntimeError(f"{context} entry {key!r} is not a tensor")
        if (actual.shape != expected.shape or actual.dtype != expected.dtype
                or actual.layout != expected.layout):
            raise RuntimeError(
                f"{context} entry {key!r} has an incompatible shape/dtype/layout")
    validated = dict(bundle)
    if set(bundle) == current_fields:
        code_identity = bundle["code_identity_sha256"]
        if (not isinstance(code_identity, str) or len(code_identity) != 64
                or any(character not in "0123456789abcdef" for character in code_identity)):
            raise RuntimeError(f"{context} has an invalid executable-code identity")
        selection_identity = bundle["selection_data_identity"]
        if selection_identity is not None:
            try:
                validated["selection_data_identity"] = _normalized_data_identity(selection_identity)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{context} has an invalid validation-data identity") from exc
    return validated


def _best_selection_is_portable(
    bundle:                           Mapping[str, object],
    cfg:                              Optional[VFE3Config],
    expected_code_identity:           str,
    expected_selection_data_identity: Optional[Mapping[str, object]],
) -> bool:
    """Return whether selected weights were measured under this code/config/validation contract."""
    if set(bundle) != {
            "model_state", "config", "config_fingerprint",
            "code_identity_sha256", "selection_data_identity"}:
        return False
    if bundle.get("code_identity_sha256") != expected_code_identity:
        return False
    if expected_selection_data_identity is None or bundle.get("selection_data_identity") is None:
        return False
    try:
        saved_selection_identity = _normalized_data_identity(bundle["selection_data_identity"])
        live_selection_identity = _normalized_data_identity(expected_selection_data_identity)
        if saved_selection_identity != live_selection_identity:
            return False
        if (cfg is not None
                and semantic_config_fingerprint(
                    _selection_semantic_config(bundle["config"]))
                != semantic_config_fingerprint(_selection_semantic_config(cfg))):
            return False
    except (TypeError, ValueError):
        return False
    return True


def _read_best_model_bundle(
    path:                 Path,
    cfg:                  VFE3Config,
    expected_model_state: Mapping[str, torch.Tensor],
    map_location:         'str | torch.device',
) -> Dict[str, object]:
    r"""Safe-load ``best_model.pt`` and validate it as a portable selected-weights bundle, NONMUTATING.

    Loaded with ``weights_only=True`` (the bundle is only tensors, an ``asdict`` config, and a
    fingerprint string). Fails closed unless every check passes: the bundle is the three-key semantic
    best-model mapping; its stored full ``config_fingerprint`` equals the recomputed fingerprint of its
    own saved config (excluded-field tampering is still caught BEFORE the selection projection); the
    SELECTION projection of the saved config matches the live config's projection; and ``model_state``
    matches ``expected_model_state`` key-for-key on tensor type, shape, and dtype. Neither ``cfg`` nor
    ``expected_model_state`` is mutated, and no state is loaded into any model. Returns the validated
    bundle as a plain ``dict``."""
    bundle = torch.load(path, map_location=map_location, weights_only=True)
    return _validate_best_model_mapping(
        bundle, cfg, expected_model_state, f"best-model bundle at {path}")


def _validate_checkpoint_model_state(
    saved_state:          object,
    expected_model_state: Mapping[str, torch.Tensor],
    checkpoint_path:      Path,
) -> Mapping[str, torch.Tensor]:
    """Validate a checkpoint model state completely before any live tensor is copied."""
    if not isinstance(saved_state, Mapping):
        raise RuntimeError(f"checkpoint {checkpoint_path} has a non-mapping model_state")
    if set(saved_state) != set(expected_model_state):
        raise RuntimeError(
            f"checkpoint {checkpoint_path} model_state keys do not match the live model")
    for key, expected in expected_model_state.items():
        actual = saved_state[key]
        if not isinstance(actual, torch.Tensor):
            raise RuntimeError(
                f"checkpoint {checkpoint_path} model_state entry {key!r} is not a tensor")
        if (actual.shape != expected.shape or actual.dtype != expected.dtype
                or actual.layout != expected.layout):
            raise RuntimeError(
                f"checkpoint {checkpoint_path} model_state entry {key!r} has an incompatible "
                f"shape/dtype/layout")
    return saved_state


def _validated_saved_successful_updates(
    saved_groups: object,
    saved_step:   int,
) -> Optional[int]:
    """Validate the all-or-none accepted-update clock on serialized optimizer groups."""
    if not isinstance(saved_groups, list) or not all(
            isinstance(group, Mapping) for group in saved_groups):
        raise RuntimeError("checkpoint optimizer_state param_groups must be a list of mappings")
    present = ["successful_updates" in group for group in saved_groups]
    if any(present) and not all(present):
        raise RuntimeError(
            "checkpoint optimizer_state successful_updates must be present on every group or none")
    if not any(present):
        return None
    values = [group["successful_updates"] for group in saved_groups]
    if any(type(value) is not int or value < 0 for value in values):
        raise RuntimeError(
            "checkpoint optimizer_state successful_updates must be exact non-negative integers")
    if len(set(values)) != 1:
        raise RuntimeError(
            "checkpoint optimizer_state successful_updates must agree across all groups")
    count = values[0]
    if count > saved_step:
        raise RuntimeError(
            "checkpoint optimizer_state successful_updates cannot exceed checkpoint step")
    return count


def _validate_optimizer_state(
    saved_state: object,
    optimizer:   torch.optim.Optimizer,
    saved_step:  int,
    slot_manifest: object,
) -> Mapping[str, object]:
    """Preflight optimizer topology and populated per-parameter slots without mutation."""
    if not isinstance(saved_state, Mapping):
        raise RuntimeError("checkpoint optimizer_state must be a mapping when optimizer is supplied")
    saved_groups = saved_state.get("param_groups")
    saved_slots = saved_state.get("state")
    has_optimizer_extra = hasattr(optimizer, "_omega_step")
    expected_top_level = {"state", "param_groups"} | (
        {"optimizer_extra"} if has_optimizer_extra else set())
    if set(saved_state) != expected_top_level:
        raise RuntimeError("checkpoint optimizer_state has an invalid top-level schema")
    successful_updates = _validated_saved_successful_updates(saved_groups, saved_step)
    maximum_slot_step = successful_updates if successful_updates is not None else saved_step
    if has_optimizer_extra:
        extra = saved_state["optimizer_extra"]
        if (not isinstance(extra, Mapping)
                or set(extra) != {"omega_step", "omega_dirty_format"}
                or type(extra["omega_step"]) is not int
                or extra["omega_step"] < 0
                or extra["omega_step"] > maximum_slot_step
                or type(extra["omega_dirty_format"]) is not int
                or extra["omega_dirty_format"] != 1):
            raise RuntimeError("checkpoint optimizer_state optimizer_extra is invalid")
    if not isinstance(saved_slots, Mapping):
        raise RuntimeError("checkpoint optimizer_state state must be a mapping")
    if (not isinstance(slot_manifest, Mapping)
            or set(slot_manifest) != {"parameter_ids", "sha256"}):
        raise RuntimeError("checkpoint optimizer populated-slot manifest is missing or invalid")
    manifest_ids = slot_manifest["parameter_ids"]
    if (not isinstance(manifest_ids, list)
            or any(type(value) is not int for value in manifest_ids)
            or manifest_ids != sorted(set(manifest_ids))):
        raise RuntimeError("checkpoint optimizer populated-slot manifest ids are invalid")
    encoded_manifest = json.dumps(
        manifest_ids, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    manifest_sha256 = hashlib.sha256(encoded_manifest).hexdigest()
    if slot_manifest["sha256"] != manifest_sha256:
        raise RuntimeError("checkpoint optimizer populated-slot manifest fingerprint mismatch")
    populated_ids = sorted(
        parameter_id for parameter_id, state in saved_slots.items()
        if isinstance(state, Mapping) and bool(state)
    )
    if populated_ids != manifest_ids:
        raise RuntimeError("checkpoint optimizer populated parameter slots do not match the manifest")
    if len(saved_groups) != len(optimizer.param_groups):
        raise RuntimeError("checkpoint optimizer_state parameter-group count mismatch")

    parameter_by_id: Dict[int, Tuple[torch.Tensor, Mapping[str, object]]] = {}
    for saved_group, live_group in zip(saved_groups, optimizer.param_groups):
        saved_ids = saved_group.get("params")
        live_parameters = live_group.get("params")
        if not isinstance(saved_ids, list) or not isinstance(live_parameters, list):
            raise RuntimeError("checkpoint optimizer_state group params must be lists")
        if len(saved_ids) != len(live_parameters):
            raise RuntimeError("checkpoint optimizer_state parameter topology mismatch")
        for parameter_id, parameter in zip(saved_ids, live_parameters):
            if type(parameter_id) is not int or parameter_id in parameter_by_id:
                raise RuntimeError("checkpoint optimizer_state parameter ids are invalid")
            parameter_by_id[parameter_id] = (parameter, saved_group)

    for parameter_id, state in saved_slots.items():
        if type(parameter_id) is not int or parameter_id not in parameter_by_id:
            raise RuntimeError("checkpoint optimizer_state contains an unknown parameter id")
        if not isinstance(state, Mapping):
            raise RuntimeError("checkpoint optimizer_state parameter slot must be a mapping")
        if not state:
            continue
        parameter, group = parameter_by_id[parameter_id]
        keys = set(state)
        if keys & {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
            required = {"step", "exp_avg", "exp_avg_sq"}
            if not required.issubset(keys):
                raise RuntimeError("checkpoint optimizer_state has incomplete AdamW slots")
            step = state["step"]
            expected_step_device = (
                parameter.device
                if bool(group.get("fused", False)) or bool(group.get("capturable", False))
                else torch.device("cpu")
            )
            valid_step = (
                isinstance(step, torch.Tensor)
                and step.device == expected_step_device
                and step.dtype == torch.float32
                and step.layout == torch.strided
                and step.ndim == 0
                and bool(torch.isfinite(step))
                and float(step) >= 0.0
                and float(step).is_integer()
                and float(step) <= maximum_slot_step
            )
            if not valid_step:
                raise RuntimeError("checkpoint optimizer_state AdamW step is invalid")
            tensor_slots = ["exp_avg", "exp_avg_sq"]
            if bool(group.get("amsgrad", False)):
                if "max_exp_avg_sq" not in state:
                    raise RuntimeError("checkpoint optimizer_state is missing max_exp_avg_sq")
                tensor_slots.append("max_exp_avg_sq")
            for name in tensor_slots:
                value = state[name]
                if (not isinstance(value, torch.Tensor)
                        or value.shape != parameter.shape
                        or value.dtype != parameter.dtype
                        or value.layout != parameter.layout
                        or not bool(torch.isfinite(value).all())):
                    raise RuntimeError(
                        f"checkpoint optimizer_state {name} is incompatible with its parameter")
        elif bool(group.get("gauge", False)):
            if "gauge_mom" in state:
                tensor_slots = ["gauge_mom"]
            elif {"gauge_m", "gauge_v", "gauge_step"}.issubset(keys):
                tensor_slots = ["gauge_m", "gauge_v"]
                gauge_step = state["gauge_step"]
                if (type(gauge_step) is not int or gauge_step < 0
                        or gauge_step > maximum_slot_step):
                    raise RuntimeError("checkpoint optimizer_state gauge_step is invalid")
            else:
                raise RuntimeError("checkpoint optimizer_state has incomplete gauge slots")
            for name in tensor_slots:
                value = state[name]
                if (not isinstance(value, torch.Tensor)
                        or value.shape != parameter.shape
                        or value.dtype != parameter.dtype
                        or value.layout != parameter.layout
                        or not bool(torch.isfinite(value).all())):
                    raise RuntimeError(
                        f"checkpoint optimizer_state {name} is incompatible with its parameter")
        elif bool(group.get("omega", False)):
            dirty = state.get("omega_dirty")
            if (keys != {"omega_dirty"} or not isinstance(dirty, torch.Tensor)
                    or dirty.dtype != torch.bool or dirty.shape != (parameter.shape[0],)):
                raise RuntimeError("checkpoint optimizer_state omega_dirty is invalid")
        else:
            raise RuntimeError("checkpoint optimizer_state contains unsupported parameter slots")
    return saved_state


def _optimizer_populated_slot_manifest(
    optimizer_state: Mapping[str, object],
) -> Dict[str, object]:
    """Bind the exact set of populated per-parameter optimizer slots independently of slot bytes."""
    slots = optimizer_state.get("state")
    if not isinstance(slots, Mapping):
        raise RuntimeError("optimizer state has no state mapping")
    parameter_ids = sorted(
        parameter_id for parameter_id, state in slots.items()
        if type(parameter_id) is int and isinstance(state, Mapping) and bool(state)
    )
    encoded = json.dumps(parameter_ids, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return {
        "parameter_ids": parameter_ids,
        "sha256":        hashlib.sha256(encoded).hexdigest(),
    }


def _validate_scaler_state(
    saved_state: object,
    scaler:      'torch.amp.GradScaler',
) -> Mapping[str, object]:
    if not isinstance(saved_state, Mapping):
        raise RuntimeError("checkpoint scaler_state is required for an enabled GradScaler")
    required = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
    if set(saved_state) != required:
        raise RuntimeError("checkpoint scaler_state has an invalid schema")
    scale = saved_state["scale"]
    growth = saved_state["growth_factor"]
    backoff = saved_state["backoff_factor"]
    interval = saved_state["growth_interval"]
    tracker = saved_state["_growth_tracker"]
    if (not isinstance(scale, Real) or isinstance(scale, bool)
            or not math.isfinite(float(scale)) or float(scale) <= 0.0):
        raise RuntimeError("checkpoint scaler_state scale must be a finite positive real")
    if (not isinstance(growth, Real) or isinstance(growth, bool)
            or not math.isfinite(float(growth)) or float(growth) <= 1.0):
        raise RuntimeError("checkpoint scaler_state growth_factor must be greater than one")
    if (not isinstance(backoff, Real) or isinstance(backoff, bool)
            or not math.isfinite(float(backoff)) or not 0.0 < float(backoff) < 1.0):
        raise RuntimeError("checkpoint scaler_state backoff_factor must lie in (0, 1)")
    if type(interval) is not int or interval <= 0:
        raise RuntimeError("checkpoint scaler_state growth_interval must be a positive integer")
    if type(tracker) is not int or not 0 <= tracker < interval:
        raise RuntimeError(
            "checkpoint scaler_state _growth_tracker must lie in [0, growth_interval)")
    return saved_state


def _validate_ema_state(
    saved_state: object,
    ema:         EMA,

    *,
    require_state: bool = False,
) -> Optional[Mapping[str, object]]:
    if saved_state is None:
        if require_state:
            raise RuntimeError("checkpoint ema_state is required for exact EMA resume")
        return None
    if not isinstance(saved_state, Mapping) or set(saved_state) != {"decay", "shadow"}:
        raise RuntimeError("checkpoint ema_state has an invalid schema")
    decay = saved_state["decay"]
    if (not isinstance(decay, Real) or isinstance(decay, bool)
            or not math.isfinite(float(decay)) or float(decay) != float(ema.decay)):
        raise RuntimeError("checkpoint ema_state decay does not match the active EMA")
    shadow = saved_state["shadow"]
    if not isinstance(shadow, Mapping) or set(shadow) != set(ema.shadow):
        raise RuntimeError("checkpoint ema_state shadow keys do not match the active EMA")
    for name, expected in ema.shadow.items():
        actual = shadow[name]
        if (not isinstance(actual, torch.Tensor)
                or actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or actual.layout != expected.layout
                or not bool(torch.isfinite(actual).all())):
            raise RuntimeError(f"checkpoint ema_state shadow entry {name!r} is incompatible")
    return saved_state


def _validate_rng_state(
    rng_state: object,
) -> Mapping[str, object]:
    if not isinstance(rng_state, Mapping) or "cpu" not in rng_state or "cuda" not in rng_state:
        raise RuntimeError("checkpoint rng_state must contain cpu and cuda states")
    cpu = rng_state["cpu"]
    if (not isinstance(cpu, torch.Tensor) or cpu.device.type != "cpu"
            or cpu.dtype != torch.uint8):
        raise RuntimeError("checkpoint rng_state cpu state must be a CPU ByteTensor")
    try:
        torch.Generator(device="cpu").set_state(cpu)
    except RuntimeError as exc:
        raise RuntimeError("checkpoint rng_state cpu state is invalid") from exc
    cuda = rng_state["cuda"]
    if torch.cuda.is_available() and cuda is None:
        raise RuntimeError("checkpoint rng_state must contain every active CUDA RNG state")
    if cuda is not None:
        if not isinstance(cuda, (list, tuple)):
            raise RuntimeError("checkpoint rng_state cuda states must be a sequence or null")
        if torch.cuda.is_available() and len(cuda) != torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint rng_state CUDA state count does not match the active device count")
        for index, state in enumerate(cuda):
            if (not isinstance(state, torch.Tensor) or state.device.type != "cpu"
                    or state.dtype != torch.uint8):
                raise RuntimeError("checkpoint rng_state CUDA states must be CPU ByteTensors")
            if torch.cuda.is_available() and index < torch.cuda.device_count():
                try:
                    torch.Generator(device=f"cuda:{index}").set_state(state)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"checkpoint rng_state CUDA state {index} is invalid") from exc
    return rng_state


def _validate_generator_state(
    state: object,
    field: str,
) -> torch.Tensor:
    if (not isinstance(state, torch.Tensor) or state.device.type != "cpu"
            or state.dtype != torch.uint8):
        raise RuntimeError(f"checkpoint {field} must be a CPU ByteTensor")
    try:
        torch.Generator(device="cpu").set_state(state)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint {field} is invalid") from exc
    return state


def _validate_epoch_generator_state(
    state:         object,
    data_identity: Mapping[str, object],
) -> Optional[torch.Tensor]:
    """Validate the epoch-start generator state against the iterator's sampler semantics."""
    iterator = data_identity.get("iterator")
    if not isinstance(iterator, Mapping):
        raise RuntimeError("checkpoint data identity iterator must be a mapping")
    sampler = iterator.get("sampler")
    if sampler == "random":
        return _validate_generator_state(state, "data_state epoch_start_generator_state")
    if sampler == "sequential":
        if state is not None:
            raise RuntimeError(
                "checkpoint sequential data_state requires a null epoch generator state")
        return None
    raise RuntimeError("checkpoint data identity carries an unsupported sampler")


def _publish_best_model_bundle(
    bundle:               Mapping[str, object],
    expected_model_state: Mapping[str, torch.Tensor],
    artifacts:            'RunArtifacts',
) -> None:
    r"""Revalidate ``bundle`` against the live run and atomically publish it as the run's ``best_model.pt``.

    Writes through a uniquely reserved same-directory temporary and ``os.replace`` so no reader sees a partial
    bundle; revalidates the exact bytes it writes (a byte-for-byte round-trip through
    :func:`_read_best_model_bundle`) so a corrupt in-memory bundle fails closed before it is published.
    The bundle is NEVER loaded into the live training model -- publication only makes the selection
    checkpoint reachable in the new run directory for later finalization."""
    with _unique_sibling_temp(artifacts.best_path) as tmp:
        torch.save(dict(bundle), tmp)
        _read_best_model_bundle(tmp, artifacts.cfg, expected_model_state, "cpu")
        _atomic_replace(artifacts.best_path, tmp)


class RunArtifacts:
    r"""Owns a run directory and the incremental writes (CSV rows, checkpoints, best model).

    Contract (m25): each instance owns a FRESH run_dir. ``__init__`` (re)writes config.json and the
    first ``log_metrics`` opens metrics.csv with ``"w"`` (truncate), so aiming a new instance at a
    populated dir would clobber it -- but no path does: resume builds a new timestamped run_dir and
    restores state from a checkpoint FILE via ``load_checkpoint``, never reusing a dir in place."""

    def __init__(
        self,
        run_dir:   'str | Path',
        cfg:       VFE3Config,
        model:     torch.nn.Module,

        *,
        dataset:   str                  = "",
        device:    'str | torch.device' = "cpu",
        timestamp: Optional[str]        = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.best_path = self.run_dir / "best_model.pt"
        self.cfg = cfg                                       # kept for figure scaling (lambda_beta) + guards
        self.code_identity_sha256 = _package_code_identity()
        self.selection_data_identity: Optional[Dict[str, object]] = None

        self.best_val_ppl: float = float("inf")
        self.best_step: Optional[int] = None
        self.history: List[Dict[str, float]] = []          # in-memory copy of the CSV rows (for figures)
        self._fieldnames: Optional[List[str]] = None

        self.save_json("config.json", {
            "config":    asdict(cfg),
            "n_params":  int(sum(p.numel() for p in model.parameters())),
            "dataset":   dataset,
            "device":    str(device),
            "timestamp": timestamp,
        })

    def bind_selection_data_identity(self, identity: Mapping[str, object]) -> None:
        """Bind every selected-weight comparison in this run to one validation data contract."""
        normalized = _normalized_data_identity(identity)
        if (self.selection_data_identity is not None
                and self.selection_data_identity != normalized):
            raise RuntimeError(
                "one RunArtifacts instance cannot compare model selections across validation data")
        self.selection_data_identity = normalized

    def save_json(self, name: str, obj: dict) -> Path:
        r"""Write ``obj`` as pretty JSON to ``run_dir/name`` (non-serializable -> str).

        Atomic: written to a unique same-directory temporary then published via ``os.replace``, so a crash
        mid-write can never leave a truncated/partial JSON at the final name."""
        candidate         = Path(name)
        windows_candidate = PureWindowsPath(name)
        if (not name or name in {".", ".."} or "/" in name or "\\" in name
                or candidate.name != name or candidate.is_absolute() or windows_candidate.drive):
            raise ValueError(f"artifact name must be a regular bare filename, got {name!r}")
        return _write_json_atomic(self.run_dir / name, obj)

    def log_metrics(self, row: Dict[str, float]) -> None:
        r"""Append one metrics row to ``metrics.csv`` (header written on the first call).

        The column set is fixed by the first row; later rows must share those keys so the CSV
        stays rectangular (the training loop emits a homogeneous row each periodic eval).

        NaN cells are written to the file as BLANK (empty string), so an eval-cadence column
        (val_*, generalization_gap, the held-out probes) -- NaN on the denser log-interval rows
        between evals -- shows an empty cell rather than a repeated value or a literal "nan".
        The IN-MEMORY ``self.history`` keeps the raw NaN float so
        the figure pass (which filters on ``math.isfinite``) is unaffected."""
        self.history.append(dict(row))                          # raw floats (incl. NaN) for the figure pass
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with open(self.csv_path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self._fieldnames).writeheader()
        csv_row = {k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        with open(self.csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self._fieldnames).writerow(csv_row)

    def maybe_save_best(self, step: int, model: torch.nn.Module, val_ppl: float) -> bool:
        r"""Save weights bound to their semantic config iff ``val_ppl`` is a new minimum.

        Atomic (same-dir tmp + ``os.replace``): a crash or Windows lock mid-save can never leave a
        corrupt/unreadable ``best_model.pt`` where a good one stood."""
        if val_ppl < self.best_val_ppl:
            self.best_val_ppl = float(val_ppl)
            self.best_step = int(step)
            config = asdict(self.cfg)
            bundle = {
                "model_state":        model.state_dict(),
                "config":             config,
                "config_fingerprint": semantic_config_fingerprint(config),
                "code_identity_sha256": self.code_identity_sha256,
                "selection_data_identity": self.selection_data_identity,
            }
            with _unique_sibling_temp(self.best_path) as tmp:
                torch.save(bundle, tmp)
                _atomic_replace(self.best_path, tmp)
            return True
        return False

    def save_attention_maps(
        self,
        step:   int,
        maps:   torch.Tensor,                 # (L, H, N, N) per-layer per-head attention
        logger: Optional[logging.Logger] = None,
    ) -> Optional[List[Path]]:
        r"""Best-effort attention heatmaps for one periodic eval: one figure per (layer, head).

        Writes ``attention/step_<N>_layer<l>_head<h>.png`` per (layer, head) -- a LOG-scaled beta
        heatmap (see :func:`vfe3.viz.figures.plot_attention_heatmap`) on a color scale shared
        across panels so heads/layers stay comparable. Mirrors ``_save_figures``: a plotting or
        dependency error is logged and swallowed (never fatal to the run), and each figure is
        closed so ~30 evals do not leak figures. Returns the paths written, or None on failure.
        """
        try:
            from vfe3.viz import figures as figs
        except Exception as exc:                                    # a viz error must never kill training
            (logger or logging.getLogger(__name__)).warning(
                "attention-map figure at step %d failed (%s); training continues", step, exc)
            return None
        before = set(figs.plt.get_fignums())
        try:
            figs.set_publication_style()
            m = maps.detach().cpu() if hasattr(maps, "detach") else torch.as_tensor(maps)
            if m.dim() == 2:                                        # (N, N) -> one layer, one head
                m = m[None, None]
            elif m.dim() == 3:                                      # (H, N, N) -> one layer
                m = m[None]
            if m.dim() != 4:
                raise ValueError(f"attention maps must be (L, H, N, N); got {tuple(m.shape)}")
            n_layers, n_heads = m.shape[0], m.shape[1]
            pos  = m[m > 0]                                         # shared log scale across all panels
            vmax = float(pos.max()) if pos.numel() else 1.0
            vmin = float(pos.min()) if pos.numel() else vmax * 1e-3
            attn_dir = self.run_dir / "attention"
            attn_dir.mkdir(exist_ok=True)
            paths = []
            for li in range(n_layers):
                for hi in range(n_heads):
                    path = attn_dir / f"step_{step}_layer{li}_head{hi}.png"
                    fig = figs.plot_attention_heatmap(
                        m[li, hi], log=True, vmin=vmin, vmax=vmax,
                        title=f"Attention (step {step}) - layer {li} head {hi}", path=str(path))
                    figs.plt.close(fig)
                    paths.append(path)
            return paths
        except Exception as exc:                                    # a viz error must never kill training
            for number in set(figs.plt.get_fignums()) - before:
                figs.plt.close(number)
            (logger or logging.getLogger(__name__)).warning(
                "attention-map figure at step %d failed (%s); training continues", step, exc)
            return None

    def save_gamma_attention_maps(
        self,
        step:   int,
        maps:   'Optional[torch.Tensor]',     # (H, N, N) per-head model-coupling gamma, or None (channel off)
        logger: Optional[logging.Logger] = None,
    ) -> Optional[List[Path]]:
        r"""Best-effort model-coupling (gamma) heatmaps for one periodic eval: one figure per head.

        The s-channel sibling of :meth:`save_attention_maps`. Writes
        ``attention/step_<N>_gamma_head<h>.png`` per head -- a LOG-scaled gamma_ij heatmap on the
        VIRIDIS color map (the belief beta channel uses magma) so the two channels read apart, on a
        scale shared across heads. ``maps`` is None when the model channel is inactive
        (``gamma_attention_maps`` returns None) -> no-op. A plotting error is logged and swallowed.
        """
        if maps is None:                                            # model channel inactive -> nothing to plot
            return None
        try:
            from vfe3.viz import figures as figs
        except Exception as exc:                                    # a viz error must never kill training
            (logger or logging.getLogger(__name__)).warning(
                "gamma-map figure at step %d failed (%s); training continues", step, exc)
            return None
        before = set(figs.plt.get_fignums())
        try:
            figs.set_publication_style()
            m = maps.detach().cpu() if hasattr(maps, "detach") else torch.as_tensor(maps)
            if m.dim() == 2:                                        # (N, N) -> one head
                m = m[None]
            if m.dim() != 3:
                raise ValueError(f"gamma maps must be (H, N, N); got {tuple(m.shape)}")
            n_heads = m.shape[0]
            pos  = m[m > 0]                                         # shared log scale across heads
            vmax = float(pos.max()) if pos.numel() else 1.0
            vmin = float(pos.min()) if pos.numel() else vmax * 1e-3
            attn_dir = self.run_dir / "attention"
            attn_dir.mkdir(exist_ok=True)
            paths = []
            for hi in range(n_heads):
                path = attn_dir / f"step_{step}_gamma_head{hi}.png"
                fig = figs.plot_attention_heatmap(
                    m[hi], log=True, vmin=vmin, vmax=vmax, cmap="viridis", symbol=r"\gamma",
                    title=f"Model-coupling attention (step {step}) - head {hi}", path=str(path))
                figs.plt.close(fig)
                paths.append(path)
            return paths
        except Exception as exc:                                    # a viz error must never kill training
            for number in set(figs.plt.get_fignums()) - before:
                figs.plt.close(number)
            (logger or logging.getLogger(__name__)).warning(
                "gamma-map figure at step %d failed (%s); training continues", step, exc)
            return None

    def save_checkpoint(
        self,
        step:      int,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg:       VFE3Config,

        *,
        scaler:               Optional['torch.amp.GradScaler'] = None,
        ema:                  Optional[EMA]                     = None,
        metropolis_generator: Optional[torch.Generator]         = None,
        data_state:            Optional[DataState]               = None,
    ) -> Path:
        r"""Write a resumable ``checkpoints/step_<N>.pt`` (model + optimizer + RNG + config + step).

        ``load_checkpoint`` reads this back to continue training: ``model_state`` and
        ``optimizer_state`` restore the weights, AdamW momentum, and successful-update scheduler
        clock; ``rng_state`` restores the
        CPU (and CUDA) generators for reproducible continuation, and ``step`` is the number of
        completed M-steps so the resumed run rebuilds the cosine ``LambdaLR`` at the right point.
        ``scaler`` (audit 2026-06-09 IE3): an ENABLED fp16 GradScaler's state (current scale +
        growth counters) is bundled so a resumed fp16 run does not restart at the init scale
        65536 and re-converge by skipped steps; a disabled/None scaler stores None.
        ``best_val_ppl``/``best_step`` (audit 2026-07-01 C2): the model-selection state is bundled
        so a resumed run reports the run-wide best, not just the continuation's best. The write is
        atomic (same-dir tmp + ``os.replace``) so a crash never leaves a corrupt ``step_<N>.pt``.
        ``metropolis_generator`` carries the private accept/reject stream independently of the
        global CPU/CUDA RNG so a resumed discrete-reflection sweep continues at the next draw.
        ``data_state`` records the current epoch's cursor and data/iterator identity. Its optional
        generator state is cloned for random sampling; deterministic sequential sampling stores null.
        """
        step = _require_nonnegative_int(step, "step")
        path = self.ckpt_dir / f"step_{step}.pt"
        rng_state = {
            "cpu":  torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        saved_data_state = None
        if data_state is not None:
            batches_consumed = _require_nonnegative_int(
                data_state["batches_consumed"], "batches_consumed")
            epoch = _require_nonnegative_int(data_state["epoch"], "epoch")
            generator_state = data_state["epoch_start_generator_state"]
            if generator_state is not None and not isinstance(generator_state, torch.Tensor):
                raise ValueError(
                    "data_state epoch_start_generator_state must be a tensor or null")
            data_identity = _normalized_data_identity(data_state.get("data_identity"))
            generator_state = _validate_epoch_generator_state(generator_state, data_identity)
            saved_data_state = {
                "epoch_start_generator_state": (generator_state.clone()
                                                  if generator_state is not None else None),
                "batches_consumed":            batches_consumed,
                "epoch":                       epoch,
                "data_identity":               data_identity,
            }
        # Portable best-model selection state (PB-03): a finite best_val_ppl means best_model.pt IS the
        # selected checkpoint, so its validated bundle is embedded here and travels with the checkpoint
        # across a cross-run-directory resume (older bundles carried only the scalar, whose file was left
        # behind). A finite best scalar without a readable, matching best_model.pt is an integrity error
        # and MUST prevent checkpoint publication; no validation best -> explicit empty selection state.
        if math.isfinite(float(self.best_val_ppl)):
            if not self.best_path.is_file():
                raise RuntimeError(
                    f"finite best_val_ppl ({float(self.best_val_ppl)}) but no readable best_model.pt "
                    f"at {self.best_path}; cannot embed a portable best-model bundle")
            best_model_bundle: Optional[Dict[str, object]] = _read_best_model_bundle(
                self.best_path, cfg, model.state_dict(), "cpu")
            saved_best_val_ppl = float(self.best_val_ppl)
            saved_best_step    = self.best_step
        else:
            best_model_bundle  = None
            saved_best_val_ppl = float("inf")
            saved_best_step    = None
        optimizer_state = optimizer.state_dict()
        optimizer_slot_manifest = _optimizer_populated_slot_manifest(optimizer_state)
        with _unique_sibling_temp(path) as tmp:
            torch.save({
                "step":            step,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer_state,
                "optimizer_populated_slot_manifest": optimizer_slot_manifest,
                "rng_state":       rng_state,
                "metropolis_rng_state": (metropolis_generator.get_state()
                                          if metropolis_generator is not None else None),
                "config":          asdict(cfg),
                "code_identity_sha256": self.code_identity_sha256,
                "selection_data_identity": self.selection_data_identity,
                "scaler_state":    (scaler.state_dict()
                                    if scaler is not None and scaler.is_enabled() else None),
                "ema_state":       (ema.state_dict() if ema is not None else None),
                "best_val_ppl":    saved_best_val_ppl,
                "best_step":       saved_best_step,
                "best_model_bundle": best_model_bundle,
                "data_state":      saved_data_state,
            }, tmp)
            _atomic_replace(path, tmp)
        return path


def _restore_best_selection(
    ckpt:                 Mapping[str, object],
    checkpoint_path:      Path,
    artifacts:            'RunArtifacts',
    expected_model_state: Mapping[str, torch.Tensor],
    map_location:         'str | torch.device',
    inherit_selection:    bool,
) -> None:
    r"""Restore portable best-model selection state into ``artifacts`` from a loaded checkpoint (PB-03).

    Precedence: (1) validate + publish a modern checkpoint's embedded ``best_model_bundle``; (2) for a
    legacy checkpoint lacking that field, validate ``<old_run>/best_model.pt`` (``old_run`` is
    ``checkpoint_path.parent.parent``) and publish it into the new run; (3) otherwise reset the
    selection state to empty, warning only when a finite-but-unreachable best scalar is being dropped.
    The best weights are only PUBLISHED (made reachable in the new run directory), never loaded into the
    live training model. After publication the scalar metadata is set and the file is required to exist.
    """
    import warnings

    embedded = ckpt.get("best_model_bundle")
    ckpt_best_ppl = ckpt.get("best_val_ppl")
    had_finite_best = ckpt_best_ppl is not None and math.isfinite(float(ckpt_best_ppl))

    if not inherit_selection:
        artifacts.best_val_ppl = float("inf")
        artifacts.best_step = None
        if had_finite_best:
            warnings.warn(
                f"resume from {checkpoint_path.name} carried a selected model measured under a "
                f"different or unbound code/config/validation-data contract; raw training state "
                f"was restored, but model selection restarts from this run.",
                UserWarning,
                stacklevel=3,
            )
        return

    # (1) Modern checkpoint carrying an embedded validated bundle.
    if isinstance(embedded, Mapping):
        _publish_best_model_bundle(embedded, expected_model_state, artifacts)
        artifacts.best_val_ppl = float(ckpt_best_ppl)
        artifacts.best_step    = ckpt.get("best_step")
        if not artifacts.best_path.is_file():
            raise RuntimeError(
                "best-model bundle is not reachable after publication into the resumed run")
        return

    # (2) Legacy checkpoint (no best_model_bundle field): import the sibling best_model.pt if reachable.
    if "best_model_bundle" not in ckpt and had_finite_best:
        old_best = checkpoint_path.parent.parent / "best_model.pt"
        if old_best.is_file():
            bundle = _read_best_model_bundle(
                old_best, artifacts.cfg, expected_model_state, map_location)
            _publish_best_model_bundle(bundle, expected_model_state, artifacts)
            artifacts.best_val_ppl = float(ckpt_best_ppl)
            artifacts.best_step    = ckpt.get("best_step")
            if not artifacts.best_path.is_file():
                raise RuntimeError(
                    "best-model bundle is not reachable after publication into the resumed run")
            return

    # (3) No reachable selected weights: reset to empty, never retaining an unreachable finite scalar.
    artifacts.best_val_ppl = float("inf")
    artifacts.best_step    = None
    if had_finite_best:
        warnings.warn(
            f"resume from {checkpoint_path.name} carried finite best-val metadata "
            f"(best_val_ppl={float(ckpt_best_ppl)}) but no reachable best-model weights (neither an "
            f"embedded bundle nor a sibling best_model.pt); dropping the unreachable selection state, "
            f"so model selection restarts from this run.",
            UserWarning,
            stacklevel=3,
        )


def _preflight_best_selection(
    ckpt:                 Mapping[str, object],
    checkpoint_path:      Path,
    cfg:                  Optional[VFE3Config],
    expected_model_state: Mapping[str, torch.Tensor],
    map_location:         'str | torch.device',
    saved_step:           int,
    expected_code_identity: str,
    expected_selection_data_identity: Optional[Mapping[str, object]],
) -> bool:
    """Validate selection scalars and all reachable selected weights before live restoration."""
    if "best_val_ppl" not in ckpt or "best_step" not in ckpt:
        raise RuntimeError("checkpoint best-model selection metadata is missing")
    best_ppl = ckpt["best_val_ppl"]
    best_step = ckpt["best_step"]
    embedded_present = "best_model_bundle" in ckpt
    embedded = ckpt.get("best_model_bundle")
    if (not isinstance(best_ppl, Real) or isinstance(best_ppl, bool)
            or math.isnan(float(best_ppl)) or float(best_ppl) == float("-inf")):
        raise RuntimeError("checkpoint best-model selection PPL must be finite or positive infinity")
    selected = math.isfinite(float(best_ppl))
    candidate: Optional[Dict[str, object]] = None
    if selected:
        if float(best_ppl) <= 0.0:
            raise RuntimeError("checkpoint best-model selection PPL must be positive")
        if type(best_step) is not int or best_step < 0 or best_step > saved_step:
            raise RuntimeError(
                "checkpoint best-model selection step must be a non-negative integer no later "
                "than the checkpoint step")
        if embedded_present:
            if not isinstance(embedded, Mapping):
                raise RuntimeError(
                    "checkpoint best-model selection has no reachable embedded weight bundle")
            candidate = _validate_best_model_mapping(
                embedded, None, expected_model_state,
                f"checkpoint {checkpoint_path} best-model selection bundle",
            )
        else:
            sibling = checkpoint_path.parent.parent / "best_model.pt"
            if not sibling.is_file():
                raise RuntimeError(
                    "checkpoint best-model selection has no reachable legacy sibling weights")
            bundle = torch.load(sibling, map_location=map_location, weights_only=True)
            candidate = _validate_best_model_mapping(
                bundle, None, expected_model_state,
                f"checkpoint {checkpoint_path} legacy best-model selection bundle",
            )
    elif best_step is not None or embedded is not None:
        raise RuntimeError(
            "checkpoint empty best-model selection must use best_step=None and no weight bundle")
    if not selected or candidate is None:
        return False
    if not _best_selection_is_portable(
            candidate, cfg, expected_code_identity, expected_selection_data_identity):
        return False
    checkpoint_code_identity = ckpt.get("code_identity_sha256")
    checkpoint_selection_identity = ckpt.get("selection_data_identity")
    if (checkpoint_code_identity != candidate.get("code_identity_sha256")
            or checkpoint_selection_identity is None):
        return False
    try:
        if (_normalized_data_identity(checkpoint_selection_identity)
                != _normalized_data_identity(candidate["selection_data_identity"])):
            return False
    except (TypeError, ValueError):
        return False
    return True


def _preflight_resume_config(
    saved_config: object,
    cfg:          VFE3Config,
) -> Tuple[Mapping[str, object], List[str]]:
    """Validate serialized config keys and compute deterministic drift before any live mutation."""
    if not isinstance(saved_config, Mapping):
        raise RuntimeError("checkpoint config must be a mapping when cfg is supplied")
    if any(not isinstance(key, str) for key in saved_config):
        raise RuntimeError("checkpoint config keys must be strings")
    known = {field.name for field in fields(VFE3Config)}
    unknown = sorted(set(saved_config) - known)
    if unknown:
        raise RuntimeError(f"checkpoint config contains unknown field(s) {unknown}")
    current = asdict(cfg)
    drift = sorted(
        key for key in (set(saved_config) | set(current))
        if key not in ("resume_from", "trust_resume_checkpoint")
        and saved_config.get(key) != current.get(key)
    )
    return saved_config, drift


def load_checkpoint(
    path:      'str | Path',
    model:     torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,

    *,
    map_location:         'Optional[str | torch.device]'   = None,
    max_step:             Optional[int]                    = None,
    restore_rng:          bool                             = True,
    scaler:               Optional['torch.amp.GradScaler'] = None,
    cfg:                  Optional[VFE3Config]             = None,
    ema:                  Optional[EMA]                    = None,
    artifacts:            'Optional[RunArtifacts]'         = None,
    metropolis_generator: Optional[torch.Generator]        = None,
    data_state:            Optional[DataStateBuffer]        = None,
    expected_data_identity: Optional[Mapping[str, object]]   = None,
    expected_selection_data_identity: Optional[Mapping[str, object]] = None,
    expected_steps_per_epoch: Optional[int]                  = None,
) -> int:
    r"""Restore a ``save_checkpoint`` bundle into ``model`` (and optionally ``optimizer``); return the saved step.

    This is the LOAD half of the resumable checkpoint. It always restores the model weights;
    it restores the AdamW optimizer state (momentum buffers + per-parameter step counts) when an
    ``optimizer`` is supplied, then reapplies that optimizer's current non-parameter group metadata
    so the current config remains authoritative. The CPU/CUDA RNG and the optional private
    ``metropolis_generator`` are restored when ``restore_rng`` is set and the bundle carries their
    states (older checkpoints simply skip absent RNG fields). The returned integer is the number of
    completed M-steps; ``train(resume_from=...)`` uses it to rebuild the cosine ``LambdaLR`` at the
    saved step and to start the loop from there.

    ``scaler`` (audit 2026-06-09 IE3): when given AND the bundle carries a saved scaler state,
    the fp16 GradScaler's scale/growth counters are restored (bundles written before the scaler
    was persisted, or written from a non-fp16 run, simply skip the step). ``cfg`` (audit IE4):
    when given, the CURRENT config is compared against the bundle's saved config and any
    differing fields are warned about -- strict ``load_state_dict`` already catches
    shape-changing divergence, but shape-preserving semantic drift (LR schedule, n_e_steps,
    e_*_lr, ...) would otherwise pass silently. ``artifacts`` (audit 2026-07-01 C2, PB-03): when
    given, the portable best-model selection state is restored into it -- a modern checkpoint's
    embedded, validated ``best_model_bundle`` (or, for a legacy checkpoint, the sibling
    ``<old_run>/best_model.pt``) is published into the resumed run's directory and the
    ``best_val_ppl``/``best_step`` scalars are set; when no reachable bundle exists the selection state
    resets to ``inf``/``None`` with one warning, so an unreachable best scalar is never retained. When a
    mutable ``data_state`` mapping is supplied, it is filled from the bundled iterator cursor.
    Supplying ``expected_data_identity`` requests exact data resume: a legacy bundle without
    ``data_state`` or a bundle whose identity differs is rejected before any mutable state is restored.

    The bundle is loaded with ``weights_only=True`` by default, which refuses to execute arbitrary
    pickle reductions: our bundle carries only tensors, an ``asdict`` config dict, and RNG tensors
    (no custom classes), so it loads safely under it (matching
    ``test_run_artifacts.py::test_save_checkpoint_is_loadable``). A bundle that fails the safe load
    (e.g. an older format) RAISES unless ``cfg.trust_resume_checkpoint`` is set, which falls back to
    the legacy ``weights_only=False`` load -- only use that for a checkpoint you trust, since that
    path can execute arbitrary code embedded in the pickle.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")
    if map_location is None:
        map_location = next(model.parameters()).device
    # A checkpoint mixes parameter tensors with CPU-only state: torch CPU RNG, DataLoader and
    # Metropolis generator states, and non-fused/non-capturable Adam step scalars. Loading the whole
    # bundle onto a CUDA target corrupts those storage contracts before validation. Validate from a
    # CPU snapshot; model.load_state_dict and optimizer.load_state_dict perform the deliberate
    # parameter-slot transfer to the live model device below.
    checkpoint_load_location = "cpu"
    trust = bool(getattr(cfg, "trust_resume_checkpoint", False))
    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location=checkpoint_load_location,
            weights_only=True,
        )
    except Exception as exc:                                    # safe load rejected a non-tensor object
        if not trust:
            raise RuntimeError(
                f"checkpoint {Path(path).name} could not be loaded under the safe weights_only=True "
                f"path ({type(exc).__name__}: {exc}). If you trust this file, set "
                f"trust_resume_checkpoint=True to allow the legacy weights_only=False load (which can "
                f"execute arbitrary code embedded in the pickle)."
            ) from exc
        ckpt = torch.load(
            checkpoint_path,
            map_location=checkpoint_load_location,
            weights_only=False,
        )
    if not isinstance(ckpt, Mapping):
        raise RuntimeError("checkpoint payload must be a mapping")
    saved_step = _require_nonnegative_int(ckpt.get("step"), "step")
    if max_step is not None:
        max_step = _require_nonnegative_int(max_step, "max_step")
        if saved_step > max_step:
            raise ValueError(
                f"checkpoint step {saved_step} exceeds requested n_steps={max_step} "
                f"(max_step={max_step})")
    saved_data_state = ckpt.get("data_state")
    saved_data_identity: Optional[Dict[str, object]] = None
    if saved_data_state is None and expected_data_identity is not None:
        raise RuntimeError(
            "checkpoint is missing data_state required for exact resume; refusing to restore "
            "model, optimizer, cursor, or RNG state")
    if saved_data_state is not None:
        if not isinstance(saved_data_state, Mapping):
            raise RuntimeError("checkpoint data_state must be a mapping")
        saved_batches_consumed = _require_nonnegative_int(
            saved_data_state["batches_consumed"], "batches_consumed")
        saved_epoch = _require_nonnegative_int(saved_data_state["epoch"], "epoch")
        saved_generator_state = saved_data_state.get("epoch_start_generator_state")
        if (saved_generator_state is not None
                and not isinstance(saved_generator_state, torch.Tensor)):
            raise RuntimeError(
                "checkpoint data_state epoch_start_generator_state must be a tensor or null")
        if "data_identity" not in saved_data_state:
            raise RuntimeError(
                "checkpoint data_state is missing its data identity contract; exact resume is unsafe")
        saved_data_identity = _normalized_data_identity(saved_data_state["data_identity"])
        if expected_data_identity is None:
            raise RuntimeError(
                "checkpoint carries a data cursor, but no live expected data identity was supplied")
        live_data_identity = _normalized_data_identity(expected_data_identity)
        if saved_data_identity != live_data_identity:
            differing = sorted(
                key for key in (saved_data_identity.keys() | live_data_identity.keys())
                if saved_data_identity.get(key) != live_data_identity.get(key)
            )
            raise RuntimeError(
                f"checkpoint data identity mismatch for field(s) {differing}; refusing to restore "
                "model, optimizer, cursor, or RNG state")
        saved_generator_state = _validate_epoch_generator_state(
            saved_generator_state, live_data_identity)
        if expected_steps_per_epoch is None:
            iterator_identity = live_data_identity["iterator"]
            n_samples = iterator_identity["sampler_num_samples"]
            batch_size = iterator_identity["batch_size"]
            expected_steps_per_epoch = (
                n_samples // batch_size
                if iterator_identity["drop_last"]
                else (n_samples + batch_size - 1) // batch_size
            )
        if type(expected_steps_per_epoch) is not int or expected_steps_per_epoch <= 0:
            raise RuntimeError("exact resume requires a positive integer steps-per-epoch")
        if saved_batches_consumed > expected_steps_per_epoch:
            raise RuntimeError(
                "checkpoint batches_consumed exceeds the live loader epoch length")
        expected_step = saved_epoch * expected_steps_per_epoch + saved_batches_consumed
        if saved_step != expected_step:
            raise RuntimeError(
                f"checkpoint step {saved_step} is inconsistent with epoch {saved_epoch} and "
                f"batches_consumed {saved_batches_consumed} for "
                f"{expected_steps_per_epoch} steps per epoch")
    saved_model_state = _validate_checkpoint_model_state(
        ckpt.get("model_state"), model.state_dict(), checkpoint_path)
    saved_optimizer_state = None
    successful_updates = None
    if optimizer is not None:
        saved_optimizer_state = _validate_optimizer_state(
            ckpt.get("optimizer_state"), optimizer, saved_step,
            ckpt.get("optimizer_populated_slot_manifest"))
        successful_updates = _validated_saved_successful_updates(
            saved_optimizer_state["param_groups"], saved_step)
    saved_scaler_state = ckpt.get("scaler_state")
    if scaler is not None and scaler.is_enabled():
        saved_scaler_state = _validate_scaler_state(saved_scaler_state, scaler)
    saved_ema_state = ckpt.get("ema_state")
    if ema is not None:
        saved_ema_state = _validate_ema_state(
            saved_ema_state,
            ema,
            require_state=(expected_data_identity is not None),
        )
    saved_rng_state = None
    if restore_rng:
        saved_rng_state = _validate_rng_state(ckpt.get("rng_state"))
    saved_metropolis_state = None
    if restore_rng and metropolis_generator is not None:
        saved_metropolis_state = _validate_generator_state(
            ckpt.get("metropolis_rng_state"), "metropolis_rng_state")
    active_selection_cfg = cfg if cfg is not None else (
        artifacts.cfg if artifacts is not None else None)
    if expected_selection_data_identity is None and artifacts is not None:
        expected_selection_data_identity = artifacts.selection_data_identity
    expected_code_identity = (
        artifacts.code_identity_sha256 if artifacts is not None else _package_code_identity())
    inherit_selection = _preflight_best_selection(
        ckpt,
        checkpoint_path,
        active_selection_cfg,
        model.state_dict(),
        map_location,
        saved_step,
        expected_code_identity,
        expected_selection_data_identity,
    )
    saved_config = None
    config_drift: List[str] = []
    if cfg is not None:
        saved_config, config_drift = _preflight_resume_config(ckpt.get("config"), cfg)

    model.load_state_dict(saved_model_state)
    if optimizer is not None:
        fresh = [{k: v for k, v in group.items() if k != "params"}
                 for group in optimizer.param_groups]
        optimizer.load_state_dict(saved_optimizer_state)
        for group, metadata in zip(optimizer.param_groups, fresh):
            params = group["params"]
            group.clear()
            group.update(metadata)
            group["params"] = params
            if successful_updates is not None:
                group["successful_updates"] = successful_updates
    if scaler is not None and scaler.is_enabled():
        scaler.load_state_dict(saved_scaler_state)
    # EMA shadow: restore it so a resumed run continues the SAME running average instead of re-seeding
    # from the resumed iterate. When the bundle carries no ema_state (a use_ema=False or legacy
    # checkpoint), the shadow was constructed from the PRE-load fresh init (EMA is built before this
    # load overwrites the model), so reseed it from the just-loaded weights -- otherwise the running
    # average blends real weights into random-init noise (audit 2026-07-01 C3).
    if ema is not None:
        if saved_ema_state is None:
            ema.reset(model)
        else:
            ema.load_state_dict(saved_ema_state)
    # Portable best-val model-selection state (PB-03, extends audit 2026-07-01 C2): restore
    # best_val_ppl/best_step AND make the selected weights reachable in the resumed run's directory,
    # so a cross-run_dir resume no longer restores best metadata whose best_model.pt is missing. The
    # scalar is never retained without reachable weights (audit m26). By precedence:
    #   (1) a modern checkpoint's embedded best_model_bundle is validated and published into the new run;
    #   (2) a legacy checkpoint (no such field) validates <old_run>/best_model.pt and publishes it;
    #   (3) neither -> the selection state resets to empty, and any unreachable finite scalar is dropped.
    if artifacts is not None:
        _restore_best_selection(
            ckpt, checkpoint_path, artifacts, model.state_dict(), map_location,
            inherit_selection)
    if cfg is not None:
        if config_drift:
            import warnings
            warnings.warn(
                f"resume config drift: the checkpoint at {Path(path).name} was written under a "
                f"different config for field(s) {config_drift}; the resumed run uses the CURRENT values "
                f"(weights/optimizer load strictly, but semantic knobs are not restored from the "
                f"bundle).",
                UserWarning,
                stacklevel=2,
            )
    if restore_rng:
        rng = saved_rng_state
        # RNG tensors must be CPU ByteTensors regardless of map_location (set_rng_state asserts this).
        torch.set_rng_state(rng["cpu"].cpu() if hasattr(rng["cpu"], "cpu") else rng["cpu"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    if restore_rng and metropolis_generator is not None:
        metro_state = saved_metropolis_state
        metropolis_generator.set_state(
            metro_state.cpu() if hasattr(metro_state, "cpu") else metro_state)
    if data_state is not None:
        data_state.clear()
        if saved_data_state is not None:
            data_state.update({
                "epoch_start_generator_state": saved_data_state["epoch_start_generator_state"],
                "batches_consumed":            saved_batches_consumed,
                "epoch":                       saved_epoch,
                "data_identity":               saved_data_identity,
            })
    return saved_step


def _git_environment(
    git_executable: str,
) -> Dict[str, str]:
    r"""Minimal noninteractive environment for bounded Git provenance probes."""
    env = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": str(Path(git_executable).resolve().parent),
    }
    for name in ("COMSPEC", "PATHEXT", "SYSTEMROOT", "WINDIR"):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


def _git_code_identity(
    root: Optional[Path] = None,
) -> Dict[str, object]:
    r"""Return HEAD plus an exact dirty-tree fingerprint, or a persisted probe error."""
    repo = Path(__file__).resolve().parent.parent if root is None else Path(root).resolve()
    identity: Dict[str, object] = {
        "git_sha":               None,
        "git_dirty":             None,
        "git_dirty_fingerprint": None,
    }
    try:
        git_executable = shutil.which("git")
        if git_executable is None:
            raise FileNotFoundError("git executable was not found on PATH")
        env = _git_environment(git_executable)

        def _git(*args: str) -> bytes:
            return subprocess.check_output(
                [git_executable,
                 "-c", "core.fsmonitor=false",
                 "-c", f"safe.directory={repo.as_posix()}",
                 *args],
                cwd=str(repo),
                env=env,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )

        identity["git_sha"] = _git("rev-parse", "HEAD").decode("ascii").strip()
        status = _git("status", "--porcelain=v1", "-z", "--untracked-files=all")
        identity["git_dirty"] = bool(status)
        if status:
            diff = _git("diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--")
            untracked = _git("ls-files", "--others", "--exclude-standard", "-z")
            digest = hashlib.sha256()
            digest.update(b"status\0")
            digest.update(status)
            digest.update(b"\0diff\0")
            digest.update(diff)
            digest.update(b"\0untracked\0")
            for raw_name in (name for name in untracked.split(b"\0") if name):
                path = repo / os.fsdecode(raw_name)
                digest.update(raw_name)
                digest.update(b"\0")
                digest.update(bytes.fromhex(_sha256_file_content(path)))
            identity["git_dirty_fingerprint"] = digest.hexdigest()
    except Exception as exc:
        identity["git_sha"] = None
        identity["git_dirty"] = None
        identity["git_dirty_fingerprint"] = None
        identity["git_error"] = repr(exc)
    return identity


def _sha256_file_content(
    path: Path,

    *,
    chunk_bytes: int = 1024 * 1024,
) -> str:
    """Hash a file without materializing it in memory."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor_content(
    tokens: torch.Tensor,

    *,
    chunk_tokens: int = 128 * 1024,
) -> str:
    r"""Hash token values canonically as int64 using bounded device-to-host chunks."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    flat = tokens.detach().reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), chunk_tokens):
        chunk = flat[start:start + chunk_tokens].to(device="cpu", dtype=torch.long).contiguous()
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def _loader_token_content_identity(
    loader: Optional[Iterable],
) -> 'Tuple[Optional[str], Optional[int]]':
    r"""Return one immutable loader split's canonical token digest and count.

    ``ablation.get_loader`` reuses the same ``TokenWindows`` instance across cells. Cache the
    canonical int64 digest on that dataset after its first bounded hash so every later finalization
    reuses the exact value instead of rereading the unchanged mapped corpus.
    """
    dataset = getattr(loader, "dataset", None)
    tokens = getattr(dataset, "tokens", None)
    if tokens is None:
        return None, None
    cached = getattr(dataset, "_vfe3_token_content_sha256", None)
    if cached is None:
        cached = _sha256_tensor_content(tokens)
        setattr(dataset, "_vfe3_token_content_sha256", cached)
    return str(cached), int(tokens.numel())


def _bincount_token_chunks(
    tokens: torch.Tensor,

    *,
    vocab_size:   int,
    chunk_tokens: int = 128 * 1024,
) -> torch.Tensor:
    r"""Accumulate exact training-token counts with a bounded host-int64 working chunk."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    flat = tokens.detach().reshape(-1)
    counts = torch.zeros(vocab_size, dtype=torch.long, device="cpu")
    for start in range(0, flat.numel(), chunk_tokens):
        chunk = flat[start:start + chunk_tokens].to(device="cpu", dtype=torch.long)
        partial = torch.bincount(chunk, minlength=vocab_size)
        if partial.numel() != vocab_size:
            raise ValueError(
                f"training token id exceeds vocab_size={vocab_size}; validate the cache first")
        counts.add_(partial)
    return counts


def _write_provenance(
    artifacts: RunArtifacts,
    cfg:       VFE3Config,
    model:     torch.nn.Module,
    logger:    logging.Logger,

    *,
    train_loader:  Optional[Iterable] = None,
    val_loader:    Optional[Iterable] = None,
    test_loader:   Optional[Iterable] = None,
    data_seed:     Optional[int]      = None,
    max_tokens:    Optional[int]      = None,
    tokenizer_tag: Optional[str]      = None,
) -> None:
    r"""Write code, environment, per-split data, and data-order provenance best-effort."""

    prov: Dict[str, object] = {
        "seed":                cfg.seed,
        "deterministic_state": deterministic_state(),
        "n_params":            int(sum(p.numel() for p in model.parameters())),
        "torch_version":       torch.__version__,
        "cuda_version":        torch.version.cuda,
        "device_name":         (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "data_seed":           (int(data_seed) if data_seed is not None else None),
        "max_tokens":          (int(max_tokens) if max_tokens is not None else None),
        "tokenizer_tag":       tokenizer_tag,
    }
    prov.update(_git_code_identity())
    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        sha_key = f"{split}_data_sha256"
        n_key = f"{split}_data_n_tokens"
        prov[sha_key], prov[n_key] = None, None
        try:
            # Hash the CONTENT, not the storage: TokenWindows may hold the stream in its native
            # cache dtype (int32 memmap) or int64 (capped load). Normalize to int64 once, then reuse
            # that immutable split digest across every ablation cell sharing the loader.
            prov[sha_key], prov[n_key] = _loader_token_content_identity(loader)
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError, MemoryError) as exc:
            # Best-effort provenance, narrowed to the realistic hash-path failures (audit
            # 2026-07-12 N2): exotic loader/dtype/allocation errors must not crash finalize, but
            # the failure is LOGGED -- previously a bare `except Exception: pass` silently
            # recorded null data hashes, indistinguishable from a loader without corpus tokens --
            # while a programming error (NameError/KeyError/...) now surfaces instead of
            # masquerading as a missing corpus.
            logger.warning("%s-split provenance data hash failed (%s); recording null", split, exc)
    # Backward-compatible held-out aliases consumed by existing scaling-analysis artifacts.
    prov["data_sha256"] = prov["test_data_sha256"]
    prov["data_n_tokens"] = prov["test_data_n_tokens"]
    artifacts.save_json("provenance.json", prov)
    logger.info("wrote provenance.json (git_sha=%s dirty=%s)", prov.get("git_sha"), prov.get("git_dirty"))


@torch.no_grad()
def _calibration_and_strata(
    corpus_counts: torch.Tensor,             # (V,) training-corpus unigram counts

    model:         torch.nn.Module,
    test_loader:   Iterable,
    device:        torch.device,

    *,
    max_batches: int = 20,
    n_bins:      int = 15,
) -> Dict[str, object]:
    r"""Decode calibration (ECE + reliability curve) and corpus-frequency-stratified CE over the test
    split. The decode is non-standard (KL-to-prior Mahalanobis or mu @ W^T with Sigma feeding the
    logit scale), so a mis-scaled ``decode_log_scale`` can leave PPL acceptable while the probability
    mass is wrong -- PPL alone cannot catch it. Bucket cutoffs are quantiles over the positive-count
    token types in the complete training corpus; sampled target duplication cannot move them, and
    evaluation targets unseen in training are rare. The aggregated values remain sampled held-out CE.
    The strata expose prior-table tail stagnation. Off-graph; capped at ``max_batches``."""
    import torch.nn.functional as F

    confs, corrects, nats, tgts = [], [], [], []
    for i, (tok, tgt) in enumerate(test_loader):
        tok, tgt = tok.to(device), tgt.to(device)
        logits = model(tok)                                     # (B, N, V) inference path
        lp = logits.reshape(-1, logits.shape[-1]).float()
        t = tgt.reshape(-1)
        valid = t != -100
        lp, t = lp[valid], t[valid]
        prob = torch.softmax(lp, dim=-1)
        p_max, pred = prob.max(dim=-1)
        confs.append(p_max)
        corrects.append((pred == t).float())
        nats.append(F.cross_entropy(lp, t, reduction="none"))
        tgts.append(t)
        if i + 1 >= max_batches:
            break
    if not confs:
        return {}
    conf, corr = torch.cat(confs), torch.cat(corrects)
    nat, tg = torch.cat(nats), torch.cat(tgts)

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=conf.device)
    ece, rel = 0.0, []
    for b in range(n_bins):                                     # expected calibration error (15-bin)
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.any():
            acc, cf, w = corr[m].mean(), conf[m].mean(), m.float().mean()
            ece += float(w * (acc - cf).abs())
            rel.append({"conf": float(cf), "acc": float(acc), "frac": float(w)})
    if corpus_counts.ndim != 1:
        raise ValueError("corpus_counts must be a one-dimensional training-corpus bincount")
    counts = corpus_counts.to(device=tg.device)
    if int(tg.max()) >= counts.numel():
        raise ValueError("corpus_counts does not cover every sampled evaluation target")
    positive_counts = counts[counts > 0].float()
    if positive_counts.numel() == 0:
        q1, q2 = 0.0, 0.0
    else:
        quantiles = positive_counts.new_tensor([1 / 3, 2 / 3])
        q1, q2 = torch.quantile(positive_counts, quantiles).tolist()
    tok_count = counts[tg].float()                              # training-corpus count of each target
    seen = tok_count > 0
    strata = {}
    for name, mask in (("rare", (~seen) | (tok_count <= q1)),
                       ("mid", seen & (tok_count > q1) & (tok_count <= q2)),
                       ("frequent", seen & (tok_count > q2))):
        strata[name] = float(nat[mask].mean()) if mask.any() else float("nan")
    return {"ece": ece, "reliability": rel, "overall_ce": float(nat.mean()),
            "corpus_freq_strata_ce": strata}


def _fd_gradient_check(
    model:       torch.nn.Module,
    test_loader: Iterable,
    device:      torch.device,

    *,
    n_coords:    int   = 4,
    fd_eps:      float = 1e-3,
) -> float:
    r"""Worst relative error between autograd-of-CE and a central finite difference on a few DECODE
    coordinates (``output_proj_weight``, else the decode log-scale) -- a parameter whose gradient does
    NOT pass through the E-step belief adjoint (which the default kernel/oracle route detaches), so a
    healthy model reads ~1e-4 and a broken decode adjoint spikes far above it. (Probing ``mu_embed``
    instead would sit at the detached-oracle's ~10-25% plateau with no headroom to flag a real bug.)
    The CLAUDE.md-mandated FD-vs-autograd check. Best-effort on one tiny batch; restores every coord."""
    batch = next(iter(test_loader))
    tok, tgt = (batch if isinstance(batch, (tuple, list)) else (batch, None))
    tok = tok[:2].to(device)
    tgt = tgt[:2].to(device)
    pb = model.prior_bank
    p = pb.output_proj_weight if getattr(pb, "output_proj_weight", None) is not None else pb.decode_log_scale
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(tok, tgt)
    loss.backward()
    if p.grad is None:                                          # decode param severed under this config
        model.zero_grad(set_to_none=True)
        return float("nan")
    flat, gflat = p.detach().view(-1), p.grad.detach().view(-1).clone()
    # Probe the LARGEST-gradient coords, not random ones: a random coord usually has near-zero
    # gradient where the central difference is dominated by fp rounding (a spurious large rel error),
    # so this checks the gradient where its signal actually dominates -- where a real adjoint bug shows.
    idx = torch.topk(gflat.abs(), min(n_coords, gflat.numel())).indices.tolist()
    worst = 0.0
    with torch.no_grad():
        for j in idx:
            orig = float(flat[j])
            # try/finally (audit 2026-07-12 N1): `flat` is a storage-sharing view of the LIVE
            # decode parameter, and the caller catches broadly -- a forward that raises between
            # the +/-eps writes must not leave the parameter perturbed for subsequent probes.
            try:
                flat[j] = orig + fd_eps
                _, lp, _ = model(tok, tgt)
                flat[j] = orig - fd_eps
                _, lm, _ = model(tok, tgt)
            finally:
                flat[j] = orig
            fd = (float(lp) - float(lm)) / (2.0 * fd_eps)
            ana = float(gflat[j])
            worst = max(worst, abs(fd - ana) / max(abs(fd), abs(ana), 1e-8))
    model.zero_grad(set_to_none=True)
    return worst


def _write_research_artifacts(
    model:       torch.nn.Module,
    artifacts:   RunArtifacts,
    cfg:         VFE3Config,
    train_loader: Optional[Iterable],
    test_loader: Optional[Iterable],
    device:      torch.device,
    logger:      logging.Logger,
) -> None:
    r"""Best-effort ``research.json``: decode calibration (ECE) + frequency-stratified loss + the FD
    gradient-check residual. Each probe is independently guarded so one failure never blocks the
    others or the saved numeric results."""
    if test_loader is None:
        return
    out: Dict[str, object] = {}
    try:
        train_dataset = getattr(train_loader, "dataset", None)
        train_tokens = getattr(train_dataset, "tokens", None)
        if train_tokens is None:
            raise ValueError("training loader dataset does not expose corpus tokens")
        corpus_counts = _bincount_token_chunks(
            train_tokens,
            vocab_size=int(cfg.vocab_size),
        )
        out.update(_calibration_and_strata(corpus_counts, model, test_loader, device))
    except Exception as exc:
        logger.warning("calibration/strata probe failed (%s); skipped", exc)
    try:
        out["fd_gradient_worst_rel_error"] = _fd_gradient_check(model, test_loader, device)
        logger.info("FD gradient-check worst rel error: %.2e", out["fd_gradient_worst_rel_error"])
    except Exception as exc:
        logger.warning("FD gradient-check failed (%s); skipped", exc)
    # B1/EXP-3 Sigma_q calibration headline: Spearman rho(tr Sigma_q, CE) and the across-token
    # spread gate CV(tr Sigma_q) > 0.10 (below it the covariance channel is inert -- reported as
    # such, NOT miscoded as "decode doesn't matter"). Off-graph; capped at a few batches.
    try:
        from vfe3.viz.extract import belief_ce_bank
        from vfe3 import metrics as _cal_metrics
        bank = belief_ce_bank(model, test_loader, device=device, max_batches=10)
        tr = bank["tr_sigma"]
        if tr.numel() >= 2:
            out["sigma_ce_spearman"] = _cal_metrics.spearman_rho(tr, bank["ce"])
            out["sigma_trace_cv"] = _cal_metrics.cv(tr)
            out["sigma_trace_cv_gate_pass"] = bool(out["sigma_trace_cv"] > 0.10)
            logger.info("Sigma_q calibration: rho(trSigma,CE)=%.3f CV(trSigma)=%.3f (gate>0.10: %s)",
                        out["sigma_ce_spearman"], out["sigma_trace_cv"], out["sigma_trace_cv_gate_pass"])
    except Exception as exc:
        logger.warning("Sigma_q calibration probe failed (%s); skipped", exc)
    if out:
        artifacts.save_json("research.json", out)


def _cost_model_fields(
    model:       torch.nn.Module,
    cfg:         VFE3Config,

    n_params:    int,
    tokens_seen: int,

    *,
    wall_time:   Optional[float] = None,
) -> Dict[str, object]:
    r"""Structural axes + a faithful compute proxy for the scaling frontier (extends scaling_point).

    The ``6ND`` rule (``6 * n_params * tokens_seen``) is LOOSE here: ``n_params`` is dominated by the
    vocab-size gauge/prior tables (``phi_embed`` is ``V * n_gen``), but only the active tokens' rows
    participate per step, while the decode reads all ``V`` rows every forward. So this records
    (a) the structural axes that set the real per-token work, (b) ``active_params_per_token`` -- the
    honest working set (decode-bound, ~``K``, NOT ``phi``/``n_gen``-bound, the mirror image of
    ``n_params`` being ``n_gen``-dominated), and (c) a transparent analytic FLOP proxy assembled from
    those drivers (order-1 constants; for a calibrated frontier use ``wall_time`` or a profiler).
    ``wall_time`` on a fixed GPU is the empirical ground truth the analytic constants calibrate
    against. ``n_gen`` / ``n_blocks`` are read from the GROUP OBJECT so they track ``cross_couplings``,
    bracket closure, and the ``so_n``/``sp_n`` decoupling of ``n_gen`` from ``K``.
    """
    V, K = int(cfg.vocab_size), int(cfg.embed_dim)
    n_gen = int(model.group.generators.shape[0])
    n_blocks = max(1, len(model.group.irrep_dims))
    d_head = K / n_blocks                                        # representative block dim
    model_channel = (cfg.lambda_h > 0.0 or cfg.lambda_gamma > 0.0
                     or cfg.prior_source == "model_channel" or cfg.s_e_step)
    # ACTIVE params per token: the single looked-up belief row is always 2K+n_gen. The decoder then
    # reads EITHER the prior-bank mean/variance rows (2VK) OR the linear output matrix (VK) and its
    # optional V-vector bias. The V*n_gen phi bulk is not touched by either full-vocabulary readout.
    token_row = 2 * K + n_gen
    if cfg.use_prior_bank:
        decode_readout = 2 * V * K
    else:
        decode_readout = V * K + (V if cfg.decode_bias else 0)
    active = token_row + decode_readout
    if model_channel:
        active += 2 * V * K                                     # s tables enter encode/decode
    # Transparent analytic FLOP proxy. Per token: decode over all V (2VK), L*T belief E-step
    # iterations, and one T-iteration model-channel refinement when s_e_step is enabled. Each E-step
    # iteration has O(N) attention energy (2NK) plus O(N) transport application (2N*d_head^2).
    # Constants are O(1); this is a proxy, not a calibrated count.
    L, T, N = int(cfg.n_layers), int(cfg.n_e_steps), int(cfg.max_seq_len)
    fpt_decode         = 2.0 * V * K
    estep_kernel       = 2.0 * N * K + 2.0 * N * d_head * d_head
    belief_estep       = L * T * estep_kernel
    model_estep        = T * estep_kernel if cfg.s_e_step else 0.0
    fpt_estep          = belief_estep + model_estep
    est_flops_analytic = (fpt_decode + fpt_estep) * float(tokens_seen)
    out: Dict[str, object] = {
        "embed_dim":               K,
        "n_heads":                 int(cfg.n_heads),
        "n_blocks":                n_blocks,
        "n_gen":                   n_gen,
        "n_layers":                L,
        "n_e_steps":               T,
        "max_seq_len":             N,
        "batch_size":              int(cfg.batch_size),
        "diagonal_covariance":     bool(cfg.diagonal_covariance),
        "gauge_group":             cfg.gauge_group,
        "use_prior_bank":          bool(cfg.use_prior_bank),
        "model_channel_active":    bool(model_channel),
        "vocab_size":              V,
        "n_learnable_params":      int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "active_params_per_token": int(active),
        "est_flops_analytic":      est_flops_analytic,
        "flops_per_token_decode":  fpt_decode,
        "flops_per_token_estep":   fpt_estep,
        "device_name":             (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "amp_dtype":               cfg.amp_dtype,
    }
    if wall_time is not None and tokens_seen > 0:
        out["wall_time_s"]         = float(wall_time)
        out["wall_time_per_token"] = float(wall_time) / float(tokens_seen)
        out["wall_time_per_step"]  = float(wall_time) / max(1, int(cfg.max_steps))
    return out


def finalize_run(
    model:       torch.nn.Module,
    artifacts:   RunArtifacts,
    cfg:         VFE3Config,

    *,
    tokens_per_char: Optional[float]           = None,  # None -> BPC unavailable; BPT remains defined
    train_loader:    Optional[Iterable]       = None,
    val_loader:      Optional[Iterable]       = None,
    test_loader:     Optional[Iterable]       = None,
    losses:          Optional[List[float]]    = None,
    data_seed:       Optional[int]            = None,
    max_tokens:      Optional[int]            = None,
    tokenizer_tag:   Optional[str]            = None,
    device:          Optional[torch.device]   = None,
    wall_time:       Optional[float]          = None,
    logger:          Optional[logging.Logger] = None,
) -> Dict[str, object]:
    r"""Reload the best-val checkpoint, score the TEST split, and write summary + figures.

    The headline metric is the test perplexity of the BEST-validation model (the periodic eval
    saved ``best_model.pt`` at the lowest val PPL); we reload it so the reported test number is
    not the final, possibly-overfit live weights. If no checkpoint was written (no validation
    configured), the live model is scored. Returns the test-results dict.
    """
    from vfe3.train import evaluate                              # local import avoids an import cycle

    logger = logger or logging.getLogger(__name__)
    if device is None:
        device = next(model.parameters()).device

    # Reachability guard (PB-03): finite best metadata REQUIRES a reachable best_model.pt; an old file
    # left with best_val_ppl=inf is ignored (not treated as selected state). The held-out test eval
    # intentionally scores the SELECTED validation checkpoint, so reload it whenever selection is live.
    has_best_metadata = math.isfinite(float(artifacts.best_val_ppl))
    if has_best_metadata and not artifacts.best_path.is_file():
        raise RuntimeError("finite best-model metadata has no reachable weights")
    reloaded_best = False
    if has_best_metadata:
        bundle = torch.load(artifacts.best_path, map_location=device, weights_only=True)
        if not isinstance(bundle, Mapping) or not {
                "model_state", "config", "config_fingerprint"}.issubset(bundle):
            raise ValueError(
                f"best checkpoint {artifacts.best_path} is not a semantic best-model bundle")
        saved_config = bundle["config"]
        if not isinstance(saved_config, Mapping):
            raise ValueError(
                f"best checkpoint {artifacts.best_path} has a non-mapping config")
        saved_fingerprint = semantic_config_fingerprint(saved_config)
        # RETAINED full internal fingerprint check (excluded-field tampering is still caught)...
        if bundle["config_fingerprint"] != saved_fingerprint:
            raise ValueError(
                f"best checkpoint {artifacts.best_path} has a config fingerprint mismatch")
        # ...but the saved-vs-live comparison is on the SELECTION PROJECTION, so a resume-path or
        # output-cadence change cannot reject otherwise-identical selected weights.
        if (semantic_config_fingerprint(_selection_semantic_config(saved_config))
                != semantic_config_fingerprint(_selection_semantic_config(cfg))):
            raise ValueError(
                f"best checkpoint {artifacts.best_path} does not match the active selection config")
        model.load_state_dict(bundle["model_state"])
        reloaded_best = True
        logger.info("Reloaded best-val checkpoint (step %s, val PPL %.3f) for test eval",
                    artifacts.best_step, artifacts.best_val_ppl)

    results: Dict[str, object] = {}                             # mixes float / Optional[float|int] / bool
    if test_loader is not None:
        m = evaluate(model, test_loader, tokens_per_char=tokens_per_char, device=device)
        results = {
            "test_ce":             m["ce"],
            "test_ppl":            m["ppl"],
            "test_bits_per_token": m["bits_per_token"],
            "test_bpc":            m["bpc"],
        }
        logger.info(
            "Test (held-out) | CE: %.4f | PPL: %.2f | BPT: %.4f | BPC: %s",
            m["ce"], m["ppl"], m["bits_per_token"],
            (f"{float(m['bpc']):.4f}" if m["bpc"] is not None else "unavailable"),
        )
    best_val_ppl = artifacts.best_val_ppl if artifacts.best_val_ppl != float("inf") else None
    results.update({"best_val_ppl": best_val_ppl, "best_step": artifacts.best_step,
                    "reloaded_best": reloaded_best})

    # E-step inference-time value: test CE with the inner E-step DISABLED (n_e_steps=0 -> belief =
    # prior, the loop runs zero iterations) minus the configured-budget test CE. NOTE this is the
    # INFERENCE-TIME marginal value of the E-step under tables that were TRAINED with it (the M-step
    # co-adapts the priors to the refinement) -- NOT a clean capacity split into table vs E-step, which
    # would need a second model trained at n_e_steps=0. A near-zero value still flags an E-step that
    # buys little at inference. Off-graph, best-effort; n_e_steps is restored in the finally.
    if test_loader is not None and results.get("test_ce") is not None:
        _saved_ne = model.cfg.n_e_steps
        try:
            model.cfg.n_e_steps = 0
            m0 = evaluate(model, test_loader, tokens_per_char=tokens_per_char, device=device)
            results["test_ce_no_estep"]    = m0["ce"]
            results["estep_capacity_gain"] = m0["ce"] - results["test_ce"]
            logger.info("E-step capacity gain (CE@n_e_steps=0 - CE@%d): %.4f",
                        _saved_ne, results["estep_capacity_gain"])
        except Exception as exc:
            logger.warning("estep capacity-gain probe failed (%s); skipped", exc)
        finally:
            model.cfg.n_e_steps = _saved_ne

    # EXP-5 (C2): the converged final E-step free energy PER TOKEN -- the E-step's OWN target-blind
    # functional value (free_energy_value sums F over the N tokens; divide by N). Persisted so a
    # cross-arm reader (scaling_analysis) can test whether final F DECORRELATES from CE across an
    # n_e_steps sweep -- the structural non-Neal-Hinton EM prediction (the E-step serves a distinct
    # functional, not the likelihood). Off-graph, best-effort, on a fixed test batch (sequence 0).
    if test_loader is not None:
        try:
            from vfe3.viz.extract import e_step_belief_trace
            _b = next(iter(test_loader))
            _tok = (_b[0] if isinstance(_b, (tuple, list)) else _b).to(device)
            _tr = e_step_belief_trace(model, _tok)              # n_iter defaults to cfg.n_e_steps
            results["estep_final_f_per_token"] = float(_tr["free_energy"][-1]) / max(1, int(_tok.shape[1]))
            logger.info("Converged final E-step F/token: %.4f", results["estep_final_f_per_token"])
        except Exception as exc:
            logger.warning("estep final-F probe failed (%s); skipped", exc)

    depth_loader = val_loader if val_loader is not None else test_loader
    if depth_loader is not None:
        try:
            _batch = next(iter(depth_loader))
            if not isinstance(_batch, (tuple, list)) or len(_batch) < 2:
                raise ValueError("depth sensitivity requires a (tokens, targets) batch")
            _depth_tokens = _batch[0].to(device)
            _depth_targets = _batch[1].to(device)
            _depth_record = collect_estep_depth_sensitivity(
                model,
                _depth_tokens,
                _depth_targets,
                depths=(0, 1, 2, 3, 5, 8),
            )
            artifacts.save_json("estep_depth_sensitivity.json", _depth_record)
            from vfe3.viz.figures import plot_estep_depth_sensitivity
            _depth_figure = plot_estep_depth_sensitivity(
                _depth_record,
                path=str(artifacts.run_dir / "estep_depth_sensitivity.png"),
            )
            if _depth_figure is not None:
                from matplotlib import pyplot as plt
                plt.close(_depth_figure)
        except Exception as exc:
            logger.warning("estep depth-sensitivity probe failed (%s); skipped", exc)
        try:
            _phi_batch = next(iter(depth_loader))
            _phi_tokens = (
                _phi_batch[0] if isinstance(_phi_batch, (tuple, list)) else _phi_batch
            ).to(device)
            _phi_record = collect_phi_numerics(model, _phi_tokens)
            artifacts.save_json("phi_numerics.json", _phi_record)
            from vfe3.viz.figures import plot_phi_numerics_reference
            _phi_figure = plot_phi_numerics_reference(
                _phi_record,
                path=str(artifacts.run_dir / "phi_numerics_reference.png"),
            )
            if _phi_figure is not None:
                from matplotlib import pyplot as plt
                plt.close(_phi_figure)
        except Exception as exc:
            logger.warning("phi numerical-reference probe failed (%s); skipped", exc)
    artifacts.save_json("test_results.json", results)

    # Reproducibility provenance (git SHA / data hash / versions) + a scaling-law data point -- the
    # externally-grounded records a config-only artifact omits (identical config.json can come from
    # different code and data, and a single run carries no (N, tokens, FLOPs, loss) frontier point).
    _write_provenance(
        artifacts,
        cfg,
        model,
        logger,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        data_seed=data_seed,
        max_tokens=max_tokens,
        tokenizer_tag=tokenizer_tag,
    )
    n_params = int(sum(p.numel() for p in model.parameters()))
    tokens_seen = int(cfg.max_steps) * int(cfg.batch_size) * int(cfg.max_seq_len)
    # scaling-law data point: the 6ND FLOP proxy is LOOSE for a no-NN E-step model, so record the
    # inputs too (a cross-run frontier can be re-fit offline with the right cost model). The
    # _cost_model_fields block adds the structural axes + active-params-per-token + a faithful
    # analytic proxy so each point is standalone; best-effort, never blocks the saved numbers.
    scaling_point: Dict[str, object] = {
        "n_params":             n_params,
        "tokens_seen":          tokens_seen,
        "est_flops_6ND":        6 * n_params * tokens_seen,
        "test_ce":              results.get("test_ce"),
        "test_ppl":             results.get("test_ppl"),
        "test_bits_per_token":  results.get("test_bits_per_token"),
        "test_bpc":             results.get("test_bpc"),
    }
    try:
        scaling_point.update(_cost_model_fields(model, cfg, n_params, tokens_seen, wall_time=wall_time))
    except Exception as exc:
        logger.warning("cost-model fields failed (%s); scaling_point keeps the 6ND proxy only", exc)
    artifacts.save_json("summary.json", {
        "n_steps":      cfg.max_steps,
        "n_params":     n_params,
        "best_val_ppl": best_val_ppl,
        "best_step":    artifacts.best_step,
        "reloaded_best": results.get("reloaded_best"),   # m26: False on a cross-run-dir resume whose best_model.pt is elsewhere
        "test_ppl":     results.get("test_ppl"),
        "test_ce":      results.get("test_ce"),
        "test_bits_per_token": results.get("test_bits_per_token"),
        "test_bpc":     results.get("test_bpc"),
        "test_ce_no_estep":    results.get("test_ce_no_estep"),
        "estep_capacity_gain": results.get("estep_capacity_gain"),
        "estep_final_f_per_token": results.get("estep_final_f_per_token"),
        "final_train_loss": (losses[-1] if losses else None),
        "wall_time_s":  wall_time,
        "use_prior_bank":  cfg.use_prior_bank,
        "use_head_mixer":  cfg.use_head_mixer,
        "phi_chart_norm_route": _phi_chart_norm_route(model, cfg),
        "scaling_point":   scaling_point,
    })

    # Pure-path certificate: the config toggles for the principal gauge / decode / free-energy purity
    # axes (flat gauge, canonical F, prior-bank decode, no head mixer, ...), plus the converged-state
    # stress metrics that say whether the numerical guards stayed inert. A REPORT of where the run sits,
    # not a judgment that any toggle is wrong (toggles are changed intentionally). Best-effort.
    try:
        artifacts.save_json("pure_path_report.json", _pure_path_report(cfg, artifacts.history))
    except Exception as exc:
        logger.warning("pure-path report failed (%s); skipped", exc)

    # Research artifacts (decode calibration / corpus-frequency-stratified loss / FD gradient check) --
    # externally-grounded probes that do NOT presuppose the gauge framework. Best-effort, AFTER the
    # test-eval n_e_steps restore so the model is in its trained state. Run before the figure pass.
    _write_research_artifacts(model, artifacts, cfg, train_loader, test_loader, device, logger)

    _save_figures(artifacts, losses, logger)
    # Single-run publication figure set (model-replay), auto-run at the end of training unless
    # cfg.generate_figures is False. Best-effort and off the hot path -- the runners are expensive
    # (UMAP, E-step replay, holonomy sampling, a belief bank over many sequences), so a failure is
    # logged and never disturbs the saved numeric results. Drives the BEST-val model reloaded above.
    if getattr(cfg, "generate_figures", True):
        try:
            from vfe3.viz.report import generate_figures
            generate_figures(
                artifacts.run_dir,
                model=model,
                loader=test_loader,
                device=device,
                allow_large=bool(getattr(cfg, "force_large_figures", False)),
                logger=logger,
            )
        except Exception as exc:
            logger.warning("publication figure generation failed (%s); numeric results are saved", exc)
    return results


@torch.no_grad()
def collect_estep_depth_sensitivity(
    model:   torch.nn.Module,
    tokens:  torch.Tensor,
    targets: torch.Tensor,
    depths:  Iterable[int],
) -> Dict[str, object]:
    r"""Evaluate current weights at several inference depths on one fixed batch.

    This is a sensitivity probe, not a retrained-depth comparison. Model mode, configured depth,
    and global CPU/CUDA RNG state are restored exactly before return.
    """
    import torch.nn.functional as F
    from vfe3.viz.extract import e_step_belief_trace

    requested = list(depths)
    if any(type(depth) is not int or depth < 0 for depth in requested):
        raise ValueError(f"depths must contain nonnegative integers, got {requested!r}")
    ordered = sorted(set(requested))
    trained_depth = int(model.cfg.n_e_steps)
    was_training = bool(model.training)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None)
    points: List[Dict[str, float | int]] = []
    try:
        model.eval()
        for depth in ordered:
            model.cfg.n_e_steps = depth
            logits = model(tokens)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                ignore_index=-100,
            )
            trace = e_step_belief_trace(model, tokens, n_iter=depth)
            points.append({
                "depth":                 depth,
                "ce":                    float(ce),
                "free_energy_per_token": float(trace["free_energy"][-1]) / max(1, int(tokens.shape[1])),
            })
    finally:
        model.cfg.n_e_steps = trained_depth
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return {
        "trained_depth": trained_depth,
        "interpretation": (
            "current-weight inference-depth sensitivity; depths other than trained_depth were not retrained"
        ),
        "points": points,
    }


@torch.no_grad()
def collect_phi_numerics(
    model:  torch.nn.Module,
    tokens: torch.Tensor,

    *,
    max_tokens: int = 8,
) -> Dict[str, object]:
    r"""Collect sampled chart, BCH, and fp32/fp64 flatness references off the hot path."""
    from vfe3 import metrics
    from vfe3.geometry.lie_ops import project_phi_to_slk
    from vfe3.model.positional_phi import positional_phi_coords

    was_training = bool(model.training)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None)
    try:
        model.eval()
        selected = tokens[:1, :max_tokens]
        belief, _ = model.forward_beliefs(selected)
        phi = belief.phi[0]
        block_dims = list(model.group.irrep_dims)
        record: Dict[str, object] = {
            "sample_tokens": int(phi.shape[0]),
            "runtime_dtype": str(phi.dtype),
            "reference_dtype": "torch.float64",
            "composition": model.cfg.pos_phi_compose,
            "chart": metrics.phi_chart_statistics(
                phi,
                model.group.generators,
                block_dims=block_dims,
            ),
        }
        if belief.right_phi is not None:
            record["right_chart"] = metrics.phi_chart_statistics(
                belief.right_phi,
                model.group.generators,
                block_dims=block_dims,
            )
        if model.cfg.transport_mode == "flat" and model.cfg.gauge_parameterization == "phi":
            record["flatness"] = metrics.flatness_reference_statistics(
                phi,
                model.group.generators,
                right_phi=belief.right_phi,
                max_triangles=8,
                block_dims=block_dims,
            )
        if model.cfg.pos_phi != "none" and model.cfg.pos_phi_compose == "bch":
            encoded = model.prior_bank.encode(selected)
            raw_phi = encoded.phi[0]
            coords = positional_phi_coords(
                model.cfg.pos_phi,
                raw_phi.shape[-2],
                raw_phi.shape[-1],
                scale=model.cfg.pos_phi_scale,
                pos_phi_free=getattr(model, "pos_phi_free", None),
                device=raw_phi.device,
                dtype=raw_phi.dtype,
            )
            if coords is not None:
                if model.cfg.pos_phi_project_slk:
                    coords = project_phi_to_slk(
                        coords,
                        model.group.generators,
                        model.group.irrep_dims,
                    )
                record["bch_fidelity"] = metrics.bch_fidelity_statistics(
                    raw_phi,
                    coords,
                    model.group.generators,
                    order=model.cfg.bch_pe_order,
                    block_dims=block_dims,
                )
        return record
    finally:
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _restore_rng_state(
    rng_state: Mapping[str, object],
) -> None:
    r"""Restore the captured CPU (and any available CUDA) global RNG states.

    Mirrors ``load_checkpoint``'s RNG restore: the CPU state must be a CPU ByteTensor and the CUDA
    per-device states are set only when CUDA is available (a CPU-only host silently skips the CUDA leg).
    """
    cpu = rng_state.get("cpu")
    if cpu is not None:
        torch.set_rng_state(cpu.cpu() if hasattr(cpu, "cpu") else cpu)
    cuda = rng_state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in cuda])


@torch.no_grad()
def finalize_validation_run(
    model:       torch.nn.Module,
    artifacts:   RunArtifacts,
    cfg:         VFE3Config,
    val_loader:  Iterable,

    *,
    tokens_per_char: Optional[float]           = None,
    train_loader:    Optional[Iterable]       = None,
    losses:          Optional[List[float]]    = None,
    data_seed:       Optional[int]            = None,
    max_tokens:      Optional[int]            = None,
    tokenizer_tag:   Optional[str]            = None,
    device:          Optional[torch.device]   = None,
    wall_time:       Optional[float]          = None,
    logger:          Optional[logging.Logger] = None,
    terminal_state:  "Optional[TrainingTerminalState]" = None,
) -> Dict[str, object]:
    r"""Score validation, save terminal artifacts, and never open a test split.

    The validation-only sibling of :func:`finalize_run` (PB-02): invoked ONCE as a ``train`` terminal
    callback so even a default ablation cell (log/eval interval above ``max_steps``,
    ``checkpoint_interval=0``) publishes a complete resumable artifact set. Model selection and every
    reported number use VALIDATION data only -- the split is labeled ``selection_split="validation"``
    and ``summary.json`` never carries ``test_ce``/``test_ppl``/``test_bpc``.

    Successful sequence (the model enters with RAW last-iterate weights): if EMA exists, copy the EMA
    shadow into the model, then evaluate ``val_loader`` once, append a terminal metrics row, publish the
    selected best weights (``best_model.pt``), write ``validation_results.json``, collect VALIDATION-only
    provenance and the pure-path report, and render the history-only figures -- all against the EMA (or
    raw) weights. A ``finally`` block then strictly reloads ``terminal_state.raw_model_state`` and
    restores the captured CPU/CUDA RNG. After a successful validation the resumable terminal checkpoint
    is written from the RESTORED raw model plus its matching optimizer/scaler/EMA/private-RNG/data cursor
    (so the checkpoint never pairs EMA weights with raw optimizer moments), ``summary.json`` is written
    only after that checkpoint exists, and EMA is copied back into the live model so ``train`` keeps its
    returned-model behavior. A validation/checkpoint failure restores the raw weights/RNG and re-raises,
    publishing no summary (and, via the caller, no success contract).

    Returns the ablation merge mapping with both ``final_val_bits_per_token`` and nullable
    ``final_val_bpc`` alongside the validation/selection/checkpoint fields.
    ``primary_val_ppl`` is the minimum of the finite run-wide best and the final validation PPL (or the
    final value when no earlier best exists); after ``maybe_save_best`` it equals the selected finite best.
    """
    from vfe3.train import _loader_data_identity, evaluate       # local import avoids an import cycle

    logger = logger or logging.getLogger(__name__)
    if device is None:
        device = next(model.parameters()).device
    ema = terminal_state.ema if terminal_state is not None else None
    completed_step = _require_nonnegative_int(
        terminal_state.step if terminal_state is not None else cfg.max_steps,
        "terminal step",
    )
    artifacts.bind_selection_data_identity(_loader_data_identity(val_loader, cfg.vocab_size))

    # Reachability guard (PB-03): entering with FINITE best metadata (a resumed periodic best) requires
    # a reachable best_model.pt. A recomputed ablation cell instead enters with INFINITE metadata even
    # when a stale file survives on disk, so that file is ignored -- terminal validation replaces it
    # below rather than the finalizer silently selecting the previous cell's weights. This finalizer
    # never loads a preexisting best into the live model before scoring the terminal EMA.
    if math.isfinite(float(artifacts.best_val_ppl)) and not artifacts.best_path.is_file():
        raise RuntimeError("finite best-model metadata has no reachable weights")

    n_params = int(sum(p.numel() for p in model.parameters()))
    final_train_loss = (float(losses[-1]) if losses else None)

    try:
        # Evaluation, best-weight publication, provenance, and figures all run against the DEPLOYED
        # averaged weights (the model entered holding the raw last-iterate weights).
        if ema is not None:
            ema.copy_to(model)

        metrics = evaluate(model, val_loader, tokens_per_char=tokens_per_char, device=device)
        final_ce = float(metrics["ce"])
        final_ppl = float(metrics["ppl"])
        final_bits_per_token = float(metrics["bits_per_token"])
        final_bpc = (float(metrics["bpc"]) if metrics["bpc"] is not None else None)

        # Terminal metrics row: compatible with an empty history (defines the schema) OR an established
        # training schema (its five keys are a subset of the training columns, so the append is clean).
        terminal_row = {
            "step":       completed_step,
            "train_loss": float(losses[-1]) if losses else float("nan"),
            "val_ce":     final_ce,
            "val_ppl":    final_ppl,
            "val_bits_per_token": final_bits_per_token,
            "val_bpc":    (final_bpc if final_bpc is not None else float("nan")),
        }
        # Run-wide best BEFORE the terminal save: primary is the better of the finite periodic best and
        # the final validation, so after maybe_save_best it equals the selected finite best.
        prior_best = artifacts.best_val_ppl
        artifacts.log_metrics(terminal_row)
        artifacts.maybe_save_best(completed_step, model, final_ppl)
        # After the terminal save the selection is live and MUST be reachable: maybe_save_best either
        # atomically replaced any stale file with the terminal weights (a recomputed cell) or left an
        # already-reachable earlier best (guarded above). Fail closed before publishing the success
        # contract so a stale-contract rerun never selects the previous cell's weights.
        if not (math.isfinite(float(artifacts.best_val_ppl)) and artifacts.best_path.is_file()):
            raise RuntimeError("finite best-model metadata has no reachable weights")
        primary_val_ppl = (min(prior_best, final_ppl) if math.isfinite(prior_best) else final_ppl)
        best_val_ppl = (artifacts.best_val_ppl if artifacts.best_val_ppl != float("inf") else None)

        artifacts.save_json("validation_results.json", {
            "selection_split": "validation",
            "val_ce":          final_ce,
            "val_ppl":         final_ppl,
            "val_bits_per_token": final_bits_per_token,
            "val_bpc":         final_bpc,
            "primary_val_ppl": float(primary_val_ppl),
            "best_val_ppl":    best_val_ppl,
            "best_step":       artifacts.best_step,
        })

        # Reproducibility provenance -- VALIDATION ONLY (test_loader=None): the ablation finalizer must
        # never open a test split.
        _write_provenance(
            artifacts, cfg, model, logger,
            train_loader=train_loader, val_loader=val_loader, test_loader=None,
            data_seed=data_seed, max_tokens=max_tokens, tokenizer_tag=tokenizer_tag,
        )
        try:
            artifacts.save_json("pure_path_report.json", _pure_path_report(cfg, artifacts.history))
        except Exception as exc:
            logger.warning("pure-path report failed (%s); skipped", exc)
        _save_figures(artifacts, losses, logger)
    finally:
        # Strict raw-state reload + RNG restore, on BOTH the success and failure paths: the terminal
        # checkpoint below pairs the RAW model with the raw optimizer moments (never the EMA weights),
        # and the global RNG is rewound to its captured post-final-step value.
        if terminal_state is not None:
            model.load_state_dict(terminal_state.raw_model_state)
            _restore_rng_state(terminal_state.rng_state)

    # Reached only after a successful validation pass. Write the resumable terminal checkpoint from the
    # restored raw model, then summary.json (only after the checkpoint exists), then copy EMA back.
    terminal_checkpoint: Optional[str] = None
    if terminal_state is not None:
        checkpoint_path = artifacts.save_checkpoint(
            completed_step, model, terminal_state.optimizer, cfg,
            scaler=terminal_state.scaler, ema=ema,
            metropolis_generator=terminal_state.metropolis_generator,
            data_state=terminal_state.data_state,
        )
        terminal_checkpoint = str(checkpoint_path)

    figures_written = sorted(p.name for p in artifacts.run_dir.glob("*.png") if p.is_file())
    artifacts.save_json("summary.json", {
        "selection_split":     "validation",
        "primary_val_ppl":     float(primary_val_ppl),
        "final_val_ce":        final_ce,
        "final_val_ppl":       final_ppl,
        "final_val_bits_per_token": final_bits_per_token,
        "final_val_bpc":       final_bpc,
        "best_val_ppl":        best_val_ppl,
        "best_step":           artifacts.best_step,
        "n_steps":             completed_step,
        "n_params":            n_params,
        "final_train_loss":    final_train_loss,
        "wall_time_s":         (float(wall_time) if wall_time is not None else None),
        "terminal_checkpoint": terminal_checkpoint,
        "figures_written":     figures_written,
        "phi_chart_norm_route": _phi_chart_norm_route(model, cfg),
    })

    if ema is not None:
        ema.copy_to(model)                                      # train() returns the deployed EMA weights

    return {
        "primary_val_ppl":     float(primary_val_ppl),
        "final_val_ppl":       final_ppl,
        "final_val_ce":        final_ce,
        "final_val_bits_per_token": final_bits_per_token,
        "final_val_bpc":       final_bpc,
        "best_val_ppl":        best_val_ppl,
        "best_step":           artifacts.best_step,
        "final_train_loss":    final_train_loss,
        "n_params":            n_params,
        "terminal_checkpoint": terminal_checkpoint,
    }


def _pure_path_report(cfg: VFE3Config, history: List[Dict]) -> Dict:
    r"""Where a run sits relative to the theoretically pure path: the toggle states that define it plus
    the converged-state stress metrics that say whether the numerical guards stayed inert.

    A REPORT, not a verdict. The pure path must EXIST under appropriate toggles, but the user changes
    toggles intentionally, so a non-pure run is recorded (``on_pure_path=False`` with the offending
    flags), never flagged as wrong. ``pure_flags`` covers the principal gauge / decode / free-energy
    purity axes (canonical attention entropy, flat transport, constant/static coupling weights,
    prior-bank decode, full sigma updates, no two-hop/fixed-prior surrogate, no head mixer,
    unweighted attention); it does NOT enumerate every default-OFF learned-scalar toggle
    (pos_phi, learnable_r, t5_learnable_bias, use_cg_coupling),
    so ``on_pure_path`` certifies these axes rather than a full no-learned-parameter audit.
    ``gauge_flags``/``on_gauge_pure_path`` is a SECOND, independent axis (audit 2026-07-01 F8): the
    gauge / model-channel path (learned gauge transport, phi parameterization, no reflection or
    positional rotation, family/group invariance, no model-channel coupling) -- a run can be pure on
    the free-energy/decode axis while a gauge setting alters the executed belief path, and vice versa.
    ``converged_stress`` reads the last finite value of each guard / flatness column (None if absent)."""
    def _last(key: str) -> Optional[float]:
        for r in reversed(history):
            v = r.get(key)
            if isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
        return None
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.transport import get_transport_registration

    group_builder = get_group(cfg.gauge_group)
    invariant_families = tuple(getattr(group_builder, "invariant_families", ()))
    family_group_invariant = cfg.family in invariant_families
    transport_registration = get_transport_registration(cfg.transport_mode)
    fixed_prior_surrogate = bool(cfg.precision_weighted_attention)
    head_mixer_compatibility = getattr(cfg, "head_mixer_compatibility", None)
    if head_mixer_compatibility is None:
        head_mixer_compatibility = VFE3Config.head_mixer_compatibility.fget(cfg)
    head_mixer_gauge_compatible = getattr(cfg, "head_mixer_gauge_compatible", None)
    if head_mixer_gauge_compatible is None:
        head_mixer_gauge_compatible = VFE3Config.head_mixer_gauge_compatible.fget(cfg)
    covariance_feature_exact = _last("regime_ii_covariant_feature_exact")
    feature_jitter_recovered = covariance_feature_exact is not None and covariance_feature_exact < 0.5
    if cfg.transport_mode != "regime_ii_covariant":
        regime_ii_covariant_exact = True
        regime_ii_covariant_exactness = "not_applicable"
    elif not family_group_invariant:
        regime_ii_covariant_exact = False
        regime_ii_covariant_exactness = "diagonal_projection_approximation"
    elif feature_jitter_recovered:
        regime_ii_covariant_exact = False
        regime_ii_covariant_exactness = "jitter_recovered_approximation"
    else:
        regime_ii_covariant_exact = True
        regime_ii_covariant_exactness = "exact_valid_spd_factorization"

    spd_retract_mode = getattr(cfg, "spd_retract_mode", "spd_affine")
    sigma_max = getattr(cfg, "sigma_max", 10.0)
    spd_retraction_exact = sigma_max is None
    if spd_retract_mode == "spd_affine":
        spd_retraction_route = (
            "airm_exact" if spd_retraction_exact else "airm_projected_spectral_cap"
        )
    else:
        spd_retraction_route = (
            "log_euclidean_exact"
            if spd_retraction_exact
            else "log_euclidean_projected_spectral_cap"
        )

    pure_flags = {
        "canonical_attention_entropy": bool(cfg.include_attention_entropy),
        "flat_transport":              cfg.transport_mode == "flat",
        "constant_lambda_alpha":       cfg.lambda_alpha_mode == "constant",
        "prior_bank_decode":           bool(cfg.use_prior_bank),
        "no_head_mixer":               not cfg.use_head_mixer,
        "unweighted_attention":        not cfg.precision_weighted_attention,
        "full_sigma_update":           not cfg.skip_belief_sigma_update,
        "no_twohop_coupling":          cfg.lambda_twohop == 0.0,
        "no_fixed_prior_surrogate":    not fixed_prior_surrogate,
    }
    # Second, INDEPENDENT purity axis (audit 2026-07-01 F8): the gauge / model-channel path. Keyed
    # on pos_rotation itself rather than the RoPE sub-toggles (rope_full_gauge / rope_on_value),
    # which are inert while RoPE is off -- those are reported in config_toggles for transparency.
    gauge_flags = {
        "learned_gauge_transport":   cfg.gauge_transport == "on",
        "no_positional_rotation":    cfg.pos_rotation == "none",
        "no_model_channel_coupling": cfg.lambda_gamma == 0.0 and not cfg.s_e_step,
        "phi_parameterization":      cfg.gauge_parameterization == "phi",
        "no_reflection_sampling":    cfg.omega_reflection == "off" and cfg.phi_reflection == "off",
        "family_group_invariant":    family_group_invariant,
        "head_mixer_intertwiner_compatible": head_mixer_gauge_compatible,
    }
    return {
        "on_pure_path":       all(pure_flags.values()),
        "pure_flags":         pure_flags,
        "gauge_flags":        gauge_flags,
        "on_gauge_pure_path": all(gauge_flags.values()),
        "config_toggles": {
            "include_attention_entropy":    bool(cfg.include_attention_entropy),
            "transport_mode":               cfg.transport_mode,
            "lambda_alpha_mode":            cfg.lambda_alpha_mode,
            "lambda_beta":                  float(cfg.lambda_beta),
            "use_prior_bank":               bool(cfg.use_prior_bank),
            "use_head_mixer":               bool(cfg.use_head_mixer),
            "head_mixer_compatibility":      head_mixer_compatibility,
            "head_mixer_gauge_compatible":   head_mixer_gauge_compatible,
            "precision_weighted_attention": bool(cfg.precision_weighted_attention),
            "gauge_transport":              cfg.gauge_transport,
            "pos_rotation":                 cfg.pos_rotation,
            "rope_full_gauge":              bool(cfg.rope_full_gauge),
            "rope_on_value":                bool(cfg.rope_on_value),
            "lambda_gamma":                 float(cfg.lambda_gamma),
            "s_e_step":                     bool(cfg.s_e_step),
            "skip_belief_sigma_update":      bool(cfg.skip_belief_sigma_update),
            "lambda_twohop":                 float(cfg.lambda_twohop),
            "gauge_parameterization":        cfg.gauge_parameterization,
            "omega_reflection":              cfg.omega_reflection,
            "phi_reflection":                cfg.phi_reflection,
            "gauge_group":                   cfg.gauge_group,
            "family":                        cfg.family,
            "group_invariant_families":      list(invariant_families),
            # Truthful fixed-surrogate ledger (C6): these derived booleans expose when the run
            # intentionally freezes a state-dependent quantity rather than following its full
            # joint objective. Defaults are False, preserving the pure path.
            "fixed_covariance_surrogate":   bool(getattr(cfg, "skip_belief_sigma_update", False)),
            "detached_precision_prior":     fixed_prior_surrogate,
            "detached_query_adaptive_tau":  bool(getattr(cfg, "query_adaptive_tau", False)),
            "state_dependent_alpha_majorizer": (
                getattr(cfg, "e_step_update", "gradient") == "mm_exact"
                and cfg.lambda_alpha_mode in ("state_dependent", "state_dependent_per_coord")
            ),
            # regime_ii_covariant under gaussian_diagonal is a CONTROLLED APPROXIMATION (the
            # diagonal cone is not closed under GL congruence Omega Sigma Omega^T -- audit C5),
            # so a diagonal covariant run is never reported as exact Route B.
            "regime_ii_covariant_exact":    regime_ii_covariant_exact,
            "regime_ii_covariant_exactness": regime_ii_covariant_exactness,
            "spd_retraction_route":         spd_retraction_route,
            "spd_retraction_exact":         spd_retraction_exact,
            # Covariance class of the ACTIVE transport (audit C7), owned by its complete registry
            # record. An unregistered mode fails closed instead of inventing report metadata.
            "transport_covariance_class":   transport_registration.covariance_class,
        },
        "converged_stress": {k: _last(k) for k in (
            "guard_sigma_floor_frac", "guard_sigma_ceil_frac", "guard_energy_klmax_frac",
            "guard_selfdiv_klmax_frac", "nonfinite_frac", "renyi_band_frac",
            "cocycle_residual", "holonomy_wilson", "gauge_invariant_spread")},
    }


def _save_figures(
    artifacts: RunArtifacts,
    losses:    Optional[List[float]],
    logger:    logging.Logger,
) -> None:
    r"""Best-effort publication figures from the logged history (no model re-run)."""
    try:
        from vfe3.viz import figures as raw_figs

        class _SafePyplot:
            r"""Delegate pyplot operations while making a failed renderer's ``None`` close a no-op."""

            def __getattr__(self, name: str) -> object:
                return getattr(raw_figs.plt, name)

            def close(self, figure: object = None) -> None:
                if figure is not None:
                    raw_figs.plt.close(figure)

        class _IsolatedFigures:
            r"""Proxy plot calls so one failed renderer cannot abort or leak into the next one."""

            def __getattr__(self, name: str) -> object:
                if name == "plt":
                    return _SafePyplot()
                value = getattr(raw_figs, name)
                if not name.startswith("plot_") or not callable(value):
                    return value

                def _isolated(*args: object, **kwargs: object) -> object:
                    before = set(raw_figs.plt.get_fignums())
                    try:
                        return value(*args, **kwargs)
                    except Exception as exc:
                        for number in set(raw_figs.plt.get_fignums()) - before:
                            raw_figs.plt.close(number)
                        logger.warning("figure %s failed (%s); remaining figures continue", name, exc)
                        return None

                return _isolated

        figs = _IsolatedFigures()
        figs.set_publication_style()
        run = artifacts.run_dir

        def _aligned(key: str) -> tuple:
            r"""Aligned (step, value) for a history column, dropping pre-first-eval NaN rows."""
            xs, ys = [], []
            for i, r in enumerate(artifacts.history):
                if key in r and math.isfinite(r[key]):
                    xs.append(r.get("step", i))
                    ys.append(r[key])
            return xs, ys

        def _hist_subset(keys: tuple) -> Optional[Dict]:
            r"""A ``{step, key: [...]}`` history dict over ``keys`` present (finite on >= 1 row), each a
            full-length column with NaN where missing so an eval-cadence key keeps its step alignment and
            the dashboard masks it per series. Returns None when no key is present (caller skips)."""
            present = [k for k in keys
                       if any(k in r and isinstance(r[k], (int, float)) and math.isfinite(r[k])
                              for r in artifacts.history)]
            if not present:
                return None
            cols: Dict = {"step": [r.get("step", i) for i, r in enumerate(artifacts.history)]}
            for k in present:
                cols[k] = [float(r[k]) if (k in r and isinstance(r[k], (int, float)) and math.isfinite(r[k]))
                           else float("nan") for r in artifacts.history]
            return cols

        if losses:
            # losses is one entry per optimizer step, so the 1-based step index IS the x-axis.
            n = len(losses)
            steps_per_epoch = next(
                (
                    int(r["steps_per_epoch"])
                    for r in artifacts.history
                    if isinstance(r.get("steps_per_epoch"), (int, float))
                    and math.isfinite(float(r["steps_per_epoch"]))
                    and int(r["steps_per_epoch"]) > 0
                ),
                0,
            )
            epoch_boundaries = (
                list(range(steps_per_epoch, n + 1, steps_per_epoch))
                if steps_per_epoch
                else None
            )
            fig = figs.plot_trajectory(
                losses, list(range(1, n + 1)), ylabel="train CE (nats/token)",
                title="Training cross-entropy", color=figs._CB[0],
                smooth=max(25, n // 240), annotate_final=True,
                epoch_boundaries=epoch_boundaries,
                path=str(run / "loss_curve.png"))
            figs.plt.close(fig)
        sx, sy = _aligned("val_ppl")
        if sy:
            fig = figs.plot_trajectory(
                sy, sx, ylabel="validation perplexity", title="Validation perplexity",
                color=figs._CB[1], logy=True, smooth=max(5, len(sy) // 80), annotate="min",
                path=str(run / "val_ppl.png"))
            figs.plt.close(fig)
        # Gauge-geometry trajectories (diagnostics tier): curvature proxy + gauge-trace spread.
        hx, hy = _aligned("holonomy_deviation")
        if hy:
            # Heavy-tailed (median ~1e-3, rare spikes ~1e3): log y + a median reference; NOT smoothed,
            # so the curvature spikes survive.
            flat_transport = getattr(getattr(artifacts, "cfg", None), "transport_mode", "flat") == "flat"
            fig = figs.plot_trajectory(
                hy, hx,
                ylabel=("numerical closure residual" if flat_transport
                         else r"$\langle\|H_{ijk}-I\|_F\rangle$"),
                title=("Numerical closure of nominally flat transport" if flat_transport
                       else "Holonomy curvature (frame-dependent Frobenius)"),
                color=figs._CB[2],
                logy=True, median_line=True, annotate="max",
                path=str(run / "holonomy.png"))
            figs.plt.close(fig)
        gx, gy = _aligned("gauge_trace_spread")
        if gy:
            fig = figs.plot_trajectory(
                gy, gx, ylabel=r"std $\log|\det\Omega|$", title="Gauge trace spread",
                color=figs._CB[3], smooth=max(5, len(gy) // 60), annotate_final=True,
                path=str(run / "gauge_trace_spread.png"))
            figs.plt.close(fig)
        # Learnable softmax-temperature trajectories: present exactly when train() logged the live
        # per-block kappa statistics for the default-off learnable_kappa_beta/gamma toggles.
        for _ch in ("beta", "gamma"):
            _hist_kappa = _hist_subset((f"kappa_{_ch}_mean", f"kappa_{_ch}_var"))
            if _hist_kappa and f"kappa_{_ch}_mean" in _hist_kappa:
                fig = figs.plot_kappa_history(
                    _hist_kappa, channel=_ch, path=str(run / f"kappa_{_ch}_history.png"))
                figs.plt.close(fig)
        # Per-irrep-block companion to the aggregate kappa_<ch>_history above: one line per block for
        # kappa AND the effective temperature tau, across the beta/gamma channels (a 2x2 grid when
        # both toggles are on). Present exactly when train() logged the per-block kappa_*/tau_* columns.
        _kb_keys = tuple(sorted({k for r in artifacts.history for k in r
                                 if k.startswith(("kappa_beta_b", "kappa_gamma_b",
                                                  "tau_beta_b", "tau_gamma_b"))}))
        if _kb_keys:
            _hist_kb = _hist_subset(_kb_keys)
            if _hist_kb:
                fig = figs.plot_kappa_block_trajectory(
                    _hist_kb, path=str(run / "kappa_block_trajectory.png"))
                figs.plt.close(fig)
        # Optimization + convergence trends (history-only; no model re-run): the pre-clip gradient
        # norm (THE optimization-health curve, previously discarded), the belief-covariance
        # conditioning, and the per-eval E-step F-descent (negative = the inner loop reduced F).
        nx, ny = _aligned("grad_norm")
        if ny:
            fig = figs.plot_trajectory(
                ny, nx, ylabel=r"$\|\nabla\|_2$ (pre-clip)", title="Gradient norm",
                color=figs._CB[5 % len(figs._CB)], logy=True, smooth=max(5, len(ny) // 80),
                annotate="max", path=str(run / "grad_norm.png"))
            figs.plt.close(fig)
        # M-step per-role gradient-norm decomposition (mu / sigma / phi): the parameter-learning
        # channels the aggregate grad_norm.png folds together. Columns logged by train_step (aggregated
        # by each optimizer group's "role" tag, so the live tables are attributed correctly under any
        # config); present only on a run that captured step_metrics, so gate on their presence (the CSV
        # stays rectangular per run). A SEPARATE figure from the aggregate, same pre-clip magnitudes.
        gd_keys = ("grad_norm_mu", "grad_norm_sigma", "grad_norm_phi")
        gd_present = [k for k in gd_keys
                      if any(k in r and math.isfinite(r[k]) for r in artifacts.history)]
        if gd_present:
            gd_rows = [r for r in artifacts.history
                       if all(k in r and math.isfinite(r[k]) for k in gd_present)]
            if gd_rows:
                hist_gd = {"step": [r.get("step", i) for i, r in enumerate(gd_rows)],
                           **{k: [r[k] for r in gd_rows] for k in gd_present}}
                fig = figs.plot_grad_norm_decomposition(hist_gd, path=str(run / "grad_norm_decomposition.png"))
                figs.plt.close(fig)
        # E-step belief-gradient decomposition (mu / sigma / phi): the INFERENCE analogue of the M-step
        # figure above -- ||grad F|| over the belief tuple per inner-loop component, logged by train_step
        # from model.forward's estep_grad_out. Accumulated runs prefer the explicitly named arithmetic
        # microbatch means; single-batch runs retain the historical column names. Same presence gate;
        # independent of the M-step columns, so build its own row set.
        eg_mean_keys = (
            "estep_grad_norm_mu_microbatch_mean",
            "estep_grad_norm_sigma_microbatch_mean",
            "estep_grad_norm_phi_microbatch_mean",
        )
        eg_keys = (eg_mean_keys if any(any(k in r for k in eg_mean_keys)
                                      for r in artifacts.history)
                   else ("estep_grad_norm_mu", "estep_grad_norm_sigma", "estep_grad_norm_phi"))
        eg_present = [k for k in eg_keys
                      if any(k in r and math.isfinite(r[k]) for r in artifacts.history)]
        if eg_present:
            eg_rows = [r for r in artifacts.history
                       if all(k in r and math.isfinite(r[k]) for k in eg_present)]
            if eg_rows:
                hist_eg = {"step": [r.get("step", i) for i, r in enumerate(eg_rows)],
                           **{k: [r[k] for r in eg_rows] for k in eg_present}}
                fig = figs.plot_estep_grad_norm_decomposition(hist_eg, path=str(run / "estep_grad_norm_decomposition.png"))
                figs.plt.close(fig)
        cx, cy = _aligned("belief_cond_median")
        if cy:
            fig = figs.plot_trajectory(
                cy, cx, ylabel=r"median $\lambda_{\max}/\lambda_{\min}$",
                title="Belief covariance conditioning", color=figs._CB[6 % len(figs._CB)],
                logy=True, smooth=max(5, len(cy) // 80), annotate="max",
                path=str(run / "belief_condition.png"))
            figs.plt.close(fig)
        ex, ey = _aligned("estep_f_drop")
        if ey:
            fig = figs.plot_trajectory(
                ey, ex, ylabel=r"$F_{\mathrm{end}}-F_{\mathrm{start}}$ (inner E-step)",
                title="E-step free-energy descent", color=figs._CB[2 % len(figs._CB)],
                median_line=True, path=str(run / "estep_convergence_trend.png"))
            figs.plt.close(fig)
        # Free-energy figures: the per-token budget DECOMPOSITION (snapshot + early/mid/late evolution)
        # and, as a SEPARATE figure, the F-vs-CE CO-DESCENT over training. Both need every plotted term
        # finite, so rows before the first eval (NaN val_*) are dropped.
        fe_keys = ("self_coupling", "belief_coupling", "attention_entropy", "val_ce")
        fe_rows = [r for r in artifacts.history
                   if all(k in r and math.isfinite(r[k]) for k in fe_keys)]
        if fe_rows:
            cfg = getattr(artifacts, "cfg", None)
            # Model-channel F components fold into the complexity-F total when the channel is live;
            # included only when present on EVERY plotted row (model-channel run). hyper_prior_weighted
            # is the EXACT weighted hyper-prior (state_dependent lambda_h != cfg.lambda_h*raw,
            # so it is read directly); the gamma blocks are scaled by cfg.lambda_gamma in the figure,
            # exactly as the belief block is scaled by lambda_beta.
            mc_fe_keys = [k for k in ("hyper_prior_weighted", "gamma_coupling", "gamma_meta_entropy")
                          if all(k in r and math.isfinite(r[k]) for r in fe_rows)]
            hist = {"step": [r.get("step", i) for i, r in enumerate(fe_rows)],
                    **{k: [r[k] for r in fe_rows] for k in (*fe_keys, *mc_fe_keys)}}
            # Scale the coupling terms by the static config lambda_beta scalar.
            lam = getattr(cfg, "lambda_beta", 1.0)
            gam = getattr(cfg, "lambda_gamma", 0.0)
            iae = getattr(cfg, "include_attention_entropy", True)
            fig = figs.plot_free_energy_decomposition(
                hist, lambda_beta=lam, lambda_gamma=gam, include_attention_entropy=iae,
                path=str(run / "free_energy_decomposition.png"))
            figs.plt.close(fig)
            fig = figs.plot_free_energy_codescent(
                hist, lambda_beta=lam, lambda_gamma=gam, include_attention_entropy=iae,
                path=str(run / "free_energy_codescent.png"))
            figs.plt.close(fig)
        # Model-channel free-energy blocks (s-channel): the hyper-prior KL(s||r), the gamma
        # model-coupling, and its meta-entropy over training. Present only when the model channel
        # is active (diagnostics logs these columns, gated on STATIC config), so the figure appears
        # exactly on the runs that have a model channel. RAW per-token blocks, a SEPARATE figure
        # since the model channel is a distinct hierarchical tier (h -> s -> p -> q).
        mc_keys = ("hyper_prior", "gamma_coupling", "gamma_meta_entropy")
        mc_present = [k for k in mc_keys
                      if any(k in r and math.isfinite(r[k]) for r in artifacts.history)]
        if mc_present:
            mc_rows = [r for r in artifacts.history
                       if all(k in r and math.isfinite(r[k]) for k in mc_present)]
            if mc_rows:
                hist_mc = {"step": [r.get("step", i) for i, r in enumerate(mc_rows)],
                           **{k: [r[k] for r in mc_rows] for k in mc_present}}
                fig = figs.plot_model_channel_terms(hist_mc, path=str(run / "model_channel_terms.png"))
                figs.plt.close(fig)
        # Geometry / SPD / Fisher health dashboard (history-only): the gauge, belief-spectrum, guard,
        # and numerical-safety scalars diagnostics() logs to metrics.csv but that no standard figure
        # surfaced. The panels self-gate, so a run missing a column simply drops that panel.
        hist_geom = _hist_subset((
            "holonomy_wilson", "cocycle_residual", "holonomy_deviation",
            "gauge_invariant_spread", "gauge_head_logdet_spread", "phi_norm_mean", "phi_norm_std",
            "phi_matrix_norm_p95", "phi_matrix_norm_p99", "phi_matrix_norm_max",
            "phi_exp_clamp_frac", "phi_exp_scale_min",
            "vertex_cond_median", "vertex_cond_p95", "vertex_cond_p99",
            "phi_chart_projected_fraction", "phi_chart_projection_scale_min",
            "phi_chart_projection_ms",
            "belief_cond_p95", "belief_cond_max", "eff_rank_p5", "eff_rank_median", "eff_rank_p95",
            "fisher_trace_mean", "guard_sigma_floor_frac", "guard_sigma_ceil_frac",
            "guard_energy_klmax_frac", "guard_selfdiv_klmax_frac", "nonfinite_frac", "renyi_band_frac",
            "attn_entropy_min", "attn_entropy_collapsed_heads"))
        if hist_geom:
            fig = figs.plot_geometry_health(
                hist_geom,
                transport_mode=getattr(getattr(artifacts, "cfg", None), "transport_mode", "flat"),
                family=getattr(getattr(artifacts, "cfg", None), "family", None),
                path=str(run / "geometry_health.png"),
            )
            figs.plt.close(fig)
        # E-step inference-quality dashboard: the inner-loop F-drop, the nondecreasing fraction, and the
        # last-iter belief residuals -- the E-step evidence the single estep_f_drop curve does not show.
        hist_estep = _hist_subset((
            "estep_f_drop", "estep_f_nondecreasing_frac",
            "estep_r_mu_last", "estep_r_sigma_last", "estep_r_phi_last",
            "estep_fp_kl", "estep_fp_mu_rms", "estep_fp_sigma_rms", "estep_fp_phi_rms",
            "estep_target_gap", "estep_beta_js", "estep_alpha_rms_delta"))
        if hist_estep:
            fig = figs.plot_estep_quality(hist_estep, path=str(run / "estep_quality.png"))
            figs.plt.close(fig)
        # Held-out validation-sanity dashboard: the per-eval probes (_val_diagnostics) that were CSV-only
        # -- generalization gap, positional loss, causal/attention sanity, and the held-out geometry.
        hist_val = _hist_subset((
            "generalization_gap", "pos_loss_first_q", "pos_loss_last_q", "pos_loss_ratio",
            "val_future_leakage", "val_row_sum_error", "val_pos_content_r2",
            "val_prev_token_mass", "val_period_match_mass", "val_head_redundancy_js",
            "val_holonomy_wilson", "val_cocycle_residual", "val_gauge_invariant_spread",
            "val_belief_cond_p95", "val_fisher_trace_mean", "val_guard_sigma_floor_frac",
            "val_guard_sigma_ceil_frac", "val_guard_energy_klmax_frac",
            "val_phi_norm_mean", "val_phi_norm_std", "val_phi_matrix_norm_p95",
            "val_phi_matrix_norm_p99", "val_phi_matrix_norm_max", "val_phi_exp_clamp_frac",
            "val_phi_exp_scale_min", "val_vertex_cond_p99"))
        if hist_val:
            fig = figs.plot_validation_sanity(
                hist_val,
                family=getattr(getattr(artifacts, "cfg", None), "family", None),
                path=str(run / "validation_sanity.png"),
            )
            figs.plt.close(fig)
        # Optimizer information-geometry dashboard: natural-gradient alignment, pullback conditioning,
        # per-role weight norms, and the synthesized update-to-weight ratio. Present only on a gauge
        # natural-grad run (cos_nat_phi / pullback) and when step_metrics captured the norms.
        hist_opt = _hist_subset((
            "cos_nat_phi", "pullback_cond_median", "pullback_cond_max",
            "weight_norm_mu", "weight_norm_sigma", "weight_norm_phi",
            "grad_norm_mu", "grad_norm_sigma", "grad_norm_phi"))
        if hist_opt:
            fig = figs.plot_optimizer_geometry(hist_opt, path=str(run / "optimizer_geometry.png"))
            figs.plt.close(fig)
    except Exception as exc:                                    # never let a plot kill a finished run
        logger.warning("figure generation failed (%s); numeric results are still saved", exc)
