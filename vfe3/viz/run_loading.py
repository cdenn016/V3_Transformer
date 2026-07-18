"""Plotting-free validation helpers for loading one finalized run."""

import json
from pathlib import Path
from typing import Mapping

import torch

from vfe3.config import VFE3Config, config_from_serialized, migrate_serialized_config
from vfe3.run_artifacts import _selection_semantic_config, semantic_config_fingerprint


def load_run_config(run_dir: Path) -> 'tuple[VFE3Config, str]':
    r"""Rebuild ``(cfg, dataset)`` from one RunArtifacts ``config.json``."""
    path = run_dir / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not isinstance(data.get("config"), Mapping):
        raise ValueError(f"run metadata {path} has no config mapping")
    raw_config = data["config"]
    stored_fingerprint = data.get("config_fingerprint")
    if (stored_fingerprint is not None
            and stored_fingerprint != semantic_config_fingerprint(raw_config)):
        raise ValueError(f"run metadata {path} has a config fingerprint mismatch")
    cfg = config_from_serialized(raw_config, source=str(path))
    dataset = data.get("dataset", "")
    if not isinstance(dataset, str):
        raise ValueError(f"run metadata {path} has a non-string dataset")
    return cfg, dataset


def load_best_model_state(
    path: Path,
    cfg:  VFE3Config,

    *,
    map_location: object,
) -> Mapping[str, torch.Tensor]:
    """Validate and unwrap a self-bound best-model bundle for strict model loading."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    required = {"model_state", "config", "config_fingerprint"}
    if not isinstance(payload, Mapping) or not payload or not required.issubset(payload):
        raise ValueError(f"best checkpoint {path} is not a self-bound model/config bundle")
    embedded = payload["config"]
    if not isinstance(embedded, Mapping) or not embedded:
        raise ValueError(f"best checkpoint {path} has no embedded config mapping")
    if payload["config_fingerprint"] != semantic_config_fingerprint(embedded):
        raise ValueError(f"best checkpoint {path} has a config fingerprint mismatch")
    embedded_cfg = migrate_serialized_config(
        embedded,
        source=f"{path} embedded config",
        strict_unknown=True,
    ).config
    if _selection_semantic_config(embedded_cfg) != _selection_semantic_config(cfg):
        raise ValueError(f"best checkpoint {path} has a semantic config mismatch with config.json")
    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError(f"best checkpoint {path} must contain a nonempty model_state mapping")
    return model_state
