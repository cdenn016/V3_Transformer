r"""Internal worker for end-of-run figures.

The parent process supplies one request path through ``VFE3_FIGURE_REQUEST``.  This module has no
user-facing argument parser; it exists only so training can contain native plotting-runtime failure
inside a disposable interpreter.
"""

import csv
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping

import torch

from vfe3.config import config_from_serialized
from vfe3.run_artifacts import _save_figures


def _history_from_csv(path: Path) -> List[Dict[str, object]]:
    """Rebuild the numeric in-memory history used by the ordinary figure pass."""
    if not path.is_file():
        return []
    history: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: Dict[str, object] = {}
            for key, raw in row.items():
                if raw is None or raw == "":
                    parsed[key] = float("nan")
                    continue
                try:
                    parsed[key] = float(raw)
                except ValueError:
                    parsed[key] = raw
            history.append(parsed)
    return history


def main() -> int:
    """Execute one validated figure request and return a process exit code."""
    raw_request = os.environ.get("VFE3_FIGURE_REQUEST")
    if not raw_request:
        raise RuntimeError("VFE3_FIGURE_REQUEST is required")
    request_path = Path(raw_request).resolve(strict=True)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, Mapping):
        raise ValueError("figure request must be a JSON object")

    run_dir = Path(request["run_dir"]).resolve(strict=True)
    metadata_path = run_dir / "config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("config"), Mapping):
        raise ValueError(f"run metadata {metadata_path} has no config mapping")
    cfg = config_from_serialized(metadata["config"], source=str(metadata_path))
    history = _history_from_csv(run_dir / "metrics.csv")
    losses_payload = request.get("losses")
    losses = (
        [float(value) for value in losses_payload]
        if isinstance(losses_payload, list)
        else None
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    artifacts = SimpleNamespace(run_dir=run_dir, cfg=cfg, history=history)

    from vfe3.viz.figures import plot_estep_depth_sensitivity, plot_phi_numerics_reference

    saved_probes = (
        (
            "estep_depth_sensitivity.json",
            "estep_depth_sensitivity.png",
            plot_estep_depth_sensitivity,
        ),
        ("phi_numerics.json", "phi_numerics_reference.png", plot_phi_numerics_reference),
    )
    for source_name, figure_name, plotter in saved_probes:
        source_path = run_dir / source_name
        if not source_path.is_file():
            continue
        try:
            record = json.loads(source_path.read_text(encoding="utf-8"))
            figure = plotter(record, path=str(run_dir / figure_name))
            if figure is not None:
                from matplotlib import pyplot as plt
                plt.close(figure)
        except Exception as exc:
            logger.warning(
                "saved probe figure %s failed (%s); remaining figures continue",
                figure_name,
                exc,
            )
    _save_figures(artifacts, losses, logger)

    if request.get("generate_publication") is True:
        from vfe3.viz.report import generate_figures

        batches_path = request.get("report_batches_path")
        model_bundle_path = request.get("model_bundle_path")
        loader = (
            torch.load(Path(batches_path), map_location="cpu", weights_only=True)
            if isinstance(batches_path, str)
            else None
        )
        generate_figures(
            run_dir,
            checkpoint_path=(
                Path(model_bundle_path) if isinstance(model_bundle_path, str) else None
            ),
            split="test",
            max_tokens=int(request.get("max_tokens", 16_384)),
            allow_large=bool(request.get("allow_large", False)),
            loader=loader,
            device=torch.device(str(request.get("device", "cpu"))),
            logger=logger,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
