"""Regression coverage for audited worker and subprocess boundary contracts."""

import json

import pytest

from vfe3 import process_utils
from vfe3.viz import figure_worker


def _finalize_request(run_dir, **overrides):
    request = {
        "run_dir":             str(run_dir),
        "losses":              [1.25],
        "generate_publication": False,
        "report_batches_path": None,
        "model_bundle_path":   None,
        "device":              "cpu",
        "max_tokens":          16_384,
        "allow_large":         False,
    }
    request.update(overrides)
    return request


@pytest.mark.parametrize(("field", "value"), [
    ("allow_large", "false"),
    ("allow_large", 1),
    ("generate_publication", 1),
    ("report_batches_path", 3),
    ("model_bundle_path", False),
    ("device", 0),
    ("losses", {"loss": 1.25}),
])
def test_finalize_request_rejects_invalid_json_fields_before_render(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    request_path = tmp_path / "figure_request.json"
    request_path.write_text(
        json.dumps(_finalize_request(tmp_path, **{field: value})),
        encoding="utf-8",
    )
    monkeypatch.setenv("VFE3_FIGURE_REQUEST", str(request_path))
    monkeypatch.setattr(figure_worker, "_load_worker_config", lambda _path: object())
    rendered = []
    monkeypatch.setattr(
        figure_worker,
        "_render_persisted_run_figures",
        lambda *_args: rendered.append(True),
    )

    with pytest.raises(ValueError, match=field):
        figure_worker.main()

    assert rendered == []


@pytest.mark.parametrize("field", [
    "run_dir",
    "losses",
    "generate_publication",
    "report_batches_path",
    "model_bundle_path",
    "device",
    "max_tokens",
    "allow_large",
])
def test_finalize_request_rejects_missing_required_field_before_render(
    tmp_path,
    monkeypatch,
    field,
):
    request = _finalize_request(tmp_path)
    del request[field]
    request_path = tmp_path / "figure_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setenv("VFE3_FIGURE_REQUEST", str(request_path))
    monkeypatch.setattr(figure_worker, "_load_worker_config", lambda _path: object())
    rendered = []
    monkeypatch.setattr(
        figure_worker,
        "_render_persisted_run_figures",
        lambda *_args: rendered.append(True),
    )

    with pytest.raises(ValueError, match=field):
        figure_worker.main()

    assert rendered == []


@pytest.mark.parametrize("losses", [[1.25], None])
def test_finalize_request_from_run_artifacts_schema_reaches_renderer(
    tmp_path,
    monkeypatch,
    losses,
):
    request_path = tmp_path / "figure_request.json"
    request_path.write_text(
        json.dumps(_finalize_request(tmp_path, losses=losses)),
        encoding="utf-8",
    )
    monkeypatch.setenv("VFE3_FIGURE_REQUEST", str(request_path))
    cfg = object()
    monkeypatch.setattr(figure_worker, "_load_worker_config", lambda _path: cfg)
    rendered = []
    monkeypatch.setattr(
        figure_worker,
        "_render_persisted_run_figures",
        lambda *args: rendered.append(args),
    )

    assert figure_worker.main() == 0
    assert rendered[0][0] == tmp_path.resolve()
    assert rendered[0][1] is cfg
    assert rendered[0][2] == losses


@pytest.mark.parametrize("command", [
    "python",
    b"python",
    [],
    ["python", ""],
    ["python", 3],
])
def test_run_process_tree_rejects_invalid_command_before_process_creation(
    monkeypatch,
    command,
):
    def _unexpected_popen(*_args, **_kwargs):
        raise AssertionError("process creation must not run for an invalid command")

    monkeypatch.setattr(process_utils.subprocess, "Popen", _unexpected_popen)

    with pytest.raises((TypeError, ValueError)):
        process_utils.run_process_tree(command)
