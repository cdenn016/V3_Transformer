r"""Tack-on accumulation for the ablation runner.

The "tack on a second value list to the first sweep's figure" behavior lives entirely in the
pure-data helpers (``_collect_sweep_results`` / ``_write_sweep_csv`` / ``_read_sweep_csv``), so it
is provable with zero training: write fake ``ablation_result.json`` markers and assert on the
accumulated frame. ``_plot_one_sweep`` is exercised only for "does not raise" (it is best-effort
and silently skips when matplotlib is unavailable), so the discriminating checks are data-level.
"""

import json
import math
from pathlib import Path

import pytest

import ablation
from vfe3.viz import figure_worker


_CODE_IDENTITY = {
    "git_sha": "a" * 40,
    "git_dirty": False,
    "git_dirty_fingerprint": None,
}


def test_runner_identity_ignores_declaration_only_tack_on_but_binds_logic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "ablation.py"
    source.write_text(
        "SWEEPS = {'probe': {'values': [1]}}\n"
        "CONFIG = {'sweep': 'probe'}\n"
        "def run():\n    return 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ablation, "__file__", str(source))
    before = ablation._ablation_runner_source_sha256()
    monkeypatch.setattr(ablation, "_PROCESS_ABLATION_RUNNER_SHA256", before)

    source.write_text(
        "SWEEPS = {'probe': {'values': [1, 2]}}\n"
        "CONFIG = {'sweep': 'probe'}\n"
        "def run():\n    return 1\n",
        encoding="utf-8",
    )
    assert ablation._ablation_runner_source_sha256() == before
    assert ablation._verified_ablation_runner_source_sha256() == before

    source.write_text(
        "SWEEPS = {'probe': {'values': [1, 2]}}\n"
        "CONFIG = {'sweep': 'probe'}\n"
        "def run():\n    return 2\n",
        encoding="utf-8",
    )
    assert ablation._ablation_runner_source_sha256() != before
    with pytest.raises(RuntimeError, match="restart the Spyder kernel"):
        ablation._verified_ablation_runner_source_sha256()


def test_validate_sweeps_rejects_duplicate_expanded_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "duplicate_labels"
    monkeypatch.setitem(ablation.SWEEPS, name, {
        "description": "duplicate labels",
        "configs": [
            {"label": "same", "kappa_beta": 0.5},
            {"label": "same", "kappa_beta": 1.0},
        ],
    })

    with pytest.raises(ValueError, match="duplicate.*label"):
        ablation.validate_sweeps([name])


@pytest.mark.parametrize("name", ("../escape", "figures", "__sensitivity__"))
def test_validate_sweeps_rejects_unsafe_or_reserved_names(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(ablation.SWEEPS, name, {
        "description": "unsafe name",
        "configs": [{"label": "cell"}],
    })

    with pytest.raises(ValueError, match="sweep name"):
        ablation.validate_sweeps([name])


def test_prepare_owned_output_child_rejects_a_junction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "sweep"
    child.mkdir()
    monkeypatch.setattr(ablation, "_path_is_junction", lambda path: path == child)

    with pytest.raises(ValueError, match="junction|reparse"):
        ablation._prepare_owned_output_child(tmp_path, "sweep", role="ablation sweep")


def test_cell_generation_rejects_and_preserves_an_unowned_nonempty_directory(
    tmp_path: Path,
) -> None:
    sweep_dir = tmp_path / "sweep"
    run_dir = sweep_dir / "cell"
    run_dir.mkdir(parents=True)
    foreign = run_dir / "user-notes.txt"
    foreign.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="ownership"):
        ablation._start_owned_cell_generation(
            sweep_dir,
            run_dir,
            sweep_name="sweep",
            label="cell",
            seed=6,
        )

    assert foreign.read_text(encoding="utf-8") == "preserve me"
    assert sorted(path.name for path in run_dir.iterdir()) == ["user-notes.txt"]


def test_cell_generation_promotes_one_exact_legacy_marker_before_cleanup(
    tmp_path: Path,
) -> None:
    sweep_dir = tmp_path / "sweep"
    run_dir = sweep_dir / "cell"
    run_dir.mkdir(parents=True)
    (run_dir / "ablation_result.json").write_text(
        json.dumps({"status": "success", "label": "cell", "seed": 6}),
        encoding="utf-8",
    )
    stale = run_dir / "stale.txt"
    stale.write_text("old generation", encoding="utf-8")

    ablation._start_owned_cell_generation(
        sweep_dir,
        run_dir,
        sweep_name="sweep",
        label="cell",
        seed=6,
    )

    owner = json.loads(
        (run_dir / "ablation_cell_owner.json").read_text(encoding="utf-8")
    )
    assert owner == {
        "schema_version": 1,
        "sweep": "sweep",
        "label": "cell",
        "seed": 6,
    }
    running = json.loads(
        (run_dir / "ablation_result.json").read_text(encoding="utf-8")
    )
    assert running == {
        "status": "running",
        "sweep": "sweep",
        "label": "cell",
        "seed": 6,
    }
    assert not stale.exists()


def test_cell_generation_rejects_a_mismatched_owner_without_legacy_fallback(
    tmp_path: Path,
) -> None:
    sweep_dir = tmp_path / "sweep"
    run_dir = sweep_dir / "cell"
    run_dir.mkdir(parents=True)
    owner_path = run_dir / "ablation_cell_owner.json"
    owner_path.write_text(
        json.dumps({
            "schema_version": 1,
            "sweep": "sweep",
            "label": "cell",
            "seed": 7,
        }),
        encoding="utf-8",
    )
    marker_path = run_dir / "ablation_result.json"
    marker_path.write_text(
        json.dumps({
            "status": "success",
            "sweep": "sweep",
            "label": "cell",
            "seed": 6,
        }),
        encoding="utf-8",
    )
    foreign = run_dir / "user-notes.txt"
    foreign.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="ownership"):
        ablation._start_owned_cell_generation(
            sweep_dir,
            run_dir,
            sweep_name="sweep",
            label="cell",
            seed=6,
        )

    assert json.loads(owner_path.read_text(encoding="utf-8"))["seed"] == 7
    assert json.loads(marker_path.read_text(encoding="utf-8"))["status"] == "success"
    assert foreign.read_text(encoding="utf-8") == "preserve me"


def test_cell_generation_preserves_a_valid_owner_while_cleaning_owned_artifacts(
    tmp_path: Path,
) -> None:
    sweep_dir = tmp_path / "sweep"
    run_dir = sweep_dir / "cell"
    run_dir.mkdir(parents=True)
    owner_path = run_dir / "ablation_cell_owner.json"
    expected_owner = {
        "schema_version": 1,
        "sweep": "sweep",
        "label": "cell",
        "seed": 6,
    }
    owner_path.write_text(json.dumps(expected_owner), encoding="utf-8")
    (run_dir / "ablation_result.json").write_text(
        json.dumps({
            "status": "failed",
            "sweep": "sweep",
            "label": "cell",
            "seed": 6,
        }),
        encoding="utf-8",
    )
    (run_dir / "stale.txt").write_text("old generation", encoding="utf-8")
    stale_dir = run_dir / "figures"
    stale_dir.mkdir()
    (stale_dir / "stale.png").write_bytes(b"old figure")

    ablation._start_owned_cell_generation(
        sweep_dir,
        run_dir,
        sweep_name="sweep",
        label="cell",
        seed=6,
    )

    assert json.loads(owner_path.read_text(encoding="utf-8")) == expected_owner
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "ablation_cell_owner.json",
        "ablation_result.json",
    ]


def test_requested_outputs_require_every_declared_diagnostic_and_real_extrapolation() -> None:
    incomplete = {
        "attn_entropy": 1.0,
        "extrap_ce": [],
    }
    assert ablation._requested_outputs_are_complete(
        incomplete,
        required_diagnostic_keys=("attn_entropy", "builder_resid"),
        min_extrapolation_points=2,
    ) is False

    complete = {
        "attn_entropy": 1.0,
        "builder_resid": 1e-7,
        "extrap_ce": [
            {"n": 16, "ce": 3.0, "ppl": 20.0},
            {"n": 32, "ce": 3.1, "ppl": 22.2},
        ],
    }
    assert ablation._requested_outputs_are_complete(
        complete,
        required_diagnostic_keys=("attn_entropy", "builder_resid"),
        min_extrapolation_points=2,
    ) is True


def test_terminal_checkpoint_identity_is_owned_and_revalidated(tmp_path: Path) -> None:
    run_dir = tmp_path / "cell"
    checkpoint = run_dir / "checkpoints" / "step_2.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint generation one")
    marker = {"terminal_checkpoint": str(checkpoint)}

    identity = ablation._terminal_checkpoint_identity(run_dir, marker)
    assert identity is not None
    marker.update(identity)
    assert ablation._terminal_checkpoint_is_current(run_dir, marker) is True

    checkpoint.write_bytes(b"checkpoint generation two")
    assert ablation._terminal_checkpoint_is_current(run_dir, marker) is False


def test_gauge_disclosure_names_the_nonintertwiner() -> None:
    disclosure = ablation._gauge_disclosure_text({
        "classifications_by_label": {
            "baseline": "independent_head_nonintertwiner",
            "pure": "disabled",
        },
        "contains_independent_head_nonintertwiner": True,
        "all_rows_on_gauge_pure_path": False,
    })

    assert "independent_head_nonintertwiner" in disclosure
    assert "not gauge-pure" in disclosure


def _source_identities() -> dict[str, dict[str, object]]:
    return {
        split: {
            "format": "pt",
            "tokenizer_tag": "tiktoken",
            "size_bytes": len(split),
            "sha256": split[0] * 64,
            "meta": None,
            "meta_sha256": None,
        }
        for split in ("train", "validation")
    }


def _aggregation_contract(sweep_dir: Path) -> dict[str, object]:
    meta_path = sweep_dir / "sweep_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))["aggregation_contract"]
    contract = ablation._sweep_aggregation_contract(
        dataset="wikitext-103",
        diagnostic_flags={
            "collect_diagnostics": False,
            "collect_extrapolation": False,
            "paired_token_bootstrap": False,
        },
        data_seed_override=ablation.DATA_SEED,
        max_tokens=None,
        max_steps=None,
        seed_design=[6],
        source_identities=_source_identities(),
        code_identity=_CODE_IDENTITY,
    )
    meta_path.write_text(
        json.dumps({"status": "complete", "aggregation_contract": contract}),
        encoding="utf-8",
    )
    return contract


def _label_overrides(label: str) -> dict[str, object]:
    if label.startswith("kappa="):
        return {"kappa_beta": float(label.split("=", 1)[1])}
    return {}


def _write_marker(
    sweep_dir: Path,
    label: str,
    ppl: float,
    *,
    seed: int = 6,
    overrides: dict[str, object] | None = None,
    contract_mutation=None,
) -> None:
    r"""Write one success marker bound to the sweep's persisted aggregation contract."""
    aggregation = _aggregation_contract(sweep_dir)
    overrides = _label_overrides(label) if overrides is None else overrides
    cfg = ablation.VFE3Config(**ablation._cell_cfg_dict(overrides, seed=seed))
    sources = aggregation["source_identities"]
    contract = ablation._cell_contract(
        cfg,
        aggregation["dataset"],
        aggregation["diagnostic_flags"],
        data_seed=(aggregation["data_seed_override"]
                   if aggregation["data_seed_override"] is not None else seed),
        max_tokens=aggregation["max_tokens"],
        source_identities=sources,
        code_identity=aggregation["code_identity"],
    )
    if contract_mutation is not None:
        contract_mutation(contract)
    gauge_fields = ablation._gauge_reporting_fields(cfg)
    cell = sweep_dir / ablation._sanitize(label)
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "ablation_cell_owner.json").write_text(
        json.dumps(ablation._ablation_cell_owner_payload(sweep_dir.name, label, seed)),
        encoding="utf-8",
    )
    checkpoint = cell / "checkpoints" / "terminal.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint:{label}:{seed}".encode("utf-8"))
    checkpoint_fields = {"terminal_checkpoint": str(checkpoint)}
    checkpoint_identity = ablation._terminal_checkpoint_identity(cell, checkpoint_fields)
    assert checkpoint_identity is not None
    checkpoint_fields.update(checkpoint_identity)
    (cell / "cell_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (cell / "ablation_result.json").write_text(
        json.dumps({
            "sweep": sweep_dir.name, "label": label, "error_kind": None,
            "status": "success", "primary_val_ppl": ppl, "final_val_ppl": ppl,
            "n_params": 1000, "seed": seed, "overrides": overrides,
            "cell_contract_fingerprint": ablation.semantic_config_fingerprint(contract),
            **aggregation["diagnostic_flags"],
            **checkpoint_fields,
            **gauge_fields,
        }),
        encoding="utf-8",
    )


def test_collect_union_and_tack_on(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()

    # First sweep: kappa = 1, 2, 3, 4.
    for v in (1, 2, 3, 4):
        _write_marker(sweep_dir, f"kappa={v}", ppl=10.0 + v)
    union = ablation._collect_sweep_results(sweep_dir)
    assert len(union) == 4
    ablation._write_sweep_csv(sweep_dir, union)
    assert len(ablation._read_sweep_csv(sweep_dir)) == 4

    # Later: tack on kappa = 0.5, 2.2, 3.7 (a DIFFERENT value list).
    for v in (0.5, 2.2, 3.7):
        _write_marker(sweep_dir, f"kappa={v}", ppl=20.0 + v)
    union2 = ablation._collect_sweep_results(sweep_dir)
    assert len(union2) == 7                                  # old four + new three, not replaced
    ablation._write_sweep_csv(sweep_dir, union2)
    rows = ablation._read_sweep_csv(sweep_dir)
    assert len(rows) == 7

    # The merged figure's x-axis: union of both value lists, sorted by numeric value.
    xs = sorted(float(r["label"].split("=")[-1]) for r in rows)
    assert xs == [0.5, 1.0, 2.0, 2.2, 3.0, 3.7, 4.0]


def test_rerun_same_label_overwrites(tmp_path: Path) -> None:
    r"""Re-running the SAME label updates that one cell (no duplicate point)."""
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    _write_marker(sweep_dir, "kappa=1", ppl=11.0)
    _write_marker(sweep_dir, "kappa=1", ppl=99.0)            # same dir -> overwrites the marker
    union = ablation._collect_sweep_results(sweep_dir)
    assert len(union) == 1
    assert union[0]["primary_val_ppl"] == 99.0


def test_int_float_spellings_stay_distinct(tmp_path: Path) -> None:
    r"""kappa=2 and kappa=2.0 sanitize to different dirs -> two points (documented caveat)."""
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    _write_marker(sweep_dir, "kappa=2", ppl=12.0)
    _write_marker(sweep_dir, "kappa=2.0", ppl=13.0)
    assert len(ablation._collect_sweep_results(sweep_dir)) == 2


def test_unreadable_marker_is_skipped(tmp_path: Path) -> None:
    r"""A partial/corrupt marker is skipped, not fatal, so the rest of the union survives."""
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    _write_marker(sweep_dir, "kappa=1", ppl=11.0)
    bad = sweep_dir / "kappa_2"
    bad.mkdir()
    (bad / "ablation_result.json").write_text("{not valid json", encoding="utf-8")
    union = ablation._collect_sweep_results(sweep_dir)
    assert len(union) == 1
    assert union[0]["label"] == "kappa=1"


def test_collect_sweep_results_rejects_malformed_failed_and_nonfinite_markers(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    _write_marker(sweep_dir, "valid", ppl=11.0)
    invalid = [
        [],
        {"label": "failed", "status": "failed", "error_kind": "train", "final_val_ppl": 11.0},
        {"label": "errored", "status": "success", "error_kind": "train", "final_val_ppl": 11.0},
        {"label": "infinite", "status": "success", "error_kind": None,
         "final_val_ppl": float("inf")},
        {"label": "missing", "status": "success", "error_kind": None},
    ]
    for i, marker in enumerate(invalid):
        cell = sweep_dir / f"invalid_{i}"
        cell.mkdir()
        (cell / "ablation_result.json").write_text(json.dumps(marker), encoding="utf-8")

    union = ablation._collect_sweep_results(sweep_dir)
    assert [marker["label"] for marker in union] == ["valid"]


def test_collect_sweep_results_rejects_a_mismatched_owner(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    _write_marker(sweep_dir, "valid-looking", ppl=11.0, seed=6)
    run_dir = sweep_dir / ablation._sanitize("valid-looking")
    (run_dir / "ablation_cell_owner.json").write_text(
        json.dumps(ablation._ablation_cell_owner_payload(sweep_dir.name, "valid-looking", 7)),
        encoding="utf-8",
    )

    assert ablation._collect_sweep_results(sweep_dir) == []


def test_collect_sweep_results_accepts_only_current_compatible_contract_cohort(
    tmp_path: Path,
) -> None:
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    _write_marker(sweep_dir, "kappa=1", ppl=11.0, seed=6)
    _write_marker(sweep_dir, "kappa=2", ppl=12.0, seed=23)
    _write_marker(
        sweep_dir,
        "wrong_code",
        ppl=1.0,
        contract_mutation=lambda contract: contract["code_identity"].update(
            {"git_sha": "b" * 40}
        ),
    )
    _write_marker(
        sweep_dir,
        "wrong_budget",
        ppl=2.0,
        contract_mutation=lambda contract: contract.update(
            {"semantic_config_fingerprint": "f" * 64}
        ),
    )
    missing = sweep_dir / ablation._sanitize("missing_contract")
    missing.mkdir()
    (missing / "ablation_result.json").write_text(
        json.dumps({
            "sweep": sweep_dir.name,
            "label": "missing_contract",
            "status": "success",
            "error_kind": None,
            "primary_val_ppl": 3.0,
            "final_val_ppl": 3.0,
            "seed": 6,
            "overrides": {},
            "cell_contract_fingerprint": "0" * 64,
        }),
        encoding="utf-8",
    )

    rows = ablation._collect_sweep_results(sweep_dir)

    assert [(row["label"], row["seed"]) for row in rows] == [
        ("kappa=1", 6),
    ]

    aggregation = _aggregation_contract(sweep_dir)
    aggregation["seed_design"] = [23]
    rows = ablation._collect_sweep_results(
        sweep_dir,
        aggregation_contract=aggregation,
    )

    assert [(row["label"], row["seed"]) for row in rows] == [
        ("kappa=2", 23),
    ]


def test_collect_sweep_results_requires_a_complete_exact_seed_panel_per_base(
    tmp_path: Path,
) -> None:
    sweep_dir = tmp_path / "seed_panel"
    sweep_dir.mkdir()
    aggregation = _aggregation_contract(sweep_dir)
    aggregation["seed_design"] = [6, 23]
    (sweep_dir / "sweep_meta.json").write_text(
        json.dumps({"status": "complete", "aggregation_contract": aggregation}),
        encoding="utf-8",
    )

    _write_marker(sweep_dir, "historical_incomplete__s6", ppl=11.0, seed=6)
    _write_marker(sweep_dir, "historical_complete__s6", ppl=12.0, seed=6)
    _write_marker(sweep_dir, "historical_complete__s23", ppl=13.0, seed=23)
    _write_marker(sweep_dir, "requested_incomplete__s6", ppl=14.0, seed=6)
    _write_marker(sweep_dir, "duplicate__s6", ppl=15.0, seed=6)
    _write_marker(sweep_dir, "duplicate__s06", ppl=16.0, seed=6)
    _write_marker(sweep_dir, "duplicate__s23", ppl=17.0, seed=23)

    rows = ablation._collect_sweep_results(sweep_dir)

    assert sorted((row["label"], row["seed"]) for row in rows) == [
        ("historical_complete__s23", 23),
        ("historical_complete__s6", 6),
    ]
    admitted_labels = {row["label"] for row in rows}
    assert "historical_incomplete__s6" not in admitted_labels
    requested_labels = {"requested_incomplete__s6", "requested_incomplete__s23"}
    assert requested_labels.isdisjoint(admitted_labels)
    assert not any(label.startswith("duplicate__s") for label in admitted_labels)


def test_plot_one_sweep_does_not_raise(tmp_path: Path) -> None:
    r"""Best-effort plotting must never raise, with or without matplotlib installed."""
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    for v in (1, 2, 3):
        _write_marker(sweep_dir, f"kappa={v}", ppl=10.0 + v)
    ablation._write_sweep_csv(sweep_dir, ablation._collect_sweep_results(sweep_dir))
    ablation._plot_one_sweep(sweep_dir, tmp_path / "figures")


def test_ablation_figures_run_in_child_with_scoped_openmp_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "ablation"
    output_dir.mkdir()
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    captured: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        captured["timeout"] = kwargs["timeout"]
        request_path = Path(kwargs["env"]["VFE3_FIGURE_REQUEST"])
        captured["request"] = json.loads(request_path.read_text(encoding="utf-8"))
        return ablation.subprocess.CompletedProcess(command, 0, "rendered", "")

    monkeypatch.setattr(ablation, "run_process_tree", _fake_run)

    assert ablation._run_ablation_figures_isolated(output_dir, scope="kappa_beta") is True
    assert "KMP_DUPLICATE_LIB_OK" not in ablation.os.environ
    assert captured["environment"]["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert captured["command"][-1] == "vfe3.viz.figure_worker"
    assert captured["timeout"] == ablation._ABLATION_FIGURE_TIMEOUT_SECONDS
    assert captured["request"] == {
        "mode": "ablation",
        "run_dir": str(output_dir.resolve()),
        "scope": "kappa_beta",
        "cohort_identity": None,
        "invalidate": False,
    }


def test_ablation_figure_invalidation_request_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "ablation"
    output_dir.mkdir()
    captured: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        del command
        request_path = Path(kwargs["env"]["VFE3_FIGURE_REQUEST"])
        captured.update(json.loads(request_path.read_text(encoding="utf-8")))
        return ablation.subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(ablation, "run_process_tree", _fake_run)

    assert ablation._run_ablation_figures_isolated(
        output_dir,
        scope="kappa_beta",
        invalidate=True,
    ) is True
    assert captured["invalidate"] is True


def test_figure_worker_invalidation_retires_manifest_and_legacy_outputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "ablation"
    output_dir.mkdir()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir()
    legacy_names = figure_worker._legacy_ablation_scope_files("gauge_transport")
    for name in legacy_names:
        (figure_dir / name).write_bytes(b"stale")

    figure_worker._render_ablation_request({
        "scope": "gauge_transport",
        "invalidate": True,
    }, output_dir)

    assert not any((figure_dir / name).exists() for name in legacy_names)
    manifest = json.loads(
        (figure_dir / "ablation_figure_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["scopes"]["gauge_transport"]["files"] == []


def test_sensitivity_invalidation_retires_pre_manifest_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "ablation"
    output_dir.mkdir()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir()
    stale = figure_dir / "sensitivity_summary.png"
    stale.write_bytes(b"stale")

    figure_worker._render_ablation_request({
        "scope": "__sensitivity__",
        "invalidate": True,
    }, output_dir)

    assert not stale.exists()


def test_figure_worker_rejects_manifest_filename_outside_scope_inventory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "ablation"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True)
    foreign = figure_dir / "foreign.txt"
    foreign.write_text("preserve me", encoding="utf-8")
    (figure_dir / "ablation_figure_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "scopes": {"gauge_transport": {"files": [foreign.name]}},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="unowned filename"):
        figure_worker._render_ablation_request({
            "scope": "gauge_transport",
            "invalidate": True,
        }, output_dir)

    assert foreign.read_text(encoding="utf-8") == "preserve me"


def test_figure_worker_rejects_malformed_existing_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "ablation"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True)
    manifest = figure_dir / "ablation_figure_manifest.json"
    manifest.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="unreadable"):
        figure_worker._render_ablation_request({
            "scope": "gauge_transport",
            "invalidate": True,
        }, output_dir)


def test_get_loader_threads_split_aware_shuffle_drop_last(monkeypatch) -> None:
    r"""ablation.get_loader must mirror train_vfe3._select_loader's F1 split-aware semantics:
    train requests shuffle=True/drop_last=True, validation/test request shuffle=False/drop_last=False,
    so the held-out metric reads the WHOLE split in a stable order (datasets.make_dataloader defaults
    to the TRAIN regime, so get_loader must pass the eval flags explicitly)."""
    captured: dict = {}

    def fake_make_dataloader(dataset, split, seq_len, batch_size, **kw):
        captured[split] = kw
        return object()                                      # a non-None sentinel get_loader caches

    monkeypatch.setattr(ablation, "make_dataloader", fake_make_dataloader)
    monkeypatch.setattr(ablation, "cache_source_identity", lambda dataset, split: {
        "format": "pt", "tokenizer_tag": "fixture", "size_bytes": len(split),
        "sha256": "0" * 64, "meta": None, "meta_sha256": None,
    })
    ablation._LOADER_CACHE.clear()
    ablation.get_loader("wikitext-103", 16, 4, "validation")
    ablation.get_loader("wikitext-103", 16, 4, "train", max_tokens=None)
    ablation._LOADER_CACHE.clear()
    assert captured["validation"].get("shuffle") is False
    assert captured["validation"].get("drop_last") is False
    assert captured["train"].get("shuffle") is True
    assert captured["train"].get("drop_last") is True


def test_run_sweep_markers_persist_requests_and_terminal_state(tmp_path: Path, monkeypatch) -> None:
    # Per-cell _cell_is_current staleness (dataset / max_tokens / diagnostic-flag / marker-validity)
    # is now bound to cell_contract.json and covered in tests/test_ablation_artifact_resume_20260712.py.
    # Stub the contract's code + corpus identity so this run-sweep marker test stays fast, deterministic,
    # and independent of any real tokenized cache on disk.
    monkeypatch.setattr(ablation, "_git_code_identity",
                        lambda: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})
    monkeypatch.setattr(ablation, "cache_source_identity",
                        lambda dataset, split, *, cache_dir=None: {
                            "format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": len(split),
                            "sha256": "0" * 64, "meta": None, "meta_sha256": None})

    sweep_name = "marker_contract"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {
        "description": "marker contract test",
        "collect_diagnostics": True,
        "collect_extrapolation": True,
        "extrapolation_lengths": [16, 32],
        "mandatory_extrapolation_lengths": [16, 32],
    })
    monkeypatch.setattr(ablation, "make_run_overrides", lambda _name: [
        ("success", {}), ("failure", {}),
    ])

    def _fake_run_single(label, _overrides, run_dir, **kwargs):
        assert kwargs["collect_diagnostics"] is True
        assert kwargs["collect_extrapolation"] is True
        if label == "success":
            checkpoint = run_dir / "checkpoints" / "terminal.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"owned terminal checkpoint")
            return {
                "label": label,
                "error_kind": None,
                "primary_val_ppl": 8.0,
                "final_val_ppl": 9.0,
                "attn_entropy": 1.0,
                "energy_klmax_frac": 0.1,
                "gauge_resid_in": 1e-7,
                "gauge_resid_out": 1e-7,
                "omega_identity_dev": 0.2,
                "rank_resid": 0.8,
                    "terminal_checkpoint": str(checkpoint),
                    "extrap_ce": [
                        {"n": 16, "status": "success", "ce": 3.0, "ppl": 20.0,
                         "effective_batch_size": 4},
                        {"n": 32, "status": "success", "ce": 3.1, "ppl": 22.2,
                         "effective_batch_size": 2},
                ],
                "_loaded_data_sources": {
                    split: {
                        "format": "pt", "tokenizer_tag": "tiktoken",
                        "size_bytes": len(split), "sha256": "0" * 64,
                        "meta": None, "meta_sha256": None,
                    }
                    for split in ("train", "validation")
                },
            }
        return {
            "label": label,
            "error_kind": "train",
            "error": "boom",
            "primary_val_ppl": float("inf"),
        }

    monkeypatch.setattr(ablation, "run_single", _fake_run_single)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    returned = ablation.run_sweep(
        sweep_name, tmp_path, dataset="wikitext-103", device=None, seed=6, resume=False,
    )

    markers = {}
    for label in ("success", "failure"):
        path = tmp_path / sweep_name / ablation._sanitize(label) / "ablation_result.json"
        markers[label] = json.loads(path.read_text(encoding="utf-8"))
        assert markers[label]["collect_diagnostics"] is True
        assert markers[label]["collect_extrapolation"] is True
        assert "error_kind" in markers[label]

    assert markers["success"]["status"] == "success"
    assert markers["success"]["error_kind"] is None
    assert math.isfinite(markers["success"]["final_val_ppl"])
    assert markers["failure"]["status"] == "failed"
    assert markers["failure"]["error_kind"] == "train"
    assert "final_val_ppl" in markers["failure"]
    assert not math.isfinite(markers["failure"]["final_val_ppl"])

    # A reuse contract is published only for the successful cell (never for the failed one).
    success_dir = tmp_path / sweep_name / ablation._sanitize("success")
    failure_dir = tmp_path / sweep_name / ablation._sanitize("failure")
    assert (success_dir / "cell_contract.json").exists()
    assert not (failure_dir / "cell_contract.json").exists()

    # A survivor-only invocation is not complete and cannot flow into reporting as if both
    # requested cells finished. The compatible success may remain in the CSV for diagnosis.
    meta = json.loads(
        (tmp_path / sweep_name / "sweep_meta.json").read_text(encoding="utf-8")
    )
    assert returned == []
    assert meta["status"] == "incomplete"
    assert meta["n_successful_requested"] == 1
    assert meta["n_runs"] == 2
    assert meta["failed_requested_labels"] == ["failure"]

    # The intentional baseline head mixer is preserved, but every generated reporting surface
    # states that its independent-head mixer is not a gauge intertwiner.
    assert markers["success"]["head_mixer_compatibility"] == "independent_head_nonintertwiner"
    assert markers["success"]["head_mixer_gauge_compatible"] is False
    assert markers["success"]["on_gauge_pure_path"] is False
    rows = ablation._read_sweep_csv(tmp_path / sweep_name)
    assert rows[0]["head_mixer_compatibility"] == "independent_head_nonintertwiner"
    assert rows[0]["head_mixer_gauge_compatible"] == "False"
    assert rows[0]["on_gauge_pure_path"] == "False"
    assert meta["gauge_purity"]["contains_independent_head_nonintertwiner"] is True
    assert meta["gauge_purity"]["all_rows_on_gauge_pure_path"] is False


def test_recompute_owns_a_clean_cell_generation_and_cached_resume_preserves_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sweep_name = "generation_cleanliness"
    sweep_dir = tmp_path / sweep_name
    run_dir = sweep_dir / ablation._sanitize("cell")
    run_dir.mkdir(parents=True)
    (run_dir / "ablation_result.json").write_text(
        json.dumps({"status": "success", "label": "cell", "seed": 6}), encoding="utf-8"
    )
    (run_dir / "cell_contract.json").write_text("{}", encoding="utf-8")
    (run_dir / "val_token_nats.pt").write_bytes(b"stale token vector")
    (run_dir / "figures").mkdir()
    (run_dir / "figures" / "stale.png").write_bytes(b"stale figure")
    outside = sweep_dir / "unrelated.txt"
    outside.write_text("preserve", encoding="utf-8")

    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "generation ownership"})
    monkeypatch.setattr(ablation, "make_run_overrides", lambda _name: [("cell", {})])
    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(_CODE_IDENTITY))
    monkeypatch.setattr(
        ablation,
        "cache_source_identity",
        lambda dataset, split, *, cache_dir=None: _source_identities()[split],
    )
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    calls: list[str] = []

    def fresh_run(label, overrides, owned_run_dir, **kwargs):
        del overrides, kwargs
        calls.append(label)
        assert owned_run_dir == run_dir
        assert sorted(path.name for path in run_dir.iterdir()) == [
            "ablation_cell_owner.json",
            "ablation_result.json",
        ]
        running = json.loads(
            (run_dir / "ablation_result.json").read_text(encoding="utf-8")
        )
        assert running["status"] == "running"
        assert running["sweep"] == sweep_name
        (run_dir / "fresh.txt").write_text("new generation", encoding="utf-8")
        checkpoint = run_dir / "checkpoints" / "terminal.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"owned terminal checkpoint")
        return {
            "label": label,
            "error_kind": None,
            "primary_val_ppl": 8.0,
            "final_val_ppl": 9.0,
            "terminal_checkpoint": str(checkpoint),
            "_loaded_data_sources": _source_identities(),
        }

    monkeypatch.setattr(ablation, "run_single", fresh_run)
    rows = ablation.run_sweep(
        sweep_name,
        tmp_path,
        dataset="wikitext-103",
        device=None,
        seed=6,
        resume=False,
    )

    assert calls == ["cell"]
    assert [row["label"] for row in rows] == ["cell"]
    assert (run_dir / "fresh.txt").read_text(encoding="utf-8") == "new generation"
    assert not (run_dir / "val_token_nats.pt").exists()
    assert not (run_dir / "figures").exists()
    assert outside.read_text(encoding="utf-8") == "preserve"

    cached_sentinel = run_dir / "cached-sentinel.txt"
    cached_sentinel.write_text("keep on valid resume", encoding="utf-8")
    monkeypatch.setattr(
        ablation,
        "run_single",
        lambda *args, **kwargs: pytest.fail("valid resume unexpectedly recomputed the cell"),
    )
    cached = ablation.run_sweep(
        sweep_name,
        tmp_path,
        dataset="wikitext-103",
        device=None,
        seed=6,
        resume=True,
    )

    assert [row["label"] for row in cached] == ["cell"]
    assert cached_sentinel.read_text(encoding="utf-8") == "keep on valid resume"
    assert outside.read_text(encoding="utf-8") == "preserve"

    owner_path = run_dir / "ablation_cell_owner.json"
    owner_path.write_text(
        json.dumps(ablation._ablation_cell_owner_payload(sweep_name, "cell", 7)),
        encoding="utf-8",
    )
    rejected = ablation.run_sweep(
        sweep_name,
        tmp_path,
        dataset="wikitext-103",
        device=None,
        seed=6,
        resume=True,
    )

    assert rejected == []
    assert json.loads(owner_path.read_text(encoding="utf-8"))["seed"] == 7
    assert json.loads(
        (run_dir / "ablation_result.json").read_text(encoding="utf-8")
    )["status"] == "success"
    assert cached_sentinel.read_text(encoding="utf-8") == "keep on valid resume"
    assert outside.read_text(encoding="utf-8") == "preserve"


def test_expand_range_sign_mismatch_raises() -> None:
    r"""A sign-mismatched [start, stop, step] must raise, not silently expand to zero cells."""
    with pytest.raises(ValueError):
        ablation._expand_range([0, 5, -1])
    with pytest.raises(ValueError):
        ablation._expand_range([5, 0, 1])


def test_expand_range_valid_directions_unchanged() -> None:
    r"""Ascending, descending, and the degenerate single-point range all still expand."""
    assert ablation._expand_range([0, 4, 2]) == [0, 2, 4]
    assert ablation._expand_range([5, 0, -1]) == [5, 4, 3, 2, 1, 0]
    assert ablation._expand_range([2, 2, 1]) == [2]


def test_main_records_global_figure_invalidation_failure_and_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep_name = "status_probe"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "status probe"})
    for key, value in {
        "list_only":  False,
        "output_dir": str(tmp_path),
        "device":     "cpu",
        "sweep":      sweep_name,
        "dataset":    "wikitext-103",
        "seed":       6,
        "resume":     False,
        "max_tokens": None,
        "max_steps":  None,
    }.items():
        monkeypatch.setitem(ablation.CONFIG, key, value)
    monkeypatch.setattr(ablation, "validate_sweeps", lambda _names: None)
    monkeypatch.setattr(
        ablation,
        "_run_ablation_figures_isolated",
        lambda *_args, **kwargs: not (
            kwargs.get("scope") == "__sensitivity__" and kwargs.get("invalidate") is True
        ),
    )
    monkeypatch.setattr(
        ablation,
        "run_sweep",
        lambda *_args, **_kwargs: pytest.fail("sweep ran after global invalidation failure"),
    )

    assert ablation.main() == 1
    status = json.loads((tmp_path / "ablation_run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "incomplete"
    assert status["requested_sweeps"] == [sweep_name]
    assert status["incomplete_sweeps"] == [sweep_name]
    assert status["failed_figure_scopes"] == ["__sensitivity__:invalidate"]


def test_main_records_requested_render_failure_and_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep_name = "render_status_probe"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "render status probe"})
    for key, value in {
        "list_only":  False,
        "output_dir": str(tmp_path),
        "device":     "cpu",
        "sweep":      sweep_name,
        "dataset":    "wikitext-103",
        "seed":       6,
        "resume":     False,
        "max_tokens": None,
        "max_steps":  None,
    }.items():
        monkeypatch.setitem(ablation.CONFIG, key, value)
    monkeypatch.setattr(ablation, "validate_sweeps", lambda _names: None)
    monkeypatch.setattr(
        ablation,
        "_cross_sweep_cohort_identity",
        lambda _contract: {"cohort": "one"},
    )
    monkeypatch.setattr(ablation, "analyze_sweep", lambda _path: None)
    monkeypatch.setattr(ablation, "summarize_sweeps", lambda *_args, **_kwargs: None)

    def fake_run_sweep(
        name:       str,
        output_dir: Path,

        **_kwargs: object,
    ) -> list:
        sweep_dir = output_dir / name
        sweep_dir.mkdir(parents=True, exist_ok=True)
        ablation._write_json_atomic(sweep_dir / "sweep_meta.json", {
            "status":               "complete",
            "aggregation_contract": {"fixture": True},
        })
        return []

    def fake_figures(
        _output_dir: Path,

        *,
        scope:           str,
        invalidate:      bool          = False,
        cohort_identity: object | None = None,
    ) -> bool:
        del cohort_identity
        return not (scope == sweep_name and not invalidate)

    monkeypatch.setattr(ablation, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(ablation, "_run_ablation_figures_isolated", fake_figures)

    assert ablation.main() == 1
    status = json.loads((tmp_path / "ablation_run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "incomplete"
    assert status["incomplete_sweeps"] == []
    assert status["failed_figure_scopes"] == [f"{sweep_name}:render"]


def test_sanitize_distinct_labels_do_not_collide() -> None:
    r"""The char-replace map is lossy ('a=b', 'a b', 'a/b' all map to 'a_b'), so the appended
    raw-label hash must keep distinct labels in distinct run dirs, deterministically."""
    assert len({ablation._sanitize("a=b"), ablation._sanitize("a b"),
                ablation._sanitize("a/b")}) == 3
    assert ablation._sanitize("kappa=2") == ablation._sanitize("kappa=2")
