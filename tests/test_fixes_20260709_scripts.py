"""Regression tests for the July 9 click-to-run script repairs."""

import inspect
from pathlib import Path

import pytest

import check_junit
from vfe3.inference.e_step import e_step


def test_read_junit_counts_accepts_testsuite_root(tmp_path):
    xml_path = tmp_path / "single.xml"
    xml_path.write_text(
        '<testsuite tests="5" failures="1" errors="1" skipped="1">'
        '<testcase name="test_param[0]"/><testcase name="test_param[1]"/>'
        '</testsuite>',
        encoding="utf-8",
    )

    assert check_junit.read_junit_counts(xml_path) == {
        "tests": 5,
        "failures": 1,
        "errors": 1,
        "skipped": 1,
        "passes": 2,
    }


def test_read_junit_counts_sums_testsuites_root(tmp_path):
    xml_path = tmp_path / "multiple.xml"
    xml_path.write_text(
        '<testsuites>'
        '<testsuite name="parameterized" tests="3" failures="0" errors="0" skipped="1"/>'
        '<testsuite name="ordinary" tests="2" failures="1" errors="0" skipped="0"/>'
        '</testsuites>',
        encoding="utf-8",
    )

    assert check_junit.read_junit_counts(xml_path) == {
        "tests": 5,
        "failures": 1,
        "errors": 0,
        "skipped": 1,
        "passes": 3,
    }


@pytest.mark.parametrize(
    "missing_field",
    ["tests", "failures", "errors", "skipped"],
)
def test_read_junit_counts_rejects_missing_required_attributes(
    tmp_path:      Path,
    missing_field: str,
) -> None:
    attributes = {
        "tests": "4",
        "failures": "0",
        "errors": "0",
        "skipped": "0",
    }
    del attributes[missing_field]
    serialized = " ".join(f'{name}="{value}"' for name, value in attributes.items())
    xml_path = tmp_path / f"missing-{missing_field}.xml"
    xml_path.write_text(f"<testsuite {serialized}/>", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"lacks the '{missing_field}' count"):
        check_junit.read_junit_counts(xml_path)


def test_run_pytest_junit_rejects_incomplete_xml_and_cleans_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_path: Path | None = None

    def _fake_pytest_main(args: list[str]) -> pytest.ExitCode:
        nonlocal observed_path
        junit_arg = next(arg for arg in args if str(arg).startswith("--junitxml="))
        observed_path = Path(str(junit_arg).split("=", 1)[1])
        observed_path.write_text('<testsuite tests="22"/>', encoding="utf-8")
        return pytest.ExitCode.OK

    monkeypatch.setattr(pytest, "main", _fake_pytest_main)

    with pytest.raises(ValueError, match="lacks the 'failures' count"):
        check_junit.run_pytest_junit(["-m", "cuda"], prefix="incomplete-")
    assert observed_path is not None
    assert not observed_path.exists()


@pytest.mark.parametrize(
    "counts",
    [
        {"tests": 0, "passes": 0, "failures": 0, "errors": 0, "skipped": 0},
        {"tests": 4, "passes": 3, "failures": 1, "errors": 0, "skipped": 0},
        {"tests": 4, "passes": 3, "failures": 0, "errors": 1, "skipped": 0},
        {"tests": 4, "passes": 3, "failures": 0, "errors": 0, "skipped": 1},
        {"tests": 4, "passes": 3, "failures": 0, "errors": 0, "skipped": 0},
        {"tests": 3, "passes": 3, "failures": 0, "errors": 0, "skipped": 0},
    ],
    ids=["zero", "failure", "error", "skip", "inconsistent", "incomplete"],
)
def test_cuda_junit_predicate_rejects_incomplete_results(
    counts: dict[str, int],
) -> None:
    predicate = getattr(check_junit, "junit_is_exact_all_pass", None)
    assert callable(predicate)
    assert predicate(counts, expected_tests=4) is False


def test_cuda_junit_predicate_accepts_only_exact_all_pass_count() -> None:
    predicate = getattr(check_junit, "junit_is_exact_all_pass", None)
    assert callable(predicate)
    counts = {
        "tests": 4,
        "passes": 4,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }

    assert predicate(counts, expected_tests=4) is True


def test_run_pytest_junit_derives_counts_and_cleans_xml(monkeypatch):
    observed = {}

    def _fake_pytest_main(args):
        observed["args"] = list(args)
        junit_arg = next(arg for arg in args if str(arg).startswith("--junitxml="))
        xml_path = Path(str(junit_arg).split("=", 1)[1])
        observed["xml_path"] = xml_path
        xml_path.write_text(
            '<testsuite tests="4" failures="1" errors="0" skipped="1"/>',
            encoding="utf-8",
        )
        return pytest.ExitCode.TESTS_FAILED

    monkeypatch.setattr(pytest, "main", _fake_pytest_main)

    code, counts = check_junit.run_pytest_junit(["tests/test_example.py"], prefix="audit-")

    assert code == int(pytest.ExitCode.TESTS_FAILED)
    assert counts == {"tests": 4, "failures": 1, "errors": 0, "skipped": 1, "passes": 2}
    assert "-q" not in observed["args"]
    assert not observed["xml_path"].exists()


def test_click_run_verifiers_use_machine_read_junit_counts():
    for script_name in ("check_audit_fixes.py", "check_gpu_tests.py"):
        source = Path(script_name).read_text(encoding="utf-8")
        assert "run_pytest_junit" in source
        assert "pytest.main" not in source


def test_public_e_step_keyword_signature_follows_repository_order():
    names = list(inspect.signature(e_step).parameters)
    expected = [
        "belief", "mu_p", "sigma_p", "group",
        "tau",
        "e_q_mu_lr", "e_q_sigma_lr", "e_phi_lr", "exp_fp64_norm_threshold",
        "n_iter", "e_steps_min", "e_steps_max", "e_steps_backprop_last",
        "e_step_gradient", "exp_fp64_mode",
        "return_trajectory", "oracle_unroll_grad", "randomize_e_steps",
        "transport_mean_per_head", "compact_phi_block_transport", "rope_on_cov", "rope_on_value",
        "training",
        "e_step_halt_tol", "grad_record", "state_record", "rope", "log_prior",
        "transport_chart_max_norm", "transport_status", "prebuilt_transport",
        "kwargs",
    ]
    assert names == expected


def test_report_source_uses_american_english_color_spelling():
    source = Path("vfe3/viz/report.py").read_text(encoding="utf-8").lower()
    for british in ("colour", "grey"):
        assert british not in source
    assert "color" in source
    assert "gray" in source
