r"""AST-based wiring tests for cfg.grad_clip propagation through every click-to-run path (PB-15).

Plan 2 Task 1 added ``VFE3Config.grad_clip`` (validated, default 1.0) and Task 2 pinned the
``train``/``train_step`` runtime behavior under every grad_clip mode (global, per-role, off).
This task closes the last gap: every click-to-run driver must actually pass
``grad_clip=cfg.grad_clip`` to ``train(...)``, and every self-contained config dict must expose
the ``grad_clip`` key rather than silently relying on the ``VFE3Config`` dataclass default.

These tests parse the driver source with ``ast`` and do NOT import the driver modules --
train_vfe3.py/ablation.py/scaling.py have heavy import-time and module-level side effects
(dataset caching, CLI-less top-level config construction) unsuited to a fast unit test -- so a
call-shape regression is caught statically, without running any training.
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_DRIVER_FILES = {
    "train_vfe3.py":      REPO_ROOT / "train_vfe3.py",
    "ablation.py":        REPO_ROOT / "ablation.py",
    "scaling.py":         REPO_ROOT / "scaling.py",
    "check_gpu_tests.py": REPO_ROOT / "check_gpu_tests.py",
}


def _train_calls(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "train"
    ]


def _grad_clip_keyword(call: ast.Call):
    r"""The ``grad_clip=...`` keyword's value expression on a ``train(...)`` call, or None."""
    for kw in call.keywords:
        if kw.arg == "grad_clip":
            return kw.value
    return None


def _is_cfg_grad_clip(value: ast.expr) -> bool:
    r"""True iff ``value`` is exactly the attribute expression ``cfg.grad_clip``."""
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "grad_clip"
        and isinstance(value.value, ast.Name)
        and value.value.id == "cfg"
    )


def _config_dict_keys(path: Path, var_name: str) -> set:
    r"""Keyword-argument names of the top-level ``var_name = dict(...)`` (or annotated) self-
    contained config-dict literal in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(t, ast.Name) and t.id == var_name for t in targets):
                continue
            value = node.value
            if (isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name) and value.func.id == "dict"):
                return {kw.arg for kw in value.keywords if kw.arg is not None}
    raise AssertionError(f"no top-level '{var_name} = dict(...)' assignment found in {path}")


def test_driver_train_calls_and_config_dicts_wire_grad_clip():
    r"""Every click-to-run driver's train(...) call passes grad_clip=cfg.grad_clip, and every
    self-contained config dict (train_vfe3.py/ablation.py/scaling.py) exposes the grad_clip key
    the caller reads. scaling.py's baseline also exposes grad_clip_per_role explicitly, since it
    otherwise relies silently on the VFE3Config dataclass default. check_gpu_tests.py builds cfg
    via _structured_cfg() (no duplicate config dict), so only its call shape is checked here."""
    problems = []

    for name, path in _DRIVER_FILES.items():
        calls = _train_calls(path)
        if not calls:
            problems.append(f"{name}: no train(...) call found")
            continue
        for call in calls:
            value = _grad_clip_keyword(call)
            if value is None:
                problems.append(f"{name}:{call.lineno}: train() call is missing the grad_clip keyword")
            elif not _is_cfg_grad_clip(value):
                problems.append(
                    f"{name}:{call.lineno}: grad_clip keyword must be the attribute expression "
                    f"cfg.grad_clip, got {ast.dump(value)}"
                )

    config_dict_specs = [
        ("train_vfe3.py", REPO_ROOT / "train_vfe3.py", "config",          {"grad_clip"}),
        ("ablation.py",   REPO_ROOT / "ablation.py",    "BASELINE_CONFIG", {"grad_clip"}),
        ("scaling.py",    REPO_ROOT / "scaling.py",     "config",          {"grad_clip", "grad_clip_per_role"}),
    ]
    for name, path, var_name, required_keys in config_dict_specs:
        keys = _config_dict_keys(path, var_name)
        missing = required_keys - keys
        if missing:
            problems.append(f"{name}: {var_name} dict is missing key(s) {sorted(missing)}")

    assert not problems, "\n".join(problems)


def test_run_training_passes_cfg_grad_clip():
    r"""vfe3.train.run_training's own train(...) call passes grad_clip=cfg.grad_clip."""
    tree = ast.parse((REPO_ROOT / "vfe3" / "train.py").read_text(encoding="utf-8"))
    run_training_def = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_training"
    )
    calls = [
        node for node in ast.walk(run_training_def)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "train"
    ]
    assert calls, "run_training: no train(...) call found"
    for call in calls:
        value = _grad_clip_keyword(call)
        assert value is not None, f"run_training:{call.lineno}: train() call is missing the grad_clip keyword"
        assert _is_cfg_grad_clip(value), (
            f"run_training:{call.lineno}: grad_clip keyword must be the attribute expression "
            f"cfg.grad_clip, got {ast.dump(value)}"
        )
