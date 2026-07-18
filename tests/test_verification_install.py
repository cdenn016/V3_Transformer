"""Behavioral tests for cross-platform verification-control installation."""

from __future__ import annotations

import json
import shutil
import tomllib
from argparse import Namespace
from pathlib import Path

import pytest

from agent_tooling.verification.install import (
    AGENT_ROLES,
    BLOCK_MARKERS,
    render_claude_agent,
    render_codex_agent,
    install,
    upsert_marked_block,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
SKILL_SOURCE = REPOSITORY_ROOT / "agent_tooling" / "verification" / "skill"


def _source_root(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    agents = source / "agents"
    blocks = source / "blocks"
    agents.mkdir(parents=True)
    blocks.mkdir()
    shutil.copytree(SKILL_SOURCE, source / "skill")
    for role in AGENT_ROLES:
        (agents / f"{role}.json").write_text(
            json.dumps(
                {
                    "name": role,
                    "description": f"Neutral {role} verification role.",
                    "instructions": f"Neutral body for {role}.\nReturn structured support, refute, or abstain.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    for filename, marker in BLOCK_MARKERS.items():
        (blocks / filename).write_text(
            f"<!-- BEGIN {marker} -->\n{filename} policy body\n<!-- END {marker} -->\n",
            encoding="utf-8",
        )
    return source


def _args(tmp_path: Path, *, project: bool = True) -> Namespace:
    claude_home = tmp_path / "claude"
    codex_home = tmp_path / "codex"
    claude_home.mkdir()
    codex_home.mkdir()
    (claude_home / "CLAUDE.md").write_text("Claude preface\n", encoding="utf-8")
    (codex_home / "AGENTS.md").write_text("Codex preface\n", encoding="utf-8")
    (claude_home / "settings.json").write_text(
        json.dumps({"theme": "dark", "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo keep"}]}]}}),
        encoding="utf-8",
    )
    (codex_home / "hooks.json").write_text(
        json.dumps({"other": {"enabled": True}, "hooks": {"Stop": [{"command": "echo keep"}]}}),
        encoding="utf-8",
    )
    (claude_home / "agents").mkdir()
    (claude_home / "agents" / "unrelated.md").write_text("keep\n", encoding="utf-8")
    (codex_home / "agents").mkdir()
    (codex_home / "agents" / "unrelated.toml").write_text('name = "keep"\n', encoding="utf-8")
    (claude_home / "skills").mkdir()
    (claude_home / "skills" / "unrelated.txt").write_text("keep\n", encoding="utf-8")
    (codex_home / "skills").mkdir()
    (codex_home / "skills" / "unrelated.txt").write_text("keep\n", encoding="utf-8")
    project_root: Path | None = None
    if project:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "CLAUDE.md").write_text("Project Claude preface\n", encoding="utf-8")
        (project_root / "AGENTS.md").write_text("Project Codex preface\n", encoding="utf-8")
    return Namespace(
        source_root=_source_root(tmp_path),
        claude_home=claude_home,
        codex_home=codex_home,
        project_root=project_root,
        shell="powershell",
    )


def _claude_instruction(text: str) -> str:
    _, _, body = text.split("---", 2)
    return body.lstrip("\n")


def test_renderers_share_identical_neutral_instruction_bodies() -> None:
    spec = {"name": "verifier-code", "description": "Code verifier.", "instructions": "Line one.\nLine two."}

    claude = render_claude_agent(spec)
    codex = render_codex_agent(spec)

    assert _claude_instruction(claude) == spec["instructions"] + "\n"
    assert tomllib.loads(codex)["instructions"] == spec["instructions"]
    assert tomllib.loads(codex)["name"] == "verifier-code"


def test_upsert_marked_block_replaces_only_the_named_region() -> None:
    text = "before\n<!-- BEGIN selected -->\nold\n<!-- END selected -->\nafter\n"

    result = upsert_marked_block(text, "selected", "new\n")

    assert result == "before\n<!-- BEGIN selected -->\nnew\n<!-- END selected -->\nafter\n"
    assert upsert_marked_block("before\n", "selected", "new\n").endswith("<!-- END selected -->\n")


def test_install_renders_six_agents_preserves_existing_surfaces_and_is_idempotent(tmp_path: Path) -> None:
    args = _args(tmp_path)

    install(args)

    for role in AGENT_ROLES:
        spec = json.loads((args.source_root / "agents" / f"{role}.json").read_text(encoding="utf-8"))
        claude_text = (args.claude_home / "agents" / f"{role}.md").read_text(encoding="utf-8")
        codex_text = (args.codex_home / "agents" / f"{role}.toml").read_text(encoding="utf-8")
        assert _claude_instruction(claude_text) == spec["instructions"] + "\n"
        assert tomllib.loads(codex_text)["instructions"] == spec["instructions"]
    assert (args.claude_home / "agents" / "unrelated.md").read_text(encoding="utf-8") == "keep\n"
    assert (args.codex_home / "agents" / "unrelated.toml").read_text(encoding="utf-8") == 'name = "keep"\n'
    assert (args.claude_home / "skills" / "unrelated.txt").read_text(encoding="utf-8") == "keep\n"
    assert (args.codex_home / "skills" / "unrelated.txt").read_text(encoding="utf-8") == "keep\n"
    assert "Claude preface" in (args.claude_home / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Codex preface" in (args.codex_home / "AGENTS.md").read_text(encoding="utf-8")
    assert "project-policy.md policy body" in (args.project_root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "project-policy.md policy body" in (args.project_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "deep-audit-integration.md policy body" in (args.claude_home / "CLAUDE.md").read_text(encoding="utf-8")
    assert "deep-audit-integration.md policy body" in (args.codex_home / "AGENTS.md").read_text(encoding="utf-8")
    assert "global-policy.md policy body" in (args.claude_home / "CLAUDE.md").read_text(encoding="utf-8")
    assert "global-policy.md policy body" in (args.codex_home / "AGENTS.md").read_text(encoding="utf-8")
    assert "{{VERIFICATION_GATE_COMMAND}}" not in (args.claude_home / "skills" / "verification" / "SKILL.md").read_text(encoding="utf-8")
    assert "{{VERIFICATION_GATE_COMMAND}}" not in (args.codex_home / "skills" / "verification" / "SKILL.md").read_text(encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file() and ".pytest_cache" not in path.parts}

    install(args)

    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file() and ".pytest_cache" not in path.parts}
    assert after == before


def test_install_merges_exactly_one_gate_stop_handler_and_preserves_json_keys(tmp_path: Path) -> None:
    args = _args(tmp_path, project=False)

    install(args)

    for path, preserved_key in ((args.claude_home / "settings.json", "theme"), (args.codex_home / "hooks.json", "other")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert preserved_key in data
        stop_entries = data["hooks"]["Stop"]
        gate_entries = [entry for entry in stop_entries if "verification_gate.py" in json.dumps(entry)]
        assert len(gate_entries) == 1
        assert "hook" in json.dumps(gate_entries[0])


@pytest.mark.parametrize("bad_source", ("missing", "invalid-json"))
def test_install_aborts_before_destination_writes_for_invalid_source(tmp_path: Path, bad_source: str) -> None:
    args = _args(tmp_path, project=False)
    target = args.claude_home / "CLAUDE.md"
    before = target.read_bytes()
    if bad_source == "missing":
        (args.source_root / "agents" / "verifier-code.json").unlink()
    else:
        (args.source_root / "agents" / "verifier-code.json").write_text("{ invalid", encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError)):
        install(args)

    assert target.read_bytes() == before


def test_install_aborts_before_destination_writes_for_invalid_destination_json(tmp_path: Path) -> None:
    args = _args(tmp_path, project=False)
    target = args.codex_home / "AGENTS.md"
    before = target.read_bytes()
    (args.codex_home / "hooks.json").write_text("{ invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        install(args)

    assert target.read_bytes() == before


def test_neutral_specs_enforce_structured_results_and_adjudicator_closure_abstention() -> None:
    for role in AGENT_ROLES:
        spec_path = REPOSITORY_ROOT / "agent_tooling" / "verification" / "agents" / f"{role}.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        body = str(spec["instructions"])
        assert "support" in body and "refute" in body and "abstain" in body
        if role == "verifier-adjudicator":
            assert "EVIDENCE_VERIFIED" in body
            assert "deterministic validator" in body
            assert "INCONCLUSIVE" in body
        else:
            assert "must not assign EVIDENCE_VERIFIED" in body
