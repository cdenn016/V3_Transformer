"""Install the verification skill, neutral agents, policies, and Stop hooks."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from agent_tooling.verification.skill.scripts.verification_gate import render_skill_markdown


AGENT_ROLES = (
    "verifier-orchestrator",
    "verifier-code",
    "verifier-math",
    "verifier-evidence",
    "verifier-skeptic",
    "verifier-adjudicator",
)
BLOCK_MARKERS = {
    "global-policy.md": "VERIFICATION GLOBAL POLICY",
    "project-policy.md": "VERIFICATION PROJECT POLICY",
    "deep-audit-integration.md": "VERIFICATION DEEP AUDIT INTEGRATION",
}


def _required_string(spec: dict[str, object], name: str) -> str:
    value = spec.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"agent spec requires nonempty {name}")
    return value


def render_claude_agent(spec: dict[str, object]) -> str:
    """Render one neutral verifier spec as Claude agent Markdown."""

    name = _required_string(spec, "name")
    description = _required_string(spec, "description")
    instructions = _required_string(spec, "instructions")
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{instructions}\n"


def render_codex_agent(spec: dict[str, object]) -> str:
    """Render one neutral verifier spec as literal-safe Codex TOML."""

    name = _required_string(spec, "name")
    description = _required_string(spec, "description")
    instructions = _required_string(spec, "instructions")
    return "".join(
        (
            f"name = {json.dumps(name)}\n",
            f"description = {json.dumps(description)}\n",
            f"instructions = {json.dumps(instructions)}\n",
        )
    )


def upsert_marked_block(text: str, marker: str, block: str) -> str:
    """Replace or append exactly one HTML-comment-delimited policy block."""

    begin = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    if begin in text or end in text:
        if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
            raise ValueError(f"invalid marked block for {marker}")
        before, remainder = text.split(begin, 1)
        _, after = remainder.split(end, 1)
        return f"{before}{begin}\n{block.rstrip()}\n{end}{after}"
    separator = "" if not text or text.endswith("\n") else "\n"
    return f"{text}{separator}{begin}\n{block.rstrip()}\n{end}\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required in {path}")
    return value


def _read_block(path: Path, marker: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    begin = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
        raise ValueError(f"source block {path} must contain one marked {marker} block")
    return text.split(begin, 1)[1].split(end, 1)[0].strip()


def _read_specs(source_root: Path) -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    for role in AGENT_ROLES:
        path = source_root / "agents" / f"{role}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"agent spec must be an object: {path}")
        if _required_string(value, "name") != role:
            raise ValueError(f"agent spec name must match filename: {path}")
        _required_string(value, "description")
        _required_string(value, "instructions")
        specs[role] = value
    return specs


def _copy_skill(source: Path, destination: Path, shell: str) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    skill_markdown = render_skill_markdown(destination, shell=shell)
    (destination / "SKILL.md").write_text(skill_markdown, encoding="utf-8")


def _gate_stop_handler(gate_path: Path) -> dict[str, object]:
    command = f'"{Path(sys.executable).resolve()}" "{gate_path.resolve()}" hook'
    return {"hooks": [{"type": "command", "command": command}]}


def _install_stop_handler(settings: dict[str, Any], gate_path: Path) -> dict[str, Any]:
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("settings hooks must be an object")
    stop = hooks.setdefault("Stop", [])
    if not isinstance(stop, list):
        raise ValueError("settings hooks.Stop must be a list")
    existing = [entry for entry in stop if "verification_gate.py" in json.dumps(entry, sort_keys=True)]
    if len(existing) > 1:
        raise ValueError("settings contains more than one verification Stop handler")
    if not existing:
        stop.append(_gate_stop_handler(gate_path))
    return settings


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def install(args: argparse.Namespace) -> None:
    """Install from explicit paths after validating every source and JSON destination."""

    source_root = Path(args.source_root)
    claude_home = Path(args.claude_home)
    codex_home = Path(args.codex_home)
    project_root = Path(args.project_root) if args.project_root is not None else None
    shell = str(args.shell)
    if shell not in {"powershell", "posix"}:
        raise ValueError("shell must be powershell or posix")
    skill_source = source_root / "skill"
    if not (skill_source / "SKILL.md").is_file() or not (skill_source / "scripts" / "verification_gate.py").is_file():
        raise FileNotFoundError(skill_source)
    specs = _read_specs(source_root)
    blocks = {filename: _read_block(source_root / "blocks" / filename, marker) for filename, marker in BLOCK_MARKERS.items()}

    claude_settings_path = claude_home / "settings.json"
    codex_hooks_path = codex_home / "hooks.json"
    claude_settings = _read_json(claude_settings_path)
    codex_hooks = _read_json(codex_hooks_path)
    claude_global = _read_text(claude_home / "CLAUDE.md")
    codex_global = _read_text(codex_home / "AGENTS.md")
    project_claude = _read_text(project_root / "CLAUDE.md") if project_root is not None else ""
    project_codex = _read_text(project_root / "AGENTS.md") if project_root is not None else ""

    claude_skill = claude_home / "skills" / "verification"
    codex_skill = codex_home / "skills" / "verification"
    claude_settings = _install_stop_handler(claude_settings, claude_skill / "scripts" / "verification_gate.py")
    codex_hooks = _install_stop_handler(codex_hooks, codex_skill / "scripts" / "verification_gate.py")
    for filename in ("global-policy.md", "deep-audit-integration.md"):
        marker = BLOCK_MARKERS[filename]
        claude_global = upsert_marked_block(claude_global, marker, blocks[filename])
        codex_global = upsert_marked_block(codex_global, marker, blocks[filename])
    if project_root is not None:
        marker = BLOCK_MARKERS["project-policy.md"]
        project_claude = upsert_marked_block(project_claude, marker, blocks["project-policy.md"])
        project_codex = upsert_marked_block(project_codex, marker, blocks["project-policy.md"])

    _copy_skill(skill_source, claude_skill, shell)
    _copy_skill(skill_source, codex_skill, shell)
    for role, spec in specs.items():
        _write_text(claude_home / "agents" / f"{role}.md", render_claude_agent(spec))
        _write_text(codex_home / "agents" / f"{role}.toml", render_codex_agent(spec))
    _write_text(claude_home / "CLAUDE.md", claude_global)
    _write_text(codex_home / "AGENTS.md", codex_global)
    if project_root is not None:
        _write_text(project_root / "CLAUDE.md", project_claude)
        _write_text(project_root / "AGENTS.md", project_codex)
    _write_text(claude_settings_path, json.dumps(claude_settings, indent=2) + "\n")
    _write_text(codex_hooks_path, json.dumps(codex_hooks, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--claude-home", required=True, type=Path)
    parser.add_argument("--codex-home", required=True, type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--shell", choices=("powershell", "posix"), required=True)
    install(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
