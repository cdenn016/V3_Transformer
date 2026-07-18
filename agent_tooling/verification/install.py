"""Install the verification skill, neutral agents, policies, and Stop hooks."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from agent_tooling.verification.skill.scripts.verification_gate import (
    GATE_COMMAND_TOKEN,
    GATE_SHELL_TOKEN,
    render_skill_markdown,
)


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
SKILL_MANIFEST = (
    "SKILL.md",
    "scripts/verification_gate.py",
    "schemas/claim-ledger.schema.json",
    "evals/evals.json",
    "references/contract.md",
    "references/criteria-code.md",
    "references/criteria-math.md",
    "references/criteria-evidence.md",
    "references/criteria-experiment.md",
    "references/criteria-general.md",
)
ADJUDICATOR_OUTPUT = {
    "required": ["claim_id", "ledger_state", "rationale", "evidence_ids", "open_obligations", "validator_errors"],
    "ledger_state": ["EVIDENCE_VERIFIED", "REFUTED", "INCONCLUSIVE"],
}
GATE_STOP_STATUS_MESSAGE = "Verification ledger gate"
_SKILL_TOKEN_COUNTS = {GATE_COMMAND_TOKEN: 2, GATE_SHELL_TOKEN: 2}


def _required_string(spec: dict[str, object], name: str) -> str:
    value = spec.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"agent spec requires nonempty {name}")
    return value


def _optional_string_list(spec: dict[str, object], name: str) -> list[str] | None:
    value = spec.get(name)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"agent spec {name} must be a list of strings")
    return value


def _literal(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def effective_instructions(spec: dict[str, object]) -> str:
    """Return the single neutral instruction body rendered for both clients."""

    instructions = _required_string(spec, "instructions")
    output = spec.get("output")
    if output is None:
        return instructions
    if not isinstance(output, dict):
        raise ValueError("agent spec output must be an object")
    return f"{instructions}\n\nRequired output schema:\n{json.dumps(output, indent=2, ensure_ascii=False)}"


def render_claude_agent(spec: dict[str, object]) -> str:
    """Render one neutral verifier spec as Claude agent Markdown."""

    name = _required_string(spec, "name")
    description = _required_string(spec, "description")
    instructions = effective_instructions(spec)
    tools = _optional_string_list(spec, "tools")
    frontmatter = f"name: {_literal(name)}\ndescription: {_literal(description)}\n"
    if tools is not None:
        frontmatter += f"tools: {_literal(tools)}\n"
    return f"---\n{frontmatter}---\n\n{instructions}\n"


def render_codex_agent(spec: dict[str, object]) -> str:
    """Render one neutral verifier spec as literal-safe Codex TOML."""

    name = _required_string(spec, "name")
    description = _required_string(spec, "description")
    instructions = effective_instructions(spec)
    tools = _optional_string_list(spec, "tools")
    rendered = f"name = {_literal(name)}\ndescription = {_literal(description)}\ninstructions = {_literal(instructions)}\n"
    if tools is not None:
        rendered += f"tools = {_literal(tools)}\n"
    return rendered


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
        _optional_string_list(value, "tools")
        if role == "verifier-adjudicator" and value.get("output") != ADJUDICATOR_OUTPUT:
            raise ValueError("verifier-adjudicator requires the exact structured output contract")
        specs[role] = value
    return specs


def _preflight_skill(source: Path) -> None:
    for relative_path in SKILL_MANIFEST:
        required = source / relative_path
        if not required.is_file():
            raise FileNotFoundError(required)


def _read_preflight_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required in {path}")
    return value


def _preflight_skill_content(source: Path, shell: str) -> None:
    schema = _read_preflight_json(source / "schemas" / "claim-ledger.schema.json")
    if schema.get("type") != "object" or not isinstance(schema.get("required"), list) or not isinstance(schema.get("properties"), dict):
        raise ValueError("claim-ledger schema must define an object root, required fields, and properties")
    evals = _read_preflight_json(source / "evals" / "evals.json")
    if evals.get("skill_name") != "verification" or not isinstance(evals.get("evals"), list):
        raise ValueError("skill evals must define verification skill_name and an evals list")
    template = (source / "SKILL.md").read_text(encoding="utf-8")
    for token, expected_count in _SKILL_TOKEN_COUNTS.items():
        if template.count(token) != expected_count:
            raise ValueError(f"skill template must contain {token} exactly {expected_count} times")
    rendered = render_skill_markdown(source, shell=shell)
    if GATE_COMMAND_TOKEN in rendered or GATE_SHELL_TOKEN in rendered:
        raise ValueError("rendered skill retains a verification render token")
    shell_label = "powershell" if shell == "powershell" else "bash"
    if rendered.count(f"```{shell_label}") != _SKILL_TOKEN_COUNTS[GATE_SHELL_TOKEN]:
        raise ValueError("rendered skill does not contain the expected shell fences")


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


def _gate_command(gate_path: Path) -> str:
    return f'"{Path(sys.executable).resolve()}" "{gate_path.resolve()}" hook'


def _gate_stop_handler(command: str, shell: str) -> dict[str, object]:
    return {
        "hooks": [
            {
                "type": "command",
                "statusMessage": GATE_STOP_STATUS_MESSAGE,
                "shell": shell,
                "command": command,
            }
        ]
    }


def _is_python_executable(value: str) -> bool:
    executable = value.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return executable == "python" or executable.startswith("python") and executable.endswith(".exe")


def _split_hook_command(command: str, shell: str) -> list[str] | None:
    try:
        if shell == "posix":
            return shlex.split(command, posix=True)
        match = re.fullmatch(r'\s*"([^"]+)"\s+"([^"]+)"\s+(\S+)\s*', command)
        return list(match.groups()) if match is not None else None
    except ValueError:
        return None


def _is_verification_handler(value: object, shell: str) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("type") != "command" or value.get("statusMessage") != GATE_STOP_STATUS_MESSAGE or value.get("shell") != shell:
        return False
    command = value.get("command")
    if not isinstance(command, str):
        return False
    parts = _split_hook_command(command, shell)
    if parts is None or len(parts) != 3 or not _is_python_executable(parts[0]) or parts[2] != "hook":
        return False
    gate_path = parts[1].replace("\\", "/")
    return gate_path.endswith("skills/verification/scripts/verification_gate.py")


def _remove_verification_handlers(stop: list[Any], shell: str) -> list[Any]:
    retained: list[Any] = []
    for entry in stop:
        if not isinstance(entry, dict):
            retained.append(entry)
            continue
        handlers = entry.get("hooks")
        if isinstance(handlers, list):
            remaining = [handler for handler in handlers if not _is_verification_handler(handler, shell)]
            if remaining:
                updated = dict(entry)
                updated["hooks"] = remaining
                retained.append(updated)
        elif not _is_verification_handler(entry, shell):
            retained.append(entry)
    return retained


def _install_stop_handler(settings: dict[str, Any], gate_path: Path, shell: str) -> dict[str, Any]:
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("settings hooks must be an object")
    stop = hooks.setdefault("Stop", [])
    if not isinstance(stop, list):
        raise ValueError("settings hooks.Stop must be a list")
    command = _gate_command(gate_path)
    stop[:] = _remove_verification_handlers(stop, shell)
    stop.append(_gate_stop_handler(command, shell))
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
    _preflight_skill(skill_source)
    _preflight_skill_content(skill_source, shell)
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
    claude_settings = _install_stop_handler(claude_settings, claude_skill / "scripts" / "verification_gate.py", shell)
    codex_hooks = _install_stop_handler(codex_hooks, codex_skill / "scripts" / "verification_gate.py", shell)
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
