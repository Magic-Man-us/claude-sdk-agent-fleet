"""Stand up (and tear down) a complex multi-scope sandbox for the scope inventory.

Every scope is a subdirectory of one root, so the whole path surface is exercised without
touching real system or home paths. Usage:

    python tests/helpers/scope_sandbox.py build ./sbx     # create the tree
    python tests/helpers/scope_sandbox.py show  ./sbx     # scan it and print the display
    python tests/helpers/scope_sandbox.py clean ./sbx     # rm -rf the tree

`build` then `show` should display artifacts in every scope, with the documented collisions
resolving correctly (nearest project agent wins; user skill beats project skill).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from capdisc.scope import ScopeInventory, ScopeRoots, render_inventory

# Run as a script; put tests/ on sys.path so the shared `helpers` package resolves under any
# launcher (uv sets PYTHONSAFEPATH, which drops the automatic script-dir entry).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import touch

# A markdown file with a YAML frontmatter hook — exercises component-scope hooks.
_GUARDED_AGENT = (
    "---\n"
    "name: guarded\n"
    "hooks:\n"
    "  PreToolUse:\n"
    "    - matcher: Bash\n"
    "      hooks:\n"
    "        - type: command\n"
    "          command: ./check.sh\n"
    "---\n"
    "A subagent that carries its own hook.\n"
)


def _settings_hook(event: str) -> str:
    """A settings.json `hooks` block carrying one command hook on `event`."""
    handler = '[{"hooks": [{"type": "command", "command": "x"}]}]'
    return f'{{"hooks": {{"{event}": {handler}}}}}'


# A plugin's bare hooks.json — events at the root, with no enclosing "hooks" key.
_PLUGIN_HOOK = '{"PostToolUse": [{"hooks": [{"type": "command", "command": "p"}]}]}'

# (relative path under the sandbox root, contents). Directories are created as needed. We launch
# the scan from project/services/api, so cases are labelled by where they sit relative to it.
_FILES: dict[str, str] = {
    # --- project repo (walk UP from the start dir to the repo root) ---
    "project/.git/HEAD": "ref: refs/heads/main\n",
    "project/.claude/agents/reviewer.md": "ROOT reviewer (loses to nearer copy)",
    "project/.claude/agents/nested/deep.md": "agent nested under the agents dir (still project)",
    "project/.claude/agents/guarded.md": _GUARDED_AGENT,
    "project/.claude/skills/linter/SKILL.md": "project linter skill (loses to user copy)",
    "project/.claude/commands/deploy.md": "project deploy command",
    "project/.claude/settings.json": _settings_hook("PreToolUse"),
    "project/.claude/settings.local.json": _settings_hook("Stop"),
    # at the start dir itself:
    "project/services/api/.claude/agents/reviewer.md": "NEAR reviewer (nearest wins)",
    # genuinely BELOW the start dir: the skill is found, the agent is not
    "project/services/api/web/.claude/skills/web-skill/SKILL.md": "skill below start (FOUND)",
    "project/services/api/web/.claude/agents/web-agent.md": "agent below start (NOT found)",
    # noise that must be pruned:
    "project/.venv/.claude/skills/noise/SKILL.md": "noise that must be pruned",
    # --- user scope (~/.claude stand-in) ---
    "home/.claude/agents/helper.md": "user helper agent",
    "home/.claude/skills/linter/SKILL.md": "user linter skill (beats project for skills)",
    "home/.claude/settings.json": _settings_hook("UserPromptSubmit"),
    # --- managed scope (settings file at root, standalone under .claude/) ---
    "managed/managed-settings.json": _settings_hook("SubagentStop"),
    "managed/.claude/agents/org-agent.md": "organization-wide managed agent",
    # --- plugin scope ---
    "plugins/acme/agents/plug-agent.md": "plugin agent",
    "plugins/acme/skills/plug-skill/SKILL.md": "plugin skill",
    "plugins/acme/hooks/hooks.json": _PLUGIN_HOOK,
    # --- an --add-dir directory ---
    "extra/.claude/commands/added.md": "command from an added directory",
}


def _sandbox_roots(root: Path) -> ScopeRoots:
    return ScopeRoots.discover(
        start=root / "project" / "services" / "api",
        home_dir=root / "home",
        managed_dir=root / "managed",
        plugin_dirs=[root / "plugins" / "acme"],
        add_dirs=[root / "extra"],
    )


def build(root: Path) -> None:
    for relative, contents in _FILES.items():
        touch(root / relative, contents)
    # A symlinked agent escaping the tree — the scan must skip it (no secret capture).
    secret = root / "outside-secret.txt"
    touch(secret, "PRETEND-SECRET")
    link = root / "project" / ".claude" / "agents" / "exfil.md"
    link.symlink_to(secret)
    print(f"built sandbox at {root}  ({len(_FILES)} files + 1 escaping symlink)")
    print(f"teardown: python {Path(__file__).name} clean {root}")


def show(root: Path) -> None:
    inventory = ScopeInventory.scan(_sandbox_roots(root))
    print(render_inventory(inventory))


def clean(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)
    print(f"removed {root}")


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in {"build", "show", "clean"}:
        print(__doc__)
        return 2
    command, root = sys.argv[1], Path(sys.argv[2]).resolve()
    {"build": build, "show": show, "clean": clean}[command](root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
