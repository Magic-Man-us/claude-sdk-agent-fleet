# The two-surface rule

There are two distinct capability surfaces in play. They are never crossed.

| Surface | Whose catalog | What it holds | Searched by |
|---------|---------------|---------------|-------------|
| **Build-time** | agent-fleet's own | skills, tools, MCP servers, plugins | the generator, to *assemble* an agent |
| **Run-time** | the emitted orchestrator's | its specialist pool, dispatch targets | the generated agent, at *its* runtime |

The build-time surface is the `CatalogEntry` union, which lives in the external
[capabilities-discovery](https://github.com/Magic-Man-us/capabilities-discovery) package
(`capabilities_discovery.catalog`) and is re-exported from `agent_fleet`. The run-time surface is
whatever capability set the generator *emits into* an orchestrator it builds — it is not part of
agent-fleet's catalog.

## The invariant

```
Every Ref type ⇔ exactly one find_/load_ entry point on the build-time surface.
No entry point → no Ref → not in the catalog.
```

Current build-time surface:

| Ref | Entry point |
|-----|-------------|
| `SkillRef` | `find_skills` / `load_skill` |
| `ToolRef` | `find_tools` |
| `McpServerRef` | `find_mcp` |
| `PluginRef` | `find_plugins` |

A plugin's bundle lists (`CatalogPlugin.skills`, `CatalogPlugin.mcp_servers`) hold only refs whose type already has a build-time entry point, so they group existing surface rather than introduce new kinds.

## Rejected for the build-time catalog

| Concept | Reason |
|---------|--------|
| Agents (subagents) | Nothing on the build-time surface equips one; a skill or orchestrator spawns it at runtime via the Agent SDK. Belongs to the run-time surface. |
| Commands | Human-invoked (`/command`); no build-time consumer searches for them. |
| Hooks | Event-triggered, never selected; no entry point. |

What an artifact does behind its ref — spawn agents, call MCP tools, fire hooks — is encapsulated runtime behavior. The catalog stops at the first hop; it never enumerates the transitive call graph.

## The tell

Reaching to add `find_agents` (or `CatalogAgent`) to the build-time `CatalogEntry` means a run-time concern has crossed into the factory. The fix is to emit that capability into the generated orchestrator instead — same invariant, applied on the run-time surface.
