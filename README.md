# Code Agent

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

A Python REPL-native agent. The model's response *is* Python.

Not a chatbot with a toolbox bolted on. Every assistant turn is source code, executed in a persistent subprocess REPL, with the filesystem, the shell, and a set of built-in functions for reading files, editing code, searching, spawning processes, and managing context. The model decides when it's done the same way a programmer does: it stops. Specifically, it calls `emit(..., release=True)`.

This is my daily driver, built and refined through real use. I can't imagine working without it.

Here's what a session looks like:

```
$ coda

──────────────────────────────────
Code Agent
Python REPL-based coding assistant
gpt-5.6-sol-medium
──────────────────────────────────
Enter = submit | Alt+Enter = newline | Ctrl+O = transcript | Esc Esc = rewind | Ctrl+C = interrupt | Ctrl+D = quit
Commands: /repl, /rewind, /exec [instructions], /resume [session_id], /fork [session_id], /skills [name], /subagents [model], /attach <file>, /detach <file>, /attachments, /model [name], /tokens
Loading AGENTS.md

> What's the current status of the database migration?

─ Python ──────────────────────────
>>> status = bash("alembic current")
>>> pending = bash("alembic history -r current:head")
>>> print(status)
>>> print(pending)
  INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
  INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
  Rev: 3a4b5c6d7e8f (head)
  (empty)

═ Output ══════════════════════════
You're already at the latest revision. Nothing pending.

═ User ════════════════════════════
> Great, then let's look at the users table schema
...

Session ended. Goodbye!
Resume session: coda --resume f5867c0c-6963-405e-9c6a-c775cea6cb6a
gpt-5.6-sol-medium: In=12847, Cached=41203, Rsn=1024, Out=843, Cost=$0.018
```

---

## Why this approach

**Model performance degrades with context length.** Most agent architectures pour fuel on that fire: native tool-call schemas, JSON marshalling, conversation-turn structure, and compaction all chew through the window and make the model worse. Code Agent is built around the opposite principle: keep the context window as small as possible for as long as possible. The REPL execution model, lossless coalescing, and attachment system all exist for this reason. Sessions that would require multiple compaction cycles in other agents run cleanly in Code Agent, typically remaining well below 200k tokens.

**Token efficiency.** Native function calling means JSON marshalling, schema validation, and protocol overhead. Plain Python has none of that. Multi-step operations happen in a single assistant turn — the model just keeps writing.

**Natural for models.** Writing Python is a core, well-trained capability, so this approach holds up across a wide range of models, not just the largest frontier ones.

**Agent-defined tools.** Tools in Code Agent are ordinary Python functions in the REPL namespace, not predefined JSON schemas stuffed into the system prompt. That is dramatically more token-efficient, and it means the agent can define its own tools at runtime — a function it writes once can be used for the rest of the session, or it can load an entire framework from a skill.

**AST-based preprocessing.** Model responses are parsed and transformed before execution, normalizing output patterns and handling common edge cases. Unrecoverable errors are retried silently, so the conversation stays uninterrupted.

**Persistent REPL state.** Variables, database connections, subprocesses, and imports persist across turns. Connect to a database once and query it for the rest of the session. No reconnecting from scratch every turn.

---

## Agents manage their own context

**Lossless coalescing.** Old turns are automatically coalesced past a three-interaction horizon (each interaction may span many turns), and the system actively coalesces as needed to stay under the context window — including mid-session on long-running interactions. Coalesced exchanges become expandable preview blobs rather than discarded text, so nothing is permanently lost. The agent can also `pin()` important turns so they survive coalescing and auto-expand when referenced.

This is fundamentally different from compaction: compaction summarizes and discards, whereas coalescing archives. And because Code Agent executes in a persistent subprocess, the full REPL state remains completely intact as context is archived.

**Agent-managed rollups.** Beyond automatic coalescing, the agent curates its own context. `observe()` records durable lessons that survive coalescing; `rollup(start, end, summary)` replaces a completed span of turns with a structured summary — recursively, so rolled-up spans can themselves be rolled up — while the original content remains archived behind expandable preview references. `pin()` protects exact turns from summarization.

**Truly lossless, effectively infinite context.** Nothing the agent has seen is ever discarded — only demoted behind references the agent can re-expand on demand. Long-running development sessions with turn counts past 50,000 continue to function, because the active context window stays lean regardless of session length.

Because the session stays lean, you can configure a *smaller* context window than you would otherwise need — where compaction would be required with a large window, Code Agent runs cleanly with a smaller one, and a smaller context window means better model performance.

**Full-file attachments, not grep fragments.** Most agents search for relevant snippets and then operate on those fragments, which is a common source of "that function doesn't even look like that." Code Agent uses an attachment system: `view(file)` attaches the full file to context with line numbers, `unview(file)` removes it. Only the most recent version of each file is ever in context, and it is updated automatically after edits. Ephemeral context, injected into the latest user message, tracks the current attachment list so the model always knows what it has loaded — without that metadata entering the conversation history and bloating the window.

---

## Features

- **REPL-native execution** — model writes Python, Python runs, state persists
- **Agent-defined tools** — tools are Python functions in the REPL namespace, not JSON schemas; the agent can define and discover its own
- **No permissions model** — full access by default; file edits are tracked as unified diffs and can be reverted on request
- **Zero dependencies** — standard library only
- **Lossless infinite context** — old exchanges coalesce automatically into expandable previews; the agent actively curates its own context with `observe`, `rollup`, and `pin`; sessions run past 50,000 turns
- **Full-file attachment system** — `view` / `unview` with ephemeral context tracking; always full context, never stale fragments
- **AST preprocessing** — output shaped for REPL execution, errors transparently corrected or retried silently
- **Session persistence** — SQLite-backed sessions, fully replayable and resumable
- **Subagents** — spawn isolated parallel Code Agent subprocesses for independent tasks
- **Remote SSH workers** — run the REPL worker on any SSH-accessible host; LLM calls and session state remain local, only Python required remotely
- **Session forking** — fork an active session to explore multiple paths in parallel; sessions are otherwise locked to one writer
- **Rewind** — step back through conversation history and continue from any earlier point
- **Auto-attached project context** — `CLAUDE.md` / `AGENTS.md` are loaded automatically at startup, with recursive `@file` reference resolution
- **MCP support** — external MCP tools exposed as Python functions in the REPL
- **Skills** — reusable markdown instruction packs; includes an AgentLib agent-builder skill, a subagent orchestration skill, and more
- **Provider-agnostic** — Anthropic, OpenAI, Google, and any compatible endpoint

---

## Terminal UX

Code Agent is a terminal tool, not a TUI that hijacks your screen. It uses the standard terminal scrollback buffer rather than taking over, so conversation history is still sitting there after you exit.

- **`Ctrl+O`** — transcript viewer: inspect the persisted textual conversation, including attachment placeholders (not attachment payloads)
- **`/repl`** — drop into the shared REPL yourself, write Python alongside the model, inspect state, inject values, then hand control back
- **`/rewind`**, **`/fork`**, **`/resume`**, **`/subagents`**, **`/skills`**, **`/model`**, **`/tokens`** — standard slash commands
- **`/exec [instructions]`** — generate an editable continuation prompt, reset the active session to the beginning under the same session ID, clear non-auto context, and preload the prompt so you can start a fresh branch with preserved task context

---

## Skills

Skills are reusable markdown instruction files that persist as durable context memory — expertise the agent can load on demand with `/skills`. They are a first-class part of the workflow. I use them constantly and have been building a growing collection of frameworks and tools designed for use by Code Agent.

The built-in **agent builder** skill is a good example of the idea: it provides guidance for building agents with [AgentLib](https://github.com/jacobsparts/agentlib), so `coda` can build, deploy, and debug an agent application end to end. The **orchestrator** skill encodes a disciplined subagent pipeline — implementation, review, steering, and verification.

---

## Quick Start

```bash
pip install git+https://github.com/jacobsparts/code-agent.git
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY, GOOGLE_API_KEY, etc.
coda
```

Or clone and install:

```bash
git clone https://github.com/jacobsparts/code-agent.git
cd code-agent
pip install -e .
coda
```

The main configuration file is `~/.code-agent/config.py`. It is plain Python and can configure API keys, register additional providers and models, and set the ordered list of default models you cycle through at the prompt. For example:

```python
# ~/.code-agent/config.py
import os

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["OPENAI_API_KEY"] = "sk-..."

code_agent_model = [
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.4",
]

```

The first entry is the default. Use `Tab` / `Shift+Tab` at the prompt to cycle forward or backward through the configured models, or select one directly with `/model` or `coda --model`.

API keys can also be supplied with the `api_key` parameter on a provider definition, including when registering another endpoint such as [codex-gateway](https://github.com/jacobsparts/codex-gateway), [cursor-gateway](https://github.com/jacobsparts/cursor-gateway), a local server, or a corporate proxy. Project `.env` files remain supported as another option. Provider definitions can configure compatible endpoints, headers, concurrency, rate limits, and other transport settings; model definitions configure aliases, context windows, costs, request parameters, and per-model overrides. See [docs/configuration.md](docs/configuration.md) for the complete reference and examples.

---

## Remote Workers

By default the REPL worker runs locally, but it can run on any host you can SSH into:

```bash
coda example.com
coda root@example.com
coda root@example.com:project-dir
```

The `:project-dir` suffix sets the worker's CWD before Python starts — relative to the remote login directory or absolute.

**All sensitive operations remain local.** LLM API calls, authentication, session persistence, and the agentic loop itself never leave your machine. The remote side runs a standard Python worker process over stdin/stdout, streamed through the SSH connection. The SSH transport is a wrapper around the local `ssh` client, so your existing SSH configuration applies automatically.

**Zero remote setup.** The only remote dependency is Python. No agent software to install, no API keys to configure, no daemons to run. If you can `ssh` into it, you can run Code Agent on it — with full functionality. Point it at any box you can reach and it becomes a complete agent environment. It has been amazing for admin work.

---

## Community

Bug reports and PRs are welcome.

---

## History

Code Agent started life as one agent paradigm inside [AgentLib](https://github.com/jacobsparts/agentlib). It grew until it was crowding out everything else, so I split it into its own project — AgentLib remains a general-purpose library for building and shipping agents, useful well beyond Code Agent. The provider gateways followed the same path more recently: `codex-gateway` and `cursor-gateway` were broken out of Code Agent's provider layer into standalone projects.

---

## Related Projects

Part of a family of developer tools for agentic coding and model gateways:

- **[Code Agent](https://github.com/jacobsparts/code-agent)** — A Python REPL-native coding agent designed around lean context, persistent execution state, and truly lossless infinite context via coalescing and agent-managed rollups.
- **[AgentLib](https://github.com/jacobsparts/agentlib)** — A lightweight, production-proven library for building and shipping LLM agents quickly, where composable agents are defined as Python classes — making it both simple and powerful.
- **[codex-gateway](https://github.com/jacobsparts/codex-gateway)** — Pure-Python OpenAI Responses API-compatible gateway for Codex/ChatGPT OAuth accounts with quota management, account rotation, and automated resets.
- **[cursor-gateway](https://github.com/jacobsparts/cursor-gateway)** — Pure-Python OpenAI-compatible Chat Completions gateway that wraps the Cursor Agent API with synthetic checkpoints to provide real native tool calling and cache-friendly session routing.

## License

MIT
