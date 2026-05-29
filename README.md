# Code Agent

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

A Python REPL-native agent. The model's response is Python.

Every assistant turn is source code
executed in a persistent subprocess REPL, with full access to the filesystem, shell, and a
set of built-in functions for reading files, editing code, searching, spawning processes, and
managing context. The model decides when it's done by calling `emit(..., release=True)`.

This is my daily driver, built and refined through real use.

Here's what a session looks like:

```
$ code-agent

──────────────────────────────────
Code Agent
Python REPL-based coding assistant
claude-sonnet-4-5
──────────────────────────────────
Loading AGENTS.md

═ User ════════════════════════════
> Summarize the error distribution in today's logs

─ Python ──────────────────────────
>>> from collections import Counter
>>> log = read("logs/app.log")
>>> errors = [l for l in log.splitlines() if "ERROR" in l]
>>> counts = Counter(l.split("]")[-1].strip() for l in errors)
>>> print("\n".join(f"{n:4d}x  {msg}" for msg, n in counts.most_common(5)))
  47x  database connection timeout
  23x  failed to parse response body
   8x  rate limit exceeded
   3x  file not found: config.yaml
   1x  unexpected keyword argument 'timeout'
>>> think("I'll review the output and ask if they want to dig in.")

═ Output ══════════════════════════
Five error types. The dominant issue is connection timeouts — 47 hits.
Worth checking pool settings.

═ User ════════════════════════════
> Yes, bump the pool size in config.py
...

Session ended. Goodbye!
Resume session: code-agent --resume f5867c0c-6963-405e-9c6a-c775cea6cb6a
claude-sonnet-4-5: In=12847, Cached=41203, Rsn=1024, Out=843, Cost=$0.118
```

---

## Why this approach

**Model performance degrades with context length.** Most agents make this worse: tool schemas,
JSON envelopes, alternating turn structure, and compaction artifacts all add tokens that dilute
the signal. Code Agent is designed around the opposite principle—keep the context lean, and the
model performs better for longer. The REPL execution model, lossless coalescing, and attachment
system all serve this goal. Long sessions that would require multiple lossy compaction passes
elsewhere run cleanly here, often staying well under 200k tokens.

**Token efficiency.** Native tool calls carry marshalling overhead—JSON wrapping, schema
validation, protocol round-trips. Plain Python has none of that. Multi-step tasks happen in
a single assistant turn.

**AST-based preprocessing.** Model responses are parsed and shaped before execution—normalizing output patterns and correcting common edge cases transparently. Failures that can't be resolved are retried silently, keeping the conversation clean.

**Persistent REPL state.** Variables, database connections, subprocess handles, and imports
survive across turns. Open a connection once, use it for the whole session.

---

## Agents manage their own context

**Lossless coalescing.** Old turns are automatically coalesced past a three-interaction horizon (each interaction may span many turns), and the system actively coalesces as needed to stay under the context window—including mid-session on long-running interactions. Coalesced exchanges become expandable preview blobs, not discarded text, so nothing is permanently lost. The agent can also `pin()` important turns so they survive coalescing and auto-expand when referenced.

This is fundamentally different from compaction: compaction summarizes and discards, coalescing archives. The full REPL state carries forward regardless.

Because sessions stay lean, you can configure a *smaller* context window than you otherwise could—where compaction would be required with a large window, Code Agent runs cleanly with a smaller one, and smaller context means better model performance.

**Full-file attachments, not grep fragments.** Most agents search for snippets and operate
on partial context, which is a consistent source of errors. Code Agent uses an attachment
system: `view(file)` pulls the full file into context with line numbers; `unview(file)`
removes it. Only the most recent version of each file is ever in context—it updates
automatically after edits. An ephemeral system prompt tracks the current attachment list
so the model always knows what it has loaded, without that metadata entering conversation
history and bloating the window.

---

## Features

- **REPL-native execution** — model writes Python, Python runs, state persists across turns
- **No permissions model** — full access by default; file edits are tracked as unified diffs and can be reverted on request
- **Zero dependencies** — standard library only
- **Lossless coalescing** — old REPL exchanges coalesce automatically; preview blobs keep large outputs accessible without staying in the window
- **Full-file attachment system** — `view` / `unview` with ephemeral context tracking; always full context, never stale fragments
- **AST preprocessing** — output shaped for REPL execution; errors transparently corrected or retried silently
- **Session persistence** — SQLite-backed sessions, fully replayable and resumable
- **Subagents** — spawn isolated parallel Code Agent subprocesses for independent tasks
- **Session forking** — fork an active session to explore multiple paths in parallel; sessions are otherwise locked to one writer
- **Rewind** — step back through conversation history and continue from any earlier point
- **Auto-attached project context** — `CLAUDE.md` / `AGENTS.md` are loaded automatically at startup, with recursive `@file` reference resolution
- **MCP support** — external MCP tools exposed as plain Python functions in the REPL
- **Skills** — attach reusable markdown instruction files as context memory
- **Provider-agnostic** — Anthropic, OpenAI, Google, OpenRouter, any OpenAI-compatible endpoint

---

## Terminal UX

Code Agent feels like a terminal tool, not a GUI app. Normal scrollback buffer, no
alternate-screen takeover.

- **`Ctrl+O`** — transcript viewer: inspect the full agent-side conversation, including the
  Python the model sent and everything it saw
- **`/repl`** — drop into the shared REPL yourself, write Python alongside the model,
  inspect state, inject values, then hand back control
- **`/rewind`**, **`/fork`**, **`/resume`**, **`/subagents`**, **`/skills`**, **`/model`**,
  **`/tokens`** — standard slash commands

---

## Quick Start

```bash
pip install git+https://github.com/jacobsparts/code-agent.git
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY, GOOGLE_API_KEY, etc.
code-agent
```

Or clone and install:

```bash
git clone https://github.com/jacobsparts/code-agent.git
cd code-agent
pip install -e .
code-agent
```

Copy `.env.example` to `.env` to persist your API key and preferred model.

Supports Anthropic, OpenAI, Google, X.AI, OpenRouter, and any OpenAI-compatible endpoint.
Custom providers and models (local Ollama, corporate proxies, etc.) are registered in
`~/.code-agent/config.py`. See [docs/configuration.md](docs/configuration.md).

---

## Roadmap

The subprocess transport is already abstracted. The default transport uses native
multiprocessing with fork, which is fast and works well for local agents. There is also an
alternate external-process transport that wraps a Python worker with a tiny `-e` stub and
communicates entirely over stdin/stdout.

That opens the door to sandboxed workers and remote Python workers over SSH, while keeping
the same REPL-native agent model.

---

## Community

Code Agent is my daily driver, but it is early and I am interested in feedback from people
who try it seriously. Issues, bug reports, PRs, design notes, and weird session transcripts
are all welcome.

---

## License

MIT
