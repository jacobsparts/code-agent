# Overlay subagents with isolated, reviewable file changes

Use this skill when delegating coding work to `OverlaySubagent`. Overlay subagents are persistent Code Agent workers whose changes are isolated in a private home-directory overlay. A worker can explicitly submit selected files for the parent to inspect and conflict-check before applying them to the real project.

## When to use overlay subagents

Use `OverlaySubagent` when you need:

- an agent to edit files without immediately changing the parent working tree;
- explicit review of submitted artifacts before accepting changes;
- deterministic diffs and conflict-checked application;
- follow-up tasks in the same isolated worker session;
- child workers based on a sealed snapshot of a parent worker's state.

Use ordinary `Subagent` instead when filesystem isolation and artifact submission are unnecessary.

## Requirements and boundaries

- The overlay backend requires Linux, procfs, user namespaces, overlayfs support, and the `cp` command.
- The project directory must be beneath the current user's home directory.
- The worker receives a private overlay of the user's home, not just the project directory.
- Changes remain private unless the worker submits paths and the parent applies the resulting artifacts.
- Submitted paths must be project-relative POSIX paths. Absolute paths, `..`, backslashes, directories, and paths escaping through symlinked parents are rejected.
- Per-file and aggregate submission limits apply.
- Keep an explicit reference to every worker for its full lifetime and call `close()` as soon as it is no longer needed. The destructor is only best-effort cleanup.
- Set `recursive=True` when a worker may create overlay subagents of its own. This attaches the worker-specific recursive orchestration skill and enables child construction in that worker.

## Providing context

An overlay worker starts with no knowledge of the parent conversation,
findings, goals, or plan. It has only its system prompt, its filesystem view,
and the task prompt. Make every task self-contained: state the goal, definition
of done, files to inspect, constraints, prior findings, and exclusions.

The worker also has its own Python instance, so parent objects are not shared.
Serialize needed data into the prompt or a temporary file and tell the worker
which path to read.

## Quick start

```python
from code_agent.overlay_subagent import OverlaySubagent

agent = OverlaySubagent(cwd="/home/me/project")
try:
    response = agent.send(
        "Fix the parser bug. Run focused tests, then submit every file you changed "
        "with emit(summary, release=True, files=[...])."
    )

    print(response.result)
    print(response.diff())

    if not response.is_error and not response.submission_error:
        response.apply()
finally:
    agent.close()
```

The worker must explicitly submit files in its terminal `emit` call:

```python
emit(
    "Fixed parser handling and added regression coverage.",
    release=True,
    files=["src/parser.py", "tests/test_parser.py"],
)
```

`files=` is valid only with `release=True`. Merely editing a file in the overlay does not include it in the response.

## Overlay hierarchy and isolation boundaries

The isolation rules depend on which layer is doing the work.

| Layer | Role | Filesystem relationship | Concurrency guidance |
|---|---|---|---|
| 1 | Top-level Code Agent | Real working tree; no overlay | Avoid files assigned to active first-level subagents |
| 2 | First-level overlay subagent | Private writable overlay over the top-level Code Agent's live files | Best orchestration layer; may keep editing while its children work from sealed snapshots |
| 3 | Recursive overlay subagent | Private writable overlay over a sealed parent snapshot | Stable independent work environment; usually a leaf worker |

The key asymmetry is between Layers 1 and 2. Layer 1 remains live beneath a first-level subagent, while every Layer 2-to-Layer 3 boundary is sealed at child creation.

### Layer 1: top-level Code Agent

The top-level Code Agent runs directly in the real working tree. It never has an overlay of its own.

When it creates a first-level `OverlaySubagent`, that subagent initially reads unchanged files through a live view of the top-level Code Agent's home directory. Top-level changes are therefore reflected in the first-level subagent immediately, but only while each file remains unmodified in the subagent's overlay.

### Copy-up is a permanent visibility boundary

The first time an overlay subagent modifies a file, overlayfs copies that file into the subagent's private writable layer. From then on, that subagent reads its private copy for the rest of its lifetime. Later edits to the same real working-tree file by the top-level Code Agent are not visible to that existing subagent.

This creates an important communication constraint:

- Before the subagent modifies a file, top-level changes to it can appear in the subagent's live lower view.
- After the subagent modifies the file, top-level changes to it cannot be demonstrated merely by editing the working tree and asking the same subagent to inspect them.
- To discuss a later top-level change with that subagent, pass the relevant patch, code, or explanation explicitly in a follow-up message.
- If the subagent needs the changed files as filesystem state rather than message context, close it and create a new first-level `OverlaySubagent`.
- Do not assume follow-up prompts refresh or rebase an existing subagent's private files.

In short: **an unmodified file tracks the live top-level working tree; a modified file remains pinned to the subagent's private copy.**

The top-level Code Agent should avoid modifying files that a first-level subagent is actively reading, editing, testing, or preparing to submit:

- Assign each file or tightly coupled file set to one side at a time.
- Continue independent work only in unrelated files.
- Avoid broad formatting, generation, dependency, or fixture operations that may touch the subagent's scope.
- Do not rely on `ApplyConflict` as the primary coordination mechanism. It can detect many before-state divergences at apply time, but top-level changes may already have affected the subagent's work before submission.

The safe default is: once a first-level subagent owns a task, the top-level Code Agent leaves the relevant files unchanged until the submission is reviewed and either applied or rejected.

### Layer 2: first-level overlay subagent

A first-level overlay subagent is any `OverlaySubagent` created directly by the top-level Code Agent. It is often the best place to orchestrate a parallel feature implementation.

This subagent has its own private writable overlay. When it creates child overlay subagents, its current state is sealed into snapshots for those children. After each child is created, the first-level subagent can freely continue modifying and integrating files in its own overlay without changing that child's filesystem view.

This makes the first-level subagent an effective orchestrator:

- establish shared groundwork;
- create children for parallel workstreams;
- continue its own implementation or integration work;
- review and apply child submissions as they arrive;
- reconcile overlapping changes;
- run integrated tests;
- submit one consolidated result to the top-level Code Agent.

### Layer 3: recursive overlay subagents

A recursive or second-level overlay subagent starts from a sealed snapshot of its parent subagent. It can work without concern that later parent edits will alter its view.

A recursive subagent can technically create further descendants, and the same snapshot rule applies. Prefer keeping the hierarchy shallow unless another level has a clear architectural benefit; a first-level orchestrator with second-level workers is usually sufficient.

## Recommended top-level workflow

1. Create a first-level worker with the intended project root.
2. Give it a coherent task, scope exclusions, file ownership, and verification requirements.
3. Avoid modifying the real working-tree files or shared inputs assigned to that worker.
4. Require it to submit every intended change explicitly.
5. Check task and submission errors.
6. Inspect `response.result`, `response.files`, and `response.diff()`.
7. Run any additional review needed before applying.
8. Apply all or selected artifacts to the real working tree.
9. Run tests in the top-level working tree.
10. Send follow-up work to the same worker when session continuity is useful.
11. Close the worker.

Treat the worker's report and tests as claims to verify. Applying artifacts transfers file changes; it does not prove the implementation is correct.

## Response model

A foreground send waits by default:

```python
response = agent.send("Implement the requested change")
print(response.result)
```

Important response attributes:

- `response.result`: final text returned by the worker.
- `response.files`: immutable mapping of submitted path to `SubmittedFile`.
- `response.progress`: snapshot of progress messages.
- `response.turns`: number of worker turns.
- `response.done`: whether the task has finished.
- `response.is_error`: whether execution failed.
- `response.error`: serialized task or worker error, if any.
- `response.submission_error`: failure while materializing submitted files.
- `response.diff(paths=None)`: combined deterministic diff.
- `response.apply(paths=None, root=None)`: conflict-check and apply artifacts.
- `response.wait(timeout=None)`: wait for a background response.

A task can finish successfully while `submission_error` is non-empty. In that case, the text result is available, but the requested files were not safely materialized. Do not apply or assume the changes were captured.

## Inspecting submitted files

`response.files` is keyed by normalized project-relative path:

```python
for path, artifact in response.files.items():
    print(path, artifact.operation)
    print(artifact.diff())
```

Each `SubmittedFile` includes:

- `path`;
- `operation`: `"create"`, `"modify"`, or `"delete"`;
- immutable `before` and `after` bytes for regular files;
- before/after mode values;
- before/after symlink targets when applicable.

For a submitted regular file:

```python
artifact = response.files["src/parser.py"]
new_text = artifact.text()
with artifact.open() as stream:
    new_bytes = stream.read()
```

`text()` and `open()` operate on after-state content. They reject deletions and symlink artifacts.

## Reviewing diffs

Review every artifact before applying:

```python
print(response.diff())
```

Select paths when only part of the submission is relevant:

```python
print(response.diff(paths=["src/parser.py"]))
```

Diff output is deterministic and path-sorted by default. Text files use unified diffs. Binary files, symlinks, and mode-only changes use stable metadata summaries.

Unknown paths raise `KeyError`; invalid paths raise `PathValidationError`.

## Applying artifacts

Apply the complete submission to the worker's original project root:

```python
response.apply()
```

Apply selected paths:

```python
response.apply(paths=["src/parser.py", "tests/test_parser.py"])
```

Apply to another root, such as a temporary verification tree:

```python
from pathlib import Path

response.apply(root=Path("/home/me/project-copy"))
```

Application checks that each destination still matches the captured before-state. It rejects cases such as:

- a create destination now exists;
- a modified or deleted file has changed;
- a destination's type, symlink target, or captured mode changed;
- an intermediate symlink escapes the destination root;
- a destination is a directory.

```python
from code_agent.overlay_subagent import ApplyConflict

try:
    response.apply()
except ApplyConflict as exc:
    for conflict in exc.conflicts or [exc]:
        print(conflict.path, conflict.reason)
```

Selected artifacts are preflighted together, so ordinary conflicts are collected before writes begin. Multi-file application is not transactionally atomic: an unexpected write-time failure can still leave a partial application.

After applying, run parent-side tests and inspect the real working-tree diff.

## Background tasks and progress

Use background execution only for actual concurrency:

```python
response = agent.send("Run the longer implementation task", bg=True)

# Do independent work in files outside this worker's scope.

response.wait()
print(response.progress)
print(response.result)
```

Do not poll in noisy waiting loops. `response.progress` is a snapshot list populated by worker calls such as:

```python
emit("Focused tests pass; reviewing the diff now.")
```

A worker can run only one task at a time. Sending another task while its current response is unfinished raises `RuntimeError` unless you pass `interrupt=True`.

## Steering an active worker

If you learn something that changes the task, redirect the worker instead of starting over:

```python
response = agent.send(
    "Stop the current approach. The real target is src/config_parser.py. "
    "Continue the same fix there and submit the changed files.",
    interrupt=True,
)
```

`interrupt=True` stops the current task and sends yours as the next task on the same worker. The worker's conversation, REPL state, and private overlay changes are kept. The interrupted response ends as an error and remains readable.

To check on a running worker, ask it to stop and report:

```python
status = agent.send("Stop and summarize what you have done and what remains.", interrupt=True)
print(status.result)
agent.send("Good. Continue with the remaining steps and submit the changed files.")
```

Use `agent.interrupt()` to stop the current task without sending new work. Use `agent.kill()` only to discard the worker entirely.

## Reuse workers

An `OverlaySubagent` is persistent and reusable. It keeps its conversation, REPL state, and private filesystem changes between tasks. Prefer following up with an existing worker over creating a new one. Create a new worker only when moving to a genuinely different task, when running tasks in parallel, or when the worker needs a fresh view of files it has already modified.

```python
agent = OverlaySubagent(cwd="/home/me/project")
try:
    first = agent.send(
        "Implement the parser fix and submit the changed files."
    )
    print(first.diff())

    second = agent.send(
        "Address the remaining edge case, rerun tests, and submit the complete "
        "set of files needed for this follow-up."
    )
    print(second.diff())
finally:
    agent.close()
```

Each response contains only paths explicitly submitted by that terminal `emit`. Do not assume a later response automatically includes files submitted earlier.

## Child overlay workers

Overlay workers use the same module API as the top-level caller:

```python
from code_agent.overlay_subagent import OverlaySubagent

child = OverlaySubagent()
try:
    response = child.send("<complete delegated task>")
    if response.is_error:
        raise RuntimeError(response.error)
    print(response.diff())
    response.apply()
finally:
    child.close()
```

Inside a worker created with `recursive=True`, constructing `OverlaySubagent` creates a child from that worker's current private overlay snapshot. A worker created with the default `recursive=False` cannot create children. The caller uses the same constructor API at every enabled level rather than a separate recursive API.

### Parent and child responsibilities

Every new child begins with only:

- its system prompt;
- the filesystem snapshot captured when it was created;
- the task prompt passed to `send()`.

A child does **not** inherit its parent's conversation, reasoning, discoveries, requirements, plans, or understanding of the user's request. Filesystem state is not task context. The child must not be expected to infer its assignment from uncommitted changes, commit history, filenames, or the parent's partially completed work.

The parent is responsible for bringing each child up to speed from nothing beyond the child's system prompt. Every delegated prompt must be self-contained and include:

- the concrete objective and required behavior;
- relevant user requirements and decisions;
- exact scope and exclusions;
- relevant files, symbols, interfaces, or starting points;
- constraints and invariants that must be preserved;
- acceptance criteria;
- required tests or verification;
- whether edits are allowed;
- the exact submission requirement, including terminal `emit(..., files=[...])`.

If the child may create recursive overlay subagents, construct it with `recursive=True`:

```python
child = OverlaySubagent(recursive=True)
try:
    response = child.send("<complete delegated task>")
finally:
    child.close()
```

`recursive=True` enables child construction in that worker and attaches the worker-specific recursive orchestration skill before the first interaction. That skill explains the recursive API, isolation model, and parent responsibilities without including the top-level caller mechanics in this skill. It does **not** transfer the parent's conversation or task context, so the delegated prompt must still be complete and self-contained.

Use `recursive=True` only for a child that may orchestrate descendants. Leaf workers should keep the default `recursive=False`. This choice is required at every delegation boundary: a parent cannot assume that descendants received the worker skill merely because the parent did.

Do not send vague delegated prompts such as:

```text
Implement the parser change and submit the changed files.
```

Instead, provide the missing context explicitly:

```text
Fix parsing of trailing commas in src/config_parser.py.

Required behavior:
- Accept one trailing comma in object and array literals.
- Continue rejecting repeated commas and missing values.
- Preserve all existing error locations and messages outside this case.

Scope:
- Edit src/config_parser.py and focused parser tests only.
- Do not change the public parser API or unrelated formatting.
- Do not commit.

Verification:
- Run pytest -q tests/test_config_parser.py.
- Review the final diff.

Submission:
- Finish with emit(
    "<summary and exact test result>",
    release=True,
    files=["src/config_parser.py", "tests/test_config_parser.py"],
  ).

If you delegate recursively, create each orchestrator child with
`recursive=True` and give every child an equally self-contained task.
```

### Concurrent fan-out and integration

Use background sends for genuinely independent workstreams:

```python
from code_agent.overlay_subagent import OverlaySubagent

parser_task = """
Fix parsing of trailing commas in src/config_parser.py.

Required behavior:
- Accept a single trailing comma in arrays and objects.
- Preserve existing failures for repeated commas and missing values.

Scope:
- Own src/config_parser.py and tests/test_config_parser.py only.
- Do not commit.

Verification:
- Run pytest -q tests/test_config_parser.py.

Submission:
- Finish with emit(summary, release=True, files=[
    "src/config_parser.py",
    "tests/test_config_parser.py",
  ]).
"""

docs_task = """
Update docs/configuration.md for the accepted trailing-comma syntax.

Scope:
- Own docs/configuration.md only.
- Do not edit parser code or tests.
- Do not commit.

Verification:
- Review the rendered section and final diff.

Submission:
- Finish with emit(summary, release=True, files=[
    "docs/configuration.md",
  ]).
"""

parser_child = OverlaySubagent()
docs_child = OverlaySubagent()
try:
    parser_response = parser_child.send(parser_task, bg=True)
    docs_response = docs_child.send(docs_task, bg=True)

    parser_response.wait()
    docs_response.wait()

    if parser_response.is_error:
        raise RuntimeError(parser_response.error)
    if docs_response.is_error:
        raise RuntimeError(docs_response.error)

    print(parser_response.diff())
    print(docs_response.diff())
    parser_response.apply()
    docs_response.apply()
finally:
    parser_child.close()
    docs_child.close()
```

The child response exposes `result`, `files`, `progress`, `turns`, `done`, `is_error`, `error`, and `submission_error`, plus `wait()`, `diff()`, and `apply()`. Applying a child response writes into the orchestrator worker's private overlay, not the real top-level working tree. After reviewing, integrating, and testing child work, the orchestrator must submit the consolidated paths in its own terminal `emit(..., release=True, files=[...])`.

Creating a child seals the parent's current state into the child's lower-layer chain:

- the child sees the parent overlay's state as it existed at child creation;
- later parent edits do not change the child's view;
- child edits stay private until the parent applies the submitted artifacts;
- a child created with `recursive=True` can create descendants through the same `OverlaySubagent` constructor;
- there is no configured recursion-depth limit;
- closing a parent closes its children.

This enables hierarchical fan-out/fan-in:

1. A parent divides a broad task into genuinely separable workstreams.
2. It gives each child a complete standalone assignment.
3. Children work concurrently from sealed snapshots.
4. The parent reviews and applies selected child submissions into its private overlay.
5. The parent reconciles interactions and runs integrated verification.
6. The parent submits one consolidated artifact set to its own caller.

Snapshot isolation does not merge overlapping sibling changes. Keep file ownership separate where possible; the parent remains responsible for conflict resolution and final correctness.

## Model and turn configuration

```python
agent = OverlaySubagent(
    cwd="/home/me/project",
    model="provider/model-name",
    max_turns=100,
    recursive=True,
)
```

Set `recursive=True` only when the worker may create children. It attaches the worker-specific recursive orchestration skill before the first interaction and enables child construction. Omit it for leaf workers.

`send()` can override task limits:

```python
response = agent.send(
    "Investigate and fix the issue",
    max_turns=80,
    timeout=1800,
)
```

The foreground `timeout` controls how long `send()` waits. If it expires, the response may still be running; inspect `response.done` or call `response.wait()` later.

`max_turns` bounds a single task and defaults to 100. Raise it for genuinely complex work; lower it for simple or delicate work you want to review early.

Reaching the turn limit ends the task with an error, but the worker stays alive and reusable and its overlay changes are intact. Ask where it got to, then decide:

```python
status = agent.send("Stop and summarize what you have done and what remains.")
print(status.result)
agent.send("Continue with the remaining steps, then submit the changed files.")
```

Continuing starts a fresh task, so the turn budget resets. If the worker has gone far in the wrong direction, kill it and start again with better instructions rather than correcting a long bad trajectory.

Snapshot controls are available for child creation:

```python
agent = OverlaySubagent(
    cwd="/home/me/project",
    snapshot_byte_limit=1024 * 1024 * 1024,
    snapshot_inode_limit=250_000,
    snapshot_timeout=120.0,
    snapshot_min_free_bytes=512 * 1024 * 1024,
)
```

Keep defaults unless the repository size and host capacity justify deliberate changes.

## Error handling

Capability and runtime failures raise exceptions such as:

- `PathValidationError`;
- `SubmissionError`;
- `ApplyConflict`;
- `OverlayRuntimeError`.

Task failures are usually represented on the response:

```python
response = agent.send("Implement the change")

if response.is_error:
    print(response.error)
elif response.submission_error:
    print("Task completed, but file submission failed:")
    print(response.submission_error)
else:
    print(response.diff())
```

Worker startup failures may include bounded trailing stdout/stderr diagnostics. A failed subagent result should not be applied blindly.

## Cleanup and cancellation

Keep an explicit reference to each worker until all of its responses have been
consumed and no follow-up work remains. Then close it directly:

```python
agent = OverlaySubagent(cwd="/home/me/project")
try:
    response = agent.send("Do the work")
finally:
    agent.close()
```

Do not create workers as temporary expressions or rely on a response object to
keep its worker alive. `__del__` attempts best-effort cleanup if the worker
reference is accidentally lost, but destructor timing is not a lifecycle
contract.

Use `agent.interrupt()` to stop active work while keeping the worker and its
overlay usable. Use `agent.kill()` to terminate the worker and clean up the
overlay runtime. `close()` is idempotent and recursively closes children. Even
though closing a parent closes any remaining children, the parent should retain
each child in a named variable or collection and close it promptly when that
child's work is finished.

## Prompt pattern

A useful task prompt makes submission explicit:

```text
Implement the requested parser fix.

Requirements:
- Inspect existing code before editing.
- Change only parser behavior and focused tests.
- Do not commit.
- Run the relevant tests.
- Review your final diff.
- Finish with:
  emit(
      "<concise summary and exact verification>",
      release=True,
      files=["<every changed file that should be reviewed and applied>"],
  )
- Submit project-relative file paths only.
- If no files should be applied, use files=[] and explain why.
```

For review-only work, tell the worker not to edit and omit `files`, or submit an empty list.

## Safety checklist

Before applying:

- [ ] `response.done` is true.
- [ ] `response.is_error` is false.
- [ ] `response.submission_error` is empty.
- [ ] Submitted paths match the intended scope.
- [ ] Every required changed file was explicitly submitted.
- [ ] `response.diff()` has been inspected.
- [ ] Binary, symlink, deletion, and mode changes are intentional.
- [ ] The destination working tree has not diverged unexpectedly.

After applying:

- [ ] Inspect the parent working-tree diff.
- [ ] Run focused tests.
- [ ] Run broader checks when warranted.
- [ ] Confirm no unrelated files changed.
- [ ] Close the worker.
