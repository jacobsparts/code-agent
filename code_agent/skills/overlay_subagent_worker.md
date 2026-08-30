# Recursive overlay subagent orchestration

This skill applies inside an `OverlaySubagent` created with `recursive=True`. You are already running in a private overlay and may create child overlay workers from sealed snapshots of your current filesystem state.

## Your role

You are responsible for every child you create. Each child starts with only:

- its system prompt;
- a sealed filesystem snapshot taken at child creation;
- the prompt passed to `send()`.

A child does not inherit your conversation, reasoning, discoveries, requirements, or plan. Give every child a complete standalone task. Do not ask it to infer requirements from the current diff, filenames, or partial work. A child also has its own Python instance; serialize parent objects into the prompt or a temporary file and give the child the path.

A delegated task should state:

- the concrete objective and required behavior;
- relevant requirements and decisions;
- exact scope and exclusions;
- relevant files, symbols, and starting points;
- invariants and acceptance criteria;
- tests and verification to run;
- whether edits are allowed;
- the exact terminal submission requirement.

## Creating children

Use the normal module API and keep an explicit reference to the child:

```python
from code_agent.overlay_subagent import OverlaySubagent

child = OverlaySubagent()
try:
    response = child.send("<complete standalone task>")
    if response.is_error:
        raise RuntimeError(response.error)
    if response.submission_error:
        raise RuntimeError(response.submission_error)
    print(response.diff())
    response.apply()
finally:
    child.close()
```

`OverlaySubagent()` creates a leaf child. If that child may create descendants, opt it into recursive orchestration and attach this skill:

```python
child = OverlaySubagent(recursive=True)
try:
    response = child.send("<complete standalone orchestration task>")
finally:
    child.close()
```

Set `recursive=True` at every boundary where the receiving child may orchestrate. Do not set it for leaf workers.

## Isolation and integration

Creating a child seals your current overlay state:

- the child sees your state as it existed at child creation;
- your later edits do not change the child's view;
- child edits stay private until you apply submitted artifacts;
- applying a child response writes into your private overlay, not the caller's working tree;
- sibling changes are not merged automatically;
- closing a parent closes its children.

Use background sends only for independent work:

```python
first = OverlaySubagent()
second = OverlaySubagent()
try:
    first_response = first.send(first_task, bg=True)
    second_response = second.send(second_task, bg=True)
    first_response.wait()
    second_response.wait()
finally:
    first.close()
    second.close()
```

Keep every child in a named variable or collection until its work and response
handling are complete, then call `close()` promptly. Do not rely on response
objects to retain children. `__del__` provides best-effort cleanup only.
Closing a parent also closes remaining children, but is not a substitute for
explicit child lifecycle management.

## Reusing and steering a child

A child is persistent and reusable. It keeps its conversation, REPL state, and
private overlay changes between tasks. Prefer sending follow-up work to an
existing child over creating a new one, unless the new work is a genuinely
different task or should run in parallel.

A child runs one task at a time. If you learn something that changes the task,
redirect it instead of starting over:

```python
response = child.send(
    "Stop the current approach. Use src/config_parser.py instead. Continue the "
    "same fix there and submit the changed files.",
    interrupt=True,
)
```

`interrupt=True` stops the child's current task and sends yours as its next
task. The interrupted response ends as an error and remains readable.

`max_turns` bounds a single task and defaults to 100. Reaching it ends the task
with an error, but the child stays alive and its overlay changes are intact. Ask
where it got to, then continue or restart:

```python
status = child.send("Stop and summarize what you have done and what remains.")
print(status.result)
child.send("Continue with the remaining steps, then submit the changed files.")
```

Continuing starts a fresh task, so the turn budget resets. Use `child.interrupt()`
to stop work without sending a new task. Close and recreate a child only when it
has gone too far in the wrong direction.

Keep sibling file ownership separate where possible. You remain responsible for reviewing diffs, resolving interactions, applying selected artifacts, and running integrated verification.

## Submission to your parent

After integrating child work into your overlay:

1. inspect the combined diff;
2. run the required tests;
3. submit every intended project-relative path in your own terminal emit.

```python
emit(
    "<summary and exact verification>",
    release=True,
    files=["<all consolidated paths for your parent>"],
)
```

Applying child artifacts does not submit them automatically. Your final `files=[...]` list is the only artifact set returned to your parent.
