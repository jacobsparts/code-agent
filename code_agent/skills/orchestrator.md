# Effective Subagent Development Workflow

Use subagents as an implementation and review pipeline, not as a one-shot delegation mechanism.

## Core principles

1. **Keep the parent agent responsible for the outcome.**
   - The parent owns scope, architecture, prioritization, and final verification.
   - Treat subagent reports as claims to verify, not as proof.
   - Inspect the working tree and test results between stages.

2. **Give each subagent one coherent role.**
   - Use an implementation agent for a distinct implementation stage.
   - Use a separate read-only review agent to challenge that implementation.
   - Use a fresh final-review agent after all stages are complete.
   - Reuse an implementation agent for follow-up fixes within the same stage so it retains context.

3. **Divide work at architectural boundaries.**
   Good stage boundaries have independently testable outcomes, such as:
   - refactor shared primitives;
   - add persistence and replay;
   - add policy or agentic behavior.

   Avoid splitting tightly coupled edits across parallel agents. Parallelism is useful only when tasks truly do not share files, invariants, or sequencing.

4. **Make exclusions explicit.**
   State what the subagent must not implement or modify. Examples:
   - no schema migration;
   - no agentic policy yet;
   - no unrelated benchmark changes;
   - no commit;
   - preserve an explicit user decision to remove a compatibility API.

## Before delegation

Orient in the repository and identify:

- governing instructions;
- relevant design documents;
- current working-tree changes;
- production call paths;
- persistence and replay paths;
- focused tests;
- known unrelated failures.

Prepare a stage prompt containing:

- the architectural problem, not merely a list of files;
- required invariants;
- required production integration;
- compatibility requirements;
- explicit non-goals;
- tests and checks to run;
- expected report contents.

If prior work is already present, tell the subagent to inspect the complete diff before editing. Explain known defects in that work precisely.

## Implementation-stage prompt pattern

A strong implementation prompt should include:

### Context

- Read the project instructions and relevant specifications.
- Inspect existing code before modifying it.
- Inspect the current diff and distinguish prior task work from unrelated user work.

### Required architecture

Describe the seam that must exist after implementation. For example:

> The existing production path and future feature must use one shared operation. Do not leave a tests-only wrapper beside a separate production orchestration path.

### Invariants

List invariants explicitly:

- atomicity;
- ordering;
- source identity;
- transaction boundaries;
- replay determinism;
- failure non-mutation;
- backward-compatible persisted format;
- absence of duplicate content.

### Integration requirement

Require a real production caller for new abstractions. A helper exercised only by unit tests is usually speculative or dead code.

### Verification

Require:

- focused tests;
- broader non-benchmark suite;
- syntax compilation;
- diff validation;
- final diff inspection;
- exact deviations and unresolved concerns.

## Review loop

After an implementation agent reports completion:

1. Inspect the diff and status yourself.
2. Spawn a separate read-only reviewer.
3. Give the reviewer the design documents, intended scope, and known risks.
4. Ask for findings ordered by severity with exact references.
5. Require direct probes for subtle state and boundary behavior.
6. Send actionable findings back to the original implementer.
7. Reuse the reviewer to verify the fixes.
8. Repeat until no blocking or actionable findings remain.

Do not ask a reviewer merely whether the code “looks good.” Give it adversarial targets.

## High-value review targets

### Shared abstraction integrity

- Is the new API used by production?
- Are there two orchestration paths doing nearly the same work?
- Is a convenience wrapper dead outside tests?
- Does the abstraction support the real production shape?

### Provenance and range handling

- Can source-aware and source-less nodes be mixed?
- Are duplicate boundaries ambiguous?
- Are ranges ordered, contiguous, atomic, and non-overlapping?
- Are mixed or invalid candidates skipped without partial mutation?

### Persistence

- Is identity distinct from a content-addressed key when necessary?
- Are event payloads minimal?
- Is content duplicated into events?
- Are sequence allocation and inserts in one transaction?
- Can stale callers commit inconsistent state?
- Does failure leave orphaned associations or partial events?

### Replay and rewind

- Does replay use the same validity rules as creation?
- Are all required state components included in snapshots?
- Does rewind restore expansion and placement state?
- Does exec reset the correct state?
- Can parent and child placements replay in order?
- Does fork retain events and blob associations?

### State-machine consistency

If both replay and transactional validation reconstruct state, require a shared transition engine or direct equivalence tests. Duplicate state machines tend to diverge on malformed history, rewind, and boundary cases.

### Integration

- Does the live agent install returned projection and state?
- Are sequence counters updated from committed values?
- Can create → resume → expand → fork → resume fork succeed?
- Does deterministic projection treat persisted nodes as atomic?

## Handling reviewer findings

Classify findings before sending them back:

- **Correctness blocker:** must fix before the next stage.
- **Architectural duplication:** fix before building another layer on top.
- **Test gap:** add a regression test together with the fix.
- **Compatibility concern:** reconcile against explicit user decisions.
- **Out of scope:** document and exclude.
- **Unrelated pre-existing failure:** verify independently and do not silently modify it.

A reviewer can be wrong when a general compatibility preference conflicts with an explicit user decision. The parent agent should resolve that conflict rather than blindly forwarding every suggestion.

## Stage transitions

Start a new implementation subagent when moving to a distinct architectural stage. The new agent must be brought up to speed with:

- prior design documents;
- the current full diff;
- Stage 1 invariants;
- accepted intentional deviations;
- APIs it must build upon;
- behaviors it must not regress.

Do not use a new agent merely for a small follow-up fix within the same stage. Context retention is valuable there.

## Final verification

Use a fresh independent reviewer after all implementation stages.

Require it to:

- read all specifications and changed files;
- inspect the complete diff;
- audit scope exclusions;
- run focused tests;
- run the broad suite;
- compile changed source;
- run diff checks;
- verify no unrelated files changed;
- report `READY` or `NOT READY`;
- list only remaining actionable findings;
- state exact intentional deviations.

The parent should then independently inspect:

- `git status`;
- changed files;
- diff statistics;
- `git diff --check`;
- any claimed unrelated test failure.

## Reporting completion

The final report should distinguish:

- what was implemented;
- what was intentionally excluded;
- verification results;
- known unrelated failures;
- intentional deviations;
- whether a commit was created.

Do not say “fully tested” when the full suite is not green. Report the exact failing test and why it is considered unrelated.

## Common failure modes

### One-shot implementation

A subagent says tests pass, but architecture is wrong or the new API is unused.

**Correction:** require an independent review of call paths and abstraction reuse.

### Tests-only abstraction

The proposed generic API has no production caller.

**Correction:** either migrate production to it or remove it until a real consumer exists.

### Implementation and review by the same context only

The implementer rationalizes its own choices and misses edge cases.

**Correction:** use a separate read-only reviewer with adversarial instructions.

### Separate replay and transaction state machines

Both appear correct on normal events but diverge on malformed history or rewind.

**Correction:** share a pure transition engine and test state equivalence.

### Optional authoritative state

A persistence API accepts omitted or stale state and commits data that replay later rejects.

**Correction:** require state ownership, reconstruct authoritative state transactionally, and compare before mutation.

### Scope creep to unrelated tests

A failing unrelated test is changed during feature work without establishing its contract.

**Correction:** reproduce it independently, leave it unchanged, and report it separately.

### Premature readiness

Focused tests pass, but review findings remain.

**Correction:** continue the implement-review-re-review loop until a fresh final reviewer reports no actionable findings.
