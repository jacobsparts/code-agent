"""Focused unit tests for overlay subagent Phase 1 artifact model."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import MappingProxyType

import pytest

from code_agent.overlay_subagent import (
    ApplyConflict,
    OverlaySubagentResponseBase,
    PathValidationError,
    SubmittedFile,
    apply_submitted_files,
    classify_operation,
    normalize_submitted_path,
    response_diff,
)


def test_normalize_rejects_absolute_and_traversal():
    with pytest.raises(PathValidationError):
        normalize_submitted_path("/etc/passwd")
    with pytest.raises(PathValidationError):
        normalize_submitted_path("../secret")
    with pytest.raises(PathValidationError):
        normalize_submitted_path("a/../../b")
    with pytest.raises(PathValidationError):
        normalize_submitted_path("")
    assert normalize_submitted_path("./src/a.py") == "src/a.py"
    assert normalize_submitted_path("src//a.py") == "src/a.py"


def test_submission_paths_reject_bare_string():
    from code_agent.overlay_subagent import normalize_submission_paths

    with pytest.raises(PathValidationError, match="not a string"):
        normalize_submission_paths("foo.txt")


def test_classify_operation_and_missing_both_sides():
    assert classify_operation(before_exists=False, after_exists=True) == "create"
    assert classify_operation(before_exists=True, after_exists=False) == "delete"
    assert classify_operation(before_exists=True, after_exists=True) == "modify"
    with pytest.raises(PathValidationError):
        classify_operation(before_exists=False, after_exists=False)


def test_create_modify_delete_text_diff_and_apply(tmp_path: Path):
    created = SubmittedFile(
        path="hello.txt",
        operation="create",
        before=None,
        after=b"hello\n",
        before_mode=None,
        after_mode=0o644,
    )
    assert "--- /dev/null" in created.diff()
    assert "+++ hello.txt" in created.diff()
    created.apply(root=tmp_path)
    assert (tmp_path / "hello.txt").read_text() == "hello\n"

    modified = SubmittedFile(
        path="hello.txt",
        operation="modify",
        before=b"hello\n",
        after=b"hello world\n",
        before_mode=0o644,
        after_mode=0o644,
    )
    diff = modified.diff()
    assert "--- hello.txt" in diff
    assert "+++ hello.txt" in diff
    assert "-hello" in diff or "-hello\n" in diff
    modified.apply(root=tmp_path)
    assert (tmp_path / "hello.txt").read_text() == "hello world\n"

    deleted = SubmittedFile(
        path="hello.txt",
        operation="delete",
        before=b"hello world\n",
        after=None,
        before_mode=0o644,
        after_mode=None,
    )
    assert "+++ /dev/null" in deleted.diff()
    deleted.apply(root=tmp_path)
    assert not (tmp_path / "hello.txt").exists()


def test_binary_diff_summary():
    artifact = SubmittedFile(
        path="blob.bin",
        operation="create",
        before=None,
        after=b"\x00\x01\x02",
        before_mode=None,
        after_mode=0o644,
    )
    text = artifact.diff()
    assert "Binary files" in text
    assert "before_size=0 after_size=3" in text


def test_symlink_create_modify_delete_and_apply(tmp_path: Path):
    created = SubmittedFile(
        path="link",
        operation="create",
        before=None,
        after=None,
        before_mode=None,
        after_mode=0o777,
        after_symlink_target="target-a",
    )
    assert "symlink create link" in created.diff()
    created.apply(root=tmp_path)
    assert (tmp_path / "link").is_symlink()
    assert os.readlink(tmp_path / "link") == "target-a"

    modified = SubmittedFile(
        path="link",
        operation="modify",
        before=None,
        after=None,
        before_mode=0o777,
        after_mode=0o777,
        before_symlink_target="target-a",
        after_symlink_target="target-b",
    )
    assert "symlink modify link" in modified.diff()
    modified.apply(root=tmp_path)
    assert os.readlink(tmp_path / "link") == "target-b"

    deleted = SubmittedFile(
        path="link",
        operation="delete",
        before=None,
        after=None,
        before_mode=0o777,
        after_mode=None,
        before_symlink_target="target-b",
    )
    deleted.apply(root=tmp_path)
    assert not (tmp_path / "link").exists() and not (tmp_path / "link").is_symlink()


def test_executable_bit_mode_change(tmp_path: Path):
    path = tmp_path / "tool.sh"
    path.write_text("#!/bin/sh\n")
    path.chmod(0o644)
    artifact = SubmittedFile(
        path="tool.sh",
        operation="modify",
        before=b"#!/bin/sh\n",
        after=b"#!/bin/sh\n",
        before_mode=0o644,
        after_mode=0o755,
    )
    assert "mode change" in artifact.diff()
    artifact.apply(root=tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o755


def test_apply_uses_secure_unique_temporary_file(tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n")
    predictable = tmp_path / "target.txt.overlay-tmp"
    predictable.symlink_to(outside)

    artifact = SubmittedFile(
        path="target.txt",
        operation="create",
        before=None,
        after=b"target\n",
        before_mode=None,
        after_mode=0o600,
    )
    artifact.apply(root=tmp_path)

    assert (tmp_path / "target.txt").read_bytes() == b"target\n"
    assert stat.S_IMODE((tmp_path / "target.txt").stat().st_mode) == 0o600
    assert outside.read_text() == "outside\n"
    assert predictable.is_symlink()


def test_apply_preserves_mode_zero(tmp_path: Path):
    artifact = SubmittedFile(
        path="locked.txt",
        operation="create",
        before=None,
        after=b"locked\n",
        before_mode=None,
        after_mode=0,
    )
    artifact.apply(root=tmp_path)

    assert stat.S_IMODE((tmp_path / "locked.txt").stat().st_mode) == 0


def test_apply_conflict_when_destination_changed(tmp_path: Path):
    (tmp_path / "a.txt").write_text("new\n")
    artifact = SubmittedFile(
        path="a.txt",
        operation="modify",
        before=b"old\n",
        after=b"newer\n",
        before_mode=0o644,
        after_mode=0o644,
    )
    with pytest.raises(ApplyConflict, match="destination no longer matches"):
        artifact.apply(root=tmp_path)


def test_apply_conflict_preflight_prevents_partial_writes(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("ok\n")
    (tmp_path / "bad.txt").write_text("changed\n")
    files = {
        "ok.txt": SubmittedFile(
            path="ok.txt",
            operation="modify",
            before=b"ok\n",
            after=b"ok2\n",
            before_mode=0o644,
            after_mode=0o644,
        ),
        "bad.txt": SubmittedFile(
            path="bad.txt",
            operation="modify",
            before=b"orig\n",
            after=b"bad2\n",
            before_mode=0o644,
            after_mode=0o644,
        ),
    }
    with pytest.raises(ApplyConflict) as excinfo:
        apply_submitted_files(files, root=tmp_path)
    assert len(excinfo.value.conflicts) == 1
    assert (tmp_path / "ok.txt").read_text() == "ok\n"
    assert (tmp_path / "bad.txt").read_text() == "changed\n"


def test_response_diff_sorted_and_subset_apply(tmp_path: Path):
    files = {
        "b.txt": SubmittedFile(
            path="b.txt",
            operation="create",
            before=None,
            after=b"b\n",
            before_mode=None,
            after_mode=0o644,
        ),
        "a.txt": SubmittedFile(
            path="a.txt",
            operation="create",
            before=None,
            after=b"a\n",
            before_mode=None,
            after_mode=0o644,
        ),
    }
    response = OverlaySubagentResponseBase(result="done", files=files)
    combined = response.diff()
    assert combined.index("a.txt") < combined.index("b.txt")
    response.apply(paths=["a.txt"], root=tmp_path)
    assert (tmp_path / "a.txt").read_text() == "a\n"
    assert not (tmp_path / "b.txt").exists()
    assert response.diff(paths=["b.txt"]).startswith("--- /dev/null")


def test_empty_submission_and_open_text_helpers(tmp_path: Path):
    response = OverlaySubagentResponseBase(result="none", files={})
    assert response.diff() == ""
    response.apply(root=tmp_path)

    artifact = SubmittedFile(
        path="x.txt",
        operation="create",
        before=None,
        after=b"hi",
        before_mode=None,
        after_mode=0o644,
    )
    assert artifact.text() == "hi"
    assert artifact.open().read() == b"hi"
    deleted = SubmittedFile(
        path="x.txt",
        operation="delete",
        before=b"hi",
        after=None,
        before_mode=0o644,
        after_mode=None,
    )
    with pytest.raises(Exception):
        deleted.open()





def test_artifact_payload_rejects_malformed_and_duplicate_entries():
    from code_agent.overlay_subagent import SubmissionError, submitted_files_from_payload

    with pytest.raises(SubmissionError, match="iterable of mappings"):
        submitted_files_from_payload({"path": "a.txt"})
    with pytest.raises(SubmissionError, match="item 0 must be a mapping"):
        submitted_files_from_payload(["a.txt"])

    item = {
        "path": "a.txt",
        "operation": "create",
        "before": None,
        "after": b"a",
        "before_mode": None,
        "after_mode": 0o644,
    }
    with pytest.raises(SubmissionError, match="duplicate artifact payload path"):
        submitted_files_from_payload([item, dict(item)])
def test_response_files_mapping_is_immutable():
    artifact = SubmittedFile(
        path="x.txt",
        operation="create",
        before=None,
        after=b"x",
        before_mode=None,
        after_mode=0o644,
    )
    response = OverlaySubagentResponseBase(result="done", files={"x.txt": artifact})
    assert isinstance(response.files, MappingProxyType)
    with pytest.raises(TypeError):
        response.files["y.txt"] = artifact
