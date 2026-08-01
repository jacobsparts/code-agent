"""Tests for OverlaySubagent and overlay runtime."""

import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock
import signal
import threading
import subprocess
import sys

from code_agent.overlay_subagent import (
    OverlaySubagent,
    OverlaySubagentResponse,
    OverlayRuntimeError,
    SubmittedFile,
    materialize_submitted_files,
    normalize_submitted_path,
    PathValidationError,
    ApplyConflict,
    set_parent_death_signal,
    start_parent_liveness_monitor,
)


def test_capability_probe_reports_missing_requirements(monkeypatch):
    from code_agent.overlay_subagent import require_overlay_capabilities

    monkeypatch.setattr(
        "code_agent.overlay_subagent.overlay_capability_diagnostics",
        lambda: {
            "platform": "Linux",
            "procfs": True,
            "user_namespace_api": True,
            "copy_command": "/usr/bin/cp",
            "unprivileged_userns_clone": False,
        },
    )
    with pytest.raises(OverlayRuntimeError, match="unprivileged_userns_clone"):
        require_overlay_capabilities()


def test_setup_overlay_worker_captures_ids_before_unshare(monkeypatch, tmp_path):
    from code_agent.overlay_subagent import setup_overlay_worker

    calls = []
    ids = iter((1000, 1000, 65534, 65534))
    monkeypatch.setattr("code_agent.overlay_subagent.os.getuid", lambda: next(ids))
    monkeypatch.setattr("code_agent.overlay_subagent.os.getgid", lambda: next(ids))
    monkeypatch.setattr(
        "code_agent.overlay_subagent.unshare_user_and_mount_namespaces",
        lambda runtime_id: calls.append(("unshare", runtime_id)),
    )
    monkeypatch.setattr(
        "code_agent.overlay_subagent.configure_id_maps",
        lambda uid, gid, runtime_id: calls.append(("maps", uid, gid, runtime_id)),
    )
    monkeypatch.setattr(
        "code_agent.overlay_subagent.make_mounts_private",
        lambda runtime_id: calls.append(("private", runtime_id)),
    )
    monkeypatch.setattr(
        "code_agent.overlay_subagent.bind_mount",
        lambda source, target, runtime_id: calls.append(("bind", source, target, runtime_id)),
    )
    monkeypatch.setattr(
        "code_agent.overlay_subagent.mount_overlay",
        lambda lowers, upper, work, home, runtime_id: calls.append(
            ("overlay", lowers, upper, work, home, runtime_id)
        ),
    )
    monkeypatch.setattr(
        "code_agent.overlay_subagent.mount_lower_view",
        lambda lowers, target, runtime_id: calls.append(
            ("lower_view", lowers, target, runtime_id)
        ),
    )
    monkeypatch.setattr(
        "code_agent.overlay_subagent.set_no_new_privs",
        lambda runtime_id: calls.append(("nnp", runtime_id)),
    )
    monkeypatch.setattr("code_agent.overlay_subagent.os.chdir", lambda path: None)

    setup_overlay_worker({
        "runtime_id": "runtime",
        "home": str(tmp_path / "home"),
        "lower_sources": [str(tmp_path / "lower")],
        "bind_initial_lower": True,
        "initial_lower_source": str(tmp_path / "source"),
        "lower_home": str(tmp_path / "lower"),
        "lower_view": str(tmp_path / "lower-view"),
        "upper_dir": str(tmp_path / "upper"),
        "work_dir": str(tmp_path / "work"),
        "project_cwd": str(tmp_path / "home" / "project"),
        "unshare": True,
    })

    assert ("maps", 1000, 1000, "runtime") in calls


def test_nested_setup_detaches_inherited_home_before_mounting_chain(monkeypatch, tmp_path):
    from code_agent.overlay_subagent import setup_overlay_worker

    calls = []
    monkeypatch.setattr(
        "code_agent.overlay_subagent.unshare_mount_namespace",
        lambda runtime_id: calls.append(("unshare", runtime_id)),
    )
    monkeypatch.setattr(
        "code_agent.overlay_subagent.make_mounts_private",
        lambda runtime_id: calls.append(("private", runtime_id)),
    )
    monkeypatch.setattr(
        "code_agent.overlay_subagent.unmount_overlay",
        lambda target, runtime_id: calls.append(("unmount", target, runtime_id)),
    )
    monkeypatch.setattr(
        "code_agent.overlay_subagent.mount_lower_view",
        lambda lowers, target, runtime_id: calls.append(
            ("lower_view", lowers, target, runtime_id)
        ),
    )
    monkeypatch.setattr(
        "code_agent.overlay_subagent.mount_overlay",
        lambda lowers, upper, work, target, runtime_id: calls.append(
            ("overlay", lowers, upper, work, target, runtime_id)
        ),
    )
    monkeypatch.setattr(
        "code_agent.overlay_subagent.set_no_new_privs",
        lambda runtime_id: calls.append(("nnp", runtime_id)),
    )
    monkeypatch.setattr("code_agent.overlay_subagent.os.chdir", lambda path: None)

    home = tmp_path / "home"
    lower_home = tmp_path / "lower-home"
    home.mkdir()
    lower_home.mkdir()
    lowers = [tmp_path / "upper", tmp_path / "base"]
    for path in lowers:
        path.mkdir()

    setup_overlay_worker({
        "runtime_id": "nested",
        "home": str(home),
        "lower_sources": [str(path) for path in lowers],
        "bind_initial_lower": False,
        "lower_home": str(lower_home),
        "lower_view": str(tmp_path / "lower-view"),
        "inherited_lower_view": str(tmp_path / "parent-lower-view"),
        "upper_dir": str(tmp_path / "child-upper"),
        "work_dir": str(tmp_path / "child-work"),
        "project_cwd": str(home / "project"),
        "unshare": False,
    })

    names = [item[0] for item in calls]
    assert names[:6] == [
        "unshare",
        "private",
        "unmount",
        "unmount",
        "lower_view",
        "overlay",
    ]




def test_overlay_subagent_initialization_requires_project_under_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    (project / "file.txt").write_text("hello")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    overlay = OverlaySubagent(cwd=str(project))
    try:
        assert overlay.project_root == project.resolve()
        assert overlay.runtime_dir.exists()
        assert overlay.upper_dir.exists()
        assert overlay.work_dir.exists()
        assert overlay.lower_home_dir.exists()
        assert overlay.lower_view_dir.exists()
        assert overlay.merged_dir == home.resolve()
        assert overlay.lower_dir == overlay.lower_view_dir / "project"
        assert overlay.runtime_config["lower_project_root"] == str(overlay.lower_dir)
        assert overlay.runtime_config["lower_sources"] == [
            str(overlay.lower_home_dir)
        ]
        assert overlay.runtime_config["initial_lower_source"] == str(home.resolve())
        assert overlay.runtime_config["session_db"] == str(
            overlay.runtime_dir / "sessions.db"
        )
        assert overlay.runtime_config["unshare"] is True
    finally:
        overlay.close()
        assert not overlay.runtime_dir.exists()


def test_overlay_subagent_response_tracks_overlay_protocol():
    agent = MagicMock()
    resp = OverlaySubagentResponse(agent)
    file_art = SubmittedFile(
        path="foo.txt",
        operation="create",
        before=None,
        after=b"foo\n",
        before_mode=None,
        after_mode=0o644,
    )
    resp._result = "Done"
    resp._progress.append("working...")
    resp._turns = 2
    resp._files = {"foo.txt": file_art}
    resp._done = True

    assert resp.done is True
    assert resp.result == "Done"
    assert resp.progress == ["working..."]
    assert resp.turns == 2
    assert resp.is_error is False
    assert resp.files == {"foo.txt": file_art}
    assert "+++ foo.txt" in resp.diff()


def test_nested_overlay_subagent_uses_sealed_parent_view(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    parent = OverlaySubagent(cwd=str(project))
    sealed_upper = parent.runtime_dir / "sealed-upper"
    sealed_base = parent.runtime_dir / "sealed-base"
    sealed_upper.mkdir()
    sealed_base.mkdir()
    monkeypatch.setattr(parent, "_seal", lambda: [sealed_upper, sealed_base])

    try:
        child = parent.create_child()
        assert child.parent_overlay is parent
        assert child in parent._children
        assert child.runtime_config["lower_sources"] == [
            str(sealed_upper),
            str(sealed_base),
        ]
        assert child.runtime_config["unshare"] is False
        child.close()
        assert child not in parent._children
    finally:
        parent.close()


def test_seal_rolls_parent_to_fresh_upper_each_time(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    parent = OverlaySubagent(cwd=str(project))
    calls = []
    monkeypatch.setattr(
        parent,
        "_control",
        lambda message, expected: calls.append((message, expected)) or {
            "lower_sources": message[1]["lower_sources"]
        },
    )
    try:
        initial_upper = parent.upper_dir
        initial_lower = parent.lower_home_dir
        (initial_upper / "project").mkdir()
        (initial_upper / "project" / "seed.txt").write_text("seed\\n")
        first = parent._seal()
        first_upper = parent.upper_dir
        assert (first[0] / "project" / "seed.txt").read_text() == "seed\\n"
        assert not initial_upper.exists()
        second = parent._seal()
        assert first == [
            parent.runtime_dir / "sealed-1",
            initial_lower,
        ]
        assert second == [
            parent.runtime_dir / "sealed-2",
            parent.runtime_dir / "sealed-1",
            initial_lower,
        ]
        assert first_upper == parent.runtime_dir / "upper-1"
        assert parent.upper_dir == parent.runtime_dir / "upper-2"
        assert [expected for _, expected in calls] == ["sealed", "sealed"]
    finally:
        parent.close()


def test_seal_failure_rolls_back_paths_and_generation(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    parent = OverlaySubagent(cwd=str(project))
    original_upper = parent.upper_dir
    original_work = parent.work_dir
    original_lowers = list(parent.lower_sources)
    monkeypatch.setattr(
        "code_agent.overlay_subagent.clone_overlay_layer",
        MagicMock(side_effect=OverlayRuntimeError("clone", "failed")),
    )
    try:
        with pytest.raises(OverlayRuntimeError, match="failed"):
            parent._seal()

        assert parent._seal_generation == 0
        assert parent.upper_dir == original_upper
        assert parent.work_dir == original_work
        assert parent.lower_sources == original_lowers
        assert original_upper.exists()
        assert not (parent.runtime_dir / "sealed-1").exists()
        assert not (parent.runtime_dir / "upper-1").exists()
        assert not (parent.runtime_dir / "work-1").exists()
    finally:
        parent.close()


def test_seal_control_failure_preserves_possible_mounted_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    parent = OverlaySubagent(cwd=str(project))
    monkeypatch.setattr(
        parent,
        "_control",
        MagicMock(side_effect=TimeoutError("lost acknowledgement")),
    )
    try:
        with pytest.raises(TimeoutError, match="lost acknowledgement"):
            parent._seal()

        assert parent._seal_generation == 0
        assert (parent.runtime_dir / "sealed-1").exists()
        assert (parent.runtime_dir / "upper-1").exists()
        assert (parent.runtime_dir / "work-1").exists()
    finally:
        parent.close()


def test_child_constructor_failure_removes_runtime(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    parent = OverlaySubagent(cwd=str(project))
    monkeypatch.setattr(
        parent,
        "_seal",
        MagicMock(side_effect=OverlayRuntimeError("seal", "failed")),
    )
    before = set(Path("/tmp").glob("overlay_subagent_*"))
    try:
        with pytest.raises(OverlayRuntimeError, match="failed"):
            parent.create_child()
        assert set(Path("/tmp").glob("overlay_subagent_*")) == before
    finally:
        parent.close()


def test_snapshot_capacity_enforces_byte_and_inode_limits(tmp_path):
    from code_agent.overlay_subagent import validate_snapshot_capacity

    source = tmp_path / "source"
    source.mkdir()
    (source / "a").write_bytes(b"1234")
    (source / "b").write_bytes(b"56")

    with pytest.raises(OverlayRuntimeError, match="byte limit"):
        validate_snapshot_capacity(
            source,
            tmp_path,
            byte_limit=5,
            inode_limit=None,
            min_free_bytes=0,
        )
    with pytest.raises(OverlayRuntimeError, match="inode limit"):
        validate_snapshot_capacity(
            source,
            tmp_path,
            byte_limit=None,
            inode_limit=1,
            min_free_bytes=0,
        )


def test_seal_quiescence_rejects_unmanaged_children(monkeypatch):
    from code_agent.overlay_subagent import assert_overlay_quiescent

    children = {
        None: [11, 12],
        11: [],
        12: [13],
        13: [],
    }
    monkeypatch.setattr(
        "code_agent.overlay_subagent.direct_child_pids",
        lambda pid=None: children[pid],
    )
    with pytest.raises(OverlayRuntimeError, match="12"):
        assert_overlay_quiescent([11], [], "runtime")
    with pytest.raises(OverlayRuntimeError, match="13"):
        assert_overlay_quiescent([11], [12], "runtime")

    children[12] = []
    assert_overlay_quiescent([11], [12], "runtime")


def test_clone_timeout_has_overlay_diagnostic(tmp_path, monkeypatch):
    from code_agent.overlay_subagent import clone_overlay_layer

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    monkeypatch.setattr(
        "subprocess.run",
        MagicMock(side_effect=subprocess.TimeoutExpired("cp", 0.1)),
    )

    with pytest.raises(OverlayRuntimeError, match="exceeded timeout"):
        clone_overlay_layer(source, target, "runtime", timeout=0.1)


def test_bounded_pipe_capture_drains_and_returns_tail():
    from code_agent.overlay_subagent import _BoundedPipeCapture

    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb")
    capture = _BoundedPipeCapture(stream, limit=10)
    os.write(write_fd, b"prefix-" + b"x" * 20 + b"-tail")
    os.close(write_fd)
    capture.join()

    assert capture.text() == (
        "[truncated to trailing output]\nxxxxx-tail"
    )


def test_overlay_runtime_error_formatting():
    err = OverlayRuntimeError(
        op="mount_overlay",
        message="permission denied",
        errno_val=1,
        paths={"lower": "/lower", "upper": "/upper"},
        mount_options="lowerdir=/lower,upperdir=/upper",
        runtime_id="sub123",
        mountinfo="1 2 3 / /mount rw",
    )
    text = str(err)
    assert "mount_overlay" in text
    assert "permission denied" in text
    assert "errno: 1" in text
    assert "runtime_id: sub123" in text
    assert "paths: lower=/lower, upper=/upper" in text
    assert "mountinfo excerpt:" in text


def test_set_parent_death_signal_configures_sigterm_and_checks_parent(monkeypatch):
    libc = MagicMock()
    libc.prctl.return_value = 0
    monkeypatch.setattr("code_agent.overlay_subagent._get_libc", lambda: libc)
    monkeypatch.setattr("code_agent.overlay_subagent.os.getppid", lambda: 123)

    set_parent_death_signal(123, "runtime")

    libc.prctl.assert_called_once_with(1, signal.SIGTERM, 0, 0, 0)


def test_set_parent_death_signal_rejects_parent_race(monkeypatch):
    libc = MagicMock()
    libc.prctl.return_value = 0
    monkeypatch.setattr("code_agent.overlay_subagent._get_libc", lambda: libc)
    monkeypatch.setattr("code_agent.overlay_subagent.os.getppid", lambda: 999)

    with pytest.raises(OverlayRuntimeError, match="owner exited"):
        set_parent_death_signal(123, "runtime")


def test_liveness_pipe_eof_runs_cleanup_and_exit(monkeypatch):
    read_fd, write_fd = os.pipe()
    cleaned = threading.Event()
    exited = threading.Event()
    monkeypatch.setattr(
        "code_agent.overlay_subagent.set_parent_death_signal",
        lambda expected_parent_pid, runtime_id="": None,
    )

    start_parent_liveness_monitor(
        read_fd,
        os.getpid(),
        cleaned.set,
        runtime_id="runtime",
        exit_func=lambda _code: exited.set(),
    )
    os.close(write_fd)

    assert cleaned.wait(1)
    assert exited.wait(1)


def test_liveness_monitor_can_reuse_preconfigured_parent_signal(monkeypatch):
    read_fd, write_fd = os.pipe()
    cleaned = threading.Event()
    exited = threading.Event()
    configure = MagicMock()
    monkeypatch.setattr(
        "code_agent.overlay_subagent.set_parent_death_signal",
        configure,
    )

    start_parent_liveness_monitor(
        read_fd,
        os.getpid(),
        cleaned.set,
        runtime_id="runtime",
        exit_func=lambda _code: exited.set(),
        configure_death_signal=False,
    )
    os.close(write_fd)

    assert cleaned.wait(1)
    assert exited.wait(1)
    configure.assert_not_called()


def test_liveness_pipe_eof_terminates_real_worker(tmp_path):
    read_fd, write_fd = os.pipe()
    marker = tmp_path / "cleaned"
    code = """
import os
import time
from pathlib import Path
from code_agent.overlay_subagent import start_parent_liveness_monitor

fd = int(os.environ["TEST_LIVENESS_FD"])
owner = int(os.environ["TEST_OWNER_PID"])
marker = Path(os.environ["TEST_MARKER"])
start_parent_liveness_monitor(
    fd,
    owner,
    lambda: marker.write_text("cleaned"),
    runtime_id="test-worker",
)
print("ready", flush=True)
while True:
    time.sleep(1)
"""
    env = os.environ.copy()
    env["TEST_LIVENESS_FD"] = str(read_fd)
    env["TEST_OWNER_PID"] = str(os.getpid())
    env["TEST_MARKER"] = str(marker)
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(read_fd,),
        env=env,
    )
    os.close(read_fd)
    try:
        assert proc.stdout.readline().strip() == "ready"
        os.close(write_fd)
        write_fd = -1
        assert proc.wait(timeout=5) == 1
        assert marker.read_text() == "cleaned"
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_close_removes_overlay_work_directory_with_restricted_mode(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    overlay = OverlaySubagent(cwd=str(project))
    work_child = overlay.work_dir / "work"
    work_child.mkdir()
    overlay.work_dir.chmod(0o700)
    work_child.chmod(0)

    runtime_dir = overlay.runtime_dir
    overlay.close()

    assert not runtime_dir.exists()


def test_overlay_subagent_context_manager(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    with OverlaySubagent(cwd=str(project)) as worker:
        assert worker.runtime_dir.exists()
        rdir = worker.runtime_dir

    assert not rdir.exists()
