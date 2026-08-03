"""Tests for OverlaySubagent and overlay runtime."""

import gc
import weakref

import pytest
import json
import os
import socket
import shutil
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
        assert overlay.runtime_config["recursive"] is False
    finally:
        overlay.close()
        assert not overlay.runtime_dir.exists()


def test_recursive_worker_code_attaches_skill_and_gates_child_construction():
    from code_agent.overlay_subagent import _build_overlay_worker_code, _overlay_repl_code

    worker_code = _build_overlay_worker_code()
    compile(worker_code, "<overlay-worker>", "exec")
    assert 'if runtime_config["recursive"]:' in worker_code
    assert 'agent.attach_skill("overlay_subagent_worker")' in worker_code
    worker_skill = Path("code_agent/skills/overlay_subagent_worker.md").read_text()
    assert worker_skill.startswith("# Recursive overlay subagent orchestration")
    assert "# Overlay subagents with isolated, reviewable file changes" not in worker_skill
    assert "A child does not inherit your conversation" in worker_skill

    recursive_repl = _overlay_repl_code(True)
    leaf_repl = _overlay_repl_code(False)
    compile(recursive_repl, "<recursive-overlay-repl>", "exec")
    compile(leaf_repl, "<leaf-overlay-repl>", "exec")
    assert "if not True:" in recursive_repl
    assert "if not False:" in leaf_repl
    assert "recursive overlay subagents require OverlaySubagent(recursive=True)" in leaf_repl
    assert "recursive=recursive" in recursive_repl
    assert "def __enter__(" not in recursive_repl
    assert "def __exit__(" not in recursive_repl
    assert "def __del__(self):" in recursive_repl


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


def test_overlay_subagent_requires_explicit_close(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    worker = OverlaySubagent(cwd=str(project))
    assert worker.runtime_dir.exists()
    rdir = worker.runtime_dir
    worker.close()

    assert not rdir.exists()

def test_overlay_subagent_has_no_context_manager_and_destructor_closes(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    worker = OverlaySubagent(cwd=str(project))
    runtime_dir = worker.runtime_dir
    worker_ref = weakref.ref(worker)

    assert not hasattr(worker, "__enter__")
    assert not hasattr(worker, "__exit__")

    del worker
    gc.collect()

    assert worker_ref() is None
    assert not runtime_dir.exists()


@pytest.mark.integration
def test_worker_can_fan_out_to_overlay_children_and_apply_results(tmp_path):
    root = Path(tempfile.mkdtemp(prefix="overlay-fanout-test-", dir=Path.home()))
    home = root / "home"
    project = home / "project"
    config_dir = home / ".code-agent"
    project.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    child_requests = set()
    skill_requests = set()
    child_requests_lock = threading.Lock()
    both_children_started = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            content = payload["messages"][-1]["content"]
            if isinstance(content, str):
                text = content
            else:
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )

            if "orchestrate-two-children" in text:
                if "# Recursive overlay subagent orchestration" in json.dumps(payload):
                    skill_requests.add("orchestrator")
                response_text = (
                    'from code_agent.overlay_subagent import OverlaySubagent\n'
                    'a = OverlaySubagent(recursive=True)\n'
                    'b = OverlaySubagent()\n'
                    'ra = a.send("make-child-a", bg=True)\n'
                    'rb = b.send("make-child-b", bg=True)\n'
                    'ra.wait(60)\n'
                    'rb.wait(60)\n'
                    'assert not ra.is_error, ra.error\n'
                    'assert not rb.is_error, rb.error\n'
                    'ra.apply()\n'
                    'rb.apply()\n'
                    'a.close()\n'
                    'b.close()\n'
                    'emit("fanout complete", release=True, files=["a.txt", "b.txt"])'
                )
            elif "make-grandchild-a" in text:
                response_text = (
                    'from pathlib import Path\n'
                    'Path("a.txt").write_text("A\\n")\n'
                    'emit("A complete", release=True, files=["a.txt"])'
                )
            elif "make-child-a" in text:
                if "# Recursive overlay subagent orchestration" in json.dumps(payload):
                    skill_requests.add("child-a")
                with child_requests_lock:
                    child_requests.add("a")
                    if child_requests == {"a", "b"}:
                        both_children_started.set()
                if not both_children_started.wait(10):
                    raise AssertionError("overlay children did not run concurrently")
                response_text = (
                    'from code_agent.overlay_subagent import OverlaySubagent\n'
                    'grandchild = OverlaySubagent()\n'
                    'response = grandchild.send("make-grandchild-a")\n'
                    'assert not response.is_error, response.error\n'
                    'response.apply()\n'
                    'grandchild.close()\n'
                    'emit("A integrated", release=True, files=["a.txt"])'
                )
            elif "make-child-b" in text:
                with child_requests_lock:
                    child_requests.add("b")
                    if child_requests == {"a", "b"}:
                        both_children_started.set()
                if not both_children_started.wait(10):
                    raise AssertionError("overlay children did not run concurrently")
                response_text = (
                    'from pathlib import Path\n'
                    'Path("b.txt").write_text("B\\n")\n'
                    'emit("B complete", release=True, files=["b.txt"])'
                )
            else:
                response_text = 'emit("unexpected prompt", release=True)'

            body = json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    (config_dir / "config.py").write_text(
        f"""
register_provider(
    "overlaytest",
    host="127.0.0.1",
    path="/v1/chat/completions",
    port={port},
    timeout=30,
    tpm=100000,
    concurrency=10,
    tools=False,
    api_type="completions",
)
register_model(
    "overlaytest",
    "worker",
    aliases="overlay-test",
    model="worker",
    input_cost=0.0,
    output_cost=0.0,
)
code_agent_model = "overlay-test"
"""
    )

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = f"""
from pathlib import Path
from code_agent.overlay_subagent import OverlaySubagent

project = Path({str(project)!r})
agent = OverlaySubagent(
    cwd=str(project),
    model="overlay-test",
    max_turns=10,
    snapshot_min_free_bytes=0,
    recursive=True,
)
try:
    response = agent.send("orchestrate-two-children", timeout=120)
    assert response.done
    assert not response.is_error, response.error
    assert not response.submission_error
    assert sorted(response.files) == ["a.txt", "b.txt"]
    assert response.files["a.txt"].text() == "A\\n"
    assert response.files["b.txt"].text() == "B\\n"
    assert not (project / "a.txt").exists()
    assert not (project / "b.txt").exists()
    response.apply()
    assert (project / "a.txt").read_text() == "A\\n"
    assert (project / "b.txt").read_text() == "B\\n"
finally:
    agent.close()
"""
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["OVERLAYTEST_API_KEY"] = "dummy"
    env["CODE_AGENT_SESSION_DB"] = str(tmp_path / "top-sessions.db")
    env["CODE_AGENT_CLI_HISTORY_DB"] = str(tmp_path / "history.db")
    env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd()), *sys.path])
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        shutil.rmtree(root, ignore_errors=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert child_requests == {"a", "b"}
    assert skill_requests == {"orchestrator", "child-a"}
