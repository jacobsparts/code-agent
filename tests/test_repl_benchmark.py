
import json
import os
import socket
import subprocess
import sys
import threading
import warnings

import pytest
from pathlib import Path
from textwrap import dedent
from http.server import BaseHTTPRequestHandler, HTTPServer

from code_agent.repl_benchmark import (
    BenchmarkTask,
    REPLBenchmarkRunner,
    build_code_agent_test_env,
    discover_tasks,
    format_summary,
    register_task,
    run_pty_session,
    strip_ansi,
)
from code_agent.repl_benchmark.code_agent_harness import strip_events
from code_agent.repl_events import ReplEvent
from code_agent.repl_benchmark.core import BenchmarkTaskContext, checker_expected_int, default_checker
from code_agent.repl_benchmark.registry import task_registry


class FakeUsageTracker:
    def __init__(self):
        self.history = []

    def _normalize(self, model_name, usage):
        return usage


class FakeLLMClient:
    def __init__(self):
        self.usage_tracker = FakeUsageTracker()


class FakeAgent:
    model = "fake-model"

    def __init__(self):
        self.llm_client = FakeLLMClient()
        self.messages = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def usermsg(self, msg):
        self.messages.append(msg)

    def run_loop(self, max_turns=10, max_syntax_retries=3):
        self.on_repl_execute(None)
        self._on_assistant_message_committed({"content": 'emit("55", release=True)'})
        self.on_statement_events([
            ReplEvent(kind="statement_started", data={"echo": "x = 1\n"}),
            ReplEvent(kind="statement_started", data={"echo": 'emit("55", release=True)\n'}),
        ])
        self.llm_client.usage_tracker.history.append((
            self.model,
            {
                "prompt_tokens": 10,
                "cached_tokens": 0,
                "completion_tokens": 5,
                "reasoning_tokens": 0,
                "cost": 0.01,
            },
        ))
        return "55"


class FakeBadAgent(FakeAgent):
    def run_loop(self, max_turns=10, max_syntax_retries=3):
        self.on_repl_execute(None)
        self.on_repl_execute(None)
        self.on_retry("syntax", 1)
        self._on_assistant_message_committed({"content": 'bash("python -c \'print(55)\'")'})
        self.on_repl_event(ReplEvent(kind="progress", text="working"))
        self.on_statement_events([
            ReplEvent(
                kind="statement_started",
                data={"echo": 'bash("python -c \'print(55)\'")\n'},
            ),
        ])
        self.llm_client.usage_tracker.history.append((
            self.model,
            {
                "prompt_tokens": 12,
                "cached_tokens": 0,
                "completion_tokens": 7,
                "reasoning_tokens": 0,
                "cost": 0.02,
            },
        ))
        return "55"


def test_repl_child_processes_inherit_devnull_stdin():
    from code_agent.tools.subrepl import SubREPL

    repl = SubREPL(echo=False)
    try:
        output = repl.execute(
            "import subprocess\n"
            "r = subprocess.run(['cat'], capture_output=True, text=True, timeout=2)\n"
            "print(r.returncode, repr(r.stdout), repr(r.stderr))",
            timeout=3,
            hard_timeout=True,
        )
    finally:
        repl.close()

    assert output.strip() == "0 '' ''"


def test_repl_python_dash_does_not_hang_on_inherited_stdin():
    from code_agent.tools.subrepl import SubREPL

    repl = SubREPL(echo=False)
    try:
        output = repl.execute(
            "import subprocess\n"
            "r = subprocess.run(['python3', '-'], capture_output=True, text=True, timeout=2)\n"
            "print(r.returncode)",
            timeout=3,
            hard_timeout=True,
        )
    finally:
        repl.close()

    assert output.strip() == "0"


def test_repl_explicit_subprocess_input_still_works():
    from code_agent.tools.subrepl import SubREPL

    repl = SubREPL(echo=False)
    try:
        output = repl.execute(
            "import subprocess\n"
            "r = subprocess.run(['python3', '-'], input=\"print('ok')\\n\", capture_output=True, text=True, timeout=2)\n"
            "print(r.returncode, r.stdout.strip())",
            timeout=3,
            hard_timeout=True,
        )
    finally:
        repl.close()

    assert output.strip() == "0 ok"


def test_repl_explicit_stdin_file_still_works(tmp_path):
    from code_agent.tools.subrepl import SubREPL

    path = tmp_path / "input.txt"
    path.write_text("hello")
    repl = SubREPL(echo=False)
    try:
        output = repl.execute(
            f"import subprocess\n"
            f"with open({str(path)!r}) as f:\n"
            f"    r = subprocess.run(['cat'], stdin=f, capture_output=True, text=True, timeout=2)\n"
            f"print(r.stdout)",
            timeout=3,
            hard_timeout=True,
        )
    finally:
        repl.close()

    assert output == "hello\n"


def test_repl_fd0_is_devnull_during_execution_and_restored_between_turns():
    from code_agent.tools.subrepl import SubREPL

    repl = SubREPL(echo=False)
    try:
        fd0 = repl.execute(
            "import os\n"
            "print(os.readlink('/proc/self/fd/0'))",
            timeout=3,
            hard_timeout=True,
        ).strip()
        second = repl.execute("print('still works')", timeout=3, hard_timeout=True).strip()
    finally:
        repl.close()

    assert fd0 in {"/dev/null", os.devnull}
    assert second == "still works"


def test_stdio_subprocess_transport_basic_subrepl():
    from code_agent.tools.subrepl import _worker_main
    from code_agent.tools.transports import StdioSubprocessTransport

    transport = StdioSubprocessTransport(target=_worker_main, args=(os.getcwd(),), queue_count=2)
    try:
        transport.start()
        cmd_queue, output_queue = transport.queues
        cmd_queue.put("x = 40 + 2\nprint(x)\nx")

        items = []
        while True:
            item = output_queue.get(timeout=5)
            items.append(item)
            if item[0] == "done":
                break
    finally:
        transport.close()

    assert items == [
        ("output", "42"),
        ("output", "\n"),
        ("output", "42"),
        ("output", "\n"),
        ("done", False),
    ]


def test_stdio_subprocess_transport_tool_repl_emit_ack():
    from queue import Empty

    from code_agent.repl_agent import ToolREPL
    from code_agent.tools.transports import StdioSubprocessTransport

    repl = ToolREPL(echo=False, transport=StdioSubprocessTransport)
    try:
        repl.inject_builtins()
        repl._running = True
        repl._cmd_seq += 1
        repl._cmd_queue.put((repl._cmd_seq, 'emit("hi", release=False)'))

        acked = False
        seen = []
        while True:
            try:
                req = repl.poll_tool_request(timeout=0.05)
            except Empty:
                req = None
            if req:
                seen.append(("request", req))
                repl.send_ack(req.get("request_id"))
                acked = True

            try:
                msg = repl._output_queue.get(timeout=0.05)
            except Empty:
                continue
            seen.append(("output", msg))
            if msg[0] == "done":
                break
    finally:
        repl.close()

    assert acked is True
    assert ("output", ("progress", "hi\n")) in seen
    assert ("request", {"tool": "__emit__", "args": {"value": "hi", "release": False}, "request_id": 1}) in seen
    assert seen[-1] == ("output", ("done", (1, False)))


def test_tool_repl_validation_silences_parent_warning_and_captures_execution_warning(monkeypatch):
    from code_agent.repl_agent import REPLMixin, ToolREPL
    from code_agent.tools.transports import StdioSubprocessTransport

    code = 'value = """\\s"""'
    monkeypatch.setenv("PYTHONWARNINGS", "always")
    repl = ToolREPL(echo=False, transport=StdioSubprocessTransport)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output, pure_syntax_error, output_chunks, corrected_code = (
                REPLMixin()._execute_with_tool_handling(repl, code)
            )
    finally:
        repl.close()

    assert [
        warning for warning in caught
        if isinstance(warning.message, (SyntaxWarning, DeprecationWarning))
    ] == []
    assert pure_syntax_error is False
    assert corrected_code == code
    assert output.count("invalid escape sequence") == 1
    assert sum(event.text.count("invalid escape sequence") for event in output_chunks) == 1


def test_tool_repl_worker_imports_from_current_directory(tmp_path, monkeypatch):
    from queue import Empty

    from code_agent.repl_agent import ToolREPL
    from code_agent.tools.transports import StdioSubprocessTransport

    (tmp_path / "worker_local_module.py").write_text("VALUE = 123\n")
    monkeypatch.chdir(tmp_path)

    repl = ToolREPL(echo=False, transport=StdioSubprocessTransport)
    try:
        repl.inject_builtins()
        repl._running = True
        repl._cmd_seq += 1
        repl._cmd_queue.put((repl._cmd_seq, "import worker_local_module\nprint(worker_local_module.VALUE)"))

        items = []
        while True:
            try:
                msg = repl._output_queue.get(timeout=5)
            except Empty:
                raise AssertionError("Timed out waiting for ToolREPL output")
            items.append(msg)
            if msg[0] == "done":
                break
    finally:
        repl.close()

    assert "".join(data for kind, data in items if kind == "output") == "123\n"
    assert items[-1] == ("done", (1, False))


def test_tool_repl_does_not_pass_local_cwd_to_ssh_transport():
    from queue import Queue

    from code_agent.repl_agent import ToolREPL
    from code_agent.tools.transports import SSHSubprocessTransport

    class FakeSSHTransport(SSHSubprocessTransport):
        def __init__(self, target, args=(), queue_count=2, maxsize=1):
            self._target = target
            self._args = args
            self._queue_count = queue_count
            self._maxsize = maxsize
            self.queues = ()
            self.worker = object()
            self.started = False

        def start(self):
            self.started = True
            self.queues = tuple(Queue(maxsize=self._maxsize) for _ in range(self._queue_count))

        def is_alive(self):
            return self.started

        def terminate(self):
            self.started = False

        def close(self):
            self.started = False

    repl = ToolREPL(echo=False, transport=FakeSSHTransport)
    try:
        repl._ensure_session()
        assert repl._transport._args == (None,)
    finally:
        repl.close()

def test_default_checker_missing_release():
    task = BenchmarkTask(id="t", prompt="p", checker=default_checker)
    ctx = BenchmarkTaskContext(
        task=task,
        agent=None,
        metrics={"turns": 1, "release_called": False, "syntax_retries": 0, "runtime_errors": 0, "saw_bash_python": False},
        result="x",
    )
    passed, violations, scores = default_checker(ctx)
    assert passed is True
    assert any(v.code == "missing_release" for v in violations)
    assert scores["completion_behavior"].earned < scores["completion_behavior"].possible


def test_discover_tasks_from_path(tmp_path):
    task_registry.clear()
    module = tmp_path / "bench_one.py"
    module.write_text(
        "from code_agent.repl_benchmark import BenchmarkTask, register_task\n"
        "from code_agent.repl_benchmark.core import default_checker\n"
        "register_task(BenchmarkTask(id='tmp/task', prompt='hi', checker=default_checker))\n"
    )
    tasks = discover_tasks(paths=[tmp_path], include_builtin=False)
    assert [task.id for task in tasks] == ["tmp/task"]


def test_builtin_task_discovery():
    task_registry.clear()
    tasks = discover_tasks(include_builtin=True)
    assert tasks == []


def test_runner_returns_summary():
    task_registry.clear()
    register_task(BenchmarkTask(id="fake/task", prompt="hello", checker=checker_expected_int(55)))
    runner = REPLBenchmarkRunner(FakeAgent, include_builtin=False)
    result = runner.run()
    assert result.model == "fake-model"
    assert len(result.task_results) == 1
    assert result.task_results[0].task_id == "fake/task"
    assert result.usage["requests"] == 1
    assert "Grand Total" in format_summary(result)


def test_penalties_keep_room_for_improvement():
    task_registry.clear()
    register_task(BenchmarkTask(id="fake/task", prompt="hello", checker=checker_expected_int(55)))
    runner = REPLBenchmarkRunner(FakeBadAgent, include_builtin=False)
    result = runner.run()
    item = result.task_results[0]
    assert item.total_score < item.total_possible
    codes = {v.code for v in item.violations}
    assert "bash_python" in codes
    assert "missing_release" in codes
    assert "syntax_retry" in codes


def test_cli_json_round_trip():
    task_registry.clear()
    register_task(BenchmarkTask(id="fake/task", prompt="hello", checker=checker_expected_int(55)))
    runner = REPLBenchmarkRunner(FakeAgent, include_builtin=False)
    payload = json.dumps(runner.run().to_dict())
    obj = json.loads(payload)
    assert obj["model"] == "fake-model"


def test_cli_help_without_warning():
    proc = subprocess.run(
        [sys.executable, "-m", "code_agent.repl_benchmark.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Run the REPL benchmark suite." in proc.stdout
    assert "RuntimeWarning" not in proc.stderr


def test_executable_without_args_shows_help():
    proc = subprocess.run(
        [str(Path("repl-benchmark").resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Run the REPL benchmark suite." in proc.stdout
    assert "usage:" in proc.stdout


def test_cli_human_summary_with_fake_agent(tmp_path):
    task_module = tmp_path / "task_mod.py"
    task_module.write_text(
        "from code_agent.repl_benchmark import BenchmarkTask, register_task\n"
        "from code_agent.repl_benchmark.core import checker_expected_int\n"
        "register_task(BenchmarkTask(id='cli/task', prompt='hello', checker=checker_expected_int(55)))\n"
    )
    agent_module = tmp_path / "fake_agent_mod.py"
    agent_module.write_text(
        dedent('''
        class FakeUsageTracker:
            def __init__(self):
                self.history = []

            def _normalize(self, model_name, usage):
                return usage

        class FakeLLMClient:
            def __init__(self):
                self.usage_tracker = FakeUsageTracker()

        from code_agent.repl_events import ReplEvent

        class FakeAgent:
            model = "fake-cli-model"

            def __init__(self):
                self.llm_client = FakeLLMClient()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def usermsg(self, msg):
                self.msg = msg

            def run_loop(self, max_turns=10, max_syntax_retries=3):
                self.on_repl_execute(None)
                self._on_assistant_message_committed({"content": 'emit("55", release=True)'})
                self.on_statement_events([
                    ReplEvent(
                        kind="statement_started",
                        data={"echo": 'emit("55", release=True)\\n'},
                    ),
                ])
                self.llm_client.usage_tracker.history.append((
                    self.model,
                    {
                        "prompt_tokens": 9,
                        "cached_tokens": 0,
                        "completion_tokens": 4,
                        "reasoning_tokens": 0,
                        "cost": 0.01,
                    },
                ))
                return "55"
        ''').strip() + "\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "code_agent.repl_benchmark.cli",
            "--agent",
            "fake_agent_mod:FakeAgent",
            "--task-path",
            str(task_module),
            "--no-builtin",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "REPL Benchmark Summary" in proc.stdout
    assert "cli/task: PASS" in proc.stdout


@pytest.mark.parametrize(
    ("prompt_args", "inputs"),
    [
        ([], ["What is 2+2?\n"]),
        (["--prompt", "What is 2+2?"], []),
    ],
)
def test_code_agent_cli_trivial_pty(tmp_path, prompt_args, inputs):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            payload = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": 'emit("4", release=True)',
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args, **kwargs):
            pass

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    home = tmp_path / "home"
    code_agent_dir = home / ".code-agent"
    code_agent_dir.mkdir(parents=True)
    (code_agent_dir / "config.py").write_text(dedent(f"""\
register_provider(
    "testlocal",
    host="127.0.0.1",
    path="/v1/chat/completions",
    port={port},
    timeout=10,
    tpm=1000,
    concurrency=5,
    tools=False,
    api_type="completions",
)
register_model(
    "testlocal",
    "tiny",
    aliases="test-code-agent",
    model="tiny",
    input_cost=0.0,
    output_cost=0.0,
)
code_agent_model = "test-code-agent"
"""))

    env = dict(os.environ)
    env["HOME"] = str(home)
    env["TESTLOCAL_API_KEY"] = "dummy"
    env["CODE_AGENT_SESSION_DB"] = str(tmp_path / "sessions.db")
    env["CODE_AGENT_CLI_HISTORY_DB"] = str(tmp_path / "cli_history.db")
    env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd()), *sys.path])

    try:
        result = run_pty_session(
            [
                sys.executable,
                "-m",
                "code_agent.agent",
                "--model",
                "test-code-agent",
                *prompt_args,
            ],
            inputs=inputs,
            env=env,
            cwd=str(Path.cwd()),
            timeout=20,
            wait_for="4",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    text = strip_ansi(result.output)
    assert result.returncode == 0
    assert "4" in text
    assert "Session ended. Goodbye!" in text
    assert "Resume session: coda --resume " in text
    assert Path(env["CODE_AGENT_SESSION_DB"]).exists()
    assert Path(env["CODE_AGENT_CLI_HISTORY_DB"]).exists()


def test_code_agent_pty_captures_syntax_retry_event(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self):
            type(self).calls += 1
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            content = "not valid python !!!" if type(self).calls == 1 else 'emit("fixed", release=True)'
            payload = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args, **kwargs):
            pass

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = build_code_agent_test_env(tmp_path, port=port, extra_env={"SHOW_EVENTS": "1"})
    try:
        result = run_pty_session(
            [sys.executable, "-m", "code_agent.agent", "--model", "test-code-agent"],
            inputs=["Return fixed\n"],
            env=env,
            cwd=str(Path.cwd()),
            timeout=20,
            wait_for="fixed",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert result.returncode == 0
    assert any(event.get("type") == "syntax_retry" and event.get("attempt") == 1 for event in result.events or [])
    assert "[[CODE_AGENT_EVENT:" in result.output
    assert "[[CODE_AGENT_EVENT:" not in strip_events(result.output)


def test_cli_includes_code_agent_builtin_suite(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        counters = {}

        @classmethod
        def next_response(cls, text):
            key_map = [
                ("YYYY-MM-DD", [f'emit("{__import__("datetime").date.today().isoformat()}", release=True)']),
                ("17 * 19 + 23", ['emit("346", release=True)']),
                ("code agent benchmark", ['emit("CODE AGENT BENCHMARK", release=True)']),
                ("sum the numbers [5, 8, 13, 21]", ['emit("47", release=True)']),
                ("2 ** 10", ['emit("1024", release=True)']),
                ("release discipline", ['emit("18", release=True)']),
                ("12345 squared", ['emit("25", release=True)']),
                ("sum(range(1, 11))", ['emit("55", release=True)']),
                ("overrides the session sqlite path", ['preview(grep("CODE_AGENT_SESSION_DB", ".", None, None, False, 2, False, False))\nemit("CODE_AGENT_SESSION_DB", release=True)']),
                ("CLI history sqlite path override", ['preview(grep("CODE_AGENT_CLI_HISTORY_DB", ".", None, None, False, 2, False, False))\nemit("CODE_AGENT_CLI_HISTORY_DB", release=True)']),
                ("first emit() call", ['_code = read("code_agent/agent.py")\npreview(_code)', 'emit("Checking today\'s date...", release=True)']),
                ("number of tool names in _preview_targets", ['_code = read("code_agent/agent.py")\nemit("3", release=True)']),
                ("sqlite state can be isolated", ['preview(grep("CODE_AGENT_SESSION_DB|CODE_AGENT_CLI_HISTORY_DB", ".", None, None, False, 2, False, False))', 'import json\n_s = read("code_agent/session_store.py")\n_h = read("code_agent/cli/mixin.py")\nemit(json.dumps({"session_db_source":"CODE_AGENT_SESSION_DB in session_store.resolve_db_path","history_db_source":"CODE_AGENT_CLI_HISTORY_DB in SQLiteHistory.__init__","reason":"Environment variable overrides force each sqlite path to a temp test db, so benchmark state stays isolated."}), release=True)']),
                ("/resume succeeds", ['_a = read("code_agent/agent.py")\n_b = read("code_agent/session_replay.py")\npreview(_a)', 'emit("resume_session -> replay_session_into_agent -> _replay_display_output; then usermsg adds system_reset / REPL session has been reset", release=True)']),
                ("functional difference between those tools", ['_code = read("code_agent/agent.py")\npreview(_code)', 'import json\nemit(json.dumps({"read":"returns file contents as text for use as a Python value","view":"shows numbered lines and attachment behavior for conversation context"}), release=True)']),
            ]
            for marker, responses in key_map:
                if marker in text:
                    idx = cls.counters.get(marker, 0)
                    cls.counters[marker] = idx + 1
                    return responses[min(idx, len(responses) - 1)]
            return 'emit("fallback", release=True)'

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            text = payload["messages"][-1]["content"]
            content = self.next_response(text)
            response = {
                "choices": [{
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args, **kwargs):
            pass

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = build_code_agent_test_env(tmp_path, port=port)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "code_agent.repl_benchmark.cli", "--model", "test-code-agent", "--json"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    payload = json.loads(proc.stdout)
    task_ids = {item["task_id"] for item in payload["task_results"]}
    assert "code-agent/sqlite-isolation-explanation" in task_ids
    assert "code-agent/resume-flow-summary" in task_ids


def test_cli_can_stream_code_agent_repl_output(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            text = payload["messages"][-1]["content"]
            if "overrides the session sqlite path" in text:
                content = 'preview(grep("CODE_AGENT_SESSION_DB", ".", None, None, False, 2, False, False))\nemit("CODE_AGENT_SESSION_DB", release=True)'
            else:
                content = 'emit("fallback", release=True)'
            response = {
                "choices": [{
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args, **kwargs):
            pass

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = build_code_agent_test_env(
        tmp_path,
        port=port,
        extra_env={"CODE_AGENT_MODEL": "test-code-agent"},
    )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "code_agent.repl_benchmark.cli",
                "--model",
                "test-code-agent",
                "--no-builtin",
                "--show-repl-output",
                "--json",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert "Code Agent" in proc.stderr
    assert "CODE_AGENT_SESSION_DB" in proc.stderr
    payload = json.loads(proc.stdout)
    task_ids = {item["task_id"] for item in payload["task_results"]}
    assert "code-agent/repo-session-db-var" in task_ids
