"""
Subagent spawning for Code Agent.

Spawn isolated Code Agent instances as subprocesses with socket communication.


## Quick Start

    from code_agent.subagent import Subagent

    # Normal: foreground, quiet, waits until done
    agent = Subagent(cwd="/path/to/project")
    response = agent.send("Fix the bug in main.py line 42")
    print(response.result)

    # Follow-up in same session
    response = agent.send("Now add tests")
    agent.close()

## Long-running Tasks

    # Default timeout is None: wait indefinitely.
    response = agent.send("Investigate the report warnings")
    print(response.result)

## Background Execution

    # Use bg=True only for real parallelism. Do not poll with noisy
    # "waiting..." loops; call wait() when you need the result.
    response = agent.send("Refactor the database module", bg=True)
    response.wait()
    print(response.result)

## Progress Updates

Subagents can report intermediate progress with `emit(value, release=False)`.
The parent receives these updates through `response.progress`:

    response = agent.send("Review the architecture", bg=True)
    response.wait()
    print(response.progress)
    print(response.result)

`response.progress` is a snapshot list of progress-update strings. Progress
updates are separate from `response.turns`, which counts subagent REPL turns.
The response representation summarizes both counters without displaying result
text, for example: `<SubagentResponse status='running', turns=3, progress_updates=2>`.

## Multiple Parallel Agents

    agents = [Subagent() for _ in range(3)]
    tasks = ["Fix bug in a.py", "Fix bug in b.py", "Fix bug in c.py"]
    responses = [a.send(t, bg=True) for a, t in zip(agents, tasks)]
    for r in responses:
        r.wait()
        print(r.result)
    for agent in agents:
        agent.close()

Keep every agent in `agents` until its task is complete. Close each agent
explicitly when it is no longer needed.

## Providing Context

A subagent starts with no knowledge of your conversation, your findings, your
files, or your goals.  It has only the Code Agent system prompt and the prompt
you send.

Every task prompt must be self-contained:

- state the goal and what "done" means;
- name the files and directories to read;
- include constraints, prior findings, and anything already ruled out;
- say what must not be touched.

The subagent runs its own Python instance, so parent objects are not shared.
To pass data, serialize it into the prompt, or write it to a temporary file and
tell the subagent to read that path:

    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(findings, fh)
        path = fh.name
    agent.send(f"Read {path} for the confirmed findings, then ...")

## Reuse Subagents

Subagents are persistent and reusable.  A subagent keeps its conversation and
its Python REPL state between tasks.

    agent = Subagent()
    agent.send("Read main.py and map the parse path")
    agent.send("Now add error handling to the parse function")  # Follows up
    agent.close()

Prefer following up with an existing subagent over spawning a new one.  Create
a new subagent only when moving to a genuinely different task, or when running
several tasks in parallel.

Keep an explicit reference to each subagent for its whole lifetime. Call
`close()` as soon as its work and any follow-ups are complete. `__del__`
provides best-effort cleanup if a reference is accidentally lost, but should
not be used as the normal lifecycle mechanism.

## Steering an Active Subagent

A subagent runs one task at a time.  If you learn something that changes the
task, redirect it instead of starting over:

    response = agent.send(
        "Stop working on 159.index.js. The real target is index.js. Continue "
        "the same extraction there.",
        interrupt=True,
    )

`interrupt=True` stops the current task and sends yours as the next task on the
same subagent.  Conversation and REPL state are kept.  The previous response
ends as an interrupted error and is still readable.

Use `agent.interrupt()` to stop the current task without sending new work.
Use `agent.kill()` only to discard the subagent entirely.

## Turn Limits

`max_turns` bounds a single task.  It defaults to 100, and can be set per
subagent or per task:

    agent = Subagent(max_turns=150)
    agent.send("Small, well-scoped fix", max_turns=20)

Raise it for genuinely complex work.  Lower it for simple tasks, or for
delicate work you want to review early.

Reaching the limit ends the task with an error, but the subagent stays alive
and reusable.  The work is not lost.  Find out where it got to:

    status = agent.send("Stop and summarize what you have done and what remains.")

From the subagent's perspective the work is still in progress, so it can report
state and then be told to continue:

    agent.send("Good. Continue with the remaining steps.")

Continuing starts a fresh task, so the turn budget resets.  If the subagent has
gone far in the wrong direction, kill it and start again with better
instructions rather than correcting a long bad trajectory.

Check on a running subagent the same way, with
`send("Stop and report progress", interrupt=True)`.

## Model Configuration

    agent = Subagent()  # Requires a parent-supplied or configured model before use
    agent = Subagent(model="opus")

## Response model

Subagents return text.  Synchronous and background calls expose the same final
text via `response.result`; `bg=True` only changes when the caller waits.  If a
task fails, the exception/traceback is serialized into that text response rather
than raised to the caller.  If the worker process dies, stderr and exit status
are serialized into the response text when available.

## Attributes

    Subagent: .id, .cwd, .model, .done, .result, .send(), .interrupt(), .wait(), .kill(), .close()
    SubagentResponse: .done, .result, .progress, .turns, .is_error, .error, .wait()
"""

import fcntl
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
import weakref
from typing import Any, Optional

try:
    import cloudpickle as pickle
except ImportError:
    import pickle


def _configured_default_model() -> Optional[str]:
    try:
        from code_agent.config import get_user_config
        config = get_user_config()
        value = getattr(config, "code_agent_model", None) if config else None
        if isinstance(value, list):
            return value[0] if value else None
        if value:
            return value
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Socket protocol
# ---------------------------------------------------------------------------

def _send_msg(sock: socket.socket, data: Any):
    """Send pickled message with length prefix."""
    payload = pickle.dumps(data)
    sock.sendall(struct.pack('!I', len(payload)) + payload)


def _recv_msg(sock: socket.socket, timeout: Optional[float] = None) -> Any:
    """Receive pickled message with length prefix."""
    if timeout is not None:
        sock.settimeout(timeout)
    else:
        sock.settimeout(None)

    raw_len = b''
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("Connection closed")
        raw_len += chunk

    msg_len = struct.unpack('!I', raw_len)[0]

    chunks = []
    remaining = msg_len
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("Connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)

    return pickle.loads(b''.join(chunks))


_NO_MESSAGE = object()


class _IncrementalMessageReceiver:
    """Retain partial socket frames across non-blocking polls and interrupts."""

    def __init__(self):
        self._buffer = bytearray()
        self._payload_size: Optional[int] = None

    def _decode_buffered(self) -> Any:
        if self._payload_size is None:
            if len(self._buffer) < 4:
                return _NO_MESSAGE
            self._payload_size = struct.unpack('!I', self._buffer[:4])[0]
            del self._buffer[:4]

        if len(self._buffer) < self._payload_size:
            return _NO_MESSAGE

        payload = bytes(self._buffer[:self._payload_size])
        del self._buffer[:self._payload_size]
        self._payload_size = None
        return pickle.loads(payload)

    def receive(self, sock: socket.socket) -> Any:
        message = self._decode_buffered()
        if message is not _NO_MESSAGE:
            return message

        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("Connection closed")
            self._buffer.extend(chunk)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

        return self._decode_buffered()


def _wrap_subagent_task(prompt: str) -> str:
    """Wrap a task with explicit REPL-completion instructions.

    Subagents run a full REPL-style CodeAgent loop, not a one-shot chat
    completion. Plain-English tasks are therefore easy for models to answer as
    chat text instead of executable Python. This wrapper reiterates the
    execution contract at task time and gives the model an explicit completion
    target: `emit(..., release=True)`.
    """
    return f"""Complete the following task in the Python REPL environment.

Task:
{prompt}

Requirements:
- Your response must be raw Python code only.
- Do not answer in plain English outside Python code.
- When you have completed the task, call emit(result, release=True).
- If the task only asks for a text answer, use emit(the_text, release=True).
- Do not ask conversational follow-up questions unless you truly need clarification.
- If clarification is required, use emit(your_question, release=True).
"""


# ---------------------------------------------------------------------------
# Worker code (runs in subprocess)
# ---------------------------------------------------------------------------

WORKER_CODE = '''
import os
import signal
import socket
import struct
import sys

try:
    import cloudpickle as pickle
except ImportError:
    import pickle


def _send_msg(sock, data):
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        payload = pickle.dumps(data)
        sock.sendall(struct.pack('!I', len(payload)) + payload)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def _recv_msg(sock, timeout=None):
    if timeout is not None:
        sock.settimeout(timeout)
    else:
        sock.settimeout(None)
    raw_len = b''
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("Connection closed")
        raw_len += chunk
    msg_len = struct.unpack('!I', raw_len)[0]
    chunks = []
    remaining = msg_len
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("Connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return pickle.loads(b''.join(chunks))


def worker_main(port, authkey, model, max_turns):
    # Request SIGTERM when parent dies (Linux-specific)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass

    # Connect to host
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(60)
    sock.connect(('127.0.0.1', port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Authenticate
    _send_msg(sock, authkey)
    ack = _recv_msg(sock)
    if ack != 'ok':
        raise RuntimeError("Authentication failed")

    # Import agent classes
    from code_agent.agent import CodeAgent

    class SubagentWorker(CodeAgent):
        """Headless Code Agent for subprocess execution."""

        interactive = True
        welcome_message = ""

        def __init__(self, host_sock, model_name, default_max_turns):
            self._host_sock = host_sock
            self.model = model_name
            self.max_turns = default_max_turns
            self._turn_count = 0
            self._task_interruptible = False
            super().__init__()
            self.output_hook = self._subagent_output_hook

        # Disable CLI display hooks, but report turns to the parent.
        def on_repl_execute(self, code):
            super().on_repl_execute(code)
            self._turn_count += 1
            _send_msg(self._host_sock, ("turn", self._turn_count))

        def on_repl_event(self, event):
            pass

        def on_repl_events_complete(self, events):
            pass

        def on_statement_events(self, events):
            pass

        def _subagent_output_hook(self, value, release):
            message = str(value) if value is not None else ""
            if release:
                self._task_interruptible = False
            _send_msg(
                self._host_sock,
                ("result" if release else "progress", message),
            )

    # Create agent
    agent = SubagentWorker(sock, model, max_turns)

    def _handle_sigint(signum, frame):
        if agent._task_interruptible:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    # Main loop - receive tasks
    while True:
        try:
            msg = _recv_msg(sock, timeout=300)

            if msg is None:
                break

            cmd_type, cmd_data = msg

            if cmd_type == "task":
                prompt = cmd_data.get("prompt", "")
                task_max_turns = cmd_data.get("max_turns", max_turns)
                agent._turn_count = 0

                try:
                    agent._task_interruptible = True
                    agent.run_interaction(prompt, max_turns=task_max_turns)
                except KeyboardInterrupt:
                    agent._task_interruptible = False
                    _send_msg(sock, ("error", "Task interrupted"))
                except Exception as e:
                    agent._task_interruptible = False
                    import traceback
                    _send_msg(sock, ("error", f"{type(e).__name__}: {e}\\n{traceback.format_exc()}"))
                finally:
                    agent._task_interruptible = False
                    agent.complete = False
                    agent._final_result = None

            elif cmd_type == "shutdown":
                break

        except socket.timeout:
            continue
        except ConnectionError:
            break

    sock.close()
'''


# ---------------------------------------------------------------------------
# SubagentError
# ---------------------------------------------------------------------------

class SubagentError(Exception):
    """Raised when a subagent returns an error."""

    def __init__(self, message: str, response: 'SubagentResponse'):
        super().__init__(message)
        self.response = response


# ---------------------------------------------------------------------------
# SubagentResponse
# ---------------------------------------------------------------------------

class SubagentResponse:
    """Result of a Subagent.send() call.

    Attributes:
        done: Whether the task has completed.
        result: Final response text. Empty until done. Errors are serialized here.
        progress: List of progress updates from emit(release=False).
        turns: Number of REPL turns started for this task.
        is_error: Whether the response text represents a task/process error.
        error: Error text if failed, else None. Same text is included in result.
    """

    def __init__(self, agent: 'Subagent'):
        self._agent_ref = weakref.ref(agent)
        self._result: Optional[str] = None
        self._error: Optional[str] = None
        self._done = False
        self._progress: list[str] = []
        self._turns = 0

    def _agent(self) -> 'Subagent':
        agent = self._agent_ref()
        if agent is None:
            raise RuntimeError("Subagent has been closed")
        return agent

    @property
    def done(self) -> bool:
        """Check if the task has completed."""
        if self._done:
            return True
        self._agent()._poll()
        return self._done

    @property
    def result(self) -> str:
        """Final response text. Empty string if not yet complete."""
        if not self.done:
            return ""
        return self._result or ""

    @property
    def progress(self) -> list[str]:
        """Progress updates received so far."""
        if not self._done:
            self._agent()._poll()
        return list(self._progress)

    @property
    def turns(self) -> int:
        """Number of REPL turns started for this task."""
        if not self._done:
            self._agent()._poll()
        return self._turns

    @property
    def is_error(self) -> bool:
        """Whether the task/process ended in an error response."""
        return self._error is not None

    @property
    def error(self) -> Optional[str]:
        """Error text if the task/process failed, else None."""
        if not self.done:
            return None
        return self._error

    def _set_result(self, text: Any) -> None:
        self._result = str(text) if text is not None else ""
        self._done = True

    def _set_error(self, text: Any) -> None:
        body = str(text) if text is not None else ""
        if not body:
            body = "Subagent failed with no error details."
        self._error = body
        self._result = body
        self._done = True

    def wait(self, timeout: Optional[float] = None) -> 'SubagentResponse':
        """Wait for the task to complete.

        Args:
            timeout: Maximum seconds to wait. None = wait indefinitely.

        Returns:
            self, for chaining. Task/process errors are serialized into result.
        """
        start = time.time()
        while not self.done:
            if timeout is not None and (time.time() - start) > timeout:
                break
            time.sleep(0.1)

        return self

    def __repr__(self) -> str:
        if not self._done:
            self._agent()._poll()
        status = "running"
        if self._done:
            status = "error" if self._error is not None else "complete"
        return (
            f"<SubagentResponse status='{status}', turns={self._turns}, "
            f"progress_updates={len(self._progress)}>"
        )


# ---------------------------------------------------------------------------
# Subagent
# ---------------------------------------------------------------------------

# Weak registry for inspection; callers must keep explicit references.
_subagents = weakref.WeakValueDictionary()


class Subagent:
    """A Code Agent running in an isolated subprocess.

    Each Subagent maintains its own session with persistent state.
    Follow-up tasks share conversation context.

    Args:
        cwd: Working directory for the agent. Defaults to current directory.
        model: LLM model to use.
        max_turns: Maximum turns per task. Default 100.

    Example:
        agent = Subagent(cwd="/path/to/project")
        response = agent.send("Fix the bug in main.py")
        print(response.result)

        # Follow-up in same session
        response = agent.send("Now add tests")
        agent.close()
    """

    # Default model set by parent agent via /subagents command.  If unset,
    # Subagent() falls back to the configured Code Agent model.
    default_model: Optional[str] = None

    def __init__(
        self,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = 100
    ):
        self.id = str(uuid.uuid4())[:8]
        self.cwd = cwd or os.getcwd()
        # Use explicit model, parent-injected default, or configured Code Agent default.
        self.model = model or Subagent.default_model or _configured_default_model()
        self.max_turns = max_turns

        self._proc: Optional[subprocess.Popen] = None
        self._conn: Optional[socket.socket] = None
        self._server: Optional[socket.socket] = None
        self._current_response: Optional[SubagentResponse] = None
        self._receiver = _IncrementalMessageReceiver()
        self._stderr_buffer = bytearray()
        self._stderr_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self._started = False

        # Register globally
        _subagents[self.id] = self

    def _ensure_started(self) -> None:
        """Start the subprocess if not already running."""
        if self._started and self._proc and self._proc.poll() is None:
            return
        if not self.model:
            raise ValueError(
                "Subagent model is required unless code_agent_model is configured "
                "or a parent agent supplies one"
            )

        if self._proc:
            self._cleanup()

        authkey = os.urandom(16)

        # Create server socket
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('127.0.0.1', 0))
        self._server.listen(1)
        port = self._server.getsockname()[1]
        self._server.settimeout(30)

        # Bootstrap code - include parent's sys.path so subprocess can find code_agent
        worker_bootstrap = f'''
import sys
sys.path = {repr(sys.path)}
exec({repr(WORKER_CODE)})
worker_main({port}, bytes.fromhex({repr(authkey.hex())}), {repr(self.model)}, {self.max_turns})
'''

        # Start subprocess
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys; exec(compile(sys.stdin.read(), '<subagent-worker>', 'exec'))",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            start_new_session=True,
        )

        self._stderr_buffer.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_process_stderr,
            args=(self._proc.stderr,),
            daemon=True,
            name=f"subagent-{self.id}-stderr",
        )
        self._stderr_thread.start()

        self._proc.stdin.write(worker_bootstrap.encode())
        self._proc.stdin.close()
        self._proc.stdin = None

        # Accept connection
        try:
            self._conn, _ = self._server.accept()
            self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except socket.timeout:
            self._proc.kill()
            self._proc.wait()
            raise TimeoutError(
                f"Subagent failed to connect. stderr: {self._read_process_stderr()}"
            )
        finally:
            self._server.close()
            self._server = None

        # Authenticate
        client_key = _recv_msg(self._conn)
        if client_key != authkey:
            self._conn.close()
            self._proc.kill()
            raise RuntimeError("Subagent authentication failed")
        _send_msg(self._conn, 'ok')

        # Set socket to non-blocking for polling
        fd = self._conn.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        self._receiver = _IncrementalMessageReceiver()
        self._started = True

    def _drain_process_stderr(self, stream) -> None:
        """Continuously drain worker stderr while retaining bounded diagnostics."""
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk or not isinstance(chunk, bytes):
                    break
                with self._stderr_lock:
                    self._stderr_buffer.extend(chunk)
                    if len(self._stderr_buffer) > 1024 * 1024:
                        del self._stderr_buffer[:-1024 * 1024]
        except (OSError, ValueError):
            pass

    def _read_process_stderr(self) -> str:
        """Return stderr retained by the worker's drain thread."""
        if self._stderr_thread and self._proc and self._proc.poll() is not None:
            self._stderr_thread.join(timeout=1)
        with self._stderr_lock:
            return bytes(self._stderr_buffer).decode(errors="replace")

    def _format_process_failure(self) -> str:
        returncode = self._proc.poll() if self._proc else None
        stderr = self._read_process_stderr()
        parts = ["Subagent process died unexpectedly."]
        if returncode is not None:
            parts.append(f"Exit code: {returncode}")
        if stderr:
            parts.append("stderr:\n" + stderr)
        return "\n\n".join(parts)

    def _poll(self) -> None:
        """Poll for messages from subprocess (non-blocking)."""
        if not self._conn or not self._current_response:
            return

        response = self._current_response

        if response._done:
            return

        if self._proc and self._proc.poll() is not None:
            response._set_error(self._format_process_failure())
            return

        while True:
            try:
                msg = self._receiver.receive(self._conn)
                if msg is _NO_MESSAGE:
                    break

                msg_type, msg_data = msg

                if msg_type == "progress":
                    response._progress.append(str(msg_data) if msg_data is not None else "")
                elif msg_type == "turn":
                    response._turns = int(msg_data)
                elif msg_type == "result":
                    response._set_result(msg_data)
                    break
                elif msg_type == "error":
                    response._set_error(f"Subagent task failed:\n\n{msg_data}")
                    break
            except socket.timeout:
                break
            except BlockingIOError:
                break
            except ConnectionError as e:
                detail = f"Connection lost: {e}"
                if self._proc and self._proc.poll() is not None:
                    detail = self._format_process_failure()
                response._set_error(detail)
                break


    def _active_response(self) -> Optional[SubagentResponse]:
        response = self._current_response
        if response is None:
            return None
        if response.done:
            return None
        return response

    def send(
        self,
        prompt: str,
        *,
        bg: bool = False,
        max_turns: Optional[int] = None,
        timeout: Optional[float] = None,
        interrupt: bool = False
    ) -> SubagentResponse:
        """Send a task to the subagent.

        Args:
            prompt: The task or message to send.
            bg: If True, return immediately without waiting.
            max_turns: Override max turns for this task.
            timeout: Seconds to wait; None waits indefinitely (ignored if bg=True).
                Ctrl+C interrupts only this foreground wait; the task continues
                and remains available through ``last`` or ``wait()``.
            interrupt: Interrupt and wait for an active task before sending.

        Returns:
            SubagentResponse object. Its .result is the final text response.
            Task/process errors are serialized into .result.
        """
        self._ensure_started()

        active = self._active_response()
        if active is not None:
            if not interrupt:
                raise RuntimeError("Subagent already has a running task. Wait for it or send with interrupt=True.")
            self.interrupt()

        response = SubagentResponse(self)
        self._current_response = response

        _send_msg(self._conn, ("task", {
            "prompt": _wrap_subagent_task(prompt),
            "max_turns": max_turns or self.max_turns
        }))

        if bg:
            return response

        try:
            response.wait(timeout)
        except KeyboardInterrupt:
            print(
                "\nSubagent task is still running in the background. "
                "Use subagent.last to inspect or wait for it."
            )
            raise
        return response

    def interrupt(self) -> Optional[SubagentResponse]:
        """Interrupt an active task without stopping the subagent."""
        active = self._active_response()
        if active is None:
            return None
        os.killpg(self._proc.pid, signal.SIGINT)
        active.wait()
        return active

    @property
    def last(self) -> Optional[SubagentResponse]:
        """Last response object."""
        return self._current_response

    @property
    def done(self) -> bool:
        """Check if the last task has completed. True if no task sent."""
        return self._current_response.done if self._current_response else True

    @property
    def result(self) -> str:
        """Result text from the last task. Empty if none or not done."""
        return self._current_response.result if self._current_response else ""

    def wait(self, timeout: Optional[float] = None) -> Optional[SubagentResponse]:
        """Wait for the last task to complete.

        Args:
            timeout: Maximum seconds to wait. None = wait indefinitely.

        Returns:
            The SubagentResponse, or None if no task sent. Task/process errors
            are serialized into response.result.
        """
        if self._current_response:
            return self._current_response.wait(timeout)
        return None

    def kill(self) -> str:
        """Kill the subagent process."""
        if self._proc:
            pid = self._proc.pid
            try:
                os.killpg(pid, signal.SIGKILL)
                return f"Killed subagent {self.id} (pid={pid})"
            except ProcessLookupError:
                return "Process already terminated"
        return "No process running"

    def _force_stop(self) -> None:
        """Force local cleanup without graceful shutdown."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None

        if self._proc:
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

        self._started = False

    def _cleanup_after_interrupt(self) -> None:
        """Clean up after foreground KeyboardInterrupt, tolerating repeated Ctrl+C."""
        try:
            self._cleanup()
        except KeyboardInterrupt:
            self._force_stop()

    def _cleanup(self) -> None:
        """Clean up subprocess and socket."""
        if self._conn:
            try:
                _send_msg(self._conn, ("shutdown", None))
            except:
                pass
            try:
                self._conn.close()
            except:
                pass
            self._conn = None

        if self._proc:
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                except:
                    self._proc.kill()
                self._proc.wait()
            if self._stderr_thread:
                self._stderr_thread.join(timeout=1)
            self._proc = None
            self._stderr_thread = None

        self._started = False

    def close(self) -> None:
        """Gracefully close the subagent."""
        self._cleanup()
        if self.id in _subagents:
            del _subagents[self.id]

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._proc and self._proc.poll() is None:
            status = "running" if (self._current_response and not self._current_response._done) else "idle"
            return f"[Subagent id={self.id} pid={self._proc.pid} status={status} cwd={self.cwd}]"
        return f"[Subagent id={self.id} status=stopped cwd={self.cwd}]"
