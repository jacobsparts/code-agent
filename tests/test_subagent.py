import gc
import pickle
import signal
import socket
import struct
import subprocess
import sys
import weakref
from unittest.mock import MagicMock

import pytest

from code_agent.subagent import (
    Subagent,
    SubagentResponse,
    WORKER_CODE,
    _IncrementalMessageReceiver,
    _NO_MESSAGE,
    _configured_default_model,
    _subagents,
)


def test_subagent_has_no_implicit_default_model(monkeypatch):
    monkeypatch.setattr("code_agent.config.get_user_config", lambda: None)
    monkeypatch.setattr(Subagent, "default_model", None)

    agent = Subagent()
    try:
        assert _configured_default_model() is None
        assert agent.model is None
        with pytest.raises(ValueError, match="Subagent model is required"):
            agent._ensure_started()
    finally:
        agent.close()


def test_worker_uses_code_agent_interaction_and_output_hooks():
    compile(WORKER_CODE, "<subagent-worker>", "exec")

    assert "self.output_hook = self._subagent_output_hook" in WORKER_CODE
    assert "agent.run_interaction(prompt, max_turns=task_max_turns)" in WORKER_CODE
    assert "super().on_repl_execute(code)" in WORKER_CODE
    assert "def _handle_tool_request(self, repl, req)" not in WORKER_CODE
    assert "agent.usermsg(prompt)" not in WORKER_CODE
    assert "agent.run_loop(max_turns=task_max_turns)" not in WORKER_CODE



def test_subagent_bootstrap_uses_c_flag_for_nested_multiprocessing(monkeypatch):
    agent = Subagent(cwd="/tmp", model="test/model")
    process = MagicMock()
    process.poll.return_value = None
    process.stdout = MagicMock()
    process.stderr = MagicMock()
    process_stdin = process.stdin
    popen = MagicMock(return_value=process)
    server = MagicMock()
    server.getsockname.return_value = ("127.0.0.1", 12345)
    connection = MagicMock()

    monkeypatch.setattr(
        "code_agent.subagent.socket.socket",
        MagicMock(return_value=server),
    )
    monkeypatch.setattr("code_agent.subagent.subprocess.Popen", popen)
    server.accept.return_value = (connection, None)
    monkeypatch.setattr("code_agent.subagent._recv_msg", lambda _sock: b"auth")
    monkeypatch.setattr("code_agent.subagent.os.urandom", lambda _size: b"auth")
    monkeypatch.setattr("code_agent.subagent._send_msg", MagicMock())
    monkeypatch.setattr(
        "code_agent.subagent.fcntl.fcntl",
        MagicMock(return_value=0),
    )

    try:
        agent._ensure_started()
    finally:
        agent._proc = None
        agent._conn = None
        agent._server = None
        agent._started = False

    command = popen.call_args.args[0]
    assert command == [
        sys.executable,
        "-c",
        "import sys; exec(compile(sys.stdin.read(), '<subagent-worker>', 'exec'))",
    ]
    assert popen.call_args.kwargs["stdin"] is subprocess.PIPE
    assert popen.call_args.kwargs["stdout"] is subprocess.DEVNULL
    assert popen.call_args.kwargs["stderr"] is subprocess.PIPE
    bootstrap = process_stdin.write.call_args.args[0].decode()
    assert "worker_main(12345" in bootstrap
    process_stdin.close.assert_called_once_with()


def test_subagent_stderr_drain_retains_only_the_latest_megabyte():
    agent = Subagent()
    stream = MagicMock()
    stream.read.side_effect = [b"a" * (1024 * 1024), b"tail", b""]

    agent._drain_process_stderr(stream)

    assert agent._read_process_stderr() == "a" * (1024 * 1024 - 4) + "tail"


def test_subagent_has_no_context_manager_and_response_does_not_own_agent():
    agent = Subagent()
    response = SubagentResponse(agent)
    agent_ref = weakref.ref(agent)

    assert not hasattr(agent, "__enter__")
    assert not hasattr(agent, "__exit__")
    assert agent.id in _subagents

    close = MagicMock()
    agent.close = close
    del agent
    gc.collect()

    assert agent_ref() is None
    assert response._agent_ref() is None
    close.assert_called_once_with()


def test_incremental_receiver_retains_partial_frame_across_polls():
    reader, writer = socket.socketpair()
    reader.setblocking(False)
    receiver = _IncrementalMessageReceiver()
    payload = pickle.dumps(("result", "finished"))
    frame = struct.pack("!I", len(payload)) + payload

    try:
        writer.sendall(frame[:2])
        assert receiver.receive(reader) is _NO_MESSAGE

        writer.sendall(frame[2:7])
        assert receiver.receive(reader) is _NO_MESSAGE

        writer.sendall(frame[7:])
        assert receiver.receive(reader) == ("result", "finished")
    finally:
        reader.close()
        writer.close()


def test_incremental_receiver_preserves_partial_frame_when_interrupted(monkeypatch):
    reader, writer = socket.socketpair()
    reader.setblocking(False)
    receiver = _IncrementalMessageReceiver()
    payload = pickle.dumps(("result", "finished"))
    frame = struct.pack("!I", len(payload)) + payload
    original_sigmask = signal.pthread_sigmask

    try:
        writer.sendall(frame[:6])
        assert receiver.receive(reader) is _NO_MESSAGE

        def interrupt_on_unblock(how, mask):
            result = original_sigmask(how, mask)
            if how == signal.SIG_SETMASK:
                raise KeyboardInterrupt
            return result

        writer.sendall(frame[6:])
        monkeypatch.setattr(signal, "pthread_sigmask", interrupt_on_unblock)
        with pytest.raises(KeyboardInterrupt):
            receiver.receive(reader)

        monkeypatch.setattr(signal, "pthread_sigmask", original_sigmask)
        assert receiver.receive(reader) == ("result", "finished")
    finally:
        reader.close()
        writer.close()

def test_foreground_interrupt_detaches_without_stopping_subagent(monkeypatch, capsys):
    agent = Subagent()
    agent._started = True
    agent._proc = MagicMock()
    agent._proc.poll.return_value = None
    agent._conn = MagicMock()
    cleanup = MagicMock()
    agent._cleanup_after_interrupt = cleanup

    monkeypatch.setattr("code_agent.subagent._send_msg", MagicMock())

    def interrupt(_response, _timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(SubagentResponse, "wait", interrupt)

    with pytest.raises(KeyboardInterrupt):
        agent.send("keep working")

    response = agent.last
    assert response is not None
    assert not response._done
    assert agent._current_response is response
    cleanup.assert_not_called()
    assert agent._proc.poll() is None
    assert capsys.readouterr().out == (
        "\nSubagent task is still running in the background. "
        "Use subagent.last to inspect or wait for it.\n"
    )

    agent._conn = None
    agent._proc = None
    agent._started = False
