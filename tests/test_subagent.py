import gc
import pickle
import signal
import socket
import struct
import weakref
from unittest.mock import MagicMock

import pytest

from code_agent.subagent import (
    Subagent,
    SubagentResponse,
    WORKER_CODE,
    _IncrementalMessageReceiver,
    _NO_MESSAGE,
    _subagents,
)


def test_worker_uses_code_agent_interaction_and_output_hooks():
    compile(WORKER_CODE, "<subagent-worker>", "exec")

    assert "self.output_hook = self._subagent_output_hook" in WORKER_CODE
    assert "agent.run_interaction(prompt, max_turns=task_max_turns)" in WORKER_CODE
    assert "super().on_repl_execute(code)" in WORKER_CODE
    assert "def _handle_tool_request(self, repl, req)" not in WORKER_CODE
    assert "agent.usermsg(prompt)" not in WORKER_CODE
    assert "agent.run_loop(max_turns=task_max_turns)" not in WORKER_CODE


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
