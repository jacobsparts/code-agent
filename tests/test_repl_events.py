import warnings
from pathlib import Path

import pytest

from code_agent.cli.rewind import (
    _extract_released_assistant_text,
    _find_last_assistant_text,
    build_exchanges_from_events,
)
from code_agent.convo import Convo
from code_agent.repl_agent import REPLMixin, ToolREPL
from code_agent.repl_events import (
    ReplEvent,
    direct_call_name,
    events_output_text,
    normalize_worker_message,
)


def test_rewind_extracts_released_text_from_canonical_blocks():
    message = {
        "role": "assistant",
        "content": [
            {"type": "reasoning", "text": "private"},
            {"type": "text", "text": "emit('done', release=True)"},
        ],
    }

    assert _extract_released_assistant_text(message) == "done"
    assert _find_last_assistant_text([message]) == "emit('done', release=True)"


def test_rewind_builds_exchange_from_canonical_assistant_blocks():
    events = [
        {
            "seq": 1,
            "event_type": "message_added",
            "payload": {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "request"}],
                    "_render_segments": [{"type": "input", "content": "request"}],
                }
            },
        },
        {
            "seq": 2,
            "event_type": "message_added",
            "payload": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "emit('finished', release=True)"}
                    ],
                }
            },
        },
    ]

    exchanges = build_exchanges_from_events(events)

    assert len(exchanges) == 1
    assert exchanges[0].assistant_preview == "finished"


def test_normalize_worker_message_maps_known_and_unknown_types():
    assert normalize_worker_message("output", "value\n") == ReplEvent(
        kind="output",
        text="value\n",
    )
    assert normalize_worker_message("emit", "done\n") == ReplEvent(
        kind="final_emit",
        text="done\n",
    )
    assert normalize_worker_message("custom", "text\n") == ReplEvent(
        kind="worker_output",
        text="text\n",
        data={"message_type": "custom"},
    )


def test_direct_call_name_is_structural_and_does_not_leak_warnings():
    source = 'edit("path", "bad' + chr(92) + 'q", "new")'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert direct_call_name(source) == "edit"

    assert caught == []
    assert direct_call_name("result = edit('a', 'b', 'c')") is None
    assert direct_call_name("print(edit('a', 'b', 'c'))") == "print"
    assert direct_call_name("obj.method()") is None


def test_events_output_text_uses_statement_echo_and_omits_metadata():
    events = [
        ReplEvent(
            kind="statement_started",
            data={"echo": ">>> value\n"},
        ),
        ReplEvent(kind="output", text="1\n"),
        ReplEvent(kind="tool_called", data={"name": "demo", "args": {}}),
        ReplEvent(kind="tool_returned", data={"name": "demo", "result": None}),
        ReplEvent(kind="statement_finished", data={"had_error": False}),
    ]

    assert events_output_text(events) == ">>> value\n1\n"


def test_execute_publishes_statement_boundaries_in_order():
    agent = REPLMixin()
    agent.complete = False
    streamed = []
    completed_statements = []
    agent.on_repl_event = streamed.append
    agent.on_statement_events = lambda events: completed_statements.append(list(events))
    repl = ToolREPL(echo=False)
    repl.inject_builtins()
    try:
        output, pure_syntax_error, events, corrected_code = agent._execute_with_tool_handling(
            repl,
            "value = 1\nprint(value)",
        )
    finally:
        repl.close()

    assert pure_syntax_error is False
    assert corrected_code == "value = 1\nprint(value)"
    assert output.endswith("1\n")
    assert [event for event in streamed if event.kind != "display"] == events
    assert [event.kind for event in streamed if event.kind == "display"] == ["display"]
    assert [event.kind for event in events] == [
        "statement_started",
        "statement_finished",
        "statement_started",
        "print",
        "statement_finished",
    ]
    assert len(completed_statements) == 2
    assert completed_statements[0][0].data["direct_call"] is None
    assert completed_statements[1][0].data["direct_call"] == "print"


class _CanonicalClient:
    def call(self, messages, **kwargs):
        raise AssertionError("not called")


class _CanonicalAgent(REPLMixin):
    system = "system"

    def __init__(self):
        self._llm_client = _CanonicalClient()

    @property
    def llm_client(self):
        return self._llm_client

    def _build_system_prompt(self):
        return self.system


def test_repl_conversation_is_canonical_and_user_messages_are_append_only():
    agent = _CanonicalAgent()

    assert isinstance(agent.conversation, Convo)
    first = agent.usermsg("REPL output", _repl_output=True)
    second = agent.usermsg("human input", _user_content="human input")

    assert agent.conversation.stored_messages()[1:] == [first, second]
    assert first["content"] == [{"type": "text", "text": "REPL output"}]
    assert first["_render_segments"] == [
        {"type": "stdout", "content": "REPL output"}
    ]
    assert second["content"] == [{"type": "text", "text": "human input"}]
    assert second["_render_segments"] == [
        {"type": "input", "content": "human input"}
    ]


def test_assistant_code_preserves_block_order_and_ignores_reasoning():
    message = {
        "role": "assistant",
        "content": [
            {"type": "reasoning", "text": "private"},
            {"type": "commentary", "text": "first\nsecond"},
            {"type": "text", "text": "emit('done', release=True)"},
        ],
    }

    assert REPLMixin._assistant_code(message) == (
        "# first\n# second\nemit('done', release=True)"
    )


def test_assistant_code_reports_unsupported_native_tool_call_details():
    with pytest.raises(
        RuntimeError,
        match=(
            r"unsupported native tool call.*"
            r"name='search'.*id='call-17'.*args=\{'query': 'coins'\}"
        ),
    ):
        REPLMixin._assistant_code({
            "role": "assistant",
            "content": [{
                "type": "tool_call",
                "id": "call-17",
                "name": "search",
                "args": {"query": "coins"},
            }],
        })


@pytest.mark.parametrize("kind", ["attachment", "unknown"])
def test_assistant_code_fails_closed_for_other_non_executable_blocks(kind):
    block = {"type": kind}
    if kind == "attachment":
        block.update(media_type="image/png", data_type="bytes", data=b"x")

    with pytest.raises(NotImplementedError):
        REPLMixin._assistant_code({
            "role": "assistant",
            "content": [block],
        })


def test_dead_tool_repl_worker_restart_is_normal_repl_output():
    agent = REPLMixin()
    streamed = []
    agent.on_repl_event = streamed.append
    repl = ToolREPL(echo=False)
    repl.inject_builtins()
    old_transport = repl._transport
    try:
        old_transport.kill()
        old_transport.join(timeout=1)

        output, pure_syntax_error, events, _ = agent._execute_with_tool_handling(repl, "1")

        notice = (
            "REPL worker exited unexpectedly; started a new worker. "
            "In-memory Python variables were lost.\n"
        )
        assert repl._transport is not old_transport
        assert pure_syntax_error is False
        assert output == notice + ">>> 1\n1\n"
        assert events[0] == ReplEvent(kind="output", text=notice)
        assert [event for event in streamed if event.kind != "display"] == events
    finally:
        repl.close()

def test_worker_interrupt_is_reported_in_statement_event():
    import threading

    agent = REPLMixin()
    repl = ToolREPL(echo=False)
    repl.inject_builtins()
    timer = threading.Timer(0.1, repl._transport.interrupt)
    try:
        timer.start()
        output, pure_syntax_error, events, _ = agent._execute_with_tool_handling(
            repl,
            "while True:\n    pass",
        )
    finally:
        timer.cancel()
        repl.close()

    assert pure_syntax_error is False
    assert output.endswith("\nKeyboardInterrupt\n")
    assert events[-1] == ReplEvent(
        kind="statement_finished",
        data={"had_error": True, "interrupted": True},
    )


def test_statement_output_spills_over_limit(monkeypatch):
    monkeypatch.setattr("code_agent.repl_agent._MAX_REPL_OUTPUT_CHARS", 10)
    agent = REPLMixin()
    repl = ToolREPL(echo=False)
    repl.inject_builtins()
    try:
        agent._execute_with_tool_handling(
            repl,
            "import code_agent.repl_agent as _repl_agent\n"
            "_repl_agent._MAX_REPL_OUTPUT_CHARS = 10",
        )
        output, _, events, _ = agent._execute_with_tool_handling(repl, "print('x' * 11)")
        path = output.split("written to ", 1)[1].split(" (", 1)[0]
        assert output == f">>> print('x' * 11)\n[large output written to {path} (0.0MB)]\n"
        assert Path(path).read_text() == "x" * 11 + "\n"
        assert "x" * 11 not in output
        assert [event.kind for event in events] == [
            "statement_started",
            "output",
            "statement_finished",
        ]
    finally:
        repl.close()
        if "path" in locals():
            Path(path).unlink()


class _ReplyRecorder:
    def __init__(self):
        self.replies = []
        self.acks = []

    def send_reply(self, request_id, result=None, error=None):
        self.replies.append((request_id, result, error))

    def send_ack(self, request_id):
        self.acks.append(request_id)


def test_relay_tool_lifecycle_events_wrap_authoritative_return_value():
    agent = REPLMixin()
    events = []
    agent._active_repl_event_publisher = events.append
    agent.toolcall = lambda name, args: args["value"] + 1
    repl = _ReplyRecorder()

    agent._handle_tool_request(
        repl,
        {"tool": "demo", "request_id": None, "args": {"value": 2}},
    )

    assert events == [
        ReplEvent(
            kind="tool_called",
            data={"name": "demo", "args": {"value": 2}},
        ),
        ReplEvent(
            kind="tool_returned",
            data={"name": "demo", "result": 3},
        ),
    ]
    assert repl.replies == [(None, 3, None)]
    assert repl.acks == []


def test_relay_tool_failure_publishes_error_metadata():
    agent = REPLMixin()
    events = []
    agent._active_repl_event_publisher = events.append

    def fail(name, args):
        raise ValueError("bad tool")

    agent.toolcall = fail
    repl = _ReplyRecorder()

    agent._handle_tool_request(
        repl,
        {"tool": "demo", "request_id": None, "args": {}},
    )

    assert events == [
        ReplEvent(
            kind="tool_called",
            data={"name": "demo", "args": {}},
        ),
        ReplEvent(
            kind="tool_failed",
            data={"name": "demo", "error": "bad tool"},
        ),
    ]
    assert repl.replies == [(None, None, "bad tool")]
    assert repl.acks == []
