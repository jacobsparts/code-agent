import json
import re

import pytest

from code_agent.agent import CodeAgent, _crash_dump_json_default
from code_agent.client import BadRequestError
from code_agent.convo import Convo, MEDIA_ATTACHMENTS_FIELD
from code_agent.repl_attachment_mixin import (
    AudioAttachment,
    ImageAttachment,
    MemoryAttachment,
    TextAttachment,
    iter_placeholders,
    make_audio_attachment,
    make_image_attachment,
    render_attachment_placeholder,
)
from code_agent.repl_events import ReplEvent
from code_agent.session_message_state import reduce_canonical_message_events
from code_agent.session_store import SessionStore
from code_agent.turn_rollups import render_semantic_labels


class DummyClient:
    on_retry = None

    def call(self, messages, tools=None):
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
        }

    def conversation(self, system):
        return Convo(self, system)

def make_agent():
    agent = CodeAgent()
    agent._conversation = Convo(DummyClient(), "system")
    agent._session_id = "session"
    agent._session_store = None
    agent._next_event_seq = 1
    agent._suspend_persistence = True
    agent._explicit_attachment_refs = {}
    agent._pending_explicit_attachment_refs = {}
    agent._pending_session_events = []
    agent._display_capture = []
    agent._pending_unviewed_files = set()
    agent._auto_context_attachment_names = set()
    agent._pending_attachments = {}
    return agent


def test_crash_dump_serializes_typed_attachment_refs_and_media_bytes():
    payload = {
        "memory": MemoryAttachment("context"),
        "media": b"\x00\xff",
    }

    restored = json.loads(json.dumps(payload, default=_crash_dump_json_default))

    assert restored == {
        "memory": {"__memory_attachment__": True, "content": "context"},
        "media": {"__bytes__": True, "hex": "00ff"},
    }


def test_grep_rejects_output_over_two_mib(monkeypatch):
    class FakeStdout:
        def read(self, size):
            assert size == 2 * 1024 * 1024 + 1
            return b"x" * size

    class FakeProcess:
        stdout = FakeStdout()

        def __init__(self):
            self.killed = False
            self.waited = False

        def kill(self):
            self.killed = True

        def wait(self):
            self.waited = True

    process = FakeProcess()
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: process)

    agent = make_agent()
    with pytest.raises(ValueError, match="grep output exceeded 2 MiB"):
        agent.grep("needle", ".")

    assert process.killed is True
    assert process.waited is True


def test_grep_returns_output_at_two_mib_limit(monkeypatch):
    limit = 2 * 1024 * 1024

    class FakeStdout:
        def read(self, size):
            assert size == limit + 1
            return b"x" * limit

    class FakeProcess:
        stdout = FakeStdout()

        def __init__(self):
            self.killed = False
            self.waited = False

        def kill(self):
            self.killed = True

        def wait(self):
            self.waited = True

    process = FakeProcess()
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: process)

    result = make_agent().grep("needle", ".", files_only=False)

    assert len(result) == limit
    assert process.killed is False
    assert process.waited is True


def test_grep_excludes_generated_and_large_data_paths_by_default(monkeypatch):
    captured = {}

    class FakeStdout:
        def read(self, size):
            return b"match"

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self):
            pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    assert make_agent().grep("needle", ".", files_only=False) == "match"

    cmd = captured["cmd"]
    globs = [cmd[index + 1] for index, value in enumerate(cmd) if value == "--glob"]
    assert "!**/.git/**" in globs
    assert "!**/.venv/**" in globs
    assert "!**/node_modules/**" in globs
    assert "!**/.cache/**" in globs
    assert "!**/dist/**" in globs
    assert "!**/build/**" in globs
    assert "!*.min.js" in globs
    assert "!*.map" in globs
    assert "!*.db" in globs
    assert "!*.sqlite3" in globs
    assert "!*.log" in globs


def test_grep_explicit_glob_can_override_default_exclusion(monkeypatch):
    captured = {}

    class FakeStdout:
        def read(self, size):
            return b"match"

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self):
            pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    make_agent().grep("needle", ".", glob="*.map")

    cmd = captured["cmd"]
    default_index = max(
        index for index, value in enumerate(cmd)
        if value == "!*.map"
    )
    explicit_index = max(
        index for index, value in enumerate(cmd)
        if value == "*.map"
    )
    assert explicit_index > default_index


def test_observe_records_runtime_text_and_commits_metadata():
    agent = make_agent()
    agent._start_assistant_execution_attempt()

    assert agent.observe(42) is None
    assert agent.observe("  first\nsecond  ") is None

    message = {"role": "assistant", "content": [{"type": "text", "text": "observe(value)"}]}
    agent._on_assistant_message_committed(message)

    assert message["_observations"] == ["42", "  first\nsecond  "]
    assert agent._pending_observations == []


def test_observe_rejects_empty_content():
    agent = make_agent()
    agent._start_assistant_execution_attempt()

    try:
        agent.observe("  \n ")
    except ValueError as exc:
        assert str(exc) == "Observation content must not be empty."
    else:
        raise AssertionError("Expected whitespace-only observation to fail")
    assert agent._pending_observations == []


def test_observe_does_not_pin_or_release():
    agent = make_agent()
    assistant = {"role": "assistant", "content": [{"type": "text", "text": "print('previous')"}]}
    agent.conversation.append_message(assistant)
    agent.complete = False
    agent._start_assistant_execution_attempt()

    assert agent.observe("Result learned.") is None

    assert "_pinned_coalesce" not in assistant
    assert agent.complete is False




def test_observe_relay_captures_runtime_expression_and_arbitrary_value():
    agent = make_agent()
    agent.complete = False
    agent._start_assistant_execution_attempt()
    repl = agent._get_tool_repl()
    try:
        output, pure_syntax_error, _, _ = agent._execute_with_tool_handling(
            repl,
            "class RuntimeValue:\n"
            "    def __str__(self):\n"
            "        return 'runtime text'\n"
            "value = RuntimeValue()\n"
            "observe(value)",
        )
    finally:
        repl.close()

    assert pure_syntax_error is False
    assert "Traceback" not in output
    assert "'[Continuing...]'" not in output
    assert agent._pending_observations == ["runtime text"]


def test_observations_before_runtime_error_remain_on_committed_message():
    agent = make_agent()
    agent.complete = False
    agent._start_assistant_execution_attempt()
    repl = agent._get_tool_repl()
    try:
        output, pure_syntax_error, _, _ = agent._execute_with_tool_handling(
            repl,
            "observe('kept')\nraise RuntimeError('later failure')",
        )
    finally:
        repl.close()

    message = {"role": "assistant", "content": [{"type": "text", "text": "observe('kept')\nraise RuntimeError('later failure')"}]}
    agent._on_assistant_message_committed(message)

    assert pure_syntax_error is False
    assert "RuntimeError: later failure" in output
    assert message["_observations"] == ["kept"]

def test_transition_before_runtime_error_remains_on_committed_message():
    agent = make_agent()
    agent.complete = False
    agent._start_assistant_execution_attempt()
    repl = agent._get_tool_repl()
    try:
        output, pure_syntax_error, _, _ = agent._execute_with_tool_handling(
            repl,
            "observe('kept transition', transition=True)\n"
            "raise RuntimeError('later failure')",
        )
    finally:
        repl.close()

    message = {
        "role": "assistant",
        "content": (
            [{"type": "text", "text": "observe('kept transition', transition=True)\n"
            "raise RuntimeError('later failure')"}]
        ),
    }
    agent._on_assistant_message_committed(message)

    assert pure_syntax_error is False
    assert "RuntimeError: later failure" in output
    assert message["_observations"] == ["kept transition"]
    assert message["_observation_transition"] is True

def test_live_ordinary_repl_output_gets_transition_association_without_stdout():
    agent = make_agent()
    agent._start_assistant_execution_attempt()
    agent.observe("ordinary output transition", transition=True)
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": "observe('ordinary output transition', transition=True)"}],
    }
    agent._on_assistant_message_committed(assistant)
    assistant["_event_seq"] = 2
    agent._pending_repl_output_for = 2

    agent.usermsg("ordinary output", _repl_output=True)

    output = agent.conversation.stored_messages()[-1]
    assert "_stdout" not in output
    assert output["_render_segments"] == [
        {"type": "stdout", "content": "ordinary output"}
    ]
    assert output["_repl_output_for"] == assistant["_event_seq"]


def test_replay_reconstructs_ordinary_output_association_without_stdout():
    events = [
        {
            "seq": 1,
            "event_type": "message_added",
            "payload": {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "task"}],
                    "_user_content": "task",
                    "_render_segments": [{"type": "input", "content": "task"}],
                }
            },
        },
        {
            "seq": 2,
            "event_type": "message_added",
            "payload": {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "observe('stage', transition=True)"}],
                    "_observation_transition": True,
                }
            },
        },
        {
            "seq": 3,
            "event_type": "message_added",
            "payload": {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "ordinary output"}],
                    "_render_segments": [
                        {"type": "stdout", "content": "ordinary output"}
                    ],
                }
            },
        },
    ]

    messages, _ = reduce_canonical_message_events(events)

    assert "_stdout" not in messages[-1]
    assert messages[-1]["_repl_output_for"] == 2
    assert [
        message["content"]
        for message in render_semantic_labels(messages)
        if message.get("_provider_checkpoint")
    ] == [[{"type": "text", "text": "# Checkpoint 2"}]]
    assert "_repl_output_for" not in events[2]["payload"]["message"]


def test_replay_does_not_associate_non_output_user_or_synthetic_messages():
    transition = {
        "seq": 1,
        "event_type": "message_added",
        "payload": {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "observe('stage', transition=True)"}],
                "_observation_transition": True,
            }
        },
    }
    candidates = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "ordinary input"}],
            "_user_content": "ordinary input",
            "_render_segments": [{"type": "input", "content": "ordinary input"}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "[PreviewRef: session://preview/x]"}],
            "_synthetic": True,
            "_render_segments": [{"type": "stdout", "content": "preview"}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "[Attachment: file.txt]"}],
            "_attachment_refs": {"file.txt": "file.txt"},
        },
    ]

    for candidate in candidates:
        messages, _ = reduce_canonical_message_events([
            transition,
            {
                "seq": 2,
                "event_type": "message_added",
                "payload": {"message": candidate},
            },
        ])
        assert "_repl_output_for" not in messages[-1]


def test_run_loop_ordinary_output_without_stdout_has_first_checkpoint_marker():
    agent = make_agent()
    calls = []
    responses = iter([
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "observe('stage', transition=True)"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "emit('done', release=True)"}],
        },
    ])

    def call(messages, **kwargs):
        calls.append(messages)
        response = next(responses)
        return response

    def execute(repl, content):
        if content.startswith("observe"):
            agent.observe("stage", transition=True)
            text = ">>> observe('stage', transition=True)\n'[Continuing...]'\n"
            return text, False, [ReplEvent(kind="output", text=text)], content
        agent.complete = True
        agent._final_result = "done"
        text = ">>> emit('done', release=True)\ndone\n"
        return text, False, [ReplEvent(kind="output", text=text)], content

    agent.conversation.llm_client.call = call
    event_seq = iter(range(1, 20))
    agent._persist_message = lambda message: agent.conversation.update_message(
        message, _event_seq=next(event_seq)
    )
    agent._ensure_setup = lambda: None
    agent._get_tool_repl = lambda: object()
    agent._execute_with_tool_handling = execute
    agent.build_output_for_llm = lambda events: events[0].text
    agent.process_output_for_llm = lambda output: output
    agent._configure_conversation(agent.conversation)

    assert agent.run_loop(max_turns=2) == "done"

    ordinary_output = agent.conversation.stored_messages()[2]
    assert "_stdout" not in ordinary_output
    transition_seq = next(
        message["_event_seq"]
        for message in agent.conversation.stored_messages()
        if message.get("_observation_transition") is True
    )
    assert ordinary_output["_repl_output_for"] == transition_seq
    assert any(
        message.get("content") == [{
            "type": "text",
            "text": f"# Checkpoint {transition_seq}",
        }]
        for message in calls[1]
    )


def test_new_execution_attempt_discards_uncommitted_observations():
    agent = make_agent()
    agent._start_assistant_execution_attempt()
    agent.observe("abandoned")

    agent._start_assistant_execution_attempt()
    message = {"role": "assistant", "content": [{"type": "text", "text": "print('replacement')"}]}
    agent._on_assistant_message_committed(message)

    assert "_observations" not in message


def test_pin_marks_previous_assistant_turn():
    agent = make_agent()
    assistant = {"role": "assistant", "content": [{"type": "text", "text": "print('important')"}]}
    agent.conversation.append_message({"role": "user", "content": [{"type": "text", "text": "Task"}], "_user_content": "Task"})
    agent.conversation.append_message(assistant)

    result = agent.pin()

    assert result == "Pinned previous turn for coalescing."
    assert agent.conversation.stored_messages()[-1]["_pinned_coalesce"] == {
        "label": "Pinned previous turn",
    }


def test_pin_no_previous_turn_is_noop():
    agent = make_agent()

    result = agent.pin()

    assert result == "No previous turn to pin."


def test_pin_can_target_previous_interaction_release_turn():
    agent = make_agent()
    old_assistant = {"role": "assistant", "content": [{"type": "text", "text": "print('old')"}]}
    release_assistant = {"role": "assistant", "content": [{"type": "text", "text": "emit('old done', release=True)"}]}
    agent.conversation.extend_messages([
        {"role": "user", "content": [{"type": "text", "text": "Old task"}], "_user_content": "Old task"},
        old_assistant,
        {"role": "user", "content": [{"type": "text", "text": ">>> print('old')\nold\n"}]},
        release_assistant,
        {"role": "user", "content": [{"type": "text", "text": ">>> emit('old done', release=True)\nold done\nNew task"}], "_user_content": "New task"},
    ])

    result = agent.pin()

    assert result == "Pinned previous turn for coalescing."
    messages = agent.conversation.stored_messages()
    assert "_pinned_coalesce" not in messages[2]
    assert messages[4]["_pinned_coalesce"] == {"label": "Pinned previous turn"}


def test_pin_persists_metadata_event_for_existing_persisted_message(tmp_path):
    from code_agent.session_replay import replay_session_into_agent

    agent = make_persistent_agent(tmp_path)
    agent.conversation.extend_messages([
        {"role": "user", "content": [{"type": "text", "text": "Task"}], "_user_content": "Task"},
        {"role": "assistant", "content": [{"type": "text", "text": "print('important')"}]},
    ])
    agent._persist_message(agent.conversation.stored_messages()[-2])
    assistant = agent.conversation.stored_messages()[-1]
    agent._persist_message(assistant)

    result = agent.pin()

    assert result == "Pinned previous turn for coalescing."
    events = agent._session_store.get_events(agent._session_id)
    assert events[-1]["event_type"] == "message_pinned"
    assert events[-1]["payload"]["message_event_seq"] == assistant["_event_seq"]

    class ReplayAgent:
        def __init__(self):
            self.conversation = Convo(DummyClient(), "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            pass

    replayed = ReplayAgent()
    replay_session_into_agent(replayed, agent._session_id, agent._session_store)
    replayed_assistant = next(
        msg for msg in replayed.conversation.stored_messages()
        if msg.get("role") == "assistant"
        and msg.get("content") == [{"type": "text", "text": "print('important')"}]
    )
    assert replayed_assistant["_pinned_coalesce"] == {"label": "Pinned previous turn"}


def test_observations_survive_message_persistence_and_replay(tmp_path):
    from code_agent.session_replay import replay_session_into_agent

    agent = make_persistent_agent(tmp_path)
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "observe(value)"}],
        "_observations": ["  persisted\nreflection  "],
    }
    agent._persist_message(message)

    class ReplayAgent:
        def __init__(self):
            self.conversation = Convo(DummyClient(), "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            pass

    replayed = ReplayAgent()
    replay_session_into_agent(replayed, agent._session_id, agent._session_store)

    assert replayed.conversation.stored_messages()[-1]["_observations"] == ["  persisted\nreflection  "]
    events = agent._session_store.get_events(agent._session_id)
    assert [event["event_type"] for event in events] == ["message_added"]


def test_replay_reconstructs_observation_counters_across_rewind_exec_and_fork(tmp_path):
    from code_agent.session_replay import replay_session_into_agent

    store = SessionStore(str(tmp_path / "counter-sessions.db"))
    session_id = store.create_session("/repo", "model")
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "one"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "observe"}], "_observations": ["note"]},
        {"role": "assistant", "content": [{"type": "text", "text": "three"}]},
    ]
    for seq, message in enumerate(messages, 1):
        store.append_event(session_id, seq, "message_added", {"message": message})

    def replay(target_id):
        agent = make_agent()
        agent._session_store = store
        replay_session_into_agent(agent, target_id, store)
        return agent

    resumed = replay(session_id)
    assert (resumed._assistant_turns_since_observation, resumed._assistant_turns_since_transition) == (1, 3)

    forked = replay(store.fork_session(session_id))
    assert (forked._assistant_turns_since_observation, forked._assistant_turns_since_transition) == (1, 3)
    forked._update_observation_counters_from_message({"role": "assistant", "content": [{"type": "text", "text": "fork"}]})
    assert resumed._assistant_turns_since_observation == 1
    assert forked._assistant_turns_since_observation == 2

    store.append_event(session_id, 4, "rewind", {"target_seq": 1})
    rewound = replay(session_id)
    assert (rewound._assistant_turns_since_observation, rewound._assistant_turns_since_transition) == (1, 1)

    store.append_event(session_id, 5, "exec", {})
    reset = replay(session_id)
    assert (reset._assistant_turns_since_observation, reset._assistant_turns_since_transition) == (0, 0)


def test_preview_uri_attachments_are_listed_by_default():
    agent = make_agent()
    agent.conversation.usermsg(
        "[Attachment: session://preview/abc]",
        _attachments={"session://preview/abc": "    1→preview content"},
        _attachment_refs={"session://preview/abc": "session://preview/abc"},
    )

    assert "session://preview/abc" in agent.list_attachments()
    assert "session://preview/abc" not in agent.list_attachments(include_session_blobs=False)


def test_preview_uri_attachments_appear_in_context_notice():
    agent = make_agent()
    notice = agent._file_context_ephemeral(["session://preview/abc"])

    assert "Current attached context:" in notice
    assert "expanded preview: session://preview/abc" in notice
    assert "unview(path_or_uri)" not in notice


def test_context_pressure_notice_when_near_limit():
    agent = make_agent()
    agent.llm_client.model_config["context_constraint"] = 100
    agent.llm_client.usage_tracker.input_tokens_per_byte = {agent.llm_client.model_name: 1.0}
    agent.conversation.usermsg("x" * 200)

    # Inventory alone does not embed guidance; management notices do.
    inventory = agent._file_context_ephemeral([])
    assert "Estimated input:" in inventory
    assert "unview(path_or_uri)" not in inventory

    notices = agent._context_management_ephemeral()
    assert re.search(r"Context usage is \d+%[.,]", notices)
    assert "Warn the user that the context window is nearly exhausted" in notices


def test_context_pressure_notice_combines_with_expanded_context():
    agent = make_agent()
    agent.llm_client.model_config["context_constraint"] = 100
    agent.llm_client.usage_tracker.input_tokens_per_byte = {agent.llm_client.model_name: 1.0}
    agent.conversation.usermsg("x" * 200)

    inventory = agent._file_context_ephemeral(["session://preview/abc"])
    assert "Current attached context:" in inventory
    assert "expanded preview: session://preview/abc" in inventory
    assert "Estimated input:" in inventory

    # With detachable context and high usage, guidance recommends unview.
    agent.conversation.usermsg(
        "[Attachment: session://preview/abc]",
        _attachments={"session://preview/abc": "preview body"},
        _attachment_refs={"session://preview/abc": "session://preview/abc"},
    )
    notices = agent._context_management_ephemeral()
    assert re.search(r"Context usage is \d+%\.", notices)
    assert "You are expected to clean up context now." in notices
    assert "unview(...)" in notices
    assert "Warn the user that the context window is nearly exhausted" not in notices


def test_attachment_listing_supports_typed_text_and_image_values():
    text = TextAttachment("hello")
    image = ImageAttachment(
        content=b"12345678",
        media_type="image/png",
        width=2,
        height=3,
    )

    assert CodeAgent._attachment_listing("notes.txt", text) == (
        "notes.txt (0.0KB)"
    )
    assert CodeAgent._attachment_listing("diagram.png", image) == (
        "diagram.png (image/png, 2×3, 0.0KB)"
    )


def test_dynamic_context_inventory_is_current_and_precedes_guidance():
    agent = make_agent()
    agent.llm_client.model_config["context_constraint"] = 100
    agent.llm_client.usage_tracker.input_tokens_per_byte = {
        agent.llm_client.model_name: 1.0
    }
    image = ImageAttachment(
        content=b"12345678",
        media_type="image/png",
        width=2,
        height=3,
    )
    agent.conversation.usermsg(
        (
            "x" * 200
            + "\n[Attachment: diagram.png, 2×3, image/png]"
            + "\n[PreviewRef: session://preview/abc]\nsummary\n[/PreviewRef]"
        ),
        _attachments={"notes.py": "body", "diagram.png": image},
    )
    agent._expanded_preview_refs = {"session://preview/abc": {"numbered": False}}
    agent._preview_blob_content = lambda uri: "preview body"
    agent._derive_rollup_eligibility = lambda: None

    notice = agent._context_management_ephemeral()

    assert "- file: notes.py" in notice
    assert "- image: diagram.png (8 bytes)" in notice
    assert "- expanded preview: session://preview/abc (12 bytes)" in notice
    assert "Attached context size: 24 bytes." in notice
    assert notice.index("Current attached context:") < notice.index("Context usage is")
    assert "unview" not in notice.split("Context usage is", 1)[0]

    agent.detach("notes.py")
    current = agent._context_management_ephemeral()
    assert "notes.py" not in current


def test_observation_counters_thresholds_precedence_and_resets():
    agent = make_agent()
    agent._reset_observation_counters()

    for _ in range(4):
        agent._update_observation_counters_from_message(
            {"role": "assistant", "content": [{"type": "text", "text": "work"}]}
        )
    assert agent._observation_reminder_ephemeral() is None

    agent._update_observation_counters_from_message(
        {"role": "assistant", "content": [{"type": "text", "text": "work"}]}
    )
    assert "last 5 assistant turns" in agent._observation_reminder_ephemeral()

    agent._assistant_turns_since_transition = 15
    assert "No observation" in agent._observation_reminder_ephemeral()

    agent._update_observation_counters_from_message({
        "role": "assistant",
        "content": [{"type": "text", "text": "observe"}],
        "_observations": ["kept"],
    })
    assert "No transition observation" in agent._observation_reminder_ephemeral()

    agent._update_observation_counters_from_message({
        "role": "assistant",
        "content": [{"type": "text", "text": "transition"}],
        "_observations": ["stage"],
        "_observation_transition": True,
    })
    assert agent._observation_reminder_ephemeral() is None
    assert agent._assistant_turns_since_observation == 0
    assert agent._assistant_turns_since_transition == 0


def test_malformed_observation_metadata_and_synthetic_boundaries_do_not_reset_or_count():
    agent = make_agent()
    agent._assistant_turns_since_observation = 7
    agent._assistant_turns_since_transition = 16

    agent._update_observation_counters_from_message({
        "role": "assistant",
        "content": [{"type": "text", "text": "malformed"}],
        "_observations": [],
        "_observation_transition": True,
    })
    assert (agent._assistant_turns_since_observation, agent._assistant_turns_since_transition) == (8, 17)

    agent._update_observation_counters_from_message({
        "role": "assistant",
        "content": [{"type": "text", "text": "emit(None, release=True)"}],
        "_synthetic": True,
        "_virtual_interaction_boundary": True,
    })
    assert (agent._assistant_turns_since_observation, agent._assistant_turns_since_transition) == (8, 17)


def test_context_constraint_resolution_validation_and_max_input_exclusion():
    agent = make_agent()
    config = agent.llm_client.model_config

    config.update(context_constraint=120, context_window=200, max_input_tokens=50)
    assert agent._resolved_context_constraint() == 120

    for invalid in (True, 0, -1, "120"):
        config["context_constraint"] = invalid
        assert agent._resolved_context_constraint() == 200

    config["context_window"] = None
    assert agent._resolved_context_constraint() is None


def test_context_guidance_tiers_and_available_actions():
    agent = make_agent()
    candidate_turns = (1, 2)

    assert agent._context_management_notices_ephemeral(
        accounting={"constraint": 100, "usage_percent": 29},
        rollup_candidate_turns=candidate_turns,
        detachable_names=[],
    ) == []

    soft = agent._context_management_notices_ephemeral(
        accounting={"constraint": 100, "usage_percent": 30},
        rollup_candidate_turns=candidate_turns,
        detachable_names=[],
    )
    assert "Eligible rollups exist" in soft[-1]

    no_rollup = agent._context_management_notices_ephemeral(
        accounting={"constraint": 100, "usage_percent": 30},
        rollup_candidate_turns=(),
        detachable_names=[],
    )
    assert no_rollup == []

    cases = [
        (candidate_turns, [], "Roll up eligible old context", "unview"),
        ((), ["file.py"], "unview(...)", "Warn the user"),
        (candidate_turns, ["file.py"], "Roll up eligible old context and detach", "Warn the user"),
        ((), [], "Warn the user", "You are expected"),
    ]
    for available, names, included, excluded in cases:
        notice = agent._context_management_notices_ephemeral(
            accounting={"constraint": 100, "usage_percent": 80},
            rollup_candidate_turns=available,
            detachable_names=names,
        )[-1]
        assert included in notice
        assert excluded not in notice


def test_context_accounting_estimation_failure_is_nonfatal():
    agent = make_agent()
    agent.llm_client.model_config["context_constraint"] = 100
    agent.llm_client._estimate_input_tokens = lambda value: (_ for _ in ()).throw(
        RuntimeError("estimate failed")
    )

    accounting = agent._context_accounting()

    assert accounting == {
        "estimated_tokens": None,
        "constraint": 100,
        "usage_percent": None,
    }
    assert agent._context_management_notices_ephemeral(
        accounting=accounting,
        rollup_candidate_turns=(),
        detachable_names=[],
    ) == []


def test_context_notices_are_ephemeral_and_do_not_mutate_messages():
    agent = make_agent()
    agent._configure_conversation(agent.conversation)
    agent.conversation.usermsg("request")
    original = json.loads(json.dumps(agent.conversation.stored_messages()))
    agent._assistant_turns_since_observation = 5

    first = agent.conversation.projected_messages()
    second = agent.conversation.projected_messages()

    assert "No observation has been recorded" in "".join(
        block.get("text", "")
        for block in first[-1]["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    )
    assert first == second
    assert agent.conversation.stored_messages() == original





def test_current_context_names_include_preview_uris():
    agent = make_agent()
    agent.conversation.usermsg(
        "[Attachment: session://preview/abc]",
        _attachments={"session://preview/abc": "    1→preview content"},
        _attachment_refs={"session://preview/abc": "session://preview/abc"},
    )

    assert "session://preview/abc" in agent._current_file_context_names()



def test_current_context_names_only_include_preview_uris_that_can_render():
    agent = make_agent()
    agent._session_id = "session"
    agent._expanded_preview_refs = {
        "session://preview/outer": {"numbered": False},
        "session://preview/inner": {"numbered": False},
    }
    blobs = {
        "session://preview/outer": (
            "outer before\n"
            "[PreviewRef: session://preview/inner]\ninner summary\n[/PreviewRef]\n"
            "outer after"
        ),
        "session://preview/inner": "INNER FULL",
    }
    agent._preview_blob_content = blobs.get

    assert "session://preview/inner" not in agent._current_file_context_names()

    agent.conversation.usermsg("[PreviewRef: session://preview/outer]\nouter summary\n[/PreviewRef]")

    assert agent._current_file_context_names() == [
        "session://preview/outer",
        "session://preview/inner",
    ]


def test_expanded_preview_context_hides_nested_preview_when_parent_not_in_context():
    agent = make_agent()
    agent._session_id = "session"
    agent._expanded_preview_refs = {
        "session://preview/outer": {"numbered": False},
        "session://preview/inner": {"numbered": False},
    }
    blobs = {
        "session://preview/outer": (
            "outer before\n"
            "[PreviewRef: session://preview/inner]\ninner summary\n[/PreviewRef]\n"
            "outer after"
        ),
        "session://preview/inner": "INNER FULL",
    }
    agent._preview_blob_content = blobs.get

    assert agent._expanded_preview_context() == {}

    agent.conversation.usermsg("[PreviewRef: session://preview/outer]\nouter summary\n[/PreviewRef]")

    assert set(agent._expanded_preview_context()) == {
        "session://preview/outer",
        "session://preview/inner",
    }

def test_unview_collapses_preview_uri():
    agent = make_agent()
    agent._expanded_preview_refs = {"session://preview/abc": {"numbered": False}}

    result = agent.unview("session://preview/abc")

    assert result == "Collapsed preview: session://preview/abc"
    assert "session://preview/abc" not in agent._expanded_preview_refs
    assert "session://preview/abc" in agent._pending_unviewed_files



def test_replay_attachment_invalidated_removes_attachment_refs(tmp_path):
    from code_agent.session_replay import replay_session_into_agent

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session(str(tmp_path), "model")
    store.append_event(session_id, 1, "message_added", {
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "[Attachment: stale.py]"}],
            "_attachment_refs": {"stale.py": "stale.py"},
        }
    })
    store.append_event(session_id, 2, "attachment_invalidated", {"name": "stale.py"})

    class ReplayAgent:
        def __init__(self):
            self.conversation = Convo(DummyClient(), "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            pass

    agent = ReplayAgent()
    replay_session_into_agent(agent, session_id, store)

    assert "_attachment_refs" not in agent.conversation.stored_messages()[1]
    assert "_attachments" not in agent.conversation.stored_messages()[1]



def test_resume_session_materializes_persisted_attachment_refs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "persisted.py").write_text("print('persisted')\n")

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session(str(repo), "model")
    store.append_event(session_id, 1, "message_added", {
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "[Attachment: persisted.py]"}],
            "_attachment_refs": {"persisted.py": "persisted.py"},
        }
    })

    class ResumeAgent(CodeAgent):
        @property
        def llm_client(self):
            return DummyClient()

        def worker_cwd(self):
            return str(repo)

        def _acquire_session_lock(self, session_id):
            return True

        def _release_session_lock(self, session_id=None):
            pass

        def _replay_display_output(self):
            pass

        def _load_file_ref_content(self, filepath, name=None):
            path = repo / filepath
            return {
                "name": name or filepath,
                "path": str(path),
                "content": path.read_text(),
            }

    agent = ResumeAgent()
    agent._ensure_setup()
    agent._session_store = store

    assert agent.resume_session(session_id) is True
    assert "persisted.py" in agent.list_attachments()
    assert "persisted.py" in agent._current_file_context_names()
    assert "print('persisted')" in agent.list_attachments()["persisted.py"].content
def make_persistent_agent(tmp_path):
    agent = CodeAgent()
    agent._ensure_setup()
    agent._session_store = SessionStore(str(tmp_path / "sessions.db"))
    agent._session_id = None
    agent._next_event_seq = 1
    return agent


_VIEW_CONTENT_EVENT_KINDS = {"read", "read_media", "read_partial", "read_attach"}

_SHARED_REPL = None


def _get_shared_repl(agent):
    global _SHARED_REPL
    if _SHARED_REPL is None:
        _SHARED_REPL = agent._get_tool_repl()
    return _SHARED_REPL


def _run_view_code(agent, code, *, start_attempt=True):
    """Execute view-related code through the real worker/parent tool path."""
    if start_attempt:
        agent._start_assistant_execution_attempt()
    agent.complete = False
    repl = _get_shared_repl(agent)
    return agent._execute_with_tool_handling(repl, code)


def _content_events(events):
    return [event for event in events if event.kind in _VIEW_CONTENT_EVENT_KINDS]


def _output_text(events):
    return "".join(event.text for event in events if event.kind == "output")


def _commit_view_attachments(agent, events, content="[view committed]"):
    previous = getattr(agent, "_suspend_persistence", False)
    agent._suspend_persistence = True
    try:
        llm_output = agent.build_output_for_llm(events)
        agent.usermsg(content if llm_output.strip() == "" else llm_output)
    finally:
        agent._suspend_persistence = previous
    return llm_output



def _png_bytes(width=2, height=3):
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


def _jpeg_bytes(width=4, height=5):
    return (
        b"\xff\xd8"
        + b"\xff\xc0\x00\x0b\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x01\x01\x11\x00"
        + b"\xff\xd9"
    )


def _wav_bytes():
    return b"RIFF\x04\x00\x00\x00WAVEdata"


def _mp3_bytes():
    return b"ID3\x04\x00\x00audio"


@pytest.mark.parametrize(
    ("data", "media_type"),
    [
        (_wav_bytes(), "audio/wav"),
        (_mp3_bytes(), "audio/mpeg"),
        (b"\xff\xfbframe", "audio/mpeg"),
    ],
)
def test_audio_detection_and_placeholder_round_trip(data, media_type):
    attachment = make_audio_attachment(data)
    assert attachment == AudioAttachment(data, media_type)
    placeholder = render_attachment_placeholder("clip, final.mp3", attachment)
    assert placeholder == f"[Attachment: clip, final.mp3, {media_type}]"
    assert list(iter_placeholders(placeholder))[0][1] == {
        "name": "clip, final.mp3",
        "media_type": media_type,
        "width": None,
        "height": None,
    }


@pytest.mark.parametrize(
    ("data", "name", "media_type", "width", "height"),
    [
        (_png_bytes(), "misleading.txt", "image/png", 2, 3),
    ],
)
def test_view_detects_images_by_magic_bytes(
    tmp_path, data, name, media_type, width, height
):
    path = tmp_path / name
    path.write_bytes(data)
    agent = make_persistent_agent(tmp_path)

    _, pure_syntax_error, events, _ = _run_view_code(
        agent, f"view({str(path)!r})"
    )
    assert pure_syntax_error is False
    assert [event.kind for event in _content_events(events)] == [
        "read_attach",
        "read_media",
    ]

    llm_output = agent.build_output_for_llm(events)
    expected = f"[Attachment: {path}, {width}×{height}, {media_type}]"
    assert expected in llm_output
    attachment = agent._read_attachments[str(path)]
    assert attachment == ImageAttachment(data, media_type, width, height)

    agent._suspend_persistence = True
    agent.usermsg(llm_output, _user_content=llm_output)
    projected = agent._projected_messages()[-1]
    assert expected in "\n".join(
        block["text"]
        for block in projected["content"]
        if block["type"] == "text"
    )
    assert projected["content"][-1] == {
        "type": "attachment",
        "media_type": media_type,
        "data_type": "bytes",
        "data": data,
    }


@pytest.mark.parametrize(
    ("data", "media_type"),
    [(_wav_bytes(), "audio/wav")],
)
def test_view_detects_audio_and_projects_attachment(tmp_path, data, media_type):
    path = tmp_path / "clip.bin"
    path.write_bytes(data)
    agent = make_persistent_agent(tmp_path)

    _, pure_syntax_error, events, _ = _run_view_code(
        agent, f"view({str(path)!r})"
    )
    assert pure_syntax_error is False
    assert [event.kind for event in _content_events(events)] == [
        "read_attach",
        "read_media",
    ]

    llm_output = agent.build_output_for_llm(events)
    expected = f"[Attachment: {path}, {media_type}]"
    assert expected in llm_output
    assert agent._read_attachments[str(path)] == AudioAttachment(data, media_type)

    agent._suspend_persistence = True
    agent.usermsg(llm_output, _user_content=llm_output)
    assert agent._projected_messages()[-1]["content"][-1] == {
        "type": "attachment",
        "media_type": media_type,
        "data_type": "bytes",
        "data": data,
    }


class _RejectAudioClient:
    on_retry = None

    def conversation(self, system):
        return Convo(self, system)

    def call(self, messages, tools=None):
        return {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}

    def validate_media_type(self, media_type):
        if media_type.startswith("audio/"):
            raise BadRequestError(f"transport does not support {media_type} attachments")


def test_attach_rejects_audio_before_mutating_existing_attachment(tmp_path):
    agent = make_persistent_agent(tmp_path)
    existing = TextAttachment("old")
    agent.conversation.append_message({
        "role": "user",
        "content": [{"type": "text", "text": "[Attachment: clip.wav]"}],
        "_attachments": {"clip.wav": existing},
    })
    agent._llm_client = _RejectAudioClient()

    with pytest.raises(BadRequestError, match="does not support audio/wav"):
        agent.attach("clip.wav", _wav_bytes())

    assert agent.conversation.stored_messages()[-1]["_attachments"]["clip.wav"] == existing
    assert "clip.wav" not in agent._pending_attachments


def test_view_rejects_audio_before_conversation_mutation(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(_wav_bytes())
    agent = make_persistent_agent(tmp_path)
    agent._llm_client = _RejectAudioClient()

    _, pure_syntax_error, events, _ = _run_view_code(
        agent, f"view({str(path)!r})"
    )
    assert pure_syntax_error is False
    with pytest.raises(BadRequestError, match="does not support audio/wav"):
        agent.build_output_for_llm(events)

    assert not getattr(agent, "_read_attachments", {})
    assert all(
        str(path) not in message.get("_attachments", {})
        for message in agent.conversation.stored_messages()
    )


def test_invalid_png_signature_has_clear_decode_error(tmp_path):
    path = tmp_path / "broken.dat"
    path.write_bytes(b"\x89PNG\r\n\x1a\ntruncated")
    agent = make_persistent_agent(tmp_path)

    _, pure_syntax_error, events, _ = _run_view_code(
        agent, f"view({str(path)!r})"
    )
    assert pure_syntax_error is False
    with pytest.raises(ValueError, match="Unable to decode media attachment"):
        agent.build_output_for_llm(events)


def test_image_placeholder_parser_supports_commas_and_stable_media_order():
    first = make_image_attachment(_png_bytes(7, 8))
    second = make_image_attachment(_jpeg_bytes(9, 10))
    first_name = "screen, final.png"
    second_name = "other image.jpg"
    first_placeholder = render_attachment_placeholder(first_name, first)
    second_placeholder = render_attachment_placeholder(second_name, second)
    content = f"{second_placeholder}\n{first_placeholder}"

    parsed = [item for _, item in iter_placeholders(content)]
    assert [item["name"] for item in parsed] == [second_name, first_name]

    conversation = Convo(DummyClient(), "system")
    conversation.usermsg(
        content,
        _attachments={
            first_name: first,
            second_name: second,
            "missing-placeholder.png": first,
            "notes.txt": TextAttachment("rendered notes"),
        },
    )
    projected = conversation.projected_messages()[-1]
    assert projected["content"] == [{"type": "text", "text": content}]
    assert [item["name"] for item in projected[MEDIA_ATTACHMENTS_FIELD]] == [
        second_name,
        first_name,
    ]


def test_text_and_image_attachments_materialize_differently():
    image = make_image_attachment(_png_bytes())
    image_placeholder = render_attachment_placeholder("diagram.png", image)
    conversation = Convo(DummyClient(), "system")
    conversation.usermsg(
        f"[Attachment: notes.txt]\n{image_placeholder}",
        _attachments={
            "notes.txt": TextAttachment("    1→hello"),
            "diagram.png": image,
        },
    )

    projected = conversation.projected_messages()[-1]
    assert projected["content"] == [
        {
            "type": "text",
            "text": f"    1→hello\n{image_placeholder}",
        }
    ]
    assert projected[MEDIA_ATTACHMENTS_FIELD][0]["content"] == image.content



def test_typed_image_attachment_persists_and_replays(tmp_path):
    from code_agent.session_replay import replay_session_into_agent

    image = make_image_attachment(_png_bytes(11, 12))
    placeholder = render_attachment_placeholder("persisted.png", image)
    agent = make_persistent_agent(tmp_path)
    message = {
        "role": "user",
        "content": [{"type": "text", "text": placeholder}],
        "_attachments": {"persisted.png": image},
        "_attachment_refs": {"persisted.png": image},
    }
    agent._persist_message(message)

    class ReplayAgent:
        def __init__(self):
            self.conversation = Convo(DummyClient(), "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            pass

    replayed = ReplayAgent()
    replay_session_into_agent(replayed, agent._session_id, agent._session_store)
    replayed_message = replayed.conversation.stored_messages()[-1]

    assert replayed_message["content"] == [
        {"type": "text", "text": placeholder}
    ]
    assert replayed_message["_attachments"]["persisted.png"] == image
    assert replayed_message["_attachment_refs"]["persisted.png"] == image
    projected = replayed.conversation.projected_messages()[-1]
    assert projected[MEDIA_ATTACHMENTS_FIELD][0]["content"] == image.content


def test_view_full_file_already_in_context_emits_notice(tmp_path):
    path = tmp_path / "already.py"
    path.write_text("print('already')\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)
    agent.conversation.usermsg(
        f"[Attachment: {file_path}]",
        _attachments={file_path: "    1→print('already')\n"},
        _attachment_refs={file_path: file_path},
    )
    output, pure_syntax_error, output_chunks, _ = _run_view_code(
        agent,
        f"view({file_path!r})",
    )

    assert pure_syntax_error is False
    assert "Already in context; use reposition=True to move it to the newest context." in output
    assert "Notice: file was already in context." not in output
    assert "Calling view() on files that are already in context is wasteful." not in output
    assert _content_events(output_chunks) == []


def test_view_attach_unattached_full_file(tmp_path):
    path = tmp_path / "fresh.py"
    path.write_text("print('fresh')\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)

    output, pure_syntax_error, events, _ = _run_view_code(agent, f"view({file_path!r})")

    assert pure_syntax_error is False
    assert "Already in context" not in output
    assert any(event.kind == "read_attach" and event.text.strip() == file_path for event in events)
    assert any(event.kind == "read" for event in events)
    assert not any(event.kind == "read_partial" for event in events)
    llm_output = agent.build_output_for_llm(events)
    assert f"[Attachment: {file_path}]" in llm_output
    assert file_path in agent._read_attachments


def test_view_deny_preattached_without_read_events(tmp_path):
    path = tmp_path / "pre.py"
    path.write_text("print('pre')\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)
    agent.conversation.usermsg(
        f"[Attachment: {file_path}]",
        _attachments={file_path: "    1→print('pre')\n"},
        _attachment_refs={file_path: file_path},
    )

    output, pure_syntax_error, events, _ = _run_view_code(agent, f"view({file_path!r})")

    assert pure_syntax_error is False
    assert "Already in context; use reposition=True to move it to the newest context." in output
    assert _content_events(events) == []


def test_view_reposition_attached_moves_to_newest_context(tmp_path):
    path = tmp_path / "move.py"
    path.write_text("print('move')\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)

    _, _, attach_events, _ = _run_view_code(agent, f"view({file_path!r})")
    _commit_view_attachments(agent, attach_events)
    assert agent._is_attached(file_path)

    agent.conversation.usermsg("spacer message between attachment and reposition")

    output, pure_syntax_error, events, _ = _run_view_code(
        agent,
        f"view({file_path!r}, reposition=True)",
    )

    assert pure_syntax_error is False
    assert "Already in context" not in output
    assert "Cannot reposition" not in output
    assert any(event.kind == "read_attach" and event.text.strip() == file_path for event in events)
    assert any(event.kind == "read" for event in events)
    llm_output = _commit_view_attachments(agent, events)
    assert f"[Attachment: {file_path}]" in llm_output
    # Invalidation clears the old placement and commit adds one newest placement.
    placements = [
        (index, msg)
        for index, msg in enumerate(agent.conversation.stored_messages())
        if file_path in (msg.get("_attachments") or {})
        or file_path in (msg.get("_attachment_refs") or {})
    ]
    assert len(placements) == 1
    index, message = placements[0]
    assert index == len(agent.conversation.stored_messages()) - 1
    assert "print('move')" in message["_attachments"][file_path].content


def test_view_deny_reposition_unattached(tmp_path):
    path = tmp_path / "missing.py"
    path.write_text("print('missing')\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)

    output, pure_syntax_error, events, _ = _run_view_code(
        agent,
        f"view({file_path!r}, reposition=True)",
    )

    assert pure_syntax_error is False
    assert "Cannot reposition: file is not in context. Call view(path) first." in output
    assert _content_events(events) == []


def test_view_deny_reposition_partial(tmp_path):
    path = tmp_path / "partial_repo.py"
    path.write_text("line1\nline2\nline3\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)
    agent.conversation.usermsg(
        f"[Attachment: {file_path}]",
        _attachments={file_path: "    1→line1\n"},
        _attachment_refs={file_path: file_path},
    )

    output, pure_syntax_error, events, _ = _run_view_code(
        agent,
        f"view({file_path!r}, offset=1, limit=1, reposition=True)",
    )

    assert pure_syntax_error is False
    assert "reposition=True is only valid for a full file view." in output
    assert _content_events(events) == []




def test_view_small_partial_promotes_with_notice_and_attachment(tmp_path):
    path = tmp_path / "small.py"
    path.write_text("print('small')\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)

    output, pure_syntax_error, events, _ = _run_view_code(
        agent,
        f"view({file_path!r}, offset=1, limit=1)",
    )

    assert pure_syntax_error is False
    assert (
        "Promoted partial view to full view: file is small and not already in context."
        in output
    )
    assert any(event.kind == "read_attach" and event.text.strip() == file_path for event in events)
    assert any(event.kind == "read" for event in events)
    assert not any(event.kind == "read_partial" for event in events)
    llm_output = agent.build_output_for_llm(events)
    assert f"[Attachment: {file_path}]" in llm_output


def test_view_large_partial_remains_partial(tmp_path):
    path = tmp_path / "large.py"
    # Exceed line threshold (>2000 lines).
    path.write_text("\n".join(f"line-{i}" for i in range(2001)) + "\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)

    output, pure_syntax_error, events, _ = _run_view_code(
        agent,
        f"view({file_path!r}, offset=1, limit=2)",
    )

    assert pure_syntax_error is False
    assert "Promoted partial view" not in output
    assert any(event.kind == "read_partial" and event.text.strip() == file_path for event in events)
    assert any(event.kind == "read" for event in events)
    assert not any(event.kind == "read_attach" for event in events)
    llm_output = agent.build_output_for_llm(events)
    assert f"[Attachment: {file_path}]" not in llm_output


def test_view_preattached_partial_remains_partial(tmp_path):
    path = tmp_path / "attached_partial.py"
    path.write_text("alpha\nbeta\ngamma\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)
    agent.conversation.usermsg(
        f"[Attachment: {file_path}]",
        _attachments={file_path: "    1→alpha\n"},
        _attachment_refs={file_path: file_path},
    )

    output, pure_syntax_error, events, _ = _run_view_code(
        agent,
        f"view({file_path!r}, offset=2, limit=1)",
    )

    assert pure_syntax_error is False
    assert "Promoted partial view" not in output
    assert any(event.kind == "read_partial" and event.text.strip() == file_path for event in events)
    assert any(event.kind == "read" and "beta" in event.text for event in events)
    assert not any(event.kind == "read_attach" for event in events)




def test_view_preview_uri_unaffected_by_reposition_rules(tmp_path):
    agent = make_persistent_agent(tmp_path)
    agent.complete = False
    original = "line\n" + ("x" * 6000)
    rendered = agent.process_output_for_llm(original)
    uri = re.search(r"session://preview/[0-9a-f]{16}", rendered).group(0)

    output, pure_syntax_error, events, _ = _run_view_code(agent, f"view({uri!r})")
    llm_output = agent.build_output_for_llm(events)

    assert pure_syntax_error is False
    assert f"Expanded preview: {uri}" in llm_output
    assert "Already in context" not in output
    assert "Cannot reposition" not in output
    assert "Promoted partial view" not in output
    assert agent._expanded_preview_refs == {uri: {"numbered": False}}
    assert _content_events(events) == []




def test_view_full_attach_context_reject_clears_pending_for_retry(tmp_path):
    """Context-limit rejection of a full attach must clear pending so retry is not ignored."""
    path = tmp_path / "retry_full.py"
    path.write_text("print('retry-full')\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)

    # Force first attach to be rejected during parent-side attachment conversion.
    agent.llm_client.model_config["context_window"] = 100
    agent.llm_client.model_config["max_input_tokens"] = 100
    agent.llm_client.usage_tracker.input_tokens_per_byte = {
        agent.llm_client.model_name: 1.0
    }

    output1, pure_syntax_error1, events1, _ = _run_view_code(
        agent,
        f"view({file_path!r})",
    )
    llm_output1 = agent.build_output_for_llm(events1)

    assert pure_syntax_error1 is False
    assert "denied because the file would exceed 90%" in llm_output1
    assert f"[Attachment: {file_path}]" not in llm_output1
    assert file_path not in getattr(agent, "_read_attachments", {})
    pending = getattr(agent, "_pending_full_views", set())
    assert (agent._logical_path(file_path), "attach") not in pending

    # Allow the same-attempt retry to succeed.
    agent.llm_client.model_config["context_window"] = 1_000_000
    agent.llm_client.model_config["max_input_tokens"] = None
    agent.llm_client.usage_tracker.input_tokens_per_byte = {
        agent.llm_client.model_name: 0.0
    }

    output2, pure_syntax_error2, events2, _ = _run_view_code(
        agent,
        f"view({file_path!r})",
        start_attempt=False,
    )
    llm_output2 = agent.build_output_for_llm(events2)

    assert pure_syntax_error2 is False
    assert "Already in context" not in output2
    assert sum(1 for event in events2 if event.kind == "read_attach") == 1
    assert sum(1 for event in events2 if event.kind == "read") == 1
    assert f"[Attachment: {file_path}]" in llm_output2
    assert file_path in agent._read_attachments


def test_view_promoted_partial_context_reject_clears_pending_for_retry(tmp_path):
    """Context-limit rejection after promotion must clear attach pending so retry is not ignored."""
    path = tmp_path / "retry_promote.py"
    path.write_text("print('retry-promote')\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)

    agent.llm_client.model_config["context_window"] = 100
    agent.llm_client.model_config["max_input_tokens"] = 100
    agent.llm_client.usage_tracker.input_tokens_per_byte = {
        agent.llm_client.model_name: 1.0
    }

    output1, pure_syntax_error1, events1, _ = _run_view_code(
        agent,
        f"view({file_path!r}, offset=1, limit=1)",
    )
    llm_output1 = agent.build_output_for_llm(events1)

    assert pure_syntax_error1 is False
    assert (
        "Promoted partial view to full view: file is small and not already in context."
        in output1
    )
    assert "denied because the file would exceed 90%" in llm_output1
    assert f"[Attachment: {file_path}]" not in llm_output1
    assert file_path not in getattr(agent, "_read_attachments", {})
    pending = getattr(agent, "_pending_full_views", set())
    assert (agent._logical_path(file_path), "attach") not in pending

    agent.llm_client.model_config["context_window"] = 1_000_000
    agent.llm_client.model_config["max_input_tokens"] = None
    agent.llm_client.usage_tracker.input_tokens_per_byte = {
        agent.llm_client.model_name: 0.0
    }

    output2, pure_syntax_error2, events2, _ = _run_view_code(
        agent,
        f"view({file_path!r}, offset=1, limit=1)",
        start_attempt=False,
    )
    llm_output2 = agent.build_output_for_llm(events2)

    assert pure_syntax_error2 is False
    assert output2.count(
        "Promoted partial view to full view: file is small and not already in context."
    ) == 1
    assert sum(1 for event in events2 if event.kind == "read_attach") == 1
    assert sum(1 for event in events2 if event.kind == "read") == 1
    assert not any(event.kind == "read_partial" for event in events2)
    assert f"[Attachment: {file_path}]" in llm_output2
    assert file_path in agent._read_attachments





def test_auto_preview_long_complete_turn_output(tmp_path):
    agent = make_persistent_agent(tmp_path)
    original = "x" * 6000

    result = agent.process_output_for_llm(original)

    assert "[PreviewRef: session://preview/" in result
    assert "x" * 6000 not in result
    match = re.search(r"session://preview/([0-9a-f]{16})", result)
    assert match is not None
    assert agent._session_store.get_preview_blob(agent._session_id, match.group(1)) == original


def test_auto_preview_boundary(tmp_path):
    agent = make_persistent_agent(tmp_path)
    agent.auto_preview_turn_chars = 5000

    assert agent.process_output_for_llm("x" * 5000) == "x" * 5000
    result = agent.process_output_for_llm("x" * 5001)
    assert "[PreviewRef: session://preview/" in result
    assert "x" * 5001 not in result


def test_auto_preview_after_attachment_conversion_does_not_expand_attachment_body(tmp_path):
    path = str(tmp_path / "large.txt")
    large_numbered_content = "\n".join(f"{i:>5}→{'x' * 100}" for i in range(100))
    agent = make_persistent_agent(tmp_path)

    output = agent.build_output_for_llm([
        ReplEvent(kind="read_attach", text=path + "\n"),
        ReplEvent(kind="read", text=large_numbered_content + "\n"),
    ])
    result = agent.process_output_for_llm(output)

    assert result == f"[Attachment: {path}]"
    assert "[PreviewRef:" not in result


def test_auto_preview_existing_preview_expansion(tmp_path):
    agent = make_persistent_agent(tmp_path)
    agent.complete = False
    original = "line\n" + ("x" * 6000)
    rendered = agent.process_output_for_llm(original)
    uri = re.search(r"session://preview/[0-9a-f]{16}", rendered).group(0)
    repl = _get_shared_repl(agent)
    output, pure_syntax_error, output_chunks, _ = agent._execute_with_tool_handling(
        repl,
        f"view({uri!r})",
    )
    assert pure_syntax_error is False
    llm_output = agent.build_output_for_llm(output_chunks)

    assert f"Expanded preview: {uri}" in llm_output
    assert agent._expanded_preview_refs == {uri: {"numbered": False}}


def test_auto_preview_can_be_disabled(tmp_path):
    agent = make_persistent_agent(tmp_path)
    agent.auto_preview_turn_chars = 0
    original = "x" * 6000

    result = agent.process_output_for_llm(original)

    assert result == original
    assert "[PreviewRef:" not in result




def test_auto_refresh_uses_last_same_turn_write_for_attached_file(tmp_path):
    path = str(tmp_path / "same.txt")
    agent = make_persistent_agent(tmp_path)
    agent.conversation.usermsg(
        f"[Attachment: {path}]",
        _attachments={path: "    1→old"},
        _attachment_refs={path: path},
    )

    output = agent.build_output_for_llm([
        ReplEvent(kind="file_written", text=json.dumps({"path": path, "content": "first\n"}) + "\n"),
        ReplEvent(kind="file_written", text=json.dumps({"path": path, "content": "second\n"}) + "\n"),
    ])


    assert output == f">>> view({path!r})\n[Attachment: {path}]\n"
    assert "second" in agent._read_attachments[path].content
    assert "first" not in agent._read_attachments[path].content


def test_auto_refresh_skips_when_file_explicitly_viewed_after_write(tmp_path):
    path = str(tmp_path / "same.txt")
    agent = make_persistent_agent(tmp_path)
    agent.conversation.usermsg(
        f"[Attachment: {path}]",
        _attachments={path: "    1→old"},
        _attachment_refs={path: path},
    )

    output = agent.build_output_for_llm([
        ReplEvent(kind="file_written", text=json.dumps({"path": path, "content": "written\n"}) + "\n"),
        ReplEvent(kind="read_attach", text=path + "\n"),
        ReplEvent(kind="read", text="    1→viewed\n"),
    ])


    assert output == f"[Attachment: {path}]\n"
    assert agent._read_attachments[path] == TextAttachment("    1→viewed")


def test_auto_refresh_after_explicit_view_uses_later_write(tmp_path):
    path = str(tmp_path / "same.txt")
    agent = make_persistent_agent(tmp_path)
    agent.conversation.usermsg(
        f"[Attachment: {path}]",
        _attachments={path: "    1→old"},
        _attachment_refs={path: path},
    )

    output = agent.build_output_for_llm([
        ReplEvent(kind="read_attach", text=path + "\n"),
        ReplEvent(kind="read", text="    1→viewed\n"),
        ReplEvent(kind="file_written", text=json.dumps({"path": path, "content": "written\n"}) + "\n"),
    ])

    assert output == f"[Attachment: {path}]\n>>> view({path!r})\n[Attachment: {path}]\n"
    assert "written" in agent._read_attachments[path].content
    assert "viewed" not in agent._read_attachments[path].content
