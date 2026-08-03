import ast
import subprocess
import sys
import warnings

from code_agent.session_replay import replay_display_text
from code_agent.repl_events import ReplEvent


def _append_preview_events(store, session_id, **kwargs):
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT seq, event_type, payload_json FROM session_events WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
    with store._connect() as conn:
        associated_preview_keys = {
            row["key"]
            for row in conn.execute(
                "SELECT key FROM session_preview_blobs WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        }
    definitions, placements, exec_start_seq = store._active_persisted_preview_state(
        rows,
        associated_preview_keys,
    )
    kwargs.setdefault("expected_exec_start_seq", exec_start_seq)
    kwargs.setdefault("expected_definitions", definitions)
    kwargs.setdefault("expected_active_placements", placements)
    return store.append_preview_events(session_id, **kwargs)


class DummyStore:
    def __init__(self, events):
        self._events = events

    def get_events(self, session_id):
        return self._events


def test_replay_display_text_respects_rewind():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> first\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("one", release=True)',
        }}},
        {"seq": 3, "event_type": "display", "payload": {"kind": "input", "text": "> second\n\n"}},
        {"seq": 4, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("two", release=True)',
        }}},
        {"seq": 5, "event_type": "rewind", "payload": {"target_seq": 2}},
        {"seq": 6, "event_type": "display", "payload": {"kind": "input", "text": "> third\n\n"}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> first\n\n═ Output ═════════════════════════\none\n> third\n\n"



def test_exec_prompt_text_unwraps_emit():
    from code_agent.agent import CodeAgentBase

    generated = 'emit("Continue investigating src/app.py.\\nRun pytest.", release=True)'

    assert CodeAgentBase._exec_prompt_text(generated) == (
        "Continue investigating src/app.py.\nRun pytest."
    )


def test_exec_prompt_text_keeps_plain_prompt():
    from code_agent.agent import CodeAgentBase

    generated = "Continue investigating src/app.py.\nRun pytest."

    assert CodeAgentBase._exec_prompt_text(generated) == generated


def test_replay_display_text_respects_exec():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> first\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("one", release=True)',
        }}},
        {"seq": 3, "event_type": "exec", "payload": {}},
        {"seq": 4, "event_type": "display", "payload": {"kind": "status", "text": "Session reset.\n"}},
        {"seq": 5, "event_type": "display", "payload": {"kind": "input", "text": "> continuation\n\n"}},
    ])

    out = replay_display_text("sid", store)
    assert out == "Session reset.\n> continuation\n\n"


def test_replay_session_into_agent_exec_resets_messages_and_preview_refs():
    from code_agent.conversation import Conversation
    from code_agent.session_replay import replay_session_into_agent

    class Client:
        pass

    class Agent:
        def __init__(self):
            self.conversation = Conversation(Client(), "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.expanded_preview_refs = self._expanded_preview_refs

    store = DummyStore([
        {"seq": 1, "event_type": "message_added", "payload": {"message": {"role": "user", "content": "first"}}},
        {"seq": 2, "event_type": "preview_expanded", "payload": {"uri": "session://preview/old", "numbered": False}},
        {"seq": 3, "event_type": "message_added", "payload": {"message": {"role": "assistant", "content": "old"}}},
        {"seq": 4, "event_type": "exec", "payload": {}},
        {"seq": 5, "event_type": "message_added", "payload": {"message": {"role": "user", "content": "continuation"}}},
    ])
    store.get_session = lambda session_id: {"cwd": "."}

    agent = Agent()
    replay_session_into_agent(agent, "sid", store)

    assert agent.conversation.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "continuation", "_event_seq": 5},
    ]
    assert agent._expanded_preview_refs == {}


def test_replay_only_deepcopies_lists_for_required_snapshots(monkeypatch):
    from code_agent.conversation import Conversation
    from code_agent import session_replay

    original_deepcopy = session_replay.copy.deepcopy
    list_copy_count = 0

    def counting_deepcopy(value):
        nonlocal list_copy_count
        if isinstance(value, list):
            list_copy_count += 1
        return original_deepcopy(value)

    monkeypatch.setattr(session_replay.copy, "deepcopy", counting_deepcopy)

    display_events = [
        {"seq": seq, "event_type": "display", "payload": {"kind": "status", "text": f"{seq}\n"}}
        for seq in range(1, 101)
    ]
    assert session_replay.replay_display_text("sid", DummyStore(display_events))
    assert list_copy_count == 1

    class Client:
        pass

    class Agent:
        def __init__(self):
            self.conversation = Conversation(Client(), "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.expanded_preview_refs = self._expanded_preview_refs

    message_events = [
        {
            "seq": seq,
            "event_type": "message_added",
            "payload": {"message": {"role": "user", "content": str(seq)}},
        }
        for seq in range(1, 101)
    ]
    store = DummyStore(message_events)
    store.get_session = lambda session_id: {"cwd": "."}

    session_replay.replay_session_into_agent(Agent(), "sid", store)
    assert list_copy_count == 2

def test_code_agent_coalesce_suppresses_invalid_escape_parse_warnings():
    from code_agent.code_agent_coalesce import coalesce_repl_messages

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": 'emit("a1\\[", release=True)'},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": 'emit("a2\\[", release=True)'},
    ]

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        coalesce_repl_messages(
            messages,
            keep_last_interactions=0,
            min_savings_chars=0,
        )

    assert not [
        warning
        for warning in rec
        if issubclass(warning.category, (SyntaxWarning, DeprecationWarning))
    ]


def test_replay_display_text_suppresses_invalid_escape_parse_warnings():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> question\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("answer\\$", release=True)',
        }}},
    ])

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = replay_display_text("sid", store)

    assert out == "> question\n\n═ Output ═════════════════════════\nanswer\\$\n"
    assert not [
        warning
        for warning in rec
        if issubclass(warning.category, (SyntaxWarning, DeprecationWarning))
    ]


def test_exec_prompt_text_suppresses_invalid_escape_parse_warnings():
    from code_agent.agent import CodeAgentBase

    generated = 'emit("Continue with `adb shell input text \\$TEST`.", release=True)'

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert CodeAgentBase._exec_prompt_text(generated) == (
            "Continue with `adb shell input text \\$TEST`."
        )

    assert not [
        warning
        for warning in rec
        if issubclass(warning.category, (SyntaxWarning, DeprecationWarning))
    ]


def test_ast_parse_still_warns_for_invalid_escape_without_replay_suppression():
    script = """
import ast
import warnings

source = 'emit("fresh' + chr(92) + 'q", release=True)'
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    ast.parse(source)
raise SystemExit(0 if any(
    issubclass(warning.category, (SyntaxWarning, DeprecationWarning))
    for warning in rec
) else 1)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-W", "always", "-c", script],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_replay_display_text_ignores_non_display_events():
    store = DummyStore([
        {"seq": 1, "event_type": "message_added", "payload": {"message": {"role": "user"}}},
        {"seq": 2, "event_type": "display", "payload": {"kind": "status", "text": "Attached x\n"}},
    ])

    out = replay_display_text("sid", store)
    assert out == "Attached x\n"


def test_replay_display_text_ignores_virtual_interaction_boundary():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> task\n\n"}},
        {
            "seq": 2,
            "event_type": "message_added",
            "payload": {
                "message": {
                    "role": "assistant",
                    "content": "emit(None, release=True)",
                    "_synthetic": True,
                    "_virtual_interaction_boundary": True,
                }
            },
        },
        {"seq": 3, "event_type": "display", "payload": {"kind": "status", "text": "still working\n"}},
    ])

    assert replay_display_text("s", store) == "> task\n\nstill working\n"


def test_replay_display_text_keeps_prior_display_after_rewind():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> first\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("one", release=True)',
        }}},
        {"seq": 3, "event_type": "display", "payload": {"kind": "input", "text": "> second\n\n"}},
        {"seq": 4, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("two", release=True)',
        }}},
        {"seq": 5, "event_type": "rewind", "payload": {"target_seq": 2}},
        {"seq": 6, "event_type": "display", "payload": {"kind": "status", "text": "Conversation rewound.\n"}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> first\n\n═ Output ═════════════════════════\none\nConversation rewound.\n"


def test_replay_display_text_stops_after_release_until_next_input():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> question\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("answer", release=True)',
        }}},
        {"seq": 3, "event_type": "display", "payload": {"kind": "python", "text": "debug work\n"}},
        {"seq": 4, "event_type": "display", "payload": {"kind": "status", "text": "Loading CLAUDE.md\n"}},
        {"seq": 5, "event_type": "display", "payload": {"kind": "input", "text": "> next\n\n"}},
        {"seq": 6, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("next answer", release=True)',
        }}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> question\n\n═ Output ═════════════════════════\nanswer\n> next\n\n═ Output ═════════════════════════\nnext answer\n"


def test_replay_display_text_prefers_persisted_final_result():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> question\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": "emit(result, release=True)",
            "_final_result": "computed answer",
        }}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> question\n\n═ Output ═════════════════════════\ncomputed answer\n"


def test_replay_display_text_extracts_multiline_concatenated_emit_literal():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> question\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit(\n    "Done.\\n\\n"\n    "Added entrypoint",\n    release=True,\n)',
        }}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> question\n\n═ Output ═════════════════════════\nDone.\n\nAdded entrypoint\n"


def test_replay_display_text_ignores_emit_inside_string_literal():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> question\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'Path("x.py").write_text("""def f():\n    emit(\"55\", release=True)\n""")',
        }}},
        {"seq": 3, "event_type": "display", "payload": {"kind": "input", "text": "> next\n\n"}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> question\n\n> next\n\n"


def test_replay_display_text_does_not_duplicate_emit_display_event():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> question\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("answer", release=True)',
        }}},
        {"seq": 3, "event_type": "display", "payload": {"kind": "emit", "text": "answer\n"}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> question\n\n═ Output ═════════════════════════\nanswer\n"



def test_code_agent_file_diff_paths_extracts_unique_paths():
    from code_agent.agent import CodeAgentBase

    diff = "\n".join([
        "--- a/src/old.py",
        "+++ b/src/old.py",
        "@@ -1 +1 @@",
        "-old",
        "+new",
        "--- /dev/null",
        "+++ b/src/new.py",
    ])

    assert CodeAgentBase._file_diff_paths(diff) == ["src/old.py", "src/new.py"]


def test_code_agent_records_file_diff_event_with_tool_and_paths():
    from code_agent.agent import CodeAgentBase

    class Agent(CodeAgentBase):
        def __init__(self):
            self.events = []
            self._statement_direct_call = "line_patch"

        def _append_session_event(self, event_type, payload, create_session=True):
            self.events.append((event_type, payload, create_session))
            return 1

    diff = "--- src/app.py\n+++ src/app.py\n@@ -1 +1 @@\n-old\n+new\n"
    agent = Agent()
    agent._record_file_diff_event(diff)

    assert agent.events == [(
        "file_diff",
        {
            "kind": "unified_diff",
            "tool": "line_patch",
            "paths": ["src/app.py"],
            "diff": diff,
        },
        False,
    )]


def test_code_agent_formats_file_diff_history_filtered_and_limited():
    from code_agent.agent import CodeAgentBase

    class Store:
        def get_events(self, session_id):
            return [
                {
                    "seq": 1,
                    "event_type": "file_diff",
                    "payload": {
                        "kind": "unified_diff",
                        "tool": "edit",
                        "paths": ["src/a.py"],
                        "diff": "--- src/a.py\n+++ src/a.py\n@@ -1 +1 @@\n-a\n+b\n",
                    },
                },
                {
                    "seq": 2,
                    "event_type": "file_diff",
                    "payload": {
                        "kind": "unified_diff",
                        "tool": "line_patch",
                        "paths": ["src/b.py"],
                        "diff": "--- src/b.py\n+++ src/b.py\n@@ -1 +1 @@\n-b\n+c\n",
                    },
                },
                {
                    "seq": 3,
                    "event_type": "file_diff",
                    "payload": {
                        "kind": "unified_diff",
                        "tool": None,
                        "paths": ["src/a.py"],
                        "diff": "--- src/a.py\n+++ src/a.py\n@@ -2 +2 @@\n-x\n+y\n",
                    },
                },
            ]

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._session_id = "sid"
    agent._session_store = Store()

    assert agent._format_file_diff_events("src/a.py", limit=1) == (
        "# file_diff seq=3 tool=unknown paths=src/a.py\n"
        "--- src/a.py\n+++ src/a.py\n@@ -2 +2 @@\n-x\n+y\n"
    )


def test_code_agent_formats_empty_file_diff_history():
    from code_agent.agent import CodeAgentBase

    class Store:
        def get_events(self, session_id):
            return []

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._session_id = "sid"
    agent._session_store = Store()

    assert agent._format_file_diff_events("src/missing.py") == "No file diffs recorded for src/missing.py."



def test_code_agent_no_repl_display_still_prints_progress_emit(capsys):
    from code_agent.agent import CodeAgentBase

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent.repl_display = False
    agent._display_capture = []

    agent.on_repl_event(ReplEvent(kind="progress", text="working\n"))

    assert capsys.readouterr().out == "\x1b[92mworking\x1b[0m\n"
    assert agent._display_capture == ["working\n"]



def test_code_agent_observe_displays_full_text_in_bright_yellow(capsys):
    from code_agent.agent import CodeAgentBase

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent.repl_display = True
    agent._display_capture = []

    text = "First observation.\n" + ("x" * 500)
    agent.on_repl_event(ReplEvent(
        kind="tool_called",
        data={"name": "observe", "args": {"content": text}},
    ))

    assert capsys.readouterr().out == (
        "\x1b[93mFirst observation.\x1b[0m\n"
        f"\x1b[93m{'x' * 500}\x1b[0m\n"
    )
    assert agent._display_capture == ["First observation.\n", f"{'x' * 500}\n"]


def test_code_agent_observe_is_hidden_when_repl_display_is_disabled(capsys):
    from code_agent.agent import CodeAgentBase

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent.repl_display = False
    agent._display_capture = []

    agent.on_repl_event(ReplEvent(
        kind="tool_called",
        data={"name": "observe", "args": {"content": "Internal observation."}},
    ))

    assert capsys.readouterr().out == ""
    assert agent._display_capture == []


def test_code_agent_observe_statement_is_quiet_without_result(capsys):
    from code_agent.agent import CodeAgentBase

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent.repl_display = True
    agent._display_capture = []
    agent.on_repl_event(ReplEvent(
        kind="statement_started",
        data={
            "direct_call": "observe",
            "source": "observe('reason')",
            "echo": "",
            "display_echo": "",
        },
    ))

    assert capsys.readouterr().out == ""
    assert agent._display_capture == []


def test_configured_models_accepts_string_and_list(monkeypatch):
    from code_agent import agent as agent_module

    monkeypatch.setattr(agent_module, "_get_config_value", lambda name, default: "model-a")
    assert agent_module._configured_models() == ["model-a"]

    monkeypatch.setattr(
        agent_module,
        "_get_config_value",
        lambda name, default: ["model-a", "model-b"],
    )
    assert agent_module._configured_models() == ["model-a", "model-b"]


def test_endpoint_registry_resolves_aliases_and_hides_them_from_error_list():
    from code_agent.llm_registry import EndpointRegistry, ModelNotFoundError

    registry = EndpointRegistry()
    registry.register_provider("provider", host="example.com", path="/v1")
    registry.register_model("provider", "model-a", aliases=["short-a", "a"])

    assert registry.resolve_model_name("short-a") == "provider/model-a"

    try:
        registry.get_model_config("missing")
    except ModelNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ModelNotFoundError")

    assert message == (
        "Unknown model 'missing'. Available models:\n"
        "  - provider/model-a"
    )
    assert "short-a" not in message


def test_code_agent_set_model_resolves_alias_at_assignment_boundary(monkeypatch):
    from code_agent.agent import CodeAgentBase
    from code_agent import llm_registry

    monkeypatch.setattr(
        llm_registry,
        "resolve_model_name",
        lambda name: "provider/model-a" if name == "short-a" else name,
    )

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent.model = "provider/model-b"
    agent._llm_client = object()

    assert agent._set_model("short-a") is True
    assert agent.model == "provider/model-a"
    assert not hasattr(agent, "_llm_client")


def test_code_agent_cycles_configured_models_and_clears_cached_client():
    from code_agent.agent import CodeAgentBase

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent.model = "model-a"
    agent.model_choices = ["model-a", "model-b"]
    agent._llm_client = object()

    status = agent._cycle_model()

    assert agent.model == "model-b"
    assert not hasattr(agent, "_llm_client")
    assert "Model changed: model-b" in status
    assert "model-a" not in status

    agent._cycle_model()
    assert agent.model == "model-a"


def test_code_agent_does_not_cycle_single_configured_model():
    from code_agent.agent import CodeAgentBase

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent.model = "model-a"
    agent.model_choices = ["model-a"]

    assert agent._cycle_model() is None
    assert agent.model == "model-a"


def test_code_agent_resume_session_command_uses_worker_target_when_present():
    from code_agent.agent import CodeAgentBase

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent.worker_target = "root@example.com:project-dir"

    assert agent.resume_session_command("663389fc") == "coda root@example.com:project-dir --resume 663389fc"


def test_code_agent_resume_session_command_without_worker_target_is_local():
    from code_agent.agent import CodeAgentBase

    agent = CodeAgentBase.__new__(CodeAgentBase)

    assert agent.resume_session_command("663389fc") == "coda --resume 663389fc"



def test_code_agent_formats_skills_list():
    from code_agent.agent import CodeAgentBase

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent.list_skills = lambda: [
        {
            "name": "debugging",
            "source": "built-in",
            "attached": False,
            "description": "Debug failures",
        },
        {
            "name": "testing",
            "source": "user",
            "attached": True,
            "description": "",
        },
    ]

    assert agent.format_skills_list() == (
        "Available skills:\n"
        "- debugging [built-in] — Debug failures\n"
        "- testing [user] (attached)\n"
        "\n"
        "Load a skill with: /skills <name>"
    )


def test_code_agent_formats_empty_skills_list():
    from code_agent.agent import CodeAgentBase

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent.list_skills = lambda: []

    assert agent.format_skills_list() == "No skills available."


def test_code_agent_diff_history_tool_bridge():
    from code_agent.agent import CodeAgent

    class Store:
        def get_events(self, session_id):
            return [
                {
                    "seq": 10,
                    "event_type": "file_diff",
                    "payload": {
                        "kind": "unified_diff",
                        "tool": "edit",
                        "paths": ["src/demo.py"],
                        "diff": "--- src/demo.py\n+++ src/demo.py\n@@ -1 +1 @@\n-old\n+new\n",
                    },
                }
            ]

    class Repl:
        def __init__(self):
            self.replies = []
            self.acks = []

        def send_reply(self, request_id, result=None, error=None):
            self.replies.append((request_id, result, error))

        def send_ack(self, request_id):
            self.acks.append(request_id)

    agent = CodeAgent.__new__(CodeAgent)
    agent._session_id = "sid"
    agent._session_store = Store()
    repl = Repl()

    agent._handle_tool_request(repl, {
        "tool": "__file_diffs__",
        "request_id": 123,
        "args": {"file_path": "src/demo.py", "limit": None},
    })

    assert repl.replies == [(
        123,
        "# file_diff seq=10 tool=edit paths=src/demo.py\n"
        "--- src/demo.py\n+++ src/demo.py\n@@ -1 +1 @@\n-old\n+new\n",
        None,
    )]
    assert repl.acks == [123]


def _replay_agent():
    from code_agent.conversation import Conversation

    class Client:
        pass

    class Agent:
        def __init__(self):
            self.conversation = Conversation(Client(), "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.expanded_preview_refs = self._expanded_preview_refs

    return Agent()


def test_persisted_preview_replay_does_not_load_blob_content(tmp_path, monkeypatch):
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.append_event(session_id, 1, "message_added", {"message": {"role": "assistant", "content": "one"}})
    store.save_preview_blob(session_id, "abc", "one")
    _append_preview_events(store,
        session_id,
        expected_next_seq=2,
        preview_key="abc",
        summary="summary",
        source_start_seq=1,
        source_end_seq=1,
    )
    monkeypatch.setattr(store, "get_preview_blob", lambda *args: (_ for _ in ()).throw(AssertionError("blob loaded")))

    agent = _replay_agent()
    replay_session_into_agent(agent, session_id, store)

    assert len(agent.conversation.messages) == 2
    assert "[PreviewRef: session://preview/abc]\nsummary\n[/PreviewRef]" in agent.conversation.messages[1]["content"]
    assert agent.conversation.messages[1]["_persisted_preview"] is True


def test_persisted_preview_replay_recursive_rewind_and_inactive_definition(tmp_path):
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.append_event(session_id, 1, "message_added", {"message": {"role": "assistant", "content": "one"}})
    store.append_event(session_id, 2, "message_added", {"message": {"role": "user", "content": "two"}})
    store.save_preview_blob(session_id, "child", "one")
    _append_preview_events(store,
        session_id, expected_next_seq=3, preview_key="child", summary="child",
        source_start_seq=1, source_end_seq=1,
    )
    store.save_preview_blob(session_id, "parent", "parent content")
    _append_preview_events(store,
        session_id, expected_next_seq=5, preview_key="parent", summary="parent",
        source_start_seq=1, source_end_seq=2,
    )
    store.append_event(session_id, 7, "rewind", {"target_seq": 5})

    agent = _replay_agent()
    replay_session_into_agent(agent, session_id, store)

    visible = "\n".join(message.get("content", "") for message in agent.conversation.messages)
    assert "session://preview/child" in visible
    assert "session://preview/parent" not in visible


def test_persisted_preview_replay_rejects_malformed_missing_partial_and_cross_exec(tmp_path):
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.append_event(session_id, 1, "message_added", {"message": {"role": "assistant", "content": "old"}})
    store.append_event(session_id, 2, "exec", {})
    store.append_event(session_id, 3, "message_added", {"message": {"role": "user", "content": "new"}})
    store.save_preview_blob(session_id, "abc", "new")
    store.append_event(session_id, 4, "preview_created", {"preview_key": "abc", "summary": "ok"})
    store.append_event(session_id, 5, "preview_placed", {
        "preview_event_seq": 4, "source_start_seq": 1, "source_end_seq": 3,
    })
    store.append_event(session_id, 6, "preview_placed", {
        "preview_event_seq": 999, "source_start_seq": 3, "source_end_seq": 3,
    })
    store.append_event(session_id, 7, "preview_created", {
        "preview_key": "missing", "summary": "bad", "extra": 1,
    })

    agent = _replay_agent()
    replay_session_into_agent(agent, session_id, store)

    assert agent.conversation.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "new", "_event_seq": 3},
    ]


def test_rewind_restores_preview_expansion_and_rendered_conversation(tmp_path):
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.save_preview_blob(session_id, "abc", "expanded body")
    uri = "session://preview/abc"
    store.append_event(session_id, 1, "message_added", {
        "message": {"role": "user", "content": f"[PreviewRef: {uri}]\nsummary\n[/PreviewRef]"}
    })
    store.append_event(session_id, 2, "preview_expanded", {"uri": uri, "numbered": False})
    store.append_event(session_id, 3, "preview_collapsed", {"uri": uri})
    store.append_event(session_id, 4, "rewind", {"target_seq": 2})

    agent = _replay_agent()
    agent.conversation.preview_loader = lambda value: store.get_preview_blob(session_id, value.rsplit("/", 1)[1])
    replay_session_into_agent(agent, session_id, store)

    assert agent._expanded_preview_refs == {uri: {"numbered": False}}
    agent.conversation.preview_loader = lambda value: store.get_preview_blob(session_id, value.rsplit("/", 1)[1])
    assert "expanded body" in agent.conversation._messages()[1]["content"]


def test_rewind_restores_collapsed_state_after_later_expand(tmp_path):
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    uri = "session://preview/abc"
    store.save_preview_blob(session_id, "abc", "expanded body")
    store.append_event(session_id, 1, "message_added", {
        "message": {"role": "user", "content": f"[PreviewRef: {uri}]\nsummary\n[/PreviewRef]"}
    })
    store.append_event(session_id, 2, "preview_expanded", {"uri": uri, "numbered": False})
    store.append_event(session_id, 3, "preview_collapsed", {"uri": uri})
    store.append_event(session_id, 4, "preview_expanded", {"uri": uri, "numbered": True})
    store.append_event(session_id, 5, "rewind", {"target_seq": 3})

    agent = _replay_agent()
    replay_session_into_agent(agent, session_id, store)

    assert agent._expanded_preview_refs == {}
    assert "expanded body" not in agent.conversation._messages()[1]["content"]


def test_replay_duplicate_is_idempotent_and_same_range_conflict_is_skipped(tmp_path):
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.append_event(session_id, 1, "message_added", {"message": {"role": "assistant", "content": "one"}})
    for key in ("a", "b"):
        store.save_preview_blob(session_id, key, "one")
    store.append_event(session_id, 2, "preview_created", {"preview_key": "a", "summary": "first"})
    placement = {"preview_event_seq": 2, "source_start_seq": 1, "source_end_seq": 1}
    store.append_event(session_id, 3, "preview_placed", placement)
    store.append_event(session_id, 4, "preview_placed", placement)
    store.append_event(session_id, 5, "preview_created", {"preview_key": "b", "summary": "second"})
    store.append_event(session_id, 6, "preview_placed", {
        "preview_event_seq": 5, "source_start_seq": 1, "source_end_seq": 1,
    })

    agent = _replay_agent()
    replay_session_into_agent(agent, session_id, store)

    visible = agent.conversation.messages[1]["content"]
    assert "session://preview/a" in visible
    assert "session://preview/b" not in visible
    assert agent._persisted_preview_state.active_placements == {(1, 1): 2}


def test_create_resume_expand_fork_resume_end_to_end(tmp_path):
    from code_agent.code_agent_coalesce import Preview, PersistedPreviewState, create_persisted_preview
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.append_event(session_id, 1, "message_added", {"message": {"role": "assistant", "content": "body"}})
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "body", "_event_seq": 1},
    ]
    projected, key, created_seq, placed_seq = create_persisted_preview(
        messages, Preview("summary"), source_start_seq=1, source_end_seq=1,
        store=store, session_id=session_id, expected_next_seq=2,
        state=PersistedPreviewState.empty(),
    )
    assert (created_seq, placed_seq) == (2, 3)
    assert f"session://preview/{key}" in projected[1]["content"]

    resumed = _replay_agent()
    replay_session_into_agent(resumed, session_id, store)
    uri = f"session://preview/{key}"
    assert uri in resumed.conversation.messages[1]["content"]
    assert store.get_preview_blob(session_id, key) == "body"

    store.append_event(session_id, 4, "preview_expanded", {"uri": uri, "numbered": False})
    expanded = _replay_agent()
    replay_session_into_agent(expanded, session_id, store)
    expanded.conversation.preview_loader = lambda value: store.get_preview_blob(session_id, value.rsplit("/", 1)[1])
    assert "body" in expanded.conversation._messages()[1]["content"]

    forked_id = store.fork_session(session_id)
    forked = _replay_agent()
    replay_session_into_agent(forked, forked_id, store)
    forked.conversation.preview_loader = lambda value: store.get_preview_blob(forked_id, value.rsplit("/", 1)[1])
    assert uri in forked.conversation.messages[1]["content"]
    assert "body" in forked.conversation._messages()[1]["content"]


def test_adversarial_history_shared_authoritative_state_matches_replay_and_allows_creation(tmp_path):
    from code_agent.agent import CodeAgentBase
    from code_agent.code_agent_coalesce import Preview
    from code_agent.conversation import Conversation
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    class Client:
        pass

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    for key, content in {
        "valid": "one",
        "other": "one",
        "post": "post",
    }.items():
        store.save_preview_blob(session_id, key, content)

    events = [
        (1, "message_added", {"message": {"role": "assistant", "content": "one"}}),
        (2, "preview_created", {"preview_key": "valid", "summary": "bad", "extra": 1}),
        (3, "preview_created", {"preview_key": "missing", "summary": "missing"}),
        (4, "preview_created", {"preview_key": "valid", "summary": ""}),
        (5, "preview_placed", {"preview_event_seq": 9, "source_start_seq": 1, "source_end_seq": 1}),
        (6, "preview_created", {"preview_key": "valid", "summary": "bool range"}),
        (7, "preview_placed", {"preview_event_seq": 6, "source_start_seq": True, "source_end_seq": 1}),
        (8, "preview_placed", {"preview_event_seq": 9, "source_start_seq": 1, "source_end_seq": 1}),
        (9, "preview_created", {"preview_key": "valid", "summary": "valid"}),
        (10, "preview_placed", {"preview_event_seq": 9, "source_start_seq": 1, "source_end_seq": 1}),
        (11, "preview_created", {"preview_key": "other", "summary": "conflict"}),
        (12, "preview_placed", {"preview_event_seq": 11, "source_start_seq": 1, "source_end_seq": 1}),
        (13, "message_added", {"message": {"role": "assistant", "content": "two"}}),
        (14, "preview_created", {"preview_key": "other", "summary": "partial"}),
        (15, "preview_placed", {"preview_event_seq": 14, "source_start_seq": 1, "source_end_seq": 13}),
        (16, "exec", {}),
        (17, "message_added", {"message": {"role": "assistant", "content": "post"}}),
        (18, "preview_created", {"preview_key": "post", "summary": "cross exec"}),
        (19, "preview_placed", {"preview_event_seq": 18, "source_start_seq": 13, "source_end_seq": 17}),
        (20, "preview_created", {"preview_key": "valid", "summary": "malformed before rewind", "extra": False}),
        (21, "rewind", {"target_seq": 13}),
    ]
    for seq, event_type, payload in events:
        store.append_event(session_id, seq, event_type, payload)

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._conversation = Conversation(Client(), "system")
    agent._expanded_preview_refs = {}
    agent._session_store = store
    agent._session_id = session_id
    agent._next_event_seq = 22
    agent._pending_session_events = []
    agent._suspend_persistence = False
    agent._ensure_live_session = lambda: None
    agent._flush_pending_session_events = lambda: None
    replay_session_into_agent(agent, session_id, store)

    with store._connect() as conn:
        rows = conn.execute(
            "SELECT seq, event_type, payload_json FROM session_events WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
        keys = {
            row["key"]
            for row in conn.execute(
                "SELECT key FROM session_preview_blobs WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        }
    definitions, placements, exec_start_seq = store._active_persisted_preview_state(rows, keys)
    assert definitions == agent._persisted_preview_state.definitions
    assert placements == agent._persisted_preview_state.active_placements
    assert exec_start_seq == agent._persisted_preview_state.exec_start_seq
    assert placements == {(1, 1): 9}
    assert exec_start_seq == 0

    key, created_seq = agent.create_persisted_preview(
        Preview("later valid"),
        source_start_seq=13,
        source_end_seq=13,
    )
    assert created_seq == 22
    assert agent._persisted_preview_state.active_placements == {
        (1, 1): 9,
        (13, 13): 22,
    }

    resumed = CodeAgentBase.__new__(CodeAgentBase)
    resumed._conversation = Conversation(Client(), "system")
    resumed._expanded_preview_refs = {}
    replay_session_into_agent(resumed, session_id, store)
    assert resumed._persisted_preview_state == agent._persisted_preview_state
    assert resumed.conversation.messages == agent.conversation.messages
    assert f"session://preview/{key}" in resumed.conversation.messages[-1]["content"]