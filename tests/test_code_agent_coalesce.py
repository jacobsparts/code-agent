import re

import pytest

import code_agent.code_agent_coalesce as coalesce_module

from code_agent.code_agent_coalesce import Preview
from code_agent.code_agent_coalesce import PreviewBoundaryError
from code_agent.code_agent_coalesce import coalesce_repl_messages
from code_agent.code_agent_coalesce import message_source_range
from code_agent.code_agent_coalesce import place_preview
from code_agent.code_agent_coalesce import preview_key
from code_agent.code_agent_coalesce import render_default_preview_summary
from code_agent.code_agent_coalesce import render_preview_ref


def interaction(i, work_size=2500):
    work = f"print({i})\n" + ("x" * work_size)
    output = f">>> print({i})\n{i}\n" + ("y" * work_size)
    return [
        {"role": "user", "content": f"Task {i}", "_user_content": f"Task {i}"},
        {"role": "assistant", "content": work},
        {"role": "user", "content": output},
        {"role": "assistant", "content": f"emit('Done {i}', release=True)"},
        {"role": "user", "content": f">>> emit('Done {i}', release=True)\nDone {i}\n"},
    ]


def five_interactions():
    messages = [{"role": "system", "content": "system"}]
    for i in range(5):
        messages.extend(interaction(i))
    return messages


def test_render_preview_ref_uses_observations_instead_of_excerpt():
    content = "head\n" + ("x" * 2500) + "\ntail"
    key = preview_key(content)
    summary = render_default_preview_summary(
        content,
        ["First result.", "Second result."],
    )
    rendered = render_preview_ref(key, summary)

    assert f"[PreviewRef: session://preview/{key}]" in rendered
    assert "Observations:\n- First result.\n- Second result." in rendered
    assert "(3 lines, 2510 chars)" in rendered
    assert "x" * 500 not in rendered
    assert "head" not in rendered
    assert "tail" not in rendered


def test_render_preview_ref_formats_multiline_observations():
    content = "canonical"
    summary = render_default_preview_summary(
        content,
        ["The attempt exposed two invariants.\n\nVirtual boundaries matter.\nFiltering broke them."],
    )
    rendered = render_preview_ref(preview_key(content), summary)

    assert (
        "Observations:\n"
        "- The attempt exposed two invariants.\n"
        "\n"
        "  Virtual boundaries matter.\n"
        "  Filtering broke them.\n"
        "(1 lines, 9 chars)"
    ) in rendered


def test_default_preview_summary_without_observations_is_unchanged():
    content = "head\nbody\ntail"

    assert render_default_preview_summary(content) == render_default_preview_summary(content, [])
    assert "Observations:" not in render_default_preview_summary(content)


def test_preview_key_depends_only_on_canonical_content():
    content = "same canonical content"

    plain = render_preview_ref(
        preview_key(content),
        render_default_preview_summary(content),
    )
    observed = render_preview_ref(
        preview_key(content),
        render_default_preview_summary(content, ["Different collapsed rendering."]),
    )

    assert plain.split("]", 1)[0] == observed.split("]", 1)[0]


def test_coalescing_collects_valid_observations_and_preserves_canonical_blob():
    saved = {}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {
            "role": "assistant",
            "content": "observe(value)\n" + ("x" * 2500),
            "_observations": ["First.", 123, "", "Second."],
        },
        {"role": "user", "content": ">>> observe(value)\n'[Continuing...]'\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
    )

    coalesced = next(msg for msg in projected if msg.get("_coalesced"))
    assert "Observations:\n- First.\n- Second." in coalesced["content"]
    assert "x" * 500 not in coalesced["content"]
    canonical = next(iter(saved.values()))
    assert canonical.startswith("observe(value)")
    assert ">>> observe(value)\n'[Continuing...]'" in canonical
    assert "Observations:" not in canonical


def test_pinned_and_normal_sections_keep_their_own_observations():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('normal')\n" + ("x" * 2500), "_observations": ["Normal only."]},
        {"role": "user", "content": ">>> print('normal')\nnormal\n" + ("y" * 2500)},
        {
            "role": "assistant",
            "content": "print('pinned')",
            "_pinned_coalesce": {"label": "Pinned previous turn"},
            "_observations": ["Pinned only."],
        },
        {"role": "user", "content": ">>> print('pinned')\npinned\n"},
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
    )
    coalesced = next(msg for msg in projected if msg.get("_coalesced"))
    blocks = re.findall(r"\[PreviewRef: .*?\[/PreviewRef\]", coalesced["content"], re.DOTALL)

    assert len(blocks) == 2
    assert "Normal only." in blocks[0] and "Pinned only." not in blocks[0]
    assert "Pinned only." in blocks[1] and "Normal only." not in blocks[1]


def test_keeps_last_three_interactions_uncoalesced():
    messages = five_interactions()

    projected = coalesce_repl_messages(messages, min_savings_chars=1)

    assert projected[0] == messages[0]
    assert sum(1 for m in projected if m.get("_coalesced")) == 2
    assert projected[-15:] == messages[-15:]


def test_preserves_real_user_inputs_and_release_assistant_messages():
    saved = {}
    messages = [{"role": "system", "content": "system"}]
    messages.extend(interaction(0))

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
    )

    assert any(m["role"] == "user" and m.get("_user_content") == "Task 0" for m in projected)
    assert any(m["role"] == "assistant" and m["content"] == "emit('Done 0', release=True)" for m in projected)
    visible = "\n".join(m.get("content") or "" for m in projected)
    assert "x" * 600 not in visible
    assert list(saved.values())[0].startswith("print(0)")


def test_saves_preview_blob_deterministically():
    messages = [{"role": "system", "content": "system"}] + interaction(0)
    saved1 = {}
    saved2 = {}

    projected1 = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1, save_preview_blob=saved1.setdefault)
    projected2 = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1, save_preview_blob=saved2.setdefault)

    assert saved1 == saved2
    assert projected1 == projected2
    key = next(iter(saved1))
    assert f"session://preview/{key}" in projected1[2]["content"]


def test_skips_small_interactions():
    messages = [{"role": "system", "content": "system"}] + interaction(0, work_size=1)

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1)

    assert projected == messages


def test_can_coalesce_last_interactions_for_resume_compaction():
    messages = [{"role": "system", "content": "system"}]
    messages.extend(five_interactions())

    default = coalesce_repl_messages(
        messages,
        keep_last_interactions=3,
        keep_last_execution_interactions=1,
        min_savings_chars=1,
    )
    aggressive = coalesce_repl_messages(
        messages,
        keep_last_interactions=3,
        keep_last_execution_interactions=1,
        min_savings_chars=1,
        protect_last_interactions=False,
    )

    assert sum(1 for msg in default if msg.get("_coalesced")) == 2
    assert sum(1 for msg in aggressive if msg.get("_coalesced")) == 5
    assert sum(len(msg.get("content") or "") for msg in aggressive) < sum(len(msg.get("content") or "") for msg in default)




def test_attachment_placeholders_and_payloads_survive():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "view('src/foo.py')\n" + ("x" * 2500)},
        {
            "role": "user",
            "content": ">>> view('src/foo.py')\n[Attachment: src/foo.py]\n" + ("y" * 2500),
            "_attachments": {"src/foo.py": "    1→print('hi')"},
            "_attachment_refs": {"src/foo.py": "src/foo.py"},
        },
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1)
    coalesced = next(m for m in projected if m.get("_coalesced"))

    assert "[Attachment: src/foo.py]" in coalesced["content"]
    assert coalesced["_attachments"] == {"src/foo.py": "    1→print('hi')"}
    assert coalesced["_attachment_refs"] == {"src/foo.py": "src/foo.py"}


def test_nested_preview_refs_remain_placeholders():
    saved = {}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "preview(value)\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> preview(value)\n[PreviewRef: session://preview/abc]\n(1 lines, 30000 chars)\n[/PreviewRef]\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1, save_preview_blob=saved.setdefault)
    content = next(iter(saved.values()))

    assert "[PreviewRef: session://preview/abc]" in content
    assert "[ExpandedPreviewRef:" not in content


def test_appended_user_content_is_preserved():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('x')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('x')\nx\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
        {
            "role": "user",
            "content": ">>> emit('Done', release=True)\nDone\nNext task",
            "_stdout": ">>> emit('Done', release=True)\nDone\nNext task",
            "_user_content": "Next task",
        },
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1)
    assert any(m["role"] == "user" and m.get("_user_content") == "Next task" for m in projected)
    coalesced = next(m for m in projected if m.get("_coalesced"))
    assert "Next task" not in coalesced["content"]


def test_release_output_is_preserved_live_not_saved_to_preview():
    saved = {}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('x')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('x')\nx\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
        {"role": "user", "content": ">>> emit('Done', release=True)\nDone\nextra line\n"},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1, save_preview_blob=saved.setdefault)
    visible = "\n".join(m.get("content") or "" for m in projected)

    assert any(m["role"] == "assistant" and m["content"] == "emit('Done', release=True)" for m in projected)
    assert ">>> emit('Done', release=True)" in visible
    assert "Done\nextra line" in visible
    assert "extra line" not in next(iter(saved.values()))


def test_coalesced_messages_are_synthetic():
    messages = [{"role": "system", "content": "system"}] + interaction(0)

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1)
    coalesced = next(m for m in projected if m.get("_coalesced"))

    assert coalesced["_synthetic"] is True
    assert coalesced["_coalesced"] is True




def test_coalesce_is_idempotent_on_projected_messages():
    messages = [{"role": "system", "content": "system"}] + interaction(0)
    saved = {}

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
    )
    projected_again = coalesce_repl_messages(
        projected,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
    )

    assert projected_again == projected
    assert sum(1 for m in projected_again if m.get("_coalesced")) == 1


def test_release_detection_accepts_emit_value_metadata_when_final_result_is_none():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('x')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('x')\nx\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit(result, release=True)", "_final_result": None, "_emit_value": ""},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1)

    assert any(m.get("_coalesced") for m in projected)
    assert projected[-1]["role"] == "assistant"


def test_release_detection_finds_top_level_emit_after_non_emit_work():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('x')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('x')\nx\n" + ("y" * 2500)},
        {"role": "assistant", "content": "result = 'done'\nemit(result, release=True)"},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1)

    assert any(m.get("_coalesced") for m in projected)


def test_render_segment_input_is_preserved_as_real_user_input():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "displayed", "_render_segments": [{"type": "input", "content": "Segment task"}]},
        {"role": "assistant", "content": "print('x')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('x')\nx\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1)

    assert projected[1]["_render_segments"] == [{"type": "input", "content": "Segment task"}]
    assert any(m.get("_coalesced") for m in projected)


def test_omitted_echo_is_reconstructed_in_preview_blob():
    saved = {}
    code = "for i in range(2):\n    print(i)"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": code + "\n" + ("x" * 2500)},
        {"role": "user", "content": "[content omitted from echo]\n0\n1\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1, save_preview_blob=saved.setdefault)
    content = next(iter(saved.values()))

    assert ">>> for i in range(2):" in content
    assert "...     print(i)" in content


def test_attachment_placeholder_order_is_first_seen_and_deduplicated():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "work\n" + ("x" * 2500)},
        {
            "role": "user",
            "content": ">>> work\n[Attachment: b.py]\n[Attachment: a.py]\n[Attachment: b.py]\n" + ("y" * 2500),
            "_attachments": {"a.py": "a", "c.py": "c"},
            "_attachment_refs": {"d.py": "d"},
        },
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_savings_chars=1)
    coalesced = next(m for m in projected if m.get("_coalesced"))
    lines = coalesced["content"].splitlines()

    assert lines[1:5] == ["[Attachment: b.py]", "[Attachment: a.py]", "[Attachment: c.py]", "[Attachment: d.py]"]


def test_small_interaction_does_not_save_preview_blob():
    saved = {}
    messages = [{"role": "system", "content": "system"}] + interaction(0, work_size=1)

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
    )

    assert projected == messages
    assert saved == {}


def test_replay_then_coalesce_keeps_raw_events_unmodified_and_reapplies_projection(tmp_path):
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    class Conversation:
        def __init__(self):
            self.messages = [{"role": "system", "content": "system"}]

    class Agent:
        def __init__(self):
            self.conversation = Conversation()
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.expanded_preview_refs = self._expanded_preview_refs

    raw_messages = interaction(0)
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    for seq, msg in enumerate(raw_messages, start=1):
        store.append_event(session_id, seq, "message_added", {"message": msg})

    agent = Agent()
    replay_session_into_agent(agent, session_id, store)
    before_projection_events = store.get_events(session_id)
    assert not any(m.get("_coalesced") for m in agent.conversation.messages)

    projected = coalesce_repl_messages(
        agent.conversation.messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=lambda key, content: store.save_preview_blob(session_id, key, content),
    )

    assert any(m.get("_synthetic") and m.get("_coalesced") for m in projected)
    assert store.get_events(session_id) == before_projection_events

    replayed_agent = Agent()
    replay_session_into_agent(replayed_agent, session_id, store)
    projected_again = coalesce_repl_messages(
        replayed_agent.conversation.messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=lambda key, content: store.save_preview_blob(session_id, key, content),
    )

    assert projected_again == projected


def test_preview_expansion_event_survives_resume_replay_for_coalesced_preview(tmp_path):
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    class Conversation:
        def __init__(self):
            self.messages = [{"role": "system", "content": "system"}]

    class Agent:
        def __init__(self):
            self.conversation = Conversation()
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.expanded_preview_refs = self._expanded_preview_refs

    raw_messages = interaction(0)
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    seq = 1
    for msg in raw_messages:
        store.append_event(session_id, seq, "message_added", {"message": msg})
        seq += 1

    agent = Agent()
    replay_session_into_agent(agent, session_id, store)
    projected = coalesce_repl_messages(
        agent.conversation.messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=lambda key, content: store.save_preview_blob(session_id, key, content),
    )
    coalesced = next(m for m in projected if m.get("_coalesced"))
    uri = coalesced["content"].split("[PreviewRef: ", 1)[1].split("]", 1)[0]
    key = uri.rsplit("/", 1)[1]
    assert store.get_preview_blob(session_id, key)

    store.append_event(session_id, seq, "preview_expanded", {"uri": uri, "numbered": True})

    resumed = Agent()
    replay_session_into_agent(resumed, session_id, store)
    assert resumed._expanded_preview_refs == {uri: {"numbered": True}}

    projected_after_resume = coalesce_repl_messages(
        resumed.conversation.messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=lambda key, content: store.save_preview_blob(session_id, key, content),
    )
    assert any(uri in (m.get("content") or "") for m in projected_after_resume if m.get("_coalesced"))
    assert resumed._expanded_preview_refs == {uri: {"numbered": True}}


def test_coalesce_preserves_expanded_preview_ref_placeholder():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('x')\n" + ("x" * 2500)},
        {
            "role": "user",
            "content": (
                ">>> print('x')\n"
                "[PreviewRef: session://preview/keep]\n"
                "(1 lines, 10 chars)\n"
                "[/PreviewRef]\n"
                + ("y" * 2500)
            ),
        },
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        preserve_preview_refs={"session://preview/keep"},
    )

    coalesced = next(m for m in projected if m.get("_coalesced"))
    assert "[PreviewRef: session://preview/keep]" in coalesced["content"]


def test_actual_expanded_preview_refs_include_replayed_expanded_state(tmp_path):
    from code_agent.agent import CodeAgent
    from code_agent.session_store import SessionStore

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session(str(tmp_path), "model")
    key = "abc123"
    uri = f"session://preview/{key}"
    store.save_preview_blob(session_id, key, "expanded body")

    agent = CodeAgent()
    agent._ensure_setup()
    agent._session_store = store
    agent._session_id = session_id
    agent._expanded_preview_refs = {uri: {"numbered": False}}
    agent._configure_conversation(agent.conversation)
    agent.conversation.messages.append({
        "role": "user",
        "content": "[Assistant work and REPL output coalesced into preview]",
        "_synthetic": True,
        "_coalesced": True,
    })

    assert agent._actual_expanded_preview_refs() == [uri]
    assert agent._expanded_preview_context() == {uri: "expanded body"}
    assert agent._current_file_context_names() == [uri]






def test_resume_replay_leaves_path_attachment_refs_for_code_agent_worker_materialization(tmp_path):
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    class Conversation:
        def __init__(self):
            self.messages = [{"role": "system", "content": "system"}]

    class Agent:
        def __init__(self):
            self.conversation = Conversation()
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.expanded_preview_refs = self._expanded_preview_refs

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "rel.py").write_text("print('from repo')\n")
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session(str(repo), "model")
    store.append_event(session_id, 1, "message_added", {
        "message": {
            "role": "user",
            "content": "[Attachment: rel.py]",
            "_attachment_refs": {"rel.py": "rel.py"},
        }
    })

    agent = Agent()
    missing = replay_session_into_agent(agent, session_id, store)

    assert missing == []
    assert "_attachments" not in agent.conversation.messages[1]
    assert agent.conversation.messages[1]["_attachment_refs"] == {"rel.py": "rel.py"}





def test_pinned_turn_creates_auto_expanded_preview_ref():
    saved = {}
    auto_expand = []
    messages = [{"role": "system", "content": "system"}] + [
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('unpinned')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('unpinned')\nunpinned\n" + ("y" * 2500)},
        {"role": "assistant", "content": "print('important')", "_pinned_coalesce": {"label": "Pinned previous turn"}},
        {"role": "user", "content": ">>> print('important')\nimportant\n"},
        {"role": "assistant", "content": "emit('Done', release=True)"},
        {"role": "user", "content": ">>> emit('Done', release=True)\nDone\n"},
    ]

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
        auto_expand_preview_refs=auto_expand,
    )
    coalesced = next(m for m in projected if m.get("_coalesced"))

    assert coalesced["content"].count("[PreviewRef: session://preview/") == 2
    assert len(auto_expand) == 1
    pinned_key = auto_expand[0].rsplit("/", 1)[1]
    assert "important" in saved[pinned_key]
    assert "unpinned" not in saved[pinned_key]


def test_pinned_turn_without_repl_output_is_preserved():
    saved = {}
    auto_expand = []
    messages = [{"role": "system", "content": "system"}] + [
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "think('important')", "_pinned_coalesce": {"label": "Pinned previous turn"}},
        {"role": "assistant", "content": "print('work')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('work')\nwork\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
        auto_expand_preview_refs=auto_expand,
    )

    assert len(auto_expand) == 1
    assert saved[auto_expand[0].rsplit("/", 1)[1]] == "think('important')"


def test_multiple_pinned_turns_keep_ordered_preview_sections():
    saved = {}
    auto_expand = []
    messages = [{"role": "system", "content": "system"}] + [
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('before')\n" + ("b" * 2500)},
        {"role": "user", "content": ">>> print('before')\nbefore\n"},
        {"role": "assistant", "content": "print('pin1')", "_pinned_coalesce": {"label": "Pinned previous turn"}},
        {"role": "user", "content": ">>> print('pin1')\npin1\n"},
        {"role": "assistant", "content": "print('middle')\n" + ("m" * 2500)},
        {"role": "user", "content": ">>> print('middle')\nmiddle\n"},
        {"role": "assistant", "content": "print('pin2')", "_pinned_coalesce": {"label": "Pinned previous turn"}},
        {"role": "user", "content": ">>> print('pin2')\npin2\n"},
        {"role": "assistant", "content": "emit('Done', release=True)"},
        {"role": "user", "content": ">>> emit('Done', release=True)\nDone\n"},
    ]

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
        auto_expand_preview_refs=auto_expand,
    )

    coalesced = next(m for m in projected if m.get("_coalesced"))
    uris = re.findall(r"session://preview/[0-9a-f]{16}", coalesced["content"])
    unique_uris = list(dict.fromkeys(uris))
    assert len(unique_uris) == 4
    assert len(auto_expand) == 2
    assert unique_uris[1] == auto_expand[0]
    assert unique_uris[3] == auto_expand[1]
    contents = [saved[uri.rsplit("/", 1)[1]] for uri in unique_uris]
    assert "before" in contents[0]
    assert contents[1].startswith("print('pin1')")
    assert "middle" in contents[2]
    assert contents[3].startswith("print('pin2')")


def chat_interaction(i):
    return [
        {"role": "user", "content": f"Question {i}", "_user_content": f"Question {i}"},
        {"role": "assistant", "content": f"emit('Answer {i}', release=True)"},
        {"role": "user", "content": f">>> emit('Answer {i}', release=True)\nAnswer {i}\n"},
    ]


def test_default_keeps_most_recent_execution_interaction_even_past_last_three():
    messages = [{"role": "system", "content": "system"}]
    for i in range(2):
        messages.extend(interaction(i))
    for i in range(3):
        messages.extend(chat_interaction(i))

    projected = coalesce_repl_messages(messages, min_savings_chars=1)

    assert sum(1 for m in projected if m.get("_coalesced")) == 1
    visible = "\n".join(m.get("content") or "" for m in projected)
    assert "Task 1" in visible
    assert "print(1)" in visible
    assert "x" * 600 in visible
    assert "Task 0" in visible
    assert "print(0)\n" in visible
    assert "x" * 600 not in projected[2]["content"]


def test_release_output_is_not_treated_as_next_interaction_start_for_execution_policy():
    messages = [{"role": "system", "content": "system"}]
    messages.extend(interaction(0))
    messages.extend(interaction(1))
    for i in range(3):
        messages.extend(chat_interaction(i))

    projected = coalesce_repl_messages(messages, min_savings_chars=1)

    visible = "\n".join(m.get("content") or "" for m in projected)
    assert sum(1 for m in projected if m.get("_coalesced")) == 1
    assert "Task 1" in visible
    assert "print(1)" in visible
    assert "x" * 600 in visible


def test_virtual_interaction_boundary_counts_as_completed_interaction_on_both_sides():
    messages = [{"role": "system", "content": "system"}]
    messages.extend(interaction(0))
    messages.extend([
        {"role": "user", "content": "Long task", "_user_content": "Long task"},
        {"role": "assistant", "content": "print('step 1')\n" + ("a" * 2500)},
        {"role": "user", "content": ">>> print('step 1')\nstep 1\n" + ("b" * 2500)},
        {
            "role": "assistant",
            "content": "emit(None, release=True)",
            "_synthetic": True,
            "_virtual_interaction_boundary": True,
        },
    ])
    messages.extend(interaction(1))
    messages.extend(interaction(2))
    messages.extend(chat_interaction(0))
    messages.extend(chat_interaction(1))
    messages.extend(chat_interaction(2))

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=3,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
    )

    assert sum(1 for m in projected if m.get("_coalesced")) == 4
    visible = "\n".join(m.get("content") or "" for m in projected)
    assert "Long task" in visible
    assert "Task 1" in visible
    assert "Task 2" in visible
    assert "a" * 600 not in visible
    assert "b" * 600 not in visible
    assert "x" * 600 not in visible
    assert "y" * 600 not in visible
    assert any(m.get("_virtual_interaction_boundary") for m in projected)


def test_virtual_interaction_boundary_survives_replay_and_still_coalesces(tmp_path):
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    class Conversation:
        def __init__(self):
            self.messages = [{"role": "system", "content": "system"}]

    class Agent:
        def __init__(self):
            self.conversation = Conversation()
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.expanded_preview_refs = self._expanded_preview_refs

    raw_messages = [{"role": "system", "content": "system"}]
    raw_messages.extend(interaction(0))
    raw_messages.extend([
        {"role": "user", "content": "Long task", "_user_content": "Long task"},
        {"role": "assistant", "content": "print('step 1')\n" + ("a" * 2500)},
        {"role": "user", "content": ">>> print('step 1')\nstep 1\n" + ("b" * 2500)},
        {
            "role": "assistant",
            "content": "emit(None, release=True)",
            "_synthetic": True,
            "_virtual_interaction_boundary": True,
        },
    ])
    raw_messages.extend(chat_interaction(0))
    raw_messages.extend(chat_interaction(1))
    raw_messages.extend(chat_interaction(2))

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    for seq, msg in enumerate(raw_messages[1:], start=1):
        store.append_event(session_id, seq, "message_added", {"message": msg})

    agent = Agent()
    replay_session_into_agent(agent, session_id, store)

    assert any(m.get("_virtual_interaction_boundary") for m in agent.conversation.messages)

    projected = coalesce_repl_messages(
        agent.conversation.messages,
        keep_last_interactions=3,
        keep_last_execution_interactions=0,
        min_savings_chars=1,
    )

    assert sum(1 for m in projected if m.get("_coalesced")) == 2
    assert any(m.get("_virtual_interaction_boundary") for m in projected)
    visible = "\n".join(m.get("content") or "" for m in projected)
    assert "a" * 600 not in visible
    assert "b" * 600 not in visible




def test_preview_validates_summary_and_content():
    assert Preview(" summary ").content is None
    assert Preview("summary", "content").content == "content"

    for summary in ("", "   ", None):
        try:
            Preview(summary)
        except ValueError:
            pass
        else:
            raise AssertionError("blank or non-string summary was accepted")

    try:
        Preview("summary", 123)
    except ValueError:
        pass
    else:
        raise AssertionError("non-string preview content was accepted")


def test_message_source_range_prefers_synthetic_coverage():
    assert message_source_range({"_event_seq": 4}) == (4, 4)
    assert message_source_range({
        "_event_seq": 4,
        "_source_start_seq": 2,
        "_source_end_seq": 6,
    }) == (2, 6)
    assert message_source_range({"role": "system"}) is None


def test_place_preview_uses_explicit_content_and_does_not_mutate_input():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "raw", "_event_seq": 10},
        {
            "role": "user",
            "content": ">>> raw\noutput",
            "_event_seq": 11,
            "_attachments": {"a.py": "body"},
            "_attachment_refs": {"a.py": "a.py"},
        },
    ]
    original = [dict(message) for message in messages]
    saved = {}

    projected, key = place_preview(
        messages,
        Preview(
            "custom summary",
            "explicit [PreviewRef: session://preview/child]\nchild\n[/PreviewRef]",
        ),
        source_start_seq=10,
        source_end_seq=11,
        save_preview_blob=saved.setdefault,
    )

    content = "explicit [PreviewRef: session://preview/child]\nchild\n[/PreviewRef]"
    assert messages == original
    assert key == preview_key(content)
    assert saved == {key: content}
    replacement = projected[1]
    assert replacement["_source_start_seq"] == 10
    assert replacement["_source_end_seq"] == 11
    assert replacement["_attachments"] == {"a.py": "body"}
    assert replacement["_attachment_refs"] == {"a.py": "a.py"}
    assert "custom summary" in replacement["content"]


def test_place_preview_derives_content_from_current_projection():
    messages = [
        {"role": "assistant", "content": "print('x')", "_event_seq": 1},
        {"role": "user", "content": ">>> print('x')\nx", "_event_seq": 2},
    ]
    saved = {}

    projected, key = place_preview(
        messages,
        Preview("derived"),
        source_start_seq=1,
        source_end_seq=2,
        save_preview_blob=saved.setdefault,
    )

    assert saved[key] == "print('x')\n\n>>> print('x')\nx"
    assert projected[0]["_source_start_seq"] == 1
    assert projected[0]["_source_end_seq"] == 2


def test_preview_summary_does_not_affect_blob_key():
    messages = [{"role": "assistant", "content": "body", "_event_seq": 1}]
    saved = {}

    first, first_key = place_preview(
        messages,
        Preview("first", "same"),
        source_start_seq=1,
        source_end_seq=1,
        save_preview_blob=saved.setdefault,
    )
    second, second_key = place_preview(
        messages,
        Preview("second", "same"),
        source_start_seq=1,
        source_end_seq=1,
        save_preview_blob=saved.setdefault,
    )

    assert first_key == second_key
    assert "first" in first[0]["content"]
    assert "second" in second[0]["content"]
    assert saved == {first_key: "same"}


def test_place_preview_rejects_split_and_partial_overlap_boundaries():
    messages = [
        {
            "role": "user",
            "content": "child",
            "_source_start_seq": 100,
            "_source_end_seq": 150,
        },
        {"role": "assistant", "content": "later", "_event_seq": 160},
    ]

    for start, end in ((110, 140), (80, 120), (120, 160)):
        try:
            place_preview(
                messages,
                Preview("invalid"),
                source_start_seq=start,
                source_end_seq=end,
            )
        except PreviewBoundaryError as exc:
            assert f"[{start}, {end}]" in str(exc)
            assert "[100, 150]" in str(exc)
        else:
            raise AssertionError(f"invalid range [{start}, {end}] was accepted")


def test_recursive_parent_preview_preserves_child_ref_verbatim():
    canonical = [
        {"role": "assistant", "content": "child work", "_event_seq": 1},
        {"role": "user", "content": ">>> child work\nresult", "_event_seq": 2},
        {"role": "assistant", "content": "parent tail", "_event_seq": 3},
    ]
    saved = {}

    with_child, child_key = place_preview(
        canonical,
        Preview("child summary"),
        source_start_seq=1,
        source_end_seq=2,
        save_preview_blob=saved.setdefault,
    )
    child_ref = with_child[0]["content"].split("\n\n", 1)[1]
    with_parent, parent_key = place_preview(
        with_child,
        Preview("parent summary"),
        source_start_seq=1,
        source_end_seq=3,
        save_preview_blob=saved.setdefault,
    )

    assert child_ref in saved[parent_key]
    assert f"session://preview/{child_key}" in saved[parent_key]
    assert with_parent[0]["_source_start_seq"] == 1
    assert with_parent[0]["_source_end_seq"] == 3


def test_disjoint_preview_placements_work():
    messages = [
        {"role": "assistant", "content": "one", "_event_seq": 1},
        {"role": "assistant", "content": "two", "_event_seq": 2},
        {"role": "assistant", "content": "three", "_event_seq": 3},
    ]

    projected, _ = place_preview(
        messages,
        Preview("first"),
        source_start_seq=1,
        source_end_seq=1,
    )
    projected, _ = place_preview(
        projected,
        Preview("third"),
        source_start_seq=3,
        source_end_seq=3,
    )

    assert message_source_range(projected[0]) == (1, 1)
    assert message_source_range(projected[1]) == (2, 2)
    assert message_source_range(projected[2]) == (3, 3)


def test_place_preview_supports_multiple_ordered_previews_in_one_replacement():
    messages = [
        {"role": "assistant", "content": "one", "_event_seq": 1},
        {"role": "user", "content": ">>> one\n1", "_event_seq": 2},
    ]
    saved = {}

    projected, keys = place_preview(
        messages,
        [Preview("first", "first content"), Preview("second", "second content")],
        source_start_seq=1,
        source_end_seq=2,
        save_preview_blob=saved.setdefault,
    )

    assert keys == [preview_key("first content"), preview_key("second content")]
    assert list(saved) == keys
    content = projected[0]["content"]
    assert content.index("first") < content.index("second")
    assert message_source_range(projected[0]) == (1, 2)


def test_source_aware_placement_rejects_source_less_node_inside_span():
    messages = [
        {"role": "assistant", "content": "one", "_event_seq": 1},
        {"role": "user", "content": "legacy output"},
        {"role": "assistant", "content": "three", "_event_seq": 3},
    ]

    try:
        place_preview(
            messages,
            Preview("invalid"),
            source_start_seq=1,
            source_end_seq=3,
        )
    except PreviewBoundaryError as exc:
        assert "source-less projected node" in str(exc)
        assert "[1, 3]" in str(exc)
    else:
        raise AssertionError("source-less interior node was silently absorbed")


def test_source_aware_placement_rejects_duplicate_boundaries_and_ranges():
    cases = [
        [
            {"role": "assistant", "content": "first", "_event_seq": 1},
            {"role": "assistant", "content": "duplicate start", "_source_start_seq": 1, "_source_end_seq": 2},
            {"role": "assistant", "content": "last", "_event_seq": 3},
        ],
        [
            {"role": "assistant", "content": "first", "_event_seq": 1},
            {"role": "assistant", "content": "duplicate end", "_source_start_seq": 2, "_source_end_seq": 3},
            {"role": "assistant", "content": "last", "_event_seq": 3},
        ],
        [
            {"role": "assistant", "content": "same", "_source_start_seq": 1, "_source_end_seq": 3},
            {"role": "assistant", "content": "same again", "_source_start_seq": 1, "_source_end_seq": 3},
        ],
    ]

    for messages in cases:
        try:
            place_preview(
                messages,
                Preview("invalid"),
                source_start_seq=1,
                source_end_seq=3,
            )
        except PreviewBoundaryError as exc:
            assert "ambiguous projected" in str(exc)
        else:
            raise AssertionError("duplicate source boundary was accepted")


def test_production_skips_mixed_provenance_candidate_without_saving():
    saved = {}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "request", "_event_seq": 1},
        {"role": "assistant", "content": "work " * 300, "_event_seq": 2},
        {"role": "user", "content": ">>> work\n" + ("output " * 200)},
        {
            "role": "assistant",
            "content": "emit('done', release=True)",
            "_final_result": "done",
            "_event_seq": 4,
        },
    ]

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
        save_preview_blob=saved.setdefault,
    )

    assert projected == messages
    assert saved == {}


def test_production_skips_invalid_duplicate_source_candidate_safely():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "request", "_event_seq": 1},
        {"role": "assistant", "content": "work " * 300, "_event_seq": 2},
        {
            "role": "user",
            "content": ">>> work\n" + ("output " * 200),
            "_source_start_seq": 2,
            "_source_end_seq": 3,
        },
        {
            "role": "assistant",
            "content": "emit('done', release=True)",
            "_final_result": "done",
            "_event_seq": 4,
        },
    ]

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    )

    assert projected == messages


def test_production_normal_pinned_normal_uses_one_ordered_shared_placement(monkeypatch):
    calls = []
    original = coalesce_module.place_preview

    def recording_place_preview(messages, previews, **kwargs):
        calls.append((list(previews), kwargs))
        return original(messages, previews, **kwargs)

    monkeypatch.setattr(coalesce_module, "place_preview", recording_place_preview)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "request", "_event_seq": 1},
        {"role": "assistant", "content": "normal before " * 200, "_event_seq": 2},
        {"role": "user", "content": ">>> before\n" + ("before output " * 100), "_event_seq": 3},
        {
            "role": "assistant",
            "content": "print('pinned')",
            "_pinned_coalesce": {"label": "Pinned previous turn"},
            "_event_seq": 4,
        },
        {"role": "user", "content": ">>> print('pinned')\npinned", "_event_seq": 5},
        {"role": "assistant", "content": "normal after " * 200, "_event_seq": 6},
        {"role": "user", "content": ">>> after\n" + ("after output " * 100), "_event_seq": 7},
        {
            "role": "assistant",
            "content": "emit('done', release=True)",
            "_final_result": "done",
            "_event_seq": 8,
        },
    ]

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    )

    assert len(calls) == 1
    previews, kwargs = calls[0]
    assert len(previews) == 3
    assert [preview.content for preview in previews] == [
        coalesce_module._preview_content(messages[2:4], None, None),
        coalesce_module._preview_content(messages[4:6], None, None),
        coalesce_module._preview_content(messages[6:8], None, None),
    ]
    assert kwargs["source_start_seq"] == 2
    assert kwargs["source_end_seq"] == 7
    coalesced = next(message for message in projected if message.get("_coalesced"))
    refs = re.findall(r"session://preview/[0-9a-f]{16}", coalesced["content"])
    assert refs == [
        f"session://preview/{preview_key(preview.content)}"
        for preview in previews
    ]




def test_coalesce_skips_release_normalization_with_nonmonotonic_event_provenance():
    saved = {}
    release_text = "done " * 400
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": "...",
            "_user_content": "...",
            "_event_seq": 4452,
        },
        {
            "role": "assistant",
            "content": "work before release " * 200,
            "_event_seq": 4453,
        },
        {
            "role": "user",
            "content": ">>> work before release\n" + ("output " * 300),
            "_event_seq": 4454,
        },
        {
            "role": "assistant",
            "content": "more work " * 200,
            "_event_seq": 4456,
        },
        {
            "role": "user",
            "content": ">>> more work\n" + ("more output " * 300),
            "_event_seq": 4457,
        },
        {
            "role": "assistant",
            "content": "emit(result, release=True)",
            "_final_result": release_text,
            "_event_seq": 4459,
        },
        {
            "role": "user",
            "content": (
                ">>> final work\n"
                + ("final output " * 300)
                + "\n>>> emit(result, release=True)\n"
                + release_text
                + "\nNext request"
            ),
            "_stdout": (
                ">>> final work\n"
                + ("final output " * 300)
                + "\n>>> emit(result, release=True)\n"
                + release_text
            ),
            "_user_content": "Next request",
            "_event_seq": 4462,
        },
    ]

    normalized = coalesce_module._split_repl_messages_with_appended_user(messages)
    assert [
        message_source_range(message)
        for message in normalized[2:8]
    ] == [
        (4453, 4453),
        (4454, 4454),
        (4456, 4456),
        (4457, 4457),
        (4462, 4462),
        (4459, 4459),
    ]

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
        save_preview_blob=saved.setdefault,
    )

    assert not any(message.get("_coalesced") for message in projected)
    assert saved == {}
    assert any(
        message.get("_event_seq") == 4459
        and message.get("role") == "assistant"
        for message in projected
    )
    assert any(
        message.get("_event_seq") == 4462
        and "final output" in (message.get("content") or "")
        for message in projected
    )
def test_create_persisted_preview_and_recursive_parent(tmp_path):
    from code_agent.code_agent_coalesce import Preview, PersistedPreviewState, create_persisted_preview
    from code_agent.session_store import SessionStore

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "one", "_event_seq": 1},
        {"role": "user", "content": "two", "_event_seq": 2},
    ]
    store.append_event(session_id, 1, "message_added", {"message": {"role": "assistant", "content": "one"}})
    store.append_event(session_id, 2, "message_added", {"message": {"role": "user", "content": "two"}})
    state = PersistedPreviewState.empty()
    child_projection, child_key, child_seq, _ = create_persisted_preview(
        messages, Preview("child summary"),
        source_start_seq=1, source_end_seq=1,
        store=store, session_id=session_id, state=state,
    )
    parent_projection, parent_key, parent_seq, _ = create_persisted_preview(
        child_projection, Preview("parent summary"),
        source_start_seq=1, source_end_seq=2,
        store=store, session_id=session_id, state=state,
    )

    assert (child_seq, parent_seq) == (3, 5)
    assert f"session://preview/{child_key}" in store.get_preview_blob(session_id, parent_key)
    assert f"session://preview/{parent_key}" in parent_projection[1]["content"]
    events = store.get_events(session_id)
    assert events[2]["payload"] == {"preview_key": child_key, "summary": "child summary"}
    assert events[3]["payload"] == {
        "preview_event_seq": child_seq,
        "source_start_seq": 1,
        "source_end_seq": 1,
    }


def test_same_blob_can_have_distinct_persisted_summaries(tmp_path):
    from code_agent.code_agent_coalesce import Preview, PersistedPreviewState, create_persisted_preview
    from code_agent.session_store import SessionStore

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "same", "_event_seq": 1},
        {"role": "assistant", "content": "same", "_event_seq": 2},
    ]
    store.append_event(session_id, 1, "message_added", {"message": {"role": "assistant", "content": "same"}})
    store.append_event(session_id, 2, "message_added", {"message": {"role": "assistant", "content": "same"}})
    state = PersistedPreviewState.empty()
    projected, first_key, first_seq, _ = create_persisted_preview(
        messages, Preview("first"), source_start_seq=1, source_end_seq=1,
        store=store, session_id=session_id, state=state,
    )
    _, second_key, second_seq, _ = create_persisted_preview(
        projected, Preview("second"), source_start_seq=2, source_end_seq=2,
        store=store, session_id=session_id, state=state,
    )

    assert first_key == second_key
    assert first_seq != second_seq
    created = [e for e in store.get_events(session_id) if e["event_type"] == "preview_created"]
    assert [e["payload"]["summary"] for e in created] == ["first", "second"]


def test_code_agent_base_create_persisted_preview_installs_only_committed_state(tmp_path):
    from code_agent.agent import CodeAgentBase
    from code_agent.code_agent_coalesce import Preview, PersistedPreviewState
    from code_agent.conversation import Conversation
    from code_agent.session_store import SessionStore

    class Client:
        pass

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.append_event(session_id, 1, "message_added", {"message": {"role": "assistant", "content": "one"}})
    store.append_event(session_id, 2, "message_added", {"message": {"role": "assistant", "content": "two"}})

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._session_store = store
    agent._session_id = session_id
    agent._next_event_seq = 3
    agent._persisted_preview_state = PersistedPreviewState.empty()
    agent._pending_session_events = []
    agent._suspend_persistence = False
    agent._conversation = Conversation(Client(), "system")
    agent.conversation.messages.extend([
        {"role": "assistant", "content": "one", "_event_seq": 1},
        {"role": "assistant", "content": "two", "_event_seq": 2},
    ])
    agent._ensure_live_session = lambda: None
    agent._flush_pending_session_events = lambda: None

    key, created_seq = agent.create_persisted_preview(
        Preview("summary"), source_start_seq=1, source_end_seq=1,
    )

    assert created_seq == 3
    assert agent._next_event_seq == 5
    assert agent._persisted_preview_state.active_placements == {(1, 1): 3}
    assert f"session://preview/{key}" in agent.conversation.messages[1]["content"]

    before_messages = list(agent.conversation.messages)
    before_state = dict(agent._persisted_preview_state.active_placements)
    before_events = store.get_events(session_id)
    agent._next_event_seq = 3
    with pytest.raises(ValueError, match="stale preview event sequence"):
        agent.create_persisted_preview(
            Preview("second"), source_start_seq=2, source_end_seq=2,
        )
    assert agent.conversation.messages == before_messages
    assert agent._persisted_preview_state.active_placements == before_state
    assert store.get_events(session_id) == before_events



def test_actual_agent_replay_then_deterministic_coalescing_keeps_persisted_node_atomic(tmp_path):
    from code_agent.agent import CodeAgentBase
    from code_agent.code_agent_coalesce import Preview, PersistedPreviewState, create_persisted_preview
    from code_agent.conversation import Conversation
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    class Client:
        pass

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    raw = [
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "work " * 600},
        {"role": "user", "content": ">>> work\n" + ("output " * 600)},
        {"role": "assistant", "content": "emit('done', release=True)"},
    ]
    for seq, message in enumerate(raw, 1):
        store.append_event(session_id, seq, "message_added", {"message": message})
    canonical = [{"role": "system", "content": "system"}] + [
        dict(message, _event_seq=seq) for seq, message in enumerate(raw, 1)
    ]
    _, key, _, _ = create_persisted_preview(
        canonical, Preview("persisted"), source_start_seq=2, source_end_seq=3,
        store=store, session_id=session_id, expected_next_seq=5,
        state=PersistedPreviewState.empty(),
    )

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._conversation = Conversation(Client(), "system")
    agent._expanded_preview_refs = {}
    agent._session_store = store
    agent._session_id = session_id
    agent.code_agent_coalesce_keep_last_execution_interactions = 0
    agent.code_agent_coalesce_min_savings_chars = 0
    replay_session_into_agent(agent, session_id, store)
    before_events = store.get_events(session_id)
    agent._coalesce_context(protect_last_interactions=False)

    persisted = [message for message in agent.conversation.messages if message.get("_persisted_preview")]
    assert len(persisted) == 1
    assert persisted[0]["_source_start_seq"] == 2
    assert persisted[0]["_source_end_seq"] == 3
    assert f"session://preview/{key}" in persisted[0]["content"]
    assert store.get_events(session_id) == before_events


def test_module_persistence_requires_complete_authoritative_state(tmp_path):
    from code_agent.code_agent_coalesce import (
        Preview,
        PersistedPreviewState,
        PreviewPlacementError,
        create_persisted_preview,
    )
    from code_agent.session_store import SessionStore

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.append_event(session_id, 1, "message_added", {"message": {"role": "assistant", "content": "one"}})
    store.append_event(session_id, 2, "message_added", {"message": {"role": "assistant", "content": "two"}})
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "one", "_event_seq": 1},
        {"role": "assistant", "content": "two", "_event_seq": 2},
    ]

    before = store.get_events(session_id)
    with pytest.raises(TypeError, match="complete persisted preview state"):
        create_persisted_preview(
            messages,
            Preview("omitted"),
            source_start_seq=1,
            source_end_seq=1,
            store=store,
            session_id=session_id,
            state=None,
        )
    assert store.get_events(session_id) == before

    state = PersistedPreviewState.empty()
    projected, _, _, _ = create_persisted_preview(
        messages,
        Preview("first"),
        source_start_seq=1,
        source_end_seq=1,
        store=store,
        session_id=session_id,
        state=state,
    )
    committed = store.get_events(session_id)
    with store._connect() as conn:
        committed_keys = [
            row["key"]
            for row in conn.execute(
                "SELECT key FROM session_preview_blobs WHERE session_id = ? ORDER BY key",
                (session_id,),
            ).fetchall()
        ]

    stale = PersistedPreviewState.empty()
    with pytest.raises(ValueError, match="stale persisted preview state"):
        create_persisted_preview(
            projected,
            Preview("disjoint with stale state"),
            source_start_seq=2,
            source_end_seq=2,
            store=store,
            session_id=session_id,
            state=stale,
        )
    assert store.get_events(session_id) == committed
    with store._connect() as conn:
        assert [
            row["key"]
            for row in conn.execute(
                "SELECT key FROM session_preview_blobs WHERE session_id = ? ORDER BY key",
                (session_id,),
            ).fetchall()
        ] == committed_keys
    assert stale.definitions == {}
    assert stale.active_placements == {}

    with pytest.raises(PreviewPlacementError, match="conflict"):
        create_persisted_preview(
            projected,
            Preview("equal-range conflict"),
            source_start_seq=1,
            source_end_seq=1,
            store=store,
            session_id=session_id,
            state=state,
        )
    assert store.get_events(session_id) == committed


def test_rewind_across_exec_then_live_create_replays_identically(tmp_path):
    from code_agent.agent import CodeAgentBase
    from code_agent.code_agent_coalesce import Preview
    from code_agent.conversation import Conversation
    from code_agent.session_replay import replay_session_into_agent
    from code_agent.session_store import SessionStore

    class Client:
        pass

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.append_event(session_id, 1, "message_added", {
        "message": {"role": "assistant", "content": "pre-exec"}
    })
    store.append_event(session_id, 2, "exec", {})
    store.append_event(session_id, 3, "message_added", {
        "message": {"role": "assistant", "content": "post-exec"}
    })
    store.append_event(session_id, 4, "rewind", {"target_seq": 1})

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._conversation = Conversation(Client(), "system")
    agent._expanded_preview_refs = {}
    agent._session_store = store
    agent._session_id = session_id
    agent._next_event_seq = 5
    agent._pending_session_events = []
    agent._suspend_persistence = False
    agent._ensure_live_session = lambda: None
    agent._flush_pending_session_events = lambda: None

    replay_session_into_agent(agent, session_id, store)
    assert agent._persisted_preview_state.exec_start_seq == 0
    assert agent.conversation.messages == [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "pre-exec", "_event_seq": 1},
    ]

    key, created_seq = agent.create_persisted_preview(
        Preview("restored branch"),
        source_start_seq=1,
        source_end_seq=1,
    )
    assert created_seq == 5
    live_messages = agent.conversation.messages
    live_state = agent._persisted_preview_state

    resumed = CodeAgentBase.__new__(CodeAgentBase)
    resumed._conversation = Conversation(Client(), "system")
    resumed._expanded_preview_refs = {}
    replay_session_into_agent(resumed, session_id, store)

    assert resumed.conversation.messages == live_messages
    assert resumed._persisted_preview_state == live_state
    assert resumed._persisted_preview_state.exec_start_seq == 0
    assert resumed._persisted_preview_state.active_placements == {(1, 1): 5}
    assert f"session://preview/{key}" in resumed.conversation.messages[1]["content"]

    before = store.get_events(session_id)
    with pytest.raises(Exception):
        agent.create_persisted_preview(
            Preview("cross boundary"),
            source_start_seq=0,
            source_end_seq=1,
        )
    assert store.get_events(session_id) == before