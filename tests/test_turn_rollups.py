import copy
import inspect
import json
from types import SimpleNamespace

import pytest

from code_agent.code_agent_coalesce import coalesce_repl_messages
from code_agent.code_agent_coalesce import message_source_range
from code_agent.conversation import Conversation
from code_agent.session_replay import replay_session_into_agent
from code_agent.persisted_preview_state import PersistedPreviewState
from code_agent.session_store import SessionStore, utc_now_iso
from code_agent.turn_rollups import (
    CompletedTurn,
    completed_turns,
    eligible_rollup_line,
    eligible_rollup_units,
    render_turn_labels,
    rollup_units,
)


def event(seq, message):
    return {
        "seq": seq,
        "event_type": "message_added",
        "payload": {"message": message},
    }


def append_raw_event(store, session_id, seq, event_type, payload):
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO session_events(
                session_id, seq, created_at, event_type, payload_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                seq,
                utc_now_iso(),
                event_type,
                json.dumps(payload),
            ),
        )
        connection.commit()


def input_message(text):
    return {
        "role": "user",
        "content": text,
        "_user_content": text,
        "_render_segments": [{"type": "input", "content": text}],
    }


def output_message(text=">>> print('x')\nx\n"):
    return {
        "role": "user",
        "content": text,
        "_stdout": text,
        "_render_segments": [{"type": "stdout", "content": text}],
    }


def release_message(text="done"):
    return {
        "role": "assistant",
        "content": f"emit({text!r}, release=True)",
    }


def completed_events(count, execution_ids=(), release_outputs=()):
    events = []
    seq = 1
    ids = []
    for index in range(count):
        turn_id = seq
        ids.append(turn_id)
        events.append(event(seq, input_message(f"request {index}")))
        seq += 1
        if turn_id in execution_ids:
            events.append(event(seq, {"role": "assistant", "content": "print('x')"}))
            seq += 1
            events.append(event(seq, output_message()))
            seq += 1
        events.append(event(seq, release_message(f"done {index}")))
        seq += 1
        if turn_id in release_outputs:
            events.append(event(seq, output_message(f">>> emit('done {index}', release=True)\ndone {index}\n")))
            seq += 1
    return events, ids


def projected_messages(events):
    messages = [{"role": "system", "content": "system"}]
    for item in events:
        if item["event_type"] != "message_added":
            continue
        message = copy.deepcopy(item["payload"]["message"])
        message["_event_seq"] = item["seq"]
        for segment in message.get("_render_segments") or []:
            segment["_event_seq"] = item["seq"]
        messages.append(message)
    return messages


def test_turn_label_is_provider_only_stable_and_not_duplicated():
    stored = input_message("hello")
    stored["_event_seq"] = 17
    stored["_render_segments"][0]["_event_seq"] = 17
    original = copy.deepcopy(stored)

    first = render_turn_labels(stored)
    second = render_turn_labels(first)

    assert first["content"] == "# Turn 17\n\nhello"
    assert second["content"] == first["content"]
    assert stored == original


def test_turn_label_handles_appended_input_segment_and_excludes_non_turns():
    combined = {
        "role": "user",
        "content": ">>> print('x')\nx\nnext request",
        "_event_seq": 9,
        "_render_segments": [
            {"type": "stdout", "content": ">>> print('x')\nx\n", "_event_seq": 8},
            {"type": "input", "content": "next request", "_event_seq": 9},
        ],
    }
    assert render_turn_labels(combined)["content"] == (
        ">>> print('x')\nx\n# Turn 9\n\nnext request"
    )
    repeated_text = {
        "role": "user",
        "content": "same\nsame",
        "_render_segments": [
            {"type": "stdout", "content": "same\n", "_event_seq": 8},
            {"type": "input", "content": "same", "_event_seq": 9},
        ],
    }
    assert render_turn_labels(repeated_text)["content"] == "same\n# Turn 9\n\nsame"
    assert render_turn_labels(output_message())["content"].startswith(">>>")
    synthetic = dict(input_message("hidden"), _synthetic=True, _event_seq=10)
    synthetic["_render_segments"][0]["_event_seq"] = 10
    assert render_turn_labels(synthetic)["content"] == "hidden"


def test_conversation_labels_active_turn_on_first_call_without_mutation():
    conversation = Conversation(None, "system")
    message = input_message("active")
    message["_event_seq"] = 23
    message["_render_segments"][0]["_event_seq"] = 23
    conversation.messages.append(message)
    conversation.message_projector = render_turn_labels

    assert conversation._messages()[-1]["content"] == "# Turn 23\n\nactive"
    assert conversation.messages[-1]["content"] == "active"
    assert conversation._messages()[-1]["content"] == "# Turn 23\n\nactive"


def test_completed_turn_ranges_use_release_output_or_release_assistant():
    from code_agent.turn_rollups import completed_turns

    events, ids = completed_events(2, execution_ids={1}, release_outputs={1})
    turns = completed_turns(events)

    assert turns == [
        CompletedTurn(ids[0], ids[0], 5, True),
        CompletedTurn(ids[1], ids[1], 7, False),
    ]


def test_incomplete_cross_exec_and_ambiguous_turns_are_omitted():
    from code_agent.turn_rollups import completed_turns

    events, ids = completed_events(1)
    events.extend([
        event(3, input_message("incomplete")),
        {"seq": 4, "event_type": "exec", "payload": {}},
        event(5, {
            "role": "user",
            "content": "a\nb",
            "_render_segments": [
                {"type": "input", "content": "a"},
                {"type": "input", "content": "b"},
            ],
        }),
        event(6, release_message()),
        event(7, input_message("current")),
    ])
    assert completed_turns(events) == []


def test_recent_and_latest_execution_turns_are_protected():
    events, ids = completed_events(6, execution_ids={1})
    units = eligible_rollup_units(
        events,
        projected_messages(events),
        PersistedPreviewState.empty(),
    )

    assert [unit.turn_ids for unit in units] == [(ids[1],), (ids[2],)]


def test_deterministic_coalescing_preserves_older_eligible_units():
    events, ids = completed_events(8, execution_ids={1, 5, 9, 13, 17, 21, 25, 29})
    for item in events:
        message = item["payload"]["message"]
        if message.get("role") == "user" and message.get("_stdout"):
            text = "large execution output " * 200
            message["content"] = text
            message["_stdout"] = text
            message["_render_segments"] = [{"type": "stdout", "content": text}]
    projection = coalesce_repl_messages(
        projected_messages(events),
        keep_last_interactions=3,
        keep_last_execution_interactions=1,
        min_savings_chars=0,
    )

    assert any(
        message.get("_synthetic")
        and message.get("_coalesced")
        and not message.get("_persisted_preview")
        for message in projection
    )
    eligible = eligible_rollup_units(
        events,
        projection,
        PersistedPreviewState.empty(),
    )
    assert [unit.turn_ids for unit in eligible] == [
        (ids[0],),
        (ids[1],),
        (ids[2],),
        (ids[3],),
        (ids[4],),
    ]
    assert eligible_rollup_line(eligible) == (
        "Eligible rollup turns: " + ", ".join(str(turn_id) for turn_id in ids[:5])
    )


def test_active_child_is_atomic_and_eligibility_line_is_single_ordered_line():
    events, ids = completed_events(8)
    projection = projected_messages(events)
    from code_agent.code_agent_coalesce import make_preview_replacement, replace_projected_span, select_projected_span

    turns = __import__("code_agent.turn_rollups", fromlist=["completed_turns"]).completed_turns(events)
    child_turns = turns[1:3]
    start = child_turns[0].source_start_seq
    end = child_turns[-1].source_end_seq
    span = select_projected_span(projection, source_start_seq=start, source_end_seq=end)
    replacement = make_preview_replacement(
        ["[PreviewRef: session://preview/child]\nchild\n[/PreviewRef]"],
        projection[span.start_index:span.end_index],
        source_start_seq=start,
        source_end_seq=end,
    )
    replacement["_persisted_preview"] = True
    replacement["_preview_event_seq"] = 100
    projection = replace_projected_span(projection, span, replacement)
    state = PersistedPreviewState(
        definitions={100: ("child", "child")},
        active_placements={(start, end): 100},
    )

    units = rollup_units(events, projection, state)
    assert units[1].turn_ids == (ids[1], ids[2])
    eligible = eligible_rollup_units(events, projection, state)
    assert eligible_rollup_line(eligible) == (
        f"Eligible rollup turns: {ids[0]}, {ids[1]}-{ids[2]}, {ids[3]}, {ids[4]}"
    )
    assert eligible_rollup_line([]) == ""


def test_structured_stdout_is_not_a_turn_and_counts_as_execution():
    from code_agent.turn_rollups import completed_turns

    events = [
        event(1, input_message("request")),
        event(2, {"role": "assistant", "content": "print('x')"}),
        event(3, {
            "role": "user",
            "content": "# [no output]",
            "_render_segments": [{"type": "stdout", "content": "# [no output]"}],
        }),
        event(4, release_message()),
        event(5, {
            "role": "user",
            "content": "plain output",
            "_render_segments": [{"type": "stdout", "content": "plain output"}],
        }),
        event(6, release_message("unrelated")),
    ]

    assert completed_turns(events) == [CompletedTurn(1, 1, 5, True)]
    projection = projected_messages(events)
    assert render_turn_labels(projection[3])["content"] == "# [no output]"
    assert render_turn_labels(projection[5])["content"] == "plain output"


def test_rollup_units_require_exact_canonical_projection_coverage():
    events, ids = completed_events(1, execution_ids={1})
    projection = projected_messages(events)
    state = PersistedPreviewState.empty()
    assert [unit.turn_ids for unit in rollup_units(events, projection, state)] == [(ids[0],)]

    missing_middle = [
        message for message in projection
        if message.get("_event_seq") != 2
    ]
    assert rollup_units(events, missing_middle, state) == []

    duplicated = projection[:3] + [copy.deepcopy(projection[2])] + projection[3:]
    assert rollup_units(events, duplicated, state) == []

    source_less = copy.deepcopy(projection)
    source_less.insert(2, {"role": "assistant", "content": "unknown"})
    assert rollup_units(events, source_less, state) == []

    extra = copy.deepcopy(projection)
    extra.insert(2, {"role": "assistant", "content": "extra", "_event_seq": 99})
    assert rollup_units(events, extra, state) == []

    mixed = copy.deepcopy(projection)
    mixed[2]["_source_start_seq"] = 2
    mixed[2]["_source_end_seq"] = 3
    assert rollup_units(events, mixed, state) == []

    fabricated = {
        "role": "user",
        "content": "arbitrary compressed content",
        "_source_start_seq": 1,
        "_source_end_seq": 4,
        "_synthetic": True,
        "_coalesced": True,
        "_render_segments": [
            {"type": "stdout", "content": "arbitrary compressed content"}
        ],
    }
    assert rollup_units(
        events,
        [{"role": "system", "content": "system"}, fabricated],
        state,
    ) == []


def test_deterministic_node_requires_exact_shared_production_derivation():
    events, ids = completed_events(1, execution_ids={1}, release_outputs={1})
    for item in events:
        message = item["payload"]["message"]
        if item["seq"] == 2:
            message["content"] = "print('x')\n" + ("assistant work " * 200)
        elif item["seq"] == 3:
            text = ">>> print('x')\n" + ("execution output " * 200)
            message["content"] = text
            message["_stdout"] = text
            message["_render_segments"] = [{"type": "stdout", "content": text}]
    canonical = projected_messages(events)
    projection = coalesce_repl_messages(
        canonical,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    )
    state = PersistedPreviewState.empty()
    node_index = next(
        index for index, message in enumerate(projection)
        if message.get("_coalesced")
    )
    exact = projection[node_index]
    assert message_source_range(exact) == (2, 3)
    assert [unit.turn_ids for unit in rollup_units(events, projection, state)] == [
        (ids[0],)
    ]

    mutations = []

    wrong_key = copy.deepcopy(projection)
    wrong_key[node_index]["content"] = wrong_key[node_index]["content"].replace(
        "session://preview/", "session://preview/wrong"
    )
    wrong_key[node_index]["_render_segments"][0]["content"] = wrong_key[node_index]["content"]
    mutations.append(wrong_key)

    wrong_summary = copy.deepcopy(projection)
    wrong_summary[node_index]["content"] = wrong_summary[node_index]["content"].replace(
        "(5 lines,", "(999 lines,"
    )
    wrong_summary[node_index]["_render_segments"][0]["content"] = wrong_summary[node_index]["content"]
    mutations.append(wrong_summary)

    extra_ref = copy.deepcopy(projection)
    extra_ref[node_index]["content"] += (
        "\n[PreviewRef: session://preview/extra]\nextra\n[/PreviewRef]"
    )
    extra_ref[node_index]["_render_segments"][0]["content"] = extra_ref[node_index]["content"]
    mutations.append(extra_ref)

    extra_payload = copy.deepcopy(projection)
    extra_payload[node_index]["content"] += "\nunrelated visible payload"
    extra_payload[node_index]["_render_segments"][0]["content"] = extra_payload[node_index]["content"]
    mutations.append(extra_payload)

    forged_attachments = copy.deepcopy(projection)
    forged_attachments[node_index]["_attachments"] = {"forged.py": "body"}
    mutations.append(forged_attachments)

    forged_refs = copy.deepcopy(projection)
    forged_refs[node_index]["_attachment_refs"] = {
        "forged.py": "session://preview/forged"
    }
    mutations.append(forged_refs)

    persisted_metadata = copy.deepcopy(projection)
    persisted_metadata[node_index]["_preview_event_seq"] = 99
    mutations.append(persisted_metadata)

    persisted_flag = copy.deepcopy(projection)
    persisted_flag[node_index]["_persisted_preview"] = True
    mutations.append(persisted_flag)

    for mutated in mutations:
        assert rollup_units(events, mutated, state) == []

    for impossible_range in [(1, 3), (2, 4), (2, 5), (1, 5)]:
        forged = copy.deepcopy(exact)
        forged["_source_start_seq"], forged["_source_end_seq"] = impossible_range
        impossible_projection = [
            canonical[0],
            forged,
            *[
                message
                for message in canonical[1:]
                if not (
                    impossible_range[0]
                    <= message.get("_event_seq", -1)
                    <= impossible_range[1]
                )
            ],
        ]
        impossible_projection.sort(
            key=lambda message: (
                -1 if message.get("role") == "system"
                else message_source_range(message)[0]
            )
        )
        assert rollup_units(events, impossible_projection, state) == []


def test_range_compression_requires_exact_active_persisted_placement():
    events, ids = completed_events(2)
    turns = __import__("code_agent.turn_rollups", fromlist=["completed_turns"]).completed_turns(events)
    start = turns[0].source_start_seq
    end = turns[-1].source_end_seq
    compressed = {
        "role": "user",
        "content": "compressed",
        "_source_start_seq": start,
        "_source_end_seq": end,
        "_synthetic": True,
        "_coalesced": True,
    }
    projection = [{"role": "system", "content": "system"}, compressed]

    empty = PersistedPreviewState.empty()
    state = PersistedPreviewState(
        definitions={20: ("child", "child")},
        active_placements={(start, end): 20},
    )
    assert rollup_units(events, projection, empty) == []
    assert rollup_units(events, projection, state) == []

    compressed["_persisted_preview"] = True
    assert rollup_units(events, projection, state) == []

    compressed["_preview_event_seq"] = 20
    compressed["content"] = "[PreviewRef: session://preview/child]\nchild\n[/PreviewRef]"
    units = rollup_units(events, projection, state)
    assert [unit.turn_ids for unit in units] == [(ids[0], ids[1])]


def test_attachment_invalidation_before_coalescing_remains_eligible():
    from code_agent.session_message_state import reduce_canonical_message_events

    events, ids = completed_events(5, execution_ids={1, 11})
    events[1]["payload"]["message"]["content"] = "assistant work " * 300
    events[1]["payload"]["message"]["_attachment_refs"] = {
        "keep.txt": "session://preview/keep",
        "stale.txt": "session://preview/stale",
    }
    events[1]["payload"]["message"]["_attachments"] = {
        "keep.txt": "keep body",
        "stale.txt": "stale body",
    }
    events[2]["payload"]["message"]["content"] = "execution output " * 300
    events[2]["payload"]["message"]["_stdout"] = "execution output " * 300
    events[2]["payload"]["message"]["_render_segments"] = [
        {"type": "stdout", "content": "execution output " * 300}
    ]
    events.insert(
        3,
        {
            "seq": 4,
            "event_type": "attachment_invalidated",
            "payload": {"name": "stale.txt"},
        },
    )
    for item in events[4:]:
        item["seq"] += 1

    canonical, _ = reduce_canonical_message_events(events)
    body = next(message for message in canonical if message.get("_event_seq") == 2)
    assert body["_attachments"] == {"keep.txt": "keep body"}
    assert body["_attachment_refs"] == {
        "keep.txt": "session://preview/keep"
    }

    projection = coalesce_repl_messages(
        [{"role": "system", "content": "system"}, *canonical],
        keep_last_interactions=3,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    )
    assert any(
        unit.turn_ids == (ids[0],)
        for unit in eligible_rollup_units(
            events,
            projection,
            PersistedPreviewState.empty(),
        )
    )


def test_materialized_attachment_identity_is_trusted_but_canonical_structure_is_rederived():
    events, ids = completed_events(5, execution_ids={1, 11})
    events[1]["payload"]["message"]["content"] = "assistant work " * 300
    events[2]["payload"]["message"]["content"] = "execution output " * 300
    events[2]["payload"]["message"]["_stdout"] = "execution output " * 300
    events[2]["payload"]["message"]["_render_segments"] = [
        {"type": "stdout", "content": "execution output " * 300}
    ]
    events[1]["payload"]["message"]["_attachment_refs"] = {
        "notes.txt": "session://preview/attachment"
    }
    live_projection = projected_messages(events)
    live_projection[2]["_attachments"] = {"notes.txt": "materialized body"}
    projection = coalesce_repl_messages(
        live_projection,
        keep_last_interactions=3,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    )
    state = PersistedPreviewState.empty()

    eligible = eligible_rollup_units(events, projection, state)
    assert eligible and eligible[0].turn_ids == (ids[0],)
    node_index = next(
        index
        for index, message in enumerate(projection)
        if message.get("_coalesced")
        and message_source_range(message) == (2, 3)
    )
    assert projection[node_index]["_attachments"] == {
        "notes.txt": "materialized body"
    }

    changed_body = copy.deepcopy(projection)
    changed_body[node_index]["_attachments"]["notes.txt"] = "forged body"
    assert not any(
        unit.turn_ids == (ids[0],)
        for unit in eligible_rollup_units(events, changed_body, state)
    )

    changed_ref = copy.deepcopy(projection)
    changed_ref[node_index]["_attachment_refs"]["notes.txt"] = (
        "session://preview/forged"
    )
    assert not any(
        unit.turn_ids == (ids[0],)
        for unit in eligible_rollup_units(events, changed_ref, state)
    )


def test_replay_materializes_attachments_then_coalesces_with_valid_identity(tmp_path):
    class ReplayAgent:
        def __init__(self):
            self.conversation = Conversation(None, "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.message_projector = render_turn_labels

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.save_preview_blob(session_id, "attachment", "attachment body")
    events, ids = completed_events(5, execution_ids={1, 11})
    events[1]["payload"]["message"]["content"] = "assistant work " * 300
    events[2]["payload"]["message"]["content"] = "execution output " * 300
    events[2]["payload"]["message"]["_stdout"] = "execution output " * 300
    events[2]["payload"]["message"]["_render_segments"] = [
        {"type": "stdout", "content": "execution output " * 300}
    ]
    events[1]["payload"]["message"]["_attachment_refs"] = {
        "notes.txt": "session://preview/attachment"
    }
    for item in events:
        store.append_event(
            session_id,
            item["seq"],
            item["event_type"],
            item["payload"],
        )

    agent = ReplayAgent()
    replay_session_into_agent(agent, session_id, store)
    agent.conversation.messages = coalesce_repl_messages(
        agent.conversation.messages,
        keep_last_interactions=3,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    )
    eligible = eligible_rollup_units(
        store.get_events(session_id),
        agent.conversation.messages,
        PersistedPreviewState.empty(),
    )

    assert eligible and eligible[0].turn_ids == (ids[0],)
    node = next(
        message
        for message in agent.conversation.messages
        if message.get("_coalesced")
    )
    assert "notes.txt" in node["_attachments"]


def _pinned_completed_events(extra_turns=4):
    events = [
        event(1, input_message("pinned task")),
        event(2, {"role": "assistant", "content": "normal before " * 200}),
        event(3, output_message(">>> before\n" + ("before " * 200))),
        event(4, {"role": "assistant", "content": "pinned middle " * 200}),
        {
            "seq": 5,
            "event_type": "message_pinned",
            "payload": {"message_event_seq": 4, "label": "Pinned middle"},
        },
        event(6, output_message(">>> pinned\n" + ("pinned " * 200))),
        event(7, {"role": "assistant", "content": "normal after " * 200}),
        event(8, output_message(">>> after\n" + ("after " * 200))),
        event(9, release_message()),
    ]
    ids = [1]
    seq = 10
    for index in range(extra_turns):
        ids.append(seq)
        events.append(event(seq, input_message(f"later {index}")))
        if index == extra_turns - 1:
            events.append(event(seq + 1, {"role": "assistant", "content": "print('recent')"}))
            events.append(event(seq + 2, output_message(">>> print('recent')\nrecent\n")))
            events.append(event(seq + 3, release_message()))
            seq += 4
        else:
            events.append(event(seq + 1, release_message()))
            seq += 2
    return events, ids


def test_message_pinned_transition_reconstructs_exact_eligible_sections():
    events, ids = _pinned_completed_events()
    from code_agent.session_message_state import reduce_canonical_message_events

    canonical, _ = reduce_canonical_message_events(events)
    projection = coalesce_repl_messages(
        [{"role": "system", "content": "system"}, *canonical],
        keep_last_interactions=3,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    )
    node = next(message for message in projection if message.get("_coalesced"))

    assert node["content"].count("[PreviewRef:") == 3
    assert any(
        unit.turn_ids == (ids[0],)
        for unit in eligible_rollup_units(
            events,
            projection,
            PersistedPreviewState.empty(),
        )
    )


def test_replay_message_pinned_sections_remain_eligible(tmp_path):
    class ReplayAgent:
        def __init__(self):
            self.conversation = Conversation(None, "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.message_projector = render_turn_labels

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    events, ids = _pinned_completed_events()
    for item in events:
        store.append_event(
            session_id,
            item["seq"],
            item["event_type"],
            item["payload"],
        )
    agent = ReplayAgent()
    replay_session_into_agent(agent, session_id, store)
    agent.conversation.messages = coalesce_repl_messages(
        agent.conversation.messages,
        keep_last_interactions=3,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    )

    assert any(
        unit.turn_ids == (ids[0],)
        for unit in eligible_rollup_units(
            store.get_events(session_id),
            agent.conversation.messages,
            PersistedPreviewState.empty(),
        )
    )


def test_attachment_invalidation_reducer_and_replay_equivalence(tmp_path):
    from code_agent.session_message_state import reduce_canonical_message_events

    message = {
        "role": "assistant",
        "content": "work",
        "_attachment_refs": {
            "keep.txt": "session://preview/keep",
            "stale.txt": "session://preview/stale",
        },
        "_attachments": {
            "keep.txt": "keep materialized",
            "stale.txt": "stale materialized",
        },
    }
    events = [
        event(1, message),
        {
            "seq": 2,
            "event_type": "attachment_invalidated",
            "payload": {"name": "stale.txt"},
        },
    ]
    reduced, _ = reduce_canonical_message_events(events)
    assert reduced[0]["_attachment_refs"] == {
        "keep.txt": "session://preview/keep"
    }
    assert reduced[0]["_attachments"] == {"keep.txt": "keep materialized"}

    class ReplayAgent:
        def __init__(self):
            self.conversation = Conversation(None, "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            pass

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.save_preview_blob(session_id, "keep", "keep body")
    store.save_preview_blob(session_id, "stale", "stale body")
    persisted_message = copy.deepcopy(message)
    persisted_message.pop("_attachments")
    for item in [event(1, persisted_message), events[1]]:
        store.append_event(
            session_id,
            item["seq"],
            item["event_type"],
            item["payload"],
        )

    agent = ReplayAgent()
    replay_session_into_agent(agent, session_id, store)
    replayed = agent.conversation.messages[1]
    assert replayed["_attachment_refs"] == reduced[0]["_attachment_refs"]
    assert set(replayed["_attachments"]) == set(reduced[0]["_attachments"])
    assert "stale.txt" not in replayed.get("_attachments", {})
    assert "stale.txt" not in replayed.get("_attachment_refs", {})


def test_attachment_invalidation_rewind_snapshots(tmp_path):
    from code_agent.session_message_state import reduce_canonical_message_events

    message = {
        "role": "assistant",
        "content": "work",
        "_attachment_refs": {"a.txt": "session://preview/a"},
        "_attachments": {"a.txt": "materialized"},
    }
    prefix = [
        event(1, message),
        {
            "seq": 2,
            "event_type": "attachment_invalidated",
            "payload": {"name": "a.txt"},
        },
    ]
    before_events = [
        *prefix,
        {"seq": 3, "event_type": "rewind", "payload": {"target_seq": 1}},
    ]
    after_events = [
        *prefix,
        {"seq": 3, "event_type": "rewind", "payload": {"target_seq": 2}},
    ]
    before, _ = reduce_canonical_message_events(before_events)
    after, _ = reduce_canonical_message_events(after_events)
    assert before[0]["_attachment_refs"] == {
        "a.txt": "session://preview/a"
    }
    assert before[0]["_attachments"] == {"a.txt": "materialized"}
    assert "_attachment_refs" not in after[0]
    assert "_attachments" not in after[0]

    class ReplayAgent:
        def __init__(self):
            self.conversation = Conversation(None, "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            pass

    for suffix, expected_present in [(before_events, True), (after_events, False)]:
        store = SessionStore(str(tmp_path / f"{expected_present}.db"))
        session_id = store.create_session("/repo", "model")
        store.save_preview_blob(session_id, "a", "body")
        persisted = copy.deepcopy(suffix)
        persisted[0]["payload"]["message"].pop("_attachments")
        for item in persisted:
            store.append_event(
                session_id,
                item["seq"],
                item["event_type"],
                item["payload"],
            )
        agent = ReplayAgent()
        replay_session_into_agent(agent, session_id, store)
        replayed = agent.conversation.messages[1]
        assert ("a.txt" in replayed.get("_attachment_refs", {})) is expected_present
        assert ("a.txt" in replayed.get("_attachments", {})) is expected_present


def test_attachment_invalidation_resume_and_fork_reconstruction(tmp_path):
    class ReplayAgent:
        def __init__(self):
            self.conversation = Conversation(None, "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            pass

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.save_preview_blob(session_id, "a", "body")
    store.append_event(
        session_id,
        1,
        "message_added",
        {
            "message": {
                "role": "assistant",
                "content": "work",
                "_attachment_refs": {"a.txt": "session://preview/a"},
            }
        },
    )
    store.append_event(
        session_id,
        2,
        "attachment_invalidated",
        {"name": "a.txt"},
    )

    for replay_id in [session_id, store.fork_session(session_id)]:
        agent = ReplayAgent()
        replay_session_into_agent(agent, replay_id, store)
        message = agent.conversation.messages[1]
        assert "_attachment_refs" not in message
        assert "_attachments" not in message


def test_malformed_attachment_invalidation_is_conservative_and_equivalent(tmp_path):
    from code_agent.session_message_state import reduce_canonical_message_events

    message = {
        "role": "assistant",
        "content": "work",
        "_attachment_refs": {"a.txt": "session://preview/a"},
        "_attachments": {"a.txt": "materialized"},
    }
    malformed_payloads = [
        {},
        {"name": None},
        {"name": 7},
        {"name": ""},
        [],
        "stale.txt",
    ]
    events = [event(1, message)]
    for seq, payload in enumerate(malformed_payloads, start=2):
        events.append(
            {
                "seq": seq,
                "event_type": "attachment_invalidated",
                "payload": payload,
            }
        )
    reduced, _ = reduce_canonical_message_events(events)
    assert reduced[0]["_attachment_refs"] == message["_attachment_refs"]
    assert reduced[0]["_attachments"] == message["_attachments"]

    class ReplayAgent:
        def __init__(self):
            self.conversation = Conversation(None, "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            pass

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.save_preview_blob(session_id, "a", "body")
    persisted_message = copy.deepcopy(message)
    persisted_message.pop("_attachments")
    store.append_event(
        session_id,
        1,
        "message_added",
        {"message": persisted_message},
    )
    for item in events[1:]:
        store.append_event(
            session_id,
            item["seq"],
            item["event_type"],
            item["payload"],
        )
    agent = ReplayAgent()
    replay_session_into_agent(agent, session_id, store)
    replayed = agent.conversation.messages[1]
    assert replayed["_attachment_refs"] == message["_attachment_refs"]
    assert "a.txt" in replayed["_attachments"]


def test_malformed_event_snapshots_match_reducer_replay_resume_and_fork(tmp_path):
    from code_agent.session_message_state import reduce_canonical_message_events

    message = {
        "role": "assistant",
        "content": "work",
        "_attachment_refs": {"a.txt": "session://preview/a"},
        "_attachments": {"a.txt": "materialized"},
    }
    events = [
        event(1, message),
        {
            "seq": 2,
            "event_type": "attachment_invalidated",
            "payload": "a.txt",
        },
        {
            "seq": 3,
            "event_type": "message_added",
            "payload": [],
        },
        {
            "seq": 4,
            "event_type": "message_pinned",
            "payload": "invalid",
        },
        {
            "seq": 5,
            "event_type": "rewind",
            "payload": "invalid",
        },
        {
            "seq": 6,
            "event_type": "exec",
            "payload": "invalid",
        },
        {
            "seq": 7,
            "event_type": "attachment_invalidated",
            "payload": {"name": "a.txt"},
        },
        {
            "seq": 8,
            "event_type": "rewind",
            "payload": {"target_seq": 6},
        },
    ]
    reduced, exec_start = reduce_canonical_message_events(events)
    assert exec_start == 0
    assert reduced[0]["_attachment_refs"] == {
        "a.txt": "session://preview/a"
    }
    assert reduced[0]["_attachments"] == {"a.txt": "materialized"}
    assert "_pinned_coalesce" not in reduced[0]

    class ReplayAgent:
        def __init__(self):
            self.conversation = Conversation(None, "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            pass

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.save_preview_blob(session_id, "a", "body")
    persisted_events = copy.deepcopy(events)
    persisted_events[0]["payload"]["message"].pop("_attachments")
    for item in persisted_events:
        append_raw_event(
            store,
            session_id,
            item["seq"],
            item["event_type"],
            item["payload"],
        )

    for replay_id in [session_id, store.fork_session(session_id)]:
        agent = ReplayAgent()
        replay_session_into_agent(agent, replay_id, store)
        replayed = agent.conversation.messages[1]
        assert replayed["_attachment_refs"] == {
            "a.txt": "session://preview/a"
        }
        assert "a.txt" in replayed["_attachments"]
        assert "_pinned_coalesce" not in replayed


def test_malformed_events_preserve_exec_boundary_and_targetable_snapshots(tmp_path):
    from code_agent.session_message_state import reduce_canonical_message_events

    events = [
        event(1, input_message("before")),
        {"seq": 2, "event_type": "exec", "payload": {}},
        event(3, input_message("after exec")),
        {"seq": 4, "event_type": "exec", "payload": None},
        {
            "seq": 5,
            "event_type": "rewind",
            "payload": {"target_seq": 4},
        },
        {"seq": 6, "event_type": "rewind", "payload": []},
        {"seq": 7, "event_type": "message_added", "payload": {"message": None}},
        {"seq": 8, "event_type": "message_pinned", "payload": {}},
        {"seq": 9, "event_type": "attachment_invalidated", "payload": 17},
        {
            "seq": 10,
            "event_type": "rewind",
            "payload": {"target_seq": 9},
        },
    ]
    reduced, exec_start = reduce_canonical_message_events(events)
    assert exec_start == 2
    assert [message["_event_seq"] for message in reduced] == [3]

    class ReplayAgent:
        def __init__(self):
            self.conversation = Conversation(None, "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            pass

    store = SessionStore(str(tmp_path / "exec.db"))
    session_id = store.create_session("/repo", "model")
    for item in events:
        append_raw_event(
            store,
            session_id,
            item["seq"],
            item["event_type"],
            item["payload"],
        )
    agent = ReplayAgent()
    replay_session_into_agent(agent, session_id, store)
    assert [
        message["_event_seq"]
        for message in agent.conversation.messages
        if message.get("_event_seq") is not None
    ] == [3]


def test_message_pin_rewind_snapshots_match_replay_semantics():
    from code_agent.session_message_state import reduce_canonical_message_events

    base_message = {"role": "assistant", "content": "work"}
    pin_then_rewind_before_pin = [
        event(1, base_message),
        {
            "seq": 2,
            "event_type": "message_pinned",
            "payload": {"message_event_seq": 1, "label": "temporary"},
        },
        {"seq": 3, "event_type": "rewind", "payload": {"target_seq": 1}},
    ]
    messages, _ = reduce_canonical_message_events(pin_then_rewind_before_pin)
    assert "_pinned_coalesce" not in messages[0]

    pin_then_rewind_after_pin = [
        event(1, base_message),
        {
            "seq": 2,
            "event_type": "message_pinned",
            "payload": {"message_event_seq": 1, "label": "retained"},
        },
        {"seq": 3, "event_type": "rewind", "payload": {"target_seq": 2}},
    ]
    messages, _ = reduce_canonical_message_events(pin_then_rewind_after_pin)
    assert messages[0]["_pinned_coalesce"] == {"label": "retained"}


def test_persisted_child_identity_validation_and_nested_outer_atomicity():
    events, ids = completed_events(4)
    turns = completed_turns(events)
    child_range = (turns[0].source_start_seq, turns[1].source_end_seq)
    parent_range = (turns[0].source_start_seq, turns[2].source_end_seq)
    child_seq = 50
    parent_seq = 60
    state = PersistedPreviewState(
        definitions={
            child_seq: ("child-key", "child summary"),
            parent_seq: ("parent-key", "parent summary"),
        },
        active_placements={
            child_range: child_seq,
            parent_range: parent_seq,
        },
    )
    parent_node = {
        "role": "user",
        "content": (
            "[Assistant work and REPL output coalesced into preview]\n\n"
            "[PreviewRef: session://preview/child-key]\nchild summary\n[/PreviewRef]\n\n"
            "[PreviewRef: session://preview/parent-key]\nparent summary\n[/PreviewRef]"
        ),
        "_source_start_seq": parent_range[0],
        "_source_end_seq": parent_range[1],
        "_synthetic": True,
        "_coalesced": True,
        "_persisted_preview": True,
        "_preview_event_seq": parent_seq,
    }
    projection = [
        {"role": "system", "content": "system"},
        parent_node,
        *[
            message
            for message in projected_messages(events)[1:]
            if message.get("_event_seq", 0) > parent_range[1]
        ],
    ]

    units = rollup_units(events, projection, state)
    assert [unit.turn_ids for unit in units] == [
        (ids[0], ids[1], ids[2]),
        (ids[3],),
    ]
    assert not any(unit.turn_ids == (ids[0], ids[1]) for unit in units)

    for field, value in [
        ("_preview_event_seq", child_seq),
        ("_preview_event_seq", 999),
        ("content", parent_node["content"].replace("parent-key", "wrong-key")),
    ]:
        fabricated = copy.deepcopy(projection)
        fabricated[1][field] = value
        assert not any(
            len(unit.turn_ids) > 1
            for unit in rollup_units(events, fabricated, state)
        )

    missing_definition = copy.deepcopy(state)
    del missing_definition.definitions[parent_seq]
    assert not any(
        len(unit.turn_ids) > 1
        for unit in rollup_units(events, projection, missing_definition)
    )

    fabricated_flag = copy.deepcopy(projection)
    fabricated_flag[1].pop("_preview_event_seq")
    assert not any(
        len(unit.turn_ids) > 1
        for unit in rollup_units(events, fabricated_flag, state)
    )


def test_ephemeral_provider_adds_exactly_one_line_and_preserves_existing_context():
    conversation = Conversation(None, "system")
    conversation.messages.append({"role": "user", "content": "request"})
    conversation.ephemeral = (
        "Context currently expanded:\n- file.py\n\n"
        "Context window is near capacity.\n"
        "Use unview(path_or_uri) to remove files or expanded previews that are no longer needed."
    )
    conversation.ephemeral_provider = lambda: "Eligible rollup turns: 1, 4-9"

    content = conversation._messages()[-1]["content"]
    assert content.count("Eligible rollup turns:") == 1
    assert "Context currently expanded:" in content
    assert "Context window is near capacity." in content
    assert content.endswith("request")


def test_code_agent_usermsg_persists_before_context_projection():
    from code_agent.agent import CodeAgent

    agent = CodeAgent()
    agent._conversation = Conversation(None, "system")
    agent._expanded_preview_refs = {}
    agent._configure_conversation(agent._conversation)
    agent._last_was_repl_output = False
    agent._pending_explicit_attachment_refs = {}
    agent._read_attachments = {}
    agent._pending_images = []
    seen = []

    class EstimatingClient:
        model_config = {"context_window": 1000}

        def _input_bytes(self, messages):
            seen.append(messages[-1]["content"])
            return b""

        def _estimate_input_tokens(self, value):
            return 0

    def persist(message):
        message["_event_seq"] = 31
        message["_render_segments"][-1]["_event_seq"] = 31

    agent._llm_client = EstimatingClient()
    agent._persist_message = persist
    agent._current_file_context_names = lambda extra=None: []

    agent.usermsg("active", _user_content="active")

    assert seen == ["# Turn 31\n\nactive"]
    assert agent.conversation.messages[-1]["content"] == "active"


def test_shared_message_reducer_rewind_and_exec_equivalence():
    from code_agent.session_message_state import reduce_canonical_message_events

    events = [
        event(1, input_message("old")),
        event(2, release_message()),
        {"seq": 3, "event_type": "rewind", "payload": {"target_seq": 0}},
        event(4, input_message("replacement")),
        event(5, release_message()),
        {"seq": 6, "event_type": "exec", "payload": {}},
        event(7, input_message("after exec")),
        event(8, release_message()),
    ]

    messages, exec_start = reduce_canonical_message_events(events)

    assert exec_start == 6
    assert [message["_event_seq"] for message in messages] == [7, 8]
    assert completed_turns(events) == [CompletedTurn(7, 7, 8, False)]


def test_replay_resume_rewind_and_fork_keep_canonical_turn_labels(tmp_path):
    class ReplayAgent:
        def __init__(self):
            self.conversation = Conversation(None, "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.message_projector = render_turn_labels

    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    store.append_event(session_id, 1, "message_added", {"message": input_message("first")})
    store.append_event(session_id, 2, "message_added", {"message": release_message("done")})
    store.append_event(session_id, 3, "rewind", {"target_seq": 0})
    store.append_event(session_id, 4, "message_added", {"message": input_message("replacement")})

    resumed = ReplayAgent()
    replay_session_into_agent(resumed, session_id, store)
    assert resumed.conversation._messages()[-1]["content"] == "# Turn 4\n\nreplacement"

    fork_id = store.fork_session(session_id)
    forked = ReplayAgent()
    replay_session_into_agent(forked, fork_id, store)
    assert forked.conversation._messages()[-1]["content"] == "# Turn 4\n\nreplacement"
    assert store.get_events(session_id)[-1]["payload"]["message"]["content"] == "replacement"
    assert store.get_events(fork_id)[-1]["payload"]["message"]["content"] == "replacement"


def test_system_prompt_contains_stable_stage_three_guidance():
    from code_agent.agent import CodeAgentBase

    prompt = CodeAgentBase.system
    assertions = [
        "# Turn N",
        "Eligible rollup turns:",
        "Both endpoints are inclusive",
        "Recent,",
        "unlisted",
        "child-internal",
        "context pressure",
        "PreviewRef",
        "user's intent",
        "preferences",
        "failed or reverted approaches",
        "observations or discoveries",
        "important decisions",
        "verification status",
        "unresolved issues",
        "future constraints",
        "work performed",
        "final result",
    ]
    for text in assertions:
        assert text in prompt


def _rollup_test_agent(events, tmp_path):
    from code_agent.agent import CodeAgentBase

    class Client:
        def text_call(self, *args, **kwargs):
            raise AssertionError("rollup must not call an LLM")

    store = SessionStore(str(tmp_path / "rollup-sessions.db"))
    session_id = store.create_session("/repo", "model")
    for item in events:
        store.append_event(
            session_id, item["seq"], item["event_type"], copy.deepcopy(item["payload"])
        )
    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._session_store = store
    agent._session_id = session_id
    agent._next_event_seq = max(item["seq"] for item in events) + 1
    agent._persisted_preview_state = PersistedPreviewState.empty()
    agent._pending_session_events = []
    agent._suspend_persistence = False
    agent._conversation = Conversation(Client(), "system")
    agent.conversation.messages[:] = copy.deepcopy(projected_messages(events))
    agent._ensure_live_session = lambda: None
    agent._flush_pending_session_events = lambda: None
    return agent, store, session_id


def _rollup_snapshot(agent, store, session_id):
    return (
        copy.deepcopy(agent.conversation.messages),
        copy.deepcopy(agent._persisted_preview_state),
        agent._next_event_seq,
        copy.deepcopy(store.get_events(session_id)),
    )


def _mock_rollup_agent(units, max_chars=20):
    state = PersistedPreviewState.empty()
    messages = []
    return SimpleNamespace(
        code_agent_rollup_summary_max_chars=max_chars,
        _session_store=SimpleNamespace(get_events=lambda session_id: [{"seq": 1}]),
        _session_id="session",
        _persisted_preview_state=state,
        _conversation=SimpleNamespace(messages=messages),
        _authoritative_persisted_projection=lambda: (messages, state),
    )


def test_rollup_tool_has_exact_callable_schema():
    from code_agent.agent import CodeAgentBase

    assert str(inspect.signature(CodeAgentBase.rollup)) == (
        "(self, start_turn: int, end_turn: int, summary: str)"
    )
    spec = CodeAgentBase._toolspec["rollup"](CodeAgentBase.__new__(CodeAgentBase))
    assert spec.name == "rollup"
    assert [(param.name, param.annotation, param.required) for param in spec.params] == [
        ("start_turn", int, True),
        ("end_turn", int, True),
        ("summary", str, True),
    ]


def test_rollup_maps_nonconsecutive_canonical_units_and_calls_persistence_once(monkeypatch):
    from code_agent.agent import CodeAgentBase
    from code_agent.code_agent_coalesce import Preview
    import code_agent.turn_rollups as turn_rollups

    units = [
        turn_rollups.RollupUnit(2, 2, 10, 12, (2,)),
        turn_rollups.RollupUnit(7, 11, 20, 30, (7, 11)),
        turn_rollups.RollupUnit(20, 20, 40, 42, (20,)),
    ]
    calls = []
    agent = _mock_rollup_agent(units)
    agent.create_persisted_preview = lambda preview, **kwargs: (
        calls.append((preview, kwargs)) or ("preview-key", 99)
    )
    monkeypatch.setattr(turn_rollups, "rollup_units", lambda *args: units)
    monkeypatch.setattr(turn_rollups, "eligible_rollup_units", lambda *args: units)
    monkeypatch.setattr(turn_rollups, "validate_rollup_interval", lambda *args: True)

    result = CodeAgentBase.rollup(agent, 2, 11, "summary")

    assert result == "Rolled up turns 2-11 into preview preview-key (event 99)."
    assert calls == [(
        Preview(summary="summary", content=None),
        {"source_start_seq": 10, "source_end_seq": 30},
    )]


def test_rollup_requires_eligible_outer_boundaries_and_every_selected_unit(monkeypatch):
    from code_agent.agent import CodeAgentBase
    import code_agent.turn_rollups as turn_rollups

    units = [
        turn_rollups.RollupUnit(2, 2, 10, 12, (2,)),
        turn_rollups.RollupUnit(7, 11, 20, 30, (7, 11)),
        turn_rollups.RollupUnit(20, 20, 40, 42, (20,)),
    ]
    agent = _mock_rollup_agent(units)
    agent.create_persisted_preview = lambda *args, **kwargs: pytest.fail("must not persist")
    monkeypatch.setattr(turn_rollups, "rollup_units", lambda *args: units)
    monkeypatch.setattr(
        turn_rollups, "eligible_rollup_units", lambda *args: [units[0], units[2]]
    )
    monkeypatch.setattr(turn_rollups, "validate_rollup_interval", lambda *args: True)

    with pytest.raises(ValueError, match="end_turn is not an eligible unit boundary"):
        CodeAgentBase.rollup(agent, 2, 11, "summary")
    with pytest.raises(ValueError, match="every unit"):
        CodeAgentBase.rollup(agent, 2, 20, "summary")
    with pytest.raises(ValueError, match="start_turn is not an eligible unit boundary"):
        CodeAgentBase.rollup(agent, 7, 20, "summary")


def test_rollup_rejects_reversed_and_single_turn_intervals(monkeypatch):
    from code_agent.agent import CodeAgentBase
    import code_agent.turn_rollups as turn_rollups

    units = [
        turn_rollups.RollupUnit(2, 2, 10, 12, (2,)),
        turn_rollups.RollupUnit(7, 7, 20, 22, (7,)),
    ]
    agent = _mock_rollup_agent(units)
    agent.create_persisted_preview = lambda *args, **kwargs: pytest.fail("must not persist")
    monkeypatch.setattr(turn_rollups, "rollup_units", lambda *args: units)
    monkeypatch.setattr(turn_rollups, "eligible_rollup_units", lambda *args: units)
    monkeypatch.setattr(turn_rollups, "validate_rollup_interval", lambda *args: True)

    with pytest.raises(ValueError, match="reversed"):
        CodeAgentBase.rollup(agent, 7, 2, "summary")
    with pytest.raises(ValueError, match="at least two completed turns"):
        CodeAgentBase.rollup(agent, 2, 2, "summary")


@pytest.mark.parametrize(
    ("summary", "error"),
    [
        (None, "must be a string"),
        (123, "must be a string"),
        ("   \n", "must not be blank"),
        ("123456", "5-character maximum"),
        ("[PreviewRef: anything]", "must not contain PreviewRef"),
        ("[/PreviewRef]", "must not contain PreviewRef"),
        ("[ExpandedPreviewRef: anything]", "must not contain PreviewRef"),
        ("session://preview/key", "must not contain PreviewRef"),
    ],
)
def test_rollup_summary_validation_is_objective_and_nonmutating(summary, error):
    from code_agent.agent import CodeAgentBase

    agent = SimpleNamespace(code_agent_rollup_summary_max_chars=5)
    before = copy.deepcopy(agent.__dict__)
    with pytest.raises((TypeError, ValueError), match=error):
        CodeAgentBase.rollup(agent, 1, 2, summary)
    assert agent.__dict__ == before


def test_rollup_summary_accepts_boundary_length(monkeypatch):
    from code_agent.agent import CodeAgentBase
    import code_agent.turn_rollups as turn_rollups

    units = [
        turn_rollups.RollupUnit(2, 2, 10, 12, (2,)),
        turn_rollups.RollupUnit(7, 7, 20, 22, (7,)),
    ]
    agent = _mock_rollup_agent(units, max_chars=5)
    agent.create_persisted_preview = lambda *args, **kwargs: ("key", 8)
    monkeypatch.setattr(turn_rollups, "rollup_units", lambda *args: units)
    monkeypatch.setattr(turn_rollups, "eligible_rollup_units", lambda *args: units)
    monkeypatch.setattr(turn_rollups, "validate_rollup_interval", lambda *args: True)
    assert "preview key" in CodeAgentBase.rollup(agent, 2, 7, "12345")


def test_rollup_rebuilds_stale_eligibility_on_every_call(monkeypatch):
    from code_agent.agent import CodeAgentBase
    import code_agent.turn_rollups as turn_rollups

    units = [
        turn_rollups.RollupUnit(2, 2, 10, 12, (2,)),
        turn_rollups.RollupUnit(7, 7, 20, 22, (7,)),
    ]
    event_reads = []
    eligibility_calls = []
    agent = _mock_rollup_agent(units)
    agent._session_store = SimpleNamespace(
        get_events=lambda session_id: event_reads.append(session_id) or []
    )
    agent.create_persisted_preview = lambda *args, **kwargs: ("key", 8)
    monkeypatch.setattr(turn_rollups, "rollup_units", lambda *args: units)
    monkeypatch.setattr(
        turn_rollups,
        "eligible_rollup_units",
        lambda *args: eligibility_calls.append(True)
        or (units if len(eligibility_calls) == 1 else [units[0]]),
    )
    monkeypatch.setattr(turn_rollups, "validate_rollup_interval", lambda *args: True)

    CodeAgentBase.rollup(agent, 2, 7, "first")
    with pytest.raises(ValueError, match="end_turn is not an eligible unit boundary"):
        CodeAgentBase.rollup(agent, 2, 7, "second")
    assert event_reads == ["session", "session"]
    assert len(eligibility_calls) == 2


def test_rollup_live_success_and_validation_failure_are_atomic(tmp_path):
    events, ids = completed_events(6)
    agent, store, session_id = _rollup_test_agent(events, tmp_path)
    before = _rollup_snapshot(agent, store, session_id)

    with pytest.raises(ValueError, match="blank"):
        agent.rollup(ids[0], ids[1], " ")
    assert _rollup_snapshot(agent, store, session_id) == before

    result = agent.rollup(ids[0], ids[1], "combined summary")

    assert f"turns {ids[0]}-{ids[1]}" in result
    assert agent._next_event_seq == max(item["seq"] for item in events) + 3
    new_events = store.get_events(session_id)[-2:]
    assert [item["event_type"] for item in new_events] == [
        "preview_created", "preview_placed"
    ]
    assert new_events[0]["payload"]["summary"] == "combined summary"
    assert agent._persisted_preview_state.active_placements == {
        (events[0]["seq"], events[3]["seq"]): new_events[0]["seq"]
    }


def test_rollup_injected_persistence_failure_is_atomic(tmp_path, monkeypatch):
    events, ids = completed_events(6)
    agent, store, session_id = _rollup_test_agent(events, tmp_path)
    before = _rollup_snapshot(agent, store, session_id)
    monkeypatch.setattr(
        store,
        "append_preview_events",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected persistence failure")
        ),
    )

    with pytest.raises(RuntimeError, match="injected persistence failure"):
        agent.rollup(ids[0], ids[1], "summary")
    assert _rollup_snapshot(agent, store, session_id) == before


def test_recursive_rollup_preserves_child_ref_and_rejects_child_internal_endpoint(tmp_path):
    events, ids = completed_events(7)
    agent, store, session_id = _rollup_test_agent(events, tmp_path)

    child_result = agent.rollup(ids[0], ids[1], "child summary")
    child_key = child_result.split("preview ", 1)[1].split(" ", 1)[0]
    child_snapshot = _rollup_snapshot(agent, store, session_id)

    with pytest.raises(ValueError, match="start_turn is not an eligible unit boundary"):
        agent.rollup(ids[1], ids[2], "partial child")
    assert _rollup_snapshot(agent, store, session_id) == child_snapshot

    parent_result = agent.rollup(ids[0], ids[2], "parent summary")
    parent_key = parent_result.split("preview ", 1)[1].split(" ", 1)[0]
    assert f"session://preview/{child_key}" in store.get_preview_blob(
        session_id, parent_key
    )


def test_rollup_ignores_stale_live_projection_and_rebuilds_authoritatively(tmp_path):
    events, ids = completed_events(6)
    agent, store, session_id = _rollup_test_agent(events, tmp_path)
    agent.conversation.messages.insert(
        3, {"role": "assistant", "content": "stale live gap", "_event_seq": 999}
    )
    agent._persisted_preview_state = PersistedPreviewState(
        definitions={777: ("stale", "stale")},
        active_placements={(900, 901): 777},
        exec_start_seq=899,
    )

    result = agent.rollup(ids[0], ids[1], "authoritative summary")

    assert f"turns {ids[0]}-{ids[1]}" in result
    new_events = store.get_events(session_id)[-2:]
    assert [item["event_type"] for item in new_events] == [
        "preview_created", "preview_placed"
    ]
    assert agent._persisted_preview_state.definitions == {
        new_events[0]["seq"]: (
            new_events[0]["payload"]["preview_key"],
            "authoritative summary",
        )
    }
    assert agent._persisted_preview_state.active_placements == {
        (events[0]["seq"], events[3]["seq"]): new_events[0]["seq"]
    }
    assert all(
        message.get("_event_seq") != 999
        for message in agent.conversation.messages
    )
