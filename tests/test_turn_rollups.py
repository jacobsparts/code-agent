import copy
import json
from types import SimpleNamespace

import pytest

from code_agent.code_agent_coalesce import coalesce_repl_messages
from code_agent.code_agent_coalesce import make_preview_replacement
from code_agent.code_agent_coalesce import message_source_range
from code_agent.code_agent_coalesce import normalize_repl_messages
from code_agent.code_agent_coalesce import replace_projected_span
from code_agent.code_agent_coalesce import select_projected_span
from code_agent.code_agent_coalesce import semantic_segments
from code_agent.conversation import Conversation
from code_agent.session_replay import replay_session_into_agent
from code_agent.persisted_preview_state import PersistedPreviewState
from code_agent.session_store import SessionStore, utc_now_iso
from code_agent.turn_rollups import (
    CompletedTurn,
    completed_turns,
    derive_rollup_candidate_turns,
    eligible_rollup_line,
    render_semantic_labels,
    render_turn_labels,
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
    conversation.append_message(message)
    conversation.message_projector = render_turn_labels

    assert conversation.projected_messages()[-1]["content"] == "# Turn 23\n\nactive"
    assert conversation.stored_messages()[-1]["content"] == "active"
    assert conversation.projected_messages()[-1]["content"] == "# Turn 23\n\nactive"

def test_transition_marker_is_provider_only_exactly_once_on_first_following_call():
    conversation = Conversation(None, "system")
    transition = {
        "role": "assistant",
        "content": "observe('stage done', transition=True)",
        "_event_seq": 7,
        "_observation_transition": True,
    }
    output = output_message(">>> observe('stage done', transition=True)\n'[Continuing...]'\n")
    output["_event_seq"] = 8
    output["_repl_output_for"] = 7
    task = {"role": "user", "content": "task", "_event_seq": 1}
    conversation.extend_messages([task, transition, output])
    conversation.messages_projector = render_semantic_labels
    original = copy.deepcopy(conversation.stored_messages())

    first = conversation.projected_messages()
    second = conversation.projected_messages()

    assert [message["content"] for message in first].count("# Checkpoint 7") == 1
    assert first[-1]["content"] == "# Checkpoint 7"
    assert second == first
    assert conversation.stored_messages() == original


def test_malformed_transition_metadata_does_not_create_marker_or_segment():
    events = [
        event(1, input_message("task")),
        event(2, {"role": "assistant", "content": "work", "_observation_transition": "true"}),
        event(3, output_message()),
        event(4, release_message()),
    ]

    assert [turn.turn_id for turn in completed_turns(events)] == [1]
    assert not any(
        message.get("_provider_checkpoint")
        for message in render_semantic_labels(projected_messages(events))
    )


def test_completed_segments_use_mixed_turn_checkpoint_identity_and_ranges():
    events = [
        event(1, input_message("task")),
        event(2, {
            "role": "assistant",
            "content": "observe('one', transition=True)",
            "_observation_transition": True,
        }),
        event(3, {
            **output_message(
                ">>> observe('one', transition=True)\n'[Continuing...]'\nfirst output"
            ),
            "_repl_output_for": 2,
        }),
        event(4, {
            "role": "assistant",
            "content": "observe('two', transition=True)",
            "_observation_transition": True,
        }),
        event(5, {
            **output_message(
                ">>> observe('two', transition=True)\n'[Continuing...]'\nsecond output"
            ),
            "_repl_output_for": 4,
        }),
        event(6, release_message()),
        event(7, output_message("release output")),
    ]

    assert completed_turns(events) == [
        CompletedTurn(1, 1, 3, True, "turn"),
        CompletedTurn(2, 4, 5, True, "checkpoint"),
        CompletedTurn(4, 6, 7, False, "checkpoint"),
    ]

def test_persisted_child_preserves_later_canonical_checkpoint_marker():
    events = [
        event(1, input_message("task")),
        event(2, {
            "role": "assistant",
            "content": "observe('one', transition=True)",
            "_observation_transition": True,
        }),
        event(3, {
            **output_message(
                ">>> observe('one', transition=True)\n'[Continuing...]'\n"
            ),
            "_repl_output_for": 2,
        }),
        event(4, {
            "role": "assistant",
            "content": "observe('two', transition=True)",
            "_observation_transition": True,
        }),
        event(5, {
            **output_message(
                ">>> observe('two', transition=True)\n'[Continuing...]'\n"
            ),
            "_repl_output_for": 4,
        }),
        event(6, release_message()),
    ]
    projection = projected_messages(events)
    span = select_projected_span(
        projection,
        source_start_seq=1,
        source_end_seq=3,
    )
    child = make_preview_replacement(
        ["[PreviewRef: session://preview/child]\nchild\n[/PreviewRef]"],
        projection[span.start_index:span.end_index],
        source_start_seq=1,
        source_end_seq=3,
    )
    child.update({
        "_persisted_preview": True,
        "_preview_event_seq": 20,
    })
    projection = replace_projected_span(projection, span, child)

    rendered = render_semantic_labels(projection)

    assert [
        message["content"]
        for message in rendered
        if message.get("_provider_checkpoint")
    ] == ["# Checkpoint 4"]
    assert "# Checkpoint 2" not in [message.get("content") for message in rendered]


def test_transition_output_mismatch_missing_and_ambiguity_are_non_authoritative():
    base = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task", "_event_seq": 1},
        {
            "role": "assistant",
            "content": "observe('stage', transition=True)",
            "_event_seq": 2,
            "_observation_transition": True,
        },
    ]
    cases = [
        {
            **output_message("unrelated output"),
            "_event_seq": 3,
        },
        {
            **output_message(
                ">>> observe('stage', transition=True)\n'[Continuing...]'\n"
            ),
            "_event_seq": 200,
        },
        output_message(
            ">>> observe('stage', transition=True)\n'[Continuing...]'\n"
        ),
    ]

    for output in cases:
        messages = [*copy.deepcopy(base), output]
        segments = semantic_segments(messages)
        assert segments and segments[0].authoritative is False
        assert not any(
            message.get("_provider_checkpoint")
            for message in render_semantic_labels(messages)
        )

    duplicated = [
        *copy.deepcopy(base),
        {
            **output_message(
                ">>> observe('stage', transition=True)\n'[Continuing...]'\n"
            ),
            "_event_seq": 3,
        },
        {"role": "assistant", "content": "duplicate", "_event_seq": 3},
    ]
    assert semantic_segments(duplicated)[0].authoritative is False


def test_duplicate_and_colliding_transition_identity_is_non_authoritative():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task", "_event_seq": 2},
        {
            "role": "assistant",
            "content": "observe('stage', transition=True)",
            "_event_seq": 2,
            "_observation_transition": True,
        },
        {
            **output_message(
                ">>> observe('stage', transition=True)\n'[Continuing...]'\n"
            ),
            "_event_seq": 3,
            "_repl_output_for": 2,
        },
    ]

    assert semantic_segments(messages)[0].authoritative is False
    assert not any(
        message.get("_provider_checkpoint")
        for message in render_semantic_labels(messages)
    )


def test_ephemeral_context_targets_canonical_user_and_keeps_checkpoint_exact():
    conversation = Conversation(None, "system")
    conversation.extend_messages([
        {"role": "user", "content": "task", "_event_seq": 1},
        {
            "role": "assistant",
            "content": "observe('stage', transition=True)",
            "_event_seq": 2,
            "_observation_transition": True,
        },
        {
            **output_message(
                ">>> observe('stage', transition=True)\n"
                "'[Continuing...]'\n"
                "[Attachment: notes.txt]\n"
                "[PreviewRef: session://preview/child]\nsummary\n[/PreviewRef]"
            ),
            "_event_seq": 3,
            "_attachments": {"notes.txt": "body"},
            "_repl_output_for": 2,
        },
    ])
    conversation.messages_projector = render_semantic_labels
    conversation.ephemeral = "EPHEMERAL CONTEXT"
    original = copy.deepcopy(conversation.stored_messages())

    rendered = conversation.projected_messages()

    assert rendered[-1]["content"] == "# Checkpoint 2"
    assert rendered[-2]["content"].startswith("EPHEMERAL CONTEXT\n\n")
    assert "body" in rendered[-2]["content"]
    assert "[PreviewRef: session://preview/child]" in rendered[-2]["content"]
    assert conversation.stored_messages() == original


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


def test_candidate_frontier_is_earlier_of_three_turns_ago_and_last_execution():
    events, ids = completed_events(8, execution_ids={7})
    candidates = derive_rollup_candidate_turns(
        events,
        projected_messages(events),
        PersistedPreviewState.empty(),
    )

    assert candidates == tuple(ids[:4])
    assert eligible_rollup_line(candidates) == (
        "Eligible normal turn boundaries: "
        + ", ".join(str(turn_id) for turn_id in ids[:4])
    )


def test_ambiguous_projected_boundaries_are_not_listed_as_candidates():
    events, ids = completed_events(7)

    projection = projected_messages(events)
    projection.insert(2, copy.deepcopy(projection[1]))
    candidates = derive_rollup_candidate_turns(
        events,
        projection,
        PersistedPreviewState.empty(),
    )
    assert ids[0] not in candidates

    projection = projected_messages(events)
    projection.insert(3, copy.deepcopy(projection[2]))
    candidates = derive_rollup_candidate_turns(
        events,
        projection,
        PersistedPreviewState.empty(),
    )
    assert ids[0] in candidates
    assert ids[1] not in candidates


def test_active_child_hides_only_internal_candidate_turns():
    events, ids = completed_events(8)
    projection = projected_messages(events)
    turns = completed_turns(events)
    child_turns = turns[1:3]
    start = child_turns[0].source_start_seq
    end = child_turns[-1].source_end_seq
    span = select_projected_span(
        projection,
        source_start_seq=start,
        source_end_seq=end,
    )
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

    candidates = derive_rollup_candidate_turns(events, projection, state)

    assert candidates == (ids[0], ids[1], ids[3], ids[4], ids[5])
    assert ids[2] not in candidates
    assert eligible_rollup_line(candidates) == (
        "Eligible normal turn boundaries: "
        + ", ".join(str(turn_id) for turn_id in candidates)
    )
    assert eligible_rollup_line(()) == ""


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
    assert ids[0] in derive_rollup_candidate_turns(
        events,
        projection,
        PersistedPreviewState.empty(),
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

    candidates = derive_rollup_candidate_turns(events, projection, state)
    assert candidates and candidates[0] == ids[0]
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
    with pytest.raises(Exception):
        derive_rollup_candidate_turns(events, changed_body, state)

    changed_ref = copy.deepcopy(projection)
    changed_ref[node_index]["_attachment_refs"]["notes.txt"] = (
        "session://preview/forged"
    )
    with pytest.raises(Exception):
        derive_rollup_candidate_turns(events, changed_ref, state)


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
    agent.conversation.replace_messages(coalesce_repl_messages(
        agent.conversation.stored_messages(),
        keep_last_interactions=3,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    ))
    candidates = derive_rollup_candidate_turns(
        store.get_events(session_id),
        agent.conversation.stored_messages(),
        PersistedPreviewState.empty(),
    )

    assert candidates and candidates[0] == ids[0]
    node = next(
        message
        for message in agent.conversation.stored_messages()
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
    assert ids[0] in derive_rollup_candidate_turns(
        events,
        projection,
        PersistedPreviewState.empty(),
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
    agent.conversation.replace_messages(coalesce_repl_messages(
        agent.conversation.stored_messages(),
        keep_last_interactions=3,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    ))

    assert ids[0] in derive_rollup_candidate_turns(
        store.get_events(session_id),
        agent.conversation.stored_messages(),
        PersistedPreviewState.empty(),
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
    replayed = agent.conversation.stored_messages()[1]
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
        replayed = agent.conversation.stored_messages()[1]
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
        message = agent.conversation.stored_messages()[1]
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
    replayed = agent.conversation.stored_messages()[1]
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
        replayed = agent.conversation.stored_messages()[1]
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
        for message in agent.conversation.stored_messages()
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


def test_ephemeral_provider_adds_exactly_one_line_and_preserves_existing_context():
    conversation = Conversation(None, "system")
    conversation.append_message({"role": "user", "content": "request"})
    conversation.ephemeral = (
        "Current attached context:\n"
        "- file: file.py\n"
        "Estimated input: 38,400 tokens of a 120,000-token context constraint."
    )
    conversation.ephemeral_provider = lambda: (
        "Eligible rollup turns: 1, 4-9\n\n"
        "Context usage is 82%. "
        "You are expected to clean up context now. "
        "Detach attachments or expanded previews that are no longer needed with unview(...)."
    )

    content = conversation.projected_messages()[-1]["content"]
    assert content.count("Eligible rollup turns:") == 1
    assert "Current attached context:" in content
    assert "Context usage is 82%." in content
    assert content.index("Current attached context:") < content.index("Eligible rollup turns:")
    assert content.index("Eligible rollup turns:") < content.index("Context usage is 82%")
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

    assert seen == []
    assert agent.ephemeral == ""
    assert agent.conversation.stored_messages()[-1]["content"] == "active"


def test_registry_normalizes_context_accounting_limits(monkeypatch):
    from code_agent.llm_registry import EndpointRegistry

    monkeypatch.setenv("TEST_API_KEY", "key")
    registry = EndpointRegistry()
    registry.register_provider("test", host="example.test", path="/v1")

    registry.register_model(
        "test",
        "valid",
        context_constraint=120,
        context_window=200,
        max_input_tokens=50,
    )
    valid = registry.get_model_config("test/valid")
    assert valid["context_constraint"] == 120
    assert valid["context_window"] == 200
    assert valid["max_input_tokens"] == 50

    for index, value in enumerate((True, 0, -1, "120")):
        registry.register_model(
            "test",
            f"invalid-{index}",
            context_constraint=value,
            context_window=value,
            max_input_tokens=value,
        )
        invalid = registry.get_model_config(f"test/invalid-{index}")
        assert invalid["context_constraint"] is None
        assert invalid["context_window"] is None
        assert invalid["max_input_tokens"] is None


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
    assert resumed.conversation.projected_messages()[-1]["content"] == "# Turn 4\n\nreplacement"

    fork_id = store.fork_session(session_id)
    forked = ReplayAgent()
    replay_session_into_agent(forked, fork_id, store)
    assert forked.conversation.projected_messages()[-1]["content"] == "# Turn 4\n\nreplacement"
    assert store.get_events(session_id)[-1]["payload"]["message"]["content"] == "replacement"
    assert store.get_events(fork_id)[-1]["payload"]["message"]["content"] == "replacement"

def test_replay_rewind_and_fork_reconstruct_transition_segments_and_markers(tmp_path):
    class ReplayAgent:
        def __init__(self):
            self.conversation = Conversation(None, "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.messages_projector = render_semantic_labels

    store = SessionStore(str(tmp_path / "transition-sessions.db"))
    session_id = store.create_session("/repo", "model")
    transition = {
        "role": "assistant",
        "content": "observe('stage', transition=True)",
        "_observation_transition": True,
    }
    for item in [
        event(1, input_message("task")),
        event(2, transition),
        event(3, {
            **output_message(
                ">>> observe('stage', transition=True)\n'[Continuing...]'\ntransition output"
            ),
            "_repl_output_for": 2,
        }),
        event(4, release_message()),
    ]:
        store.append_event(
            session_id, item["seq"], item["event_type"], copy.deepcopy(item["payload"])
        )

    resumed = ReplayAgent()
    replay_session_into_agent(resumed, session_id, store)
    fork_id = store.fork_session(session_id)
    forked = ReplayAgent()
    replay_session_into_agent(forked, fork_id, store)

    assert completed_turns(store.get_events(session_id)) == [
        CompletedTurn(1, 1, 3, True, "turn"),
        CompletedTurn(2, 4, 4, False, "checkpoint"),
    ]
    assert resumed.conversation.projected_messages() == forked.conversation.projected_messages()
    assert sum(
        message.get("content") == "# Checkpoint 2"
        for message in resumed.conversation.projected_messages()
    ) == 1

    store.append_event(session_id, 5, "rewind", {"target_seq": 1})
    rewound = ReplayAgent()
    replay_session_into_agent(rewound, session_id, store)
    assert completed_turns(store.get_events(session_id)) == []
    assert not any(
        message.get("_provider_checkpoint")
        for message in rewound.conversation.projected_messages()
    )




def _rollup_test_agent(events, tmp_path):
    from code_agent.agent import CodeAgentBase

    class Client:
        def call(self, *args, **kwargs):
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
    agent.conversation.replace_messages(copy.deepcopy(projected_messages(events)))
    agent._ensure_live_session = lambda: None
    agent._flush_pending_session_events = lambda: None
    return agent, store, session_id


def _rollup_snapshot(agent, store, session_id):
    return (
        copy.deepcopy(agent.conversation.stored_messages()),
        copy.deepcopy(agent._persisted_preview_state),
        agent._next_event_seq,
        copy.deepcopy(store.get_events(session_id)),
    )



def test_observe_transition_schema_is_strict_bool_and_commits_atomically():
    from code_agent.agent import CodeAgent

    agent = CodeAgent.__new__(CodeAgent)
    agent._pending_observations = []
    agent._pending_observation_transition = False
    persisted = []
    agent._persist_message = persisted.append


    for value in (1, 0, "true", None):
        with pytest.raises(TypeError, match="must be a boolean"):
            agent.observe("invalid", transition=value)
    assert agent._pending_observations == []
    assert agent._pending_observation_transition is False

    agent.observe("stage complete", transition=True)
    message = {"role": "assistant", "content": "work"}
    agent._on_assistant_message_committed(message)

    assert message == {
        "role": "assistant",
        "content": "work",
        "_observations": ["stage complete"],
        "_observation_transition": True,
    }
    assert persisted == [message]
    assert agent._pending_observations == []
    assert agent._pending_observation_transition is False


def test_observe_transition_retry_and_abandonment_state_does_not_leak():
    from code_agent.agent import CodeAgent

    agent = CodeAgent.__new__(CodeAgent)
    agent._pending_observations = []
    agent._pending_observation_transition = False
    agent._persist_message = lambda message: None

    agent.observe("discarded attempt", transition=True)
    agent._start_assistant_execution_attempt()
    retry_message = {"role": "assistant", "content": "retry"}
    agent._on_assistant_message_committed(retry_message)
    assert "_observations" not in retry_message
    assert "_observation_transition" not in retry_message

    agent.observe("interrupted attempt", transition=True)
    agent._start_assistant_execution_attempt()
    later_message = {"role": "assistant", "content": "later"}
    agent._on_assistant_message_committed(later_message)
    assert "_observation_transition" not in later_message



@pytest.mark.parametrize(
    ("summary", "error"),
    [
        (None, "must be a string"),
        (123, "must be a string"),
        ("   \n", "must not be blank"),
        ("[PreviewRef: anything]", "must not contain PreviewRef"),
        ("[/PreviewRef]", "must not contain PreviewRef"),
        ("[ExpandedPreviewRef: anything]", "must not contain PreviewRef"),
        ("session://preview/key", "must not contain PreviewRef"),
    ],
)
def test_rollup_summary_validation_is_objective_and_nonmutating(
    summary, error, capsys
):
    from code_agent.agent import CodeAgentBase

    agent = SimpleNamespace()
    before = copy.deepcopy(agent.__dict__)
    CodeAgentBase.rollup(agent, 1, 2, summary)
    output = capsys.readouterr().out
    assert output.startswith("Rollup rejected: ")
    assert error in output
    assert agent.__dict__ == before




def test_rollup_live_success_and_validation_failure_are_atomic(tmp_path, capsys):
    events, ids = completed_events(6)
    agent, store, session_id = _rollup_test_agent(events, tmp_path)
    before = _rollup_snapshot(agent, store, session_id)

    assert agent.rollup(ids[0], ids[1], " ") is None
    assert "Rollup rejected: rollup summary must not be blank." in capsys.readouterr().out
    assert _rollup_snapshot(agent, store, session_id) == before

    result = agent.rollup(ids[0], ids[1], "combined summary")

    assert result is None
    assert agent._next_event_seq == max(item["seq"] for item in events) + 3
    new_events = store.get_events(session_id)[-2:]
    assert [item["event_type"] for item in new_events] == [
        "preview_created", "preview_placed"
    ]
    assert new_events[0]["payload"]["summary"] == "combined summary"
    assert agent._persisted_preview_state.active_placements == {
        (events[0]["seq"], events[1]["seq"]): new_events[0]["seq"]
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


def test_recursive_and_adjacent_rollups_preserve_outer_boundaries(tmp_path, capsys):
    events, ids = completed_events(7)
    agent, store, session_id = _rollup_test_agent(events, tmp_path)

    assert agent.rollup(ids[0], ids[1], "child summary") is None
    child_output = capsys.readouterr().out
    child_key = child_output.split("preview ", 1)[1].split(" ", 1)[0]

    assert agent.rollup(ids[1], ids[2], "adjacent summary") is None
    adjacent_output = capsys.readouterr().out
    adjacent_key = adjacent_output.split("preview ", 1)[1].split(" ", 1)[0]

    assert agent.rollup(ids[0], ids[2], "parent summary") is None
    parent_output = capsys.readouterr().out
    parent_key = parent_output.split("preview ", 1)[1].split(" ", 1)[0]
    parent_content = store.get_preview_blob(session_id, parent_key)
    assert f"session://preview/{child_key}" in parent_content
    assert f"session://preview/{adjacent_key}" in parent_content


def test_rollup_ignores_stale_live_projection_and_rebuilds_authoritatively(tmp_path):
    events, ids = completed_events(6)
    agent, store, session_id = _rollup_test_agent(events, tmp_path)
    agent.conversation.insert_message(
        3, {"role": "assistant", "content": "stale live gap", "_event_seq": 999}
    )
    agent._persisted_preview_state = PersistedPreviewState(
        definitions={777: ("stale", "stale")},
        active_placements={(900, 901): 777},
        exec_start_seq=899,
    )

    result = agent.rollup(ids[0], ids[1], "authoritative summary")

    assert result is None
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
        (events[0]["seq"], events[1]["seq"]): new_events[0]["seq"]
    }
    assert all(
        message.get("_event_seq") != 999
        for message in agent.conversation.stored_messages()
    )


def test_production_shape_turn_212_release_output_slices_form_one_authoritative_turn():
    events = [
        event(141, input_message("preceding turn")),
        event(209, release_message("preceding done")),
        event(210, output_message(
            ">>> emit('preceding done', release=True)\npreceding done\n"
        )),
        event(212, input_message("ordinary work")),
    ]
    for seq in (213, 215, 217):
        events.append(event(seq, {"role": "assistant", "content": f"print({seq})"}))
        events.append(event(seq + 1, output_message(f">>> print({seq})\n{seq}\n")))
    events.extend([
        event(310, {
            "role": "assistant",
            "content": (
                "observe('stage complete', transition=True)\n"
                "emit('done', release=True)"
            ),
            "_observation_transition": True,
            "_final_result": "done",
        }),
        event(311, {
            "role": "user",
            "content": (
                ">>> observe('stage complete', transition=True)\n"
                "'[Continuing...]'\n"
                ">>> emit('done', release=True)\n"
                "done\n"
            ),
            "_stdout": "combined output",
            "_render_segments": [{
                "type": "stdout",
                "content": (
                    ">>> observe('stage complete', transition=True)\n"
                    "'[Continuing...]'\n"
                    ">>> emit('done', release=True)\n"
                    "done\n"
                ),
            }],
        }),
        event(313, input_message("next turn")),
    ])

    turns = completed_turns(events)

    assert turns == [
        CompletedTurn(141, 141, 210, False),
        CompletedTurn(212, 212, 311, True),
    ]
    projection = coalesce_repl_messages(
        projected_messages(events),
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    )
    normalized = __import__(
        "code_agent.code_agent_coalesce",
        fromlist=["normalize_repl_messages"],
    ).normalize_repl_messages(
        __import__(
            "code_agent.session_message_state",
            fromlist=["reduce_canonical_message_events"],
        ).reduce_canonical_message_events(events)[0]
    )
    event_311_nodes = [
        message for message in normalized if message.get("_event_seq") == 311
    ]
    assert len(event_311_nodes) == 2
    assert [message.get("_event_seq") for message in normalized[-4:]] == [
        311, 310, 311, 313
    ]


    span = select_projected_span(
        normalized,
        source_start_seq=212,
        source_end_seq=311,
    )
    assert normalized[span.start_index].get("_event_seq") == 212
    assert normalized[span.end_index - 1].get("_event_seq") == 311
    assert normalized[span.end_index].get("_event_seq") == 313

    duplicated = normalized[:span.end_index]
    duplicated.insert(span.end_index - 1, copy.deepcopy(normalized[span.end_index - 1]))
    with pytest.raises(Exception, match="non-contiguous|ambiguous"):
        select_projected_span(
            duplicated,
            source_start_seq=212,
            source_end_seq=311,
        )

    mixed = copy.deepcopy(normalized)
    mixed[span.end_index - 3]["role"] = "assistant"
    with pytest.raises(Exception, match="non-contiguous"):
        select_projected_span(
            mixed,
            source_start_seq=212,
            source_end_seq=311,
        )


@pytest.mark.parametrize("ending", ["interruption", "max turns", "runtime error"])
def test_next_real_input_fallback_closes_unreleased_history_but_eof_keeps_active(ending):
    prefix = [
        event(1, input_message("first")),
        event(2, {"role": "assistant", "content": "print('one')"}),
        event(3, output_message(">>> print('one')\none\n")),
        event(4, {"role": "assistant", "content": f"# {ending}\nprint('two')"}),
        event(5, output_message(">>> print('two')\ntwo\n")),
    ]

    assert completed_turns(prefix) == []
    assert completed_turns([*prefix, event(9, input_message("next"))]) == [
        CompletedTurn(1, 1, 5, True)
    ]


def test_transition_release_same_execution_has_one_release_closure_and_no_checkpoint():
    events = [
        event(1, input_message("task")),
        event(2, {
            "role": "assistant",
            "content": "observe('done', transition=True)\nemit('released', release=True)",
            "_observation_transition": True,
            "_final_result": "released",
        }),
        event(3, output_message(
            ">>> observe('done', transition=True)\n"
            "'[Continuing...]'\n"
            ">>> emit('released', release=True)\nreleased\n"
        )),
    ]

    assert completed_turns(events) == [CompletedTurn(1, 1, 3, False)]
    assert not any(
        message.get("_provider_checkpoint")
        for message in render_semantic_labels(projected_messages(events))
    )


def test_ordered_same_event_slices_are_valid_but_duplicate_or_mixed_provenance_is_not():
    events = [
        event(1, input_message("task")),
        event(2, release_message()),
        event(3, output_message("prefix\n>>> emit('done', release=True)\ndone\n")),
        event(4, input_message("next")),
    ]
    canonical = __import__(
        "code_agent.session_message_state",
        fromlist=["reduce_canonical_message_events"],
    ).reduce_canonical_message_events(events)[0]
    normalized = __import__(
        "code_agent.code_agent_coalesce",
        fromlist=["normalize_repl_messages"],
    ).normalize_repl_messages(canonical)
    assert len([message for message in normalized if message.get("_event_seq") == 3]) == 2
    assert semantic_segments(normalized)[0].authoritative is True

    duplicated = normalized[:]
    duplicated.insert(-1, copy.deepcopy(normalized[-2]))
    assert semantic_segments(duplicated)[0].authoritative is False

    mixed = copy.deepcopy(normalized)
    mixed[-2]["role"] = "assistant"
    assert semantic_segments(mixed)[0].authoritative is False


def test_completed_partition_is_coverage_complete_before_active_segment():
    events = [
        event(1, input_message("fallback")),
        event(2, {"role": "assistant", "content": "work"}),
        event(3, output_message()),
        event(4, input_message("transition turn")),
        event(5, {
            "role": "assistant",
            "content": "observe('stage', transition=True)",
            "_observation_transition": True,
        }),
        event(6, {**output_message("transition output"), "_repl_output_for": 5}),
        event(7, release_message()),
        event(8, output_message("release output")),
        event(9, input_message("active")),
    ]
    turns = completed_turns(events)

    assert [(turn.source_start_seq, turn.source_end_seq) for turn in turns] == [
        (1, 3),
        (4, 6),
        (7, 8),
    ]
    attributable = {
        item["seq"]
        for item in events
        if item["event_type"] == "message_added" and item["seq"] < 9
    }
    covered = {
        seq
        for turn in turns
        for seq in range(turn.source_start_seq, turn.source_end_seq + 1)
        if seq in attributable
    }
    assert covered == attributable

def test_adjacent_release_output_and_multiple_real_inputs_preserve_turn_identities():
    from code_agent.session_message_state import reduce_canonical_message_events

    events = [
        event(1414, input_message("released turn")),
        event(1415, release_message("released")),
        event(1416, output_message(
            ">>> emit('released', release=True)\nreleased\n"
        )),
        event(1418, input_message("input A")),
        event(1420, input_message("input B")),
        event(1422, {"role": "assistant", "content": "print('work')"}),
        event(1423, output_message(">>> print('work')\nwork\n")),
    ]
    canonical, _ = reduce_canonical_message_events(events)
    merged = next(
        message
        for message in canonical
        if any(
            segment.get("_event_seq") == 1418
            for segment in message.get("_render_segments") or []
        )
    )
    assert [
        (segment.get("type"), segment.get("_event_seq"))
        for segment in merged["_render_segments"]
    ] == [
        ("stdout", 1416),
        ("input", 1418),
        ("input", 1420),
    ]

    normalized = normalize_repl_messages(canonical)
    assert [
        (
            message.get("_event_seq"),
            message.get("_render_segments", [{}])[0].get("type"),
        )
        for message in normalized
        if message.get("_event_seq") in {1416, 1418, 1420}
    ] == [
        (1416, "stdout"),
        (1418, "input"),
        (1420, "input"),
    ]
    assert completed_turns(events) == [
        CompletedTurn(1414, 1414, 1416, False),
        CompletedTurn(1418, 1418, 1418, False),
    ]
    assert not any(turn.turn_id == 1416 for turn in completed_turns(events))

    closed_events = [
        *events,
        event(1452, release_message("done")),
        event(1453, output_message(
            ">>> emit('done', release=True)\ndone\n"
        )),
    ]
    assert completed_turns(closed_events) == [
        CompletedTurn(1414, 1414, 1416, False),
        CompletedTurn(1418, 1418, 1418, False),
        CompletedTurn(1420, 1420, 1453, True),
    ]
    projection = coalesce_repl_messages(
        projected_messages(closed_events),
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    )


def test_structured_multi_input_normalization_rejects_ambiguous_provenance():
    base = {
        "role": "user",
        "content": "output\nA\nB",
        "_render_segments": [
            {"type": "stdout", "content": "output\n", "_event_seq": 1},
            {"type": "input", "content": "A", "_event_seq": 2},
            {"type": "input", "content": "B", "_event_seq": 3},
        ],
    }
    assert [message.get("_event_seq") for message in normalize_repl_messages([base])] == [
        1, 2, 3
    ]

    duplicate = copy.deepcopy(base)
    duplicate["_render_segments"][2]["_event_seq"] = 2
    duplicate_nodes = normalize_repl_messages([duplicate])
    assert len(duplicate_nodes) == 1
    assert semantic_segments(duplicate_nodes) == []

    mixed = copy.deepcopy(base)
    mixed["_render_segments"][2].pop("_event_seq")
    mixed_nodes = normalize_repl_messages([mixed])
    assert len(mixed_nodes) == 1
    assert semantic_segments(mixed_nodes) == []

    descending = copy.deepcopy(base)
    descending["_render_segments"] = [
        {"type": "input", "content": "B", "_event_seq": 2},
        {"type": "input", "content": "A", "_event_seq": 1},
    ]
    descending_nodes = normalize_repl_messages([descending])
    assert len(descending_nodes) == 1
    assert semantic_segments(descending_nodes) == []