import copy
import inspect
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
    RollupUnit,
    completed_turns,
    derive_rollup_eligibility,
    eligible_rollup_groups,
    eligible_rollup_line,
    eligible_rollup_units,
    render_semantic_labels,
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
    conversation.messages.extend([task, transition, output])
    conversation.messages_projector = render_semantic_labels
    original = copy.deepcopy(conversation.messages)

    first = conversation._messages()
    second = conversation._messages()

    assert [message["content"] for message in first].count("# Checkpoint 7") == 1
    assert first[-1]["content"] == "# Checkpoint 7"
    assert second == first
    assert conversation.messages == original


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
    conversation.messages.extend([
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
    original = copy.deepcopy(conversation.messages)

    rendered = conversation._messages()

    assert rendered[-1]["content"] == "# Checkpoint 2"
    assert rendered[-2]["content"].startswith("EPHEMERAL CONTEXT\n\n")
    assert "body" in rendered[-2]["content"]
    assert "[PreviewRef: session://preview/child]" in rendered[-2]["content"]
    assert conversation.messages == original


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
    groups = eligible_rollup_groups(
        events,
        projection,
        PersistedPreviewState.empty(),
    )
    assert eligible_rollup_line(groups) == (
        "Eligible rollup turns: ["
        + ", ".join(str(turn_id) for turn_id in ids[:5])
        + "] (combine units only within one bracketed group)"
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
    assert [unit.turn_ids for unit in eligible] == [
        (ids[0],), (ids[1], ids[2]), (ids[3],), (ids[4],)
    ]
    groups = eligible_rollup_groups(events, projection, state)
    assert eligible_rollup_line(groups) == (
        f"Eligible rollup turns: [{ids[0]}, {ids[1]}-{ids[2]}, {ids[3]}, {ids[4]}] "
        "(combine units only within one bracketed group)"
    )
    assert eligible_rollup_line([]) == ""


def test_eligibility_line_hides_a_single_completed_turn():
    unit = RollupUnit(6, 6, 6, 7, (6,))

    assert eligible_rollup_line([]) == ""


def test_eligibility_line_keeps_one_atomic_multi_turn_unit():
    unit = RollupUnit(6, 14, 6, 15, (6, 10, 14))

    assert eligible_rollup_line([[unit]]) == (
        "Eligible rollup turns: [6-14] "
        "(combine units only within one bracketed group)"
    )



def test_sparse_individually_covered_units_without_valid_pairs_are_not_advertised(
    monkeypatch,
):
    import code_agent.turn_rollups as turn_rollups

    units = [
        RollupUnit(141, 141, 141, 142, (141,)),
        RollupUnit(313, 313, 313, 314, (313,)),
        RollupUnit(317, 317, 317, 318, (317,)),
        RollupUnit(321, 321, 321, 322, (321,)),
        RollupUnit(325, 325, 325, 326, (325,)),
    ]
    turns = [
        CompletedTurn(unit.start_turn, unit.source_start_seq, unit.source_end_seq, False)
        for unit in units
    ] + [
        CompletedTurn(turn_id, turn_id, turn_id + 1, False)
        for turn_id in (400, 404, 408)
    ]
    monkeypatch.setattr(
        turn_rollups,
        "_canonical_messages",
        lambda events: ([{"role": "user", "_event_seq": unit.start_turn} for unit in units], 0),
    )
    monkeypatch.setattr(turn_rollups, "semantic_segments", lambda messages: [
        SimpleNamespace(
            segment_id=turn.turn_id,
            source_start_seq=turn.source_start_seq,
            source_end_seq=turn.source_end_seq,
            has_execution=False,
            identity_kind="turn",
            authoritative=True,
        )
        for turn in turns
    ])
    monkeypatch.setattr(turn_rollups, "_rollup_units_from_snapshot", lambda *args: units)
    spans = {
        141: SimpleNamespace(start_index=0, end_index=1),
        313: SimpleNamespace(start_index=2, end_index=3),
        317: SimpleNamespace(start_index=4, end_index=5),
        321: SimpleNamespace(start_index=6, end_index=7),
        325: SimpleNamespace(start_index=7, end_index=8),
    }
    monkeypatch.setattr(
        turn_rollups,
        "select_projected_span",
        lambda projected, source_start_seq, source_end_seq: spans[source_start_seq],
    )

    groups = eligible_rollup_groups([], [], PersistedPreviewState.empty())
    assert groups == [[units[3], units[4]]]
    assert eligible_rollup_units([], [], PersistedPreviewState.empty()) == [
        units[3],
        units[4],
    ]
    assert eligible_rollup_line(groups) == (
        "Eligible rollup turns: [321, 325] "
        "(combine units only within one bracketed group)"
    )



@pytest.mark.parametrize(
    ("count", "execution_ids", "coalesce"),
    [
        (8, set(), False),
        (10, {1, 9, 17}, False),
        (12, {1, 9, 17, 25}, True),
    ],
)
def test_group_derivation_matches_authoritative_adjacent_validation(
    count,
    execution_ids,
    coalesce,
):
    from code_agent.turn_rollups import validate_rollup_interval

    events, _ = completed_events(count, execution_ids=execution_ids)
    projection = projected_messages(events)
    if coalesce:
        projection = coalesce_repl_messages(
            projection,
            keep_last_interactions=3,
            keep_last_execution_interactions=1,
            min_savings_chars=0,
        )
    state = PersistedPreviewState.empty()
    eligibility = derive_rollup_eligibility(events, projection, state)
    turns = completed_turns(events)
    protected = {turn.turn_id for turn in turns[-3:]}
    execution_turns = [turn for turn in turns if turn.has_execution]
    if execution_turns:
        protected.add(execution_turns[-1].turn_id)
    candidates = [
        unit
        for unit in eligibility.all_units
        if not any(turn_id in protected for turn_id in unit.turn_ids)
    ]

    expected = []
    current = []
    for unit in candidates:
        if current and not validate_rollup_interval(
            events,
            projection,
            state,
            [current[-1], unit],
        ):
            if sum(len(item.turn_ids) for item in current) >= 2:
                expected.append(current)
            current = []
        current.append(unit)
    if current and sum(len(item.turn_ids) for item in current) >= 2:
        expected.append(current)

    assert [list(group) for group in eligibility.groups] == expected


def test_group_derivation_builds_one_snapshot_without_candidate_revalidation(
    monkeypatch,
):
    import code_agent.turn_rollups as turn_rollups

    events, _ = completed_events(20)
    projection = projected_messages(events)
    canonical_calls = 0
    validation_calls = 0
    original_canonical = turn_rollups._canonical_messages

    def counted_canonical(snapshot_events):
        nonlocal canonical_calls
        canonical_calls += 1
        return original_canonical(snapshot_events)

    def counted_validation(*args):
        nonlocal validation_calls
        validation_calls += 1
        return True

    monkeypatch.setattr(turn_rollups, "_canonical_messages", counted_canonical)
    monkeypatch.setattr(
        turn_rollups,
        "_validate_rollup_interval_snapshot",
        counted_validation,
    )

    eligibility = derive_rollup_eligibility(
        events,
        projection,
        PersistedPreviewState.empty(),
    )

    assert eligibility.groups
    assert canonical_calls == 1
    assert validation_calls == 0


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

    content = conversation._messages()[-1]["content"]
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

    assert seen == []
    assert agent.ephemeral == ""
    assert agent.conversation.messages[-1]["content"] == "active"


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
    assert resumed.conversation._messages()[-1]["content"] == "# Turn 4\n\nreplacement"

    fork_id = store.fork_session(session_id)
    forked = ReplayAgent()
    replay_session_into_agent(forked, fork_id, store)
    assert forked.conversation._messages()[-1]["content"] == "# Turn 4\n\nreplacement"
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
    assert resumed.conversation._messages() == forked.conversation._messages()
    assert sum(
        message.get("content") == "# Checkpoint 2"
        for message in resumed.conversation._messages()
    ) == 1

    store.append_event(session_id, 5, "rewind", {"target_seq": 1})
    rewound = ReplayAgent()
    replay_session_into_agent(rewound, session_id, store)
    assert completed_turns(store.get_events(session_id)) == []
    assert not any(
        message.get("_provider_checkpoint")
        for message in rewound.conversation._messages()
    )


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

    assert "# Checkpoint" in prompt
    assert "transition=True" in prompt
    assert "does not release control" in prompt


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

def test_observe_transition_schema_is_strict_bool_and_commits_atomically():
    from code_agent.agent import CodeAgent

    agent = CodeAgent.__new__(CodeAgent)
    agent._pending_observations = []
    agent._pending_observation_transition = False
    persisted = []
    agent._persist_message = persisted.append

    signature = inspect.signature(CodeAgent.observe)
    assert signature.parameters["content"].annotation is str
    assert signature.parameters["content"].default == "Reflection on previous substantive work"
    assert signature.parameters["transition"].annotation is bool
    assert signature.parameters["transition"].default is False

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
    monkeypatch.setattr(turn_rollups, "derive_rollup_eligibility", lambda *args: turn_rollups.RollupEligibility(tuple(units), tuple(units), (tuple(units),)))
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
        turn_rollups,
        "derive_rollup_eligibility",
        lambda *args: turn_rollups.RollupEligibility(
            tuple(units),
            (units[0], units[2]),
            ((units[0],), (units[2],)),
        ),
    )
    monkeypatch.setattr(turn_rollups, "validate_rollup_interval", lambda *args: True)

    with pytest.raises(ValueError, match="end_turn is not an eligible unit boundary"):
        CodeAgentBase.rollup(agent, 2, 11, "summary")
    with pytest.raises(ValueError, match="every unit"):
        CodeAgentBase.rollup(agent, 2, 20, "summary")
    with pytest.raises(ValueError, match="start_turn is not an eligible unit boundary"):
        CodeAgentBase.rollup(agent, 7, 20, "summary")



def test_rollup_reports_uncovered_gap_between_individually_eligible_groups(
    monkeypatch,
):
    from code_agent.agent import CodeAgentBase
    import code_agent.turn_rollups as turn_rollups

    units = [
        turn_rollups.RollupUnit(141, 141, 10, 12, (141,)),
        turn_rollups.RollupUnit(313, 313, 20, 22, (313,)),
        turn_rollups.RollupUnit(317, 317, 30, 32, (317,)),
        turn_rollups.RollupUnit(321, 321, 40, 42, (321,)),
    ]
    agent = _mock_rollup_agent(units)
    agent.create_persisted_preview = lambda *args, **kwargs: pytest.fail(
        "must not persist"
    )
    monkeypatch.setattr(turn_rollups, "rollup_units", lambda *args: units)
    monkeypatch.setattr(
        turn_rollups,
        "derive_rollup_eligibility",
        lambda *args: turn_rollups.RollupEligibility(
            tuple(units),
            tuple(units),
            (tuple(units[:2]), tuple(units[2:])),
        ),
    )
    monkeypatch.setattr(turn_rollups, "validate_rollup_interval", lambda *args: True)

    with pytest.raises(ValueError, match="uncovered canonical/projected gap"):
        CodeAgentBase.rollup(agent, 141, 321, "summary")


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
    monkeypatch.setattr(turn_rollups, "derive_rollup_eligibility", lambda *args: turn_rollups.RollupEligibility(tuple(units), tuple(units), (tuple(units),)))
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
    monkeypatch.setattr(turn_rollups, "derive_rollup_eligibility", lambda *args: turn_rollups.RollupEligibility(tuple(units), tuple(units), (tuple(units),)))
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
        "derive_rollup_eligibility",
        lambda *args: eligibility_calls.append(True)
        or (
            turn_rollups.RollupEligibility(tuple(units), tuple(units), (tuple(units),))
            if len(eligibility_calls) == 1
            else turn_rollups.RollupEligibility(tuple(units), (), ())
        ),
    )
    monkeypatch.setattr(turn_rollups, "validate_rollup_interval", lambda *args: True)

    CodeAgentBase.rollup(agent, 2, 7, "first")
    with pytest.raises(ValueError, match="no rollup units are currently eligible"):
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
    units = rollup_units(events, projection, PersistedPreviewState.empty())
    assert [unit for unit in units if unit.turn_ids == (212,)] == [
        RollupUnit(212, 212, 212, 311, (212,))
    ]
    adjacent_events = [*events, event(314, release_message("next done"))]
    adjacent_projection = coalesce_repl_messages(
        projected_messages(adjacent_events),
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_savings_chars=0,
    )
    adjacent_units = rollup_units(
        adjacent_events,
        adjacent_projection,
        PersistedPreviewState.empty(),
    )
    bridge = [
        unit for unit in adjacent_units
        if unit.start_turn in {141, 212, 313}
    ]
    assert bridge == [
        RollupUnit(141, 141, 141, 210, (141,)),
        RollupUnit(212, 212, 212, 311, (212,)),
        RollupUnit(313, 313, 313, 314, (313,)),
    ]
    assert __import__(
        "code_agent.turn_rollups",
        fromlist=["validate_rollup_interval"],
    ).validate_rollup_interval(
        adjacent_events,
        adjacent_projection,
        PersistedPreviewState.empty(),
        bridge,
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
    assert rollup_units(
        closed_events,
        projection,
        PersistedPreviewState.empty(),
    ) == [
        RollupUnit(1414, 1414, 1414, 1416, (1414,)),
        RollupUnit(1418, 1418, 1418, 1418, (1418,)),
        RollupUnit(1420, 1420, 1420, 1453, (1420,)),
    ]


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
    assert rollup_units(
        [
            event(2, input_message("B")),
            event(1, input_message("A")),
        ],
        descending_nodes,
        PersistedPreviewState.empty(),
    ) == []
