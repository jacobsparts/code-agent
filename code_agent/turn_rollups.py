import copy
import re
from dataclasses import dataclass

from code_agent.code_agent_coalesce import (
    PreviewPlacementError,
    has_valid_deterministic_identity,
    is_release_assistant_message,
    is_repl_output_message,
    message_source_range,
    normalize_repl_messages,
    render_preview_ref,
    select_projected_span,
    semantic_boundaries,
    semantic_segments,
)
from code_agent.persisted_preview_state import PersistedPreviewState
from code_agent.session_message_state import reduce_canonical_message_events


@dataclass(frozen=True)
class CompletedTurn:
    turn_id: int
    source_start_seq: int
    source_end_seq: int
    has_execution: bool
    identity_kind: str = "turn"


@dataclass(frozen=True)
class RollupUnit:
    start_turn: int
    end_turn: int
    source_start_seq: int
    source_end_seq: int
    turn_ids: tuple[int, ...]


@dataclass(frozen=True)
class RollupEligibility:
    all_units: tuple[RollupUnit, ...]
    units: tuple[RollupUnit, ...]
    groups: tuple[tuple[RollupUnit, ...], ...]


def _input_segments(message: dict) -> list[tuple[str, int]]:
    if message.get("role") != "user" or message.get("_synthetic"):
        return []
    if "_render_segments" in message:
        render_segments = message.get("_render_segments")
        if not isinstance(render_segments, list) or any(
            not isinstance(segment, dict) for segment in render_segments
        ):
            return []
        segments = []
        for segment in render_segments:
            if segment.get("type") != "input":
                continue
            content = segment.get("content")
            seq = segment.get("_event_seq")
            if isinstance(content, str) and content:
                if type(seq) is not int:
                    return []
                segments.append((content, seq))
        return segments
    content = message.get("_user_content")
    seq = message.get("_event_seq")
    if isinstance(content, str) and content and type(seq) is int:
        return [(content, seq)]
    if is_repl_output_message(message):
        return []
    content = message.get("content")
    if isinstance(content, str) and content and type(seq) is int:
        return [(content, seq)]
    return []


def render_semantic_labels(messages: list[dict]) -> list[dict]:
    projected = [copy.deepcopy(message) for message in messages]
    markers = {}
    for boundary in semantic_boundaries(projected).values():
        if (
            boundary.authoritative
            and boundary.is_transition
            and type(boundary.transition_seq) is int
        ):
            marker_index = (
                boundary.boundary_output_index
                if boundary.boundary_output_index is not None
                else boundary.boundary_index
            )
            markers[marker_index] = boundary.transition_seq

    out = []
    for index, message in enumerate(projected):
        out.append(render_turn_labels(message))
        checkpoint = markers.get(index)
        if checkpoint is not None:
            out.append({
                "role": "user",
                "content": f"# Checkpoint {checkpoint}",
                "_synthetic": True,
                "_provider_checkpoint": True,
            })
    return out


def render_turn_labels(message: dict) -> dict:
    out = copy.deepcopy(message)
    if out.get("role") != "user" or out.get("_synthetic"):
        return out
    content = out.get("content") or ""
    render_segments = out.get("_render_segments") or []
    located_inputs = []
    cursor = 0
    if render_segments:
        for segment in render_segments:
            if not isinstance(segment, dict):
                return out
            text = segment.get("content")
            if not isinstance(text, str) or not text:
                continue
            index = content.find(text, cursor)
            if index < 0:
                return out
            if segment.get("type") == "input":
                seq = segment.get("_event_seq")
                if type(seq) is not int:
                    return out
                located_inputs.append((index, text, seq))
            cursor = index + len(text)
    else:
        inputs = _input_segments(out)
        if len(inputs) != 1:
            return out
        text, seq = inputs[0]
        index = content.find(text)
        if index < 0:
            return out
        located_inputs.append((index, text, seq))

    inserts = []
    for index, _, seq in located_inputs:
        marker = f"# Turn {seq}\n\n"
        if content[max(0, index - len(marker)):index] != marker:
            inserts.append((index, marker))
    for index, marker in reversed(inserts):
        content = content[:index] + marker + content[index:]
    out["content"] = content
    return out


def _canonical_messages(events: list[dict]) -> tuple[list[dict], int]:
    messages, exec_start_seq = reduce_canonical_message_events(events)
    return normalize_repl_messages(messages), exec_start_seq


def _canonical_input_seq(message: dict) -> int | None:
    segments = _input_segments(message)
    if len(segments) != 1:
        return None
    _, seq = segments[0]
    if seq != message.get("_event_seq"):
        return None
    return seq


def completed_turns(events: list[dict]) -> list[CompletedTurn]:
    messages, exec_start_seq = _canonical_messages(events)
    return [
        CompletedTurn(
            segment.segment_id,
            segment.source_start_seq,
            segment.source_end_seq,
            segment.has_execution,
            segment.identity_kind,
        )
        for segment in semantic_segments(messages)
        if (
            segment.authoritative
            and exec_start_seq < segment.source_start_seq <= segment.source_end_seq
        )
    ]


def _outer_active_placements(active_placements: dict) -> list[tuple[int, int]]:
    ranges = list(active_placements)
    return [
        current
        for current in ranges
        if not any(
            other != current
            and other[0] <= current[0]
            and current[1] <= other[1]
            for other in ranges
        )
    ]


def _range_turns(
    turns: list[CompletedTurn],
    source_start_seq: int,
    source_end_seq: int,
) -> list[CompletedTurn]:
    overlapping = [
        turn
        for turn in turns
        if turn.source_start_seq <= source_end_seq
        and source_start_seq <= turn.source_end_seq
    ]
    if not overlapping:
        return []
    if (
        overlapping[0].source_start_seq != source_start_seq
        or overlapping[-1].source_end_seq != source_end_seq
    ):
        return []
    if any(
        turn.source_start_seq < source_start_seq
        or turn.source_end_seq > source_end_seq
        for turn in overlapping
    ):
        return []
    return overlapping


def _validate_projected_turn_coverage(
    projected_messages: list[dict],
    canonical_messages: list[dict],
    turns: list[CompletedTurn],
    *,
    persisted_state: PersistedPreviewState,
    active_placement: tuple[int, int] | None = None,
) -> bool:
    if not turns:
        return False
    source_start = turns[0].source_start_seq
    source_end = turns[-1].source_end_seq
    try:
        span = select_projected_span(
            projected_messages,
            source_start_seq=source_start,
            source_end_seq=source_end,
        )
    except PreviewPlacementError:
        return False

    selected_ranges = [
        (turn.source_start_seq, turn.source_end_seq)
        for turn in turns
    ]
    outer_messages = [
        message
        for message in canonical_messages
        if message.get("role") in {"user", "assistant"}
        and type(message.get("_event_seq")) is int
        and source_start <= message["_event_seq"] <= source_end
    ]
    if any(
        not any(start <= message["_event_seq"] <= end for start, end in selected_ranges)
        for message in outer_messages
    ):
        return False
    expected = tuple(sorted({message["_event_seq"] for message in outer_messages}))
    if not expected or expected[0] != source_start or expected[-1] != source_end:
        return False

    selected = projected_messages[span.start_index:span.end_index]
    singleton_indexes = {}
    for index, message in enumerate(selected):
        source_range = message_source_range(message)
        if source_range is not None and source_range[0] == source_range[1]:
            singleton_indexes.setdefault(source_range[0], []).append(index)

    deferred_singletons = set()
    ignored_singletons = set()
    for seq, indexes in singleton_indexes.items():
        if len(indexes) == 1:
            continue
        nodes = [selected[index] for index in indexes]
        structural = [
            (
                node.get("role"),
                node.get("content"),
                repr(node.get("_render_segments")),
            )
            for node in nodes
        ]
        if len(set(structural)) != len(structural):
            return False
        if not all(
            node.get("role") == "user"
            and is_repl_output_message(node)
            and not _input_segments(node)
            for node in nodes
        ):
            return False
        contiguous = indexes == list(range(indexes[0], indexes[-1] + 1))
        split_around_assistant = (
            len(indexes) == 2
            and indexes[1] == indexes[0] + 2
            and selected[indexes[0] + 1].get("role") == "assistant"
            and message_source_range(selected[indexes[0] + 1]) is not None
            and message_source_range(selected[indexes[0] + 1])[1] < seq
        )
        if contiguous:
            ignored_singletons.update(indexes[1:])
        elif split_around_assistant:
            deferred_singletons.add(indexes[0])
        else:
            return False

    cursor = 0
    for index, message in enumerate(selected):
        if index in deferred_singletons or index in ignored_singletons:
            continue
        source_range = message_source_range(message)
        if source_range is None:
            return False
        node_start, node_end = source_range
        if node_start == node_end:
            if (
                cursor >= len(expected)
                or expected[cursor] != node_start
                or message.get("_persisted_preview")
                or message.get("_preview_event_seq") is not None
            ):
                return False
            cursor += 1
            continue

        try:
            end_cursor = expected.index(node_end, cursor) + 1
        except ValueError:
            return False
        if expected[cursor] != node_start:
            return False
        node_expected = expected[cursor:end_cursor]

        if message.get("_persisted_preview"):
            preview_event_seq = message.get("_preview_event_seq")
            definition = persisted_state.definitions.get(preview_event_seq)
            if (
                active_placement != source_range
                or source_range != (source_start, source_end)
                or len(selected) != 1
                or type(preview_event_seq) is not int
                or persisted_state.active_placements.get(source_range)
                != preview_event_seq
                or definition is None
                or not (message.get("content") or "").rstrip().endswith(
                    render_preview_ref(definition[0], definition[1])
                )
                or node_expected != expected
            ):
                return False
        else:
            if active_placement is not None:
                return False
            if not has_valid_deterministic_identity(message):
                return False
            if node_expected != tuple(
                seq for seq in expected if node_start <= seq <= node_end
            ):
                return False
        cursor = end_cursor

    return cursor == len(expected)


def _rollup_units_from_snapshot(
    turns: list[CompletedTurn],
    messages: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
) -> list[RollupUnit]:
    units = []
    consumed = set()

    for source_start, source_end in sorted(
        _outer_active_placements(persisted_state.active_placements)
    ):
        covered = _range_turns(turns, source_start, source_end)
        if not covered:
            continue
        if not _validate_projected_turn_coverage(
            projected_messages,
            messages,
            covered,
            persisted_state=persisted_state,
            active_placement=(source_start, source_end),
        ):
            continue
        ids = tuple(turn.turn_id for turn in covered)
        units.append(RollupUnit(ids[0], ids[-1], source_start, source_end, ids))
        consumed.update(ids)

    for turn in turns:
        if turn.turn_id in consumed:
            continue
        if not _validate_projected_turn_coverage(
            projected_messages,
            messages,
            [turn],
            persisted_state=persisted_state,
        ):
            continue
        units.append(
            RollupUnit(
                turn.turn_id,
                turn.turn_id,
                turn.source_start_seq,
                turn.source_end_seq,
                (turn.turn_id,),
            )
        )

    order = {turn.turn_id: index for index, turn in enumerate(turns)}
    return sorted(units, key=lambda unit: order[unit.start_turn])


def rollup_units(
    events: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
) -> list[RollupUnit]:
    if not isinstance(persisted_state, PersistedPreviewState):
        raise TypeError("complete persisted preview state is required")
    messages, exec_start_seq = _canonical_messages(events)
    turns = [
        CompletedTurn(
            segment.segment_id,
            segment.source_start_seq,
            segment.source_end_seq,
            segment.has_execution,
            segment.identity_kind,
        )
        for segment in semantic_segments(messages)
        if (
            segment.authoritative
            and exec_start_seq < segment.source_start_seq <= segment.source_end_seq
        )
    ]
    return _rollup_units_from_snapshot(
        turns,
        messages,
        projected_messages,
        persisted_state,
    )


def _validate_rollup_interval_snapshot(
    turns: list[CompletedTurn],
    canonical_messages: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
    units: list[RollupUnit],
) -> bool:
    """Validate that units form an exact complete canonical/projected source interval."""
    if not units:
        return False
    source_start = units[0].source_start_seq
    source_end = units[-1].source_end_seq
    covered = _range_turns(turns, source_start, source_end)
    if tuple(turn.turn_id for turn in covered) != tuple(
        turn_id for unit in units for turn_id in unit.turn_ids
    ):
        return False

    for unit in units:
        unit_turns = _range_turns(
            turns,
            unit.source_start_seq,
            unit.source_end_seq,
        )
        active_placement = (
            (unit.source_start_seq, unit.source_end_seq)
            if persisted_state.active_placements.get(
                (unit.source_start_seq, unit.source_end_seq)
            )
            is not None
            else None
        )
        if not _validate_projected_turn_coverage(
            projected_messages,
            canonical_messages,
            unit_turns,
            persisted_state=persisted_state,
            active_placement=active_placement,
        ):
            return False

    try:
        outer_span = select_projected_span(
            projected_messages,
            source_start_seq=source_start,
            source_end_seq=source_end,
        )
        unit_spans = [
            select_projected_span(
                projected_messages,
                source_start_seq=unit.source_start_seq,
                source_end_seq=unit.source_end_seq,
            )
            for unit in units
        ]
    except PreviewPlacementError:
        return False
    return (
        unit_spans[0].start_index == outer_span.start_index
        and unit_spans[-1].end_index == outer_span.end_index
        and all(
            left.end_index == right.start_index
            for left, right in zip(unit_spans, unit_spans[1:])
        )
    )


def validate_rollup_interval(
    events: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
    units: list[RollupUnit],
) -> bool:
    """Validate that units form an exact complete canonical/projected source interval."""
    turns = completed_turns(events)
    canonical_messages, _ = _canonical_messages(events)
    return _validate_rollup_interval_snapshot(
        turns,
        canonical_messages,
        projected_messages,
        persisted_state,
        units,
    )


def derive_rollup_eligibility(
    events: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
) -> RollupEligibility:
    if not isinstance(persisted_state, PersistedPreviewState):
        raise TypeError("complete persisted preview state is required")
    canonical_messages, exec_start_seq = _canonical_messages(events)
    turns = [
        CompletedTurn(
            segment.segment_id,
            segment.source_start_seq,
            segment.source_end_seq,
            segment.has_execution,
            segment.identity_kind,
        )
        for segment in semantic_segments(canonical_messages)
        if (
            segment.authoritative
            and exec_start_seq < segment.source_start_seq <= segment.source_end_seq
        )
    ]
    protected = {turn.turn_id for turn in turns[-3:]}
    execution_turns = [turn for turn in turns if turn.has_execution]
    if execution_turns:
        protected.add(execution_turns[-1].turn_id)
    all_units = _rollup_units_from_snapshot(
        turns,
        canonical_messages,
        projected_messages,
        persisted_state,
    )
    units = [
        unit
        for unit in all_units
        if not any(turn_id in protected for turn_id in unit.turn_ids)
    ]

    turn_order = {turn.turn_id: index for index, turn in enumerate(turns)}
    unit_spans = {}
    for unit in units:
        try:
            unit_spans[unit] = select_projected_span(
                projected_messages,
                source_start_seq=unit.source_start_seq,
                source_end_seq=unit.source_end_seq,
            )
        except PreviewPlacementError:
            continue

    def shares_exact_boundary(left, right):
        left_span = unit_spans.get(left)
        right_span = unit_spans.get(right)
        if left_span is None or right_span is None:
            return False
        return (
            turn_order[left.end_turn] + 1 == turn_order[right.start_turn]
            and left_span.end_index == right_span.start_index
        )

    groups = []
    current = []
    for unit in units:
        if current and not shares_exact_boundary(current[-1], unit):
            if sum(len(item.turn_ids) for item in current) >= 2:
                groups.append(current)
            current = []
        current.append(unit)
    if current and sum(len(item.turn_ids) for item in current) >= 2:
        groups.append(current)
    return RollupEligibility(
        all_units=tuple(all_units),
        units=tuple(unit for group in groups for unit in group),
        groups=tuple(tuple(group) for group in groups),
    )



def _replacement_source_ranges(
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
) -> list[tuple[int, int]]:
    ranges = list(_outer_active_placements(persisted_state.active_placements))
    for message in projected_messages:
        if message.get("_persisted_preview"):
            continue
        if not has_valid_deterministic_identity(message):
            continue
        source_range = message_source_range(message)
        if source_range is not None:
            ranges.append(source_range)
    return ranges


def _surviving_rollup_boundary_indexes(
    turns: list[CompletedTurn],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
) -> list[int]:
    connected = {index: {index} for index in range(len(turns))}

    for source_start, source_end in _replacement_source_ranges(
        projected_messages,
        persisted_state,
    ):
        overlapping = [
            index
            for index, turn in enumerate(turns)
            if turn.source_start_seq <= source_end
            and source_start <= turn.source_end_seq
        ]
        if len(overlapping) < 2:
            continue
        merged = set().union(*(connected[index] for index in overlapping))
        for index in merged:
            connected[index] = merged

    hidden = set()
    seen = set()
    for index in range(len(turns)):
        component = connected[index]
        if index in seen or len(component) < 2:
            continue
        seen.update(component)
        hidden.update(range(min(component) + 1, max(component)))

    return [
        index
        for index in range(len(turns))
        if index not in hidden
    ]


def _boundary_unit(turn: CompletedTurn) -> RollupUnit:
    return RollupUnit(
        turn.turn_id,
        turn.turn_id,
        turn.source_start_seq,
        turn.source_end_seq,
        (turn.turn_id,),
    )


def derive_rollup_boundary_eligibility(
    events: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
    *,
    exact_messages: list[dict] | None = None,
) -> RollupEligibility:
    if not isinstance(persisted_state, PersistedPreviewState):
        raise TypeError("complete persisted preview state is required")
    turns = completed_turns(events)
    if not turns:
        return RollupEligibility((), (), ())

    if exact_messages is None:
        exact_messages = projected_messages
    canonical_messages, _ = _canonical_messages(events)
    exact_units = _rollup_units_from_snapshot(
        turns,
        canonical_messages,
        exact_messages,
        persisted_state,
    )
    valid_turn_ids = {
        turn_id
        for unit in exact_units
        for turn_id in unit.turn_ids
    }
    surviving = _surviving_rollup_boundary_indexes(
        turns,
        projected_messages,
        persisted_state,
    )

    protected = {turn.turn_id for turn in turns[-3:]}
    execution_turns = [turn for turn in turns if turn.has_execution]
    if execution_turns:
        protected.add(execution_turns[-1].turn_id)

    all_boundaries = tuple(_boundary_unit(turns[index]) for index in surviving)
    groups = []
    current = []
    for index in surviving:
        turn = turns[index]
        if turn.turn_id in protected:
            if len(current) >= 2:
                groups.append(tuple(current))
            current = []
            continue
        if current:
            previous_index = next(
                candidate
                for candidate in reversed(surviving)
                if candidate < index
            )
            covered_ids = {
                item.turn_id
                for item in turns[previous_index:index]
            }
            if not covered_ids or not covered_ids.issubset(valid_turn_ids):
                if len(current) >= 2:
                    groups.append(tuple(current))
                current = []
        current.append(_boundary_unit(turn))
    if len(current) >= 2:
        groups.append(tuple(current))

    units = tuple(unit for group in groups for unit in group)
    return RollupEligibility(all_boundaries, units, tuple(groups))


def resolve_rollup_boundary_interval(
    eligibility: RollupEligibility,
    turns: list[CompletedTurn],
    start_turn: int,
    end_turn: int,
) -> tuple[int, int, tuple[int, ...]]:
    selected_group = next(
        (
            group
            for group in eligibility.groups
            if any(unit.start_turn == start_turn for unit in group)
            and any(unit.start_turn == end_turn for unit in group)
        ),
        None,
    )
    if selected_group is None:
        starts = {
            unit.start_turn
            for group in eligibility.groups
            for unit in group
        }
        if start_turn not in starts:
            raise ValueError("start_turn is not an eligible boundary.")
        if end_turn not in starts:
            raise ValueError("end_turn is not an eligible boundary.")
        raise ValueError(
            "rollup boundaries must come from one advertised bracketed group."
        )

    order = {turn.turn_id: index for index, turn in enumerate(turns)}
    start_index = order[start_turn]
    end_index = order[end_turn]
    if start_index >= end_index:
        raise ValueError("rollup endpoints are reversed or identical.")

    covered = turns[start_index:end_index]
    return (
        covered[0].source_start_seq,
        covered[-1].source_end_seq,
        tuple(turn.turn_id for turn in covered),
    )


def derive_agent_rollup_context(agent):
    from code_agent.code_agent_coalesce import coalesce_repl_messages

    store = getattr(agent, "_session_store", None)
    session_id = getattr(agent, "_session_id", None)
    if store is None or session_id is None:
        return None

    events = store.get_events(session_id)
    exact_messages, state = agent._authoritative_persisted_projection(
        coalesce=False
    )
    projected_messages = coalesce_repl_messages(
        exact_messages,
        keep_last_execution_interactions=(
            agent.code_agent_coalesce_keep_last_execution_interactions
        ),
        min_savings_chars=agent.code_agent_coalesce_min_savings_chars,
    )
    eligibility = derive_rollup_boundary_eligibility(
        events,
        projected_messages,
        state,
        exact_messages=exact_messages,
    )
    return events, exact_messages, state, eligibility


def eligible_rollup_groups(
    events: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
) -> list[list[RollupUnit]]:
    return [
        list(group)
        for group in derive_rollup_eligibility(
            events,
            projected_messages,
            persisted_state,
        ).groups
    ]


def eligible_rollup_units(
    events: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
) -> list[RollupUnit]:
    return list(
        derive_rollup_eligibility(
            events,
            projected_messages,
            persisted_state,
        ).units
    )


def eligible_rollup_line(groups: list[list[RollupUnit]]) -> str:
    if not groups:
        return ""

    def render_group(group):
        entries = [
            str(unit.start_turn)
            if unit.start_turn == unit.end_turn
            else f"{unit.start_turn}-{unit.end_turn}"
            for unit in group
        ]
        return "[" + ", ".join(entries) + "]"

    return (
        "Eligible rollup turns: "
        + " | ".join(render_group(group) for group in groups)
        + " (combine boundaries only within one bracketed group)"
    )