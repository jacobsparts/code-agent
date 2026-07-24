import copy
import re
from dataclasses import dataclass

from code_agent.code_agent_coalesce import (
    PreviewPlacementError,
    deterministic_interaction_replacement,
    deterministic_structural_identity,
    has_valid_deterministic_identity,
    is_release_assistant_message,
    is_repl_output_message,
    message_source_range,
    normalize_repl_messages,
    render_preview_ref,
    select_projected_span,
)
from code_agent.persisted_preview_state import PersistedPreviewState
from code_agent.session_message_state import reduce_canonical_message_events


@dataclass(frozen=True)
class CompletedTurn:
    turn_id: int
    source_start_seq: int
    source_end_seq: int
    has_execution: bool


@dataclass(frozen=True)
class RollupUnit:
    start_turn: int
    end_turn: int
    source_start_seq: int
    source_end_seq: int
    turn_ids: tuple[int, ...]


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
    completed = []
    start_index = None
    start_seq = None
    previous_release_output = None
    for index, message in enumerate(messages):
        seq = message.get("_event_seq")
        if index == previous_release_output:
            continue
        if message.get("_synthetic") and not message.get("_virtual_interaction_boundary"):
            continue
        input_seq = _canonical_input_seq(message)
        if input_seq is not None:
            start_index = index
            start_seq = input_seq
            continue
        if (
            start_index is not None
            and message.get("role") == "assistant"
            and is_release_assistant_message(message)
        ):
            release_output_index = None
            if index + 1 < len(messages):
                following = messages[index + 1]
                if (
                    following.get("role") == "user"
                    and is_repl_output_message(following)
                    and not _input_segments(following)
                    and not (
                        following.get("_synthetic")
                        and not following.get("_coalesced")
                    )
                ):
                    release_output_index = index + 1
            end_message = (
                messages[release_output_index]
                if release_output_index is not None
                else message
            )
            end_seq = end_message.get("_event_seq")
            between = messages[start_index + 1:index]
            has_execution = (
                any(item.get("role") == "assistant" for item in between)
                and any(
                    item.get("role") == "user" and is_repl_output_message(item)
                    for item in between
                )
            )
            if (
                type(start_seq) is int
                and type(end_seq) is int
                and exec_start_seq < start_seq <= end_seq
            ):
                completed.append(
                    CompletedTurn(start_seq, start_seq, end_seq, has_execution)
                )
            start_index = None
            start_seq = None
            previous_release_output = release_output_index
    return completed


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
    expected = tuple(message["_event_seq"] for message in outer_messages)
    if not expected or expected[0] != source_start or expected[-1] != source_end:
        return False

    selected = projected_messages[span.start_index:span.end_index]
    cursor = 0
    for message in selected:
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
            canonical_projection = [
                {"role": "system", "content": ""},
                *copy.deepcopy(canonical_messages),
            ]
            canonical_blocks = {}
            for canonical_message in canonical_messages:
                for match in re.finditer(
                    r"\[PreviewRef: (?P<uri>session://preview/[^\]\n]+)\]\n"
                    r".*?"
                    r"\[/PreviewRef\]",
                    canonical_message.get("content") or "",
                    re.DOTALL,
                ):
                    canonical_blocks[match.group("uri")] = match.group(0)
            preserve_preview_refs = {
                uri: block
                for uri, block in canonical_blocks.items()
                if block in (message.get("content") or "")
            }
            if not has_valid_deterministic_identity(message):
                return False
            derived = deterministic_interaction_replacement(
                canonical_projection,
                source_start_seq=node_start,
                source_end_seq=node_end,
                preserve_preview_refs=preserve_preview_refs,
            )
            if (
                derived is None
                or deterministic_structural_identity(message)
                != deterministic_structural_identity(derived[0])
            ):
                return False
        cursor = end_cursor

    return cursor == len(expected)


def rollup_units(
    events: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
) -> list[RollupUnit]:
    if not isinstance(persisted_state, PersistedPreviewState):
        raise TypeError("complete persisted preview state is required")
    turns = completed_turns(events)
    messages, _ = _canonical_messages(events)
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


def validate_rollup_interval(
    events: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
    units: list[RollupUnit],
) -> bool:
    """Validate that units form an exact complete canonical/projected source interval."""
    if not units:
        return False
    turns = completed_turns(events)
    canonical_messages, _ = _canonical_messages(events)
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


def eligible_rollup_units(
    events: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
) -> list[RollupUnit]:
    turns = completed_turns(events)
    protected = {turn.turn_id for turn in turns[-3:]}
    execution_turns = [turn for turn in turns if turn.has_execution]
    if execution_turns:
        protected.add(execution_turns[-1].turn_id)
    return [
        unit
        for unit in rollup_units(events, projected_messages, persisted_state)
        if not any(turn_id in protected for turn_id in unit.turn_ids)
    ]


def eligible_rollup_line(units: list[RollupUnit]) -> str:
    if sum(len(unit.turn_ids) for unit in units) < 2:
        return ""
    entries = [
        str(unit.start_turn)
        if unit.start_turn == unit.end_turn
        else f"{unit.start_turn}-{unit.end_turn}"
        for unit in units
    ]
    return "Eligible rollup turns: " + ", ".join(entries)
