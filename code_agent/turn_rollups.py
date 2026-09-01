import copy
import re
from dataclasses import dataclass

from code_agent.code_agent_coalesce import (
    has_valid_deterministic_identity,
    is_repl_output_message,
    message_source_range,
    normalize_repl_messages,
    select_projected_span,
    semantic_boundaries,
    semantic_segments,
)
from code_agent.persisted_preview_state import PersistedPreviewState
from code_agent.session_message_state import reduce_canonical_message_events


def _content_text(message: dict) -> str:
    content = message.get("content") or []
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _replace_text_content(message: dict, text: str) -> None:
    message["content"] = [{"type": "text", "text": text}]


@dataclass(frozen=True)
class CompletedTurn:
    turn_id: int
    source_start_seq: int
    source_end_seq: int
    has_execution: bool
    identity_kind: str = "turn"


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
    content = _content_text(message)
    if content and type(seq) is int:
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
                "content": [{"type": "text", "text": f"# Checkpoint {checkpoint}"}],
                "_synthetic": True,
                "_provider_checkpoint": True,
            })
    return out


def render_turn_labels(message: dict) -> dict:
    out = copy.deepcopy(message)
    if out.get("role") != "user" or out.get("_synthetic"):
        return out
    content = _content_text(out)
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
    _replace_text_content(out, content)
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


def _replacement_source_ranges(
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
) -> list[tuple[int, int]]:
    ranges = list(_outer_active_placements(persisted_state.active_placements))
    for message in projected_messages:
        if message.get("_persisted_preview"):
            continue
        if message.get("_coalesced") and not has_valid_deterministic_identity(message):
            raise RuntimeError(
                "deterministic coalesced node has invalid identity while building "
                "rollup candidates"
            )
        if not has_valid_deterministic_identity(message):
            continue
        source_range = message_source_range(message)
        if source_range is None:
            raise RuntimeError(
                "deterministic coalesced node has no source range while building "
                "rollup candidates"
            )
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
        hidden.update(range(min(component) + 1, max(component) + 1))

    return [
        index
        for index in range(len(turns))
        if index not in hidden
    ]


def _exact_projected_boundary_turns(
    turns: list[CompletedTurn],
    projected_messages: list[dict],
) -> list[int]:
    """Return indexes of turns usable as either endpoint of a rollup range.

    A range resolves to the selected start turn's projected-node start and the
    projected-node end of the turn immediately before the selected end turn, so
    both boundaries must match exactly one projected node.
    """
    starts: dict[int, list[int]] = {}
    ends: dict[int, list[int]] = {}
    for index, message in enumerate(projected_messages):
        source_range = message_source_range(message)
        if source_range is None:
            continue
        starts.setdefault(source_range[0], []).append(index)
        ends.setdefault(source_range[1], []).append(index)

    usable = []
    for index, turn in enumerate(turns):
        if len(starts.get(turn.source_start_seq, [])) != 1:
            continue
        if index and len(ends.get(turns[index - 1].source_end_seq, [])) != 1:
            continue
        usable.append(index)
    return usable


def derive_rollup_candidate_turns(
    events: list[dict],
    projected_messages: list[dict],
    persisted_state: PersistedPreviewState,
) -> tuple[int, ...]:
    """Return every turn number that may be passed to rollup().

    Any two listed numbers, in listed order, form a valid rollup range.
    """
    if not isinstance(persisted_state, PersistedPreviewState):
        raise TypeError("complete persisted preview state is required")
    turns = completed_turns(events)
    if not turns:
        return ()

    frontier = len(turns) - 3
    execution_indexes = [
        index for index, turn in enumerate(turns) if turn.has_execution
    ]
    if execution_indexes:
        frontier = min(frontier, execution_indexes[-1])
    if frontier < 1:
        return ()

    usable = set(_exact_projected_boundary_turns(turns, projected_messages))
    indexes = [
        index
        for index in _surviving_rollup_boundary_indexes(
            turns,
            projected_messages,
            persisted_state,
        )
        if index <= frontier and index in usable
    ]
    if len(indexes) < 2:
        return ()

    return tuple(turns[index].turn_id for index in indexes)


def resolve_rollup_interval(
    candidate_turns: tuple[int, ...],
    turns: list[CompletedTurn],
    start_turn: int,
    end_turn: int,
) -> tuple[int, int, tuple[int, ...]]:
    listed = ", ".join(str(turn_id) for turn_id in candidate_turns)
    for value in (start_turn, end_turn):
        if value not in candidate_turns:
            raise ValueError(
                f"Turn {value} is not available for rollup. "
                f"Choose two of these turn numbers: {listed}."
            )
    if start_turn == end_turn:
        raise ValueError(
            "A rollup needs two different turn numbers. "
            f"Choose two of these turn numbers: {listed}."
        )

    order = {turn.turn_id: index for index, turn in enumerate(turns)}
    start_index = order[start_turn]
    end_index = order[end_turn]
    if start_index > end_index:
        raise ValueError(
            f"Turn {start_turn} comes after turn {end_turn}. "
            "Pass the earlier turn first."
        )

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
    candidate_turns = derive_rollup_candidate_turns(
        events,
        projected_messages,
        state,
    )
    rolled_up_upper_turns = rolled_up_upper_boundary_turns(
        events,
        state,
        candidate_turns,
    )
    return (
        events,
        exact_messages,
        state,
        candidate_turns,
        rolled_up_upper_turns,
    )


def rolled_up_upper_boundary_turns(
    events: list[dict],
    persisted_state: PersistedPreviewState,
    candidate_turns: tuple[int, ...],
) -> tuple[int, ...]:
    turns = completed_turns(events)
    next_turn_by_end = {
        turns[index].source_end_seq: turns[index + 1].turn_id
        for index in range(len(turns) - 1)
    }
    candidate_set = set(candidate_turns)
    return tuple(
        turn_id
        for _, source_end in _outer_active_placements(
            persisted_state.active_placements
        )
        if (turn_id := next_turn_by_end.get(source_end)) in candidate_set
    )


def eligible_rollup_line(
    candidate_turns: tuple[int, ...],
    rolled_up_upper_turns: tuple[int, ...] = (),
) -> str:
    if len(candidate_turns) < 2:
        return ""
    rolled_up = set(rolled_up_upper_turns)
    ordinary = tuple(turn for turn in candidate_turns if turn not in rolled_up)
    lines = []
    if ordinary:
        lines.append(
            "Eligible normal turn boundaries: "
            + ", ".join(str(turn_id) for turn_id in ordinary)
        )
    if rolled_up_upper_turns:
        lines.append(
            "Eligible existing-rollup upper boundaries: "
            + ", ".join(str(turn_id) for turn_id in rolled_up_upper_turns)
        )
    return "\n".join(lines)