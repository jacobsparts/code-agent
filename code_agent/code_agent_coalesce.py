import ast
import copy
import hashlib
import hmac
import pickle
import re
import secrets
from dataclasses import dataclass

from code_agent.persisted_preview_state import PersistedPreviewState
from code_agent.session_replay import _parse_silently


OMITTED_ECHO_MARKER = "[content omitted from echo]"
_DETERMINISTIC_IDENTITY_FIELD = "_deterministic_preview_identity"
_DETERMINISTIC_IDENTITY_KEY = secrets.token_bytes(32)


def _deterministic_identity_payload(message: dict) -> bytes:
    payload = copy.deepcopy(message)
    payload.pop(_DETERMINISTIC_IDENTITY_FIELD, None)
    return pickle.dumps(payload, protocol=5)


def _sign_deterministic_replacement(message: dict) -> None:
    message[_DETERMINISTIC_IDENTITY_FIELD] = hmac.digest(
        _DETERMINISTIC_IDENTITY_KEY,
        _deterministic_identity_payload(message),
        "sha256",
    )


def has_valid_deterministic_identity(message: dict) -> bool:
    identity = message.get(_DETERMINISTIC_IDENTITY_FIELD)
    if not isinstance(identity, bytes):
        return False
    expected = hmac.digest(
        _DETERMINISTIC_IDENTITY_KEY,
        _deterministic_identity_payload(message),
        "sha256",
    )
    return hmac.compare_digest(identity, expected)


def deterministic_structural_identity(message: dict) -> dict:
    identity = copy.deepcopy(message)
    identity.pop(_DETERMINISTIC_IDENTITY_FIELD, None)
    identity.pop("_attachments", None)
    return identity


def _content_preserves_context_refs(content: str) -> bool:
    return (
        "[Attachment:" in content
        or "[PreviewRef:" in content
        or "[ExpandedPreviewRef:" in content
    )


@dataclass(frozen=True)
class Preview:
    summary: str
    content: str | None = None

    def __post_init__(self):
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("preview summary must be a non-empty string")
        if self.content is not None and not isinstance(self.content, str):
            raise ValueError("preview content must be a string or None")


@dataclass(frozen=True)
class ProjectedSpan:
    start_index: int
    end_index: int
    source_start_seq: int
    source_end_seq: int


@dataclass(frozen=True)
class SemanticBoundary:
    boundary_index: int
    boundary_output_index: int | None
    transition_seq: int | None
    is_transition: bool
    authoritative: bool


@dataclass(frozen=True)
class SemanticSegment:
    segment_id: int
    identity_kind: str
    anchor_index: int
    boundary_index: int
    boundary_output_index: int | None
    source_start_seq: int
    source_end_seq: int
    has_execution: bool
    authoritative: bool


class PreviewPlacementError(ValueError):
    pass


class PreviewBoundaryError(PreviewPlacementError):
    pass


def message_source_range(message: dict) -> tuple[int, int] | None:
    start = message.get("_source_start_seq")
    end = message.get("_source_end_seq")
    if start is not None and end is not None:
        return start, end
    seq = message.get("_event_seq")
    if seq is not None:
        return seq, seq
    return None


def preview_key(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def render_preview_ref(key: str, summary: str) -> str:
    uri = f"session://preview/{key}"
    return f"[PreviewRef: {uri}]\n{summary}\n[/PreviewRef]"


def render_default_preview_summary(content: str, observations=None) -> str:
    lines = content.split("\n")
    nlines = len(lines)
    nchars = len(content)
    valid_observations = [
        observation
        for observation in observations or []
        if isinstance(observation, str) and observation.strip()
    ]
    if valid_observations:
        parts = ["Observations:"]
        for observation in valid_observations:
            observation_lines = observation.split("\n")
            parts.append(f"- {observation_lines[0]}")
            parts.extend(
                f"  {line}" if line else ""
                for line in observation_lines[1:]
            )
        parts.append(f"({nlines} lines, {nchars} chars)")
        return "\n".join(parts)

    def render_preview_line(line):
        max_preview_line = 500
        if len(line) <= max_preview_line:
            return line
        return f"{line[:max_preview_line]}... [line truncated, {len(line)} chars total]"

    head = 8
    tail = 4
    head_indexes = list(range(min(head, nlines)))
    tail_start = max(len(head_indexes), nlines - tail)
    omitted = nlines - len(head_indexes) - (nlines - tail_start)
    parts = [f"({nlines} lines, {nchars} chars)"]
    parts.extend(render_preview_line(lines[i]) for i in head_indexes)
    if omitted:
        parts.append(f"  ... ({omitted} lines omitted)")
    parts.extend(render_preview_line(lines[i]) for i in range(tail_start, nlines))
    return "\n".join(parts)




def select_projected_span(
    messages: list[dict],
    *,
    source_start_seq: int,
    source_end_seq: int,
) -> ProjectedSpan:
    if type(source_start_seq) is not int or type(source_end_seq) is not int:
        raise PreviewPlacementError("preview source boundaries must be integers")
    if source_start_seq > source_end_seq:
        raise PreviewPlacementError(
            f"invalid preview source range [{source_start_seq}, {source_end_seq}]"
        )

    requested = f"[{source_start_seq}, {source_end_seq}]"
    overlapping = []
    start_matches = []
    end_matches = []
    for index, message in enumerate(messages):
        source_range = message_source_range(message)
        if source_range is None:
            continue
        node_start, node_end = source_range
        if type(node_start) is not int or type(node_end) is not int:
            raise PreviewPlacementError("projected-node source boundaries must be integers")
        if node_start > node_end:
            raise PreviewPlacementError(
                f"invalid projected-node source range [{node_start}, {node_end}]"
            )
        if node_end < source_start_seq or node_start > source_end_seq:
            continue
        overlapping.append((index, node_start, node_end))
        if node_start == source_start_seq:
            start_matches.append(index)
        if node_end == source_end_seq:
            end_matches.append(index)
        if node_start < source_start_seq or node_end > source_end_seq:
            raise PreviewBoundaryError(
                f"requested range {requested} splits or partially overlaps "
                f"projected node [{node_start}, {node_end}]"
            )

    if len(start_matches) > 1:
        raise PreviewBoundaryError(
            f"ambiguous projected start boundary for requested range {requested}"
        )
    if len(end_matches) > 1:
        raise PreviewBoundaryError(
            f"ambiguous projected end boundary for requested range {requested}"
        )
    if not start_matches or not end_matches:
        raise PreviewBoundaryError(
            f"missing projected boundary for requested range {requested}"
        )

    start_index = start_matches[0]
    end_index = end_matches[0] + 1
    if start_index >= end_index:
        raise PreviewBoundaryError(
            f"source-aware projected nodes are not ordered for requested range {requested}"
        )

    expected_indexes = list(range(start_index, end_index))
    overlapping_indexes = [index for index, _, _ in overlapping]
    if overlapping_indexes != expected_indexes:
        for index in expected_indexes:
            if message_source_range(messages[index]) is None:
                raise PreviewBoundaryError(
                    f"source-less projected node at index {index} lies inside "
                    f"source-aware requested range {requested}"
                )
        raise PreviewBoundaryError(
            f"source-aware nodes do not form one exact projected span for "
            f"requested range {requested}"
        )

    previous = None
    for _, node_start, node_end in overlapping:
        if previous is not None and node_start <= previous[1]:
            raise PreviewPlacementError(
                f"non-contiguous projected span: node [{node_start}, {node_end}] "
                f"overlaps or precedes [{previous[0]}, {previous[1]}]"
            )
        previous = (node_start, node_end)

    if overlapping[0][1] != source_start_seq or overlapping[-1][2] != source_end_seq:
        raise PreviewBoundaryError(
            f"selected projected nodes do not uniquely cover requested range {requested}"
        )

    return ProjectedSpan(
        start_index=start_index,
        end_index=end_index,
        source_start_seq=source_start_seq,
        source_end_seq=source_end_seq,
    )


def is_release_assistant_message(msg: dict) -> bool:
    if msg.get("_final_result") is not None or msg.get("_emit_value") is not None:
        return True

    content = msg.get("content") or ""
    try:
        tree = _parse_silently(content)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            expr = node.value if isinstance(node, ast.Expr) else None
            if not isinstance(expr, ast.Call):
                continue
            if not isinstance(expr.func, ast.Name) or expr.func.id != "emit":
                continue
            for kw in expr.keywords:
                if kw.arg == "release":
                    try:
                        return bool(ast.literal_eval(kw.value))
                    except Exception:
                        return False
            return False

    stripped = content.lstrip()
    return stripped.startswith("emit(") and "release=True" in stripped.replace(" ", "")


def released_assistant_text(msg: dict) -> str:
    value = msg.get("_final_result")
    if value is None:
        value = msg.get("_emit_value")
    if value is not None:
        return str(value)

    content = msg.get("content") or ""
    try:
        tree = _parse_silently(content)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            expr = node.value if isinstance(node, ast.Expr) else None
            if not isinstance(expr, ast.Call):
                continue
            if not isinstance(expr.func, ast.Name) or expr.func.id != "emit":
                continue
            released = False
            for kw in expr.keywords:
                if kw.arg == "release":
                    try:
                        released = bool(ast.literal_eval(kw.value))
                    except Exception:
                        released = False
                    break
            if not released or not expr.args:
                continue
            try:
                return str(ast.literal_eval(expr.args[0]))
            except Exception:
                return ""

    return ""


def message_stdout(msg: dict) -> str:
    content = msg.get("content") or ""
    if _content_preserves_context_refs(content):
        return content
    return msg.get("_stdout") or content


def _structured_render_segments(msg: dict) -> list[dict] | None:
    if "_render_segments" not in msg:
        return None
    segments = msg.get("_render_segments")
    if not isinstance(segments, list) or any(not isinstance(seg, dict) for seg in segments):
        return None
    return segments


def is_repl_output_message(msg: dict) -> bool:
    segments = _structured_render_segments(msg)
    if segments is not None:
        return any(seg.get("type") == "stdout" for seg in segments)
    content = msg.get("content") or ""
    return content.lstrip().startswith(">>>") or OMITTED_ECHO_MARKER in content


def human_inputs(msg: dict) -> list[str]:
    segments = _structured_render_segments(msg)
    if segments is not None:
        return [
            str(seg.get("content"))
            for seg in segments
            if seg.get("type") == "input"
            and seg.get("content") is not None
            and str(seg.get("content"))
        ]
    if msg.get("_user_content") is not None:
        return [str(msg.get("_user_content"))]
    if is_repl_output_message(msg):
        return []
    content = msg.get("content") or ""
    return [content] if content else []


def split_appended_user_content(msg: dict, text: str) -> tuple[str, str | None]:
    user_content = msg.get("_user_content")
    if user_content is None:
        return text, None
    suffix = str(user_content)
    if not suffix:
        return text, suffix
    candidates = [suffix, suffix.rstrip("\n")]
    for candidate in candidates:
        if candidate and text.endswith(candidate):
            stripped = text[: -len(candidate)].rstrip("\n")
            return stripped, suffix
    return text, suffix

def split_release_repl_output(text: str) -> tuple[str, str | None]:
    if not text:
        return text, None
    offset = 0
    split_at = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        compact = stripped.replace(" ", "")
        if stripped.startswith(">>>") and "emit(" in stripped and "release=True" in compact:
            split_at = offset
        offset += len(line)
    if split_at is None or split_at <= 0:
        return text, None
    prefix = text[:split_at].rstrip("\n")
    release = text[split_at:]
    if not prefix or not release.strip():
        return text, None
    return prefix, release


def reconstruct_omitted_echo(assistant_code: str, repl_output: str) -> str:
    if OMITTED_ECHO_MARKER not in repl_output:
        return repl_output
    assistant_code = assistant_code or ""
    if not assistant_code.strip():
        return repl_output
    echo_lines = []
    for i, line in enumerate(assistant_code.rstrip("\n").splitlines()):
        prefix = ">>> " if i == 0 else "... "
        echo_lines.append(prefix + line)
    return repl_output.replace(OMITTED_ECHO_MARKER, "\n".join(echo_lines), 1)


def _append_block(parts: list[str], text: str):
    if text is None:
        return
    text = str(text).strip("\n")
    if not text:
        return
    if parts:
        parts.append("")
    parts.append(text)


def _real_user_message(msg: dict) -> bool:
    if msg.get("_synthetic"):
        return False
    return msg.get("role") == "user" and bool(human_inputs(msg))


def _message_from_segments(template: dict, segments: list[dict], *, user_input: bool) -> dict:
    msg = copy.deepcopy(template)
    msg["_render_segments"] = copy.deepcopy(segments)
    content = "".join(str(segment.get("content") or "") for segment in segments)
    msg["content"] = content
    seqs = {
        segment.get("_event_seq")
        for segment in segments
        if type(segment.get("_event_seq")) is int
    }
    if len(seqs) == 1:
        msg["_event_seq"] = next(iter(seqs))
        msg.pop("_source_start_seq", None)
        msg.pop("_source_end_seq", None)
    elif seqs:
        msg.pop("_event_seq", None)
        msg["_source_start_seq"] = min(seqs)
        msg["_source_end_seq"] = max(seqs)
    if user_input:
        msg["_user_content"] = content
        msg.pop("_stdout", None)
    else:
        msg.pop("_user_content", None)
        if msg.get("_stdout") is not None:
            msg["_stdout"] = content
    return msg


def _stdout_message_from(template: dict, content: str, segment: dict | None = None) -> dict:
    stdout_segment = copy.deepcopy(segment) if segment is not None else {
        "type": "stdout",
        "content": content,
    }
    stdout_segment["content"] = content
    return _message_from_segments(template, [stdout_segment], user_input=False)


def _split_structured_appended_user(msg: dict) -> tuple[list[dict], dict] | None:
    segments = _structured_render_segments(msg)
    if segments is None:
        return None
    input_indexes = [
        index
        for index, segment in enumerate(segments)
        if segment.get("type") == "input"
    ]
    if len(input_indexes) != 1:
        return None
    input_index = input_indexes[0]
    if any(
        segment.get("type") == "input"
        for segment in segments[:input_index]
    ) or any(
        segment.get("type") != "input"
        and str(segment.get("content") or "")
        for segment in segments[input_index + 1:]
    ):
        return None
    stdout_segments = [
        segment for segment in segments[:input_index]
        if segment.get("type") == "stdout"
    ]
    if not stdout_segments:
        return None
    input_message = _message_from_segments(
        msg,
        [segments[input_index]],
        user_input=True,
    )
    return stdout_segments, input_message


def normalize_repl_messages(messages: list[dict]) -> list[dict]:
    out = []
    for msg in messages:
        structured = (
            _split_structured_appended_user(msg)
            if msg.get("role") == "user"
            else None
        )
        if structured is not None:
            stdout_segments, input_message = structured
            stdout_content = "".join(
                str(segment.get("content") or "")
                for segment in stdout_segments
            )
            prefix, release_text = split_release_repl_output(stdout_content)
            if (
                release_text is not None
                and out
                and out[-1].get("role") == "assistant"
                and is_release_assistant_message(out[-1])
            ):
                release_msg = out.pop()
                out.append(_stdout_message_from(msg, prefix, stdout_segments[0]))
                out.append(release_msg)
                out.append(_stdout_message_from(msg, release_text, stdout_segments[-1]))
            else:
                out.append(_message_from_segments(msg, stdout_segments, user_input=False))
            out.append(input_message)
            continue

        if msg.get("role") == "user" and is_repl_output_message(msg) and msg.get("_user_content") is not None:
            copied = copy.deepcopy(msg)
            text, appended = split_appended_user_content(copied, copied.get("content") or "")
            stdout = None
            if copied.get("_stdout") is not None:
                stdout, _ = split_appended_user_content(copied, copied.get("_stdout") or "")

            prefix, release_text = split_release_repl_output(stdout if stdout is not None else text)
            if release_text is not None and out and out[-1].get("role") == "assistant" and is_release_assistant_message(out[-1]):
                release_msg = out.pop()
                out.append(_stdout_message_from(copied, prefix))
                out.append(release_msg)
                out.append(_stdout_message_from(copied, release_text))
            else:
                copied["content"] = text
                if stdout is not None:
                    copied["_stdout"] = stdout
                copied.pop("_user_content", None)
                out.append(copied)

            if appended is not None:
                appended_msg = {"role": "user", "content": appended, "_user_content": appended}
                if type(copied.get("_event_seq")) is int:
                    appended_msg["_event_seq"] = copied["_event_seq"]
                out.append(appended_msg)
        else:
            out.append(copy.deepcopy(msg))
    return out



def _interaction_has_execution(messages: list[dict], anchor: int, boundary: int) -> bool:
    range_messages = messages[anchor + 1:boundary]
    has_work = any(msg.get("role") == "assistant" for msg in range_messages)
    has_output = any(
        msg.get("role") == "user" and is_repl_output_message(msg)
        for msg in range_messages
    )
    return has_work and has_output


def _literal_observation_transition(message: dict) -> bool:
    return message.get("_observation_transition") is True


def _associated_repl_output(
    messages: list[dict],
    assistant_index: int,
) -> tuple[int | None, bool]:
    output_index = assistant_index + 1
    if output_index >= len(messages):
        return None, True
    output = messages[output_index]
    if (
        output.get("role") != "user"
        or not is_repl_output_message(output)
        or _real_user_message(output)
        or (output.get("_synthetic") and not output.get("_coalesced"))
    ):
        return None, True

    assistant_range = message_source_range(messages[assistant_index])
    output_range = message_source_range(output)
    if (
        assistant_range is None
        or output_range is None
        or assistant_range[0] != assistant_range[1]
        or output_range[0] != output_range[1]
        or type(assistant_range[0]) is not int
        or type(output_range[0]) is not int
        or assistant_range[0] >= output_range[0]
    ):
        return None, False

    if output.get("_repl_output_for") != assistant_range[0]:
        return None, False
    return output_index, True


def semantic_boundaries(messages: list[dict]) -> dict[int, SemanticBoundary]:
    boundaries = {}
    event_counts = {}
    for message in messages:
        seq = message.get("_event_seq")
        if type(seq) is int:
            event_counts[seq] = event_counts.get(seq, 0) + 1

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        is_release = is_release_assistant_message(message)
        transition_seq = message.get("_event_seq")
        is_transition = (
            not is_release
            and _literal_observation_transition(message)
            and type(transition_seq) is int
        )
        if not (is_release or is_transition):
            continue
        if is_transition:
            output_index, output_valid = _associated_repl_output(messages, index)
        else:
            output_index = None
            output_valid = True
            if index + 1 < len(messages):
                following = messages[index + 1]
                if (
                    following.get("role") == "user"
                    and is_repl_output_message(following)
                    and not _real_user_message(following)
                    and not (
                        following.get("_synthetic")
                        and not following.get("_coalesced")
                    )
                ):
                    output_index = index + 1
        boundary_seq = message.get("_event_seq")
        authoritative = (
            output_valid
            and type(boundary_seq) is int
            and event_counts.get(boundary_seq) == 1
            and (
                output_index is None
                or event_counts.get(messages[output_index].get("_event_seq")) == 1
            )
        )
        boundaries[index] = SemanticBoundary(
            boundary_index=index,
            boundary_output_index=output_index,
            transition_seq=transition_seq if is_transition else None,
            is_transition=is_transition,
            authoritative=authoritative,
        )
    return boundaries


def semantic_segments(messages: list[dict]) -> list[SemanticSegment]:
    segments = []
    identity = None
    identity_kind = None
    authoritative = False
    anchor = None
    segment_start = None
    skip_output = None

    boundaries = semantic_boundaries(messages)
    identity_counts = {}
    for message in messages:
        seq = message.get("_event_seq")
        if type(seq) is int and (
            _real_user_message(message)
            or _literal_observation_transition(message)
        ):
            identity_counts[seq] = identity_counts.get(seq, 0) + 1

    first_index = 1 if messages and messages[0].get("role") == "system" else 0
    for index, message in enumerate(messages[first_index:], start=first_index):
        if index == skip_output:
            continue
        if (
            message.get("_synthetic")
            and not message.get("_coalesced")
            and not message.get("_virtual_interaction_boundary")
        ):
            continue

        starts_interaction = _real_user_message(message)
        if starts_interaction:
            if identity is not None:
                identity = identity_kind = anchor = segment_start = None
                authoritative = False
                continue
            input_segments = [
                segment
                for segment in message.get("_render_segments") or []
                if isinstance(segment, dict) and segment.get("type") == "input"
            ]
            seq = message.get("_event_seq")
            authoritative = (
                len(input_segments) in (0, 1)
                and type(seq) is int
                and (
                    not input_segments
                    or input_segments[0].get("_event_seq") == seq
                )
            )
            identity = seq if authoritative else index
            identity_kind = "turn"
            anchor = index
            segment_start = index
            continue

        if identity is None or index not in boundaries:
            continue

        boundary = boundaries[index]
        is_transition = boundary.is_transition
        transition_seq = boundary.transition_seq
        boundary_output = boundary.boundary_output_index
        end_index = boundary_output if boundary_output is not None else index
        start_range = message_source_range(messages[segment_start])
        end_range = message_source_range(messages[end_index])
        source_valid = (
            authoritative
            and boundary.authoritative
            and identity_counts.get(identity) == 1
            and start_range is not None
            and end_range is not None
            and type(start_range[0]) is int
            and type(end_range[1]) is int
            and start_range[0] <= end_range[1]
        )
        segments.append(SemanticSegment(
            segment_id=identity,
            identity_kind=identity_kind,
            anchor_index=anchor,
            boundary_index=index,
            boundary_output_index=boundary_output,
            source_start_seq=start_range[0] if source_valid else index,
            source_end_seq=end_range[1] if source_valid else end_index,
            has_execution=(
                is_transition
                or _interaction_has_execution(messages, anchor, index)
            ),
            authoritative=source_valid,
        ))

        skip_output = boundary_output
        if is_transition:
            identity = transition_seq
            identity_kind = "checkpoint"
            authoritative = True
            anchor = end_index
            segment_start = end_index + 1
        else:
            identity = identity_kind = anchor = segment_start = None
            authoritative = False

    return segments


def _completed_interactions(messages: list[dict]) -> list[SemanticSegment]:
    return semantic_segments(messages)




def _merge_message_attachments(messages: list[dict]) -> tuple[dict, dict]:
    attachments = {}
    attachment_refs = {}
    for msg in messages:
        attachments.update(msg.get("_attachments") or {})
        attachment_refs.update(msg.get("_attachment_refs") or {})
    return attachments, attachment_refs


def _attachment_placeholders(messages: list[dict]) -> list[str]:
    seen = set()
    placeholders = []
    for msg in messages:
        for match in re.finditer(r"\[Attachment: ([^\]\n]+)\]", msg.get("content") or ""):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                placeholders.append(f"[Attachment: {name}]")
        for name in (msg.get("_attachments") or {}):
            if name not in seen:
                seen.add(name)
                placeholders.append(f"[Attachment: {name}]")
        for name in (msg.get("_attachment_refs") or {}):
            if name not in seen:
                seen.add(name)
                placeholders.append(f"[Attachment: {name}]")
    return placeholders


_PREVIEW_REF_BLOCK_RE = re.compile(
    r"\[PreviewRef: (?P<uri>session://preview/[^\]\n]+)\]\n"
    r".*?"
    r"\[/PreviewRef\]",
    re.DOTALL,
)


def _preserved_preview_ref_blocks(messages: list[dict], preserved_preview_refs) -> list[str]:
    if not preserved_preview_refs:
        return []
    rendered_by_uri = preserved_preview_refs if isinstance(preserved_preview_refs, dict) else {}
    uris = set(rendered_by_uri) or set(preserved_preview_refs)
    seen = set()
    blocks = []
    for msg in messages:
        for match in _PREVIEW_REF_BLOCK_RE.finditer(msg.get("content") or ""):
            uri = match.group("uri")
            if uri in uris and uri not in seen:
                seen.add(uri)
                blocks.append(match.group(0))
    for uri, rendered in rendered_by_uri.items():
        if uri not in seen and rendered:
            seen.add(uri)
            blocks.append(rendered)
    return blocks


def _strip_release_output_boilerplate(output: str, release_text: str) -> str:
    if not output:
        return ""
    lines = output.strip("\n").split("\n")
    kept = []
    release_text = (release_text or "").strip()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">>> emit(") and "release=True" in stripped.replace(" ", ""):
            continue
        if release_text and stripped == release_text:
            continue
        kept.append(line)
    return "\n".join(kept).strip("\n")


def _preview_content(range_messages: list[dict], release_msg: dict | None, release_output_msg: dict | None) -> str:
    parts = []
    prev_assistant = None
    for msg in range_messages:
        role = msg.get("role")
        if role == "assistant":
            _append_block(parts, msg.get("content") or "")
            prev_assistant = msg
        elif role == "user" and is_repl_output_message(msg):
            text = message_stdout(msg)
            if prev_assistant is not None:
                text = reconstruct_omitted_echo(prev_assistant.get("content") or "", text)
            _append_block(parts, text)
            prev_assistant = None
        elif role == "user":
            for text in human_inputs(msg):
                _append_block(parts, text)
            prev_assistant = None
        else:
            prev_assistant = None

    if release_output_msg is not None:
        text = message_stdout(release_output_msg)
        text = _strip_release_output_boilerplate(text, released_assistant_text(release_msg or {}))
        _append_block(parts, text)

    return "\n".join(parts).rstrip("\n")


def _coalesced_message_from_refs(
    rendered_refs: list[str],
    range_messages: list[dict],
    preserved_preview_refs=None,
    *,
    replacement_prefix: str | None = None,
    source_start_seq: int | None = None,
    source_end_seq: int | None = None,
) -> dict:
    placeholders = _attachment_placeholders(range_messages)
    preview_placeholders = _preserved_preview_ref_blocks(range_messages, preserved_preview_refs or set())
    visible_parts = [
        replacement_prefix
        if replacement_prefix is not None
        else "[Assistant work and REPL output coalesced into preview]"
    ]
    visible_parts.extend(placeholders)
    visible_parts.extend(preview_placeholders)
    visible_parts.extend(["", *rendered_refs])
    visible = "\n".join(visible_parts).rstrip("\n")
    msg = {
        "role": "user",
        "content": visible,
        "_render_segments": [{"type": "stdout", "content": visible}],
        "_synthetic": True,
        "_coalesced": True,
    }
    if source_start_seq is not None and source_end_seq is not None:
        msg["_source_start_seq"] = source_start_seq
        msg["_source_end_seq"] = source_end_seq
    attachments, attachment_refs = _merge_message_attachments(range_messages)
    if attachments:
        msg["_attachments"] = attachments
    if attachment_refs:
        msg["_attachment_refs"] = attachment_refs
    return msg



def render_projected_span(messages: list[dict]) -> str:
    return _preview_content(messages, None, None)


def materialize_preview(preview: Preview, derived_content: str) -> tuple[str, str, str]:
    content = derived_content if preview.content is None else preview.content
    key = preview_key(content)
    return key, content, render_preview_ref(key, preview.summary)


def make_preview_replacement(
    rendered_refs: list[str],
    selected_messages: list[dict],
    *,
    source_start_seq: int | None = None,
    source_end_seq: int | None = None,
    replacement_prefix: str | None = None,
    preserve_preview_refs=None,
) -> dict:
    return _coalesced_message_from_refs(
        rendered_refs,
        selected_messages,
        preserve_preview_refs,
        replacement_prefix=replacement_prefix,
        source_start_seq=source_start_seq,
        source_end_seq=source_end_seq,
    )


def replace_projected_span(
    messages: list[dict],
    span: ProjectedSpan,
    replacement: dict,
) -> list[dict]:
    return [
        *copy.deepcopy(messages[:span.start_index]),
        copy.deepcopy(replacement),
        *copy.deepcopy(messages[span.end_index:]),
    ]


def _resolved_preview_placement(
    messages: list[dict],
    *,
    source_start_seq: int,
    source_end_seq: int,
    preview_key: str,
    summary: str,
    content: str | None = None,
    replacement_prefix: str | None = None,
    preserve_preview_refs=None,
    persisted: bool = False,
    preview_event_seq: int | None = None,
) -> tuple[list[dict], str | None]:
    span = select_projected_span(
        messages,
        source_start_seq=source_start_seq,
        source_end_seq=source_end_seq,
    )
    selected = messages[span.start_index:span.end_index]
    replacement = make_preview_replacement(
        [render_preview_ref(preview_key, summary)],
        selected,
        source_start_seq=source_start_seq,
        source_end_seq=source_end_seq,
        replacement_prefix=replacement_prefix,
        preserve_preview_refs=preserve_preview_refs,
    )
    if persisted:
        replacement["_persisted_preview"] = True
        if preview_event_seq is not None:
            replacement["_preview_event_seq"] = preview_event_seq
    return replace_projected_span(messages, span, replacement), content


def _resolve_preview(
    messages: list[dict],
    preview: Preview,
    *,
    source_start_seq: int,
    source_end_seq: int,
    persisted: bool = False,
) -> tuple[list[dict], str, str]:
    span = select_projected_span(
        messages,
        source_start_seq=source_start_seq,
        source_end_seq=source_end_seq,
    )
    selected = messages[span.start_index:span.end_index]
    key, content, _ = materialize_preview(preview, render_projected_span(selected))
    projected, _ = _resolved_preview_placement(
        messages,
        source_start_seq=source_start_seq,
        source_end_seq=source_end_seq,
        preview_key=key,
        summary=preview.summary,
        content=content,
        persisted=persisted,
    )
    return projected, key, content


def apply_preview_placement(
    messages: list[dict],
    *,
    preview_event_seq: int,
    preview_key: str,
    summary: str,
    source_start_seq: int,
    source_end_seq: int,
) -> list[dict]:
    if type(preview_event_seq) is not int:
        raise PreviewPlacementError("preview event sequence must be an integer")
    if not isinstance(preview_key, str) or not preview_key:
        raise PreviewPlacementError("preview key must be a non-empty string")
    preview = Preview(summary=summary)
    projected, _ = _resolved_preview_placement(
        messages,
        source_start_seq=source_start_seq,
        source_end_seq=source_end_seq,
        preview_key=preview_key,
        summary=preview.summary,
        persisted=True,
        preview_event_seq=preview_event_seq,
    )
    return projected


def create_persisted_preview(
    messages: list[dict],
    preview: Preview,
    *,
    source_start_seq: int,
    source_end_seq: int,
    store,
    session_id: str,
    expected_next_seq: int | None = None,
    state: PersistedPreviewState,
) -> tuple[list[dict], str, int, int]:
    if not isinstance(state, PersistedPreviewState):
        raise TypeError("complete persisted preview state is required")
    if type(source_start_seq) is not int or type(source_end_seq) is not int:
        raise PreviewPlacementError("preview source boundaries must be integers")
    if source_start_seq <= state.exec_start_seq:
        raise PreviewPlacementError("preview placement crosses the active exec boundary")
    status = state.placement_status(-1, source_start_seq, source_end_seq)
    if status != "apply":
        raise PreviewPlacementError(f"persisted preview range is not available: {status}")
    projected, key, content = _resolve_preview(
        messages,
        preview,
        source_start_seq=source_start_seq,
        source_end_seq=source_end_seq,
        persisted=True,
    )
    created_seq, placed_seq = store.append_preview_events(
        session_id,
        expected_next_seq=expected_next_seq,
        preview_key=key,
        summary=preview.summary,
        source_start_seq=source_start_seq,
        source_end_seq=source_end_seq,
        expected_exec_start_seq=state.exec_start_seq,
        expected_definitions=state.definitions,
        expected_active_placements=state.active_placements,
        preview_content=content,
    )
    state.definitions[created_seq] = (key, preview.summary)
    state.install_placement(created_seq, source_start_seq, source_end_seq)
    span = select_projected_span(
        projected,
        source_start_seq=source_start_seq,
        source_end_seq=source_end_seq,
    )
    replacement = projected[span.start_index:span.end_index]
    if len(replacement) != 1 or not replacement[0].get("_persisted_preview"):
        raise PreviewPlacementError("persisted preview projection metadata is missing")
    replacement[0]["_preview_event_seq"] = created_seq
    return projected, key, created_seq, placed_seq


def place_preview(
    messages: list[dict],
    previews: Preview | list[Preview],
    *,
    source_start_seq: int | None = None,
    source_end_seq: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    save_preview_blob=None,
    replacement_prefix: str | None = None,
    preserve_preview_refs=None,
) -> tuple[list[dict], list[str]]:
    single_preview = isinstance(previews, Preview)
    previews = [previews] if single_preview else list(previews)
    if not previews or any(not isinstance(preview, Preview) for preview in previews):
        raise TypeError("previews must be a Preview or non-empty sequence of Preview values")

    has_source_boundary = source_start_seq is not None or source_end_seq is not None
    has_index_boundary = start_index is not None or end_index is not None
    if has_source_boundary:
        if source_start_seq is None or source_end_seq is None or has_index_boundary:
            raise PreviewPlacementError(
                "provide either a complete canonical source range or a complete legacy index span"
            )
        span = select_projected_span(
            messages,
            source_start_seq=source_start_seq,
            source_end_seq=source_end_seq,
        )
    else:
        if start_index is None or end_index is None:
            raise PreviewPlacementError(
                "provide either a complete canonical source range or a complete legacy index span"
            )
        if not 0 <= start_index < end_index <= len(messages):
            raise PreviewPlacementError(
                f"invalid legacy projected span [{start_index}, {end_index})"
            )
        selected_ranges = [
            message_source_range(message)
            for message in messages[start_index:end_index]
            if message_source_range(message) is not None
        ]
        if selected_ranges:
            raise PreviewPlacementError(
                "legacy index placement is only supported for source-less projected nodes"
            )
        span = ProjectedSpan(start_index, end_index, 0, 0)

    selected = messages[span.start_index:span.end_index]
    if single_preview and has_source_boundary:
        projected, key, content = _resolve_preview(
            messages,
            previews[0],
            source_start_seq=source_start_seq,
            source_end_seq=source_end_seq,
        )
        if replacement_prefix is not None or preserve_preview_refs:
            projected, _ = _resolved_preview_placement(
                messages,
                source_start_seq=source_start_seq,
                source_end_seq=source_end_seq,
                preview_key=key,
                summary=previews[0].summary,
                content=content,
                replacement_prefix=replacement_prefix,
                preserve_preview_refs=preserve_preview_refs,
            )
        if save_preview_blob is not None:
            save_preview_blob(key, content)
        return projected, key

    derived_content = render_projected_span(selected)
    materialized = [
        materialize_preview(preview, derived_content)
        for preview in previews
    ]
    replacement = make_preview_replacement(
        [rendered for _, _, rendered in materialized],
        selected,
        source_start_seq=source_start_seq,
        source_end_seq=source_end_seq,
        replacement_prefix=replacement_prefix,
        preserve_preview_refs=preserve_preview_refs,
    )
    projected = replace_projected_span(messages, span, replacement)
    if save_preview_blob is not None:
        for key, content, _ in materialized:
            save_preview_blob(key, content)
    keys = [key for key, _, _ in materialized]
    return projected, keys[0] if single_preview else keys


def _message_observations(messages: list[dict]) -> list[str]:
    observations = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        values = msg.get("_observations")
        if not isinstance(values, list):
            continue
        observations.extend(
            value
            for value in values
            if isinstance(value, str) and value.strip()
        )
    return observations


def _coalesced_ordered_sections(
    range_messages: list[dict],
    release_msg: dict | None,
    release_output_msg: dict | None,
    boundary_observations: list[str] | None = None,
) -> list[tuple[bool, str, list[str]]]:
    sections = []
    normal = []

    def flush_normal(include_release_output: bool = False):
        nonlocal normal
        content = _preview_content(
            normal,
            release_msg if include_release_output else None,
            release_output_msg if include_release_output else None,
        )
        if content:
            sections.append((False, content, _message_observations(normal)))
        normal = []

    i = 0
    while i < len(range_messages):
        msg = range_messages[i]
        if msg.get("role") == "assistant" and msg.get("_pinned_coalesce"):
            flush_normal()
            turn = [msg]
            if i + 1 < len(range_messages):
                nxt = range_messages[i + 1]
                if nxt.get("role") == "user" and is_repl_output_message(nxt):
                    turn.append(nxt)
                    i += 1
            content = _preview_content(turn, None, None)
            if content:
                sections.append((True, content, _message_observations(turn)))
        else:
            normal.append(msg)
        i += 1

    flush_normal()
    if boundary_observations and sections:
        pinned, content, observations = sections[-1]
        sections[-1] = (
            pinned,
            content,
            [*observations, *boundary_observations],
        )
    return sections




def deterministic_interaction_replacement(
    messages: list[dict],
    *,
    source_start_seq: int | None = None,
    source_end_seq: int | None = None,
    interaction_index: int | None = None,
    preserve_preview_refs=None,
) -> tuple[dict, list[str], dict[str, str]] | None:
    normalized = normalize_repl_messages(messages)
    interactions = _completed_interactions(normalized)
    if interaction_index is not None:
        if not 0 <= interaction_index < len(interactions):
            return None
        candidates = [interactions[interaction_index]]
    else:
        if source_start_seq is None or source_end_seq is None:
            raise PreviewPlacementError(
                "provide an interaction index or complete canonical body range"
            )
        candidates = interactions
    for item in candidates:
        start = item.anchor_index
        release = item.boundary_index
        range_messages = normalized[start + 1:release]
        if not range_messages or any(message.get("_coalesced") for message in range_messages):
            continue
        source_ranges = [message_source_range(message) for message in range_messages]
        if all(source_range is not None for source_range in source_ranges):
            if any(
                current[0] <= previous[1]
                for previous, current in zip(source_ranges, source_ranges[1:])
            ):
                continue
            body_start = source_ranges[0][0]
            body_end = source_ranges[-1][1]
            if (
                source_start_seq is not None
                and (body_start != source_start_seq or body_end != source_end_seq)
            ):
                continue
            placement_kwargs = {
                "source_start_seq": body_start,
                "source_end_seq": body_end,
            }
        elif all(source_range is None for source_range in source_ranges):
            if source_start_seq is not None:
                continue
            placement_kwargs = {
                "start_index": start + 1,
                "end_index": release,
            }
        else:
            continue

        release_output = item.boundary_output_index
        release_msg = normalized[release]
        release_output_msg = (
            normalized[release_output] if release_output is not None else None
        )
        section_contents = _coalesced_ordered_sections(
            range_messages,
            release_msg,
            release_output_msg,
            (
                _message_observations([release_msg])
                if _literal_observation_transition(release_msg)
                else None
            ),
        )
        if not section_contents:
            return None
        previews = [
            Preview(
                summary=render_default_preview_summary(section_content, observations),
                content=section_content,
            )
            for _, section_content, observations in section_contents
        ]
        materialized_contents = {}
        try:
            candidate, keys = place_preview(
                normalized,
                previews,
                save_preview_blob=materialized_contents.setdefault,
                preserve_preview_refs=preserve_preview_refs,
                **placement_kwargs,
            )
        except PreviewPlacementError:
            return None
        replacement = candidate[start + 1]
        _sign_deterministic_replacement(replacement)
        return replacement, keys, materialized_contents
    return None


def coalesce_repl_messages(
    messages: list[dict],
    *,
    keep_last_interactions: int = 3,
    keep_last_execution_interactions: int = 1,
    min_savings_chars: int = 1000,
    save_preview_blob=None,
    auto_expand_preview_refs: list[str] | None = None,
    preserve_preview_refs=None,
    protect_last_interactions: bool = True,
) -> list[dict]:


    if not messages:
        return []
    if keep_last_interactions < 0:
        raise ValueError("keep_last_interactions must be >= 0")
    if keep_last_execution_interactions < 0:
        raise ValueError("keep_last_execution_interactions must be >= 0")

    messages = normalize_repl_messages(messages)
    interactions = _completed_interactions(messages)


    if len(interactions) <= keep_last_interactions:
        return messages


    protected = set()
    if protect_last_interactions:
        protected.update(range(max(0, len(interactions) - keep_last_interactions), len(interactions)))
        if keep_last_execution_interactions:
            execution_indexes = [
                i
                for i, item in enumerate(interactions)
                if item.has_execution
            ]
            protected.update(execution_indexes[-keep_last_execution_interactions:])


    by_start = {item.anchor_index: (idx, item) for idx, item in enumerate(interactions)}
    skip_indexes = set()
    replacements = {}

    for idx, item in enumerate(interactions):
        if idx in protected:
            continue
        start = item.anchor_index
        release = item.boundary_index
        range_messages = messages[start + 1:item.boundary_index]
        if not range_messages:
            continue
        derived = deterministic_interaction_replacement(
            messages,
            interaction_index=idx,
            preserve_preview_refs=preserve_preview_refs,
        )
        if derived is None:
            continue
        projected_replacement, keys, materialized_contents = derived
        boundary_message = messages[item.boundary_index]
        section_contents = _coalesced_ordered_sections(
            range_messages,
            boundary_message,
            (
                messages[item.boundary_output_index]
                if item.boundary_output_index is not None
                else None
            ),
            (
                _message_observations([boundary_message])
                if _literal_observation_transition(boundary_message)
                else None
            ),
        )
        original_chars = sum(len(m.get("content") or "") for m in range_messages)
        replacement_chars = len(projected_replacement.get("content") or "")
        if original_chars - replacement_chars < min_savings_chars:
            continue

        if save_preview_blob is not None:
            for key in keys:
                save_preview_blob(key, materialized_contents[key])
        for (pinned, _, _), key in zip(section_contents, keys):
            if pinned and auto_expand_preview_refs is not None:
                auto_expand_preview_refs.append(f"session://preview/{key}")

        replacements[start] = projected_replacement
        skip_indexes.update(range(start + 1, release))

    projected = [copy.deepcopy(messages[0])]
    for i, msg in enumerate(messages[1:], start=1):
        if i in skip_indexes:
            continue
        projected.append(copy.deepcopy(msg))
        if i in by_start and i in replacements:
            projected.append(replacements[i])

    return projected
