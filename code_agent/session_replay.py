import ast
import base64
import contextlib
import copy
import re
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from .persisted_preview_state import PersistedPreviewState, PersistedPreviewTransitions
from .repl_attachment_mixin import (
    ImageAttachment,
    MemoryAttachment,
    TextAttachment,
    decode_attachment_refs,
    normalize_message_attachments,
)
from .session_message_state import (
    apply_canonical_message_transition,
    coalesce_adjacent_user_messages,
)


@contextlib.contextmanager
def _silence_parse_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def _parse_silently(content: str) -> ast.AST:
    with _silence_parse_warnings():
        return ast.parse(content)


def _preview_key(path: str) -> str | None:
    prefix = "session://preview/"
    if isinstance(path, str) and path.startswith(prefix):
        key = path[len(prefix):]
        return key or None
    return None


def render_attachment_content(ref, store=None, session_id: str | None = None, base_dir: str | None = None):
    if isinstance(ref, (TextAttachment, ImageAttachment)):
        return ref
    if isinstance(ref, MemoryAttachment):
        content = ref.content
    elif key := _preview_key(ref):
        if store is None or session_id is None:
            raise FileNotFoundError(ref)
        content = store.get_preview_blob(session_id, key)
        if content is None:
            raise FileNotFoundError(ref)
    else:
        raise FileNotFoundError(ref)
    lines = content.split('\n')
    return TextAttachment(
        '\n'.join(f"{i+1:>5}→{line}" for i, line in enumerate(lines))
    )


def _load_attachment_map(refs: dict, missing: list[tuple[str, object]], store=None, session_id: str | None = None, base_dir: str | None = None) -> dict:
    loaded = {}
    for name, ref in refs.items():
        if not isinstance(
            ref, (MemoryAttachment, TextAttachment, ImageAttachment)
        ) and not _preview_key(ref):
            continue
        try:
            loaded[name] = render_attachment_content(ref, store, session_id, base_dir)
        except Exception:
            missing.append((name, ref))
    return loaded


def _coalesce_user_messages(messages: list[dict]) -> list[dict]:
    return coalesce_adjacent_user_messages(messages)


def _decode_media(value):
    if isinstance(value, dict) and "__b64__" in value:
        return base64.b64decode(value["__b64__"])
    if isinstance(value, list):
        return [_decode_media(item) for item in value]
    return value


def _replay_session_into_target(agent, session_id: str, store):
    events = store.get_events(session_id)
    session = store.get_session(session_id) if hasattr(store, "get_session") else None
    base_dir = (session or {}).get("cwd")

    snapshots = {}
    persisted_transitions = PersistedPreviewTransitions()
    persisted_preview_state = persisted_transitions.state
    snapshot_seqs = {
        event["payload"]["target_seq"]
        for event in events
        if event.get("event_type") == "rewind"
        and isinstance(event.get("payload"), dict)
        and type(event["payload"].get("target_seq")) is int
    }
    snapshot_seqs.add(0)
    messages = [copy.deepcopy(agent.conversation.messages[0])]
    agent._expanded_preview_refs = {}
    if hasattr(agent, "_configure_conversation"):
        agent._configure_conversation(agent.conversation)

    missing_seen = set()

    def snapshot(seq):
        if seq in snapshot_seqs:
            snapshots[seq] = (
                copy.deepcopy(messages),
                copy.deepcopy(persisted_preview_state),
                copy.deepcopy(agent._expanded_preview_refs),
            )

    snapshot(0)
    for event in events:
        seq = event.get("seq")
        if type(seq) is not int:
            continue
        payload = event.get("payload")
        event_type = event.get("event_type")
        if event_type == "message_added" and isinstance(payload, dict):
            raw_message = payload.get("message")
            if not isinstance(raw_message, dict):
                snapshot(seq)
                continue
            msg = copy.deepcopy(raw_message)
            if msg.get("audio"):
                msg["audio"] = _decode_media(msg["audio"])
            refs = decode_attachment_refs(msg.pop("_attachment_refs", None) or {})
            local_missing = []
            if refs:
                loaded = _load_attachment_map(refs, local_missing, store, session_id, base_dir)
                if loaded:
                    msg["_attachments"] = loaded
                msg["_attachment_refs"] = refs
                for item in local_missing:
                    missing_seen.add(item)
            apply_canonical_message_transition(
                messages,
                event_type=event_type,
                payload={"message": msg},
                event_seq=seq,
            )
        elif event_type in {"attachment_invalidated", "message_pinned"}:
            apply_canonical_message_transition(
                messages,
                event_type=event_type,
                payload=payload,
                event_seq=seq,
            )
        elif event_type in {"preview_created", "preview_placed"} and isinstance(payload, dict):
            def apply_persisted_placement(
                _preview_event_seq,
                definition,
                source_start_seq,
                source_end_seq,
            ):
                nonlocal messages
                from code_agent.code_agent_coalesce import (
                    PreviewPlacementError,
                    apply_preview_placement,
                )
                try:
                    messages = apply_preview_placement(
                        messages,
                        preview_event_seq=_preview_event_seq,
                        preview_key=definition[0],
                        summary=definition[1],
                        source_start_seq=source_start_seq,
                        source_end_seq=source_end_seq,
                    )
                except PreviewPlacementError:
                    return False
                return True

            persisted_transitions.apply(
                seq=seq,
                event_type=event_type,
                payload=payload,
                has_preview_blob=lambda key: store.has_preview_blob(session_id, key),
                apply_placement=apply_persisted_placement,
            )
            persisted_preview_state = persisted_transitions.state
        elif event_type == "preview_expanded" and isinstance(payload, dict):
            uri = payload.get("uri")
            if uri:
                agent._expanded_preview_refs[uri] = {"numbered": bool(payload.get("numbered", False))}
        elif event_type == "preview_collapsed" and isinstance(payload, dict):
            uri = payload.get("uri")
            if uri:
                agent._expanded_preview_refs.pop(uri, None)
        elif event_type == "rewind" and isinstance(payload, dict):
            target_seq = payload.get("target_seq")
            if type(target_seq) is int:
                restored = copy.deepcopy(snapshots.get(target_seq, snapshots[0]))
                messages, _, expanded_preview_refs = restored
                persisted_transitions.apply(
                    seq=seq,
                    event_type=event_type,
                    payload=payload,
                    has_preview_blob=lambda key: store.has_preview_blob(session_id, key),
                )
                persisted_preview_state = persisted_transitions.state
                agent._expanded_preview_refs = expanded_preview_refs
                if hasattr(agent, "_configure_conversation"):
                    agent._configure_conversation(agent.conversation)
        elif event_type == "exec" and isinstance(payload, dict):
            messages, _, _ = copy.deepcopy(snapshots[0])
            persisted_transitions.apply(
                seq=seq,
                event_type=event_type,
                payload=payload,
                has_preview_blob=lambda key: store.has_preview_blob(session_id, key),
            )
            persisted_preview_state = persisted_transitions.state
            agent._expanded_preview_refs.clear()
        if isinstance(payload, dict) and event_type not in {
            "preview_created",
            "preview_placed",
            "rewind",
            "exec",
        }:
            persisted_transitions.apply(
                seq=seq,
                event_type=event_type,
                payload=payload,
                has_preview_blob=lambda key: store.has_preview_blob(session_id, key),
            )
            persisted_preview_state = persisted_transitions.state
        snapshot(seq)

    final_missing = []
    for msg in messages:
        refs = decode_attachment_refs(msg.get("_attachment_refs") or {})
        attachments = msg.get("_attachments", {})
        if refs:
            msg["_attachment_refs"] = refs
        else:
            msg.pop("_attachment_refs", None)
        for name, ref in refs.items():
            if isinstance(
                ref, (MemoryAttachment, TextAttachment, ImageAttachment)
            ):
                continue
            if _preview_key(ref):
                if store.get_preview_blob(session_id, _preview_key(ref)) is None:
                    attachments.pop(name, None)
                    final_missing.append((name, ref))
        if "_attachments" in msg and not msg["_attachments"]:
            del msg["_attachments"]
    agent.conversation.messages = [
        normalize_message_attachments(message) for message in messages
    ]
    agent._persisted_preview_state = persisted_preview_state
    if hasattr(agent, "_reconstruct_observation_counters"):
        agent._reconstruct_observation_counters(messages)
    deduped = []
    seen = set()
    for item in final_missing:
        key = (item[0], repr(item[1]))
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def replay_session_into_agent(agent, session_id: str, store):
    return _replay_session_into_target(agent, session_id, store)


def _extract_released_assistant_text(msg: dict) -> str:
    value = msg.get("_final_result", msg.get("_emit_value"))
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
                break

    stripped = content.lstrip()
    if not stripped.startswith("emit("):
        return ""

    match = re.search(
        r'emit\(\s*(?P<q>["\']{1,3})(?P<text>.*?)(?P=q)\s*,\s*release\s*=\s*True\s*\)',
        content,
        re.DOTALL,
    )
    if not match:
        return ""
    return bytes(match.group("text"), "utf-8").decode("unicode_escape")


def _is_released_assistant_message(msg: dict) -> bool:
    if msg.get("_final_result", msg.get("_emit_value")) is not None:
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


def _section_header(label: str, char: str = "═", width: int = 34) -> str:
    prefix = f"{char} {label} "
    return prefix + (char * max(0, width - len(prefix)))


def replay_display_text(session_id: str, store, format_response=None) -> str:
    events = store.get_events(session_id)
    snapshots = {}
    snapshot_seqs = {
        event["payload"]["target_seq"]
        for event in events
        if event["event_type"] == "rewind"
    }
    snapshot_seqs.add(0)
    chunks: list[str] = []
    released_to_user = False
    just_rewound = False

    def snapshot(seq):
        if seq in snapshot_seqs:
            snapshots[seq] = (copy.deepcopy(chunks), released_to_user, just_rewound)

    snapshot(0)
    for event in events:
        seq = event["seq"]
        payload = event["payload"]
        event_type = event["event_type"]
        if event_type == "message_added":
            msg = payload.get("message", {})
            if msg.get("_virtual_interaction_boundary"):
                snapshot(seq)
                continue
            if msg.get("role") == "assistant" and _is_released_assistant_message(msg):
                released_to_user = True
                text = _extract_released_assistant_text(msg)
                if text:
                    chunks.append(_section_header("Output") + "\n")
                    if format_response is not None:
                        text = format_response(text)
                    if not text.endswith("\n"):
                        text += "\n"
                    chunks.append(text)
        elif event_type == "display":
            kind = payload.get("kind")
            if released_to_user and kind != "input" and not (just_rewound and kind == "status"):
                snapshot(seq)
                continue
            text = payload.get("text", "")
            if text:
                chunks.append(text)
            if kind == "input":
                released_to_user = False
            just_rewound = False
        elif event_type == "rewind":
            target_seq = payload["target_seq"]
            chunks, released_to_user, _ = copy.deepcopy(snapshots.get(target_seq, snapshots[0]))
            just_rewound = True
        elif event_type == "exec":
            chunks, released_to_user, _ = copy.deepcopy(snapshots[0])
            just_rewound = True
        snapshot(seq)

    return "".join(chunks)
