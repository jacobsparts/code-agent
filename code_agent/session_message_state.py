import copy


def coalesce_adjacent_user_messages(messages: list[dict]) -> list[dict]:
    out = []
    for message in messages:
        if (
            message.get("role") == "user"
            and out
            and out[-1].get("role") == "user"
        ):
            previous = out[-1]
            previous.setdefault("_render_segments", []).extend(
                copy.deepcopy(message.get("_render_segments") or [])
            )
            previous["content"].extend(copy.deepcopy(message["content"]))
            if message.get("_stdout"):
                previous_stdout = previous.get("_stdout", "")
                stdout_separator = (
                    ""
                    if not previous_stdout or previous_stdout.endswith("\n")
                    else "\n"
                )
                stdout = previous_stdout + stdout_separator + message["_stdout"]
                if not stdout.endswith("\n"):
                    stdout += "\n"
                previous["_stdout"] = stdout
            if message.get("_user_content") is not None:
                previous["_user_content"] = message["_user_content"]
            for key in ("_attachment_refs", "_attachments"):
                if message.get(key):
                    values = previous.get(key) or {}
                    values.update(copy.deepcopy(message[key]))
                    previous[key] = values
        else:
            out.append(copy.deepcopy(message))
    return out


def is_canonical_repl_output(message: dict) -> bool:
    if message.get("role") != "user" or message.get("_synthetic"):
        return False
    segments = message.get("_render_segments")
    if not isinstance(segments, list):
        return False
    segment_types = [
        segment.get("type")
        for segment in segments
        if isinstance(segment, dict)
    ]
    return bool(segment_types) and all(
        segment_type == "stdout"
        for segment_type in segment_types
    )


def add_canonical_message(
    messages: list[dict],
    message: dict,
    event_seq: int,
) -> None:
    canonical = copy.deepcopy(message)
    content = canonical.get("content")
    if not isinstance(content, list):
        raise TypeError("canonical message content must be a list of typed blocks")
    if any(
        not isinstance(block, dict) or not isinstance(block.get("type"), str)
        for block in content
    ):
        raise TypeError("canonical message content must contain typed blocks")
    canonical["_event_seq"] = event_seq
    for segment in reversed(canonical.get("_render_segments") or []):
        if "_event_seq" not in segment:
            segment["_event_seq"] = event_seq
            break
    messages.append(canonical)


def pin_canonical_message(
    messages: list[dict],
    message_event_seq: int,
    label: str | None,
) -> None:
    for message in reversed(messages):
        if message.get("_event_seq") == message_event_seq:
            message["_pinned_coalesce"] = {
                "label": label or "Pinned previous turn"
            }
            break


def invalidate_canonical_attachment(
    messages: list[dict],
    name: str,
) -> None:
    for message in messages:
        for key in ("_attachments", "_attachment_refs"):
            values = message.get(key)
            if isinstance(values, dict) and name in values:
                del values[name]
                if not values:
                    del message[key]


def apply_canonical_message_transition(
    messages: list[dict],
    *,
    event_type: str,
    payload,
    event_seq: int,
) -> bool:
    if not isinstance(payload, dict):
        return event_type in {
            "message_added",
            "message_pinned",
            "attachment_invalidated",
        }
    if event_type == "message_added":
        message = payload.get("message")
        if isinstance(message, dict):
            add_canonical_message(messages, message, event_seq)
        return True
    if event_type == "message_pinned":
        target = payload.get("message_event_seq")
        if type(target) is int:
            pin_canonical_message(messages, target, payload.get("label"))
        return True
    if event_type == "attachment_invalidated":
        name = payload.get("name")
        if isinstance(name, str) and name:
            invalidate_canonical_attachment(messages, name)
        return True
    return False


def reduce_canonical_message_events(
    events: list[dict],
) -> tuple[list[dict], int]:
    messages = []
    exec_start_seq = 0
    pending_transition_seq = None
    rewind_targets = {
        payload.get("target_seq")
        for event in events
        if event.get("event_type") == "rewind"
        and isinstance((payload := event.get("payload")), dict)
        and type(payload.get("target_seq")) is int
    }
    snapshots = {0: ([], 0, None)}
    for event in events:
        seq = event.get("seq")
        event_type = event.get("event_type")
        payload = event.get("payload")
        if type(seq) is not int:
            continue
        applied_payload = payload
        if event_type == "message_added" and isinstance(payload, dict):
            message = payload.get("message")
            if isinstance(message, dict):
                applied_payload = copy.deepcopy(payload)
                canonical_message = applied_payload["message"]
                if (
                    pending_transition_seq is not None
                    and is_canonical_repl_output(canonical_message)
                ):
                    canonical_message["_repl_output_for"] = pending_transition_seq
                if (
                    canonical_message.get("role") == "assistant"
                    and canonical_message.get("_observation_transition") is True
                ):
                    pending_transition_seq = seq
                else:
                    pending_transition_seq = None
        if apply_canonical_message_transition(
            messages,
            event_type=event_type,
            payload=applied_payload,
            event_seq=seq,
        ):
            pass
        elif event_type == "rewind":
            target = payload.get("target_seq") if isinstance(payload, dict) else None
            if type(target) is int:
                messages, exec_start_seq, pending_transition_seq = copy.deepcopy(
                    snapshots.get(target, snapshots[0])
                )
        elif event_type == "exec" and isinstance(payload, dict):
            messages = []
            exec_start_seq = seq
            pending_transition_seq = None
        if seq in rewind_targets:
            snapshots[seq] = (
                copy.deepcopy(messages),
                exec_start_seq,
                pending_transition_seq,
            )
    return coalesce_adjacent_user_messages(messages), exec_start_seq
