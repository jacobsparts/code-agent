import json

from .preview_refs import render_preview_refs
from .repl_attachment_mixin import (
    AudioAttachment,
    ImageAttachment,
    TextAttachment,
    iter_placeholders,
    normalize_attachments,
)


def content_blocks(content):
    """Normalize a value to canonical content blocks.

    A non-empty list is already canonical only when every item is a mapping
    with a string ``type``. Ordinary list payloads are JSON text, just like
    other structured values.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if (
        isinstance(content, list)
        and content
        and all(
            isinstance(block, dict) and isinstance(block.get("type"), str)
            for block in content
        )
    ):
        return content
    return [{"type": "text", "text": json.dumps(content)}]


def _blocks_text(blocks):
    return "".join(
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )


MEDIA_ATTACHMENTS_FIELD = "_media_attachments"


def materialize_attachments(content: str, attachments: dict) -> tuple[str, list[dict]]:
    """Expand text attachments and collect projected media, in placeholder order.

    Text attachments replace their placeholder with rendered text. Image
    attachments keep their placeholder and produce a provider-neutral media
    entry. An attachment value without a matching placeholder is not
    materialized.
    """
    content = content or ""
    attachments = attachments or {}
    parts = []
    media = []
    seen = set()
    end = 0
    for match, parsed in iter_placeholders(content):
        parts.append(content[end:match.start()])
        name = parsed["name"]
        value = attachments.get(name)
        if isinstance(value, TextAttachment):
            parts.append(value.content)
        else:
            parts.append(match.group(0))
            if isinstance(value, (ImageAttachment, AudioAttachment)) and name not in seen:
                seen.add(name)
                item = {
                    "name": name,
                    "media_type": value.media_type,
                    "content": value.content,
                }
                if isinstance(value, ImageAttachment):
                    item.update(width=value.width, height=value.height)
                media.append(item)
        end = match.end()
    parts.append(content[end:])
    return "".join(parts), media


def _canonical_message(message):
    content = message.get("content")
    if not isinstance(content, list):
        raise TypeError("canonical message content must be a list of typed blocks")
    if any(
        not isinstance(block, dict) or not isinstance(block.get("type"), str)
        for block in content
    ):
        raise TypeError("canonical message content must contain typed blocks")
    return message


class Convo:
    def __init__(self, llm_client, system_prompt):
        self.llm_client = llm_client
        content = content_blocks(system_prompt)
        self._messages = [{"role": "system", "content": content}]
        self.ephemeral = ""
        self._prompt_cache = []
        self._prompt_cache_model = getattr(llm_client, "model_name", None)

    def _with_cache_breakpoints(self, messages):
        if self.llm_client is None:
            return messages
        model_config = getattr(self.llm_client, "model_config", None) or {}
        if not model_config.get("explicit_prompt_cache"):
            return messages
        model_name = getattr(self.llm_client, "model_name", None)
        if model_name != self._prompt_cache_model:
            self._prompt_cache = []
            self._prompt_cache_model = model_name
        cache, self._prompt_cache = self._prompt_cache, []
        annotated = []
        for message in messages:
            out = dict(message)
            content = out.get("content")
            if (
                cache is not False
                and isinstance(content, list)
                and out.get("role") in ("system", "user")
            ):
                content_hash = hash(repr(content))
                if cache:
                    if (expected := cache.pop(0)) is None:
                        if not cache:
                            cache = False
                    elif content_hash != expected:
                        cache = [None] * 3
                    elif not cache:
                        cache = [None] * 4
                out["_prompt_cache_breakpoint"] = True
                self._prompt_cache.append(content_hash)
            annotated.append(out)
        return annotated

    def stored_messages(self):
        return self._messages

    def replace_messages(self, messages):
        messages = list(messages)
        for message in messages:
            _canonical_message(message)
        self._messages = messages

    def append_message(self, message):
        self._messages.append(_canonical_message(message))
        return message

    def extend_messages(self, messages):
        messages = list(messages)
        for message in messages:
            _canonical_message(message)
        self._messages.extend(messages)
        return messages

    def insert_message(self, index, message):
        self._messages.insert(index, _canonical_message(message))
        return message

    def pop_message(self, index=-1):
        return self._messages.pop(index)

    def update_message(self, message, **changes):
        updated = {**message, **changes}
        _canonical_message(updated)
        message.update(changes)
        return message

    def remove_message_fields(self, message, *fields):
        for field in fields:
            message.pop(field, None)
        return message

    def projected_messages(self):
        messages_projector = getattr(self, "messages_projector", None)
        messages = (
            messages_projector(self._messages)
            if messages_projector is not None
            else self._messages
        )
        message_projector = getattr(self, "message_projector", None)
        result = [
            message_projector(message)
            if message_projector is not None
            else dict(message)
            for message in messages
        ]

        # Provider boundary: `_attachments` are materialized per text block
        # exactly once. Stored messages are never mutated.
        for index, message in enumerate(result):
            content = message.get("content", [])
            if message.get("role") == "assistant":
                content = [block for block in content if block["type"] != "attachment"]
            attachments = message.pop("_attachments", None)
            if attachments:
                normalized_attachments = normalize_attachments(attachments)
                blocks = []
                media_entries = []
                seen_media = set()
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text, media = materialize_attachments(
                            block.get("text", ""), normalized_attachments
                        )
                        blocks.append({**block, "text": text})
                        for item in media:
                            key = item["name"]
                            if key not in seen_media:
                                seen_media.add(key)
                                media_entries.append(item)
                    else:
                        blocks.append(block)
                content = blocks
                if media_entries:
                    message[MEDIA_ATTACHMENTS_FIELD] = media_entries
            result[index] = {**message, "content": content}

        system_prompt_prefix_provider = getattr(
            self, "system_prompt_prefix_provider", None
        )
        if system_prompt_prefix_provider is not None:
            prefix = system_prompt_prefix_provider()
            if prefix:
                for index, message in enumerate(result):
                    if message.get("role") == "system":
                        out = dict(message)
                        out["content"] = [
                            {"type": "text", "text": prefix},
                            *out.get("content", []),
                        ]
                        result[index] = out
                        break

        ephemeral_parts = []
        if self.ephemeral:
            ephemeral_parts.append(self.ephemeral)
        ephemeral_provider = getattr(self, "ephemeral_provider", None)
        if ephemeral_provider is not None:
            dynamic_ephemeral = ephemeral_provider()
            if dynamic_ephemeral:
                ephemeral_parts.append(dynamic_ephemeral)
        if ephemeral_parts:
            ephemeral = "\n\n".join(ephemeral_parts)
            for index in range(len(result) - 1, -1, -1):
                if (
                    result[index].get("role") == "user"
                    and not result[index].get("_provider_checkpoint")
                ):
                    out = dict(result[index])
                    out["content"] = [
                        {"type": "text", "text": ephemeral},
                        *out.get("content", []),
                    ]
                    result[index] = out
                    break

        return self._with_cache_breakpoints(result)

    def call(self, messages=None, additional_messages=(), **kwargs):
        if messages is None:
            messages = self.projected_messages()
        else:
            messages = list(messages)
            for message in messages:
                _canonical_message(message)
        additional_messages = list(additional_messages)
        for message in additional_messages:
            _canonical_message(message)
        messages.extend(additional_messages)
        return self.llm_client.call(messages, **kwargs)

    def add_assistant_response(self):
        return self.append_message(self.call())

    def usermsg(self, content, **kwargs):
        return self.append_message({
            "role": "user",
            "content": content_blocks(content),
            **kwargs,
        })
