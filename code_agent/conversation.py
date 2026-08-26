import base64
import json
from .preview_refs import render_preview_refs
from .repl_attachment_mixin import (
    AudioAttachment,
    ImageAttachment,
    TextAttachment,
    iter_placeholders,
    normalize_message_attachments,
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


class Conversation:
    def __init__(self, llm_client, system_prompt, convo=None):
        self.convo = convo or Convo(llm_client, system_prompt)
        if llm_client is not None:
            self.convo.llm_client = llm_client
        self.ephemeral = ""
        self._prompt_cache = []
        self._prompt_cache_model = getattr(llm_client, "model_name", None)
        self._legacy_targets = {}

    @property
    def llm_client(self):
        return self.convo.llm_client

    @llm_client.setter
    def llm_client(self, value):
        self.convo.llm_client = value

    def _remember(self, legacy, canonical):
        self._legacy_targets[id(legacy)] = (legacy, canonical)
        return legacy

    @staticmethod
    def _attachment(media_type, data_type, data):
        return {
            "type": "attachment",
            "media_type": media_type,
            "data_type": data_type,
            "data": data,
        }

    def _content_block_to_canonical(self, block):
        kind = block["type"]
        if kind in ("text", "input_text", "output_text"):
            return {"type": "text", "text": block["text"]}
        if kind == "input_file":
            return self._attachment(
                block.get("media_type"),
                "provider_id",
                block["file_id"],
            )
        if kind == "image_url":
            image_url = block["image_url"]
            return self._attachment(
                block.get("media_type"),
                "url",
                image_url["url"] if isinstance(image_url, dict) else image_url,
            )
        if kind == "input_audio":
            audio = block["input_audio"]
            media_type = {
                "wav": "audio/wav",
                "mp3": "audio/mpeg",
            }.get(audio["format"])
            if media_type is None:
                raise NotImplementedError(
                    f"Unknown input_audio format: {audio['format']!r}"
                )
            return self._attachment(
                media_type,
                "bytes",
                base64.b64decode(audio["data"]),
            )
        raise NotImplementedError(f"Unknown legacy content type: {kind!r}")

    def _to_canonical(self, message):
        from .client import BadRequestError

        blocks = []
        content = message["content"]
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif content is not None:
            blocks.extend(
                self._content_block_to_canonical(block)
                for block in content
            )
        for item in message.get(MEDIA_ATTACHMENTS_FIELD) or []:
            if not isinstance(item, dict):
                raise BadRequestError("Invalid projected media attachment")
            data = item.get("content")
            if not isinstance(data, bytes):
                raise BadRequestError(
                    "Projected media attachment has no binary content"
                )
            blocks.append(
                self._attachment(item.get("media_type"), "bytes", data)
            )
        if message["role"] == "assistant":
            for tool_call in message.get("tool_calls") or []:
                function = tool_call["function"]
                blocks.append({
                    "type": "tool_call",
                    "id": tool_call["id"],
                    "name": function["name"],
                    "args": json.loads(function["arguments"]),
                })
        result = {"role": message["role"], "content": blocks}
        for key in ("tool_call_id", "name"):
            if key in message:
                result[key] = message[key]
        result.update({
            key: value
            for key, value in message.items()
            if key.startswith("_") and key != MEDIA_ATTACHMENTS_FIELD
        })
        return result

    def _to_legacy(self, message, *, response=False):
        text = []
        calls = []
        legacy_blocks = []
        for block in message["content"]:
            kind = block["type"]
            if kind == "text":
                text.append(block["text"])
                legacy_blocks.append({"type": "text", "text": block["text"]})
            elif kind == "commentary":
                rendered = "# " + "\n# ".join(block["text"].split("\n"))
                text.append(rendered)
                legacy_blocks.append({"type": "text", "text": rendered})
            elif kind == "tool_call":
                calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["args"]),
                    },
                })
            elif kind == "reasoning":
                continue
            elif kind == "attachment":
                if response:
                    print(f"Legacy Conversation cannot store attachment responses, ignoring")
                    continue
                data_type = block["data_type"]
                if data_type == "provider_id":
                    legacy_blocks.append({
                        "type": "input_file",
                        "file_id": block["data"],
                        **({
                            "media_type": block["media_type"],
                        } if block.get("media_type") is not None else {}),
                    })
                elif data_type == "url":
                    legacy_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": block["data"]},
                        **({
                            "media_type": block["media_type"],
                        } if block.get("media_type") is not None else {}),
                    })
                else:
                    raise NotImplementedError(
                        "Legacy Conversation cannot represent stored binary "
                        "attachment blocks"
                    )
            else:
                raise NotImplementedError(
                    f"Unknown transport content type: {kind!r}"
                )

        if response or message["role"] == "assistant" or all(
            block["type"] == "text" for block in legacy_blocks
        ):
            content = "\n".join(text)
        else:
            content = legacy_blocks
        result = {"role": message["role"], "content": content}
        if calls:
            result["tool_calls"] = calls
        for key in ("tool_call_id", "name"):
            if key in message:
                result[key] = message[key]
        result.update({
            key: value
            for key, value in message.items()
            if key.startswith("_")
        })
        metadata = message.get("provider_metadata")
        if metadata and "stop_reason" in metadata:
            result["_stop_reason"] = metadata["stop_reason"]
        return result

    def _with_cache_breakpoints(self, messages):
        """Annotate projected messages; update continuity for the next call."""
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
                and isinstance(content, str)
                and out.get("role") in ("system", "user")
            ):
                content_hash = hash(content)
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
        return [
            self._remember(self._to_legacy(message), message)
            for message in self.convo.stored_messages()
        ]

    def replace_messages(self, messages):
        self._legacy_targets.clear()
        canonical_messages = []
        for message in messages:
            message = normalize_message_attachments(message)
            canonical = self._to_canonical(message)
            canonical_messages.append(canonical)
            self._remember(message, canonical)
        self.convo.replace_messages(canonical_messages)

    def append_message(self, message):
        message = normalize_message_attachments(message)
        canonical = self.convo.append_message(self._to_canonical(message))
        return self._remember(message, canonical)

    def extend_messages(self, messages):
        return [self.append_message(message) for message in messages]

    def insert_message(self, index, message):
        message = normalize_message_attachments(message)
        canonical = self.convo.insert_message(index, self._to_canonical(message))
        return self._remember(message, canonical)

    def pop_message(self, index=-1):
        canonical = self.convo.pop_message(index)
        return self._remember(self._to_legacy(canonical), canonical)

    def update_message(self, message, **changes):
        message.update(changes)
        remembered = self._legacy_targets.get(id(message))
        if remembered is not None and remembered[0] is message:
            canonical = remembered[1]
            canonical.clear()
            canonical.update(self._to_canonical(message))
        return message

    def remove_message_fields(self, message, *fields):
        for field in fields:
            message.pop(field, None)
        remembered = self._legacy_targets.get(id(message))
        if remembered is not None and remembered[0] is message:
            canonical = remembered[1]
            canonical.clear()
            canonical.update(self._to_canonical(message))
        return message

    def projected_messages(self):
        result = []
        expanded_preview_refs = getattr(self, "expanded_preview_refs", {})
        preview_loader = getattr(self, "preview_loader", None)
        rendered_preview_refs = []

        messages_projector = getattr(self, "messages_projector", None)
        stored = self.stored_messages()
        projected_messages = (
            messages_projector(stored)
            if messages_projector is not None
            else stored
        )
        message_projector = getattr(self, "message_projector", None)
        for msg in projected_messages:
            out = message_projector(msg) if message_projector is not None else dict(msg)
            normalize_message_attachments(out)
            attachments = out.pop("_attachments", None)
            if attachments:
                content, media = materialize_attachments(
                    out.get("content", ""), attachments
                )
                out["content"] = content
                if media:
                    out[MEDIA_ATTACHMENTS_FIELD] = media
            if preview_loader is not None:
                out["content"] = render_preview_refs(
                    out.get("content", ""),
                    expanded_preview_refs,
                    preview_loader,
                    rendered_preview_refs,
                )
            result.append(out)
        self.rendered_preview_refs = rendered_preview_refs

        system_prompt_prefix_provider = getattr(
            self, "system_prompt_prefix_provider", None
        )
        if system_prompt_prefix_provider is not None:
            system_prompt_prefix = system_prompt_prefix_provider()
            if system_prompt_prefix:
                for i, message in enumerate(result):
                    if message.get("role") == "system":
                        out = dict(message)
                        content = out.get("content", "")
                        out["content"] = system_prompt_prefix + (
                            "\n\n" + content if content else ""
                        )
                        result[i] = out
                        break

        ephemeral_parts = []
        if self.ephemeral:
            ephemeral_parts.append(self.ephemeral)
        ephemeral_provider = getattr(self, "ephemeral_provider", None)
        if ephemeral_provider is not None:
            dynamic_ephemeral = ephemeral_provider()
            if dynamic_ephemeral:
                ephemeral_parts.append(dynamic_ephemeral)
        ephemeral = "\n\n".join(ephemeral_parts)
        if ephemeral:
            for i in range(len(result) - 1, -1, -1):
                if (
                    result[i].get("role") == "user"
                    and not result[i].get("_provider_checkpoint")
                ):
                    out = dict(result[i])
                    content = out.get("content", "")
                    out["content"] = ephemeral + (
                        "\n\n" + content if content else ""
                    )
                    result[i] = out
                    break

        return self._with_cache_breakpoints(result)

    def call(self, messages=None, additional_messages=(), **kwargs):
        if messages is None:
            messages = self.projected_messages()
        else:
            messages = list(messages)
        messages.extend(additional_messages)
        response = self.convo.call([
            self._to_canonical(message)
            for message in messages
        ], **kwargs)
        return self._to_legacy(response, response=True)

    def add_assistant_response(self):
        return self.append_message(self.call())

    def usermsg(self, content, **kwargs):
        content = content if type(content) is str else json.dumps(content)
        return self.append_message({"role": "user", "content": content, **kwargs})


class Convo:
    def __init__(self, llm_client, system_prompt):
        self.llm_client = llm_client
        content = (
            system_prompt
            if isinstance(system_prompt, list)
            else [{"type": "text", "text": system_prompt}]
        )
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
        self._messages = list(messages)

    def append_message(self, message):
        self._messages.append(message)
        return message

    def extend_messages(self, messages):
        messages = list(messages)
        self._messages.extend(messages)
        return messages

    def insert_message(self, index, message):
        self._messages.insert(index, message)
        return message

    def pop_message(self, index=-1):
        return self._messages.pop(index)

    def update_message(self, message, **changes):
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
        messages.extend(additional_messages)
        return self.llm_client.call(messages, **kwargs)

    def add_assistant_response(self):
        return self.append_message(self.call())

    def usermsg(self, content, **kwargs):
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif not isinstance(content, list):
            content = [{"type": "text", "text": json.dumps(content)}]
        return self.append_message({"role": "user", "content": content, **kwargs})
