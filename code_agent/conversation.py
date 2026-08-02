import json
from .preview_refs import render_preview_refs


class Conversation:
    def __init__(self, llm_client, system_prompt):
        self.llm_client = llm_client
        self.messages = [ {"role": "system", "content": system_prompt} ]
        self.ephemeral = ""
        self._prompt_cache = []
        self._prompt_cache_model = getattr(llm_client, "model_name", None)

    def _with_cache_breakpoints(self, messages):
        """Annotate projected messages; update continuity for the next call."""
        if self.llm_client is None:
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

    def _messages(self):
        result = []
        expanded_preview_refs = getattr(self, "expanded_preview_refs", {})
        preview_loader = getattr(self, "preview_loader", None)

        rendered_preview_refs = []

        messages_projector = getattr(self, "messages_projector", None)
        projected_messages = (
            messages_projector(self.messages)
            if messages_projector is not None
            else self.messages
        )
        message_projector = getattr(self, "message_projector", None)
        for msg in projected_messages:
            out = message_projector(msg) if message_projector is not None else dict(msg)
            attachments = out.pop('_attachments', None)
            if attachments:
                for name, content in attachments.items():
                    out['content'] = out.get('content', '').replace(f'[Attachment: {name}]', content)
            if preview_loader is not None:
                out['content'] = render_preview_refs(
                    out.get('content', ''),
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
                    out["content"] = ephemeral + ("\n\n" + content if content else "")
                    result[i] = out
                    break

        return self._with_cache_breakpoints(result)

    def _append_message(self, message):
        self.messages.append(message)

    def add_assistant_response(self):
        resp_msg = self.llm_client.text_call(self._messages())
        self.messages.append(resp_msg)
        return resp_msg

    def usermsg(self, content, **kwargs):
        content = content if type(content) is str else json.dumps(content)
        message = {"role": 'user', "content": content, **kwargs}
        self._append_message(message)

