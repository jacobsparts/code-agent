import sys
assert sys.version_info >= (3, 8), "Requires Python 3.8+"
import os
import json
import http.client
import socket
import urllib.parse
import time
import logging
import base64
import contextlib
from collections import defaultdict

_NO_DEADLINE = object()


class DeadlineHTTPResponse(http.client.HTTPResponse):
    def __init__(self, sock, *, deadline, **kwargs):
        self._deadline = deadline
        self._deadline_socket = sock
        super().__init__(sock, **kwargs)

    def _apply_deadline(self):
        timeout = self._deadline()
        if timeout is not _NO_DEADLINE:
            self._deadline_socket.settimeout(timeout)

    def begin(self):
        self._apply_deadline()
        return super().begin()

    def read1(self, amt=-1):
        self._apply_deadline()
        return super().read1(amt)

    def read(self, amt=None):
        if amt is not None and amt < 0:
            amt = None
        chunks = []
        remaining = amt
        while remaining is None or remaining:
            size = 64 * 1024 if remaining is None else min(64 * 1024, remaining)
            chunk = self.read1(size)
            if not chunk:
                break
            chunks.append(chunk)
            if remaining is not None:
                remaining -= len(chunk)
        return b"".join(chunks)


class _DeadlineConnectionMixin:
    def __init__(self, *args, deadline=_NO_DEADLINE, **kwargs):
        self._deadline = (
            deadline
            if deadline is _NO_DEADLINE or deadline is None
            else time.monotonic() + deadline
        )
        if deadline is not _NO_DEADLINE:
            kwargs["timeout"] = self._remaining()
        super().__init__(*args, **kwargs)
        self.response_class = lambda sock, **response_kwargs: DeadlineHTTPResponse(
            sock, deadline=self._remaining, **response_kwargs
        )

    def _remaining(self):
        if self._deadline is _NO_DEADLINE or self._deadline is None:
            return self._deadline
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request deadline exceeded")
        return remaining

    def connect(self):
        timeout = self._remaining()
        if timeout is not _NO_DEADLINE:
            self.timeout = timeout
        return super().connect()

    def send(self, data):
        timeout = self._remaining()
        if timeout is not _NO_DEADLINE and self.sock is not None:
            self.sock.settimeout(timeout)
        return super().send(data)


class DeadlineHTTPConnection(_DeadlineConnectionMixin, http.client.HTTPConnection):
    pass


class DeadlineHTTPSConnection(_DeadlineConnectionMixin, http.client.HTTPSConnection):
    pass


from .provider_admission import ProviderAdmission

from .utils import UsageTracker
from .llm_registry import get_model_config
from .conversation import Conversation, MEDIA_ATTACHMENTS_FIELD
from . import codex, cursor
from .streaming import wrap_chat_completions_streaming_response
from .repl_tool_adapter import REPL_EXECUTE_TOOL, ReplExecuteResponseError, normalize_openai_repl_response, project_openai_repl_messages

# Define TCP keepalive constants for cross-platform compatibility
try:
    TCP_KEEPIDLE = socket.TCP_KEEPIDLE
except AttributeError:
    TCP_KEEPIDLE = getattr(socket, "TCP_KEEPALIVE", None)  # macOS uses TCP_KEEPALIVE

# Message keys passed through in addition to the standard public keys.
EXTRA_KEYS = set()

logger = logging.getLogger('code_agent')

class BadRequestError(Exception): pass
class MaxTokensError(Exception): pass
class ContextOverflowError(Exception): pass
class EmptyResponseError(Exception): pass

CONTEXT_INPUT_BUFFER = 4_000
CONTEXT_OUTPUT_HEADROOM = 16_000
TOKEN_RATIO_EMA_ALPHA = 0.2

IMAGE_MEDIA_TYPES = {"image/png", "image/jpeg"}
AUDIO_MEDIA_TYPES = {"audio/wav", "audio/mpeg"}

TRANSPORT_MEDIA_TYPES = {
    "completions": IMAGE_MEDIA_TYPES | AUDIO_MEDIA_TYPES,
    "responses": IMAGE_MEDIA_TYPES,
    "messages": IMAGE_MEDIA_TYPES,
    "gemini": IMAGE_MEDIA_TYPES | AUDIO_MEDIA_TYPES,
    "cursor": set(),
    "codex": IMAGE_MEDIA_TYPES,
}



def _parse_completions_response(response_json):
    if 'choices' not in response_json:
        raise Exception(f"choices missing from response: {response_json}")
    choice = response_json['choices'][0]
    return choice.get('message', {}), choice.get('finish_reason'), response_json.get('usage')


class LLMClient:
    usage_tracker = UsageTracker()

    def __init__(self, model_name, native=None):
        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        self.timeout = self.model_config.get('timeout')
        self.provider_admission = ProviderAdmission.from_model_config(
            model_name, self.model_config
        )
        self.native = self.model_config.get('tools') if native is None else native
        self.tool_mode = self.model_config.get('tool_mode')
        self.on_retry = None
        self._current_input_bytes = None


    def _input_bytes(self, messages, tools=None):
        messages = self._public_messages(messages)
        payload = {"messages": messages}
        if tools:
            payload["tools"] = tools
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=self._json_size_default).encode("utf-8"))

    @staticmethod
    def _public_messages(messages):
        return [{k: v for k, v in m.items() if not k.startswith('_')} for m in messages]

    def validate_media_type(self, media_type):
        if not isinstance(media_type, str) or "/" not in media_type:
            raise BadRequestError("Invalid media attachment type")
        api_type = self.model_config["api_type"]
        if media_type not in TRANSPORT_MEDIA_TYPES.get(api_type, set()):
            raise BadRequestError(
                f"{api_type} transport does not support {media_type} attachments"
            )

    def _pop_media_attachments(self, message):
        media = message.pop(MEDIA_ATTACHMENTS_FIELD, None) or []
        for item in media:
            if not isinstance(item, dict):
                raise BadRequestError("Invalid projected media attachment")
            self.validate_media_type(item.get("media_type"))
            if not isinstance(item.get("content"), bytes):
                raise BadRequestError("Projected media attachment has no binary content")
        return media

    @staticmethod
    def _strip_response_media(message):
        message.pop("images", None)
        return message

    @staticmethod
    def _json_size_default(value):
        if isinstance(value, bytes):
            return f"<{len(value)} bytes>"
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _input_tokens_per_byte(self):
        return getattr(self.usage_tracker, "input_tokens_per_byte", {}).get(self.model_name)

    def _update_input_tokens_per_byte(self, input_bytes, usage):
        if not usage or not input_bytes:
            return
        normalized = self.usage_tracker._normalize(self.model_name, usage)
        input_tokens = normalized['prompt_tokens'] + normalized['cached_tokens']
        if input_tokens <= 0:
            return
        ratios = getattr(self.usage_tracker, "input_tokens_per_byte", None)
        if ratios is None:
            ratios = {}
            self.usage_tracker.input_tokens_per_byte = ratios
        observed = input_tokens / input_bytes
        old = ratios.get(self.model_name)
        ratios[self.model_name] = observed if old is None else (
            old * (1 - TOKEN_RATIO_EMA_ALPHA) + observed * TOKEN_RATIO_EMA_ALPHA
        )

    def _estimate_input_tokens(self, input_bytes):
        ratio = self._input_tokens_per_byte()
        if ratio is None:
            return None
        return int(input_bytes * ratio) + CONTEXT_INPUT_BUFFER

    def _validate_context_budget(self, input_bytes):
        self._current_input_bytes = input_bytes
        estimated_input = self._estimate_input_tokens(input_bytes)
        if estimated_input is None:
            return
        max_input_tokens = self.model_config.get('max_input_tokens')
        if max_input_tokens is not None and estimated_input > max_input_tokens:
            raise ContextOverflowError(
                f"estimated input {estimated_input:,} tokens exceeds max_input_tokens "
                f"{max_input_tokens:,} for {self.model_name}"
            )
        context_window = self.model_config.get('context_window')
        if context_window is not None and estimated_input + CONTEXT_OUTPUT_HEADROOM > context_window:
            raise ContextOverflowError(
                f"estimated input {estimated_input:,} tokens + output headroom "
                f"{CONTEXT_OUTPUT_HEADROOM:,} exceeds context_window "
                f"{context_window:,} for {self.model_name}"
            )

    def _call_completions(self, messages, tools):
        """
        Call OpenAI Completions API.

        Args:
            messages: List of projected message dicts.
            tools: Optional tool specifications.
        """
        # OpenAI Completions API-compatible format
        prepared = []
        for m in messages:
            out = dict(m)
            breakpoint = out.pop('_prompt_cache_breakpoint', None)
            if media := self._pop_media_attachments(out):
                blocks = []
                if out['content']:
                    blocks.append({"type": "text", "text": out['content']})
                for item in media:
                    encoded = base64.b64encode(item['content']).decode()
                    if item['media_type'].startswith('image/'):
                        blocks.append({"type": "image_url", "image_url": {
                            "url": f"data:{item['media_type']};base64,{encoded}"
                        }})
                    else:
                        blocks.append({"type": "input_audio", "input_audio": {
                            "data": encoded,
                            "format": "wav" if item['media_type'] == "audio/wav" else "mp3",
                        }})
                out['content'] = blocks
            elif breakpoint and self.model_config.get('explicit_prompt_cache') and isinstance(out.get('content'), str):
                out['content'] = [{
                    "type": "text",
                    "text": out['content'],
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }]
            prepared.append(out)
        messages = self._public_messages(prepared)

        context_window = self.model_config.get('context_window')
        extra_config = dict(self.model_config.get('config', {}))
        current_max_tokens = extra_config.get('max_tokens')
        messages = list(messages)
        max_tokens_retry = 0

        while True:
            req = {
                "model": self.model_config['model'],
                "messages": messages,
                **extra_config,
            }
            if self.tool_mode == "repl_execute":
                req["tools"] = [REPL_EXECUTE_TOOL]
            elif tools:
                raise TypeError("Provider-side tool calls are not supported by Code Agent")
            if self.model_config['port'] == 443:
                conn = DeadlineHTTPSConnection(self.model_config['host'], timeout=self.timeout, deadline=self.timeout)
                conn.connect()
                sock = conn.sock
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if TCP_KEEPIDLE is not None:
                    sock.setsockopt(
                        socket.IPPROTO_TCP, TCP_KEEPIDLE, 60
                    )  # 60 sec idle before keepalive
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)    # 10 sec between probes
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)       # 3 probes before giving up
            else:
                conn = DeadlineHTTPConnection(self.model_config['host'], self.model_config['port'], timeout=self.timeout, deadline=self.timeout)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.model_config['api_key']}",
            }
            body = json.dumps(req)
            request_path = self.model_config.get('request_path', self.model_config['path'])
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("----------- TO LLM -----------")
                    logger.debug(f"POST {request_path} {headers}")
                    logger.debug(body)
                conn.request("POST", request_path, body, headers)
                response = conn.getresponse()
                content_type = ""
                if getattr(response, "headers", None):
                    content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type.lower():
                    response = wrap_chat_completions_streaming_response(response)
                response_data = response.read().decode()
                if logger.isEnabledFor(logging.INFO):
                    logger.info("---------- FROM LLM ----------")
                    logger.info(response_data)
                if response.status == 429:
                    print(response)
                    logger.warning("Throttled. Waiting 20s")
                    time.sleep(20)
                    raise Exception("Throttled")
                if response.status == 400:
                    logger.debug(req)
                    raise BadRequestError(response_data.strip())
                elif response.status != 200:
                    raise Exception(f"API Error {response.status}: {response_data}")

                response_json = json.loads(response_data)
                parser = self.model_config.get('response_parser') or _parse_completions_response
                message, stop_reason, usage = parser(response_json)
                if self.tool_mode == "repl_execute":
                    message = normalize_openai_repl_response(message)
                if usage:
                    self.usage_tracker.log(self.model_name, usage)
                    self._update_input_tokens_per_byte(self._current_input_bytes, usage)
                message['_stop_reason'] = stop_reason

                # Truncated response: feed it back and retry with doubled max_tokens.
                # Keeps doubling until prompt + output would exceed context_window.
                # Retry messages stay local — they never reach the Conversation history.
                if stop_reason in ('max_tokens', 'length', 'MAX_TOKENS') and context_window and current_max_tokens and usage:
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    next_max_tokens = current_max_tokens * 2
                    if prompt_tokens + next_max_tokens <= context_window:
                        max_tokens_retry += 1
                        if self.on_retry:
                            self.on_retry("max_tokens", max_tokens_retry)
                        messages.append(message)
                        messages.append({'role': 'user', 'content': 'Incomplete response detected. Resubmit your response.'})
                        current_max_tokens = next_max_tokens
                        extra_config['max_tokens'] = current_max_tokens
                        logger.warning(f"stop_reason={stop_reason}, doubling max_tokens to {current_max_tokens}")
                        continue

                return message
            finally:
                conn.close()

    def _responses_request(self, messages, tools):
        config = dict(self.model_config.get('config', {}))
        if 'reasoning_effort' in config:
            config['reasoning'] = {'effort': config.pop('reasoning_effort')}
        if 'max_tokens' in config:
            config['max_output_tokens'] = config.pop('max_tokens')
        input_items = []
        has_cache_breakpoint = False
        for message in messages:
            message = dict(message)
            media = self._pop_media_attachments(message)
            role = message.get('role')
            cache_breakpoint = bool(message.get('_prompt_cache_breakpoint')) and self.model_config.get('explicit_prompt_cache')
            if role == 'tool':
                input_items.append({
                    'type': 'function_call_output',
                    'call_id': message.get('tool_call_id'),
                    'output': message.get('content', ''),
                })
                continue
            content = message.get('content')
            if isinstance(content, str):
                content = [{'type': 'text', 'text': content}]
            elif not isinstance(content, list):
                content = []
            blocks = []
            for block in content:
                if not isinstance(block, dict) or block.get('type') == 'reasoning':
                    continue
                kind = block.get('type')
                if kind == 'text':
                    blocks.append({
                        'type': 'output_text' if role == 'assistant' else 'input_text',
                        'text': block.get('text', ''),
                    })
                elif kind == 'image_url':
                    image_url = block.get('image_url')
                    if isinstance(image_url, dict):
                        image_url = image_url.get('url')
                    blocks.append({'type': 'input_image', 'image_url': image_url})
                elif kind == 'output_text' and role != 'assistant':
                    blocks.append({'type': 'input_text', 'text': block.get('text', '')})
                else:
                    blocks.append(dict(block))
            blocks.extend({
                'type': 'input_image',
                'image_url': (
                    f"data:{item['media_type']};base64,"
                    f"{base64.b64encode(item['content']).decode()}"
                ),
            } for item in media)
            if cache_breakpoint and blocks:
                blocks[-1] = {
                    **blocks[-1],
                    'prompt_cache_breakpoint': {'mode': 'explicit'},
                }
                has_cache_breakpoint = True
            if blocks:
                input_items.append({'role': role, 'content': blocks})
            for call in message.get('tool_calls') or []:
                function = call.get('function') or {}
                input_items.append({
                    'type': 'function_call',
                    'call_id': call.get('id'),
                    'name': function.get('name'),
                    'arguments': function.get('arguments', ''),
                })
        response_tools = []
        for tool in ([REPL_EXECUTE_TOOL] if self.tool_mode == 'repl_execute' else tools or []):
            response_tools.append({'type': 'function', **tool.get('function', tool)})
        req = {'model': self.model_config['model'], 'input': input_items, **config}
        if has_cache_breakpoint:
            req.setdefault('prompt_cache_options', {'mode': 'explicit'})
        if response_tools:
            req['tools'] = response_tools
        return req

    def _parse_responses_result(self, response_json):
        output = response_json.get('output')
        if not isinstance(output, list):
            raise Exception(f"output missing from response: {response_json}")
        text = []
        calls = []
        for item in output:
            if item.get('type') == 'message':
                msg_text = '\n'.join(
                    block['text']
                    for block in item.get('content', [])
                    if block.get('type') in ('output_text', 'text') and block.get('text')
                )
                phase = item.get('phase')
                if phase not in (None, 'final'):
                    if phase != 'commentary':
                        sys.stderr.write(f"Warning: unrecognized Responses API message phase: {phase!r}\n")
                    if msg_text:
                        msg_text = '# ' + '\n# '.join(msg_text.split('\n'))
                if msg_text:
                    text.append(msg_text)
            elif item.get('type') == 'output_text' and item.get('text'):
                text.append(item['text'])
            elif item.get('type') == 'function_call':
                calls.append({
                    'id': item.get('call_id') or item.get('id'),
                    'type': 'function',
                    'function': {
                        'name': item.get('name'),
                        'arguments': item.get('arguments', ''),
                    },
                })
        if not text and isinstance(response_json.get('output_text'), str):
            text.append(response_json['output_text'])
        message = {'role': 'assistant', 'content': '\n'.join(part for part in text if part)}
        if calls:
            message['tool_calls'] = calls
        if self.tool_mode == 'repl_execute':
            message = normalize_openai_repl_response(message)
        if usage := response_json.get('usage'):
            self.usage_tracker.log(self.model_name, usage)
            self._update_input_tokens_per_byte(self._current_input_bytes, usage)
        incomplete = response_json.get('incomplete_details') or {}
        message['_stop_reason'] = incomplete.get('reason')
        if message['_stop_reason'] is None:
            message['_stop_reason'] = (
                'tool_calls' if calls
                else 'stop' if response_json.get('status') == 'completed'
                else response_json.get('status')
            )
        return message

    def _call_responses(self, messages, tools):
        req = self._responses_request(messages, tools)
        if self.model_config['port'] == 443:
            conn = DeadlineHTTPSConnection(self.model_config['host'], timeout=self.timeout, deadline=self.timeout)
            conn.connect()
        else:
            conn = DeadlineHTTPConnection(
                self.model_config['host'], self.model_config['port'], timeout=self.timeout, deadline=self.timeout
            )
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.model_config['api_key']}",
        }
        try:
            request_path = self.model_config.get('request_path', self.model_config['path'])
            body = json.dumps(req)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("----------- TO LLM -----------")
                logger.debug(f"POST {request_path} {headers}")
                logger.debug(body)
            conn.request('POST', request_path, body, headers)
            response = conn.getresponse()
            response_data = response.read().decode()
            if logger.isEnabledFor(logging.INFO):
                logger.info("---------- FROM LLM ----------")
                logger.info(response_data)
            if response.status == 400:
                raise BadRequestError(response_data.strip())
            if response.status != 200:
                raise Exception(f"API Error {response.status}: {response_data}")
            return self._parse_responses_result(json.loads(response_data))
        finally:
            conn.close()

    def _call_cursor(self, messages, tools):
        if self.tool_mode != "repl_execute":
            raise TypeError("Cursor transport requires tool_mode='repl_execute'")
        prepared = []
        for message in messages:
            message = dict(message)
            self._pop_media_attachments(message)
            prepared.append(message)
        messages = self._public_messages(prepared)
        req = {
            "model": self.model_config["model"],
            "messages": messages,
            "tools": [REPL_EXECUTE_TOOL],
            **self.model_config.get("config", {}),
        }
        response_json = cursor.chat_completions(
            self.model_config["api_key"],
            req,
        )
        message, stop_reason, usage = _parse_completions_response(response_json)
        message = normalize_openai_repl_response(message)
        if usage:
            self.usage_tracker.log(self.model_name, usage)
            self._update_input_tokens_per_byte(self._current_input_bytes, usage)
        message["_stop_reason"] = stop_reason
        return message

    def _call_codex(self, messages, tools):
        req = self._responses_request(messages, tools)
        # Stage idle budgets live in code_agent.codex (60s/30s/30s).
        # Do not impose a total wall-clock timeout here.
        return self._parse_responses_result(codex.responses(req))

    def _call_messages(self, messages, tools):
        """
        Call Anthropic Messages API.

        Args:
            messages: List of projected message dicts.
            tools: Optional tool specifications.
        """
        # Anthropic Messages API-compatible format
        prepared = []
        for message in messages:
            m = dict(message)
            if media := self._pop_media_attachments(m):
                m['content'] = [
                    *([{"type": "text", "text": m['content']}] if m['content'] else []),
                    *[{"type": "image", "source": {
                        "type": "base64",
                        "media_type": item["media_type"],
                        "data": base64.b64encode(item["content"]).decode()
                    }} for item in media]
                ]
            prepared.append(m)
        messages = self._public_messages(prepared)
        system_message = None
        _messages = []
        for msg in messages:
            if msg['role'] == 'system':
                if system_message is None:
                    system_message = msg['content']
                else:
                    _messages.append({**msg, 'role': 'user'})
            elif msg['role'] == 'tool':
                _messages.append({'role': 'user', 'content': msg.get('content', '')})
            else:
                _messages.append({k: v for k, v in msg.items() if k != 'tool_calls'})
        req = {
            "model": self.model_config['model'],
            "messages": _messages,
            "max_tokens": self.model_config.get('config', {}).get('max_tokens', 4096),
            **{k: v for k, v in self.model_config.get('config', {}).items() if k != 'max_tokens'}
        }
        if system_message:
            req["system"] = system_message
        if tools:
            raise TypeError("Provider-side tool calls are not supported by Code Agent")
        conn = DeadlineHTTPSConnection(self.model_config['host'], timeout=self.timeout, deadline=self.timeout)
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.model_config['api_key'],
            "anthropic-version": "2023-06-01",
        }
        body = json.dumps(req)
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("----------- TO LLM -----------")
                logger.debug(f"POST {self.model_config['path']} {headers}")
                logger.debug(body)
            conn.request("POST", self.model_config['path'], body, headers)
            response = conn.getresponse()
            response_data = response.read().decode()
            if logger.isEnabledFor(logging.INFO):
                logger.info("---------- FROM LLM ----------")
                logger.info(response_data)
            if response.status == 429:
                logger.warning("Throttled. Waiting 20s")
                time.sleep(20)
                raise Exception("Throttled")
            if response.status == 400:
                logger.debug(req)
                raise BadRequestError(response_data.strip())
            elif response.status != 200:
                raise Exception(f"API Error {response.status}: {response_data}")
            response_json = json.loads(response_data)
            if usage := response_json.get('usage'):
                self.usage_tracker.log(self.model_name, usage)
                self._update_input_tokens_per_byte(self._current_input_bytes, usage)
            text = [
                block['text'] for block in response_json.get('content', [])
                if block['type'] == 'text' and block.get('text')
            ]
            message = {
                'role': 'assistant',
                'content': '\n'.join(text)
            }
            message['_stop_reason'] = response_json.get('stop_reason')
            return message
        finally:
            conn.close()

    def _call_gemini(self, messages, tools):
        """
        Call Gemini native generateContent API.

        Args:
            messages: List of projected message dicts.
                          tools: Optional tool specifications.
        """
        contents = []
        system_parts = []
        for original in messages:
            m = dict(original)
            media = self._pop_media_attachments(m)
            role = m['role']
            if role == 'system':
                system_parts.append({"text": m['content']})
                continue
            if role == 'tool':
                contents.append({
                    "role": "user",
                    "parts": [{"text": m.get('content', '')}]
                })
                continue
            if role == 'assistant':
                parts = []
                if m.get('content'):
                    parts.append({"text": m['content']})
                contents.append({"role": "model", "parts": parts})
                continue
            # role == 'user'
            parts = []
            if m.get('content'):
                parts.append({"text": m['content']})
            for item in media:
                parts.append({"inlineData": {
                    "mimeType": item["media_type"],
                    "data": base64.b64encode(item["content"]).decode()
                }})
            contents.append({"role": "user", "parts": parts})
        # Merge consecutive same-role messages (required by Gemini API)
        merged = []
        for entry in contents:
            if merged and merged[-1]['role'] == entry['role']:
                merged[-1]['parts'].extend(entry['parts'])
            else:
                merged.append(entry)
        # Build request
        model_name = self.model_config['model']
        path = f"{self.model_config['path']}/models/{model_name}:generateContent"
        req = {"contents": merged}
        if system_parts:
            req["systemInstruction"] = {"parts": system_parts}
        # Map config keys to generationConfig
        generation_config = {}
        thinking_config = {}
        for k, v in self.model_config.get('config', {}).items():
            if k == 'max_tokens':
                generation_config['maxOutputTokens'] = v
            elif k in ('thinkingBudget', 'thinkingLevel'):
                thinking_config[k] = v
            else:
                generation_config[k] = v
        if thinking_config:
            generation_config['thinkingConfig'] = thinking_config
        if generation_config:
            req["generationConfig"] = generation_config
        if tools:
            raise TypeError("Provider-side tool calls are not supported by Code Agent")
        conn = DeadlineHTTPSConnection(self.model_config['host'], timeout=self.timeout, deadline=self.timeout)
        conn.connect()
        sock = conn.sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if TCP_KEEPIDLE is not None:
            sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPIDLE, 60)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.model_config['api_key'],
        }
        body = json.dumps(req)
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("----------- TO LLM -----------")
                logger.debug(f"POST {path} {headers}")
                logger.debug(body)
            conn.request("POST", path, body, headers)
            response = conn.getresponse()
            response_data = response.read().decode()
            if logger.isEnabledFor(logging.INFO):
                logger.info("---------- FROM LLM ----------")
                logger.info(response_data)
            if response.status == 429:
                logger.warning("Throttled. Waiting 20s")
                time.sleep(20)
                raise Exception("Throttled")
            if response.status == 400:
                logger.debug(req)
                raise BadRequestError(response_data.strip())
            elif response.status != 200:
                raise Exception(f"API Error {response.status}: {response_data}")
            response_json = json.loads(response_data)
            if usage := response_json.get('usageMetadata'):
                self.usage_tracker.log(self.model_name, usage)
                self._update_input_tokens_per_byte(self._current_input_bytes, usage)
            if not response_json.get('candidates'):
                raise Exception(f"candidates missing from response: {response_json}")
            candidate = response_json['candidates'][0]
            text = [
                part.get('text') or json.dumps(part['functionCall'])
                for part in candidate.get('content', {}).get('parts', [])
                if part.get('text') or 'functionCall' in part
            ]
            message = {'role': 'assistant', 'content': '\n'.join(text)}
            message['_stop_reason'] = candidate.get('finishReason')
            return message
        finally:
            conn.close()

    def prepare_message(self, m):
        if m['role'] == 'tool':
            return {
                'role': 'user',
                'content': f"{m.get('name', 'tool')}: {m.get('content', '')}",
                **{k: v for k, v in m.items() if k in EXTRA_KEYS}
            }
        return {k: v for k, v in m.items() if k != 'tool_calls'}

    @staticmethod
    def _strip_tool_metadata(messages):
        return [{k: v for k, v in msg.items() if k != 'tool_calls'} for msg in messages]

    def _call(self, messages, tools=None):
        if self.tool_mode == "repl_execute":
            messages = project_openai_repl_messages(messages)
        else:
            if not self.native:
                messages = [self.prepare_message(msg) for msg in messages]
            messages = self._strip_tool_metadata(messages)
        size_tools = [REPL_EXECUTE_TOOL] if self.tool_mode == "repl_execute" else tools
        self._validate_context_budget(self._input_bytes(messages, size_tools))
        api_type = self.model_config['api_type']
        callers = {
            "completions": self._call_completions,
            "responses": self._call_responses,
            "messages": self._call_messages,
            "cursor": self._call_cursor,
            "codex": self._call_codex,
            "gemini": self._call_gemini,
        }
        try:
            caller = callers[api_type]
        except KeyError:
            raise NotImplementedError(api_type)
        return self._strip_response_media(caller(messages, tools))

    @staticmethod
    def _sleep_backoff(attempt, base=15):
        """
        Exponential back-off helper. Sleeps for `base * 2**attempt` seconds.
        """
        time.sleep(base * (2 ** attempt))

    def text_call(self, messages, retry=3, attempt=0):
        try:
            context = (
                self.provider_admission.admitted()
                if self.provider_admission is not None
                else contextlib.nullcontext()
            )
            with context:
                response = self._call(messages)
                content = response.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise EmptyResponseError("LLM returned an empty response")
                return response
        except (ContextOverflowError, ReplExecuteResponseError):
            raise
        except Exception as e:
            err = (str(e) if len(str(e)) < 1000 else str(e)[:1000]+'...').replace("\n"," ")
            logger.error(f"text_call {type(e).__name__}: {err}", exc_info=True)
            if retry:
                self._sleep_backoff(attempt)
                return self.text_call(messages, retry-1, attempt+1)
            raise



    def conversation(self, system_prompt):
        return Conversation(self, system_prompt)
