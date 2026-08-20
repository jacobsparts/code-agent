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
from .conversation import Conversation
from . import codex, cursor
from .streaming import wrap_chat_completions_streaming_response
from .repl_tool_adapter import REPL_EXECUTE_TOOL, ReplExecuteResponseError, repl_response_to_text, project_repl_tool_history

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
    return choice['message'], choice.get('finish_reason'), response_json.get('usage')


def _attachment(media_type, data_type, data):
    return {
        'type': 'attachment',
        'media_type': media_type,
        'data_type': data_type,
        'data': data,
    }


def _text_only_content(blocks, context):
    text = []
    for block in blocks:
        kind = block['type']
        if kind == 'text':
            text.append(block['text'])
        else:
            raise NotImplementedError(
                f"Unknown {context} content type: {kind!r}"
            )
    return '\n'.join(text)


def _apply_native_policy(messages, native):
    projected = []
    for message in messages:
        if not native and message['role'] == 'tool':
            text = _text_only_content(
                message['content'], 'tool result'
            )
            projected.append({
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': f"{message.get('name', 'tool')}: {text}",
                }],
            })
            continue
        content = []
        for block in message['content']:
            kind = block['type']
            if kind == 'tool_call':
                continue
            if kind in (
                'text', 'commentary', 'reasoning', 'attachment',
            ):
                content.append(block)
            else:
                raise NotImplementedError(
                    f"Unknown transport content type: {kind!r}"
                )
        projected.append({**message, 'content': content})
    return projected


def _openai_compatible_message_to_transport_blocks(message):
    content = message['content']
    if isinstance(content, str):
        blocks = [{'type': 'text', 'text': content}] if content else []
    elif content is None:
        blocks = []
    else:
        blocks = []
        for block in content:
            kind = block['type']
            if kind in ('text', 'commentary', 'reasoning'):
                blocks.append({'type': kind, 'text': block['text']})
            elif kind == 'image_url':
                image_url = block['image_url']
                blocks.append(_attachment(
                    block.get('media_type'),
                    'url',
                    image_url['url'] if isinstance(image_url, dict) else image_url,
                ))
            elif kind == 'input_audio':
                audio = block['input_audio']
                media_type = {
                    'wav': 'audio/wav',
                    'mp3': 'audio/mpeg',
                }.get(audio['format'])
                if media_type is None:
                    raise NotImplementedError(
                        f"Unknown input_audio format: {audio['format']!r}"
                    )
                blocks.append(_attachment(
                    media_type,
                    'bytes',
                    base64.b64decode(audio['data']),
                ))
            else:
                raise NotImplementedError(
                    f"Unknown OpenAI-compatible response content type: {kind!r}"
                )
    for call in message.get('tool_calls') or []:
        function = call['function']
        blocks.append({
            'type': 'tool_call',
            'id': call['id'],
            'name': function['name'],
            'args': json.loads(function['arguments']),
        })
    for image in message.get('images') or []:
        kind = image['type']
        if kind != 'image_url':
            raise NotImplementedError(
                f"Unknown OpenAI-compatible image type: {kind!r}"
            )
        image_url = image['image_url']
        blocks.append(_attachment(
            image.get('media_type'),
            'url',
            image_url['url'] if isinstance(image_url, dict) else image_url,
        ))
    return blocks


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
            raise NotImplementedError(
                f"{api_type} transport does not support {media_type} attachments"
            )

    def _attachment_data(self, block):
        data_type = block['data_type']
        data = block['data']
        if data_type == 'bytes':
            if not isinstance(data, bytes):
                raise BadRequestError("Attachment bytes data must be bytes")
            return data
        if data_type == 'filepath':
            if not isinstance(data, str):
                raise BadRequestError("Attachment filepath data must be a string")
            with open(data, 'rb') as file:
                return file.read()
        if data_type in ('url', 'provider_id'):
            if not isinstance(data, str):
                raise BadRequestError(
                    f"Attachment {data_type} data must be a string"
                )
            return data
        raise NotImplementedError(
            f"Unknown attachment data type: {data_type!r}"
        )

    def _binary_attachment(self, block):
        if block['data_type'] not in ('bytes', 'filepath'):
            raise NotImplementedError(
                f"{self.model_config['api_type']} transport does not support "
                f"{block['data_type']} attachments"
            )
        self.validate_media_type(block.get('media_type'))
        return self._attachment_data(block)

    def _openai_attachment(self, block):
        if self.model_config['api_type'] == 'cursor':
            raise NotImplementedError(
                "Cursor transport does not support attachments"
            )
        data_type = block['data_type']
        media_type = block.get('media_type')
        if data_type == 'provider_id':
            return {
                'type': 'input_file',
                'file_id': self._attachment_data(block),
            }
        if data_type == 'url':
            if media_type is not None and not media_type.startswith('image/'):
                raise NotImplementedError(
                    "OpenAI-compatible transport only supports image URLs"
                )
            return {
                'type': 'image_url',
                'image_url': {'url': self._attachment_data(block)},
            }
        data = self._binary_attachment(block)
        encoded = base64.b64encode(data).decode()
        if media_type.startswith('image/'):
            return {
                'type': 'image_url',
                'image_url': {
                    'url': f"data:{media_type};base64,{encoded}",
                },
            }
        if media_type in AUDIO_MEDIA_TYPES:
            return {
                'type': 'input_audio',
                'input_audio': {
                    'data': encoded,
                    'format': 'wav' if media_type == 'audio/wav' else 'mp3',
                },
            }
        raise NotImplementedError(
            f"OpenAI-compatible transport does not support {media_type}"
        )

    def _responses_attachment(self, block):
        data_type = block['data_type']
        if data_type == 'provider_id':
            return {
                'type': 'input_file',
                'file_id': self._attachment_data(block),
            }
        if data_type == 'url':
            media_type = block.get('media_type')
            if media_type is not None and not media_type.startswith('image/'):
                raise NotImplementedError(
                    "Responses transport only supports image URLs"
                )
            return {
                'type': 'input_image',
                'image_url': self._attachment_data(block),
            }
        data = self._binary_attachment(block)
        media_type = block['media_type']
        if not media_type.startswith('image/'):
            raise NotImplementedError(
                f"Responses transport does not support {media_type}"
            )
        return {
            'type': 'input_image',
            'image_url': (
                f"data:{media_type};base64,"
                f"{base64.b64encode(data).decode()}"
            ),
        }

    def _anthropic_attachment(self, block):
        data = self._binary_attachment(block)
        media_type = block['media_type']
        if not media_type.startswith('image/'):
            raise NotImplementedError(
                f"Anthropic transport does not support {media_type}"
            )
        return {
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': media_type,
                'data': base64.b64encode(data).decode(),
            },
        }

    def _gemini_attachment(self, block):
        data = self._binary_attachment(block)
        return {
            'inlineData': {
                'mimeType': block['media_type'],
                'data': base64.b64encode(data).decode(),
            },
        }

    @staticmethod
    def _strip_response_media(message):
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

    def _completions_messages(self, messages):
        projected = []
        for original in messages:
            message = dict(original)
            role = message['role']
            blocks = message['content']
            if role == 'tool':
                output = []
                for block in blocks:
                    kind = block['type']
                    if kind == 'text':
                        output.append(block['text'])
                    else:
                        raise NotImplementedError(
                            f"Unknown tool result content type: {kind!r}"
                        )
                projected.append({
                    'role': 'tool',
                    'tool_call_id': message['tool_call_id'],
                    'content': '\n'.join(output),
                })
                continue
            content = []
            calls = []
            for block in blocks:
                kind = block['type']
                if kind in ('text', 'commentary'):
                    content.append({'type': 'text', 'text': block['text']})
                elif kind == 'attachment':
                    content.append(self._openai_attachment(block))
                elif kind == 'tool_call':
                    calls.append({
                        'id': block['id'],
                        'type': 'function',
                        'function': {
                            'name': block['name'],
                            'arguments': json.dumps(block['args']),
                        },
                    })
                elif kind == 'reasoning':
                    continue
                else:
                    raise NotImplementedError(
                        f"Unknown transport content type: {kind!r}"
                    )
            out = {
                'role': role,
                'content': (
                    content[0]['text']
                    if len(content) == 1 and content[0]['type'] == 'text'
                    else content
                ),
            }
            if calls:
                out['tool_calls'] = calls
            breakpoint = message.get('_prompt_cache_breakpoint')
            if (
                breakpoint
                and self.model_config.get('explicit_prompt_cache')
                and out['content']
            ):
                if isinstance(out['content'], str):
                    out['content'] = [{
                        'type': 'text',
                        'text': out['content'],
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }]
                else:
                    out['content'][-1] = {
                        **out['content'][-1],
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
            projected.append(out)
        return projected


    def _call_completions(self, messages, tools):
        """
        Call OpenAI Completions API.

        Args:
            messages: List of projected message dicts.
            tools: Optional tool specifications.
        """
        transport_messages = list(messages)
        context_window = self.model_config.get('context_window')
        extra_config = dict(self.model_config.get('config', {}))
        current_max_tokens = extra_config.get('max_tokens')
        max_tokens_retry = 0

        while True:
            req = {
                "model": self.model_config['model'],
                "messages": self._completions_messages(transport_messages),
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
                provider_message, stop_reason, usage = parser(response_json)
                message = {
                    'role': 'assistant',
                    'content': _openai_compatible_message_to_transport_blocks(provider_message),
                    'provider_metadata': {'stop_reason': stop_reason},
                }
                if self.tool_mode == "repl_execute":
                    message = repl_response_to_text(message)
                if usage:
                    self.usage_tracker.log(self.model_name, usage)
                    self._update_input_tokens_per_byte(self._current_input_bytes, usage)

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
                        transport_messages.append(message)
                        transport_messages.append({
                            'role': 'user',
                            'content': [{
                                'type': 'text',
                                'text': (
                                    'Incomplete response detected. '
                                    'Resubmit your response.'
                                ),
                            }],
                        })
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
            role = message['role']
            blocks = message['content']
            if role == 'tool':
                output = []
                for block in blocks:
                    kind = block['type']
                    if kind == 'text':
                        output.append(block['text'])
                    else:
                        raise NotImplementedError(
                            f"Unknown tool result content type: {kind!r}"
                        )
                input_items.append({
                    'type': 'function_call_output',
                    'call_id': message['tool_call_id'],
                    'output': '\n'.join(output),
                })
                continue

            message_items = []
            content = []

            def flush_content(phase=None):
                nonlocal content
                if content:
                    item = {'role': role, 'content': content}
                    if phase is not None:
                        item['phase'] = phase
                    message_items.append(item)
                    content = []

            for block in blocks:
                kind = block['type']
                if kind in ('text', 'commentary'):
                    phase = 'commentary' if kind == 'commentary' else None
                    if content and phase is not None:
                        flush_content()
                    content.append({
                        'type': (
                            'output_text'
                            if role == 'assistant'
                            else 'input_text'
                        ),
                        'text': block['text'],
                    })
                    if phase is not None:
                        flush_content(phase)
                elif kind == 'attachment':
                    content.append(self._responses_attachment(block))
                elif kind == 'reasoning':
                    flush_content()
                    metadata = block.get('provider_metadata') or {}
                    if 'encrypted_content' in metadata:
                        message_items.append({
                            'type': 'reasoning',
                            'encrypted_content': metadata['encrypted_content'],
                            'summary': [],
                        })
                elif kind == 'tool_call':
                    flush_content()
                    message_items.append({
                        'type': 'function_call',
                        'call_id': block['id'],
                        'name': block['name'],
                        'arguments': json.dumps(block['args']),
                    })
                else:
                    raise NotImplementedError(
                        f"Unknown transport content type: {kind!r}"
                    )
            flush_content()
            if (
                message.get('_prompt_cache_breakpoint')
                and self.model_config.get('explicit_prompt_cache')
            ):
                message_items[-1]['content'][-1] = {
                    **message_items[-1]['content'][-1],
                    'prompt_cache_breakpoint': {'mode': 'explicit'},
                }
                has_cache_breakpoint = True
            input_items.extend(message_items)
        response_tools = [
            {'type': 'function', **tool.get('function', tool)}
            for tool in (
                [REPL_EXECUTE_TOOL]
                if self.tool_mode == 'repl_execute'
                else tools or []
            )
        ]
        req = {'model': self.model_config['model'], 'input': input_items, **config}
        if has_cache_breakpoint:
            req.setdefault('prompt_cache_options', {'mode': 'explicit'})
        if response_tools:
            req['tools'] = response_tools
        return req

    def _parse_responses_result(self, response_json):
        output = response_json['output']
        blocks = []
        for item in output:
            kind = item['type']
            if kind == 'message':
                phase = item.get('phase')
                block_type = 'text' if phase in (None, 'final') else 'commentary'
                if phase not in (None, 'final', 'commentary'):
                    sys.stderr.write(
                        f"Warning: unrecognized Responses API message phase: {phase!r}\n"
                    )
                for content in item['content']:
                    content_kind = content['type']
                    if content_kind in ('output_text', 'text'):
                        blocks.append({
                            'type': block_type,
                            'text': content['text'],
                        })
                    elif content_kind == 'input_file':
                        blocks.append(_attachment(
                            content.get('media_type'),
                            'provider_id',
                            content['file_id'],
                        ))
                    elif content_kind in ('input_image', 'image_url'):
                        image_url = content.get(
                            'image_url', content.get('url')
                        )
                        blocks.append(_attachment(
                            content.get('media_type'),
                            'url',
                            image_url,
                        ))
                    else:
                        raise NotImplementedError(
                            "Unknown Responses message content type: "
                            f"{content_kind!r}"
                        )
            elif kind == 'output_text':
                blocks.append({'type': 'text', 'text': item['text']})
            elif kind == 'reasoning':
                summary = []
                for part in item.get('summary', []):
                    part_kind = part['type']
                    if part_kind in ('summary_text', 'text'):
                        summary.append(part['text'])
                    else:
                        raise NotImplementedError(
                            "Unknown Responses reasoning summary type: "
                            f"{part_kind!r}"
                        )
                block = {
                    'type': 'reasoning',
                    'text': '\n'.join(summary),
                }
                if 'encrypted_content' in item:
                    block['provider_metadata'] = {
                        'encrypted_content': item['encrypted_content'],
                    }
                blocks.append(block)
            elif kind == 'function_call':
                blocks.append({
                    'type': 'tool_call',
                    'id': item.get('call_id') or item['id'],
                    'name': item['name'],
                    'args': json.loads(item['arguments']),
                })
            elif kind == 'input_file':
                blocks.append(_attachment(
                    item.get('media_type'),
                    'provider_id',
                    item['file_id'],
                ))
            elif kind in ('input_image', 'image_url'):
                blocks.append(_attachment(
                    item.get('media_type'),
                    'url',
                    item.get('image_url', item.get('url')),
                ))
            else:
                raise NotImplementedError(
                    f"Unknown Responses output type: {kind!r}"
                )
        if not blocks and isinstance(response_json.get('output_text'), str):
            blocks.append({'type': 'text', 'text': response_json['output_text']})
        if usage := response_json.get('usage'):
            self.usage_tracker.log(self.model_name, usage)
            self._update_input_tokens_per_byte(self._current_input_bytes, usage)
        incomplete = response_json.get('incomplete_details') or {}
        stop_reason = incomplete.get('reason')
        if stop_reason is None:
            stop_reason = (
                'tool_calls' if any(block['type'] == 'tool_call' for block in blocks)
                else 'stop' if response_json.get('status') == 'completed'
                else response_json.get('status')
            )
        message = {
            'role': 'assistant',
            'content': blocks,
            'provider_metadata': {'stop_reason': stop_reason},
        }
        if self.tool_mode == 'repl_execute':
            message = repl_response_to_text(message)
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
        messages = self._completions_messages(messages)
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
        provider_message, stop_reason, usage = _parse_completions_response(response_json)
        message = {
            'role': 'assistant',
            'content': _openai_compatible_message_to_transport_blocks(provider_message),
            'provider_metadata': {'stop_reason': stop_reason},
        }
        message = repl_response_to_text(message)
        if usage:
            self.usage_tracker.log(self.model_name, usage)
            self._update_input_tokens_per_byte(self._current_input_bytes, usage)
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
        system_message = None
        projected = []
        for message in messages:
            role = message['role']
            blocks = message['content']
            if role == 'system':
                text = _text_only_content(blocks, 'Anthropic system')
                if system_message is None:
                    system_message = text
                else:
                    projected.append({'role': 'user', 'content': text})
                continue
            content = []
            if role == 'tool':
                content.append({
                    'type': 'tool_result',
                    'tool_use_id': message['tool_call_id'],
                    'content': _text_only_content(
                        blocks, 'Anthropic tool result'
                    ),
                })
                role = 'user'
            else:
                for block in blocks:
                    kind = block['type']
                    if kind in ('text', 'commentary'):
                        content.append({'type': 'text', 'text': block['text']})
                    elif kind == 'reasoning':
                        item = {
                            'type': 'thinking',
                            'thinking': block['text'],
                        }
                        metadata = block.get('provider_metadata') or {}
                        if 'signature' in metadata:
                            item['signature'] = metadata['signature']
                        content.append(item)
                    elif kind == 'tool_call':
                        content.append({
                            'type': 'tool_use',
                            'id': block['id'],
                            'name': block['name'],
                            'input': block['args'],
                        })
                    elif kind == 'attachment':
                        content.append(self._anthropic_attachment(block))
                    else:
                        raise NotImplementedError(
                            f"Unknown transport content type: {kind!r}"
                        )
            projected.append({'role': role, 'content': content})
        req = {
            "model": self.model_config['model'],
            "messages": projected,
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
            blocks = []
            for block in response_json['content']:
                kind = block['type']
                if kind == 'text':
                    blocks.append({'type': 'text', 'text': block['text']})
                elif kind in ('thinking', 'reasoning'):
                    reasoning = {'type': 'reasoning', 'text': block['thinking']}
                    if 'signature' in block:
                        reasoning['provider_metadata'] = {
                            'signature': block['signature'],
                        }
                    blocks.append(reasoning)
                elif kind == 'tool_use':
                    blocks.append({
                        'type': 'tool_call',
                        'id': block['id'],
                        'name': block['name'],
                        'args': block['input'],
                    })
                elif kind == 'image':
                    source = block['source']
                    source_type = source['type']
                    if source_type == 'base64':
                        blocks.append(_attachment(
                            source['media_type'],
                            'bytes',
                            base64.b64decode(source['data']),
                        ))
                    elif source_type == 'url':
                        blocks.append(_attachment(
                            source.get('media_type'),
                            'url',
                            source['url'],
                        ))
                    else:
                        raise NotImplementedError(
                            "Unknown Anthropic image source type: "
                            f"{source_type!r}"
                        )
                else:
                    raise NotImplementedError(
                        f"Unknown Anthropic response content type: {kind!r}"
                    )
            return {
                'role': 'assistant',
                'content': blocks,
                'provider_metadata': {
                    'stop_reason': response_json['stop_reason'],
                },
            }
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
        for message in messages:
            role = message['role']
            blocks = message['content']
            parts = []
            if role == 'system':
                system_parts.extend(
                    {'text': text}
                    for text in [_text_only_content(
                        blocks, 'Gemini system'
                    )]
                    if text
                )
                continue
            if role == 'tool':
                parts.append({
                    'functionResponse': {
                        'id': message['tool_call_id'],
                        'name': message['name'],
                        'response': {
                            'output': _text_only_content(
                                blocks, 'Gemini tool result'
                            ),
                        },
                    },
                })
                role = 'user'
            else:
                for block in blocks:
                    kind = block['type']
                    if kind in ('text', 'commentary'):
                        parts.append({'text': block['text']})
                    elif kind == 'reasoning':
                        part = {'text': block['text'], 'thought': True}
                        metadata = block.get('provider_metadata') or {}
                        if 'thought_signature' in metadata:
                            part['thoughtSignature'] = metadata[
                                'thought_signature'
                            ]
                        parts.append(part)
                    elif kind == 'tool_call':
                        parts.append({
                            'functionCall': {
                                'id': block['id'],
                                'name': block['name'],
                                'args': block['args'],
                            },
                        })
                    elif kind == 'attachment':
                        parts.append(self._gemini_attachment(block))
                    else:
                        raise NotImplementedError(
                            f"Unknown transport content type: {kind!r}"
                        )
            contents.append({
                'role': 'model' if role == 'assistant' else 'user',
                'parts': parts,
            })
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
            blocks = []
            for part in candidate['content']['parts']:
                if 'functionCall' in part:
                    call = part['functionCall']
                    blocks.append({
                        'type': 'tool_call',
                        'id': call['id'],
                        'name': call['name'],
                        'args': call['args'],
                    })
                elif 'inlineData' in part:
                    inline = part['inlineData']
                    blocks.append(_attachment(
                        inline['mimeType'],
                        'bytes',
                        base64.b64decode(inline['data']),
                    ))
                elif 'fileData' in part:
                    file_data = part['fileData']
                    blocks.append(_attachment(
                        file_data.get('mimeType'),
                        'url',
                        file_data['fileUri'],
                    ))
                elif part.get('thought'):
                    block = {'type': 'reasoning', 'text': part['text']}
                    if 'thoughtSignature' in part:
                        block['provider_metadata'] = {
                            'thought_signature': part['thoughtSignature'],
                        }
                    blocks.append(block)
                elif 'text' in part:
                    blocks.append({'type': 'text', 'text': part['text']})
                else:
                    raise NotImplementedError(
                        "Unknown Gemini response part type: "
                        f"{tuple(part)!r}"
                    )
            return {
                'role': 'assistant',
                'content': blocks,
                'provider_metadata': {
                    'stop_reason': candidate['finishReason'],
                },
            }
        finally:
            conn.close()


    def _call(self, messages, tools=None):
        if self.tool_mode == "repl_execute":
            messages = project_repl_tool_history(messages)
        else:
            messages = _apply_native_policy(messages, self.native)
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

    def call(self, messages, retry=3, attempt=0):
        try:
            context = (
                self.provider_admission.admitted()
                if self.provider_admission is not None
                else contextlib.nullcontext()
            )
            with context:
                return self._call(messages)
        except (ContextOverflowError, ReplExecuteResponseError):
            raise
        except Exception as e:
            err = (str(e) if len(str(e)) < 1000 else str(e)[:1000]+'...').replace("\n"," ")
            logger.error(f"call {type(e).__name__}: {err}", exc_info=True)
            if retry:
                self._sleep_backoff(attempt)
                return self.call(messages, retry-1, attempt+1)
            raise



    def conversation(self, system_prompt):
        return Conversation(self, system_prompt)
