import json
import socket
import logging
import os
import stat
import subprocess
import sys
import threading
import uuid
from io import BytesIO

import pytest

from code_agent import codex
from code_agent.client import LLMClient
from code_agent.repl_tool_adapter import REPL_EXECUTE_TOOL
from code_agent.client import legacy_to_transport_messages, transport_to_legacy_message


def _jwt(payload):
    import base64
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).decode().rstrip("=")
    return f"x.{encoded}.x"


def _credential(
    *,
    access_token=None,
    refresh_token="refresh",
    account_id="account-from-token",
    rate_limits=None,
):
    credential = {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": access_token or _jwt({
                "exp": 4_000_000_000,
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": account_id
                },
            }),
            "id_token": "id",
            "refresh_token": refresh_token,
            "account_id": account_id,
        },
        "last_refresh": "2026-01-01T00:00:00+00:00",
    }
    if rate_limits is not None:
        credential["rate_limits"] = rate_limits
    return credential


def _auth_file(path, credentials=None):
    path.write_text(json.dumps({
        "credentials": credentials or [_credential()],
    }))
    return path


def test_codex_registry_does_not_require_api_key(tmp_path):
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PYTHONPATH": os.getcwd(),
    }
    env.pop("CODEX_API_KEY", None)
    code = """
import json
from code_agent.llm_registry import get_model_config
print(json.dumps(get_model_config("codex/gpt-5.6-luna-xhigh"), default=str))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    config = json.loads(result.stdout)
    assert config["api_type"] == "codex"
    assert config["tool_mode"] == "repl_execute"
    assert config["api_key"] is None
    assert "prompt_cache_key" not in config.get("config", {})


def test_auth_save_is_private_and_atomic(tmp_path):
    path = _auth_file(tmp_path / "auth.json")
    auth = codex.CodexAuth(str(path))
    auth.data["tokens"]["access_token"] = "new"
    auth._save()
    assert json.loads(path.read_text())["credentials"][0]["tokens"]["access_token"] == "new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not (tmp_path / "auth.json.tmp").exists()


def test_auth_read_blocks_on_exclusive_lock(tmp_path):
    path = _auth_file(tmp_path / "auth.json")
    lock_fd = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    codex.fcntl.flock(lock_fd, codex.fcntl.LOCK_EX)
    started = threading.Event()
    finished = threading.Event()
    errors = []

    def load():
        started.set()
        try:
            codex.CodexAuth(str(path))
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=load)
    thread.start()
    assert started.wait(1)
    assert not finished.wait(0.1)

    codex.fcntl.flock(lock_fd, codex.fcntl.LOCK_UN)
    os.close(lock_fd)
    thread.join(1)

    assert finished.is_set()
    assert errors == []


def test_auth_write_blocks_on_shared_lock(tmp_path):
    path = _auth_file(tmp_path / "auth.json")
    auth = codex.CodexAuth(str(path))
    auth.data["tokens"]["access_token"] = "new"
    lock_fd = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    codex.fcntl.flock(lock_fd, codex.fcntl.LOCK_SH)
    started = threading.Event()
    finished = threading.Event()
    errors = []

    def save():
        started.set()
        try:
            auth._save()
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=save)
    thread.start()
    assert started.wait(1)
    assert not finished.wait(0.1)

    codex.fcntl.flock(lock_fd, codex.fcntl.LOCK_UN)
    os.close(lock_fd)
    thread.join(1)

    assert finished.is_set()
    assert errors == []
    assert json.loads(path.read_text())["credentials"][0]["tokens"]["access_token"] == "new"


def test_concurrent_refresh_uses_updated_credentials(monkeypatch, tmp_path):
    path = _auth_file(tmp_path / "auth.json")
    first = codex.CodexAuth(str(path))
    second = codex.CodexAuth(str(path))
    barrier = threading.Barrier(2)
    refresh_requests = []
    errors = []

    class RefreshResponse(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def urlopen(request, timeout=None):
        refresh_requests.append(json.loads(request.data))
        return RefreshResponse(json.dumps({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
        }).encode())

    def refresh(auth):
        try:
            barrier.wait()
            auth.refresh()
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(codex.urllib.request, "urlopen", urlopen)
    threads = [
        threading.Thread(target=refresh, args=(first,)),
        threading.Thread(target=refresh, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(refresh_requests) == 1
    assert refresh_requests[0]["refresh_token"] == "refresh"
    assert first.access_token == "new-access"
    assert second.access_token == "new-access"
    saved = json.loads(path.read_text())
    assert saved["credentials"][0]["tokens"]["access_token"] == "new-access"
    assert saved["credentials"][0]["tokens"]["refresh_token"] == "new-refresh"


def test_parse_sse_supports_multiline_and_usage():
    stream = BytesIO(
        b'event: message\n'
        b'data: {"type":"response.output_text.delta",\n'
        b'data: "delta":"hello"}\n\n'
        b'data: {"type":"response.completed","response":{"usage":'
        b'{"input_tokens":12,"output_tokens":3,'
        b'"input_tokens_details":{"cached_tokens":4}}}}\n\n'
        b'data: [DONE]\n\n'
    )
    response = codex.parse_sse(stream)
    assert response["output"] == [{
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hello"}],
    }]
    assert response["usage"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        "input_tokens_details": {"cached_tokens": 4},
    }
    assert response["status"] == "completed"


def test_parse_sse_reconstructs_function_call_arguments_by_item_id():
    events = [
        {
            "type": "response.output_item.added",
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "repl_execute",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": '{"code": "emit(\'ok\', release=True)"}',
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "repl_execute",
                    "arguments": "",
                }],
            },
        },
    ]
    payload = b"".join(
        f"data: {json.dumps(event)}\n\n".encode() for event in events
    ) + b"data: [DONE]\n\n"
    response = codex.parse_sse(BytesIO(payload))
    assert response["output"][0]["arguments"] == '{"code": "emit(\'ok\', release=True)"}'


def test_parse_response_body_accepts_json_without_content_type():
    assert codex._parse_response_body("", b'{"status":"completed","output":[]}') == {
        "status": "completed",
        "output": [],
    }


def test_parse_response_body_accepts_sse_without_content_type():
    payload = b'data: {\"type\":\"response.completed\",\"response\":{\"output\":[]}}\n\n'
    assert codex._parse_response_body("", payload)["output"] == []


def test_responses_adds_transport_defaults(monkeypatch, tmp_path):
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json")))
    captured = {}
    native_response = {
        "id": "response-1",
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }

    def request(auth_arg, body, timeouts):
        captured.update(auth=auth_arg, body=body, timeouts=timeouts)
        return native_response

    monkeypatch.setattr(codex, "_request", request)
    request_body = {
        "model": "gpt-5.6-luna",
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "hello"},
        ]}],
        "tools": [{
            "type": "function",
            "name": "repl_execute",
            "description": "execute",
            "parameters": {"type": "object"},
        }],
    }
    response = codex.responses(request_body, auth=auth, timeout=17)

    assert response is native_response
    assert captured["timeouts"] == codex.StreamTimeouts(
        first_byte=17,
        thinking_idle=codex.DEFAULT_THINKING_IDLE_TIMEOUT,
        answering_idle=codex.DEFAULT_ANSWERING_IDLE_TIMEOUT,
    )
    assert captured["body"]["input"] == request_body["input"]
    assert captured["body"]["instructions"] == ""
    assert captured["body"]["tools"] == request_body["tools"]
    assert captured["body"]["stream"] is True
    assert captured["body"]["store"] is False
    assert captured["body"]["tool_choice"] == "auto"
    assert captured["body"]["parallel_tool_calls"] is True
    assert captured["body"]["prompt_cache_key"] == codex.SESSION_ID
    assert captured["body"]["include"] == ["reasoning.encrypted_content"]


def test_responses_maps_system_to_developer_without_mutating_request(monkeypatch, tmp_path):
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json")))
    captured = {}

    def request(auth_arg, body, timeouts):
        captured["body"] = body
        return {"status": "completed", "output": []}

    monkeypatch.setattr(codex, "_request", request)
    request_body = {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "contract"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "task"}],
            },
        ],
    }

    codex.responses(request_body, auth=auth)

    assert captured["body"]["instructions"] == ""
    assert [item["role"] for item in captured["body"]["input"]] == [
        "developer",
        "user",
    ]
    assert request_body["input"][0]["role"] == "system"


def test_headers_reuse_process_session_id(tmp_path):
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json")))

    first = codex._headers(auth)
    second = codex._headers(auth)

    assert first["Session_id"] == codex.SESSION_ID
    assert second["Session_id"] == codex.SESSION_ID
    assert uuid.UUID(codex.SESSION_ID).version == 7


def test_legacy_input_file_survives_responses_projection(monkeypatch):
    config = {
        "provider": "openai",
        "model": "gpt-test",
        "api_key": "key",
        "api_type": "responses",
        "tool_mode": None,
        "tools": True,
        "concurrency": 1,
        "timeout": 20,
        "config": {},
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("openai/gpt-test")

    transport = legacy_to_transport_messages([{
        "role": "user",
        "content": [
            {"type": "input_file", "file_id": "file_123"},
            {"type": "text", "text": "Use this file."},
        ],
    }])
    request = client._responses_request(transport, None)

    assert transport[0]["content"] == [
        {
            "type": "attachment",
            "media_type": None,
            "data_type": "provider_id",
            "data": "file_123",
        },
        {"type": "text", "text": "Use this file."},
    ]
    assert [block["type"] for block in request["input"][0]["content"]] == [
        "input_file", "input_text",
    ]
    assert request["input"][0]["content"][0]["file_id"] == "file_123"


def test_responses_request_adds_explicit_cache_breakpoints(monkeypatch):
    config = {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "api_key": "key",
        "api_type": "responses",
        "tool_mode": "repl_execute",
        "tools": True,
        "concurrency": 1,
        "timeout": 20,
        "port": 443,
        "host": "api.openai.com",
        "path": "/v1/responses",
        "config": {"prompt_cache_key": "tenant:acme:knowledge-base-v1"},
        "explicit_prompt_cache": True,
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("openai/gpt-5.6-luna")
    request = client._responses_request([
        {
            "role": "system",
            "content": [{"type": "text", "text": "stable instructions"}],
            "_prompt_cache_breakpoint": True,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "attachment",
                    "media_type": None,
                    "data_type": "provider_id",
                    "data": "file_123",
                },
                {"type": "text", "text": "Answer the current question."},
            ],
            "_prompt_cache_breakpoint": True,
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "uncached"}],
        },
    ], None)

    assert request["prompt_cache_key"] == "tenant:acme:knowledge-base-v1"
    assert request["prompt_cache_options"] == {"mode": "explicit"}
    assert request["input"][0]["content"][-1]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert request["input"][1]["content"][-1]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert request["input"][1]["content"][0]["type"] == "input_file"
    assert "prompt_cache_breakpoint" not in request["input"][1]["content"][0]
    assert "prompt_cache_breakpoint" not in request["input"][2]["content"][0]


def test_codex_strips_unsupported_prompt_cache_fields(monkeypatch, tmp_path):
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json")))
    captured = {}

    def request(auth_arg, body, timeouts):
        captured.update(body=body)
        return {"status": "completed", "output": []}

    monkeypatch.setattr(codex, "_request", request)
    request_body = {
        "model": "gpt-5.6-luna",
        "prompt_cache_key": "tenant:acme:knowledge-base-v1",
        "prompt_cache_options": {"mode": "explicit"},
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": "stable",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }],
        }],
    }
    codex.responses(request_body, auth=auth)

    assert captured["body"]["prompt_cache_key"] == "tenant:acme:knowledge-base-v1"
    assert "prompt_cache_options" not in captured["body"]
    assert "prompt_cache_breakpoint" not in captured["body"]["input"][0]["content"][0]
    assert request_body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}


def test_responses_order_survives_to_lossy_legacy_boundary(monkeypatch):
    config = {
        "provider": "openai",
        "model": "gpt-test",
        "api_key": "key",
        "api_type": "responses",
        "tool_mode": None,
        "tools": True,
        "concurrency": 1,
        "timeout": 20,
        "config": {},
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("openai/gpt-test")

    response = client._parse_responses_result({
        "status": "completed",
        "output": [
            {
                "type": "message",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "first"}],
            },
            {
                "type": "function_call",
                "call_id": "call",
                "name": "read",
                "arguments": '{"file_path": "app.py"}',
            },
            {
                "type": "message",
                "phase": "final",
                "content": [{"type": "output_text", "text": "last"}],
            },
        ],
    })

    assert [block["type"] for block in response["content"]] == [
        "commentary", "tool_call", "text",
    ]
    assert response["content"][1]["args"] == {"file_path": "app.py"}

    legacy = transport_to_legacy_message(response)
    assert legacy["content"].startswith("# first")
    assert legacy["content"].endswith("last")
    assert legacy["tool_calls"][0]["function"]["name"] == "read"



def test_non_native_history_preserves_legacy_tool_policy(monkeypatch):
    config = {
        "provider": "codex",
        "model": "gpt-test",
        "api_key": None,
        "api_type": "codex",
        "tool_mode": None,
        "tools": True,
        "concurrency": 1,
        "timeout": 20,
        "config": {},
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("codex/gpt-test", native=False)
    captured = {}

    def call(messages, tools):
        captured["messages"] = messages
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
        }

    client._call_codex = call
    client._call([
        {
            "role": "assistant",
            "content": "working",
            "tool_calls": [{
                "id": "call",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "name": "read",
            "tool_call_id": "call",
            "content": "file contents",
            "_private": "discard",
        },
    ])

    assistant, result = captured["messages"]
    assert assistant["content"] == [{"type": "text", "text": "working"}]
    assert result == {
        "role": "user",
        "content": [{"type": "text", "text": "read: file contents"}],
    }


def test_codex_client_adapter_without_repl_tool_mode(monkeypatch):
    config = {
        "provider": "codex",
        "model": "gpt-5.6-luna",
        "api_key": None,
        "api_type": "codex",
        "tool_mode": None,
        "rpm": 17,
        "tools": True,
        "concurrency": 1,
        "timeout": 20,
        "config": {},
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    captured = {}

    def complete(body, auth=None, **kwargs):
        captured.update(body=body, auth=auth, kwargs=kwargs)
        return {
            "id": "response-1",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }],
        }

    monkeypatch.setattr(codex, "responses", complete)
    client = LLMClient("codex/gpt-5.6-luna")
    result = client._call([{"role": "user", "content": "hello"}])
    assert captured["body"]["input"] == [{
        "role": "user",
        "content": [{"type": "input_text", "text": "hello"}],
    }]
    assert "tools" not in captured["body"]
    assert "timeout" not in captured["kwargs"]
    assert result == {
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "provider_metadata": {"stop_reason": "stop"},
    }


def test_codex_client_adapter(monkeypatch, tmp_path):
    config = {
        "provider": "codex",
        "model": "gpt-5.6-luna",
        "api_key": None,
        "api_type": "codex",
        "tool_mode": "repl_execute",
        "rpm": 17,
        "tools": True,
        "concurrency": 1,
        "timeout": 20,
        "config": {},
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    monkeypatch.setattr("code_agent.utils.get_model_config", lambda name: config)
    monkeypatch.setattr(
        codex, "CodexAuth",
        lambda path=codex.CRED_FILE: object(),
    )
    captured = {}

    def complete(body, auth=None, **kwargs):
        captured.update(body=body, auth=auth, kwargs=kwargs)
        return {
            "id": "response-1",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "call_id": "call-1",
                "name": "repl_execute",
                "arguments": json.dumps({"code": "emit('ok', release=True)"}),
            }],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }

    monkeypatch.setattr(codex, "responses", complete)
    client = LLMClient("codex/gpt-5.6-luna-xhigh")
    result = client._call([{"role": "user", "content": "hello"}])
    assert captured["body"]["model"] == "gpt-5.6-luna"
    assert captured["body"]["tools"] == [{
        "type": "function",
        **REPL_EXECUTE_TOOL["function"],
    }]
    assert "timeout" not in captured["kwargs"]
    assert result["content"] == [{
        "type": "text",
        "text": "emit('ok', release=True)",
    }]
    assert result["provider_metadata"]["stop_reason"] == "tool_calls"


def test_chat_completions_stream_reassembles_tool_call_arguments():
    from code_agent.streaming import reassemble_chat_completions_stream

    events = [
        {
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "repl_",
                            "arguments": '{"code":"print',
                        },
                    }],
                },
            }],
        },
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "name": "execute",
                            "arguments": '(1)"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        },
    ]
    body = "".join(
        f"data: {json.dumps(event)}\n\n" for event in events
    ) + "data: [DONE]\n\n"

    response = reassemble_chat_completions_stream(body)

    assert response["choices"][0]["message"]["tool_calls"] == [{
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "repl_execute",
            "arguments": '{"code":"print(1)"}',
        },
    }]


class _FakeHeaders(dict):
    def get(self, key, default=None):
        for candidate, value in self.items():
            if candidate.lower() == key.lower():
                return value
        return default


class _FakeResponse:
    def __init__(self, payload: bytes, content_type: str = "text/event-stream"):
        self.headers = _FakeHeaders({"Content-Type": content_type} if content_type else {})
        self._stream = BytesIO(payload)
        self.status = 200

    def read(self, size=-1):
        return self._stream.read(size)

    def readline(self, size=-1):
        return self._stream.readline(size)

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_parse_sse_logs_events_as_they_arrive(caplog):
    events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_text.delta", "delta": "hi"},
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                }],
            },
        },
    ]
    payload = b"".join(
        f"data: {json.dumps(event)}\n\n".encode() for event in events
    ) + b"data: [DONE]\n\n"

    with caplog.at_level(logging.INFO, logger="code_agent"):
        response = codex.parse_sse(BytesIO(payload))

    assert response["output"][0]["content"][0]["text"] == "hi"
    messages = [record.getMessage() for record in caplog.records]
    assert "---------- FROM LLM ----------" in messages
    assert json.dumps(events[0], separators=(",", ":")) in messages
    assert json.dumps(events[1], separators=(",", ":")) in messages
    assert json.dumps(events[2], separators=(",", ":")) in messages


def test_request_logs_to_llm_and_streams_sse(monkeypatch, tmp_path, caplog):
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json")))
    events = [
        {"type": "response.output_text.delta", "delta": "ok"},
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }],
            },
        },
    ]
    payload = b"".join(
        f"data: {json.dumps(event)}\n\n".encode() for event in events
    ) + b"data: [DONE]\n\n"
    fake = _FakeResponse(payload, content_type="text/event-stream")

    def fake_urlopen(request, timeout=None):
        assert timeout == 17
        assert request.full_url == codex.RESPONSES_URL
        fake.request = request
        return fake

    monkeypatch.setattr(codex.urllib.request, "urlopen", fake_urlopen)

    body = {
        "model": "gpt-5.6-luna",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "stream": True,
    }
    with caplog.at_level(logging.DEBUG, logger="code_agent"):
        response = codex._request(
            auth,
            body,
            codex.StreamTimeouts(
                first_byte=17,
                thinking_idle=codex.DEFAULT_THINKING_IDLE_TIMEOUT,
                answering_idle=codex.DEFAULT_ANSWERING_IDLE_TIMEOUT,
            ),
        )

    assert response["output"][0]["content"][0]["text"] == "ok"
    messages = [record.getMessage() for record in caplog.records]
    assert "----------- TO LLM -----------" in messages
    assert any(msg.startswith(f"POST {codex.RESPONSES_URL} ") for msg in messages)
    assert json.dumps(body, separators=(",", ":")) in messages
    assert "---------- FROM LLM ----------" in messages
    assert json.dumps(events[0], separators=(",", ":")) in messages
    assert json.dumps(events[1], separators=(",", ":")) in messages


def test_parse_http_response_streams_sse_without_content_type():
    payload = (
        b'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n'
        b"data: [DONE]\n\n"
    )
    response = codex._parse_http_response(_FakeResponse(payload, content_type=""))
    assert response["status"] == "completed"
    assert response["output"] == []


def test_http_rate_limit_headers_are_saved_before_stream_read(tmp_path):
    path = _auth_file(tmp_path / "auth.json")
    auth = codex.CodexAuth(str(path))
    payload = (
        b'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n'
        b"data: [DONE]\n\n"
    )

    class RateLimitResponse(_FakeResponse):
        def __init__(self):
            super().__init__(payload)
            self.headers.update({
                "x-codex-primary-used-percent": "12.5",
                "x-codex-primary-window-minutes": "300",
                "x-codex-primary-reset-at": "1704069000",
                "x-codex-secondary-used-percent": "30",
                "x-codex-secondary-window-minutes": "10080",
                "x-codex-secondary-reset-at": "1704673800",
                "x-codex-credits-has-credits": "true",
                "x-codex-credits-unlimited": "false",
                "x-codex-credits-balance": "4.25",
            })

        def readline(self, size=-1):
            saved = json.loads(path.read_text())
            assert saved["credentials"][0]["rate_limits"]["limits"]["codex_primary"][
                "used_percent"
            ] == 12.5
            return super().readline(size)

    codex._parse_http_response(RateLimitResponse(), auth=auth)

    saved = json.loads(path.read_text())["credentials"][0]["rate_limits"]
    assert isinstance(saved["fetched_at"], int)
    assert saved["limits"] == {
        "codex_primary": {
            "used_percent": 12.5,
            "reset_at": 1704069000,
        },
        "codex_secondary": {
            "used_percent": 30.0,
            "reset_at": 1704673800,
        },
    }


def test_sse_rate_limit_event_is_saved_immediately(tmp_path):
    path = _auth_file(tmp_path / "auth.json")
    auth = codex.CodexAuth(str(path))
    rate_limits = {
        "type": "codex.rate_limits",
        "plan_type": "plus",
        "rate_limits": {
            "primary": {
                "used_percent": 18.0,
                "window_minutes": 300,
                "reset_at": 1704069000,
            },
        },
    }
    payload = (
        f"data: {json.dumps(rate_limits)}\n\n"
        'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n'
        "data: [DONE]\n\n"
    ).encode()

    codex._parse_http_response(_FakeResponse(payload), auth=auth)

    saved = json.loads(path.read_text())["credentials"][0]["rate_limits"]
    assert isinstance(saved["fetched_at"], int)
    assert saved["limits"] == {
        "codex_primary": {
            "used_percent": 18.0,
            "reset_at": 1704069000,
        },
    }


def test_saving_rate_limits_preserves_concurrently_refreshed_tokens(tmp_path):
    path = _auth_file(tmp_path / "auth.json")
    stale = codex.CodexAuth(str(path))
    current = codex.CodexAuth(str(path))
    current.data["tokens"]["access_token"] = "new-access"
    current.data["tokens"]["refresh_token"] = "new-refresh"
    current._save()

    stale.save_rate_limits({
        "type": "codex.rate_limits",
        "rate_limits": {
            "primary": {"used_percent": 25, "reset_at": 1704069000},
        },
    })

    saved = json.loads(path.read_text())
    assert saved["credentials"][0]["tokens"]["access_token"] == "new-access"
    assert saved["credentials"][0]["tokens"]["refresh_token"] == "new-refresh"
    assert saved["credentials"][0]["rate_limits"]["limits"] == {
        "codex_primary": {"used_percent": 25.0, "reset_at": 1704069000},
    }



class _FakeSock:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)

    def recv(self, size=0):
        return b""


class _SocketBackedStream:
    """Yield SSE lines for the given events, then raise socket.timeout."""

    def __init__(self, events, sock, timeout_exc=None):
        payload = b"".join(
            f"data: {json.dumps(event)}\n\n".encode() for event in events
        )
        self._lines = payload.splitlines(keepends=True)
        self._index = 0
        self._timeout_exc = timeout_exc or socket.timeout("timed out")
        self._sock = sock

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._lines):
            raise self._timeout_exc
        line = self._lines[self._index]
        self._index += 1
        return line


def test_normalize_stream_timeouts_defaults_and_overrides():
    defaults = codex._normalize_stream_timeouts()
    assert defaults == codex.StreamTimeouts(
        first_byte=codex.DEFAULT_FIRST_BYTE_TIMEOUT,
        thinking_idle=codex.DEFAULT_THINKING_IDLE_TIMEOUT,
        answering_idle=codex.DEFAULT_ANSWERING_IDLE_TIMEOUT,
    )
    assert defaults.first_byte == 60.0
    assert defaults.thinking_idle == 30.0
    assert defaults.answering_idle == 30.0

    legacy = codex._normalize_stream_timeouts(timeout=17)
    assert legacy.first_byte == 17.0
    assert legacy.thinking_idle == 30.0
    assert legacy.answering_idle == 30.0

    custom = codex._normalize_stream_timeouts(
        first_byte_timeout=12,
        thinking_idle_timeout=34,
        answering_idle_timeout=56,
    )
    assert custom == codex.StreamTimeouts(first_byte=12, thinking_idle=34, answering_idle=56)


def test_is_answering_event_distinguishes_message_and_tool_from_reasoning():
    assert codex._is_answering_event({"type": "response.output_text.delta", "delta": "x"})
    assert codex._is_answering_event({
        "type": "response.output_item.added",
        "item": {"type": "message"},
    })
    assert codex._is_answering_event({
        "type": "response.output_item.added",
        "item": {"type": "function_call", "name": "echo"},
    })
    assert not codex._is_answering_event({
        "type": "response.output_item.added",
        "item": {"type": "reasoning"},
    })
    assert not codex._is_answering_event({"type": "response.created"})


def test_parse_sse_stalls_during_thinking_with_stage_timeouts(monkeypatch):
    sock = _FakeSock()
    events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {
            "type": "response.output_item.added",
            "item": {"id": "rs_1", "type": "reasoning", "summary": []},
        },
    ]
    stream = _SocketBackedStream(events, sock)
    monkeypatch.setattr(codex, "_response_socket", lambda s: sock)

    with pytest.raises(codex.CodexStallError) as exc_info:
        codex.parse_sse(
            stream,
            timeouts=codex.StreamTimeouts(first_byte=60, thinking_idle=30, answering_idle=30),
        )

    err = exc_info.value
    assert err.stage == "thinking"
    assert err.idle_timeout == 30
    assert err.last_event_type == "response.output_item.added"
    assert "stalled during thinking" in str(err)
    assert sock.timeouts[0] == 60
    assert sock.timeouts[-1] == 30


def test_parse_sse_moves_to_answering_idle_timeout(monkeypatch):
    sock = _FakeSock()
    events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {
            "type": "response.output_item.added",
            "item": {"id": "rs_1", "type": "reasoning", "summary": []},
        },
        {
            "type": "response.output_item.added",
            "item": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
            },
        },
        {"type": "response.output_text.delta", "delta": "hi"},
    ]
    stream = _SocketBackedStream(events, sock)
    monkeypatch.setattr(codex, "_response_socket", lambda s: sock)

    with pytest.raises(codex.CodexStallError) as exc_info:
        codex.parse_sse(
            stream,
            timeouts=codex.StreamTimeouts(first_byte=60, thinking_idle=30, answering_idle=25),
        )

    err = exc_info.value
    assert err.stage == "answering"
    assert err.idle_timeout == 25
    assert err.last_event_type == "response.output_text.delta"
    assert sock.timeouts[0] == 60
    assert 30 in sock.timeouts
    assert sock.timeouts[-1] == 25


def test_request_maps_first_byte_timeout_to_stall(monkeypatch, tmp_path):
    import urllib.error

    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json")))

    def fake_urlopen(request, timeout=None):
        assert timeout == 11
        raise urllib.error.URLError(socket.timeout("timed out"))

    monkeypatch.setattr(codex.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(codex.CodexStallError) as exc_info:
        codex._request(
            auth,
            {
                "model": "gpt-5.6-luna",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
                "stream": True,
            },
            codex.StreamTimeouts(first_byte=11, thinking_idle=30, answering_idle=30),
        )

    err = exc_info.value
    assert err.stage == "first_byte"
    assert err.idle_timeout == 11
    assert err.last_event_type is None


def _quota(fetched_at, *windows):
    return {
        "fetched_at": fetched_at,
        "limits": {
            name: {"used_percent": used, "reset_at": reset}
            for name, used, reset in windows
        },
    }


def test_auth_requires_credentials_list(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_credential()))

    with pytest.raises(codex.CodexError, match="credentials list"):
        codex.CodexAuth(str(path))


def test_single_credential_skips_quota_preflight(monkeypatch, tmp_path):
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json")))

    monkeypatch.setattr(
        auth,
        "_usage_request_unlocked",
        lambda *args, **kwargs: pytest.fail("single credential ran quota preflight"),
    )

    assert auth.select_credential() == 0


def test_multi_credential_selection_uses_pool_reset_remaining_and_index(tmp_path):
    now = int(codex.time.time())
    credentials = [
        _credential(
            account_id="zero",
            rate_limits=_quota(now, ("codex_primary", 50, 200)),
        ),
        _credential(
            account_id="one",
            rate_limits=_quota(now, ("codex_primary", 60, 100)),
        ),
        _credential(
            account_id="two",
            rate_limits=_quota(now, ("codex_primary", 70, 100)),
        ),
    ]
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json", credentials)))

    assert auth.select_credential() == 2
    assert auth.index == 2
    assert auth.account_id == "two"

    credentials[1]["rate_limits"] = _quota(now, ("codex_primary", 70, 100))
    credentials[2]["rate_limits"] = _quota(now, ("codex_primary", 70, 100))
    _auth_file(tmp_path / "auth.json", credentials)
    auth = codex.CodexAuth(str(tmp_path / "auth.json"))

    assert auth.select_credential() == 1


def test_normal_pool_is_preferred_over_earlier_reset_reserve(tmp_path):
    now = int(codex.time.time())
    credentials = [
        _credential(
            account_id="reserve",
            rate_limits=_quota(now, ("codex_primary", 96, 100)),
        ),
        _credential(
            account_id="normal",
            rate_limits=_quota(now, ("codex_primary", 94, 200)),
        ),
    ]
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json", credentials)))

    assert auth.select_credential() == 1


def test_longer_horizon_limit_suppresses_earlier_reset(tmp_path):
    now = int(codex.time.time())
    credentials = [
        _credential(
            account_id="suppressed",
            rate_limits=_quota(
                now,
                ("codex_primary", 50, 100),
                ("codex_secondary", 90, 300),
            ),
        ),
        _credential(
            account_id="earlier-effective-reset",
            rate_limits=_quota(now, ("codex_primary", 80, 200)),
        ),
    ]
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json", credentials)))

    assert codex._effective_quota(credentials[0]) == (300, 10.0)
    assert auth.select_credential() == 1


def test_stale_quotas_are_refreshed_and_failed_refresh_is_unusable(
    monkeypatch, tmp_path
):
    now = int(codex.time.time())
    credentials = [
        _credential(
            account_id="stale",
            rate_limits=_quota(now - 3601, ("codex_primary", 10, 50)),
        ),
        _credential(
            account_id="fresh",
            rate_limits=_quota(now, ("codex_primary", 20, 100)),
        ),
    ]
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json", credentials)))
    calls = []

    def refresh(index, timeout=60):
        calls.append(index)
        raise codex.CodexError("unavailable")

    monkeypatch.setattr(auth, "_usage_request_unlocked", refresh)

    assert auth.select_credential() == 1
    assert calls == [0]


def test_all_exhausted_forces_full_quota_recheck(monkeypatch, tmp_path):
    now = int(codex.time.time())
    credentials = [
        _credential(
            account_id="zero",
            rate_limits=_quota(now, ("codex_primary", 100, 100)),
        ),
        _credential(
            account_id="one",
            rate_limits=_quota(now, ("codex_primary", 100, 200)),
        ),
    ]
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json", credentials)))
    calls = []

    def refresh(index, timeout=60):
        calls.append(index)
        snapshot = _quota(
            int(codex.time.time()),
            ("codex_primary", 80 if index == 1 else 100, 300 + index),
        )
        auth.root["credentials"][index]["rate_limits"] = snapshot
        return snapshot

    monkeypatch.setattr(auth, "_usage_request_unlocked", refresh)

    assert auth.select_credential() == 1
    assert calls == [0, 1]


def test_usage_payload_is_reduced_to_selection_fields():
    snapshot = codex._quota_snapshot_from_usage({
        "plan_type": "plus",
        "rate_limit": {
            "allowed": True,
            "primary_window": {
                "used_percent": 12,
                "limit_window_seconds": 18000,
                "reset_after_seconds": 60,
                "reset_at": 100,
            },
            "secondary_window": {
                "used_percent": 30,
                "reset_at": 200,
            },
        },
        "additional_rate_limits": [{
            "limit_name": "Luna",
            "metered_feature": "luna",
            "rate_limit": {
                "primary_window": {
                    "used_percent": 40,
                    "reset_at": 300,
                },
            },
        }],
    }, fetched_at=50)

    assert snapshot == {
        "fetched_at": 50,
        "limits": {
            "codex_primary": {"used_percent": 12.0, "reset_at": 100},
            "codex_secondary": {"used_percent": 30.0, "reset_at": 200},
            "luna_primary": {"used_percent": 40.0, "reset_at": 300},
        },
    }


def test_http_429_expires_selected_quota(monkeypatch, tmp_path):
    import urllib.error

    now = int(codex.time.time())
    credential = _credential(
        rate_limits=_quota(now, ("codex_primary", 20, now + 100)),
    )
    path = _auth_file(tmp_path / "auth.json", [credential])
    auth = codex.CodexAuth(str(path))

    def fail(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "rate limited",
            {},
            BytesIO(b'{"error":"rate_limit"}'),
        )

    monkeypatch.setattr(codex.urllib.request, "urlopen", fail)

    with pytest.raises(codex.CodexError, match="HTTP 429"):
        codex._request(
            auth,
            {
                "model": "gpt-5.6-luna",
                "input": [{"role": "user", "content": []}],
                "stream": True,
            },
            codex.StreamTimeouts(),
        )

    saved = json.loads(path.read_text())
    assert saved["credentials"][0]["rate_limits"]["fetched_at"] == 0



def test_refresh_401_marks_credential_invalid(monkeypatch, tmp_path):
    import urllib.error

    path = _auth_file(tmp_path / "auth.json")
    auth = codex.CodexAuth(str(path))

    def fail(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "unauthorized",
            {},
            BytesIO(b'{"error":"invalid_grant"}'),
        )

    monkeypatch.setattr(codex.urllib.request, "urlopen", fail)

    with pytest.raises(codex.CredentialInvalidError, match="Token refresh failed: HTTP 401") as exc_info:
        auth.refresh()

    assert exc_info.value.index == 0
    saved = json.loads(path.read_text())
    assert saved["credentials"][0]["invalid"] is True


def test_select_credential_ignores_invalid_credentials(tmp_path):
    now = int(codex.time.time())
    credentials = [
        _credential(
            account_id="invalid",
            rate_limits=_quota(now, ("codex_primary", 10, 50)),
        ),
        _credential(
            account_id="usable",
            rate_limits=_quota(now, ("codex_primary", 20, 100)),
        ),
    ]
    credentials[0]["invalid"] = True
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json", credentials)))

    assert auth.select_credential() == 1
    assert auth.account_id == "usable"


def test_select_credential_errors_when_all_invalid(tmp_path):
    credentials = [_credential(account_id="invalid")]
    credentials[0]["invalid"] = True
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json", credentials)))

    with pytest.raises(codex.CodexError, match="all marked invalid"):
        auth.select_credential()


def test_successful_refresh_clears_invalid_flag(monkeypatch, tmp_path):
    path = _auth_file(tmp_path / "auth.json")
    auth = codex.CodexAuth(str(path))
    auth.data["invalid"] = True
    auth._save()

    class RefreshResponse(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def urlopen(request, timeout=None):
        return RefreshResponse(json.dumps({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
        }).encode())

    monkeypatch.setattr(codex.urllib.request, "urlopen", urlopen)
    auth.refresh()

    saved = json.loads(path.read_text())
    assert "invalid" not in saved["credentials"][0]
    assert saved["credentials"][0]["tokens"]["access_token"] == "new-access"


def test_ensure_valid_switches_credential_after_refresh_401(monkeypatch, tmp_path):
    import urllib.error

    now = int(codex.time.time())
    expired = _jwt({"exp": 1, "https://api.openai.com/auth": {"chatgpt_account_id": "bad"}})
    credentials = [
        _credential(
            access_token=expired,
            account_id="bad",
            rate_limits=_quota(now, ("codex_primary", 10, 50)),
        ),
        _credential(
            account_id="good",
            rate_limits=_quota(now, ("codex_primary", 20, 100)),
        ),
    ]
    path = _auth_file(tmp_path / "auth.json", credentials)
    auth = codex.CodexAuth(str(path))
    assert auth.index == 0

    def fail_refresh(request, timeout=None):
        assert request.full_url == codex.REFRESH_URL
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "unauthorized",
            {},
            BytesIO(b'{"error":"invalid_grant"}'),
        )

    monkeypatch.setattr(codex.urllib.request, "urlopen", fail_refresh)
    auth.ensure_valid()

    assert auth.index == 1
    assert auth.account_id == "good"
    saved = json.loads(path.read_text())
    assert saved["credentials"][0]["invalid"] is True
    assert "invalid" not in saved["credentials"][1]


def test_request_switches_credential_after_refresh_401(monkeypatch, tmp_path):
    import urllib.error

    now = int(codex.time.time())
    credentials = [
        _credential(
            account_id="bad",
            rate_limits=_quota(now, ("codex_primary", 10, 50)),
        ),
        _credential(
            account_id="good",
            rate_limits=_quota(now, ("codex_primary", 20, 100)),
        ),
    ]
    path = _auth_file(tmp_path / "auth.json", credentials)
    auth = codex.CodexAuth(str(path))
    seen_accounts = []
    events = [
        {"type": "response.output_text.delta", "delta": "ok"},
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }],
            },
        },
    ]
    payload = b"".join(
        f"data: {json.dumps(event)}\n\n".encode() for event in events
    ) + b"data: [DONE]\n\n"

    def urlopen(request, timeout=None):
        if request.full_url == codex.REFRESH_URL:
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "unauthorized",
                {},
                BytesIO(b'{"error":"invalid_grant"}'),
            )
        if request.full_url == codex.RESPONSES_URL:
            account = request.get_header("Chatgpt-account-id") or request.get_header("ChatGPT-Account-ID")
            # urllib Request normalizes header names; inspect headers dict instead
            headers = {k.lower(): v for k, v in request.header_items()}
            account = headers.get("chatgpt-account-id")
            seen_accounts.append(account)
            if account == "bad":
                raise urllib.error.HTTPError(
                    request.full_url,
                    401,
                    "unauthorized",
                    {},
                    BytesIO(b'{"error":"unauthorized"}'),
                )
            return _FakeResponse(payload)
        raise AssertionError(f"unexpected url: {request.full_url}")

    monkeypatch.setattr(codex.urllib.request, "urlopen", urlopen)
    response = codex._request(
        auth,
        {
            "model": "gpt-5.6-luna",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "stream": True,
        },
        codex.StreamTimeouts(),
    )

    assert response["output"][0]["content"][0]["text"] == "ok"
    assert seen_accounts == ["bad", "good"]
    assert auth.account_id == "good"
    saved = json.loads(path.read_text())
    assert saved["credentials"][0]["invalid"] is True
