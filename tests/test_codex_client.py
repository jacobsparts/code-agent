import json
import os
import stat
import subprocess
import sys
from io import BytesIO

import pytest

from code_agent import codex
from code_agent.client import LLMClient
from code_agent.repl_tool_adapter import REPL_EXECUTE_TOOL


def _jwt(payload):
    import base64
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).decode().rstrip("=")
    return f"x.{encoded}.x"


def _auth_file(path):
    path.write_text(json.dumps({
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": _jwt({
                "exp": 4_000_000_000,
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": "account-from-token"
                },
            }),
            "id_token": "id",
            "refresh_token": "refresh",
        },
        "last_refresh": "2026-01-01T00:00:00+00:00",
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
    assert json.loads(path.read_text())["tokens"]["access_token"] == "new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not (tmp_path / "auth.json.tmp").exists()


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

    def request(auth_arg, body, timeout):
        captured.update(auth=auth_arg, body=body, timeout=timeout)
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
    assert captured["timeout"] == 17
    assert captured["body"]["input"] == request_body["input"]
    assert captured["body"]["tools"] == request_body["tools"]
    assert captured["body"]["stream"] is True
    assert captured["body"]["store"] is False
    assert captured["body"]["parallel_tool_calls"] is True
    assert captured["body"]["include"] == ["reasoning.encrypted_content"]


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
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("openai/gpt-5.6-luna")
    request = client._responses_request([
        {
            "role": "system",
            "content": "stable instructions",
            "_prompt_cache_breakpoint": True,
        },
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": "file_123"},
                {"type": "text", "text": "Answer the current question."},
            ],
            "_prompt_cache_breakpoint": True,
        },
        {"role": "user", "content": "uncached"},
    ], None)

    assert request["prompt_cache_key"] == "tenant:acme:knowledge-base-v1"
    assert request["prompt_cache_options"] == {"mode": "explicit"}
    assert request["input"][0]["content"][-1]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert request["input"][1]["content"][-1]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert "prompt_cache_breakpoint" not in request["input"][1]["content"][0]
    assert "prompt_cache_breakpoint" not in request["input"][2]["content"][0]


def test_codex_strips_unsupported_prompt_cache_fields(monkeypatch, tmp_path):
    auth = codex.CodexAuth(str(_auth_file(tmp_path / "auth.json")))
    captured = {}

    def request(auth_arg, body, timeout):
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


def test_codex_client_adapter_without_repl_tool_mode(monkeypatch):
    config = {
        "provider": "codex",
        "model": "gpt-5.6-luna",
        "api_key": None,
        "api_type": "codex",
        "tool_mode": None,
        "tpm": 17,
        "tools": True,
        "concurrency": 1,
        "timeout": 20,
        "config": {},
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    captured = {}

    def complete(body, auth=None, timeout=codex.DEFAULT_TIMEOUT):
        captured.update(body=body, auth=auth, timeout=timeout)
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
    monkeypatch.setattr("code_agent.client.throttle", lambda *args: None)
    client = LLMClient("codex/gpt-5.6-luna")
    result = client._call([{"role": "user", "content": "hello"}])
    assert captured["body"]["input"] == [{
        "role": "user",
        "content": [{"type": "input_text", "text": "hello"}],
    }]
    assert "tools" not in captured["body"]
    assert result == {"role": "assistant", "content": "hello", "_stop_reason": "stop"}


def test_codex_client_adapter(monkeypatch, tmp_path):
    config = {
        "provider": "codex",
        "model": "gpt-5.6-luna",
        "api_key": None,
        "api_type": "codex",
        "tool_mode": "repl_execute",
        "tpm": 17,
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

    def complete(body, auth=None, timeout=codex.DEFAULT_TIMEOUT):
        captured.update(body=body, auth=auth, timeout=timeout)
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
    monkeypatch.setattr("code_agent.client.throttle", lambda *args: None)
    client = LLMClient("codex/gpt-5.6-luna-xhigh")
    result = client._call([{"role": "user", "content": "hello"}])
    assert captured["body"]["model"] == "gpt-5.6-luna"
    assert captured["body"]["tools"] == [{
        "type": "function",
        **REPL_EXECUTE_TOOL["function"],
    }]
    assert captured["timeout"] == 20
    assert result["content"] == "emit('ok', release=True)"
    assert result["_stop_reason"] == "tool_calls"
