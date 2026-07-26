
import json
import os
import stat
import subprocess
import sys
import time

import pytest

from code_agent import cursor
from code_agent.client import LLMClient
from code_agent.llm_registry import get_model_config, resolve_model_name
from code_agent.repl_tool_adapter import REPL_EXECUTE_TOOL


def test_cursor_registry(tmp_path):
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "CURSOR_API_KEY": "key",
        "PYTHONPATH": os.getcwd(),
    }
    code = """
import json
from code_agent.llm_registry import get_model_config, resolve_model_name
config = get_model_config("cursor")
grok = get_model_config("cursor/grok-4.5")
print(json.dumps({
    "resolved": resolve_model_name("cursor"),
    "api_type": config["api_type"],
    "tool_mode": config["tool_mode"],
    "host": config["host"],
    "path": config["path"],
    "grok_model": grok["model"],
    "grok_api_type": grok["api_type"],
    "grok_tool_mode": grok["tool_mode"],
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    config = json.loads(result.stdout)
    assert config == {
        "resolved": "cursor/composer-2.5",
        "api_type": "cursor",
        "tool_mode": "repl_execute",
        "host": None,
        "path": None,
        "grok_model": "grok-4.5",
        "grok_api_type": "cursor",
        "grok_tool_mode": "repl_execute",
    }


def test_cached_access_token_is_reused(monkeypatch, tmp_path):
    cache = tmp_path / "cursor-auth.json"
    lock = tmp_path / "cursor-auth.lock"
    cache.write_text(json.dumps({
        "access_token": "cached",
        "expires_at": time.time() + 600,
    }))
    monkeypatch.setattr(cursor, "exchange_api_key", lambda key: pytest.fail("exchange"))
    assert cursor.get_access_token("key", cache_path=str(cache), lock_path=str(lock)) == "cached"
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600


def test_access_token_refreshes_and_writes_securely(monkeypatch, tmp_path):
    cache = tmp_path / "cursor-auth.json"
    lock = tmp_path / "cursor-auth.lock"
    cache.write_text(json.dumps({
        "access_token": "old",
        "expires_at": time.time() + 299,
    }))
    monkeypatch.setattr(
        cursor,
        "exchange_api_key",
        lambda key: {"accessToken": "new", "expiresIn": 3600},
    )
    assert cursor.get_access_token("key", cache_path=str(cache), lock_path=str(lock)) == "new"
    saved = json.loads(cache.read_text())
    assert saved["access_token"] == "new"
    assert saved["expires_at"] > time.time() + 3500
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600


def test_exchange_failure_preserves_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cursor-auth.json"
    lock = tmp_path / "cursor-auth.lock"
    original = json.dumps({"access_token": "old", "expires_at": 0})
    cache.write_text(original)
    def fail(key):
        raise RuntimeError("exchange failed")
    monkeypatch.setattr(cursor, "exchange_api_key", fail)
    with pytest.raises(RuntimeError, match="exchange failed"):
        cursor.get_access_token("key", cache_path=str(cache), lock_path=str(lock))
    assert cache.read_text() == original


@pytest.mark.parametrize("message", [
    {"role": "user", "content": "x", "images": [b"image"]},
    {"role": "user", "content": "x", "audio": [b"audio"]},
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
])
def test_cursor_rejects_media(message):
    with pytest.raises(ValueError, match="does not support"):
        cursor._openai_messages([message])


def test_cursor_client_adapter(monkeypatch):
    config = {
        "provider": "cursor",
        "model": "composer-2.5",
        "api_key": "key",
        "api_type": "cursor",
        "tool_mode": "repl_execute",
        "tpm": 17,
        "tools": True,
        "concurrency": 1,
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    throttled = []
    monkeypatch.setattr("code_agent.client.throttle", lambda key, tpm: throttled.append((key, tpm)))
    captured = {}
    def chat(api_key, body):
        captured.update(api_key=api_key, body=body)
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "repl_execute",
                            "arguments": json.dumps({"code": "emit('ok', release=True)"}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }]
        }
    monkeypatch.setattr(cursor, "chat_completions", chat)
    client = LLMClient("cursor")
    result = client._call([{"role": "user", "content": "hello"}])
    assert captured["api_key"] == "key"
    assert captured["body"]["model"] == "composer-2.5"
    assert captured["body"]["tools"] == [REPL_EXECUTE_TOOL]
    assert "timeout" not in captured["body"]
    assert throttled == [("cursor", 17)]
    assert result["content"] == "emit('ok', release=True)"
    assert result["_stop_reason"] == "tool_calls"


def test_unsupported_native_tool_renders_flattened_known_parameters():
    call = cursor.ToolCall(
        id="tool-1",
        name="shell_stream",
        arguments={
            "1": "date",
            "10": 40000,
            "3": 30000,
            "15": "Get current system date",
            "8": b"opaque",
        },
        native=True,
        oneof_name="shell_stream_args",
    )

    assert cursor._native_repl_code(call) == (
        "# unsupported tool call: ShellStream("
        "{'command': 'date', 'file_output_threshold_bytes': 40000, "
        "'timeout': 30000, 'description': 'Get current system date', "
        "'parsing_result': b'opaque'})"
    )



def test_shell_stream_wire_arguments_use_recovered_schema():
    arguments = cursor.protobuf_message(
        cursor.Field.bytes(1, b"date"),
        cursor.Field.bytes(2, b"/tmp"),
        cursor.Field.varint(3, 30000),
        cursor.Field.varint(11, 1),
        cursor.Field.bytes(15, b"Get current system date"),
        cursor.Field.varint(17, 1),
    )

    assert cursor._generic_arguments(arguments, "shell_stream_args") == {
        "command": "date",
        "working_directory": "/tmp",
        "timeout": 30000,
        "is_background": True,
        "description": "Get current system date",
        "close_stdin": True,
    }


def test_cursor_client_requires_repl_execute(monkeypatch):
    config = {
        "provider": "cursor",
        "model": "composer-2.5",
        "api_key": "key",
        "api_type": "cursor",
        "tool_mode": None,
        "concurrency": 1,
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("cursor")
    with pytest.raises(TypeError, match="requires tool_mode"):
        client._call_cursor([], None)


def _decode_model_metadata(payload: bytes):
    client = cursor.RawMessage.decode(payload)
    run = cursor.RawMessage.decode(client.first_bytes(1))
    meta = cursor.RawMessage.decode(run.first_bytes(14))
    model = (meta.first_bytes(1) or b"").decode()
    entries = {}
    for field in meta.matching(3, 2):
        entry = cursor.RawMessage.decode(field.value)
        key = (entry.first_bytes(1) or b"").decode()
        value = (entry.first_bytes(2) or b"").decode()
        entries[key] = value
    return model, entries


def test_encode_model_metadata_defaults_omit_fast_and_effort():
    raw = cursor.RawMessage.decode(cursor.encode_model_metadata("grok-4.5"))
    assert (raw.first_bytes(1) or b"").decode() == "grok-4.5"
    assert raw.matching(3, 2) == ()


def test_encode_model_metadata_fast_and_effort():
    raw = cursor.RawMessage.decode(
        cursor.encode_model_metadata(
            "grok-4.5",
            fast=True,
            reasoning_effort="medium",
        )
    )
    assert (raw.first_bytes(1) or b"").decode() == "grok-4.5"
    entries = {}
    for field in raw.matching(3, 2):
        entry = cursor.RawMessage.decode(field.value)
        entries[(entry.first_bytes(1) or b"").decode()] = (
            entry.first_bytes(2) or b""
        ).decode()
    assert entries == {"fast": "true", "effort": "medium"}


def test_build_run_request_model_metadata_defaults():
    payload = cursor.build_run_request("hi", "grok-4.5")
    assert _decode_model_metadata(payload) == ("grok-4.5", {})


def test_build_run_request_model_metadata_fast_and_effort():
    payload = cursor.build_run_request(
        "hi",
        "grok-4.5",
        fast=True,
        reasoning_effort="high",
    )
    assert _decode_model_metadata(payload) == (
        "grok-4.5",
        {"fast": "true", "effort": "high"},
    )


def test_chat_completions_passes_fast_and_reasoning_effort(monkeypatch):
    captured = {}

    def fake_run(prompt, **kwargs):
        captured.update(kwargs)
        captured["prompt"] = prompt
        return cursor.RunResult(
            frames=[],
            text="ok",
            tool_calls=[],
            turn_ended=True,
            checkpoint_updates=[],
            eos_metadata=None,
            eos_error=None,
        )

    monkeypatch.setattr(cursor, "run", fake_run)
    response = cursor.chat_completions("key", {
        "model": "grok-4.5",
        "messages": [{"role": "user", "content": "hello"}],
        "fast": True,
        "reasoning_effort": "medium",
    })
    assert captured["model"] == "grok-4.5"
    assert captured["fast"] is True
    assert captured["reasoning_effort"] == "medium"
    assert response["choices"][0]["message"]["content"] == "ok"


def test_chat_completions_rejects_non_bool_fast():
    with pytest.raises(TypeError, match="fast must be bool"):
        cursor.chat_completions("key", {
            "model": "grok-4.5",
            "messages": [{"role": "user", "content": "hello"}],
            "fast": "true",
        })


def test_cursor_client_adapter_passes_config(monkeypatch):
    config = {
        "provider": "cursor",
        "model": "grok-4.5",
        "api_key": "key",
        "api_type": "cursor",
        "tool_mode": "repl_execute",
        "tpm": 17,
        "tools": True,
        "concurrency": 1,
        "config": {
            "fast": True,
            "reasoning_effort": "medium",
        },
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    monkeypatch.setattr("code_agent.client.throttle", lambda key, tpm: None)
    captured = {}

    def chat(api_key, body):
        captured.update(api_key=api_key, body=body)
        return {
            "choices": [{
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }]
        }

    monkeypatch.setattr(cursor, "chat_completions", chat)
    client = LLMClient("cursor/grok-4.5")
    result = client._call([{"role": "user", "content": "hello"}])
    assert captured["body"]["model"] == "grok-4.5"
    assert captured["body"]["fast"] is True
    assert captured["body"]["reasoning_effort"] == "medium"
    assert result["content"] == "done"
