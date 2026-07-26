
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
