
import json
import os
import stat
import subprocess
import sys
import time

import pytest

from code_agent import cursor
from code_agent.client import LLMClient
from code_agent.llm_registry import get_model_config
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
from code_agent.llm_registry import get_model_config
config = get_model_config("cursor/composer-2.5")
grok = get_model_config("cursor/grok-4.5")
print(json.dumps({
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
        "api_type": "cursor",
        "tool_mode": "repl_execute",
        "host": None,
        "path": None,
        "grok_model": "cursor-grok-4.5-high",
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


def test_shell_stream_native_tool_translates_to_bash():
    call = cursor.ToolCall(
        id="tool-1",
        name="shell_stream",
        arguments={
            "command": "python3.11 -m py_compile app.py",
            "working_directory": "/tmp/project dir",
            "timeout": 30000,
            "is_background": False,
            "description": "Compile app",
        },
        native=True,
        oneof_name="shell_stream_args",
    )

    assert cursor._native_repl_code(call) == (
        "bash(command=\"cd -- '/tmp/project dir' && "
        "python3.11 -m py_compile app.py\", timeout=30, bg=False)"
    )


def test_grep_native_tool_translates_to_grep():
    call = cursor.ToolCall(
        id="tool-2",
        name="grep",
        arguments={
            "pattern": "matched INTEGER|def find_pending",
            "path": "/tmp/project",
            "glob": "*.{py,md}",
            "case_insensitive": False,
            "multiline": False,
            "tool_call_id": "tool-2",
        },
        native=True,
        oneof_name="grep_args",
    )

    assert cursor._native_repl_code(call) == (
        "grep(pattern='matched INTEGER|def find_pending', "
        "path='/tmp/project', glob='*.{py,md}', "
        "case_insensitive=False, multiline=False)"
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


def _decode_model_details(payload: bytes):
    client = cursor.RawMessage.decode(payload)
    run = cursor.RawMessage.decode(client.first_bytes(1))
    details = cursor.RawMessage.decode(run.first_bytes(3))
    return {
        number: (details.first_bytes(number) or b"").decode()
        for number in (1, 3, 4, 5)
    }, cursor._int(details, 7), run


def test_encode_model_details():
    raw = cursor.RawMessage.decode(
        cursor.encode_model_details("cursor-grok-4.5-high")
    )
    assert {
        number: (raw.first_bytes(number) or b"").decode()
        for number in (1, 3, 4, 5)
    } == {
        1: "cursor-grok-4.5-high",
        3: "cursor-grok-4.5-high",
        4: "cursor-grok-4.5-high",
        5: "cursor-grok-4.5-high",
    }
    assert cursor._int(raw, 7) == 1


def test_build_run_request_uses_legacy_model_details_only():
    payload = cursor.build_run_request("hi", "cursor-grok-4.5-high")
    values, mode, run = _decode_model_details(payload)
    assert values == {
        1: "cursor-grok-4.5-high",
        3: "cursor-grok-4.5-high",
        4: "cursor-grok-4.5-high",
        5: "cursor-grok-4.5-high",
    }
    assert mode == 1
    assert not run.has(9)
    assert not run.has(14)


def test_chat_completions_basic(monkeypatch):
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
        "model": "composer-2.5",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert captured["model"] == "composer-2.5"
    assert response["choices"][0]["message"]["content"] == "ok"


def _turn_structure_frame(request_id):
    structure = cursor.protobuf_message(
        cursor.Field.bytes(1, b"u" * 32),
        cursor.Field.bytes(2, b"s" * 32),
        cursor.Field.bytes(3, request_id.encode()),
        cursor.Field.varint(5, 0),
    )
    blob = cursor.protobuf_message(cursor.Field.bytes(1, structure))
    set_args = cursor.protobuf_message(cursor.Field.bytes(2, blob))
    kv = cursor.protobuf_message(
        cursor.Field.varint(1, 9),
        cursor.Field.bytes(3, set_args),
    )
    return cursor.ConnectFrame.from_decoded(
        cursor.protobuf_message(cursor.Field.bytes(4, kv))
    )


def test_agent_conversation_turn_structure_detection():
    request_id = "5270c3d8-821c-4c8b-bdf4-2d880504206c"
    frame = _turn_structure_frame(request_id)

    assert cursor.is_agent_conversation_turn_structure(
        frame.decoded_payload, request_id
    )
    assert not cursor.is_agent_conversation_turn_structure(
        frame.decoded_payload, "other-request"
    )


def test_cursor_turn_structure_closes_without_tool_call(monkeypatch):
    request_id = "5270c3d8-821c-4c8b-bdf4-2d880504206c"
    closed = []

    class FakeSSEClient:
        def __init__(self, *args, stream_callback, headers_callback, **kwargs):
            self.stream_callback = stream_callback
            self.headers_callback = headers_callback
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def post(self, url, body=b"", headers=None, callback=None):
            callback({"status": 200, "headers": {}, "body": b""})

        def reset_heartbeat_timeout(self):
            pass

        def close(self):
            if not self.closed:
                self.closed = True
                closed.append(True)

        def run_forever(self, timeout=None, *, heartbeat_timeout=None):
            self.headers_callback(200, {})
            answer = cursor.ConnectFrame.from_decoded(
                cursor.AnswerText.create("done").encode()
            )
            self.stream_callback(answer.encode())
            self.stream_callback(_turn_structure_frame(request_id).encode())
            assert self.closed

    monkeypatch.setattr(cursor.uuid, "uuid4", lambda: request_id)
    monkeypatch.setattr(cursor, "SSEClient", FakeSSEClient)
    result = cursor.CursorClient("token").run("hello")

    assert result.text == "done"
    assert result.tool_calls == []
    assert result.turn_ended is True
    assert closed == [True]


def test_cursor_turn_structure_collects_tools_and_closes(monkeypatch):
    request_id = "5270c3d8-821c-4c8b-bdf4-2d880504206c"
    closed = []

    def tool_frame(call_id, code):
        arguments = cursor.protobuf_message(
            cursor.Field.bytes(1, b"repl_execute"),
            cursor.Field.bytes(
                2,
                cursor.protobuf_message(
                    cursor.Field.bytes(1, b"code"),
                    cursor.Field.bytes(2, cursor._protobuf_value(code)),
                ),
            ),
            cursor.Field.bytes(3, call_id.encode()),
            cursor.Field.bytes(5, b"repl_execute"),
        )
        execution = cursor.protobuf_message(
            cursor.Field.varint(1, 1),
            cursor.Field.bytes(11, arguments),
            cursor.Field.bytes(15, (call_id + "-exec").encode()),
        )
        return cursor.ConnectFrame.from_decoded(
            cursor.protobuf_message(cursor.Field.bytes(2, execution))
        ).encode()

    class FakeSSEClient:
        def __init__(self, *args, stream_callback, headers_callback, **kwargs):
            self.stream_callback = stream_callback
            self.headers_callback = headers_callback
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def post(self, url, body=b"", headers=None, callback=None):
            callback({"status": 200, "headers": {}, "body": b""})

        def reset_heartbeat_timeout(self):
            pass

        def close(self):
            if not self.closed:
                self.closed = True
                closed.append(True)

        def run_forever(self, timeout=None, *, heartbeat_timeout=None):
            self.headers_callback(200, {})
            for payload in (
                tool_frame("call-1", "emit('one')"),
                tool_frame("call-2", "emit('two')"),
                _turn_structure_frame(request_id).encode(),
            ):
                self.stream_callback(payload)
            assert self.closed

    monkeypatch.setattr(cursor.uuid, "uuid4", lambda: request_id)
    monkeypatch.setattr(cursor, "SSEClient", FakeSSEClient)
    result = cursor.CursorClient("token").run("hello")

    assert [call.id for call in result.tool_calls] == ["call-1", "call-2"]
    assert result.turn_ended is True
    assert closed == [True]



def test_cursor_turn_ended_still_closes_immediately(monkeypatch):
    closed = []

    class FakeSSEClient:
        def __init__(self, *args, stream_callback, headers_callback, **kwargs):
            self.stream_callback = stream_callback
            self.headers_callback = headers_callback
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def post(self, url, body=b"", headers=None, callback=None):
            callback({"status": 200, "headers": {}, "body": b""})

        def reset_heartbeat_timeout(self):
            pass

        def close(self):
            if not self.closed:
                self.closed = True
                closed.append(True)

        def run_forever(self, timeout=None, *, heartbeat_timeout=None):
            self.headers_callback(200, {})
            frame = cursor.ConnectFrame.from_decoded(
                cursor.InteractionUpdate.create("turn_ended").encode()
            )
            self.stream_callback(frame.encode())
            assert self.closed

    monkeypatch.setattr(cursor, "SSEClient", FakeSSEClient)
    result = cursor.CursorClient("token").run("hello")

    assert result.turn_ended is True
    assert closed == [True]