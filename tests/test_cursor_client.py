
import json
import os
import selectors
import stat
import struct
import subprocess
import sys
import time

import pytest

from code_agent.transports import cursor
from code_agent.client import LLMClient
from code_agent.llm_registry import get_model_config
from code_agent.repl_tool_adapter import REPL_EXECUTE_TOOL


# --- Test-only protobuf wire helpers (production no longer ships these) ---

def _encode_varint(value: int) -> bytes:
    assert 0 <= value < (1 << 64)
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _wire_tag(number: int, wire_type: int) -> bytes:
    return _encode_varint((number << 3) | wire_type)


def pb_varint(number: int, value: int) -> bytes:
    return _wire_tag(number, 0) + _encode_varint(value)


def pb_fixed64(number: int, value: int) -> bytes:
    return _wire_tag(number, 1) + struct.pack("<Q", value)


def pb_bytes(number: int, value: bytes) -> bytes:
    return _wire_tag(number, 2) + _encode_varint(len(value)) + value


def pb_message(*fields: bytes) -> bytes:
    return b"".join(fields)


def pb_fields(data: bytes) -> list[tuple[int, int, object]]:
    """Decode a protobuf message into (number, wire_type, value) tuples."""
    out = []
    i = 0
    while i < len(data):
        tag, i = _read_one(data, i)
        number, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = _read_one(data, i)
        elif wt == 1:
            v, i = struct.unpack("<Q", data[i:i + 8])[0], i + 8
        elif wt == 2:
            n, i = _read_one(data, i)
            v, i = data[i:i + n], i + n
        elif wt == 5:
            v, i = struct.unpack("<I", data[i:i + 4])[0], i + 4
        else:
            raise ValueError(f"bad wire type {wt}")
        out.append((number, wt, v))
    return out


def _read_one(data: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = data[i]
        i += 1
        r |= (c & 0x7F) << s
        if not c & 0x80:
            return r, i
        s += 7


def pb_first_bytes(data: bytes, number: int) -> bytes | None:
    for num, wt, v in pb_fields(data):
        if num == number and wt == 2:
            return v
    return None


def pb_ints(data: bytes, number: int) -> list[int]:
    return [v for num, wt, v in pb_fields(data) if num == number and wt == 0]


def pb_texts(data: bytes, number: int) -> list[str]:
    values = []
    for num, wt, v in pb_fields(data):
        if num == number and wt == 2 and isinstance(v, bytes):
            values.append(v.decode())
    return values


def answer_text_frame(text: str) -> bytes:
    leaf = pb_bytes(1, text.encode())
    middle = pb_bytes(1, leaf)
    return pb_bytes(1, middle)


def interaction_update_frame(subtype_number: int, payload: bytes = b"") -> bytes:
    interaction = pb_bytes(subtype_number, payload)
    return pb_bytes(1, interaction)


def pb_value(value) -> bytes:
    """Build a Google.protobuf Value-style wrapper: str->bytes field 1,
    bool->varint field 2, number->varint field 3."""
    if isinstance(value, bool):
        return pb_varint(2, int(value))
    if isinstance(value, (int, float)):
        return pb_varint(3, int(value))
    return pb_bytes(1, str(value).encode())


_subtype_numbers = {name: num for num, name in cursor.INTERACTION_UPDATE_FIELD_NAMES.items()}
_THINKING_DELTA_NUMBER = _subtype_numbers["thinking_delta"]
_TOOL_CALL_COMPLETED_NUMBER = _subtype_numbers["tool_call_completed"]
_HEARTBEAT_NUMBER = _subtype_numbers["heartbeat"]
_TURN_ENDED_NUMBER = _subtype_numbers["turn_ended"]


def decode_cursor(payload: bytes) -> "cursor.CursorMessage":
    return cursor.decode_cursor_payload(payload, "IN")


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
    events = []
    monkeypatch.setattr(
        cursor,
        "exchange_api_key",
        lambda key: (
            events.append(("exchange", key))
            or {"accessToken": "new", "expiresIn": 3600}
        ),
    )
    monkeypatch.setattr(
        cursor,
        "discover_models",
        lambda token: events.append(("discover", token)),
    )
    assert cursor.get_access_token("key", cache_path=str(cache), lock_path=str(lock)) == "new"
    assert events == [("exchange", "key"), ("discover", "new")]
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
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
])
def test_cursor_rejects_media(message):
    with pytest.raises(ValueError, match="does not support"):
        cursor._openai_messages([message])


def test_cursor_tool_continuation_prompt():
    messages = [
        {"role": "user", "content": "Run the tool."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "repl_execute",
                    "arguments": json.dumps({"code": "print('ok')"}),
                },
            }],
        },
        {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "call-1",
            "name": "repl_execute",
        },
    ]

    prompt, history = cursor._openai_messages(messages)

    assert prompt == "This is a tool result. Do not mention this reminder to the user."
    assert len(history) == 3
    assert history[-1]["role"] == "tool"
    assert history[-1]["content"] == "ok"


def test_cursor_client_adapter(monkeypatch):
    config = {
        "provider": "cursor",
        "model": "composer-2.5",
        "api_key": "key",
        "api_type": "cursor",
        "tool_mode": "repl_execute",
        "rpm": 17,
        "tools": True,
        "concurrency": 1,
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    monkeypatch.setattr("code_agent.utils.get_model_config", lambda name: config)
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
            }],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 157,
                "prompt_tokens_details": {"cached_tokens": 20},
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        }
    monkeypatch.setattr(cursor, "chat_completions", chat)
    client = LLMClient("cursor/composer-2.5")
    history_length = len(client.usage_tracker.history)
    result = client._call([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
    assert captured["api_key"] == "key"
    assert captured["body"]["model"] == "composer-2.5"
    assert captured["body"]["tools"] == [REPL_EXECUTE_TOOL]
    assert "timeout" not in captured["body"]
    assert result["content"] == [{
        "type": "text",
        "text": "emit('ok', release=True)",
    }]
    assert result["provider_metadata"]["stop_reason"] == "tool_calls"
    assert client.usage_tracker.history[history_length:] == [(
        "cursor/composer-2.5",
        {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 157,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    )]
    assert client.usage_tracker._normalize(
        "cursor/composer-2.5", client.usage_tracker.history[-1][1]
    ) == {
        "prompt_tokens": 100,
        "cached_tokens": 20,
        "completion_tokens": 30,
        "reasoning_tokens": 7,
        "cost": 0.0,
    }
    assert client._input_tokens_per_byte() is not None
    del client.usage_tracker.history[history_length:]


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
    arguments = pb_message(
        pb_bytes(1, b"date"),
        pb_bytes(2, b"/tmp"),
        pb_varint(3, 30000),
        pb_varint(11, 1),
        pb_bytes(15, b"Get current system date"),
        pb_varint(17, 1),
    )

    decoded_arguments = cursor.cursor_schema.decode(arguments, "ShellArgs")
    assert cursor._generic_arguments(decoded_arguments) == {
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
    client = payload
    run = pb_first_bytes(client, 1)
    details = pb_first_bytes(run, 3)
    return {
        number: (pb_first_bytes(details, number) or b"").decode()
        for number in (1, 3, 4, 5)
    }, pb_ints(details, 7)[0], run


def test_encode_model_details():
    raw = (
        cursor.encode_model_details("cursor-grok-4.5-high")
    )
    assert {
        number: (pb_first_bytes(raw, number) or b"").decode()
        for number in (1, 3, 4, 5)
    } == {
        1: "cursor-grok-4.5-high",
        3: "cursor-grok-4.5-high",
        4: "cursor-grok-4.5-high",
        5: "cursor-grok-4.5-high",
    }
    assert pb_ints(raw, 7) == [1]


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
    assert pb_first_bytes(run, 9) is None
    assert pb_first_bytes(run, 14) is None


def test_build_run_request_keeps_stable_conversation_id(monkeypatch):
    monkeypatch.setattr(cursor, "_SESSION_CONVERSATION_ID", "stable-conversation")

    first = cursor.build_run_request(
        "one", "composer-2.5", message_id="message-1"
    )
    second = cursor.build_run_request(
        "two", "composer-2.5", message_id="message-2"
    )
    first_run = pb_first_bytes(first, 1)
    second_run = pb_first_bytes(second, 1)

    assert pb_texts(first_run, 5) == ["stable-conversation"]
    assert pb_texts(second_run, 5) == ["stable-conversation"]
    assert pb_first_bytes(first_run, 16) is None
    assert pb_first_bytes(second_run, 16) is None


def test_build_run_request_allows_explicit_conversation_ids():
    payload = cursor.build_run_request(
        "hi",
        "composer-2.5",
        conversation_id="explicit-conversation",
        message_id="message",
        run_config=cursor.RunConfig(
            conversation_group_id="explicit-group"
        ),
    )
    run = pb_first_bytes(payload, 1)

    assert pb_texts(run, 5) == ["explicit-conversation"]
    assert pb_texts(run, 16) == ["explicit-group"]


def _run_result(usage):
    return cursor.RunResult(
        frames=[],
        text="ok",
        tool_calls=[],
        turn_ended=True,
        checkpoint_updates=[],
        eos_metadata=None,
        eos_error=None,
        usage=usage,
    )


def test_chat_completions_basic(monkeypatch):
    captured = {}

    def fake_run(prompt, **kwargs):
        captured.update(kwargs)
        captured["prompt"] = prompt
        return _run_result(cursor.TurnUsage(
            input_tokens=12_000,
            output_tokens=30,
            cache_read_tokens=2_000,
            reasoning_tokens=7,
        ))

    model = "cursor-grok-4.5-high"
    monkeypatch.setattr(cursor, "run", fake_run)
    monkeypatch.delitem(cursor._MODEL_CURSOR_STATS, model, raising=False)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    }
    input_bytes = cursor._cursor_input_bytes(body)
    expected_input = round(
        cursor.CURSOR_MODEL_CALIBRATION[model]["system_prompt_tokens"]
        + input_bytes
        * cursor.CURSOR_MODEL_CALIBRATION[model]["variable_tokens_per_byte"]
    )

    response = cursor.chat_completions("key", body)

    assert captured["model"] == model
    assert response["choices"][0]["message"]["content"] == "ok"
    assert response["usage"] == {
        "prompt_tokens": expected_input,
        "completion_tokens": 30,
    }


def test_chat_completions_requires_model():
    with pytest.raises(ValueError, match="model is required"):
        cursor.chat_completions("key", {
            "messages": [{"role": "user", "content": "hello"}],
        })


def test_unknown_model_uses_two_fresh_conversation_probes(
    monkeypatch, capsys,
):
    model = "unknown-model"
    conversation_ids = []
    prompts = []
    sample = cursor._CURSOR_CALIBRATION_SAMPLE
    sample_bytes = len(sample.encode("utf-8"))
    added_bytes = len(("\n\n" + sample).encode("utf-8"))
    ratio = 0.25
    system_tokens = 12_345

    def fake_run(prompt, **kwargs):
        prompts.append(prompt)
        conversation_ids.append(kwargs["run_config"].conversation_id)
        if len(prompts) == 1:
            input_tokens = round(system_tokens + ratio * sample_bytes)
        elif len(prompts) == 2:
            input_tokens = round(
                system_tokens + ratio * (sample_bytes + added_bytes)
            )
        else:
            input_tokens = 20_000
        return _run_result(cursor.TurnUsage(input_tokens=input_tokens))

    monkeypatch.delitem(cursor.CURSOR_MODEL_CALIBRATION, model, raising=False)
    monkeypatch.delitem(cursor._MODEL_CURSOR_STATS, model, raising=False)
    monkeypatch.setattr(cursor, "run", fake_run)

    cursor.chat_completions("key", {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert prompts[:2] == [sample, sample + "\n\n" + sample]
    assert len(set(conversation_ids)) == 3
    calibration = cursor.CURSOR_MODEL_CALIBRATION[model]
    assert calibration["system_prompt_tokens"] == pytest.approx(
        system_tokens, abs=1
    )
    expected_ratio = (
        round(system_tokens + ratio * (sample_bytes + added_bytes))
        - round(system_tokens + ratio * sample_bytes)
    ) / added_bytes
    assert calibration["variable_tokens_per_byte"] == pytest.approx(
        expected_ratio
    )
    assert cursor._MODEL_CURSOR_STATS[model][
        "previous_request_bytes"
    ] is not None
    assert "Detected Cursor calibration for unknown-model" in capsys.readouterr().out


def test_unknown_model_probe_requires_usage(monkeypatch):
    model = "unknown-without-usage"
    monkeypatch.delitem(cursor.CURSOR_MODEL_CALIBRATION, model, raising=False)
    monkeypatch.setattr(cursor, "run", lambda *args, **kwargs: _run_result(None))

    with pytest.raises(ValueError, match="returned no usage"):
        cursor.chat_completions("key", {
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
        })


def test_cursor_ratio_uses_lower_third_and_weights_first_request():
    model = "cursor-grok-4.5-high"
    stats = {
        "previous_request_bytes": None,
        "previous_reported_cost": None,
        "accumulated_excess": 0.0,
        "ratio_samples": [],
    }
    system_tokens = cursor.CURSOR_MODEL_CALIBRATION[model][
        "system_prompt_tokens"
    ]

    cursor._record_cursor_usage(
        model,
        stats,
        10_000,
        cursor.TurnUsage(input_tokens=system_tokens + 2_000),
    )
    assert stats["ratio_samples"] == [0.2, 0.2, 0.2]
    assert stats["request_count"] == 1

    for request_bytes, uncached_tokens in (
        (12_000, 500),
        (14_000, 600),
        (16_000, 1_600),
    ):
        cursor._record_cursor_usage(
            model,
            stats,
            request_bytes,
            cursor.TurnUsage(input_tokens=uncached_tokens),
        )

    assert cursor._cursor_tokens_per_byte(model, stats) == pytest.approx(0.2)
    assert stats["request_count"] == 4


def test_cursor_rotates_before_request_when_accumulated_excess_is_high(
    monkeypatch,
):
    model = "cursor-grok-4.5-high"
    old_id = "old-conversation"
    stats = {
        "conversation_id": old_id,
        "previous_request_bytes": 50_000,
        "previous_reported_cost": 100_000.0,
        "accumulated_excess": 100_000.0,
        "request_count": 4,
        "ratio_samples": [0.2, 0.2, 0.2],
    }
    conversation_ids = []

    monkeypatch.setitem(cursor._MODEL_CURSOR_STATS, model, stats)
    monkeypatch.setattr(
        cursor,
        "run",
        lambda prompt, **kwargs: (
            conversation_ids.append(kwargs["run_config"].conversation_id)
            or _run_result(cursor.TurnUsage(input_tokens=12_000))
        ),
    )

    cursor.chat_completions("key", {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert conversation_ids[0] != old_id
    assert stats["conversation_id"] == conversation_ids[0]
    assert stats["accumulated_excess"] >= 0
    assert stats["previous_request_bytes"] is not None
    assert stats["request_count"] == 1


def test_cursor_does_not_rotate_before_four_completed_requests(monkeypatch):
    model = "cursor-grok-4.5-high"
    conversation_id = "young-conversation"
    stats = {
        "conversation_id": conversation_id,
        "previous_request_bytes": 50_000,
        "previous_reported_cost": 100_000.0,
        "accumulated_excess": 100_000.0,
        "request_count": 3,
        "ratio_samples": [0.2, 0.2, 0.2],
    }
    conversation_ids = []

    monkeypatch.setitem(cursor._MODEL_CURSOR_STATS, model, stats)
    monkeypatch.setattr(
        cursor,
        "run",
        lambda prompt, **kwargs: (
            conversation_ids.append(kwargs["run_config"].conversation_id)
            or _run_result(cursor.TurnUsage(input_tokens=12_000))
        ),
    )

    cursor.chat_completions("key", {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert conversation_ids == [conversation_id]
    assert stats["request_count"] == 4


def test_cursor_rotates_before_shrunken_request_when_predicted_cost_is_high(
    monkeypatch,
):
    model = "cursor-grok-4.5-high"
    old_id = "inflated-conversation"
    stats = {
        "conversation_id": old_id,
        "previous_request_bytes": 100_000,
        "previous_reported_cost": 80_000.0,
        "accumulated_excess": 10_000.0,
        "request_count": 4,
        "ratio_samples": [0.2, 0.2, 0.2],
    }
    conversation_ids = []

    monkeypatch.setitem(cursor._MODEL_CURSOR_STATS, model, stats)
    monkeypatch.setattr(
        cursor,
        "run",
        lambda prompt, **kwargs: (
            conversation_ids.append(kwargs["run_config"].conversation_id)
            or _run_result(cursor.TurnUsage(input_tokens=12_000))
        ),
    )

    cursor.chat_completions("key", {
        "model": model,
        "messages": [{"role": "user", "content": "rewound"}],
    })

    assert conversation_ids[0] != old_id


def test_cursor_reuses_conversation_before_threshold(monkeypatch):
    model = "cursor-grok-4.5-high"
    conversation_id = "stable-conversation"
    stats = {
        "conversation_id": conversation_id,
        "previous_request_bytes": 1_000,
        "previous_reported_cost": 1_000.0,
        "accumulated_excess": 1.0,
        "request_count": 4,
        "ratio_samples": [0.2, 0.2, 0.2],
    }
    conversation_ids = []

    monkeypatch.setitem(cursor._MODEL_CURSOR_STATS, model, stats)
    monkeypatch.setattr(
        cursor,
        "run",
        lambda prompt, **kwargs: (
            conversation_ids.append(kwargs["run_config"].conversation_id)
            or _run_result(cursor.TurnUsage(input_tokens=11_000))
        ),
    )

    cursor.chat_completions("key", {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert conversation_ids == [conversation_id]


def test_cursor_accumulates_reported_cost_excess():
    model = "cursor-grok-4.5-high"
    stats = {
        "conversation_id": "conversation",
        "previous_request_bytes": 10_000,
        "previous_reported_cost": 12_000.0,
        "accumulated_excess": 0.0,
        "ratio_samples": [0.2, 0.2, 0.2],
    }

    cursor._record_cursor_usage(
        model,
        stats,
        12_000,
        cursor.TurnUsage(
            input_tokens=30_000,
            cache_read_tokens=10_000,
        ),
    )

    expected_fresh = (
        cursor.CURSOR_MODEL_CALIBRATION[model]["system_prompt_tokens"]
        + 12_000 * 0.2
    )
    reported_cost = 20_000 + 10_000 * 0.25
    assert stats["accumulated_excess"] == pytest.approx(
        reported_cost - expected_fresh
    )


def _response_boundary_frame(request_id):
    structure = pb_message(
        pb_bytes(1, b"u" * 32),
        pb_bytes(2, b"s" * 32),
        pb_bytes(3, request_id.encode()),
        pb_varint(5, 0),
    )
    blob = pb_message(pb_bytes(1, structure))
    set_args = pb_message(pb_bytes(2, blob))
    kv = pb_message(
        pb_varint(1, 9),
        pb_bytes(3, set_args),
    )
    return cursor.ConnectFrame.from_decoded(pb_message(pb_bytes(4, kv)))


def test_response_boundary_blob_write_detection():
    request_id = "5270c3d8-821c-4c8b-bdf4-2d880504206c"
    frame = _response_boundary_frame(request_id)

    decoded = cursor.cursor_schema.decode(frame.decoded_payload, "Run_res")
    assert cursor.is_response_boundary_blob_write(decoded, request_id)
    assert not cursor.is_response_boundary_blob_write(decoded, "other-request")


def test_build_kv_response_acknowledges_set_blob():
    set_args = pb_message(
        pb_bytes(1, b"blob-id"),
        pb_bytes(2, b"blob-data"),
    )
    server = pb_message(
        pb_bytes(
            4,
            pb_message(
                pb_varint(1, 9),
                pb_bytes(3, set_args),
            ),
        )
    )

    response = cursor.build_kv_response(
        cursor.cursor_schema.decode(server, "Run_res"), {}
    )
    kv = pb_first_bytes(response, 3)

    assert pb_ints(kv, 1) == [9]
    assert pb_first_bytes(kv, 3) == b""
    assert pb_first_bytes(kv, 2) is None


def test_parse_turn_usage():
    payload = pb_message(
        pb_varint(1, 25311),
        pb_varint(2, 220),
        pb_varint(3, 12625),
        pb_varint(4, 7),
        pb_varint(5, 11),
    )

    update = cursor.cursor_schema.decode(payload, "TurnEnded")
    assert cursor.parse_turn_usage(update) == cursor.TurnUsage(
        input_tokens=25311,
        output_tokens=220,
        cache_read_tokens=12625,
        cache_write_tokens=7,
        reasoning_tokens=11,
    )

def test_build_user_cancelled_message_matches_official_payload():
    assert cursor.build_user_cancelled_message().hex() == (
        "22121a100a0e757365725f63616e63656c6c6564"
    )


def test_parse_filtered_usage_selects_earliest_event_after_request_start():
    conversation_id = "conversation-1"

    def event(timestamp, conversation, input_tokens, output_tokens, cache_read):
        token_usage = pb_message(
            pb_varint(1, input_tokens),
            pb_varint(2, output_tokens),
            pb_varint(4, cache_read),
        )
        return pb_message(
            pb_varint(1, timestamp),
            pb_varint(8, 1),
            pb_bytes(9, token_usage),
            pb_bytes(23, conversation.encode()),
        )

    response = pb_message(
        pb_bytes(3, event(10_090, conversation_id, 100, 20, 30)),
        pb_bytes(3, event(10_050, conversation_id, 200, 40, 50)),
        pb_bytes(3, event(9_990, conversation_id, 700, 70, 60)),
        pb_bytes(3, event(10_100, "other-conversation", 900, 90, 80)),
    )

    assert cursor.parse_filtered_usage(
        response, conversation_id, 10_000
    ) == cursor.TurnUsage(
        input_tokens=250,
        output_tokens=40,
        cache_read_tokens=50,
    )


def test_closed_post_ignores_stale_selector_event():
    request = object.__new__(cursor._PostRequest)
    request.finished = False
    request.sock = None
    request.connected = True
    request.tls_handshake_done = True

    request.run(selectors.EVENT_READ | selectors.EVENT_WRITE)


def test_sse_client_ignores_events_after_callback_closes_transport():
    client = object.__new__(cursor.SSEClient)
    client.closed = False
    client.timeout = None
    client.sock = type(
        "Socket",
        (),
        {
            "close": lambda self: None,
            "recv": lambda self, size: pytest.fail(
                "closed transport should not receive"
            ),
        },
    )()
    closer = type("Closer", (), {"run": lambda self, mask: client.close()})()
    key = lambda data: type("Key", (), {"data": data})()
    client.selector = type(
        "Selector",
        (),
        {
            "select": lambda self, wait: [
                (key(closer), selectors.EVENT_READ),
                (key(client), selectors.EVENT_READ),
            ],
            "unregister": lambda self, sock: None,
            "close": lambda self: None,
        },
    )()
    client._posts = set()
    client._post_pipelines = {}
    client._idle_post_connections = {}

    assert client.run_once() is False


def test_post_request_does_not_force_connection_close():
    request = object.__new__(cursor._PostRequest)
    request.parts = cursor.urlsplit("https://api.example.test/append")
    request.port = 443
    request.body = b"payload"
    request.headers = {}
    request.client = type("Client", (), {"headers": {}})()
    request.outgoing = bytearray()

    request._prepare_request()

    headers = bytes(request.outgoing).split(b"\r\n\r\n", 1)[0]
    assert b"Connection: close" not in headers


def test_post_response_returns_persistent_socket_to_pool():
    returned = []
    responses = []
    sock = type(
        "Socket",
        (),
        {"close": lambda self: pytest.fail("socket should be reused")},
    )()
    selector = type(
        "Selector",
        (),
        {"unregister": lambda self, value: None},
    )()
    client = type(
        "Client",
        (),
        {
            "_posts": set(),
            "_return_post_connection": (
                lambda self, parts, value: returned.append((parts, value))
            ),
        },
    )()
    request = object.__new__(cursor._PostRequest)
    request.finished = False
    request.response_headers = {"content-length": "0"}
    request.chunked = False
    request.content_remaining = 0
    request.sock = sock
    request.selector = selector
    request.client = client
    request.parts = cursor.urlsplit("https://api.example.test/append")
    request.status = 200
    request.response_body = bytearray()
    request.callback = responses.append
    client._posts.add(request)

    request._complete()

    assert returned == [(request.parts, sock)]
    assert request.sock is None
    assert request not in client._posts
    assert responses == [{
        "status": 200,
        "headers": {"content-length": "0"},
        "body": b"",
    }]


def test_post_pipeline_writes_eight_requests_before_responses():
    callbacks = []
    pipeline = object.__new__(cursor._PostPipeline)
    pipeline.client = type("Client", (), {"headers": {}})()
    pipeline.selector = type(
        "Selector",
        (),
        {"modify": lambda self, sock, events, handler: None},
    )()
    pipeline.parts = cursor.urlsplit("https://api.example.test/append")
    pipeline.port = 443
    pipeline.sock = object()
    pipeline.connected = True
    pipeline.tls_handshake_done = True
    pipeline.outgoing = bytearray()
    pipeline.incoming = bytearray()
    pipeline.pending = cursor.deque()
    pipeline.closed = False
    pipeline._reset_response()

    for index in range(8):
        pipeline.submit(
            pipeline.parts,
            f"payload-{index}".encode(),
            {},
            lambda response, index=index: callbacks.append(
                (index, response["status"])
            ),
        )

    assert bytes(pipeline.outgoing).count(
        b"POST /append HTTP/1.1\r\n"
    ) == 8
    assert len(pipeline.pending) == 8

    pipeline.incoming.extend(
        b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n" * 8
    )
    pipeline._parse_responses()

    assert callbacks == [(index, 200) for index in range(8)]
    assert pipeline.pending == cursor.deque()


def test_sse_grace_period_closes_transport(monkeypatch):
    client = object.__new__(cursor.SSEClient)
    client.closed = False
    client.timeout = None
    client._heartbeat_deadline = None
    client._grace_deadline = None
    client.selector = type(
        "Selector",
        (),
        {"close": lambda self: None, "unregister": lambda self, sock: None},
    )()
    client.sock = type("Socket", (), {"close": lambda self: None})()
    client._posts = set()
    client.start_grace_period(0)
    client.run_once = lambda timeout=None: pytest.fail("expired timer should not poll")

    client.run_forever()

    assert client.closed is True


def test_sse_post_blob_timeout_raises_without_polling(monkeypatch):
    client = object.__new__(cursor.SSEClient)
    client.closed = False
    client.timeout = None
    client._heartbeat_deadline = None
    client._grace_deadline = None
    client._post_blob_deadline = 10.0
    client.run_once = lambda timeout=None: pytest.fail(
        "expired post-blob timer should not poll"
    )
    monkeypatch.setattr(cursor.time, "monotonic", lambda: 10.0)

    with pytest.raises(
        cursor.SSEError, match="no model progress after blob hydration"
    ):
        client.run_forever()


def test_generation_progress_classification():
    assert cursor.is_generation_progress(decode_cursor(answer_text_frame("x")))
    assert not cursor.is_generation_progress(decode_cursor(answer_text_frame("")))
    assert cursor.is_generation_progress(
        decode_cursor(interaction_update_frame(_THINKING_DELTA_NUMBER, pb_bytes(1, b"x")))
    )
    assert cursor.is_generation_progress(
        decode_cursor(interaction_update_frame(_TOOL_CALL_COMPLETED_NUMBER))
    )
    assert not cursor.is_generation_progress(
        decode_cursor(interaction_update_frame(_HEARTBEAT_NUMBER))
    )
    assert not cursor.is_generation_progress(
        decode_cursor(interaction_update_frame(_TURN_ENDED_NUMBER))
    )



def _get_blob_frame(request_id, blob_id=b"b" * 32):
    args = pb_message(pb_bytes(1, blob_id))
    kv = pb_message(
        pb_varint(1, request_id),
        pb_bytes(2, args),
    )
    return cursor.ConnectFrame.from_decoded(pb_message(pb_bytes(4, kv))).encode()


def test_latest_blob_response_completion_arms_timeout(monkeypatch):
    armed = []
    cleared = []
    blob_callbacks = []

    class FakeSSEClient:
        def __init__(self, *args, stream_callback, headers_callback, **kwargs):
            self.stream_callback = stream_callback
            self.headers_callback = headers_callback
            self.closed = False
            self.posts = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def post(self, url, body=b"", headers=None, callback=None):
            self.posts += 1
            if self.posts == 1:
                callback({"status": 200, "headers": {}, "body": b""})
            else:
                blob_callbacks.append(callback)

        def arm_post_blob_timeout(self, timeout):
            armed.append(timeout)

        def clear_post_blob_timeout(self):
            cleared.append(True)

        def reset_heartbeat_timeout(self):
            pass

        def close(self):
            self.closed = True

        def run_forever(self, timeout=None, *, heartbeat_timeout=None):
            self.headers_callback(200, {})
            self.stream_callback(_get_blob_frame(1))
            self.stream_callback(_get_blob_frame(2, b"c" * 32))
            assert len(blob_callbacks) == 2
            blob_callbacks[0]({"status": 200, "headers": {}, "body": b""})
            assert armed == []
            blob_callbacks[1]({"status": 200, "headers": {}, "body": b""})
            assert armed == [cursor.POST_BLOB_PROGRESS_TIMEOUT]
            self.close()

    monkeypatch.setattr(cursor, "SSEClient", FakeSSEClient)
    cursor.CursorClient("token", model="composer-2.5").run("hello")

    assert len(cleared) == 2


def test_generation_progress_invalidates_pending_blob_response(monkeypatch):
    armed = []
    blob_callbacks = []

    class FakeSSEClient:
        def __init__(self, *args, stream_callback, headers_callback, **kwargs):
            self.stream_callback = stream_callback
            self.headers_callback = headers_callback
            self.closed = False
            self.posts = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def post(self, url, body=b"", headers=None, callback=None):
            self.posts += 1
            if self.posts == 1:
                callback({"status": 200, "headers": {}, "body": b""})
            else:
                blob_callbacks.append(callback)

        def arm_post_blob_timeout(self, timeout):
            armed.append(timeout)

        def clear_post_blob_timeout(self):
            pass

        def reset_heartbeat_timeout(self):
            pass

        def close(self):
            self.closed = True

        def run_forever(self, timeout=None, *, heartbeat_timeout=None):
            self.headers_callback(200, {})
            self.stream_callback(_get_blob_frame(1))
            answer = cursor.ConnectFrame.from_decoded(
                answer_text_frame("started")
            )
            self.stream_callback(answer.encode())
            blob_callbacks[0]({"status": 200, "headers": {}, "body": b""})
            assert armed == []
            self.close()

    monkeypatch.setattr(cursor, "SSEClient", FakeSSEClient)
    result = cursor.CursorClient("token", model="composer-2.5").run("hello")

    assert result.text == "started"


def test_cursor_text_boundary_starts_usage_grace(monkeypatch):
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

        def start_grace_period(self, timeout):
            self.grace_timeout = timeout

        def close(self):
            if not self.closed:
                self.closed = True
                closed.append(True)

        def run_forever(self, timeout=None, *, heartbeat_timeout=None):
            self.headers_callback(200, {})
            answer = cursor.ConnectFrame.from_decoded(
                answer_text_frame("done")
            )
            self.stream_callback(answer.encode())
            assert not hasattr(self, "grace_timeout")
            self.stream_callback(_response_boundary_frame(request_id).encode())
            assert self.grace_timeout == cursor.RESPONSE_USAGE_GRACE_TIMEOUT
            self.close()

    monkeypatch.setattr(cursor.uuid, "uuid4", lambda: request_id)
    monkeypatch.setattr(cursor, "SSEClient", FakeSSEClient)
    monkeypatch.setattr(
        cursor,
        "get_filtered_usage",
        lambda *args, **kwargs: cursor.TurnUsage(input_tokens=7),
    )
    result = cursor.CursorClient("token", model="composer-2.5").run("hello")

    assert result.text == "done"
    assert result.tool_calls == []
    assert result.turn_ended is True
    assert result.usage == cursor.TurnUsage(input_tokens=7)
    assert closed == [True]


def test_cursor_turn_ended_closes_and_records_usage(monkeypatch):
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

        def start_grace_period(self, timeout):
            pass

        def close(self):
            if not self.closed:
                self.closed = True
                closed.append(True)

        def run_forever(self, timeout=None, *, heartbeat_timeout=None):
            self.headers_callback(200, {})
            self.stream_callback(_response_boundary_frame(request_id).encode())
            update = interaction_update_frame(
                _TURN_ENDED_NUMBER,
                pb_message(
                    pb_varint(1, 25311),
                    pb_varint(2, 220),
                    pb_varint(3, 12625),
                ),
            )
            self.stream_callback(
                cursor.ConnectFrame.from_decoded(update).encode()
            )
            assert self.closed

    monkeypatch.setattr(cursor.uuid, "uuid4", lambda: request_id)
    monkeypatch.setattr(cursor, "SSEClient", FakeSSEClient)
    result = cursor.CursorClient("token", model="composer-2.5").run("hello")

    assert result.turn_ended is True
    assert result.usage == cursor.TurnUsage(
        input_tokens=25311,
        output_tokens=220,
        cache_read_tokens=12625,
    )
    assert closed == [True]


def test_cursor_tool_boundary_sends_cancellation_then_closes(monkeypatch):
    request_id = "5270c3d8-821c-4c8b-bdf4-2d880504206c"
    closed = []
    posted_payloads = []

    def tool_frame(call_id, code):
        arguments = pb_message(
            pb_bytes(1, b"repl_execute"),
            pb_bytes(
                2,
                pb_message(
                    pb_bytes(1, b"code"),
                    pb_bytes(2, pb_value(code)),
                ),
            ),
            pb_bytes(3, call_id.encode()),
            pb_bytes(5, b"repl_execute"),
        )
        execution = pb_message(
            pb_varint(1, 1),
            pb_bytes(11, arguments),
            pb_bytes(15, (call_id + "-exec").encode()),
        )
        return cursor.ConnectFrame.from_decoded(
            pb_message(pb_bytes(2, execution))
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
            payload = pb_first_bytes(body, 4)
            if payload is not None:
                posted_payloads.append(payload)
            callback({"status": 200, "headers": {}, "body": b""})

        def reset_heartbeat_timeout(self):
            pass

        def start_grace_period(self, timeout):
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
                _response_boundary_frame(request_id).encode(),
            ):
                self.stream_callback(payload)
            assert self.closed

    monkeypatch.setattr(cursor.uuid, "uuid4", lambda: request_id)
    monkeypatch.setattr(cursor, "SSEClient", FakeSSEClient)
    monkeypatch.setattr(
        cursor,
        "get_filtered_usage",
        lambda *args, **kwargs: cursor.TurnUsage(output_tokens=9),
    )
    result = cursor.CursorClient("token", model="composer-2.5").run("hello")

    assert [call.id for call in result.tool_calls] == ["call-1", "call-2"]
    assert result.turn_ended is True
    assert closed == [True]
    assert result.usage == cursor.TurnUsage(output_tokens=9)
    assert cursor.build_user_cancelled_message() in posted_payloads
