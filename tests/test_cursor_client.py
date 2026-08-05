
import json
import os
import selectors
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
    monkeypatch.setattr("code_agent.utils.get_model_config", lambda name: config)
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
    result = client._call([{"role": "user", "content": "hello"}])
    assert captured["api_key"] == "key"
    assert captured["body"]["model"] == "composer-2.5"
    assert captured["body"]["tools"] == [REPL_EXECUTE_TOOL]
    assert "timeout" not in captured["body"]
    assert throttled == [("cursor", 17)]
    assert result["content"] == "emit('ok', release=True)"
    assert result["_stop_reason"] == "tool_calls"
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


def test_build_run_request_keeps_stable_conversation_id(monkeypatch):
    monkeypatch.setattr(cursor, "_SESSION_CONVERSATION_ID", "stable-conversation")

    first = cursor.build_run_request(
        "one", "composer-2.5", message_id="message-1"
    )
    second = cursor.build_run_request(
        "two", "composer-2.5", message_id="message-2"
    )
    first_run = cursor.RawMessage.decode(
        cursor.RawMessage.decode(first).first_bytes(1)
    )
    second_run = cursor.RawMessage.decode(
        cursor.RawMessage.decode(second).first_bytes(1)
    )

    assert cursor._text(first_run, 5) == "stable-conversation"
    assert cursor._text(second_run, 5) == "stable-conversation"
    assert not first_run.has(16)
    assert not second_run.has(16)


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
    run = cursor.RawMessage.decode(
        cursor.RawMessage.decode(payload).first_bytes(1)
    )

    assert cursor._text(run, 5) == "explicit-conversation"
    assert cursor._text(run, 16) == "explicit-group"


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


def test_response_boundary_blob_write_detection():
    request_id = "5270c3d8-821c-4c8b-bdf4-2d880504206c"
    frame = _response_boundary_frame(request_id)

    assert cursor.is_response_boundary_blob_write(
        frame.decoded_payload, request_id
    )
    assert not cursor.is_response_boundary_blob_write(
        frame.decoded_payload, "other-request"
    )


def test_build_kv_response_acknowledges_set_blob():
    set_args = cursor.protobuf_message(
        cursor.Field.bytes(1, b"blob-id"),
        cursor.Field.bytes(2, b"blob-data"),
    )
    server = cursor.protobuf_message(
        cursor.Field.bytes(
            4,
            cursor.protobuf_message(
                cursor.Field.varint(1, 9),
                cursor.Field.bytes(3, set_args),
            ),
        )
    )

    response = cursor.RawMessage.decode(
        cursor.build_kv_response(server, {})
    )
    kv = cursor.RawMessage.decode(response.first_bytes(3))

    assert cursor._int(kv, 1) == 9
    assert kv.first_bytes(3) == b""
    assert kv.first_bytes(2) is None


def test_parse_turn_usage():
    payload = cursor.protobuf_message(
        cursor.Field.varint(1, 25311),
        cursor.Field.varint(2, 220),
        cursor.Field.varint(3, 12625),
        cursor.Field.varint(4, 7),
        cursor.Field.varint(5, 11),
    )

    assert cursor.parse_turn_usage(payload) == cursor.TurnUsage(
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
        token_usage = cursor.protobuf_message(
            cursor.Field.varint(1, input_tokens),
            cursor.Field.varint(2, output_tokens),
            cursor.Field.varint(4, cache_read),
        )
        return cursor.protobuf_message(
            cursor.Field.varint(1, timestamp),
            cursor.Field.varint(8, 1),
            cursor.Field.bytes(9, token_usage),
            cursor.Field.bytes(23, conversation.encode()),
        )

    response = cursor.protobuf_message(
        cursor.Field.bytes(
            3, event(10_090, conversation_id, 100, 20, 30)
        ),
        cursor.Field.bytes(
            3, event(10_050, conversation_id, 200, 40, 50)
        ),
        cursor.Field.bytes(
            3, event(9_990, conversation_id, 700, 70, 60)
        ),
        cursor.Field.bytes(
            3, event(10_100, "other-conversation", 900, 90, 80)
        ),
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
    assert cursor.is_generation_progress(cursor.AnswerText.create("x"))
    assert not cursor.is_generation_progress(cursor.AnswerText.create(""))
    assert cursor.is_generation_progress(
        cursor.InteractionUpdate.create("thinking_delta", b"x")
    )
    assert cursor.is_generation_progress(
        cursor.InteractionUpdate.create("tool_call_completed")
    )
    assert not cursor.is_generation_progress(
        cursor.InteractionUpdate.create("heartbeat")
    )
    assert not cursor.is_generation_progress(
        cursor.InteractionUpdate.create("turn_ended")
    )



def _get_blob_frame(request_id, blob_id=b"b" * 32):
    args = cursor.protobuf_message(cursor.Field.bytes(1, blob_id))
    kv = cursor.protobuf_message(
        cursor.Field.varint(1, request_id),
        cursor.Field.bytes(2, args),
    )
    return cursor.ConnectFrame.from_decoded(
        cursor.protobuf_message(cursor.Field.bytes(4, kv))
    ).encode()


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
                cursor.AnswerText.create("started").encode()
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
                cursor.AnswerText.create("done").encode()
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
            update = cursor.InteractionUpdate.create(
                "turn_ended",
                cursor.protobuf_message(
                    cursor.Field.varint(1, 25311),
                    cursor.Field.varint(2, 220),
                    cursor.Field.varint(3, 12625),
                ),
            )
            self.stream_callback(
                cursor.ConnectFrame.from_decoded(update.encode()).encode()
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
            bidi = cursor.RawMessage.decode(body)
            payload = bidi.first_bytes(4)
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
