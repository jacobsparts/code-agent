import json

import pytest

from code_agent.repl_tool_adapter import (
    REPL_EXECUTE_TOOL,
    ReplExecuteResponseError,
    normalize_openai_repl_response,
    project_openai_repl_messages,
)


def call(code, name="repl_execute"):
    return {
        "id": "provider-id",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"code": code})},
    }


def test_tool_call_normalizes_to_python():
    assert normalize_openai_repl_response({
        "role": "assistant", "content": None, "tool_calls": [call("x = 1")]
    }) == {
        "role": "assistant",
        "content": "x = 1",
        "_tool_call_ids": ["provider-id"],
    }


def test_text_and_multiple_calls_are_joined_safely():
    result = normalize_openai_repl_response({
        "role": "assistant",
        "content": "I'll inspect it.",
        "tool_calls": [call("x = 1"), call("print(x)")],
    })
    assert result["content"] == "emit(\"I'll inspect it.\")\nx = 1\nprint(x)"
    assert result["_tool_call_ids"] == ["provider-id", "provider-id"]


def test_raw_python_and_text_fallbacks():
    assert normalize_openai_repl_response({
        "role": "assistant", "content": "x = 1"
    })["content"] == "x = 1"
    text = 'A "quoted" answer\nwith Unicode: café'
    assert normalize_openai_repl_response({
        "role": "assistant", "content": text
    })["content"] == f"emit({text!r}, release=True)"


@pytest.mark.parametrize("tool_call", [
    call("x = 1", name="other"),
    {"function": {"name": "repl_execute", "arguments": "{"}},
    {"function": {"name": "repl_execute", "arguments": "{}"}},
    {"function": {"name": "repl_execute", "arguments": '{"code": 1}'}},
])
def test_invalid_calls_raise(tool_call):
    with pytest.raises(ReplExecuteResponseError):
        normalize_openai_repl_response({
            "role": "assistant", "content": None, "tool_calls": [tool_call]
        })


def test_projection_uses_stable_calls_and_results():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "x = 1", "_private": True},
        {"role": "user", "content": ">>> x = 1\n", "_stdout": ">>> x = 1\n"},
        {"role": "assistant", "content": "print(x)"},
    ]
    result = project_openai_repl_messages(messages)
    assert result[2]["tool_calls"][0]["id"] == "repl_000001"
    assert json.loads(result[2]["tool_calls"][0]["function"]["arguments"]) == {"code": "x = 1"}
    assert result[3] == {
        "role": "tool", "tool_call_id": "repl_000001", "content": ">>> x = 1\n"
    }
    assert result[4]["tool_calls"][0]["id"] == "repl_000002"
    assert result[5] == {
        "role": "tool", "tool_call_id": "repl_000002", "content": ""
    }


def test_projection_reuses_provider_tool_call_id():
    messages = [
        {
            "role": "assistant",
            "content": "x = 1",
            "_tool_call_ids": ["provider-call-123"],
        },
        {"role": "user", "content": ">>> x = 1\n", "_stdout": ">>> x = 1\n"},
    ]

    result = project_openai_repl_messages(messages)

    assert result[0]["tool_calls"][0]["id"] == "provider-call-123"
    assert result[1]["tool_call_id"] == "provider-call-123"


def test_projection_splits_appended_human_input():
    messages = [
        {"role": "assistant", "content": "emit('done', release=True)"},
        {
            "role": "user",
            "content": ">>> emit('done', release=True)\ndone\nnext task\n",
            "_stdout": ">>> emit('done', release=True)\ndone\nnext task\n",
            "_user_content": "next task",
            "_render_segments": [
                {"type": "stdout", "content": ">>> emit('done', release=True)\ndone\n"},
                {"type": "input", "content": "next task"},
            ],
        },
    ]
    result = project_openai_repl_messages(messages)
    assert result[1]["role"] == "tool"
    assert result[1]["content"].endswith("done\n")
    assert result[2] == {"role": "user", "content": "next task"}


def test_projection_preserves_synthetic_release_output_before_first_real_input():
    final_output = (
        '>>> emit("Today is Friday, July 17, 2026.", release=True)\n'
        'Today is Friday, July 17, 2026.\n'
    )
    messages = [
        {"role": "assistant", "content": 'emit("Today is Friday, July 17, 2026.", release=True)'},
        {
            "role": "user",
            "content": final_output + "hi\n",
            "_stdout": final_output + "hi\n",
            "_user_content": "hi",
            "_render_segments": [
                {"type": "stdout", "content": final_output},
                {"type": "input", "content": "hi"},
            ],
        },
    ]

    result = project_openai_repl_messages(messages)

    assert result[1] == {
        "role": "tool",
        "tool_call_id": "repl_000001",
        "content": final_output,
    }
    assert result[2] == {"role": "user", "content": "hi"}


def test_projection_strips_echoed_preceding_human_input_from_tool_result():
    messages = [
        {
            "role": "user",
            "content": "What's today's date?\n",
            "_user_content": "What's today's date?",
            "_render_segments": [
                {"type": "input", "content": "What's today's date?"},
            ],
        },
        {"role": "assistant", "content": "print('2026-07-13')"},
        {
            "role": "user",
            "content": "What's today's date?\n>>> print('2026-07-13')\n2026-07-13\n",
            "_stdout": "What's today's date?\n>>> print('2026-07-13')\n2026-07-13\n",
            "_render_segments": [{
                "type": "stdout",
                "content": "What's today's date?\n>>> print('2026-07-13')\n2026-07-13\n",
            }],
        },
    ]

    result = project_openai_repl_messages(messages)

    assert result[2] == {
        "role": "tool",
        "tool_call_id": "repl_000001",
        "content": ">>> print('2026-07-13')\n2026-07-13\n",
    }


def test_schema_is_exactly_one_function():
    assert REPL_EXECUTE_TOOL["function"]["name"] == "repl_execute"
    assert REPL_EXECUTE_TOOL["function"]["parameters"]["additionalProperties"] is False
