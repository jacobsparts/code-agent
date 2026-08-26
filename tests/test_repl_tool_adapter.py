import pytest

from code_agent.repl_tool_adapter import (
    ReplExecuteResponseError,
    project_repl_tool_history,
    repl_response_to_text,
)


def text(value):
    return {"type": "text", "text": value}


def repl_call(code):
    return {
        "type": "tool_call",
        "id": "provider-call",
        "name": "repl_execute",
        "args": {"code": code},
    }


def test_transport_repl_response_preserves_executable_block_order():
    result = repl_response_to_text({
        "role": "assistant",
        "content": [
            text("working"),
            {"type": "commentary", "text": "checking"},
            repl_call("x = 1"),
            repl_call("print(x)"),
        ],
    })

    source = result["content"][0]["text"]
    assert source.index("emit(") < source.index("# checking")
    assert source.index("# checking") < source.index("x = 1")
    assert source.index("x = 1") < source.index("print(x)")


def test_transport_repl_response_requires_python_without_a_call():
    assert repl_response_to_text({
        "role": "assistant",
        "content": [text("x = 1")],
    })["content"] == [text("x = 1")]

    with pytest.raises(ReplExecuteResponseError):
        repl_response_to_text({
            "role": "assistant",
            "content": [text("not Python prose")],
        })


def message(role, value, **metadata):
    return {
        "role": role,
        "content": value if isinstance(value, list) else [text(value)],
        **metadata,
    }


def project_history(messages):
    assert all(isinstance(item.get("content"), list) for item in messages)
    return project_repl_tool_history(messages)


def test_repl_history_uses_stable_synthetic_calls_and_results():
    projected = project_history([
        message("system", "system"),
        message("user", "task"),
        message("assistant", "x = 1", _private=True),
        message("user", ">>> x = 1\n", _stdout="ignored"),
        message("assistant", "print(x)"),
    ])

    calls = [
        message["content"][0]
        for message in projected
        if message["role"] == "assistant"
    ]
    results = [
        message for message in projected if message["role"] == "tool"
    ]
    assert [call["id"] for call in calls] == ["repl_000001", "repl_000002"]
    assert [result["tool_call_id"] for result in results] == [
        call["id"] for call in calls
    ]
    assert calls[0]["args"]["code"] == "x = 1"
    assert results[-1]["content"] == [text("")]


def test_repl_history_recovers_appended_user_input():
    projected = project_history([
        message("assistant", "emit('done', release=True)"),
        message(
            "user",
            ">>> emit('done', release=True)\ndone\nnext task\n",
            _user_content="next task",
            _render_segments=[
                {
                    "type": "stdout",
                    "content": ">>> emit('done', release=True)\ndone\n",
                },
                {"type": "input", "content": "next task"},
            ],
        ),
    ])

    assert projected[1]["content"][0]["text"].endswith("done\n")
    assert projected[2]["content"] == [text("next task")]


def test_repl_history_uses_preview_instead_of_raw_output():
    preview = "[PreviewRef: session://preview/example]\n[/PreviewRef]\n"
    projected = project_history([
        message("assistant", "print(large_value)"),
        message(
            "user",
            preview,
            _stdout="x" * 200_000,
            _render_segments=[
                {"type": "stdout", "content": "x" * 200_000},
            ],
        ),
    ])

    assert projected[1]["content"] == [text(preview)]


def test_repl_history_preserves_release_output_before_user_input():
    output = (
        '>>> emit("done", release=True)\n'
        'done\n'
    )
    projected = project_history([
        message("assistant", 'emit("done", release=True)'),
        message(
            "user",
            output + "hi\n",
            _user_content="hi",
            _render_segments=[
                {"type": "stdout", "content": output},
                {"type": "input", "content": "hi"},
            ],
        ),
    ])

    assert projected[1]["content"] == [text(output)]
    assert projected[2]["content"] == [text("hi")]


def test_repl_history_strips_echoed_preceding_user_input():
    projected = project_history([
        message(
            "user",
            "What's today's date?\n",
            _user_content="What's today's date?",
            _render_segments=[
                {"type": "input", "content": "What's today's date?"},
            ],
        ),
        message("assistant", "print('2026-07-13')"),
        message(
            "user",
            (
                "What's today's date?\n"
                ">>> print('2026-07-13')\n2026-07-13\n"
            ),
            _render_segments=[{
                "type": "stdout",
                "content": (
                    "What's today's date?\n"
                    ">>> print('2026-07-13')\n2026-07-13\n"
                ),
            }],
        ),
    ])

    assert projected[2]["content"] == [
        text(">>> print('2026-07-13')\n2026-07-13\n")
    ]


def test_repl_history_preserves_attachments_on_recovered_user_input():
    attachment = {
        "type": "attachment",
        "media_type": "image/png",
        "data_type": "bytes",
        "data": b"image",
    }
    projected = project_repl_tool_history([
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "print('done')"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "done\nnext task\n"},
                attachment,
            ],
            "_user_content": "next task",
            "_render_segments": [
                {"type": "stdout", "content": "done\n"},
                {"type": "input", "content": "next task"},
            ],
        },
    ])

    assert projected[2]["content"] == [
        {"type": "text", "text": "next task"},
        attachment,
    ]


def test_repl_history_ignores_obsolete_provider_metadata():
    preview = "[PreviewRef: session://preview/example]\n[/PreviewRef]\n"
    projected = project_history([
        message(
            "assistant",
            "print(large_value)",
            _tool_call_ids=["provider-id"],
            _repl_execute_calls=[{
                "id": "provider-id",
                "code": "different_code()",
            }],
            _tool_call_outputs=["raw output"],
        ),
        message("user", preview, _stdout="raw output"),
    ])

    assert projected[0]["content"][0]["id"] == "repl_000001"
    assert projected[0]["content"][0]["args"]["code"] == "print(large_value)"
    assert projected[1]["content"] == [text(preview)]
