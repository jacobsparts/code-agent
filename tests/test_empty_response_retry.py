import pytest

from code_agent import client as client_module


@pytest.fixture
def make_client(monkeypatch):
    monkeypatch.setattr(
        client_module,
        "get_model_config",
        lambda _name: {
            "timeout": 1,
            "concurrency": 1,
            "tools": False,
            "api_type": "completions",
        },
    )

    def factory():
        c = client_module.LLMClient("test/model")
        c.provider_admission = None
        sleeps = []
        c._sleep_backoff = lambda attempt, base=15: sleeps.append(attempt)
        return c, sleeps

    return factory


def test_empty_response_is_retried(make_client):
    c, sleeps = make_client()
    calls = []

    def flaky(messages, tools):
        calls.append(1)
        if len(calls) == 1:
            return {"role": "assistant", "content": [{"type": "text", "text": "   "}]}
        return {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}

    c._call_completions = flaky
    result = c.call(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], retry=2
    )
    assert result["content"] == [{"type": "text", "text": "hello"}]
    assert len(calls) == 2
    assert sleeps == [0]


def test_persistent_empty_response_raises_after_retries(make_client):
    c, sleeps = make_client()
    calls = []

    def always_empty(messages, tools):
        calls.append(1)
        return {"role": "assistant", "content": []}

    c._call_completions = always_empty
    with pytest.raises(client_module.EmptyResponseError):
        c.call([], retry=1)
    assert len(calls) == 2
    assert sleeps == [0]


def test_tool_call_only_response_is_not_empty(make_client):
    c, _ = make_client()

    def tool_only(messages, tools):
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "x",
                    "name": "repl_execute",
                    "args": {"code": "emit(1)"},
                },
            ],
        }

    c._call_completions = tool_only
    result = c.call([])
    assert result["content"][0]["name"] == "repl_execute"
