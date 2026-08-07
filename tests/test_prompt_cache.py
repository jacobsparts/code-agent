import json
import os
import subprocess
import sys

from code_agent.client import LLMClient
from code_agent.conversation import Conversation
from code_agent.repl_attachment_mixin import ImageAttachment


def test_responses_request_projects_private_image_attachment(monkeypatch):
    config = {
        "provider": "openai",
        "model": "gpt-image-capable",
        "api_key": "key",
        "api_type": "responses",
        "tool_mode": "repl_execute",
        "tools": True,
        "concurrency": 1,
        "timeout": 20,
        "port": 443,
        "host": "api.openai.com",
        "path": "/v1/responses",
        "config": {},
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("openai/gpt-image-capable")
    image = ImageAttachment(b"png-bytes", "image/png", 2, 3)

    conversation = Conversation(client, "system")
    conversation.usermsg(
        "[Attachment: diagram.png, 2×3, image/png]",
        _attachments={"diagram.png": image},
    )
    request = client._responses_request(conversation._messages(), None)

    user = request["input"][1]
    assert user["content"][0] == {
        "type": "input_text",
        "text": "[Attachment: diagram.png, 2×3, image/png]",
    }
    assert user["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,cG5nLWJ5dGVz",
    }
    assert all(
        "_media_attachments" not in item
        for item in request["input"]
    )


def test_provider_response_media_is_removed_from_conversation_history():
    message = {
        "role": "assistant",
        "content": "done",
        "images": [{"type": "image_url", "image_url": {"url": "data:..."}}],
        "audio": [b"audio"],
    }

    result = LLMClient._strip_response_media(message)

    assert result is message
    assert message == {"role": "assistant", "content": "done"}


def test_responses_request_skips_explicit_cache_when_not_opted_in(monkeypatch):
    config = {
        "provider": "openai",
        "model": "gpt-5.5",
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
        "explicit_prompt_cache": False,
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("openai/gpt-5.5")
    request = client._responses_request([
        {
            "role": "system",
            "content": "stable instructions",
            "_prompt_cache_breakpoint": True,
        },
        {
            "role": "user",
            "content": "cached-looking",
            "_prompt_cache_breakpoint": True,
        },
    ], None)

    assert request["prompt_cache_key"] == "tenant:acme:knowledge-base-v1"
    assert "prompt_cache_options" not in request
    assert "prompt_cache_breakpoint" not in request["input"][0]["content"][0]
    assert "prompt_cache_breakpoint" not in request["input"][1]["content"][0]


def test_conversation_skips_cache_breakpoints_when_not_opted_in():
    class Client:
        model_name = "openai/gpt-5.5"
        model_config = {"explicit_prompt_cache": False}

    conversation = Conversation(Client(), "system")
    conversation.usermsg("hello")
    messages = conversation._messages()
    assert all("_prompt_cache_breakpoint" not in message for message in messages)


def test_conversation_adds_cache_breakpoints_when_opted_in():
    class Client:
        model_name = "openai/gpt-5.6-luna-high"
        model_config = {"explicit_prompt_cache": True}

    conversation = Conversation(Client(), "system")
    conversation.usermsg("hello")
    messages = conversation._messages()
    assert messages[0].get("_prompt_cache_breakpoint") is True
    assert messages[1].get("_prompt_cache_breakpoint") is True


def test_openai_explicit_prompt_cache_is_gpt_5_6_only(tmp_path):
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PYTHONPATH": os.getcwd(),
        "OPENAI_API_KEY": "test-key",
    }
    code = """
import json
from code_agent.llm_registry import list_models, get_model_config
rows = []
for model in list_models():
    if model["provider"] != "openai":
        continue
    config = get_model_config(model["full_name"])
    rows.append({
        "full_name": model["full_name"],
        "model": config["model"],
        "explicit_prompt_cache": bool(config.get("explicit_prompt_cache")),
        "prompt_cache_key": config.get("config", {}).get("prompt_cache_key"),
    })
print(json.dumps(rows))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    rows = json.loads(result.stdout)
    for row in rows:
        enabled = row["model"].startswith("gpt-5.6")
        assert row["explicit_prompt_cache"] is enabled
        if enabled:
            assert row["prompt_cache_key"] == "jp-code-agent-001"
        else:
            assert row["prompt_cache_key"] is None
