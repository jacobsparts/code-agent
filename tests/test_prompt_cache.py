import json
import os
import subprocess
import sys

import pytest

from code_agent.client import LLMClient, legacy_to_transport_messages
from code_agent.conversation import Conversation, MEDIA_ATTACHMENTS_FIELD
from code_agent.repl_attachment_mixin import AudioAttachment, ImageAttachment


def test_mixed_legacy_attachments_preserve_order_into_completions(monkeypatch):
    config = {
        "model": "media-model",
        "api_type": "completions",
        "concurrency": 1,
        "timeout": 20,
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("test/media-model")
    messages = legacy_to_transport_messages([{
        "role": "user",
        "content": [
            {"type": "text", "text": "first"},
            {"type": "input_file", "file_id": "file_123"},
            {
                "type": "image_url",
                "media_type": "image/png",
                "image_url": {"url": "https://example.test/image.png"},
            },
            {
                "type": "input_audio",
                "input_audio": {"data": "YXVkaW8=", "format": "wav"},
            },
            {"type": "text", "text": "last"},
        ],
        MEDIA_ATTACHMENTS_FIELD: [{
            "media_type": "image/png",
            "content": b"binary-image",
        }],
    }])

    assert [
        (block["type"], block.get("data_type"))
        for block in messages[0]["content"]
    ] == [
        ("text", None),
        ("attachment", "provider_id"),
        ("attachment", "url"),
        ("attachment", "bytes"),
        ("text", None),
        ("attachment", "bytes"),
    ]
    assert messages[0]["content"][3]["data"] == b"audio"
    assert messages[0]["content"][-1]["data"] == b"binary-image"

    projected = client._completions_messages(messages)[0]["content"]
    assert [block["type"] for block in projected] == [
        "text", "input_file", "image_url", "input_audio", "text", "image_url",
    ]


def test_unknown_legacy_and_transport_types_raise(monkeypatch):
    with pytest.raises(NotImplementedError, match="Unknown legacy content type"):
        legacy_to_transport_messages([{
            "role": "user",
            "content": [{"type": "future_media"}],
        }])

    config = {
        "model": "media-model",
        "api_type": "completions",
        "concurrency": 1,
        "timeout": 20,
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("test/media-model")
    with pytest.raises(NotImplementedError, match="Unknown transport content type"):
        client._completions_messages([{
            "role": "user",
            "content": [{"type": "future_transport"}],
        }])


def test_unknown_responses_output_types_raise(monkeypatch):
    config = {
        "model": "media-model",
        "api_type": "responses",
        "concurrency": 1,
        "timeout": 20,
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("test/media-model")

    with pytest.raises(NotImplementedError, match="Unknown Responses output type"):
        client._parse_responses_result({
            "output": [{"type": "future_output"}],
        })


def test_anthropic_and_gemini_project_binary_and_filepath_attachments(
    monkeypatch, tmp_path
):
    path = tmp_path / "image.png"
    path.write_bytes(b"file-image")
    attachment = {
        "type": "attachment",
        "media_type": "image/png",
        "data_type": "bytes",
        "data": b"image",
    }

    config = {
        "model": "media-model",
        "api_type": "messages",
        "concurrency": 1,
        "timeout": 20,
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    anthropic = LLMClient("test/media-model")
    assert anthropic._anthropic_attachment(attachment)["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "aW1hZ2U=",
    }

    config["api_type"] = "gemini"
    gemini = LLMClient("test/media-model")
    attachment = {
        **attachment,
        "data_type": "filepath",
        "data": str(path),
    }
    assert gemini._gemini_attachment(attachment) == {
        "inlineData": {
            "mimeType": "image/png",
            "data": "ZmlsZS1pbWFnZQ==",
        },
    }


def test_responses_attachment_output_decodes_to_transport(monkeypatch):
    config = {
        "model": "media-model",
        "api_type": "responses",
        "concurrency": 1,
        "timeout": 20,
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("test/media-model")

    result = client._parse_responses_result({
        "status": "completed",
        "output": [{
            "type": "message",
            "content": [{
                "type": "input_file",
                "file_id": "file_result",
                "media_type": "application/pdf",
            }],
        }],
    })

    assert result["content"] == [{
        "type": "attachment",
        "media_type": "application/pdf",
        "data_type": "provider_id",
        "data": "file_result",
    }]


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
    request = client._responses_request(
        legacy_to_transport_messages(conversation._messages()), None
    )

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


class _Response:
    status = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()


class _Connection:
    requests = []
    response_payload = {}

    def __init__(self, *args, **kwargs):
        self.sock = self

    def connect(self):
        pass

    def setsockopt(self, *args):
        pass

    def request(self, method, path, body, headers):
        self.requests.append(json.loads(body))

    def getresponse(self):
        return _Response(self.response_payload)

    def close(self):
        pass


@pytest.mark.parametrize(
    ("api_type", "response_payload", "expected"),
    [
        (
            "completions",
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            {
                "type": "input_audio",
                "input_audio": {"data": "UklGRgQAAABXQVZFZGF0YQ==", "format": "wav"},
            },
        ),
        (
            "gemini",
            {
                "candidates": [{
                    "content": {"parts": [{"text": "ok"}]},
                    "finishReason": "STOP",
                }],
            },
            {
                "inlineData": {
                    "mimeType": "audio/wav",
                    "data": "UklGRgQAAABXQVZFZGF0YQ==",
                },
            },
        ),
    ],
)
def test_audio_attachment_transport_payload(
    monkeypatch, api_type, response_payload, expected
):
    config = {
        "model": "audio-model",
        "api_key": "key",
        "api_type": api_type,
        "tool_mode": None,
        "tools": False,
        "concurrency": 1,
        "timeout": 20,
        "port": 443,
        "host": "example.test",
        "path": "/v1",
        "config": {},
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    monkeypatch.setattr("code_agent.client.DeadlineHTTPSConnection", _Connection)
    _Connection.requests = []
    _Connection.response_payload = response_payload
    client = LLMClient("test/audio-model")
    media = {
        "name": "clip.wav",
        "media_type": "audio/wav",
        "content": b"RIFF\x04\x00\x00\x00WAVEdata",
    }

    client._call([{
        "role": "user",
        "content": "listen",
        MEDIA_ATTACHMENTS_FIELD: [media],
    }])

    request = _Connection.requests[0]
    if api_type == "completions":
        assert request["messages"][0]["content"][1] == expected
    else:
        assert request["contents"][0]["parts"][1] == expected


@pytest.mark.parametrize(
    ("api_type", "projector"),
    [
        ("responses", "_responses_attachment"),
        ("messages", "_anthropic_attachment"),
        ("cursor", "_openai_attachment"),
        ("codex", "_responses_attachment"),
    ],
)
def test_transport_rejects_unsupported_audio_generically(
    monkeypatch, api_type, projector
):
    config = {
        "model": "text-model",
        "api_type": api_type,
        "concurrency": 1,
        "timeout": 20,
    }
    monkeypatch.setattr("code_agent.client.get_model_config", lambda name: config)
    client = LLMClient("test/text-model")
    attachment = {
        "type": "attachment",
        "media_type": "audio/wav",
        "data_type": "bytes",
        "data": b"audio",
    }

    with pytest.raises(NotImplementedError):
        getattr(client, projector)(attachment)


def test_openai_response_image_decodes_then_legacy_projection_raises():
    from code_agent.client import (
        _openai_compatible_message_to_transport_blocks,
        transport_to_legacy_message,
    )

    blocks = _openai_compatible_message_to_transport_blocks({
        "role": "assistant",
        "content": "done",
        "images": [{
            "type": "image_url",
            "image_url": {"url": "https://example.test/result.png"},
        }],
    })

    assert blocks[1] == {
        "type": "attachment",
        "media_type": None,
        "data_type": "url",
        "data": "https://example.test/result.png",
    }
    with pytest.raises(NotImplementedError, match="cannot store attachment"):
        transport_to_legacy_message({
            "role": "assistant",
            "content": blocks,
        })


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
            "content": [{"type": "text", "text": "stable instructions"}],
            "_prompt_cache_breakpoint": True,
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "cached-looking"}],
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
