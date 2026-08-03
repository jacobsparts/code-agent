"""Codex OAuth transport for the ChatGPT Codex Responses endpoint."""

from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from io import BytesIO
import tempfile
import urllib.error
import urllib.request
import uuid

CRED_FILE = os.path.expanduser("~/.code-agent/codex-auth.json")
REFRESH_URL = "https://auth.openai.com/oauth/token"
RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CLIENT_VERSION = "0.146.0"
MODEL = "gpt-5.6-luna"
DEFAULT_TIMEOUT = 300
REFRESH_WINDOW_MINUTES = 5
MAX_REFRESH_AGE_DAYS = 8


def _uuid7() -> str:
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    random_bits = int.from_bytes(os.urandom(10), "big")
    value = (
        (timestamp_ms & ((1 << 48) - 1)) << 80
        | 0x7 << 76
        | ((random_bits >> 62) & 0xFFF) << 64
        | 0b10 << 62
        | (random_bits & ((1 << 62) - 1))
    )
    return str(uuid.UUID(int=value))


SESSION_ID = _uuid7()


class CodexError(Exception):
    pass


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _jwt_payload(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        payload = json.loads(_b64url_decode(parts[1]))
        return payload if isinstance(payload, dict) else {}
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def jwt_exp(token: str) -> datetime | None:
    value = _jwt_payload(token).get("exp")
    if not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


class CodexAuth:
    def __init__(self, path: str = CRED_FILE):
        self.path = os.path.expanduser(path)
        self.data = self._load()

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as stream:
                data = json.load(stream)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No Codex credentials at {self.path}. Copy the official "
                "~/.codex/auth.json file or create the equivalent token schema."
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CodexError(f"Could not read Codex credentials at {self.path}: {exc}") from exc
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if not isinstance(tokens, dict):
            raise CodexError("Codex credentials are missing the tokens object")
        for name in ("access_token", "refresh_token"):
            if not isinstance(tokens.get(name), str) or not tokens[name]:
                raise CodexError(f"Codex credentials are missing {name}")
        return data

    def _save(self) -> None:
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, mode=0o700, exist_ok=True)
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        fd, temporary = tempfile.mkstemp(prefix=".codex-auth-", dir=directory)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.data, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    @property
    def access_token(self) -> str:
        return self.data["tokens"]["access_token"]

    @property
    def account_id(self) -> str:
        value = self.data["tokens"].get("account_id")
        if isinstance(value, str) and value:
            return value
        auth = _jwt_payload(self.access_token).get("https://api.openai.com/auth", {})
        return auth.get("chatgpt_account_id", "") if isinstance(auth, dict) else ""

    def needs_refresh(self) -> bool:
        expiration = jwt_exp(self.access_token)
        now = datetime.now(timezone.utc)
        if expiration is not None:
            return expiration <= now + timedelta(minutes=REFRESH_WINDOW_MINUTES)
        value = self.data.get("last_refresh")
        if isinstance(value, str):
            try:
                refreshed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if refreshed.tzinfo is None:
                    refreshed = refreshed.replace(tzinfo=timezone.utc)
                return refreshed < now - timedelta(days=MAX_REFRESH_AGE_DAYS)
            except ValueError:
                pass
        return False

    def refresh(self, timeout: float = 60) -> None:
        request = urllib.request.Request(
            REFRESH_URL,
            data=json.dumps({
                "client_id": CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": self.data["tokens"]["refresh_token"],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise CodexError(f"Token refresh failed: HTTP {exc.code}: {detail}") from exc
        except (OSError, ValueError) as exc:
            raise CodexError(f"Token refresh failed: {exc}") from exc
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise CodexError("Token refresh returned no access_token")
        tokens = self.data["tokens"]
        tokens["access_token"] = access_token
        for name in ("refresh_token", "id_token"):
            if isinstance(payload.get(name), str) and payload[name]:
                tokens[name] = payload[name]
        self.data["last_refresh"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def ensure_valid(self) -> None:
        if self.needs_refresh():
            self.refresh()


def _event_result(events: list[dict]) -> dict:
    response = None
    text = []
    calls = {}
    call_order = []
    aliases = {}
    pending_arguments = {}

    def find_call(*identifiers):
        for identifier in identifiers:
            if identifier is not None and identifier in calls:
                return identifier
            if identifier is not None and identifier in aliases:
                return aliases[identifier]
        return None

    def add_call(item):
        item = dict(item)
        identifiers = (item.get("id"), item.get("call_id"))
        key = find_call(*identifiers)
        if key is None:
            key = next((value for value in identifiers if value), None)
            key = key or "call_" + uuid.uuid4().hex
            calls[key] = {"type": "function_call"}
            call_order.append(key)
        existing_arguments = calls[key].get("arguments", "")
        calls[key].update(item)
        calls[key]["type"] = "function_call"
        if not calls[key].get("arguments") and existing_arguments:
            calls[key]["arguments"] = existing_arguments
        for identifier in identifiers:
            if identifier is not None:
                aliases[identifier] = key
        for identifier in identifiers:
            if identifier in pending_arguments:
                calls[key]["arguments"] = pending_arguments.pop(identifier)
                break
        calls[key].setdefault("call_id", str(item.get("call_id") or key))
        calls[key].setdefault("name", "")
        calls[key].setdefault("arguments", "")
        return key

    def append_arguments(identifier, delta):
        key = find_call(identifier)
        if key is None:
            pending_arguments[identifier] = pending_arguments.get(identifier, "") + delta
        else:
            calls[key]["arguments"] = str(calls[key].get("arguments") or "") + delta

    def set_arguments(identifier, arguments):
        key = find_call(identifier)
        if key is None:
            pending_arguments[identifier] = arguments
        else:
            calls[key]["arguments"] = arguments

    for event in events:
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            text.append(str(event.get("delta", "")))
        elif event_type in ("response.output_item.added", "response.output_item.done"):
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                add_call(item)
        elif event_type == "response.function_call_arguments.delta":
            identifier = event.get("item_id") or event.get("call_id")
            if identifier is not None:
                append_arguments(identifier, str(event.get("delta", "")))
        elif event_type == "response.function_call_arguments.done":
            identifier = event.get("item_id") or event.get("call_id")
            arguments = event.get("arguments")
            if identifier is not None and isinstance(arguments, str):
                set_arguments(identifier, arguments)
        if event_type in ("response.completed", "response.incomplete", "response.failed"):
            value = event.get("response")
            if isinstance(value, dict):
                response = dict(value)

    response = response or {}
    output = response.get("output")
    if not isinstance(output, list):
        output = []

    merged_output = []
    seen_calls = set()
    for item in output:
        if isinstance(item, dict) and item.get("type") == "function_call":
            key = find_call(item.get("id"), item.get("call_id"))
            if key is not None:
                merged = dict(item)
                tracked = calls[key]
                for name, value in tracked.items():
                    if name not in merged or (not merged[name] and value):
                        merged[name] = value
                if tracked.get("arguments"):
                    merged["arguments"] = tracked["arguments"]
                item = merged
                seen_calls.add(key)
        merged_output.append(item)

    if not merged_output and text:
        merged_output.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "".join(text)}],
        })
    for key in call_order:
        if key not in seen_calls:
            merged_output.append(calls[key])
    response["output"] = merged_output
    response.setdefault("status", "completed")
    return response


def parse_sse(stream) -> dict:
    events = []
    data_lines = []
    for raw_line in stream:
        line = raw_line.decode("utf-8", "strict").rstrip("\r\n")
        if line:
            if line.startswith("data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(" ") else value)
            continue
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        data_lines.clear()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise CodexError(f"Malformed Codex SSE event: {exc}") from exc
        if isinstance(event, dict):
            events.append(event)
    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise CodexError(f"Malformed Codex SSE event: {exc}") from exc
            if isinstance(event, dict):
                events.append(event)
    return _event_result(events)


def _headers(auth: CodexAuth) -> dict[str, str]:
    headers = {
        "Authorization": "Bearer " + auth.access_token,
        "Content-Type": "application/json",
        "Openai-Beta": "responses=experimental",
        "Originator": "codex_cli_rs",
        "User-Agent": f"codex_cli_rs/{CLIENT_VERSION}",
        "Session_id": SESSION_ID,
        "Version": CLIENT_VERSION,
        "Accept": "text/event-stream",
    }
    if auth.account_id:
        headers["ChatGPT-Account-ID"] = auth.account_id
    return headers


def _parse_response_body(content_type: str, payload: bytes) -> dict:
    content_type = content_type.lower()
    if "text/event-stream" in content_type:
        return parse_sse(BytesIO(payload))
    if "json" in content_type:
        try:
            result = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CodexError(f"Malformed Codex JSON response: {exc}") from exc
        if not isinstance(result, dict):
            raise CodexError("Codex JSON response is not an object")
        return result
    if not content_type:
        # The Codex endpoint can omit Content-Type on an otherwise valid SSE
        # response.  Prefer JSON when the body is clearly JSON, then fall back
        # to the event parser used by the normal streaming response.
        if payload.lstrip().startswith(b"{"):
            try:
                result = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                result = None
            if isinstance(result, dict):
                return result
        return parse_sse(BytesIO(payload))
    raise CodexError(f"Unexpected Codex Content-Type: {content_type!r}")


def _request(auth: CodexAuth, body: dict, timeout: float) -> dict:
    encoded = json.dumps(body, separators=(",", ":")).encode()
    for attempt in range(2):
        auth.ensure_valid()
        request = urllib.request.Request(
            RESPONSES_URL,
            data=encoded,
            headers=_headers(auth),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                return _parse_response_body(content_type, response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if exc.code == 401 and attempt == 0:
                auth.refresh()
                continue
            raise CodexError(f"Codex request failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise CodexError(f"Codex request failed: {exc.reason}") from exc
    raise CodexError("Codex request failed after token refresh")


def responses(
    body: dict,
    auth: CodexAuth | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    if not isinstance(body, dict):
        raise TypeError("body must be a dictionary")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("model is required")
    if not isinstance(body.get("input"), list) or not body["input"]:
        raise ValueError("input must be a non-empty list")
    request_body = dict(body)
    request_body.pop("prompt_cache_options", None)
    request_body["input"] = copy.deepcopy(body["input"])
    request_body.setdefault("instructions", "")
    for item in request_body["input"]:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "system":
            item["role"] = "developer"
        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("prompt_cache_breakpoint", None)
    request_body["stream"] = True
    request_body.setdefault("store", False)
    request_body["tool_choice"] = "auto"
    request_body.setdefault("parallel_tool_calls", True)
    request_body.setdefault("prompt_cache_key", SESSION_ID)
    request_body.setdefault("include", ["reasoning.encrypted_content"])
    return _request(auth or CodexAuth(), request_body, timeout)
