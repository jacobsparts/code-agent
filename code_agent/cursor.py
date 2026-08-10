"""Minimal synchronous Cursor AgentService client using HTTP/1.1 compatibility RPC.

Single-file stdlib-only library: protobuf wire codec, Connect/SSE transport,
and stateless Cursor Agent client.

Runtime-reported native tools for composer-2.5:
    ExecServerMessage, Shell, Grep, Delete, WebSearch, WebFetch,
    GenerateImage, ReadLints, EditNotebook, TodoWrite, StrReplace, Write,
    Read, Glob, AskQuestion, Task, Await, ListMcpResources,
    FetchMcpResource, SwitchMode.

This model-reported inventory is distinct from the complete protobuf
ExecServerMessage oneof enumeration in EXEC_SERVER_TOOL_FIELDS.
"""

from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass, field
import errno
import fcntl
import gzip
import hashlib
import json
import os
import selectors
import shlex
import tempfile
import socket
import ssl
import struct
import sys
import time
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping
import uuid
from urllib.parse import urljoin, urlsplit
from urllib.error import HTTPError
from urllib.request import Request, urlopen


# === Configuration globals ===

DEFAULT_BASE_URL = "https://api2.cursor.sh"
DEFAULT_CLIENT_VERSION = "cli-2026.07.08-0c04a8a"
DEFAULT_TIMEOUT = 30 * 60
KEY_EXCHANGE_TIMEOUT = 30
HEARTBEAT_TIMEOUT = 30
POST_BLOB_PROGRESS_TIMEOUT = 30
BIDI_APPEND_PIPELINE_DEPTH = 8
RESPONSE_USAGE_GRACE_TIMEOUT = 3
USAGE_LOOKUP_ATTEMPTS = 4
USAGE_LOOKUP_RETRY_DELAY = 1
USAGE_LOOKUP_WINDOW_MS = 5_000

CURSOR_MODEL_CALIBRATION = {
    "composer-2.5": {
        "system_prompt_tokens": 10_752,
        "variable_tokens_per_byte": 0.205204021289178,
        "input_cost": 0.5,
        "cache_read_cost": 0.2,
    },
    "cursor-grok-4.5-high": {
        "system_prompt_tokens": 10_960,
        "variable_tokens_per_byte": 0.23536369012418687,
        "input_cost": 2.0,
        "cache_read_cost": 0.5,
    },
    "kimi-k3-high": {
        "system_prompt_tokens": 15_042,
        "variable_tokens_per_byte": 0.205204021289178,
        "input_cost": 3.0,
        "cache_read_cost": 0.3,
    },
}

_CURSOR_RATIO_MIN_BYTES = 1_000
_CURSOR_RATIO_MIN = 0.05
_CURSOR_RATIO_MAX = 1.0
_CURSOR_RATIO_SAMPLE_LIMIT = 32
_CURSOR_ROTATION_MIN_REQUESTS = 4
_CURSOR_CALIBRATION_SAMPLE = 'from collections import defaultdict\nfrom pathlib import Path\n\n\ndef summarize_python_files(root):\n    grouped = defaultdict(list)\n    for path in Path(root).rglob("*.py"):\n        if "__pycache__" in path.parts or path.name.startswith("."):\n            continue\n        try:\n            source = path.read_text(encoding="utf-8")\n        except (OSError, UnicodeDecodeError):\n            continue\n        grouped[str(path.parent)].append({\n            "path": str(path),\n            "bytes": len(source.encode("utf-8")),\n            "lines": len(source.splitlines()),\n        })\n    return dict(grouped)\n'
_CURSOR_DEFAULT_INPUT_COST = 0.5
_CURSOR_DEFAULT_CACHE_READ_COST = 0.2
_MODEL_CURSOR_STATS = {}

DEFAULT_AGENT_MODE = 1
KEY_EXCHANGE_URL = "https://api2.cursor.sh/auth/exchange_user_api_key"
AUTH_CACHE_PATH = os.path.expanduser("~/.code-agent/cursor-auth.json")
AUTH_LOCK_PATH = os.path.expanduser("~/.code-agent/cursor-auth.lock")
ACCESS_TOKEN_LIFETIME = 60 * 60
ACCESS_TOKEN_REFRESH_MARGIN = 5 * 60
AGENT_RUNSSE_PATH = "agent.v1.AgentService/RunSSE"
BIDI_APPEND_PATH = "aiserver.v1.BidiService/BidiAppend"
FILTERED_USAGE_PATH = "aiserver.v1.DashboardService/GetFilteredUsageEvents"
AVAILABLE_MODELS_PATH = "aiserver.v1.AiService/AvailableModels"

# Cursor uses this identifier for inference routing/cache affinity.
# A code-agent process represents one conversation session.
_SESSION_CONVERSATION_ID = str(uuid.uuid4())

DEBUG = True


def _debug_bidi_event(event: str, **details) -> None:
    if not DEBUG:
        return
    record = {
        "timestamp_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "event": event,
        **details,
    }
    path = f"/tmp/coda-cursor-bidi-{_SESSION_CONVERSATION_ID[:8]}.jsonl"
    try:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        pass


# === Protobuf wire codec ===




UINT64_MAX = (1 << 64) - 1
MAX_FIELD_NUMBER = (1 << 29) - 1
RESERVED_FIELD_RANGE = range(19000, 20000)

NATIVE_EXEC_FIELD_NAMES = {
    2: "shell_args", 3: "write_args", 4: "delete_args", 5: "grep_args",
    7: "read_args", 8: "ls_args", 9: "diagnostics_args",
    10: "request_context_args", 14: "shell_stream_args",
    16: "background_shell_spawn_args", 17: "list_mcp_resources_exec_args",
    18: "read_mcp_resource_exec_args", 20: "fetch_args",
    21: "record_screen_args", 22: "computer_use_args",
    23: "write_shell_stdin_args", 27: "execute_hook_args",
    28: "subagent_args", 29: "redacted_read_args",
    30: "force_background_shell_args", 31: "force_background_subagent_args",
    36: "mcp_state_exec_args", 37: "subagent_await_args",
    38: "smart_mode_classifier_args", 40: "canvas_diagnostics_args",
    41: "shell_allowlist_precheck_args", 42: "mcp_allowlist_precheck_args",
    43: "web_fetch_allowlist_precheck_args", 44: "git_diff_request",
    45: "pi_read_args", 46: "pi_bash_args", 47: "pi_edit_args",
    48: "pi_write_args", 49: "pi_grep_args", 50: "pi_find_args",
    51: "pi_ls_args", 53: "conversation_search_args",
}

INTERACTION_UPDATE_FIELD_NAMES = {
    1: "text_delta", 2: "tool_call_started", 3: "tool_call_completed",
    4: "thinking_delta", 5: "thinking_completed",
    6: "user_message_appended", 7: "partial_tool_call", 8: "token_delta",
    9: "summary", 10: "summary_started", 11: "summary_completed",
    12: "shell_output_delta", 13: "heartbeat", 14: "turn_ended",
    15: "tool_call_delta", 16: "step_started", 17: "step_completed",
    18: "prompt_suggestion", 19: "post_request_prompt",
    20: "active_branch_change", 21: "feedback_request",
    22: "response_comparison",
}

def _validate_field_number(number: int) -> None:
    if not isinstance(number, int):
        raise TypeError("field number must be int")
    if not 1 <= number <= MAX_FIELD_NUMBER:
        raise ValueError(f"field number must be 1..{MAX_FIELD_NUMBER}")
    # Reserved numbers are accepted: this is a schema-free wire library, and
    # rejecting them would prevent lossless handling of otherwise valid bytes.


def encode_varint(value: int) -> bytes:
    if not isinstance(value, int):
        raise TypeError("varint must be int")
    if not 0 <= value <= UINT64_MAX:
        raise ValueError("varint must fit uint64")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7f) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _read_varint(data: bytes, offset: int) -> tuple[int, int, bytes]:
    start = offset
    value = 0
    for index in range(10):
        if offset >= len(data):
            raise ValueError("truncated varint")
        byte = data[offset]
        offset += 1
        if index == 9 and byte > 1:
            raise ValueError("varint exceeds uint64")
        value |= (byte & 0x7f) << (index * 7)
        if byte < 0x80:
            return value, offset, data[start:offset]
    raise ValueError("unterminated varint exceeds 10 bytes")


@dataclass(frozen=True, slots=True)
class Field:
    number: int
    wire_type: int
    value: int | bytes
    _tag_bytes: bytes | None = field(default=None, repr=False, compare=False)
    _value_bytes: bytes | None = field(default=None, repr=False, compare=False)
    _length_bytes: bytes | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_field_number(self.number)
        if self.wire_type not in (0, 1, 2, 5):
            raise ValueError(f"unsupported wire type {self.wire_type}")
        if self.wire_type == 2:
            if not isinstance(self.value, bytes):
                raise TypeError("length-delimited value must be bytes")
        elif not isinstance(self.value, int):
            raise TypeError("numeric wire value must be int")
        elif self.wire_type == 0 and not 0 <= self.value <= UINT64_MAX:
            raise ValueError("varint value must fit uint64")
        elif self.wire_type == 1 and not 0 <= self.value < (1 << 64):
            raise ValueError("fixed64 value must fit uint64")
        elif self.wire_type == 5 and not 0 <= self.value < (1 << 32):
            raise ValueError("fixed32 value must fit uint32")

    @classmethod
    def varint(cls, number: int, value: int) -> "Field":
        return cls(number, 0, value)

    @classmethod
    def fixed64(cls, number: int, value: int) -> "Field":
        return cls(number, 1, value)

    @classmethod
    def bytes(cls, number: int, value: bytes) -> "Field":
        return cls(number, 2, value)

    @classmethod
    def fixed32(cls, number: int, value: int) -> "Field":
        return cls(number, 5, value)

    def replacing(self, *, number: int | None = None, value: int | bytes | None = None) -> "Field":
        return Field(
            self.number if number is None else number,
            self.wire_type,
            self.value if value is None else value,
        )

    def encode(self) -> bytes:
        tag = self._tag_bytes or encode_varint((self.number << 3) | self.wire_type)
        if self.wire_type == 0:
            return tag + (self._value_bytes or encode_varint(self.value))
        if self.wire_type == 1:
            return tag + (self._value_bytes or struct.pack("<Q", self.value))
        if self.wire_type == 5:
            return tag + (self._value_bytes or struct.pack("<I", self.value))
        return tag + (self._length_bytes or encode_varint(len(self.value))) + self.value

    def nested(self) -> "RawMessage":
        if self.wire_type != 2:
            raise TypeError("only length-delimited fields can be decoded as nested")
        return RawMessage.decode(self.value)

    def __repr__(self) -> str:
        value = (
            "b''" if self.value == b"" else
            f"0x{self.value.hex()}" if isinstance(self.value, bytes) else
            repr(self.value)
        )
        canonical = Field(self.number, self.wire_type, self.value).encode()
        exact = self.encode()
        suffix = "" if exact == canonical else f", encoded=0x{exact.hex()}"
        return f"Field({self.number}, {self.wire_type}, value={value}{suffix})"


@dataclass(frozen=True, slots=True)
class RawMessage:
    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple):
            object.__setattr__(self, "fields", tuple(self.fields))

    @classmethod
    def decode(cls, data: bytes) -> "RawMessage":
        if not isinstance(data, bytes):
            raise TypeError("protobuf input must be bytes")
        fields = []
        offset = 0
        while offset < len(data):
            tag, offset, tag_bytes = _read_varint(data, offset)
            number, wire_type = tag >> 3, tag & 7
            _validate_field_number(number)
            if wire_type == 0:
                value, offset, raw = _read_varint(data, offset)
                fields.append(Field(number, 0, value, tag_bytes, raw))
            elif wire_type == 1:
                end = offset + 8
                if end > len(data):
                    raise ValueError("truncated fixed64")
                raw = data[offset:end]
                fields.append(Field(number, 1, struct.unpack("<Q", raw)[0], tag_bytes, raw))
                offset = end
            elif wire_type == 2:
                length, offset, length_bytes = _read_varint(data, offset)
                end = offset + length
                if end > len(data):
                    raise ValueError("truncated length-delimited value")
                fields.append(Field(number, 2, data[offset:end], tag_bytes, None, length_bytes))
                offset = end
            elif wire_type == 5:
                end = offset + 4
                if end > len(data):
                    raise ValueError("truncated fixed32")
                raw = data[offset:end]
                fields.append(Field(number, 5, struct.unpack("<I", raw)[0], tag_bytes, raw))
                offset = end
            else:
                raise ValueError(f"unsupported wire type {wire_type}")
        return cls(tuple(fields))

    @classmethod
    def canonical(cls, fields: Iterable[Field]) -> "RawMessage":
        return cls(tuple(Field(item.number, item.wire_type, item.value) for item in fields))

    def replacing(self, index: int, new_field: Field) -> "RawMessage":
        fields = list(self.fields)
        fields[index] = new_field
        return RawMessage(tuple(fields))

    def encode(self) -> bytes:
        return b"".join(item.encode() for item in self.fields)

    def matching(self, number: int, wire_type: int | None = None) -> tuple[Field, ...]:
        return tuple(
            item for item in self.fields
            if item.number == number and (wire_type is None or item.wire_type == wire_type)
        )

    def first_bytes(self, number: int) -> bytes | None:
        fields = self.matching(number, 2)
        return fields[0].value if fields else None

    def has(self, number: int, wire_type: int | None = None) -> bool:
        return bool(self.matching(number, wire_type))

    def __repr__(self) -> str:
        return f"RawMessage(fields={list(self.fields)!r})"


def protobuf_field(number: int, wire_type: int, value: int | bytes) -> Field:
    return Field(number, wire_type, value)


def protobuf_message(*fields: Field) -> bytes:
    return RawMessage(tuple(fields)).encode()


@dataclass(frozen=True, slots=True)
class ConnectFrame:
    """A Connect frame whose wire_payload is exactly the bytes after its header."""

    flags: int
    wire_payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.flags, int) or not 0 <= self.flags <= 0xff:
            raise ValueError("flags must be an integer in 0..255")
        if not isinstance(self.wire_payload, bytes):
            raise TypeError("wire_payload must be bytes")

    @property
    def compressed(self) -> bool:
        return bool(self.flags & 1)

    @property
    def eos(self) -> bool:
        return bool(self.flags & 2)

    @property
    def decoded_payload(self) -> bytes:
        return gzip.decompress(self.wire_payload) if self.compressed else self.wire_payload

    @classmethod
    def from_decoded(cls, payload: bytes, *, flags: int = 0, compress: bool = False) -> "ConnectFrame":
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if compress:
            flags |= 1
            return cls(flags, gzip.compress(payload, mtime=0))
        if flags & 1:
            raise ValueError("compressed flag requires compress=True or precompressed wire_payload")
        return cls(flags, payload)

    def encode(self) -> bytes:
        return bytes((self.flags,)) + len(self.wire_payload).to_bytes(4, "big") + self.wire_payload

    def __repr__(self) -> str:
        return f"ConnectFrame(flags=0x{self.flags:02x}, wire_payload=0x{self.wire_payload.hex()})"


def decode_connect_frames(data: bytes) -> tuple[ConnectFrame, ...]:
    if not isinstance(data, bytes):
        raise TypeError("Connect stream must be bytes")
    frames = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < 5:
            raise ValueError("truncated Connect frame header")
        flags = data[offset]
        length = int.from_bytes(data[offset + 1:offset + 5], "big")
        end = offset + 5 + length
        if end > len(data):
            raise ValueError("truncated Connect frame payload")
        frames.append(ConnectFrame(flags, data[offset + 5:end]))
        offset = end
    return tuple(frames)


def encode_connect_frame(payload: bytes, flags: int = 0, compress: bool = False) -> bytes:
    return ConnectFrame.from_decoded(payload, flags=flags, compress=compress).encode()


@dataclass(frozen=True, slots=True)
class SemanticField:
    """A visible, self-contained semantic replacement for one wire token."""

    number: int
    wire_type: int
    name: str
    value: object
    tag_bytes: bytes | None = None
    value_bytes: bytes | None = None
    length_bytes: bytes | None = None

    @classmethod
    def from_field(cls, name: str, value: object, source: Field) -> "SemanticField":
        canonical_tag = encode_varint((source.number << 3) | source.wire_type)
        canonical_value = _semantic_wire_value(value, source.wire_type)
        canonical_value_bytes = (
            encode_varint(canonical_value) if source.wire_type == 0 else
            struct.pack("<Q", canonical_value) if source.wire_type == 1 else
            struct.pack("<I", canonical_value) if source.wire_type == 5 else None
        )
        canonical_length = (
            encode_varint(len(canonical_value)) if source.wire_type == 2 else None
        )
        return cls(
            source.number,
            source.wire_type,
            name,
            value,
            source._tag_bytes if source._tag_bytes != canonical_tag else None,
            source._value_bytes if source._value_bytes != canonical_value_bytes else None,
            source._length_bytes if source._length_bytes != canonical_length else None,
        )

    def encode(self) -> bytes:
        value = _semantic_wire_value(self.value, self.wire_type)
        tag = self.tag_bytes or encode_varint((self.number << 3) | self.wire_type)
        if self.wire_type == 0:
            return tag + (self.value_bytes or encode_varint(value))
        if self.wire_type == 1:
            return tag + (self.value_bytes or struct.pack("<Q", value))
        if self.wire_type == 5:
            return tag + (self.value_bytes or struct.pack("<I", value))
        return tag + (self.length_bytes or encode_varint(len(value))) + value

    def __repr__(self) -> str:
        metadata = ""
        if self.tag_bytes is not None:
            metadata += f", tag=0x{self.tag_bytes.hex()}"
        if self.value_bytes is not None:
            metadata += f", value_encoding=0x{self.value_bytes.hex()}"
        if self.length_bytes is not None:
            metadata += f", length=0x{self.length_bytes.hex()}"
        return (
            f"SemanticField({self.number}, {self.wire_type}, "
            f"{self.name}={self.value!r}{metadata})"
        )


@dataclass(frozen=True, slots=True)
class NestedField:
    """A visible, self-contained length-delimited semantic envelope."""

    number: int
    content: "MessageRepresentation"
    tag_bytes: bytes | None = None
    length_bytes: bytes | None = None

    @classmethod
    def from_field(
        cls, source: Field, content: "MessageRepresentation",
    ) -> "NestedField":
        canonical_tag = encode_varint((source.number << 3) | 2)
        canonical_length = encode_varint(len(source.value))
        return cls(
            source.number,
            content,
            source._tag_bytes if source._tag_bytes != canonical_tag else None,
            source._length_bytes if source._length_bytes != canonical_length else None,
        )

    def encode(self) -> bytes:
        value = self.content.encode()
        tag = self.tag_bytes or encode_varint((self.number << 3) | 2)
        length = self.length_bytes or encode_varint(len(value))
        return tag + length + value

    def __repr__(self) -> str:
        metadata = ""
        if self.tag_bytes is not None:
            metadata += f", tag=0x{self.tag_bytes.hex()}"
        if self.length_bytes is not None:
            metadata += f", length=0x{self.length_bytes.hex()}"
        return f"NestedField({self.number}, {self.content!r}{metadata})"


@dataclass(frozen=True, slots=True)
class MessageRepresentation:
    """Ordered lossless semantic/raw token stream."""

    parts: tuple[Field | SemanticField | NestedField, ...]

    def encode(self) -> bytes:
        return b"".join(part.encode() for part in self.parts)

    def __repr__(self) -> str:
        return f"MessageRepresentation({list(self.parts)!r})"


def _semantic_wire_value(value: object, wire_type: int) -> int | bytes:
    if wire_type == 2:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8", "surrogateescape")
        raise TypeError("length-delimited semantic value must be bytes or str")
    if not isinstance(value, int):
        raise TypeError("numeric semantic value must be int")
    return value


def _field_index(raw: RawMessage, number: int, wire_type: int | None = None) -> int | None:
    for index, item in enumerate(raw.fields):
        if item.number == number and (wire_type is None or item.wire_type == wire_type):
            return index
    return None


def _representation(
    raw: RawMessage,
    replacements: Mapping[tuple[int, ...], tuple[str, object]],
) -> MessageRepresentation:
    parts: list[Field | SemanticField | NestedField] = []
    for index, item in enumerate(raw.fields):
        direct = replacements.get((index,))
        children = {
            path[1:]: replacement
            for path, replacement in replacements.items()
            if len(path) > 1 and path[0] == index
        }
        if direct is not None:
            parts.append(SemanticField.from_field(direct[0], direct[1], item))
        elif children and item.wire_type == 2:
            nested = RawMessage.decode(item.value)
            content = _representation(nested, children)
            parts.append(NestedField.from_field(item, content))
        else:
            parts.append(item)
    return MessageRepresentation(tuple(parts))


@dataclass(frozen=True, slots=True, repr=False)
class CursorMessage:
    raw: RawMessage
    classification: str
    direction: str

    def encode(self) -> bytes:
        return self.raw.encode()

    @property
    def all_fields(self) -> tuple[Field, ...]:
        return self.raw.fields

    def representation(self) -> MessageRepresentation:
        replacements: dict[tuple[int, ...], tuple[str, object]] = {}

        if isinstance(self, AnswerText):
            outer = _field_index(self.raw, 1, 2)
            one = _decode_or_none(self.raw.fields[outer].value) if outer is not None else None
            middle = _field_index(one, 1, 2) if one is not None else None
            two = (
                _decode_or_none(one.fields[middle].value)
                if one is not None and middle is not None else None
            )
            leaf = _field_index(two, 1, 2) if two is not None else None
            if None not in (outer, middle, leaf):
                replacements[(outer, middle, leaf)] = ("text", self.text)

        elif isinstance(self, NativeExec):
            outer = _field_index(self.raw, 2, 2)
            execution = (
                _decode_or_none(self.raw.fields[outer].value)
                if outer is not None else None
            )
            leaf = (
                _field_index(execution, self.field_number, 2)
                if execution is not None else None
            )
            if outer is not None and leaf is not None:
                replacements[(outer, leaf)] = (
                    self.subtype,
                    self.arguments_payload,
                )

        elif isinstance(self, InteractionUpdate) and self.subtype_number:
            outer = _field_index(self.raw, 1, 2)
            interaction = (
                _decode_or_none(self.raw.fields[outer].value)
                if outer is not None else None
            )
            leaf = (
                _field_index(interaction, self.subtype_number)
                if interaction is not None else None
            )
            if (
                outer is not None
                and leaf is not None
                and interaction.fields[leaf].wire_type == 2
            ):
                replacements[(outer, leaf)] = (
                    self.subtype,
                    self.update_payload,
                )

        elif isinstance(self, LiveMCPCall):
            outer = _field_index(self.raw, 2, 2)
            execution = (
                _decode_or_none(self.raw.fields[outer].value)
                if outer is not None else None
            )
            if outer is not None and execution is not None:
                server_id = _field_index(execution, 1, 0)
                args_index = _field_index(execution, 11, 2)
                execution_id = _field_index(execution, 15, 2)
                if server_id is not None:
                    replacements[(outer, server_id)] = (
                        "server_message_id", self.server_message_id,
                    )
                if execution_id is not None:
                    replacements[(outer, execution_id)] = (
                        "execution_id", self.execution_id,
                    )
                if args_index is not None:
                    args = _decode_or_none(execution.fields[args_index].value)
                    if args is not None:
                        semantic_args = (
                            (1, "name", self.name),
                            (3, "tool_call_id", self.tool_call_id),
                            (4, "provider_identifier", self.provider_identifier),
                            (5, "tool_name", self.tool_name),
                            (9, "server_identifier", self.server_identifier),
                        )
                        for number, name, value in semantic_args:
                            index = _field_index(args, number, 2)
                            if index is not None:
                                replacements[(outer, args_index, index)] = (
                                    name, value,
                                )

        elif isinstance(self, CompletedMCPUpdate):
            path_numbers = (1, 2, 2, 15, 1)
            current = self.raw
            path: list[int] = []
            for depth, number in enumerate(path_numbers):
                index = _field_index(current, number, 2)
                if index is None:
                    break
                path.append(index)
                if depth < len(path_numbers) - 1:
                    current = _decode_or_none(current.fields[index].value)
                    if current is None:
                        break
            if len(path) == len(path_numbers):
                args = _decode_or_none(current.fields[path[-1]].value)
                if args is not None:
                    semantic_args = (
                        (1, "name", self.name),
                        (3, "tool_call_id", self.tool_call_id),
                        (4, "provider_identifier", self.provider_identifier),
                        (5, "tool_name", self.tool_name),
                    )
                    for number, name, value in semantic_args:
                        index = _field_index(args, number, 2)
                        if index is not None:
                            replacements[tuple(path + [index])] = (name, value)

        return _representation(self.raw, replacements)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"classification={self.classification!r}, "
            f"direction={self.direction!r}, "
            f"content={self.representation()!r})"
        )


def parse_message_repr(text: str) -> MessageRepresentation:
    """Parse a CursorMessage repr and reconstruct its visible representation."""

    tree = ast.parse(text, mode="eval")

    def hex_bytes(node: ast.AST) -> bytes:
        if not isinstance(node, ast.Constant) or not isinstance(node.value, int):
            raise ValueError("encoding metadata must be a hexadecimal integer")
        source = ast.get_source_segment(text, node)
        if source is None or not source.lower().startswith("0x"):
            raise ValueError("encoding metadata must use hexadecimal notation")
        token = source[2:].replace("_", "")
        if len(token) % 2:
            token = "0" + token
        return bytes.fromhex(token)

    def parse_node(node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return tuple(parse_node(item) for item in node.elts)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError("unsupported repr syntax")
        name = node.func.id
        keywords = {item.arg: item.value for item in node.keywords}
        if name not in {
            "MessageRepresentation", "Field", "SemanticField", "NestedField",
        }:
            if "content" not in keywords:
                raise ValueError("message repr has no content")
            result = parse_node(keywords["content"])
            if not isinstance(result, MessageRepresentation):
                raise ValueError("message content is not a representation")
            return result
        if name == "MessageRepresentation":
            if len(node.args) != 1:
                raise ValueError("invalid MessageRepresentation repr")
            return MessageRepresentation(tuple(parse_node(node.args[0])))
        if name == "Field":
            if len(node.args) != 2 or "value" not in keywords:
                raise ValueError("invalid Field repr")
            number = ast.literal_eval(node.args[0])
            wire_type = ast.literal_eval(node.args[1])
            value_node = keywords["value"]
            if (
                wire_type == 2 and isinstance(value_node, ast.Constant)
                and isinstance(value_node.value, int)
            ):
                value = hex_bytes(value_node)
            else:
                value = ast.literal_eval(value_node)
            encoded = keywords.get("encoded")
            if encoded is None:
                return Field(number, wire_type, value)
            exact = RawMessage.decode(hex_bytes(encoded))
            if len(exact.fields) != 1:
                raise ValueError("Field encoded metadata is not one field")
            return exact.fields[0]
        if name == "SemanticField":
            if len(node.args) != 2 or len(keywords) < 1:
                raise ValueError("invalid SemanticField repr")
            semantic = [
                (key, value) for key, value in keywords.items()
                if key not in {"tag", "value_encoding", "length"}
            ]
            if len(semantic) != 1:
                raise ValueError("SemanticField requires one named value")
            semantic_name, value_node = semantic[0]
            return SemanticField(
                ast.literal_eval(node.args[0]),
                ast.literal_eval(node.args[1]),
                semantic_name,
                ast.literal_eval(value_node),
                hex_bytes(keywords["tag"]) if "tag" in keywords else None,
                hex_bytes(keywords["value_encoding"])
                if "value_encoding" in keywords else None,
                hex_bytes(keywords["length"]) if "length" in keywords else None,
            )
        if len(node.args) != 2:
            raise ValueError("invalid NestedField repr")
        content = parse_node(node.args[1])
        if not isinstance(content, MessageRepresentation):
            raise ValueError("NestedField content is not a representation")
        return NestedField(
            ast.literal_eval(node.args[0]),
            content,
            hex_bytes(keywords["tag"]) if "tag" in keywords else None,
            hex_bytes(keywords["length"]) if "length" in keywords else None,
        )

    result = parse_node(tree.body)
    if not isinstance(result, MessageRepresentation):
        raise ValueError("repr did not contain a message representation")
    return result


@dataclass(frozen=True, slots=True, repr=False)
class NativeExec(CursorMessage):
    field_number: int
    subtype: str
    arguments_payload: bytes


@dataclass(frozen=True, slots=True, repr=False)
class LiveMCPCall(CursorMessage):
    server_message_id: int
    execution_id: bytes
    tool_call_id: str
    name: str
    provider_identifier: str
    tool_name: str
    server_identifier: str
    arguments_raw: bytes


@dataclass(frozen=True, slots=True, repr=False)
class CompletedMCPUpdate(CursorMessage):
    tool_call_id: str
    name: str
    provider_identifier: str
    tool_name: str
    arguments_raw: bytes


@dataclass(frozen=True, slots=True, repr=False)
class AnswerText(CursorMessage):
    text: str

    @classmethod
    def create(cls, text: str) -> "AnswerText":
        leaf = protobuf_message(Field.bytes(1, text.encode()))
        middle = protobuf_message(Field.bytes(1, leaf))
        raw = RawMessage((Field.bytes(1, middle),))
        return cls(raw, "agent_server.answer_text", "IN", text)


@dataclass(frozen=True, slots=True, repr=False)
class InteractionUpdate(CursorMessage):
    subtype_number: int
    subtype: str
    update_payload: bytes

    @classmethod
    def create(cls, subtype: str, payload: bytes = b"") -> "InteractionUpdate":
        reverse = {name: number for number, name in INTERACTION_UPDATE_FIELD_NAMES.items()}
        number = reverse[subtype]
        interaction = protobuf_message(Field.bytes(number, payload))
        raw = RawMessage((Field.bytes(1, interaction),))
        return cls(raw, f"agent_server.interaction_update.{subtype}", "IN",
                   number, subtype, payload)


@dataclass(frozen=True, slots=True, repr=False)
class AgentExecMessage(CursorMessage):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class CheckpointUpdate(CursorMessage):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class KVServerMessage(CursorMessage):
    subtype: str


@dataclass(frozen=True, slots=True, repr=False)
class ExecControlMessage(CursorMessage):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class InteractionQuery(CursorMessage):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class RunRequest(CursorMessage):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class ClientExecMessage(CursorMessage):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class KVResponse(CursorMessage):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class Control(CursorMessage):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class ClientHeartbeat(CursorMessage):
    @classmethod
    def create(cls) -> "ClientHeartbeat":
        raw = RawMessage((Field.bytes(7, b""),))
        return cls(raw, "agent_client.heartbeat", "OUT")


def _decode_or_none(data: bytes | None) -> RawMessage | None:
    if data is None:
        return None
    try:
        return RawMessage.decode(data)
    except ValueError:
        return None


def _nested(raw: RawMessage | None, number: int) -> RawMessage | None:
    return _decode_or_none(raw.first_bytes(number)) if raw is not None else None


def _decode_text(data: bytes | None) -> str:
    return "" if data is None else data.decode("utf-8", "surrogateescape")


def _first_varint(raw: RawMessage, number: int) -> int:
    fields = raw.matching(number, 0)
    return fields[0].value if fields else 0


def _answer_decode_text(raw: RawMessage) -> str:
    one = _nested(raw, 1)
    two = _nested(one, 1)
    value = two.first_bytes(1) if two else None
    return _decode_text(value) if value else ""


def _live_mcp(raw: RawMessage) -> dict[str, object] | None:
    execution = _nested(raw, 2)
    if execution is None:
        return None
    args_raw = execution.first_bytes(11)
    execution_id = execution.first_bytes(15)
    args = _decode_or_none(args_raw)
    if args is None or not execution_id:
        return None
    name = args.first_bytes(1) or args.first_bytes(5)
    call_id = args.first_bytes(3)
    if not name or not call_id:
        return None
    return {
        "server_message_id": _first_varint(execution, 1),
        "execution_id": execution_id,
        "tool_call_id": _decode_text(call_id),
        "name": _decode_text(name),
        "provider_identifier": _decode_text(args.first_bytes(4)),
        "tool_name": _decode_text(args.first_bytes(5) or name),
        "server_identifier": _decode_text(args.first_bytes(9)),
        "arguments_raw": args_raw,
    }


def _completed_mcp(raw: RawMessage) -> dict[str, object] | None:
    update = _nested(raw, 1)
    completed = _nested(update, 2)
    tool = _nested(completed, 2)
    mcp = _nested(tool, 15)
    args_raw = mcp.first_bytes(1) if mcp else None
    args = _decode_or_none(args_raw)
    if args is None:
        return None
    name = args.first_bytes(1) or args.first_bytes(5)
    call_id = args.first_bytes(3)
    if not name or not call_id:
        return None
    return {
        "tool_call_id": _decode_text(call_id),
        "name": _decode_text(name),
        "provider_identifier": _decode_text(args.first_bytes(4)),
        "tool_name": _decode_text(args.first_bytes(5) or name),
        "arguments_raw": args_raw,
    }


def classify(raw: RawMessage, direction: str) -> CursorMessage:
    direction = direction.upper()
    if direction not in ("IN", "OUT"):
        raise ValueError("direction must be IN or OUT")
    if direction == "OUT":
        # Go intentionally tests nonempty byte values for fields 1/2/3/5,
        # while heartbeat field 7 is presence-based.
        for number, kind, name in (
            (1, RunRequest, "agent_client.run_request"),
            (2, ClientExecMessage, "agent_client.exec_message"),
            (3, KVResponse, "agent_client.kv_response"),
            (5, Control, "agent_client.control"),
        ):
            value = raw.first_bytes(number)
            if value is not None and len(value) > 0:
                return kind(raw, name, direction)
        if raw.has(7):
            return ClientHeartbeat(raw, "agent_client.heartbeat", direction)
        return CursorMessage(raw, "unknown", direction)

    execution = _nested(raw, 2)
    if execution is not None:
        for item in execution.fields:
            if item.wire_type == 2 and item.number in NATIVE_EXEC_FIELD_NAMES:
                subtype = NATIVE_EXEC_FIELD_NAMES[item.number]
                return NativeExec(raw, f"agent_server.native_exec.{subtype}", direction,
                                  item.number, subtype, item.value)
    live = _live_mcp(raw)
    if live:
        return LiveMCPCall(raw, f"agent_server.mcp_exec.{live['name']}", direction, **live)
    completed = _completed_mcp(raw)
    if completed:
        return CompletedMCPUpdate(
            raw, f"agent_server.completed_mcp_update.{completed['name']}",
            direction, **completed,
        )
    answer = _answer_decode_text(raw)
    if answer:
        return AnswerText(raw, "agent_server.answer_text", direction, answer)
    value = raw.first_bytes(1)
    if value is not None and len(value) > 0:
        interaction = _decode_or_none(value)
        if interaction is not None:
            for item in interaction.fields:
                if item.number in INTERACTION_UPDATE_FIELD_NAMES:
                    subtype = INTERACTION_UPDATE_FIELD_NAMES[item.number]
                    payload = item.value if item.wire_type == 2 else item.encode()
                    return InteractionUpdate(
                        raw, f"agent_server.interaction_update.{subtype}", direction,
                        item.number, subtype, payload,
                    )
        return InteractionUpdate(raw, "agent_server.interaction_update.unclassified",
                                 direction, 0, "unclassified", value)
    for number, kind, name in (
        (2, AgentExecMessage, "agent_server.exec_message.unclassified"),
        (3, CheckpointUpdate, "agent_server.conversation_checkpoint_update"),
    ):
        value = raw.first_bytes(number)
        if value is not None and len(value) > 0:
            return kind(raw, name, direction)
    value = raw.first_bytes(4)
    if value is not None and len(value) > 0:
        kv = _decode_or_none(value)
        subtype = "unclassified"
        if kv and kv.has(2):
            subtype = "get_blob_args"
        elif kv and kv.has(3):
            subtype = "set_blob_args"
        return KVServerMessage(raw, f"agent_server.kv_server_message.{subtype}",
                               direction, subtype)
    for number, kind, name in (
        (5, ExecControlMessage, "agent_server.exec_control_message"),
        (7, InteractionQuery, "agent_server.interaction_query"),
    ):
        value = raw.first_bytes(number)
        if value is not None and len(value) > 0:
            return kind(raw, name, direction)
    return CursorMessage(raw, "unknown", direction)


def decode_cursor_payload(payload: bytes, direction: str) -> CursorMessage:
    return classify(RawMessage.decode(payload), direction)


def parse_eos_metadata(payload: bytes) -> tuple[object | None, str | None]:
    if not payload or not payload.strip():
        return None, None
    try:
        metadata = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed Connect EOS JSON payload") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Connect EOS metadata must be a JSON object")
    error = metadata.get("error")
    if error is None:
        return metadata, None
    if not isinstance(error, dict):
        return metadata, str(error)
    for detail in error.get("details", ()):
        if not isinstance(detail, dict):
            continue
        details = detail.get("debug", {}).get("details", {})
        if isinstance(details, dict):
            title = details.get("title", "")
            text = details.get("detail", "")
            if title or text:
                return metadata, " ".join(part for part in (title, text) if part)
    return metadata, str(error.get("message") or error)


@dataclass(frozen=True, slots=True)
class CursorFrame:
    connect: ConnectFrame
    direction: str
    classification: str
    message: CursorMessage | None
    eos_metadata: object | None = None
    eos_error: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        direction = self.direction.upper()
        if direction not in ("IN", "OUT"):
            raise ValueError("direction must be IN or OUT")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.connect.eos and self.classification != "connect_end_stream":
            raise ValueError("EOS frame classification must be connect_end_stream")
        if not self.connect.eos and self.message is None:
            raise ValueError("non-EOS frame requires a message")

    @classmethod
    def decode(cls, connect: ConnectFrame, direction: str,
               metadata: Mapping[str, object] | None = None) -> "CursorFrame":
        payload = connect.decoded_payload
        if connect.eos:
            eos_metadata, eos_error = parse_eos_metadata(payload)
            return cls(connect, direction, "connect_end_stream", None,
                       eos_metadata, eos_error, metadata or {})
        message = decode_cursor_payload(payload, direction)
        return cls(connect, direction, message.classification, message,
                   metadata=metadata or {})

    @property
    def flags(self) -> int:
        return self.connect.flags

    @property
    def wire_payload(self) -> bytes:
        return self.connect.wire_payload

    @property
    def decoded_payload(self) -> bytes:
        return self.connect.decoded_payload

    def encode_connect(self) -> bytes:
        return self.connect.encode()


@dataclass(frozen=True, slots=True)
class FrameGroup:
    """A contiguous lossless group under the README's simplified rules."""

    kind: str
    frames: tuple[CursorFrame, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("frame group cannot be empty")

    def encode_connect(self) -> bytes:
        return b"".join(frame.encode_connect() for frame in self.frames)


class LosslessFrameGrouper:
    """Groups one ordered connection; it is not full AgentRun.Next state."""

    def __init__(self) -> None:
        self._pending: list[CursorFrame] = []
        self._connection_id: str | None = None

    @property
    def connection_id(self) -> str | None:
        return self._connection_id

    def _bind_stream(self, frame: CursorFrame) -> None:
        value = frame.metadata.get("connection_id")
        if value is None:
            return
        identity = str(value).strip()
        if not identity:
            return
        if self._connection_id is None:
            self._connection_id = identity
        elif self._connection_id != identity:
            raise ValueError(
                f"frame connection_id {identity!r} conflicts with "
                f"grouper connection_id {self._connection_id!r}"
            )

    def _flush(self) -> list[FrameGroup]:
        if not self._pending:
            return []
        group = FrameGroup("interaction", tuple(self._pending))
        self._pending.clear()
        return [group]

    def feed(self, frame: CursorFrame) -> list[FrameGroup]:
        if not isinstance(frame, CursorFrame):
            raise TypeError("frame must be CursorFrame")
        self._bind_stream(frame)
        if frame.connect.eos:
            return self._flush() + [FrameGroup("eos_error" if frame.eos_error else "eos",
                                                (frame,))]
        message = frame.message
        if isinstance(message, InteractionUpdate):
            self._pending.append(frame)
            return []
        if isinstance(message, (AnswerText, CompletedMCPUpdate)):
            self._pending.append(frame)
            return self._flush()
        # Flush before an intervening immediate, KV, or singleton frame so
        # flattened output remains byte-for-byte ordered.
        kind = (
            "native_exec" if isinstance(message, NativeExec) else
            "live_mcp" if isinstance(message, LiveMCPCall) else
            "kv_internal" if isinstance(message, KVServerMessage) else
            "singleton"
        )
        return self._flush() + [FrameGroup(kind, (frame,))]

    def finish(self) -> list[FrameGroup]:
        return self._flush()


def flatten_groups(groups: Iterable[FrameGroup]) -> tuple[CursorFrame, ...]:
    return tuple(frame for group in groups for frame in group.frames)


def reconstitute_connect(groups: Iterable[FrameGroup]) -> bytes:
    return b"".join(frame.encode_connect() for frame in flatten_groups(groups))


def _record_to_frame(record: Mapping[str, object], default_direction: str | None) -> CursorFrame:
    direction_value = record.get("direction", default_direction)
    if direction_value is None:
        raise ValueError("frame record is missing direction")
    direction = str(direction_value).upper()
    flags_value = record.get("flags")
    if flags_value is None:
        raise ValueError("frame record is missing flags")
    flags = int(flags_value, 0) if isinstance(flags_value, str) else int(flags_value)
    wire_hex = record.get("wire_payload_hex")
    decoded_hex = record.get("decoded_payload_hex")
    if isinstance(wire_hex, str):
        wire_payload = bytes.fromhex(wire_hex)
        connect = ConnectFrame(flags, wire_payload)
        if isinstance(decoded_hex, str) and connect.decoded_payload != bytes.fromhex(decoded_hex):
            raise ValueError("wire and decoded payload metadata disagree")
    elif isinstance(decoded_hex, str):
        decoded = bytes.fromhex(decoded_hex)
        if flags & 1:
            raise ValueError("compressed record requires wire_payload_hex")
        connect = ConnectFrame(flags, decoded)
    else:
        raise ValueError("frame record has no payload hex")
    return CursorFrame.decode(connect, direction, record)


def load_log(path: str, default_direction: str | None = None) -> Iterator[CursorFrame]:
    """Stream marker-delimited production records or one-object-per-line JSONL."""

    begin = "========== CURSOR FRAME BEGIN"
    end = "========== CURSOR FRAME END"
    in_record = False
    record_lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if stripped.startswith(begin):
                if in_record:
                    raise ValueError(f"nested frame marker at line {line_number}")
                in_record = True
                record_lines = []
                continue
            if stripped.startswith(end):
                if not in_record:
                    raise ValueError(f"frame end without begin at line {line_number}")
                try:
                    record = json.loads("".join(record_lines))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed frame JSON ending at line {line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"frame JSON ending at line {line_number} is not an object")
                yield _record_to_frame(record, default_direction)
                in_record = False
                record_lines = []
                continue
            if in_record:
                record_lines.append(line)
                continue
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed JSONL frame at line {line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL frame at line {line_number} is not an object")
                if "decoded_payload_hex" in record or "wire_payload_hex" in record:
                    yield _record_to_frame(record, default_direction)
            elif (stripped.startswith("{")
                  and ("decoded_payload_hex" in stripped or "wire_payload_hex" in stripped)):
                raise ValueError(f"malformed JSONL frame at line {line_number}")
        if in_record:
            raise ValueError("unterminated CURSOR FRAME record")


def production_conformance(path: str) -> dict[str, object]:
    counts: dict[str, int] = {}
    total = connect_round_trips = raw_round_trips = typed_round_trips = 0
    for frame in load_log(path):
        total += 1
        expected_connect = (
            bytes((frame.flags,))
            + len(frame.wire_payload).to_bytes(4, "big")
            + frame.wire_payload
        )
        if frame.encode_connect() != expected_connect:
            raise AssertionError(f"Connect round-trip failed at frame {total}")
        connect_round_trips += 1
        counts[frame.classification] = counts.get(frame.classification, 0) + 1
        if frame.connect.eos:
            continue
        raw = RawMessage.decode(frame.decoded_payload)
        if raw.encode() != frame.decoded_payload:
            raise AssertionError(f"raw round-trip failed at frame {total}")
        raw_round_trips += 1
        if frame.message is None or frame.message.encode() != frame.decoded_payload:
            raise AssertionError(f"typed round-trip failed at frame {total}")
        typed_round_trips += 1
    return {
        "total_frames": total,
        "connect_round_trips": connect_round_trips,
        "raw_round_trips": raw_round_trips,
        "typed_round_trips": typed_round_trips,
        "classifications": counts,
    }


__all__ = [
    "UINT64_MAX", "MAX_FIELD_NUMBER", "RESERVED_FIELD_RANGE",
    "Field", "RawMessage", "SemanticField", "NestedField",
    "MessageRepresentation", "ConnectFrame", "CursorFrame",
    "decode_connect_frames", "encode_connect_frame", "encode_varint",
    "protobuf_field", "protobuf_message", "CursorMessage", "NativeExec",
    "LiveMCPCall", "CompletedMCPUpdate", "AnswerText",
    "InteractionUpdate", "AgentExecMessage", "CheckpointUpdate",
    "KVServerMessage", "ExecControlMessage", "InteractionQuery",
    "RunRequest", "ClientExecMessage", "KVResponse", "Control",
    "ClientHeartbeat", "FrameGroup", "LosslessFrameGrouper",
    "flatten_groups", "reconstitute_connect", "classify",
    "decode_cursor_payload", "parse_message_repr", "parse_eos_metadata", "load_log",
    "production_conformance", "NATIVE_EXEC_FIELD_NAMES",
    "INTERACTION_UPDATE_FIELD_NAMES",
]


# === Connect/SSE transport ===




class SSEError(Exception):
    pass


class _PostRequest:
    def __init__(self, client, url, body, headers, callback):
        self.client = client
        self.selector = client.selector
        self.parts = urlsplit(url)
        if self.parts.scheme not in ("http", "https"):
            raise ValueError("URL scheme must be http or https")
        if not self.parts.hostname:
            raise ValueError("URL must include a hostname")

        self.body = bytes(body)
        self.headers = dict(headers or {})
        self.callback = callback
        self.port = self.parts.port or (
            443 if self.parts.scheme == "https" else 80
        )
        self.sock = None
        self.connected = False
        self.tls_handshake_done = False
        self.outgoing = bytearray()
        self.incoming = bytearray()
        self.response_body = bytearray()
        self.headers_done = False
        self.status = None
        self.response_headers = {}
        self.chunked = False
        self.content_remaining = None
        self.chunk_remaining = None
        self.finished = False
        pooled = self.client._take_post_connection(self.parts)
        if pooled is None:
            self._connect()
        else:
            self.sock = pooled
            self.connected = True
            self.tls_handshake_done = True
            self.selector.register(
                self.sock,
                selectors.EVENT_READ | selectors.EVENT_WRITE,
                self,
            )
            self._prepare_request()

    def _connect(self):
        addresses = socket.getaddrinfo(
            self.parts.hostname,
            self.port,
            type=socket.SOCK_STREAM,
        )
        last_error = None
        for family, socktype, proto, _, address in addresses:
            sock = socket.socket(family, socktype, proto)
            sock.setblocking(False)
            error = sock.connect_ex(address)
            if error in (
                0,
                errno.EINPROGRESS,
                errno.EWOULDBLOCK,
                errno.EALREADY,
            ):
                self.sock = sock
                break
            last_error = OSError(error, errno.errorcode.get(error, "connect"))
            sock.close()
        else:
            raise last_error or SSEError("No usable address found")

        self.selector.register(
            self.sock,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            self,
        )

    def _finish_connect(self):
        error = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if error:
            raise OSError(error, errno.errorcode.get(error, "connect"))
        self.connected = True

        if self.parts.scheme == "https":
            self.selector.unregister(self.sock)
            context = self.client.ssl_context or ssl.create_default_context()
            self.sock = context.wrap_socket(
                self.sock,
                server_hostname=self.parts.hostname,
                do_handshake_on_connect=False,
            )
            self.sock.setblocking(False)
            self.selector.register(
                self.sock,
                selectors.EVENT_READ | selectors.EVENT_WRITE,
                self,
            )
        else:
            self.tls_handshake_done = True
            self._prepare_request()

    def _do_tls_handshake(self):
        try:
            self.sock.do_handshake()
        except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
            return
        self.tls_handshake_done = True
        self._prepare_request()

    def _prepare_request(self):
        target = self.parts.path or "/"
        if self.parts.query:
            target += "?" + self.parts.query

        default_port = 443 if self.parts.scheme == "https" else 80
        host = self.parts.hostname
        if self.port != default_port:
            host = f"{host}:{self.port}"

        headers = {
            "Host": host,
            "Content-Length": str(len(self.body)),
            "Content-Type": "application/octet-stream",
            "User-Agent": "stdlib-sse-client/1.0",
        }
        headers.update(self.client.headers)
        headers.update(self.headers)
        request = [f"POST {target} HTTP/1.1"]
        request.extend(f"{name}: {value}" for name, value in headers.items())
        request.extend(("", ""))
        self.outgoing.extend(
            "\r\n".join(request).encode("iso-8859-1") + self.body
        )

    def _set_interest(self):
        if self.finished:
            return
        events = selectors.EVENT_READ
        if (
            not self.connected
            or not self.tls_handshake_done
            or self.outgoing
        ):
            events |= selectors.EVENT_WRITE
        self.selector.modify(self.sock, events, self)

    def _send(self):
        if not self.outgoing:
            return
        try:
            sent = self.sock.send(self.outgoing)
            del self.outgoing[:sent]
        except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            pass

    def _receive(self):
        try:
            data = self.sock.recv(65536)
        except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            return
        if not data:
            if not self.headers_done:
                raise SSEError("Uplink closed before HTTP response headers")
            if self.content_remaining not in (None, 0) or self.chunk_remaining is not None:
                raise SSEError("Uplink response body was truncated")
            self._complete()
            return

        self.incoming.extend(data)
        if not self.headers_done:
            self._parse_headers()
        if self.headers_done:
            self._parse_body()

    def _parse_headers(self):
        marker = self.incoming.find(b"\r\n\r\n")
        if marker < 0:
            if len(self.incoming) > 65536:
                raise SSEError("HTTP response headers are too large")
            return

        raw_headers = bytes(self.incoming[:marker])
        del self.incoming[:marker + 4]
        lines = raw_headers.decode("iso-8859-1").split("\r\n")
        parts = lines[0].split(" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            raise SSEError(f"Invalid HTTP status line: {lines[0]!r}")
        self.status = int(parts[1])

        for line in lines[1:]:
            if ":" not in line:
                raise SSEError(f"Invalid HTTP header: {line!r}")
            name, value = line.split(":", 1)
            self.response_headers[name.strip().lower()] = value.strip()

        transfer_encoding = self.response_headers.get("transfer-encoding", "")
        self.chunked = "chunked" in {
            item.strip().lower() for item in transfer_encoding.split(",")
        }
        content_length = self.response_headers.get("content-length")
        if content_length is not None and not self.chunked:
            try:
                self.content_remaining = int(content_length)
            except ValueError:
                raise SSEError("Invalid Content-Length")
            if self.content_remaining < 0:
                raise SSEError("Invalid Content-Length")
        self.headers_done = True

    def _parse_body(self):
        if self.chunked:
            self._parse_chunked_body()
            return
        if self.content_remaining is None:
            self.response_body.extend(self.incoming)
            self.incoming.clear()
            return

        count = min(len(self.incoming), self.content_remaining)
        self.response_body.extend(self.incoming[:count])
        del self.incoming[:count]
        self.content_remaining -= count
        if self.content_remaining == 0:
            self._complete()

    def _parse_chunked_body(self):
        while not self.finished:
            if self.chunk_remaining is None:
                marker = self.incoming.find(b"\r\n")
                if marker < 0:
                    return
                line = bytes(self.incoming[:marker])
                del self.incoming[:marker + 2]
                try:
                    self.chunk_remaining = int(line.split(b";", 1)[0], 16)
                except ValueError:
                    raise SSEError("Invalid chunk size")
                if self.chunk_remaining == 0:
                    self._complete()
                    return

            needed = self.chunk_remaining + 2
            if len(self.incoming) < needed:
                return
            if self.incoming[self.chunk_remaining:needed] != b"\r\n":
                raise SSEError("Invalid chunk terminator")
            self.response_body.extend(self.incoming[:self.chunk_remaining])
            del self.incoming[:needed]
            self.chunk_remaining = None

    def _complete(self):
        if self.finished:
            return
        self.finished = True
        reusable = (
            self.response_headers.get("connection", "").lower() != "close"
            and not self.chunked
            and self.content_remaining == 0
        )
        if reusable:
            sock = self.sock
            try:
                self.selector.unregister(sock)
            except (KeyError, ValueError):
                pass
            self.sock = None
            self.client._return_post_connection(self.parts, sock)
        else:
            self.close()
        self.client._posts.discard(self)
        if self.callback is not None:
            self.callback({
                "status": self.status,
                "headers": dict(self.response_headers),
                "body": bytes(self.response_body),
            })

    def run(self, mask):
        if self.finished or self.sock is None:
            return
        if not self.connected:
            self._finish_connect()
        if self.finished or self.sock is None:
            return
        if not self.tls_handshake_done:
            self._do_tls_handshake()
        else:
            if mask & selectors.EVENT_WRITE:
                self._send()
            if (
                not self.finished
                and self.sock is not None
                and mask & selectors.EVENT_READ
            ):
                self._receive()
        if not self.finished and self.sock is not None:
            self._set_interest()

    def close(self):
        if self.sock is None:
            return
        try:
            self.selector.unregister(self.sock)
        except (KeyError, ValueError):
            pass
        self.sock.close()
        self.sock = None


class _PostPipeline:
    def __init__(self, client, parts):
        self.client = client
        self.selector = client.selector
        self.parts = parts
        self.port = self.parts.port or (
            443 if self.parts.scheme == "https" else 80
        )
        self.sock = None
        self.connected = False
        self.tls_handshake_done = False
        self.outgoing = bytearray()
        self.incoming = bytearray()
        self.pending = deque()
        self.closed = False
        self._reset_response()
        self._connect()

    def _reset_response(self):
        self.response_body = bytearray()
        self.headers_done = False
        self.status = None
        self.response_headers = {}
        self.chunked = False
        self.content_remaining = None
        self.chunk_remaining = None

    def _connect(self):
        addresses = socket.getaddrinfo(
            self.parts.hostname,
            self.port,
            type=socket.SOCK_STREAM,
        )
        last_error = None
        for family, socktype, proto, _, address in addresses:
            sock = socket.socket(family, socktype, proto)
            sock.setblocking(False)
            error = sock.connect_ex(address)
            if error in (
                0,
                errno.EINPROGRESS,
                errno.EWOULDBLOCK,
                errno.EALREADY,
            ):
                self.sock = sock
                break
            last_error = OSError(
                error, errno.errorcode.get(error, "connect")
            )
            sock.close()
        else:
            raise last_error or SSEError("No usable address found")

        self.selector.register(
            self.sock,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            self,
        )

    def _finish_connect(self):
        error = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if error:
            raise OSError(error, errno.errorcode.get(error, "connect"))
        self.connected = True

        if self.parts.scheme == "https":
            self.selector.unregister(self.sock)
            context = self.client.ssl_context or ssl.create_default_context()
            self.sock = context.wrap_socket(
                self.sock,
                server_hostname=self.parts.hostname,
                do_handshake_on_connect=False,
            )
            self.sock.setblocking(False)
            self.selector.register(
                self.sock,
                selectors.EVENT_READ | selectors.EVENT_WRITE,
                self,
            )
        else:
            self.tls_handshake_done = True

    def _do_tls_handshake(self):
        try:
            self.sock.do_handshake()
        except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
            return
        self.tls_handshake_done = True

    def submit(self, parts, body, headers, callback):
        if self.closed:
            raise SSEError("Uplink pipeline is closed")
        body = bytes(body)
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query

        default_port = 443 if self.parts.scheme == "https" else 80
        host = self.parts.hostname
        if self.port != default_port:
            host = f"{host}:{self.port}"

        request_headers = {
            "Host": host,
            "Content-Length": str(len(body)),
            "Content-Type": "application/octet-stream",
            "User-Agent": "stdlib-sse-client/1.0",
        }
        request_headers.update(self.client.headers)
        request_headers.update(headers or {})
        request = [f"POST {target} HTTP/1.1"]
        request.extend(
            f"{name}: {value}" for name, value in request_headers.items()
        )
        request.extend(("", ""))
        self.outgoing.extend(
            "\r\n".join(request).encode("iso-8859-1") + body
        )
        self.pending.append(callback)
        self._set_interest()
        return self

    def _set_interest(self):
        if self.closed or self.sock is None:
            return
        events = selectors.EVENT_READ
        if (
            not self.connected
            or not self.tls_handshake_done
            or self.outgoing
        ):
            events |= selectors.EVENT_WRITE
        self.selector.modify(self.sock, events, self)

    def _send(self):
        if not self.outgoing:
            return
        try:
            sent = self.sock.send(self.outgoing)
            del self.outgoing[:sent]
        except (
            BlockingIOError,
            ssl.SSLWantReadError,
            ssl.SSLWantWriteError,
        ):
            pass

    def _receive(self):
        try:
            data = self.sock.recv(65536)
        except (
            BlockingIOError,
            ssl.SSLWantReadError,
            ssl.SSLWantWriteError,
        ):
            return
        if not data:
            if self.pending:
                raise SSEError(
                    "Uplink pipeline closed with responses outstanding"
                )
            self.close()
            return

        self.incoming.extend(data)
        self._parse_responses()

    def _parse_responses(self):
        while self.pending and not self.closed:
            if not self.headers_done:
                marker = self.incoming.find(b"\r\n\r\n")
                if marker < 0:
                    if len(self.incoming) > 65536:
                        raise SSEError(
                            "HTTP response headers are too large"
                        )
                    return
                raw_headers = bytes(self.incoming[:marker])
                del self.incoming[:marker + 4]
                lines = raw_headers.decode("iso-8859-1").split("\r\n")
                parts = lines[0].split(" ", 2)
                if len(parts) < 2 or not parts[1].isdigit():
                    raise SSEError(
                        f"Invalid HTTP status line: {lines[0]!r}"
                    )
                self.status = int(parts[1])
                for line in lines[1:]:
                    if ":" not in line:
                        raise SSEError(f"Invalid HTTP header: {line!r}")
                    name, value = line.split(":", 1)
                    self.response_headers[
                        name.strip().lower()
                    ] = value.strip()

                transfer_encoding = self.response_headers.get(
                    "transfer-encoding", ""
                )
                self.chunked = "chunked" in {
                    item.strip().lower()
                    for item in transfer_encoding.split(",")
                }
                content_length = self.response_headers.get(
                    "content-length"
                )
                if content_length is not None and not self.chunked:
                    try:
                        self.content_remaining = int(content_length)
                    except ValueError:
                        raise SSEError("Invalid Content-Length")
                    if self.content_remaining < 0:
                        raise SSEError("Invalid Content-Length")
                elif not self.chunked:
                    raise SSEError(
                        "Pipelined response requires Content-Length "
                        "or chunked encoding"
                    )
                self.headers_done = True

            if self.chunked:
                if not self._parse_chunked_body():
                    return
            else:
                count = min(
                    len(self.incoming), self.content_remaining
                )
                self.response_body.extend(self.incoming[:count])
                del self.incoming[:count]
                self.content_remaining -= count
                if self.content_remaining:
                    return

            self._complete_response()

    def _parse_chunked_body(self):
        while True:
            if self.chunk_remaining is None:
                marker = self.incoming.find(b"\r\n")
                if marker < 0:
                    return False
                line = bytes(self.incoming[:marker])
                del self.incoming[:marker + 2]
                try:
                    self.chunk_remaining = int(
                        line.split(b";", 1)[0], 16
                    )
                except ValueError:
                    raise SSEError("Invalid chunk size")
                if self.chunk_remaining == 0:
                    if len(self.incoming) < 2:
                        return False
                    if self.incoming[:2] != b"\r\n":
                        raise SSEError("Invalid chunk terminator")
                    del self.incoming[:2]
                    return True

            needed = self.chunk_remaining + 2
            if len(self.incoming) < needed:
                return False
            if (
                self.incoming[self.chunk_remaining:needed]
                != b"\r\n"
            ):
                raise SSEError("Invalid chunk terminator")
            self.response_body.extend(
                self.incoming[:self.chunk_remaining]
            )
            del self.incoming[:needed]
            self.chunk_remaining = None

    def _complete_response(self):
        callback = self.pending.popleft()
        response = {
            "status": self.status,
            "headers": dict(self.response_headers),
            "body": bytes(self.response_body),
        }
        closing = (
            self.response_headers.get("connection", "").lower()
            == "close"
        )
        self._reset_response()
        if closing:
            if self.pending:
                raise SSEError(
                    "Uplink closed with pipelined responses outstanding"
                )
            self.close()
        if callback is not None:
            callback(response)

    def run(self, mask):
        if self.closed or self.sock is None:
            return
        if not self.connected:
            self._finish_connect()
        if self.closed or self.sock is None:
            return
        if not self.tls_handshake_done:
            self._do_tls_handshake()
        else:
            if mask & selectors.EVENT_WRITE:
                self._send()
            if (
                not self.closed
                and self.sock is not None
                and mask & selectors.EVENT_READ
            ):
                self._receive()
        if not self.closed and self.sock is not None:
            self._set_interest()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.sock is not None:
            try:
                self.selector.unregister(self.sock)
            except (KeyError, ValueError):
                pass
            self.sock.close()
            self.sock = None
        self.client._post_pipeline_closed(self.parts, self)


class SSEClient:
    def __init__(
        self, url, callback, headers=None, timeout=None, ssl_context=None,
        stream_callback=None, accepted_content_types=("text/event-stream",),
        method="GET", body=b"", headers_callback=None,
    ):
        self.url = url
        self.callback = callback
        self.method = method.upper()
        self.request_body = bytes(body)
        if self.method not in ("GET", "POST"):
            raise ValueError("method must be GET or POST")
        self.stream_callback = stream_callback
        self.headers_callback = headers_callback
        self.accepted_content_types = tuple(
            item.lower() for item in accepted_content_types
        )
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.ssl_context = ssl_context

        self.selector = selectors.DefaultSelector()
        self.sock = None
        self.closed = False
        self.connected = False

        self._tls_handshake_done = False
        self._want_read = False
        self._want_write = True
        self._outgoing = bytearray()
        self._incoming = bytearray()
        self._body = bytearray()
        self._headers_done = False
        self._chunked = False
        self._content_remaining = None
        self._chunk_remaining = None
        self._chunk_finished = False
        self._event_lines = []
        self.last_event_id = None
        self.retry = None
        self._posts = set()
        self._idle_post_connections = {}
        self._post_pipelines = {}
        self._grace_deadline = None
        self._post_blob_deadline = None
        self._post_blob_debug = {}

        self._parts = urlsplit(url)
        if self._parts.scheme not in ("http", "https"):
            raise ValueError("URL scheme must be http or https")
        if not self._parts.hostname:
            raise ValueError("URL must include a hostname")

        self._port = self._parts.port or (
            443 if self._parts.scheme == "https" else 80
        )
        self._connect()

    def _connect(self):
        addresses = socket.getaddrinfo(
            self._parts.hostname,
            self._port,
            type=socket.SOCK_STREAM,
        )
        last_error = None

        for family, socktype, proto, _, address in addresses:
            sock = socket.socket(family, socktype, proto)
            sock.setblocking(False)
            error = sock.connect_ex(address)
            if error in (
                0,
                errno.EINPROGRESS,
                errno.EWOULDBLOCK,
                errno.EALREADY,
            ):
                self.sock = sock
                break
            last_error = OSError(error, errno.errorcode.get(error, "connect"))
            sock.close()
        else:
            raise last_error or SSEError("No usable address found")

        self.selector.register(
            self.sock,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            self,
        )

    def _finish_connect(self):
        error = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if error:
            raise OSError(error, errno.errorcode.get(error, "connect"))

        self.connected = True

        if self._parts.scheme == "https":
            self.selector.unregister(self.sock)
            context = self.ssl_context or ssl.create_default_context()
            self.sock = context.wrap_socket(
                self.sock,
                server_hostname=self._parts.hostname,
                do_handshake_on_connect=False,
            )
            self.sock.setblocking(False)
            self.selector.register(
                self.sock,
                selectors.EVENT_READ | selectors.EVENT_WRITE,
                self,
            )
        else:
            self._tls_handshake_done = True
            self._prepare_request()

    def _do_tls_handshake(self):
        try:
            self.sock.do_handshake()
        except ssl.SSLWantReadError:
            self._want_read = True
            self._want_write = False
            return
        except ssl.SSLWantWriteError:
            self._want_read = False
            self._want_write = True
            return

        self._tls_handshake_done = True
        self._want_read = False
        self._want_write = True
        self._prepare_request()

    def _prepare_request(self):
        target = self._parts.path or "/"
        if self._parts.query:
            target += "?" + self._parts.query

        default_port = 443 if self._parts.scheme == "https" else 80
        host = self._parts.hostname
        if self._port != default_port:
            host = f"{host}:{self._port}"

        headers = {
            "Host": host,
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "User-Agent": "stdlib-sse-client/1.0",
        }
        if self.method == "POST":
            headers["Content-Length"] = str(len(self.request_body))
        headers.update(self.headers)
        if self.last_event_id is not None:
            headers.setdefault("Last-Event-ID", self.last_event_id)

        request = [f"{self.method} {target} HTTP/1.1"]
        request.extend(f"{name}: {value}" for name, value in headers.items())
        request.extend(("", ""))
        self._outgoing.extend(
            "\r\n".join(request).encode("iso-8859-1") + self.request_body
        )

    def _set_interest(self):
        if self.closed:
            return
        events = selectors.EVENT_READ
        if (
            not self.connected
            or not self._tls_handshake_done
            or self._outgoing
        ):
            events |= selectors.EVENT_WRITE
        self.selector.modify(self.sock, events, self)

    def _send(self):
        if not self._outgoing:
            return
        try:
            sent = self.sock.send(self._outgoing)
            del self._outgoing[:sent]
        except (BlockingIOError, ssl.SSLWantWriteError):
            pass
        except ssl.SSLWantReadError:
            pass

    def _receive(self):
        try:
            data = self.sock.recv(65536)
            if DEBUG:
                with open(f"/tmp/coda-cursor-protobuf-{_SESSION_CONVERSATION_ID[:8]}.log",'ab') as f:
                    f.write(data)
        except (BlockingIOError, ssl.SSLWantReadError):
            return
        except ssl.SSLWantWriteError:
            return

        if not data:
            if self._event_lines:
                self._dispatch_event()
            self.close()
            return

        self._incoming.extend(data)
        if not self._headers_done:
            self._parse_headers()
        if self._headers_done:
            self._parse_body()

    def _parse_headers(self):
        marker = self._incoming.find(b"\r\n\r\n")
        if marker < 0:
            if len(self._incoming) > 65536:
                raise SSEError("HTTP response headers are too large")
            return

        raw_headers = bytes(self._incoming[:marker])
        del self._incoming[:marker + 4]
        lines = raw_headers.decode("iso-8859-1").split("\r\n")

        parts = lines[0].split(" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            raise SSEError(f"Invalid HTTP status line: {lines[0]!r}")

        status = int(parts[1])
        if status != 200:
            raise SSEError(f"SSE request returned HTTP {status}")

        headers = {}
        for line in lines[1:]:
            if ":" not in line:
                raise SSEError(f"Invalid HTTP header: {line!r}")
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()

        content_type = headers.get("content-type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type not in self.accepted_content_types:
            raise SSEError(f"Unexpected Content-Type: {content_type!r}")

        transfer_encoding = headers.get("transfer-encoding", "")
        self._chunked = "chunked" in {
            item.strip().lower() for item in transfer_encoding.split(",")
        }

        content_length = headers.get("content-length")
        if content_length is not None and not self._chunked:
            try:
                self._content_remaining = int(content_length)
            except ValueError:
                raise SSEError("Invalid Content-Length")

        self._headers_done = True
        if self.headers_callback is not None:
            self.headers_callback(status, dict(headers))

    def _parse_body(self):
        if self._chunked:
            self._parse_chunked_body()
            return

        if self._content_remaining is None:
            if self._incoming:
                self._feed_sse(bytes(self._incoming))
                self._incoming.clear()
            return

        count = min(len(self._incoming), self._content_remaining)
        if count:
            self._feed_sse(bytes(self._incoming[:count]))
            del self._incoming[:count]
            self._content_remaining -= count
        if self._content_remaining == 0:
            self.close()

    def _parse_chunked_body(self):
        while not self._chunk_finished:
            if self._chunk_remaining is None:
                marker = self._incoming.find(b"\r\n")
                if marker < 0:
                    return
                line = bytes(self._incoming[:marker])
                del self._incoming[:marker + 2]
                size_text = line.split(b";", 1)[0]
                try:
                    self._chunk_remaining = int(size_text, 16)
                except ValueError:
                    raise SSEError("Invalid chunk size")
                if self._chunk_remaining == 0:
                    self._chunk_finished = True
                    self.close()
                    return

            needed = self._chunk_remaining + 2
            if len(self._incoming) < needed:
                return
            if self._incoming[self._chunk_remaining:needed] != b"\r\n":
                raise SSEError("Invalid chunk terminator")

            data = bytes(self._incoming[:self._chunk_remaining])
            del self._incoming[:needed]
            self._chunk_remaining = None
            self._feed_sse(data)

    def _feed_sse(self, data):
        if self.stream_callback is not None:
            self.stream_callback(data)
            return
        self._body.extend(data)
        while True:
            newline = self._body.find(b"\n")
            if newline < 0:
                return
            raw_line = bytes(self._body[:newline])
            del self._body[:newline + 1]
            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            line = raw_line.decode("utf-8", errors="replace")
            if line == "":
                self._dispatch_event()
            else:
                self._event_lines.append(line)

    def _dispatch_event(self):
        data_lines = []
        event_type = None
        event_id = None
        retry = None

        for line in self._event_lines:
            if line.startswith(":"):
                continue

            if ":" in line:
                field, value = line.split(":", 1)
                if value.startswith(" "):
                    value = value[1:]
            else:
                field, value = line, ""

            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event_type = value
            elif field == "id" and "\x00" not in value:
                event_id = value
            elif field == "retry" and value.isdigit():
                retry = int(value)

        self._event_lines.clear()

        if event_id is not None:
            self.last_event_id = event_id
        if retry is not None:
            self.retry = retry

        if not data_lines:
            return

        event = {
            "data": "\n".join(data_lines),
            "event": event_type or "message",
        }
        if event_id is not None:
            event["id"] = event_id
        if retry is not None:
            event["retry"] = retry
        self.callback(event)

    @staticmethod
    def _post_connection_key(parts):
        return (
            parts.scheme,
            parts.hostname,
            parts.port or (443 if parts.scheme == "https" else 80),
        )

    def _take_post_connection(self, parts):
        connections = self._idle_post_connections.get(
            self._post_connection_key(parts)
        )
        if not connections:
            return None
        sock = connections.pop()
        if not connections:
            del self._idle_post_connections[
                self._post_connection_key(parts)
            ]
        return sock

    def _return_post_connection(self, parts, sock):
        if self.closed:
            sock.close()
            return
        self._idle_post_connections.setdefault(
            self._post_connection_key(parts), []
        ).append(sock)

    def _post_pipeline_closed(self, parts, pipeline):
        key = self._post_connection_key(parts)
        if self._post_pipelines.get(key) is pipeline:
            del self._post_pipelines[key]

    def post(self, url, body=b"", headers=None, callback=None):
        """Pipeline a POST request on the origin's HTTP/1.1 connection."""
        if self.closed:
            raise SSEError("SSE client is closed")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise ValueError("URL scheme must be http or https")
        if not parts.hostname:
            raise ValueError("URL must include a hostname")
        key = self._post_connection_key(parts)
        pipeline = self._post_pipelines.get(key)
        if pipeline is None or pipeline.closed:
            pipeline = _PostPipeline(self, parts)
            self._post_pipelines[key] = pipeline
        return pipeline.submit(parts, body, headers, callback)

    def run_once(self, timeout=None):
        if self.closed:
            return False

        wait = self.timeout if timeout is None else timeout
        for key, mask in self.selector.select(wait):
            if self.closed:
                break
            handler = key.data
            if handler is self:
                if not self.connected:
                    self._finish_connect()

                if not self._tls_handshake_done:
                    self._do_tls_handshake()
                else:
                    if mask & selectors.EVENT_WRITE:
                        self._send()
                    if mask & selectors.EVENT_READ:
                        self._receive()

                if not self.closed:
                    self._set_interest()
            else:
                handler.run(mask)

        return not self.closed

    def reset_heartbeat_timeout(self):
        self._heartbeat_deadline = time.monotonic() + HEARTBEAT_TIMEOUT

    def start_grace_period(self, timeout):
        self._grace_deadline = time.monotonic() + timeout

    def arm_post_blob_timeout(self, timeout):
        self._post_blob_deadline = time.monotonic() + timeout
        details = getattr(self, "_post_blob_debug", {})
        _debug_bidi_event(
            "post_blob_timeout_armed", timeout=timeout, **details
        )

    def clear_post_blob_timeout(self):
        details = getattr(self, "_post_blob_debug", {})
        if self._post_blob_deadline is not None:
            _debug_bidi_event(
                "post_blob_timeout_cleared", **details
            )
        self._post_blob_deadline = None
        self._post_blob_debug = {}

    def run_forever(
        self, timeout=None, *, heartbeat_timeout=None,
    ):
        now = time.monotonic()
        deadline = None if timeout is None else now + timeout
        self._heartbeat_deadline = (
            None if heartbeat_timeout is None else now + heartbeat_timeout
        )
        while not self.closed:
            now = time.monotonic()
            waits = []
            if deadline is not None:
                remaining = deadline - now
                if remaining <= 0:
                    raise SSEError("request deadline exceeded")
                waits.append(remaining)
            if self._heartbeat_deadline is not None:
                remaining = self._heartbeat_deadline - now
                if remaining <= 0:
                    raise SSEError("server heartbeat timeout")
                waits.append(remaining)
            if self._grace_deadline is not None:
                remaining = self._grace_deadline - now
                if remaining <= 0:
                    self._grace_deadline = None
                    self.close()
                    break
                waits.append(remaining)
            if getattr(self, "_post_blob_deadline", None) is not None:
                remaining = self._post_blob_deadline - now
                if remaining <= 0:
                    _debug_bidi_event(
                        "post_blob_timeout_expired",
                        **getattr(self, "_post_blob_debug", {}),
                    )
                    raise SSEError(
                        "no model progress after blob hydration"
                    )
                waits.append(remaining)
            wait = min(waits) if waits else None
            if not self.run_once(wait):
                break

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.selector.unregister(self.sock)
        except (KeyError, ValueError):
            pass
        self.sock.close()
        for request in tuple(self._posts):
            request.close()
        self._posts.clear()
        pipelines = getattr(self, "_post_pipelines", {})
        for pipeline in tuple(pipelines.values()):
            pipeline.close()
        pipelines.clear()
        idle_connections = getattr(
            self, "_idle_post_connections", {}
        )
        for connections in idle_connections.values():
            for sock in connections:
                sock.close()
        idle_connections.clear()
        self.selector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# === Cursor Agent client ===


def _message(*fields: Field) -> bytes:
    return RawMessage(tuple(fields)).encode()


def _bytes(number: int, payload: bytes) -> Field:
    return Field.bytes(number, payload)


def _string(number: int, value: str) -> Field:
    return _bytes(number, value.encode())


def extract_prefetched_blobs(client_payload: bytes) -> dict[bytes, bytes]:
    client = RawMessage.decode(client_payload)
    run_payload = client.first_bytes(1)
    if run_payload is None:
        return {}
    run = RawMessage.decode(run_payload)
    blobs = {}
    for field in run.matching(17, 2):
        blob = RawMessage.decode(field.value)
        blob_id = blob.first_bytes(1)
        value = blob.first_bytes(2)
        if blob_id is not None and value is not None:
            blobs[blob_id] = value
    return blobs


def is_generation_progress(message: CursorMessage | None) -> bool:
    if isinstance(message, AnswerText):
        return bool(message.text)
    if isinstance(message, (NativeExec, LiveMCPCall, CompletedMCPUpdate)):
        return True
    if not isinstance(message, InteractionUpdate):
        return False
    return message.subtype in {
        "text_delta",
        "tool_call_started",
        "tool_call_completed",
        "thinking_delta",
        "thinking_completed",
        "partial_tool_call",
        "token_delta",
        "tool_call_delta",
    }


def build_kv_response(server_payload: bytes, blobs) -> bytes | None:
    server = RawMessage.decode(server_payload)
    kv_payload = server.first_bytes(4)
    if kv_payload is None:
        return None
    kv = RawMessage.decode(kv_payload)
    request_id_fields = kv.matching(1, 0)
    request_id = (
        int(request_id_fields[0].value) if request_id_fields else 0
    )
    get_args_payload = kv.first_bytes(2)
    if get_args_payload is not None:
        args = RawMessage.decode(get_args_payload)
        blob_id = args.first_bytes(1)
        if blob_id is None:
            return None
        result_fields = []
        value = blobs.get(blob_id)
        if value is not None:
            result_fields.append(_bytes(1, value))
        kv_client = _message(
            Field.varint(1, request_id),
            _bytes(2, _message(*result_fields)),
        )
        return _message(_bytes(3, kv_client))
    if kv.first_bytes(3) is not None:
        kv_client = _message(
            Field.varint(1, request_id),
            _bytes(3, b""),
        )
        return _message(_bytes(3, kv_client))
    return None


def is_response_boundary_blob_write(
    server_payload: bytes, request_id: str
) -> bool:
    try:
        server = RawMessage.decode(server_payload)
        kv_payload = server.first_bytes(4)
        if kv_payload is None:
            return False
        kv = RawMessage.decode(kv_payload)
        set_args_payload = kv.first_bytes(3)
        if set_args_payload is None:
            return False
        set_args = RawMessage.decode(set_args_payload)
        blob_payload = set_args.first_bytes(2)
        if blob_payload is None:
            return False
        blob = RawMessage.decode(blob_payload)
        structure_payload = blob.first_bytes(1)
        if structure_payload is None:
            return False
        structure = RawMessage.decode(structure_payload)
    except ValueError:
        return False

    if any(
        field.number not in (1, 2, 3, 4, 5)
        or field.wire_type != (0 if field.number == 5 else 2)
        for field in structure.fields
    ):
        return False
    user_messages = structure.matching(1, 2)
    steps = structure.matching(2, 2)
    request_ids = structure.matching(3, 2)
    if (
        len(user_messages) != 1
        or len(user_messages[0].value) != 32
        or not steps
        or any(len(step.value) != 32 for step in steps)
        or len(request_ids) != 1
    ):
        return False
    try:
        return request_ids[0].value.decode() == request_id
    except UnicodeDecodeError:
        return False


def build_user_cancelled_message() -> bytes:
    return _message(
        _bytes(4, _message(_bytes(3, _message(_string(1, "user_cancelled")))))
    )


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str = ""
    tool_calls: tuple["ToolCall", ...] = ()
    tool_call_id: str = ""
    tool_name: str = ""


def _protobuf_value(value) -> bytes:
    if value is None:
        return _message(Field.varint(1, 0))
    if isinstance(value, bool):
        return _message(Field.varint(4, int(value)))
    if isinstance(value, (int, float)):
        return _message(Field(2, 1, struct.pack("<d", float(value))))
    if isinstance(value, str):
        return _message(_string(3, value))
    if isinstance(value, dict):
        entries = [
            _bytes(
                1,
                _message(
                    _string(1, str(key)),
                    _bytes(2, _protobuf_value(item)),
                ),
            )
            for key, item in value.items()
        ]
        return _message(_bytes(5, _message(*entries)))
    if isinstance(value, (list, tuple)):
        items = [_bytes(1, _protobuf_value(item)) for item in value]
        return _message(_bytes(6, _message(*items)))
    return _message(_string(3, json.dumps(value, separators=(",", ":"))))


def _historical_tool_step(
    call: "ToolCall", result: ConversationMessage | None
) -> bytes:
    argument_fields = [
        _string(1, call.name),
        _string(3, call.id),
        _string(4, call.provider_identifier),
        _string(5, call.name),
    ]
    for key, value in call.arguments.items():
        argument_fields.append(
            _bytes(
                2,
                _message(
                    _string(1, key),
                    _bytes(2, _protobuf_value(value)),
                ),
            )
        )
    mcp_fields = [_bytes(1, _message(*argument_fields))]
    if result is not None:
        text = _message(_string(1, result.content))
        item = _message(_bytes(1, text))
        success = _message(_bytes(1, item), Field.varint(2, 0))
        mcp_fields.append(_bytes(2, _message(_bytes(1, success))))
    return _message(
        _bytes(2, _message(_bytes(15, _message(*mcp_fields))))
    )


def _conversation_message_json(message: ConversationMessage) -> bytes | None:
    if message.role == "assistant" and message.tool_calls:
        if not message.content or message.content == "[empty]":
            return None
        payload = {
            "role": "assistant",
            "id": _conversation_message_id("assistant", message),
            "content": message.content,
        }
    elif message.role == "tool":
        try:
            result = json.loads(message.content)
        except json.JSONDecodeError:
            result = message.content
        payload = {
            "role": "tool",
            "id": _conversation_message_id("tool", message),
            "content": [{
                "type": "tool-result",
                "toolCallId": message.tool_call_id,
                "toolName": message.tool_name,
                "result": result,
            }],
        }
    else:
        payload = {
            "role": message.role,
            "content": message.content,
        }
    return json.dumps(payload, separators=(",", ":")).encode()


def _conversation_message_id(
    role: str, message: ConversationMessage
) -> str:
    key = role + "\0" + message.tool_call_id + "\0" + message.content
    for call in message.tool_calls:
        arguments = json.dumps(
            call.arguments, separators=(",", ":"), sort_keys=True
        )
        key += "\0" + call.id + "\0" + call.name + "\0" + arguments
    return hashlib.sha256(key.encode()).hexdigest()


def encode_conversation_state(
    history, mode: int = 1
) -> tuple[bytes, list[bytes]]:
    history = list(history)
    state_fields = [Field.varint(10, mode)]
    prefetched = []

    def add_blob(value: bytes) -> bytes:
        blob_id = hashlib.sha256(value).digest()
        prefetched.append(_message(_bytes(1, blob_id), _bytes(2, value)))
        return blob_id

    for message in history:
        encoded = _conversation_message_json(message)
        if encoded:
            state_fields.append(_bytes(1, add_blob(encoded)))

    index = 0
    while index < len(history):
        if history[index].role != "user":
            index += 1
            continue
        end = index + 1
        while end < len(history) and history[end].role != "user":
            end += 1
        messages = history[index:end]
        user_id = "history-" + hashlib.sha256(
            messages[0].content.encode()
        ).hexdigest()[:16]
        user_message = _message(
            _string(1, messages[0].content),
            _string(2, user_id),
            Field.varint(4, mode),
        )
        results = {
            message.tool_call_id: message
            for message in messages[1:]
            if message.role == "tool"
        }
        step_ids = []
        for message in messages[1:]:
            if message.role != "assistant":
                continue
            if message.content and message.content != "[empty]":
                step = _message(
                    _bytes(1, _message(_string(1, message.content)))
                )
                step_ids.append(add_blob(step))
            for call in message.tool_calls:
                step_ids.append(
                    add_blob(
                        _historical_tool_step(call, results.get(call.id))
                    )
                )
        agent_turn = _message(
            _bytes(1, add_blob(user_message)),
            *(_bytes(2, step_id) for step_id in step_ids),
        )
        state_fields.append(
            _bytes(8, add_blob(_message(_bytes(1, agent_turn))))
        )
        index = end
    return _message(*state_fields), prefetched


def encode_model_details(model: str) -> bytes:
    """Encode AgentRunRequest model details (field 3)."""
    return _message(
        _string(1, model),
        _string(3, model),
        _string(4, model),
        _string(5, model),
        Field.varint(7, 1),
    )


def build_run_request(
    prompt: str,
    model: str,
    *,
    tools=(),
    history=(),
    conversation_id: str | None = None,
    message_id: str | None = None,
    user_config: UserMessageConfig | None = None,
    run_config: RunConfig | None = None,
    workspace_uri: str | None = None,
    client_name: str | None = None,
) -> bytes:
    """Build the smallest useful AgentClientMessage.run_request."""

    user_config = user_config or UserMessageConfig()
    run_config = run_config or RunConfig()
    conversation_id = (
        conversation_id
        or run_config.conversation_id
        or _SESSION_CONVERSATION_ID
    )
    message_id = message_id or str(uuid.uuid4())

    user_fields: list[Field] = [
        _string(1, prompt),
        _string(2, message_id),
        _bytes(3, b""),
        Field.varint(4, user_config.mode),
    ]
    if user_config.selected_context is not None:
        user_fields.append(_bytes(3, user_config.selected_context))
    if user_config.is_simulated_msg is not None:
        user_fields.append(Field.varint(5, int(user_config.is_simulated_msg)))
    if user_config.best_of_n_group_id is not None:
        user_fields.append(_string(6, user_config.best_of_n_group_id))
    if user_config.try_use_best_of_n_promotion is not None:
        user_fields.append(
            Field.varint(7, int(user_config.try_use_best_of_n_promotion))
        )
    if user_config.rich_text is not None:
        user_fields.append(_string(8, user_config.rich_text))
    if user_config.simulated_msg_reason is not None:
        user_fields.append(Field.varint(9, user_config.simulated_msg_reason))
    if user_config.conversation_state_blob_id:
        user_fields.append(_bytes(10, user_config.conversation_state_blob_id))
    if user_config.subagent_system_reminder is not None:
        user_fields.append(_string(11, user_config.subagent_system_reminder))
    if user_config.triggering_user_info is not None:
        user_fields.append(_bytes(13, user_config.triggering_user_info))
    if user_config.execute_plan_info is not None:
        user_fields.append(_bytes(14, user_config.execute_plan_info))
    if user_config.simulated_message_metadata is not None:
        user_fields.append(_bytes(15, user_config.simulated_message_metadata))
    if user_config.prompt_reference_id is not None:
        user_fields.append(_string(16, user_config.prompt_reference_id))
    if user_config.thread_id is not None:
        user_fields.append(_string(17, user_config.thread_id))
    if user_config.text_blob_id is not None:
        user_fields.append(_bytes(18, user_config.text_blob_id))
    if user_config.rich_text_blob_id is not None:
        user_fields.append(_bytes(19, user_config.rich_text_blob_id))
    user_fields.extend(
        _bytes(21, value)
        for value in user_config.hook_additional_contexts
    )
    if user_config.custom_mode_intent is not None:
        user_fields.append(_bytes(22, user_config.custom_mode_intent))
    user_message = _message(*user_fields)
    user_message_action = _message(
        _bytes(1, user_message),
        _bytes(2, b""),
    )
    action = _message(_bytes(1, user_message_action))
    conversation_state, prefetched = encode_conversation_state(
        history, user_config.mode
    )
    if run_config.conversation_state is not None:
        conversation_state = run_config.conversation_state
    model_details = encode_model_details(model)
    run_fields = [
        _bytes(1, conversation_state),
        _bytes(2, run_config.action or action),
        _bytes(3, run_config.model_details or model_details),
        _bytes(4, encode_mcp_tools(tools)),
        _string(5, conversation_id),
        *(_bytes(17, blob) for blob in prefetched),
    ]
    for number, value in (
        (6, run_config.mcp_file_system_options),
        (7, run_config.skill_options),
    ):
        if value is not None:
            run_fields.append(_bytes(number, value))
    for number, value in (
        (8, run_config.custom_system_prompt),
        (11, run_config.subagent_type_name),
        (13, run_config.harness),
        (16, run_config.conversation_group_id),
        (18, run_config.dev_raw_model_slug),
    ):
        if value is not None:
            run_fields.append(_string(number, value))
    for number, value in (
        (10, run_config.suggest_next_prompt),
        (12, run_config.exclude_workspace_context),
        (19, run_config.client_supports_inline_images),
        (21, run_config.can_create_cloud_subagents),
        (22, run_config.suppress_subagent_progress_update_tool),
        (23, run_config.client_supports_send_to_user),
    ):
        if value is not None:
            run_fields.append(Field.varint(number, int(value)))
    run_fields.extend(
        _bytes(14, value)
        for value in run_config.selected_subagent_models
    )
    run_fields.extend(
        _bytes(15, value)
        for value in run_config.selected_subagent_model_details
    )
    run_fields.extend(
        _bytes(20, value)
        for value in run_config.subagent_model_overrides
    )
    run_fields.extend(run_config.extra_fields)
    run_request = _message(*run_fields)
    return _message(_bytes(1, run_request))


def build_bidi_request_id(request_id: str) -> bytes:
    return _message(_string(1, request_id))


def build_bidi_append(
    request_id: str,
    payload: bytes,
    *,
    append_seqno: int = 0,
    binary: bool = True,
) -> bytes:
    fields = []
    if not binary:
        fields.append(_string(1, payload.hex()))
    fields.append(_bytes(2, build_bidi_request_id(request_id)))
    if append_seqno:
        fields.append(Field.varint(3, append_seqno))
    if binary:
        fields.append(_bytes(4, payload))
    return _message(*fields)


class ConnectStreamDecoder:
    def __init__(self, callback):
        self.callback = callback
        self.buffer = bytearray()

    def feed(self, data: bytes) -> None:
        self.buffer.extend(data)
        while len(self.buffer) >= 5:
            length = int.from_bytes(self.buffer[1:5], "big")
            if len(self.buffer) < 5 + length:
                return
            frame = ConnectFrame(self.buffer[0], bytes(self.buffer[5:5 + length]))
            del self.buffer[:5 + length]
            self.callback(frame)

    def finish(self) -> None:
        if self.buffer:
            raise ValueError("truncated Connect frame")


@dataclass(frozen=True)
class TurnUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


def parse_turn_usage(payload: bytes) -> TurnUsage:
    update = RawMessage.decode(payload)
    return TurnUsage(
        input_tokens=_first_varint(update, 1),
        output_tokens=_first_varint(update, 2),
        cache_read_tokens=_first_varint(update, 3),
        cache_write_tokens=_first_varint(update, 4),
        reasoning_tokens=_first_varint(update, 5),
    )


def openai_usage(usage: TurnUsage) -> dict:
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
    }


def _checkpoint_timestamp(frame: CursorFrame) -> int | None:
    if not isinstance(frame.message, CheckpointUpdate):
        return None
    checkpoint = RawMessage.decode(frame.message.raw.first_bytes(3) or b"")
    value = _int(checkpoint, 26)
    return value or None


def build_filtered_usage_request(
    start_date: int,
    end_date: int,
    *,
    page: int = 1,
    page_size: int = 20,
) -> bytes:
    return _message(
        Field.varint(1, 0),
        Field.varint(2, start_date),
        Field.varint(3, end_date),
        Field.varint(6, page),
        Field.varint(7, page_size),
    )


def parse_filtered_usage(
    payload: bytes,
    conversation_id: str,
    request_started_ms: int,
) -> TurnUsage | None:
    response = RawMessage.decode(payload)
    candidates = []
    for field in response.matching(3, 2):
        event = RawMessage.decode(field.value)
        timestamp = _int(event, 1)
        if (
            _text(event, 23) != conversation_id
            or not _int(event, 8)
            or timestamp < request_started_ms
        ):
            continue
        token_payload = event.first_bytes(9)
        if token_payload is None:
            continue
        token = RawMessage.decode(token_payload)
        uncached_input = _int(token, 1)
        cache_read = _int(token, 4)
        candidates.append((
            timestamp,
            TurnUsage(
                input_tokens=uncached_input + cache_read,
                output_tokens=_int(token, 2),
                cache_read_tokens=cache_read,
                cache_write_tokens=_int(token, 3),
            ),
        ))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def get_filtered_usage(
    token: str,
    base_url: str,
    conversation_id: str,
    anchor_timestamp: int,
    request_started_ms: int,
    *,
    timeout: float = KEY_EXCHANGE_TIMEOUT,
) -> TurnUsage | None:
    payload = build_filtered_usage_request(
        anchor_timestamp - USAGE_LOOKUP_WINDOW_MS,
        anchor_timestamp + USAGE_LOOKUP_WINDOW_MS,
    )
    request = Request(
        urljoin(base_url.rstrip("/") + "/", FILTERED_USAGE_PATH),
        data=payload,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/proto",
            "Accept": "application/proto",
            "Connect-Protocol-Version": "1",
            "User-Agent": "connect-es/1.6.1",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return None
            return parse_filtered_usage(
                response.read(), conversation_id, request_started_ms
            )
    except (HTTPError, OSError, ValueError):
        return None


@dataclass
class RunResult:
    frames: list[CursorFrame]
    text: str
    tool_calls: list["ToolCall | UnknownToolCall"]
    turn_ended: bool
    checkpoint_updates: list[CursorFrame]
    eos_metadata: object | None
    eos_error: str | None
    usage: TurnUsage | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str = ""
    parameters: dict | str | None = None
    provider_identifier: str = ""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict
    provider_identifier: str = ""
    tool_name: str = ""
    server_identifier: str = ""
    native: bool = False
    server_message_id: int = 0
    exec_id: str = ""
    field_number: int = 0
    oneof_name: str = ""
    payload_type: str = ""
    arguments_raw: bytes = b""


@dataclass(frozen=True)
class UnknownToolCall:
    field_number: int
    oneof_name: str
    arguments_raw: bytes
    server_message_id: int = 0
    exec_id: str = ""


@dataclass(frozen=True)
class UserMessageConfig:
    selected_context: bytes | None = None
    mode: int = 1
    is_simulated_msg: bool | None = None
    best_of_n_group_id: str | None = None
    try_use_best_of_n_promotion: bool | None = None
    rich_text: str | None = None
    simulated_msg_reason: int | None = None
    conversation_state_blob_id: bytes = b""
    subagent_system_reminder: str | None = None
    triggering_user_info: bytes | None = None
    execute_plan_info: bytes | None = None
    simulated_message_metadata: bytes | None = None
    prompt_reference_id: str | None = None
    thread_id: str | None = None
    text_blob_id: bytes | None = None
    rich_text_blob_id: bytes | None = None
    hook_additional_contexts: tuple[bytes, ...] = ()
    custom_mode_intent: bytes | None = None


@dataclass(frozen=True)
class RunConfig:
    conversation_state: bytes | None = None
    action: bytes | None = None
    model_details: bytes | None = None
    conversation_id: str | None = None
    mcp_file_system_options: bytes | None = None
    skill_options: bytes | None = None
    custom_system_prompt: str | None = None
    suggest_next_prompt: bool | None = None
    subagent_type_name: str | None = None
    exclude_workspace_context: bool | None = None
    harness: str | None = None
    selected_subagent_models: tuple[bytes, ...] = ()
    selected_subagent_model_details: tuple[bytes, ...] = ()
    conversation_group_id: str | None = None
    dev_raw_model_slug: str | None = None
    client_supports_inline_images: bool | None = None
    subagent_model_overrides: tuple[bytes, ...] = ()
    can_create_cloud_subagents: bool | None = None
    suppress_subagent_progress_update_tool: bool | None = None
    client_supports_send_to_user: bool | None = False
    extra_fields: tuple[Field, ...] = ()


def _int(raw: RawMessage, number: int, default: int = 0) -> int:
    fields = raw.matching(number, 0)
    return int(fields[0].value) if fields else default


def _text(raw: RawMessage, number: int) -> str:
    value = raw.first_bytes(number)
    return value.decode(errors="replace") if value is not None else ""


def _decode_value(payload: bytes):
    value = RawMessage.decode(payload)
    if value.matching(1, 0):
        return None
    if value.matching(2, 1):
        return struct.unpack("<d", value.matching(2, 1)[0].encoded_value)[0]
    text = value.first_bytes(3)
    if text is not None:
        return text.decode(errors="replace")
    boolean = value.matching(4, 0)
    if boolean:
        return bool(boolean[0].value)
    struct_value = value.first_bytes(5)
    if struct_value is not None:
        return _decode_map(RawMessage.decode(struct_value), 1)
    list_value = value.first_bytes(6)
    if list_value is not None:
        items = RawMessage.decode(list_value)
        return [
            _decode_value(field.value)
            for field in items.matching(1, 2)
        ]
    return None


def _decode_map(raw: RawMessage, number: int) -> dict:
    result = {}
    for field in raw.matching(number, 2):
        entry = RawMessage.decode(field.value)
        key = _text(entry, 1)
        value = entry.first_bytes(2)
        if key:
            result[key] = _decode_value(value) if value is not None else None
    return result


EXEC_SERVER_TOOL_FIELDS = {
    2: ("shell_args", "agent.v1.ShellArgs"),
    3: ("write_args", "agent.v1.WriteArgs"),
    4: ("delete_args", "agent.v1.DeleteArgs"),
    5: ("grep_args", "agent.v1.GrepArgs"),
    7: ("read_args", "agent.v1.ReadArgs"),
    29: ("redacted_read_args", "agent.v1.ReadArgs"),
    8: ("ls_args", "agent.v1.LsArgs"),
    9: ("diagnostics_args", "agent.v1.DiagnosticsArgs"),
    10: ("request_context_args", "agent.v1.RequestContextArgs"),
    11: ("mcp_args", "agent.v1.McpArgs"),
    14: ("shell_stream_args", "agent.v1.ShellArgs"),
    16: ("background_shell_spawn_args", "agent.v1.BackgroundShellSpawnArgs"),
    17: ("list_mcp_resources_exec_args", "agent.v1.ListMcpResourcesExecArgs"),
    18: ("read_mcp_resource_exec_args", "agent.v1.ReadMcpResourceExecArgs"),
    36: ("mcp_state_exec_args", "agent.v1.McpStateExecArgs"),
    20: ("fetch_args", "agent.v1.FetchArgs"),
    21: ("record_screen_args", "agent.v1.RecordScreenArgs"),
    22: ("computer_use_args", "agent.v1.ComputerUseArgs"),
    23: ("write_shell_stdin_args", "agent.v1.WriteShellStdinArgs"),
    27: ("execute_hook_args", "agent.v1.ExecuteHookArgs"),
    28: ("subagent_args", "agent.v1.SubagentArgs"),
    30: ("force_background_shell_args", "agent.v1.ForceBackgroundShellArgs"),
    31: ("force_background_subagent_args", "agent.v1.ForceBackgroundSubagentArgs"),
    37: ("subagent_await_args", "agent.v1.SubagentAwaitArgs"),
    38: ("smart_mode_classifier_args", "agent.v1.SmartModeClassifierArgs"),
    40: ("canvas_diagnostics_args", "agent.v1.CanvasDiagnosticsArgs"),
    41: ("shell_allowlist_precheck_args", "agent.v1.ShellAllowlistPrecheckArgs"),
    42: ("mcp_allowlist_precheck_args", "agent.v1.McpAllowlistPrecheckArgs"),
    43: ("web_fetch_allowlist_precheck_args", "agent.v1.WebFetchAllowlistPrecheckArgs"),
    44: ("git_diff_request", "aiserver.v1.GetDiffRequest"),
    45: ("pi_read_args", "agent.v1.PiReadExecArgs"),
    46: ("pi_bash_args", "agent.v1.PiBashExecArgs"),
    47: ("pi_edit_args", "agent.v1.PiEditExecArgs"),
    48: ("pi_write_args", "agent.v1.PiWriteExecArgs"),
    49: ("pi_grep_args", "agent.v1.PiGrepExecArgs"),
    50: ("pi_find_args", "agent.v1.PiFindExecArgs"),
    51: ("pi_ls_args", "agent.v1.PiLsExecArgs"),
    53: ("conversation_search_args", "agent.v1.ConversationSearchArgs"),
}


NATIVE_ARGUMENT_SCHEMAS = {
    "shell_args": {
        1: ("command", "string"), 2: ("working_directory", "string"),
        3: ("timeout", "int32"), 4: ("tool_call_id", "string"),
        5: ("simple_commands", "string"), 6: ("has_input_redirect", "bool"),
        7: ("has_output_redirect", "bool"), 8: ("parsing_result", "message"),
        9: ("requested_sandbox_policy", "message"),
        10: ("file_output_threshold_bytes", "uint64"),
        11: ("is_background", "bool"), 12: ("skip_approval", "bool"),
        13: ("timeout_behavior", "enum"), 14: ("hard_timeout", "int32"),
        15: ("description", "string"), 16: ("classifier_result", "message"),
        17: ("close_stdin", "bool"), 18: ("output_notification", "message"),
        19: ("smart_mode_approval", "message"),
        20: ("hook_approval_requirement", "message"),
        21: ("conversation_id", "string"),
    },
    "shell_stream_args": "shell_args",
    "write_args": {
        1: ("path", "string"), 2: ("file_text", "string"),
        3: ("tool_call_id", "string"),
        4: ("return_file_content_after_write", "bool"),
        5: ("file_bytes", "bytes"), 6: ("encoding_hint", "string"),
    },
    "delete_args": {1: ("path", "string"), 2: ("tool_call_id", "string")},
    "grep_args": {
        1: ("pattern", "string"), 2: ("path", "string"),
        3: ("glob", "string"), 4: ("output_mode", "string"),
        5: ("context_before", "int32"), 6: ("context_after", "int32"),
        7: ("context", "int32"), 8: ("case_insensitive", "bool"),
        9: ("type", "string"), 10: ("head_limit", "int32"),
        11: ("multiline", "bool"), 12: ("sort", "string"),
        13: ("sort_ascending", "bool"), 14: ("tool_call_id", "string"),
        15: ("sandbox_policy", "message"), 16: ("offset", "int32"),
    },
    "read_args": {
        1: ("path", "string"), 2: ("tool_call_id", "string"),
        4: ("offset", "int32"), 5: ("limit", "uint32"),
        6: ("encoding_hint", "string"),
    },
    "redacted_read_args": "read_args",
    "ls_args": {
        1: ("path", "string"), 2: ("ignore", "string"),
        3: ("tool_call_id", "string"), 4: ("sandbox_policy", "message"),
        5: ("timeout_ms", "uint32"),
    },
    "diagnostics_args": {
        1: ("path", "string"), 2: ("tool_call_id", "string"),
    },
    "request_context_args": {
        2: ("notes_session_id", "string"), 3: ("workspace_id", "string"),
        4: ("read_only_pinned_tree_sha", "string"),
        5: ("read_only_plugin_cache_root", "string"),
        7: ("use_cached", "bool"),
    },
    "mcp_args": {
        1: ("name", "string"), 3: ("tool_call_id", "string"),
        4: ("provider_identifier", "string"), 5: ("tool_name", "string"),
        6: ("smart_mode_approval", "message"),
        7: ("smart_mode_approval_only", "bool"),
        8: ("skip_approval", "bool"), 9: ("server_identifier", "string"),
    },
    "background_shell_spawn_args": {
        1: ("command", "string"), 2: ("working_directory", "string"),
        3: ("tool_call_id", "string"), 4: ("parsing_result", "message"),
        5: ("sandbox_policy", "message"),
        6: ("enable_write_shell_stdin_tool", "bool"),
        7: ("description", "string"), 8: ("classifier_result", "message"),
        9: ("output_notification", "message"),
        10: ("smart_mode_approval", "message"),
        11: ("hook_approval_requirement", "message"),
        12: ("skip_approval", "bool"), 13: ("conversation_id", "string"),
    },
    "list_mcp_resources_exec_args": {1: ("server", "string")},
    "read_mcp_resource_exec_args": {
        1: ("server", "string"), 2: ("uri", "string"),
        3: ("download_path", "string"), 4: ("tool_call_id", "string"),
        5: ("smart_mode_approval", "message"),
    },
    "fetch_args": {1: ("url", "string"), 2: ("tool_call_id", "string")},
    "record_screen_args": {
        1: ("mode", "enum"), 2: ("tool_call_id", "string"),
        3: ("save_as_filename", "string"),
    },
    "computer_use_args": {
        1: ("tool_call_id", "string"), 2: ("actions", "message"),
    },
    "write_shell_stdin_args": {
        1: ("shell_id", "uint32"), 2: ("chars", "string"),
    },
    "execute_hook_args": {1: ("request", "message")},
    "subagent_args": {
        1: ("tool_call_id", "string"), 2: ("subagent_type", "string"),
        3: ("model_id", "string"), 4: ("prompt", "string"),
        5: ("readonly", "bool"), 6: ("resume_agent_id", "string"),
        7: ("run_in_background", "bool"),
        8: ("continuation_config", "message"),
        9: ("parent_conversation_id", "string"),
        10: ("api_key_credentials", "message"),
        11: ("azure_credentials", "message"),
        12: ("bedrock_credentials", "message"), 13: ("interrupt", "bool"),
        14: ("mode", "enum"), 15: ("fork_agent_id", "string"),
        16: ("root_parent_conversation_id", "string"),
        17: ("selected_context", "message"),
        18: ("direct_meta_parent_child_subagent", "bool"),
        19: ("environment", "enum"), 20: ("cloud_base_branch", "string"),
    },
    "force_background_shell_args": {1: ("tool_call_id", "string")},
    "force_background_subagent_args": {1: ("tool_call_id", "string")},
    "mcp_state_exec_args": {
        1: ("server_identifiers", "string"), 2: ("kick_only", "bool"),
    },
    "subagent_await_args": {
        1: ("agent_id", "string"), 2: ("timeout_ms", "uint32"),
    },
    "smart_mode_classifier_args": {
        1: ("tool_call_id", "string"),
        2: ("parent_conversation_id", "string"), 3: ("target", "message"),
        4: ("conversation_context", "message"),
    },
    "canvas_diagnostics_args": {
        1: ("path", "string"), 2: ("tool_call_id", "string"),
    },
    "shell_allowlist_precheck_args": {
        1: ("command", "string"), 2: ("working_directory", "string"),
        3: ("parsing_result", "message"),
        4: ("classifier_result", "message"), 5: ("tool_call_id", "string"),
    },
    "mcp_allowlist_precheck_args": {
        1: ("provider_identifier", "string"), 2: ("tool_name", "string"),
        3: ("tool_call_id", "string"),
    },
    "web_fetch_allowlist_precheck_args": {
        1: ("url", "string"), 2: ("tool_call_id", "string"),
    },
    "git_diff_request": {
        1: ("cwd", "string"), 2: ("ref", "string"),
        3: ("base_ref", "string"), 4: ("merge_base", "bool"),
        5: ("target_paths", "string"),
        6: ("unified_context_lines", "int32"),
        7: ("max_untracked_files", "int32"), 8: ("output_format", "enum"),
        9: ("submodule_recurse_depth", "int32"),
        10: ("include_space_changes", "bool"),
        11: ("committed_only", "bool"), 12: ("compute_patch_id", "bool"),
        13: ("return_head_sha", "bool"), 14: ("max_response_bytes", "int32"),
    },
    "pi_read_args": {
        1: ("path", "string"), 2: ("offset", "int32"),
        3: ("limit", "int32"),
    },
    "pi_bash_args": {1: ("command", "string"), 2: ("timeout", "double")},
    "pi_edit_args": {1: ("path", "string"), 2: ("edits", "message")},
    "pi_write_args": {1: ("path", "string"), 2: ("content", "string")},
    "pi_grep_args": {
        1: ("pattern", "string"), 2: ("path", "string"),
        3: ("glob", "string"), 4: ("ignore_case", "bool"),
        5: ("literal", "bool"), 6: ("context", "int32"),
        7: ("limit", "int32"),
    },
    "pi_find_args": {
        1: ("pattern", "string"), 2: ("path", "string"),
        3: ("limit", "int32"),
    },
    "pi_ls_args": {1: ("path", "string"), 2: ("limit", "int32")},
    "conversation_search_args": {
        1: ("query", "string"), 2: ("tool_call_id", "string"),
        3: ("limit", "int32"),
    },
}


def _native_argument_schema(oneof_name: str) -> dict:
    schema = NATIVE_ARGUMENT_SCHEMAS.get(oneof_name, {})
    if isinstance(schema, str):
        return NATIVE_ARGUMENT_SCHEMAS[schema]
    return schema


def _generic_arguments(payload: bytes, oneof_name: str = "") -> dict:
    raw = RawMessage.decode(payload)
    schema = _native_argument_schema(oneof_name)
    arguments = {}
    for field in raw.fields:
        name, value_type = schema.get(
            field.number, (str(field.number), None)
        )
        if value_type in ("message", "bytes"):
            value = field.value
        elif value_type == "double" and field.wire_type == 1:
            value = struct.unpack("<d", struct.pack("<Q", field.value))[0]
        elif value_type == "bool":
            value = bool(field.value)
        elif field.wire_type == 0:
            value = int(field.value)
        elif field.wire_type in (1, 5):
            value = field.value
        elif field.wire_type == 2:
            try:
                value = field.value.decode()
            except UnicodeDecodeError:
                value = field.value
        else:
            value = field.value
        if name in arguments:
            current = arguments[name]
            if not isinstance(current, list):
                current = [current]
            current.append(value)
            arguments[name] = current
        else:
            arguments[name] = value
    return arguments


def decode_tool_call(payload: bytes) -> ToolCall | UnknownToolCall | None:
    message = RawMessage.decode(payload)
    exec_payload = message.first_bytes(2)
    if not exec_payload:
        return None
    execution = RawMessage.decode(exec_payload)
    server_message_id = _int(execution, 1)
    exec_id = _text(execution, 15)

    mcp_payload = execution.first_bytes(11)
    if mcp_payload is not None:
        mcp = RawMessage.decode(mcp_payload)
        name = _text(mcp, 1) or _text(mcp, 5)
        tool_name = _text(mcp, 5) or name
        call_id = _text(mcp, 3) or exec_id
        return ToolCall(
            id=call_id,
            name=name or tool_name or "mcp",
            arguments=_decode_map(mcp, 2),
            provider_identifier=_text(mcp, 4),
            tool_name=tool_name,
            server_identifier=_text(mcp, 9),
            native=False,
            server_message_id=server_message_id,
            exec_id=exec_id,
            field_number=11,
            oneof_name="mcp_args",
            payload_type="agent.v1.McpArgs",
            arguments_raw=mcp_payload,
        )

    for field in execution.fields:
        if field.wire_type != 2 or field.number in (15, 19):
            continue
        definition = EXEC_SERVER_TOOL_FIELDS.get(field.number)
        if definition is None:
            return UnknownToolCall(
                field.number,
                "field_" + str(field.number),
                field.value,
                server_message_id,
                exec_id,
            )
        oneof_name, payload_type = definition
        return ToolCall(
            id=exec_id,
            name=oneof_name.removesuffix("_args"),
            arguments=_generic_arguments(field.value, oneof_name),
            tool_name=oneof_name.removesuffix("_args"),
            native=True,
            server_message_id=server_message_id,
            exec_id=exec_id,
            field_number=field.number,
            oneof_name=oneof_name,
            payload_type=payload_type,
            arguments_raw=field.value,
        )
    return None


def encode_mcp_tools(tools) -> bytes:
    definitions = []
    for tool in tools:
        parameters = tool.parameters
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, str):
            parameters = json.dumps(parameters, separators=(",", ":"))
        definition = _message(
            _string(1, tool.name),
            _string(4, tool.provider_identifier),
            _string(5, tool.name),
            _string(2, tool.description),
            _string(6, parameters),
        )
        definitions.append(_bytes(1, definition))
    return _message(*definitions)




def exchange_api_key(
    api_key: str,
    *,
    url: str = KEY_EXCHANGE_URL,
) -> dict:
    request = Request(
        url,
        data=b"{}",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=KEY_EXCHANGE_TIMEOUT) as response:
        return json.load(response)


def discover_models(
    token: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    client_version: str = DEFAULT_CLIENT_VERSION,
) -> None:
    request = Request(
        urljoin(base_url.rstrip("/") + "/", AVAILABLE_MODELS_PATH),
        data=bytes.fromhex("28013801"),
        headers={
            "Authorization": "Bearer " + token,
            "Connect-Protocol-Version": "1",
            "Content-Type": "application/proto",
            "User-Agent": "connect-es/1.6.1",
            "x-cursor-client-type": "cli",
            "x-cursor-client-version": client_version,
            "x-ghost-mode": "true",
            "x-request-id": str(uuid.uuid4()),
        },
        method="POST",
    )
    with urlopen(request, timeout=KEY_EXCHANGE_TIMEOUT) as response:
        response.read()


def _token_expiration(payload: Mapping[str, object], exchanged_at: float) -> float:
    for key in ("expiresAt", "expires_at"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for key in ("expiresIn", "expires_in"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return exchanged_at + float(value)
    token = payload.get("accessToken")
    if isinstance(token, str):
        parts = token.split(".")
        if len(parts) == 3:
            try:
                import base64
                encoded = parts[1] + "=" * (-len(parts[1]) % 4)
                claims = json.loads(base64.urlsafe_b64decode(encoded).decode())
                expiration = claims.get("exp")
                if isinstance(expiration, (int, float)):
                    return float(expiration)
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                pass
    return exchanged_at + ACCESS_TOKEN_LIFETIME


def _read_cached_token(path: str, now: float) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            cached = json.load(stream)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(cached, dict):
        return None
    token = cached.get("access_token")
    expires_at = cached.get("expires_at")
    if (
        isinstance(token, str)
        and token
        and isinstance(expires_at, (int, float))
        and expires_at > now + ACCESS_TOKEN_REFRESH_MARGIN
    ):
        return token
    return None


def _write_cached_token(path: str, token: str, expires_at: float) -> None:
    directory = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix=".cursor-auth-", dir=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"access_token": token, "expires_at": expires_at}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
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


def get_access_token(
    api_key: str,
    *,
    cache_path: str = AUTH_CACHE_PATH,
    lock_path: str = AUTH_LOCK_PATH,
) -> str:
    if not api_key:
        raise ValueError("api_key is required")
    directory = os.path.dirname(cache_path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        now = time.time()
        token = _read_cached_token(cache_path, now)
        if token is not None:
            return token
        payload = exchange_api_key(api_key)
        token = payload.get("accessToken")
        if not isinstance(token, str) or not token:
            raise ValueError("Cursor key exchange returned no access token")
        # The official client discovers models immediately after authentication.
        # This may warm shared account/model routing before generation, but we do
        # not yet know whether it affects the observed post-idle backend stalls.
        discover_models(token)
        _write_cached_token(cache_path, token, _token_expiration(payload, now))
        return token
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


class CursorClient:
    def __init__(
        self,
        token: str,
        *,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        client_version: str = DEFAULT_CLIENT_VERSION,
        timeout: float | None = DEFAULT_TIMEOUT,
        tools=(),
        user_config: UserMessageConfig | None = None,
        run_config: RunConfig | None = None,

    ):
        if not token:
            raise ValueError("token is required")
        self.token = token
        self.base_url = base_url.rstrip("/") + "/"
        self.client_version = client_version
        self.timeout = timeout
        self.model = model
        self.tools = tuple(tools)
        self.user_config = user_config or UserMessageConfig()
        self.run_config = run_config or RunConfig()


    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": "Bearer " + self.token,
            "x-cursor-client-type": "cli",
            "x-cursor-client-version": self.client_version,
        }

    def handle_frame(self, frame: CursorFrame) -> None:
        pass


    def run(
        self,
        prompt: str,
        model: str | None = None,
        *,
        tools=None,
        history=(),
        user_config: UserMessageConfig | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResult:
        request_id = str(uuid.uuid4())
        model = self.model if model is None else model
        tools = self.tools if tools is None else tuple(tools)
        effective_run_config = run_config or self.run_config
        conversation_id = (
            effective_run_config.conversation_id or _SESSION_CONVERSATION_ID
        )
        request_started_ms = int(time.time() * 1000)
        _debug_bidi_event(
            "request_started",
            request_id=request_id,
            conversation_id=conversation_id,
            model=model,
        )
        input_payload = build_run_request(
            prompt,
            model,
            tools=tools,
            history=history,
            user_config=user_config or self.user_config,
            run_config=effective_run_config,
            conversation_id=conversation_id,
        )
        prefetched_blobs = extract_prefetched_blobs(input_payload)
        downlink_body = ConnectFrame.from_decoded(
            build_bidi_request_id(request_id)
        ).encode()
        frames: list[CursorFrame] = []
        text_parts: list[str] = []
        tool_calls: list[ToolCall | UnknownToolCall] = []
        seen_tool_calls = set()
        turn_ended = False
        checkpoint_updates = []
        eos_metadata = None
        eos_error = None
        usage = None
        response_boundary_seen = False
        accounting_anchor_ms = request_started_ms
        blob_response_generation = 0

        def receive(connect: ConnectFrame) -> None:
            nonlocal turn_ended, eos_metadata, eos_error, usage
            nonlocal response_boundary_seen, accounting_anchor_ms
            nonlocal blob_response_generation
            frame = CursorFrame.decode(
                connect, "IN", {"connection_id": request_id}
            )
            frames.append(frame)
            self.handle_frame(frame)
            if DEBUG:
                _debug_bidi_event(
                    "downlink_frame",
                    request_id=request_id,
                    conversation_id=conversation_id,
                    classification=frame.classification,
                    flags=connect.flags,
                    decoded_payload_hex=connect.decoded_payload.hex(),
                )
            if is_generation_progress(frame.message):
                blob_response_generation += 1
                clear_timeout = getattr(
                    transport, "clear_post_blob_timeout", None
                )
                if clear_timeout is not None:
                    clear_timeout()
            if isinstance(frame.message, AnswerText):
                text_parts.append(frame.message.text)
            if (
                frame.classification
                == "agent_server.interaction_update.heartbeat"
            ):
                transport.reset_heartbeat_timeout()
            if not connect.eos:
                kv_response = build_kv_response(
                    connect.decoded_payload, prefetched_blobs
                )
                if kv_response is not None:
                    if (
                        isinstance(frame.message, KVServerMessage)
                        and frame.message.subtype == "get_blob_args"
                    ):
                        blob_response_generation += 1
                        generation = blob_response_generation
                        clear_timeout = getattr(
                            transport, "clear_post_blob_timeout", None
                        )
                        if clear_timeout is not None:
                            clear_timeout()

                        def blob_appended(response, generation=generation):
                            appended(response)
                            if generation == blob_response_generation:
                                arm_timeout = getattr(
                                    transport, "arm_post_blob_timeout", None
                                )
                                if arm_timeout is not None:
                                    transport._post_blob_debug = {
                                        "request_id": request_id,
                                        "conversation_id": conversation_id,
                                        "append_seqno": generation,
                                    }
                                    arm_timeout(
                                        POST_BLOB_PROGRESS_TIMEOUT
                                    )

                        append(kv_response, callback=blob_appended)
                    else:
                        append(kv_response)
                call = decode_tool_call(connect.decoded_payload)
                if call is not None:
                    identity = (
                        type(call),
                        call.field_number,
                        getattr(call, "id", ""),
                        call.exec_id,
                    )
                    if identity not in seen_tool_calls:
                        seen_tool_calls.add(identity)
                        tool_calls.append(call)
            if (
                frame.classification
                == "agent_server.interaction_update.turn_ended"
            ):
                turn_ended = True
                usage = parse_turn_usage(frame.message.update_payload)
                transport.close()
            elif (
                frame.classification
                == "agent_server.conversation_checkpoint_update"
            ):
                checkpoint_updates.append(frame)
                timestamp = _checkpoint_timestamp(frame)
                if timestamp is not None:
                    accounting_anchor_ms = timestamp
            elif (
                not response_boundary_seen
                and not connect.eos
                and is_response_boundary_blob_write(
                    connect.decoded_payload, request_id
                )
            ):
                response_boundary_seen = True
                turn_ended = True
                if tool_calls:
                    append(
                        build_user_cancelled_message(),
                        callback=cancelled,
                    )
                else:
                    transport.start_grace_period(
                        RESPONSE_USAGE_GRACE_TIMEOUT
                    )
            if connect.eos:
                eos_metadata = frame.eos_metadata
                eos_error = frame.eos_error

        decoder = ConnectStreamDecoder(receive)
        downlink_url = urljoin(
            self.base_url, AGENT_RUNSSE_PATH
        )
        append_url = urljoin(
            self.base_url, BIDI_APPEND_PATH
        )
        request_headers = {
            **self.headers,
            "x-ghost-mode": "true",
            "x-request-id": request_id,
            "x-original-request-id": request_id,
        }
        downlink_headers = {
            **request_headers,
            "Accept": "application/connect+proto",
            "Content-Type": "application/connect+proto",
            "Connect-Protocol-Version": "1",
            "Connect-Accept-Encoding": "gzip",
        }
        append_result = []
        append_started = False
        append_seqno = -1
        append_queue = deque()
        append_in_flight = 0

        def appended(response):
            append_result.append(response)
            if response["status"] != 200:
                raise SSEError(
                    f"BidiAppend returned HTTP {response['status']}"
                )

        def cancelled(response):
            appended(response)
            transport.close()

        def pump_append_queue():
            nonlocal append_in_flight
            while (
                append_in_flight < BIDI_APPEND_PIPELINE_DEPTH
                and append_queue
                and not transport.closed
            ):
                (
                    seqno,
                    payload,
                    wrapped_payload,
                    classification,
                    callback,
                ) = append_queue.popleft()
                append_in_flight += 1
                _debug_bidi_event(
                    "bidi_append_started",
                    request_id=request_id,
                    conversation_id=conversation_id,
                    append_seqno=seqno,
                    classification=classification,
                    pipeline_depth=append_in_flight,
                    payload_hex=payload.hex(),
                    wrapped_payload_hex=wrapped_payload.hex(),
                )

                def completed(
                    response,
                    seqno=seqno,
                    classification=classification,
                    callback=callback,
                ):
                    nonlocal append_in_flight
                    _debug_bidi_event(
                        "bidi_append_completed",
                        request_id=request_id,
                        conversation_id=conversation_id,
                        append_seqno=seqno,
                        classification=classification,
                        pipeline_depth=append_in_flight,
                        status=response["status"],
                        headers=response["headers"],
                        body_hex=response["body"].hex(),
                    )
                    append_in_flight -= 1
                    callback(response)
                    pump_append_queue()

                transport.post(
                    append_url,
                    wrapped_payload,
                    headers={
                        **request_headers,
                        "Content-Type": "application/proto",
                        "Accept": "application/proto",
                    },
                    callback=completed,
                )

        def append(payload, callback=appended):
            nonlocal append_seqno
            append_seqno += 1
            seqno = append_seqno
            wrapped_payload = build_bidi_append(
                request_id,
                payload,
                append_seqno=seqno,
            )
            try:
                classification = decode_cursor_payload(
                    payload, "OUT"
                ).classification
            except ValueError:
                classification = "decode_error"
            append_queue.append((
                seqno,
                payload,
                wrapped_payload,
                classification,
                callback,
            ))
            pump_append_queue()

        def downlink_ready(status, headers):
            nonlocal append_started
            _debug_bidi_event(
                "downlink_ready",
                request_id=request_id,
                conversation_id=conversation_id,
                status=status,
                headers=headers,
            )
            if append_started:
                return
            append_started = True
            append(input_payload)

        with SSEClient(
            downlink_url,
            callback=lambda event: None,
            headers=downlink_headers,
            timeout=self.timeout,
            stream_callback=decoder.feed,
            accepted_content_types=("application/connect+proto",),
            method="POST",
            body=downlink_body,
            headers_callback=downlink_ready,
        ) as transport:
            transport.run_forever(
                timeout=self.timeout,
                heartbeat_timeout=HEARTBEAT_TIMEOUT,
            )
            if not append_result and not frames:
                raise SSEError("BidiAppend did not complete")
        decoder.finish()

        if eos_error and not response_boundary_seen:
            raise SSEError(eos_error)
        if turn_ended and usage is None:
            for attempt in range(USAGE_LOOKUP_ATTEMPTS):
                usage = get_filtered_usage(
                    self.token,
                    self.base_url,
                    conversation_id,
                    accounting_anchor_ms,
                    request_started_ms,
                )
                if usage is not None:
                    break
                if attempt + 1 < USAGE_LOOKUP_ATTEMPTS:
                    time.sleep(USAGE_LOOKUP_RETRY_DELAY)
        _debug_bidi_event(
            "request_completed",
            request_id=request_id,
            conversation_id=conversation_id,
            frame_count=len(frames),
            turn_ended=turn_ended,
            eos_error=eos_error,
        )
        return RunResult(
            frames,
            "".join(text_parts),
            tool_calls,
            turn_ended,
            checkpoint_updates,
            eos_metadata,
            eos_error,
            usage,
        )


def run(
    prompt: str,
    *,
    api_key: str,
    model: str,
    tools=(),
    history=(),
    timeout: float | None = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_BASE_URL,
    client_version: str = DEFAULT_CLIENT_VERSION,
    user_config: UserMessageConfig | None = None,
    run_config: RunConfig | None = None,
) -> RunResult:
    """Perform one independent stateless Cursor Agent request."""
    deadline = (
        None if timeout is None else time.monotonic() + timeout
    )
    token = get_access_token(api_key)

    remaining_timeout = None
    if deadline is not None:
        remaining_timeout = deadline - time.monotonic()
        if remaining_timeout <= 0:
            raise SSEError("request deadline exceeded")

    normalized_tools = tuple(
        item if isinstance(item, ToolDefinition)
        else ToolDefinition(**item)
        for item in tools
    )
    normalized_history = []
    for item in history:
        if isinstance(item, ConversationMessage):
            normalized_history.append(item)
            continue
        values = dict(item)
        values["tool_calls"] = tuple(
            call if isinstance(call, ToolCall) else ToolCall(**call)
            for call in values.get("tool_calls", ())
        )
        normalized_history.append(ConversationMessage(**values))

    return CursorClient(
        token,
        base_url=base_url,
        client_version=client_version,
        timeout=remaining_timeout,
        model=model,
        tools=normalized_tools,
        user_config=user_config,
        run_config=run_config,
    ).run(prompt, history=normalized_history)


def _chat_content_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError("message content must be a string, list, or None")
    parts = []
    for item in content:
        if not isinstance(item, dict):
            raise TypeError("content parts must be objects")
        if item.get("type") not in ("text", "input_text"):
            raise ValueError(
                f"Cursor does not support content type {item.get('type')!r}"
            )
        parts.append(str(item.get("text", "")))
    return "".join(parts)


def _openai_tools(tools):
    result = []
    for item in tools or ():
        if not isinstance(item, dict) or item.get("type") != "function":
            raise ValueError("only OpenAI function tools are supported")
        function = item.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            raise ValueError("function tool requires a name")
        result.append({
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
            "provider_identifier": "openai",
        })
    return result


def _openai_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")

    parsed = []
    system_parts = []
    active_user_index = -1

    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("messages must contain objects")
        if message.get("audio"):
            raise ValueError("Cursor does not support audio input")
        role = message.get("role") or "user"
        content = _chat_content_text(message.get("content"))

        if role == "system":
            system_parts.append(content)
            continue

        converted = {"role": role, "content": content}
        if role == "assistant":
            calls = []
            for item in message.get("tool_calls") or ():
                function = item.get("function") or {}
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append({
                    "id": item.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": arguments,
                    "tool_name": function.get("name", ""),
                    "provider_identifier": "openai",
                })
            converted["tool_calls"] = calls
        elif role == "tool":
            converted["tool_call_id"] = message.get("tool_call_id", "")
            converted["tool_name"] = message.get("name", "")
        elif role != "user":
            raise ValueError(f"unsupported message role: {role!r}")

        parsed.append(converted)
        if role == "user":
            active_user_index = len(parsed) - 1

    if not parsed:
        raise ValueError("messages must contain a non-system message")

    if system_parts:
        addendum = (
            "<system-addendum>\n"
            + "\n\n".join(system_parts)
            + "\n</system-addendum>"
        )
        for message in parsed:
            if message["role"] == "user":
                message["content"] = addendum + "\n\n" + message["content"]
                break

    if active_user_index == len(parsed) - 1:
        prompt = parsed[active_user_index]["content"]
        history = parsed[:active_user_index]
    elif parsed[-1]["role"] == "tool":
        prompt = parsed[-1]["content"]
        history = [
            *parsed[:-1],
            {**parsed[-1], "content": "Input received."},
        ]
    else:
        prompt = ""
        history = parsed

    return prompt, history


def _native_call_name(call: ToolCall) -> str:
    if call.oneof_name:
        words = call.oneof_name.removesuffix("_args").split("_")
        return "".join(word.capitalize() for word in words)
    return call.name or "ExecServerMessage"


def _native_call_content(call: ToolCall):
    for key in ("content", "command", "code", "text", "1"):
        if key in call.arguments:
            return call.arguments[key]
    return call.arguments


def _native_call_params(call: ToolCall) -> dict:
    schema = _native_argument_schema(call.oneof_name)
    return {
        schema.get(int(key), (str(key), None))[0]
        if str(key).isdigit() else str(key): value
        for key, value in call.arguments.items()
    }


def _native_repl_code(call: ToolCall) -> str:
    name = _native_call_name(call)
    params = _native_call_params(call)
    content = _native_call_content(call)
    content_literal = json.dumps(str(content), ensure_ascii=False)
    if name == "ExecServerMessage":
        return f"think({content_literal})"
    if name == "ShellStream":
        command = params.get("command", "")
        working_directory = params.get("working_directory")
        if working_directory:
            command = f"cd -- {shlex.quote(working_directory)} && {command}"
        timeout = params.get("timeout")
        if timeout is not None:
            timeout = timeout / 1000
            if timeout.is_integer():
                timeout = int(timeout)
        return (
            f"bash(command={command!r}, timeout={timeout!r}, "
            f"bg={bool(params.get('is_background', False))!r})"
        )
    if name == "Grep":
        arguments = {
            "pattern": params.get("pattern", ""),
            "path": params.get("path"),
            "glob": params.get("glob"),
            "file_type": params.get("type"),
            "context": params.get("context"),
            "case_insensitive": params.get("case_insensitive"),
            "multiline": params.get("multiline"),
        }
        rendered = ", ".join(
            f"{key}={value!r}"
            for key, value in arguments.items()
            if value is not None
        )
        return f"grep({rendered})"
    if name == "Read":
        path = params.get("path", "")
        offset = params.get("offset")
        limit = params.get("limit")
        if offset is None and limit is None:
            return f"view({path!r})"
        if offset is None:
            slice_spec = f":{limit}"
        elif limit is None:
            slice_spec = f"{offset}:"
        else:
            slice_spec = f"{offset}:{offset + limit}"
        return f"print(read({path!r})[{slice_spec}])"
    return f"# unsupported tool call: {name}({params!r})"


def _openai_tool_call(call: ToolCall) -> dict:
    if call.native:
        name = "repl_execute"
        arguments = {"code": _native_repl_code(call)}
    else:
        name = call.name
        arguments = call.arguments
    return {
        "id": call.id or call.exec_id or "tool_" + uuid.uuid4().hex,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, separators=(",", ":")),
        },
    }


def _cursor_input_bytes(body: dict) -> int:
    payload = {"messages": body.get("messages")}
    if body.get("tools"):
        payload["tools"] = body["tools"]
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def _detect_cursor_model_calibration(
    api_key: str, model: str, timeout: float | None
) -> None:
    sample = _CURSOR_CALIBRATION_SAMPLE
    prompts = (sample, sample + "\n\n" + sample)
    usages = []
    for prompt in prompts:
        result = run(
            prompt,
            api_key=api_key,
            model=model,
            timeout=timeout,
            run_config=RunConfig(conversation_id=str(uuid.uuid4())),
        )
        if result.usage is None:
            raise ValueError(
                f"Cursor returned no usage while calibrating {model}"
            )
        usages.append(result.usage.input_tokens)

    added_bytes = len(("\n\n" + sample).encode("utf-8"))
    variable_tokens = usages[1] - usages[0]
    if variable_tokens <= 0:
        raise ValueError(
            f"Cursor returned invalid differential usage while calibrating {model}"
        )
    ratio = variable_tokens / added_bytes
    first_bytes = len(sample.encode("utf-8"))
    system_tokens = round(usages[0] - ratio * first_bytes)
    if system_tokens < 0:
        raise ValueError(
            f"Cursor returned invalid system prompt usage while calibrating {model}"
        )

    CURSOR_MODEL_CALIBRATION[model] = {
        "system_prompt_tokens": system_tokens,
        "variable_tokens_per_byte": ratio,
        "input_cost": _CURSOR_DEFAULT_INPUT_COST,
        "cache_read_cost": _CURSOR_DEFAULT_CACHE_READ_COST,
    }
    print(
        f"Detected Cursor calibration for {model}: "
        f"system_prompt_tokens={system_tokens}, "
        f"variable_tokens_per_byte={ratio}"
    )


def _cursor_model_stats(model: str) -> dict:
    stats = _MODEL_CURSOR_STATS.get(model)
    if stats is None:
        stats = {
            "conversation_id": str(uuid.uuid4()),
            "previous_request_bytes": None,
            "previous_reported_cost": None,
            "accumulated_excess": 0.0,
            "request_count": 0,
            "ratio_samples": [],
        }
        _MODEL_CURSOR_STATS[model] = stats
    return stats


def _cursor_tokens_per_byte(model: str, stats: dict) -> float:
    samples = sorted(stats["ratio_samples"])
    if not samples:
        return CURSOR_MODEL_CALIBRATION[model]["variable_tokens_per_byte"]
    count = max(1, (len(samples) + 2) // 3)
    return sum(samples[:count]) / count


def _estimated_fresh_input_tokens(
    model: str, input_bytes: int, stats: dict
) -> float:
    calibration = CURSOR_MODEL_CALIBRATION[model]
    return (
        calibration["system_prompt_tokens"]
        + input_bytes * _cursor_tokens_per_byte(model, stats)
    )


def _reported_cursor_cost_equivalent(
    model: str, usage: TurnUsage
) -> float:
    calibration = CURSOR_MODEL_CALIBRATION[model]
    cache_ratio = (
        calibration["cache_read_cost"] / calibration["input_cost"]
    )
    return (
        usage.input_tokens
        - usage.cache_read_tokens
        + usage.cache_read_tokens * cache_ratio
    )


def _should_rotate_cursor_conversation(
    model: str, stats: dict, input_bytes: int
) -> bool:
    previous_bytes = stats["previous_request_bytes"]
    previous_cost = stats["previous_reported_cost"]
    if previous_bytes is None or previous_cost is None:
        return False
    if stats.get("request_count", 0) < _CURSOR_ROTATION_MIN_REQUESTS:
        return False
    ratio = _cursor_tokens_per_byte(model, stats)
    predicted_cost = max(
        0.0, previous_cost + ratio * (input_bytes - previous_bytes)
    )
    estimated_fresh = _estimated_fresh_input_tokens(
        model, input_bytes, stats
    )
    predicted_excess = max(0.0, predicted_cost - estimated_fresh)
    return (
        stats["accumulated_excess"] + predicted_excess
        > estimated_fresh
    )


def _record_cursor_usage(
    model: str, stats: dict, input_bytes: int, usage: TurnUsage
) -> None:
    calibration = CURSOR_MODEL_CALIBRATION[model]
    previous_bytes = stats["previous_request_bytes"]
    if previous_bytes is None:
        variable_tokens = (
            usage.input_tokens - calibration["system_prompt_tokens"]
        )
        sample = variable_tokens / input_bytes if input_bytes else 0
        copies = 3
    else:
        added_bytes = input_bytes - previous_bytes
        uncached_tokens = usage.input_tokens - usage.cache_read_tokens
        sample = (
            uncached_tokens / added_bytes
            if added_bytes >= _CURSOR_RATIO_MIN_BYTES else 0
        )
        copies = 1
    if _CURSOR_RATIO_MIN <= sample <= _CURSOR_RATIO_MAX:
        stats["ratio_samples"].extend([sample] * copies)
        del stats["ratio_samples"][:-_CURSOR_RATIO_SAMPLE_LIMIT]

    estimated_fresh = _estimated_fresh_input_tokens(
        model, input_bytes, stats
    )
    stats["accumulated_excess"] = max(
        0.0,
        stats["accumulated_excess"]
        + _reported_cursor_cost_equivalent(model, usage)
        - estimated_fresh,
    )
    stats["previous_request_bytes"] = input_bytes
    stats["previous_reported_cost"] = _reported_cursor_cost_equivalent(
        model, usage
    )
    stats["request_count"] = stats.get("request_count", 0) + 1


def _rotate_cursor_conversation(stats: dict) -> None:
    stats["conversation_id"] = str(uuid.uuid4())
    stats["previous_request_bytes"] = None
    stats["previous_reported_cost"] = None
    stats["accumulated_excess"] = 0.0
    stats["request_count"] = 0


def chat_completions(api_key: str, body: dict) -> dict:
    """Run one non-streaming OpenAI Chat Completions-compatible request."""
    if not isinstance(body, dict):
        raise TypeError("body must be a dictionary")
    if body.get("stream"):
        raise ValueError("streaming chat completions are not supported")
    prompt, history = _openai_messages(body.get("messages"))
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("model is required")
    timeout = body.get("timeout", DEFAULT_TIMEOUT)
    if model not in CURSOR_MODEL_CALIBRATION:
        _detect_cursor_model_calibration(api_key, model, timeout)
    stats = _cursor_model_stats(model)
    input_bytes = _cursor_input_bytes(body)
    estimated_input_tokens = round(
        _estimated_fresh_input_tokens(model, input_bytes, stats)
    )
    if _should_rotate_cursor_conversation(model, stats, input_bytes):
        _rotate_cursor_conversation(stats)
        estimated_input_tokens = round(
            _estimated_fresh_input_tokens(model, input_bytes, stats)
        )
    conversation_id = stats["conversation_id"]
    result = run(
        prompt,
        api_key=api_key,
        model=model,
        tools=_openai_tools(body.get("tools")),
        history=history,
        timeout=timeout,
        run_config=RunConfig(conversation_id=conversation_id),
    )
    if result.usage is not None:
        _record_cursor_usage(model, stats, input_bytes, result.usage)
    tool_calls = [
        _openai_tool_call(call)
        for call in result.tool_calls
        if isinstance(call, ToolCall)
    ]
    message = {"role": "assistant", "content": result.text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    response = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
    }
    if result.usage is not None:
        response["usage"] = {
            "prompt_tokens": estimated_input_tokens,
            "completion_tokens": result.usage.output_tokens,
        }
    return response


