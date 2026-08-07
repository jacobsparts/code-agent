"""
REPL attachments rendered as file reads.

For REPLAgent-based agents, attachments appear as if the user had read the file
on the REPL, maintaining the illusion of a continuous REPL session.

Example:
    agent.attach("config.json", '{"debug": true}')
    agent.usermsg("What's in the config?")
    result = agent.run_loop(max_turns=10)


When the attachment is invalidated (re-attached or detached), the content is
removed and only a small placeholder remains: [Attachment: config.json]

Attachments are typed. Text attachments carry rendered text that replaces their
placeholder; image attachments keep their placeholder in the text and are
projected separately as provider-neutral media.
"""

import base64
import json
import re
import struct

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryAttachment:
    content: str


@dataclass(frozen=True)
class TextAttachment:
    content: str


@dataclass(frozen=True)
class ImageAttachment:
    content: bytes
    media_type: str
    width: int
    height: int


class ImageDecodeError(ValueError):
    """Raised when a file looks like a supported image but cannot be parsed."""


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"


def detect_image_media_type(data: bytes) -> str | None:
    """Return the MIME type for supported image bytes, else None."""
    if not isinstance(data, (bytes, bytearray)):
        return None
    if data[:8] == PNG_SIGNATURE:
        return "image/png"
    if data[:3] == JPEG_SIGNATURE:
        return "image/jpeg"
    return None


def _png_dimensions(data: bytes) -> tuple[int, int]:
    # IHDR is required to be the first chunk: 8-byte signature, 4-byte length,
    # 4-byte type, then width and height as big-endian unsigned integers.
    if len(data) < 24 or data[12:16] != b"IHDR":
        raise ImageDecodeError("invalid PNG: missing IHDR header")
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        raise ImageDecodeError("invalid PNG: non-positive dimensions")
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    index = 2
    size = len(data)
    while index + 1 < size:
        if data[index] != 0xFF:
            raise ImageDecodeError("invalid JPEG: expected marker")
        marker = data[index + 1]
        index += 2
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:
            break
        if index + 2 > size:
            raise ImageDecodeError("invalid JPEG: truncated segment")
        (segment_length,) = struct.unpack(">H", data[index:index + 2])
        if segment_length < 2 or index + segment_length > size:
            raise ImageDecodeError("invalid JPEG: truncated segment")
        # SOF markers carry the frame dimensions; DHT/JPG/DAC are not SOF.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if segment_length < 7:
                raise ImageDecodeError("invalid JPEG: truncated frame header")
            height, width = struct.unpack(">HH", data[index + 3:index + 7])
            if width <= 0 or height <= 0:
                raise ImageDecodeError("invalid JPEG: non-positive dimensions")
            return width, height
        index += segment_length
    raise ImageDecodeError("invalid JPEG: no frame header found")


def image_dimensions(data: bytes, media_type: str) -> tuple[int, int]:
    """Parse pixel dimensions from image bytes without re-encoding them."""
    if media_type == "image/png":
        return _png_dimensions(data)
    if media_type == "image/jpeg":
        return _jpeg_dimensions(data)
    raise ImageDecodeError(f"unsupported image media type: {media_type}")


def make_image_attachment(data: bytes) -> ImageAttachment:
    """Build a typed image attachment from raw bytes."""
    media_type = detect_image_media_type(data)
    if media_type is None:
        raise ImageDecodeError("unsupported image format")
    data = bytes(data)
    width, height = image_dimensions(data, media_type)
    return ImageAttachment(
        content=data,
        media_type=media_type,
        width=width,
        height=height,
    )


def normalize_attachment_value(value) -> TextAttachment | ImageAttachment:
    """Normalize an attachment at the message boundary."""
    if isinstance(value, (TextAttachment, ImageAttachment)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return make_image_attachment(value)
    return TextAttachment(value if isinstance(value, str) else str(value))


def normalize_attachments(attachments) -> dict[str, TextAttachment | ImageAttachment]:
    return {
        name: normalize_attachment_value(value)
        for name, value in (attachments or {}).items()
    }


def normalize_message_attachments(message: dict) -> dict:
    if message.get("_attachments"):
        message["_attachments"] = normalize_attachments(message["_attachments"])
    return message


# ---------------------------------------------------------------------------
# Placeholder grammar
# ---------------------------------------------------------------------------
#
# Text:  [Attachment: app.py]
# Image: [Attachment: screenshot.png, 1440x1100, image/png]  (x is U+00D7)
#
# Attachment names may contain commas and spaces, so the image metadata suffix
# is only recognized when it strictly matches the grammar, anchored at the end.

_IMAGE_SUFFIX_RE = re.compile(
    r", (?P<width>[1-9][0-9]*)\u00d7(?P<height>[1-9][0-9]*), "
    r"(?P<media_type>image/[A-Za-z0-9][A-Za-z0-9.+-]*)$"
)

_PLACEHOLDER_RE = re.compile(r"\[Attachment: (?P<body>[^\]\n]+)\]")


def render_attachment_placeholder(name: str, value=None) -> str:
    """Render an attachment placeholder, including image metadata when present."""
    if isinstance(value, ImageAttachment):
        name = f"{name}, {value.width}\u00d7{value.height}, {value.media_type}"
    return f"[Attachment: {name}]"


def iter_placeholders(content: str):
    """Yield (match, parsed metadata) for every placeholder in text."""
    if not isinstance(content, str):
        return
    for match in _PLACEHOLDER_RE.finditer(content):
        body = match.group("body")
        suffix = _IMAGE_SUFFIX_RE.search(body)
        if suffix is None:
            parsed = {
                "name": body,
                "media_type": None,
                "width": None,
                "height": None,
            }
        else:
            parsed = {
                "name": body[:suffix.start()],
                "media_type": suffix.group("media_type"),
                "width": int(suffix.group("width")),
                "height": int(suffix.group("height")),
            }
        yield match, parsed


def encode_attachment_ref(ref):
    if isinstance(ref, MemoryAttachment):
        return {"__memory_attachment__": True, "content": ref.content}
    if isinstance(ref, TextAttachment):
        return {"__text_attachment__": True, "content": ref.content}
    if isinstance(ref, ImageAttachment):
        return {
            "__image_attachment__": True,
            "content": base64.b64encode(ref.content).decode("ascii"),
            "media_type": ref.media_type,
            "width": ref.width,
            "height": ref.height,
        }
    return ref


def decode_attachment_ref(ref):
    if not isinstance(ref, dict):
        return ref
    if ref.get("__memory_attachment__"):
        return MemoryAttachment(ref.get("content", ""))
    if ref.get("__text_attachment__"):
        return TextAttachment(ref.get("content", ""))
    if ref.get("__image_attachment__"):
        return ImageAttachment(
            content=base64.b64decode(ref.get("content") or ""),
            media_type=ref.get("media_type") or "",
            width=ref.get("width") or 0,
            height=ref.get("height") or 0,
        )
    return ref


def encode_attachment_refs(refs):
    return {name: encode_attachment_ref(ref) for name, ref in (refs or {}).items()}


def decode_attachment_refs(refs):
    return {name: decode_attachment_ref(ref) for name, ref in (refs or {}).items()}


class REPLAttachmentMixin:
    """Adds REPL-style attachment support."""

    def _ensure_setup(self):
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        if hasattr(self, '_pending_attachments'):
            return

        self._pending_attachments = {}

    def attach(self, name: str, content):
        """
        Add or update an attachment.

        Args:
            name: Identifier for this attachment (used as filename in synthetic read)
            content: Text, bytes for a supported image, a typed attachment value,
                or dict/list content (dicts/lists are JSON-serialized)
        """
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)

        self._invalidate_attachment(name)
        self._pending_attachments[name] = self._render_attachment(name, content)

    def detach(self, name: str):
        """
        Remove an attachment from context.

        Args:
            name: Identifier of attachment to remove
        """
        self._invalidate_attachment(name)
        self._pending_attachments.pop(name, None)

    def list_attachments(self) -> dict[str, TextAttachment | ImageAttachment]:
        """Get currently active attachments as typed values."""
        active = {}
        for msg in self.conversation.messages:
            for name, content in msg.get('_attachments', {}).items():
                active[name] = content
        active.update(self._pending_attachments)
        return active

    def _invalidate_attachment(self, name: str):
        """Remove an attachment from all messages."""
        for msg in self.conversation.messages:
            attachments = msg.get('_attachments')
            if attachments and name in attachments:
                del attachments[name]
                if not attachments:
                    del msg['_attachments']

    def _render_attachment(self, name: str, content):
        """Build a typed attachment value, numbering lines for text content."""
        if isinstance(content, (TextAttachment, ImageAttachment)):
            return content
        if isinstance(content, (bytes, bytearray)):
            return make_image_attachment(content)
        lines = str(content).split('\n')
        return TextAttachment(
            '\n'.join(f"{i+1:>5}→{line}" for i, line in enumerate(lines))
        )

    def _render_placeholder(self, name: str, value=None) -> str:
        """Render a placeholder that looks like a REPL read call."""
        return f">>> read({name!r})\n{render_attachment_placeholder(name, value)}"

    def usermsg(self, content, **kwargs):
        if self._pending_attachments:
            # Force new message — don't append to previous REPL output
            self._last_was_repl_output = False

            placeholders = "\n\n".join(
                self._render_placeholder(name, value)
                for name, value in self._pending_attachments.items()
            )
            content = placeholders + "\n\n" + (content if isinstance(content, str) else json.dumps(content))
            # Merge with any existing _attachments (e.g. from read-as-attach)
            existing = kwargs.get('_attachments', {})
            existing.update(self._pending_attachments)
            kwargs['_attachments'] = existing
            self._pending_attachments.clear()
        super().usermsg(content, **kwargs)
