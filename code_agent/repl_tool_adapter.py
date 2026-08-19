import ast
import json


REPL_EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "repl_execute",
        "description": (
            "Execute Python source in Code Agent's persistent Python REPL. "
            "Put all Python statements and calls to Code Agent functions in the code argument."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute in the persistent REPL.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


class ReplExecuteResponseError(Exception):
    pass


def _python_source(text):
    try:
        ast.parse(text, mode="exec")
        return True
    except SyntaxError:
        return False


def _join_python(parts):
    return "\n".join(part.rstrip("\n") for part in parts if part != "")


def repl_response_to_text(message):
    blocks = message["content"]
    calls = []
    for block in blocks:
        kind = block["type"]
        if kind == "tool_call":
            calls.append(block)
        elif kind not in ("text", "commentary"):
            raise NotImplementedError(
                f"Unknown REPL response content type: {kind!r}"
            )

    if not calls:
        text = _join_python(
            block["text"] for block in blocks
        )
        if not text:
            raise ReplExecuteResponseError(
                "Response must include a repl_execute tool call with valid Python."
            )
        if _python_source(text):
            return {**message, "content": [{"type": "text", "text": text}]}
        raise ReplExecuteResponseError(
            "Response without a repl_execute tool call must be valid Python. "
            "Use repl_execute with emit(...) for prose responses."
        )

    parts = []
    for block in blocks:
        kind = block["type"]
        if kind == "text":
            if block["text"]:
                parts.append(f"emit({block['text']!r})")
        elif kind == "commentary":
            if block["text"]:
                parts.append("# " + "\n# ".join(block["text"].split("\n")))
        elif kind == "tool_call":
            if block["name"] != "repl_execute":
                raise ReplExecuteResponseError(
                    f"Unexpected native tool: {block['name']!r}"
                )
            code = block["args"]["code"]
            if not isinstance(code, str):
                raise ReplExecuteResponseError("repl_execute code must be a string")
            parts.append(code)
        else:
            raise NotImplementedError(
                f"Unknown REPL response content type: {kind!r}"
            )

    text = _join_python(parts)
    if not _python_source(text):
        raise ReplExecuteResponseError(
            "repl_execute response must produce valid Python."
        )
    return {**message, "content": [{"type": "text", "text": text}]}


def _message_text(message):
    text = []
    for block in message["content"]:
        kind = block["type"]
        if kind == "text":
            text.append(block["text"])
        elif kind == "attachment":
            continue
        else:
            raise NotImplementedError(
                f"Unknown REPL history content type: {kind!r}"
            )
    return "\n".join(text)


def _strip_leading_input_echo(output, human_input):
    if not output or not human_input:
        return output
    echo = human_input.rstrip("\r\n")
    for newline in ("\r\n", "\n", "\r"):
        prefix = echo + newline
        if output.startswith(prefix):
            return output[len(prefix):]
    return output


def _split_repl_output_and_input(message, preceding_human_input=None):
    segments = message.get("_render_segments") or []
    inputs = [
        segment.get("content", "")
        for segment in segments
        if segment.get("type") == "input" and segment.get("content")
    ]
    human = message.get("_user_content")
    if human is not None and not inputs:
        inputs = [human]

    output = _message_text(message)
    for value in reversed(inputs):
        suffix = value + "\n"
        if output.endswith(suffix):
            before = output[:-len(suffix)]
            output = before.rstrip("\n") + ("\n" if before else "")
    attachments = [
        block for block in message["content"]
        if block["type"] == "attachment"
    ]
    return (
        _strip_leading_input_echo(output, preceding_human_input),
        inputs,
        attachments,
    )


def project_repl_tool_history(messages):
    projected = []
    call_number = 0
    index = 0
    pending_user_input = None

    while index < len(messages):
        message = messages[index]
        role = message["role"]

        if role != "assistant":
            projected.append(message)
            if role == "user":
                pending_user_input = message.get("_user_content", _message_text(message))
            index += 1
            continue

        call_number += 1
        call_id = f"repl_{call_number:06d}"
        projected.append({
            "role": "assistant",
            "content": [{
                "type": "tool_call",
                "id": call_id,
                "name": "repl_execute",
                "args": {"code": _message_text(message)},
            }],
        })

        output = ""
        human_inputs = []
        attachments = []
        if index + 1 < len(messages) and messages[index + 1]["role"] == "user":
            output, human_inputs, attachments = _split_repl_output_and_input(
                messages[index + 1],
                pending_user_input,
            )
            index += 1
        projected.append({
            "role": "tool",
            "content": [{"type": "text", "text": output}],
            "tool_call_id": call_id,
            "name": "repl_execute",
        })
        recovered = [{
            "role": "user",
            "content": [{"type": "text", "text": value}],
        } for value in human_inputs]
        if attachments:
            if recovered:
                recovered[-1]["content"].extend(attachments)
            else:
                recovered.append({
                    "role": "user",
                    "content": attachments,
                })
        projected.extend(recovered)
        pending_user_input = human_inputs[-1] if human_inputs else None
        index += 1

    return projected


def repl_protocol_prompt(tool_mode):
    if tool_mode != "repl_execute":
        return ""
    return """Every response must include a repl_execute tool call.
The code argument is executed in a persistent Python REPL; variables, imports, connections, and tool state persist.
Code Agent functions such as emit, read, view, bash, and edit are Python functions available inside submitted code, not separate native tools.
Use valid Python in the repl_execute code argument.
Accompanying assistant prose is allowed, but use emit(...) in executed Python for prose communication whenever possible.
Use emit(..., release=True) only when work is complete or user input is required; otherwise continue working without releasing control.
Do not return a response without a repl_execute tool call.
Do not wrap Python in markdown fences or invent native tools other than repl_execute."""


def repl_retry_hint(tool_mode, error=None):
    if tool_mode != "repl_execute":
        return ""
    detail = f"\nError: {error}" if error else ""
    return (
        "Every response must include a repl_execute tool call containing valid Python. "
        "Use emit(...) in executed Python for prose communication, and use "
        "emit(..., release=True) only when work is complete or user input is required. "
        "Do not use any other native tool and do not wrap the code in markdown."
        + detail
    )
