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


def normalize_openai_repl_response(message):
    content = message.get("content")
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        text = content or ""
        if not text:
            return {"role": "assistant", "content": ""}
        if _python_source(text):
            return {"role": "assistant", "content": text}
        return {"role": "assistant", "content": f"emit({text!r}, release=True)"}

    parts = []
    if content:
        parts.append(f"emit({content!r})")

    for call in tool_calls:
        function = call.get("function")
        if not isinstance(function, dict):
            raise ReplExecuteResponseError("Invalid native tool call: missing function object")
        name = function.get("name")
        if name != "repl_execute":
            raise ReplExecuteResponseError(f"Unexpected native tool: {name!r}")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError) as exc:
                raise ReplExecuteResponseError("Invalid repl_execute arguments JSON") from exc
        if not isinstance(arguments, dict):
            raise ReplExecuteResponseError("repl_execute arguments must be an object")
        code = arguments.get("code")
        if not isinstance(code, str):
            raise ReplExecuteResponseError("repl_execute code must be a string")
        parts.append(code)

    return {
        "role": "assistant",
        "content": _join_python(parts),
        "_tool_call_ids": [call.get("id") for call in tool_calls],
    }


def _public_message(message):
    return {k: v for k, v in message.items() if not k.startswith("_") and k != "tool_calls"}


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
    if segments:
        stdout = "".join(
            segment.get("content", "")
            for segment in segments
            if segment.get("type") != "input"
        )
        inputs = [
            segment.get("content", "")
            for segment in segments
            if segment.get("type") == "input" and segment.get("content")
        ]
        if stdout or inputs:
            return _strip_leading_input_echo(stdout, preceding_human_input), inputs

    human = message.get("_user_content")
    if human is not None:
        stdout = message.get("_stdout", message.get("content", ""))
        suffix = human + "\n"
        if stdout.endswith(suffix):
            stdout = stdout[:-len(suffix)].rstrip("\n") + ("\n" if stdout[:-len(suffix)] else "")
        return _strip_leading_input_echo(stdout, preceding_human_input), [human]
    stdout = message.get("_stdout", message.get("content", ""))
    return _strip_leading_input_echo(stdout, preceding_human_input), []


def project_openai_repl_messages(messages):
    projected = []
    call_number = 0
    index = 0
    pending_human_input = None

    while index < len(messages):
        message = messages[index]
        role = message.get("role")

        if role != "assistant":
            projected.append(_public_message(message))
            if role == "user":
                pending_human_input = message.get("_user_content", message.get("content"))
            index += 1
            continue

        code = message.get("content") or ""
        call_number += 1
        call_id = (message.get("_tool_call_ids") or [None])[0] or f"repl_{call_number:06d}"
        projected.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "repl_execute",
                    "arguments": json.dumps({"code": code}),
                },
            }],
        })

        output = ""
        human_inputs = []
        if index + 1 < len(messages) and messages[index + 1].get("role") == "user":
            output, human_inputs = _split_repl_output_and_input(
                messages[index + 1],
                pending_human_input,
            )
            index += 1
        projected.append({"role": "tool", "tool_call_id": call_id, "content": output})
        projected.extend({"role": "user", "content": value} for value in human_inputs)
        pending_human_input = human_inputs[-1] if human_inputs else None
        index += 1

    return projected


def repl_protocol_prompt(tool_mode):
    if tool_mode != "repl_execute":
        return ""
    return """Use the repl_execute tool for Python execution.
The code argument is executed in a persistent Python REPL; variables, imports, connections, and tool state persist.
Code Agent functions such as emit, read, view, bash, and edit are Python functions available inside submitted code, not separate native tools.
Use ordinary assistant text only for communication accompanying an execution or when no execution is needed.
Use emit(..., release=True) in executed Python when completing through the REPL.
Do not wrap Python in markdown fences or invent native tools other than repl_execute."""


def repl_retry_hint(tool_mode, error=None):
    if tool_mode != "repl_execute":
        return ""
    detail = f"\nError: {error}" if error else ""
    return (
        "Submit corrected valid Python using the repl_execute tool. "
        "Do not use any other native tool and do not wrap the code in markdown."
        + detail
    )
