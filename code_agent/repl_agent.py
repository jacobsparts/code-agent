"""
REPL runtime for Code Agent.

The model writes Python source directly. Code Agent executes that code in a
persistent subprocess REPL. Tools are ordinary Python functions injected into
that REPL, with host-side implementations reached through request/response
queues.

The provider does not receive JSON tool schemas for these functions.
"""

from __future__ import annotations

import ast
import contextlib
import inspect

import json
import logging
import os
import signal
import sys
import tempfile
import warnings
from multiprocessing import Queue

from queue import Empty
from typing import Any, Callable, Optional, Union

logger = logging.getLogger('code_agent')


@contextlib.contextmanager
def _silence_compile_warnings():
    """Suppress warnings emitted by parent-side parsing and validation."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        warnings.simplefilter('ignore', DeprecationWarning)
        yield


def _compile_silently(code: str, filename: str = '<repl>', mode: str = 'exec'):
    """Compile code without leaking warnings from internal validation."""
    with _silence_compile_warnings():
        return compile(code, filename, mode)


def _parse_silently(code: str, filename: str = '<unknown>') -> ast.AST:
    """Parse code without leaking warnings from internal transformation."""
    with _silence_compile_warnings():
        return ast.parse(code, filename=filename)


def _join_native_repl_code(*parts: str) -> str:
    return "\n".join(part.rstrip("\n") for part in parts if part)


def _strip_native_repl_prefix(code: str, prefix: str) -> str:
    if not prefix:
        return code
    if code == prefix:
        return ""
    prefix_with_newline = prefix.rstrip("\n") + "\n"
    if code.startswith(prefix_with_newline):
        return code[len(prefix_with_newline):]
    return code


_EVENT_PREFIX = "[[CODE_AGENT_EVENT:"
_EVENT_SUFFIX = "]]"


def _show_events_enabled() -> bool:
    value = os.getenv("SHOW_EVENTS", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _emit_event(event_type: str, **payload) -> None:
    if not _show_events_enabled():
        return
    event = {"type": event_type, **payload}
    print(f"{_EVENT_PREFIX}{json.dumps(event, separators=(',', ':'))}{_EVENT_SUFFIX}", file=sys.stderr, flush=True)

from code_agent.base_agent import BaseAgent, _CompleteException
from code_agent.client import BadRequestError, ContextOverflowError
from code_agent.repl_tool_adapter import ReplExecuteResponseError, repl_protocol_prompt, repl_retry_hint
from code_agent.repl_events import (
    ReplEvent,
    direct_call_name,
    events_output_text,
    normalize_worker_message,
)


class _InterruptedError(KeyboardInterrupt):
    """Raised when user interrupts execution with Ctrl+C."""
    def __init__(self, output: str = ""):
        self.output = output
        super().__init__("Interrupted by user")


from code_agent.tools.subrepl import (
    MultiprocessingTransport,
    SubREPL,
    WORKER_RESTART_NOTICE,
    _format_echo,
    _format_echo_stdout,
    _noninteractive_stdin,
    _split_into_statements,
)


from code_agent.tools.source_extract import extract_method_source as _extract_tool_source


def _is_optional_annotation(annotation: Any) -> bool:
    """Return True for Optional[T] / Union[T, None] / T | None."""
    if annotation in (inspect.Parameter.empty, None):
        return False

    if isinstance(annotation, str):
        text = annotation.replace(" ", "")
        return (
            text.startswith("Optional[")
            or "None" in text.split("|")
            or "NoneType" in text
        )

    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if origin is Union and type(None) in args:
        return True

    try:
        import types as _types
        if origin is _types.UnionType and type(None) in args:
            return True
    except Exception:
        pass

    return type(None) in args




# =============================================================================
# Shared REPL builtins code
# =============================================================================


BUILTINS_CODE = '''
# Request ID counter for matching requests with responses
_request_id = 0

# Tagged print for top-level REPL output (normal print stays available for functions)
def _print(*args, **kwargs):
    """Print with output tagging for display control."""
    import io
    # If printing to a file, use original print
    if kwargs.get('file') is not None:
        return print(*args, **kwargs)
    # Capture output and send tagged
    buf = io.StringIO()
    print(*args, **{**kwargs, 'file': buf})
    _send_output("print", buf.getvalue())

def emit(value, release=False):
    """Emit a value to the output.

    Args:
        value: The value to emit.
        release: If True, release control to the user (for questions or completion).
                 If False (default), the value is emitted but execution continues.
    """
    global _request_id
    _request_id += 1
    req_id = _request_id

    # Use different message types: "emit" for release=True (shown at turn end),
    # "progress" for release=False (shown inline immediately)
    msg_type = "emit" if release else "progress"
    _send_output(msg_type, str(value) + "\\n")
    import json as _json
    _send_tool_request(_json.dumps({
        "tool": "__emit__",
        "args": {"value": value, "release": release},
        "request_id": req_id
    }))
    _wait_for_ack(req_id)

def _wait_for_ack(expected_id):
    """Wait for ACK with matching request_id, capturing reply data."""
    import json as _json
    _result = None
    _error = None
    while True:
        raw = _recv_tool_response()
        # Parse JSON if string
        if isinstance(raw, str):
            try:
                msg = _json.loads(raw)
            except (ValueError, _json.JSONDecodeError):
                continue  # Not JSON, discard
        elif isinstance(raw, dict):
            msg = raw
        else:
            continue  # Unknown format, discard

        # Check for matching request_id
        if msg.get("request_id") != expected_id:
            continue  # Wrong ID, discard stale message

        msg_type = msg.get("type")
        if msg_type == "reply":
            # Capture result/error from reply
            _result = msg.get("result")
            _error = msg.get("error")
        elif msg_type == "ack":
            # ACK received - return result or raise error
            if _error is not None:
                raise Exception(_error)
            return _result
'''

# Queue-based transport for ToolREPL
QUEUE_TRANSPORT_CODE = '''
def _send_output(msg_type, data):
    _statement_output.send(msg_type, data)
def _send_tool_request(msg):
    _tool_request_queue.put(msg)
def _recv_tool_response():
    return _tool_response_queue.get()
'''



_MAX_REPL_OUTPUT_CHARS = 2_000_000


class _StatementOutput:
    def __init__(self, queue: Queue) -> None:
        self.queue = queue
        self.chunks: list[tuple[str, str]] = []
        self.size = 0
        self.file = None
        self._display_buf = ""
        self._display_kind = None

    def send(self, kind: str, text: str) -> int:
        if self.file is None and self.size + len(text) > _MAX_REPL_OUTPUT_CHARS:
            self.file = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix="code-agent-output-", suffix=".txt", delete=False
            )
            for _, chunk in self.chunks:
                self.file.write(chunk)
            self.chunks.clear()
        if self.file:
            self.file.write(text)
        else:
            self.chunks.append((kind, text))
        self.size += len(text)
        if self._display_kind is None:
            self._display_kind = kind
        elif self._display_kind != kind:
            if self._display_buf:
                self.queue.put(("display", (self._display_kind, self._display_buf)))
                self._display_buf = ""
            self._display_kind = kind
        combined = self._display_buf + text
        idx = combined.rfind("\n")
        if idx != -1:
            complete = combined[: idx + 1]
            remainder = combined[idx + 1 :]
            if complete:
                self.queue.put(("display", (kind, complete)))
            self._display_buf = remainder
        else:
            self._display_buf = combined
        return len(text)

    def write(self, text: str) -> int:
        return self.send("output", text)

    def flush(self) -> None:
        pass

    def finish(self) -> None:
        if self._display_buf:
            self.queue.put(("display", (self._display_kind, self._display_buf)))
            self._display_buf = ""
            self._display_kind = None
        if self.file:
            path = self.file.name
            self.file.close()
            spill = f"[large output written to {path} ({self.size / 1_000_000:.1f}MB)]\n"
            self.queue.put(("output", spill))
            self.queue.put(("display", ("spill", spill)))
        else:
            for chunk in self.chunks:
                self.queue.put(chunk)


class _OutputWriter:
    def __init__(self, output: _StatementOutput, original: Any) -> None:
        self.output = output
        self.original = original

    def write(self, text: str) -> int:
        return self.output.write(text)

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        return self.original.fileno()


# ---------------------------------------------------------------------------
# Tool-aware worker process
# ---------------------------------------------------------------------------

def _tool_worker_main(
    cmd_queue: Queue,
    output_queue: Queue,
    tool_request_queue: Queue,
    tool_response_queue: Queue,
    cwd: Optional[str] = None,
) -> None:
    """
    Worker process that can make tool calls back to main process.

    Tool stubs in repl_locals use the request/response queues to
    execute tools in the main process where the actual implementations live.
    """
    if cwd is not None:
        os.chdir(cwd)
    current_cwd = os.getcwd()
    if current_cwd not in sys.path:
        sys.path.insert(0, current_cwd)

    repl_locals: dict[str, Any] = {
        '_output_queue': output_queue,
        '_tool_request_queue': tool_request_queue,
        '_tool_response_queue': tool_response_queue,
    }
    command_active = False
    interrupt_in_progress = False

    def sigint_handler(signum: int, frame: Any) -> None:
        nonlocal interrupt_in_progress
        if command_active and not interrupt_in_progress:
            interrupt_in_progress = True
            raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, sigint_handler)

    while True:
        try:
            msg = cmd_queue.get()

            if msg is None:
                break

            # Commands can be (seq_id, cmd) tuples or plain strings (legacy/injection)
            if isinstance(msg, tuple):
                seq_id, cmd = msg
            else:
                seq_id, cmd = None, msg

            command_active = True
            interrupt_in_progress = False
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            statement_output = _StatementOutput(output_queue)
            repl_locals["_statement_output"] = statement_output
            sys.stdout = _OutputWriter(statement_output, old_stdout)
            sys.stderr = _OutputWriter(statement_output, old_stderr)

            had_error = False
            was_interrupted = False
            try:
                try:
                    with _noninteractive_stdin():
                        # Parse and execute each node, displaying expression results
                        tree = ast.parse(cmd, "<repl>", "exec")
                        for node in tree.body:
                            if isinstance(node, ast.Expr):
                                # Expression statement - eval and display result
                                code_obj = compile(ast.Expression(node.value), "<repl>", "eval")
                                result = eval(code_obj, repl_locals)
                                if result is not None:
                                    statement_output.send("output", repr(result) + "\n")
                            else:
                                # Other statement - just exec
                                mod = ast.Module(body=[node], type_ignores=[])
                                code_obj = compile(mod, "<repl>", "exec")
                                exec(code_obj, repl_locals)
                except SyntaxError as e:
                    had_error = True
                    sys.stderr.write(f"  File \"<repl>\", line {e.lineno}\n")
                    if e.text:
                        sys.stderr.write(f"    {e.text}")
                        if e.offset:
                            sys.stderr.write(" " * (e.offset + 3) + "^\n")
                    sys.stderr.write(f"SyntaxError: {e.msg}\n")
            except KeyboardInterrupt:
                had_error = True
                was_interrupted = True
                sys.stderr.write("\nKeyboardInterrupt\n")
            except Exception as e:
                had_error = True
                import traceback
                tb = e.__traceback__
                while tb is not None and tb.tb_frame.f_code.co_filename != "<repl>":
                    tb = tb.tb_next
                if tb is not None:
                    sys.stderr.write("Traceback (most recent call last):\n")
                    sys.stderr.write("".join(traceback.format_tb(tb)))
                sys.stderr.write(f"{type(e).__name__}: {e}\n")
            finally:
                interrupt_in_progress = True
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                statement_output.finish()

            output_queue.put(("done", (seq_id, had_error, was_interrupted)))
            command_active = False
            interrupt_in_progress = False

        except KeyboardInterrupt:
            continue


# ---------------------------------------------------------------------------
# ToolREPL: SubREPL with tool call bridging
# ---------------------------------------------------------------------------

class ToolREPL(SubREPL):
    """
    SubREPL extended with bidirectional tool call support.

    Tools are injected as Python functions that call back to the main
    process for execution. This allows the LLM to use tools by writing
    normal Python code instead of structured tool calls.
    """

    def __init__(self, echo: bool = True, transport: Optional[type] = None) -> None:
        super().__init__(echo=echo)
        self._transport_class = transport or MultiprocessingTransport
        self._tool_request_queue: Optional[Queue] = None
        self._tool_response_queue: Optional[Queue] = None
        self._tools_injected: bool = False
        self._cmd_seq: int = 0  # Command sequence number for detecting stale messages
    def _ensure_session(self) -> None:
        """Override to use tool-aware worker and add tool queues."""
        if self._transport is None or not self._transport.is_alive():
            replacing_dead_worker = self._transport is not None
            cwd = None
            try:
                from code_agent.tools.transports import SSHSubprocessTransport
                is_remote_transport = issubclass(self._transport_class, SSHSubprocessTransport)
            except TypeError:
                is_remote_transport = False
            if not is_remote_transport:
                cwd = os.getcwd()

            self._transport = self._transport_class(
                target=_tool_worker_main,
                args=(cwd,),
                queue_count=4,
            )
            self._transport.start()
            (
                self._cmd_queue,
                self._output_queue,
                self._tool_request_queue,
                self._tool_response_queue,
            ) = self._transport.queues
            self._worker = self._transport.worker
            self._running = False
            self._tools_injected = False
            self._builtins_injected = False
            self._cmd_seq = 0  # Reset sequence on new session
            if replacing_dead_worker:
                self._pending_output += WORKER_RESTART_NOTICE + "\n"


    def inject_tools(self, tools: dict[str, tuple[Callable, Any]]) -> None:
        """
        Inject tool functions into the REPL.

        Tools marked with inject=True have their source extracted and run
        directly in the subprocess. Other tools get relay stubs that call
        back to the host process.

        Args:
            tools: Dict of tool_name -> (implementation, lightweight ToolSpec)
        """
        self._ensure_session()

        if self._tools_injected:
            return

        # Inject common imports needed by tools
        self._inject_code('from pathlib import Path')

        for name, (impl, spec) in tools.items():
            should_inject = impl is not None and getattr(impl, '_tool_inject', False)
            
            if should_inject:
                code = _extract_tool_source(impl, name)
            else:
                # Relay stub - calls back to host process
                code = _generate_tool_stub(name, impl, spec)
            
            self._inject_code(code)

        self._tools_injected = True

    def inject_builtins(self) -> None:
        """Inject built-in functions like emit() and patched print()."""
        self._ensure_session()
        if getattr(self, '_builtins_injected', False):
            return
        self._inject_code(QUEUE_TRANSPORT_CODE)
        self._inject_code(BUILTINS_CODE)
        self._builtins_injected = True

    def _inject_code(self, code: str, timeout: float = 10.0) -> None:
        """Override to use queue directly and handle tool worker."""
        # Note: timeout parameter accepted for interface compatibility but
        # we use a fixed 5.0s poll timeout internally
        self._ensure_session()
        self._running = True
        self._cmd_queue.put(code)  # Plain string, worker uses seq_id=None

        error_output = []
        while True:
            try:
                msg_type, msg_data = self._output_queue.get(timeout=5.0)
                if msg_type == "done":
                    # Injection uses seq_id=None and cannot be user-interrupted.
                    seq_id, had_error, _ = msg_data
                    self._running = False
                    if had_error:
                        detail = "\n".join(error_output) if error_output else "(no output captured)"
                        raise RuntimeError(f"Startup code failed in REPL subprocess:\n{detail}")
                    return
                if msg_type == "output" and msg_data.strip():
                    error_output.append(msg_data.rstrip())
                    logger.debug("[ToolREPL inject] %s", msg_data.rstrip())
            except Empty:
                raise RuntimeError("Timeout injecting code into REPL")

    def poll_tool_request(self, timeout: float = 0.0) -> Optional[dict]:
        """
        Check for pending tool request from REPL.

        Returns:
            Tool request dict {"tool": name, "args": {...}} or None
        """
        if self._tool_request_queue is None:
            return None
        try:
            request_json = self._tool_request_queue.get(timeout=timeout)
            return json.loads(request_json)
        except Empty:
            return None

    def _check_worker_alive(self) -> None:
        """Check if worker is alive, drain queues and raise if dead."""
        if self._transport is None or not self._transport.is_alive():
            transport = self._transport
            if transport is not None:
                transport.join(timeout=0)

            # Drain all queues to prevent blocking
            for q in (self._tool_response_queue, self._tool_request_queue, self._output_queue):
                if q is not None:
                    try:
                        while True:
                            q.get_nowait()
                    except Empty:
                        pass
            raise RuntimeError("Worker process died")

    def send_reply(self, request_id: int, result: Any = None, error: str = None) -> None:
        """Send application-level reply with result or error.

        Args:
            request_id: The request ID this reply is for.
            result: The result value (if successful).
            error: The error message (if failed).
        """
        if self._tool_response_queue is None:
            raise RuntimeError("No active session")
        self._check_worker_alive()

        msg = {"type": "reply", "request_id": request_id}
        if error is not None:
            msg["error"] = error
        else:
            try:
                json.dumps(result)  # Test serializability
                msg["result"] = result
            except TypeError:
                msg["result"] = repr(result)
        self._tool_response_queue.put(json.dumps(msg))

    def send_ack(self, request_id: int) -> None:
        """Send transport-level ACK to unblock the sender.

        Args:
            request_id: The request ID this ACK is for.
        """
        if self._tool_response_queue is None:
            raise RuntimeError("No active session")
        self._check_worker_alive()
        self._tool_response_queue.put(json.dumps({"type": "ack", "request_id": request_id}))


# ---------------------------------------------------------------------------
# Relay stub generation (for relay tools and MCP)
# ---------------------------------------------------------------------------

def _extract_stub_signature(name: str, impl: Optional[Callable], spec: Any) -> tuple[str, str, str]:
    """
    Extract signature info for generating relay stubs.
    
    Returns:
        (signature_str, docstring, args_dict) - components for stub template
    """
    param_names: list[str] = []
    doc_parts: list[str] = []
    required_sig_parts: list[str] = []
    optional_sig_parts: list[str] = []

    # If impl uses **kwargs and a spec is provided, prefer the spec for
    # signature generation — the spec has the real parameter names/types
    # while **kwargs would produce a broken single-arg stub.
    if impl is not None and spec is not None:
        sig = inspect.signature(impl)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if has_var_keyword:
            impl = None  # fall through to spec-based extraction below

    if impl is not None:
        # Extract signature from implementation
        sig = inspect.signature(impl)

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            param_names.append(param_name)

            if param.default != inspect.Parameter.empty:
                if isinstance(param.default, str):
                    # In Code Agent, string defaults are descriptions, not actual defaults
                    doc_parts.append(f"        {param_name}: {param.default}")
                    if _is_optional_annotation(param.annotation):
                        optional_sig_parts.append(f"{param_name}=None")
                    else:
                        required_sig_parts.append(param_name)
                else:
                    # Actual default value
                    doc_parts.append(f"        {param_name}: (default: {repr(param.default)})")
                    optional_sig_parts.append(f"{param_name}={repr(param.default)}")
            else:
                doc_parts.append(f"        {param_name}")
                required_sig_parts.append(param_name)

        signature_parts = required_sig_parts + optional_sig_parts
        docstring = (impl.__doc__ or f"Call the {name} tool.").strip()

    else:
        # Dynamic tool - extract from lightweight ToolSpec
        signature_parts = []
        required_params = []
        optional_params = []

        for param in getattr(spec, "params", []):
            sanitized = param.name
            original = param.original_name or param.name
            desc = param.description or ''
            doc_parts.append(f"        {sanitized}: {desc}")

            if param.required:
                required_params.append((original, sanitized))
            else:
                default = repr(param.default) if param.default is not inspect.Parameter.empty else 'None'
                optional_params.append((original, sanitized, default))

        for orig, sanitized in required_params:
            param_names.append((orig, sanitized))
            signature_parts.append(sanitized)
        for orig, sanitized, default in optional_params:
            param_names.append((orig, sanitized))
            signature_parts.append(f"{sanitized}={default}")

        docstring = (getattr(spec, "description", None) or f"Call the {name} tool.").strip()

    signature_str = ", ".join(signature_parts)

    if doc_parts:
        docstring += "\n\n    Args:\n" + "\n".join(doc_parts)

    # Build args dict - handle both tuple and string param_names
    if param_names and isinstance(param_names[0], tuple):
        args_items = ", ".join(f'"{orig}": {sanitized}' for orig, sanitized in param_names)
    else:
        args_items = ", ".join(f'"{n}": {n}' for n in param_names)
    args_dict = f"{{{args_items}}}"

    return signature_str, docstring, args_dict


def _generate_tool_stub(name: str, impl: Optional[Callable], spec: Any) -> str:
    """Generate relay stub using queue transport (for ToolREPL)."""
    sig, doc, args = _extract_stub_signature(name, impl, spec)

    # Check for files_param marker - adds path-to-bytes conversion
    files_param = getattr(impl, '_tool_files_param', None) if impl else None
    preprocess = ""
    if files_param:
        preprocess = f'''
    # Convert file paths to bytes
    from pathlib import Path as _Path
    def _read_file(f):
        if isinstance(f, bytes):
            return f
        p = _Path(f).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {{f}}")
        return p.read_bytes()
    if isinstance({files_param}, list):
        {files_param} = [_read_file(f) for f in {files_param}]
    elif isinstance({files_param}, str):
        {files_param} = [_read_file({files_param})]
'''

    return f'''
def {name}({sig}):
    """{doc}"""
    import json as _json
    {preprocess}
    def _serialize(x):
        if isinstance(x, bytes):
            import base64
            return {{"__b64__": base64.b64encode(x).decode()}}
        if isinstance(x, (list, tuple)):
            return [_serialize(i) for i in x]
        if isinstance(x, dict):
            return {{k: _serialize(v) for k, v in x.items()}}
        return x

    _args = {args}
    if "{name}" == "observe":
        _args["content"] = str(_args["content"])
    _safe_args = {{k: _serialize(v) for k, v in _args.items()}}

    _tool_request_queue.put(_json.dumps({{"tool": "{name}", "args": _safe_args}}))
    while True:
        _response = _json.loads(_tool_response_queue.get())
        if _response.get("type") != "ack":
            break
    if "error" in _response:
        raise Exception(_response["error"])
    return _response.get("result")
'''




def _type_to_str(type_hint: Any) -> str:
    """Convert type annotation to string for stub."""
    if type_hint is str:
        return "str"
    elif type_hint is int:
        return "int"
    elif type_hint is float:
        return "float"
    elif type_hint is bool:
        return "bool"
    elif type_hint is list:
        return "list"
    elif type_hint is dict:
        return "dict"
    elif hasattr(type_hint, "__origin__"):
        return str(type_hint)
    else:
        return ""


# ---------------------------------------------------------------------------
# REPLMixin: The main agent mixin
# ---------------------------------------------------------------------------

class REPLMixin:
    """
    Runtime mixin for direct REPL code execution.

    The LLM writes Python code which is executed in a persistent REPL.
    Tools defined with @BaseAgent.tool become callable functions in
    the REPL that bridge back to the main process for execution.

    The model's response is passed directly to the REPL - no markdown
    code blocks, no extraction. The response IS the code.

    Use emit(value, release=False) to output values. Only emit(..., release=True)
    releases control to the user.
    """

    advertise_emit: bool = True

    repl_transport = MultiprocessingTransport

    direct_repl_protocol_prompt = """You are in a Python REPL. Your response body is executed directly as Python source code.

Respond with raw Python only.
- Do not wrap your code in markdown fences
- Do not wrap it in <function_calls>, XML, JSON, or any tool-call protocol
- Do not add prose before or after the code
- Do not describe what you are about to do

Important:
- There is no separate tool-calling layer for your response
- The text of your response itself is what gets executed
- If you need to communicate text, do it from Python using print(...) or an appropriate provided Python function"""

    def _repl_system_prompt_prefix(self) -> str:
        tool_mode = getattr(self.llm_client, "tool_mode", None)
        native_protocol = repl_protocol_prompt(tool_mode)
        if native_protocol:
            return (
                "You are operating Code Agent through a provider-native execution tool.\n\n"
                + native_protocol
            )
        return self.direct_repl_protocol_prompt

    def _configure_conversation(self, conversation):
        if hasattr(super(), "_configure_conversation"):
            super()._configure_conversation(conversation)
        conversation.system_prompt_prefix_provider = self._repl_system_prompt_prefix

    def build_output_for_llm(self, events: list[ReplEvent]) -> str:
        """Build the model-visible output from REPL events."""
        return events_output_text(events)

    def _publish_repl_event(self, event: ReplEvent) -> None:
        publisher = getattr(self, "_active_repl_event_publisher", None)
        if publisher is not None:
            publisher(event)

    def _publish_tool_event(self, kind: str, name: str, **data) -> None:
        self._publish_repl_event(ReplEvent(kind=kind, data={"name": name, **data}))

    def usermsg(self, content, **kwargs):
        """
        Add a user message, appending to REPL output if continuing a session.

        When the last message was REPL output (from a previous turn), new user
        messages are appended to maintain the illusion of a continuous REPL
        session. The message appears as if emitted via emit().
        """
        # Check if we should append to last REPL output
        if getattr(self, '_last_was_repl_output', False) and self.conversation.messages:
            last_msg = self.conversation.messages[-1]
            if last_msg.get("role") == "user":
                # Append as output of the previous emit() call
                # Add trailing \n since this simulates print() output from emit()
                prev = last_msg["content"]
                sep = "" if prev.endswith("\n") else "\n"
                last_msg["content"] = prev + sep + content + "\n"
                last_msg["_user_content"] = content

                # Also update _stdout with the appended content
                if '_stdout' in last_msg:
                    prev_stdout = last_msg['_stdout']
                    sep_stdout = "" if prev_stdout.endswith("\n") else "\n"
                    last_msg['_stdout'] = prev_stdout + sep_stdout + content + "\n"

                # If new content has images or audio, append them too
                if 'images' in kwargs:
                    last_msg['images'] = last_msg.get('images', []) + kwargs['images']
                if 'audio' in kwargs:
                    last_msg['audio'] = last_msg.get('audio', []) + kwargs['audio']
                if '_attachments' in kwargs:
                    attachments = last_msg.get('_attachments', {})
                    attachments.update(kwargs['_attachments'])
                    last_msg['_attachments'] = attachments
                if '_attachment_refs' in kwargs:
                    refs = last_msg.get('_attachment_refs', {})
                    refs.update(kwargs['_attachment_refs'])
                    last_msg['_attachment_refs'] = refs

                last_msg.setdefault("_render_segments", []).append(
                    {"type": "input", "content": content}
                )

                if hasattr(self, '_on_append_last_user_message'):
                    self._on_append_last_user_message(last_msg, content, kwargs)

                self._last_was_repl_output = False
                return

        # Fall back to normal message append
        self._last_was_repl_output = False
        result = super().usermsg(content, **kwargs)
        new_msg = self.conversation.messages[-1]
        if new_msg.get("role") == "user":
            if kwargs.get('_user_content') is not None:
                seg = {"type": "input", "content": kwargs['_user_content']}
            else:
                # Prefer _stdout (unfiltered REPL output) over content (which may
                # have emit chunks filtered out for the LLM).
                seg = {"type": "stdout", "content": kwargs.get('_stdout') or content}
            new_msg["_render_segments"] = [seg]
        return result

    def _ensure_setup(self) -> None:
        """Initialize the ToolREPL."""
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        if not hasattr(self, '_tool_repl'):
            self._tool_repl = ToolREPL(echo=True, transport=self.repl_transport)

    def _get_tool_repl(self) -> ToolREPL:
        """Get or create the ToolREPL instance, with tools and startup code injected."""
        self._ensure_setup()
        # Inject builtins first so tools can use _send_output, _original_print, etc.
        self._tool_repl.inject_builtins()
        tools = {
            name: (self._toolimpl.get(name), spec)
            for name, spec in self.toolspecs.items()
        }
        self._tool_repl.inject_tools(tools)
        
        # Inject startup code if defined
        if not getattr(self, '_repl_startup_injected', False):
            startup = getattr(self, 'repl_startup', None)
            if startup:
                if callable(startup):
                    startup = startup()
                self._tool_repl.inject_startup(startup)
            self._repl_startup_injected = True
        
        return self._tool_repl

    def _cleanup(self) -> None:
        """Clean up REPL resources."""
        if hasattr(self, '_tool_repl'):
            self._tool_repl.close()
        if hasattr(super(), '_cleanup'):
            super()._cleanup()

    def _build_system_prompt(self) -> str:
        """Build REPL-style system prompt."""
        # Ensure mixins are set up (e.g., MCPMixin connects to servers)
        self._ensure_setup()

        # Get base prompt from parent (includes mixin additions)
        if hasattr(super(), '_build_system_prompt'):
            base_prompt = super()._build_system_prompt()
        else:
            base_prompt = getattr(self, 'system', '') or ''

        # List available tools (static + dynamic)
        tool_list = []
        for name, spec in self.toolspecs.items():
            impl = self._toolimpl.get(name)
            param_strs = []
            # If impl uses **kwargs and a spec exists, prefer spec for params
            use_impl = impl is not None
            if use_impl and spec is not None:
                sig_check = inspect.signature(impl)
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_check.parameters.values()):
                    use_impl = False
            if use_impl:
                # Static tool - get params from signature
                sig = inspect.signature(impl)
                for pname, param in sig.parameters.items():
                    if pname == 'self' or param.kind == inspect.Parameter.VAR_KEYWORD:
                        continue
                    type_str = _type_to_str(param.annotation) if param.annotation != inspect.Parameter.empty else ''
                    param_str = f"{pname}: {type_str}" if type_str else pname
                    if param.default != inspect.Parameter.empty:
                        if isinstance(param.default, str):
                            if _is_optional_annotation(param.annotation):
                                param_str += " = None"
                        else:
                            param_str += f" = {repr(param.default)}"
                    param_strs.append(param_str)
                doc = (impl.__doc__ or '').strip()
                first_line = doc.split('\n')[0] if doc else ''
            else:
                # Dynamic tool or **kwargs with spec - get params from lightweight ToolSpec
                for param in getattr(spec, "params", []):
                    type_str = _type_to_str(param.annotation) if param.annotation else ''
                    param_str = f"{param.name}: {type_str}" if type_str else param.name
                    if not param.required:
                        if param.default is not inspect.Parameter.empty and param.default is not None:
                            param_str += f" = {repr(param.default)}"
                        else:
                            param_str += " = None"
                    param_strs.append(param_str)
                doc = (getattr(spec, "description", "") or "").strip()
                first_line = doc.split('\n')[0] if doc else ''

            # Format: name(param: type, ...) - description
            params_str = ", ".join(param_strs)
            tool_entry = f"{name}({params_str})"
            if first_line:
                # Strip leading non-alphanumeric (bullets, dashes, etc.)
                desc = first_line.lstrip()
                while desc and not desc[0].isalnum():
                    desc = desc[1:]
                if desc:
                    tool_entry += f" - {desc}"
            tool_list.append(tool_entry)

        tools_str = "\n".join(tool_list) if tool_list else "(no tools defined)"

        emit_line = "\nemit(value, release=False) - Emit output. release=True yields control." if self.advertise_emit else ""

        return f"""{base_prompt}

You have full access to Python. These additional functions are available:
{tools_str}{emit_line}

Call help(function_name) for parameter descriptions.
"""

    def _llm_text_call_with_context_recovery(self, messages):
        try:
            return self.llm_client.text_call(messages)
        except ContextOverflowError:
            if not hasattr(self, "_coalesce_context"):
                raise
            self.conversation.messages.append({
                "role": "assistant",
                "content": "emit(None, release=True)",
                "_synthetic": True,
                "_virtual_interaction_boundary": True,
            })
            if hasattr(self, '_on_assistant_message_committed'):
                self._on_assistant_message_committed(self.conversation.messages[-1])
            self._coalesce_context()
            return self.llm_client.text_call(
                self.conversation._messages()
            )



    def run_loop(self, max_turns: int = 50, max_syntax_retries: int = 3) -> Any:
        """
        Main agent loop using REPL-first paradigm.

        The LLM writes Python code, we execute it, feed output back.
        Loop continues until emit(release=True) is called or max_turns reached.

        Pure syntax errors (where no statements executed) are retried without
        polluting the conversation history - only successful attempts are committed.
        """
        self._ensure_setup()

        self.complete = False

        for turn in range(max_turns):
            # Get REPL each turn to handle session restart (re-injects tools if needed)
            repl = self._get_tool_repl()
            messages = self.conversation._messages()

            # Fire execute hook once at start of turn (before any attempts)
            if hasattr(self, 'on_repl_execute'):
                self.on_repl_execute(None)

            # Retry loop for pure syntax errors (nothing executed)
            output = ""
            events = []
            try:
                native_retry = 0
                for syntax_retry in range(max_syntax_retries):
                    if hasattr(self, '_start_assistant_execution_attempt'):
                        self._start_assistant_execution_attempt()
                    try:
                        try:
                            resp = self._llm_text_call_with_context_recovery(messages)
                        except ReplExecuteResponseError as exc:
                            native_retry += 1
                            if native_retry >= max_syntax_retries:
                                raise
                            logger.debug("Invalid repl_execute response, retry #%s: %s", native_retry, exc)
                            if hasattr(self, 'on_retry'):
                                self.on_retry("native_repl", native_retry)
                            _emit_event("native_repl_retry", attempt=native_retry)
                            messages = self.conversation._messages() + [{
                                    "role": "user",
                                    "content": repl_retry_hint(
                                        getattr(self.llm_client, "tool_mode", None),
                                        exc,
                                    ),
                                }]
                            continue
                    except KeyboardInterrupt:
                        # User interrupted LLM call - subprocess may also have received SIGINT
                        # Wait briefly for subprocess to handle signal, then drain
                        if hasattr(repl, '_output_queue'):
                            while True:
                                try:
                                    msg_type, _ = repl._output_queue.get(timeout=0.1)
                                    if msg_type == "done":
                                        break
                                except Empty:
                                    break
                        raise _InterruptedError("")
                    except BadRequestError:
                        raise

                    content = (resp.get('content') or '').strip()
                    resp['content'] = content
                    if not content:
                        break

                    native_calls = resp.get("_repl_execute_calls")
                    if native_calls:
                        prefix = resp.get("_repl_execute_prefix") or ""
                        outputs = []
                        events = []
                        corrected_calls = []
                        pure_syntax_error = False
                        for call_index, call in enumerate(native_calls):
                            call_code = call.get("code") or ""
                            execution_code = _join_native_repl_code(
                                prefix if call_index == 0 else "",
                                call_code,
                            )
                            call_output, call_syntax_error, call_events, corrected_code = (
                                self._execute_with_tool_handling(repl, execution_code)
                            )
                            outputs.append(call_output)
                            events.extend(call_events)
                            corrected_calls.append({
                                "id": call.get("id"),
                                "code": _strip_native_repl_prefix(
                                    corrected_code,
                                    prefix if call_index == 0 else "",
                                ),
                            })
                            pure_syntax_error = pure_syntax_error or call_syntax_error
                        output = "".join(outputs)
                        resp["_tool_call_outputs"] = outputs
                        resp["_repl_execute_calls"] = corrected_calls
                        content = _join_native_repl_code(
                            prefix,
                            *(call["code"] for call in corrected_calls),
                        )
                        resp["content"] = content
                    else:
                        output, pure_syntax_error, events, corrected_code = self._execute_with_tool_handling(repl, content)

                        # Apply silent corrections to conversation (both sides see corrected code)
                        if corrected_code != content:
                            resp['content'] = corrected_code
                            content = corrected_code

                    if not pure_syntax_error:
                        break
                    # Pure syntax error - retry with temporary error context
                    logger.debug(f"SyntaxError, retry #{syntax_retry + 1}")
                    if hasattr(self, 'on_retry'):
                        self.on_retry("syntax", syntax_retry + 1)
                    _emit_event("syntax_retry", attempt=syntax_retry + 1)
                    native_hint = repl_retry_hint(
                        getattr(self.llm_client, "tool_mode", None),
                        output,
                    )
                    hint = native_hint or (
                        "Your previous response was rejected because the response body itself must be valid Python source code.\n\n"
                        "Write raw Python only.\n"
                        "- Do not wrap it in <function_calls>, XML, JSON, markdown fences, or any tool-call protocol\n"
                        "- Do not describe what you are going to do\n"
                        "- Do not return prose before or after the code\n\n"
                        "Important:\n"
                        "- Your response is executed directly as Python\n"
                        "- It is not submitted through a separate tool-calling layer\n"
                        "- So respond with the Python statements themselves, exactly as they should be executed\n\n"
                        "If you need to communicate text, do it from Python, for example with print(...) or the appropriate provided Python function.\n\n"
                        "Return only valid Python code."
                    )
                    messages = self.conversation._messages() + [
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": f"{output}\n{hint}"}
                        ]
                else:
                    # All retries exhausted
                    print(f"\n*** CRASH: Model failed to produce valid Python after {max_syntax_retries} retries ***", file=sys.stderr)
                    raise SyntaxError(
                        f"Your response must be valid Python code without preamble or markdown.\n\n{output}"
                    )
            except _InterruptedError:
                # Interrupted output is discarded - don't pollute conversation
                raise
            except _CompleteException:
                # Record the terminal assistant message before returning
                if getattr(self, "complete", False) and getattr(self, "_final_result", None) is not None:
                    resp["_final_result"] = self._final_result
                self.conversation.messages.append(resp)
                if hasattr(self, '_on_assistant_message_committed'):
                    self._on_assistant_message_committed(resp)
                if hasattr(self, '_complete_value'):
                    value = self._complete_value
                    del self._complete_value
                    return value
                return self._final_result

            # Fire output hook after successful execution
            if hasattr(self, "on_repl_events_complete"):
                self.on_repl_events_complete(events)

            # Commit successful response to conversation
            if getattr(self, "complete", False) and getattr(self, "_final_result", None) is not None:
                resp["_final_result"] = self._final_result
            self.conversation.messages.append(resp)
            if hasattr(self, '_on_assistant_message_committed'):
                self._on_assistant_message_committed(resp)

            if not content:
                continue

            # Feed output back to LLM as the REPL response.
            output_for_llm = self.build_output_for_llm(events)
            if hasattr(self, 'process_output_for_llm'):
                output_for_llm = self.process_output_for_llm(output_for_llm)
            elif hasattr(self, 'process_repl_output'):
                output_for_llm = self.process_repl_output(output_for_llm)

            # Store full output as _stdout when it differs from filtered content
            kwargs = {"_repl_output": True}
            if output_for_llm != output:
                kwargs['_stdout'] = output

            if output_for_llm.strip():
                self._last_was_repl_output = False  # Clear before usermsg check
                self.usermsg(output_for_llm, **kwargs)
            else:
                self._last_was_repl_output = False
                kwargs.pop("_repl_output", None)
                self.usermsg("# [no output]", **kwargs)

            if self.complete:
                # Mark that last message is REPL output - next user message appends
                self._last_was_repl_output = True
                if hasattr(self, '_complete_value'):
                    value = self._complete_value
                    del self._complete_value
                    return value
                return self._final_result

            if any(
                event.kind == "statement_finished" and event.data.get("interrupted")
                for event in events
            ):
                raise _InterruptedError(output)

        raise Exception(f"{type(self).__name__} did not complete within {max_turns} turns")

    def _format_syntax_error(self, e: SyntaxError) -> str:
        """Format a SyntaxError like Python's REPL does."""
        lines = []
        lines.append(f'  File "<repl>", line {e.lineno}')
        if e.text:
            lines.append(f'    {e.text.rstrip()}')
            if e.offset:
                # Python 3.10+ has end_offset for better caret positioning
                end_offset = getattr(e, 'end_offset', None)
                if end_offset and end_offset > e.offset:
                    lines.append('    ' + ' ' * (e.offset - 1) + '^' * (end_offset - e.offset))
                else:
                    lines.append('    ' + ' ' * (e.offset - 1) + '^')
        lines.append(f'SyntaxError: {e.msg}')
        return '\n'.join(lines) + '\n'

    def _transform_toplevel_print(self, code: str) -> str:
        """Transform top-level print() calls to _print() for output tagging.

        Only transforms print calls at module level, not inside function/class definitions.
        This allows tools and user-defined functions to use normal print().
        """
        import ast

        try:
            tree = _parse_silently(code)
        except SyntaxError:
            return code  # Let the REPL handle syntax errors

        class PrintTransformer(ast.NodeTransformer):
            def __init__(self):
                self.in_function = False
                self.changed = False

            def visit_FunctionDef(self, node):
                # Don't transform inside function bodies
                old = self.in_function
                self.in_function = True
                self.generic_visit(node)
                self.in_function = old
                return node

            def visit_AsyncFunctionDef(self, node):
                return self.visit_FunctionDef(node)

            def visit_ClassDef(self, node):
                # Don't transform inside class bodies
                old = self.in_function
                self.in_function = True
                self.generic_visit(node)
                self.in_function = old
                return node

            def visit_Call(self, node):
                self.generic_visit(node)
                # Transform print() to _print() at top level only
                if not self.in_function:
                    if isinstance(node.func, ast.Name) and node.func.id == 'print':
                        node.func.id = '_print'
                        self.changed = True
                return node

        transformer = PrintTransformer()
        new_tree = transformer.visit(tree)
        if not transformer.changed:
            return code
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)

    def preprocess_code(self, code: str) -> str:
        """Preprocess LLM-generated code before execution.

        Applies generic fixes for common model mistakes. Subclasses can
        override to add additional preprocessing (call super() to chain).
        """
        from code_agent.preprocess import preprocess
        return preprocess(code)

    def _execute_with_tool_handling(self, repl: ToolREPL, code: str) -> tuple[str, bool, list[ReplEvent], str]:
        """Execute code statement-by-statement while publishing ordered REPL events."""
        code = self.preprocess_code(code)
        validation_code = self._transform_toplevel_print(code)
        try:
            _compile_silently(validation_code)
        except SyntaxError as e:
            output = self._format_syntax_error(e)
            events = [ReplEvent(kind="error", text=output)]
            if hasattr(self, "on_repl_event"):
                self.on_repl_event(events[0])
            return output, True, events, code

        from code_agent.execution_policy import ExecutionPolicyError, check_execution_policy
        try:
            with _silence_compile_warnings():
                check_execution_policy(validation_code)
        except ExecutionPolicyError as e:
            output = f"{type(e).__name__}: {e}\n"
            events = [ReplEvent(kind="error", text=output)]
            if hasattr(self, "on_repl_event"):
                self.on_repl_event(events[0])
            return output, False, events, code

        with _silence_compile_warnings():
            original_statements = _split_into_statements(code)
        if not original_statements:
            return "", False, [], code

        repl._ensure_session()
        events: list[ReplEvent] = []
        statement_events: list[ReplEvent] = []
        any_executed = False

        def publish(event: ReplEvent) -> None:
            events.append(event)
            statement_events.append(event)
            if hasattr(self, "on_repl_event"):
                self.on_repl_event(event)

        def publish_worker(message_type: str, text: str) -> None:
            event = normalize_worker_message(message_type, text)
            if event.kind == "display":
                if hasattr(self, "on_repl_event"):
                    self.on_repl_event(event)
                return
            publish(event)

        previous_publisher = getattr(self, "_active_repl_event_publisher", None)
        self._active_repl_event_publisher = publish
        try:
            if repl._pending_output:
                publish(ReplEvent(kind="output", text=repl._pending_output))
                repl._pending_output = ""
            for original_stmt in original_statements:
                exec_stmt = self._transform_toplevel_print(original_stmt)
                with _silence_compile_warnings():
                    echo = _format_echo(original_stmt)
                    display_echo = _format_echo_stdout(original_stmt)
                publish(ReplEvent(
                    kind="statement_started",
                    data={
                        "source": original_stmt,
                        "echo": echo,
                        "display_echo": display_echo,
                        "direct_call": direct_call_name(original_stmt),
                    },
                ))

                try:
                    _compile_silently(exec_stmt)
                except SyntaxError as e:
                    publish(ReplEvent(kind="error", text=self._format_syntax_error(e)))
                    publish(ReplEvent(kind="statement_finished", data={"had_error": True}))
                    if hasattr(self, "on_statement_events"):
                        self.on_statement_events(list(statement_events))
                    statement_events.clear()
                    break

                any_executed = True
                repl._running = True
                repl._cmd_seq += 1
                current_seq = repl._cmd_seq
                repl._cmd_queue.put((current_seq, exec_stmt))
                statement_had_error = False
                statement_was_interrupted = False
                completed_early = False

                while True:
                    try:
                        tool_req = repl.poll_tool_request(timeout=0)
                        if tool_req:
                            self._handle_tool_request(repl, tool_req)
                            if self.complete:
                                while True:
                                    try:
                                        msg_type, msg_data = repl._output_queue.get(timeout=0.1)
                                        if msg_type == "done":
                                            seq_id, had_error, was_interrupted = msg_data
                                            if seq_id == current_seq:
                                                statement_had_error = had_error
                                                statement_was_interrupted = was_interrupted
                                                break
                                            continue
                                        publish_worker(msg_type, msg_data)
                                    except Empty:
                                        break
                                repl._running = False
                                completed_early = True
                                break

                        try:
                            msg_type, msg_data = repl._output_queue.get(timeout=0.05)
                            if msg_type == "done":
                                seq_id, had_error, was_interrupted = msg_data
                                if seq_id == current_seq:
                                    repl._running = False
                                    statement_had_error = had_error
                                    statement_was_interrupted = was_interrupted
                                    break
                                continue
                            publish_worker(msg_type, msg_data)
                        except Empty:
                            repl._check_worker_alive()
                    except KeyboardInterrupt:
                        if not repl._transport.receives_terminal_sigint:
                            repl._transport.interrupt()

                publish(ReplEvent(
                    kind="statement_finished",
                    data={
                        "had_error": statement_had_error,
                        "interrupted": statement_was_interrupted,
                    },
                ))
                if hasattr(self, "on_statement_events"):
                    self.on_statement_events(list(statement_events))
                statement_events.clear()

                if completed_early:
                    output = events_output_text(events)
                    return output, False, events, code
                if statement_had_error:
                    break
        finally:
            if previous_publisher is None:
                try:
                    del self._active_repl_event_publisher
                except AttributeError:
                    pass
            else:
                self._active_repl_event_publisher = previous_publisher

        output = events_output_text(events)
        is_pure_syntax_error = not any_executed and bool(events)
        return output, is_pure_syntax_error, events, code

    def _handle_tool_request(self, repl: ToolREPL, req: dict) -> None:
        """Handle a tool request from the REPL."""
        tool_name = req.get("tool")
        request_id = req.get("request_id")
        args = req.get("args", {})

        def _deserialize(x):
            if isinstance(x, dict) and "__b64__" in x:
                import base64
                return base64.b64decode(x["__b64__"])
            if isinstance(x, list):
                return [_deserialize(i) for i in x]
            if isinstance(x, dict):
                return {k: _deserialize(v) for k, v in x.items()}
            return x

        args = {k: _deserialize(v) for k, v in args.items()}
        event_name = "emit" if tool_name == "__emit__" else tool_name
        if event_name:
            self._publish_tool_event("tool_called", event_name, args=args)

        try:
            if tool_name == "__emit__":
                value = args.get("value")
                release = args.get("release", False)
                self._final_result = value
                if release:
                    self.complete = True
                self._publish_tool_event("tool_returned", "emit", result=None)
            elif tool_name:
                try:
                    result = self.toolcall(tool_name, args)
                    repl.send_reply(request_id, result=result)
                    self._publish_tool_event("tool_returned", tool_name, result=result)
                except _CompleteException:
                    if request_id is None:
                        repl.send_reply(request_id, result=None)
                    raise
                except Exception as e:
                    repl.send_reply(request_id, error=str(e))
                    self._publish_tool_event("tool_failed", tool_name, error=str(e))
        finally:
            if request_id is not None:
                repl.send_ack(request_id)


class REPLAgent(REPLMixin, BaseAgent):
    """
    Agent that operates in a Python REPL instead of using tool calls.

    Example:
        CodeAgent uses this as its REPL execution base.
    """
    pass
