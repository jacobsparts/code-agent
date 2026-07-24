#!/usr/bin/env python3
"""Code assistant with Python REPL execution and REPL-proxied tools.

Combines REPLAgent with CLIMixin to create an interactive coding assistant
that executes Python code directly. Uses local implementations for file
operations and ripgrep for search.

Project filesystem access belongs in the worker process. Parent-side code
should treat project files as worker-owned resources and receive content/paths
over the REPL queues instead of reading or statting them directly.

Dependencies:
    - ripgrep (rg) enables grep(); when missing in the worker, grep is disabled.
"""

from code_agent.code_agent_preprocess import preprocess_code_agent
import ast
import base64
import hashlib
import json
import os
import sys
import termios
import copy
import re
from typing import Optional
from pathlib import Path
from queue import Empty
from code_agent.repl_agent import REPLAgent
from code_agent.repl_events import ReplEvent
from code_agent.repl_attachment_mixin import REPLAttachmentMixin
from code_agent.mcp_mixin import MCPMixin
from code_agent.repl_attachment_mixin import MemoryAttachment, encode_attachment_refs

from code_agent.cli import CLIMixin
from code_agent.llm_registry import ModelNotFoundError
from code_agent.client import ContextOverflowError
from code_agent.cli.terminal import DIM, RESET, Panel, strip_ansi, TEXT
from code_agent.session_store import SessionStore
from code_agent.session_replay import replay_session_into_agent, replay_display_text
from code_agent.preview_refs import is_preview_uri, preview_key, numbered_content, render_preview_refs

from code_agent.dotenv import load_dotenv

load_dotenv()

#import logging; logging.getLogger('code_agent').setLevel(logging.DEBUG)


def _get_config_value(attr_name, default):
    """Lazy load user config value."""
    from code_agent.config import get_user_config
    config = get_user_config()
    return getattr(config, attr_name, default) if config else default


def _configured_models() -> list[str]:
    value = _get_config_value("code_agent_model", "sonnet")
    if isinstance(value, list):
        return value or ["sonnet"]
    return [value]


def _skill_description(path: Path) -> str:
    try:
        for line in path.read_text().splitlines():
            text = line.strip()
            if not text:
                continue
            if text.startswith('#'):
                text = text.lstrip('#').strip()
            return text
    except Exception:
        pass
    return ""


class CodeAgentBase(REPLAttachmentMixin, CLIMixin, REPLAgent):
    """Code assistant with Python REPL execution."""

    model_choices = _configured_models()
    model = model_choices[0]
    worker_host = "local"
    worker_target = None


    def _ensure_setup(self):
        super()._ensure_setup()
        if not hasattr(self, '_session_store'):
            self._session_store = SessionStore()
            self._session_id = None
            self._next_event_seq = 1
            self._suspend_persistence = False
            self._explicit_attachment_refs = {}
            self._pending_explicit_attachment_refs = {}
            self._pending_session_events = []
            self._display_capture = []
            self._pending_unviewed_files = set()
            self._pending_observations = []
            self._auto_context_attachment_names = set()
            self._expanded_preview_refs = {}
            from code_agent.code_agent_coalesce import PersistedPreviewState
            self._persisted_preview_state = PersistedPreviewState.empty()
            self._rg_available = None
            self._rg_warning_printed = False

        if hasattr(self, "_conversation"):
            self._configure_conversation(self._conversation)

        self.llm_client.on_retry = self.on_retry

    def _set_model(self, new_model: str) -> bool:
        from code_agent.llm_registry import resolve_model_name
        new_model = resolve_model_name(new_model)
        old_model = self.model
        if new_model == old_model:
            return False
        self.model = new_model
        if hasattr(self, "_llm_client"):
            delattr(self, "_llm_client")
        if hasattr(self, "_conversation"):
            self._conversation.llm_client = self.llm_client
        return True

    def _cycle_model(self, direction: int = 1) -> str | None:
        choices = getattr(self, "model_choices", [])
        if len(choices) < 2:
            return None
        try:
            index = choices.index(self.model)
        except ValueError:
            index = -1 if direction > 0 else 0
        new_model = choices[(index + direction) % len(choices)]
        self._set_model(new_model)
        return f"{DIM}Model changed: {self.model}{RESET}"

    def _cycle_model_reverse(self) -> str | None:
        return self._cycle_model(-1)

    def session_host(self) -> str:
        return getattr(self, "worker_host", "local") or "local"

    def resume_session_command(self, session_id: str) -> str:
        worker_target = getattr(self, "worker_target", None)
        target = f" {worker_target}" if worker_target else ""
        return f"coda{target} --resume {session_id}"

    def _lock_status_text(self, lock: dict | None) -> str:
        if not lock:
            return "Session is locked by another process."
        owner = lock.get("owner") or "unknown"
        host = lock.get("hostname") or "unknown host"
        pid = lock.get("pid") or "unknown pid"
        expires = lock.get("expires_at") or "unknown"
        return f"Session is currently open by {owner} (pid {pid} on {host}, expires {expires}). Fork it to work in parallel."

    def _acquire_session_lock(self, session_id: str) -> bool:
        owner = getattr(self, "_session_lock_owner", None)
        if owner is None:
            owner = self._session_store.default_lock_owner()
            self._session_lock_owner = owner
        ok, lock = self._session_store.acquire_session_lock(session_id, owner)
        if not ok:
            print(f"{DIM}{self._lock_status_text(lock)}{RESET}")
            return False
        return True

    def _heartbeat_session_lock(self) -> bool:
        session_id = getattr(self, "_session_id", None)
        owner = getattr(self, "_session_lock_owner", None)
        if not session_id or not owner:
            return True
        return self._session_store.heartbeat_session_lock(session_id, owner)

    def _release_session_lock(self, session_id: str | None = None):
        owner = getattr(self, "_session_lock_owner", None)
        target = session_id or getattr(self, "_session_id", None)
        if target and owner:
            self._session_store.release_session_lock(target, owner)

    def fork_session(self, session_id: str) -> str | None:
        try:
            new_session_id = self._session_store.fork_session(
                session_id,
                cwd=self.worker_cwd(),
                model=getattr(self, "model", None),
                host=self.session_host(),
            )
        except Exception as e:
            print(f"{DIM}Error forking session: {type(e).__name__}: {e}{RESET}")
            return None
        print(f"{DIM}Forked session: {new_session_id}{RESET}")
        return new_session_id
    code_agent_coalesce_min_savings_chars = _get_config_value("code_agent_coalesce_min_savings_chars", 1000)
    code_agent_coalesce_keep_last_execution_interactions = _get_config_value("code_agent_coalesce_keep_last_execution_interactions", 1)

    def _coalesce_context(
        self,
        *,
        protect_last_interactions: bool = True,
    ):


        from code_agent.code_agent_coalesce import coalesce_repl_messages

        if not self._session_id:
            return
        auto_expand_preview_refs = []
        preserve_preview_refs = {}
        for uri in getattr(self, "_expanded_preview_refs", {}):
            content = self._preview_blob_content(uri)
            if content is not None:
                preserve_preview_refs[uri] = self._render_preview_ref(content)[1]
        self.conversation.messages = coalesce_repl_messages(
            self.conversation.messages,
            keep_last_execution_interactions=self.code_agent_coalesce_keep_last_execution_interactions,
            min_savings_chars=self.code_agent_coalesce_min_savings_chars,
            save_preview_blob=lambda key, content: self._session_store.save_preview_blob(self._session_id, key, content),
            auto_expand_preview_refs=auto_expand_preview_refs,
            preserve_preview_refs=preserve_preview_refs,
            protect_last_interactions=protect_last_interactions,
        )
        for uri in auto_expand_preview_refs:
            self._expanded_preview_refs[uri] = {"numbered": False}
            self._append_session_event("preview_expanded", {"uri": uri, "numbered": False}, create_session=False)



    def _resolve_session_selection(self, selection) -> str | None:
        if not selection:
            return None
        if isinstance(selection, str):
            return selection
        action = selection.get("action")
        session_id = selection.get("session_id")
        if not session_id:
            return None
        if action == "fork":
            return self.fork_session(session_id)
        return session_id

    def _ensure_live_session(self):
        if self._session_id is None:
            self._session_id = self._session_store.create_session(self.worker_cwd(), getattr(self, 'model', None), self.session_host())
            self._next_event_seq = 1
            if not self._acquire_session_lock(self._session_id):
                raise RuntimeError("Could not acquire session lock.")
        elif not self._heartbeat_session_lock():
            if not self._acquire_session_lock(self._session_id):
                raise RuntimeError("Session lock was lost.")

    def _flush_pending_session_events(self):
        if not self._session_id:
            return
        pending = self._pending_session_events
        self._pending_session_events = []
        for event_type, payload in pending:
            seq = self._next_event_seq
            self._session_store.append_event(self._session_id, seq, event_type, payload)
            self._next_event_seq += 1

    def _bootstrap_persisted_conversation(self):
        if getattr(self, '_suspend_persistence', False):
            return
        if self._session_id is None:
            self._ensure_live_session()
            self._flush_pending_session_events()
        for msg in self.conversation.messages[1:]:
            if msg.get('_synthetic') and not msg.get("_virtual_interaction_boundary"):
                continue

            if msg.get('_event_seq') is None:
                self._persist_message(msg)

    def _append_session_event(self, event_type: str, payload: dict, create_session: bool = True) -> int | None:
        if getattr(self, '_suspend_persistence', False):
            return None
        if self._session_id is None and not create_session:
            self._pending_session_events.append((event_type, copy.deepcopy(payload)))
            return None
        self._ensure_live_session()
        self._flush_pending_session_events()
        seq = self._next_event_seq
        self._session_store.append_event(self._session_id, seq, event_type, payload)
        self._next_event_seq += 1
        return seq

    def create_persisted_preview(
        self,
        preview,
        *,
        source_start_seq: int,
        source_end_seq: int,
    ) -> tuple[str, int]:
        from code_agent.code_agent_coalesce import create_persisted_preview

        self._ensure_live_session()
        self._flush_pending_session_events()
        state = getattr(self, "_persisted_preview_state", None)
        if state is None:
            from code_agent.code_agent_coalesce import PersistedPreviewState
            state = PersistedPreviewState.empty()
        projected, key, preview_event_seq, placed_event_seq = create_persisted_preview(
            self.conversation.messages,
            preview,
            source_start_seq=source_start_seq,
            source_end_seq=source_end_seq,
            store=self._session_store,
            session_id=self._session_id,
            expected_next_seq=self._next_event_seq,
            state=state,
        )
        self.conversation.messages = projected
        self._persisted_preview_state = state
        self._next_event_seq = placed_event_seq + 1
        return key, preview_event_seq

    def _record_display_event(self, kind: str, text: str, create_session: bool = False):
        if not text:
            return
        self._append_session_event("display", {"kind": kind, "text": text}, create_session=create_session)

    def _display_text(self, text: str, kind: str = "status", end: str = "\n", create_session: bool = False):
        print(text, end=end, flush=True)
        self._record_display_event(kind, strip_ansi(text) + end, create_session=create_session)

    def _section_header(self, label: str, char: str = "═", color: str = "\x1b[1;36m", width: int = 34) -> str:
        prefix = f"{char} {label} "
        return f"{color}{prefix}{char * max(0, width - len(prefix))}{RESET}"

    def _display_input_block(self, text: str, include_header: bool = False):
        lines = text.rstrip("\n").split("\n") if text else [""]
        rendered = []
        if include_header:
            rendered.append(strip_ansi(self._section_header("User")))
        rendered.append(f"{self.cli_prompt}{lines[0]}")
        rendered.extend(lines[1:])
        self._record_display_event("input", "\n".join(rendered) + "\n\n")
    def _replay_display_output(self):
        if not self._session_id:
            return
        display_text = replay_display_text(self._session_id, self._session_store, format_response=self.format_response)
        sys.stdout.write(display_text)
        if display_text and not display_text.endswith("\n"):
            print()

    def _reset_display_capture(self):
        self._display_capture = []

    def _capture_display_line(self, text: str = ""):
        self._display_capture.append(strip_ansi(text) + "\n")

    def _show_python_header_if_pending(self):
        if getattr(self, '_header_pending', False):
            header = self._section_header("Python", "─", "\x1b[1;94m")
            print(f"\x1b[1G\x1b[K{header}")
            self._capture_display_line(strip_ansi(header))
            self._header_pending = False
            self._repl_printed_header = True

    def _flush_statement_echo(self) -> None:
        echo = getattr(self, "_statement_echo", "")
        if not echo or getattr(self, "_statement_echo_displayed", False):
            return
        self._show_python_header_if_pending()
        for line in echo.rstrip("\n").split("\n"):
            print(line, flush=True)
            self._capture_display_line(line)
        self._statement_echo_displayed = True

    def _compact_edit_echo(self, diff: str, tool: str | None = None) -> str:
        func = tool if tool in {"edit", "line_patch"} else "edit"

        filename = None
        for prefix in ("+++ ", "--- "):
            for line in diff.splitlines():
                if line.startswith(prefix):
                    candidate = line[len(prefix):].strip()
                    if candidate != "/dev/null":
                        filename = candidate
                        break
            if filename is not None:
                break

        if filename is None:
            return f">>> {func}(...)"
        return f">>> {func}({filename!r}, ...)"

    @staticmethod
    def _file_diff_paths(diff: str) -> list[str]:
        paths = []
        seen = set()
        for line in diff.splitlines():
            if not (line.startswith("--- ") or line.startswith("+++ ")):
                continue
            path = line[4:].strip().split("\t", 1)[0]
            if path == "/dev/null":
                continue
            if path.startswith("a/") or path.startswith("b/"):
                path = path[2:]
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
        return paths

    def _record_file_diff_event(self, diff: str, tool: str | None = None):
        if not diff:
            return
        if tool is None:
            tool = getattr(self, "_statement_direct_call", None)
        if tool not in {"edit", "line_patch"}:
            tool = None
        self._append_session_event("file_diff", {
            "kind": "unified_diff",
            "tool": tool,
            "paths": self._file_diff_paths(diff),
            "diff": diff,
        }, create_session=False)

    def _matching_file_diff_events(self, file_path: str | None = None, limit: int | None = None) -> list[dict]:
        if not getattr(self, "_session_id", None):
            return []
        events = []
        for event in self._session_store.get_events(self._session_id):
            if event.get("event_type") != "file_diff":
                continue
            payload = event.get("payload") or {}
            if file_path:
                paths = payload.get("paths") or []
                if not any(path == file_path or self._same_file(path, file_path) for path in paths):
                    continue
            events.append(event)
        if limit is not None:
            events = events[-int(limit):]
        return events

    def _format_file_diff_events(self, file_path: str | None = None, limit: int | None = None) -> str:
        events = self._matching_file_diff_events(file_path, limit)
        if not events:
            target = f" for {file_path}" if file_path else ""
            return f"No file diffs recorded{target}."
        chunks = []
        for event in events:
            payload = event.get("payload") or {}
            paths = ", ".join(payload.get("paths") or [])
            tool = payload.get("tool") or "unknown"
            chunks.append(
                f"# file_diff seq={event.get('seq')} tool={tool} paths={paths}\n"
                f"{payload.get('diff', '').rstrip()}\n"
            )
        return "\n".join(chunks).rstrip() + "\n"


    def _flush_display_capture(self):
        if not self._display_capture:
            return
        self._record_display_event("python", "".join(self._display_capture))
        self._display_capture = []

    def _sanitize_message_for_persistence(self, message: dict) -> dict:
        def encode_media(value):
            if isinstance(value, bytes):
                return {"__b64__": base64.b64encode(value).decode("ascii")}
            if isinstance(value, list):
                return [encode_media(item) for item in value]
            return copy.deepcopy(value)

        msg = {}
        for key, value in message.items():
            if key in {'images', 'audio'}:
                msg[key] = encode_media(value)
            elif key in {'role', 'content', '_stdout', '_user_content', 'name', 'tool_call_id', '_synthetic', '_render_segments', '_final_result', '_emit_value', '_pinned_coalesce', '_virtual_interaction_boundary', '_observations'}:
                msg[key] = copy.deepcopy(value)

        refs = message.get('_attachment_refs')
        if refs:
            msg['_attachment_refs'] = encode_attachment_refs(refs)
        return msg

    def _tag_latest_segment_seq(self, message: dict, seq: int):
        for seg in reversed(message.get("_render_segments") or []):
            if "_event_seq" not in seg:
                seg["_event_seq"] = seq
                break

    def _persist_message(self, message: dict):
        if (message.get('_synthetic') and not message.get("_virtual_interaction_boundary")) or message.get("_event_seq") is not None:
            return
        if self._session_id is None:
            self._bootstrap_persisted_conversation()
            if message.get("_event_seq") is not None:
                return
        seq = self._append_session_event("message_added", {"message": self._sanitize_message_for_persistence(message)})
        if seq is not None:
            message["_event_seq"] = seq
            self._tag_latest_segment_seq(message, seq)

    def _persist_append_to_last_user_message(self, target_message: dict, content: str, kwargs: dict):
        user_content = target_message.get('_user_content', content)
        latest_segment = (target_message.get("_render_segments") or [None])[-1]
        new_msg = {"role": "user", "content": content, "_user_content": user_content}
        if kwargs.get('_stdout') is not None:
            new_msg["_stdout"] = kwargs['_stdout']
        if kwargs.get('_attachment_refs'):
            new_msg["_attachment_refs"] = copy.deepcopy(kwargs['_attachment_refs'])
        for key in ('images', 'audio'):
            if kwargs.get(key):
                new_msg[key] = kwargs[key]
        if latest_segment is not None:
            new_msg["_render_segments"] = [{k: v for k, v in latest_segment.items() if k != "_event_seq"}]
        seq = self._append_session_event("message_added", {"message": self._sanitize_message_for_persistence(new_msg)})
        if seq is not None:
            target_message["_event_seq"] = seq
            self._tag_latest_segment_seq(target_message, seq)

    def _on_append_last_user_message(self, target_message: dict, content, kwargs):
        self._persist_append_to_last_user_message(target_message, content, kwargs)

    _worker_attachment_helpers = r"""
import json as _code_agent_json
import os as _code_agent_os
from pathlib import Path as _code_agent_Path

def _code_agent_attach_file_ref(filepath, name=None):
    path = _code_agent_Path(filepath).expanduser()
    content = path.read_text()
    attach_name = name or filepath
    _send_output("attachment_ref", _code_agent_json.dumps({
        "name": attach_name,
        "path": str(path),
        "content": content,
    }) + "\n")

def _code_agent_gather_auto_attach_files():
    current = _code_agent_Path.cwd()
    found_files = []
    seen_paths = set()

    def add_file_and_imports(file_path):
        abs_path = file_path.resolve()
        if abs_path in seen_paths:
            return
        seen_paths.add(abs_path)
        found_files.append(_code_agent_os.path.relpath(abs_path, current))
        try:
            content = file_path.read_text()
            for line in content.split("\n"):
                if line.startswith("@"):
                    import_name = line[1:].strip()
                    if import_name:
                        import_path = (file_path.parent / import_name).resolve()
                        if import_path.exists() and import_path.is_file():
                            add_file_and_imports(import_path)
        except Exception:
            pass

    md_files = []
    search_dir = current
    while True:
        for name in ["CLAUDE.md", "AGENTS.md"]:
            candidate = search_dir / name
            if candidate.exists():
                md_files.append(candidate)
        parent = search_dir.parent
        if parent == search_dir:
            break
        search_dir = parent

    md_files.reverse()
    for md_file in md_files:
        add_file_and_imports(md_file)
    return found_files

def _code_agent_send_auto_context_files():
    _send_output("auto_context_files", _code_agent_json.dumps(_code_agent_gather_auto_attach_files()) + "\n")

def _code_agent_send_worker_cwd():
    _send_output("worker_cwd", str(_code_agent_Path.cwd()) + "\n")

def _code_agent_send_rg_available():
    import shutil as _code_agent_shutil
    _send_output("rg_available", _code_agent_json.dumps(bool(_code_agent_shutil.which("rg"))) + "\n")
"""

    def _ensure_worker_attachment_helpers(self):
        self._ensure_setup()
        repl = self._tool_repl
        repl.inject_builtins()
        if not getattr(self, '_worker_attachment_helpers_injected', False):
            import inspect
            from code_agent.tools.subshell import ensure_python_on_path
            repl._inject_code(inspect.getsource(ensure_python_on_path) + "\n" + self._worker_attachment_helpers)
            self._worker_attachment_helpers_injected = True
        return repl

    def _run_worker_control_code(self, code: str, capture_types: set[str]) -> list[tuple[str, str]]:
        repl = self._ensure_worker_attachment_helpers()
        repl._ensure_session()
        repl._running = True
        repl._cmd_seq += 1
        current_seq = repl._cmd_seq
        repl._cmd_queue.put((current_seq, code))
        chunks = []
        error_output = []

        try:
            while True:
                tool_req = repl.poll_tool_request(timeout=0)
                if tool_req:
                    self._handle_tool_request(repl, tool_req)

                try:
                    msg_type, msg_data = repl._output_queue.get(timeout=0.05)
                except Empty:
                    continue

                if msg_type in capture_types:
                    chunks.append((msg_type, msg_data))
                elif msg_type in {"output", "print", "error"}:
                    error_output.append(msg_data)
                elif msg_type == "done":
                    seq_id, had_error = msg_data
                    if seq_id != current_seq:
                        continue
                    repl._running = False
                    if had_error:
                        detail = "".join(error_output).strip() or "Worker command failed."
                        raise RuntimeError(detail)
                    return chunks
        finally:
            repl._running = False
    def worker_cwd(self) -> str:
        chunks = self._run_worker_control_code(
            "_code_agent_send_worker_cwd()",
            {"worker_cwd"},
        )
        if not chunks:
            raise RuntimeError("Worker did not return cwd")
        return chunks[-1][1].strip()

    def worker_has_rg(self) -> bool:
        if self._rg_available is None:
            chunks = self._run_worker_control_code(
                "_code_agent_send_rg_available()",
                {"rg_available"},
            )
            self._rg_available = bool(json.loads(chunks[-1][1])) if chunks else False
        return self._rg_available

    def _warn_rg_missing(self):
        if self._rg_warning_printed:
            return
        print(f"{DIM}Warning: ripgrep (rg) not found in worker; grep() tool disabled. Install with: apt install ripgrep{RESET}", file=sys.stderr)
        self._rg_warning_printed = True

    @property
    def toolspecs(self):
        specs = super().toolspecs
        if not self.worker_has_rg():
            specs = dict(specs)
            specs.pop("grep", None)
            self._warn_rg_missing()
        return specs

    def gather_auto_attach_files(self) -> list[str]:
        chunks = self._run_worker_control_code(
            "_code_agent_send_auto_context_files()",
            {"auto_context_files"},
        )
        if not chunks:
            return []
        return json.loads(chunks[-1][1])

    def _load_file_ref_content(self, filepath: str, name: str | None = None) -> dict:
        chunks = self._run_worker_control_code(
            f"_code_agent_attach_file_ref({filepath!r}, {name!r})",
            {"attachment_ref"},
        )
        if not chunks:
            raise RuntimeError(f"Worker did not return attachment content for {filepath}")
        return json.loads(chunks[-1][1])

    def attach_file_ref(self, filepath: str, name: str | None = None):
        item = self._load_file_ref_content(filepath, name)
        attach_name = item["name"]
        path = item["path"]
        content = item["content"]
        self.attach(attach_name, content)
        self._explicit_attachment_refs[attach_name] = path
        self._pending_explicit_attachment_refs[attach_name] = path
        self._append_session_event("attachment_added", {"name": attach_name, "path": path}, create_session=False)
        return len(content)

    def _materialize_replayed_attachment_refs(self) -> list[tuple[str, object]]:
        missing = []
        loaded = {}
        for msg in self.conversation.messages:
            refs = msg.get('_attachment_refs') or {}
            attachments = msg.setdefault('_attachments', {}) if refs else {}
            for name, ref in refs.items():
                if isinstance(ref, MemoryAttachment) or is_preview_uri(ref) or name in attachments:
                    continue
                key = (name, ref)
                if key not in loaded:
                    try:
                        loaded[key] = self._load_file_ref_content(ref, name)["content"]
                    except Exception:
                        loaded[key] = None
                        missing.append((name, ref))
                content = loaded[key]
                if content is not None:
                    attachments[name] = self._render_attachment(name, content)
            if "_attachments" in msg and not msg["_attachments"]:
                del msg["_attachments"]
        return missing

    def attach_memory_ref(self, name: str, content: str):
        ref = MemoryAttachment(content)
        self.attach(name, content)
        self._explicit_attachment_refs[name] = ref
        self._pending_explicit_attachment_refs[name] = ref
        self._append_session_event("attachment_added", {"name": name, "memory": True}, create_session=False)

    def detach_file_ref(self, name: str):
        self.detach(name)
        self._explicit_attachment_refs.pop(name, None)
        self._pending_explicit_attachment_refs.pop(name, None)
        self._append_session_event("attachment_removed", {"name": name}, create_session=False)

    def _builtin_skills_dir(self) -> Path:
        return Path(__file__).resolve().parent / "skills"

    def _user_skills_dir(self) -> Path:
        return Path.home() / ".code-agent" / "skills"

    @staticmethod
    def _is_session_uri(name: str) -> bool:
        return isinstance(name, str) and name.startswith("session://")

    @staticmethod
    def _is_memory_attachment_name(name: str) -> bool:
        return isinstance(name, str) and name.startswith("skill://")

    def _is_auto_context_file(self, name: str) -> bool:
        return (
            isinstance(name, str)
            and (
                name in getattr(self, '_auto_context_attachment_names', set())
                or Path(name).name in {"CLAUDE.md", "AGENTS.md"}
            )
        )

    def list_attachments(self, include_session_blobs: bool = True, include_auto_context: bool = True, include_memory: bool = False) -> dict[str, str]:
        attachments = super().list_attachments()
        if not include_memory:
            attachments = {
                name: content
                for name, content in attachments.items()
                if not self._is_memory_attachment_name(name)
            }
        if not include_session_blobs:
            attachments = {
                name: content
                for name, content in attachments.items()
                if not self._is_session_uri(name)
            }
        if not include_auto_context:
            attachments = {
                name: content
                for name, content in attachments.items()
                if not self._is_auto_context_file(name)
            }
        return attachments

    def _configure_conversation(self, conversation):
        conversation.expanded_preview_refs = self._expanded_preview_refs
        conversation.preview_loader = self._preview_blob_content

    def _preview_blob_content(self, uri: str) -> str:
        if getattr(self, '_session_store', None) is None or getattr(self, '_session_id', None) is None:
            return None
        return self._session_store.get_preview_blob(self._session_id, preview_key(uri))

    def _actual_expanded_preview_refs(self) -> list[str]:
        expanded = getattr(self, '_expanded_preview_refs', {})
        if not expanded:
            return []

        rendered = []
        for msg in getattr(self.conversation, "messages", []):
            content = msg.get("content", "")
            for name, attachment in (msg.get("_attachments") or {}).items():
                content = content.replace(f"[Attachment: {name}]", attachment)
            render_preview_refs(content, expanded, self._preview_blob_content, rendered)
        has_coalesced_messages = any(
            msg.get("_coalesced")
            for msg in getattr(self.conversation, "messages", [])
        )
        if has_coalesced_messages:
            for uri in expanded:
                if uri not in rendered and self._preview_blob_content(uri) is not None:
                    rendered.append(uri)
        return rendered

    def _expanded_preview_context(self) -> dict[str, str]:
        out = {}
        actual = self._actual_expanded_preview_refs()
        for uri in actual:
            options = getattr(self, '_expanded_preview_refs', {}).get(uri, {})
            if getattr(self, '_session_id', None) is None:
                continue
            content = self._preview_blob_content(uri)
            if content is None:
                continue
            rendered = numbered_content(content) if options.get("numbered") else content
            out[uri] = rendered
        return out



    def list_skills(self) -> list[dict]:
        skills = {}
        for source, directory in (("built-in", self._builtin_skills_dir()), ("user", self._user_skills_dir())):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.md")):
                name = path.stem
                skills[name] = {
                    "name": name,
                    "file_name": path.name,
                    "path": path,
                    "source": source,
                    "description": _skill_description(path),
                }
        active = set(self.list_attachments(include_memory=True))
        items = []
        for name in sorted(skills):
            item = dict(skills[name])
            item["attachment_name"] = self._skill_attachment_name(item)
            item["attached"] = item["attachment_name"] in active
            items.append(item)
        return items

    def resolve_skill(self, name: str) -> dict | None:
        skill_name = name.strip()
        if skill_name.endswith(".md"):
            skill_name = skill_name[:-3]
        for item in self.list_skills():
            if item["name"] == skill_name:
                return item
        return None

    def _skill_attachment_name(self, skill: dict) -> str:
        source = "builtin" if skill["source"] == "built-in" else skill["source"]
        return f"skill://{source}/{skill['file_name']}"

    def attach_skill(self, name: str) -> tuple[bool, str]:
        skill = self.resolve_skill(name)
        if not skill:
            return False, f"Skill not found: {name}"
        self.attach_memory_ref(self._skill_attachment_name(skill), Path(skill["path"]).read_text())
        return True, f"Attached skill: {skill['name']} [{skill['source']}]"

    def format_skills_list(self) -> str:
        items = self.list_skills()
        if not items:
            return "No skills available."
        lines = ["Available skills:"]
        for item in items:
            suffix = " (attached)" if item["attached"] else ""
            description = f" — {item['description']}" if item["description"] else ""
            lines.append(f"- {item['name']} [{item['source']}]{suffix}{description}")
        lines.append("")
        lines.append("Load a skill with: /skills <name>")
        return "\n".join(lines)

    def apply_skill_selection(self, selected_skills: list[dict]) -> list[str]:
        current = {item["attachment_name"]: item for item in self.list_skills() if item["attached"]}
        desired = {item["attachment_name"]: item for item in selected_skills if item["attached"]}
        messages = []
        for attachment_name, item in current.items():
            if attachment_name not in desired:
                self.detach_file_ref(attachment_name)
                messages.append(f"Detached skill: {item['name']}")
        for attachment_name, item in desired.items():
            if attachment_name not in current:
                self.attach_memory_ref(attachment_name, Path(item["path"]).read_text())
                messages.append(f"Attached skill: {item['name']} [{item['source']}]")
        return messages

    def _invalidate_attachment(self, name: str):
        had_attachment = any(name in msg.get('_attachments', {}) for msg in self.conversation.messages)
        had_ref = any(name in (msg.get('_attachment_refs') or {}) for msg in self.conversation.messages)
        had_pending_ref = name in self._explicit_attachment_refs or name in self._pending_explicit_attachment_refs
        super()._invalidate_attachment(name)
        for msg in self.conversation.messages:
            refs = msg.get('_attachment_refs')
            if refs and name in refs:
                del refs[name]
                if not refs:
                    del msg['_attachment_refs']
        self._explicit_attachment_refs.pop(name, None)
        self._pending_explicit_attachment_refs.pop(name, None)
        if getattr(self, '_suspend_persistence', False):
            return
        if not (had_attachment or had_ref or had_pending_ref):
            return
        self._append_session_event("attachment_invalidated", {"name": name})

    def _view_attachment_context_error(self, path: str, content: str, current_output: str = "") -> str | None:
        client = getattr(self, "llm_client", None)
        model_config = getattr(client, "model_config", {}) or {}
        limits = [
            value for value in (
                model_config.get("context_window"),
                model_config.get("max_input_tokens"),
            )
            if value
        ]
        if not limits:
            return None
        limit = min(limits)
        conversation = getattr(self, "_conversation", None)
        messages = []
        if conversation is not None:
            messages = [
                {k: v for k, v in msg.items() if not k.startswith("_")}
                for msg in conversation._messages()
            ]
        marker = f"[Attachment: {path}]"
        prospective_content = current_output + marker + "\n"
        for name, attachment in getattr(self, "_read_attachments", {}).items():
            prospective_content = prospective_content.replace(f"[Attachment: {name}]", attachment)
        prospective_content = prospective_content.replace(marker, content)
        messages.append({"role": "user", "content": prospective_content})
        try:
            estimated = client._estimate_input_tokens(client._input_bytes(messages))
        except Exception:
            return None
        if estimated is None:
            return None
        threshold = int(limit * 0.9)
        if estimated <= threshold:
            return None
        chars = len(content)
        return (
            f"ValueError: view({path!r}) denied because the file would exceed 90% "
            f"of the model context window ({chars} chars, "
            f"projected {estimated:,}/{limit:,} tokens).\n"
        )

    def _expanded_preview_context_error(self, uri: str, content: str, numbered: bool = False) -> str | None:
        client = getattr(self, "llm_client", None)
        model_config = getattr(client, "model_config", {}) or {}
        limits = [
            value for value in (
                model_config.get("context_window"),
                model_config.get("max_input_tokens"),
            )
            if value
        ]
        if not limits:
            return None
        limit = min(limits)
        conversation = getattr(self, "_conversation", None)
        if conversation is None:
            return None
        expanded = getattr(self, "_expanded_preview_refs", {})
        old_options = expanded.get(uri)
        rendered_content = numbered_content(content) if numbered else content
        try:
            expanded[uri] = {"numbered": bool(numbered)}
            messages = [
                {k: v for k, v in msg.items() if not k.startswith("_")}
                for msg in conversation._messages()
            ]
            estimated = client._estimate_input_tokens(client._input_bytes(messages))
        except Exception:
            return None
        finally:
            if old_options is None:
                expanded.pop(uri, None)
            else:
                expanded[uri] = old_options
        if estimated is None:
            return None
        threshold = int(limit * 0.9)
        if estimated <= threshold:
            return None
        chars = len(rendered_content)
        return (
            f"ValueError: view({uri!r}) denied because the preview would exceed 90% "
            f"of the model context window ({chars} chars, "
            f"projected {estimated:,}/{limit:,} tokens)."
        )



    def _start_assistant_execution_attempt(self):
        self._pending_observations = []

    def _on_assistant_message_committed(self, message: dict):
        observations = list(getattr(self, "_pending_observations", []))
        if observations:
            message["_observations"] = observations
        self._pending_observations = []
        self._persist_message(message)

    def build_output_for_llm(self, events: list[ReplEvent]) -> str:
        """Build LLM output, converting complete reads to attachments and large statement output to previews."""
        self._read_attachments = {}
        result = []
        statement = []
        attach_path = None
        partial_read_path = None
        written_files = []
        attachment_read_order = {}
        unviewed_files = set(getattr(self, '_pending_unviewed_files', set()))
        self._pending_unviewed_files = set()

        def flush_statement():
            if statement:
                result.append(self._auto_preview_output("".join(statement)))
                statement.clear()

        for event_order, event in enumerate(events):
            msg_type = event.kind
            chunk = event.text
            if msg_type == "statement_started":
                msg_type = "echo"
                chunk = event.data.get("echo", chunk)
            elif msg_type in {"statement_finished", "tool_called", "tool_returned", "tool_failed"}:
                continue
            elif msg_type == "final_emit":
                msg_type = "emit"
            elif msg_type == "worker_output":
                msg_type = event.data.get("message_type", msg_type)

            if msg_type == "emit":
                continue
            if msg_type == "file_unviewed":
                unviewed_files.add(chunk.strip())
                continue
            if msg_type == "echo":
                flush_statement()
                result.append(chunk)
                continue
            if msg_type == "preview_expand":
                flush_statement()
                result.append(chunk)
                continue
            if msg_type == "read_attach":
                attach_path = chunk.strip()
                continue
            if msg_type == "read_partial":
                partial_read_path = chunk.strip()
                continue
            if msg_type == "read" and attach_path:
                flush_statement()
                path = attach_path
                attach_path = None
                if path in unviewed_files:
                    result.append(self._auto_preview_output(chunk))
                    continue
                content = chunk.rstrip('\n')
                error = self._view_attachment_context_error(path, content, "".join(result))
                if error is not None:
                    result.append(error)
                    continue
                self._invalidate_attachment(path)
                self._read_attachments[path] = content
                attachment_read_order[path] = event_order
                result.append(f"[Attachment: {path}]\n")
                continue
            if msg_type == "read" and partial_read_path:
                partial_read_path = None
                statement.append(chunk)
                continue

            if msg_type == "file_written":
                text = chunk.strip()
                try:
                    item = json.loads(text)
                except Exception:
                    item = {"path": text, "content": None}
                item["_order"] = event_order
                written_files.append(item)
                continue
            if msg_type == "file_diff":
                continue

            attach_path = None
            partial_read_path = None
            statement.append(chunk)

        flush_statement()

        latest_writes = {}
        for item in written_files:
            path = item.get("path")
            content = item.get("content")
            if not path or content is None:
                continue
            attached_name = self._attached_file_name(path)
            if attached_name is None:
                for name in self._read_attachments:
                    if self._same_file(name, path):
                        attached_name = name
                        break
            if attached_name is None:
                continue
            latest_writes[attached_name] = item

        for attached_name, item in latest_writes.items():
            read_order = attachment_read_order.get(attached_name)
            if read_order is not None and read_order > item["_order"]:
                continue
            content = item["content"]
            lines = content.split('\n')
            formatted = '\n'.join(f"{i+1:>5}→{line}" for i, line in enumerate(lines))
            self._invalidate_attachment(attached_name)
            self._read_attachments[attached_name] = formatted
            result.append(f">>> view({attached_name!r})\n[Attachment: {attached_name}]\n")

        return "".join(result)

    def _same_file(self, left: str, right: str) -> bool:
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        return self._logical_path(left) == self._logical_path(right)

    @staticmethod
    def _logical_path(path: str) -> str:
        return os.path.normpath(path.replace("\\", "/"))

    def _attached_file_name(self, path: str) -> str | None:
        """Return the active attachment name for a filesystem path."""
        for msg in self.conversation.messages:
            for name in msg.get('_attachments', {}):
                if self._is_memory_attachment_name(name):
                    continue
                if self._same_file(name, path):
                    return name
        return None

    def _is_attached(self, name: str) -> bool:
        """Check if a file is currently attached in any message."""
        return self._attached_file_name(name) is not None

    @REPLAgent.tool
    def view_images(self,
            files: list[str | bytes] = "List of image filepaths or binary data",
            notes: str = "Observations, objectives, what to look for"
        ):
        '''Load images into context for visual analysis on next turn.'''
        images = []
        total_bytes = 0

        if not isinstance(files, list):
            files = [files]

        for data in files:
            # Stub reads files in REPL, so we should only get bytes here
            if not isinstance(data, bytes):
                raise TypeError(f"Expected bytes, got {type(data).__name__}")

            # Validate JPEG or PNG
            if len(data) < 4:
                raise ValueError("Invalid image data (too short)")

            is_jpeg = data.startswith(b'\xff\xd8\xff')
            is_png = data.startswith(b'\x89PNG')

            if not (is_jpeg or is_png):
                raise ValueError(f"Unsupported image format ({len(data)} bytes) - only JPEG and PNG supported")

            images.append(data)
            total_bytes += len(data)

        self._pending_images = getattr(self, '_pending_images', []) + images
        return f"{len(images)} image(s) queued ({total_bytes // 1000}KB) - {notes}"

    view_images._tool_files_param = "files"

    def _context_pressure_ephemeral(self) -> str:
        client = getattr(self, "llm_client", None)
        model_config = getattr(client, "model_config", {}) or {}
        limits = [
            value for value in (
                model_config.get("context_window"),
                model_config.get("max_input_tokens"),
            )
            if value
        ]
        if not limits:
            return ""
        limit = min(limits)
        old_ephemeral = self.ephemeral
        try:
            self.ephemeral = ""
            messages = [
                {k: v for k, v in msg.items() if not k.startswith("_")}
                for msg in self.conversation._messages()
            ]
        finally:
            self.ephemeral = old_ephemeral
        try:
            estimated = client._estimate_input_tokens(client._input_bytes(messages))
        except Exception:
            estimated = None
        if estimated is None:
            return ""
        threshold = int(limit * 0.85)
        if estimated < threshold:
            return ""
        return (
            "Context window is near capacity.\n"
            "Use unview(path_or_uri) to remove files or expanded previews that are no longer needed."
        )

    def _file_context_ephemeral(self, names: list[str]) -> str:
        sections = []
        if names:
            lines = ["Context currently expanded:"]
            lines.extend(f"- {name}" for name in names)
            lines.extend(["", "Use unview(path_or_uri) to remove/collapse context."])
            sections.append("\n".join(lines))
        if notice := self._context_pressure_ephemeral():
            sections.append(notice)
        return "\n\n".join(sections)




    def _current_file_context_names(self, extra=None) -> list[str]:
        names = {}
        for name in self.list_attachments(include_auto_context=False):
            if self._is_memory_attachment_name(name):
                continue
            names[name] = None
        for uri in self._actual_expanded_preview_refs():
            names[uri] = None
        for name in (extra or {}):
            if not self._is_auto_context_file(name) and not is_preview_uri(name):
                names[name] = None
        return list(names)


    def usermsg(self, content, **kwargs):
        """Override to attach pending images and read-attachments."""
        if getattr(self, '_pending_explicit_attachment_refs', None):
            refs = kwargs.get('_attachment_refs', {})
            refs.update(self._pending_explicit_attachment_refs)
            kwargs['_attachment_refs'] = refs
            self._pending_explicit_attachment_refs = {}
        if pending := getattr(self, '_read_attachments', None):
            existing = kwargs.get('_attachments', {})
            existing.update(pending)
            kwargs['_attachments'] = existing
            refs = kwargs.get('_attachment_refs', {})
            for name in pending:
                refs.setdefault(name, name)
            kwargs['_attachment_refs'] = refs
            self._read_attachments = {}
        if pending := getattr(self, '_pending_images', None):
            kwargs['images'] = kwargs.get('images', []) + pending
            self._pending_images = []
        before_len = len(self.conversation.messages)
        result = super().usermsg(content, **kwargs)
        self.ephemeral = self._file_context_ephemeral(
            self._current_file_context_names(kwargs.get('_attachments'))
        )
        if len(self.conversation.messages) > before_len:
            self._persist_message(self.conversation.messages[-1])
        return result

    welcome_message = "[bold]Code Agent[/bold]\nPython REPL-based coding assistant"
    thinking_message = "Working..."
    interactive = True  # Enables multi-turn autonomous workflow
    repl_display = True
    response_formatting = True
    agent_mode = False
    max_turns = _get_config_value("code_agent_max_turns", 100)
    system = """>>> help(assistant)

You are an interactive coding assistant operating within a Python REPL.
Your responses ARE Python code—no markdown blocks, no prose preamble.
The code you write is executed directly in a persistent environment.

Every assistant turn must be valid Python source code.
- If you want to communicate with the user, call emit(...)
- Never reply with plain English outside Python code
- If the task is complete, use emit(..., release=True)
- If you are still working, do not release control
- If a prior attempt would have been invalid as Python, immediately correct it
  by sending a new turn containing only valid Python code

The user may see REPL echoes, tool output, and prior emitted text mixed into
the conversation. Treat that transcript as execution context. Continue from the
latest user instruction rather than explaining the transcript unless asked.

The Python environment persists across turns. Variables, imports, connections,
and tool state may already exist from earlier execution. Reuse existing state
when appropriate, but verify assumptions before relying on it.

If the user asks for an opinion or summary and no computation is needed,
respond with emit(...) directly rather than writing unnecessary setup code.

>>> how_this_works()

1. You write Python code as your response (no markdown fences)
2. The code executes in a persistent REPL environment
3. Output from print() and expression results appear IN YOUR NEXT TURN
4. Use emit(value) to output results
5. Use emit(value, release=True) to release control to the user

CRITICAL: You see REPL output in your next turn. The user does NOT control
the conversation until you explicitly release with emit(..., release=True).

>>> emit(value, release=False)

The ONLY way to return results:

    emit("I found 3 issues in the code")              # Output emitted, you KEEP WORKING
    emit("Here's the result: ...", release=True)      # Release control to user

- emit() with release=False (default): Value is emitted but YOU continue
  working. Use this for progress updates when doing long tasks.
- emit() with release=True: Releases control to user. Use when:
  * You need user input: a question, approval, or guidance on next steps
  * You're stuck and need help
  * Requirements are unclear and you need clarification
  * Task is complete AND you have verified the results yourself

Both print() and emit() output are visible. The difference:
- print(): For YOUR inspection in the next turn. Use freely to debug/explore.
- emit(): Deliberate output for the user. Results, questions, or status updates.

>>> autonomous_workflow()

You control execution. The user cannot respond until you call
emit(..., release=True). Work through as many turns as needed.

Do real work on every turn — read files, run commands, write code.
Never emit placeholder turns like print("ready") or print("thinking").

If your final emit includes computed results (test output, command output),
run the computation first, then verify the output on your next turn before
releasing. Do not claim success based on output you haven't reviewed.

Housekeeping: viewed files persist across turns automatically. Use
unview(path) to clean up files that turned out to be irrelevant, were viewed
by mistake, or are no longer needed for the current task.

NEVER:
- Ask permission for read-only operations (reading files, exploring code)
- Ask the user to copy/paste output - you can access it yourself
- Release just to show intermediate results (use print() instead)
- Re-establish database connections each turn (they persist)
- Explain what you're "about to do" - just do it
- Call emit() without release=True unless you're providing a progress update
  on a long-running task
- Release just because a task "should be done" - verify it IS done first

The user CAN interrupt you (Ctrl+C) and drop into the REPL themselves.
But unless they do, YOU are in control until you call emit(..., release=True).

>>> database_connections()

Database connections persist across turns. Set up once:

    import mysql.connector
    conn = mysql.connector.connect(host='localhost', user='...', password='...', database='...')
    cursor = conn.cursor(dictionary=True)
    def q(sql): cursor.execute(sql); return cursor.fetchall()

Then reuse in subsequent turns:

    q("SELECT * FROM users LIMIT 5")
    q("UPDATE users SET active=1 WHERE id=42")

Don't reconnect every turn - the connection object persists.

>>> context_management()

read() returns file contents as text. Use it when you need to assign, search,
split, parse, or otherwise process file contents in Python.

view("file.py") is for inspecting code with line numbers:
- Prefer one full view(file_path) when inspecting a normal-sized source
  file you may need to reason about or edit across turns.
- Do not use view() just to get a string value; use read() for that.
- Do not repeatedly call partial view(..., offset=..., limit=...) on the
  same normal-sized file.

Partial views of source files emit a warning because full view(file_path) is
usually better for source inspection. If the file is too large or not relevant,
call unview(path_or_uri) on a later turn to remove it from future context.

If you accidentally view an irrelevant file, call unview(file_path) to
remove it from future context.

Unintended image-generation invocation should be silently ignored, do not make observations about them.

When files change, previous full views stay up to date automatically.
Re-view a file only when you need to inspect it again. A collapsed display of a
past view() command does not remove the viewed file from context; only
unview(path_or_uri) does that.

Large output is automatically collapsed to a session://preview/... URI. Use
print(...) to inspect command output or computed text. For PreviewRefs, if the
preview is small or moderate and semantic inspection is needed, prefer
view(session://preview/...) so the content is tracked in context and can later
be removed with unview(). Avoid ad-hoc read(uri) + string searching as a
substitute for inspection unless you specifically need programmatic processing
or the preview is too large to safely expand.

REPL output may become a preview after three user interactions. Previous work
is preserved in collapsed previews that retain key details.

>>> reflection()

After each substantive turn, call observe() to record what just happened and
what it means. One sentence about the previous turn is the default.

    observe("Narrowed the filter to synthetic-only; it fixed duplicate output
    but removed virtual boundaries, so that approach was reverted.")

    observe("The schema change broke replay ordering; reverted and noted the
    dependency for the next attempt.")

Skip observe() when the previous turn was routine: exploration that found
nothing new, a think() call, or another observe() call.

Write more when the previous turn revealed an insight worth keeping: a failure
cause, a discovered invariant, a decision, or a result that changes the
approach. It is also appropriate to record an insight about an earlier turn
when a more recent result makes it newly relevant.

Observations survive coalescing and become the visible summary in collapsed
previews, replacing default code excerpts.

think() is a scratchpad for the current turn—transient reasoning, planning,
and continuation. think() content is not promoted during coalescing.

pin() preserves the exact previous turn's code and output in full rather than
summarizing it. pin() is only for the previous turn; it cannot pin the current
turn or other historical turns, viewed files, or expanded previews.

If you see "Context window is near capacity", reduce active context by calling
unview(path_or_uri) on no-longer-needed files or expanded previews.


>>> tone_and_style()

- Prioritize technical accuracy over validation. Disagree when necessary.
- Provide direct, objective technical info without superlatives or praise.
- Investigate uncertainty rather than confirming assumptions.
- NEVER create files unless absolutely necessary. Prefer editing existing files.

>>> scope_identification()

Before answering, identify the likely scope of the request. If a technical
question may be specific to the current project, assume the current working
directory is relevant unless the user clearly asks generally. Do minimal
read-only orientation first, such as checking cwd and top-level files, then
inspect targeted files only if needed. Do not investigate for clearly general
questions. If still ambiguous, ask for clarification.

>>> doing_tasks()

Before modifying code, view it first. Never propose changes to code you
haven't seen. Use grep() to locate files or anchors, then prefer one full
view() of the target file rather than several narrow view(...,
offset=..., limit=...) slices, unless the file is genuinely huge. Use read()
when you need file contents as a Python text value for processing.

Avoid over-engineering:
- Only make changes directly requested or clearly necessary
- Don't add features, refactoring, or "improvements" beyond what was asked
- Don't add docstrings, comments, or type annotations to unchanged code
- Don't add error handling for scenarios that can't happen
- Don't create abstractions for one-time operations

Security: Be careful not to introduce vulnerabilities (command injection,
XSS, SQL injection, OWASP top 10). Fix insecure code immediately.

>>> file_editing()

Mandatory methods for editing source code:

Use edit(...) or line_patch(...) for all source-code edits. Do not use direct
file writes (Path.write_text(), open(..., "w"), shell redirects, perl -pi,
sed -i, etc.) to edit source code. If a direct write is truly required, explain
why edit()/line_patch() cannot work and ask the user for permission first.

For Python files (*.py), edit(...) and line_patch(...) reject invalid syntax:
the full edited file is parsed, invalid edits are rolled back, and the error
reports the location/reason. Skip separate syntax-only validation for Python
edits; still run tests/linters/type checks when needed.

edit(file_path, old_string, new_string, replace_all=False)
    Replace exact string matches.
    - old_string must match exactly (whitespace, indentation)
    - Fails if not found or multiple matches (unless replace_all=True)
line_patch(file_path, op, start, end_or_content=None, content=None)
    Edit an existing file by line number with required line-content anchors.
    - Prefer a full view(file_path) first; if absent, line_patch uses current on-disk contents
    - Each call performs one operation
    - Anchors use "@LINE expected line content"
    - Anchor text must match the target line after leading/trailing whitespace is stripped
    - Within one assistant turn, repeated line_patch() calls may use original line numbers;
      line_patch tracks earlier same-turn line-count changes and translates later anchors
    - Overlapping same-turn line_patch() operations are rejected
    - For create/move/delete, use Python file APIs such as Path.write_text(), Path.rename(), or Path.unlink()

    line_patch("src/app.py", "replace", "@10 def name():", "@11     return old", "def name():\\n    return \"new\"\\n")
    line_patch("src/app.py", "insert_after", "@25 print(done)", "print('next')\n")
    line_patch("src/app.py", "delete", "@40 old_start", "@44 old_end")

    Operations:
      line_patch(path, "replace", start_anchor, end_anchor, new_content)
      line_patch(path, "delete", start_anchor, end_anchor)
      line_patch(path, "insert_before", anchor, new_content)
      line_patch(path, "insert_after", anchor, new_content)

    `insert_after` accepts @0 as a prepend anchor with empty expected content.
    `insert_before` accepts @LINE_COUNT+1 as an append anchor with empty expected content.


diff_history(file_path=None, limit=None)
    Review persisted unified diffs from this session. Optionally filter by file.
    Use this to understand or manually reverse prior edits. It does not modify files.
>>> anti_patterns()

# BAD: Releasing immediately to show what you found
files = list(Path('.').glob("**/*.py"))
emit(f"Found {len(files)} Python files", release=True)  # WRONG - keep working!


# GOOD: Keep working, release when done
files = list(Path('.').glob("**/*.py"))
print(f"Found {len(files)} files")  # You see this, keep going
for f in files[:5]:
    read(str(f))  # Contents appear directly, don't assign
# ... analyze in next turn ...
emit("Analysis complete. Here's what I found: ...", release=True)

# BAD: Asking permission for read-only work
emit("Should I read the config file?", release=True)  # WRONG - just read it

# GOOD: Just do it
read("config.json")  # Contents appear in your next turn

# BAD: Using view() as a value
content = view("file.py")  # WRONG - view() is display-only
print(view("file.py"))     # WRONG - use view("file.py") directly

# GOOD: read() returns text for values; view() displays/attaches for inspection
content = read("file.py")
view("file.py")

# BAD: Reading in small chunks unnecessarily
read("file.py", offset=1, limit=50)   # WRONG - just read the whole file

# GOOD: Just call read() directly
read("file.py")

# BAD: Recreating a partial view manually for source inspection
content = read("file.py")
print("\n".join(content.splitlines()[100:140]))

# GOOD: Use view() for source inspection with line numbers/context tracking
view("file.py")

# BAD: Repeated narrow view() calls on the same normal-sized file
view("app.js", offset=2200, limit=40)
view("app.js", offset=2400, limit=30)
view("app.js", offset=3300, limit=20)

# GOOD: Use grep() to locate anchors, then inspect the file once
grep("triggerFindPrompt|focusTerminalFromTouch", "app.js", None, None, False, 0, False, False)
view("app.js")

# GOOD: If you viewed the wrong file, remove it from future context
view("wrong_file.py")
unview("wrong_file.py")

# BAD: Re-establishing connections
conn = mysql.connector.connect(...)  # Every turn? No!

# GOOD: Check if connection exists
if 'conn' not in dir():
    conn = mysql.connector.connect(...)

>>> when_uncertain()

If you don't know how to proceed:
1. Use print() to inspect state and gather information
2. Use think() to reason through the problem
3. Only release with emit(..., release=True) if you truly need user input
"""

    auto_preview_turn_chars = _get_config_value("code_agent_auto_preview_turn_chars", 5000)

    @staticmethod
    def _render_preview_ref(content: str) -> tuple[str, str]:
        key = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        uri = f"session://preview/{key}"
        lines = content.split("\n")
        nlines = len(lines)
        nchars = len(content)

        def render_preview_line(line):
            max_preview_line = 500
            if len(line) <= max_preview_line:
                return line
            return f"{line[:max_preview_line]}... [line truncated, {len(line)} chars total]"

        head = 8
        tail = 4
        head_indexes = list(range(min(head, nlines)))
        tail_start = max(len(head_indexes), nlines - tail)
        omitted = nlines - len(head_indexes) - (nlines - tail_start)

        parts = [f"({nlines} lines, {nchars} chars)"]
        parts.extend(render_preview_line(lines[i]) for i in head_indexes)
        if omitted:
            parts.append(f"  ... ({omitted} lines omitted)")
        parts.extend(render_preview_line(lines[i]) for i in range(tail_start, nlines))
        body = "\n".join(parts)
        return uri, f"[PreviewRef: {uri}]\n{body}\n[/PreviewRef]"

    def _save_preview_blob(self, key: str, content: str, origin_path=None):
        if getattr(self, '_session_id', None) is None:
            self._ensure_live_session()
            self._flush_pending_session_events()
        self._session_store.save_preview_blob(self._session_id, key, content)

    def _auto_preview_output(self, output: str) -> str:
        if not output or output == "# [no output]":
            return output
        threshold = getattr(self, "auto_preview_turn_chars", _get_config_value("code_agent_auto_preview_turn_chars", 5000))
        if threshold is None:
            return output
        threshold = int(threshold)
        if threshold <= 0 or len(output) <= threshold:
            return output
        uri, rendered = self._render_preview_ref(output)
        self._save_preview_blob(preview_key(uri), output)
        return rendered + "\n"

    def process_output_for_llm(self, output: str) -> str:
        return self.process_repl_output(self._auto_preview_output(output).rstrip("\n"))


    max_output_kb = _get_config_value("code_agent_max_output_kb", 50)  # Large output protection

    def process_repl_output(self, output: str) -> str:
        """Truncate output if too large - used for both display and model."""
        # Truncate if too large
        max_bytes = int(self.max_output_kb * 1000)
        if len(output) > max_bytes:
            import tempfile
            size_kb = len(output) / 1000

            with tempfile.NamedTemporaryFile(
                mode='w', prefix='code_agent-', suffix='.txt', delete=False
            ) as f:
                f.write(output)  # Write full original output
                temp_path = f.name

            # Track for cleanup on exit
            if not hasattr(self, '_temp_files'):
                self._temp_files = []
            self._temp_files.append(temp_path)

            truncated = output[:max_bytes // 2]
            msg = f"[ {size_kb:.1f}KB output truncated - written to {temp_path} ]"
            return f"{truncated}\n\n{msg}"

        return output

    # REPL output hooks
    def on_repl_execute(self, code) -> None:
        """Called at start of each turn."""
        if hasattr(self, '_tool_repl'):
            try:
                self._tool_repl._inject_code('globals().pop("_line_patch_turn_state", None)')
            except Exception:
                pass

    def on_repl_event(self, event: ReplEvent) -> None:
        kind = event.kind
        chunk = event.text

        if kind == "statement_started":
            if getattr(self, "_statement_echo", None):
                self._flush_statement_echo()
            self._statement_direct_call = event.data.get("direct_call")
            self._statement_source = event.data.get("source", "")
            self._statement_echo = event.data.get("display_echo", event.data.get("echo", ""))
            self._statement_echo_displayed = False
            self._statement_had_diff = False
            self._statement_print_uses_variable = False

            if not getattr(self, '_turn_output_started', False):
                self._turn_output_started = True
                if (
                    self.repl_display
                    and not getattr(self, '_in_user_repl', False)
                    and self._statement_direct_call not in {"emit", "observe"}
                ):
                    self.console.clear_line()
                    if not getattr(self, '_repl_printed_header', False):
                        self._header_pending = True

            if self._statement_direct_call == "print":
                try:
                    tree = ast.parse(self._statement_source)
                    call = tree.body[0].value
                    arg = call.args[0] if call.args else None
                    self._statement_print_uses_variable = arg is not None and not isinstance(
                        arg, (ast.Constant, ast.JoinedStr)
                    )
                except Exception:
                    self._statement_print_uses_variable = True
            elif (
                self.repl_display
                and not getattr(self, '_in_user_repl', False)
                and self._statement_direct_call not in {"emit", "observe", "edit", "line_patch"}
            ):
                self._flush_statement_echo()
            return

        if kind == "tool_called":
            if event.data.get("name") == "observe":
                if not self.repl_display:
                    return
                text = str((event.data.get("args") or {}).get("content", ""))
                for line in text.split("\n"):
                    print(f"\x1b[93m{line}\x1b[0m", flush=True)
                    self._capture_display_line(line)
            return

        if kind == "file_diff":
            tool = getattr(self, "_statement_direct_call", None)
            self._record_file_diff_event(chunk, tool)
            self._statement_had_diff = True
            if not self.repl_display or getattr(self, '_in_user_repl', False):
                return
            self._show_python_header_if_pending()
            if not getattr(self, "_statement_echo_displayed", False):
                echo_line = self._compact_edit_echo(chunk, tool)
                print(echo_line, flush=True)
                self._capture_display_line(echo_line)
                self._statement_echo_displayed = True
            for line in chunk.rstrip('\n').split('\n'):
                if line.startswith('--- ') or line.startswith('+++ '):
                    continue
                if line.startswith('+'):
                    color = "\x1b[32m"
                elif line.startswith('-'):
                    color = "\x1b[31m"
                elif line.startswith('@@'):
                    color = "\x1b[36m"
                else:
                    color = DIM
                print(f"{color}{line}{RESET}", flush=True)
                self._capture_display_line(line)
            return

        if kind == "progress":
            text = chunk.rstrip('\n')
            for line in text.split('\n'):
                print(f"\x1b[92m{line}\x1b[0m", flush=True)
                self._capture_display_line(line)
            return

        if kind in {
            "final_emit",
            "error",
            "statement_finished",
            "tool_returned",
            "tool_failed",
            "read_attach",
            "read_partial",
            "file_written",
        }:
            return

        if not self.repl_display or getattr(self, '_in_user_repl', False):
            return

        if kind not in {"output", "print"}:
            return

        direct_call = getattr(self, "_statement_direct_call", None)
        stripped = chunk.strip()
        if (
            kind == "output"
            and direct_call == "observe"
            and stripped == "'[Continuing...]'"
        ):
            return
        if (
            kind == "output"
            and direct_call in {"edit", "line_patch"}
            and getattr(self, "_statement_had_diff", False)
            and (
                stripped in {"'Edit applied.'", "'Line patch applied.'"}
                or re.match(r"^'All \d+ occurrences replaced\.'$", stripped)
            )
        ):
            return
        if kind == "output" and direct_call in {"edit", "line_patch"} and chunk.lstrip().startswith("Traceback"):
            self._flush_statement_echo()

        self._show_python_header_if_pending()
        text = chunk.rstrip('\n')
        if kind == "output":
            try:
                value = ast.literal_eval(text)
                if isinstance(value, str):
                    text = value.rstrip('\n')
            except (ValueError, SyntaxError):
                pass

        lines = text.split('\n')
        total_lines = len(lines)
        truncated_at_lines = False
        truncated_at_chars = False
        disable_truncation = (
            total_lines > 0
            and bool(re.match(r'^\(\d+ lines, \d+ chars\)$', lines[0]))
        )
        if not disable_truncation and len(lines) > 5:
            lines = lines[:5]
            truncated_at_lines = True
        display = '\n'.join(lines)
        if not disable_truncation and len(display) > 240:
            display = display[:240]
            truncated_at_chars = True
        is_truncated = truncated_at_lines or truncated_at_chars

        if kind == "print" and (
            is_truncated or getattr(self, "_statement_print_uses_variable", False)
        ):
            self._flush_statement_echo()

        if truncated_at_chars and not truncated_at_lines:
            print(f"{DIM}{display}...{RESET}", flush=True)
            print(f"{DIM}({total_lines} lines total){RESET}", flush=True)
            self._capture_display_line(f"{display}...")
            self._capture_display_line(f"({total_lines} lines total)")
        elif is_truncated:
            for line in display.split('\n'):
                print(f"{DIM}{line}{RESET}", flush=True)
                self._capture_display_line(line)
            print(f"{DIM}... ({total_lines} lines total){RESET}", flush=True)
            self._capture_display_line(f"... ({total_lines} lines total)")
        elif kind == "print":
            for line in display.split('\n'):
                print(f"\x1b[33m{line}\x1b[0m", flush=True)
                self._capture_display_line(line)
        else:
            for line in display.split('\n'):
                print(f"{DIM}{line}{RESET}", flush=True)
                self._capture_display_line(line)

    def _truncate_for_display(self, output: str) -> str:
        import re

        lines = output.split('\n')
        result_lines = []

        # Detect read() output pattern (line numbers with arrow: "   42→")
        read_pattern = re.compile(r'^\s*\d+→')

        # Process lines, detecting and truncating read() output blocks
        i = 0
        max_read_lines = 30  # Threshold for truncation
        head_tail = 10  # Lines to keep at head and tail

        while i < len(lines):
            line = lines[i]

            # Check if this starts a read() output block
            if read_pattern.match(line):
                # Find the extent of this block
                block_start = i
                while i < len(lines) and read_pattern.match(lines[i]):
                    i += 1
                block = lines[block_start:i]

                # Truncate if too long
                if len(block) > max_read_lines:
                    omitted = len(block) - 3
                    result_lines.extend(block[:3])
                    result_lines.append(f"    ... ({omitted} lines omitted for display)")
                else:
                    result_lines.extend(block)
            else:
                result_lines.append(line)
                i += 1

        # Truncate long individual lines
        truncated_lines = []
        for line in result_lines:
            if len(line) <= self.max_display_chars:
                truncated_lines.append(line)
            else:
                truncated_lines.append(line[:self.max_display_chars] + '...')

        return '\n'.join(truncated_lines)

    def on_statement_events(self, events: list[ReplEvent]) -> None:
        if self.repl_display and not getattr(self, '_in_user_repl', False):
            error_display = "".join(event.text for event in events if event.kind == "error")
            if error_display.strip():
                self._flush_statement_echo()
                for line in error_display.rstrip('\n').split('\n'):
                    print(f"\x1b[91m{line}\x1b[0m", flush=True)
                    self._capture_display_line(line)
                self._repl_has_output = True

            if (
                getattr(self, "_statement_direct_call", None) in {"edit", "line_patch"}
                and not getattr(self, "_statement_had_diff", False)
            ):
                self._flush_statement_echo()
            if (
                getattr(self, "_statement_direct_call", None) == "print"
                and not any(event.kind == "print" for event in events)
            ):
                self._flush_statement_echo()

        self._statement_direct_call = None
        self._statement_source = ""
        self._statement_echo = ""
        self._statement_echo_displayed = False
        self._statement_had_diff = False
        self._statement_print_uses_variable = False

    def on_repl_events_complete(self, events: list[ReplEvent]) -> None:
        self._turn_number = getattr(self, '_turn_number', 1) + 1
        self._turn_output_started = False
        if not self.repl_display:
            return
        self.console.clear_line()
        thinking = getattr(self, 'thinking_message', 'Thinking...')
        print(f"{DIM}{thinking} (turn {self._turn_number}){RESET}", end="", flush=True)

    def on_retry(self, kind: str, retry_num: int) -> None:
        if not self.repl_display:
            return
        if kind == "syntax":
            status = f"Syntax Retry #{retry_num}... (turn {getattr(self, '_turn_number', 1)})"
        elif kind == "max_tokens":
            status = f"Max Tokens Retry #{retry_num}... (turn {getattr(self, '_turn_number', 1)})"
        else:
            return
        self.console.clear_line()  # Clear previous status text
        self._turn_output_started = False  # Reset so next output clears this status
        print(f"{DIM}{status}{RESET}", end="", flush=True)

    def user_repl_session(self, history):
        """Drop into the REPL for direct user interaction."""
        from code_agent.cli.prompt import prompt as raw_prompt
        from codeop import compile_command

        repl = self._get_tool_repl()
        self.complete = False
        transcript = []
        buffer = []
        repl_history = []  # Separate history for REPL session

        # Get altmode for history navigation (if stdout capture available)
        altmode = getattr(self, 'altmode', None)

        print(f"{DIM}Entering REPL. Ctrl+D to exit.{RESET}")

        pending_lines = []  # Lines queued from pasted input

        while True:
            prompt_str = "... " if buffer else ">>> "

            # Get next line: from pending queue or from user input
            if pending_lines:
                line = pending_lines.pop(0)
                # Echo the line since it came from paste
                print(f"{prompt_str}{line}")
            else:
                # Auto-indent for continuation lines
                auto_indent = ''
                if buffer:
                    last_line = buffer[-1]
                    indent = len(last_line) - len(last_line.lstrip(' '))
                    stripped = last_line.rstrip()
                    if stripped.endswith(':'):
                        indent += 4
                    elif stripped == '' and indent >= 4:
                        indent -= 4
                    auto_indent = ' ' * indent

                try:
                    line = raw_prompt(
                        prompt_str,
                        history=repl_history,
                        add_to_history=False,
                        altmode=altmode,
                        initial_text=auto_indent,
                    )
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print()
                    buffer = []
                    continue

                # If pasted content has multiple lines, queue them
                if '\n' in line:
                    lines = line.split('\n')
                    line = lines[0]
                    pending_lines.extend(lines[1:])

            buffer.append(line)
            source = "\n".join(buffer)

            try:
                result = compile_command(source)
                if result is not None:
                    # Complete statement - execute with tool handling
                    # Suppress REPL event display during direct REPL mode
                    self._in_user_repl = True
                    try:
                        output, _, _, _ = self._execute_with_tool_handling(repl, source)
                    except KeyboardInterrupt:
                        self._in_user_repl = False
                        print()
                        buffer = []
                        continue
                    finally:
                        self._in_user_repl = False
                    processed = self.process_repl_output(output)
                    # Strip echo for display (user already typed it)
                    display_lines = []
                    for ln in processed.split('\n'):
                        if not ln.startswith('>>> ') and not ln.startswith('... '):
                            display_lines.append(ln)
                    display = '\n'.join(display_lines).strip()
                    if display:
                        print(f"\x1b[92m{display}\x1b[0m")
                    transcript.append(processed)
                    if source.strip():
                        repl_history.append(source)
                    buffer = []
                # else: incomplete, continue accumulating
            except SyntaxError as e:
                print(f"\x1b[91mSyntaxError: {e}\x1b[0m")
                buffer = []

        if transcript:
            # Strip trailing newlines from each entry to avoid double spacing
            cleaned = [t.rstrip('\n') for t in transcript]
            self.usermsg("##### USER REPL SESSION #####\n" + "\n".join(cleaned) + "\n##### END SESSION #####")

        return bool(transcript)

    def save_session(self, filename: str):
        raise NotImplementedError("Session export has been removed; use /resume for persisted sessions.")

    def load_session(self, filename: str):
        raise NotImplementedError("Session import has been removed; use /resume for persisted sessions.")

    def resume_session(self, session_id: str):
        session = self._session_store.get_session(session_id)
        if not session:
            print(f"{DIM}Session not found: {session_id}{RESET}")
            return False
        session_host = session.get("host") or "local"
        if session_host != self.session_host():
            print(f"{DIM}Session {session_id} belongs to host {session_host}; current worker host is {self.session_host()}.{RESET}")
            return False
        if not self._acquire_session_lock(session_id):
            return False
        old_session_id = getattr(self, "_session_id", None)
        resumed = False

        if hasattr(self, '_conversation'):
            del self._conversation
        if hasattr(self, '_tool_repl'):
            self._tool_repl.close()
            del self._tool_repl
        if hasattr(self, '_repl_startup_injected'):
            del self._repl_startup_injected
        if hasattr(self, '_worker_attachment_helpers_injected'):
            del self._worker_attachment_helpers_injected
        _ = self.conversation
        self._suspend_persistence = True
        try:
            missing = replay_session_into_agent(self, session_id, self._session_store)
            self._session_id = session_id
            self._next_event_seq = self._session_store.get_next_seq(session_id)
            self._explicit_attachment_refs = {}
            self._pending_explicit_attachment_refs = {}
            for msg in self.conversation.messages:
                for name, ref in (msg.get('_attachment_refs') or {}).items():
                    self._explicit_attachment_refs[name] = ref
            missing.extend(self._materialize_replayed_attachment_refs())
            self._replay_display_output()
            self._coalesce_context(protect_last_interactions=False)
            self.usermsg(">>> system_reset()\nREPL session has been reset\n")
            self._coalesce_context()
            if missing:
                lines = ["[Resume warning: attachment file missing and detached]"]
                lines.extend(f"- {name}: {path}" for name, path in missing)
                self.usermsg("\n".join(lines))
            resumed = True
        finally:
            self._suspend_persistence = False
            if not resumed and old_session_id != session_id:
                self._release_session_lock(session_id)
        if old_session_id and old_session_id != session_id:
            self._release_session_lock(old_session_id)

        print(f"{DIM}Session resumed: {session_id}{RESET}")
        return True

    @staticmethod
    def _exec_prompt_text(content: str) -> str:
        text = (content or "").strip()
        try:
            import ast
            from code_agent.session_replay import _silence_parse_warnings
            with _silence_parse_warnings():
                tree = ast.parse(text)
        except SyntaxError:
            return text
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            return text
        call = tree.body[0].value
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "emit"
            or not call.args
        ):
            return text
        try:
            value = ast.literal_eval(call.args[0])
        except Exception:
            return text
        return str(value).strip()

    def _generate_exec_prompt(self, extra_instructions: str = "") -> str:
        instruction = """Write a continuation prompt for a fresh Code Agent session.

The prompt will replace the first user message after rewinding this session to the beginning.
Include all context needed to continue from a new session:
- current task or pending question
- relevant project/files/functions inspected
- decisions and constraints
- changes already made, if any
- commands/tests run and results
- unresolved issues and next steps
- attachments, previews, or files that should be reattached or viewed again
- any connection parameters, credential names, environment details, or operational constraints that are necessary and safe to include

Also include kickoff instructions for the next agent. If no task is pending, instruct it to ask the user what to do next.

Return only the replacement user prompt text.
"""
        if extra_instructions:
            instruction += f"\nAdditional user instructions for the continuation prompt:\n{extra_instructions}\n"

        old_ephemeral = self.ephemeral
        try:
            self.ephemeral = ""
            msg = self.llm_client.text_call(
                self.conversation._messages() + [{"role": "user", "content": instruction}]
            )
        finally:
            self.ephemeral = old_ephemeral
        return self._exec_prompt_text(msg.get("content") or "")

    def _restore_auto_context_attachments(self):
        names = list(getattr(self, "_auto_context_attachment_names", set()))
        self._explicit_attachment_refs = {}
        self._pending_explicit_attachment_refs = {}
        self._pending_attachments = {}
        self._expanded_preview_refs.clear()
        for name in names:
            try:
                self.attach_file_ref(name, name)
            except Exception as e:
                self._display_text(f"{DIM}Error reattaching {name}: {e}{RESET}", kind="status")

    def _quiet_replay_session(self):
        """Re-replay the persisted session into this agent without UI noise."""
        if not self._session_id:
            return
        if hasattr(self, '_conversation'):
            del self._conversation
        _ = self.conversation
        self._suspend_persistence = True
        try:
            replay_session_into_agent(self, self._session_id, self._session_store)
            self._next_event_seq = self._session_store.get_next_seq(self._session_id)
            self._explicit_attachment_refs = {}
            self._pending_explicit_attachment_refs = {}
            for msg in self.conversation.messages:
                for name, ref in (msg.get('_attachment_refs') or {}).items():
                    self._explicit_attachment_refs[name] = ref
            self._materialize_replayed_attachment_refs()
            self._coalesce_context()
        finally:
            self._suspend_persistence = False

    def _synthetic_exchange(self):
        self._ensure_setup()
        repl = self._get_tool_repl()
        repl._inject_code("from datetime import date")
        today = __import__('datetime').date.today().isoformat()
        parsed_today = __import__('datetime').date.fromisoformat(today) if today else None
        formatted_today = (
            f"{parsed_today.strftime('%A, %B')} {parsed_today.day}, {parsed_today.year}"
            if parsed_today else today
        )
        assistant_probe = 'from datetime import date\nemit("Checking today\'s date...")\nprint(date.today().isoformat())'
        assistant_emit = f'emit("Today is {formatted_today}.", release=True)'
        user_probe = f"What's today's date?\n"
        user_probe_output = (
            user_probe
            + '>>> emit("Checking today\'s date...")\n'
            + "Checking today's date...\n"
            + f">>> print(date.today().isoformat())\n"
            + f"{today}\n"
        )
        user_emit_output = (
            f">>> {assistant_emit}\n"
            f"Today is {formatted_today}.\n"
        )
        for role, content in (
            ('user', user_probe),
            ('assistant', assistant_probe),
            ('user', user_probe_output),
            ('assistant', assistant_emit),
            ('user', user_emit_output),
        ):
            self.conversation.messages.append({"role": role, "content": content, "_synthetic": True})
        self.conversation.messages[-1]["_stdout"] = user_emit_output
        self.conversation.messages[-1]["_render_segments"] = [
            {"type": "stdout", "content": user_emit_output}
        ]
        self._last_was_repl_output = True

    def cli_run(
        self,
        max_turns: int | None = None,
        resume: str | bool = False,
        initial_prompt: str | None = None,
    ):
        """Run CLI loop with Python block delimiters."""
        from code_agent.cli.mixin import SQLiteHistory, InputSession
        from code_agent.cli.altmode import AltMode

        self._ensure_setup()

        if max_turns is None:
            max_turns = getattr(self, 'max_turns', 10)

        # Set up stdout capture for alt-buffer replay
        altmode = AltMode()
        altmode.install()
        self.altmode = altmode  # Make available to user_repl_session

        # Set up history
        history_path = getattr(self, 'history_db', None)
        history = SQLiteHistory(history_path)
        session = InputSession(history, altmode=altmode)

        # Display welcome banner with model info
        welcome = getattr(self, 'welcome_message', '')
        if welcome:
            from code_agent.llm_registry import resolve_model_name
            full_model_name = resolve_model_name(self.model)
            banner_lines = [welcome, f"[dim]{full_model_name}[/dim]"]
            if self.session_host() != "local":
                banner_lines.append(f"[dim]{self.session_host()}[/dim]")
            banner = "\n".join(banner_lines)
            self.console.print(Panel.fit(banner, border_style="cyan"))


        prompt_str = getattr(self, 'cli_prompt', '> ')
        thinking = getattr(self, 'thinking_message', 'Thinking...')

        key_help = "Enter = submit | Alt+Enter = newline | Ctrl+O = transcript | Ctrl+C = interrupt | Ctrl+D = quit"
        if not self.agent_mode:
            key_help = "Enter = submit | Alt+Enter = newline | Ctrl+O = transcript | Esc Esc = rewind | Ctrl+C = interrupt | Ctrl+D = quit"
        if len(getattr(self, "model_choices", [])) > 1:
            key_help = f"Tab/Shift+Tab = model | {key_help}"
        self.console.print(f"[dim]{key_help}[/dim]")
        commands = ["/repl"]
        if not self.agent_mode:
            commands.append("/rewind")
        commands.extend(["/exec [instructions]", "/resume [session_id]"])
        if not self.agent_mode:
            commands.append("/fork [session_id]")
        commands.extend(["/skills [name]", "/subagents [model]", "/attach <file>", "/detach <file>", "/attachments", "/model [name]", "/tokens"])
        startup_commands = f"Commands: {', '.join(commands)}"
        startup_help = f"[dim]{startup_commands}[/dim]"
        self.console.print(startup_help)

        resumed_on_start = False
        if resume:
            if resume is True:
                from code_agent.cli.sessions import select_session_ui
                selection = select_session_ui(altmode, self._session_store, self.worker_cwd(), host=self.session_host())
                session_id = self._resolve_session_selection(selection)
            else:
                session_id = resume
            if session_id:
                resumed_on_start = self.resume_session(session_id)


        if not resumed_on_start:
            if files := self.gather_auto_attach_files():
                self._display_text(f"Loading {', '.join(files)}", kind="status", create_session=False)
                self._auto_context_attachment_names.update(files)
                for filename in files:
                    self.attach_file_ref(filename, filename)

        if startup_attachments := getattr(self, "startup_attachments", None):
            for filename in startup_attachments:
                try:
                    size_kb = self.attach_file_ref(filename, filename) / 1000
                    self._display_text(f"{DIM}Attached {filename} ({size_kb:.1f}KB){RESET}", kind="status", create_session=False)
                except Exception as e:
                    self._display_text(f"{DIM}Error attaching {filename}: {e}{RESET}", kind="status", create_session=False)

        synth = not resumed_on_start

        try:
            preload_input = ""
            pending_initial_prompt = initial_prompt
            user_header_pending = False
            flush_input_before_prompt = False
            while True:
                rewind_shortcut = False

                if flush_input_before_prompt:
                    try:
                        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
                    except (OSError, termios.error):
                        pass
                    flush_input_before_prompt = False

                def open_transcript(_buffer: str, _cursor: int):
                    from code_agent.cli.transcript import transcript_viewer_ui
                    self._ensure_live_session()
                    self._flush_pending_session_events()
                    events = self._session_store.get_transcript_events(self._session_id) if self._session_id else []
                    transcript_viewer_ui(altmode, events)

                def trigger_rewind():
                    nonlocal rewind_shortcut
                    rewind_shortcut = True
                    return "/rewind"

                def accepted_user_prefix(line: str) -> str | None:
                    if self.agent_mode or not self.response_formatting:
                        return None
                    if not user_header_pending:
                        return None
                    stripped = line.strip()
                    if not stripped or stripped.startswith("/"):
                        return None
                    return self._section_header("User")

                if pending_initial_prompt is not None:
                    user_input = pending_initial_prompt
                    pending_initial_prompt = None
                    lines = user_input.splitlines() or [""]
                    print(f"{prompt_str}{lines[0]}")
                    for line in lines[1:]:
                        print(line)
                else:
                    try:
                        user_input = session.prompt(
                            prompt_str,
                            initial_text=preload_input,
                            on_ctrl_o=open_transcript,
                            on_esc_esc=None if self.agent_mode else trigger_rewind,
                            on_tab=self._cycle_model,
                            on_shift_tab=self._cycle_model_reverse,
                            accepted_prefix=accepted_user_prefix,
                        )
                    except KeyboardInterrupt:
                        print()
                        preload_input = ""
                        continue
                    except EOFError:
                        if not self._run_pre_exit_hooks():
                            self.console.print("[yellow]Returning to prompt. Try Ctrl+D again to exit.[/yellow]")
                            continue
                        break
                preload_input = ""

                if not user_input.strip():
                    continue

                self._ensure_live_session()
                self._flush_pending_session_events()
                if user_input.strip() == "/repl":
                    self._display_input_block(user_input, include_header=False)
                    try:
                        if self.user_repl_session(history):
                            self._coalesce_context()
                    except Exception as e:
                        print(f"\n{DIM}Error: {type(e).__name__}: {e}{RESET}", file=sys.stderr)
                    continue

                if user_input.strip() == "/rewind" and not self.agent_mode:
                    if not rewind_shortcut:
                        self._display_input_block(user_input, include_header=False)
                    from code_agent.cli.rewind import rewind_ui
                    self._ensure_live_session()
                    self._flush_pending_session_events()
                    events = self._session_store.get_events(self._session_id) if self._session_id else []
                    rewind_result = rewind_ui(altmode, events)
                    if rewind_result is not None:
                        if rewind_shortcut:
                            sys.stdout.write("\x1b[1A\r\x1b[K")
                            sys.stdout.flush()
                        target_seq = rewind_result.get("target_seq")
                        if target_seq is not None:
                            self._append_session_event("rewind", {"target_seq": target_seq})
                        self._quiet_replay_session()
                        self._replay_display_output()
                        self._display_text(f"{DIM}Conversation rewound.{RESET}", kind="status")
                        last = self.conversation.messages[-1] if self.conversation.messages else None
                        self._last_was_repl_output = bool(last and last.get('role') == 'user')
                        preload_input = rewind_result.get("preload_input", "") or ""
                    elif rewind_shortcut:
                        sys.stdout.write("\x1b[1A\r\x1b[K")
                        sys.stdout.flush()
                    continue

                if user_input.strip() == "/exec" or user_input.strip().startswith("/exec "):
                    self._display_input_block(user_input, include_header=False)
                    extra_instructions = user_input.strip()[5:].strip()
                    try:
                        continuation_prompt = self._generate_exec_prompt(extra_instructions)
                    except Exception as e:
                        self._display_text(f"{DIM}Error generating continuation prompt: {type(e).__name__}: {e}{RESET}", kind="status")
                        continue
                    if not continuation_prompt:
                        self._display_text(f"{DIM}No continuation prompt generated.{RESET}", kind="status")
                        continue
                    self._append_session_event("exec", {})
                    self._quiet_replay_session()
                    self._restore_auto_context_attachments()
                    self._replay_display_output()
                    self._display_text(f"{DIM}Session reset. Edit the continuation prompt, then press Enter.{RESET}", kind="status")
                    preload_input = continuation_prompt
                    synth = False
                    user_header_pending = False
                    self._last_was_repl_output = False
                    continue

                if user_input.strip().startswith("/resume"):
                    self._display_input_block(user_input, include_header=False)
                    resumed = False
                    parts = user_input.strip().split(None, 1)
                    if len(parts) == 1:
                        from code_agent.cli.sessions import select_session_ui
                        selection = select_session_ui(altmode, self._session_store, self.worker_cwd(), host=self.session_host())
                        session_id = self._resolve_session_selection(selection)
                        if session_id:
                            resumed = self.resume_session(session_id)

                    else:
                        resumed = self.resume_session(parts[1].strip())
                    if resumed:
                        synth = False
                    continue

                if user_input.strip().startswith("/fork") and not self.agent_mode:
                    self._display_input_block(user_input, include_header=False)
                    parts = user_input.strip().split(None, 1)
                    source_id = parts[1].strip() if len(parts) > 1 else getattr(self, "_session_id", None)
                    if not source_id:
                        self._display_text(f"{DIM}No active session to fork{RESET}", kind="status")
                        continue
                    forked_id = self.fork_session(source_id)
                    if forked_id:
                        resumed = self.resume_session(forked_id)
                        if resumed:
                            synth = False
                    continue


                if user_input.strip().startswith("/skills"):
                    self._display_input_block(user_input, include_header=False)
                    parts = user_input.strip().split(None, 1)
                    if len(parts) == 1:
                        if self.agent_mode:
                            self._display_text(self.format_skills_list(), kind="status")
                        else:
                            from code_agent.cli.skills import select_skills_ui
                            skill_items = self.list_skills()
                            result = select_skills_ui(altmode, skill_items)
                            if result is not None:
                                changes = self.apply_skill_selection(result)
                                if changes:
                                    for line in changes:
                                        self._display_text(f"{DIM}{line}{RESET}", kind="status")
                                else:
                                    self._display_text(f"{DIM}No skill changes{RESET}", kind="status")
                    else:
                        ok, msg = self.attach_skill(parts[1].strip())
                        self._display_text(f"{DIM}{msg}{RESET}", kind="status")
                    continue

                if user_input.strip().startswith("/attach "):
                    self._display_input_block(user_input, include_header=False)
                    filename = user_input.strip()[8:].strip()
                    if filename:
                        try:
                            size_kb = self.attach_file_ref(filename, filename) / 1000
                            self._display_text(f"{DIM}Attached {filename} ({size_kb:.1f}KB){RESET}", kind="status")
                        except Exception as e:
                            self._display_text(f"{DIM}Error attaching {filename}: {e}{RESET}", kind="status")
                    continue

                if user_input.strip().startswith("/detach "):
                    self._display_input_block(user_input, include_header=False)
                    filename = user_input.strip()[8:].strip()
                    if filename:
                        if is_preview_uri(filename):
                            self._expanded_preview_refs.pop(filename, None)
                            self._append_session_event("preview_collapsed", {"uri": filename})
                            if filename in self.list_attachments():
                                self.detach(filename)
                            self._display_text(f"{DIM}Collapsed preview: {filename}{RESET}", kind="status")
                        else:
                            self.detach_file_ref(filename)
                            self._display_text(f"{DIM}Detached {filename}{RESET}", kind="status")
                    continue


                if user_input.strip() == "/attachments":
                    self._display_input_block(user_input, include_header=False)
                    attachments = self.list_attachments()
                    expanded = self._expanded_preview_context()
                    if not attachments and not expanded:
                        self._display_text(f"{DIM}No attachments/context{RESET}", kind="status")
                    else:
                        self._display_text(f"{DIM}Attachments/context:{RESET}", kind="status")
                        for name, content in attachments.items():
                            size_kb = len(content) / 1000
                            self._display_text(f"{DIM}  {name} ({size_kb:.1f}KB){RESET}", kind="status")
                        for name, content in expanded.items():
                            size_kb = len(content) / 1000
                            line_count = len(content.split("\n"))
                            self._display_text(f"{DIM}  {name} (expanded preview, {size_kb:.1f}KB, {line_count} lines){RESET}", kind="status")
                    continue

                if user_input.strip().startswith("/model"):
                    self._display_input_block(user_input, include_header=False)
                    parts = user_input.strip().split(None, 1)
                    if len(parts) == 1:
                        from code_agent.llm_registry import list_models, resolve_model_name
                        from code_agent.cli.models import select_model_ui
                        selected_model = select_model_ui(altmode, list_models(), resolve_model_name(self.model))
                        if selected_model is None:
                            continue
                        new_model = selected_model
                    else:
                        new_model = parts[1].strip()

                    old_model = self.model
                    try:
                        # Validate the model exists by trying to get its config
                        from code_agent.llm_registry import get_model_config, resolve_model_name
                        get_model_config(new_model)
                        new_model = resolve_model_name(new_model)
                        if new_model == old_model:
                            self._display_text(f"{DIM}Current model: {old_model}{RESET}", kind="status")
                            continue
                        self._set_model(new_model)
                        self._display_text(f"{DIM}Model changed: {new_model}{RESET}", kind="status")
                    except ModelNotFoundError as e:
                        self._display_text(f"{DIM}{str(e)}{RESET}", kind="status")
                    continue

                if user_input.strip() == "/tokens":
                    self._display_input_block(user_input, include_header=False)
                    tracker = self.llm_client.usage_tracker
                    if not tracker.history:
                        self._display_text(f"{DIM}No API calls yet{RESET}", kind="status")
                    else:
                        n = tracker._normalize(*tracker.history[-1])
                        total = n['prompt_tokens'] + n['cached_tokens'] + n['completion_tokens'] + n['reasoning_tokens']
                        parts = [p for p in [
                            f"{n['prompt_tokens']:,} in" if n['prompt_tokens'] else None,
                            f"{n['cached_tokens']:,} cached" if n['cached_tokens'] else None,
                            f"{n['reasoning_tokens']:,} reasoning" if n['reasoning_tokens'] else None,
                            f"{n['completion_tokens']:,} out" if n['completion_tokens'] else None,
                        ] if p]
                        self._display_text(f"{DIM}[Last request: {total:,} tokens ({', '.join(parts)})]{RESET}", kind="status")
                    continue

                if user_input.strip().startswith("/subagents"):
                    self._display_input_block(user_input, include_header=False)
                    try:
                        # Import subagent module into REPL and show docstring to agent
                        # Optional model parameter: /subagents [model]
                        parts = user_input.strip().split(None, 1)
                        if len(parts) > 1:
                            # Model specified administratively
                            from code_agent.llm_registry import get_model_config, resolve_model_name
                            subagent_model = parts[1].strip()
                            get_model_config(subagent_model)
                            subagent_model = resolve_model_name(subagent_model)
                            model_locked = True
                        else:
                            # Inherit parent's model
                            subagent_model = self.model
                            model_locked = False

                        repl = self._get_tool_repl()
                        # Import and set default model (silent injection)
                        repl._inject_code(f"from code_agent.subagent import Subagent, SubagentError, SubagentResponse, _subagents; Subagent.default_model = {repr(subagent_model)}")

                        # Only show docstring on first load; subsequent calls just update model
                        already_loaded = getattr(self, '_subagents_loaded', False)
                        if already_loaded:
                            self._display_text(f"{DIM}Subagent default model changed to: {subagent_model}{RESET}", kind="status")
                        else:
                            self._subagents_loaded = True
                            # Build docstring, optionally hiding model config section
                            from code_agent import subagent
                            docstring = subagent.__doc__
                            if model_locked:
                                # Strip "## Model Configuration" section so agent doesn't try to override
                                import re
                                docstring = re.sub(r'## Model Configuration\n.*?(?=\n## |\n"""|\Z)', '', docstring, flags=re.DOTALL)
                            self.usermsg(f">>> # Subagent module loaded (model: {subagent_model})\n{docstring}")
                            self._display_text(f"{DIM}Subagent module loaded into REPL (model: {subagent_model}){RESET}", kind="status")
                    except Exception as e:
                        print(f"\n{DIM}Error: {type(e).__name__}: {e}{RESET}", file=sys.stderr)
                    continue

                if user_input.lstrip().startswith("/"):
                    self._display_input_block(user_input, include_header=False)
                    self.console.print(startup_help)
                    self._record_display_event("status", startup_commands + "\n")
                    continue

                self._display_input_block(user_input, include_header=user_header_pending)
                user_header_pending = False

                if synth:
                    try:
                        self._synthetic_exchange()
                    except Exception as e:
                        print(f"\n{DIM}Error: {type(e).__name__}: {e}{RESET}", file=sys.stderr)
                    synth = False

                self.usermsg(user_input, _user_content=user_input)

                # Reset state for new user interaction
                self._repl_printed_header = False
                self._repl_has_output = False
                self._turn_number = 1
                self._turn_output_started = False
                self._header_pending = False
                self._statement_direct_call = None
                self._statement_source = ""
                self._statement_echo = ""
                self._statement_echo_displayed = False
                self._statement_had_diff = False
                self._statement_print_uses_variable = False
                self._reset_display_capture()
                if self.repl_display:
                    print()  # Blank line after user input
                    print(f"{DIM}{thinking} (turn 1){RESET}", end="", flush=True)
                elif self.agent_mode:
                    print()  # Keep agent-mode output from overwriting the submitted prompt line

                flush_input_before_prompt = True
                try:
                    response = self.run_loop(max_turns=max_turns)
                except KeyboardInterrupt:
                    self.console.clear_line()
                    print()
                    continue
                except ContextOverflowError as e:
                    self.console.clear_line()
                    print(f"\n{DIM}Warning: {e}{RESET}", file=sys.stderr)
                    continue
                except Exception as e:
                    self.console.clear_line()
                    print(f"\n{DIM}Error: {type(e).__name__}: {e}{RESET}", file=sys.stderr)
                    continue

                self._coalesce_context()

                if self.repl_display:
                    self.console.clear_line()  # Clear thinking message

                # The next section header delineates the end of Python output.

                # Display response
                response_str = str(response) if response is not None else ""
                formatted = self.format_response(response_str) if self.response_formatting else response_str
                if formatted:
                    user_header_pending = bool(self.response_formatting) and not self.agent_mode
                    if self.response_formatting:
                        output_header = self._section_header("Output", "═", TEXT)
                        print(output_header)
                    print(formatted)
        finally:
            altmode.uninstall()
            # Save conversation on crash
            if sys.exc_info()[1] is not None:
                import tempfile
                crash_file = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.json', prefix='repl_crash_', delete=False
                )
                json.dump(self.conversation._messages(), crash_file, indent=2)
                crash_file.close()
                print(f"\n*** Conversation saved to: {crash_file.name} ***", file=sys.stderr)
            # Clean up temp files from truncated output
            for path in getattr(self, '_temp_files', []):
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self.console.print("\n[dim]Session ended. Goodbye![/dim]")
            session_id = getattr(self, "_session_id", None)
            if session_id:
                self._release_session_lock(session_id)
                self.console.print(f"[dim]Resume session: {self.resume_session_command(session_id)}[/dim]")


class CodeAgent(MCPMixin, CodeAgentBase):
    """Code agent with REPL-proxied tools."""

    def _last_pinnable_turn(self):
        for msg in reversed(self.conversation.messages):
            if msg.get("role") != "assistant" or msg.get("_synthetic"):
                continue
            return msg
        return None

    @REPLAgent.tool
    def pin(self):
        """Pin the immediately previous completed assistant turn for preview expansion after coalescing."""
        turn = self._last_pinnable_turn()
        if turn is None:
            return "No previous turn to pin."
        turn["_pinned_coalesce"] = {"label": "Pinned previous turn"}
        event_seq = turn.get("_event_seq")
        if event_seq is not None:
            old_suspend = getattr(self, "_suspend_persistence", False)
            try:
                self._suspend_persistence = True
                self._ensure_live_session()
            finally:
                self._suspend_persistence = old_suspend
            self._append_session_event(
                "message_pinned",
                {"message_event_seq": event_seq, "label": "Pinned previous turn"},
            )
        else:
            self._persist_message(turn)
        return "Pinned previous turn for coalescing."

    mcp_servers = []

    _preview_counter = 0  # Kept for backwards-compatible instance state.


    @REPLAgent.tool
    def observe(self, content: str = "Reflection on previous substantive work"):
        """Record a reflective observation about previous work."""
        text = str(content)
        if not text.strip():
            raise ValueError("Observation content must not be empty.")
        self._pending_observations.append(text)
        return "[Continuing...]"

    @REPLAgent.tool(inject=True)
    def think(self, content: str = "All relevant observations and reasoning"):
        """Think through the problem and yield to a new turn.

        Call this when you're uncertain how to proceed or need to reason
        through a problem. Write down your observations, hypotheses,
        open questions, and options you're considering.
        """
        return "[Continuing...]"

    # preview() is intentionally not exposed: large output is auto-previewed.



    def preprocess_code(self, code: str) -> str:
        """Apply base preprocessing and CodeAgent-specific file/context rewrites."""

        if getattr(self, '_in_user_repl', False):
            return code

        code = super().preprocess_code(code)

        code, self._preview_counter = preprocess_code_agent(
            code,
            preview_counter=getattr(self, '_preview_counter', 0),
            preview_origins=getattr(self, '_preview_full_file_origins', {}),
        )
        return code

    def _handle_tool_request(self, repl, req: dict) -> None:
        tool_name = req.get('tool')
        if tool_name in {'__preview_blob_save__', '__preview_blob_read__', '__line_patch_is_attached__', '__preview_ref_expand__', '__preview_ref_collapse__', '__file_diffs__'}:

            request_id = req.get('request_id')
            args = req.get('args', {})
            try:
                if tool_name == '__file_diffs__':
                    file_path = args.get("file_path")
                    limit = args.get("limit")
                    repl.send_reply(request_id, result=self._format_file_diff_events(file_path, limit))
                    return
                if tool_name == '__preview_ref_expand__':
                    uri = args.get('uri')
                    if uri:
                        content = self._preview_blob_content(uri)
                        if content is not None:
                            error = self._expanded_preview_context_error(uri, content, bool(args.get("numbered", False)))
                            if error is not None:
                                repl.send_reply(request_id, error=error)
                                return
                        self._expanded_preview_refs[uri] = {"numbered": bool(args.get("numbered", False))}
                        self._append_session_event("preview_expanded", {"uri": uri, "numbered": bool(args.get("numbered", False))})
                    repl.send_reply(request_id, result=True)
                    return
                if tool_name == '__preview_ref_collapse__':
                    uri = args.get('uri')
                    if uri:
                        self._expanded_preview_refs.pop(uri, None)
                        self._append_session_event("preview_collapsed", {"uri": uri})
                    repl.send_reply(request_id, result=True)
                    return
                if tool_name == '__line_patch_is_attached__':
                    repl.send_reply(request_id, result=self._is_attached(args.get('path')))
                    return
                if getattr(self, '_session_id', None) is None:
                    self._ensure_live_session()
                    self._flush_pending_session_events()
                if tool_name == '__preview_blob_save__':
                    origin_path = args.get('origin_path')
                    if origin_path is not None:
                        self._preview_full_file_origins = getattr(self, '_preview_full_file_origins', {})
                        self._preview_full_file_origins[args.get('key')] = origin_path
                    self._session_store.save_preview_blob(self._session_id, args.get('key'), args.get('content', ''))
                    repl.send_reply(request_id, result=True)
                else:
                    repl.send_reply(request_id, result=self._session_store.get_preview_blob(self._session_id, args.get('key')))
            except Exception as e:
                repl.send_reply(request_id, error=str(e))
            finally:
                repl.send_ack(request_id)
            return
        return super()._handle_tool_request(repl, req)


    @REPLAgent.tool(inject=True)
    def grep(self,
            pattern: str = "Regex pattern to search for",
            path: Optional[str] = "File or directory to search in",
            glob: Optional[str] = "Glob pattern to filter files (e.g., '*.js')",
            file_type: Optional[str] = "File type to search (e.g., 'py', 'js', 'rust')",
            files_only: Optional[bool] = "Only return filenames, not matching lines",
            context: Optional[int] = "Lines of context around matches (-C)",
            case_insensitive: Optional[bool] = "Case insensitive search (-i)",
            multiline: Optional[bool] = "Enable multiline matching (-U)"
        ):
        """Search file contents with ripgrep."""
        import subprocess
        cmd = ['rg', '--color=never', '-n']  # Always show line numbers

        default_excludes = [
            '.git', '.hg', '.svn', '.venv', 'venv', 'env', 'node_modules',
            '__pycache__', '.mypy_cache', '.pytest_cache', '.ruff_cache',
            '.tox', '.nox', '.cache', 'dist', 'build', 'coverage', '.coverage',
        ]
        default_file_excludes = [
            '*.min.js', '*.min.css', '*.map', '*.pyc', '*.pyo',
            '*.db', '*.db-*', '*.sqlite', '*.sqlite-*',
            '*.sqlite3', '*.sqlite3-*', '*.log',
        ]
        for directory in default_excludes:
            cmd.extend(['--glob', f'!**/{directory}/**'])
        for file_glob in default_file_excludes:
            cmd.extend(['--glob', f'!{file_glob}'])

        if files_only:
            cmd.append('-l')
        if case_insensitive:
            cmd.append('-i')
        if multiline:
            cmd.append('-U')
        if context:
            cmd.extend(['-C', str(context)])
        if glob:
            cmd.extend(['--glob', glob])
        if file_type:
            cmd.extend(['--type', file_type])

        cmd.append(pattern)
        if path:
            cmd.append(path)

        max_output_bytes = 2 * 1024 * 1024
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        raw_output = process.stdout.read(max_output_bytes + 1)
        if len(raw_output) > max_output_bytes:
            process.kill()
            process.wait()
            raise ValueError(
                "grep output exceeded 2 MiB; narrow the path or pattern."
            )
        process.wait()
        output = raw_output.decode(errors="replace").strip()

        if not output:
            return "No matches found"
        if files_only:
            return output.split('\n')
        return output

    @REPLAgent.tool(inject=True)
    def read(self,
            file_path: str = "Path to the file",
            offset: Optional[int] = "Line number to start from (1-indexed)",
            limit: Optional[int] = "Number of lines to read (default: 5000)"
        ):
        """Read a file or session://preview/... URI and return its contents as text.

        Use read() when you want contents as a Python value:
            content = read("file.py")
            lines = read("file.py").splitlines()
            snippet = read("file.py", offset=100, limit=20)

        Path-like objects are accepted for filesystem paths:
            content = read(Path("file.py"))

        Long preview() output is saved to a session://preview/... URI:
            full_output = read("session://preview/abc123")

        Use view() when you want numbered file output:
            view("file.py")
            view("file.py", offset=100, limit=20)
            view("session://preview/abc123", offset=100, limit=20)
        """
        import os
        if not isinstance(file_path, str):
            file_path = os.fspath(file_path)
        prefix = "session://preview/"

        if isinstance(file_path, str) and file_path.startswith(prefix):
            key = file_path[len(prefix):]
            import json as _json
            global _request_id
            _request_id += 1
            _req_id = _request_id
            _send_tool_request(_json.dumps({
                "tool": "__preview_blob_read__",
                "args": {"key": key},
                "request_id": _req_id,
            }))
            content = _wait_for_ack(_req_id)
            if content is None:
                raise FileNotFoundError(file_path)
        else:
            content = Path(file_path).expanduser().read_text()
        if offset is None and limit is None:
            if not isinstance(file_path, str) or not file_path.startswith("session://"):
                _FullFileRead = globals().get("_FullFileRead")
                if _FullFileRead is None:
                    class _FullFileRead(str):
                        def __new__(cls, value, path):
                            obj = str.__new__(cls, value)
                            obj._code_agent_full_file_read_path = path
                            return obj
                    globals()["_FullFileRead"] = _FullFileRead
                return _FullFileRead(content, file_path)
            return content
        all_lines = content.split('\n')
        start = (offset or 1) - 1
        end = start + (limit or 5000)
        return '\n'.join(all_lines[start:end])

    @REPLAgent.tool(inject=True)
    def view(self,
            file_path: str = "Path to the file",
            offset: Optional[int] = "Line number to start from (1-indexed)",
            limit: Optional[int] = "Number of lines to read (default: 5000)",
            numbered: Optional[bool] = "Number output lines. Defaults true for files, false for full preview URIs."
        ):

        """Display a file or session://preview/... URI with line numbers.

        Use view() for inspection with numbered lines:
            view("file.py")
            view("file.py", offset=100, limit=20)
            view("session://preview/abc123", offset=100, limit=20)

        Preview URI reads are for inspecting saved preview() output and are
        not filesystem paths.

        WRONG — view() is not a value:
            content = view("file.py")
            print(view("file.py"))
            preview(view("file.py"))

        Use read() if you need contents as text:
            content = read("file.py")
            full_output = read("session://preview/abc123")
        """
        import os
        if not isinstance(file_path, str):
            file_path = os.fspath(file_path)

        prefix = "session://preview/"
        is_preview_uri = isinstance(file_path, str) and file_path.startswith(prefix)
        if numbered is None:
            numbered = not is_preview_uri
        path = Path(file_path).expanduser()
        if not is_preview_uri and offset is None and limit is None:
            import json as _json
            global _request_id
            _request_id += 1
            _context_req_id = _request_id
            _send_tool_request(_json.dumps({
                "tool": "__line_patch_is_attached__",
                "args": {"path": file_path},
                "request_id": _context_req_id,
            }))
            if _wait_for_ack(_context_req_id):
                _send_output(
                    "output",
                    "\nNotice: file was already in context. Calling view() on files that are already in context is wasteful.\n\n",
                )
        if is_preview_uri:
            key = file_path[len(prefix):]
            import json as _json
            _request_id += 1
            _req_id = _request_id
            _send_tool_request(_json.dumps({
                "tool": "__preview_blob_read__",
                "args": {"key": key},
                "request_id": _req_id,
            }))
            content = _wait_for_ack(_req_id)
            if content is None:
                raise FileNotFoundError(file_path)
            if offset is None and limit is None:
                _request_id += 1
                _expand_req_id = _request_id
                _send_tool_request(_json.dumps({
                    "tool": "__preview_ref_expand__",
                    "args": {"uri": file_path, "numbered": bool(numbered)},
                    "request_id": _expand_req_id,
                }))
                _wait_for_ack(_expand_req_id)
                _send_output("preview_expand", f"Expanded preview: {file_path} (full content is now available in current context)\n")
                return
        else:
            content = path.read_text()

        all_lines = content.split('\n')
        total_lines = len(all_lines)
        is_partial = offset is not None or limit is not None
        source_extensions = {
            ".py", ".pyi", ".pyx", ".pxd", ".pxi",
            ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts",
            ".html", ".htm", ".xhtml", ".css", ".scss", ".sass", ".less",
            ".vue", ".svelte", ".astro",
            ".java", ".kt", ".kts", ".scala", ".groovy",
            ".c", ".h", ".cc", ".hh", ".cpp", ".cxx", ".hpp", ".hxx",
            ".cs", ".go", ".rs", ".swift", ".m", ".mm",
            ".php", ".rb", ".rake", ".pl", ".pm", ".t", ".lua",
            ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
            ".sql", ".graphql", ".gql",
            ".xml", ".xsl", ".xslt", ".svg",
            ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
            ".md", ".mdx", ".rst", ".tex",
            ".dockerfile", ".containerfile",
            ".vim", ".el", ".clj", ".cljs", ".cljc", ".ex", ".exs", ".erl", ".hrl",
            ".fs", ".fsx", ".fsi", ".hs", ".lhs", ".ml", ".mli", ".nim", ".zig",
            ".r", ".R", ".jl", ".dart", ".sol", ".tf", ".tfvars", ".hcl",
        }
        source_names = {
            "Dockerfile", "Containerfile", "Makefile", "Rakefile", "Gemfile",
            "Podfile", "Brewfile", "Justfile", "Taskfile", "Jenkinsfile",
            "BUILD", "WORKSPACE", "CMakeLists.txt",
        }
        if not is_preview_uri and is_partial and (path.suffix in source_extensions or path.name in source_names):
            _send_output(
                "output",
                "\nNotice: Direct or partial file reads bypass file inspection tools. Prefer full view(file_path) for inspection; use read(file_path) only when you need a Python string value.\n\n",
            )

        start = (offset or 1) - 1
        end = start + (limit or 5000)
        lines = all_lines[start:end]
        start_line = start + 1

        if numbered:
            formatted = [f"{start_line + i:>5}→{line}" for i, line in enumerate(lines)]
            output = '\n'.join(formatted)
        else:
            output = '\n'.join(lines)


        remaining = total_lines - end
        if remaining > 0 and limit is None:
            output += f"\n... ({remaining} more lines)"

        if offset is None and limit is None:
            if not is_preview_uri:
                import hashlib
                snapshots = globals().setdefault("_line_patch_snapshots", {})
                snapshots[file_path] = {


                    "path": file_path,
                    "resolved_path": str(path.resolve()),
                    "content": content,
                    "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "line_count": len(all_lines),
                    "line_patch_stale": False,
                }
            _send_output("read_attach", file_path + "\n")
        else:
            _send_output("read_partial", file_path + "\n")

        _send_output("read", output + "\n")

    @REPLAgent.tool(inject=True)
    def diff_history(self,
            file_path: Optional[str] = "Optional file path to filter diffs",
            limit: Optional[int] = "Maximum number of matching diff events to return"
        ):
        '''Review persisted file diffs from this session. Does not modify files.'''
        import json as _json
        global _request_id
        _request_id += 1
        _req_id = _request_id
        _send_tool_request(_json.dumps({
            "tool": "__file_diffs__",
            "args": {"file_path": file_path, "limit": limit},
            "request_id": _req_id,
        }))
        return _wait_for_ack(_req_id)

    @REPLAgent.tool
    def unview(self,
            file_path: str = "Path to a file or session://preview/... URI previously viewed with view()"
        ):

        """Remove a previously viewed attachment from future context.

        Use this if you viewed the wrong file or preview URI with view(), or no
        longer need it in context. This only affects future turns.
        """
        if is_preview_uri(file_path):
            self._expanded_preview_refs.pop(file_path, None)
            self._append_session_event("preview_collapsed", {"uri": file_path})
            if file_path in self.list_attachments():
                self.detach(file_path)
            self._pending_unviewed_files.add(file_path)
            return f"Collapsed preview: {file_path}"

        attachments = self.list_attachments()
        explicit_refs = getattr(self, '_explicit_attachment_refs', {})
        if file_path in explicit_refs:
            self.detach_file_ref(file_path)
        elif file_path in attachments:
            self.detach(file_path)
        globals().get("_line_patch_snapshots", {}).pop(file_path, None)
        self._pending_unviewed_files.add(file_path)
        return f"Removed from future context: {file_path}"


    @REPLAgent.tool(inject=True)
    def edit(self,
            file_path: str = "Path to the file",
            old_string: str = "Text to replace (must be unique unless replace_all)",
            new_string: str = "Replacement text",
            replace_all: Optional[bool] = "Replace all occurrences"
        ):
        """Edit a file by replacing text."""
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError("File does not exist.")
        if old_string == new_string:
            raise ValueError("No changes to make: old_string and new_string are exactly the same.")

        content = path.read_text()
        count = content.count(old_string)

        if count == 0:
            raise ValueError(f"String to replace not found in file.")
        if count > 1 and not replace_all:
            raise ValueError(f"Found {count} matches of the string to replace, but replace_all is false.")

        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)
        path.write_text(new_content)
        import hashlib
        snapshots = globals().get("_line_patch_snapshots", {})
        resolved_path = str(path.resolve())
        for key, snapshot in list(snapshots.items()):
            if key == file_path or snapshot.get("resolved_path") == resolved_path:
                snapshot.update({
                    "content": new_content,
                    "sha256": hashlib.sha256(new_content.encode()).hexdigest(),
                    "line_count": len(new_content.split('\n')),
                    "line_patch_stale": False,
                })
        globals().get("_line_patch_turn_state", {}).pop(resolved_path, None)
        import difflib
        diff = ''.join(difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=file_path,
            tofile=file_path,
        ))
        if diff:
            _send_output("file_diff", diff.rstrip('\n') + "\n")
        import json as _json
        _send_output("file_written", _json.dumps({
            "path": file_path,
            "content": new_content,
        }) + "\n")

        if replace_all and count > 1:
            return f"All {count} occurrences replaced."
        return "Edit applied."

    @REPLAgent.tool(inject=True)
    def line_patch(self,
            file_path: str = "Path to an existing file",
            op: str = "Operation: replace, delete, insert_before, or insert_after",
            start: str = "Anchor in the form '@LINE expected line content'",
            end_or_content: Optional[str] = "End anchor for replace/delete, or content for insert operations",
            content: Optional[str] = "Replacement content for replace operations"
        ):
        """Edit an existing file by line number with line-content anchors.

        Prefer a full view(file_path) first. If no current view snapshot exists,
        line_patch uses the file's current on-disk contents as the line-number
        baseline and attaches the edited file for future context.

        Each call performs one operation. Line anchors use the form
        "@LINE expected line content"; the expected content must match the target
        line after leading/trailing whitespace is stripped.

        Within one assistant turn, repeated line_patch() calls for the same file
        may continue to use line numbers from the file as it existed at the start
        of the turn. The tool tracks earlier same-turn line-count changes and
        translates later anchors to the current file before applying them.

        Operations:
          line_patch(path, "replace", "@START first line", "@END last line", new_content)
          line_patch(path, "delete", "@START first line", "@END last line")
          line_patch(path, "insert_before", "@LINE anchor line", new_content)
          line_patch(path, "insert_after", "@LINE anchor line", new_content)

        `insert_after` accepts @0 as a prepend anchor. `insert_before` accepts
        @LINE_COUNT+1 as an append anchor; the expected line content must be
        empty for that virtual EOF anchor.
        """
        import ast
        import hashlib
        import json as _json
        import os
        import re

        if isinstance(file_path, str) and file_path.startswith("session://"):
            raise ValueError("line_patch cannot edit session:// preview URIs.")

        if isinstance(op, str):
            legacy = re.match(r"^(replace|delete|insert_before|insert_after)\s+(\d+)(?::(\d+))?\n(.*)\Z", op, re.DOTALL)
            if legacy:
                legacy_op = legacy.group(1)
                legacy_start = int(legacy.group(2))
                legacy_end = int(legacy.group(3) or legacy.group(2))
                legacy_content = legacy.group(4)
                legacy_lines = Path(file_path).expanduser().read_text().split("\n")

                def _legacy_anchor(line):
                    if 1 <= line <= len(legacy_lines):
                        return f"@{line} {legacy_lines[line - 1]}"
                    return f"@{line}"

                op = legacy_op
                start = _legacy_anchor(legacy_start)
                if legacy_op in {"replace", "delete"}:
                    end_or_content = _legacy_anchor(legacy_end)
                    content = legacy_content if legacy_op == "replace" else None
                else:
                    end_or_content = legacy_content
                    content = None

        valid_ops = {"replace", "delete", "insert_before", "insert_after"}
        if op not in valid_ops:
            raise ValueError(f"Unsupported line_patch operation {op!r}. Expected one of: {', '.join(sorted(valid_ops))}.")

        def _parse_anchor(anchor, label):
            if not isinstance(anchor, str):
                raise TypeError(f"{label} anchor must be a string like '@12 expected content'.")
            match = re.match(r"^@(\d+)(?:\s(.*))?$", anchor)
            if not match:
                raise ValueError(f"{label} anchor must be in the form '@LINE expected line content'.")
            return int(match.group(1)), match.group(2) or ""

        def _line_matches(actual, expected):
            return actual.strip() == expected.strip()

        def _anchor_mismatch(path_text, original_line, current_line, expected, actual):
            return (
                f"Anchor mismatch in {path_text}\n"
                f"Expected original @{original_line}: {expected}\n"
                f"Translated current @{current_line}: {actual}"
            )

        def _content_lines(text):
            if text is None:
                return []
            if not isinstance(text, str):
                raise TypeError("line_patch content must be a string.")
            lines = text.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            return lines

        global _request_id
        _request_id += 1
        _req_id = _request_id
        _send_tool_request(_json.dumps({
            "tool": "__line_patch_is_attached__",
            "args": {"path": file_path},
            "request_id": _req_id,
        }))
        was_attached = bool(_wait_for_ack(_req_id))

        path = Path(file_path).expanduser()
        old_text = path.read_text()
        old_hash = hashlib.sha256(old_text.encode()).hexdigest()
        resolved_path = str(path.resolve())

        snapshots = globals().setdefault("_line_patch_snapshots", {})
        snapshot = snapshots.get(file_path)
        if snapshot is None:
            snapshot = {
                "path": file_path,
                "resolved_path": resolved_path,
                "content": old_text,
                "sha256": old_hash,
                "line_count": len(old_text.split("\n")),
                "line_patch_stale": False,
            }
            snapshots[file_path] = snapshot
            was_attached = False
        if snapshot.get("line_patch_stale"):
            raise ValueError(f"Call view({file_path!r}) before using line_patch().")
        if old_hash != snapshot.get("sha256"):
            if not was_attached:
                snapshot.update({
                    "content": old_text,
                    "sha256": old_hash,
                    "line_count": len(old_text.split("\n")),
                    "line_patch_stale": False,
                })
            else:
                raise ValueError(f"{file_path} changed on disk since it was viewed. Call view({file_path!r}) again before line_patch().")

        turn_states = globals().setdefault("_line_patch_turn_state", {})
        state = turn_states.get(resolved_path)
        if state is None:
            baseline_text = old_text
            state = {
                "path": file_path,
                "resolved_path": resolved_path,
                "content": baseline_text,
                "sha256": hashlib.sha256(baseline_text.encode()).hexdigest(),
                "line_count": len(baseline_text.split("\n")),
                "edits": [],
            }
            turn_states[resolved_path] = state
        else:
            baseline_text = state["content"]

        baseline_lines = baseline_text.split("\n")
        current_lines = old_text.split("\n")
        baseline_line_count = len(baseline_lines)
        current_line_count = len(current_lines)

        def _baseline_line(line):
            if line == 0:
                return ""
            if line == baseline_line_count + 1:
                return ""
            return baseline_lines[line - 1]

        def _current_line(line):
            if line == 0:
                return ""
            if line == current_line_count + 1:
                return ""
            return current_lines[line - 1]

        def _check_baseline_anchor(original_line, expected, label, allow_prepend=False, allow_append=False):
            valid = 1 <= original_line <= baseline_line_count
            if allow_prepend and original_line == 0:
                valid = True
            if allow_append and original_line == baseline_line_count + 1:
                valid = True
            if not valid:
                raise ValueError(f"{label} anchor @{original_line} is outside {file_path} with {baseline_line_count} baseline lines.")
            actual = _baseline_line(original_line)
            if not _line_matches(actual, expected):
                raise ValueError(
                    f"Anchor mismatch in {file_path}\n"
                    f"Expected baseline @{original_line}: {expected}\n"
                    f"Actual baseline   @{original_line}: {actual}"
                )

        def _insert_affects_existing(edit, original_line):
            if edit["kind"] != "insert":
                return False
            if edit["where"] == "before":
                return original_line >= edit["line"]
            return original_line > edit["line"]

        def _map_existing_line(original_line):
            mapped = original_line
            for edit in state["edits"]:
                if edit["kind"] == "range":
                    if edit["start"] <= original_line <= edit["end"]:
                        raise ValueError(f"Original line @{original_line} was already modified by earlier same-turn line_patch operation {edit['header']}.")
                    if original_line > edit["end"]:
                        mapped += edit["delta"]
                elif _insert_affects_existing(edit, original_line):
                    mapped += edit["delta"]
            return mapped

        def _map_insert_point(where, original_line):
            mapped = original_line
            for edit in state["edits"]:
                if edit["kind"] == "range":
                    if edit["start"] <= original_line <= edit["end"]:
                        raise ValueError(f"Insert anchor @{original_line} was already modified by earlier same-turn line_patch operation {edit['header']}.")
                    if original_line > edit["end"]:
                        mapped += edit["delta"]
                elif edit["kind"] == "insert":
                    if edit["where"] == "before":
                        if original_line >= edit["line"]:
                            mapped += edit["delta"]
                    elif edit["where"] == "after":
                        if original_line > edit["line"] or (where == "after" and original_line == edit["line"]):
                            mapped += edit["delta"]
            return mapped

        def _reject_range_overlap(start_line, end_line, header):
            for edit in state["edits"]:
                if edit["kind"] == "range":
                    if start_line <= edit["end"] and edit["start"] <= end_line:
                        raise ValueError(f"line_patch operation {header} overlaps earlier same-turn operation {edit['header']}.")
                elif edit["kind"] == "insert":
                    inside = (
                        (edit["where"] == "before" and start_line <= edit["line"] <= end_line)
                        or (edit["where"] == "after" and start_line <= edit["line"] < end_line)
                    )
                    if inside:
                        raise ValueError(f"line_patch operation {header} overlaps earlier same-turn operation {edit['header']}.")

        if op in {"replace", "delete"}:
            start_line, start_expected = _parse_anchor(start, "start")
            end_line, end_expected = _parse_anchor(end_or_content, "end")
            if start_line < 1 or end_line < start_line or end_line > baseline_line_count:
                raise ValueError(f"Invalid range {start_line}:{end_line} for {file_path} with {baseline_line_count} baseline lines.")
            _check_baseline_anchor(start_line, start_expected, "start")
            _check_baseline_anchor(end_line, end_expected, "end")
            header = f"{op} @{start_line}..@{end_line}"
            _reject_range_overlap(start_line, end_line, header)
            current_start = _map_existing_line(start_line)
            current_end = _map_existing_line(end_line)
            if current_start < 1 or current_end < current_start or current_end > current_line_count:
                raise ValueError(f"Translated range @{start_line}:@{end_line} no longer fits {file_path}.")
            actual_start = _current_line(current_start)
            actual_end = _current_line(current_end)
            if not _line_matches(actual_start, start_expected):
                raise ValueError(_anchor_mismatch(file_path, start_line, current_start, start_expected, actual_start))
            if not _line_matches(actual_end, end_expected):
                raise ValueError(_anchor_mismatch(file_path, end_line, current_end, end_expected, actual_end))
            replacement = _content_lines(content if op == "replace" else "")
            if op == "delete" and content is not None:
                raise ValueError("delete does not accept replacement content.")
            new_lines = current_lines[:current_start - 1] + replacement + current_lines[current_end:]
            delta = len(replacement) - (end_line - start_line + 1)
            edit_record = {
                "kind": "range",
                "start": start_line,
                "end": end_line,
                "delta": delta,
                "header": header,
            }
        else:
            anchor_line, expected = _parse_anchor(start, "insert")
            insert_content = end_or_content if content is None else content
            body = _content_lines(insert_content)
            if not body:
                raise ValueError(f"{op} requires non-empty content.")
            where = "before" if op == "insert_before" else "after"
            _check_baseline_anchor(
                anchor_line,
                expected,
                "insert",
                allow_prepend=(where == "after"),
                allow_append=(where == "before"),
            )
            header = f"{op} @{anchor_line}"
            current_anchor = _map_insert_point(where, anchor_line)
            valid = 1 <= current_anchor <= current_line_count + 1 if where == "before" else 0 <= current_anchor <= current_line_count
            if not valid:
                raise ValueError(f"Translated insert anchor @{anchor_line} no longer fits {file_path}.")
            if anchor_line in (0, baseline_line_count + 1):
                validation_line = current_anchor
            else:
                validation_line = _map_existing_line(anchor_line)
            actual = _current_line(validation_line)
            if not _line_matches(actual, expected):
                raise ValueError(_anchor_mismatch(file_path, anchor_line, validation_line, expected, actual))
            if where == "before":
                new_lines = current_lines[:current_anchor - 1] + body + current_lines[current_anchor - 1:]
            else:
                new_lines = current_lines[:current_anchor] + body + current_lines[current_anchor:]
            edit_record = {
                "kind": "insert",
                "where": where,
                "line": anchor_line,
                "delta": len(body),
                "header": header,
            }

        new_text = "\n".join(new_lines)
        if new_text == old_text:
            raise ValueError("line_patch produced no changes.")

        if path.suffix == ".py":
            try:
                ast.parse(new_text, filename=str(path))
            except SyntaxError as exc:
                location = f"{exc.lineno}:{exc.offset}" if exc.lineno is not None else "unknown"
                raise SyntaxError(f"line_patch would make invalid Python syntax at {location}: {exc.msg}") from exc

        import difflib
        path.write_text(new_text)
        snapshot.update({
            "content": new_text,
            "sha256": hashlib.sha256(new_text.encode()).hexdigest(),
            "line_count": len(new_text.split("\n")),
            "line_patch_stale": False,
        })
        state["edits"].append(edit_record)

        diff = "".join(difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=file_path,
            tofile=file_path,
        ))
        if diff:
            _send_output("file_diff", diff.rstrip("\n") + "\n")
        _send_output("file_written", _json.dumps({
            "path": file_path,
            "content": new_text,
        }) + "\n")
        if not was_attached:
            lines = new_text.split("\n")
            formatted = "\n".join(f"{i+1:>5}→{line}" for i, line in enumerate(lines))
            _send_output("read_attach", file_path + "\n")
            _send_output("read", formatted + "\n")
        return "Line patch applied."

    @REPLAgent.tool(inject=True)
    def bash(self,
            command: str = "The command to execute",
            timeout: Optional[int] = "Timeout in seconds (default: 120)",
            bg: bool = "Run in background (returns BashProcess object)"
        ):
        """Execute a bash command.

        Returns string output if successful within timeout.
        Returns BashProcess object if bg=True or if command times out.
        """

        import subprocess
        import os
        import fcntl
        import signal
        import time

        # Keep process registry private to the injected bash function. It is
        # only used for cleanup, not for agent-facing process recovery.
        registry = getattr(bash, '_procs', None)
        if registry is None:
            registry = {}
            bash._procs = registry

        if not getattr(bash, '_cleanup_registered', False):
            import atexit
            def _cleanup_bash_procs():
                for pid, bp in list(registry.items()):
                    if bp.returncode is None:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
            atexit.register(_cleanup_bash_procs)
            bash._cleanup_registered = True


        class BashProcess:
            """Running bash command returned by bash(..., bg=True) or timeout.

            Use the returned handle directly instead of polling with ps/kill:

                proc.wait(timeout=1800)
                print(proc.output)

            For incremental monitoring:

                chunk = proc.read(timeout=30)
                if chunk:
                    print(chunk)

            Attributes:
                pid: Process group id.
                command: Original shell command.
                returncode: None while running, otherwise the process exit code.
                output: All stdout/stderr captured so far.

            Methods:
                read(timeout=...): Read newly available output and append it to
                    .output. Blocks until the process exits or timeout expires.
                wait(timeout=...): Wait for exit while draining output into
                    .output. Returns True if the process exited.
                write(text): Write text to stdin.
                kill(): Kill the process group.
            """

            def __init__(self, popen, command, timeout=None):
                self.proc = popen
                self.command = command
                self.pid = popen.pid
                self.timeout = timeout  # None or number
                self._output = ""
                self._set_nonblocking(self.proc.stdout)
                registry[self.pid] = self


            def _set_nonblocking(self, f):
                if f:
                    fd = f.fileno()
                    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            def read(self, timeout=-1):
                """Read newly available output and append it to .output.

                Blocks until process completes or timeout expires.
                Default: uses self.timeout if set, else 120s.
                """
                if timeout == -1:
                    timeout = self.timeout if self.timeout is not None else 120
                output = []
                start = time.time()

                def _read_chunk():
                    found_any = False
                    while True:
                        try:
                            chunk = self.proc.stdout.read(4096)
                            if not chunk:
                                break
                            output.append(chunk)
                            found_any = True
                        except Exception:
                            break
                    return found_any

                while True:
                    _read_chunk()

                    if self.proc.poll() is not None:
                        while _read_chunk():
                            pass
                        break

                    if timeout is not None and time.time() - start > timeout:
                        break

                    time.sleep(0.01)


                new_output = b"".join(output).decode('utf-8', errors='replace')
                self._output += new_output
                return new_output


            def write(self, text):
                """Write text to stdin."""
                if self.proc.stdin:
                    self.proc.stdin.write(text.encode())
                    self.proc.stdin.flush()
            
            def kill(self):
                """Kill the process group."""
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return f"Sent SIGKILL to process group {self.pid}"

            def wait(self, timeout=-1):
                """Wait for process to exit, draining output into .output.

                Default: uses self.timeout if set, else waits forever.
                Returns True if the process exited.
                """
                if timeout == -1:
                    timeout = self.timeout  # None means forever

                start = time.time()
                while True:
                    chunk_timeout = 60 if timeout is None else min(60, timeout - (time.time() - start))
                    if chunk_timeout <= 0:
                        break
                    self.read(timeout=chunk_timeout)
                    if self.returncode is not None:
                        break
                    if timeout is not None and time.time() - start >= timeout:
                        break

                if self.returncode is None:
                    elapsed = time.time() - start
                    output_info = f", {len(self._output)} bytes captured" if self._output else ""
                    print(f"[wait() timed out after {elapsed:.1f}s{output_info}]")
                return self.returncode is not None

            @property
            def output(self):
                """All stdout/stderr captured so far."""
                return self._output


            @property
            def returncode(self):
                code = self.proc.poll()
                if code is not None:
                    registry.pop(self.pid, None)
                return code


            def __repr__(self):
                status = "running" if self.returncode is None else f"exited code={self.returncode}"
                output_info = f" output={len(self._output)}B" if self._output else ""
                return f"[BashProcess pid={self.pid} status={status}{output_info} cmd={self.command!r}]"

        ensure_python_on_path()

        # Set up preexec_fn for Linux to kill child when parent dies
        # PR_SET_PDEATHSIG makes kernel send signal to child on parent death
        def _set_pdeathsig():
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                PR_SET_PDEATHSIG = 1
                libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
            except Exception:
                pass  # Non-Linux or ctypes unavailable

        # Start process
        # start_new_session=True creates a new process group, so we can kill the whole tree
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            start_new_session=True,
            preexec_fn=_set_pdeathsig
        )
        
        bp = BashProcess(proc, command, timeout=timeout)

        # Immediate return if background requested
        if bg:
            if not getattr(bash, '_shown_bashprocess_help', False):
                print(
                    "BashProcess returned. Use proc.wait(timeout=...), "
                    "proc.read(timeout=...), proc.output, proc.returncode, "
                    "or proc.kill()."
                )
                bash._shown_bashprocess_help = True
            return bp


        # Foreground: wait for completion (read uses configured/default timeout)
        output = bp.read()
        
        if bp.returncode is None:
            # Timeout occurred. Preserve a handle even if the caller wrote
            # print(bash(...)) or otherwise failed to assign the return value.
            globals()["bash_proc"] = bp
            globals()[f"bash_proc_{bp.pid}"] = bp
            print(
                f"Timed-out BashProcess assigned to bash_proc "
                f"(also bash_proc_{bp.pid})."
            )
            if not getattr(bash, '_shown_bashprocess_help', False):
                print(
                    "BashProcess returned. Use bash_proc.wait(timeout=...), "
                    "bash_proc.read(timeout=...), bash_proc.output, "
                    "bash_proc.returncode, or bash_proc.kill()."
                )
                bash._shown_bashprocess_help = True
            return bp

        registry.pop(bp.pid, None)
        if bp.returncode != 0:
            return f"[Exit code {bp.returncode}]\n{bp.output}"

        return bp.output.strip()



def _parse_worker_target(value: str) -> tuple[str, str | None]:
    """Parse [user@]host[:path] worker target syntax."""
    if not value:
        raise ValueError("worker target cannot be empty")
    if "/" in value.split(":", 1)[0]:
        raise ValueError(f"Invalid worker target: {value}")
    if ":" not in value:
        return value, None
    host, path = value.split(":", 1)
    if not host or not path:
        raise ValueError(f"Invalid worker target: {value}")
    return host, path


def main():
    """CLI entry point for code-agent."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Code Agent - Python REPL-based coding assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  coda                                # Start with default settings
  coda --model sonnet                 # Use Claude
  coda --max-turns 50                 # Limit conversation turns
  coda --resume                       # Open session picker on startup
  coda --resume <session_id>          # Resume specific session directly
  coda --attach AGENTS.md             # Attach a file on startup
  coda --prompt "Fix the failing tests"  # Submit an initial prompt on startup
"""
    )
    configured_models = _configured_models()
    parser.add_argument(
        "--model", "-m",
        default=configured_models[0],
        help="LLM model to use (default from config or sonnet)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=_get_config_value("code_agent_max_turns", 100),
        help="Maximum turns per interaction (default from config or 100)"
    )
    parser.add_argument(
        "--resume", "-r",
        nargs="?",
        const=True,
        default=False,
        metavar="SESSION_ID",
        help="Resume a session. With no argument, opens the session picker."
    )
    parser.add_argument(
        "--prompt", "-p",
        metavar="TEXT",
        help="Submit an initial prompt automatically on startup.",
    )
    parser.add_argument(
        "--no-repl-display",
        action="store_true",
        help="Suppress intermediary REPL display output while the agent is working.",
    )
    parser.add_argument(
        "--no-response-formatting",
        action="store_true",
        help="Print final emit(release=True) values as plain text without markdown rendering.",
    )
    parser.add_argument(
        "--agent-mode",
        action="store_true",
        help="Use agent-oriented output: suppress REPL display, print plain final responses, and list skills without the interactive picker.",
    )

    parser.add_argument(
        "--debug",
        metavar="LOG_FILE",
        help="Write LLM request bodies and provider responses to LOG_FILE.",
    )
    parser.add_argument(
        "--attach",
        action="append",
        default=[],
        metavar="FILE",
        help="Attach a file on startup. May be passed multiple times.",
    )
    parser.add_argument(
        "worker_target",
        nargs="?",
        help="Optional SSH worker target: [user@]host or [user@]host:path"
    )

    args = parser.parse_args()

    if args.debug:
        log_path = Path(args.debug).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger = logging.getLogger("code_agent")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.propagate = False

    try:
        from code_agent.llm_registry import get_model_config, resolve_model_name
        for model in configured_models:
            get_model_config(model)
        get_model_config(args.model)
        configured_models = [resolve_model_name(model) for model in configured_models]
        args.model = resolve_model_name(args.model)
    except ModelNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    remote_transport = None
    remote_host = "local"
    if args.worker_target:
        try:
            ssh_target, remote_cwd = _parse_worker_target(args.worker_target)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

        from code_agent.tools.transports import SSHSubprocessTransport

        class ConfiguredSSHTransport(SSHSubprocessTransport):
            def __init__(self, *transport_args, **transport_kwargs):
                transport_kwargs.setdefault("ssh_target", ssh_target)
                transport_kwargs.setdefault("remote_cwd", remote_cwd)
                super().__init__(*transport_args, **transport_kwargs)

        remote_transport = ConfiguredSSHTransport
        remote_host = ssh_target

    class ConfiguredAgent(CodeAgent):
        model = args.model
        model_choices = configured_models
        max_turns = args.max_turns
        worker_host = remote_host
        worker_target = args.worker_target
        startup_attachments = args.attach
        agent_mode = args.agent_mode
        repl_display = not (args.no_repl_display or args.agent_mode)
        response_formatting = not (args.no_response_formatting or args.agent_mode)
        if remote_transport is not None:
            repl_transport = remote_transport
    try:
        with ConfiguredAgent() as agent:
            agent.cli_run(resume=args.resume, initial_prompt=args.prompt)
    except ModelNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
