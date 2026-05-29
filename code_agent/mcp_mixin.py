"""
MCP integration for CodeAgent.

MCP tools are exposed as proxied Python functions inside the REPL, not as
provider-native tool calls.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Optional

from .base_agent import ToolParam, ToolSpec
from .tools.mcp import create_stdio_client, create_sse_client, MCPError


def _safe_identifier(name: str) -> str:
    name = re.sub(r"\W", "_", str(name))
    if not name or name[0].isdigit():
        name = f"p_{name}"
    return name


class MCPMixin:
    """Mixin that adds MCP server tools as proxied REPL functions."""

    mcp_servers: list = []

    def _ensure_setup(self):
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        if getattr(self, '_mcp_initialized', False):
            return

        if not hasattr(self, '_mcp_lock'):
            self._mcp_lock = threading.Lock()

        with self._mcp_lock:
            if getattr(self, '_mcp_initialized', False):
                return

            self._mcp_clients: dict[str, Any] = {}
            self._mcp_tools: dict[str, tuple[Any, str, dict]] = {}
            self._mcp_instructions: dict[str, str] = {}

            try:
                for server_def in getattr(self, 'mcp_servers', []):
                    self.connect_mcp(*server_def)
                self._mcp_initialized = True
            except Exception:
                for client in self._mcp_clients.values():
                    try:
                        client.close()
                    except Exception:
                        pass
                self._mcp_clients = {}
                self._mcp_tools = {}
                self._mcp_instructions = {}
                raise

    def _build_system_prompt(self):
        if hasattr(super(), '_build_system_prompt'):
            system = super()._build_system_prompt()
        else:
            system = getattr(self, 'system', '')

        instructions = getattr(self, '_mcp_instructions', {})
        if instructions:
            parts = [f"=== {n} ===\n{i}" for n, i in instructions.items()]
            system += "\n\nMCP SERVER INSTRUCTIONS:\n" + "\n\n".join(parts)

        return system

    def _get_dynamic_toolspecs(self):
        specs = super()._get_dynamic_toolspecs() if hasattr(super(), '_get_dynamic_toolspecs') else {}
        for tool_name, (_client, _orig_name, tool_def) in getattr(self, '_mcp_tools', {}).items():
            specs[tool_name] = self._make_mcp_spec(tool_name, tool_def)
        return specs

    def toolcall(self, toolname, function_args):
        if toolname in getattr(self, '_mcp_tools', {}):
            try:
                result = self._call_mcp_raw(toolname, **function_args)
                return self._format_mcp_result(result)
            except MCPError as e:
                return f"[MCP Error] {e}"
        return super().toolcall(toolname, function_args)

    def _call_mcp_raw(self, toolname, **function_args):
        client, orig_name, tool_def = self._mcp_tools[toolname]
        args = {}
        props = (tool_def.get("inputSchema") or {}).get("properties") or {}
        for original in props:
            safe = _safe_identifier(original)
            if safe in function_args:
                args[original] = function_args[safe]
            elif original in function_args:
                args[original] = function_args[original]
        for key, value in function_args.items():
            args.setdefault(key, value)
        return client.call_tool(orig_name, args)

    def _cleanup(self):
        for client in getattr(self, '_mcp_clients', {}).values():
            try:
                client.close()
            except Exception:
                pass

        if hasattr(self, '_mcp_clients'):
            self._mcp_clients = {}
        if hasattr(self, '_mcp_tools'):
            self._mcp_tools = {}
        if hasattr(self, '_mcp_instructions'):
            self._mcp_instructions = {}
        if hasattr(self, '_mcp_initialized'):
            self._mcp_initialized = False

        if hasattr(super(), '_cleanup'):
            super()._cleanup()

    def connect_mcp(self, name: str, server: str, options: Optional[dict] = None):
        opts = options.copy() if options else {}
        include = opts.pop('include', None)
        exclude = opts.pop('exclude', None)

        if server.startswith('http://') or server.startswith('https://'):
            client = create_sse_client(server, **opts)
        else:
            opts.setdefault('forward_stderr', False)
            client = create_stdio_client(server.split(), **opts)

        self._register_mcp_client(name, client, include=include, exclude=exclude)

    def connect_mcp_stdio(
        self,
        name: str,
        command: list[str],
        env: Optional[dict[str, str]] = None,
        timeout: float = 300.0,
        forward_stderr: bool = True
    ):
        client = create_stdio_client(
            command,
            env=env,
            timeout=timeout,
            forward_stderr=forward_stderr
        )
        self._register_mcp_client(name, client)

    def connect_mcp_sse(
        self,
        name: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 300.0
    ):
        client = create_sse_client(
            url,
            headers=headers,
            timeout=timeout
        )
        self._register_mcp_client(name, client)

    def _register_mcp_client(self, name: str, client, include: list = None, exclude: list = None):
        self._mcp_clients[name] = client

        if client.instructions:
            self._mcp_instructions[name] = client.instructions

        for tool_def in client.list_tools():
            orig_name = tool_def['name']
            if include is not None and orig_name not in include:
                continue
            if exclude is not None and orig_name in exclude:
                continue
            tool_name = f"{name}_{orig_name}"
            self._mcp_tools[tool_name] = (client, orig_name, tool_def)

    def disconnect_mcp(self, name: str):
        client = self._mcp_clients.get(name)
        if not client:
            return

        self._mcp_tools = {
            k: v for k, v in self._mcp_tools.items()
            if v[0] is not client
        }
        self._mcp_instructions.pop(name, None)
        del self._mcp_clients[name]

        try:
            client.close()
        except Exception:
            pass

    def _make_mcp_spec(self, tool_name: str, tool_def: dict) -> ToolSpec:
        schema = tool_def.get('inputSchema') or {}
        props = schema.get('properties') or {}
        required = set(schema.get('required') or [])
        params = []

        type_map = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
        }

        for original_name, pschema in props.items():
            safe_name = _safe_identifier(original_name)
            params.append(ToolParam(
                name=safe_name,
                original_name=original_name,
                annotation=type_map.get(pschema.get('type', 'string'), str),
                default=None,
                description=pschema.get('description', ''),
                required=original_name in required,
            ))

        return ToolSpec(
            name=tool_name,
            description=tool_def.get('description') or tool_name,
            params=params,
        )

    def _format_mcp_result(self, result: dict) -> str:
        parts = []
        for item in result.get('content', []):
            if item.get('type') == 'text':
                parts.append(item['text'])
            elif item.get('type') == 'image':
                parts.append(f"[Image: {item.get('mimeType', 'unknown')}]")
            else:
                parts.append(str(item))
        text = '\n'.join(parts) if parts else str(result)
        return f"[MCP Error] {text}" if result.get('isError') else text