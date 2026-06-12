"""
Execution policy checks for REPL-submitted Python code.

Policy checks reject code before it is sent to the worker process. They are not
source-to-source preprocessors: the submitted code remains unchanged for echo and
history, and the framework emits a synthetic execution error.
"""

from __future__ import annotations

import ast


DIRECT_SUBPROCESS_ERROR = (
    "Direct subprocess usage is not supported in this environment. "
    "Use bash(command, timeout=..., bg=False) instead; it provides framework-managed "
    "timeouts, output capture, and cancellation."
)


class ExecutionPolicyError(RuntimeError):
    """Raised when submitted REPL code violates an execution policy."""


def check_execution_policy(code: str) -> None:
    """Raise ExecutionPolicyError if code violates a REPL execution policy."""
    check_no_direct_subprocess_usage(code)


def check_no_direct_subprocess_usage(code: str) -> None:
    """Reject direct subprocess usage in REPL code."""
    try:
        tree = ast.parse(code, "<repl>", "exec")
    except SyntaxError:
        return

    subprocess_names: set[str] = set()
    subprocess_members: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess" or alias.name.startswith("subprocess."):
                    subprocess_names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            for alias in node.names:
                if alias.name == "*":
                    raise ExecutionPolicyError(DIRECT_SUBPROCESS_ERROR)
                subprocess_members.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in subprocess_names:
                raise ExecutionPolicyError(DIRECT_SUBPROCESS_ERROR)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in subprocess_members:
                    raise ExecutionPolicyError(DIRECT_SUBPROCESS_ERROR)
                if func.id == "__import__" and node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant) and first_arg.value == "subprocess":
                        raise ExecutionPolicyError(DIRECT_SUBPROCESS_ERROR)
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id in subprocess_names:
                    raise ExecutionPolicyError(DIRECT_SUBPROCESS_ERROR)

    if subprocess_names or subprocess_members:
        raise ExecutionPolicyError(DIRECT_SUBPROCESS_ERROR)
