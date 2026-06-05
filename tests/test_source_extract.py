import inspect

import pytest

from code_agent.tools.source_extract import extract_method_source


def test_extract_method_source_accepts_single_top_level_function_with_different_name():
    def original(self, value: str = "description") -> str:
        return value

    source = extract_method_source(original, "renamed")

    assert source.startswith("def renamed(")
    assert "self" not in source.splitlines()[0]
    assert "value=None" in source


def test_extract_method_source_rejects_missing_name_when_source_has_multiple_top_level_functions(monkeypatch):
    def original():
        return 1

    def fake_getsource(_impl):
        return """
def first():
    return 1

def second():
    return 2
"""

    monkeypatch.setattr("code_agent.tools.source_extract.inspect.getsource", fake_getsource)

    with pytest.raises(ValueError, match="Cannot find function 'missing'"):
        extract_method_source(original, "missing")


def test_extract_method_source_prefers_decorated_source_over_current_file(monkeypatch):
    def target(self, pattern="original", path="original"):
        """Original doc."""
        return pattern, path

    target._tool_source = inspect.getsource(target)

    def replacement(self, value, path):
        """Wrong current source."""
        return value, path

    monkeypatch.setattr("code_agent.tools.source_extract.inspect.getsource", lambda _impl: inspect.getsource(replacement))

    source = extract_method_source(target, "target")
    namespace = {}
    exec(source, namespace)

    assert str(inspect.signature(namespace["target"])) == "(pattern=None, path=None)"
    assert namespace["target"]("p", "q") == ("p", "q")


def test_extract_method_source_ignores_nested_functions_with_target_name(monkeypatch):
    def outer(self, file_path="Path"):
        """Outer doc."""
        def outer(value, path):
            return value, path
        return file_path

    monkeypatch.setattr("code_agent.tools.source_extract.inspect.getsource", lambda _impl: """
def outer(self, file_path="Path"):
    \"""Outer doc.\"""
    def outer(value, path):
        return value, path
    return file_path
""")

    source = extract_method_source(outer, "outer")
    namespace = {}
    exec(source, namespace)

    assert str(inspect.signature(namespace["outer"])) == "(file_path=None)"
    assert namespace["outer"]("demo") == "demo"
