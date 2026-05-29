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
