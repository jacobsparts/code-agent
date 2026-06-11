import pytest

from code_agent.preprocess import preprocess


def test_preprocess_rejects_import_subprocess():
    code = preprocess("import subprocess\nsubprocess.run(['echo', 'x'])")

    assert code.startswith("raise RuntimeError(")
    assert "Direct subprocess usage is not supported" in code
    assert "Use bash(command" in code


def test_preprocess_rejects_subprocess_alias():
    code = preprocess("import subprocess as sp\nsp.run(['echo', 'x'])")

    assert code.startswith("raise RuntimeError(")
    assert "Use bash(command" in code


def test_preprocess_rejects_from_subprocess_import():
    code = preprocess("from subprocess import run\nrun(['echo', 'x'])")

    assert code.startswith("raise RuntimeError(")
    assert "Direct subprocess usage is not supported" in code


def test_preprocess_rejects_subprocess_after_markdown_fix():
    code = preprocess("""```python
import subprocess
subprocess.run(['echo', 'x'])
```""")

    assert code.startswith("raise RuntimeError(")
    assert "Use bash(command" in code


def test_preprocess_allows_subprocess_in_text_only():
    code = 'emit("subprocess is not supported; use bash()", release=True)'

    assert preprocess(code) == code


def test_preprocess_rejection_executes_with_helpful_error():
    code = preprocess("import subprocess\nsubprocess.run(['echo', 'x'])")

    with pytest.raises(RuntimeError, match="Direct subprocess usage is not supported.*Use bash"):
        exec(code, {})
