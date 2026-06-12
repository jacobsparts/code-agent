import pytest

from code_agent.execution_policy import ExecutionPolicyError, check_execution_policy
from code_agent.preprocess import preprocess


def test_execution_policy_rejects_import_subprocess():
    with pytest.raises(ExecutionPolicyError, match="Direct subprocess usage is not supported.*Use bash"):
        check_execution_policy("import subprocess\nsubprocess.run(['echo', 'x'])")


def test_execution_policy_rejects_subprocess_alias():
    with pytest.raises(ExecutionPolicyError, match="Use bash"):
        check_execution_policy("import subprocess as sp\nsp.run(['echo', 'x'])")


def test_execution_policy_rejects_from_subprocess_import():
    with pytest.raises(ExecutionPolicyError, match="Direct subprocess usage is not supported"):
        check_execution_policy("from subprocess import run\nrun(['echo', 'x'])")


def test_execution_policy_rejects_subprocess_after_markdown_fix():
    code = preprocess("""```python
import subprocess
subprocess.run(['echo', 'x'])
```""")

    assert code == "import subprocess\nsubprocess.run(['echo', 'x'])"
    with pytest.raises(ExecutionPolicyError, match="Use bash"):
        check_execution_policy(code)


def test_preprocess_does_not_replace_subprocess_source():
    code = "import subprocess\nsubprocess.run(['echo', 'x'])"

    assert preprocess(code) == code


def test_execution_policy_allows_subprocess_in_text_only():
    check_execution_policy('emit("subprocess is not supported; use bash()", release=True)')
