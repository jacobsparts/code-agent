
from pathlib import Path

import pytest

from code_agent.agent import CodeAgent
from code_agent.tools.source_extract import extract_method_source


def build_line_patch(tmp_path):
    source = extract_method_source(CodeAgent._toolimpl["line_patch"], "line_patch")
    outputs = []
    requests = []

    def _send_tool_request(payload):
        requests.append(payload)

    def _wait_for_ack(_request_id):
        return False

    def _send_output(kind, data):
        outputs.append((kind, data))

    namespace = {
        "Path": Path,
        "Optional": object,
        "_request_id": 0,
        "_send_tool_request": _send_tool_request,
        "_wait_for_ack": _wait_for_ack,
        "_send_output": _send_output,
    }
    exec(source, namespace)
    namespace["outputs"] = outputs
    return namespace


def write_file(tmp_path, text, name="sample.txt"):
    path = tmp_path / name
    path.write_text(text)
    return path


def lines(path):
    return path.read_text().split("\n")


def test_insert_before_then_replace_later_original_line(tmp_path):
    ns = build_line_patch(tmp_path)
    path = write_file(tmp_path, "a\nb\nc")

    assert ns["line_patch"](str(path), "insert_before", "@2 b", "x\n") == "Line patch applied."
    assert ns["line_patch"](str(path), "replace", "@3 c", "@3 c", "C\n") == "Line patch applied."

    assert lines(path) == ["a", "x", "b", "C"]


def test_replace_with_different_line_count_then_insert_after_later_original_line(tmp_path):
    ns = build_line_patch(tmp_path)
    path = write_file(tmp_path, "a\nb\nc\nd")

    ns["line_patch"](str(path), "replace", "@2 b", "@3 c", "B\nC\nCC\n")
    ns["line_patch"](str(path), "insert_after", "@4 d", "x\n")

    assert lines(path) == ["a", "B", "C", "CC", "d", "x"]


def test_delete_before_later_target(tmp_path):
    ns = build_line_patch(tmp_path)
    path = write_file(tmp_path, "a\nb\nc\nd")

    ns["line_patch"](str(path), "delete", "@2 b", "@2 b")
    ns["line_patch"](str(path), "insert_before", "@4 d", "x\n")

    assert lines(path) == ["a", "c", "x", "d"]


def test_reject_overlap_with_replaced_range(tmp_path):
    ns = build_line_patch(tmp_path)
    path = write_file(tmp_path, "a\nb\nc\nd")

    ns["line_patch"](str(path), "replace", "@2 b", "@3 c", "B\n")

    with pytest.raises(ValueError, match="already modified|overlaps"):
        ns["line_patch"](str(path), "replace", "@3 c", "@3 c", "C\n")

    assert lines(path) == ["a", "B", "d"]


def test_reject_targeting_deleted_line(tmp_path):
    ns = build_line_patch(tmp_path)
    path = write_file(tmp_path, "a\nb\nc\nd")

    ns["line_patch"](str(path), "delete", "@2 b", "@3 c")

    with pytest.raises(ValueError, match="already modified"):
        ns["line_patch"](str(path), "insert_after", "@2 b", "x\n")

    assert lines(path) == ["a", "d"]


def test_anchor_mismatch_rejects_without_change(tmp_path):
    ns = build_line_patch(tmp_path)
    path = write_file(tmp_path, "a\nb\nc")

    with pytest.raises(ValueError, match="Anchor mismatch"):
        ns["line_patch"](str(path), "replace", "@2 wrong", "@2 b", "B\n")

    assert path.read_text() == "a\nb\nc"


def test_insert_after_same_anchor_preserves_call_order(tmp_path):
    ns = build_line_patch(tmp_path)
    path = write_file(tmp_path, "a\nb\nc")

    ns["line_patch"](str(path), "insert_after", "@2 b", "x\n")
    ns["line_patch"](str(path), "insert_after", "@2 b", "y\n")

    assert lines(path) == ["a", "b", "x", "y", "c"]


def test_insert_before_and_after_same_anchor(tmp_path):
    ns = build_line_patch(tmp_path)
    path = write_file(tmp_path, "a\nb\nc")

    ns["line_patch"](str(path), "insert_before", "@2 b", "x\n")
    ns["line_patch"](str(path), "insert_after", "@2 b", "y\n")

    assert lines(path) == ["a", "x", "b", "y", "c"]


def test_invalid_python_rolls_back_only_failed_call(tmp_path):
    ns = build_line_patch(tmp_path)
    path = write_file(tmp_path, "def f():\n    return 1\n", "sample.py")

    ns["line_patch"](str(path), "insert_after", "@1 def f():", "    x = 1\n")

    with pytest.raises(SyntaxError, match="invalid Python syntax"):
        ns["line_patch"](str(path), "replace", "@2     return 1", "@2     return 1", "    if True:\n")

    assert path.read_text() == "def f():\n    x = 1\n    return 1\n"


@pytest.mark.parametrize(
    "args, error",
    [
        ((None, "insert_after", "@1 a", "x\n"), TypeError),
        (("sample.txt", "unknown", "@1 a", "x\n"), ValueError),
        (("sample.txt", "insert_after", "1 a", "x\n"), ValueError),
        (("sample.txt", "insert_after", "@99 z", "x\n"), ValueError),
        (("sample.txt", "replace", "@2 b", "@1 a", "x\n"), ValueError),
    ],
)
def test_invalid_arguments(tmp_path, args, error):
    ns = build_line_patch(tmp_path)
    write_file(tmp_path, "a\nb\nc")
    fixed = tuple(str(tmp_path / arg) if i == 0 and isinstance(arg, str) else arg for i, arg in enumerate(args))

    with pytest.raises(error):
        ns["line_patch"](*fixed)


def test_whitespace_stripped_anchor_match(tmp_path):
    ns = build_line_patch(tmp_path)
    path = write_file(tmp_path, "def f():\n    return 1\n")

    ns["line_patch"](str(path), "replace", "@2 return 1", "@2 return 1", "    return 2\n")

    assert path.read_text() == "def f():\n    return 2\n"


def test_prepend_and_append_virtual_anchors(tmp_path):
    ns = build_line_patch(tmp_path)
    path = write_file(tmp_path, "a\nb")

    ns["line_patch"](str(path), "insert_after", "@0", "start\n")
    ns["line_patch"](str(path), "insert_before", "@3", "end\n")

    assert lines(path) == ["start", "a", "b", "end"]
