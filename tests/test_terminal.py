import os

from code_agent.cli.terminal import render_markdown, strip_ansi


def test_table_cells_wrap_instead_of_truncating():
    markdown = """| key | value |
| --- | --- |
| a | 1234567890abcdef |
| b | short |"""

    rendered = strip_ansi(render_markdown(markdown, width=20))
    lines = rendered.splitlines()

    assert "…" not in rendered
    assert "1234567890ab" in rendered
    assert "cdef" in rendered
    assert max(len(line) for line in lines) <= 22


def test_table_columns_keep_readable_minimum_width():
    markdown = """| c1 | c2 | c3 | c4 |
| --- | --- | --- | --- |
| abcdefghijklmnop | qrstuvwxyz012345 | ABCDEFGHIJKLMNOP | QRSTUVWXYZ012345 |"""

    rendered = strip_ansi(render_markdown(markdown, width=30))
    lines = rendered.splitlines()

    assert "…" not in rendered
    assert "abcdefghijkl" in rendered
    assert "mnop" in rendered
    assert max(len(line) for line in lines) > 30



class _NoopRawMode:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass



class _FakeStdin:
    def fileno(self):
        return 0


def test_prompt_tab_callback_only_runs_with_empty_input(monkeypatch):
    import io
    from code_agent.cli import prompt as prompt_module

    reads = iter([b"x", b"\t", b"\r"])
    statuses = []
    output = io.StringIO()

    monkeypatch.setattr(prompt_module, "RawMode", _NoopRawMode)
    monkeypatch.setattr(prompt_module.os, "read", lambda fd, size: next(reads))
    monkeypatch.setattr(prompt_module.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(prompt_module.sys, "stdout", output)

    result = prompt_module.prompt(
        prompt_str="> ",
        on_tab=lambda: statuses.append("called") or "changed",
    )

    assert result == "x"
    assert statuses == []


def test_prompt_moves_cursor_to_end_before_submitting(monkeypatch):
    import io
    from code_agent.cli import prompt as prompt_module

    reads = iter([b"\x1b[D", b"\x1b[D", b"\x1b[D", b"\x1b[D", b"\r"])
    output = io.StringIO()

    monkeypatch.setattr(prompt_module, "RawMode", _NoopRawMode)
    monkeypatch.setattr(prompt_module.os, "read", lambda fd, size: next(reads))
    monkeypatch.setattr(prompt_module.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(prompt_module.sys, "stdout", output)
    monkeypatch.setattr(
        prompt_module.os,
        "get_terminal_size",
        lambda: os.terminal_size((80, 24)),
    )

    result = prompt_module.prompt(prompt_str="> ", initial_text="one\ntwo")

    assert result == "one\ntwo"
    assert output.getvalue().endswith("\r\x1b[3C\x1b[?25h\n")


def test_repeated_prompt_tabs_replace_model_status_line(monkeypatch):
    import io
    from code_agent.cli import prompt as prompt_module

    reads = iter([b"\t", b"\t", b"\r"])
    statuses = iter(["model-b", "model-c"])
    output = io.StringIO()

    monkeypatch.setattr(prompt_module, "RawMode", _NoopRawMode)
    monkeypatch.setattr(prompt_module.os, "read", lambda fd, size: next(reads))
    monkeypatch.setattr(prompt_module.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(prompt_module.sys, "stdout", output)
    monkeypatch.setattr(
        prompt_module.os,
        "get_terminal_size",
        lambda: os.terminal_size((80, 24)),
    )

    result = prompt_module.prompt(
        prompt_str="> ",
        on_tab=lambda: next(statuses),
    )

    rendered = output.getvalue()
    assert result == ""
    assert rendered.count("model-b\n") == 1
    assert "\x1b[1A\r\x1b[2Kmodel-c\x1b[1B\r\x1b[2C" in rendered
    assert "\nmodel-c\n" not in rendered


def test_prompt_shift_tab_callback_only_runs_with_empty_input(monkeypatch):
    import io
    from code_agent.cli import prompt as prompt_module

    reads = iter([b"x", b"\x1b[Z", b"\r"])
    statuses = []
    output = io.StringIO()

    monkeypatch.setattr(prompt_module, "RawMode", _NoopRawMode)
    monkeypatch.setattr(prompt_module.os, "read", lambda fd, size: next(reads))
    monkeypatch.setattr(prompt_module.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(prompt_module.sys, "stdout", output)

    result = prompt_module.prompt(
        prompt_str="> ",
        on_shift_tab=lambda: statuses.append("called") or "changed",
    )

    assert result == "x"
    assert statuses == []


def test_prompt_shift_tab_runs_reverse_callback(monkeypatch):
    import io
    from code_agent.cli import prompt as prompt_module

    reads = iter([b"\x1b[Z", b"\r"])
    statuses = []
    output = io.StringIO()

    monkeypatch.setattr(prompt_module, "RawMode", _NoopRawMode)
    monkeypatch.setattr(prompt_module.os, "read", lambda fd, size: next(reads))
    monkeypatch.setattr(prompt_module.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(prompt_module.sys, "stdout", output)

    result = prompt_module.prompt(
        prompt_str="> ",
        on_shift_tab=lambda: statuses.append("called") or "model-c",
    )

    assert result == ""
    assert statuses == ["called"]
    assert "model-c\n" in output.getvalue()


def test_cycle_model_reverse_wraps_to_last_choice():
    from code_agent.agent import CodeAgentBase

    agent = object.__new__(CodeAgentBase)
    agent.model_choices = ["model-a", "model-b", "model-c"]
    agent.model = "model-a"

    status = agent._cycle_model_reverse()

    assert agent.model == "model-c"
    assert "Model changed: model-c" in status
