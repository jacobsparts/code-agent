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
