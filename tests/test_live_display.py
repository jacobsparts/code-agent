from code_agent.repl_events import ReplEvent
from code_agent.repl_agent import _StatementOutput
from code_agent.agent import CodeAgentBase
from code_agent.cli.terminal import strip_ansi
from unittest.mock import MagicMock

class ListQueue:
    def __init__(self):
        self.items = []
    def put(self, item):
        self.items.append(item)
    def get_nowait(self):
        return self.items.pop(0)

def test_display_events_never_reach_model():
    """The display channel is structurally separate: display events are handed to
    on_repl_event only, and never enter the event list the model sees."""
    from code_agent.repl_agent import REPLMixin, ToolREPL

    agent = REPLMixin()
    streamed = []
    agent.on_repl_event = streamed.append
    repl = ToolREPL(echo=False)
    repl.inject_builtins()
    try:
        output, _, events, _ = agent._execute_with_tool_handling(repl, "print('hello')")
    finally:
        repl.close()

    assert output == ">>> print('hello')\nhello\n"
    assert [event.kind for event in events if event.kind == "display"] == []
    assert [event.text for event in streamed if event.kind == "display"] == ["hello\n"]

def test_display_reaches_before_done():
    q = ListQueue()
    so = _StatementOutput(q)
    so.send("output", "hello\n")
    assert any(kind == "display" for kind, _ in q.items), "display not queued before finish"
    display_items = [it for it in q.items if it[0] == "display"]
    assert len(display_items) >= 1
    assert not any(k == "output" and v == "hello\n" for k, v in q.items)
    so.finish()
    assert ("output", "hello\n") in q.items
    idx_display = next(i for i,(k,v) in enumerate(q.items) if k=="display")
    idx_output = next(i for i,(k,v) in enumerate(q.items) if k=="output" and v=="hello\n")
    assert idx_display < idx_output

def test_line_buffering():
    q = ListQueue()
    so = _StatementOutput(q)
    so.send("output", "hell")
    assert not any(k=="display" for k,_ in q.items), "should not emit incomplete line"
    so.send("output", "o")
    assert not any(k=="display" for k,_ in q.items)
    so.send("output", "\n")
    display_texts = [v[1] for k,v in q.items if k=="display"]
    assert display_texts == ["hello\n"]
    q2 = ListQueue()
    so2 = _StatementOutput(q2)
    so2.send("output", "partial")
    assert not any(k=="display" for k,_ in q2.items)
    so2.send("print", "next\n")
    kinds = [v[0] for k,v in q2.items if k=="display"]
    texts = [v[1] for k,v in q2.items if k=="display"]
    assert texts[0] == "partial"
    assert kinds[0] == "output"
    assert texts[1] == "next\n"
    assert kinds[1] == "print"
    q3 = ListQueue()
    so3 = _StatementOutput(q3)
    so3.send("output", "trailing")
    assert not any(k=="display" for k,_ in q3.items)
    so3.finish()
    texts3 = [v[1] for k,v in q3.items if k=="display"]
    assert "trailing" in texts3

def test_per_turn_cap_emits_once_and_suppresses():
    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._display_capture = []
    agent._statement_echo = ">>> x=1\n"
    agent._statement_echo_displayed = False
    agent._statement_direct_call = None
    agent._statement_source = "x=1"
    agent._statement_had_diff = False
    agent._statement_print_uses_variable = False
    agent._turn_output_started = True
    agent._repl_printed_header = True
    agent._header_pending = False
    agent.repl_display = True
    agent._in_user_repl = False
    agent.MAX_DISPLAY_LINES = 2
    agent._cli_console = MagicMock()
    agent._cli_console.clear_line = lambda: None
    orig_capture = agent._capture_display_line
    agent._capture_display_line = lambda text="": agent._display_capture.append(text+"\n" if not str(text).endswith("\n") else str(text))
    import builtins
    orig_print = builtins.print
    printed = []
    def mock_print(*args, **kwargs):
        printed.append(" ".join(str(a) for a in args))
    builtins.print = mock_print
    try:
        agent.on_repl_execute(None)
        agent.on_repl_event(ReplEvent(kind="statement_started", data={"source":"x=1","echo":">>> x=1\n","display_echo":">>> x=1\n","direct_call":None}))
        agent.on_repl_event(ReplEvent(kind="display", text="line1\n", data={"display_kind":"output"}))
        agent.on_repl_event(ReplEvent(kind="display", text="line2\n", data={"display_kind":"output"}))
        printed.clear()
        agent._display_capture.clear()
        agent.on_repl_execute(None)
        agent._display_capture.clear()
        agent.MAX_DISPLAY_LINES = 3
        agent.on_repl_event(ReplEvent(kind="statement_started", data={"source":"a=1","echo":">>> a=1\n","display_echo":">>> a=1\n","direct_call":None}))
        agent.on_repl_event(ReplEvent(kind="display", text="1\n", data={"display_kind":"output"}))
        assert any(">>> a=1" in c for c in agent._display_capture), "echo not printed"
        printed.clear()
        agent._display_capture.clear()
        agent.on_repl_execute(None)
        agent._display_capture.clear()
        agent._display_line_count = 3
        agent.MAX_DISPLAY_LINES = 3
        agent.on_repl_event(ReplEvent(kind="statement_started", data={"source":"b=1","echo":">>> b=1\n","display_echo":">>> b=1\n","direct_call":None}))
        agent.on_repl_event(ReplEvent(kind="display", text="should_not_appear\n", data={"display_kind":"output"}))
        truncation = [p for p in printed if "display truncated" in strip_ansi(p)]
        truncation_cap = [c for c in agent._display_capture if "display truncated" in c]
        assert len(truncation) == 1, f"truncation not printed exactly once: {printed}"
        assert len(truncation_cap) == 1, f"truncation not captured exactly once: {agent._display_capture}"
        printed_before = len(printed)
        capture_before = len(agent._display_capture)
        agent.on_repl_event(ReplEvent(kind="display", text="also_suppressed\n", data={"display_kind":"output"}))
        assert len(printed) == printed_before and len(agent._display_capture) == capture_before, "further output not suppressed"
        agent.on_repl_execute(None)
        assert agent._display_line_count == 0
        assert not agent._display_truncated
        printed.clear()
        agent._display_capture.clear()
        agent.on_repl_event(ReplEvent(kind="statement_started", data={"source":"c=1","echo":">>> c=1\n","display_echo":">>> c=1\n","direct_call":None}))
        agent.on_repl_event(ReplEvent(kind="display", text="after_reset\n", data={"display_kind":"output"}))
        assert any("after_reset" in c for c in agent._display_capture)
    finally:
        builtins.print = orig_print
        agent._capture_display_line = orig_capture

def test_spill_always_rendered_even_when_suppressed():
    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._display_capture = []
    agent._statement_direct_call = None
    agent._statement_echo = ""
    agent._statement_echo_displayed = True
    agent._statement_had_diff = False
    agent._statement_print_uses_variable = False
    agent._turn_output_started = True
    agent._repl_printed_header = True
    agent._header_pending = False
    agent.repl_display = True
    agent._in_user_repl = False
    agent.MAX_DISPLAY_LINES = 1
    agent._cli_console = MagicMock()
    agent._cli_console.clear_line = lambda: None
    agent._capture_display_line = lambda text="": agent._display_capture.append(text+"\n" if not str(text).endswith("\n") else str(text))
    import builtins
    orig_print = builtins.print
    printed = []
    def mock_print(*args, **kwargs):
        printed.append(" ".join(str(a) for a in args))
    builtins.print = mock_print
    try:
        agent.on_repl_execute(None)
        agent.on_repl_event(ReplEvent(kind="statement_started", data={"source":"x=1","echo":">>> x=1\n","display_echo":">>> x=1\n","direct_call":None}))
        agent.on_repl_event(ReplEvent(kind="display", text="line1\n", data={"display_kind":"output"}))
        agent.on_repl_event(ReplEvent(kind="display", text="line2\n", data={"display_kind":"output"}))
        spill = "[large output written to /tmp/fake.txt (1.0MB)]\n"
        agent.on_repl_event(ReplEvent(kind="display", text=spill, data={"display_kind":"spill"}))
        assert spill.strip() in strip_ansi("\n".join(printed)), "spill not printed"
        assert spill.strip() in "".join(agent._display_capture), "spill not captured"
        idx_trunc = next(i for i, c in enumerate(agent._display_capture) if "display truncated" in c)
        idx_spill = next(i for i, c in enumerate(agent._display_capture) if "large output" in c)
        assert idx_trunc < idx_spill, "spill should be after truncation notice"
    finally:
        builtins.print = orig_print

def test_statement_echo_interleaving():
    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._display_capture = []
    agent._statement_echo = ""
    agent._statement_echo_displayed = False
    agent._statement_direct_call = None
    agent._statement_source = ""
    agent._statement_had_diff = False
    agent._statement_print_uses_variable = False
    agent._turn_output_started = False
    agent._repl_printed_header = True
    agent._header_pending = False
    agent.repl_display = True
    agent._in_user_repl = False
    agent.MAX_DISPLAY_LINES = 1000
    agent._cli_console = MagicMock()
    agent._cli_console.clear_line = lambda: None
    agent._capture_display_line = lambda text="": agent._display_capture.append(text)
    import builtins
    orig_print = builtins.print
    printed = []
    def mock_print(*args, **kwargs):
        printed.append(" ".join(str(a) for a in args))
    builtins.print = mock_print
    try:
        agent.on_repl_execute(None)
        agent.on_repl_event(ReplEvent(kind="statement_started", data={"source":"x = 1","echo":">>> x = 1\n","display_echo":">>> x = 1\n","direct_call":None}))
        assert any(">>> x = 1" in strip_ansi(c) for c in agent._display_capture), "echo not immediately flushed"
        printed.clear()
        agent._display_capture.clear()
        agent.on_repl_event(ReplEvent(kind="statement_started", data={"source":'print("hi")', "echo":'>>> print("hi")\n',"display_echo":'>>> print("hi")\n',"direct_call":"print"}))
        assert not any(">>> print" in strip_ansi(p) for p in printed), "literal print echo should be suppressed initially"
        agent.on_repl_event(ReplEvent(kind="display", text="hi\n", data={"display_kind":"print"}))
        assert not any(">>> print" in strip_ansi(p) for p in printed) and not any(">>> print" in strip_ansi(c) for c in agent._display_capture), "literal echo incorrectly flushed"
        assert any("hi" in strip_ansi(c) for c in agent._display_capture)
        printed.clear()
        agent._display_capture.clear()
        agent.on_repl_event(ReplEvent(kind="statement_started", data={"source":"print(x)", "echo":">>> print(x)\n","display_echo":">>> print(x)\n","direct_call":"print"}))
        assert not any(">>> print(x)" in strip_ansi(p) for p in printed)
        agent.on_repl_event(ReplEvent(kind="display", text="1\n", data={"display_kind":"print"}))
        assert any(">>> print(x)" in strip_ansi(c) for c in agent._display_capture), "variable print echo not flushed"
        cap_str = "\n".join(agent._display_capture)
        assert cap_str.find(">>> print(x)") < cap_str.find("1"), "echo not before output"
    finally:
        builtins.print = orig_print

def test_benchmark_metrics_ignore_display():
    from code_agent.repl_benchmark.core import InstrumentedREPLBenchmarkMixin
    mixin = InstrumentedREPLBenchmarkMixin()
    mixin._benchmark_reset_metrics()
    mixin.on_repl_event(ReplEvent(kind="display", text="hi\n", data={"display_kind":"output"}))
    assert mixin._benchmark_metrics["chunk_counts"] == {}
    mixin.on_repl_event(ReplEvent(kind="output", text="hi\n"))
    assert mixin._benchmark_metrics["chunk_counts"].get("output") == 1
    mixin.on_repl_event(ReplEvent(kind="print", text="hi\n"))
    assert mixin._benchmark_metrics["chunk_counts"].get("print") == 1
