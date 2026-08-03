import gc
import weakref
from unittest.mock import MagicMock

from code_agent.subagent import Subagent, SubagentResponse, WORKER_CODE, _subagents


def test_worker_uses_code_agent_interaction_and_output_hooks():
    compile(WORKER_CODE, "<subagent-worker>", "exec")

    assert "self.output_hook = self._subagent_output_hook" in WORKER_CODE
    assert "agent.run_interaction(prompt, max_turns=task_max_turns)" in WORKER_CODE
    assert "super().on_repl_execute(code)" in WORKER_CODE
    assert "def _handle_tool_request(self, repl, req)" not in WORKER_CODE
    assert "agent.usermsg(prompt)" not in WORKER_CODE
    assert "agent.run_loop(max_turns=task_max_turns)" not in WORKER_CODE


def test_subagent_has_no_context_manager_and_response_does_not_own_agent():
    agent = Subagent()
    response = SubagentResponse(agent)
    agent_ref = weakref.ref(agent)

    assert not hasattr(agent, "__enter__")
    assert not hasattr(agent, "__exit__")
    assert agent.id in _subagents

    close = MagicMock()
    agent.close = close
    del agent
    gc.collect()

    assert agent_ref() is None
    assert response._agent_ref() is None
    close.assert_called_once_with()
