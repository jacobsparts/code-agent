from code_agent.client import ContextOverflowError
from code_agent.conversation import Conversation
from code_agent.repl_agent import REPLAgent


class OverflowThenSuccessClient:
    def __init__(self):
        self.calls = 0

    def text_call(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise ContextOverflowError("too large")
        return {"role": "assistant", "content": "emit('ok', release=True)"}


class RecoveringAgent(REPLAgent):
    system = "system"

    def __init__(self):
        super().__init__()
        self._llm_client = OverflowThenSuccessClient()
        self._session_id = "test-session"
        self._conversation = Conversation(self._llm_client, self.system)
        self.coalesced = []
        self.committed = []


    def _ensure_setup(self):
        pass

    def _coalesce_context(self, **kwargs):
        self.coalesced.append(kwargs)

    def _on_assistant_message_committed(self, message):
        self.committed.append(message)



def test_text_call_coalesces_open_interaction_and_retries_after_context_overflow():
    agent = RecoveringAgent()
    agent.conversation.usermsg("start")
    result = agent._llm_text_call_with_context_recovery(agent.conversation._messages())


    assert result["content"] == "emit('ok', release=True)"
    assert agent.llm_client.calls == 2
    assert agent.coalesced == [{}]
    assert agent.conversation.messages[-1].get("_virtual_interaction_boundary") is True
    assert agent.committed == [agent.conversation.messages[-1]]
