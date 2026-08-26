from code_agent.client import ContextOverflowError
from code_agent.convo import Convo
from code_agent.repl_agent import REPLAgent


class OverflowThenSuccessClient:
    def __init__(self):
        self.calls = 0
        self.tool_mode = "repl_execute"
        self.requests = []

    def call(self, messages, **kwargs):
        self.calls += 1
        self.requests.append(messages)
        if self.calls == 1:
            raise ContextOverflowError("too large")
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "emit('ok', release=True)"}],
        }


class RecoveringAgent(REPLAgent):
    system = "system"

    def __init__(self):
        super().__init__()
        self._llm_client = OverflowThenSuccessClient()
        self._session_id = "test-session"
        self._conversation = Convo(self._llm_client, self.system)
        self._configure_conversation(self._conversation)
        self.coalesced = []
        self.committed = []


    def _ensure_setup(self):
        pass

    def _coalesce_context(self, **kwargs):
        self.coalesced.append(kwargs)

    def _on_assistant_message_committed(self, message):
        self.committed.append(message)



def test_conversation_call_coalesces_open_interaction_and_retries_after_context_overflow():
    agent = RecoveringAgent()
    agent.conversation.usermsg("start")
    result = agent._conversation_call_with_context_recovery(agent.conversation.projected_messages())


    assert result["content"] == [
        {"type": "text", "text": "emit('ok', release=True)"}
    ]
    assert agent.llm_client.calls == 2
    assert agent.coalesced == [{}]
    assert agent.conversation.stored_messages()[-1].get("_virtual_interaction_boundary") is True
    assert agent.committed == [agent.conversation.stored_messages()[-1]]
    assert [
        request[0]["content"][0]["text"].count(
            "Every response must include a repl_execute tool call."
        )
        for request in agent.llm_client.requests
    ] == [1, 1]
    assert all(
        "Every response must include a repl_execute tool call." not in block["text"]
        for block in agent.conversation.stored_messages()[0]["content"]
    )
