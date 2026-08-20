from code_agent.agent import CodeAgent
from code_agent.base_agent import BaseAgent
from code_agent.conversation import Conversation
from code_agent.repl_agent import REPLMixin
from code_agent.repl_tool_adapter import repl_protocol_prompt
from code_agent.session_replay import replay_session_into_agent


class Client:
    on_retry = None

    def __init__(self, tool_mode=None, name=None):
        self.tool_mode = tool_mode
        self.name = name
        self.calls = []

    def conversation(self, system_prompt):
        return Conversation(self, system_prompt)

    def call(self, messages, **kwargs):
        self.calls.append(messages)
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
        }


class PromptAgent(REPLMixin, BaseAgent):
    system = "stable base"

    def __init__(self, client):
        self._llm_client = client

    def _ensure_setup(self):
        pass


def test_conversation_prefix_is_request_time_only():
    conversation = Conversation(None, "persisted system")
    prefixes = iter(["first prefix", "second prefix"])
    conversation.system_prompt_prefix_provider = lambda: next(prefixes)
    persisted = [dict(message) for message in conversation.stored_messages()]

    first = conversation.projected_messages()
    second = conversation.projected_messages()

    assert first[0] == {
        "role": "system",
        "content": "first prefix\n\npersisted system",
    }
    assert second[0] == {
        "role": "system",
        "content": "second prefix\n\npersisted system",
    }
    assert conversation.stored_messages() == persisted
    assert first[0] is not conversation.stored_messages()[0]
    assert second[0] is not conversation.stored_messages()[0]


def test_conversation_ignores_empty_prefix():
    conversation = Conversation(None, "persisted system")
    conversation.system_prompt_prefix_provider = lambda: ""

    outgoing = conversation.projected_messages()

    assert outgoing == [{"role": "system", "content": "persisted system"}]
    assert conversation.stored_messages() == [{"role": "system", "content": "persisted system"}]


def test_repl_direct_mode_uses_existing_direct_python_preamble():
    agent = PromptAgent(Client())
    conversation = Conversation(agent.llm_client, agent._build_system_prompt())
    agent._configure_conversation(conversation)

    outgoing = conversation.projected_messages()

    assert outgoing[0]["content"].startswith(
        "You are in a Python REPL. Your response body is executed directly as Python source code.\n\n"
        "Respond with raw Python only."
    )
    assert "There is no separate tool-calling layer for your response" in outgoing[0]["content"]
    assert conversation.stored_messages()[0]["content"].startswith("stable base")
    assert "Respond with raw Python only." not in conversation.stored_messages()[0]["content"]


def test_repl_execute_mode_uses_native_preamble():
    agent = PromptAgent(Client("repl_execute"))
    conversation = Conversation(agent.llm_client, agent._build_system_prompt())
    agent._configure_conversation(conversation)

    outgoing = conversation.projected_messages()

    expected = (
        "You are operating Code Agent through a provider-native execution tool.\n\n"
        + repl_protocol_prompt("repl_execute")
    )
    assert outgoing[0]["content"].startswith(expected + "\n\n")
    assert "Respond with raw Python only." not in outgoing[0]["content"]
    assert conversation.stored_messages()[0]["content"].startswith("stable base")
    assert "Every response must include a repl_execute tool call." not in conversation.stored_messages()[0]["content"]


def test_production_code_agent_conversation_has_one_runtime_prefix():
    agent = CodeAgent()
    agent._llm_client = Client()
    try:
        conversation = agent.conversation
        persisted_system = conversation.stored_messages()[0]["content"]

        outgoing_system = conversation.projected_messages()[0]["content"]

        assert "Respond with raw Python only." not in persisted_system
        assert outgoing_system.count("You are in a Python REPL.") == 1
        assert outgoing_system.count("Respond with raw Python only.") == 1
        assert outgoing_system.endswith(persisted_system)
        assert conversation.stored_messages()[0]["content"] == persisted_system
        assert conversation.system_prompt_prefix_provider.__self__ is agent
    finally:
        agent.close()


def test_production_model_switch_updates_dispatch_and_runtime_prefix(monkeypatch):
    from code_agent import base_agent, llm_registry

    clients = {
        "provider/direct": Client(name="direct"),
        "provider/native": Client("repl_execute", name="native"),
    }
    monkeypatch.setattr(llm_registry, "resolve_model_name", lambda name: name)
    monkeypatch.setattr(base_agent, "LLMClient", lambda model: clients[model])
    monkeypatch.setattr(CodeAgent, "model", "provider/direct")

    agent = CodeAgent()
    try:
        conversation = agent.conversation
        conversation.usermsg("unchanged history")
        persisted = [dict(message) for message in conversation.stored_messages()]

        conversation.add_assistant_response()
        assert clients["provider/direct"].calls[-1][0]["content"][0]["text"].count(
            "You are in a Python REPL."
        ) == 1
        conversation.pop_message()

        assert agent._set_model("provider/native") is True
        assert conversation.llm_client is clients["provider/native"]
        assert conversation.stored_messages() == persisted

        conversation.add_assistant_response()
        native_system = clients["provider/native"].calls[-1][0]["content"][0]["text"]
        assert native_system.count(
            "You are operating Code Agent through a provider-native execution tool."
        ) == 1
        assert native_system.count(
            "Every response must include a repl_execute tool call."
        ) == 1
        assert "Respond with raw Python only." not in native_system
        assert conversation.stored_messages()[:-1] == persisted
        assert "Every response must include a repl_execute tool call." not in (
            conversation.stored_messages()[0]["content"]
        )
    finally:
        agent.close()


def test_production_replay_reinstalls_runtime_prefix_provider():
    class Store:
        def get_events(self, session_id):
            return [{
                "seq": 1,
                "event_type": "message_added",
                "payload": {"message": {"role": "user", "content": "resumed"}},
            }]

        def get_session(self, session_id):
            return {"cwd": "."}

    agent = CodeAgent()
    agent._llm_client = Client("repl_execute")
    try:
        conversation = agent.conversation
        persisted_system = conversation.stored_messages()[0]["content"]
        conversation.system_prompt_prefix_provider = None

        replay_session_into_agent(agent, "session", Store())

        assert conversation.system_prompt_prefix_provider.__self__ is agent
        assert conversation.stored_messages()[0]["content"] == persisted_system
        assert "Every response must include a repl_execute tool call." not in persisted_system
        outgoing_system = conversation.projected_messages()[0]["content"]
        assert outgoing_system.count(
            "Every response must include a repl_execute tool call."
        ) == 1
    finally:
        agent.close()


def test_repl_configure_conversation_chains_cooperatively():
    class Parent:
        def _configure_conversation(self, conversation):
            conversation.parent_configured = True

    class ChainedAgent(REPLMixin, Parent):
        _llm_client = Client()

    agent = ChainedAgent()
    conversation = Conversation(agent._llm_client, "system")

    agent._configure_conversation(conversation)

    assert conversation.parent_configured is True
    assert conversation.system_prompt_prefix_provider.__self__ is agent
