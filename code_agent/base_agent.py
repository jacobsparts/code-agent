import inspect
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any

from .client import LLMClient

logger = logging.getLogger('code_agent')


class _CompleteException(BaseException):
    pass


@dataclass
class ToolParam:
    name: str
    annotation: Any = inspect.Parameter.empty
    default: Any = inspect.Parameter.empty
    description: str = ""
    required: bool = True
    original_name: str | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)


def _is_optional_annotation(annotation):
    """Return True for Optional[T] / Union[T, None] / T | None."""
    if annotation in (inspect.Parameter.empty, None):
        return False

    if isinstance(annotation, str):
        text = annotation.replace(" ", "")
        return (
            text.startswith("Optional[")
            or "None" in text.split("|")
            or "NoneType" in text
        )

    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    try:
        from typing import Union
        if origin is Union and type(None) in args:
            return True
    except Exception:
        pass

    try:
        import types as _types
        if origin is _types.UnionType and type(None) in args:
            return True
    except Exception:
        pass

    return type(None) in args


def _tool_spec_from_function(fn, toolname: str) -> ToolSpec:
    params = []
    for p in list(inspect.signature(fn).parameters.values())[1:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        description = p.default if isinstance(p.default, str) else ""
        default = inspect.Parameter.empty if isinstance(p.default, str) else p.default
        required = default is inspect.Parameter.empty and not _is_optional_annotation(p.annotation)
        params.append(ToolParam(
            name=p.name,
            annotation=p.annotation,
            default=default,
            description=description,
            required=required,
        ))
    return ToolSpec(
        name=toolname,
        description=textwrap.dedent(fn.__doc__ or "").strip(),
        params=params,
    )


class AgentMeta(type):
    def __new__(mcls, name, bases, clsdict):
        local_tools = {}
        local_specs = {}

        for attr_name, attr_value in clsdict.items():
            if callable(attr_value) and hasattr(attr_value, '_tool_name'):
                tool_name = getattr(attr_value, '_tool_name')
                tool_spec = getattr(attr_value, '_tool_spec')
                local_tools[tool_name] = attr_value
                local_specs[tool_name] = tool_spec
                delattr(attr_value, '_tool_name')
                delattr(attr_value, '_tool_spec')

        cls = super().__new__(mcls, name, bases, clsdict)

        final_tool_registry = {}
        for base in reversed(cls.__mro__[1:]):
            if base_registry := getattr(base, '_toolimpl', None):
                final_tool_registry.update(base_registry)
        final_tool_registry.update(local_tools)
        cls._toolimpl = final_tool_registry

        final_spec_registry = {}
        for base in reversed(cls.__mro__[1:]):
            if base_specs := getattr(base, '_toolspec', None):
                final_spec_registry.update(base_specs)
        final_spec_registry.update(local_specs)
        cls._toolspec = final_spec_registry

        return cls


class BaseAgent(metaclass=AgentMeta):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        user_init = cls.__dict__.get('__init__')
        if user_init:
            def wrapped(self, *args, _user_init=user_init, **kwargs):
                _user_init(self, *args, **kwargs)
                self._ensure_setup()
            cls.__init__ = wrapped
        else:
            def default_init(self):
                self._ensure_setup()
            cls.__init__ = default_init

    def _ensure_setup(self):
        """Hook for mixins to do lazy initialization. Override and call super()."""
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

    def _build_system_prompt(self):
        """Hook for mixins to modify system prompt. Override and call super()."""
        prompt = getattr(self, 'system', '')
        if hasattr(super(), '_build_system_prompt'):
            prompt = super()._build_system_prompt() + prompt
        return prompt

    def _get_dynamic_toolspecs(self):
        """Hook for mixins to add dynamic REPL tools. Override and call super()."""
        if hasattr(super(), '_get_dynamic_toolspecs'):
            return super()._get_dynamic_toolspecs()
        return {}

    def _cleanup(self):
        """Hook for mixins to clean up resources. Override and call super()."""
        if hasattr(super(), '_cleanup'):
            super()._cleanup()

    def close(self):
        """Clean up resources. Call when done with agent, or use as context manager."""
        self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def tool(_input=None, model=None, inject=False):  # decorator
        if model is not None:
            raise TypeError("model= tool schemas are not supported by Code Agent")

        def decorator(fn):
            if fn.__doc__ is None:
                raise ValueError(f"Missing docstring: {fn.__name__}")
            fn._tool_name = toolname = fn.__name__
            fn._tool_inject = inject
            if inject:
                try:
                    fn._tool_source = inspect.getsource(fn)
                except (OSError, TypeError):
                    fn._tool_source = None
            fn._tool_spec = lambda self, fn=fn, toolname=toolname: _tool_spec_from_function(fn, toolname)
            return fn

        if _input is not None and callable(_input):
            return decorator(_input)
        return decorator

    @property
    def toolspecs(self):
        result = {}
        for k, v in self.__class__._toolspec.items():
            result[k] = v(self) if callable(v) else v
        result.update(self._get_dynamic_toolspecs())
        return result

    def toolcall(self, toolname, function_args):
        if func := self.__class__._toolimpl.get(toolname):
            return func(self, **function_args)
        raise KeyError(f"No tool '{toolname}' registered for class {self.__class__.__name__}")

    @property
    def llm_client(self):
        try:
            return self._llm_client
        except AttributeError:
            assert hasattr(self, 'model'), "model must be defined"
            self._llm_client = LLMClient(self.model)
            return self._llm_client

    @property
    def conversation(self):
        try:
            return self._conversation
        except AttributeError:
            system = self._build_system_prompt()
            assert system, "system must be defined"
            self._conversation = self.llm_client.conversation(system)
            if hasattr(self, "_configure_conversation"):
                self._configure_conversation(self._conversation)
            return self._conversation

    def next_assistant_message(self):
        return self.conversation.add_assistant_response()

    def usermsg(self, *args, **kwargs):
        return self.conversation.usermsg(*args, **kwargs)

    @property
    def ephemeral(self):
        return self.conversation.ephemeral

    @ephemeral.setter
    def ephemeral(self, value):
        self.conversation.ephemeral = value

    def run_loop(self, max_turns):
        self.complete = False
        for _ in range(max_turns):
            resp_msg = self.next_assistant_message()
            content = resp_msg.get("content")
            if content:
                return content
        raise Exception(f"{type(self).__name__} did not complete within {max_turns} turns")


    def respond(self, value=None):
        self._complete_value = value
        raise _CompleteException()