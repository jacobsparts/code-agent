import os
from dataclasses import dataclass, field
from code_agent.dotenv import load_dotenv

load_dotenv()

class ModelNotFoundError(Exception):
    """Raised when an unknown model is requested."""
    pass

@dataclass
class ProviderConfig:
    provider: str
    host: str
    path: str
    port: int = 443
    rpm: float = None
    concurrency: int = None
    timeout: int = 900
    tools: bool = False
    api_type: str = "completions"
    token_transform: object = None    # callable(usage_dict) -> usage_dict
    cost_transform: object = None     # callable(prompt, cached, completion, reasoning, in_cost, cached_cost, out_cost, rsn_cost) -> (in_cost, cached_cost, out_cost, rsn_cost)
    response_parser: object = None    # callable(response_json) -> (message, stop_reason, usage)
    headers: dict = field(default_factory=dict)
    api_key: str = None

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv(f"{self.provider.upper()}_API_KEY")

@dataclass
class ModelConfig:
    model: str
    provider: ProviderConfig
    path: str = None
    config: dict = field(default_factory=dict)
    context_window: int = None
    context_constraint: int = None
    max_input_tokens: int = None
    input_cost: float = None
    output_cost: float = None
    cached_cost: float = None
    reasoning_cost: float = None
    timeout: int = None
    tools: bool = None
    tool_mode: str = None
    explicit_prompt_cache: bool = False
    headers: dict = field(default_factory=dict)

    @property
    def request_path(self):
        return self.path or self.provider.path

class EndpointRegistry:
    def __init__(self):
        self._models = {}
        self._providers = {}
        self._aliases = {}

    def list_models(self):
        aliases_by_model = {}
        for alias, full_name in self._aliases.items():
            aliases_by_model.setdefault(full_name, []).append(alias)
        return [
            {
                "full_name": full_name,
                "provider": model.provider.provider,
                "alias": full_name.split("/", 1)[1],
                "model": model.model,
                "aliases": sorted(aliases_by_model.get(full_name, [])),
            }
            for full_name, model in sorted(self._models.items())
            if model.provider.api_key
        ]

    def register_provider(self, name, **kwargs):
        kwargs['provider'] = name
        self._providers[name] = ProviderConfig(**kwargs)

    def register_model(self, provider, alias, aliases=None, **kwargs):
        if not (prov_obj := self._providers.get(provider)):
            raise ValueError(f"unknown provider: {provider}")
        kwargs.setdefault('model', alias)
        for key in ("context_window", "context_constraint", "max_input_tokens"):
            if key in kwargs:
                value = kwargs[key]
                if type(value) is not int or value <= 0:
                    kwargs[key] = None
        full_name = f"{provider}/{alias}"
        self._models[full_name] = ModelConfig(provider=prov_obj, **kwargs)
        if aliases:
            for a in (aliases if isinstance(aliases, list) else [aliases]):
                self._aliases[a] = full_name

    def resolve_model_name(self, name):
        """Resolve an alias or short name to the full model name (provider/model).
        Returns the input unchanged if not an alias."""
        return self._aliases.get(name, name)

    def get_model_config(self, name):
        try:
            model_obj = self._models[self._aliases.get(name, name)]
        except KeyError:
            raise ModelNotFoundError(
                f"Unknown model '{name}'\nAvailable models:\n"
                + "\n".join(f"  - {model['full_name']}" for model in self.list_models())
            )
        if not model_obj.provider.api_key:
            raise Exception(f"{model_obj.provider.provider.upper()}_API_KEY not set")
        return {
            **{k:v for k,v in model_obj.__dict__.items() if not k == 'provider'},
            **{k:v for k,v in model_obj.provider.__dict__.items() if model_obj.__dict__.get(k) is None},
            'provider': model_obj.provider.provider,
            'headers': {**model_obj.provider.headers, **model_obj.headers},
            'path': model_obj.provider.path,
            'request_path': model_obj.request_path,
            'api_key': model_obj.provider.api_key,
        }

registry = EndpointRegistry()
register_provider = registry.register_provider
register_model = registry.register_model
get_model_config = registry.get_model_config
resolve_model_name = registry.resolve_model_name
list_models = registry.list_models

# --- OpenAI Responses ---
register_provider("openai",
    host="api.openai.com",
    path="/v1/responses",
    rpm=100,
    concurrency=30,
    timeout=300,
    tools=True,
    api_type="responses",
)

for conf, efforts in (
    ({'model': 'gpt-5.4', 'input_cost': 2.5, 'cached_cost': 0.25, 'output_cost': 15.0}, ('none', 'medium', 'high')),
    ({'model': 'gpt-5.5', 'input_cost': 5.0, 'cached_cost': 0.5, 'output_cost': 30.0}, ('none', 'medium', 'high')),
    ({'model': 'gpt-5.6-luna', 'input_cost': 0.2, 'cached_cost': 0.02, 'output_cost': 1.2}, ('high','xhigh')),
    ({'model': 'gpt-5.6-sol', 'input_cost': 5.0, 'cached_cost': 0.5, 'output_cost': 30.0}, ('medium', 'high', 'xhigh', 'max')),
):
    for effort in efforts:
        suffix = '' if (effort == 'none' or len(efforts) == 1) else f"-{effort}"
        kwargs = {**conf, "config":{"reasoning_effort": effort}} if effort else conf
        if conf['model'].startswith('gpt-5.6'):
            kwargs = {**kwargs, 'explicit_prompt_cache': True}
            kwargs.setdefault('config', {})['prompt_cache_key'] = 'jp-code-agent-001'
        register_model("openai", f"{conf['model']}{suffix}", **kwargs)


### Anthropic ###

register_provider("anthropic",
    host="api.anthropic.com",
    path="/v1/messages",
    rpm=100,
    concurrency=30,
    timeout=300,
    tools=True,
    api_type="messages"
)
register_model("anthropic","claude-sonnet-4-6",
    model="claude-sonnet-4-6",
    aliases="sonnet",
    input_cost=3.00,
    cached_cost=0.3,
    output_cost=15.0,
)
register_model("anthropic","claude-opus-5",
    model="claude-opus-5",
    aliases="opus",
    input_cost=5.00,
    cached_cost=0.5,
    output_cost=25.0,
)


### Google ###

def gemini_token_transform(usage):
    return {
        'prompt_tokens': usage.get('promptTokenCount', 0),
        'completion_tokens': usage.get('candidatesTokenCount', 0),
        'completion_tokens_details': {
            'reasoning_tokens': usage.get('thoughtsTokenCount', 0),
        },
        'prompt_tokens_details': {
            'cached_tokens': usage.get('cachedContentTokenCount', 0),
        },
    }

register_provider("google",
    host="generativelanguage.googleapis.com",
    path="/v1beta",
    rpm=60,
    concurrency=10,
    timeout=None,
    tools=True,
    api_type="gemini",
    token_transform=gemini_token_transform,
)
register_model("google","gemini-3.7-flash",
    model="gemini-3.7-flash",
    aliases="google/gemini-3.6-flash",
    config={"thinkingLevel": "high"},
    input_cost=0.75,
    cached_cost=0.15,
    output_cost=0.075,
)


# --- User Configuration ---
from code_agent.config import get_config_spec, get_user_config

def load_user_config():
    """Load user's custom model configurations from ~/.code-agent/config.py"""
    # Get spec and module without executing
    spec, user_config = get_config_spec()
    
    if spec is None and user_config is None:
        # Config doesn't exist
        return
    
    if spec is None:
        # Already loaded by someone else
        return
    
    # Inject registry functions before execution
    user_config.register_provider = register_provider
    user_config.register_model = register_model
    user_config.registry = registry
    
    # Now execute with injected functions
    get_user_config()

load_user_config()
