# Configuration

## `~/.code-agent/config.py`

Code Agent loads `~/.code-agent/config.py` on startup, if it exists. This is the main
configuration file: use it to set API keys and default models, or to register custom
providers and models—any OpenAI-compatible endpoint, a local Ollama instance, a corporate
proxy, a custom API wrapper, etc.

API keys can be set through `<PROVIDER>_API_KEY` environment variables here. Code Agent
also loads environment variables from a project `.env` file, and `.env.example` lists the
standard names.

Provider definitions accept `api_key` directly. This is useful for endpoints such as Codex
or Cursor gateways, local servers, and corporate proxies. If `api_key` is omitted, a
provider falls back to its `<PROVIDER>_API_KEY` environment variable.

The following functions are injected into the config's namespace before it runs:

| Name | Description |
|------|-------------|
| `register_provider(name, **kwargs)` | Register a new API provider |
| `register_model(provider, alias, **kwargs)` | Register a model under a provider |
| `registry` | The `EndpointRegistry` instance (for advanced use) |

### `register_provider` parameters

| Parameter | Description |
|-----------|-------------|
| `host` | API hostname (e.g. `"api.openai.com"`) |
| `path` | Default request path (e.g. `"/v1/chat/completions"`) |
| `api_key` | API key for the provider; defaults to `<PROVIDER>_API_KEY` from the environment |
| `port` | Port, default `443` |
| `api_type` | `"completions"` (OpenAI-compatible) or `"messages"` (Anthropic) |
| `timeout` | Request timeout in seconds |
| `rpm` | Host-wide requests-per-minute limit |
| `concurrency` | Strict host-wide active-request limit; enables admission when configured |
| `tools` | Whether this provider supports native tool calls |
| `headers` | Extra HTTP headers for completions and responses requests |

### `register_model` parameters

| Parameter | Description |
|-----------|-------------|
| `provider` | Provider name (must match a registered provider) |
| `alias` | Short name to refer to this model (also used as `/model` argument) |
| `aliases` | Additional short names (string or list) |
| `model` | Actual model identifier sent to the API (defaults to `alias`) |
| `path` | Override the provider's default path for this model |
| `context_window` | Context window size in tokens |
| `input_cost` | Cost per million input tokens (USD) |
| `output_cost` | Cost per million output tokens (USD) |
| `cached_cost` | Cost per million cached input tokens (USD) |
| `reasoning_cost` | Cost per million reasoning tokens (USD) |
| `config` | Extra body parameters merged into every request (e.g. `{"temperature": 0.7}`) |
| `timeout` | Per-model timeout override |
| `headers` | Extra HTTP headers; merged over the provider headers |

### Examples

**API keys and default model cycle:**

```python
import os

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["OPENAI_API_KEY"] = "sk-..."

code_agent_model = [
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.4",
]
```

The first entry is used at startup. At an empty prompt, use `Tab` and `Shift+Tab` to cycle
through the list.

**Local Ollama:**

```python
register_provider("ollama",
    host="localhost",
    port=11434,
    path="/v1/chat/completions",
    api_type="completions",
    timeout=300,
)
register_model("ollama", "my-coder",
    model="my-coder-model:latest",
    context_window=32768,
)
```

**Corporate OpenAI proxy:**

```python
register_provider("corp",
    host="my-company-proxy.internal",
    path="/openai/v1/chat/completions",
    api_type="completions",
    timeout=120,
    headers={"X-Proxy-Token": "..."},
)
register_model("corp", "fast",
    model="your-model-name",
    headers={"X-Route": "fast"},
    input_cost=5.0,
    output_cost=15.0,
)
```

Provider headers apply to all of its models. Model headers are merged on top,
overriding provider values with the same name. Custom headers are currently
sent only by the `completions` and `responses` transports.

**Set environment variables:**

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

The config file is plain Python executed at startup, so you can set environment
variables, read from secrets managers, or do anything else you need.

### Selecting a model

Once registered, select any model with:

```bash
coda --model ollama/my-coder
```

Or interactively at the prompt with `/model`.

For a deeper understanding of how providers and models are resolved, see [`code_agent/llm_registry.py`](../code_agent/llm_registry.py).
