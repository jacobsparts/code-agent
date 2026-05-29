# Configuration

## API Keys

Set your API key in a `.env` file in your working directory, or as an environment variable.

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
XAI_API_KEY=...
OPENROUTER_API_KEY=...
```

Copy `.env.example` to get started:

```bash
cp .env.example .env
```

## `~/.code-agent/config.py`

Code Agent loads `~/.code-agent/config.py` on startup, if it exists. Use it to register
custom providers and models—any OpenAI-compatible endpoint, a local Ollama instance,
a corporate proxy, a custom API wrapper, etc.

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
| `port` | Port, default `443` |
| `api_type` | `"completions"` (OpenAI-compatible) or `"messages"` (Anthropic) |
| `timeout` | Request timeout in seconds |
| `tpm` | Requests per minute limit |
| `concurrency` | Max concurrent requests |
| `tools` | Whether this provider supports native tool calls (not used by Code Agent, but tracked) |

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

### Examples

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
)
register_model("corp", "fast",
    model="your-model-name",
    input_cost=5.0,
    output_cost=15.0,
)
```

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
code-agent --model ollama/my-coder
```

Or interactively at the prompt with `/model`.

For a deeper understanding of how providers and models are resolved, see [`code_agent/llm_registry.py`](../code_agent/llm_registry.py).
