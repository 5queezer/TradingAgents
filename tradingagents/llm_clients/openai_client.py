import os
from typing import Any, List, Optional

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class FallbackChatModel(Runnable):
    """Primary chat model + ordered fallbacks with `.bind_tools()` support.

    Why this exists
    ---------------
    LangChain's native `.with_fallbacks()` returns a `RunnableWithFallbacks`
    that is a valid `Runnable` (pipes, invoke, batch — all work). But it
    does *not* expose `.bind_tools()`, because tools are bound on the
    underlying chat-model class, not on the runnable wrapper. TradingAgents
    agents call `llm.bind_tools(tools)` at construction time, so we need
    `.bind_tools()` on the same object the factory returns from `get_llm()`.

    Behaviour
    ---------
    - As a `Runnable`: forwards `invoke` / `stream` / `batch` to the
      fallback-wrapped runnable, so `prompt | llm` works even without
      binding tools.
    - `.bind_tools(tools)` returns a native `RunnableWithFallbacks` over
      each chain member with tools bound — the caller gets a proper
      LangChain Runnable straight away.
    """

    def __init__(self, primary, fallbacks: List[Any]):
        super().__init__()
        self._primary = primary
        self._fallbacks = list(fallbacks)
        self._runnable = (
            primary.with_fallbacks(self._fallbacks) if self._fallbacks else primary
        )

    def bind_tools(self, tools, **kwargs):
        bound_primary = self._primary.bind_tools(tools, **kwargs)
        if not self._fallbacks:
            return bound_primary
        bound_fallbacks = [fb.bind_tools(tools, **kwargs) for fb in self._fallbacks]
        return bound_primary.with_fallbacks(bound_fallbacks)

    def invoke(self, input, config=None, **kwargs):
        return self._runnable.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        return await self._runnable.ainvoke(input, config=config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        return self._runnable.stream(input, config=config, **kwargs)

    async def astream(self, input, config=None, **kwargs):
        async for chunk in self._runnable.astream(input, config=config, **kwargs):
            yield chunk

    def batch(self, inputs, config=None, **kwargs):
        return self._runnable.batch(inputs, config=config, **kwargs)

    def __getattr__(self, name):
        # Unknown attributes — try primary first (it's the real chat model),
        # then the fallback-wrapped runnable.
        try:
            return getattr(self._primary, name)
        except AttributeError:
            return getattr(self._runnable, name)

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://api.z.ai/api/paas/v4/", "ZHIPU_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth
        if self.provider in _PROVIDER_CONFIG:
            base_url, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = base_url
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        # Env-driven retry bump (tenacity-based exponential backoff inside
        # ChatOpenAI). Default 2 retries is too aggressive for flaky
        # free-tier providers.
        env_retries = os.environ.get("TRADINGAGENTS_LLM_MAX_RETRIES")
        if env_retries and "max_retries" not in llm_kwargs:
            llm_kwargs["max_retries"] = int(env_retries)

        primary = NormalizedChatOpenAI(**llm_kwargs)

        # Env-driven fallback chain. Same provider + base_url + api_key;
        # only the model name differs. When primary errors (e.g. 429), the
        # runnable tries each fallback in order.
        fallback_models = [
            m.strip()
            for m in os.environ.get("TRADINGAGENTS_FALLBACK_MODELS", "").split(",")
            if m.strip() and m.strip() != self.model
        ]
        if not fallback_models:
            return primary

        fallbacks = [
            NormalizedChatOpenAI(**{**llm_kwargs, "model": m}) for m in fallback_models
        ]
        return FallbackChatModel(primary, fallbacks)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
