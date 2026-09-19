"""Provider abstraction for the extraction engine.

The rest of the pipeline talks to :class:`LLMProvider` only, so no module outside
this file imports a vendor SDK. Two implementations ship here:

* :class:`OpenAIChatProvider` - the real OpenAI-compatible chat/completions call.
* :class:`FakeLLMProvider`    - a deterministic scripted provider for tests.

This module never builds prompts. It receives the exact messages rendered by
``prompts.render_messages`` and returns the raw model text unchanged.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union

import src.config as config


class ProviderConfigError(RuntimeError):
    """Configuration or contract problem. Never retried: retrying cannot help."""


class ProviderCallError(RuntimeError):
    """Transient provider failure (timeout, rate limit, 5xx). Safe to retry."""


class LLMProvider(Protocol):
    """Minimal surface the extraction engine depends on."""

    name: str

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        max_output_tokens: int,
        response_schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "extraction",
    ) -> str:
        """Return the raw assistant text for ``messages``."""
        ...


class OpenAIChatProvider:
    """OpenAI-compatible chat/completions provider.

    The API key is read from the environment and is never logged, never stored on
    a public attribute and never placed in a prompt. The SDK is imported lazily so
    that importing this module (as the offline test suite does) needs no key and
    no network.
    """

    name = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = config.LLM_TIMEOUT_S,
        temperature: float = config.LLM_TEMPERATURE,
        seed: int = config.LLM_SEED,
    ):
        key = api_key or os.getenv(config.LLM_API_KEY_ENV, "")
        if not key or not key.strip() or key.startswith("your_"):
            raise ProviderConfigError(
                f"{config.LLM_API_KEY_ENV} is not set; configure it or use a different provider"
            )

        self.model = model or os.getenv(config.LLM_MODEL_ENV) or config.DEFAULT_LLM_MODEL
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed

        try:
            import openai
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderConfigError(f"openai SDK is not installed: {exc}") from exc

        self._sdk = openai
        # The client holds the credential; nothing else in this object does.
        self._client = openai.OpenAI(api_key=key, base_url=base_url) if base_url else openai.OpenAI(api_key=key)

    def __repr__(self) -> str:  # keeps the key out of tracebacks and logs
        return f"OpenAIChatProvider(model={self.model!r})"

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        max_output_tokens: int,
        response_schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "extraction",
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_completion_tokens": max_output_tokens,
            "timeout": self.timeout,
        }
        if response_schema is not None:
            # Strict structured output: the model can only emit the package's
            # own field names and evidence handles.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": response_schema},
            }

        sdk = self._sdk
        try:
            response = self._client.chat.completions.create(**kwargs)
        except (sdk.AuthenticationError, sdk.PermissionDeniedError) as exc:
            raise ProviderConfigError(f"authentication rejected: {type(exc).__name__}") from exc
        except sdk.BadRequestError as exc:
            # A malformed schema or unsupported parameter. Retrying sends the
            # same thing again, so surface it instead.
            raise ProviderConfigError(f"request rejected: {type(exc).__name__}: {exc}") from exc
        except (
            sdk.RateLimitError,
            sdk.APITimeoutError,
            sdk.APIConnectionError,
            sdk.InternalServerError,
        ) as exc:
            raise ProviderCallError(f"transient provider failure: {type(exc).__name__}") from exc
        except sdk.APIStatusError as exc:
            status = getattr(exc, "status_code", 0) or 0
            if status >= 500:
                raise ProviderCallError(f"provider {status} error") from exc
            raise ProviderConfigError(f"provider {status} error: {type(exc).__name__}") from exc

        choice = response.choices[0]
        # A refusal or an empty completion is returned as empty text; the strict
        # parser then rejects it and the correction path takes over.
        return choice.message.content or ""


ScriptEntry = Union[str, Exception, Callable[..., str]]


class FakeLLMProvider:
    """Deterministic provider used by the offline tests.

    ``script`` is replayed in order. A string is returned as raw model text, an
    exception instance is raised, and a callable is invoked with the messages.
    Every call is recorded so tests can assert exactly what was sent.
    """

    name = "fake"

    def __init__(self, script: Sequence[ScriptEntry] = (), default: Optional[str] = None):
        self.script: List[ScriptEntry] = list(script)
        self.default = default
        self.calls: List[Dict[str, Any]] = []

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        max_output_tokens: int,
        response_schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "extraction",
    ) -> str:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "max_output_tokens": max_output_tokens,
                "response_schema": response_schema,
                "schema_name": schema_name,
            }
        )

        if self.script:
            entry = self.script.pop(0)
        elif self.default is not None:
            entry = self.default
        else:
            raise ProviderCallError("FakeLLMProvider script exhausted")

        if isinstance(entry, Exception):
            raise entry
        if callable(entry):
            return entry(messages)
        return entry


def build_provider(name: Optional[str] = None, **kwargs) -> LLMProvider:
    """Construct the configured provider. Only 'openai' is wired today."""
    provider_name = (name or os.getenv(config.LLM_PROVIDER_ENV) or config.DEFAULT_LLM_PROVIDER).lower()
    if provider_name == "openai":
        return OpenAIChatProvider(**kwargs)
    raise ProviderConfigError(f"unknown LLM provider '{provider_name}'")
