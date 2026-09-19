"""Provider abstraction for the extraction engine.

The rest of the pipeline talks to :class:`LLMProvider` only, so no module outside
this file imports a vendor SDK. Four implementations ship here:

* :class:`OpenRouterChatProvider` - the active provider; OpenAI-compatible API
  at https://openrouter.ai/api/v1, reached with the openai SDK.
* :class:`OpenAIChatProvider`     - optional, OpenAI directly.
* :class:`MistralChatProvider`    - optional, native Mistral SDK.
* :class:`FakeLLMProvider`        - a deterministic scripted provider for tests.

Exactly one provider is selected by LLM_PROVIDER. Each validates only its own
credentials and there is never a silent fallback between providers.

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


#: Markers OpenAI uses when a 429 means "no balance" rather than "slow down".
_QUOTA_MARKERS = ("insufficient_quota", "credit_balance_exhausted", "no credits remaining")


def _is_quota_exhausted(exc: Exception) -> bool:
    """True when a 429 reports an exhausted balance, which retrying cannot fix."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") or {}
        for field in ("code", "type"):
            value = str(error.get(field) or "").lower()
            if any(marker in value for marker in _QUOTA_MARKERS):
                return True
    text = str(exc).lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


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
        # max_retries=0 because ExtractionEngine owns the retry policy; letting
        # the SDK retry as well multiplies attempts and hides the real count.
        client_kwargs = {"api_key": key, "max_retries": 0}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)

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
        except sdk.RateLimitError as exc:
            # A 429 means either "slow down" or "you have no credits". Only the
            # first is worth retrying; retrying an exhausted balance just burns
            # time and produces a misleading "provider unavailable" result.
            if _is_quota_exhausted(exc):
                raise ProviderConfigError(
                    "the API account has no remaining credits or quota; "
                    "add credits or use a different account"
                ) from exc
            raise ProviderCallError("transient provider failure: RateLimitError") from exc
        except sdk.BadRequestError as exc:
            # A malformed schema or unsupported parameter. Retrying sends the
            # same thing again, so surface it instead.
            raise ProviderConfigError(f"request rejected: {type(exc).__name__}: {exc}") from exc
        except (
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


class MistralChatProvider:
    """Mistral API provider."""

    name = "mistral"

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
        key = api_key or os.getenv(config.MISTRAL_API_KEY_ENV, "")
        if not key or not key.strip() or key.startswith("your_"):
            raise ProviderConfigError(
                f"{config.MISTRAL_API_KEY_ENV} is not set; configure it or use a different provider"
            )

        self.model = model or os.getenv(config.MISTRAL_MODEL_ENV) or config.DEFAULT_MISTRAL_MODEL
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed

        try:
            from mistralai.client import Mistral
        except ImportError as exc:
            raise ProviderConfigError(f"mistralai SDK is not installed: {exc}") from exc

        client_kwargs = {"api_key": key}
        effective_base_url = base_url or os.getenv(config.MISTRAL_BASE_URL_ENV)
        if effective_base_url:
            client_kwargs["server_url"] = effective_base_url
        self._client = Mistral(**client_kwargs)

    def __repr__(self) -> str:
        return f"MistralChatProvider(model={self.model!r})"

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
            "random_seed": self.seed,
            "max_tokens": max_output_tokens,
        }

        if response_schema is not None:
            # Mistral structured output format
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.complete(**kwargs)
        except Exception as exc:
            exc_str = str(exc).lower()
            status = getattr(exc, "status_code", 0) or 0

            if status == 401 or status == 403 or "unauthorized" in exc_str or "authentication" in exc_str:
                raise ProviderConfigError(f"authentication rejected: {type(exc).__name__}") from exc
            if status == 400 or "validation" in exc_str:
                raise ProviderConfigError(f"request rejected: {type(exc).__name__}: {exc}") from exc
            if status == 429 or "rate limit" in exc_str or "429" in exc_str:
                if _is_quota_exhausted(exc):
                    raise ProviderConfigError(
                        "the API account has no remaining credits or quota; "
                        "add credits or use a different account"
                    ) from exc
                raise ProviderCallError("transient provider failure: RateLimitError") from exc

            if status >= 500:
                raise ProviderCallError(f"provider {status} error") from exc
            if "timeout" in exc_str or "connection" in exc_str:
                raise ProviderCallError(f"transient provider failure: {type(exc).__name__}") from exc

            raise ProviderCallError(f"transient provider failure: {type(exc).__name__}") from exc

        choice = response.choices[0]
        return choice.message.content or ""


class OpenRouterChatProvider:
    """OpenRouter provider (the active one).

    OpenRouter exposes an OpenAI-compatible ``/chat/completions`` endpoint, so
    this reuses the already-declared ``openai`` SDK pointed at a different base
    URL instead of adding another vendor dependency. The SDK sets
    ``Authorization: Bearer <key>``; the key lives only inside the client.
    """

    name = "openrouter"

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
        key = api_key or os.getenv(config.OPENROUTER_API_KEY_ENV, "")
        if not key or not key.strip() or key.startswith("your_"):
            raise ProviderConfigError(
                f"{config.OPENROUTER_API_KEY_ENV} is not set; configure it or use a different provider"
            )

        self.model = model or os.getenv(config.OPENROUTER_MODEL_ENV) or config.DEFAULT_OPENROUTER_MODEL
        self.base_url = (
            base_url
            or os.getenv(config.OPENROUTER_BASE_URL_ENV)
            or config.DEFAULT_OPENROUTER_BASE_URL
        )
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed

        try:
            import openai
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderConfigError(f"openai SDK is not installed: {exc}") from exc

        self._sdk = openai
        # max_retries=0: ExtractionEngine owns the retry policy.
        self._client = openai.OpenAI(api_key=key, base_url=self.base_url, max_retries=0)

    def __repr__(self) -> str:  # keeps the key out of tracebacks and logs
        return f"OpenRouterChatProvider(model={self.model!r}, base_url={self.base_url!r})"

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
            # OpenRouter's OpenAI-compatible surface uses max_tokens.
            "max_tokens": max_output_tokens,
            "timeout": self.timeout,
        }
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": response_schema},
            }

        sdk = self._sdk
        try:
            response = self._client.chat.completions.create(**kwargs)
        except (sdk.AuthenticationError, sdk.PermissionDeniedError) as exc:
            raise ProviderConfigError(
                f"OpenRouter rejected the credential ({type(exc).__name__}); "
                f"check {config.OPENROUTER_API_KEY_ENV}"
            ) from exc
        except sdk.RateLimitError as exc:
            if _is_quota_exhausted(exc):
                raise ProviderConfigError(
                    "the OpenRouter account has no remaining credits or quota; "
                    "add credits or use a different model/account"
                ) from exc
            raise ProviderCallError("transient provider failure: RateLimitError") from exc
        except sdk.BadRequestError as exc:
            # Includes a model that cannot honour the strict schema. Retrying
            # resends the same unsupported request, so surface it instead.
            raise ProviderConfigError(f"OpenRouter rejected the request: {exc}") from exc
        except (
            sdk.APITimeoutError,
            sdk.APIConnectionError,
            sdk.InternalServerError,
        ) as exc:
            raise ProviderCallError(f"transient provider failure: {type(exc).__name__}") from exc
        except sdk.APIStatusError as exc:
            status = getattr(exc, "status_code", 0) or 0
            if status in (408, 409, 425) or status >= 500:
                raise ProviderCallError(f"provider {status} error") from exc
            raise ProviderConfigError(f"provider {status} error: {type(exc).__name__}") from exc

        # OpenRouter can answer HTTP 200 while reporting an upstream failure in
        # the body, which the SDK does not raise for. Treat that as a real error
        # rather than letting an empty completion look like a valid answer.
        error = getattr(response, "error", None)
        if error:
            code = 0
            message = str(error)
            if isinstance(error, dict):
                code = error.get("code") or 0
                message = str(error.get("message") or error)
            try:
                code = int(code)
            except (TypeError, ValueError):
                code = 0
            if code in (401, 403):
                raise ProviderConfigError(f"OpenRouter rejected the credential: {message}")
            if code == 429:
                if any(m in message.lower() for m in _QUOTA_MARKERS) or "quota" in message.lower():
                    raise ProviderConfigError(f"OpenRouter quota exhausted: {message}")
                raise ProviderCallError(f"OpenRouter rate limited: {message}")
            if code >= 500 or code == 408:
                raise ProviderCallError(f"OpenRouter upstream error {code}: {message}")
            raise ProviderConfigError(f"OpenRouter returned an error: {message}")

        if not getattr(response, "choices", None):
            raise ProviderCallError("OpenRouter returned no choices")

        choice = response.choices[0]
        return getattr(choice.message, "content", None) or ""


def build_provider(name: Optional[str] = None, **kwargs) -> LLMProvider:
    """Construct exactly one configured provider.

    Each provider validates only its own credentials, so selecting OpenRouter
    never requires an OpenAI or Mistral key, and vice versa. There is no
    fallback between providers: a misconfigured provider fails loudly.
    """
    raw_env = os.getenv(config.LLM_PROVIDER_ENV)
    provider_name = (name or raw_env or config.DEFAULT_LLM_PROVIDER).strip().lower()
    if provider_name == "openrouter":
        return OpenRouterChatProvider(**kwargs)
    if provider_name == "openai":
        return OpenAIChatProvider(**kwargs)
    if provider_name == "mistral":
        return MistralChatProvider(**kwargs)
    raise ProviderConfigError(
        f"unknown LLM provider '{provider_name}'; expected one of: openrouter, openai, mistral"
    )
