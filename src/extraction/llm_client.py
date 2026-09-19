"""Provider abstraction for the extraction engine.

The rest of the pipeline talks to :class:`LLMProvider` only, so no module outside
this file imports a vendor SDK. Two implementations ship here:

* :class:`GeminiChatProvider` - the active provider; Google Gemini API using google-generativeai SDK.
* :class:`FakeLLMProvider`    - a deterministic scripted provider for tests.

Exactly one provider is selected by LLM_PROVIDER.

This module never builds prompts. It receives the exact messages rendered by
``prompts.render_messages`` and returns the raw model text unchanged.
"""

import json
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


class GeminiChatProvider:
    """Google Gemini API chat provider.

    The API key is read from the environment and is never logged, never stored on
    a public attribute and never placed in a prompt. The SDK is imported lazily so
    that importing this module (as the offline test suite does) needs no key and
    no network.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        temperature: float = config.LLM_TEMPERATURE,
    ):
        key = api_key or os.getenv(config.GEMINI_API_KEY_ENV, "")
        if not key or not key.strip() or key.startswith("your_"):
            raise ProviderConfigError(
                f"{config.GEMINI_API_KEY_ENV} is not set; configure it or use a different provider"
            )

        self.model_name = model or os.getenv(config.GEMINI_MODEL_ENV) or config.DEFAULT_GEMINI_MODEL
        self.temperature = temperature

        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise ProviderConfigError(f"google-generativeai SDK is not installed: {exc}") from exc

        self._sdk = genai
        self._api_key = key
        # Setup is deferred to complete() because genai.configure affects global state,
        # but we set it up here as well.
        genai.configure(api_key=key)

    def __repr__(self) -> str:  # keeps the key out of tracebacks and logs
        return f"GeminiChatProvider(model={self.model_name!r})"

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        max_output_tokens: int,
        response_schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "extraction",
    ) -> str:
        genai = self._sdk
        
        # In the context of extraction, we often have a 'system' prompt and 'user' prompt.
        # Gemini expects system_instruction on the GenerativeModel, and user/model in contents.
        system_instruction = None
        contents = []
        
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                # Only take the first system message, or combine them
                if system_instruction is None:
                    system_instruction = content
                else:
                    system_instruction += "\n\n" + content
            else:
                # Map standard roles to gemini roles
                g_role = "user" if role == "user" else "model"
                contents.append({"role": g_role, "parts": [content]})
                
        # If no system_instruction was provided in messages, it's None.
        
        # Configure model
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction
        )
        
        # Configure generation config
        gen_config_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": max_output_tokens,
        }
        
        if response_schema is not None:
            gen_config_kwargs["response_mime_type"] = "application/json"
            # We omit passing response_schema to the SDK because Pydantic's JSON Schema
            # contains advanced constructs (like `anyOf`, `additionalProperties`)
            # that the google-generativeai SDK's strict Schema proto does not support.
            # The prompt already contains the full schema definition, and the mime_type
            # ensures it outputs valid JSON.

        generation_config = genai.types.GenerationConfig(**gen_config_kwargs)
        
        try:
            from google.api_core.exceptions import (
                InvalidArgument,
                PermissionDenied,
                ResourceExhausted,
                RetryError,
                InternalServerError,
                ServiceUnavailable,
                GoogleAPIError
            )
        except ImportError:
            # Fallback if api_core is missing
            raise ProviderConfigError("google.api_core is not installed")

        try:
            response = model.generate_content(
                contents=contents,
                generation_config=generation_config
            )
        except PermissionDenied as exc:
            raise ProviderConfigError(f"authentication rejected: {type(exc).__name__}") from exc
        except ResourceExhausted as exc:
            # 429 quota or rate limit (Free tier is 15 RPM)
            # Sleep aggressively to allow quota to refill, then throw transient error for retry
            import time
            time.sleep(15)
            raise ProviderCallError("transient provider failure: RateLimitError") from exc
        except InvalidArgument as exc:
            raise ProviderConfigError(f"request rejected: {type(exc).__name__}: {exc}") from exc
        except (RetryError, InternalServerError, ServiceUnavailable) as exc:
            raise ProviderCallError(f"transient provider failure: {type(exc).__name__}") from exc
        except GoogleAPIError as exc:
            raise ProviderCallError(f"transient provider failure: {type(exc).__name__}") from exc
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise ProviderCallError(f"transient provider failure: {type(exc).__name__}") from exc

        # Handle the case where the model returned no text (e.g., blocked by safety settings)
        try:
            return response.text
        except ValueError:
            # response.text raises ValueError if there are no valid text parts (e.g. safety block)
            return ""


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


def build_provider() -> LLMProvider:
    """Instantiate the active LLM provider configured in the environment.

    Returns:
        LLMProvider: A configured provider ready for extraction calls.
    Raises:
        ProviderConfigError: If the configured provider name is unknown or if it
            fails to initialize (e.g. missing API key).
    """
    provider_name = (os.getenv(config.LLM_PROVIDER_ENV) or config.DEFAULT_LLM_PROVIDER).strip().lower()

    if provider_name == "gemini":
        return GeminiChatProvider()
    elif provider_name == "fake":
        return FakeLLMProvider()

    raise ProviderConfigError(
        f"unknown {config.LLM_PROVIDER_ENV} {provider_name!r}; valid options: "
        f"'gemini', 'fake'"
    )
