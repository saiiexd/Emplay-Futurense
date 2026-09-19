"""Provider selection and OpenRouter adapter tests.

Everything here is offline: the network is replaced by mocks and no test needs a
real key. Covers provider isolation (one provider never requires another's
credentials), OpenRouter request construction, and error classification.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
from src.extraction.llm_client import (
    MistralChatProvider,
    OpenAIChatProvider,
    OpenRouterChatProvider,
    ProviderCallError,
    ProviderConfigError,
    build_provider,
)

FAKE_KEY = "sk-or-v1-FAKE-KEY-FOR-TESTS-ONLY"


def openrouter_env(**overrides):
    env = {
        config.LLM_PROVIDER_ENV: "openrouter",
        config.OPENROUTER_API_KEY_ENV: FAKE_KEY,
    }
    env.update(overrides)
    return env


class StubResponse:
    """Mimics the OpenAI SDK response object shape."""

    def __init__(self, content="{}", error=None, choices=True):
        self.error = error
        if choices:
            message = MagicMock()
            message.content = content
            choice = MagicMock()
            choice.message = message
            self.choices = [choice]
        else:
            self.choices = []


def make_provider(**kwargs):
    """Build an OpenRouter provider whose HTTP client is a mock."""
    provider = OpenRouterChatProvider(api_key=FAKE_KEY, **kwargs)
    provider._client = MagicMock()
    return provider


# --- provider selection ------------------------------------------------------

class TestProviderSelection(unittest.TestCase):
    def test_default_provider_is_openrouter(self):
        self.assertEqual(config.DEFAULT_LLM_PROVIDER, "openrouter")
        with patch.dict(os.environ, openrouter_env(), clear=True):
            os.environ.pop(config.LLM_PROVIDER_ENV)
            self.assertEqual(build_provider().name, "openrouter")

    def test_explicit_selection_of_each_provider(self):
        with patch.dict(os.environ, openrouter_env(), clear=True):
            self.assertIsInstance(build_provider(), OpenRouterChatProvider)
        with patch.dict(os.environ, {config.LLM_PROVIDER_ENV: "openai",
                                     config.LLM_API_KEY_ENV: "sk-openai"}, clear=True):
            self.assertIsInstance(build_provider(), OpenAIChatProvider)
        with patch.dict(os.environ, {config.LLM_PROVIDER_ENV: "mistral",
                                     config.MISTRAL_API_KEY_ENV: "mistral-key"}, clear=True):
            self.assertIsInstance(build_provider(), MistralChatProvider)

    def test_provider_name_is_case_and_space_insensitive(self):
        with patch.dict(os.environ, openrouter_env(**{config.LLM_PROVIDER_ENV: "  OpenRouter "}), clear=True):
            self.assertEqual(build_provider().name, "openrouter")

    def test_unknown_provider_is_rejected(self):
        with patch.dict(os.environ, {config.LLM_PROVIDER_ENV: "not-a-provider"}, clear=True):
            with self.assertRaises(ProviderConfigError) as ctx:
                build_provider()
            self.assertIn("openrouter", str(ctx.exception))


class TestCredentialIsolation(unittest.TestCase):
    """Selecting one provider must never demand another provider's key."""

    def test_openrouter_does_not_need_openai_or_mistral_keys(self):
        with patch.dict(os.environ, openrouter_env(), clear=True):
            provider = build_provider()
            self.assertEqual(provider.name, "openrouter")

    def test_openrouter_missing_key_fails_with_its_own_variable(self):
        with patch.dict(os.environ, {config.LLM_PROVIDER_ENV: "openrouter",
                                     config.LLM_API_KEY_ENV: "sk-openai-present"}, clear=True):
            with self.assertRaises(ProviderConfigError) as ctx:
                build_provider()
            self.assertIn(config.OPENROUTER_API_KEY_ENV, str(ctx.exception))

    def test_openai_does_not_need_openrouter_key(self):
        with patch.dict(os.environ, {config.LLM_PROVIDER_ENV: "openai",
                                     config.LLM_API_KEY_ENV: "sk-openai"}, clear=True):
            self.assertEqual(build_provider().name, "openai")

    def test_openai_missing_key_fails_with_its_own_variable(self):
        with patch.dict(os.environ, {config.LLM_PROVIDER_ENV: "openai",
                                     config.OPENROUTER_API_KEY_ENV: FAKE_KEY}, clear=True):
            with self.assertRaises(ProviderConfigError) as ctx:
                build_provider()
            self.assertIn(config.LLM_API_KEY_ENV, str(ctx.exception))

    def test_mistral_does_not_need_openrouter_or_openai_keys(self):
        with patch.dict(os.environ, {config.LLM_PROVIDER_ENV: "mistral",
                                     config.MISTRAL_API_KEY_ENV: "mistral-key"}, clear=True):
            self.assertEqual(build_provider().name, "mistral")

    def test_mistral_missing_key_fails_with_its_own_variable(self):
        with patch.dict(os.environ, {config.LLM_PROVIDER_ENV: "mistral",
                                     config.OPENROUTER_API_KEY_ENV: FAKE_KEY}, clear=True):
            with self.assertRaises(ProviderConfigError) as ctx:
                build_provider()
            self.assertIn(config.MISTRAL_API_KEY_ENV, str(ctx.exception))

    def test_placeholder_key_is_rejected(self):
        with self.assertRaises(ProviderConfigError):
            OpenRouterChatProvider(api_key="your_openrouter_api_key_here")


# --- configuration -----------------------------------------------------------

class TestOpenRouterConfiguration(unittest.TestCase):
    def test_default_base_url_and_model(self):
        provider = OpenRouterChatProvider(api_key=FAKE_KEY)
        self.assertEqual(provider.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(provider.model, config.DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(provider.model, "qwen/qwen3-30b-a3b:free")

    def test_env_overrides_base_url_and_model(self):
        with patch.dict(os.environ, {
            config.OPENROUTER_BASE_URL_ENV: "https://example.test/api/v1",
            config.OPENROUTER_MODEL_ENV: "vendor/other-model:free",
        }, clear=True):
            provider = OpenRouterChatProvider(api_key=FAKE_KEY)
            self.assertEqual(provider.base_url, "https://example.test/api/v1")
            self.assertEqual(provider.model, "vendor/other-model:free")

    def test_deterministic_settings_preserved(self):
        provider = OpenRouterChatProvider(api_key=FAKE_KEY)
        self.assertEqual(provider.temperature, 0.0)
        self.assertEqual(provider.temperature, config.LLM_TEMPERATURE)
        self.assertEqual(provider.seed, config.LLM_SEED)
        self.assertEqual(provider.timeout, config.LLM_TIMEOUT_S)

    def test_sdk_retries_disabled_engine_owns_budget(self):
        provider = OpenRouterChatProvider(api_key=FAKE_KEY)
        self.assertEqual(provider._client.max_retries, 0)

    def test_repr_never_exposes_the_key(self):
        provider = OpenRouterChatProvider(api_key=FAKE_KEY)
        self.assertNotIn("FAKE-KEY", repr(provider))
        self.assertIn("qwen", repr(provider))


# --- request construction ----------------------------------------------------

class TestOpenRouterRequest(unittest.TestCase):
    def test_request_carries_model_messages_and_determinism(self):
        provider = make_provider()
        provider._client.chat.completions.create.return_value = StubResponse('{"ok":1}')
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]

        provider.complete(messages, max_output_tokens=1234)

        kwargs = provider._client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen/qwen3-30b-a3b:free")
        self.assertEqual(kwargs["messages"], messages)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["seed"], config.LLM_SEED)
        self.assertEqual(kwargs["timeout"], config.LLM_TIMEOUT_S)
        # OpenRouter's OpenAI-compatible surface uses max_tokens.
        self.assertEqual(kwargs["max_tokens"], 1234)

    def test_strict_json_schema_is_sent_when_a_schema_is_supplied(self):
        provider = make_provider()
        provider._client.chat.completions.create.return_value = StubResponse('{"ok":1}')
        schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

        provider.complete([{"role": "user", "content": "x"}],
                          max_output_tokens=10, response_schema=schema, schema_name="extract_identity")

        rf = provider._client.chat.completions.create.call_args.kwargs["response_format"]
        self.assertEqual(rf["type"], "json_schema")
        self.assertTrue(rf["json_schema"]["strict"])
        self.assertEqual(rf["json_schema"]["name"], "extract_identity")
        self.assertEqual(rf["json_schema"]["schema"], schema)

    def test_no_response_format_when_no_schema(self):
        provider = make_provider()
        provider._client.chat.completions.create.return_value = StubResponse('{"ok":1}')
        provider.complete([{"role": "user", "content": "x"}], max_output_tokens=10)
        self.assertNotIn("response_format",
                         provider._client.chat.completions.create.call_args.kwargs)

    def test_successful_response_returns_raw_text_unchanged(self):
        provider = make_provider()
        payload = '{"group":"identity","candidates":[],"not_found":[]}'
        provider._client.chat.completions.create.return_value = StubResponse(payload)
        self.assertEqual(provider.complete([{"role": "user", "content": "x"}], max_output_tokens=10),
                         payload)

    def test_empty_content_returns_empty_string_for_the_parser_to_reject(self):
        provider = make_provider()
        provider._client.chat.completions.create.return_value = StubResponse(None)
        self.assertEqual(provider.complete([{"role": "user", "content": "x"}], max_output_tokens=10), "")


# --- error classification ----------------------------------------------------

def sdk_error(cls_name, **attrs):
    """Build a real SDK exception instance without performing a request."""
    import openai
    import httpx
    cls = getattr(openai, cls_name)
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    body = attrs.pop("body", None)
    status = attrs.pop("status_code", 400)
    message = attrs.pop("message", cls_name)
    response = httpx.Response(status, request=request, json=body or {})
    if cls_name in ("APITimeoutError", "APIConnectionError"):
        return cls(request=request)
    return cls(message=message, response=response, body=body)


class TestOpenRouterErrorClassification(unittest.TestCase):
    def _raise(self, exc):
        provider = make_provider()
        provider._client.chat.completions.create.side_effect = exc
        return provider

    def test_401_is_fatal_configuration_error(self):
        provider = self._raise(sdk_error("AuthenticationError", status_code=401))
        with self.assertRaises(ProviderConfigError) as ctx:
            provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)
        self.assertIn(config.OPENROUTER_API_KEY_ENV, str(ctx.exception))

    def test_403_is_fatal_configuration_error(self):
        provider = self._raise(sdk_error("PermissionDeniedError", status_code=403))
        with self.assertRaises(ProviderConfigError):
            provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)

    def test_429_rate_limit_is_transient(self):
        exc = sdk_error("RateLimitError", status_code=429,
                        body={"error": {"code": "rate_limit_exceeded", "message": "slow down"}})
        provider = self._raise(exc)
        with self.assertRaises(ProviderCallError):
            provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)

    def test_429_exhausted_quota_is_fatal(self):
        exc = sdk_error("RateLimitError", status_code=429,
                        body={"error": {"code": "insufficient_quota", "message": "no credits remaining"}})
        provider = self._raise(exc)
        with self.assertRaises(ProviderConfigError):
            provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)

    def test_400_bad_request_is_fatal(self):
        provider = self._raise(sdk_error("BadRequestError", status_code=400,
                                         message="model does not support response_format"))
        with self.assertRaises(ProviderConfigError):
            provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)

    def test_timeout_and_connection_errors_are_transient(self):
        for name in ("APITimeoutError", "APIConnectionError"):
            provider = self._raise(sdk_error(name))
            with self.assertRaises(ProviderCallError):
                provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)

    def test_5xx_is_transient(self):
        provider = self._raise(sdk_error("InternalServerError", status_code=503))
        with self.assertRaises(ProviderCallError):
            provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)


class TestOpenRouterErrorInBody(unittest.TestCase):
    """OpenRouter can answer HTTP 200 with an error payload; that is not success."""

    def _complete_with_error(self, error):
        provider = make_provider()
        provider._client.chat.completions.create.return_value = StubResponse(error=error)
        return provider

    def test_body_401_is_fatal(self):
        provider = self._complete_with_error({"code": 401, "message": "No auth credentials found"})
        with self.assertRaises(ProviderConfigError):
            provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)

    def test_body_429_rate_limit_is_transient(self):
        provider = self._complete_with_error({"code": 429, "message": "temporarily rate-limited upstream"})
        with self.assertRaises(ProviderCallError):
            provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)

    def test_body_429_quota_is_fatal(self):
        provider = self._complete_with_error({"code": 429, "message": "insufficient_quota for this key"})
        with self.assertRaises(ProviderConfigError):
            provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)

    def test_body_502_is_transient(self):
        provider = self._complete_with_error({"code": 502, "message": "upstream provider error"})
        with self.assertRaises(ProviderCallError):
            provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)

    def test_no_choices_is_transient_not_a_silent_empty_answer(self):
        provider = make_provider()
        provider._client.chat.completions.create.return_value = StubResponse(choices=False)
        with self.assertRaises(ProviderCallError):
            provider.complete([{"role": "user", "content": "x"}], max_output_tokens=5)


if __name__ == '__main__':
    unittest.main()
