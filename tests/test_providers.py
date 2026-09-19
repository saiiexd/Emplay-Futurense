import json
import os
from unittest import mock

import pytest
from src.extraction import llm_client
from src.extraction.llm_client import (
    FakeLLMProvider,
    GeminiChatProvider,
    ProviderCallError,
    ProviderConfigError,
    build_provider,
)


class TestProviderSelection:
    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "fake_key"}, clear=True)
    def test_default_provider_is_gemini(self):
        # Even with no LLM_PROVIDER set, it defaults to gemini
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}, clear=True):
            provider = build_provider()
        assert isinstance(provider, GeminiChatProvider)

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "fake"}, clear=True)
    def test_fake_provider(self):
        provider = build_provider()
        assert isinstance(provider, FakeLLMProvider)

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}, clear=True)
    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ProviderConfigError, match="unknown LLM_PROVIDER 'unknown'"):
            build_provider()


class TestGeminiConfiguration:
    @mock.patch.dict(os.environ, {"GEMINI_API_KEY": "secret_key"}, clear=True)
    def test_repr_never_exposes_the_key(self):
        provider = GeminiChatProvider()
        rep = repr(provider)
        assert "secret_key" not in rep
        assert "GeminiChatProvider" in rep

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_gemini_missing_key_fails(self):
        with pytest.raises(ProviderConfigError, match="GEMINI_API_KEY is not set"):
            GeminiChatProvider()
