import os
import sys
import json
import logging
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
from src.extraction import prompts as P
from src.extraction.context_builder import ContextBuilder
from src.extraction.extraction_engine import (
    STATUS_EMPTY,
    STATUS_INVALID,
    STATUS_OK,
    STATUS_PROVIDER_ERROR,
    ExtractionEngine,
)
from src.extraction.llm_client import (
    FakeLLMProvider,
    OpenAIChatProvider,
    ProviderCallError,
    ProviderConfigError,
    build_provider,
)
from src.retrieval.registry import DocumentRegistry
from src.schemas.enums import RejectionCode


class NoSearch:
    """The documents used here are small, so they are included whole."""

    def search_field(self, field, doc_filter, top_k, lexical_floor):
        return []


def build_package(group="identity", chunks=None):
    registry = DocumentRegistry()
    registry.register_document(
        {"metadata": {"doc_id": "docA", "document_type": "rfp_main", "source_filename": "Main.pdf"}},
        chunks or [
            {"chunk_id": "ck1", "text": "Solicitation Number ABC-123 issued by Example County."},
            {"chunk_id": "ck2", "text": "Title: Supply of Widgets."},
        ],
    )
    builder = ContextBuilder(registry, NoSearch())
    return registry, builder.build_group("bidX", group)[0]


def good_response(package, field="bid_number", value="ABC-123", quote="Solicitation Number ABC-123"):
    return json.dumps({
        "group": package.group,
        "candidates": [{
            "field": field,
            "value": value,
            "contact": None,
            "evidence": [{"evidence_id": package.excerpts[0].handle, "quote": quote}],
        }],
        "not_found": [],
    })


def no_sleep(_seconds):
    return None


# --- provider abstraction ----------------------------------------------------

class TestProviderAbstraction(unittest.TestCase):
    def test_missing_key_fails_fast_without_network(self):
        saved = os.environ.pop(config.LLM_API_KEY_ENV, None)
        try:
            with self.assertRaises(ProviderConfigError):
                OpenAIChatProvider()
        finally:
            if saved is not None:
                os.environ[config.LLM_API_KEY_ENV] = saved

    def test_placeholder_key_is_rejected(self):
        with self.assertRaises(ProviderConfigError):
            OpenAIChatProvider(api_key="your_openai_api_key_here")

    def test_repr_never_exposes_the_key(self):
        provider = OpenAIChatProvider(api_key="sk-secret-value", model="test-model")
        self.assertNotIn("sk-secret-value", repr(provider))
        self.assertNotIn("sk-secret-value", str(provider.__dict__.get("model", "")))

    def test_unknown_provider_name(self):
        with self.assertRaises(ProviderConfigError):
            build_provider("not-a-provider")

    def test_exhausted_quota_is_detected_as_fatal(self):
        """Regression: a 429 for an empty balance must not be retried.

        Observed live: every call returned 429 insufficient_quota, and treating
        it as transient burned the full retry budget on every group.
        """
        from src.extraction.llm_client import _is_quota_exhausted

        class FakeQuotaError(Exception):
            body = {"error": {"code": "credit_balance_exhausted", "type": "insufficient_quota",
                              "message": "You have no credits remaining."}}

        class FakeRateLimit(Exception):
            body = {"error": {"code": "rate_limit_exceeded", "type": "requests",
                              "message": "Rate limit reached, please slow down."}}

        self.assertTrue(_is_quota_exhausted(FakeQuotaError()))
        self.assertFalse(_is_quota_exhausted(FakeRateLimit()))
        # Also detectable when the SDK exposes only a message string.
        self.assertTrue(_is_quota_exhausted(Exception("429 - insufficient_quota")))

    def test_sdk_retries_are_disabled_so_the_engine_owns_the_budget(self):
        provider = OpenAIChatProvider(api_key="sk-not-a-real-key", model="test-model")
        self.assertEqual(provider._client.max_retries, 0)

    def test_mistral_config_does_not_require_openai_key(self):
        # Regression test: Mistral configuration must not require OPENAI_API_KEY
        with patch.dict(os.environ, {
            config.LLM_PROVIDER_ENV: "mistral",
            config.MISTRAL_API_KEY_ENV: "mistral-key",
            config.MISTRAL_MODEL_ENV: "mistral-small-latest"
        }, clear=True):
            try:
                provider = build_provider()
                self.assertEqual(provider.name, "mistral")
                self.assertEqual(getattr(provider, "model", ""), "mistral-small-latest")
            except Exception as exc:
                self.fail(f"build_provider() raised {type(exc).__name__} unexpectedly!")

    def test_openai_config_still_requires_openai_key(self):
        # Regression test: OpenAI configuration must still require OPENAI_API_KEY
        with patch.dict(os.environ, {
            config.LLM_PROVIDER_ENV: "openai"
        }, clear=True):
            with self.assertRaises(ProviderConfigError) as ctx:
                build_provider()
            self.assertIn(config.LLM_API_KEY_ENV, str(ctx.exception))

    def test_fake_provider_records_calls(self):
        provider = FakeLLMProvider(script=["{}"])
        provider.complete([{"role": "user", "content": "hi"}], max_output_tokens=10)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["max_output_tokens"], 10)


# --- happy path --------------------------------------------------------------

class TestExtractionHappyPath(unittest.TestCase):
    def setUp(self):
        self.registry, self.pkg = build_package()

    def test_exact_prompts_messages_reach_provider(self):
        provider = FakeLLMProvider(script=[good_response(self.pkg)])
        engine = ExtractionEngine(provider, registry=self.registry, sleep=no_sleep)
        engine.extract_group(self.pkg)

        expected = P.render_messages(self.pkg)
        self.assertEqual(provider.calls[0]["messages"], expected)
        # The engine must not invent a second prompt.
        self.assertEqual(len(provider.calls[0]["messages"]), 2)
        self.assertEqual(provider.calls[0]["messages"][0]["content"], P.SYSTEM_PROMPT)

    def test_schema_and_token_budget_come_from_the_package(self):
        provider = FakeLLMProvider(script=[good_response(self.pkg)])
        ExtractionEngine(provider, sleep=no_sleep).extract_group(self.pkg)
        call = provider.calls[0]
        self.assertEqual(call["response_schema"], P.response_schema(self.pkg))
        self.assertEqual(call["max_output_tokens"], self.pkg.max_output_tokens)
        self.assertEqual(call["schema_name"], f"extract_{self.pkg.group}")

    def test_valid_response_is_parsed(self):
        provider = FakeLLMProvider(script=[good_response(self.pkg)])
        result = ExtractionEngine(provider, sleep=no_sleep).extract_group(self.pkg)

        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].field_name, "bid_number")
        self.assertIsNotNone(result.prompt_hash)

    def test_candidates_keep_evidence_handles_for_grounding(self):
        provider = FakeLLMProvider(script=[good_response(self.pkg)])
        result = ExtractionEngine(provider, sleep=no_sleep).extract_group(self.pkg)

        expected_chunk_id = self.pkg.handle_map[self.pkg.excerpts[0].handle]
        evidence = result.candidates[0].candidate.evidence
        self.assertEqual(evidence[0].chunk_id, expected_chunk_id)
        self.assertTrue(evidence[0].quote)

    def test_confidence_is_never_populated(self):
        provider = FakeLLMProvider(script=[good_response(self.pkg)])
        result = ExtractionEngine(provider, sleep=no_sleep).extract_group(self.pkg)
        candidate = result.candidates[0].candidate
        self.assertIsNone(candidate.confidence_score)
        self.assertFalse(candidate.is_superseding)

    def test_empty_package_makes_no_provider_call(self):
        registry = DocumentRegistry()
        pkg = ContextBuilder(registry, NoSearch()).build_group("bidX", "identity")[0]
        provider = FakeLLMProvider(script=["should not be used"])
        result = ExtractionEngine(provider, sleep=no_sleep).extract_group(pkg)

        self.assertEqual(result.status, STATUS_EMPTY)
        self.assertEqual(provider.calls, [])


# --- correction path ---------------------------------------------------------

class TestCorrectionBehaviour(unittest.TestCase):
    def setUp(self):
        self.registry, self.pkg = build_package()

    def test_malformed_json_triggers_one_correction_then_succeeds(self):
        provider = FakeLLMProvider(script=["{not json", good_response(self.pkg)])
        engine = ExtractionEngine(provider, max_correction_retries=1, registry=self.registry, sleep=no_sleep)
        result = engine.extract_group(self.pkg)

        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(provider.calls), 2)

    def test_correction_reuses_render_correction_and_keeps_original_contract(self):
        provider = FakeLLMProvider(script=["{not json", good_response(self.pkg)])
        engine = ExtractionEngine(provider, max_correction_retries=1, registry=self.registry, sleep=no_sleep)
        engine.extract_group(self.pkg)

        first, second = provider.calls[0]["messages"], provider.calls[1]["messages"]
        # Original system + user are resent byte-identical.
        self.assertEqual(second[:2], first)
        self.assertEqual(len(second), 3)
        self.assertEqual(second[2]["role"], "user")
        self.assertTrue(second[2]["content"].startswith("CORRECTION"))

    def test_correction_retries_are_bounded(self):
        provider = FakeLLMProvider(script=["bad", "still bad", "worse"])
        engine = ExtractionEngine(provider, max_correction_retries=1, sleep=no_sleep)
        result = engine.extract_group(self.pkg)

        self.assertEqual(result.status, STATUS_INVALID)
        self.assertEqual(len(provider.calls), 2)  # initial + one correction
        self.assertEqual(result.attempts, 2)

    def test_zero_correction_retries_is_configurable(self):
        provider = FakeLLMProvider(script=["bad", good_response(self.pkg)])
        engine = ExtractionEngine(provider, max_correction_retries=0, sleep=no_sleep)
        result = engine.extract_group(self.pkg)

        self.assertEqual(result.status, STATUS_INVALID)
        self.assertEqual(len(provider.calls), 1)

    def test_parser_failure_is_reported_not_swallowed(self):
        provider = FakeLLMProvider(script=["{not json", "{still not json"])
        engine = ExtractionEngine(provider, max_correction_retries=1, sleep=no_sleep)
        result = engine.extract_group(self.pkg)

        self.assertEqual(result.status, STATUS_INVALID)
        self.assertIsNotNone(result.parsed)
        self.assertEqual(result.parsed.status, "invalid_json")
        self.assertTrue(result.error)
        self.assertEqual(result.candidates, [])
        self.assertEqual(len(result.raw_responses), 2)

    def test_correction_prompt_contains_no_hidden_metadata(self):
        provider = FakeLLMProvider(script=["{not json", good_response(self.pkg)])
        engine = ExtractionEngine(provider, max_correction_retries=1, registry=self.registry, sleep=no_sleep)
        engine.extract_group(self.pkg)

        correction = provider.calls[1]["messages"][2]["content"]
        for chunk_id in self.pkg.handle_map.values():
            self.assertNotIn(chunk_id, correction)
        for literal in P.FORBIDDEN_METADATA_LITERALS:
            self.assertNotIn(literal, correction)
        self.assertNotIn("Main.pdf", correction)


# --- provider failures -------------------------------------------------------

class TestProviderFailureHandling(unittest.TestCase):
    def setUp(self):
        self.registry, self.pkg = build_package()

    def test_transient_error_is_retried_then_succeeds(self):
        provider = FakeLLMProvider(script=[ProviderCallError("429"), good_response(self.pkg)])
        engine = ExtractionEngine(provider, max_transient_retries=2, sleep=no_sleep)
        result = engine.extract_group(self.pkg)

        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(len(provider.calls), 2)

    def test_transient_retries_are_bounded(self):
        provider = FakeLLMProvider(script=[ProviderCallError("boom")] * 10)
        engine = ExtractionEngine(provider, max_transient_retries=2, sleep=no_sleep)
        result = engine.extract_group(self.pkg)

        self.assertEqual(result.status, STATUS_PROVIDER_ERROR)
        self.assertEqual(len(provider.calls), 3)  # initial + 2 retries
        self.assertIn("boom", result.error)

    def test_backoff_delays_are_deterministic(self):
        delays = []
        provider = FakeLLMProvider(script=[ProviderCallError("x")] * 5)
        engine = ExtractionEngine(
            provider, max_transient_retries=3, backoff_factor=2.0, sleep=delays.append
        )
        engine.extract_group(self.pkg)
        self.assertEqual(delays, [1.0, 2.0, 4.0])

    def test_config_error_is_not_retried(self):
        provider = FakeLLMProvider(script=[ProviderConfigError("bad key"), good_response(self.pkg)])
        engine = ExtractionEngine(provider, max_transient_retries=3, sleep=no_sleep)
        result = engine.extract_group(self.pkg)

        self.assertEqual(result.status, STATUS_PROVIDER_ERROR)
        self.assertEqual(len(provider.calls), 1)

    def test_provider_failure_does_not_raise_to_caller(self):
        provider = FakeLLMProvider(script=[ProviderCallError("down")] * 5)
        engine = ExtractionEngine(provider, max_transient_retries=1, sleep=no_sleep)
        result = engine.extract_group(self.pkg)  # must not raise
        self.assertEqual(result.status, STATUS_PROVIDER_ERROR)


# --- safety properties -------------------------------------------------------

class TestSafetyProperties(unittest.TestCase):
    def setUp(self):
        self.registry, self.pkg = build_package()

    def test_unknown_handle_cannot_bypass_the_parser(self):
        fabricated = json.dumps({
            "group": self.pkg.group,
            "candidates": [{
                "field": "bid_number",
                "value": "FAKE-999",
                "contact": None,
                "evidence": [{"evidence_id": "Edeadbeef", "quote": "Solicitation Number FAKE-999"}],
            }],
            "not_found": [],
        })
        provider = FakeLLMProvider(script=[fabricated])
        result = ExtractionEngine(provider, max_correction_retries=0, sleep=no_sleep).extract_group(self.pkg)

        self.assertEqual(result.status, STATUS_OK)  # envelope parsed
        self.assertEqual(result.candidates, [])     # but nothing was accepted
        self.assertEqual(len(result.parsed.rejected), 1)
        self.assertEqual(result.parsed.rejected[0].rejection_code, RejectionCode.unknown_evidence_id)

    def test_fabricated_field_is_rejected(self):
        fabricated = json.dumps({
            "group": self.pkg.group,
            "candidates": [{
                "field": "payment_terms",  # not offered in the identity group
                "value": "Net 30",
                "contact": None,
                "evidence": [{"evidence_id": self.pkg.excerpts[0].handle, "quote": "Net 30"}],
            }],
            "not_found": [],
        })
        provider = FakeLLMProvider(script=[fabricated])
        result = ExtractionEngine(provider, max_correction_retries=0, sleep=no_sleep).extract_group(self.pkg)
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.parsed.rejected[0].rejection_code, RejectionCode.unknown_field)

    def test_engine_adds_no_hidden_document_metadata_to_messages(self):
        provider = FakeLLMProvider(script=[good_response(self.pkg)])
        ExtractionEngine(provider, registry=self.registry, sleep=no_sleep).extract_group(self.pkg)

        sent = "\n".join(m["content"] for m in provider.calls[0]["messages"])
        self.assertNotIn("Main.pdf", sent)
        self.assertNotIn("docA", sent)
        for chunk_id in self.pkg.handle_map.values():
            self.assertNotIn(chunk_id, sent)
        for literal in P.FORBIDDEN_METADATA_LITERALS:
            self.assertNotIn(literal, sent)

    def test_leakage_guard_blocks_the_call(self):
        registry, pkg = build_package(chunks=[{"chunk_id": "ck1", "text": "See Main.pdf for details."}])
        provider = FakeLLMProvider(script=[good_response(pkg)])
        engine = ExtractionEngine(provider, registry=registry, sleep=no_sleep)

        with self.assertRaises(P.PromptLeakageError):
            engine.extract_group(pkg)
        self.assertEqual(provider.calls, [], "no call may be made once leakage is detected")

    def test_api_key_never_appears_in_logs_or_prompts(self):
        secret = "sk-this-must-not-appear"
        os.environ[config.LLM_API_KEY_ENV] = secret
        try:
            provider = FakeLLMProvider(script=["bad", good_response(self.pkg)])
            engine = ExtractionEngine(provider, max_correction_retries=1, registry=self.registry, sleep=no_sleep)
            with self.assertLogs("src.extraction.extraction_engine", level=logging.DEBUG) as logs:
                engine.extract_group(self.pkg)
            self.assertNotIn(secret, "\n".join(logs.output))
            for call in provider.calls:
                self.assertNotIn(secret, "\n".join(m["content"] for m in call["messages"]))
        finally:
            os.environ.pop(config.LLM_API_KEY_ENV, None)


# --- all six groups ----------------------------------------------------------

class TestBidLevelExtraction(unittest.TestCase):
    def test_extract_bid_runs_every_group_in_locked_order(self):
        registry = DocumentRegistry()
        registry.register_document(
            {"metadata": {"doc_id": "docA", "document_type": "rfp_main", "source_filename": "Main.pdf"}},
            [{"chunk_id": "ck1", "text": "Solicitation Number ABC-123 for widgets."}],
        )
        builder = ContextBuilder(registry, NoSearch())

        def respond(messages):
            # Echo an empty but valid envelope for whichever group was asked.
            group_line = [l for l in messages[1]["content"].splitlines() if l.startswith("Group: ")][0]
            return json.dumps({"group": group_line.split("Group: ")[1], "candidates": [], "not_found": []})

        provider = FakeLLMProvider(default=respond)
        results = ExtractionEngine(provider, registry=registry, sleep=no_sleep).extract_bid(builder, "bidX")

        self.assertEqual(list(results.keys()), config.GROUP_ORDER)
        for group, result in results.items():
            self.assertIn(result.status, (STATUS_OK, STATUS_EMPTY), group)


if __name__ == '__main__':
    unittest.main()
