"""Offline pipeline tests.

The LLM is always a deterministic fake. Where the supplied corpus is available,
the tests exercise the real ingestion/retrieval/grounding/resolution chain over
real documents; expected corpus literals live in tests/fixtures.
"""

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
from src.extraction.extraction_engine import ExtractionEngine
from src.extraction.llm_client import FakeLLMProvider
from src.extraction.pipeline import (  # noqa: E402
    build_bid_context,
    group_by_field,
    ground_group_results,
    run_pipeline,
    write_outputs,
)
from src.resolution.resolver import CandidateResolver, to_public_json, build_bid_record  # noqa: E402
from src.schemas.candidates import LLMCandidate, LLMEvidenceRef  # noqa: E402
from src.schemas.enums import FieldStatus  # noqa: E402
from src.schemas.field_catalog import FIELD_CATALOG  # noqa: E402
from src.validation.canonical import find_canonical_match  # noqa: E402
from src.validation.grounding import GroundingValidator  # noqa: E402

from tests.fixtures.corpus_expectations import BID1, BID2  # noqa: E402
from tests.fixtures.offline_provider import SourceBackedOfflineResponder  # noqa: E402

BID1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Bid1"))
BID2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Bid2"))


def no_sleep(_seconds):
    return None


def find_chunk(registry, marker):
    """First chunk (in stable id order) whose text contains the marker."""
    for chunk_id in sorted(registry.chunk_metadata):
        text = registry.chunk_metadata[chunk_id].get("text", "")
        if marker in text:
            return chunk_id, text
    return None, None


def quote_around(text, marker, trailing=60):
    """An exact substring of the chunk, starting at the marker."""
    start = text.index(marker)
    return text[start: start + len(marker) + trailing]


def evidence_responder(messages):
    """Answer each field with a real line from an excerpt it was actually given."""
    user = messages[1]["content"]
    group = next(l for l in user.splitlines() if l.startswith("Group: ")).split("Group: ")[1]
    fields = re.findall(r'^\[(\w+)\] kind:', user, re.M)
    blocks = re.findall(
        r'<excerpt id="(E[0-9a-f]+)"[^>]*fields="([^"]*)">\n(.*?)\n</excerpt>', user, re.DOTALL
    )

    candidates, not_found = [], []
    for field_name in fields:
        picked = None
        for handle, eligible, text in blocks:
            if field_name not in eligible.split(","):
                continue
            line = next((l.strip() for l in text.splitlines() if len(l.strip()) > 25), None)
            if line:
                picked = (handle, line[:200])
                break
        if picked and field_name != "contact_info":
            candidates.append({
                "field": field_name, "value": picked[1], "contact": None,
                "evidence": [{"evidence_id": picked[0], "quote": picked[1]}],
            })
        else:
            not_found.append({"field": field_name, "reason_code": "not_stated", "reason": "fake"})
    return json.dumps({"group": group, "candidates": candidates, "not_found": not_found})


def fake_engine_factory(registry):
    return ExtractionEngine(
        FakeLLMProvider(default=evidence_responder), registry=registry, sleep=no_sleep
    )


def source_backed_engine_factory(bid_id, providers):
    """Build the real extraction engine with a test-only source-backed provider."""
    def factory(registry):
        responder = SourceBackedOfflineResponder(bid_id)
        provider = FakeLLMProvider(default=responder)
        providers.append(provider)
        return ExtractionEngine(provider, registry=registry, sleep=no_sleep)

    return factory


@unittest.skipUnless(os.path.isdir(BID1_DIR), "Bid1 corpus not available")
class TestAddendumRegressionOnRealCorpus(unittest.TestCase):
    """The amended-due-date case, driven by real document provenance.

    The expected literals come from tests/fixtures; production code never sees
    them. What is asserted is that an addendum carrying explicit amendment
    language wins over the main solicitation for that field.
    """

    @classmethod
    def setUpClass(cls):
        cls.context = build_bid_context(BID1_DIR, embeddings="fake")
        cls.registry = cls.context.registry

    def _grounded(self, field_name, value, chunk_id, quote):
        candidate = LLMCandidate(
            value=value, evidence=[LLMEvidenceRef(chunk_id=chunk_id, quote=quote, explanation="")]
        )
        return GroundingValidator(self.registry).validate(field_name, candidate)

    def test_addendum_metadata_is_available_to_the_resolver(self):
        numbers = {
            ref.addendum_number
            for ref in self.registry.doc_refs.values()
            if ref.document_type.value == "addendum"
        }
        self.assertIn(BID1["addendum_number_that_amends"], numbers)

    def test_amended_due_date_wins_over_original(self):
        main_chunk, main_text = find_chunk(self.registry, BID1["main_due_date_marker"])
        add_chunk, add_text = find_chunk(self.registry, BID1["addendum_due_date_marker"])
        self.assertIsNotNone(main_chunk, "original due date evidence missing from corpus")
        self.assertIsNotNone(add_chunk, "amended due date evidence missing from corpus")

        original = self._grounded(
            "due_date", BID1["main_due_date_value"], main_chunk,
            quote_around(main_text, BID1["main_due_date_marker"]),
        )
        amended = self._grounded(
            "due_date", BID1["addendum_due_date_value"], add_chunk,
            quote_around(add_text, BID1["addendum_due_date_marker"]),
        )
        self.assertTrue(original.is_valid, original.rejection_reason)
        self.assertTrue(amended.is_valid, amended.rejection_reason)

        resolver = CandidateResolver(self.registry, doc_label_for=self.context.builder.doc_label_for)
        resolution = resolver.resolve_field("due_date", [original, amended])

        self.assertEqual(resolution.rule, "addendum_amendment")
        self.assertIn(BID1["addendum_due_date_value"], str(resolution.value))
        self.assertEqual(
            resolution.chosen.provenance.addendum_number, BID1["addendum_number_that_amends"]
        )
        # Both candidates survive for audit; the original is marked superseded.
        self.assertEqual(len(resolution.considered), 2)
        self.assertTrue(resolution.superseded)
        self.assertIn(BID1["main_due_date_value"], str(resolution.superseded[0].value))

    def test_original_due_date_stands_when_no_addendum_amends_it(self):
        """Same corpus, addendum evidence withheld: the main document wins."""
        main_chunk, main_text = find_chunk(self.registry, BID1["main_due_date_marker"])
        original = self._grounded(
            "due_date", BID1["main_due_date_value"], main_chunk,
            quote_around(main_text, BID1["main_due_date_marker"]),
        )
        resolution = CandidateResolver(self.registry).resolve_field("due_date", [original])

        self.assertEqual(resolution.rule, "highest_authority")
        self.assertIn(BID1["main_due_date_value"], str(resolution.value))
        self.assertFalse(resolution.superseded)

    def test_question_and_answer_addendum_text_is_a_clarification(self):
        chunk_id, text = find_chunk(self.registry, BID1["addendum_qa_marker"])
        self.assertIsNotNone(chunk_id, "Q&A addendum evidence missing from corpus")

        grounded = self._grounded(
            "product_specification", BID1["addendum_qa_marker"], chunk_id,
            quote_around(text, BID1["addendum_qa_marker"], trailing=0),
        )
        self.assertTrue(grounded.is_valid, grounded.rejection_reason)

        resolution = CandidateResolver(self.registry).resolve_field("product_specification", [grounded])
        self.assertEqual(resolution.rule, "clarification_only")
        self.assertEqual(resolution.status, FieldStatus.not_found.value)
        self.assertEqual(len(resolution.clarifications), 1)


@unittest.skipUnless(os.path.isdir(BID2_DIR), "Bid2 corpus not available")
class TestFullPipelineOffline(unittest.TestCase):
    """Whole pipeline over real documents with a deterministic fake model."""

    @classmethod
    def setUpClass(cls):
        cls.result = run_pipeline(BID2_DIR, fake_engine_factory, embeddings="fake")

    def test_every_group_was_extracted(self):
        self.assertEqual(list(self.result.group_results.keys()), config.GROUP_ORDER)
        for group, group_result in self.result.group_results.items():
            self.assertIn(group_result.status, ("ok", "empty_context"), group)

    def test_public_json_has_exactly_the_twenty_assignment_fields(self):
        public = self.result.public_json
        self.assertEqual(
            list(public.keys()), [spec.assignment_label for spec in FIELD_CATALOG.values()]
        )
        self.assertEqual(len(public), 20)

    def test_every_resolved_value_is_backed_by_real_chunk_text(self):
        registry = build_bid_context(BID2_DIR, embeddings="fake").registry
        checked = 0
        for field_name, resolution in self.result.resolutions.items():
            if resolution.status == FieldStatus.not_found.value:
                continue
            for candidate in resolution.contributing:
                for chunk_id, quote in zip(candidate.provenance.chunk_ids, candidate.provenance.quotes):
                    chunk = registry.chunk_metadata.get(chunk_id)
                    self.assertIsNotNone(chunk, f"{field_name} cites an unknown chunk")
                    self.assertTrue(
                        find_canonical_match(quote, chunk["text"]).matched,
                        f"{field_name} quote is not present in its chunk",
                    )
                    checked += 1
        self.assertGreater(checked, 0, "expected at least one grounded field")

    def test_unresolved_fields_are_null_and_never_invented(self):
        for field_name, resolution in self.result.resolutions.items():
            label = FIELD_CATALOG[field_name].assignment_label
            if resolution.status == FieldStatus.not_found.value:
                self.assertIsNone(self.result.public_json[label])

    def test_diagnostics_carry_provenance_for_every_resolved_field(self):
        for field_name, entry in self.result.diagnostics["fields"].items():
            if entry["status"] == FieldStatus.not_found.value:
                continue
            self.assertIsNotNone(entry["chosen"], field_name)
            self.assertTrue(entry["chosen"]["doc_label"], field_name)
            self.assertTrue(entry["chosen"]["chunk_ids"], field_name)
            self.assertIn("rule", entry)

    def test_outputs_are_written_without_credentials(self):
        secret = "sk-must-not-be-written"
        os.environ[config.GEMINI_API_KEY_ENV] = secret
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                paths = write_outputs(self.result, out_dir)
                public_blob = open(paths["public"], encoding="utf-8").read()
                diag_blob = open(paths["diagnostics"], encoding="utf-8").read()
                self.assertNotIn(secret, public_blob)
                self.assertNotIn(secret, diag_blob)
                self.assertEqual(len(json.loads(public_blob)), 20)
        finally:
            os.environ.pop(config.GEMINI_API_KEY_ENV, None)

    def test_grounding_rejects_fabricated_candidates_from_the_engine(self):
        """A model answer citing a handle it was never given cannot reach resolution."""
        registry = build_bid_context(BID2_DIR, embeddings="fake").registry
        fabricated = LLMCandidate(
            value="Totally Invented Value",
            evidence=[LLMEvidenceRef(chunk_id="not-a-real-chunk", quote="Totally Invented Value",
                                     explanation="")],
        )
        grounded = GroundingValidator(registry).validate("bid_number", fabricated)
        self.assertFalse(grounded.is_valid)

        resolution = CandidateResolver(registry).resolve_field("bid_number", [grounded])
        self.assertEqual(resolution.status, FieldStatus.not_found.value)


@unittest.skipUnless(os.path.isdir(BID1_DIR) and os.path.isdir(BID2_DIR), "bid corpora not available")
class TestBothBidsCompleteOfflinePipeline(unittest.TestCase):
    """Both supplied corpora complete all six real extraction groups offline."""

    @classmethod
    def setUpClass(cls):
        cls.results = {}
        cls.providers = {}
        for bid_id, bid_dir in (("Bid1", BID1_DIR), ("Bid2", BID2_DIR)):
            providers = []
            cls.results[bid_id] = run_pipeline(
                bid_dir,
                source_backed_engine_factory(bid_id, providers),
                embeddings="fake",
            )
            cls.providers[bid_id] = providers[0]

    def test_all_six_groups_execute_for_both_bids(self):
        for bid_id, result in self.results.items():
            self.assertEqual(list(result.group_results), config.GROUP_ORDER, bid_id)
            self.assertEqual(len(self.providers[bid_id].calls), 6, bid_id)
            for group, group_result in result.group_results.items():
                self.assertEqual(group_result.status, "ok", f"{bid_id}/{group}: {group_result.error}")

    def test_outputs_have_the_exact_assignment_contract(self):
        expected_labels = [spec.assignment_label for spec in FIELD_CATALOG.values()]
        for bid_id, result in self.results.items():
            self.assertEqual(list(result.public_json), expected_labels, bid_id)
            self.assertEqual(len(result.public_json), 20)

    def test_every_non_null_output_value_is_string_or_null(self):
        def flat_value(value):
            if value is None or isinstance(value, str):
                return value
            if isinstance(value, list):
                if value and all(isinstance(item, dict) for item in value):
                    contacts = []
                    for contact in value:
                        parts = [contact.get("phone"), contact.get("email")]
                        details = "; ".join(part for part in parts if part)
                        contacts.append(f"{contact.get('name')} ({details})")
                    return "; ".join(contacts)
                return "; ".join(str(item) for item in value)
            self.fail(f"unexpected structured output value: {value!r}")

        for bid_id, result in self.results.items():
            for label, value in result.public_json.items():
                self.assertTrue(
                    isinstance(flat_value(value), (str, type(None))),
                    f"{bid_id}/{label}: {value!r}",
                )

    def test_retrieval_context_grounding_and_resolution_ran(self):
        for bid_id, result in self.results.items():
            self.assertGreater(len(result.grounded), 0, bid_id)
            self.assertTrue(any(r.contributing for r in result.resolutions.values()), bid_id)
            for group_result in result.group_results.values():
                self.assertTrue(group_result.parsed.coverage, bid_id)


class TestFailedRunWritesNoArtifact(unittest.TestCase):
    """A provider failure must never produce a null-filled 'result' file."""

    def test_cli_refuses_to_write_when_every_group_fails(self):
        from unittest.mock import patch

        import main as cli
        from src.extraction.llm_client import FakeLLMProvider, ProviderConfigError

        # A provider whose every call fails the way a rejected credential does.
        provider = FakeLLMProvider(script=[ProviderConfigError("credential rejected")] * 50)

        with tempfile.TemporaryDirectory() as out_dir:
            argv = ["main.py", "--bid", BID2_DIR, "--out", out_dir]
            with patch.object(sys, "argv", argv), \
                 patch.object(cli, "build_provider", return_value=provider):
                exit_code = cli.main()

            produced = [f for f in os.listdir(out_dir) if f.endswith(".json")]
            self.assertEqual(produced, [], "a failed run must not write JSON artifacts")
            self.assertEqual(exit_code, 3, "a failed run must not exit 0")


class TestPipelineHelpers(unittest.TestCase):
    def test_group_by_field_partitions_candidates(self):
        from src.schemas.candidates import GroundedCandidate

        a = GroundedCandidate(candidate_id="1", field_name="title", raw_value="x", chunk_ids=[], excerpts=[])
        b = GroundedCandidate(candidate_id="2", field_name="title", raw_value="y", chunk_ids=[], excerpts=[])
        c = GroundedCandidate(candidate_id="3", field_name="bid_number", raw_value="z", chunk_ids=[], excerpts=[])
        grouped = group_by_field([a, b, c])
        self.assertEqual(len(grouped["title"]), 2)
        self.assertEqual(len(grouped["bid_number"]), 1)


if __name__ == '__main__':
    unittest.main()
