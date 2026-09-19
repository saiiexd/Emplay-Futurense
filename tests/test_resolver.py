import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.resolution.resolver import (
    ROLE_AMENDMENT,
    ROLE_CLARIFICATION,
    TIER_ADDENDUM_AMENDMENT,
    TIER_MAIN,
    TIER_PORTAL,
    CandidateResolver,
    build_bid_record,
    build_diagnostics,
    classify_addendum_candidate,
    to_public_json,
)
from src.retrieval.registry import DocumentRegistry
from src.schemas.candidates import GroundedCandidate, LLMCandidate, LLMEvidenceRef
from src.schemas.enums import FieldStatus, RejectionCode
from src.schemas.field_catalog import FIELD_CATALOG
from src.validation.grounding import GroundingValidator


def make_registry(docs):
    """docs: list of (doc_id, doc_type, addendum_number, [(chunk_id, text), ...])"""
    registry = DocumentRegistry()
    for doc_id, doc_type, addendum_number, chunks in docs:
        metadata = {
            "doc_id": doc_id,
            "document_type": doc_type,
            "source_filename": f"{doc_id}.pdf",
        }
        if addendum_number is not None:
            metadata["addendum_number"] = addendum_number
        registry.register_document(
            {"metadata": metadata, "sections": []},
            [{"chunk_id": cid, "text": text} for cid, text in chunks],
        )
    return registry


def ground(registry, field_name, value, chunk_id, quote):
    """Produce a genuinely grounded candidate via the Stage 4.1 validator."""
    candidate = LLMCandidate(
        value=value,
        evidence=[LLMEvidenceRef(chunk_id=chunk_id, quote=quote, explanation="")],
    )
    return GroundingValidator(registry).validate(field_name, candidate)


class TestGroundingIntegration(unittest.TestCase):
    """Resolution consumes only what Stage 4.1 accepted."""

    def setUp(self):
        self.registry = make_registry([
            ("main", "rfp_main", None, [("c1", "Solicitation Number ABC-123 for widgets.")]),
        ])

    def test_valid_candidate_is_grounded(self):
        grounded = ground(self.registry, "bid_number", "ABC-123", "c1", "Solicitation Number ABC-123")
        self.assertTrue(grounded.is_valid)
        self.assertEqual(grounded.chunk_ids, ["c1"])

    def test_fabricated_quote_is_rejected(self):
        grounded = ground(self.registry, "bid_number", "XYZ-999", "c1", "Solicitation Number XYZ-999")
        self.assertFalse(grounded.is_valid)
        self.assertEqual(grounded.rejection_code, RejectionCode.quote_not_in_chunk)

    def test_unsupported_value_is_rejected(self):
        grounded = ground(self.registry, "bid_number", "ZZZ-000", "c1", "Solicitation Number ABC-123")
        self.assertFalse(grounded.is_valid)
        self.assertEqual(grounded.rejection_code, RejectionCode.value_not_in_quote)

    def test_fabricated_handle_is_rejected(self):
        grounded = ground(self.registry, "bid_number", "ABC-123", "does-not-exist", "Solicitation Number ABC-123")
        self.assertFalse(grounded.is_valid)
        self.assertEqual(grounded.rejection_code, RejectionCode.unknown_evidence_id)

    def test_resolver_ignores_invalid_candidates(self):
        bad = ground(self.registry, "bid_number", "XYZ-999", "c1", "Solicitation Number XYZ-999")
        resolver = CandidateResolver(self.registry)
        resolution = resolver.resolve_field("bid_number", [bad])
        self.assertEqual(resolution.status, FieldStatus.not_found.value)


class TestAmendmentClassification(unittest.TestCase):
    def test_amendment_cue_is_detected(self):
        text = "ADDENDUM No. 2. The new due date for this RFP will be the first Friday of next month."
        role, cue = classify_addendum_candidate(text, "the first Friday of next month")
        self.assertEqual(role, ROLE_AMENDMENT)
        self.assertIsNotNone(cue)

    def test_question_and_answer_is_a_clarification(self):
        text = (
            "1. Does the district require etching on monitors?\n"
            "Answer:\nEtching is required on laptops only.\n"
        )
        role, cue = classify_addendum_candidate(text, "Etching is required on laptops only.")
        self.assertEqual(role, ROLE_CLARIFICATION)
        self.assertIsNone(cue)

    def test_restatement_without_cue_is_not_an_amendment(self):
        text = "This addendum re-issues the specification. Delivery is required at the loading dock."
        role, _ = classify_addendum_candidate(text, "Delivery is required at the loading dock.")
        self.assertNotEqual(role, ROLE_AMENDMENT)


class TestPrecedence(unittest.TestCase):
    def test_amending_addendum_supersedes_main(self):
        registry = make_registry([
            ("main", "rfp_main", None, [("c1", "Solicitation Due 01-JAN-2030 10:00:00")]),
            ("add2", "addendum", 2, [("c2", "The new due date for this RFP will be 15-FEB-2030 14:00:00.")]),
        ])
        candidates = [
            ground(registry, "due_date", "01-JAN-2030 10:00:00", "c1", "Solicitation Due 01-JAN-2030 10:00:00"),
            ground(registry, "due_date", "15-FEB-2030 14:00:00", "c2",
                   "The new due date for this RFP will be 15-FEB-2030 14:00:00."),
        ]
        resolution = CandidateResolver(registry).resolve_field("due_date", candidates)

        self.assertEqual(resolution.status, FieldStatus.found.value)
        self.assertEqual(resolution.value, "15-FEB-2030 14:00:00")
        self.assertEqual(resolution.rule, "addendum_amendment")
        self.assertEqual(resolution.chosen.tier, TIER_ADDENDUM_AMENDMENT)
        # The original value is retained for audit, not discarded.
        self.assertEqual(len(resolution.superseded), 1)
        self.assertEqual(resolution.superseded[0].value, "01-JAN-2030 10:00:00")

    def test_higher_addendum_number_wins_between_amendments(self):
        registry = make_registry([
            ("main", "rfp_main", None, [("c1", "Solicitation Due 01-JAN-2030 10:00:00")]),
            ("add1", "addendum", 1, [("c2", "The new due date for this RFP will be 05-JAN-2030 10:00:00.")]),
            ("add3", "addendum", 3, [("c3", "The new due date for this RFP will be 09-MAR-2030 09:00:00.")]),
        ])
        candidates = [
            ground(registry, "due_date", "01-JAN-2030 10:00:00", "c1", "Solicitation Due 01-JAN-2030 10:00:00"),
            ground(registry, "due_date", "05-JAN-2030 10:00:00", "c2",
                   "The new due date for this RFP will be 05-JAN-2030 10:00:00."),
            ground(registry, "due_date", "09-MAR-2030 09:00:00", "c3",
                   "The new due date for this RFP will be 09-MAR-2030 09:00:00."),
        ]
        resolution = CandidateResolver(registry).resolve_field("due_date", candidates)
        self.assertEqual(resolution.value, "09-MAR-2030 09:00:00")
        self.assertEqual(resolution.chosen.provenance.addendum_number, 3)

    def test_addendum_that_does_not_address_the_field_has_no_effect(self):
        """A restatement inside an addendum must not displace the main document."""
        registry = make_registry([
            ("main", "rfp_main", None, [("c1", "Payment is made Net 30 after invoice receipt.")]),
            ("add1", "addendum", 1, [("c2", "For reference, payment is made Net 45 after invoice receipt.")]),
        ])
        candidates = [
            ground(registry, "payment_terms", "Net 30 after invoice receipt",
                   "c1", "Payment is made Net 30 after invoice receipt."),
            ground(registry, "payment_terms", "Net 45 after invoice receipt",
                   "c2", "For reference, payment is made Net 45 after invoice receipt."),
        ]
        resolution = CandidateResolver(registry).resolve_field("payment_terms", candidates)

        self.assertEqual(resolution.value, "Net 30 after invoice receipt")
        self.assertEqual(resolution.chosen.provenance.document_type.value, "rfp_main")
        self.assertNotEqual(resolution.rule, "addendum_amendment")

    def test_qa_clarification_does_not_change_the_value(self):
        registry = make_registry([
            ("main", "rfp_main", None, [("c1", "All deliveries include white glove services and etching.")]),
            ("add1", "addendum", 1, [("c2",
                "6. Does the district require etching on monitors?\nAnswer:\nEtching is required on laptops only.\n")]),
        ])
        candidates = [
            ground(registry, "installation", "white glove services and etching",
                   "c1", "All deliveries include white glove services and etching."),
        ]
        # The clarification is eligible only where the catalog allows it; use a
        # field that accepts addenda so the clarification path is exercised.
        clarification = ground(registry, "product_specification", "Etching is required on laptops only.",
                               "c2", "Etching is required on laptops only.")
        resolver = CandidateResolver(registry)

        installation = resolver.resolve_field("installation", candidates)
        self.assertEqual(installation.value, "white glove services and etching")

        spec = resolver.resolve_field("product_specification", [clarification])
        self.assertEqual(spec.status, FieldStatus.not_found.value)
        self.assertEqual(spec.rule, "clarification_only")
        self.assertEqual(len(spec.clarifications), 1)

    def test_portal_does_not_override_solicitation(self):
        registry = make_registry([
            ("main", "rfp_main", None, [("c1", "Title: Supply of Classroom Furniture")]),
            ("portal", "portal_listing", None, [("c2", "Title Supply of Furniture **LISTING #99**")]),
        ])
        candidates = [
            ground(registry, "title", "Supply of Classroom Furniture", "c1", "Title: Supply of Classroom Furniture"),
            ground(registry, "title", "Supply of Furniture", "c2", "Title Supply of Furniture"),
        ]
        resolution = CandidateResolver(registry).resolve_field("title", candidates)
        self.assertEqual(resolution.value, "Supply of Classroom Furniture")
        self.assertEqual(resolution.chosen.tier, TIER_MAIN)

    def test_portal_is_used_when_it_is_the_only_eligible_source(self):
        registry = make_registry([
            ("portal", "portal_listing", None, [("c1", "Issuing Organization Example County Schools")]),
        ])
        candidates = [ground(registry, "company_name", "Example County Schools", "c1",
                             "Issuing Organization Example County Schools")]
        resolution = CandidateResolver(registry).resolve_field("company_name", candidates)
        self.assertEqual(resolution.value, "Example County Schools")
        self.assertEqual(resolution.chosen.tier, TIER_PORTAL)
        self.assertEqual(resolution.rule, "portal_only")

    def test_ineligible_document_type_cannot_win(self):
        """installation accepts only the main document per the catalog."""
        registry = make_registry([
            ("portal", "portal_listing", None, [("c1", "Installation is included at no charge.")]),
        ])
        grounded = ground(registry, "installation", "Installation is included at no charge.",
                          "c1", "Installation is included at no charge.")
        self.assertFalse(grounded.is_valid)  # grounding already blocks it
        resolution = CandidateResolver(registry).resolve_field("installation", [grounded])
        self.assertEqual(resolution.status, FieldStatus.not_found.value)

    def test_same_tier_disagreement_is_reported_as_conflicting(self):
        registry = make_registry([
            ("main", "rfp_main", None, [
                ("c1", "Payment terms are Net 30 days."),
                ("c2", "Payment terms are Net 60 days."),
            ]),
        ])
        candidates = [
            ground(registry, "payment_terms", "Net 30 days", "c1", "Payment terms are Net 30 days."),
            ground(registry, "payment_terms", "Net 60 days", "c2", "Payment terms are Net 60 days."),
        ]
        resolution = CandidateResolver(registry).resolve_field("payment_terms", candidates)
        self.assertEqual(resolution.status, FieldStatus.conflicting.value)
        self.assertTrue(resolution.conflicts)

    def test_duplicate_values_do_not_create_a_conflict(self):
        registry = make_registry([
            ("main", "rfp_main", None, [
                ("c1", "Payment terms are Net 30 days."),
                ("c2", "Again: payment terms are Net 30 days."),
            ]),
        ])
        candidates = [
            ground(registry, "payment_terms", "Net 30 days", "c1", "Payment terms are Net 30 days."),
            ground(registry, "payment_terms", "Net 30 days", "c2", "Again: payment terms are Net 30 days."),
        ]
        resolution = CandidateResolver(registry).resolve_field("payment_terms", candidates)
        self.assertEqual(resolution.status, FieldStatus.found.value)
        self.assertEqual(resolution.conflicts, [])

    def test_aggregated_field_keeps_every_contributor(self):
        registry = make_registry([
            ("main", "rfp_main", None, [
                ("c1", "Submit the W9 Form with your response."),
                ("c2", "Submit the MWBE Forms with your response."),
            ]),
        ])
        candidates = [
            ground(registry, "additional_documentation", "W9 Form", "c1", "Submit the W9 Form with your response."),
            ground(registry, "additional_documentation", "MWBE Forms", "c2", "Submit the MWBE Forms with your response."),
        ]
        resolution = CandidateResolver(registry).resolve_field("additional_documentation", candidates)
        self.assertEqual(resolution.status, FieldStatus.found.value)
        self.assertIn("W9 Form", resolution.value)
        self.assertIn("MWBE Forms", resolution.value)
        self.assertEqual(len(resolution.contributing), 2)

    def test_aggregated_field_spans_document_tiers(self):
        """A required-forms list legitimately lives in several documents, so a
        lower-tier form must not be dropped because the main document also
        names one."""
        registry = make_registry([
            ("main", "rfp_main", None, [("c1", "Submit the Mercury Affidavit with your response.")]),
            ("aff", "affidavit", None, [("c2", "This Contract Affidavit must accompany the bid.")]),
        ])
        candidates = [
            ground(registry, "additional_documentation", "Mercury Affidavit", "c1",
                   "Submit the Mercury Affidavit with your response."),
            ground(registry, "additional_documentation", "Contract Affidavit", "c2",
                   "This Contract Affidavit must accompany the bid."),
        ]
        resolution = CandidateResolver(registry).resolve_field("additional_documentation", candidates)

        self.assertEqual(resolution.rule, "aggregated_all_tiers")
        self.assertIn("Mercury Affidavit", resolution.value)
        self.assertIn("Contract Affidavit", resolution.value)

    def test_amending_addendum_replaces_an_aggregated_list(self):
        registry = make_registry([
            ("main", "rfp_main", None, [("c1", "Submit the W9 Form with your response.")]),
            ("add1", "addendum", 1, [("c2",
                "The attachment list is hereby amended: submit the Revised Pricing Sheet.")]),
        ])
        candidates = [
            ground(registry, "additional_documentation", "W9 Form", "c1",
                   "Submit the W9 Form with your response."),
            ground(registry, "additional_documentation", "Revised Pricing Sheet", "c2",
                   "The attachment list is hereby amended: submit the Revised Pricing Sheet."),
        ]
        resolution = CandidateResolver(registry).resolve_field("additional_documentation", candidates)

        self.assertEqual(resolution.rule, "addendum_amendment_aggregated")
        self.assertIn("Revised Pricing Sheet", resolution.value)
        self.assertNotIn("W9 Form", resolution.value)
        self.assertTrue(resolution.superseded)

    def test_resolution_is_deterministic(self):
        registry = make_registry([
            ("main", "rfp_main", None, [("c1", "Solicitation Number ABC-123 for widgets.")]),
        ])
        candidates = [ground(registry, "bid_number", "ABC-123", "c1", "Solicitation Number ABC-123")]
        resolver = CandidateResolver(registry)
        first = resolver.resolve_field("bid_number", candidates)
        second = resolver.resolve_field("bid_number", candidates)
        self.assertEqual(first.value, second.value)
        self.assertEqual(first.chosen.grounded.candidate_id, second.chosen.grounded.candidate_id)


class TestFinalAssembly(unittest.TestCase):
    def setUp(self):
        self.registry = make_registry([
            ("main", "rfp_main", None, [
                ("c1", "Solicitation Number ABC-123 for widgets."),
                ("c2", "Manufacturer Name Acme Industrial"),
            ]),
        ])
        self.resolver = CandidateResolver(self.registry)

    def _resolve(self, candidates):
        by_field = {}
        for grounded in candidates:
            by_field.setdefault(grounded.field_name, []).append(grounded)
        return self.resolver.resolve_bid(by_field)

    def test_all_twenty_fields_present_in_public_json(self):
        resolutions = self._resolve([])
        record = build_bid_record("bidX", resolutions, self.registry)
        public = to_public_json(record)

        expected_labels = [spec.assignment_label for spec in FIELD_CATALOG.values()]
        self.assertEqual(list(public.keys()), expected_labels)
        self.assertEqual(len(public), 20)

    def test_missing_fields_become_null_not_invented(self):
        record = build_bid_record("bidX", self._resolve([]), self.registry)
        public = to_public_json(record)
        self.assertTrue(all(value is None for value in public.values()))
        self.assertEqual(record.summary.found_fields, 0)
        self.assertEqual(record.summary.missing_fields, 20)

    def test_found_field_keeps_evidence_and_value(self):
        candidates = [ground(self.registry, "bid_number", "ABC-123", "c1", "Solicitation Number ABC-123")]
        resolutions = self._resolve(candidates)
        record = build_bid_record("bidX", resolutions, self.registry)

        field = record.fields.bid_number
        self.assertEqual(field.status, FieldStatus.found)
        self.assertEqual(field.normalized_value, "ABC-123")
        self.assertEqual(field.evidence.chunk_ids, ["c1"])
        self.assertTrue(field.evidence.document_refs)
        self.assertEqual(to_public_json(record)["Bid Number"], "ABC-123")

    def test_manufacturer_is_not_emitted_as_company_name(self):
        candidates = [ground(self.registry, "mfg_for_registration", "Acme Industrial",
                             "c2", "Manufacturer Name Acme Industrial")]
        record = build_bid_record("bidX", self._resolve(candidates), self.registry)
        public = to_public_json(record)

        self.assertEqual(public["MFG for Registration"], ["Acme Industrial"])
        self.assertIsNone(public["company_name"])

    def test_model_and_part_numbers_stay_separate(self):
        registry = make_registry([
            ("main", "rfp_main", None, [("c1", "Model XPS-15 with part number PN-7788.")]),
        ])
        resolver = CandidateResolver(registry)
        by_field = {
            "model_no": [ground(registry, "model_no", "XPS-15", "c1", "Model XPS-15 with part number PN-7788.")],
        }
        record = build_bid_record("bidX", resolver.resolve_bid(by_field), registry)
        public = to_public_json(record)

        self.assertEqual(public["Model_no"], ["XPS-15"])
        self.assertIsNone(public["Part_no"])

    def test_superseded_field_records_the_prior_value(self):
        registry = make_registry([
            ("main", "rfp_main", None, [("c1", "Solicitation Due 01-JAN-2030 10:00:00")]),
            ("add2", "addendum", 2, [("c2", "The new due date for this RFP will be 15-FEB-2030 14:00:00.")]),
        ])
        resolver = CandidateResolver(registry)
        by_field = {"due_date": [
            ground(registry, "due_date", "01-JAN-2030 10:00:00", "c1", "Solicitation Due 01-JAN-2030 10:00:00"),
            ground(registry, "due_date", "15-FEB-2030 14:00:00", "c2",
                   "The new due date for this RFP will be 15-FEB-2030 14:00:00."),
        ]}
        record = build_bid_record("bidX", resolver.resolve_bid(by_field), registry)

        field = record.fields.due_date
        self.assertTrue(field.is_superseded)
        self.assertIn("01-JAN-2030", field.supersession.reason)
        self.assertEqual(to_public_json(record)["Due Date"], "15-FEB-2030 14:00:00")

    def test_dates_are_not_rewritten_by_normalization(self):
        """Historic solicitations stay as written; no current-date reasoning."""
        registry = make_registry([
            ("main", "rfp_main", None, [("c1", "Solicitation Due 27-JUN-2024 14:00:00")]),
        ])
        resolver = CandidateResolver(registry)
        by_field = {"due_date": [ground(registry, "due_date", "27-JUN-2024 14:00:00",
                                        "c1", "Solicitation Due 27-JUN-2024 14:00:00")]}
        record = build_bid_record("bidX", resolver.resolve_bid(by_field), registry)
        self.assertEqual(to_public_json(record)["Due Date"], "27-JUN-2024 14:00:00")

    def test_diagnostics_are_auditable(self):
        candidates = [ground(self.registry, "bid_number", "ABC-123", "c1", "Solicitation Number ABC-123")]
        resolutions = self._resolve(candidates)
        diagnostics = build_diagnostics("bidX", resolutions, [])

        entry = diagnostics["fields"]["bid_number"]
        self.assertEqual(entry["status"], FieldStatus.found.value)
        self.assertEqual(entry["chosen"]["value"], "ABC-123")
        self.assertEqual(entry["chosen"]["document_type"], "rfp_main")
        self.assertEqual(entry["chosen"]["chunk_ids"], ["c1"])
        self.assertTrue(entry["chosen"]["quotes"])
        self.assertIsNotNone(entry["chosen"]["normalized"])

    def test_diagnostics_retain_rejected_candidates(self):
        rejected = [ground(self.registry, "bid_number", "XYZ-999", "c1", "Solicitation Number XYZ-999")]
        diagnostics = build_diagnostics("bidX", self._resolve([]), rejected)
        self.assertEqual(len(diagnostics["rejected_candidates"]), 1)
        self.assertEqual(diagnostics["rejected_candidates"][0]["rejection_code"], "quote_not_in_chunk")


if __name__ == '__main__':
    unittest.main()
