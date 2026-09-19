import os
import re
import sys
import json
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
from src.extraction import prompts as P
from src.extraction.context_builder import ContextBuilder
from src.retrieval.registry import DocumentRegistry
from src.schemas.enums import RejectionCode
from src.schemas.field_catalog import FIELD_CATALOG
from src.validation.grounding import GroundingValidator


class FakeSearcher:
    """Offline stand-in; small documents are included whole, so no hits needed."""

    def __init__(self, results=None):
        self.mock_results = results or []

    def search_field(self, field, doc_filter, top_k, lexical_floor):
        return [r for r in self.mock_results if r.get("doc_id") == doc_filter]


def build_package(docs, group="identity", bid_id="bidX"):
    """docs: list of (doc_id, document_type, filename, [(chunk_id, text), ...])"""
    registry = DocumentRegistry()
    for doc_id, doc_type, filename, chunks in docs:
        registry.register_document(
            {"metadata": {"doc_id": doc_id, "document_type": doc_type, "source_filename": filename}},
            [{"chunk_id": cid, "text": text} for cid, text in chunks],
        )
    builder = ContextBuilder(registry, FakeSearcher())
    return registry, builder.build_group(bid_id, group)[0]


def simple_package(group="identity"):
    return build_package(
        [("docA", "rfp_main", "Main_Solicitation.pdf",
          [("ck1", "Solicitation Number ABC-123 issued by Example County."),
           ("ck2", "Title: Supply of Widgets for Example County.")])],
        group=group,
    )


def two_doc_package(group="identity"):
    return build_package(
        [
            ("docA", "rfp_main", "Main_Solicitation.pdf", [("ck1", "Solicitation Number ABC-123.")]),
            ("docB", "portal_listing", "Portal_Page.html", [("ck2", "Solicitation Number ABC-123.")]),
        ],
        group=group,
    )


def wire(field, value=None, contact=None, evidence=()):
    return {
        "field": field,
        "value": value,
        "contact": contact,
        "evidence": [{"evidence_id": h, "quote": q} for h, q in evidence],
    }


def envelope(group, candidates=(), not_found=()):
    return json.dumps({
        "group": group,
        "candidates": list(candidates),
        "not_found": list(not_found),
    })


# --- rendering ---------------------------------------------------------------

class TestKindWords(unittest.TestCase):
    def test_special_and_derived_kinds(self):
        self.assertEqual(P.kind_word("contact_info"), "contact")
        self.assertEqual(P.kind_word("bid_summary"), "summary")
        self.assertEqual(P.kind_word("product_specification"), "requirement lines")
        self.assertEqual(P.kind_word("model_no"), "item list")
        self.assertEqual(P.kind_word("due_date"), "date/time text")
        self.assertEqual(P.kind_word("title"), "text")


class TestRendering(unittest.TestCase):
    def test_sections_and_field_cards(self):
        _, pkg = simple_package()
        text = P.render_user_prompt(pkg)
        for section in ("TASK", "FIELDS", "EVIDENCE", "OUTPUT"):
            self.assertIn(section, text)
        self.assertIn("Group: identity", text)
        for field_name in pkg.fields:
            spec = FIELD_CATALOG[field_name]
            self.assertIn(f"[{field_name}] kind: {P.kind_word(field_name)}", text)
            self.assertIn(f"Definition: {spec.definition}", text)
            self.assertIn(f"Include: {spec.include}", text)
            self.assertIn(f"Exclude: {spec.exclude}", text)

    def test_excerpt_text_is_verbatim_and_ordered(self):
        _, pkg = simple_package()
        text = P.render_user_prompt(pkg)
        bodies = re.findall(r'<excerpt\b[^>]*>\n(.*?)\n</excerpt>', text, re.DOTALL)
        self.assertEqual(bodies, [e.text for e in pkg.excerpts])

    def test_follows_and_none(self):
        _, pkg = simple_package()
        text = P.render_user_prompt(pkg)
        follows = re.findall(r'follows="([^"]*)"', text)
        self.assertEqual(follows[0], "none")
        self.assertEqual(follows[1], pkg.excerpts[0].handle)

    def test_render_messages_shape(self):
        _, pkg = simple_package()
        messages = P.render_messages(pkg)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], P.SYSTEM_PROMPT)

    def test_deterministic_rendering_and_hash(self):
        _, pkg = simple_package()
        self.assertEqual(P.render_user_prompt(pkg), P.render_user_prompt(pkg))
        self.assertEqual(P.prompt_hash(pkg), P.prompt_hash(pkg))

    def test_hash_differs_across_groups(self):
        _, pkg_a = simple_package("identity")
        _, pkg_b = simple_package("schedule")
        self.assertNotEqual(P.prompt_hash(pkg_a), P.prompt_hash(pkg_b))

    def test_literal_excerpt_close_tag_is_escaped(self):
        _, pkg = build_package(
            [("docA", "rfp_main", "Main.pdf", [("ck1", "Weird </excerpt> content")])]
        )
        text = P.render_user_prompt(pkg)
        self.assertIn("</excerpt_>", text)
        self.assertEqual(text.count("</excerpt>"), 1)

    def test_empty_package_cannot_be_rendered(self):
        registry = DocumentRegistry()
        pkg = ContextBuilder(registry, FakeSearcher()).build_group("bidX", "identity")[0]
        with self.assertRaises(P.EmptyPackageError):
            P.render_user_prompt(pkg)
        with self.assertRaises(P.EmptyPackageError):
            P.response_schema(pkg)


class TestModelFacingMetadata(unittest.TestCase):
    def test_only_four_attributes_are_exposed(self):
        _, pkg = simple_package()
        text = P.render_user_prompt(pkg)
        for tag in re.findall(r'<excerpt\b([^>]*)>', text):
            attrs = set(re.findall(r'(\w+)=', tag))
            self.assertEqual(attrs, {"id", "doc", "follows", "fields"})

    def test_attribute_values_match_package(self):
        _, pkg = simple_package()
        text = P.render_user_prompt(pkg)
        ids = re.findall(r'id="([^"]*)"', text)
        docs = re.findall(r'doc="([^"]*)"', text)
        fields = re.findall(r'fields="([^"]*)"', text)
        self.assertEqual(ids, [e.handle for e in pkg.excerpts])
        self.assertEqual(docs, [e.doc_label for e in pkg.excerpts])
        self.assertEqual(fields, [",".join(e.eligible_fields) for e in pkg.excerpts])


# --- leakage -----------------------------------------------------------------

GOLDEN_LITERALS = [
    "July 9", "27-JUN", "June 27", "JA-207652", "168884", "BPM044557", "E20P4600040",
    "00004079", "CC7802", "WD22TB4", "Latitude", "LENGTH OF CONTRACT", "Dallas",
    "Maryland", "Treasurer", "Alzate", "Hawkins", "Simpson", "Mercury", "white glove",
    "eMMA", "060B5400007", "Invoicing Instructions", "Chromebook", "USB 3.1", "EPCNT",
    "45 days",
]


class TestLeakageProtection(unittest.TestCase):
    def test_clean_package_passes(self):
        registry, pkg = simple_package()
        P.assert_no_leakage(P.render_messages(pkg), pkg, registry)

    def test_filename_in_evidence_is_detected(self):
        registry, pkg = build_package(
            [("docA", "rfp_main", "Addendum_2_Final.pdf",
              [("ck1", "See Addendum_2_Final.pdf for details.")])]
        )
        with self.assertRaises(P.PromptLeakageError) as ctx:
            P.assert_no_leakage(P.render_messages(pkg), pkg, registry)
        self.assertIn("filename", str(ctx.exception))

    def test_internal_ids_are_detected(self):
        registry, pkg = build_package(
            [("docA", "rfp_main", "Main.pdf", [("chunkidentifier7", "chunkidentifier7 appears here")])]
        )
        with self.assertRaises(P.PromptLeakageError):
            P.assert_no_leakage(P.render_messages(pkg), pkg, registry)

    def test_metadata_literals_are_detected(self):
        for literal in ("rfp_main", "portal_listing", "addendum_number", "page_numbers",
                        "fused_score", "overall_rank", "retrieval_metadata"):
            registry, pkg = build_package(
                [("docA", "rfp_main", "Main.pdf", [("ck1", f"leaky {literal} value")])]
            )
            with self.assertRaises(P.PromptLeakageError):
                P.assert_no_leakage(P.render_messages(pkg), pkg, registry)

    def test_banned_word_in_instructions_is_detected(self):
        registry, pkg = simple_package()
        spec = FIELD_CATALOG["title"]
        original = spec.include
        spec.include = "Use the latest value stated."
        try:
            with self.assertRaises(P.PromptLeakageError) as ctx:
                P.assert_no_leakage(P.render_messages(pkg), pkg, registry)
            self.assertIn("latest", str(ctx.exception))
        finally:
            spec.include = original

    def test_banned_word_inside_evidence_body_is_allowed(self):
        registry, pkg = build_package(
            [("docA", "rfp_main", "Main.pdf",
              [("ck1", "ADDENDUM No. 2 states the latest due date and supersedes page 3.")])]
        )
        P.assert_no_leakage(P.render_messages(pkg), pkg, registry)

    def test_no_hidden_provenance_words_in_rendered_prompt(self):
        _, pkg = simple_package()
        text = P.render_user_prompt(pkg)
        for token in ("page_number", "doc_id", "chunk_id", "document_type",
                      "lexical_rank", "semantic_rank", "html_label_hit"):
            self.assertNotIn(token, text)

    def test_no_golden_literals_in_instruction_text(self):
        _, pkg = simple_package()
        instructions = P._instruction_text(P.render_messages(pkg))
        for literal in GOLDEN_LITERALS:
            self.assertNotIn(literal.lower(), instructions.lower())

    def test_catalog_prompt_text_has_no_golden_literals(self):
        for name, spec in FIELD_CATALOG.items():
            blob = " ".join([spec.definition, spec.include, spec.exclude]).lower()
            for literal in GOLDEN_LITERALS:
                self.assertNotIn(literal.lower(), blob, f"{literal} leaked into {name}")

    def test_system_prompt_has_no_banned_words(self):
        lowered = P.SYSTEM_PROMPT.lower()
        for word in P.BANNED_INSTRUCTION_WORDS:
            self.assertIsNone(re.search(r"\b" + re.escape(word) + r"\b", lowered), word)

    def test_system_prompt_states_extraction_only_rules(self):
        text = P.SYSTEM_PROMPT
        self.assertIn("extraction only", text)
        self.assertIn("not_found", text)
        self.assertIn("Do not decide which value applies", text)
        self.assertIn("Never calculate, convert", text)


# --- response schema ---------------------------------------------------------

ALLOWED_KEYWORDS = {"type", "properties", "required", "additionalProperties", "enum", "items", "anyOf"}


class TestResponseSchema(unittest.TestCase):
    def _walk(self, node, visit):
        if isinstance(node, dict):
            visit(node)
            for value in node.values():
                self._walk(value, visit)
        elif isinstance(node, list):
            for item in node:
                self._walk(item, visit)

    def test_only_allowed_keywords(self):
        _, pkg = simple_package()
        schema = P.response_schema(pkg)
        seen = []
        self._walk(schema, lambda n: seen.extend(k for k in n if k not in ("properties",)))
        # keys under "properties" are field names, so inspect schema nodes only
        def visit(node):
            if "type" in node or "anyOf" in node:
                for key in node:
                    self.assertIn(key, ALLOWED_KEYWORDS)
        self._walk(schema, visit)

    def test_objects_are_closed_and_fully_required(self):
        _, pkg = simple_package()
        schema = P.response_schema(pkg)

        def visit(node):
            if node.get("type") == "object":
                self.assertIs(node.get("additionalProperties"), False)
                self.assertEqual(set(node.get("required", [])), set(node.get("properties", {})))

        self._walk(schema, visit)

    def test_enums_are_package_controlled(self):
        _, pkg = simple_package()
        schema = P.response_schema(pkg)
        props = schema["properties"]
        self.assertEqual(props["group"]["enum"], [pkg.group])
        cand = props["candidates"]["items"]["properties"]
        self.assertEqual(cand["field"]["enum"], list(pkg.fields))
        self.assertEqual(
            cand["evidence"]["items"]["properties"]["evidence_id"]["enum"],
            [e.handle for e in pkg.excerpts],
        )
        nf = props["not_found"]["items"]["properties"]
        self.assertEqual(nf["field"]["enum"], list(pkg.fields))
        self.assertEqual(
            nf["reason_code"]["enum"],
            ["not_stated", "mentioned_without_value", "only_excluded_subjects"],
        )

    def test_nullable_value_and_contact(self):
        _, pkg = simple_package()
        cand = P.response_schema(pkg)["properties"]["candidates"]["items"]["properties"]
        self.assertEqual(cand["value"]["type"], ["string", "null"])
        self.assertIn({"type": "null"}, cand["contact"]["anyOf"])


# --- parser: envelope --------------------------------------------------------

class TestParserEnvelope(unittest.TestCase):
    def setUp(self):
        _, self.pkg = simple_package()

    def test_invalid_json(self):
        parsed = P.parse_response("{not json", self.pkg)
        self.assertEqual(parsed.status, "invalid_json")
        self.assertTrue(parsed.errors)
        self.assertEqual(parsed.for_grounding, [])

    def test_non_object_json(self):
        self.assertEqual(P.parse_response("[1, 2]", self.pkg).status, "schema_error")

    def test_missing_key(self):
        parsed = P.parse_response(json.dumps({"group": "identity", "candidates": []}), self.pkg)
        self.assertEqual(parsed.status, "schema_error")

    def test_extra_key_rejected(self):
        raw = json.dumps({"group": "identity", "candidates": [], "not_found": [], "extra": 1})
        self.assertEqual(P.parse_response(raw, self.pkg).status, "schema_error")

    def test_wrong_group(self):
        parsed = P.parse_response(envelope("schedule"), self.pkg)
        self.assertEqual(parsed.status, "schema_error")
        self.assertIn("group", parsed.errors[0])

    def test_errors_do_not_leak_ids(self):
        parsed = P.parse_response("{oops", self.pkg)
        joined = " ".join(parsed.errors)
        for chunk_id in self.pkg.handle_map.values():
            self.assertNotIn(chunk_id, joined)


# --- parser: valid responses -------------------------------------------------

class TestParserValid(unittest.TestCase):
    def setUp(self):
        self.registry, self.pkg = simple_package()
        self.h = [e.handle for e in self.pkg.excerpts]

    def test_candidate_is_mapped_to_chunk_id(self):
        raw = envelope("identity", [wire("bid_number", "ABC-123",
                                         evidence=[(self.h[0], "Solicitation Number ABC-123")])])
        parsed = P.parse_response(raw, self.pkg)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(len(parsed.for_grounding), 1)
        fc = parsed.for_grounding[0]
        self.assertEqual(fc.field_name, "bid_number")
        self.assertEqual(fc.doc_label, "D1")
        self.assertEqual(fc.candidate.evidence[0].chunk_id, self.pkg.handle_map[self.h[0]])
        self.assertEqual(fc.candidate.evidence[0].explanation, "")

    def test_no_resolution_metadata_is_ever_set(self):
        raw = envelope("identity", [wire("bid_number", "ABC-123",
                                         evidence=[(self.h[0], "Solicitation Number ABC-123")])])
        candidate = P.parse_response(raw, self.pkg).for_grounding[0].candidate
        self.assertIsNone(candidate.confidence_score)
        self.assertFalse(candidate.is_superseding)
        self.assertIsNone(candidate.supersedes_reason)

    def test_multiple_candidates_for_one_field_are_all_kept(self):
        raw = envelope("identity", [
            wire("bid_number", "ABC-123", evidence=[(self.h[0], "Solicitation Number ABC-123")]),
            wire("bid_number", "Example County", evidence=[(self.h[0], "issued by Example County")]),
        ])
        parsed = P.parse_response(raw, self.pkg)
        self.assertEqual(len(parsed.for_grounding), 2)
        self.assertEqual(parsed.coverage["bid_number"], "candidates")

    def test_not_found_and_coverage_states(self):
        raw = envelope("identity",
                       [wire("bid_number", "ABC-123", evidence=[(self.h[0], "Solicitation Number ABC-123")])],
                       [{"field": "title", "reason_code": "not_stated", "reason": "absent"}])
        parsed = P.parse_response(raw, self.pkg)
        self.assertEqual(parsed.coverage["bid_number"], "candidates")
        self.assertEqual(parsed.coverage["title"], "not_found")
        self.assertEqual(parsed.coverage["company_name"], "missing")
        self.assertEqual(parsed.not_found[0].field_name, "title")

    def test_both_states_warn_and_keep_candidates(self):
        raw = envelope("identity",
                       [wire("bid_number", "ABC-123", evidence=[(self.h[0], "Solicitation Number ABC-123")])],
                       [{"field": "bid_number", "reason_code": "not_stated", "reason": "x"}])
        parsed = P.parse_response(raw, self.pkg)
        self.assertEqual(parsed.coverage["bid_number"], "both")
        self.assertIn("not_found_contradicted:bid_number", parsed.warnings)
        self.assertEqual(len(parsed.for_grounding), 1)

    def test_not_found_reason_truncated_and_unknown_field_ignored(self):
        raw = envelope("identity", [], [
            {"field": "title", "reason_code": "not_stated", "reason": "x" * 400},
            {"field": "due_date", "reason_code": "not_stated", "reason": "wrong group"},
        ])
        parsed = P.parse_response(raw, self.pkg)
        self.assertEqual(len(parsed.not_found), 1)
        self.assertEqual(len(parsed.not_found[0].reason), config.MAX_REASON_CHARS)
        self.assertIn("not_found_unknown_field", parsed.warnings)

    def test_contact_candidate(self):
        registry, pkg = build_package(
            [("docA", "rfp_main", "Main.pdf",
              [("ck1", "Buyer: Jane Roe, jane@example.gov, 555-0100")])],
            group="contacts_summary",
        )
        handle = pkg.excerpts[0].handle
        raw = envelope("contacts_summary", [
            wire("contact_info",
                 contact={"name": "Jane Roe", "email": "jane@example.gov", "phone": "555-0100"},
                 evidence=[(handle, "Buyer: Jane Roe, jane@example.gov, 555-0100")])
        ])
        parsed = P.parse_response(raw, pkg)
        self.assertEqual(parsed.status, "ok")
        value = parsed.for_grounding[0].candidate.value
        self.assertEqual(value, {"name": "Jane Roe", "email": "jane@example.gov", "phone": "555-0100"})

    def test_bid_summary_candidate(self):
        registry, pkg = build_package(
            [("docA", "rfp_main", "Main.pdf",
              [("ck1", "This solicitation seeks widgets for county offices.")])],
            group="contacts_summary",
        )
        handle = pkg.excerpts[0].handle
        raw = envelope("contacts_summary", [
            wire("bid_summary", "This solicitation seeks widgets for county offices.",
                 evidence=[(handle, "This solicitation seeks widgets for county offices.")])
        ])
        parsed = P.parse_response(raw, pkg)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(len(parsed.for_grounding), 1)


# --- parser: rejections ------------------------------------------------------

class TestParserRejections(unittest.TestCase):
    def setUp(self):
        self.registry, self.pkg = simple_package()
        self.h = [e.handle for e in self.pkg.excerpts]

    def _reject_code(self, raw):
        parsed = P.parse_response(raw, self.pkg)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(len(parsed.rejected), 1)
        self.assertEqual(parsed.for_grounding, [])
        return parsed.rejected[0].rejection_code

    def test_unknown_field(self):
        raw = envelope("identity", [wire("due_date", "x", evidence=[(self.h[0], "q")])])
        self.assertEqual(self._reject_code(raw), RejectionCode.unknown_field)

    def test_unknown_handle(self):
        raw = envelope("identity", [wire("bid_number", "ABC-123", evidence=[("Edeadbeef", "q")])])
        self.assertEqual(self._reject_code(raw), RejectionCode.unknown_evidence_id)

    def test_no_evidence(self):
        raw = envelope("identity", [wire("bid_number", "ABC-123", evidence=[])])
        self.assertEqual(self._reject_code(raw), RejectionCode.no_evidence)

    def test_null_value_for_normal_field(self):
        raw = envelope("identity", [wire("bid_number", None, evidence=[(self.h[0], "q")])])
        self.assertEqual(self._reject_code(raw), RejectionCode.format_error)

    def test_contact_object_on_normal_field(self):
        raw = envelope("identity", [wire("bid_number", "ABC-123",
                                         contact={"name": "X", "email": None, "phone": None},
                                         evidence=[(self.h[0], "q")])])
        self.assertEqual(self._reject_code(raw), RejectionCode.format_error)

    def test_value_too_long(self):
        raw = envelope("identity", [wire("bid_number", "a" * (config.MAX_VALUE_CHARS + 1),
                                         evidence=[(self.h[0], "q")])])
        self.assertEqual(self._reject_code(raw), RejectionCode.format_error)

    def test_placeholder_values(self):
        for placeholder in ("N/A", "none", "TBD", "____", "(name of business entity)", "   "):
            raw = envelope("identity", [wire("bid_number", placeholder, evidence=[(self.h[0], "q")])])
            parsed = P.parse_response(raw, self.pkg)
            self.assertEqual(len(parsed.rejected), 1, placeholder)
            self.assertEqual(parsed.rejected[0].rejection_code, RejectionCode.format_error, placeholder)

    def test_empty_and_oversized_quotes(self):
        raw = envelope("identity", [wire("bid_number", "ABC-123", evidence=[(self.h[0], "   ")])])
        self.assertEqual(self._reject_code(raw), RejectionCode.invalid_quote)

        long_quote = "q" * (config.MAX_QUOTE_CHARS + 1)
        raw = envelope("identity", [wire("bid_number", "ABC-123", evidence=[(self.h[0], long_quote)])])
        self.assertEqual(self._reject_code(raw), RejectionCode.invalid_quote)

    def test_contact_shape_errors(self):
        _, pkg = build_package(
            [("docA", "rfp_main", "Main.pdf", [("ck1", "Buyer: Jane Roe")])],
            group="contacts_summary",
        )
        handle = pkg.excerpts[0].handle

        raw = envelope("contacts_summary", [wire("contact_info", "Jane Roe", evidence=[(handle, "Buyer: Jane Roe")])])
        parsed = P.parse_response(raw, pkg)
        self.assertEqual(parsed.rejected[0].rejection_code, RejectionCode.format_error)

        raw = envelope("contacts_summary", [
            wire("contact_info", contact={"name": None, "email": None, "phone": None},
                 evidence=[(handle, "Buyer: Jane Roe")])
        ])
        parsed = P.parse_response(raw, pkg)
        self.assertEqual(parsed.rejected[0].rejection_code, RejectionCode.contact_empty)

    def test_placeholder_contact_part(self):
        _, pkg = build_package(
            [("docA", "rfp_main", "Main.pdf", [("ck1", "Name: (name of affiant)")])],
            group="contacts_summary",
        )
        handle = pkg.excerpts[0].handle
        raw = envelope("contacts_summary", [
            wire("contact_info", contact={"name": "(name of affiant)", "email": None, "phone": None},
                 evidence=[(handle, "Name: (name of affiant)")])
        ])
        parsed = P.parse_response(raw, pkg)
        self.assertEqual(parsed.rejected[0].rejection_code, RejectionCode.format_error)

    def test_malformed_candidate_object_is_schema_error(self):
        raw = json.dumps({
            "group": "identity",
            "candidates": [{"field": "bid_number", "value": "ABC-123"}],  # missing contact/evidence
            "not_found": [],
        })
        self.assertEqual(P.parse_response(raw, self.pkg).status, "schema_error")

    def test_rejected_candidates_never_reach_grounding_list(self):
        raw = envelope("identity", [
            wire("bid_number", "ABC-123", evidence=[(self.h[0], "Solicitation Number ABC-123")]),
            wire("bid_number", "N/A", evidence=[(self.h[0], "q")]),
        ])
        parsed = P.parse_response(raw, self.pkg)
        self.assertEqual(len(parsed.for_grounding), 1)
        self.assertEqual(len(parsed.rejected), 1)


# --- parser: transformations -------------------------------------------------

class TestParserTransforms(unittest.TestCase):
    def setUp(self):
        self.registry, self.pkg = simple_package()
        self.h = [e.handle for e in self.pkg.excerpts]

    def test_duplicate_evidence_deduplicated(self):
        raw = envelope("identity", [wire("bid_number", "ABC-123", evidence=[
            (self.h[0], "Solicitation Number ABC-123"),
            (self.h[0], "Solicitation Number ABC-123"),
        ])])
        parsed = P.parse_response(raw, self.pkg)
        self.assertEqual(len(parsed.for_grounding[0].candidate.evidence), 1)
        self.assertIn("duplicate_evidence:bid_number", parsed.warnings)

    def test_evidence_capped_at_three(self):
        _, pkg = build_package(
            [("docA", "rfp_main", "Main.pdf",
              [(f"ck{i}", f"Solicitation Number ABC-123 mention {i}") for i in range(1, 6)])]
        )
        handles = [e.handle for e in pkg.excerpts]
        raw = envelope("identity", [wire("bid_number", "ABC-123",
                                         evidence=[(h, f"Solicitation Number ABC-123 mention {i}")
                                                   for i, h in enumerate(handles, start=1)])])
        parsed = P.parse_response(raw, pkg)
        self.assertEqual(len(parsed.for_grounding[0].candidate.evidence), config.MAX_EVIDENCE_PER_CANDIDATE)
        self.assertIn("evidence_truncated:bid_number", parsed.warnings)

    def test_contact_and_summary_capped_at_one(self):
        _, pkg = build_package(
            [("docA", "rfp_main", "Main.pdf",
              [("ck1", "Buyer: Jane Roe"), ("ck2", "Buyer: Jane Roe again")])],
            group="contacts_summary",
        )
        handles = [e.handle for e in pkg.excerpts]
        raw = envelope("contacts_summary", [
            wire("contact_info", contact={"name": "Jane Roe", "email": None, "phone": None},
                 evidence=[(handles[0], "Buyer: Jane Roe"), (handles[1], "Buyer: Jane Roe again")])
        ])
        parsed = P.parse_response(raw, pkg)
        self.assertEqual(len(parsed.for_grounding), 1)
        self.assertEqual(len(parsed.for_grounding[0].candidate.evidence), 1)

    def test_cross_document_evidence_is_split_not_merged(self):
        _, pkg = two_doc_package()
        handles = [e.handle for e in pkg.excerpts]
        labels = {e.handle: e.doc_label for e in pkg.excerpts}
        self.assertNotEqual(labels[handles[0]], labels[handles[1]])
        raw = envelope("identity", [wire("bid_number", "ABC-123", evidence=[
            (handles[0], "Solicitation Number ABC-123."),
            (handles[1], "Solicitation Number ABC-123."),
        ])])
        parsed = P.parse_response(raw, pkg)
        self.assertEqual(len(parsed.for_grounding), 2)
        self.assertEqual({fc.doc_label for fc in parsed.for_grounding}, {"D1", "D2"})
        for fc in parsed.for_grounding:
            self.assertEqual(len(fc.candidate.evidence), 1)
        self.assertIn("cross_document_split:bid_number", parsed.warnings)

    def test_exact_duplicate_candidates_dropped(self):
        raw = envelope("identity", [
            wire("bid_number", "ABC-123", evidence=[(self.h[0], "Solicitation Number ABC-123")]),
            wire("bid_number", "ABC-123", evidence=[(self.h[0], "Solicitation Number ABC-123")]),
        ])
        parsed = P.parse_response(raw, self.pkg)
        self.assertEqual(len(parsed.for_grounding), 1)
        self.assertIn("duplicate_candidate:bid_number", parsed.warnings)

    def test_candidate_cap(self):
        candidates = [
            wire("bid_number", f"ABC-{i}", evidence=[(self.h[0], f"Solicitation Number ABC-{i}")])
            for i in range(config.MAX_CANDIDATES_PER_RESPONSE + 5)
        ]
        parsed = P.parse_response(envelope("identity", candidates), self.pkg)
        self.assertIn("candidate_cap", parsed.warnings)
        self.assertLessEqual(len(parsed.for_grounding), config.MAX_CANDIDATES_PER_RESPONSE)


class TestParserDeterminism(unittest.TestCase):
    def test_identical_input_gives_identical_output(self):
        _, pkg = simple_package()
        h = [e.handle for e in pkg.excerpts]
        raw = envelope("identity", [
            wire("title", "Supply of Widgets", evidence=[(h[1], "Title: Supply of Widgets")]),
            wire("bid_number", "ABC-123", evidence=[(h[0], "Solicitation Number ABC-123")]),
        ])
        first = P.parse_response(raw, pkg).model_dump_json()
        second = P.parse_response(raw, pkg).model_dump_json()
        self.assertEqual(first, second)

    def test_candidates_sorted_by_catalog_order(self):
        _, pkg = simple_package()
        h = [e.handle for e in pkg.excerpts]
        raw = envelope("identity", [
            wire("title", "Supply of Widgets", evidence=[(h[1], "Title: Supply of Widgets")]),
            wire("bid_number", "ABC-123", evidence=[(h[0], "Solicitation Number ABC-123")]),
        ])
        parsed = P.parse_response(raw, pkg)
        self.assertEqual([fc.field_name for fc in parsed.for_grounding], ["bid_number", "title"])


# --- correction prompt -------------------------------------------------------

class TestCorrectionPrompt(unittest.TestCase):
    def test_contains_errors_and_is_deterministic(self):
        text = P.render_correction(["group: expected 'identity'", "candidates.0: missing"])
        self.assertTrue(text.startswith("CORRECTION"))
        self.assertIn("group: expected 'identity'", text)
        self.assertEqual(text, P.render_correction(["group: expected 'identity'", "candidates.0: missing"]))

    def test_truncates_long_error_lists(self):
        text = P.render_correction(["e" * 2000])
        body = text.split(": ", 1)[1]
        self.assertLessEqual(len(body), config.MAX_CORRECTION_CHARS + 120)

    def test_correction_has_no_banned_words(self):
        lowered = P.render_correction(["bad shape"]).lower()
        for word in P.BANNED_INSTRUCTION_WORDS:
            self.assertIsNone(re.search(r"\b" + re.escape(word) + r"\b", lowered), word)


# --- hand-off to Stage 4.1 ---------------------------------------------------

class TestGroundingCompatibility(unittest.TestCase):
    """The parser validates contract shape only; grounding remains the authority."""

    def setUp(self):
        self.registry, self.pkg = build_package(
            [("docA", "rfp_main", "Main.pdf",
              [("ck1", "PROPOSAL DUE \nDATE and TIME: \n06/10/2024"),
               ("ck2", "Delivery within 45 days of Award.")])],
            group="schedule",
        )
        self.h = [e.handle for e in self.pkg.excerpts]
        self.validator = GroundingValidator(self.registry)

    def test_parsed_candidate_grounds_successfully(self):
        raw = envelope("schedule", [
            wire("due_date", "06/10/2024",
                 evidence=[(self.h[0], "PROPOSAL DUE DATE and TIME: 06/10/2024")])
        ])
        parsed = P.parse_response(raw, self.pkg)
        fc = parsed.for_grounding[0]
        grounded = self.validator.validate(fc.field_name, fc.candidate)
        self.assertTrue(grounded.is_valid)
        self.assertIsNone(grounded.rejection_code)

    def test_parser_accepts_but_grounding_rejects_fabricated_quote(self):
        raw = envelope("schedule", [
            wire("delivery_date", "Delivery within 30 days of Award.",
                 evidence=[(self.h[1], "Delivery within 30 days of Award.")])
        ])
        parsed = P.parse_response(raw, self.pkg)
        self.assertEqual(len(parsed.for_grounding), 1, "parser must not do grounding")
        fc = parsed.for_grounding[0]
        grounded = self.validator.validate(fc.field_name, fc.candidate)
        self.assertFalse(grounded.is_valid)
        self.assertEqual(grounded.rejection_code, RejectionCode.quote_not_in_chunk)


if __name__ == '__main__':
    unittest.main()
