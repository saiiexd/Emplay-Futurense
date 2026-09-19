"""Deterministic, non-LLM candidate generation.

Some required forms only name themselves inside their own body text, so a model
reading excerpts may never state them. This rule captures that case without
guessing: the value is the phrase found in the document, quoted verbatim from
the chunk it came from, and it then travels the same resolution path as any
LLM-produced candidate.

The pattern is generic procurement language. No document name from the supplied
corpus is encoded here.
"""

import re
from typing import List

from src.retrieval.registry import DocumentRegistry
from src.schemas.candidates import GroundedCandidate, NormalizedValue
from src.schemas.enums import CandidateOrigin, MatchLevel
from src.schemas.field_catalog import FIELD_CATALOG

#: A form that names itself, such as "<Something> Affidavit". The negative
#: lookahead drops determiners, back-references and connectors ("this
#: affidavit", "or Affidavit"), which are not document titles.
#: ContextBuilder uses the same pattern when it decides which affidavit chunks
#: to show the model, so both layers agree on what counts as a named form.
_NOT_A_FORM_NAME = (
    "this|the|an?|said|such|each|any|or|and|of|to|for|by|with|is|be|no|not|that|which"
)
NAMED_FORM_PATTERN = re.compile(
    r'(?i)\b(?!(?:' + _NOT_A_FORM_NAME + r')\b)[a-z][a-z\-]+\s+affidavit\b'
)

#: The only field these rules contribute to today.
RULE_FIELD = "additional_documentation"


class RuleCandidateGenerator:
    def __init__(self, registry: DocumentRegistry):
        self.registry = registry

    def generate_candidates(self, field_name: str) -> List[GroundedCandidate]:
        spec = FIELD_CATALOG.get(field_name)
        if spec is None or field_name != RULE_FIELD:
            return []

        candidates: List[GroundedCandidate] = []
        for doc_id in sorted(self.registry.doc_refs):
            doc_ref = self.registry.get_document_ref(doc_id)
            if not doc_ref or doc_ref.document_type not in spec.eligible_document_types:
                continue

            # A document can name several required forms, and this field is a
            # list, so every distinct name counts. Each is attributed to the
            # first chunk that states it.
            seen: set = set()
            for chunk_id in self.registry.get_ordered_chunk_ids(doc_id):
                chunk = self.registry.chunk_metadata.get(chunk_id, {})
                for match in NAMED_FORM_PATTERN.finditer(chunk.get("text", "")):
                    phrase = match.group(0)
                    key = " ".join(phrase.split()).casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(self._build_candidate(field_name, phrase, chunk_id))

        return candidates

    def _build_candidate(self, field_name: str, value: str, chunk_id: str) -> GroundedCandidate:
        grounded = GroundedCandidate(
            candidate_id=GroundedCandidate.generate_id(field_name, [chunk_id], [value]),
            field_name=field_name,
            raw_value=[value],
            chunk_ids=[chunk_id],
            # The matched phrase is itself an exact substring of the chunk, so
            # the evidence stays verifiable against the source text.
            excerpts=[value],
            origin=CandidateOrigin.rule,
            is_valid=True,
            match_level=MatchLevel.exact,
        )
        grounded.normalized_value = NormalizedValue(value=[value], data_type="list")
        return grounded
