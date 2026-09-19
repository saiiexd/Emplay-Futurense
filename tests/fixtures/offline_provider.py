"""Source-backed offline provider responses for end-to-end pipeline tests.

This module is test-only. It reads the audited flat artifacts as expected values,
then emits candidates only when those values (or their individual list items) are
present in the ContextBuilder excerpts supplied to the fake provider. It never
runs in production and does not replace retrieval, grounding, or resolution.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.validation.canonical import find_canonical_match


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


class SourceBackedOfflineResponder:
    """Render deterministic, evidence-linked responses for one supplied bid."""

    def __init__(self, bid_id: str):
        self.bid_id = bid_id
        path = OUTPUT_DIR / f"{bid_id}.flat.json"
        self.expected: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        self.calls: List[str] = []

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        user = messages[1]["content"]
        group = self._single_line(user, r"^Group: (.+)$")
        self.calls.append(group)
        fields = re.findall(r"^\[(\w+)\] kind:", user, re.MULTILINE)
        blocks = self._blocks(user)
        candidates: List[Dict[str, Any]] = []
        not_found: List[Dict[str, str]] = []

        for field_name in fields:
            label = self._label_for(field_name)
            expected = self.expected.get(label)
            if expected is None:
                not_found.append(self._not_found(field_name))
                continue

            if field_name == "contact_info":
                contacts = self._contacts(expected)
                emitted = 0
                for contact in contacts:
                    evidence = self._find_contact_evidence(contact, blocks, field_name)
                    if evidence:
                        handle, quote = evidence
                        candidates.append({
                            "field": field_name,
                            "value": None,
                            "contact": contact,
                            "evidence": [{"evidence_id": handle, "quote": quote}],
                        })
                        emitted += 1
                if not emitted:
                    not_found.append(self._not_found(field_name))
                continue

            values = self._list_items(expected) if field_name in {
                "additional_documentation", "mfg_for_registration", "model_no",
                "part_no", "product",
            } else [str(expected)]
            emitted = 0
            for value in values:
                evidence = self._find_value_evidence(field_name, value, blocks)
                if not evidence and field_name == "bid_summary":
                    evidence = self._find_summary_evidence(value, blocks, field_name)
                if evidence:
                    handle, quote = evidence
                    candidates.append({
                        "field": field_name,
                        "value": value,
                        "contact": None,
                        "evidence": [{"evidence_id": handle, "quote": quote}],
                    })
                    emitted += 1
            if not emitted:
                not_found.append(self._not_found(field_name))

        return json.dumps({"group": group, "candidates": candidates, "not_found": not_found})

    @staticmethod
    def _single_line(text: str, pattern: str) -> str:
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise AssertionError(f"offline fixture could not identify {pattern!r}")
        return match.group(1).strip()

    @staticmethod
    def _blocks(user: str) -> List[Tuple[str, str, List[str]]]:
        pattern = re.compile(
            r'<excerpt id="(E[0-9a-f]+)"[^>]*fields="([^"]*)">\n(.*?)\n</excerpt>',
            re.DOTALL,
        )
        return [
            (handle, text, eligible.split(","))
            for handle, eligible, text in pattern.findall(user)
        ]

    @staticmethod
    def _label_for(field_name: str) -> str:
        from src.schemas.field_catalog import FIELD_CATALOG
        return FIELD_CATALOG[field_name].assignment_label

    @staticmethod
    def _not_found(field_name: str) -> Dict[str, str]:
        return {
            "field": field_name,
            "reason_code": "not_stated",
            "reason": "The supplied excerpts do not state a supported value.",
        }

    @staticmethod
    def _list_items(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        return [item.strip() for item in str(value).split(";") if item.strip()]

    @staticmethod
    def _contacts(value: str) -> List[Dict[str, Optional[str]]]:
        contacts = []
        for match in re.finditer(r"([^;(]+?)\s*\(([^)]*)\)", value):
            name = match.group(1).strip()
            parts = [part.strip() for part in match.group(2).split(";")]
            email = next((part for part in parts if "@" in part), None)
            phone = next((part for part in parts if "@" not in part), None)
            contacts.append({"name": name, "email": email, "phone": phone})
        return contacts

    @staticmethod
    def _find_value_evidence(
        field_name: str, value: str, blocks: Iterable[Tuple[str, str, List[str]]]
    ) -> Optional[Tuple[str, str]]:
        for handle, text, eligible in blocks:
            if field_name not in eligible:
                continue
            match = find_canonical_match(value, text)
            if match.matched:
                return handle, match.original_substring
        return None

    @staticmethod
    def _find_summary_evidence(
        value: str, blocks: Iterable[Tuple[str, str, List[str]]], field_name: str
    ) -> Optional[Tuple[str, str]]:
        tokens = [token.casefold() for token in re.findall(r"[A-Za-z0-9]+", value) if len(token) > 4]
        for handle, text, eligible in blocks:
            if field_name not in eligible:
                continue
            if all(token in text.casefold() for token in tokens):
                return handle, text[:600]
        return None

    @classmethod
    def _find_contact_evidence(
        cls, contact: Dict[str, Optional[str]], blocks: Iterable[Tuple[str, str, List[str]]], field_name: str
    ) -> Optional[Tuple[str, str]]:
        for handle, text, eligible in blocks:
            if field_name not in eligible:
                continue
            parts = [value for value in contact.values() if value]
            if all(find_canonical_match(value, text).matched for value in parts):
                return handle, text[:600]
        return None
