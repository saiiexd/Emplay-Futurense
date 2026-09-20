"""End-to-end pipeline for one bid directory.

    ingest -> chunk -> registry -> retrieval index
           -> ContextBuilder (per group)
           -> ExtractionEngine (LLM) -> prompts.parse_response
           -> GroundingValidator (Stage 4.1, unchanged)
           -> CandidateResolver (deterministic precedence)
           -> 20-field record + public JSON + diagnostics

Each stage keeps its own responsibility. Grounding is the only place quotes are
checked against source text; resolution is the only place a winner is chosen.
"""

import json
import logging
import os
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional

import src.config as config
from src.extraction.context_builder import ContextBuilder
from src.extraction.extraction_engine import ExtractionEngine, GroupExtractionResult
from src.extraction.rule_candidates import RuleCandidateGenerator
from src.parsers.bid_manager import BidManager
from src.resolution.resolver import (
    CandidateResolver,
    FieldResolution,
    build_aggregate_document,
    build_bid_record,
    build_diagnostics,
    to_flat_json,
    to_public_json,
)
from src.retrieval.chunker import DocumentChunker
from src.retrieval.embeddings import (
    CachedEmbeddingClient,
    FakeEmbeddingClient,
    OpenAIEmbeddingClient,
)
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.registry import DocumentRegistry
from src.schemas.candidates import GroundedCandidate
from src.schemas.field_catalog import FIELD_CATALOG
from src.schemas.models import BidRecord
from src.validation.grounding import GroundingValidator

logger = logging.getLogger(__name__)


@dataclass
class BidContext:
    """Everything ingestion and retrieval produce for one bid."""

    bid_id: str
    registry: DocumentRegistry
    builder: ContextBuilder
    searcher: HybridSearcher


@dataclass
class PipelineResult:
    bid_id: str
    record: BidRecord
    public_json: Dict[str, Any]
    flat_json: Dict[str, Any]
    diagnostics: Dict[str, Any]
    group_results: Dict[str, GroupExtractionResult] = dc_field(default_factory=dict)
    grounded: List[GroundedCandidate] = dc_field(default_factory=list)
    rejected: List[GroundedCandidate] = dc_field(default_factory=list)
    resolutions: Dict[str, FieldResolution] = dc_field(default_factory=dict)


def build_bid_context(bid_dir: str, embeddings: str = "fake") -> BidContext:
    """Ingest, chunk, register and index one bid folder."""
    bid_id = os.path.basename(os.path.normpath(bid_dir))
    bid_data = BidManager(output_dir=os.path.join(config.CACHE_ROOT, "parsed")).process_bid_directory(bid_dir)

    registry = DocumentRegistry()
    chunker = DocumentChunker(max_chunk_size=config.CHUNK_MAX, overlap=config.CHUNK_OVERLAP)
    all_chunks = []
    for document in bid_data.get("documents", []):
        chunks = chunker.chunk_document(document)
        registry.register_document(document, chunks)
        all_chunks.extend(chunks)

    if embeddings == "openai":
        client = CachedEmbeddingClient(OpenAIEmbeddingClient())
    else:
        client = FakeEmbeddingClient(dim=256)

    searcher = HybridSearcher(embedding_client=client)
    searcher.add_chunks(all_chunks)
    searcher.build_index()

    return BidContext(
        bid_id=bid_id,
        registry=registry,
        builder=ContextBuilder(registry, searcher),
        searcher=searcher,
    )


def ground_group_results(
    group_results: Dict[str, GroupExtractionResult], registry: DocumentRegistry
) -> tuple:
    """Run every parsed candidate through the Stage 4.1 grounding validator.

    Returns (accepted, rejected). Rejected candidates keep their code and reason
    so a failure is auditable instead of disappearing.
    """
    validator = GroundingValidator(registry)
    accepted: List[GroundedCandidate] = []
    rejected: List[GroundedCandidate] = []

    for group, result in group_results.items():
        for field_candidate in result.candidates:
            grounded = validator.validate(field_candidate.field_name, field_candidate.candidate)
            if grounded.is_valid:
                accepted.append(grounded)
            else:
                rejected.append(grounded)
                logger.info(
                    "group %s: rejected %s candidate (%s)",
                    group, field_candidate.field_name, grounded.rejection_code,
                )

    return accepted, rejected


def group_by_field(candidates: List[GroundedCandidate]) -> Dict[str, List[GroundedCandidate]]:
    by_field: Dict[str, List[GroundedCandidate]] = {}
    for candidate in candidates:
        by_field.setdefault(candidate.field_name, []).append(candidate)
    return by_field


def run_pipeline(bid_dir: str, engine_factory: Callable[[DocumentRegistry], ExtractionEngine],
                 embeddings: str = "fake") -> PipelineResult:
    """Run one bid from raw documents to the final 20-field answer."""
    context = build_bid_context(bid_dir, embeddings=embeddings)

    engine = engine_factory(context.registry)
    group_results = engine.extract_bid(context.builder, context.bid_id)

    accepted, rejected = ground_group_results(group_results, context.registry)

    # Deterministic candidates for forms that only name themselves in their own
    # body text, which a model reading excerpts may never state. They are built
    # from source text, so they are already grounded.
    rule_generator = RuleCandidateGenerator(context.registry)
    for field_name in FIELD_CATALOG:
        accepted.extend(rule_generator.generate_candidates(field_name))

    resolver = CandidateResolver(context.registry, doc_label_for=context.builder.doc_label_for)
    resolutions = resolver.resolve_bid(group_by_field(accepted))

    record = build_bid_record(context.bid_id, resolutions, context.registry)
    diagnostics = build_diagnostics(context.bid_id, resolutions, rejected)
    diagnostics["extraction"] = {
        group: {
            "status": result.status,
            "attempts": result.attempts,
            "prompt_hash": result.prompt_hash,
            "error": result.error,
            "coverage": dict(result.parsed.coverage) if result.parsed else {},
            "warnings": list(result.parsed.warnings) if result.parsed else [],
        }
        for group, result in group_results.items()
    }

    return PipelineResult(
        bid_id=context.bid_id,
        record=record,
        public_json=to_public_json(record),
        flat_json=to_flat_json(record),
        diagnostics=diagnostics,
        group_results=group_results,
        grounded=accepted,
        rejected=rejected,
        resolutions=resolutions,
    )


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


def write_outputs(result: PipelineResult, out_dir: str) -> Dict[str, str]:
    """Write one bid's artifacts. Never writes credentials.

    ``<bid>.flat.json``  the assignment answer: 20 labels, string or null
    ``<bid>.json``       the same answer with typed values (lists, contacts)
    ``<bid>.diagnostics.json``  the audit trail, kept out of the answer
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "flat": os.path.join(out_dir, f"{result.bid_id}.flat.json"),
        "public": os.path.join(out_dir, f"{result.bid_id}.json"),
        "diagnostics": os.path.join(out_dir, f"{result.bid_id}.diagnostics.json"),
    }

    _write_json(paths["flat"], result.flat_json)
    _write_json(paths["public"], result.public_json)
    _write_json(paths["diagnostics"], result.diagnostics)

    logger.info("wrote %s, %s and %s", paths["flat"], paths["public"], paths["diagnostics"])
    return paths


def write_aggregate(results: List[PipelineResult], out_dir: str) -> str:
    """Write the single aggregate deliverable covering every bid in this run."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "extracted_data.json")
    document = build_aggregate_document({r.bid_id: r.flat_json for r in results})
    _write_json(path, document)
    logger.info("wrote %s", path)
    return path
