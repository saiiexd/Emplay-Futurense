"""Primary extraction entry point for the assignment.

This is the Python script the assignment asks for. It turns one or more folders
of procurement documents (PDF + HTML) into the required 20-field JSON:

    python main.py --bid data/input/Bid1 --bid data/input/Bid2

For each bid folder it writes, into --out (default data/output):

    <bid>.flat.json          the assignment answer: 20 labels, string or null
    <bid>.json               the same answer with typed values (lists, contacts)
    <bid>.diagnostics.json   the audit trail, kept out of the answer

and one aggregate file covering every bid in the run:

    extracted_data.json

Credentials are read from the environment and are never written to disk. If the
provider fails for every group of a bid, no answer file is written for that bid,
because a null-filled file would misrepresent a failed run as a result.
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

import src.config as config
from src.extraction.extraction_engine import ExtractionEngine
from src.extraction.llm_client import ProviderConfigError, build_provider
from src.extraction.pipeline import run_pipeline, write_aggregate, write_outputs

logger = logging.getLogger("rfp")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the 20 assignment fields from one or more bid folders."
    )
    parser.add_argument(
        "--bid", action="append", required=True,
        help="path to a folder of bid documents; repeat for more bids",
    )
    parser.add_argument("--out", default=config.OUTPUT_ROOT, help="output directory")
    parser.add_argument(
        "--embeddings", choices=["fake", "openai"], default="fake",
        help="embedding channel for hybrid retrieval (default: fake, offline and deterministic)",
    )
    parser.add_argument(
        "--max-correction-retries", type=int, default=config.LLM_MAX_CORRECTION_RETRIES,
        help="how many times a malformed model response is re-requested",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
    )

    try:
        provider = build_provider()
    except ProviderConfigError as exc:
        print(f"Cannot start extraction: {exc}", file=sys.stderr)
        print("Check your .env file and ensure the required API key and provider are set.", file=sys.stderr)
        return 2

    def engine_factory(registry):
        return ExtractionEngine(
            provider, registry=registry, max_correction_retries=args.max_correction_retries
        )

    exit_code = 0
    completed = []

    for bid_dir in args.bid:
        result = run_pipeline(bid_dir, engine_factory, embeddings=args.embeddings)

        groups = result.group_results
        failed = [g for g, r in groups.items() if r.status == "provider_error"]

        # If the provider never answered, the empty record reflects a failed
        # call, not an empty corpus. Writing it would produce a null-filled
        # file that looks like a real extraction, so write nothing at all.
        if groups and len(failed) == len(groups):
            reason = next((r.error for r in groups.values() if r.error), "provider unavailable")
            print(f"\n{result.bid_id}: extraction FAILED - every group failed at the provider.",
                  file=sys.stderr)
            print(f"  reason: {reason}", file=sys.stderr)
            print(f"  no output written for {result.bid_id}: a null-filled file would "
                  f"misrepresent a failed run as a result.", file=sys.stderr)
            exit_code = 3
            continue

        paths = write_outputs(result, args.out)
        completed.append(result)

        summary = result.record.summary
        print(f"\n{result.bid_id}: {summary.found_fields}/{summary.total_fields} fields resolved "
              f"({summary.conflicting_fields} conflicting) -> {paths['flat']}")
        for field_name, resolution in result.resolutions.items():
            marker = "-" if resolution.status == "not_found" else "+"
            source = resolution.chosen.provenance.doc_label if resolution.chosen else ""
            print(f"  {marker} {field_name:26} {resolution.status:12} {resolution.rule:28} {source}")

        if failed:
            print(f"  WARNING: {len(failed)}/{len(groups)} group(s) failed at the "
                  f"provider: {', '.join(failed)}; this output is incomplete.", file=sys.stderr)
            exit_code = 3

    if completed:
        print(f"\naggregate -> {write_aggregate(completed, args.out)}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
