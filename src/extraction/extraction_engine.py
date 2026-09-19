"""Group-level LLM extraction.

One ``ContextPackage`` in, one parsed candidate structure out. The engine is the
only place that talks to a provider, and it is deliberately thin:

    render_messages(package)  ->  provider.complete(...)  ->  parse_response(raw)

It does not build prompts of its own, does not inspect documents, and does not
touch grounding, normalization, precedence or final assembly. Raw model text
always goes through the strict parser; model JSON is never trusted directly.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import src.config as config
from src.extraction import prompts
from src.extraction.llm_client import LLMProvider, ProviderCallError, ProviderConfigError
from src.schemas.candidates import ParsedGroupResponse

logger = logging.getLogger(__name__)

# Status meanings:
#   ok             - provider answered and the strict parser accepted the envelope
#   empty_context  - nothing eligible to send, so no call was made
#   invalid_output - provider answered but the parser rejected it after retries
#   provider_error - provider kept failing, or is misconfigured
STATUS_OK = "ok"
STATUS_EMPTY = "empty_context"
STATUS_INVALID = "invalid_output"
STATUS_PROVIDER_ERROR = "provider_error"


@dataclass
class GroupExtractionResult:
    """Outcome for a single group. Parser errors are kept, never swallowed."""

    group: str
    status: str
    parsed: Optional[ParsedGroupResponse] = None
    attempts: int = 0
    raw_responses: List[str] = field(default_factory=list)
    error: Optional[str] = None
    prompt_hash: Optional[str] = None

    @property
    def candidates(self):
        """Structurally valid candidates awaiting the Stage 4.1 grounding layer."""
        return self.parsed.for_grounding if self.parsed else []


class ExtractionEngine:
    """Runs the six extraction groups through a configurable provider."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_correction_retries: int = config.LLM_MAX_CORRECTION_RETRIES,
        max_transient_retries: int = config.LLM_MAX_TRANSIENT_RETRIES,
        backoff_factor: float = config.LLM_BACKOFF_FACTOR,
        registry: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.provider = provider
        self.max_correction_retries = max(0, int(max_correction_retries))
        self.max_transient_retries = max(0, int(max_transient_retries))
        self.backoff_factor = backoff_factor
        # Optional: when supplied, the leakage guard can also check filenames and
        # document ids from the registry, not just the package's own handles.
        self.registry = registry
        self._sleep = sleep

    # --- one group -----------------------------------------------------------

    def extract_group(self, package) -> GroupExtractionResult:
        group = package.group

        # An empty package cannot be rendered, so there is nothing to ask.
        if not package.fields or not package.excerpts:
            logger.info("group %s has no eligible evidence; skipping provider call", group)
            return GroupExtractionResult(group=group, status=STATUS_EMPTY)

        base_messages = prompts.render_messages(package)
        schema = prompts.response_schema(package)
        messages = list(base_messages)

        result = GroupExtractionResult(
            group=group, status=STATUS_PROVIDER_ERROR, prompt_hash=prompts.prompt_hash(package)
        )

        for attempt in range(self.max_correction_retries + 1):
            # Re-checked on every call, including correction calls, so a retry
            # can never smuggle hidden metadata into the conversation.
            prompts.assert_no_leakage(messages, package, self.registry)

            try:
                raw = self._call_provider(messages, package, schema)
            except ProviderConfigError as exc:
                result.status = STATUS_PROVIDER_ERROR
                result.error = str(exc)
                logger.error("group %s: provider configuration error: %s", group, exc)
                return result
            except ProviderCallError as exc:
                result.status = STATUS_PROVIDER_ERROR
                result.error = str(exc)
                logger.error("group %s: provider unavailable after retries: %s", group, exc)
                return result

            result.attempts += 1
            result.raw_responses.append(raw)

            parsed = prompts.parse_response(raw, package)
            result.parsed = parsed

            if parsed.status == "ok":
                result.status = STATUS_OK
                result.error = None
                return result

            # Malformed envelope. Re-send the original contract plus the
            # parser's complaint, using the existing correction format.
            if attempt < self.max_correction_retries:
                logger.warning(
                    "group %s: response rejected (%s); sending correction attempt %d",
                    group, parsed.status, attempt + 1,
                )
                messages = base_messages + [
                    {"role": "user", "content": prompts.render_correction(parsed.errors)}
                ]
                continue

            result.status = STATUS_INVALID
            result.error = "; ".join(parsed.errors) or parsed.status
            logger.error("group %s: unusable response after %d attempt(s)", group, result.attempts)
            return result

        return result  # pragma: no cover - loop always returns

    def _call_provider(self, messages, package, schema) -> str:
        """Call the provider, retrying only transient failures."""
        last_error: Optional[ProviderCallError] = None

        for attempt in range(self.max_transient_retries + 1):
            try:
                return self.provider.complete(
                    messages,
                    max_output_tokens=package.max_output_tokens,
                    response_schema=schema,
                    schema_name=f"extract_{package.group}",
                )
            except ProviderCallError as exc:
                last_error = exc
                if attempt >= self.max_transient_retries:
                    break
                delay = self.backoff_factor ** attempt
                logger.warning(
                    "group %s: transient provider failure (%s); retry %d in %.1fs",
                    package.group, exc, attempt + 1, delay,
                )
                self._sleep(delay)

        raise last_error if last_error else ProviderCallError("provider call failed")

    # --- all groups ----------------------------------------------------------

    def extract_bid(self, builder, bid_id: str) -> Dict[str, GroupExtractionResult]:
        """Run every configured group for one bid, in the locked group order."""
        results: Dict[str, GroupExtractionResult] = {}
        for group in config.GROUP_ORDER:
            for package in builder.build_group(bid_id, group):
                results[group] = self.extract_group(package)
                # One package per group today; batching stays a later concern.
                break
        return results
