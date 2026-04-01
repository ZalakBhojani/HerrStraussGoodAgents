from __future__ import annotations

import json
import logging

from herrstraussgoodagents.config import LLMConfig, get_llm_client, get_settings
from herrstraussgoodagents.handoff.context import enforce_token_budget
from herrstraussgoodagents.llm import LLMClient, Message, get_cost_tracker
from herrstraussgoodagents.models import AgentStage, HandoffContext, ResolutionPath

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM = """\
You are a handoff summarizer for a debt collections system. \
Given a conversation transcript between a collections agent and a borrower, \
produce a JSON summary with these exact fields:

{
  "debt_amount": <float>,
  "months_overdue": <int>,
  "offers_made": [<str>, ...],
  "objections_raised": [<str>, ...],
  "resolution_path": <"lump_sum" | "payment_plan" | "hardship_referral" | "unresolved">,
  "tone_summary": "<100 tokens max — borrower emotional state, cooperativeness, key dynamics>"
}

Rules:
- Be accurate about facts stated in the transcript.
- tone_summary must be under 100 tokens and qualitative only.
- Return ONLY valid JSON, no commentary.
"""


class HandoffSummarizer:
    """Converts a completed conversation transcript into a HandoffContext.

    Uses a single LLM call (batch-at-handoff strategy). If the resulting
    serialized context exceeds the 500-token budget, a second truncation
    call is made.
    """

    def __init__(self, llm_client: LLMClient, llm_config: LLMConfig) -> None:
        self.client = llm_client
        self.config = llm_config

    @classmethod
    def from_config(cls, llm_config: LLMConfig) -> "HandoffSummarizer":
        return cls(get_llm_client(llm_config), llm_config)

    async def summarize(
        self,
        messages: list[Message],
        source_stage: AgentStage,
        debt_amount: float,
        months_overdue: int,
    ) -> HandoffContext:
        """Summarize a conversation into a HandoffContext within the 500-token budget.

        Args:
            messages: Full conversation history (system + user/assistant turns).
            source_stage: Which agent stage produced this transcript.
            debt_amount: Known debt amount to validate/fill if LLM misses it.
            months_overdue: Known months overdue to validate/fill.
        """
        transcript = _format_transcript(messages)

        summary_messages: list[Message] = [
            {"role": "system", "content": _SUMMARIZE_SYSTEM},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ]

        llm_response = await self.client.complete(
            summary_messages,
            model=self.config.model,
            temperature=0.0,
            max_tokens=512,
        )
        get_cost_tracker().record(llm_response, "handoff:summarization")

        ctx = _parse_llm_response(llm_response.text, source_stage, debt_amount, months_overdue)

        # Enforce 500-token budget; re-summarize if needed
        settings = get_settings()
        ctx = await enforce_token_budget(ctx, self.client, self.config.model, settings.handoff_context_tokens)

        if ctx.token_count > settings.handoff_context_tokens:
            logger.warning(
                "Handoff context still over budget (%d tokens) after truncation; forcing hard trim.",
                ctx.token_count,
            )
            ctx = ctx.model_copy(update={"tone_summary": ctx.tone_summary[:120] + "…"})
            ctx.token_count = settings.handoff_context_tokens  # optimistic estimate

        logger.info(
            "Handoff summarized: stage=%s path=%s tokens=%d",
            source_stage,
            ctx.resolution_path,
            ctx.token_count,
        )
        return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_transcript(messages: list[Message]) -> str:
    """Convert message list to a readable transcript, skipping system messages."""
    lines: list[str] = []
    for m in messages:
        if m["role"] == "system":
            continue
        speaker = "Agent" if m["role"] == "assistant" else "Borrower"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def _parse_llm_response(
    raw: str,
    source_stage: AgentStage,
    debt_amount: float,
    months_overdue: int,
) -> HandoffContext:
    """Parse LLM JSON response into a HandoffContext, with safe fallbacks."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Handoff summarizer: failed to parse JSON, using defaults. Raw: %r", raw[:200])
        data = {}

    # Resolve path with fallback
    raw_path = data.get("resolution_path", "unresolved")
    try:
        path = ResolutionPath(raw_path)
    except ValueError:
        path = ResolutionPath.UNRESOLVED

    return HandoffContext(
        identity_verified=bool(data.get("identity_verified", False)),
        debt_amount=float(data.get("debt_amount") or debt_amount),
        months_overdue=int(data.get("months_overdue") or months_overdue),
        offers_made=list(data.get("offers_made") or []),
        objections_raised=list(data.get("objections_raised") or []),
        resolution_path=path,
        tone_summary=str(data.get("tone_summary") or ""),
        source_stage=source_stage,
    )
