from __future__ import annotations

import json

from herrstraussgoodagents.llm import LLMClient, Message
from herrstraussgoodagents.models import HandoffContext

_HANDOFF_TOKEN_BUDGET = 500


def handoff_to_text(ctx: HandoffContext) -> str:
    """Serialize a HandoffContext to a compact text string for injection into agent prompts."""
    lines = [
        f"identity_verified: {ctx.identity_verified}",
        f"debt_amount: ${ctx.debt_amount:,.2f}",
        f"months_overdue: {ctx.months_overdue}",
        f"resolution_path: {ctx.resolution_path.value if ctx.resolution_path else 'unknown'}",
    ]
    if ctx.offers_made:
        lines.append(f"offers_made: {'; '.join(ctx.offers_made)}")
    if ctx.objections_raised:
        lines.append(f"objections_raised: {'; '.join(ctx.objections_raised)}")
    if ctx.tone_summary:
        lines.append(f"tone_summary: {ctx.tone_summary}")
    return "\n".join(lines)


async def enforce_token_budget(
    ctx: HandoffContext,
    client: LLMClient,
    model: str,
    budget: int = _HANDOFF_TOKEN_BUDGET,
) -> HandoffContext:
    """Check token count of serialized handoff. If over budget, truncate prose fields."""
    text = handoff_to_text(ctx)
    messages: list[Message] = [{"role": "user", "content": text}]
    count = await client.count_tokens(messages, model)

    if count <= budget:
        ctx.token_count = count
        return ctx

    # Truncate tone_summary to bring within budget. Hard-truncate to ~200 chars.
    truncated = ctx.tone_summary[:200] + "…" if len(ctx.tone_summary) > 200 else ctx.tone_summary
    ctx = ctx.model_copy(update={"tone_summary": truncated})

    # Re-check; if still over, trim objections list
    text = handoff_to_text(ctx)
    messages = [{"role": "user", "content": text}]
    count = await client.count_tokens(messages, model)
    if count > budget and ctx.objections_raised:
        ctx = ctx.model_copy(update={"objections_raised": ctx.objections_raised[:2]})
        text = handoff_to_text(ctx)
        messages = [{"role": "user", "content": text}]
        count = await client.count_tokens(messages, model)

    ctx.token_count = count
    return ctx
