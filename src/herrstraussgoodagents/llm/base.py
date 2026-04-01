from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, TypedDict

logger = logging.getLogger(__name__)


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """Return type for all LLM calls — carries text + usage metadata."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float


# ---------------------------------------------------------------------------
# Pricing table (per 1M tokens)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-20250414": {"input": 0.80, "output": 4.00},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from token counts. Returns 0.0 for unknown models."""
    rates = PRICING.get(model)
    if rates is None:
        logger.warning("No pricing data for model %r — cost will be 0.0", model)
        return 0.0
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Cost tracker (singleton)
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tag: str


class BudgetExhausted(Exception):
    """Raised when the LLM spend budget has been exceeded."""


@dataclass
class CostTracker:
    budget_usd: float = 20.0
    abort_at_usd: float = 18.0
    total_usd: float = 0.0
    calls: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, response: LLMResponse, tag: str) -> None:
        record = CallRecord(
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            tag=tag,
        )
        with self._lock:
            self.calls.append(record)
            self.total_usd += response.cost_usd

    def check_budget(self) -> None:
        """Raise BudgetExhausted if spend has reached the abort threshold."""
        if self.total_usd >= self.abort_at_usd:
            raise BudgetExhausted(
                f"LLM spend ${self.total_usd:.4f} has reached abort threshold "
                f"${self.abort_at_usd:.2f} (budget ${self.budget_usd:.2f})"
            )

    def breakdown(self) -> dict[str, dict]:
        """Return cost breakdown grouped by tag."""
        groups: dict[str, dict] = {}
        for call in self.calls:
            if call.tag not in groups:
                groups[call.tag] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            g = groups[call.tag]
            g["calls"] += 1
            g["input_tokens"] += call.input_tokens
            g["output_tokens"] += call.output_tokens
            g["cost_usd"] += call.cost_usd
        return groups

    def report(self) -> str:
        """Formatted cost breakdown string."""
        lines = [f"Total LLM spend: ${self.total_usd:.4f} / ${self.budget_usd:.2f}"]
        lines.append(f"Total calls: {len(self.calls)}")
        lines.append("")
        for tag, info in sorted(self.breakdown().items()):
            lines.append(
                f"  {tag:40s}  calls={info['calls']:4d}  "
                f"in={info['input_tokens']:8d}  out={info['output_tokens']:8d}  "
                f"${info['cost_usd']:.4f}"
            )
        return "\n".join(lines)


# Module-level singleton — import and use directly
_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def reset_cost_tracker(**kwargs) -> CostTracker:
    """Reset the singleton (useful for tests and new learning loop runs)."""
    global _cost_tracker
    _cost_tracker = CostTracker(**kwargs)
    return _cost_tracker


# ---------------------------------------------------------------------------
# Abstract LLM client
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse: ...

    @abstractmethod
    async def count_tokens(
        self,
        messages: list[Message],
        model: str,
    ) -> int: ...
