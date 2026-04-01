from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

from herrstraussgoodagents.agents.base import BaseAgent
from herrstraussgoodagents.config import AgentConfig, get_llm_client
from herrstraussgoodagents.handoff.context import handoff_to_text
from herrstraussgoodagents.models import (
    AgentOutcome,
    AgentStage,
    BorrowerCase,
    HandoffContext,
    OutcomeStatus,
    ResolutionPath,
    TurnSource,
)

logger = logging.getLogger(__name__)

# Lump-sum settlement tiers (% of principal). Steps down after 2 objections.
_LUMP_SUM_TIERS = [0.80, 0.70, 0.60]

_AGREEMENT_KEYWORDS = [
    "yes", "i'll do it", "i agree", "sounds good", "let's do it",
    "i can do that", "deal", "ok let's", "i accept", "i'll pay",
    "i will pay", "set it up", "go ahead",
]
_HANGUP_KEYWORDS = [
    "goodbye", "i'm done", "stop", "not interested", "leave me alone",
    "don't call me", "hang up", "bye",
]
_CEASE_KEYWORDS = ["stop calling", "cease", "don't contact", "do not contact", "remove me"]
_CALLBACK_KEYWORDS = ["call me back", "call back", "later", "not right now", "another time"]
_OBJECTION_SIGNALS = [
    "can't afford", "cannot afford", "too much", "too high", "not possible",
    "no", "i don't have", "don't have that", "what else", "lower",
]


class SessionSignal(Enum):
    """What the agent wants the pipeline to do after this turn."""
    CONTINUE = auto()
    END_RESOLVED = auto()
    END_HUNG_UP = auto()
    END_CEASE = auto()
    END_CALLBACK = auto()
    END_UNRESOLVED = auto()


@dataclass
class TurnResult:
    response: str
    signal: SessionSignal


class ResolutionAgent(BaseAgent):
    """Agent 2 (voice): negotiates settlement inside the Pipecat voice pipeline.

    This agent does NOT run its own event loop.  Instead the Pipecat
    VertexAILLMProcessor calls ``process_turn()`` on each finalized
    transcription and acts on the returned ``SessionSignal``.

    Settlement ladder (lump-sum path):
      80% → 70% → 60% of principal.  Steps down after 2 objections per tier.
    """

    def __init__(
        self,
        config: AgentConfig,
        case: BorrowerCase,
        handoff: HandoffContext,
    ) -> None:
        super().__init__(get_llm_client(config.llm), config)
        self.case = case
        self.handoff = handoff

        self._resolution_path: ResolutionPath = (
            handoff.resolution_path or ResolutionPath.PAYMENT_PLAN
        )
        self._lump_sum_tier_idx: int = 0
        self._objection_count: int = 0
        self._offers_made: list[str] = list(handoff.offers_made)
        self._objections: list[str] = list(handoff.objections_raised)
        self._settlement_amount: float | None = None
        self._payment_plan_months: int | None = None
        self._turns: int = 0

    # ------------------------------------------------------------------
    # Prompt assembly — injects handoff context into system prompt
    # ------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        base = super().build_system_prompt()
        handoff_block = (
            "\n\n--- HANDOFF CONTEXT FROM ASSESSMENT ---\n"
            + handoff_to_text(self.handoff)
            + "\n--- END HANDOFF CONTEXT ---"
        )
        return base + handoff_block

    # ------------------------------------------------------------------
    # Initialise conversation (called once when pipeline starts)
    # ------------------------------------------------------------------

    def opening_line(self) -> str:
        """Return the agent's first spoken line; initialises message history."""
        system_prompt = self.build_system_prompt()
        self.init_messages(system_prompt)

        opening = self.config.prompt.opening_script.format(
            borrower_name=self.case.borrower_name
        )
        offer_line = self._build_offer_line()
        full_opening = f"{opening} {offer_line}"
        self.add_assistant_message(full_opening, source=TurnSource.DETERMINISTIC)
        return full_opening

    # ------------------------------------------------------------------
    # Per-turn handler — called by VertexAILLMProcessor
    # ------------------------------------------------------------------

    async def process_turn(self, borrower_text: str) -> TurnResult:
        """Process one borrower utterance and return a spoken response + signal.

        The Pipecat processor should call this on every finalized transcription
        and act on the ``SessionSignal``:
          - CONTINUE     → keep the pipeline running
          - END_*        → push EndFrame and collect outcome
        """
        self._turns += 1
        msg_lower = borrower_text.lower()

        # --- Hard-exit conditions (no LLM needed) ---
        if any(kw in msg_lower for kw in _CEASE_KEYWORDS):
            response = (
                "I understand and will honor your request. "
                "I will note the cease-communication request on your account. Goodbye."
            )
            self.add_user_message(borrower_text)
            self.add_assistant_message(response, source=TurnSource.DETERMINISTIC)
            return TurnResult(response=response, signal=SessionSignal.END_CEASE)

        if any(kw in msg_lower for kw in _HANGUP_KEYWORDS):
            response = "Thank you for your time. Goodbye."
            self.add_user_message(borrower_text)
            self.add_assistant_message(response, source=TurnSource.DETERMINISTIC)
            return TurnResult(response=response, signal=SessionSignal.END_HUNG_UP)

        if any(kw in msg_lower for kw in _CALLBACK_KEYWORDS):
            response = (
                "Understood. I'll note a callback request on your account. "
                "Please be aware this offer is time-sensitive. Goodbye."
            )
            self.add_user_message(borrower_text)
            self.add_assistant_message(response, source=TurnSource.DETERMINISTIC)
            return TurnResult(response=response, signal=SessionSignal.END_CALLBACK)

        # --- Agreement detection ---
        if any(kw in msg_lower for kw in _AGREEMENT_KEYWORDS):
            self._record_agreement()
            response = self._build_confirmation()
            self.add_user_message(borrower_text)
            self.add_assistant_message(response, source=TurnSource.DETERMINISTIC)
            return TurnResult(response=response, signal=SessionSignal.END_RESOLVED)

        # --- Objection handling ---
        if self._is_objection(msg_lower):
            self._objections.append(borrower_text[:100])
            self._objection_count += 1
            if self._objection_count >= 2 and self._resolution_path == ResolutionPath.LUMP_SUM:
                self._step_down_offer()

        # --- Standard LLM turn ---
        self.add_user_message(borrower_text)
        response = await self.generate(cost_tag="agent:resolution")
        self.add_assistant_message(response)
        return TurnResult(response=response, signal=SessionSignal.CONTINUE)

    # ------------------------------------------------------------------
    # Outcome builder — called by pipeline when session ends
    # ------------------------------------------------------------------

    def build_outcome(self, signal: SessionSignal) -> AgentOutcome:
        status_map = {
            SessionSignal.END_RESOLVED: OutcomeStatus.RESOLVED,
            SessionSignal.END_HUNG_UP: OutcomeStatus.HUNG_UP,
            SessionSignal.END_CEASE: OutcomeStatus.CEASE_REQUESTED,
            SessionSignal.END_CALLBACK: OutcomeStatus.ESCALATED,
            SessionSignal.END_UNRESOLVED: OutcomeStatus.UNRESOLVED,
            SessionSignal.CONTINUE: OutcomeStatus.UNRESOLVED,
        }
        return AgentOutcome(
            stage=AgentStage.RESOLUTION,
            status=status_map.get(signal, OutcomeStatus.UNRESOLVED),
            resolution_path=self._resolution_path,
            settlement_amount=self._settlement_amount,
            payment_plan_months=self._payment_plan_months,
            handoff_context=HandoffContext(
                identity_verified=self.handoff.identity_verified,
                debt_amount=self.case.debt_amount,
                months_overdue=self.case.months_overdue,
                offers_made=self._offers_made,
                objections_raised=self._objections,
                resolution_path=self._resolution_path,
                tone_summary=self.handoff.tone_summary,
                source_stage=AgentStage.RESOLUTION,
            ),
            transcript=self.transcript,
            turns_taken=self._turns,
        )

    # ------------------------------------------------------------------
    # Settlement helpers
    # ------------------------------------------------------------------

    def _build_offer_line(self) -> str:
        if self._resolution_path == ResolutionPath.HARDSHIP_REFERRAL:
            return (
                "Based on your situation, I'd like to connect you with our hardship assistance program, "
                "which can pause collection activity while we find a sustainable solution."
            )
        if self._resolution_path == ResolutionPath.LUMP_SUM:
            pct = _LUMP_SUM_TIERS[self._lump_sum_tier_idx]
            amount = self.case.debt_amount * pct
            self._offers_made.append(f"Lump-sum {int(pct * 100)}%: ${amount:,.2f}")
            return (
                f"I can offer a one-time settlement of ${amount:,.2f} "
                f"— that's {int(pct * 100)}% of your balance — to fully resolve your account today."
            )
        # Payment plan (default)
        monthly = self.case.debt_amount / 6
        self._offers_made.append(f"Payment plan 6mo: ${monthly:,.2f}/mo")
        return (
            f"I can set up a 6-month payment plan at ${monthly:,.2f} per month. "
            "We also have 3-month and 12-month options if that works better."
        )

    def _step_down_offer(self) -> None:
        if self._lump_sum_tier_idx < len(_LUMP_SUM_TIERS) - 1:
            self._lump_sum_tier_idx += 1
            self._objection_count = 0
            pct = _LUMP_SUM_TIERS[self._lump_sum_tier_idx]
            amount = self.case.debt_amount * pct
            self._offers_made.append(f"Lump-sum {int(pct * 100)}%: ${amount:,.2f}")
            logger.info("Stepped lump-sum offer down to %.0f%%", pct * 100)

    def _record_agreement(self) -> None:
        if self._resolution_path == ResolutionPath.LUMP_SUM:
            pct = _LUMP_SUM_TIERS[self._lump_sum_tier_idx]
            self._settlement_amount = self.case.debt_amount * pct
        elif self._resolution_path == ResolutionPath.PAYMENT_PLAN:
            self._payment_plan_months = 6

    def _build_confirmation(self) -> str:
        if self._resolution_path == ResolutionPath.LUMP_SUM and self._settlement_amount:
            return (
                f"Excellent. To confirm: you've agreed to a settlement of "
                f"${self._settlement_amount:,.2f} to fully resolve your account. "
                "You'll receive written confirmation within 24 hours. Thank you."
            )
        if self._resolution_path == ResolutionPath.PAYMENT_PLAN:
            months = self._payment_plan_months or 6
            monthly = self.case.debt_amount / months
            return (
                f"Confirmed — ${monthly:,.2f} per month for {months} months. "
                "Written confirmation within 24 hours. Thank you."
            )
        return "Thank you for agreeing to resolve your account. We'll send confirmation shortly."

    @staticmethod
    def _is_objection(msg_lower: str) -> bool:
        return any(kw in msg_lower for kw in _OBJECTION_SIGNALS)
