from __future__ import annotations

import asyncio
import logging

from herrstraussgoodagents.agents.base import BaseAgent
from herrstraussgoodagents.config import AgentConfig, get_llm_client, get_settings
from herrstraussgoodagents.handoff.context import handoff_to_text
from herrstraussgoodagents.models import (
    AgentOutcome,
    AgentStage,
    BorrowerCase,
    HandoffContext,
    OutcomeStatus,
    ResolutionPath,
)

logger = logging.getLogger(__name__)

MAX_TURNS = 15
_BORROWER_TIMEOUT_SECONDS = 300

_AGREEMENT_KEYWORDS = [
    "yes", "i'll pay", "i will pay", "i agree", "i accept",
    "go ahead", "proceed", "ok", "fine", "deal", "i'll do it",
]
_REFUSAL_KEYWORDS = [
    "no", "refuse", "won't pay", "will not pay", "not going to",
    "forget it", "do your worst", "sue me", "i don't care",
]
_CEASE_KEYWORDS = ["stop calling", "cease", "don't contact", "do not contact", "remove me"]

# Hard deadline communicated to borrower
_DEADLINE_HOURS = 72


class FinalNoticeAgent(BaseAgent):
    """Agent 3 (chat): last-chance offer with hard 72-hour deadline.

    Receives the full handoff summary from Resolution. Outlines factual
    consequences (credit bureau reporting, legal referral, asset recovery)
    without threats. Offers one final settlement figure — does not negotiate
    below it.
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

        self._resolution_path = handoff.resolution_path or ResolutionPath.UNRESOLVED
        self._offers_made: list[str] = list(handoff.offers_made)
        self._objections: list[str] = list(handoff.objections_raised)
        self._final_offer_amount = self._compute_final_offer()
        self._committed = False
        self._refused = False

    def _compute_final_offer(self) -> float:
        """Final offer is 60% of principal — the lowest lump-sum tier."""
        return round(self.case.debt_amount * 0.60, 2)

    # ------------------------------------------------------------------
    # Prompt assembly — injects full handoff history
    # ------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        base = super().build_system_prompt()
        handoff_block = (
            "\n\n--- FULL CONVERSATION HANDOFF ---\n"
            + handoff_to_text(self.handoff)
            + f"\nfinal_offer_amount: ${self._final_offer_amount:,.2f}"
            + f"\ndeadline_hours: {_DEADLINE_HOURS}"
            + "\n--- END HANDOFF ---"
        )
        return base + handoff_block

    # ------------------------------------------------------------------
    # Main run loop (chat modality)
    # ------------------------------------------------------------------

    async def run(
        self,
        inbound: asyncio.Queue[str],
        outbound: asyncio.Queue[str],
    ) -> AgentOutcome:
        settings = get_settings()
        system_prompt = self.build_system_prompt()
        sp_tokens = await self.client.count_tokens(
            [{"role": "system", "content": system_prompt}], self.config.llm.model
        )
        base_budget = settings.main_context_tokens - settings.handoff_context_tokens  # 1500
        if sp_tokens > settings.main_context_tokens:
            logger.warning(
                "FinalNotice system prompt is %d tokens (total budget: %d)",
                sp_tokens,
                settings.main_context_tokens,
            )
        elif sp_tokens > base_budget:
            logger.info(
                "FinalNotice system prompt: %d tokens (using %d of 500-token handoff allocation)",
                sp_tokens,
                sp_tokens - base_budget,
            )
        else:
            logger.info("FinalNotice system prompt: %d / %d tokens", sp_tokens, settings.main_context_tokens)

        self.init_messages(system_prompt)

        opening = self.config.prompt.opening_script.format(
            account_last_four=self.case.account_last_four
        )
        offer_line = self._build_offer_line()
        full_opening = f"{opening}\n\n{offer_line}"
        self.add_assistant_message(full_opening)
        await outbound.put(full_opening)

        turns = 0
        while turns < MAX_TURNS:
            try:
                borrower_msg = await asyncio.wait_for(
                    inbound.get(), timeout=_BORROWER_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                timeout_msg = (
                    "As we have not heard from you, this notice will be processed "
                    "according to our standard escalation procedure. Goodbye."
                )
                await outbound.put(timeout_msg)
                await outbound.put("__DONE__")
                return self._make_outcome(OutcomeStatus.UNRESOLVED, turns)

            turns += 1
            msg_lower = borrower_msg.lower()

            # Cease-and-desist
            if any(kw in msg_lower for kw in _CEASE_KEYWORDS):
                farewell = (
                    "Your cease-communication request has been noted and will be honored. "
                    "Goodbye."
                )
                await outbound.put(farewell)
                await outbound.put("__DONE__")
                return self._make_outcome(OutcomeStatus.CEASE_REQUESTED, turns)

            # Agreement
            if any(kw in msg_lower for kw in _AGREEMENT_KEYWORDS) and not self._refused:
                self._committed = True
                confirmation = self._build_confirmation()
                self.add_user_message(borrower_msg)
                self.add_assistant_message(confirmation)
                await outbound.put(confirmation)
                await outbound.put("__DONE__")
                return self._make_outcome(OutcomeStatus.RESOLVED, turns)

            # Explicit refusal
            if any(kw in msg_lower for kw in _REFUSAL_KEYWORDS):
                self._refused = True
                refusal_acknowledgment = self._build_refusal_acknowledgment()
                self.add_user_message(borrower_msg)
                self.add_assistant_message(refusal_acknowledgment)
                await outbound.put(refusal_acknowledgment)
                await outbound.put("__DONE__")
                return self._make_outcome(OutcomeStatus.UNRESOLVED, turns)

            self.add_user_message(borrower_msg)
            response = await self.generate()
            self.add_assistant_message(response)
            await outbound.put(response)

        await outbound.put("__DONE__")
        return self._make_outcome(OutcomeStatus.UNRESOLVED, turns)

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    def _build_offer_line(self) -> str:
        self._offers_made.append(f"Final notice offer: ${self._final_offer_amount:,.2f}")
        return (
            f"Our final offer is a one-time settlement of ${self._final_offer_amount:,.2f} "
            f"to fully resolve your account. This offer expires in {_DEADLINE_HOURS} hours.\n\n"
            "Should this account remain unresolved after the deadline, the following will occur "
            "as standard procedure: the account will be reported to the major credit bureaus, "
            "the matter may be referred for legal review, and asset recovery processes may be initiated. "
            "These are procedural outcomes, not threats."
        )

    def _build_confirmation(self) -> str:
        return (
            f"Confirmed. Your settlement of ${self._final_offer_amount:,.2f} has been accepted. "
            "You will receive written confirmation with payment instructions within 24 hours. "
            "Once payment is received, all collection activity will cease and the account "
            "will be marked settled in full. Thank you."
        )

    def _build_refusal_acknowledgment(self) -> str:
        return (
            "Your response has been documented. The account will now proceed through "
            "the standard escalation process as outlined. "
            "You may contact us if you change your decision before the deadline. Goodbye."
        )

    def _make_outcome(self, status: OutcomeStatus, turns: int) -> AgentOutcome:
        return AgentOutcome(
            stage=AgentStage.FINAL_NOTICE,
            status=status,
            resolution_path=self._resolution_path,
            settlement_amount=self._final_offer_amount if self._committed else None,
            handoff_context=HandoffContext(
                identity_verified=self.handoff.identity_verified,
                debt_amount=self.case.debt_amount,
                months_overdue=self.case.months_overdue,
                offers_made=self._offers_made,
                objections_raised=self._objections,
                resolution_path=self._resolution_path,
                tone_summary=self.handoff.tone_summary,
                source_stage=AgentStage.FINAL_NOTICE,
            ),
            turns_taken=turns,
        )
