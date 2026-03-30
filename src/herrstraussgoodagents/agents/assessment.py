from __future__ import annotations

import asyncio

from herrstraussgoodagents.agents.base import BaseAgent
from herrstraussgoodagents.config import AgentConfig, get_llm_client
from herrstraussgoodagents.models import (
    AgentOutcome,
    AgentStage,
    BorrowerCase,
    HandoffContext,
    OutcomeStatus,
    ResolutionPath,
)

MAX_TURNS = 20
_BORROWER_TIMEOUT_SECONDS = 300  # 5 min idle timeout

# Keywords detected in borrower messages to infer state
_CEASE_KEYWORDS = ["stop calling", "cease", "don't contact", "do not contact", "remove me"]
_HARDSHIP_KEYWORDS = ["can't afford", "cannot afford", "lost my job", "medical", "hardship", "disability", "unemployed"]
_LUMP_SUM_KEYWORDS = ["pay in full", "lump sum", "pay it all", "full amount", "one payment"]
_PAYMENT_PLAN_KEYWORDS = ["payment plan", "monthly", "installment", "over time", "split"]


class AssessmentAgent(BaseAgent):
    """Agent 1 (chat): verifies identity, gathers situation, determines resolution path."""

    def __init__(self, config: AgentConfig, case: BorrowerCase) -> None:
        super().__init__(get_llm_client(config.llm), config)
        self.case = case
        self._identity_verified = False
        self._identity_attempts = 0
        self._resolution_path: ResolutionPath | None = None
        self._objections: list[str] = []
        self._tone_notes: list[str] = []

    async def run(
        self,
        inbound: asyncio.Queue[str],
        outbound: asyncio.Queue[str],
    ) -> AgentOutcome:
        system_prompt = self.build_system_prompt()
        self.init_messages(system_prompt)

        # Send opening message immediately
        opening = self.config.prompt.opening_script.format(
            borrower_name=self.case.borrower_name
        )
        self.add_assistant_message(opening)
        await outbound.put(opening)

        turns = 0
        while turns < MAX_TURNS:
            try:
                borrower_msg = await asyncio.wait_for(
                    inbound.get(), timeout=_BORROWER_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                await outbound.put("It appears you may have stepped away. Feel free to reconnect when you are ready.")
                await outbound.put("__DONE__")
                return AgentOutcome(
                    stage=AgentStage.ASSESSMENT,
                    status=OutcomeStatus.UNRESOLVED,
                    turns_taken=turns,
                )

            turns += 1
            msg_lower = borrower_msg.lower()

            # Cease-and-desist detection
            # todo/discuss: imo we should just flag the account in real scenarios, but the mermaid diagram in the problem says to 
            # move to the voice agent
            if any(kw in msg_lower for kw in _CEASE_KEYWORDS):
                farewell = (
                    "I understand and will honor your request. "
                    "I will note that you have requested no further contact. "
                    "Thank you. Goodbye."
                )
                await outbound.put(farewell)
                await outbound.put("__DONE__")
                return AgentOutcome(
                    stage=AgentStage.ASSESSMENT,
                    status=OutcomeStatus.CEASE_REQUESTED,
                    turns_taken=turns,
                )

            # Identity verification: check if borrower provided the last 4 digits
            if not self._identity_verified:
                # todo(gap): strategyPattern to identify using other partial accounts information
                if self.case.account_last_four in borrower_msg:
                    self._identity_verified = True
                    self._tone_notes.append("Identity verified via account last 4.")
                else:
                    self._identity_attempts += 1
                    if self._identity_attempts >= 3:
                        farewell = (
                            "I was unable to verify your identity after several attempts. "
                            "For the security of your account, I cannot continue. "
                            "Please call us back with your account information. Goodbye."
                        )
                        await outbound.put(farewell)
                        await outbound.put("__DONE__")
                        return AgentOutcome(
                            stage=AgentStage.ASSESSMENT,
                            status=OutcomeStatus.IDENTITY_FAILED,
                            turns_taken=turns,
                        )

            # Infer resolution path from borrower language
            if self._resolution_path is None:
                if any(kw in msg_lower for kw in _HARDSHIP_KEYWORDS):
                    self._resolution_path = ResolutionPath.HARDSHIP_REFERRAL
                    self._tone_notes.append("Borrower indicated financial hardship.")
                elif any(kw in msg_lower for kw in _LUMP_SUM_KEYWORDS):
                    self._resolution_path = ResolutionPath.LUMP_SUM
                elif any(kw in msg_lower for kw in _PAYMENT_PLAN_KEYWORDS):
                    self._resolution_path = ResolutionPath.PAYMENT_PLAN

            self.add_user_message(borrower_msg)
            response = await self.generate()
            self.add_assistant_message(response)
            await outbound.put(response)

            # Complete assessment once identity is confirmed and we have enough turns
            if self._identity_verified and turns >= 4:
                break

        # Default resolution path if none detected
        if self._resolution_path is None:
            self._resolution_path = ResolutionPath.PAYMENT_PLAN

        handoff = HandoffContext(
            identity_verified=self._identity_verified,
            debt_amount=self.case.debt_amount,
            months_overdue=self.case.months_overdue,
            offers_made=[],
            objections_raised=self._objections,
            resolution_path=self._resolution_path,
            tone_summary=" ".join(self._tone_notes) if self._tone_notes else "Borrower was cooperative.",
            source_stage=AgentStage.ASSESSMENT,
        )

        await outbound.put("__DONE__")
        return AgentOutcome(
            stage=AgentStage.ASSESSMENT,
            status=OutcomeStatus.ESCALATED,
            resolution_path=self._resolution_path,
            handoff_context=handoff,
            turns_taken=turns,
        )
