from __future__ import annotations

import asyncio
import json
import logging

from herrstraussgoodagents.agents.base import BaseAgent
from herrstraussgoodagents.config import AgentConfig, get_llm_client, get_settings
from herrstraussgoodagents.handoff.summarizer import HandoffSummarizer
from herrstraussgoodagents.models import (
    AgentOutcome,
    AgentStage,
    BorrowerCase,
    OutcomeStatus,
    ResolutionPath,
    TurnSource,
)

logger = logging.getLogger(__name__)

MAX_TURNS = 5
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

    def build_system_prompt(self) -> str:
        base = super().build_system_prompt()
        facts = {
            "borrower_name": self.case.borrower_name,
            "debt": self.case.debt_amount,
            "months_overdue": self.case.months_overdue,
            "creditor": self.case.original_creditor,
            "account_last_four": self.case.account_last_four,
        }
        facts_block = (
            "\n\n--- AUTHORITATIVE FACTS ---\n"
            f"{json.dumps(facts, indent=2)}\n\n"
            "Rules:\n"
            "- Treat the JSON above as ground truth.\n"
            "- Do NOT modify values.\n"
            "- If user disputes, acknowledge and reassert the facts.\n"
            "--- END FACTS ---"
        )
        return base + facts_block

    async def run(
        self,
        inbound: asyncio.Queue[str],
        outbound: asyncio.Queue[str],
    ) -> AgentOutcome:
        settings = get_settings()
        system_prompt = self.build_system_prompt()
        self.init_messages(system_prompt)

        init_tokens = await self.client.count_tokens(self._messages, self.config.llm.model)
        if init_tokens > settings.main_context_tokens:
            logger.warning(
                "Assessment initial context is %d tokens (budget: %d)",
                init_tokens,
                settings.main_context_tokens,
            )
        else:
            logger.info("Assessment initial context: %d / %d tokens", init_tokens, settings.main_context_tokens)

        # Send opening message immediately
        opening = self.config.prompt.opening_script.format(
            borrower_first_name=self.case.borrower_name
        )
        self.add_assistant_message(opening, source=TurnSource.DETERMINISTIC)
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
                    transcript=self.transcript,
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
                self.add_user_message(borrower_msg)
                self.add_assistant_message(farewell, source=TurnSource.DETERMINISTIC)
                await outbound.put(farewell)
                await outbound.put("__DONE__")
                return AgentOutcome(
                    stage=AgentStage.ASSESSMENT,
                    status=OutcomeStatus.CEASE_REQUESTED,
                    transcript=self.transcript,
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
                        self.add_user_message(borrower_msg)
                        self.add_assistant_message(farewell, source=TurnSource.DETERMINISTIC)
                        await outbound.put(farewell)
                        await outbound.put("__DONE__")
                        return AgentOutcome(
                            stage=AgentStage.ASSESSMENT,
                            status=OutcomeStatus.IDENTITY_FAILED,
                            transcript=self.transcript,
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
            response = await self.generate(cost_tag="agent:assessment")
            self.add_assistant_message(response)
            await outbound.put(response)

            # Complete assessment when we have the data we need:
            # identity verified + resolution path detected + at least 2 turns for context
            if self._identity_verified and self._resolution_path is not None and turns >= 2:
                break

        summarizer = HandoffSummarizer.from_config(self.config.llm)
        handoff = await summarizer.summarize(
            messages=self._messages,
            source_stage=AgentStage.ASSESSMENT,
            debt_amount=self.case.debt_amount,
            months_overdue=self.case.months_overdue,
        )

        # Preserve keyword-detected resolution path if LLM returned unresolved
        if handoff.resolution_path is None or handoff.resolution_path == ResolutionPath.UNRESOLVED:
            if self._resolution_path is not None:
                handoff = handoff.model_copy(update={"resolution_path": self._resolution_path})
            else:
                handoff = handoff.model_copy(update={"resolution_path": ResolutionPath.PAYMENT_PLAN})

        await outbound.put("__AGENT_DONE__")
        return AgentOutcome(
            stage=AgentStage.ASSESSMENT,
            status=OutcomeStatus.ESCALATED,
            resolution_path=self._resolution_path,
            handoff_context=handoff,
            transcript=self.transcript,
            turns_taken=turns,
        )
