from __future__ import annotations

import logging

from herrstraussgoodagents.compliance.rules import ComplianceResult, check_all_with_context
from herrstraussgoodagents.config import AgentConfig, get_settings
from herrstraussgoodagents.llm import LLMClient, Message, get_cost_tracker
from herrstraussgoodagents.models import ConversationMessage, TurnSource

logger = logging.getLogger(__name__)

_MAX_REGENERATE_ATTEMPTS = 3
_COMPLIANCE_FALLBACK = (
    "I need to pause for a moment. Please hold while I review the details of your account."
)


class BaseAgent:
    def __init__(self, llm_client: LLMClient, config: AgentConfig) -> None:
        self.client = llm_client
        self.config = config
        self._messages: list[Message] = []
        self._transcript: list[ConversationMessage] = []

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        """Concatenate all non-empty mutable prompt sections into a system prompt.
        compliance_rules is included but never mutated by the learning loop."""
        p = self.config.prompt
        sections = [
            p.persona_header,
            p.goal_statement,
            p.behavioral_guidelines,
            p.compliance_rules,
            p.conversation_style,
        ]
        return "\n\n".join(s.strip() for s in sections if s.strip())

    # ------------------------------------------------------------------
    # Message history helpers
    # ------------------------------------------------------------------

    def init_messages(self, system_prompt: str) -> None:
        self._messages = [{"role": "system", "content": system_prompt}]
        self._transcript = []

    def add_system_message(self, system_prompt: str) -> None:
        self._messages.append({"role": "system", "content": system_prompt})

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._transcript.append(
            ConversationMessage(role="user", content=content, source=TurnSource.BORROWER)
        )

    def add_assistant_message(
        self, content: str, source: TurnSource = TurnSource.LLM
    ) -> None:
        self._messages.append({"role": "assistant", "content": content})
        self._transcript.append(
            ConversationMessage(role="assistant", content=content, source=source)
        )

    @property
    def transcript(self) -> list[ConversationMessage]:
        return list(self._transcript)

    # ------------------------------------------------------------------
    # History compaction
    # ------------------------------------------------------------------

    async def _maybe_compact_history(self) -> None:
        """Slide the message window when total tokens exceed the main context budget.

        Keeps system prompt + first 2 non-system messages (opening context) +
        last 4 messages (recent conversation). This preserves identity
        verification and initial context while keeping recent turns.
        """
        settings = get_settings()
        budget = settings.main_context_tokens
        total = await self.client.count_tokens(self._messages, self.config.llm.model)
        if total <= budget:
            return

        # Separate system messages from conversation turns
        system_msgs = [m for m in self._messages if m["role"] == "system"]
        conversation = [m for m in self._messages if m["role"] != "system"]

        if len(conversation) <= 6:
            # Not enough messages to compact meaningfully
            return

        opening = conversation[:2]
        tail = conversation[-4:]
        self._messages = system_msgs + opening + tail
        logger.warning(
            "History compacted: was ~%d tokens (budget %d), kept %d system + "
            "2 opening + 4 recent = %d messages",
            total,
            budget,
            len(system_msgs),
            len(self._messages),
        )

    # ------------------------------------------------------------------
    # LLM call with compliance pre-check
    # ------------------------------------------------------------------

    async def generate(self, cost_tag: str = "") -> str:
        """Call LLM and compliance-check the response.
        Regenerates up to 3 times on violation; returns a safe fallback if all fail.
        Token budget is enforced only at handoff, not on every call."""
        await self._maybe_compact_history()

        # Find last borrower message for context-sensitive checks (rules 6 + 7)
        prior_borrower = ""
        for m in reversed(self._messages):
            if m["role"] == "user":
                prior_borrower = m["content"]
                break

        tracker = get_cost_tracker()
        tag = cost_tag or f"agent:{self.config.prompt.persona_header[:20]}"

        for attempt in range(_MAX_REGENERATE_ATTEMPTS):
            llm_response = await self.client.complete(
                self._messages,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            tracker.record(llm_response, tag)

            result: ComplianceResult = check_all_with_context(llm_response.text, prior_borrower)
            if result.passed:
                return llm_response.text

            # Inject a corrective hint and retry
            if attempt < _MAX_REGENERATE_ATTEMPTS - 1:
                self._messages.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM CORRECTION: Your last response violated a compliance rule "
                        f"({result.violation}). Rewrite it without this violation.]"
                    ),
                })

        return _COMPLIANCE_FALLBACK
