from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from herrstraussgoodagents.compliance.rules import ComplianceResult, check_all_with_context
from herrstraussgoodagents.config import AgentConfig, get_settings
from herrstraussgoodagents.llm import LLMClient, Message, get_cost_tracker
from herrstraussgoodagents.models import ConversationMessage, TurnSource

if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_context import LLMContext

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
        self._ext_context: LLMContext | None = None

    def set_llm_context(self, context: LLMContext) -> None:
        """Wire a Pipecat LLMContext as the backing store for conversation history.

        When set, ``add_user_message`` / ``add_assistant_message`` will skip
        updating ``_messages`` (the aggregators handle that), and ``generate``
        will read messages from the context instead of ``_messages``.
        """
        self._ext_context = context

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
        if self._ext_context is not None:
            self._ext_context.set_messages([{"role": "system", "content": system_prompt}])
        else:
            self._messages = [{"role": "system", "content": system_prompt}]
        self._transcript = []

    def add_system_message(self, system_prompt: str) -> None:
        self._messages.append({"role": "system", "content": system_prompt})

    def add_user_message(self, content: str) -> None:
        self._transcript.append(
            ConversationMessage(role="user", content=content, source=TurnSource.BORROWER)
        )
        # When using external context, the LLMUserAggregator has already added
        # this message to LLMContext — skip to avoid duplicates.
        if self._ext_context is None:
            self._messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self, content: str, source: TurnSource = TurnSource.LLM
    ) -> None:
        self._transcript.append(
            ConversationMessage(role="assistant", content=content, source=source)
        )
        # When using external context, the LLMAssistantAggregator captures the
        # response from the emitted frames — skip to avoid duplicates.
        if self._ext_context is None:
            self._messages.append({"role": "assistant", "content": content})

    @property
    def transcript(self) -> list[ConversationMessage]:
        return list(self._transcript)

    # ------------------------------------------------------------------
    # History compaction
    # ------------------------------------------------------------------

    # This is currently used by text agents as we are managing their own context
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
        When an external LLMContext is wired in (voice path), it is used as-is;
        otherwise the internal _messages list is compacted first."""
        if self._ext_context is not None:
            messages = self._ext_context.get_messages()
        else:
            await self._maybe_compact_history()
            messages = self._messages

        # Find last borrower message for context-sensitive checks (rules 6 + 7)
        prior_borrower = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                prior_borrower = m.get("content", "")
                break

        tracker = get_cost_tracker()
        tag = cost_tag or f"agent:{self.config.prompt.persona_header[:20]}"

        for attempt in range(_MAX_REGENERATE_ATTEMPTS):
            llm_response = await self.client.complete(
                messages,
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
                correction = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM CORRECTION: Your last response violated a compliance rule "
                        f"({result.violation}). Rewrite it without this violation.]"
                    ),
                }
                if self._ext_context is not None:
                    self._ext_context.add_message(correction)
                    messages = self._ext_context.get_messages()
                else:
                    self._messages.append(correction)
                    messages = self._messages

        return _COMPLIANCE_FALLBACK
