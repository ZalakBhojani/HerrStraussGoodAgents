from __future__ import annotations

from herrstraussgoodagents.compliance.rules import ComplianceResult, check_all_with_context
from herrstraussgoodagents.config import AgentConfig
from herrstraussgoodagents.llm import LLMClient, Message

_MAX_REGENERATE_ATTEMPTS = 3
_COMPLIANCE_FALLBACK = (
    "I need to pause for a moment. Please hold while I review the details of your account."
)


class BaseAgent:
    def __init__(self, llm_client: LLMClient, config: AgentConfig) -> None:
        self.client = llm_client
        self.config = config
        self._messages: list[Message] = []

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

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    # ------------------------------------------------------------------
    # LLM call with compliance pre-check
    # ------------------------------------------------------------------

    async def generate(self) -> str:
        """Call LLM and compliance-check the response.
        Regenerates up to 3 times on violation; returns a safe fallback if all fail.
        Token budget is enforced only at handoff, not on every call."""
        # Find last borrower message for context-sensitive checks (rules 6 + 7)
        prior_borrower = ""
        for m in reversed(self._messages):
            if m["role"] == "user":
                prior_borrower = m["content"]
                break

        for attempt in range(_MAX_REGENERATE_ATTEMPTS):
            response = await self.client.complete(
                self._messages,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            result: ComplianceResult = check_all_with_context(response, prior_borrower)
            if result.passed:
                return response

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
