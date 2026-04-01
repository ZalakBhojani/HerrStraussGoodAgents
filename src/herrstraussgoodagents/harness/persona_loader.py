"""Persona loader and LLM-backed borrower simulator.

Loads YAML persona configs and provides an LLM-backed borrower that
generates responses in character.  Uses the cheap model (Flash) to
keep simulation costs low.
"""

from __future__ import annotations

import logging
from pathlib import Path

from herrstraussgoodagents.config import (
    LLMConfig,
    PersonaConfig,
    get_llm_client,
    load_persona_config,
    CONFIGS_DIR,
)
from herrstraussgoodagents.llm import LLMClient, Message, get_cost_tracker

logger = logging.getLogger(__name__)

# All available persona IDs (stem of YAML filenames in configs/personas/)
PERSONA_IDS: list[str] = [
    p.stem for p in sorted((CONFIGS_DIR / "personas").glob("*.yaml"))
]

# Default LLM config for persona simulation (cheap + creative)
PERSONA_LLM_CONFIG = LLMConfig(
    provider="vertexai",
    model="gemini-2.5-flash",
    temperature=0.8,
    max_tokens=256,
)


def load_all_personas() -> list[PersonaConfig]:
    """Load all persona configs from the personas directory."""
    return [load_persona_config(pid) for pid in PERSONA_IDS]


class BorrowerSimulator:
    """LLM-backed borrower that responds in character based on a persona config.

    Maintains its own conversation history (persona perspective) separate
    from the agent's history.  This dual-perspective approach means the
    persona sees the conversation from the borrower's point of view.
    """

    def __init__(
        self,
        persona: PersonaConfig,
        llm_client: LLMClient | None = None,
        llm_config: LLMConfig | None = None,
        account_last_four: str = "1234",
    ) -> None:
        config = llm_config or PERSONA_LLM_CONFIG
        self.client = llm_client or get_llm_client(config)
        self.config = config
        self.persona = persona
        self.account_last_four = account_last_four
        self._messages: list[Message] = []
        self._init_system_prompt()

    def _init_system_prompt(self) -> None:
        system_prompt = (
            f"{self.persona.system_prompt.strip()}\n\n"
            f"Your account's last 4 digits are: {self.account_last_four}\n\n"
            "Rules for your responses:\n"
            "- Stay in character at all times.\n"
            "- Respond naturally as a real person would on a collections call.\n"
            "- Keep responses under 60 words (people don't monologue on calls).\n"
            "- Do NOT break character or mention you are an AI.\n"
            "- Do NOT repeat yourself verbatim across turns.\n"
            "- If you decide to agree, use one of your resolution keywords naturally.\n"
            "- If you decide to hang up, use one of your hangup keywords naturally.\n"
        )
        self._messages = [{"role": "system", "content": system_prompt}]

    async def respond(self, agent_message: str) -> str:
        """Generate a borrower response to an agent message."""
        self._messages.append({"role": "user", "content": f"[Agent says]: {agent_message}"})

        tracker = get_cost_tracker()
        tracker.check_budget()

        llm_response = await self.client.complete(
            self._messages,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        tracker.record(llm_response, "simulation:persona")

        response = llm_response.text.strip()
        self._messages.append({"role": "assistant", "content": response})

        return response

    @property
    def history(self) -> list[Message]:
        """Return the persona's conversation history (for debugging)."""
        return list(self._messages)

    @classmethod
    def from_persona_id(
        cls,
        persona_id: str,
        account_last_four: str = "1234",
        llm_config: LLMConfig | None = None,
    ) -> "BorrowerSimulator":
        """Create a BorrowerSimulator from a persona ID string."""
        persona = load_persona_config(persona_id)
        return cls(
            persona=persona,
            llm_config=llm_config,
            account_last_four=account_last_four,
        )
