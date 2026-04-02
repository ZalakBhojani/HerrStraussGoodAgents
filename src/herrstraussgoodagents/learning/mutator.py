"""Failure-driven prompt mutator.

Receives a FailureReport from the failure analyzer and proposes targeted
prompt mutations.  Uses the strong model (Gemini Pro) to generate changes.

Constraints:
  - Max 2 prompt sections mutated per iteration.
  - compliance_rules section is LOCKED — never mutated.
  - Token budget validated post-mutation.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field

from herrstraussgoodagents.config import AgentConfig, LLMConfig, get_llm_client
from herrstraussgoodagents.learning.failure_analyzer import FailureReport, WeakSession
from herrstraussgoodagents.llm import LLMClient, Message, get_cost_tracker

logger = logging.getLogger(__name__)

# Sections the mutator is allowed to change
MUTABLE_SECTIONS = [
    "persona_header",
    "goal_statement",
    "behavioral_guidelines",
    "conversation_style",
    "opening_script",
]

# compliance_rules is explicitly excluded
LOCKED_SECTIONS = ["compliance_rules"]

MAX_SECTIONS_PER_MUTATION = 2

MUTATOR_LLM_CONFIG = LLMConfig(
    provider="vertexai",
    model="gemini-2.5-pro",
    temperature=0.3,
    max_tokens=2048,
)


@dataclass
class PromptMutation:
    """A proposed change to an agent's prompt."""
    sections_changed: list[str]
    changes: dict[str, str]  # section_name -> new_value
    rationale: str
    parent_version: str


_MUTATION_SYSTEM = """\
You are a prompt engineer improving a debt collections AI agent. You will \
receive a failure analysis showing the agent's worst conversations and the \
specific metrics that are weak.

Your job: propose targeted changes to the agent's prompt to fix these failures.

The agent prompt has these mutable sections:
- persona_header: The agent's personality and tone
- goal_statement: What the agent is trying to achieve
- behavioral_guidelines: Specific rules for how the agent behaves
- conversation_style: Tone, register, verbosity preferences
- opening_script: The agent's first message (may contain {format_variables})

LOCKED (do NOT change): compliance_rules

Constraints:
- Change at most 2 sections.
- Keep changes targeted — fix the specific failure patterns, don't rewrite everything.
- Preserve any {format_variables} in opening_script (e.g., {borrower_first_name}).
- The agent must remain professional and FDCPA-compliant.
- Each section should be under 200 words.

Return ONLY valid JSON:
{
  "sections_to_mutate": ["section_name_1", "section_name_2"],
  "new_values": {
    "section_name_1": "new text for this section",
    "section_name_2": "new text for this section"
  },
  "rationale": "2-3 sentences explaining why these changes address the failures"
}
"""


def _build_mutation_prompt(
    current_config: AgentConfig,
    failure_report: FailureReport,
) -> str:
    """Build the user prompt with current config + failure analysis."""
    # Current prompt sections
    prompt = current_config.prompt
    current_sections = (
        f"=== CURRENT PROMPT SECTIONS ===\n"
        f"persona_header:\n{prompt.persona_header.strip()}\n\n"
        f"goal_statement:\n{prompt.goal_statement.strip()}\n\n"
        f"behavioral_guidelines:\n{prompt.behavioral_guidelines.strip()}\n\n"
        f"conversation_style:\n{prompt.conversation_style.strip()}\n\n"
        f"opening_script:\n{prompt.opening_script.strip()}\n"
        f"=== END CURRENT PROMPT ===\n\n"
    )

    # Failure analysis summary
    analysis = (
        f"=== FAILURE ANALYSIS ===\n"
        f"Agent stage: {failure_report.stage}\n"
        f"Evaluated: {failure_report.total_evaluated} conversations\n"
        f"Weak sessions: {failure_report.weak_count}\n"
        f"Recommendation: {failure_report.recommendation}\n\n"
        f"Weak metric averages across failures:\n"
    )
    for name, avg in failure_report.overall_weak_metrics.items():
        analysis += f"  {name}: {avg:.2f}/5.0\n"

    # Persona breakdown
    analysis += "\nPersona breakdown:\n"
    for pp in failure_report.persona_patterns:
        analysis += (
            f"  {pp.persona_id}: {pp.session_count} failures, "
            f"avg fitness={pp.avg_fitness:.2f}, "
            f"weak metrics={pp.common_weak_metrics}\n"
        )

    # Sample weak transcripts (top 3 worst)
    worst = sorted(failure_report.weak_sessions, key=lambda ws: ws.fitness)[:3]
    analysis += "\n=== SAMPLE WEAK TRANSCRIPTS ===\n"
    for ws in worst:
        analysis += (
            f"\n--- Session {ws.session_id} (fitness={ws.fitness:.2f}, "
            f"persona={ws.persona_id}, weakest={ws.weakest_metric}={ws.weakest_score:.1f}) ---\n"
            f"Evaluator reasoning: {ws.evaluation.overall_reasoning}\n"
            f"Transcript:\n{ws.transcript_summary}\n"
        )
    analysis += "=== END FAILURE ANALYSIS ===\n"

    return current_sections + analysis


def _parse_mutation(raw: str, parent_version: str) -> PromptMutation | None:
    """Parse LLM JSON response into a PromptMutation."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Mutator: failed to parse JSON: %r", raw[:300])
        return None

    sections = data.get("sections_to_mutate", [])
    new_values = data.get("new_values", {})
    rationale = data.get("rationale", "")

    # Validate
    if not sections or not new_values:
        logger.warning("Mutator: empty mutation proposal")
        return None

    # Filter out locked sections
    sections = [s for s in sections if s in MUTABLE_SECTIONS]
    new_values = {k: v for k, v in new_values.items() if k in MUTABLE_SECTIONS}

    if not sections:
        logger.warning("Mutator: all proposed sections are locked")
        return None

    # Enforce max sections
    if len(sections) > MAX_SECTIONS_PER_MUTATION:
        sections = sections[:MAX_SECTIONS_PER_MUTATION]
        new_values = {k: v for k, v in new_values.items() if k in sections}

    return PromptMutation(
        sections_changed=sections,
        changes=new_values,
        rationale=rationale,
        parent_version=parent_version,
    )


def apply_mutation(
    config: AgentConfig,
    mutation: PromptMutation,
) -> AgentConfig:
    """Apply a mutation to an AgentConfig, returning a new copy."""
    new_config = config.model_copy(deep=True)
    for section_name, new_value in mutation.changes.items():
        if hasattr(new_config.prompt, section_name):
            setattr(new_config.prompt, section_name, new_value)
        else:
            logger.warning("Mutator: unknown section %r, skipping", section_name)
    return new_config


class Mutator:
    """Proposes prompt mutations based on failure analysis."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        config = llm_config or MUTATOR_LLM_CONFIG
        self.client = llm_client or get_llm_client(config)
        self.config = config

    async def propose_mutation(
        self,
        current_config: AgentConfig,
        failure_report: FailureReport,
        parent_version: str = "v1",
    ) -> PromptMutation | None:
        """Propose a prompt mutation based on failure analysis.

        Returns None if the LLM response can't be parsed or all proposed
        sections are locked.
        """
        messages: list[Message] = [
            {"role": "system", "content": _MUTATION_SYSTEM},
            {"role": "user", "content": _build_mutation_prompt(current_config, failure_report)},
        ]

        tracker = get_cost_tracker()
        tracker.check_budget()

        llm_response = await self.client.complete(
            messages,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        tracker.record(llm_response, "mutation:proposal")

        mutation = _parse_mutation(llm_response.text, parent_version)
        if mutation:
            logger.info(
                "Mutation proposed: sections=%s, rationale=%s",
                mutation.sections_changed, mutation.rationale[:100],
            )
        else:
            logger.warning("Mutator failed to produce a valid mutation")

        return mutation
