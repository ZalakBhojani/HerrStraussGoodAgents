"""LLM-as-judge evaluator for agent conversations.

Scores conversations on per-agent metrics (Tier 1) and system-level
metrics (Tier 2).  Uses the strong model (Gemini Pro) for evaluation.
Returns structured results with reasoning for the failure analyzer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from herrstraussgoodagents.compliance.rules import check_all
from herrstraussgoodagents.config import (
    EvaluationRubric,
    LLMConfig,
    MetricConfig,
    get_llm_client,
    load_evaluation_rubric,
)
from herrstraussgoodagents.llm import LLMClient, Message, get_cost_tracker
from herrstraussgoodagents.models import (
    ConversationMessage,
    ConversationRecord,
    TurnSource,
)

logger = logging.getLogger(__name__)

# Strong model for evaluation
EVAL_LLM_CONFIG = LLMConfig(
    provider="vertexai",
    model="gemini-2.5-pro",
    temperature=0.0,
    max_tokens=1024,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MetricScore:
    name: str
    score: float
    weight: float
    reasoning: str


@dataclass
class ConversationEvaluation:
    """Full evaluation result for one conversation."""

    session_id: str
    stage: str
    persona_id: str
    # Per-agent metrics
    metric_scores: list[MetricScore] = field(default_factory=list)
    fitness: float = 0.0
    # Compliance
    compliance_violations: list[str] = field(default_factory=list)
    compliance_capped: bool = False
    # For failure analysis
    weakest_metric: str = ""
    overall_reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "stage": self.stage,
            "persona_id": self.persona_id,
            "metric_scores": {m.name: {"score": m.score, "weight": m.weight, "reasoning": m.reasoning} for m in self.metric_scores},
            "fitness": self.fitness,
            "compliance_violations": self.compliance_violations,
            "compliance_capped": self.compliance_capped,
            "weakest_metric": self.weakest_metric,
            "overall_reasoning": self.overall_reasoning,
        }


@dataclass
class PipelineEvaluation:
    """Evaluation result for a full pipeline run (system-level metrics)."""

    case_id: str
    persona_id: str
    per_stage: list[ConversationEvaluation] = field(default_factory=list)
    system_scores: list[MetricScore] = field(default_factory=list)
    system_fitness: float = 0.0
    combined_fitness: float = 0.0


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------

def _format_transcript(messages: list[ConversationMessage]) -> str:
    """Format transcript for the evaluator, annotating turn sources."""
    lines: list[str] = []
    for msg in messages:
        if msg.role == "user":
            lines.append(f"Borrower: {msg.content}")
        elif msg.role == "assistant":
            tag = " [deterministic]" if msg.source == TurnSource.DETERMINISTIC else ""
            lines.append(f"Agent{tag}: {msg.content}")
    return "\n".join(lines)


def _format_pipeline_transcript(records: list[ConversationRecord]) -> str:
    """Format a multi-stage transcript for system-level evaluation."""
    sections: list[str] = []
    for record in records:
        sections.append(f"=== Stage: {record.stage.value.upper()} ===")
        sections.append(_format_transcript(record.messages))
        if record.outcome:
            sections.append(f"[Outcome: {record.outcome.status.value}]")
        sections.append("")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Compliance pre-check (deterministic)
# ---------------------------------------------------------------------------

def _check_compliance_violations(
    messages: list[ConversationMessage],
    forbidden_phrases: list[str],
) -> list[str]:
    """Run deterministic compliance checks on all agent messages."""
    violations: list[str] = []
    for msg in messages:
        if msg.role != "assistant":
            continue
        # Forbidden phrase check
        lower = msg.content.lower()
        for phrase in forbidden_phrases:
            if phrase.lower() in lower:
                violations.append(f"Forbidden phrase: '{phrase}'")
        # Rule-based compliance check
        result = check_all(msg.content)
        if not result.passed:
            violations.append(result.violation or "Unknown violation")
    return list(set(violations))


# ---------------------------------------------------------------------------
# LLM evaluation prompts
# ---------------------------------------------------------------------------

_METRIC_EVAL_SYSTEM = """\
You are an expert evaluator for a debt collections AI agent. You will score \
a conversation transcript on a specific metric.

Return ONLY valid JSON with this exact format:
{
  "score": <float between SCALE_MIN and SCALE_MAX>,
  "reasoning": "<2-3 sentences explaining your score>"
}

Important:
- Turns marked [deterministic] are canned responses, not LLM-generated. \
For conversational_quality, weight LLM-generated turns more heavily.
- Be specific in your reasoning — cite parts of the conversation.
- A score of 5.0 means flawless, 3.0 is adequate, 1.0 is poor, 0.0 is failing.
"""

_SYSTEM_METRIC_EVAL_SYSTEM = """\
You are an expert evaluator for a multi-agent debt collections pipeline. \
You will score the full pipeline transcript on a system-level metric that \
measures cross-agent quality.

The pipeline has 3 stages:
1. Assessment (chat): verifies identity, gathers facts, determines resolution path
2. Resolution (voice): negotiates settlement
3. Final Notice (chat): last-chance offer with deadline

Return ONLY valid JSON with this exact format:
{
  "score": <float between SCALE_MIN and SCALE_MAX>,
  "reasoning": "<2-3 sentences explaining your score>"
}
"""


def _build_metric_prompt(
    metric: MetricConfig,
    transcript: str,
    stage: str,
    outcome_status: str,
) -> str:
    return (
        f"Metric: {metric.name}\n"
        f"Description: {metric.description}\n"
        f"Scale: {metric.scale_min} to {metric.scale_max}\n\n"
        f"Agent stage: {stage}\n"
        f"Outcome: {outcome_status}\n\n"
        f"Transcript:\n{transcript}"
    )


def _build_system_metric_prompt(
    metric: MetricConfig,
    pipeline_transcript: str,
    final_outcome: str,
) -> str:
    return (
        f"Metric: {metric.name}\n"
        f"Description: {metric.description}\n"
        f"Scale: {metric.scale_min} to {metric.scale_max}\n\n"
        f"Final pipeline outcome: {final_outcome}\n\n"
        f"Full pipeline transcript:\n{pipeline_transcript}"
    )


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def _parse_score(raw: str, metric: MetricConfig, fallback: float = 3.0) -> MetricScore:
    """Parse LLM JSON response into a MetricScore."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        data = json.loads(text)
        score = float(data.get("score", fallback))
        score = max(metric.scale_min, min(metric.scale_max, score))
        reasoning = str(data.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Failed to parse eval response for %s: %r", metric.name, raw[:200])
        score = fallback
        reasoning = f"Parse failure, defaulted to {fallback}"

    return MetricScore(
        name=metric.name,
        score=score,
        weight=metric.weight,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """Scores conversations using LLM-as-judge with rubric-driven metrics."""

    def __init__(
        self,
        rubric: EvaluationRubric | None = None,
        llm_client: LLMClient | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.rubric = rubric or load_evaluation_rubric()
        config = llm_config or EVAL_LLM_CONFIG
        self.client = llm_client or get_llm_client(config)
        self.config = config

    async def _score_metric(
        self,
        metric: MetricConfig,
        transcript: str,
        stage: str,
        outcome_status: str,
        cost_tag: str,
    ) -> MetricScore:
        """Score a single metric via one LLM call."""
        messages: list[Message] = [
            {"role": "system", "content": _METRIC_EVAL_SYSTEM},
            {"role": "user", "content": _build_metric_prompt(metric, transcript, stage, outcome_status)},
        ]
        tracker = get_cost_tracker()
        tracker.check_budget()

        llm_response = await self.client.complete(
            messages,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        tracker.record(llm_response, cost_tag)
        return _parse_score(llm_response.text, metric)

    async def _score_system_metric(
        self,
        metric: MetricConfig,
        pipeline_transcript: str,
        final_outcome: str,
    ) -> MetricScore:
        """Score a single system-level metric via one LLM call."""
        messages: list[Message] = [
            {"role": "system", "content": _SYSTEM_METRIC_EVAL_SYSTEM},
            {"role": "user", "content": _build_system_metric_prompt(metric, pipeline_transcript, final_outcome)},
        ]
        tracker = get_cost_tracker()
        tracker.check_budget()

        llm_response = await self.client.complete(
            messages,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        tracker.record(llm_response, "evaluation:tier2")
        return _parse_score(llm_response.text, metric)

    # ------------------------------------------------------------------
    # Tier 1: Per-agent evaluation
    # ------------------------------------------------------------------

    async def evaluate_conversation(
        self,
        record: ConversationRecord,
        cost_tag: str = "evaluation:tier1",
    ) -> ConversationEvaluation:
        """Evaluate a single conversation on all per-agent metrics.

        Runs compliance pre-check (deterministic), then scores all metrics
        concurrently via LLM.  Auto-caps compliance score on violation.
        """
        transcript = _format_transcript(record.messages)
        outcome_status = record.outcome.status.value if record.outcome else "unknown"
        stage = record.stage.value

        # Deterministic compliance check
        violations = _check_compliance_violations(
            record.messages, self.rubric.forbidden_phrases
        )

        # Score all metrics concurrently
        tasks = [
            self._score_metric(metric, transcript, stage, outcome_status, cost_tag)
            for metric in self.rubric.metrics
        ]
        metric_scores = await asyncio.gather(*tasks)

        # Auto-cap compliance on violation
        compliance_capped = False
        if violations:
            compliance_capped = True
            for ms in metric_scores:
                if ms.name == "compliance":
                    ms.score = min(ms.score, self.rubric.compliance_auto_cap)
                    ms.reasoning += f" [AUTO-CAPPED to {self.rubric.compliance_auto_cap} due to: {'; '.join(violations)}]"

        # Compute weighted fitness
        fitness = sum(ms.score * ms.weight for ms in metric_scores)

        # Find weakest metric
        weakest = min(metric_scores, key=lambda ms: ms.score)

        evaluation = ConversationEvaluation(
            session_id=record.session_id,
            stage=stage,
            persona_id=record.persona_id or "",
            metric_scores=list(metric_scores),
            fitness=round(fitness, 4),
            compliance_violations=violations,
            compliance_capped=compliance_capped,
            weakest_metric=weakest.name,
            overall_reasoning=weakest.reasoning,
        )

        logger.info(
            "Evaluated %s | persona=%s | fitness=%.2f | weakest=%s (%.1f) | violations=%d",
            stage, record.persona_id, fitness, weakest.name, weakest.score, len(violations),
        )
        return evaluation

    async def evaluate_batch(
        self,
        records: list[ConversationRecord],
        cost_tag: str = "evaluation:tier1",
    ) -> list[ConversationEvaluation]:
        """Evaluate a batch of conversations sequentially.

        Sequential to avoid overwhelming the LLM API with concurrent Pro calls.
        """
        evaluations: list[ConversationEvaluation] = []
        for i, record in enumerate(records):
            logger.info("Evaluating %d/%d: %s", i + 1, len(records), record.session_id)
            try:
                evaluation = await self.evaluate_conversation(record, cost_tag)
                evaluations.append(evaluation)
            except Exception:
                logger.exception("Evaluation failed for %s", record.session_id)
        return evaluations

    # ------------------------------------------------------------------
    # Tier 2: System-level evaluation (full pipeline)
    # ------------------------------------------------------------------

    async def evaluate_pipeline(
        self,
        records: list[ConversationRecord],
    ) -> PipelineEvaluation:
        """Evaluate a full pipeline run: per-agent scores + system-level metrics.

        Args:
            records: List of 1-3 ConversationRecords from a pipeline run.
        """
        if not records:
            raise ValueError("No records to evaluate")

        case_id = records[0].case_id
        persona_id = records[0].persona_id or ""

        # Tier 1: Per-agent evaluation for each stage
        per_stage: list[ConversationEvaluation] = []
        for record in records:
            evaluation = await self.evaluate_conversation(record, cost_tag="evaluation:tier2")
            per_stage.append(evaluation)

        # Tier 2: System-level metrics (only if we have 2+ stages)
        system_scores: list[MetricScore] = []
        if len(records) >= 2 and self.rubric.system_metrics:
            pipeline_transcript = _format_pipeline_transcript(records)
            final_outcome = records[-1].outcome.status.value if records[-1].outcome else "unknown"

            system_tasks = [
                self._score_system_metric(metric, pipeline_transcript, final_outcome)
                for metric in self.rubric.system_metrics
            ]
            system_scores = await asyncio.gather(*system_tasks)

        # Compute fitnesses
        avg_agent_fitness = sum(e.fitness for e in per_stage) / len(per_stage)
        system_fitness = sum(ms.score * ms.weight for ms in system_scores) if system_scores else 0.0

        # Combined: 70% agent fitness + 30% system fitness (if available)
        if system_scores:
            combined = 0.7 * avg_agent_fitness + 0.3 * system_fitness
        else:
            combined = avg_agent_fitness

        logger.info(
            "Pipeline evaluated | persona=%s | agent_fitness=%.2f | system_fitness=%.2f | combined=%.2f",
            persona_id, avg_agent_fitness, system_fitness, combined,
        )

        return PipelineEvaluation(
            case_id=case_id,
            persona_id=persona_id,
            per_stage=per_stage,
            system_scores=list(system_scores),
            system_fitness=round(system_fitness, 4),
            combined_fitness=round(combined, 4),
        )
