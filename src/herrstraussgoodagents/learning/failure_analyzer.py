"""Failure analyzer — extracts weak sessions and groups failure patterns.

Takes evaluation results from the Evaluator, extracts the bottom 25%
by fitness, groups by persona and weak metric, and produces a structured
FailureReport that feeds the prompt mutator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from herrstraussgoodagents.learning.evaluator import ConversationEvaluation
from herrstraussgoodagents.models import ConversationRecord

logger = logging.getLogger(__name__)

# Minimum number of weak sessions to extract (even if <25%)
_MIN_WEAK_SESSIONS = 2

# Metric threshold below which a metric is flagged
_WEAK_METRIC_THRESHOLD = 4.0


@dataclass
class WeakSession:
    """A single weak conversation with its evaluation."""
    session_id: str
    persona_id: str
    fitness: float
    weakest_metric: str
    weakest_score: float
    evaluation: ConversationEvaluation
    transcript_summary: str  # condensed transcript for mutator context


@dataclass
class PersonaPattern:
    """Aggregated failure pattern for one persona."""
    persona_id: str
    session_count: int
    avg_fitness: float
    common_weak_metrics: dict[str, int]  # metric_name -> count
    sample_reasonings: list[str]  # evaluator reasoning from worst sessions


@dataclass
class FailureReport:
    """Output of the failure analyzer — input to the prompt mutator."""
    stage: str
    agent_version: str
    total_evaluated: int
    weak_count: int
    weak_sessions: list[WeakSession]
    persona_patterns: list[PersonaPattern]
    overall_weak_metrics: dict[str, float]  # metric_name -> avg score across weak sessions
    recommendation: str  # short text summary for the mutator


def _summarize_transcript(record: ConversationRecord, max_turns: int = 6) -> str:
    """Condense a transcript to the most relevant turns for the mutator."""
    lines: list[str] = []
    turn_count = 0
    for msg in record.messages:
        if msg.role == "user":
            lines.append(f"Borrower: {msg.content[:150]}")
            turn_count += 1
        elif msg.role == "assistant":
            tag = " [det]" if msg.source.value == "deterministic" else ""
            lines.append(f"Agent{tag}: {msg.content[:150]}")
        if turn_count >= max_turns:
            lines.append(f"... ({len(record.messages)} total messages)")
            break
    return "\n".join(lines)


def analyze_failures(
    evaluations: list[ConversationEvaluation],
    records: list[ConversationRecord],
    stage: str = "",
    agent_version: str = "v1",
) -> FailureReport:
    """Extract bottom 25% sessions and group failure patterns.

    Args:
        evaluations: Scored evaluations from the Evaluator.
        records: Corresponding ConversationRecords (same order / matched by session_id).
        stage: Agent stage name (for the report).
        agent_version: Current prompt version.
    """
    if not evaluations:
        return FailureReport(
            stage=stage,
            agent_version=agent_version,
            total_evaluated=0,
            weak_count=0,
            weak_sessions=[],
            persona_patterns=[],
            overall_weak_metrics={},
            recommendation="No evaluations to analyze.",
        )

    # Build session_id -> record lookup
    record_map = {r.session_id: r for r in records}

    # Sort by fitness ascending (worst first)
    sorted_evals = sorted(evaluations, key=lambda e: e.fitness)

    # Extract bottom 25% (minimum _MIN_WEAK_SESSIONS)
    cutoff = max(_MIN_WEAK_SESSIONS, len(sorted_evals) // 4)
    weak_evals = sorted_evals[:cutoff]

    # Build WeakSession objects
    weak_sessions: list[WeakSession] = []
    for ev in weak_evals:
        record = record_map.get(ev.session_id)
        transcript_summary = _summarize_transcript(record) if record else "(transcript unavailable)"

        # Find the lowest-scoring metric
        worst_metric = min(ev.metric_scores, key=lambda ms: ms.score)

        weak_sessions.append(WeakSession(
            session_id=ev.session_id,
            persona_id=ev.persona_id,
            fitness=ev.fitness,
            weakest_metric=worst_metric.name,
            weakest_score=worst_metric.score,
            evaluation=ev,
            transcript_summary=transcript_summary,
        ))

    # Group by persona
    persona_groups: dict[str, list[WeakSession]] = {}
    for ws in weak_sessions:
        persona_groups.setdefault(ws.persona_id, []).append(ws)

    persona_patterns: list[PersonaPattern] = []
    for persona_id, sessions in persona_groups.items():
        metric_counts: dict[str, int] = {}
        for s in sessions:
            metric_counts[s.weakest_metric] = metric_counts.get(s.weakest_metric, 0) + 1

        avg_fitness = sum(s.fitness for s in sessions) / len(sessions)

        # Collect worst reasoning samples (up to 3)
        sorted_sessions = sorted(sessions, key=lambda s: s.fitness)
        sample_reasonings = [s.evaluation.overall_reasoning for s in sorted_sessions[:3]]

        persona_patterns.append(PersonaPattern(
            persona_id=persona_id,
            session_count=len(sessions),
            avg_fitness=round(avg_fitness, 4),
            common_weak_metrics=metric_counts,
            sample_reasonings=sample_reasonings,
        ))

    # Overall weak metric averages
    overall_metrics: dict[str, list[float]] = {}
    for ws in weak_sessions:
        for ms in ws.evaluation.metric_scores:
            overall_metrics.setdefault(ms.name, []).append(ms.score)
    overall_weak_metrics = {
        name: round(sum(scores) / len(scores), 4)
        for name, scores in overall_metrics.items()
    }

    # Build recommendation
    flagged = [
        f"{name} (avg {avg:.1f})"
        for name, avg in overall_weak_metrics.items()
        if avg < _WEAK_METRIC_THRESHOLD
    ]
    if flagged:
        recommendation = f"Weak metrics across bottom {cutoff} sessions: {', '.join(flagged)}. "
    else:
        recommendation = f"No metric consistently below {_WEAK_METRIC_THRESHOLD} threshold. "

    # Most common persona in failures
    worst_persona = max(persona_patterns, key=lambda p: p.session_count) if persona_patterns else None
    if worst_persona:
        recommendation += (
            f"Persona '{worst_persona.persona_id}' appears most often in failures "
            f"({worst_persona.session_count}/{cutoff} weak sessions). "
            f"Most common weak metric: {max(worst_persona.common_weak_metrics, key=worst_persona.common_weak_metrics.get)}."
        )

    report = FailureReport(
        stage=stage or (evaluations[0].stage if evaluations else ""),
        agent_version=agent_version,
        total_evaluated=len(evaluations),
        weak_count=len(weak_sessions),
        weak_sessions=weak_sessions,
        persona_patterns=persona_patterns,
        overall_weak_metrics=overall_weak_metrics,
        recommendation=recommendation,
    )

    logger.info(
        "Failure analysis: %d/%d weak sessions, flagged metrics: %s",
        report.weak_count, report.total_evaluated,
        ", ".join(flagged) or "none",
    )
    return report
