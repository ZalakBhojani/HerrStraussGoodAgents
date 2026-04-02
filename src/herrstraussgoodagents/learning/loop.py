"""Self-learning loop orchestrator.

Ties together the evaluator, failure analyzer, mutator, bootstrap CI,
test harness runner, and version archive into an iterative prompt
improvement loop.

Loop body (per iteration):
  1. Simulate baseline with current prompt
  2. Evaluate baseline conversations
  3. Analyze failures (bottom 25%)
  4. Propose prompt mutation
  5. Simulate candidate with mutated prompt
  6. Evaluate candidate conversations
  7. Bootstrap CI comparison
  8. Promote or reject
  9. Every Nth iteration: Tier 2 full-pipeline gate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from herrstraussgoodagents.config import AgentConfig, load_agent_config
from herrstraussgoodagents.harness.runner import (
    run_full_pipeline_batch,
    run_single_agent_batch,
)
from herrstraussgoodagents.learning.archive import Archive, VersionRecord
from herrstraussgoodagents.learning.evaluator import (
    ConversationEvaluation,
    Evaluator,
)
from herrstraussgoodagents.learning.failure_analyzer import analyze_failures
from herrstraussgoodagents.learning.mutator import Mutator, apply_mutation
from herrstraussgoodagents.compliance.auditor import ComplianceAuditor
from herrstraussgoodagents.learning.statistics import BootstrapResult, bootstrap_ci
from herrstraussgoodagents.llm import BudgetExhausted, get_cost_tracker
from herrstraussgoodagents.models import AgentStage

logger = logging.getLogger(__name__)

# Map AgentStage to config name used by load_agent_config
_STAGE_CONFIG_NAME = {
    AgentStage.ASSESSMENT: "assessment",
    AgentStage.RESOLUTION: "resolution",
    AgentStage.FINAL_NOTICE: "final_notice",
}


@dataclass
class IterationResult:
    """Result of a single learning iteration."""

    iteration: int
    baseline_fitness: float
    candidate_fitness: float
    bootstrap: BootstrapResult
    adopted: bool
    version_id: str
    mutation_rationale: str
    audit_hard_violations: int = 0
    audit_blocked: bool = False


@dataclass
class LoopResult:
    """Final result of the full learning loop."""

    stage: str
    iterations_completed: int
    iterations: list[IterationResult] = field(default_factory=list)
    versions: list[VersionRecord] = field(default_factory=list)
    final_version: str = ""
    cost_report: str = ""
    budget_exhausted: bool = False


def _next_version_id(parent: str, iteration: int) -> str:
    """Generate a version id like v1.1, v1.2, etc."""
    base = parent.split(".")[0]  # strip any existing sub-version
    return f"{base}.{iteration}"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _fitness_scores(evals: list[ConversationEvaluation]) -> list[float]:
    return [e.fitness for e in evals]


def _per_metric_avg(evals: list[ConversationEvaluation]) -> dict[str, float]:
    """Compute average score for each metric across evaluations."""
    totals: dict[str, list[float]] = {}
    for ev in evals:
        for ms in ev.metric_scores:
            totals.setdefault(ms.name, []).append(ms.score)
    return {name: round(_mean(scores), 4) for name, scores in totals.items()}


def _per_persona_avg(evals: list[ConversationEvaluation]) -> dict[str, float]:
    """Compute average fitness per persona."""
    groups: dict[str, list[float]] = {}
    for ev in evals:
        groups.setdefault(ev.persona_id, []).append(ev.fitness)
    return {pid: round(_mean(scores), 4) for pid, scores in groups.items()}


class LearningLoop:
    """Orchestrates the iterative self-learning loop for one agent stage."""

    def __init__(
        self,
        stage: AgentStage,
        agent_version: str = "v1",
        max_iterations: int = 4,
        conversations_per_persona: int = 6,
        full_pipeline_every: int = 3,
        seed: int = 42,
        archive: Archive | None = None,
        evaluator: Evaluator | None = None,
        mutator: Mutator | None = None,
        auditor: ComplianceAuditor | None = None,
    ) -> None:
        self.stage = stage
        self.agent_version = agent_version
        self.max_iterations = max_iterations
        self.conversations_per_persona = conversations_per_persona
        self.full_pipeline_every = full_pipeline_every
        self.seed = seed

        self.archive = archive or Archive()
        self.evaluator = evaluator or Evaluator()
        self.mutator = mutator or Mutator()
        self.auditor = auditor or ComplianceAuditor()

        self._config_name = _STAGE_CONFIG_NAME[stage]

    async def run(self) -> LoopResult:
        """Run the full learning loop. Returns results even if budget runs out."""
        result = LoopResult(stage=self.stage.value)
        tracker = get_cost_tracker()

        current_version = self.agent_version
        current_config = load_agent_config(self._config_name, current_version)

        # Archive baseline
        self._archive_baseline(current_version, current_config)

        try:
            for iteration in range(1, self.max_iterations + 1):
                logger.info(
                    "=== Iteration %d/%d | stage=%s | version=%s ===",
                    iteration, self.max_iterations, self.stage.value, current_version,
                )

                iter_result = await self._run_iteration(
                    iteration=iteration,
                    current_version=current_version,
                    current_config=current_config,
                )

                if iter_result is None:
                    logger.info("Iteration %d produced no result (mutation failed), stopping.", iteration)
                    break

                result.iterations.append(iter_result)
                result.iterations_completed = iteration

                if iter_result.adopted:
                    current_version = iter_result.version_id
                    current_config = load_agent_config(self._config_name, current_version)
                    logger.info("Adopted version %s", current_version)

                    # Tier 2 gate check
                    if iteration % self.full_pipeline_every == 0:
                        passed = await self._tier2_gate(current_version)
                        if not passed:
                            logger.warning(
                                "Tier 2 gate FAILED for %s — rolling back", current_version
                            )
                            prev = self.archive.rollback(self.stage.value)
                            current_version = prev.version_id
                            current_config = load_agent_config(self._config_name, current_version)
                else:
                    logger.info("Rejected candidate — keeping %s", current_version)

        except BudgetExhausted:
            logger.warning("Budget exhausted during learning loop")
            result.budget_exhausted = True

        result.final_version = current_version
        result.cost_report = tracker.report()
        result.versions = self.archive.list_versions(self.stage.value)

        logger.info(
            "Learning loop complete: %d iterations, final=%s, budget_exhausted=%s",
            result.iterations_completed, result.final_version, result.budget_exhausted,
        )
        return result

    async def _run_iteration(
        self,
        iteration: int,
        current_version: str,
        current_config: AgentConfig,
    ) -> IterationResult | None:
        """Run one iteration of the learning loop."""

        # 1. Simulate baseline
        logger.info("Simulating baseline (version=%s)...", current_version)
        baseline_records = await run_single_agent_batch(
            stage=self.stage,
            agent_version=current_version,
            conversations_per_persona=self.conversations_per_persona,
            seed=self.seed + iteration * 10_000,
            agent_config=current_config,
        )
        if not baseline_records:
            logger.warning("No baseline conversations produced")
            return None

        # 2. Evaluate baseline
        logger.info("Evaluating baseline (%d conversations)...", len(baseline_records))
        baseline_evals = await self.evaluator.evaluate_batch(
            baseline_records, cost_tag="evaluation:tier1:baseline"
        )

        # 3. Analyze failures
        failure_report = analyze_failures(
            baseline_evals,
            baseline_records,
            stage=self.stage.value,
            agent_version=current_version,
        )

        # 4. Propose mutation
        logger.info("Proposing mutation based on %d weak sessions...", failure_report.weak_count)
        mutation = await self.mutator.propose_mutation(
            current_config, failure_report, parent_version=current_version
        )
        if mutation is None:
            logger.warning("Mutator failed to produce a valid mutation")
            return None

        # 5. Apply mutation
        candidate_config = apply_mutation(current_config, mutation)
        candidate_version = _next_version_id(current_version, iteration)

        # 6. Simulate candidate
        logger.info("Simulating candidate (version=%s)...", candidate_version)
        candidate_records = await run_single_agent_batch(
            stage=self.stage,
            agent_version=candidate_version,
            conversations_per_persona=self.conversations_per_persona,
            seed=self.seed + iteration * 10_000,  # same seed for fair comparison
            agent_config=candidate_config,
        )
        if not candidate_records:
            logger.warning("No candidate conversations produced")
            return None

        # 7. Evaluate candidate
        logger.info("Evaluating candidate (%d conversations)...", len(candidate_records))
        candidate_evals = await self.evaluator.evaluate_batch(
            candidate_records, cost_tag="evaluation:tier1:candidate"
        )

        # 8. Bootstrap CI comparison
        baseline_fitness = _fitness_scores(baseline_evals)
        candidate_fitness = _fitness_scores(candidate_evals)
        bs_result = bootstrap_ci(baseline_fitness, candidate_fitness, seed=self.seed + iteration)

        # 9. Decision — bootstrap CI
        adopted = bs_result.recommend_adoption

        # 9b. Compliance audit gate — runs only if bootstrap passes
        hard_violations = 0
        audit_blocked = False
        if adopted:
            logger.info("Running compliance audit on %d candidate conversations...", len(candidate_records))
            audit_results = await self.auditor.audit_batch(candidate_records)
            hard_violations = sum(r.hard_violations for r in audit_results)
            if hard_violations > 0:
                adopted = False
                audit_blocked = True
                logger.warning(
                    "Compliance audit blocked promotion of %s: %d hard violations across %d conversations",
                    candidate_version, hard_violations, len(candidate_records),
                )

        # Archive the candidate
        version_record = VersionRecord(
            version_id=candidate_version,
            parent_version=current_version,
            stage=self.stage.value,
            status="promoted" if adopted else "rejected",
            created_at=datetime.now(timezone.utc).isoformat(),
            mutation_rationale=mutation.rationale,
            fitness_mean=round(_mean(candidate_fitness), 4),
            fitness_ci=(bs_result.ci_lower, bs_result.ci_upper),
            per_metric=_per_metric_avg(candidate_evals),
            per_persona=_per_persona_avg(candidate_evals),
            config_snapshot=candidate_config.model_dump(),
        )
        self.archive.save_version(version_record)

        if adopted:
            self.archive.promote(self.stage.value, candidate_version)

        return IterationResult(
            iteration=iteration,
            baseline_fitness=round(_mean(baseline_fitness), 4),
            candidate_fitness=round(_mean(candidate_fitness), 4),
            bootstrap=bs_result,
            adopted=adopted,
            version_id=candidate_version,
            mutation_rationale=mutation.rationale,
            audit_hard_violations=hard_violations,
            audit_blocked=audit_blocked,
        )

    async def _tier2_gate(self, version: str) -> bool:
        """Run Tier 2 full-pipeline evaluation as a promotion gate.

        Returns True if the version passes, False if it should be rolled back.
        """
        logger.info("Running Tier 2 full-pipeline gate for version %s...", version)

        pipeline_runs = await run_full_pipeline_batch(
            conversations_per_persona=2,  # fewer convos for gate check
            seed=self.seed,
            **{f"{self._config_name}_version": version},
        )

        if not pipeline_runs:
            logger.warning("Tier 2 gate: no pipeline runs produced")
            return False

        # Evaluate each pipeline run
        total_combined = 0.0
        for records in pipeline_runs:
            pipeline_eval = await self.evaluator.evaluate_pipeline(records)
            total_combined += pipeline_eval.combined_fitness

        avg_combined = total_combined / len(pipeline_runs)
        fitness_passed = avg_combined >= 3.0  # minimum acceptable system-level fitness

        # Compliance audit on all pipeline stage records
        all_records = [record for run in pipeline_runs for record in run]
        audit_results = await self.auditor.audit_batch(all_records)
        hard_violations = sum(r.hard_violations for r in audit_results)
        audit_passed = hard_violations == 0

        passed = fitness_passed and audit_passed
        logger.info(
            "Tier 2 gate: avg_combined_fitness=%.2f (threshold=3.0, pass=%s), "
            "compliance hard_violations=%d (pass=%s), overall=%s",
            avg_combined, fitness_passed, hard_violations, audit_passed, passed,
        )
        return passed

    def _archive_baseline(self, version: str, config: AgentConfig) -> None:
        """Archive the initial baseline version if not already archived."""
        try:
            self.archive.load_version(self.stage.value, version)
        except FileNotFoundError:
            record = VersionRecord(
                version_id=version,
                parent_version=None,
                stage=self.stage.value,
                status="baseline",
                created_at=datetime.now(timezone.utc).isoformat(),
                mutation_rationale="Initial baseline",
                fitness_mean=0.0,
                fitness_ci=(0.0, 0.0),
                config_snapshot=config.model_dump(),
            )
            self.archive.save_version(record)
            self.archive.promote(self.stage.value, version)
