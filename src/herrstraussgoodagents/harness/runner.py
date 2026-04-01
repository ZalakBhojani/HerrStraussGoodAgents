"""Batch runner for the test harness.

Runs N conversations per persona per agent (or full-pipeline), with
seed-based reproducibility, cost tracking integration, and JSON output.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path

from herrstraussgoodagents.config import AgentConfig, PersonaConfig, load_agent_config
from herrstraussgoodagents.harness.persona_loader import (
    BorrowerSimulator,
    PERSONA_LLM_CONFIG,
    load_all_personas,
)
from herrstraussgoodagents.harness.simulator import (
    simulate_single_agent,
    simulate_full_pipeline,
)
from herrstraussgoodagents.llm import BudgetExhausted, get_cost_tracker
from herrstraussgoodagents.models import (
    AgentStage,
    BorrowerCase,
    ConversationRecord,
    HandoffContext,
    ResolutionPath,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "results"


def _make_case(persona: PersonaConfig, seed: int) -> BorrowerCase:
    """Build a BorrowerCase from a persona config with a deterministic ID."""
    rng = random.Random(seed)
    last_four = f"{rng.randint(1000, 9999)}"
    return BorrowerCase(
        case_id=f"sim-{persona.id}-{seed}",
        borrower_id=f"borrower-{persona.id}-{seed}",
        borrower_name=persona.context.name,
        account_last_four=last_four,
        debt_amount=persona.context.loan_amount,
        months_overdue=persona.context.months_overdue,
        original_creditor="Original Lender Corp",
    )


def _save_record(record: ConversationRecord, output_dir: Path) -> Path:
    """Save a ConversationRecord as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{record.session_id}.json"
    path.write_text(record.model_dump_json(indent=2))
    return path


def _default_handoff(case: BorrowerCase) -> HandoffContext:
    """Build a minimal handoff context for standalone Resolution/FinalNotice sims."""
    return HandoffContext(
        identity_verified=True,
        debt_amount=case.debt_amount,
        months_overdue=case.months_overdue,
        resolution_path=ResolutionPath.PAYMENT_PLAN,
        tone_summary="Borrower verified identity. Standard assessment completed.",
        source_stage=AgentStage.ASSESSMENT,
    )


# ---------------------------------------------------------------------------
# Per-agent batch runner
# ---------------------------------------------------------------------------

async def run_single_agent_batch(
    stage: AgentStage,
    agent_version: str = "v1",
    conversations_per_persona: int = 6,
    seed: int = 42,
    personas: list[PersonaConfig] | None = None,
    agent_config: AgentConfig | None = None,
    handoff: HandoffContext | None = None,
) -> list[ConversationRecord]:
    """Run N simulated conversations per persona for a single agent stage.

    Returns all ConversationRecords. Stops early if budget is exhausted.
    """
    personas = personas or load_all_personas()
    tracker = get_cost_tracker()
    records: list[ConversationRecord] = []
    output_dir = DATA_DIR / f"{stage.value}_{agent_version}"

    total = len(personas) * conversations_per_persona
    logger.info(
        "Starting %s batch: %d personas x %d convos = %d total",
        stage.value, len(personas), conversations_per_persona, total,
    )

    for p_idx, persona in enumerate(personas):
        for c_idx in range(conversations_per_persona):
            conversation_seed = seed + p_idx * 1000 + c_idx
            case = _make_case(persona, conversation_seed)

            try:
                tracker.check_budget()
            except BudgetExhausted:
                logger.warning("Budget exhausted after %d conversations", len(records))
                return records

            try:
                # For Resolution/FinalNotice, use provided handoff or generate default
                sim_handoff = handoff
                if stage != AgentStage.ASSESSMENT and sim_handoff is None:
                    sim_handoff = _default_handoff(case)

                borrower = BorrowerSimulator(
                    persona=persona,
                    account_last_four=case.account_last_four,
                )
                record = await simulate_single_agent(
                    stage=stage,
                    case=case,
                    borrower=borrower,
                    agent_config=agent_config,
                    handoff=sim_handoff,
                    agent_version=agent_version,
                )
                records.append(record)
                _save_record(record, output_dir)

                logger.info(
                    "[%d/%d] %s | persona=%s | status=%s | turns=%d",
                    len(records), total,
                    stage.value, persona.id,
                    record.outcome.status if record.outcome else "?",
                    record.outcome.turns_taken if record.outcome else 0,
                )
            except BudgetExhausted:
                logger.warning("Budget exhausted during conversation %d", len(records) + 1)
                return records
            except Exception:
                logger.exception(
                    "Conversation failed: stage=%s persona=%s seed=%d",
                    stage.value, persona.id, conversation_seed,
                )

    logger.info(
        "Batch complete: %d conversations, $%.4f spent",
        len(records), tracker.total_usd,
    )
    return records


# ---------------------------------------------------------------------------
# Full-pipeline batch runner
# ---------------------------------------------------------------------------

async def run_full_pipeline_batch(
    conversations_per_persona: int = 3,
    seed: int = 42,
    personas: list[PersonaConfig] | None = None,
    assessment_version: str = "v1",
    resolution_version: str = "v1",
    final_notice_version: str = "v1",
) -> list[list[ConversationRecord]]:
    """Run full 3-agent pipeline simulations.

    Returns a list of pipeline runs, each being a list of 1-3 ConversationRecords.
    """
    personas = personas or load_all_personas()
    tracker = get_cost_tracker()
    all_runs: list[list[ConversationRecord]] = []
    output_dir = DATA_DIR / f"pipeline_{assessment_version}_{resolution_version}_{final_notice_version}"

    total = len(personas) * conversations_per_persona
    logger.info(
        "Starting pipeline batch: %d personas x %d convos = %d total",
        len(personas), conversations_per_persona, total,
    )

    for p_idx, persona in enumerate(personas):
        for c_idx in range(conversations_per_persona):
            conversation_seed = seed + p_idx * 1000 + c_idx
            case = _make_case(persona, conversation_seed)

            try:
                tracker.check_budget()
            except BudgetExhausted:
                logger.warning("Budget exhausted after %d pipeline runs", len(all_runs))
                return all_runs

            try:
                borrower = BorrowerSimulator(
                    persona=persona,
                    account_last_four=case.account_last_four,
                )
                records = await simulate_full_pipeline(
                    case=case,
                    borrower=borrower,
                    assessment_version=assessment_version,
                    resolution_version=resolution_version,
                    final_notice_version=final_notice_version,
                )
                all_runs.append(records)

                # Save each stage's record
                for record in records:
                    _save_record(record, output_dir)

                final = records[-1]
                logger.info(
                    "[%d/%d] pipeline | persona=%s | stages=%d | final_status=%s",
                    len(all_runs), total,
                    persona.id, len(records),
                    final.outcome.status if final.outcome else "?",
                )
            except BudgetExhausted:
                logger.warning("Budget exhausted during pipeline run %d", len(all_runs) + 1)
                return all_runs
            except Exception:
                logger.exception(
                    "Pipeline failed: persona=%s seed=%d",
                    persona.id, conversation_seed,
                )

    logger.info(
        "Pipeline batch complete: %d runs, $%.4f spent",
        len(all_runs), tracker.total_usd,
    )
    return all_runs
