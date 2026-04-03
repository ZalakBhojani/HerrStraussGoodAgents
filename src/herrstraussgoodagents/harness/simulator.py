"""Conversation simulator for the test harness.

Supports two modes:
  - Per-agent:     Run a single agent against a persona LLM.
  - Full-pipeline: Run all 3 agents sequentially with real handoff summarization.

Both modes return a ConversationRecord with tagged transcripts.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime

from herrstraussgoodagents.agents.assessment import AssessmentAgent
from herrstraussgoodagents.agents.resolution import ResolutionAgent, SessionSignal
from herrstraussgoodagents.agents.final_notice import FinalNoticeAgent
from herrstraussgoodagents.config import AgentConfig, load_agent_config
from herrstraussgoodagents.harness.persona_loader import BorrowerSimulator
from herrstraussgoodagents.models import (
    AgentOutcome,
    AgentStage,
    BorrowerCase,
    ConversationRecord,
    HandoffContext,
    OutcomeStatus,
)

logger = logging.getLogger(__name__)

MAX_TURNS = 20


# ---------------------------------------------------------------------------
# Per-agent simulators
# ---------------------------------------------------------------------------

async def simulate_assessment(
    agent: AssessmentAgent,
    borrower: BorrowerSimulator,
) -> AgentOutcome:
    """Drive the assessment agent with a persona LLM via asyncio queues."""
    inbound: asyncio.Queue[str] = asyncio.Queue()
    outbound: asyncio.Queue[str] = asyncio.Queue()

    async def _feed_borrower():
        """Read agent messages from outbound, generate persona replies, push to inbound."""
        while True:
            msg = await outbound.get()
            if msg in ("__DONE__", "__AGENT_DONE__"):
                break
            response = await borrower.respond(msg)
            await inbound.put(response)

    feeder = asyncio.create_task(_feed_borrower())
    outcome = await agent.run(inbound, outbound)

    # Signal feeder to stop if still waiting
    if not feeder.done():
        await outbound.put("__DONE__")
    await feeder
    return outcome


async def simulate_resolution(
    agent: ResolutionAgent,
    borrower: BorrowerSimulator,
) -> AgentOutcome:
    """Drive the resolution agent with a persona LLM via process_turn()."""
    opening = agent.opening_line()

    for turn in range(MAX_TURNS):
        borrower_text = await borrower.respond(opening if turn == 0 else result.response)
        result = await agent.process_turn(borrower_text)

        if result.signal != SessionSignal.CONTINUE:
            return agent.build_outcome(result.signal)

    return agent.build_outcome(SessionSignal.END_UNRESOLVED)


async def simulate_final_notice(
    agent: FinalNoticeAgent,
    borrower: BorrowerSimulator,
) -> AgentOutcome:
    """Drive the final notice agent with a persona LLM via asyncio queues."""
    inbound: asyncio.Queue[str] = asyncio.Queue()
    outbound: asyncio.Queue[str] = asyncio.Queue()

    async def _feed_borrower():
        while True:
            msg = await outbound.get()
            if msg in ("__DONE__", "__AGENT_DONE__"):
                break
            response = await borrower.respond(msg)
            await inbound.put(response)

    feeder = asyncio.create_task(_feed_borrower())
    outcome = await agent.run(inbound, outbound)

    if not feeder.done():
        await outbound.put("__DONE__")
    await feeder
    return outcome


# ---------------------------------------------------------------------------
# Single-agent simulation
# ---------------------------------------------------------------------------

async def simulate_single_agent(
    stage: AgentStage,
    case: BorrowerCase,
    borrower: BorrowerSimulator,
    agent_config: AgentConfig | None = None,
    handoff: HandoffContext | None = None,
    agent_version: str = "v1",
) -> ConversationRecord:
    """Run one agent against a persona LLM and return a ConversationRecord.

    For Resolution and FinalNotice, a handoff context must be provided.
    """
    started_at = datetime.utcnow()

    if stage == AgentStage.ASSESSMENT:
        config = agent_config or load_agent_config("assessment", agent_version)
        agent = AssessmentAgent(config, case)
        outcome = await simulate_assessment(agent, borrower)

    elif stage == AgentStage.RESOLUTION:
        if handoff is None:
            raise ValueError("Resolution simulation requires a handoff context")
        config = agent_config or load_agent_config("resolution", agent_version)
        agent = ResolutionAgent(config, case, handoff)
        outcome = await simulate_resolution(agent, borrower)

    elif stage == AgentStage.FINAL_NOTICE:
        if handoff is None:
            raise ValueError("FinalNotice simulation requires a handoff context")
        config = agent_config or load_agent_config("final_notice", agent_version)
        agent = FinalNoticeAgent(config, case, handoff)
        outcome = await simulate_final_notice(agent, borrower)

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return ConversationRecord(
        session_id=str(uuid.uuid4()),
        case_id=case.case_id,
        stage=stage,
        agent_version=agent_version,
        persona_id=borrower.persona.id,
        # messages=outcome.transcript, # (todo): transcript does not contain the conversation
        outcome=outcome,
        started_at=started_at,
        ended_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Full-pipeline simulation (all 3 agents)
# ---------------------------------------------------------------------------

async def simulate_full_pipeline(
    case: BorrowerCase,
    borrower: BorrowerSimulator,
    assessment_version: str = "v1",
    resolution_version: str = "v1",
    final_notice_version: str = "v1",
) -> list[ConversationRecord]:
    """Run the full 3-agent pipeline: Assessment -> Resolution -> FinalNotice.

    Uses real handoff summarization between stages. Returns a list of 1-3
    ConversationRecords (pipeline may exit early on cease/identity failure).
    """
    records: list[ConversationRecord] = []

    # --- Stage 1: Assessment ---
    assessment_config = load_agent_config("assessment", assessment_version)
    assessment_agent = AssessmentAgent(assessment_config, case)
    started = datetime.utcnow()
    assessment_outcome = await simulate_assessment(assessment_agent, borrower)

    records.append(ConversationRecord(
        case_id=case.case_id,
        stage=AgentStage.ASSESSMENT,
        agent_version=assessment_version,
        persona_id=borrower.persona.id,
        messages=assessment_outcome.transcript,
        outcome=assessment_outcome,
        started_at=started,
        ended_at=datetime.utcnow(),
    ))

    if assessment_outcome.status in (
        OutcomeStatus.CEASE_REQUESTED,
        OutcomeStatus.IDENTITY_FAILED,
    ):
        logger.info("Pipeline ended at assessment: %s", assessment_outcome.status)
        return records

    handoff = assessment_outcome.handoff_context
    if handoff is None:
        logger.warning("Assessment produced no handoff context, stopping pipeline.")
        return records

    # --- Stage 2: Resolution ---
    resolution_config = load_agent_config("resolution", resolution_version)
    resolution_agent = ResolutionAgent(resolution_config, case, handoff)
    started = datetime.utcnow()
    resolution_outcome = await simulate_resolution(resolution_agent, borrower)

    records.append(ConversationRecord(
        case_id=case.case_id,
        stage=AgentStage.RESOLUTION,
        agent_version=resolution_version,
        persona_id=borrower.persona.id,
        messages=resolution_outcome.transcript,
        outcome=resolution_outcome,
        started_at=started,
        ended_at=datetime.utcnow(),
    ))

    if resolution_outcome.status in (
        OutcomeStatus.RESOLVED,
        OutcomeStatus.CEASE_REQUESTED,
    ):
        logger.info("Pipeline ended at resolution: %s", resolution_outcome.status)
        return records

    resolution_handoff = resolution_outcome.handoff_context
    if resolution_handoff is None:
        logger.warning("Resolution produced no handoff context, stopping pipeline.")
        return records

    # --- Stage 3: Final Notice ---
    final_config = load_agent_config("final_notice", final_notice_version)
    final_agent = FinalNoticeAgent(final_config, case, resolution_handoff)
    started = datetime.utcnow()
    final_outcome = await simulate_final_notice(final_agent, borrower)

    records.append(ConversationRecord(
        case_id=case.case_id,
        stage=AgentStage.FINAL_NOTICE,
        agent_version=final_notice_version,
        persona_id=borrower.persona.id,
        messages=final_outcome.transcript,
        outcome=final_outcome,
        started_at=started,
        ended_at=datetime.utcnow(),
    ))

    logger.info("Pipeline complete: final status=%s", final_outcome.status)
    return records
