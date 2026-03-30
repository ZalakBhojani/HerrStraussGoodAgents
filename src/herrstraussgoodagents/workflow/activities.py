from __future__ import annotations

from temporalio import activity

from herrstraussgoodagents.models import (
    AgentOutcome,
    AgentStage,
    BorrowerCase,
    OutcomeStatus,
    ResolutionPath,
)

# ResolutionPath imported for stub activities below; remove when stubs are replaced


@activity.defn
async def run_assessment(case: BorrowerCase) -> AgentOutcome:
    """Assessment agent — multi-turn chat via WebSocket bridge."""
    from herrstraussgoodagents.agents.assessment import AssessmentAgent
    from herrstraussgoodagents.api.bridge import bridge
    from herrstraussgoodagents.config import load_agent_config

    activity.logger.info(f"Assessment started for case {case.case_id} ({case.borrower_name})")
    inbound, outbound = bridge.get_or_create(case.borrower_id)
    config = load_agent_config("assessment")
    agent = AssessmentAgent(config, case)
    outcome = await agent.run(inbound, outbound)
    activity.logger.info(f"Assessment complete: {outcome.status}, path={outcome.resolution_path}")
    return outcome


@activity.defn
async def run_resolution(prior_outcome: AgentOutcome) -> AgentOutcome:
    """[STUB] Resolution agent — voice call, negotiates settlement."""
    activity.logger.info(f"[STUB] Resolution after {prior_outcome.stage}")
    return AgentOutcome(
        stage=AgentStage.RESOLUTION,
        status=OutcomeStatus.ESCALATED,
        resolution_path=prior_outcome.resolution_path,
        turns_taken=6,
    )


@activity.defn
async def run_final_notice(prior_outcome: AgentOutcome) -> AgentOutcome:
    """[STUB] Final notice agent — last offer, documents outcome."""
    activity.logger.info(f"[STUB] Final notice after {prior_outcome.stage}")
    return AgentOutcome(
        stage=AgentStage.FINAL_NOTICE,
        status=OutcomeStatus.UNRESOLVED,
        resolution_path=prior_outcome.resolution_path,
        turns_taken=3,
    )
