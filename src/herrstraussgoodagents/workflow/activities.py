from __future__ import annotations

from temporalio import activity

from herrstraussgoodagents.models import (
    AgentOutcome,
    AgentStage,
    BorrowerCase,
    OutcomeStatus,
)


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
async def run_resolution_for_case(case: BorrowerCase, prior_outcome: AgentOutcome) -> AgentOutcome:
    """Resolution agent — voice call via Pipecat pipeline.

    Flow:
      1. Stash BorrowerCase + HandoffContext in the bridge so the
         /voice/{borrower_id} WebSocket handler can construct the agent.
      2. Put __VOICE_CALL_READY__ on the chat outbound queue → frontend
         shows the "Start Call" button.
      3. Block until the Pipecat pipeline completes.
      4. Return the AgentOutcome from the pipeline.
    """
    from herrstraussgoodagents.api.bridge import bridge

    handoff = prior_outcome.handoff_context
    if handoff is None:
        activity.logger.warning("Resolution: no handoff context from assessment, using unresolved.")
        return AgentOutcome(
            stage=AgentStage.RESOLUTION,
            status=OutcomeStatus.UNRESOLVED,
            resolution_path=prior_outcome.resolution_path,
            turns_taken=0,
        )

    activity.logger.info(
        f"Resolution: preparing voice session for {case.borrower_id!r} ({case.borrower_name})"
    )
    bridge.prepare_voice_session(case.borrower_id, case, handoff)
    await bridge.request_voice_call(case.borrower_id)

    activity.logger.info("Resolution: waiting for voice session to complete…")
    outcome = await bridge.wait_for_voice_session(case.borrower_id)
    activity.logger.info(f"Resolution complete: {outcome.status}")
    return outcome


@activity.defn
async def run_final_notice(case: BorrowerCase, prior_outcome: AgentOutcome) -> AgentOutcome:
    """Final notice agent — last-chance chat offer with 72h deadline."""
    from herrstraussgoodagents.agents.final_notice import FinalNoticeAgent
    from herrstraussgoodagents.api.bridge import bridge
    from herrstraussgoodagents.config import load_agent_config

    handoff = prior_outcome.handoff_context
    if handoff is None:
        activity.logger.warning("Final notice: no handoff context, using unresolved.")
        return AgentOutcome(
            stage=AgentStage.FINAL_NOTICE,
            status=OutcomeStatus.UNRESOLVED,
            resolution_path=prior_outcome.resolution_path,
            turns_taken=0,
        )

    activity.logger.info(
        f"Final notice started for case {case.case_id} ({case.borrower_name})"
    )
    inbound, outbound = bridge.get_or_create(case.borrower_id)
    config = load_agent_config("final_notice")
    agent = FinalNoticeAgent(config, case, handoff)
    outcome = await agent.run(inbound, outbound)
    activity.logger.info(f"Final notice complete: {outcome.status}")
    return outcome
