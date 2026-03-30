from __future__ import annotations

from temporalio import activity

from herrstraussgoodagents.models import (
    AgentOutcome,
    AgentStage,
    BorrowerCase,
    OutcomeStatus,
    ResolutionPath,
)


@activity.defn
async def run_assessment(case: BorrowerCase) -> AgentOutcome:
    """[STUB] Assessment agent — identifies borrower, gathers info, routes to resolution."""
    activity.logger.info(f"[STUB] Assessment for case {case.case_id} ({case.borrower_name})")
    return AgentOutcome(
        stage=AgentStage.ASSESSMENT,
        status=OutcomeStatus.ESCALATED,
        resolution_path=ResolutionPath.PAYMENT_PLAN,
        turns_taken=4,
    )


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
