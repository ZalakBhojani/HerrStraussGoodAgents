from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from herrstraussgoodagents.models import (
        AgentOutcome,
        BorrowerCase,
        OutcomeStatus,
    )
    from herrstraussgoodagents.workflow.activities import (
        run_assessment,
        run_final_notice,
        run_resolution_for_case,
    )


@workflow.defn
class CollectionsWorkflow:
    @workflow.run
    async def run(self, case: BorrowerCase) -> AgentOutcome:
        # Stage 1: Assessment (chat) — up to 3 retries
        assessment_outcome: AgentOutcome = await workflow.execute_activity(
            run_assessment,
            case,
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=2),
            ),
        )

        if assessment_outcome.status in (
            OutcomeStatus.CEASE_REQUESTED,
            OutcomeStatus.IDENTITY_FAILED,
        ):
            return assessment_outcome

        # Stage 2: Resolution (voice) — no retries on voice calls
        resolution_outcome: AgentOutcome = await workflow.execute_activity(
            run_resolution_for_case,
            args=[case, assessment_outcome],
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )

        if resolution_outcome.status in (
            OutcomeStatus.RESOLVED,
            OutcomeStatus.CEASE_REQUESTED,
        ):
            return resolution_outcome

        # Stage 3: Final Notice (chat)
        final_outcome: AgentOutcome = await workflow.execute_activity(
            run_final_notice,
            args=[case, resolution_outcome],
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )

        return final_outcome
