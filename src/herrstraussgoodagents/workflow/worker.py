from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from herrstraussgoodagents.config import get_settings
from herrstraussgoodagents.workflow.activities import (
    run_assessment,
    run_final_notice,
    run_resolution_for_case,
)
from herrstraussgoodagents.workflow.collections_workflow import CollectionsWorkflow


async def create_worker() -> Worker:
    settings = get_settings()
    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )
    # Use asyncio activity executor (not thread pool) — required for queue sharing with FastAPI
    return Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[CollectionsWorkflow],
        activities=[run_assessment, run_resolution_for_case, run_final_notice],
    )


async def run_worker() -> None:
    worker = await create_worker()
    await worker.run()
