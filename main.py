from __future__ import annotations

import asyncio
import logging

import uvicorn

from herrstraussgoodagents.config import get_settings
from herrstraussgoodagents.workflow.worker import create_worker

logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()

    from herrstraussgoodagents.api.server import app

    config = uvicorn.Config(
        app,
        host=settings.app_host,
        port=settings.app_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    worker = await create_worker()
    worker_task = asyncio.create_task(worker.run())

    try:
        await server.serve()
    finally:
        # Unblock any bridge waiters so in-flight activities can finish
        from herrstraussgoodagents.api.bridge import bridge
        bridge.cancel_all()

        # Give the worker a bounded window to drain, then force-cancel
        try:
            await asyncio.wait_for(worker.shutdown(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Temporal worker did not shut down within 5 s, cancelling.")
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
