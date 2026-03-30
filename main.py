from __future__ import annotations

import asyncio

import uvicorn

from herrstraussgoodagents.config import get_settings
from herrstraussgoodagents.workflow.worker import run_worker


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

    # Start the Temporal worker as a background task
    worker_task = asyncio.create_task(run_worker())

    try:
        # uvicorn.serve() handles SIGINT/SIGTERM itself and exits cleanly
        await server.serve()
    finally:
        # Once uvicorn exits (graceful or forced), cancel the worker immediately
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
