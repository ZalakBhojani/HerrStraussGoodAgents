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

    # Run FastAPI and Temporal worker concurrently in the same event loop.
    # The asyncio activity executor is used by default in the worker,
    # which allows queue sharing between FastAPI WebSocket handlers and activities.
    await asyncio.gather(
        server.serve(),
        run_worker(),
    )


if __name__ == "__main__":
    asyncio.run(main())
