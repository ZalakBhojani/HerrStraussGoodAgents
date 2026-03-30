from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from herrstraussgoodagents.api.bridge import bridge
from herrstraussgoodagents.config import get_settings
from herrstraussgoodagents.models import BorrowerCase
from herrstraussgoodagents.workflow.collections_workflow import CollectionsWorkflow

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="HerrStrauss Collections", lifespan=lifespan)


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.websocket("/ws/{borrower_id}")
async def chat_websocket(websocket: WebSocket, borrower_id: str):
    await websocket.accept()
    inbound, outbound = bridge.get_or_create(borrower_id)

    async def receive_loop() -> None:
        try:
            while True:
                data = await websocket.receive_text()
                await inbound.put(data)
        except WebSocketDisconnect:
            pass

    async def send_loop() -> None:
        try:
            while True:
                message = await outbound.get()
                if message == "__DONE__":
                    break
                await websocket.send_text(message)
        except WebSocketDisconnect:
            pass

    await asyncio.gather(receive_loop(), send_loop())
    bridge.teardown(borrower_id)


@app.post("/workflow/start")
async def start_workflow(case: BorrowerCase):
    from temporalio.client import Client

    settings = get_settings()
    client = await Client.connect(
        settings.temporal_host, namespace=settings.temporal_namespace
    )
    handle = await client.start_workflow(
        CollectionsWorkflow.run,
        case,
        id=f"collections-{case.case_id}",
        task_queue=settings.temporal_task_queue,
    )
    return {"workflow_id": handle.id, "run_id": handle.first_execution_run_id}
