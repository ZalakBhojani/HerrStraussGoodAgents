from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from herrstraussgoodagents.api.bridge import bridge
from herrstraussgoodagents.config import get_settings
from herrstraussgoodagents.models import BorrowerCase
from herrstraussgoodagents.workflow.collections_workflow import CollectionsWorkflow

logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="HerrStrauss Collections", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


# ---------------------------------------------------------------------------
# Chat WebSocket — Assessment and Final Notice (text-based)
# ---------------------------------------------------------------------------

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
                if message == "__AGENT_DONE__":
                    continue  # stage boundary — keep WebSocket open
                if message == "__DONE__":
                    break     # full session over — close
                await websocket.send_text(message)
        except WebSocketDisconnect:
            pass

    tasks = {
        asyncio.create_task(receive_loop()),
        asyncio.create_task(send_loop()),
    }
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    bridge.teardown(borrower_id)


# ---------------------------------------------------------------------------
# Voice WebSocket — Resolution agent via Pipecat pipeline
# ---------------------------------------------------------------------------

@app.websocket("/voice/{borrower_id}")
async def voice_websocket(websocket: WebSocket, borrower_id: str):
    """Receives raw PCM audio from the browser, runs the Pipecat pipeline,
    and streams TTS audio back.  When the pipeline ends, signals the
    Temporal activity waiting in bridge.wait_for_voice_session().
    """
    from herrstraussgoodagents.agents.resolution import ResolutionAgent
    from herrstraussgoodagents.config import load_agent_config, get_settings
    from herrstraussgoodagents.voice.pipeline import run_voice_session

    await websocket.accept()

    # Retrieve the handoff context that was stashed by run_resolution_for_case
    # via bridge.  The outbound queue contains __VOICE_CALL_READY__ which the
    # chat frontend already consumed; the handoff is in the bridge's voice slot.
    # We need the BorrowerCase + HandoffContext to construct the agent.
    # These are passed via a small side-channel: the activity stashes them
    # in the bridge before signaling VOICE_CALL_READY.
    voice_ctx = bridge.get_voice_context(borrower_id)
    if voice_ctx is None:
        logger.warning("Voice WebSocket: no context for borrower_id=%r, closing.", borrower_id)
        await websocket.close(code=1008)
        return

    case, handoff = voice_ctx
    settings = get_settings()
    config = load_agent_config("resolution")
    agent = ResolutionAgent(config, case, handoff)

    try:
        result = await run_voice_session(
            agent,
            websocket,
            deepgram_api_key=settings.deepgram_api_key,
            cartesia_api_key=settings.cartesia_api_key,
        )
        bridge.complete_voice_session(borrower_id, result.outcome)
        logger.info(
            "Voice session complete for %r: status=%s", borrower_id, result.outcome.status
        )
    except Exception:
        logger.exception("Voice pipeline error for borrower_id=%r", borrower_id)
        from herrstraussgoodagents.models import AgentOutcome, AgentStage, OutcomeStatus
        bridge.complete_voice_session(
            borrower_id,
            AgentOutcome(stage=AgentStage.RESOLUTION, status=OutcomeStatus.UNRESOLVED),
        )


# ---------------------------------------------------------------------------
# Workflow trigger
# ---------------------------------------------------------------------------

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
