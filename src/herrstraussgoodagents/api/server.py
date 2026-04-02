from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from herrstraussgoodagents.api.bridge import bridge
from herrstraussgoodagents.config import get_settings
from herrstraussgoodagents.models import AgentStage, BorrowerCase
from herrstraussgoodagents.workflow.collections_workflow import CollectionsWorkflow

logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"


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


# ---------------------------------------------------------------------------
# Background job tracking
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BackgroundJob:
    __slots__ = ("id", "kind", "status", "started_at", "ended_at", "result", "error")

    def __init__(self, job_id: str, kind: str) -> None:
        self.id = job_id
        self.kind = kind
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at: str | None = None
        self.result: Any = None
        self.error: str | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "result": self.result,
            "error": self.error,
        }


_jobs: dict[str, BackgroundJob] = {}


def _register_job(kind: str) -> BackgroundJob:
    job = BackgroundJob(str(uuid.uuid4()), kind)
    _jobs[job.id] = job
    return job


def _complete_job(job: BackgroundJob, result: Any) -> None:
    job.status = JobStatus.COMPLETED
    job.ended_at = datetime.now(timezone.utc).isoformat()
    job.result = result


def _fail_job(job: BackgroundJob, error: str) -> None:
    job.status = JobStatus.FAILED
    job.ended_at = datetime.now(timezone.utc).isoformat()
    job.error = error


_STAGE_MAP = {
    "assessment": AgentStage.ASSESSMENT,
    "resolution": AgentStage.RESOLUTION,
    "final_notice": AgentStage.FINAL_NOTICE,
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RunBatchRequest(BaseModel):
    stage: str = Field(..., pattern="^(assessment|resolution|final_notice)$")
    version: str = "v1"
    conversations_per_persona: int = Field(6, ge=1, le=50)
    seed: int = 42


class RunPipelineRequest(BaseModel):
    conversations_per_persona: int = Field(3, ge=1, le=20)
    seed: int = 42


class LearningStartRequest(BaseModel):
    stage: str = Field(..., pattern="^(assessment|resolution|final_notice)$")
    max_iterations: int = Field(3, ge=1, le=10)
    conversations_per_persona: int = Field(6, ge=1, le=50)
    full_pipeline_every: int = Field(3, ge=1, le=10)
    seed: int = 42


# ---------------------------------------------------------------------------
# Harness endpoints
# ---------------------------------------------------------------------------

@app.post("/harness/run-batch")
async def harness_run_batch(req: RunBatchRequest):
    """Run N simulated conversations per persona for a single agent stage."""
    from herrstraussgoodagents.harness.runner import run_single_agent_batch

    job = _register_job("harness:batch")
    stage = _STAGE_MAP[req.stage]

    async def _run():
        try:
            records = await run_single_agent_batch(
                stage=stage,
                agent_version=req.version,
                conversations_per_persona=req.conversations_per_persona,
                seed=req.seed,
            )
            _complete_job(job, {
                "conversations": len(records),
                "sessions": [r.session_id for r in records],
            })
        except Exception as exc:
            logger.exception("Harness batch job %s failed", job.id)
            _fail_job(job, str(exc))

    asyncio.create_task(_run())
    return {"job_id": job.id}


@app.post("/harness/run-pipeline")
async def harness_run_pipeline(req: RunPipelineRequest):
    """Run full 3-agent pipeline simulations."""
    from herrstraussgoodagents.harness.runner import run_full_pipeline_batch

    job = _register_job("harness:pipeline")

    async def _run():
        try:
            runs = await run_full_pipeline_batch(
                conversations_per_persona=req.conversations_per_persona,
                seed=req.seed,
            )
            _complete_job(job, {
                "pipeline_runs": len(runs),
                "stages_per_run": [len(r) for r in runs],
            })
        except Exception as exc:
            logger.exception("Harness pipeline job %s failed", job.id)
            _fail_job(job, str(exc))

    asyncio.create_task(_run())
    return {"job_id": job.id}


@app.get("/harness/jobs/{job_id}")
async def harness_job_status(job_id: str):
    """Check the status of a harness job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return job.to_dict()


# ---------------------------------------------------------------------------
# Learning loop endpoints
# ---------------------------------------------------------------------------

@app.post("/learning/start")
async def learning_start(req: LearningStartRequest):
    """Start the self-learning loop for an agent stage."""
    from herrstraussgoodagents.learning.loop import LearningLoop

    job = _register_job("learning:loop")
    stage = _STAGE_MAP[req.stage]

    async def _run():
        try:
            loop = LearningLoop(
                stage=stage,
                max_iterations=req.max_iterations,
                conversations_per_persona=req.conversations_per_persona,
                full_pipeline_every=req.full_pipeline_every,
                seed=req.seed,
            )
            result = await loop.run()
            _complete_job(job, {
                "iterations_completed": result.iterations_completed,
                "final_version": result.final_version,
                "budget_exhausted": result.budget_exhausted,
                "cost_report": result.cost_report,
                "iterations": [
                    {
                        "iteration": ir.iteration,
                        "baseline_fitness": ir.baseline_fitness,
                        "candidate_fitness": ir.candidate_fitness,
                        "adopted": ir.adopted,
                        "version_id": ir.version_id,
                        "mutation_rationale": ir.mutation_rationale,
                        "effect_size": ir.bootstrap.effect_size,
                        "ci": [ir.bootstrap.ci_lower, ir.bootstrap.ci_upper],
                    }
                    for ir in result.iterations
                ],
            })
        except Exception as exc:
            logger.exception("Learning loop job %s failed", job.id)
            _fail_job(job, str(exc))

    asyncio.create_task(_run())
    return {"job_id": job.id}


@app.get("/learning/status/{job_id}")
async def learning_status(job_id: str):
    """Check the status of a learning loop job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return job.to_dict()


@app.get("/learning/versions/{stage}")
async def learning_versions(stage: str):
    """List all archived prompt versions for a stage."""
    from herrstraussgoodagents.learning.archive import Archive

    if stage not in _STAGE_MAP:
        raise HTTPException(400, f"Unknown stage: {stage}")

    archive = Archive()
    versions = archive.list_versions(stage)
    current = archive.get_current(stage)
    return {
        "stage": stage,
        "current_version": current.version_id if current else None,
        "versions": [v.to_dict() for v in versions],
    }


@app.post("/learning/rollback/{stage}")
async def learning_rollback(stage: str):
    """Roll back to the previous prompt version for a stage."""
    from herrstraussgoodagents.learning.archive import Archive

    if stage not in _STAGE_MAP:
        raise HTTPException(400, f"Unknown stage: {stage}")

    archive = Archive()
    try:
        previous = archive.rollback(stage)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))

    return {
        "rolled_back_to": previous.version_id,
        "stage": stage,
    }


# ---------------------------------------------------------------------------
# Compliance audit endpoints
# ---------------------------------------------------------------------------

class AuditRequest(BaseModel):
    job_id: str


@app.post("/compliance/audit")
async def compliance_audit(req: AuditRequest):
    """Run LLM compliance audit on all records from a completed harness job."""
    from herrstraussgoodagents.compliance.auditor import ComplianceAuditor
    from herrstraussgoodagents.models import ConversationRecord

    job = _jobs.get(req.job_id)
    if job is None:
        raise HTTPException(404, f"Job {req.job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job {req.job_id} is not completed (status={job.status.value})")

    # Reload ConversationRecords from job result session_ids
    result = job.result or {}
    session_ids: list[str] = result.get("sessions", [])
    if not session_ids:
        raise HTTPException(400, "No sessions found in job result")

    audit_job = _register_job("compliance:audit")

    async def _run():
        try:
            from herrstraussgoodagents.harness.runner import DATA_DIR
            import json

            # Reconstruct records from saved JSON files
            records: list[ConversationRecord] = []
            stage = result.get("stage", "")
            version = result.get("version", "v1")
            output_dir = DATA_DIR / f"{stage}_{version}"
            for sid in session_ids:
                path = output_dir / f"{sid}.json"
                if path.exists():
                    records.append(ConversationRecord.model_validate_json(path.read_text()))

            if not records:
                _fail_job(audit_job, "No records could be loaded from disk")
                return

            auditor = ComplianceAuditor()
            audit_results = await auditor.audit_batch(records)

            total = len(audit_results)
            failed = sum(1 for r in audit_results if not r.passed)
            by_stage: dict[str, dict] = {}
            for r in audit_results:
                entry = by_stage.setdefault(r.stage, {"total": 0, "failed": 0, "hard_violations": 0})
                entry["total"] += 1
                if not r.passed:
                    entry["failed"] += 1
                entry["hard_violations"] += r.hard_violations

            _complete_job(audit_job, {
                "total": total,
                "passed": total - failed,
                "failed": failed,
                "by_stage": by_stage,
                "results": [r.to_dict() for r in audit_results],
            })
        except Exception as exc:
            logger.exception("Compliance audit job %s failed", audit_job.id)
            _fail_job(audit_job, str(exc))

    asyncio.create_task(_run())
    return {"job_id": audit_job.id}


@app.get("/compliance/audit/{job_id}")
async def compliance_audit_status(job_id: str):
    """Check the status of a compliance audit job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return job.to_dict()


# ---------------------------------------------------------------------------
# Cost endpoint
# ---------------------------------------------------------------------------

@app.get("/cost/report")
async def cost_report():
    """Return the current cost tracker report and breakdown."""
    from herrstraussgoodagents.llm import get_cost_tracker

    tracker = get_cost_tracker()
    return {
        "total_usd": round(tracker.total_usd, 6),
        "budget_usd": tracker.budget_usd,
        "remaining_usd": round(tracker.budget_usd - tracker.total_usd, 6),
        "breakdown": tracker.breakdown(),
        "report": tracker.report(),
    }
