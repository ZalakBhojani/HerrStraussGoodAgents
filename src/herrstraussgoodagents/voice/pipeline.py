from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from starlette.websockets import WebSocket

from herrstraussgoodagents.agents.resolution import ResolutionAgent, SessionSignal
from herrstraussgoodagents.models import AgentOutcome
from herrstraussgoodagents.voice.hangup_detector import HangupPhraseDetector
from herrstraussgoodagents.voice.transport import build_fastapi_transport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom LLM processor — wraps ResolutionAgent inside a FrameProcessor
# ---------------------------------------------------------------------------

class VertexAILLMProcessor(FrameProcessor):
    """Pipecat FrameProcessor that drives the ResolutionAgent per voice turn.

    Receives ``TranscriptionFrame`` (finalized) from Deepgram, calls
    ``agent.process_turn()``, and emits:
      LLMFullResponseStartFrame → TextFrame(response) → LLMFullResponseEndFrame

    On a terminal ``SessionSignal`` it pushes an ``EndFrame`` to tear down
    the pipeline and stores the final outcome on ``self.outcome``.
    """

    def __init__(self, agent: ResolutionAgent, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.outcome: AgentOutcome | None = None
        self._transcript: list[dict[str, str]] = []

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.finalized:
            borrower_text = frame.text.strip()
            if not borrower_text:
                await self.push_frame(frame, direction)
                return

            logger.debug("Borrower (voice): %s", borrower_text)
            self._transcript.append({"role": "user", "content": borrower_text})

            turn_result = await self.agent.process_turn(borrower_text)

            logger.debug("Agent (voice): %s | signal=%s", turn_result.response[:80], turn_result.signal)
            self._transcript.append({"role": "assistant", "content": turn_result.response})

            # Emit response frames for TTS
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TextFrame(turn_result.response))
            await self.push_frame(LLMFullResponseEndFrame())

            # Terminal signal → build outcome and end pipeline
            if turn_result.signal != SessionSignal.CONTINUE:
                self.outcome = self.agent.build_outcome(turn_result.signal)
                await self.push_frame(EndFrame())
            return

        await self.push_frame(frame, direction)

    @property
    def transcript(self) -> list[dict[str, str]]:
        return self._transcript


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

@dataclass
class VoiceSessionResult:
    outcome: AgentOutcome
    transcript: list[dict[str, str]] = field(default_factory=list)

def build_stt_service(apiKey: str) -> DeepgramSTTService:
    """Return a DeepgramSTTService configured from AgentConfig."""
    return DeepgramSTTService(
        api_key=apiKey,
        settings=DeepgramSTTService.Settings(
            model="nova-2",
            smart_format=True,
            punctuate=True,
        ),
    )

async def run_voice_session(
    agent: ResolutionAgent,
    websocket: WebSocket,
    *,
    deepgram_api_key: str = "",
    cartesia_api_key: str = "",
    cartesia_voice_id: str = "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
) -> VoiceSessionResult:
    """Assemble and run the full voice pipeline for one borrower session.

    Accepts an already-accepted FastAPI WebSocket. Blocks until the session
    ends (agreement, hangup, cease, or client disconnect).

    Pipeline:
      FastAPI WebSocket (in)
        → Deepgram STT
        → VertexAILLMProcessor  ← drives ResolutionAgent
        → HangupPhraseDetector
        → Cartesia TTS
        → FastAPI WebSocket (out)
    """
    transport = build_fastapi_transport(websocket)
    stt = build_stt_service(deepgram_api_key)
    llm_processor = VertexAILLMProcessor(agent=agent, name="resolution-llm")
    hangup_detector = HangupPhraseDetector(name="hangup-detector")
    tts = CartesiaTTSService(api_key=cartesia_api_key, voice_id=cartesia_voice_id)

    pipeline = Pipeline([
        transport.input(),
        stt,
        llm_processor,
        hangup_detector,
        tts,
        transport.output(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Speak the opening line as soon as the client connects
    opening = agent.opening_line()
    logger.info("Voice session starting: %s…", opening[:80])

    @transport.event_handler("on_client_connected")
    async def on_connected(transport_ref, client):
        await task.queue_frames([
            LLMFullResponseStartFrame(),
            TextFrame(opening),
            LLMFullResponseEndFrame(),
        ])

    runner = PipelineRunner()
    await runner.run(task)

    outcome = llm_processor.outcome or agent.build_outcome(SessionSignal.END_UNRESOLVED)
    return VoiceSessionResult(outcome=outcome, transcript=llm_processor.transcript)
