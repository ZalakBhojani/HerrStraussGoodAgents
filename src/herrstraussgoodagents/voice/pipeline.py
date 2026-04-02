from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    EndFrame,
    LLMContextFrame,
    LLMContextSummaryRequestFrame,
    LLMContextSummaryResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.utils.context.llm_context_summarization import LLMContextSummarizationUtil
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

    Sits between ``LLMUserAggregator`` and ``LLMAssistantAggregator`` in the
    pipeline. Receives ``LLMContextFrame`` (pushed by the user aggregator after
    it adds the borrower's utterance to ``LLMContext``), calls
    ``agent.process_turn()``, and emits:
      LLMFullResponseStartFrame → TextFrame(response) → LLMFullResponseEndFrame

    The ``LLMAssistantAggregator`` downstream captures those frames and appends
    the assistant response to ``LLMContext``, completing the turn.

    When the assistant aggregator triggers auto context summarization it pushes
    ``LLMContextSummaryRequestFrame`` upstream; this processor handles that by
    generating a summary via the agent's LLM client and broadcasting the result
    back as ``LLMContextSummaryResultFrame``.

    On a terminal ``SessionSignal`` it pushes an ``EndFrame`` to tear down
    the pipeline and stores the final outcome on ``self.outcome``.
    """

    def __init__(self, agent: ResolutionAgent, context: LLMContext, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self._context = context
        self.outcome: AgentOutcome | None = None
        self._transcript: list[dict[str, str]] = []

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # --- User turn: triggered by LLMUserAggregator after it adds to context ---
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            messages = frame.context.get_messages()
            user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
            if not user_msgs:
                await self.push_frame(frame, direction)
                return
            borrower_text = user_msgs[-1].get("content", "").strip()
            if not borrower_text:
                return

            logger.debug("Borrower (voice): %s", borrower_text)
            self._transcript.append({"role": "user", "content": borrower_text})

            turn_result = await self.agent.process_turn(borrower_text)

            logger.debug("Agent (voice): %s | signal=%s", turn_result.response[:80], turn_result.signal)
            self._transcript.append({"role": "assistant", "content": turn_result.response})

            # Emit response frames — LLMAssistantAggregator captures these
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TextFrame(turn_result.response))
            await self.push_frame(LLMFullResponseEndFrame())

            if turn_result.signal != SessionSignal.CONTINUE:
                self.outcome = self.agent.build_outcome(turn_result.signal)
                await self.push_frame(EndFrame())
            return

        # --- Context summarization request from LLMAssistantAggregator (upstream) ---
        if isinstance(frame, LLMContextSummaryRequestFrame) and direction == FrameDirection.UPSTREAM:
            asyncio.ensure_future(self._handle_summary_request(frame))
            return

        await self.push_frame(frame, direction)

    async def _handle_summary_request(self, frame: LLMContextSummaryRequestFrame) -> None:
        """Generate a context summary and broadcast the result back to the aggregator."""
        summary = ""
        last_index = -1
        error = None
        try:
            result = LLMContextSummarizationUtil.get_messages_to_summarize(
                frame.context, frame.min_messages_to_keep
            )
            if not result.messages:
                logger.debug("Context summarization: no messages to summarize")
                return

            transcript = LLMContextSummarizationUtil.format_messages_for_summary(result.messages)
            msgs = [
                {"role": "system", "content": frame.summarization_prompt},
                {"role": "user", "content": f"Conversation history:\n{transcript}"},
            ]
            llm_response = await self.agent.client.complete(
                msgs,
                model=self.agent.config.llm.model,
                temperature=0.3,
                max_tokens=frame.target_context_tokens,
            )
            summary = llm_response.text.strip()
            last_index = result.last_summarized_index
            logger.info("Context summarized: %d chars covering %d messages", len(summary), len(result.messages))
        except Exception as exc:
            error = f"Context summarization failed: {exc}"
            logger.warning(error)

        await self.broadcast_frame(
            LLMContextSummaryResultFrame,
            request_id=frame.request_id,
            summary=summary,
            last_summarized_index=last_index,
            error=error,
        )

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
        → LLMUserAggregator      ← adds user utterance to LLMContext, pushes LLMContextFrame
        → VertexAILLMProcessor   ← drives ResolutionAgent, handles summarization requests
        → HangupPhraseDetector
        → Cartesia TTS
        → FastAPI WebSocket (out)
        → LLMAssistantAggregator ← captures response frames into LLMContext after output
    """
    # Shared context — will be seeded with the system prompt by agent.opening_line()
    context = LLMContext()
    agent.set_llm_context(context)

    transport = build_fastapi_transport(websocket)
    stt = build_stt_service(deepgram_api_key)
    llm_processor = VertexAILLMProcessor(agent=agent, context=context, name="resolution-llm")
    hangup_detector = HangupPhraseDetector(name="hangup-detector")
    tts = CartesiaTTSService(api_key=cartesia_api_key, voice_id=cartesia_voice_id)

    ctx_pair = LLMContextAggregatorPair(
        context,
        assistant_params=LLMAssistantAggregatorParams(enable_auto_context_summarization=True),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        ctx_pair.user(),       # TranscriptionFrame → LLMContext → LLMContextFrame
        llm_processor,
        hangup_detector,
        tts,
        transport.output(),
        ctx_pair.assistant(),  # LLMFull*Frame → LLMContext (placed after output per Pipecat convention)
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
