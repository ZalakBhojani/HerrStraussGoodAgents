from __future__ import annotations

import logging

from pipecat.frames.frames import BotStoppedSpeakingFrame, EndTaskFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)

# Phrases in agent output that should cleanly end the call
_HANGUP_PHRASES = [
    "goodbye",
    "good bye",
    "thank you for your time",
    "have a good day",
    "take care",
    "we'll be in touch",
    "written confirmation",  # agreement confirmations always end the call
]


class HangupPhraseDetector(FrameProcessor):
    """Monitors outbound TextFrames for terminal phrases and pushes EndFrame.

    Sits between the LLM processor and TTS so the TTS still speaks the
    final sentence before the pipeline tears down.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._phrases = [p.lower() for p in _HANGUP_PHRASES]
        self._triggered = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Step 1: detect terminal phrase in LLM output, pass it through for TTS
        if isinstance(frame, TextFrame) and not self._triggered:
            if any(phrase in frame.text.lower() for phrase in self._phrases):
                self._triggered = True
                logger.info(
                    f"[HangupDetector] Terminal phrase detected: {frame.text!r} — "
                    "will close after bot finishes speaking"
                )

        # Step 2: once TTS has fully played, close the pipeline
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._triggered:
            logger.info("[HangupDetector] Bot finished speaking — sending EndFrame")
            await self.push_frame(frame, direction)
            await self.push_frame(EndTaskFrame("Bot came up with the hangup_phase"), FrameDirection.UPSTREAM)
            return

        await self.push_frame(frame, direction)
