from __future__ import annotations

import logging

from pipecat.frames.frames import EndFrame, TextFrame
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
        self._triggered = False

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and not self._triggered:
            text_lower = frame.text.lower()
            if any(phrase in text_lower for phrase in _HANGUP_PHRASES):
                self._triggered = True
                logger.info("HangupPhraseDetector: hangup phrase found, scheduling EndFrame")
                # Pass the text frame through so TTS speaks it first
                await self.push_frame(frame, direction)
                # Then end the pipeline
                await self.push_frame(EndFrame())
                return

        await self.push_frame(frame, direction)
