from __future__ import annotations

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from starlette.websockets import WebSocket

_AUDIO_SAMPLE_RATE = 16_000


def build_fastapi_transport(websocket: WebSocket) -> FastAPIWebsocketTransport:
    """Create a pipecat transport from an existing FastAPI WebSocket connection.

    This is the production path: the voice pipeline runs directly inside
    FastAPI's event loop with no separate port. Audio flows through the
    same origin as the chat UI.
    """
    params = FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        serializer=ProtobufFrameSerializer(),
        add_wav_header=False,
    )
    return FastAPIWebsocketTransport(websocket=websocket, params=params)
