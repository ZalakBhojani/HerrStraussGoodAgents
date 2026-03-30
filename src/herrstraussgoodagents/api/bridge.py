from __future__ import annotations

import asyncio


class MessageBridge:
    """In-memory asyncio queue bridge: borrower_id → (inbound, outbound) queues.

    The FastAPI WebSocket handler puts borrower messages on inbound and reads
    agent responses from outbound. Temporal activities do the reverse.
    """

    def __init__(self) -> None:
        self._inbound: dict[str, asyncio.Queue[str]] = {}
        self._outbound: dict[str, asyncio.Queue[str]] = {}

    def get_or_create(
        self, borrower_id: str
    ) -> tuple[asyncio.Queue[str], asyncio.Queue[str]]:
        if borrower_id not in self._inbound:
            self._inbound[borrower_id] = asyncio.Queue()
            self._outbound[borrower_id] = asyncio.Queue()
        return self._inbound[borrower_id], self._outbound[borrower_id]

    def get_inbound(self, borrower_id: str) -> asyncio.Queue[str] | None:
        return self._inbound.get(borrower_id)

    def get_outbound(self, borrower_id: str) -> asyncio.Queue[str] | None:
        return self._outbound.get(borrower_id)

    def teardown(self, borrower_id: str) -> None:
        self._inbound.pop(borrower_id, None)
        self._outbound.pop(borrower_id, None)


# Module-level singleton shared between FastAPI and Temporal activities
bridge = MessageBridge()
