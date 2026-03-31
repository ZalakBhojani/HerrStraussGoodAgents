from __future__ import annotations

import asyncio

from herrstraussgoodagents.models import (
    AgentOutcome,
    AgentStage,
    BorrowerCase,
    HandoffContext,
    OutcomeStatus,
)


class MessageBridge:
    """In-memory asyncio queue bridge: borrower_id → (inbound, outbound) queues.

    Chat agents (Assessment, Final Notice) use the inbound/outbound queues.
    The voice session (Resolution) uses a separate coordination mechanism:

      1. Activity calls ``prepare_voice_session(borrower_id, case, handoff)``
         to stash the context, then ``request_voice_call(borrower_id)`` to
         put ``__VOICE_CALL_READY__`` on the chat outbound queue.
      2. Frontend sees the sentinel and shows the "Start Call" button.
      3. User clicks → browser connects to ``/voice/{borrower_id}``.
      4. FastAPI handler calls ``get_voice_context()`` to retrieve the stash,
         runs the Pipecat pipeline, then calls ``complete_voice_session()``.
      5. The waiting activity unblocks and receives the outcome.
    """

    def __init__(self) -> None:
        self._inbound: dict[str, asyncio.Queue[str]] = {}
        self._outbound: dict[str, asyncio.Queue[str]] = {}
        # Voice session coordination
        self._voice_events: dict[str, asyncio.Event] = {}
        self._voice_outcomes: dict[str, AgentOutcome] = {}
        self._voice_contexts: dict[str, tuple[BorrowerCase, HandoffContext]] = {}

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

    # ------------------------------------------------------------------
    # Voice session coordination
    # ------------------------------------------------------------------

    def prepare_voice_session(
        self,
        borrower_id: str,
        case: BorrowerCase,
        handoff: HandoffContext,
    ) -> None:
        """Stash the case + handoff so the /voice WebSocket handler can retrieve them."""
        self._voice_contexts[borrower_id] = (case, handoff)
        self._voice_events[borrower_id] = asyncio.Event()

    def get_voice_context(
        self, borrower_id: str
    ) -> tuple[BorrowerCase, HandoffContext] | None:
        return self._voice_contexts.get(borrower_id)

    async def request_voice_call(self, borrower_id: str) -> None:
        """Signal the frontend to show the 'Start Call' button."""
        _, outbound = self.get_or_create(borrower_id)
        await outbound.put("__VOICE_CALL_READY__")

    async def wait_for_voice_session(self, borrower_id: str) -> AgentOutcome:
        """Block until the Pipecat pipeline completes and returns the outcome."""
        event = self._voice_events.get(borrower_id)
        if event is None:
            raise RuntimeError(
                f"No voice session registered for borrower_id={borrower_id!r}. "
                "Call prepare_voice_session() first."
            )
        await event.wait()
        outcome = self._voice_outcomes.pop(borrower_id)
        return outcome

    def complete_voice_session(self, borrower_id: str, outcome: AgentOutcome) -> None:
        """Called by the FastAPI voice WebSocket handler when the pipeline ends."""
        self._voice_outcomes[borrower_id] = outcome
        event = self._voice_events.get(borrower_id)
        if event:
            event.set()

    def cancel_all(self) -> None:
        """Unblock all pending waiters so the process can exit.

        Sets all voice events with a fallback UNRESOLVED outcome and
        sends __DONE__ to all outbound queues.
        """
        for bid in list(self._voice_events):
            if not self._voice_events[bid].is_set():
                self._voice_outcomes.setdefault(
                    bid,
                    AgentOutcome(
                        stage=AgentStage.RESOLUTION,
                        status=OutcomeStatus.UNRESOLVED,
                    ),
                )
                self._voice_events[bid].set()
        for q in self._outbound.values():
            q.put_nowait("__DONE__")

    def teardown(self, borrower_id: str) -> None:
        self._inbound.pop(borrower_id, None)
        self._outbound.pop(borrower_id, None)
        self._voice_events.pop(borrower_id, None)
        self._voice_outcomes.pop(borrower_id, None)
        self._voice_contexts.pop(borrower_id, None)


# Module-level singleton shared between FastAPI and Temporal activities
bridge = MessageBridge()
