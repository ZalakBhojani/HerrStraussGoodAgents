from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AgentStage(str, Enum):
    ASSESSMENT = "assessment"
    RESOLUTION = "resolution"
    FINAL_NOTICE = "final_notice"


class AgentModality(str, Enum):
    CHAT = "chat"
    VOICE = "voice"


class ResolutionPath(str, Enum):
    LUMP_SUM = "lump_sum"
    PAYMENT_PLAN = "payment_plan"
    HARDSHIP_REFERRAL = "hardship_referral"
    UNRESOLVED = "unresolved"


class OutcomeStatus(str, Enum):
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    HUNG_UP = "hung_up"
    CEASE_REQUESTED = "cease_requested"
    IDENTITY_FAILED = "identity_failed"
    UNRESOLVED = "unresolved"
    IN_PROGRESS = "in_progress"


class BorrowerCase(BaseModel):
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    borrower_id: str
    borrower_name: str
    account_last_four: str
    debt_amount: float
    months_overdue: int
    original_creditor: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TurnSource(str, Enum):
    LLM = "llm"
    DETERMINISTIC = "deterministic"
    BORROWER = "borrower"


class ConversationMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str
    source: TurnSource = TurnSource.LLM
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: int | None = None


class Turn(BaseModel):
    turn_number: int
    agent_message: str
    borrower_message: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HandoffContext(BaseModel):
    # Structured fields
    identity_verified: bool
    debt_amount: float
    months_overdue: int
    offers_made: list[str] = Field(default_factory=list)
    objections_raised: list[str] = Field(default_factory=list)
    resolution_path: ResolutionPath | None = None
    # Qualitative prose (~100 tokens)
    tone_summary: str = ""
    # Metadata
    source_stage: AgentStage
    token_count: int = 0


class AgentOutcome(BaseModel):
    stage: AgentStage
    status: OutcomeStatus
    resolution_path: ResolutionPath | None = None
    settlement_amount: float | None = None
    payment_plan_months: int | None = None
    handoff_context: HandoffContext | None = None
    # transcript: list[ConversationMessage] = Field(default_factory=list)
    turns_taken: int = 0
    # tokens_used: int = 0


class ConversationRecord(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str
    stage: AgentStage
    agent_version: str
    persona_id: str | None = None
    turns: list[Turn] = Field(default_factory=list)
    messages: list[ConversationMessage] = Field(default_factory=list)
    outcome: AgentOutcome | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
