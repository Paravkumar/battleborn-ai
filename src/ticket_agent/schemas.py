from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ExecutionStatus = Literal["completed", "escalated", "failed", "waiting_on_customer"]
StepOutcome = Literal["success", "failed", "escalated", "skipped"]


class Ticket(BaseModel):
    ticket_id: str
    customer_id: str
    subject: str
    body: str
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    status: str = "open"
    issue_type: str | None = None
    resolution_summary: str | None = None
    customer_message_history: list[str] = Field(default_factory=list)
    pending_customer_replies: list[str] = Field(default_factory=list)
    response_history: list[str] = Field(default_factory=list)
    tool_failures: dict[str, int] = Field(default_factory=dict)


class KnowledgeBaseArticle(BaseModel):
    article_id: str
    title: str
    issue_type: str
    information_type: Literal["policy", "product", "hybrid"] = "hybrid"
    keywords: list[str]
    summary: str
    resolution_steps: list[str]
    customer_reply_template: str


class KnowledgeBaseHit(BaseModel):
    article_id: str
    title: str
    issue_type: str
    information_type: Literal["policy", "product", "hybrid"] = "hybrid"
    summary: str
    matched_terms: list[str] = Field(default_factory=list)
    score: float
    confidence_score: float


class PlanStep(BaseModel):
    step_id: str
    tool: str
    purpose: str


class Plan(BaseModel):
    summary: str
    steps: list[PlanStep]


class ToolOutcome(BaseModel):
    success: bool
    retryable: bool = False
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class DraftedReply(BaseModel):
    body: str
    citations: list[str] = Field(default_factory=list)
    resolution_confidence: Literal["high", "medium", "low"] = "medium"
    needs_escalation: bool = False
    next_action: Literal["await_customer", "resolve", "escalate"] = "await_customer"


class ExecutionStepRecord(BaseModel):
    step_id: str
    tool: str
    purpose: str
    conversation_turn: int = 1
    attempt: int
    arguments: dict[str, Any] = Field(default_factory=dict)
    outcome: StepOutcome
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False


class ExecutionLog(BaseModel):
    goal: str
    plan_source: Literal["ollama", "fallback"]
    response_source: Literal["ollama", "fallback", "none"] = "none"
    ticket_id: str
    ticket_snapshot: dict[str, Any]
    model_used: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    status: ExecutionStatus = "failed"
    final_ticket_status: str | None = None
    final_response: str | None = None
    plan: Plan
    steps: list[ExecutionStepRecord] = Field(default_factory=list)
