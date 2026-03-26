import json
import tempfile
from pathlib import Path

from ticket_agent.agent import TicketResolutionAgent
from ticket_agent.config import Settings
from ticket_agent.knowledge_base import KnowledgeBase
from ticket_agent.repository import TicketRepository
from ticket_agent.schemas import DraftedReply, Plan, PlanStep
from ticket_agent.tools import TicketResolutionToolkit


FIXTURE_TICKETS = Path(__file__).resolve().parent / "fixtures" / "sample_tickets.json"


def build_agent(tickets_path: Path | None = None) -> TicketResolutionAgent:
    settings = Settings(use_ollama=False)
    resolved_tickets_path = tickets_path
    if resolved_tickets_path is None:
        temp_dir = Path(tempfile.mkdtemp())
        resolved_tickets_path = temp_dir / "tickets.json"
        resolved_tickets_path.write_text(FIXTURE_TICKETS.read_text(encoding="utf-8"), encoding="utf-8")
    repository = TicketRepository(resolved_tickets_path)
    kb = KnowledgeBase(settings.data_dir / "knowledge_base.json")
    toolkit = TicketResolutionToolkit(
        repository=repository,
        kb=kb,
        llm=None,
        student_model=settings.student_model,
        use_ollama=False,
        min_kb_confidence=settings.min_kb_confidence,
        min_classification_confidence=settings.min_classification_confidence,
    )
    return TicketResolutionAgent(settings=settings, repository=repository, toolkit=toolkit, llm=None)


def test_password_ticket_resolves() -> None:
    agent = build_agent()
    log = agent.run("TICK-1001", "Resolve the customer ticket end-to-end.")
    assert log.status == "completed"
    assert log.final_ticket_status == "resolved"
    classification = next(step for step in log.steps if step.tool == "classify_ticket")
    assert classification.data["intent"] == "restore_account_access"
    search = next(step for step in log.steps if step.tool == "search_kb" and step.outcome == "success")
    assert search.data["hits"][0]["article_id"] == "KB-100"
    assert "product" in search.data["information_types"]
    assert "matched_terms" in search.data["hits"][0]
    assert any(step.tool == "customer_reply" for step in log.steps)
    post_steps = [step for step in log.steps if step.tool == "post_response"]
    assert len(post_steps) == 2
    posted_messages = [step.arguments["response_body"] for step in post_steps]
    assert all("References:" not in message for message in posted_messages)
    assert all("knowledge base" not in message.lower() for message in posted_messages)
    assert len(post_steps[0].data["response_history"]) == 1
    assert len(post_steps[1].data["response_history"]) == 2


def test_unknown_ticket_escalates() -> None:
    agent = build_agent()
    log = agent.run("TICK-1003", "Resolve the customer ticket end-to-end.")
    assert log.status == "escalated"
    assert log.final_ticket_status == "escalated"


def test_billing_ticket_retries_then_resolves() -> None:
    agent = build_agent()
    log = agent.run("TICK-1002", "Resolve the customer ticket end-to-end.")
    assert log.status == "completed"
    assert log.final_ticket_status == "resolved"
    search_steps = [step for step in log.steps if step.tool == "search_kb"]
    assert len(search_steps) == 3
    assert search_steps[0].outcome == "failed"
    assert search_steps[1].outcome == "success"
    assert any(step.tool == "customer_reply" for step in log.steps)


def test_agent_waits_for_customer_when_no_scripted_reply_is_available() -> None:
    agent = build_agent()
    agent.repository.clear_pending_customer_replies("TICK-1001")

    log = agent.run("TICK-1001", "Resolve the customer ticket end-to-end.")

    assert log.status == "waiting_on_customer"
    assert log.final_ticket_status == "pending_customer"
    assert any(step.tool == "post_response" for step in log.steps)
    assert not any(step.tool == "customer_reply" for step in log.steps)


def test_plan_normalization_repairs_missing_status_step() -> None:
    agent = build_agent()
    raw_plan = Plan(
        summary="Model plan",
        steps=[
            PlanStep(step_id="a", tool="read_ticket", purpose="Read."),
            PlanStep(step_id="b", tool="classify_ticket", purpose="Classify."),
            PlanStep(step_id="c", tool="search_kb", purpose="Search."),
            PlanStep(step_id="d", tool="draft_response", purpose="Draft."),
            PlanStep(step_id="e", tool="post_response", purpose="Post."),
            PlanStep(step_id="f", tool="escalate_ticket", purpose="Escalate if needed."),
        ],
    )
    assert agent._plan_is_usable(raw_plan) is True
    normalized = agent._normalize_plan(raw_plan)
    assert [step.tool for step in normalized.steps] == [
        "read_ticket",
        "classify_ticket",
        "search_kb",
        "draft_response",
        "post_response",
        "update_ticket_status",
    ]


def test_supported_first_turn_does_not_escalate_too_early(monkeypatch) -> None:
    agent = build_agent()
    agent.repository.clear_pending_customer_replies("TICK-1001")

    def eager_escalation(*args, **kwargs) -> DraftedReply:
        return DraftedReply(
            body="This needs escalation immediately.",
            citations=["KB-100"],
            resolution_confidence="low",
            needs_escalation=True,
            next_action="escalate",
        )

    monkeypatch.setattr(agent.toolkit, "_draft_with_model", eager_escalation)

    log = agent.run("TICK-1001", "Resolve the customer ticket end-to-end.")

    assert log.status == "waiting_on_customer"
    draft_step = next(step for step in log.steps if step.tool == "draft_response" and step.outcome == "success")
    assert draft_step.data["next_action"] == "await_customer"
    assert draft_step.data["needs_escalation"] is False


def test_unresolved_follow_up_escalates_instead_of_looping(monkeypatch) -> None:
    agent = build_agent()
    agent.repository.get("TICK-1001").pending_customer_replies = [
        "I followed the reset steps but I still cannot sign in to the account."
    ]

    def hesitant_model(ticket_id: str, articles) -> DraftedReply:
        ticket = agent.repository.get(ticket_id)
        if not ticket.response_history:
            return DraftedReply(
                body="Please use the password reset link tied to your registered email and then try signing in again.",
                citations=["KB-100"],
                resolution_confidence="high",
                needs_escalation=False,
                next_action="await_customer",
            )
        return DraftedReply(
            body="Please retry the same reset steps once more and let us know what happens.",
            citations=["KB-100"],
            resolution_confidence="high",
            needs_escalation=False,
            next_action="await_customer",
        )

    monkeypatch.setattr(agent.toolkit, "_draft_with_model", hesitant_model)

    log = agent.run("TICK-1001", "Resolve the customer ticket end-to-end.")

    assert log.status == "escalated"
    failed_draft = next(
        step
        for step in log.steps
        if step.tool == "draft_response" and step.outcome == "failed" and step.conversation_turn == 2
    )
    assert failed_draft.data["next_action"] == "escalate"


def test_subscription_refund_ticket_gets_grounded_billing_response(tmp_path: Path) -> None:
    tickets_path = tmp_path / "tickets.json"
    tickets_path.write_text(
        json.dumps(
            [
                {
                    "ticket_id": "TICK-2001",
                    "customer_id": "CUST-2001",
                    "subject": "Worst subscription service",
                    "body": "i bought your subscription and it is the worst one there is i want my money back.",
                    "priority": "high",
                    "status": "open",
                    "issue_type": None,
                    "resolution_summary": None,
                    "customer_message_history": [],
                    "pending_customer_replies": [],
                    "response_history": [],
                    "tool_failures": {},
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    agent = build_agent(tickets_path=tickets_path)
    log = agent.run("TICK-2001", "Resolve the customer ticket end-to-end.")

    assert log.status == "waiting_on_customer"
    classification = next(step for step in log.steps if step.tool == "classify_ticket")
    assert classification.data["issue_type"] == "billing"
    assert classification.data["intent"] == "request_subscription_refund"
    search = next(step for step in log.steps if step.tool == "search_kb" and step.outcome == "success")
    assert search.data["hits"][0]["article_id"] == "KB-115"
    draft = next(step for step in log.steps if step.tool == "draft_response" and step.outcome == "success")
    assert "refund review" in draft.data["body"].lower()


def test_vague_ticket_gets_clarification_before_escalation(tmp_path: Path) -> None:
    tickets_path = tmp_path / "tickets.json"
    tickets_path.write_text(
        json.dumps(
            [
                {
                    "ticket_id": "TICK-3001",
                    "customer_id": "CUST-3001",
                    "subject": "Need help urgently",
                    "body": "This is bad and not working.",
                    "priority": "high",
                    "status": "open",
                    "issue_type": None,
                    "resolution_summary": None,
                    "customer_message_history": [],
                    "pending_customer_replies": [],
                    "response_history": [],
                    "tool_failures": {},
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    agent = build_agent(tickets_path=tickets_path)
    log = agent.run("TICK-3001", "Resolve the customer ticket end-to-end.")

    assert log.status == "waiting_on_customer"
    search = next(step for step in log.steps if step.tool == "search_kb" and step.outcome == "success")
    assert search.data["needs_clarification"] is True
    assert search.data["hits"][0]["article_id"] == "KB-143"
    draft = next(step for step in log.steps if step.tool == "draft_response" and step.outcome == "success")
    assert draft.data["next_action"] == "await_customer"
    assert "what you were trying to do" in draft.data["body"].lower()


def test_escalation_summary_includes_context_after_clarification_turn(tmp_path: Path) -> None:
    tickets_path = tmp_path / "tickets.json"
    tickets_path.write_text(
        json.dumps(
            [
                {
                    "ticket_id": "TICK-3002",
                    "customer_id": "CUST-3002",
                    "subject": "Need help urgently",
                    "body": "This is bad and not working.",
                    "priority": "high",
                    "status": "open",
                    "issue_type": None,
                    "resolution_summary": None,
                    "customer_message_history": [],
                    "pending_customer_replies": ["Still bad. Still not working."],
                    "response_history": [],
                    "tool_failures": {},
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    agent = build_agent(tickets_path=tickets_path)
    log = agent.run("TICK-3002", "Resolve the customer ticket end-to-end.")

    assert log.status == "escalated"
    assert log.final_response is not None
    assert "Classification:" in log.final_response
    assert "Latest customer message:" in log.final_response
    assert "KB attempts:" in log.final_response
    escalation_step = next(step for step in log.steps if step.tool == "escalate_ticket")
    assert "customer_message" in escalation_step.data
