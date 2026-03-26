from __future__ import annotations

from pathlib import Path

from ticket_agent.agent import TicketResolutionAgent
from ticket_agent.config import Settings
from ticket_agent.knowledge_base import KnowledgeBase
from ticket_agent.reporting import build_demo_report, build_execution_log_report
from ticket_agent.repository import TicketRepository
from ticket_agent.tools import TicketResolutionToolkit


FIXTURE_TICKETS = Path(__file__).resolve().parent / "fixtures" / "sample_tickets.json"


def build_agent() -> TicketResolutionAgent:
    settings = Settings(use_ollama=False)
    repository = TicketRepository(FIXTURE_TICKETS)
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


def test_build_execution_log_report_includes_grounding_and_outcome() -> None:
    agent = build_agent()
    log = agent.run("TICK-1001", "Resolve the customer ticket end-to-end.")

    report = build_execution_log_report(log)

    assert "# Ticket Report: TICK-1001" in report
    assert "matched terms" in report
    assert "KB-100" in report
    assert "Final ticket status" in report


def test_build_demo_report_includes_overview_table() -> None:
    agent = build_agent()
    logs = [
        agent.run("TICK-1001", "Resolve the customer ticket end-to-end."),
        agent.run("TICK-1003", "Resolve the customer ticket end-to-end."),
    ]

    report = build_demo_report(logs)

    assert "| Ticket | Status | Plan | Response | Retries | Final Ticket Status |" in report
    assert "### TICK-1001" in report
    assert "### TICK-1003" in report
